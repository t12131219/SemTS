"""官方 ShuffleMTM 模型在 AD/PTB 分类上的复现（论文 Table 3 / Table 4）。

官方仓库无分类入口；本驱动严格复用官方模型（models.shufflemtm.Model）与官方预训练逻辑
（RevIN → patch → 通道打乱(permute) → 随机掩码 → 孪生编码器 + 交叉注意力解码器 → MSE 重建），
微调采用官方 freeze-then-finetune 策略（先线性探针后全量微调）+ 交叉熵。
数据：COMET 处理后 AD/PTB（figshare 直链），患者独立划分与 COMET 论文一致
（AD: val 17/18, test 19/20；PTB: 按标签排序的负/正患者列表取后 14/7 与 42/21）。

协议（论文附录 Table 7/8，单次运行；论文为 5-run 平均）：
  预训练 100 epoch、lr 5e-4、batch 256、mask_ratio 0.4；
  微调 lr 5e-4、batch 128、LP/FT 20/40（AD）、30/60（PTB）；
  少量标签比例 {100%, 20%, 10%, 5%}（Table 3 + Table 4）。

用法（在 SemTS 根目录）:
    python third_party/run_shufflemtm_classify.py --data_root data/shufflemtm/AD --name AD --device cuda
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SHUF = os.path.join(HERE, "ShuffleMTM")
sys.path.insert(0, SHUF)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

import patch_mask
from models.layers.revin import RevIN
from models.shufflemtm import Model

ARCH = {
    #            patch_len  d_model  d_ff  heads  LP  FT
    "AD":  dict(pl=32, dm=64,  df=128, nh=8,  lp=20, ft=40),
    "PTB": dict(pl=30, dm=128, df=256, nh=8,  lp=30, ft=60),
}


def load_comet(name, data_root):
    """返回 (X_tr, y_tr, X_va, y_va, X_te, y_te)，每个 X: (N, T, C)。"""
    feat_dir = os.path.join(data_root, "Feature")
    label = np.load(os.path.join(data_root, "Label", "label.npy"))  # (N_p, 2): [label, id]
    files = sorted(os.listdir(feat_dir))
    assert len(files) == len(label), f"{name}: feature 文件数与 label 行数不一致"
    if name == "AD":
        val_ids, test_ids = [17, 18], [19, 20]
    else:  # PTB: 负/正患者按出现顺序（即文件顺序）取尾部
        neg = list(np.where(label[:, 0] == 0)[0] + 1)
        pos = list(np.where(label[:, 0] == 1)[0] + 1)
        val_ids = neg[-14:-7] + pos[-42:-21]
        test_ids = neg[-7:] + pos[-21:]
    Xs, ys = {"train": [], "val": [], "test": []}, {"train": [], "val": [], "test": []}
    for j, fn in enumerate(files):
        subj = np.load(os.path.join(feat_dir, fn))  # (N_r, T, C)
        lab = int(label[j, 0])
        split = "val" if (j + 1) in val_ids else ("test" if (j + 1) in test_ids else "train")
        for trial in subj:
            Xs[split].append(trial.astype(np.float32))
            ys[split].append(lab)
    out = {}
    for s in ("train", "val", "test"):
        out[s] = (np.stack(Xs[s]), np.array(ys[s]))
        print(f"{name} {s}: X{out[s][0].shape} y{out[s][1].shape}")
    return out


def build_model(name, C, T, head_type):
    a = ARCH[name]
    num_patch = (T - a["pl"]) // a["pl"] + 1
    return Model(c_in=C, target_dim=0 if head_type == "pretrain" else 2,
                 patch_len=a["pl"], stride=a["pl"], num_patch=num_patch,
                 e_layers=3, d_layers=1, n_heads=a["nh"], d_model=a["dm"],
                 shared_embedding=True, d_ff=a["df"], dropout=0.2, head_dropout=0.2,
                 act="relu", head_type=head_type, res_attention=False)


def pretrain_iter(model, revin, x, mask_ratio, pl, dev):
    x = x.to(dev)
    normed = revin(x, "norm")
    x_patch, _ = patch_mask.create_patch(normed, pl)
    x_perm = patch_mask.permute_patch(x_patch)
    x_patch_m, mask = patch_mask.random_masking(x_patch, mask_ratio)
    x_perm_m = patch_mask.random_masking(x_perm, mask=mask)
    pred = model(x_patch_m, x_perm_m, ca_query="original")
    return ((pred - x_patch) ** 2).mean()


def pretrain(args, data, dev):
    X, y = data["train"]
    Xv, yv = data["val"]
    C, T = X.shape[2], X.shape[1]
    pl = ARCH[args.name]["pl"]
    model = build_model(args.name, C, T, "pretrain").to(dev)
    revin = RevIN(C).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(revin.parameters()), lr=5e-4)
    tr = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=256, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xv)), batch_size=256)
    best_val, best_state, t0 = 1e18, None, time.time()
    for ep in range(args.pretrain_epochs):
        model.train()
        tot = 0.0
        for (xb,) in tr:
            opt.zero_grad()
            loss = pretrain_iter(model, revin, xb, 0.4, pl, dev)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        model.eval()
        vtot = sum(pretrain_iter(model, revin, xb, 0.4, pl, dev).item() * xb.size(0)
                   for (xb,) in va) / len(Xv)
        print(f"[{args.name}] pretrain ep{ep + 1}: train={tot / len(X):.4f} val={vtot:.4f} "
              f"({time.time() - t0:.0f}s)")
        if vtot < best_val:
            best_val, best_state = vtot, {k: v.clone() for k, v in model.state_dict().items()}
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"pretrain_{args.name}.pt")
    torch.save({"model": best_state, "arch": ARCH[args.name], "C": C, "T": T}, ckpt)
    print(f"[{args.name}] pretrain saved -> {ckpt}")
    return ckpt


def metrics(y, p):
    return {"acc": accuracy_score(y, p), "prec": precision_score(y, p, zero_division=0),
            "rec": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0)}


def finetune_fraction(args, data, ckpt, fraction, dev):
    X, y = data["train"]
    Xv, yv = data["val"]
    Xt, yt = data["test"]
    C, T = X.shape[2], X.shape[1]
    # 少量标签子集（论文 Table 4）
    if fraction < 1.0:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=int(len(X) * fraction), replace=False)
        X, y = X[idx], y[idx]

    a = ARCH[args.name]
    model = build_model(args.name, C, T, "classification").to(dev)
    state = torch.load(ckpt, map_location=dev)["model"]
    model.backbone.load_state_dict({k[len("backbone."):]: v for k, v in state.items()
                                    if k.startswith("backbone.")})
    crit = nn.CrossEntropyLoss()
    tr = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
                    batch_size=128, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv).long()), batch_size=256)

    def one_epoch(opt, sched, train=True):
        model.train() if train else model.eval()
        tot, corr, n = 0.0, 0, 0
        for xb, yb in tr if train else va:
            if train:
                opt.zero_grad()
            xb = xb.to(dev)
            z = model.backbone(patch_mask.create_patch(RevIN_norm(xb), a["pl"])[0])
            out = model.head(z)
            loss = crit(out, yb.to(dev))
            if train:
                loss.backward()
                opt.step()
                sched.step()
            tot += loss.item() * len(yb)
            corr += (out.argmax(1).cpu() == yb).sum().item()
            n += len(yb)
        return tot / n, corr / n

    # freeze（线性探针）
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, steps_per_epoch=len(tr),
                                                pct_start=0.3, epochs=a["lp"] + a["ft"],
                                                max_lr=5e-4)
    for ep in range(a["lp"]):
        tl, ta = one_epoch(opt, sched, True)
        vl, va_acc = one_epoch(None, sched, False)
        print(f"[{args.name} frac={fraction}] LP ep{ep + 1}: tr_loss={tl:.4f} va_acc={va_acc:.4f}")
    for p in model.backbone.parameters():
        p.requires_grad = True
    for ep in range(a["ft"]):
        tl, ta = one_epoch(opt, sched, True)
        vl, va_acc = one_epoch(None, sched, False)
        print(f"[{args.name} frac={fraction}] FT ep{ep + 1}: tr_loss={tl:.4f} va_acc={va_acc:.4f}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt).long()),
                                 batch_size=256):
            xb = xb.to(dev)
            z = model.backbone(patch_mask.create_patch(RevIN_norm(xb), a["pl"])[0])
            preds.append(model.head(z).argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    return metrics(np.concatenate(trues), np.concatenate(preds))


def RevIN_norm(x):
    """per-sample RevIN（与官方一致：statistics 在样本内计算）。"""
    mean = x.mean(dim=1, keepdim=True)
    std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return (x - mean) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--name", type=str, required=True, choices=["AD", "PTB"])
    ap.add_argument("--fractions", type=str, default="1.0,0.2,0.1,0.05")
    ap.add_argument("--pretrain_epochs", type=int, default=100, help="冒烟可设小值（论文 100）")
    ap.add_argument("--out", type=str, default=os.path.join("experiments", "shufflemtm_classify"))
    ap.add_argument("--pretrain_ckpt", type=str, default=None, help="跳过预训练直接微调")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    data = load_comet(args.name, args.data_root)
    ckpt = args.pretrain_ckpt or pretrain(args, data, dev)
    results = {}
    for f in [float(x) for x in args.fractions.split(",")]:
        m = finetune_fraction(args, data, ckpt, f, dev)
        results[str(f)] = m
        print(f"[{args.name}] fraction={f}: {m}")
    out_json = os.path.join(args.out, f"{args.name}_results.json")
    os.makedirs(args.out, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[{args.name}] done -> {out_json}")


if __name__ == "__main__":
    main()
