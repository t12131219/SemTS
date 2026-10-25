"""官方 ShuffleMTM 模型在 AD/PTB 分类上的论文配置复现驱动（Table 3 / Table 4）。

背景：官方仓库无分类入口；本驱动复用官方 Model/预训练逻辑，并按论文附录
Table 7/8 的搜索空间做"验证集选优 + 5 次平均"（与论文"determine the best
configuration on a pre-defined validation dataset / 5-run average"一致）。

论文配置（Table 7 模型 / Table 8 训练）：
  AD:  patch_len {32, 64}, d_model {64, 128}, d_ff {64, 128}, heads {8, 16},
       mask 0.4, 预训练 lr {5e-4, 1e-3}×batch 256×100ep,
       微调 lr {5e-4, 1e-3}×batch 128, LP/FT {(20,40), (30,60)},
       池化 {last（官方代码）, max（论文文字）}
  PTB: patch_len {15, 30}, d_model {64, 128, 256}, d_ff {128, 256, 512},
       mask {0.4, 0.8}, 其余同上

搜索策略（分阶段，验证集选优；全网格 256/1152 不可行，论文本身亦为关键参数搜索）：
  stage=arch   扫 patch×d_model(×d_ff 耦合)×heads（lr=5e-4, mask=0.4, LPFT 默认, pool=last）
  stage=train  在最佳架构上扫 mask(PTB)/微调lr/LPFT
  stage=pool   在最佳训练配置上扫 pool∈{last,max}
  stage=final  最佳配置 × --seeds 次 + 标签比例 {100%,20%,10%,5%}（Table 3+4）
  stage=all    依次执行以上全部

用法（在 SemTS 根目录）:
    python third_party/run_shufflemtm_classify.py --name AD --data_root data/shufflemtm/AD \
        --stage all --seeds 5
"""
import argparse
import copy
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

# 论文 Table 7/8 搜索空间
SPACE = {
    "AD": dict(
        pl=[32, 64], dm=[64, 128], df=[64, 128], nh=[8, 16],
        mask=[0.4], plr=[5e-4, 1e-3], lpft=[(20, 40), (30, 60)],
        ftlr=[5e-4, 1e-3], pool=["last", "max"], df_coupled=True,   # d_ff 随 d_model 配对
    ),
    "PTB": dict(
        pl=[15, 30], dm=[64, 128, 256], df=[128, 256, 512], nh=[8, 16],
        mask=[0.4, 0.8], plr=[5e-4, 1e-3], lpft=[(30, 60), (20, 40)],
        ftlr=[5e-4, 1e-3], pool=["last", "max"], df_coupled=True,
    ),
}
DEFAULTS = {"pl": None, "dm": None, "df": None, "nh": 8, "mask": 0.4, "plr": 5e-4,
            "lpft": None, "ftlr": 5e-4, "pool": "last"}


class MaxPoolHead(nn.Module):
    """论文文字口径：每通道沿 patch 维 max-pool 后拼接（与官方 last-patch 头对照）。"""

    def __init__(self, n_vars, d_model, n_classes, head_dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):  # x: [bs x nvars x d_model x num_patch]
        x = x.max(dim=-1).values      # [bs x nvars x d_model]
        x = self.dropout(self.flatten(x))
        return self.linear(x)


def load_comet(name, data_root):
    feat_dir = os.path.join(data_root, "Feature")
    label = np.load(os.path.join(data_root, "Label", "label.npy"))
    files = sorted(os.listdir(feat_dir))
    assert len(files) == len(label), f"{name}: feature 文件数与 label 行数不一致"
    if name == "AD":
        val_ids, test_ids = [17, 18], [19, 20]
    else:
        neg = list(np.where(label[:, 0] == 0)[0] + 1)
        pos = list(np.where(label[:, 0] == 1)[0] + 1)
        val_ids = neg[-14:-7] + pos[-42:-21]
        test_ids = neg[-7:] + pos[-21:]
    Xs, ys = {"train": [], "val": [], "test": []}, {"train": [], "val": [], "test": []}
    for j, fn in enumerate(files):
        subj = np.load(os.path.join(feat_dir, fn))
        lab = int(label[j, 0])
        split = "val" if (j + 1) in val_ids else ("test" if (j + 1) in test_ids else "train")
        for trial in subj:
            Xs[split].append(trial.astype(np.float32))
            ys[split].append(lab)
    out = {}
    for s in ("train", "val", "test"):
        out[s] = (np.stack(Xs[s]), np.array(ys[s]))
    return out


def build_model(name, C, T, cfg):
    num_patch = (T - cfg["pl"]) // cfg["pl"] + 1
    return Model(c_in=C, target_dim=0, patch_len=cfg["pl"], stride=cfg["pl"],
                 num_patch=num_patch, e_layers=3, d_layers=1, n_heads=cfg["nh"],
                 d_model=cfg["dm"], shared_embedding=True, d_ff=cfg["df"],
                 dropout=0.2, head_dropout=0.2, act="relu",
                 head_type="pretrain", res_attention=False)


def pretrain_one(name, data, cfg, seed, dev, ckpt_dir, epochs=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y = data["train"]
    Xv, yv = data["val"]
    C, T = X.shape[2], X.shape[1]
    model = build_model(name, C, T, cfg).to(dev)
    revin = RevIN(C).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(revin.parameters()), lr=cfg["plr"])
    tr = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=256, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xv)), batch_size=256)

    def step(xb, train):
        if train:
            opt.zero_grad()
        xb = xb.to(dev)
        normed = revin(xb, "norm")
        x_patch, _ = patch_mask.create_patch(normed, cfg["pl"])
        x_perm = patch_mask.permute_patch(x_patch)
        x_patch_m, mask = patch_mask.random_masking(x_patch, cfg["mask"])
        x_perm_m = patch_mask.random_masking(x_perm, mask=mask)
        pred = model(x_patch_m, x_perm_m, ca_query="original")
        loss = ((pred - x_patch) ** 2).mean()
        if train:
            loss.backward()
            opt.step()
        return loss.item()

    best_val, best_state = 1e18, None
    for ep in range(epochs):
        model.train()
        tot = sum(step(xb, True) * xb.size(0) for (xb,) in tr) / len(X)
        model.eval()
        with torch.no_grad():
            vtot = sum(step(xb, False) * xb.size(0) for (xb,) in va) / len(Xv)
        print(f"  [pretrain] ep{ep + 1}: tr={tot:.4f} va={vtot:.4f}", flush=True)
        if vtot < best_val:
            best_val, best_state = vtot, {k: v.clone() for k, v in model.state_dict().items()}
    key = f"{name}_pl{cfg['pl']}_dm{cfg['dm']}_df{cfg['df']}_nh{cfg['nh']}_m{cfg['mask']}_lr{cfg['plr']}_s{seed}"
    ckpt = os.path.join(ckpt_dir, f"{key}.pt")
    torch.save({"model": best_state, "cfg": cfg, "C": C, "T": T}, ckpt)
    return ckpt


def finetune_eval(name, data, ckpt, fraction, seed, cfg, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y = data["train"]
    Xv, yv = data["val"]
    Xt, yt = data["test"]
    C, T = X.shape[2], X.shape[1]
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=int(len(X) * fraction), replace=False)
        X, y = X[idx], y[idx]
    st = torch.load(ckpt, map_location=dev)
    model = build_model(name, C, T, st["cfg"]).to(dev)
    model.backbone.load_state_dict({k[len("backbone."):]: v for k, v in st["model"].items()
                                    if k.startswith("backbone.")})
    num_patch = (T - cfg["pl"]) // cfg["pl"] + 1
    if cfg["pool"] == "max":
        model.head = MaxPoolHead(C, cfg["dm"], 2, head_dropout=0.2).to(dev)  # 替换官方 head
    else:
        # 官方 head_type='classification'（last patch）；pretrain 模型无 head → 手动挂
        model.head = Model(c_in=C, target_dim=2, patch_len=cfg["pl"], stride=cfg["pl"],
                           num_patch=num_patch, e_layers=3, d_layers=1, n_heads=cfg["nh"],
                           d_model=cfg["dm"], shared_embedding=True, d_ff=cfg["df"],
                           dropout=0.2, head_dropout=0.2, act="relu",
                           head_type="classification", res_attention=False).head.to(dev)
    crit = nn.CrossEntropyLoss()
    tr = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
                    batch_size=128, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv).long()), batch_size=256)

    def norm(xb):
        m = xb.mean(1, keepdim=True)
        s = torch.sqrt(torch.var(xb, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return (xb - m) / s

    def one_epoch(opt, sched, train=True):
        model.train() if train else model.eval()
        tot, corr, n = 0.0, 0, 0
        loader = tr if train else va
        for xb, yb in loader:
            if train:
                opt.zero_grad()
            xb = xb.to(dev)
            z = model.backbone(patch_mask.create_patch(norm(xb), cfg["pl"])[0])
            out = model.head(z)
            loss = crit(out, yb.to(dev))
            if train:
                loss.backward()
                opt.step()
                if sched is not None:
                    sched.step()
            tot += loss.item() * len(yb)
            corr += (out.argmax(1).cpu() == yb).sum().item()
            n += len(yb)
        return tot / n, corr / n

    lp, ft = cfg["lpft"]
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["ftlr"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, steps_per_epoch=len(tr),
                                                pct_start=0.3, epochs=lp + ft, max_lr=cfg["ftlr"])
    for ep in range(lp):
        one_epoch(opt, sched, True)
    for p in model.backbone.parameters():
        p.requires_grad = True
    best_va = 0.0
    best_head = None
    for ep in range(ft):
        _, va_acc = one_epoch(opt, sched, False)
        if va_acc > best_va:
            best_va = va_acc
            best_head = {k: v.clone() for k, v in model.head.state_dict().items()}
    if best_head is not None:
        model.head.load_state_dict(best_head)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt).long()),
                                 batch_size=256):
            xb = xb.to(dev)
            z = model.backbone(patch_mask.create_patch(norm(xb), cfg["pl"])[0])
            preds.append(model.head(z).argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    return {"acc": float(accuracy_score(trues, preds)),
            "prec": float(precision_score(trues, preds, zero_division=0)),
            "rec": float(recall_score(trues, preds, zero_division=0)),
            "f1": float(f1_score(trues, preds, zero_division=0)), "val_acc": best_va}


def default_lpft(name):
    return (20, 40) if name == "AD" else (30, 60)


def arch_configs(name):
    """stage=arch：patch × d_model（d_ff 耦合 2×dm，PTB 用搜索空间配对）× heads。"""
    s = SPACE[name]
    out = []
    for pl in s["pl"]:
        for dm in s["dm"]:
            for df in (s["df"] if not s["df_coupled"] else [2 * dm]):
                for nh in s["nh"]:
                    c = dict(DEFAULTS)
                    c.update(pl=pl, dm=dm, df=df, nh=nh, mask=0.4, plr=5e-4,
                             lpft=default_lpft(name), ftlr=5e-4, pool="last")
                    out.append(c)
    return out


def train_configs(name, best):
    """stage=train：在最佳架构上扫 mask(PTB) / ftlr / lpft / plr。"""
    s = SPACE[name]
    out = []
    for m in s["mask"]:
        for ftlr in s["ftlr"]:
            for lpft in s["lpft"]:
                c = copy.deepcopy(best)
                c.update(mask=m, ftlr=ftlr, lpft=lpft)
                out.append(c)
    return out


def pool_configs(name, best):
    out = []
    for p in SPACE[name]["pool"]:
        c = copy.deepcopy(best)
        c["pool"] = p
        out.append(c)
    return out


def cfg_key(name, cfg):
    return (f"{name}_pl{cfg['pl']}_dm{cfg['dm']}_df{cfg['df']}_nh{cfg['nh']}"
            f"_m{cfg['mask']}_lr{cfg['plr']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True, choices=["AD", "PTB"])
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--stage", type=str, default="all", choices=["arch", "train", "pool", "final", "all"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="冒烟：预训练 5 ep + LP/FT (2,4)")
    ap.add_argument("--out", type=str, default=os.path.join("experiments", "shufflemtm_classify"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_comet(args.name, args.data_root)
    ckpt_dir = os.path.join(args.out, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    results = {"configs": [], "best": None, "final": {}}
    best = None
    t0 = time.time()

    if args.stage in ("arch", "all"):
        print(f"===== stage=arch: {args.name} =====", flush=True)
        for c in arch_configs(args.name):
            if args.quick:
                c = dict(c, lpft=(2, 4))
            ckpt = pretrain_one(args.name, data, c, 42, dev, ckpt_dir,
                                epochs=5 if args.quick else 100)
            m = finetune_eval(args.name, data, ckpt, 1.0, 42, c, dev)
            results["configs"].append({"cfg": c, "metrics": m})
            print(f"  [arch] {c} -> acc={m['acc']:.3f} (va {m['val_acc']:.3f})", flush=True)
            if best is None or m["val_acc"] > best[1]:
                best = (c, m["val_acc"], m["acc"])

    if args.stage in ("train", "all"):
        print(f"===== stage=train (best arch {cfg_key(args.name, best[0])}) =====", flush=True)
        for c in train_configs(args.name, best[0]):
            if args.quick:
                c = dict(c, lpft=(2, 4))
            ckpt = pretrain_one(args.name, data, c, 42, dev, ckpt_dir,
                                epochs=5 if args.quick else 100)
            m = finetune_eval(args.name, data, ckpt, 1.0, 42, c, dev)
            results["configs"].append({"cfg": c, "metrics": m})
            print(f"  [train] {c} -> acc={m['acc']:.3f} (va {m['val_acc']:.3f})", flush=True)
            if m["val_acc"] > best[1]:
                best = (c, m["val_acc"], m["acc"])

    if args.stage in ("pool", "all"):
        print(f"===== stage=pool =====", flush=True)
        for c in pool_configs(args.name, best[0]):
            m = finetune_eval(args.name, data, os.path.join(
                ckpt_dir, cfg_key(args.name, c) + "_s42.pt"), 1.0, 42, c, dev)
            results["configs"].append({"cfg": c, "metrics": m})
            print(f"  [pool] {c} -> acc={m['acc']:.3f} (va {m['val_acc']:.3f})", flush=True)
            if m["val_acc"] > best[1]:
                best = (c, m["val_acc"], m["acc"])

    results["best"] = {"cfg": best[0], "val_acc": best[1], "test_acc": best[2]}
    print(f"===== BEST: {best[0]} va={best[1]:.3f} test={best[2]:.3f} =====", flush=True)

    if args.stage in ("final", "all"):
        print(f"===== stage=final: {args.seeds} seeds × fractions =====", flush=True)
        best_cfg = best[0]
        for frac in (1.0, 0.2, 0.1, 0.05):
            runs = []
            for seed in range(args.seeds):
                ckpt = pretrain_one(args.name, data, best_cfg, seed, dev, ckpt_dir)
                m = finetune_eval(args.name, data, ckpt, frac, seed, best_cfg, dev)
                runs.append(m)
                print(f"  [final] frac={frac} seed={seed}: acc={m['acc']:.3f}", flush=True)
            avg = {k: float(np.mean([r[k] for r in runs])) for k in ("acc", "prec", "rec", "f1")}
            avg["std_acc"] = float(np.std([r["acc"] for r in runs]))
            results["final"][str(frac)] = {"avg": avg, "runs": runs}

    out_json = os.path.join(args.out, f"{args.name}_search.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"[{args.name}] done in {(time.time() - t0) / 3600:.1f}h -> {out_json}")


if __name__ == "__main__":
    main()
