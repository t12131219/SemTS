"""SemTS 分类探针（ShuffleMTM 论文 Table 3/4 的 SemTS 行）。

冻结 SemTS 主配置编码器（ablation_w2）→ 每通道沿时间 maxpool 的实例表示 →
逻辑回归线性探针；标签比例 {100%, 20%, 10%, 5%}，四指标输出为 JSON。

用法: python scripts/run_semts_classify_probe.py [--ckpt experiments/ablation_w2/ckpt_ep9.pt]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from semts.model_factory import load_encoder_for_eval

FRACTIONS = {"1.0": 1.0, "0.2": 0.2, "0.1": 0.1, "0.05": 0.05}


@torch.no_grad()
def extract(model, x, dev, batch=64):
    reps = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(dev)
        z = model.encode(xb)          # (B, C, P, d)
        r = z.max(dim=2).values       # (B, C, d)
        reps.append(r.reshape(r.size(0), -1).cpu())
    return torch.cat(reps, 0)


def probe(Xtr, ytr, Xva, yva, Xte, yte, fraction, dev, seed=42, epochs=60):
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xtr), size=int(len(Xtr) * fraction), replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    clf = nn.Linear(Xtr.shape[1], 2).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
    best = (0.0, None)
    for ep in range(epochs):
        clf.train()
        for xb, yb in tr:
            opt.zero_grad()
            loss = ce(clf(xb.to(dev)), yb.to(dev))
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            acc = (clf(Xva.to(dev)).argmax(1).cpu() == yva).float().mean().item()
        if acc > best[0]:
            best = (acc, {k: v.clone() for k, v in clf.state_dict().items()})
    clf.load_state_dict(best[1])
    with torch.no_grad():
        pred = clf(Xte.to(dev)).argmax(1).cpu().numpy()
    return {"acc": float(accuracy_score(yte.numpy(), pred)),
            "prec": float(precision_score(yte.numpy(), pred, zero_division=0)),
            "rec": float(recall_score(yte.numpy(), pred, zero_division=0)),
            "f1": float(f1_score(yte.numpy(), pred, zero_division=0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="experiments/ablation_w2/ckpt_ep9.pt")
    ap.add_argument("--out", type=str, default="experiments/semts_classify_probe")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    model = load_encoder_for_eval(args.ckpt, dev)
    os.makedirs(args.out, exist_ok=True)
    for name in ("AD", "PTB"):
        d = os.path.join("data", "shufflemtm", f"{name}_classify")
        x_tr = np.load(os.path.join(d, "train.npy"))
        y_tr = torch.from_numpy(np.load(os.path.join(d, "train_label.npy")).astype(np.int64))
        x_va = np.load(os.path.join(d, "val.npy"))
        y_va = torch.from_numpy(np.load(os.path.join(d, "val_label.npy")).astype(np.int64))
        x_te = np.load(os.path.join(d, "test.npy"))
        y_te = torch.from_numpy(np.load(os.path.join(d, "test_label.npy")).astype(np.int64))
        print(f"[probe] {name}: extract features {x_tr.shape}...", flush=True)
        Xtr = extract(model, x_tr, dev)
        Xva = extract(model, x_va, dev)
        Xte = extract(model, x_te, dev)
        res = {}
        for fk, f in FRACTIONS.items():
            m = probe(Xtr, y_tr, Xva, y_va, Xte, y_te, f, dev)
            res[fk] = m
            print(f"[probe] {name} fraction={fk}: {m}", flush=True)
        with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    print(f"[probe] done -> {args.out}")


if __name__ == "__main__":
    main()
