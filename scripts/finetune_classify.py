"""时序分类线性探针：冻结编码器 → 实例表示（每通道 maxpool 后拼接）→ 逻辑回归。

数据约定：--data_dir 下含
    train.npy / val.npy / test.npy        (N, L, C)
    train_label.npy / val_label.npy / test_label.npy   (N,) 类别 0..K-1
（SleepEEG/Epilepsy/FD-B/Gesture/EMG 可由 SimMTM 官方仓库下载后转换为此格式；
  UEA 数据集同理，按官方 train/test 划分放置。）

用法:
    python scripts/finetune_classify.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt \
        --data_dir data/SleepEEG --epochs 100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from semts.model_factory import load_encoder_for_eval


@torch.no_grad()
def extract_instance_reps(model, x, dev, batch=256):
    """x: (N, L, C) numpy → reps: (N, C*d)（每通道沿时间 maxpool 后拼接，TS2Vec 式实例表示）。"""
    reps = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(dev)
        z = model.encode(xb)          # (B, C, P, d)
        r = z.max(dim=2).values       # (B, C, d)
        reps.append(r.reshape(r.size(0), -1).cpu())
    return torch.cat(reps, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[finetune_classify] device={dev}")

    model = load_encoder_for_eval(args.ckpt, dev)
    cfg = model.cfg

    def load(split):
        x = np.load(os.path.join(args.data_dir, f"{split}.npy"))
        y = np.load(os.path.join(args.data_dir, f"{split}_label.npy")).astype(np.int64)
        return x, y

    x_tr, y_tr = load("train")
    x_va, y_va = load("val")
    x_te, y_te = load("test")

    Xtr = extract_instance_reps(model, x_tr, dev)
    Xva = extract_instance_reps(model, x_va, dev)
    Xte = extract_instance_reps(model, x_te, dev)
    K = int(y_tr.max()) + 1
    print(f"[finetune_classify] dim={Xtr.shape[1]}, classes={K}, "
          f"n_train={len(x_tr)} n_val={len(x_va)} n_test={len(x_te)}")

    clf = nn.Linear(Xtr.shape[1], K).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    tr_loader = DataLoader(TensorDataset(Xtr, torch.from_numpy(y_tr)),
                           batch_size=args.batch_size, shuffle=True)
    best_acc, best_state = 0.0, None
    for ep in range(args.epochs):
        clf.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = ce(clf(xb), yb)
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            acc = (clf(Xva.to(dev)).argmax(1).cpu() == torch.from_numpy(y_va)).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in clf.state_dict().items()}
    clf.load_state_dict(best_state)
    with torch.no_grad():
        test_acc = (clf(Xte.to(dev)).argmax(1).cpu() == torch.from_numpy(y_te)).float().mean().item()
    print(f"[finetune_classify] best val acc={best_acc:.4f}, test acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
