"""用预计算特征做预测线性评估（与 finetune_forecast.py 完全相同的岭回归协议）。

特征目录 --feats 需含：
    train.npy / val.npy / test.npy
    - (N, C, d)：逐通道特征（SemTS 协议）
    - (N, d)  ：整窗共享特征（官方 TS2Vec/CoST 输出，通道合并）→ 自动按通道复制扩展
窗口与目标由 --data 的 SlidingWindowDataset 按相同切分重建（必须与特征行数一致）。

用法:
    python scripts/finetune_from_features.py --feats experiments/official_ts2vec/feats \
        --data data/ETTm1.csv --horizons 96,192,336,720
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

from semts.datasets.csv_dataset import SlidingWindowDataset


def ridge_fit(X, Y, lam, dev):
    d = X.size(1)
    A = X.t() @ X + lam * torch.eye(d, device=dev)
    b = X.t() @ Y
    return torch.linalg.solve(A, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--max_windows", type=int, default=None,
                    help="每 split 最多窗口数（与特征导出时一致）")
    ap.add_argument("--horizons", type=str, default="96,192,336,720")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    F = {}
    for split in ("train", "val", "test"):
        f = np.load(os.path.join(args.feats, f"{split}.npy"))
        F[split] = f
        print(f"[finetune_from_features] {split} feats: {f.shape}")

    ds = SlidingWindowDataset(args.data, L=args.L)
    horizons = [int(h) for h in args.horizons.split(",")]
    Hmax = max(horizons)
    _, y_tr = ds.windows_with_target("train", Hmax)
    _, y_va = ds.windows_with_target("val", Hmax)
    _, y_te = ds.windows_with_target("test", Hmax)
    if args.max_windows is not None:
        y_tr, y_va, y_te = y_tr[:args.max_windows], y_va[:args.max_windows], y_te[:args.max_windows]

    C = ds.n_channels
    for split, feat in F.items():
        if feat.ndim == 2:  # (N, d) → 逐通道复制 (N, C, d)
            F[split] = np.repeat(feat[:, None, :], C, axis=1)
    # 适配器导出的是全部窗口（n-L+1），评估只用"有未来目标"的窗口（n-L-H+1）：
    # 两者窗口顺序一致（均从切分起点开始），按目标窗口数截断特征即对齐
    for split in ("train", "val", "test"):
        y = {"train": y_tr, "val": y_va, "test": y_te}[split]
        if F[split].shape[0] > y.shape[0]:
            F[split] = F[split][:y.shape[0]]
        assert F[split].shape[0] == y.shape[0], \
            f"{split} 特征行数 {F[split].shape[0]} 与窗口数 {y.shape[0]} 不一致"

    Xtr = torch.from_numpy(F["train"].reshape(-1, F["train"].shape[-1])).to(dev)
    Xva = torch.from_numpy(F["val"].reshape(-1, F["val"].shape[-1])).to(dev)
    Xte = torch.from_numpy(F["test"].reshape(-1, F["test"].shape[-1])).to(dev)

    for H in horizons:
        Ytr = torch.from_numpy(y_tr[:, :H]).permute(0, 2, 1).reshape(-1, H).to(dev)
        Yva = torch.from_numpy(y_va[:, :H]).permute(0, 2, 1).reshape(-1, H).to(dev)
        Yte = torch.from_numpy(y_te[:, :H]).permute(0, 2, 1).reshape(-1, H).to(dev)
        best = (None, 1e18)
        for lam in (0.1, 1.0, 10.0, 100.0, 1000.0):
            W = ridge_fit(Xtr, Ytr, lam, dev)
            mse = ((Xva @ W - Yva) ** 2).mean().item()
            if mse < best[1]:
                best = (W, mse)
        W = best[0]
        mse = ((Xte @ W - Yte) ** 2).mean().item()
        mae = (Xte @ W - Yte).abs().mean().item()
        print(f"H={H:>4}: MSE={mse:.4f} MAE={mae:.4f} (best val MSE={best[1]:.4f})")


if __name__ == "__main__":
    main()
