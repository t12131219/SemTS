"""零样本/跨域评估：源域拟合岭回归 → 目标域直接预测（不接触目标域训练数据）。

v0.1 约定：源/目标通道数相同；不同时取前 min(C) 个通道（打印警告）。

用法:
    python scripts/eval_zeroshot.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt \
        --data_src data/ETTm1.csv --data_tgt data/ETTm2.csv --horizons 96,336
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

from semts.datasets.csv_dataset import SlidingWindowDataset
from semts.model_factory import load_encoder_for_eval


@torch.no_grad()
def extract_features(model, x, dev, batch=512):
    reps = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(dev)
        z = model.encode(xb)
        reps.append(z[:, :, -1, :].cpu())
    return torch.cat(reps, 0)


def ridge_fit(X, Y, lam, dev):
    d = X.size(1)
    A = X.t() @ X + lam * torch.eye(d, device=dev)
    b = X.t() @ Y
    return torch.linalg.solve(A, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_src", type=str, required=True)
    ap.add_argument("--data_tgt", type=str, required=True)
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--horizons", type=str, default="96,336")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_encoder_for_eval(args.ckpt, dev)
    cfg = model.cfg

    src = SlidingWindowDataset(args.data_src, L=args.L)
    tgt = SlidingWindowDataset(args.data_tgt, L=args.L)
    C = min(src.n_channels, tgt.n_channels)
    if src.n_channels != tgt.n_channels:
        print(f"[eval_zeroshot] 通道数不同（{src.n_channels} vs {tgt.n_channels}），"
              f"取前 {C} 个通道")

    horizons = [int(h) for h in args.horizons.split(",")]
    Hmax = max(horizons)
    x_src, y_src = src.windows_with_target("train", Hmax)
    x_tgt, y_tgt = tgt.windows_with_target("test", Hmax)

    Xsrc = extract_features(model, x_src[:, :, :C], dev).reshape(-1, cfg.d_model)
    Xtgt = extract_features(model, x_tgt[:, :, :C], dev).reshape(-1, cfg.d_model)

    for H in horizons:
        Ysrc = torch.from_numpy(y_src[:, :H, :C]).permute(0, 2, 1).reshape(-1, H).to(dev)
        Ytgt = torch.from_numpy(y_tgt[:, :H, :C]).permute(0, 2, 1).reshape(-1, H).to(dev)
        best = (None, 1e18)
        for lam in (0.1, 1.0, 10.0, 100.0, 1000.0):
            W = ridge_fit(Xsrc.to(dev), Ysrc, lam, dev)
            mse = ((Xsrc.to(dev) @ W - Ysrc) ** 2).mean().item()  # 源域选择 lam（无目标域信息）
            if mse < best[1]:
                best = (W, mse)
        W = best[0]
        mse = ((Xtgt.to(dev) @ W - Ytgt) ** 2).mean().item()
        mae = (Xtgt.to(dev) @ W - Ytgt).abs().mean().item()
        print(f"[eval_zeroshot] {args.data_src} → {args.data_tgt}, "
              f"H={H}: MSE={mse:.4f} MAE={mae:.4f}")


if __name__ == "__main__":
    main()
