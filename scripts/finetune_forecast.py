"""预测线性评估（岭回归闭式解，TS2Vec 协议）：冻结编码器 → 末 patch 表示 → 未来 H 步。

用法:
    python scripts/finetune_forecast.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt \
        --data data/ETTm1.csv [--horizons 96,192,336,720]
指标在 z-score 标准化空间报告（与 TS2Vec/CoST/SimMTM 协议一致）。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

from semts.datasets.csv_dataset import SlidingWindowDataset
from semts.model_factory import load_encoder_for_eval


@torch.no_grad()
def extract_features(model, x, dev, batch=512):
    """x: (N, L, C) numpy → reps: (N, C, d)（末 patch 表示）。"""
    reps = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(dev)
        z = model.encode(xb)
        reps.append(z[:, :, -1, :].cpu())
    return torch.cat(reps, 0)


def ridge_fit(X, Y, lam, dev):
    """岭回归闭式解（正规方程）：min ||XW - Y||^2 + lam ||W||^2。"""
    d = X.size(1)
    A = X.t() @ X + lam * torch.eye(d, device=dev)
    b = X.t() @ Y
    return torch.linalg.solve(A, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--max_channels", type=int, default=None,
                    help="通道截断（大通道数据集防止特征矩阵过大）")
    ap.add_argument("--horizons", type=str, default="96,192,336,720")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[finetune_forecast] device={dev}")

    ckpt_path = args.ckpt
    model = load_encoder_for_eval(ckpt_path, dev)
    cfg = model.cfg

    ds = SlidingWindowDataset(args.data, L=args.L, max_channels=args.max_channels)
    horizons = [int(h) for h in args.horizons.split(",")]
    Hmax = max(horizons)

    x_tr, y_tr = ds.windows_with_target("train", Hmax)
    x_va, y_va = ds.windows_with_target("val", Hmax)
    x_te, y_te = ds.windows_with_target("test", Hmax)

    Xtr = extract_features(model, x_tr, dev).reshape(-1, cfg.d_model)
    Xva = extract_features(model, x_va, dev).reshape(-1, cfg.d_model)
    Xte = extract_features(model, x_te, dev).reshape(-1, cfg.d_model)

    print(f"[finetune_forecast] features: train={Xtr.shape[0]} val={Xva.shape[0]} "
          f"test={Xte.shape[0]}, dim={cfg.d_model}")

    for H in horizons:
        Ytr = torch.from_numpy(y_tr[:, :H]).permute(0, 2, 1).reshape(-1, H).to(dev)
        Yva = torch.from_numpy(y_va[:, :H]).permute(0, 2, 1).reshape(-1, H).to(dev)
        Yte = torch.from_numpy(y_te[:, :H]).permute(0, 2, 1).reshape(-1, H).to(dev)
        best = (None, 1e18)
        for lam in (0.1, 1.0, 10.0, 100.0, 1000.0):
            W = ridge_fit(Xtr.to(dev), Ytr, lam, dev)
            mse = ((Xva.to(dev) @ W - Yva) ** 2).mean().item()
            if mse < best[1]:
                best = (W, mse)
        W = best[0]
        mse = ((Xte.to(dev) @ W - Yte) ** 2).mean().item()
        mae = (Xte.to(dev) @ W - Yte).abs().mean().item()
        print(f"H={H:>4}: MSE={mse:.4f} MAE={mae:.4f} (best val MSE={best[1]:.4f})")


if __name__ == "__main__":
    main()
