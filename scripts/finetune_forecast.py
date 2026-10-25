"""预测线性评估（岭回归闭式解，TS2Vec 协议）：冻结编码器 → 末 patch 表示 → 未来 H 步。

内存安全设计（24GB 显存 / 32GB 内存机器友好）：
  - 特征提取按自适应批量（按通道数收缩），特征保存在 CPU；
  - 岭回归采用分块正规方程：A = Σ XᵢᵀXᵢ、b = Σ XᵢᵀYᵢ，X/Y 均分块流式（GPU 峰值 < 1GB），
    Y 目标矩阵按窗口流式生成（不再一次性物化 N×H×C）；
  - 大通道数据集（如 Traffic 862 通道、H=720）下内存仍可控。

--revin：逐窗口 RevIN（改进方案 A1）。每个窗口先按通道去均值/除标准差再编码；
目标用同一窗口统计量归一化；最终指标反归一化回原空间（与默认协议数字可直接对比）。
文献依据：RevIN/SAN（非平稳归一化）；arXiv:2605.19462（线性探针对预测的增益高度
依赖归一化处理）。

用法:
    python scripts/finetune_forecast.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt \
        --data data/ETTm1.csv [--horizons 96,192,336,720] [--revin]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

from semts.datasets.csv_dataset import SlidingWindowDataset
from semts.model_factory import load_encoder_for_eval


def window_iter(ds, split, L, H=None, chunk=256):
    """流式生成窗口：yield (x, y)，x: (c, L, C) float32；y: None 或 (c, H, C)。"""
    t0, t1 = ds.bounds[split]
    n = t1 - t0 - L - (H if H else 0) + 1
    assert n > 0, split
    rL = np.arange(L)
    rH = np.arange(H) if H else None
    for s in range(0, n, chunk):
        idx = np.arange(s, min(s + chunk, n))
        x = ds.data[t0 + idx[:, None] + rL]
        y = None if H is None else ds.data[t0 + idx[:, None] + L + rH]
        yield x, y


def _stats(x, dev):
    """逐窗口逐通道统计量：返回 (mu, sd)，形状 (c, 1, C)。"""
    xb = torch.from_numpy(x).to(dev)
    mu = xb.mean(dim=1, keepdim=True)
    sd = torch.sqrt(torch.var(xb, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return mu, sd


@torch.no_grad()
def extract_features(model, ds, split, L, dev, feat_batch=512, revin=False):
    """流式提取特征 → CPU 大数组 (N*C, d)。"""
    C = ds.n_channels
    P = (L - model.cfg.patch_len) // model.cfg.stride + 1
    b = min(feat_batch,
            max(1, int(0.25e9 // (C * P * model.cfg.d_model * 4))),
            max(1, 8192 // C))
    reps = []
    for x, _ in window_iter(ds, split, L, None, chunk=b):
        xb = torch.from_numpy(x).to(dev)
        if revin:
            mu, sd = _stats(x, dev)
            xb = (xb - mu) / sd
        z = model.encode(xb)
        reps.append(z[:, :, -1, :].cpu())
    X = torch.cat(reps, 0)
    return X.reshape(-1, model.cfg.d_model)


def _targets(x, y, dev, revin):
    """(c, H, C) → 按窗口×通道展开 (c*C, H)；revin 时用 x 的窗口统计量归一化，
    并返回 (mu, sd) 展开形式供反归一化。"""
    Yi = torch.from_numpy(y.transpose(0, 2, 1).reshape(-1, y.shape[1])).to(dev)
    mu = sd = None
    if revin:
        mu, sd = _stats(x, dev)                      # (c, 1, C)
        mu = mu.transpose(1, 2).reshape(-1, 1)       # (c*C, 1)
        sd = sd.transpose(1, 2).reshape(-1, 1)
        Yi = (Yi - mu) / sd
    return Yi, mu, sd


def ridge_fit_chunked(X, ds, split, L, H, lam, dev, revin=False, chunk=256):
    """A = XᵀX + λI；b = XᵀY（Y 流式），X 分块上传 GPU。"""
    d = X.size(1)
    C = ds.n_channels
    A = torch.zeros(d, d, device=dev)
    b = torch.zeros(d, H, device=dev)
    off = 0
    for x, y in window_iter(ds, split, L, H, chunk=chunk):
        rows = y.shape[0] * C
        Xi = X[off:off + rows].to(dev)
        Yi, _, _ = _targets(x, y, dev, revin)
        A += Xi.t() @ Xi
        b += Xi.t() @ Yi
        off += rows
    A += lam * torch.eye(d, device=dev)
    return torch.linalg.solve(A, b)


def eval_chunked(X, ds, split, L, H, W, dev, revin=False, chunk=256):
    """返回 (mse, mae)（反归一化回原空间，可与默认协议直接对比）。"""
    sse = sae = 0.0
    n = 0
    C = ds.n_channels
    off = 0
    for x, y in window_iter(ds, split, L, H, chunk=chunk):
        rows = y.shape[0] * C
        Xi = X[off:off + rows].to(dev)
        Yi, mu, sd = _targets(x, y, dev, revin)
        pred = Xi @ W                                   # (rows, H) 归一化空间
        if revin:
            pred = pred * sd + mu                       # 反归一化回原空间
            Yi = Yi * sd + mu
        e = pred - Yi
        sse += (e * e).sum().item()
        sae += e.abs().sum().item()
        n += e.numel()
        off += rows
    return sse / n, sae / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--max_channels", type=int, default=None,
                    help="通道截断（大通道数据集防止特征矩阵过大）")
    ap.add_argument("--horizons", type=str, default="96,192,336,720")
    ap.add_argument("--revin", action="store_true",
                    help="逐窗口 RevIN 归一化（改进方案 A1）")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[finetune_forecast] device={dev} revin={args.revin}")

    model = load_encoder_for_eval(args.ckpt, dev)
    cfg = model.cfg

    ds = SlidingWindowDataset(args.data, L=args.L, max_channels=args.max_channels)
    horizons = [int(h) for h in args.horizons.split(",")]

    Xtr = extract_features(model, ds, "train", args.L, dev, revin=args.revin)
    Xva = extract_features(model, ds, "val", args.L, dev, revin=args.revin)
    Xte = extract_features(model, ds, "test", args.L, dev, revin=args.revin)
    print(f"[finetune_forecast] features: train={Xtr.shape[0]} val={Xva.shape[0]} "
          f"test={Xte.shape[0]}, dim={cfg.d_model}")

    for H in horizons:
        best = (None, 1e18)
        for lam in (0.1, 1.0, 10.0, 100.0, 1000.0):
            W = ridge_fit_chunked(Xtr, ds, "train", args.L, H, lam, dev, revin=args.revin)
            mse, _ = eval_chunked(Xva, ds, "val", args.L, H, W, dev, revin=args.revin)
            if mse < best[1]:
                best = (W, mse)
        W = best[0]
        mse, mae = eval_chunked(Xte, ds, "test", args.L, H, W, dev, revin=args.revin)
        print(f"H={H:>4}: MSE={mse:.4f} MAE={mae:.4f} (best val MSE={best[1]:.4f})")


if __name__ == "__main__":
    main()
