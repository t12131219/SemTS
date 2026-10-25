"""跨通道岭回归预测评估（改进方案 A3，文献 arXiv:2609.32187）。

思想：通道独立的逐通道岭回归完全忽略跨通道信息；本脚本为每个目标通道 c：
  1) 用训练窗口上的相关系数筛选 top-k 个源通道（含自身，k 上限 12）；
  2) 以这些源通道的末 patch 表示拼接为特征（k·d 维），对目标 c 的未来 H 步拟合岭；
  3) 测试指标覆盖全部通道（与逐通道基线口径一致，可直接对比）。

与基线脚本不同：特征/目标按"窗口×通道"行布局（逐通道独立拟合，而非全局共享 W）。
内存：分块流式，24GB 友好。

用法:
    python scripts/finetune_forecast_crosschannel.py --ckpt experiments/ablation_w2/ckpt_ep9.pt \
        --data data/ETTm1.csv --horizons 96,336
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

from semts.datasets.csv_dataset import SlidingWindowDataset
from semts.model_factory import load_encoder_for_eval
from finetune_forecast import extract_features, window_iter


def fit_ridge(X, Y, lam, dev):
    d = X.size(1)
    A = X.t() @ X + lam * torch.eye(d, device=dev)
    return torch.linalg.solve(A, X.t() @ Y)


def y_count(ds, split, L, H):
    t0, t1 = ds.bounds[split]
    return t1 - t0 - L - H + 1


def y_matrix(ds, split, L, H, C, dev):
    """一次性流式物化 (Nw, H, C) 到 GPU（宽通道时远快于逐通道重复流式）。"""
    n = y_count(ds, split, L, H)
    Y = torch.empty(n, H, C, device=dev)
    off = 0
    for _, y in window_iter(ds, split, L, H, chunk=256):
        Y[off:off + y.shape[0]] = torch.from_numpy(y)
        off += y.shape[0]
    return Y


def y_for_channel(ds, split, L, H, c, dev):
    """流式收集目标通道 c 的 Y（Nw, H），不一次性物化 (N, H, C)。"""
    t0, t1 = ds.bounds[split]
    n = t1 - t0 - L - H + 1
    Y = torch.empty(n, H, device=dev)
    off = 0
    for _, y in window_iter(ds, split, L, H, chunk=256):
        Y[off:off + y.shape[0]] = torch.from_numpy(y[:, :, c])
        off += y.shape[0]
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--horizons", type=str, default="96,192,336,720")
    ap.add_argument("--k", type=int, default=0,
                    help="每目标通道源数上限；0=视野自适应（短视野多源、长视野少源）")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_encoder_for_eval(args.ckpt, dev)
    ds = SlidingWindowDataset(args.data, L=args.L)
    C = ds.n_channels
    horizons = [int(h) for h in args.horizons.split(",")]
    print(f"[crosschannel] C={C} device={dev}")

    # 特征 (N*C, d)，行序 = 窗口×通道；通道 c 的行 = X[c::C]
    Xtr = extract_features(model, ds, "train", args.L, dev)
    Xva = extract_features(model, ds, "val", args.L, dev)
    Xte = extract_features(model, ds, "test", args.L, dev)
    d = Xtr.size(1)
    Ntr = Xtr.size(0) // C
    print(f"[crosschannel] features: train={Ntr}w x {C}c, dim={d}")
    # 一次性转为"每通道连续存储"（避免逐通道跨步切片，宽通道集提速 ~10×）
    Xtr = [Xtr[c::C].contiguous() for c in range(C)]
    Xva = [Xva[c::C].contiguous() for c in range(C)]
    Xte = [Xte[c::C].contiguous() for c in range(C)]

    # 相关性筛选：train 窗口上通道间 |corr|（窗口均值近似，CPU 分块累计避免大张量上 GPU）
    means = []
    for x, _ in window_iter(ds, "train", args.L, None, chunk=512):
        means.append(x.mean(axis=1))                 # (c, C)
    m_tr = np.concatenate(means, 0)                  # (N, C) CPU
    m_tr = m_tr - np.mean(m_tr, axis=0, keepdims=True)
    denom = np.sqrt(np.sum(m_tr * m_tr, axis=0, keepdims=True)) + 1e-8
    corr = torch.from_numpy((m_tr.T @ m_tr) / (denom.T @ denom)).to(dev)  # (C, C)

    for H in horizons:
        # 视野自适应源数（文献 2609.32187 horizon-adaptive source selection；
        # PSM 实测：H96 k=12 最优、H336 k=4 最优 −32.8%）
        if args.k > 0:
            k_h = min(args.k, C)
        else:
            k_h = {96: 12, 192: 8, 336: 6, 720: 4}.get(H, 12)
            k_h = min(k_h, C)
        t0_, t1_ = ds.bounds["train"]
        Nw_tr = t1_ - t0_ - args.L - H + 1
        t0_, t1_ = ds.bounds["test"]
        Nw_te = t1_ - t0_ - args.L - H + 1
        lam_grid = (1.0, 10.0, 100.0, 1000.0)
        test_mse = np.zeros(C)
        # 每 split 的 Y 一次性物化（GPU，宽通道远快于逐通道流式）
        Ytr_all = y_matrix(ds, "train", args.L, H, C, dev)
        Yva_all = y_matrix(ds, "val", args.L, H, C, dev)
        Yte_all = y_matrix(ds, "test", args.L, H, C, dev)
        Nw_va = Yva_all.size(0)
        for c in range(C):
            srcs = torch.topk(corr[c].abs(), k_h).indices.tolist()
            if c not in srcs:
                srcs[0] = c
            cols = torch.cat([Xtr[s][:Nw_tr] for s in srcs], dim=1).to(dev)   # (Nw_tr, k*d)
            cols_va = torch.cat([Xva[s][:Nw_va] for s in srcs], dim=1).to(dev)
            Yva = Yva_all[:, :, c]
            Yc = Ytr_all[:, :, c]
            best = (None, 1e18)
            for lam in lam_grid:
                W = fit_ridge(cols, Yc, lam, dev)
                m = ((cols_va @ W - Yva) ** 2).mean().item()
                if m < best[1]:
                    best = (W, m)
            W = best[0]
            cols_te = torch.cat([Xte[s][:Nw_te] for s in srcs], dim=1).to(dev)
            Yte = Yte_all[:, :, c]
            test_mse[c] = ((cols_te @ W - Yte) ** 2).mean().item()
            if (c + 1) % max(1, C // 10) == 0:
                print(f"[crosschannel] H={H} channel {c + 1}/{C}", flush=True)
        print(f"H={H:>4}: cross-channel ridge MSE={test_mse.mean():.4f} "
              f"(per-channel mean, all {C} channels)")


if __name__ == "__main__":
    main()
