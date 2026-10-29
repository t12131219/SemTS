"""软通道选择岭回归（改进方案 A3b，借鉴 DUET KDD'25 的 Channel-Soft-Clustering）。

与 A3 硬 top-k 源选择不同：每个目标通道 c 对全通道计算连续权重
    w_c = softmax(|corr[c,:]| / tau)，tau 由验证集网格选择；
聚合为单个"软源原型" P_c = Σ_s w_cs · X[s]（(Nw, d)），
岭回归特征 = [X[c], P_c]（2d 维，远低于硬选择的 k·d 维）→
高自由度（全通道参与、权重连续）+ 低维度（聚合消除共线性）。

用法:
    python scripts/finetune_forecast_softcc.py --ckpt experiments/semts_tables_Electricity/ckpt_ep9.pt \
        --data data/shufflemtm/electricity.csv --horizons 96
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
from finetune_forecast_crosschannel import y_count, y_for_channel, y_matrix


def fit_ridge(X, Y, lam, dev):
    d = X.size(1)
    A = X.t() @ X + lam * torch.eye(d, device=dev)
    return torch.linalg.solve(A, X.t() @ Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--horizons", type=str, default="96,192,336,720")
    ap.add_argument("--taus", type=str, default="0.05,0.1,0.2,0.5",
                    help="软权重温度网格（越小越接近硬选择）")
    ap.add_argument("--self_weight", type=float, default=1.0,
                    help="目标通道自身表示相对软源原型的权重（>1 强调自身历史）")
    ap.add_argument("--topk_sparse", type=int, default=32,
                    help="软权重仅保留 |corr| 最大的 top-k 个源（有界支持，宽通道提速 O(C²)→O(kC)）")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    model = load_encoder_for_eval(args.ckpt, dev)
    ds = SlidingWindowDataset(args.data, L=args.L)
    C = ds.n_channels
    horizons = [int(h) for h in args.horizons.split(",")]
    print(f"[softcc] C={C} device={dev}")

    Xtr = extract_features(model, ds, "train", args.L, dev)
    Xva = extract_features(model, ds, "val", args.L, dev)
    Xte = extract_features(model, ds, "test", args.L, dev)
    d = Xtr.size(1)
    Xtr = [Xtr[c::C].contiguous() for c in range(C)]
    Xva = [Xva[c::C].contiguous() for c in range(C)]
    Xte = [Xte[c::C].contiguous() for c in range(C)]

    # 通道相关性（窗口均值，CPU 分块）
    means = []
    for x, _ in window_iter(ds, "train", args.L, None, chunk=512):
        means.append(x.mean(axis=1))
    m_tr = np.concatenate(means, 0)
    m_tr = m_tr - np.mean(m_tr, axis=0, keepdims=True)
    denom = np.sqrt(np.sum(m_tr * m_tr, axis=0, keepdims=True)) + 1e-8
    corr = (m_tr.T @ m_tr) / (denom.T @ denom)          # (C, C)
    corr_abs = np.abs(corr)

    taus = [float(t) for t in args.taus.split(",")]
    lam_grid = (1.0, 10.0, 100.0, 1000.0)
    for H in horizons:
        Nw_tr = y_count(ds, "train", args.L, H)
        Nw_va = y_count(ds, "val", args.L, H)
        Nw_te = y_count(ds, "test", args.L, H)
        # 内存自适应：全量 y 矩阵 ≤ 2GB 用快路径，否则逐通道流式（小显存机器长视野安全）
        big = (Nw_tr + Nw_va + Nw_te) * H * C * 4 > 2e9
        if not big:
            Ytr_all = y_matrix(ds, "train", args.L, H, C, dev)
            Yva_all = y_matrix(ds, "val", args.L, H, C, dev)
            Yte_all = y_matrix(ds, "test", args.L, H, C, dev)
        test_mse = np.zeros(C)
        for c in range(C):
            Xc_tr = Xtr[c][:Nw_tr].to(dev)
            Xc_va = Xva[c][:Nw_va].to(dev)
            Xc_te = Xte[c][:Nw_te].to(dev)
            if big:
                Yc = y_for_channel(ds, "train", args.L, H, c, dev)
                Yva = y_for_channel(ds, "val", args.L, H, c, dev)
                Yte = y_for_channel(ds, "test", args.L, H, c, dev)
            else:
                Yc = Ytr_all[:, :, c]
                Yva = Yva_all[:, :, c]
                Yte = Yte_all[:, :, c]
            best = (None, 1e18)
            # 软源原型（按 tau 网格；权重先稀疏化到 top-k 源，宽通道 O(kC) 而非 O(C²)）
            topk = np.argsort(-corr_abs[c])[:min(args.topk_sparse, C - 1)]
            for tau in taus:
                w = np.exp(corr_abs[c] / tau)            # (C,)
                w[c] = 0.0                               # 排除自身
                keep = np.zeros(C, dtype=bool)
                keep[topk] = True
                w = np.where(keep, w, 0.0)
                w = w / max(w.sum(), 1e-12)
                w_t = torch.from_numpy(w).float()        # CPU：与 Xtr[s]（CPU）同设备加权
                srcs = topk.tolist()
                P_tr = sum(w_t[s] * Xtr[s][:Nw_tr] for s in srcs).to(dev)  # (Nw, d)
                P_va = sum(w_t[s] * Xva[s][:Nw_va] for s in srcs).to(dev)
                P_te = sum(w_t[s] * Xte[s][:Nw_te] for s in srcs).to(dev)
                cols = torch.cat([args.self_weight * Xc_tr, P_tr], dim=1)
                cols_va = torch.cat([args.self_weight * Xc_va, P_va], dim=1)
                cols_te = torch.cat([args.self_weight * Xc_te, P_te], dim=1)
                for lam in lam_grid:
                    W = fit_ridge(cols, Yc, lam, dev)
                    m = ((cols_va @ W - Yva) ** 2).mean().item()
                    if m < best[1]:
                        best = (W, m, cols_te)
            mse = ((best[2] @ best[0] - Yte) ** 2).mean().item()
            test_mse[c] = mse
            if (c + 1) % max(1, C // 10) == 0:
                print(f"[softcc] H={H} channel {c + 1}/{C}", flush=True)
        print(f"H={H:>4}: soft-channel ridge MSE={test_mse.mean():.4f} "
              f"(per-channel mean, all {C} channels)")


if __name__ == "__main__":
    main()
