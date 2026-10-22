"""语义探针实验：验证四条语义流各自学到了对应语义（论文第二层证据，全部自包含）。

probe_s1_shuffle : ①时间/顺序 —— 原始 vs 乱序序列的表示线性可分度（需 --data CSV）
probe_s2_pattern : ④模式 —— TFD/SFD 输出对趋势/季节分量的回归 R² 分离度（合成数据）
probe_s3_lag     : ⑤关系 —— 冻结表示 [h_i;h_j] 线性探针预测滞后 bin（合成滞后对）
probe_s4_quality : ⑦质量 —— patch 表示线性探针预测噪声等级 q（合成数据）

用法:
    python scripts/probes.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt --probe s3_lag
    python scripts/probes.py --ckpt <ckpt> --probe s1_shuffle --data data/ETTm1.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

from semts.augs import add_state_quality_emb, three_state_mask
from semts.datasets.csv_dataset import SlidingWindowDataset
from semts.model_factory import load_encoder_for_eval


def _ridge_r2(X, y, Xt, yt, lam=1.0, dev="cpu"):
    """线性回归 + R²。X/Xt: (N, d)，y/yt: (N,)（接受 tensor 或 numpy）。"""
    X = torch.as_tensor(X, dtype=torch.float32).to(dev)
    y = torch.as_tensor(y, dtype=torch.float32).to(dev)
    Xt = torch.as_tensor(Xt, dtype=torch.float32).to(dev)
    yt = torch.as_tensor(yt, dtype=torch.float32).to(dev)
    d = X.size(1)
    A = X.t() @ X + lam * torch.eye(d, device=dev)
    W = torch.linalg.solve(A, X.t() @ y)
    pred = Xt.to(dev) @ W
    ss_res = ((pred - yt.to(dev)) ** 2).sum()
    ss_tot = ((yt.to(dev) - yt.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


@torch.no_grad()
def _encode(model, x, dev, batch=256):
    outs = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(dev)
        outs.append(model.encode(xb).cpu())  # (B, C, P, d)
    return torch.cat(outs, 0)


# ---------------------------------------------------------------- S1 乱序探针
def probe_s1_shuffle(model, args):
    ds = SlidingWindowDataset(args.data, L=args.L, max_channels=args.max_channels)
    x = ds.windows("train", stride=4)[: args.n_samples]
    N, L, C = x.shape
    block = max(2, model.cfg.patch_len * 2)
    n_blk = L // block
    x_shuf = np.zeros_like(x)
    rng = np.random.default_rng(0)
    for i in range(N):
        perm = rng.permutation(n_blk)
        for b in range(n_blk):
            x_shuf[i, b * block:(b + 1) * block] = x[i, perm[b] * block:(perm[b] + 1) * block]
    z1 = _encode(model, x, args.dev)          # (N, C, P, d)
    z2 = _encode(model, x_shuf, args.dev)
    r1 = z1.mean(dim=(1, 2))  # (N, d)
    r2 = z2.mean(dim=(1, 2))
    X = torch.cat([r1, r2], 0)
    y = torch.cat([torch.zeros(N), torch.ones(N)], 0)
    perm = torch.randperm(2 * N)
    n_tr = int(2 * N * 0.7)
    Xtr, ytr = X[perm[:n_tr]], y[perm[:n_tr]]
    Xte, yte = X[perm[n_tr:]], y[perm[n_tr:]]
    d = X.size(1)
    A = Xtr.t() @ Xtr + torch.eye(d) * 0.1
    W = torch.linalg.solve(A, Xtr.t() @ ytr)
    acc = ((Xte @ W > 0.5) == yte).float().mean().item()
    print(f"[probe_s1_shuffle] ordered-vs-shuffled 线性可分 acc={acc:.4f} "
          f"(>0.8 视为顺序语义可读)")


# ---------------------------------------------------------------- S2 趋势/季节探针
def probe_s2_pattern(model, args):
    N, L, C = args.n_samples, args.L, 1
    t = np.arange(L)[None, :, None] / L  # (1, L, 1) 归一化时间
    rng = np.random.default_rng(1)
    # 逐样本随机斜率与相位，避免目标退化为常数
    slope = rng.uniform(0.2, 0.8, (N, 1, 1))
    ph_tr = rng.uniform(0, 2 * np.pi, (N, 1, 1))
    ph_se = rng.uniform(0, 2 * np.pi, (N, 1, 1))
    trend = slope * t + 0.3 * np.sin(2 * np.pi * t / 3 + ph_tr)
    season = 0.6 * np.sin(2 * np.pi * t * 8 + ph_se)
    noise = rng.normal(0, 0.1, (N, L, C))
    x = (trend + season + noise).astype(np.float32)
    z = _encode(model, x, args.dev)  # (N, C, P, d)
    # 趋势/季节表示（复用 S2 的解耦器，权重冻结）
    z_dev = z.to(args.dev)
    t_rep = model.stream2.tfd(z_dev).mean(2).reshape(N, -1).cpu()      # (N, dT)
    s_rep = model.stream2.sfd(z_dev, n_time=z_dev.size(2)).mean(2).reshape(N, -1).cpu()  # (N, dS)
    y_tr = trend[:, -1, 0]   # (N,) 末点趋势值（逐样本不同）
    y_se = season[:, -1, 0]  # (N,) 末点季节值
    n_tr = int(N * 0.7)
    print(f"[probe_s2_pattern] trend_rep R2(trend)={_ridge_r2(t_rep[:n_tr], y_tr[:n_tr], t_rep[n_tr:], y_tr[n_tr:], dev=args.dev):.3f} "
          f"R2(season)={_ridge_r2(t_rep[:n_tr], y_se[:n_tr], t_rep[n_tr:], y_se[n_tr:], dev=args.dev):.3f}")
    print(f"[probe_s2_pattern] season_rep R2(trend)={_ridge_r2(s_rep[:n_tr], y_tr[:n_tr], s_rep[n_tr:], y_tr[n_tr:], dev=args.dev):.3f} "
          f"R2(season)={_ridge_r2(s_rep[:n_tr], y_se[:n_tr], s_rep[n_tr:], y_se[n_tr:], dev=args.dev):.3f}")
    print("[probe_s2_pattern] expected: trend rep higher R2 on trend, season rep higher R2 on season")


# ---------------------------------------------------------------- S3 滞后探针
def probe_s3_lag(model, args):
    N, L = args.n_samples, args.L
    rng = np.random.default_rng(2)
    base = np.cumsum(rng.normal(0, 1, (N, L)), axis=1)  # 随机游走基底
    ks = rng.integers(-24, 25, N)
    ks[np.abs(ks) < 3] = 3
    x2 = np.zeros_like(base)
    for i in range(N):
        k = ks[i]
        if k > 0:
            x2[i, k:] = base[i, :-k]
        else:
            x2[i, :L + k] = base[i, -k:]
    x = np.stack([base, x2], axis=-1).astype(np.float32)  # (N, L, 2)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-5)
    z = _encode(model, x, args.dev)  # (N, 2, P, d)
    h = z.mean(2)                    # (N, 2, d)
    feat = torch.cat([h[:, 0], h[:, 1], h[:, 0] * h[:, 1]], -1)  # (N, 3d)
    bins = ((ks + 24) // 8).astype(np.int64)  # 6 bins（与 bin_width=8 一致）
    K = int(bins.max()) + 1
    n_tr = int(N * 0.7)
    W = torch.zeros(feat.size(1), K)
    A = feat[:n_tr].t() @ feat[:n_tr] + torch.eye(feat.size(1)) * 0.1
    b = feat[:n_tr].t() @ torch.nn.functional.one_hot(torch.from_numpy(bins[:n_tr]), K).float()
    W = torch.linalg.solve(A, b)
    pred = (feat[n_tr:] @ W).argmax(1)
    acc = (pred == torch.from_numpy(bins[n_tr:])).float().mean().item()
    print(f"[probe_s3_lag] 冻结表示滞后 bin 分类 acc={acc:.4f} "
          f"(随机基线≈{1/K:.3f}，>0.5 视为滞后语义可读)")


# ---------------------------------------------------------------- S4 质量探针
def probe_s4_quality(model, args):
    N, L, C = args.n_samples, args.L, 1
    rng = np.random.default_rng(3)
    x = (np.sin(np.arange(L)[None, :, None] / 3) + rng.normal(0, 0.2, (N, L, C))).astype(np.float32)
    xt = torch.from_numpy(x).to(args.dev)
    x_aug, state, q = three_state_mask(xt, mask_ratio=0.0, noise_ratio=0.3)
    # 关键设计：只喂带噪数值，不喂状态/质量嵌入 —— 探针测的是"编码器权重本身
    # 学到的质量语义"，而非输入嵌入捷径（喂嵌入则线性探针可平凡读出等级）。
    z = model.encode(x_aug)  # (N, C, P, d)
    P = z.size(2)
    labels = []
    for p in range(P):
        t = np.arange(p * model.cfg.stride, min(p * model.cfg.stride + model.cfg.patch_len, L))
        qp = q[:, t, 0]          # (N, k)
        sp = state[:, t, 0] == 2  # (N, k)
        cnt = torch.zeros(N, model.cfg.n_q, device=q.device)
        for ql in range(model.cfg.n_q):
            cnt[:, ql] = ((qp == ql) & sp).sum(-1)  # 只统计 N 点，one-hot 计数
        best = cnt.argmax(-1)     # (N,)
        has = sp.any(-1)
        labels.append(torch.where(has, best, torch.full_like(best, -1)))
    labels = torch.stack(labels, dim=1)  # (N, P)
    # 逐 patch 粒度线性探针：特征与标签统一到 (N*P) 行，避免样本级/逐 patch 粒度错位
    feat = z[:, 0].reshape(N * P, -1)    # (N*P, d)，C=1
    lab_flat = labels.reshape(-1)        # (N*P,)
    valid = lab_flat >= 0
    feat, lab_flat = feat[valid], lab_flat[valid]
    K = model.cfg.n_q
    n_tr = int(len(feat) * 0.7)
    A = feat[:n_tr].t() @ feat[:n_tr] + torch.eye(feat.size(1), device=feat.device) * 0.1
    b = torch.zeros(feat.size(1), K, device=feat.device)
    for k in range(K):
        sel = lab_flat[:n_tr] == k
        if sel.any():
            b[:, k] = feat[:n_tr][sel].sum(0)
    W = torch.linalg.solve(A, b)
    pred = (feat[n_tr:] @ W).argmax(1)
    acc = (pred == lab_flat[n_tr:]).float().mean().item()
    print(f"[probe_s4_quality] 质量等级线性可分 acc={acc:.4f} (随机基线=0.25，>0.6 视为质量语义可读)")


PROBES = {
    "s1_shuffle": probe_s1_shuffle,
    "s2_pattern": probe_s2_pattern,
    "s3_lag": probe_s3_lag,
    "s4_quality": probe_s4_quality,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--probe", type=str, required=True, choices=sorted(PROBES))
    ap.add_argument("--data", type=str, default=None, help="s1_shuffle 需要")
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--max_channels", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=4000)
    args = ap.parse_args()
    args.dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_encoder_for_eval(args.ckpt, args.dev)
    if args.probe in ("s2_pattern", "s4_quality") and not hasattr(model, "stream2"):
        print(f"[probes] {args.probe} 需要 SemTS 检查点（含 S2/S4 头），当前检查点不支持")
        return
    PROBES[args.probe](model, args)


if __name__ == "__main__":
    main()
