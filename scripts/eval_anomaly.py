"""异常检测评估（v0.1 基础协议）：重建误差评分 + 阈值 F1。

数据约定（后续按 SMD/SWaT/PSM 官方格式扩展）：
    --data    CSV，首列为时间戳（date/timestamp，自动丢弃），其余列为数值通道
    --labels  .npy，(T,) 二值 0/1 异常标签，与 CSV 行对齐
评分：逐时间戳重建误差（滑窗多窗口预测平均）→ 在训练段后 20% 扫描阈值 → 测试段 F1/P/R。
注：v0.1 不做 point-adjust（该协议后续按 SMD 惯例补充）。

用法:
    python scripts/eval_anomaly.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt \
        --data data/SMD.csv --labels data/SMD_label.npy --L 336
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import torch

from semts.model_factory import load_encoder_for_eval


@torch.no_grad()
def reconstruction_scores(model, data, L, dev, stride=8, batch=128):
    """data: (T, C) numpy → score: (T,) 每时间戳平均重建误差（多窗口预测平均）。

    窗口 [s, s+L) 的解码覆盖该窗口全部时间戳；相邻窗口 stride=8 交叠，
    每个时间戳的误差取覆盖它的所有窗口的平均。"""
    T, C = data.shape
    acc = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)
    starts = list(range(0, max(1, T - L + 1), stride))
    pl = model.cfg.patch_len
    st = model.cfg.stride
    model.eval()
    for i in range(0, len(starts), batch):
        idx = starts[i:i + batch]
        xs = torch.from_numpy(np.stack([data[s:s + L] for s in idx])).to(dev)  # (B, L, C)
        z = model.encode(xs)  # (B, C, P, d)
        vals, _ = model.stream4_head(z)  # (B, C, P, pl)
        P = vals.size(2)
        off = torch.arange(pl, device=dev).view(1, -1)
        ts = torch.arange(P, device=dev).view(P, 1) * st + off  # (P, pl) 窗口内时间戳
        # xs 为 (B, L, C)：先转置到 (B, C, L) 再按时间维索引（否则网格命中通道维）
        tgt = xs.permute(0, 2, 1)[:, :, ts]  # (B, C, P, pl)（ts 最大值恰为 L-1，无越界）
        err = ((vals - tgt) ** 2).mean(dim=1)  # (B, P, pl) 通道平均
        for b, s in enumerate(idx):
            for p in range(P):
                t = ts[p]
                acc[s + t.cpu().numpy()] += err[b, p].cpu().numpy()
                cnt[s + t.cpu().numpy()] += 1.0
    score = np.zeros(T)
    ok = cnt > 0
    score[ok] = acc[ok] / cnt[ok]
    return score


def best_f1(score, label):
    """分位数网格阈值扫描，返回 (best_th, f1, prec, rec)。"""
    pos = label > 0
    ths = np.quantile(score, np.linspace(0.5, 0.999, 200))
    best = (0.0, 0.0, 0.0, 0.0)
    for th in ths:
        pred = score > th
        tp = (pred & pos).sum()
        fp = (pred & ~pos).sum()
        fn = (~pred & pos).sum()
        if tp == 0:
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        if f1 > best[1]:
            best = (float(th), float(f1), float(p), float(r))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--val_ratio", type=float, default=0.2, help="训练段用于选阈值的比例")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_encoder_for_eval(args.ckpt, dev)
    cfg = model.cfg
    if not hasattr(model, "stream4_head"):
        raise ValueError("异常评估需要 SemTS 检查点（含 S4 质量头解码器）；基线检查点不支持")

    df = pd.read_csv(args.data)
    for c in ("date", "timestamp", "Date", "Timestamp"):
        if c in df.columns:
            df = df.drop(columns=[c])
    # PSM 等数据含 NaN：前向/后向填充 + 兜底置 0（与 SlidingWindowDataset 一致）
    df = df.ffill().bfill().fillna(0.0)
    data = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    label = np.load(args.labels).astype(np.int64).ravel()
    assert len(data) == len(label), f"长度不一致: {len(data)} vs {len(label)}"
    if len(data) < args.L:
        raise ValueError(f"序列长度 {len(data)} 小于窗口 L={args.L}")

    n_tr = int(len(data) * 0.6)  # 前 60% 视为训练段（与预训练切分一致）
    mu, sd = data[:n_tr].mean(0, keepdims=True), data[:n_tr].std(0, keepdims=True) + 1e-5
    data = (data - mu) / sd

    score = reconstruction_scores(model, data, args.L, dev)
    # 阈值在测试段内选（训练段无异常，无法提供正样本）：
    # 取测试段前 val_ratio 部分选阈值，其余部分报 F1
    n_te = len(data) - n_tr
    n_val = max(int(n_te * args.val_ratio), 1)
    seg_val = slice(n_tr, n_tr + n_val)
    seg_test = slice(n_tr + n_val, len(data))
    th, f1_v, p_v, r_v = best_f1(score[seg_val], label[seg_val])
    pred = score[seg_test] > th
    tp = (pred & (label[seg_test] > 0)).sum()
    fp = (pred & ~(label[seg_test] > 0)).sum()
    fn = (~pred & (label[seg_test] > 0)).sum()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    print(f"[eval_anomaly] 阈值段(测试前 {args.val_ratio:.0%}) F1={f1_v:.4f} (th={th:.4f}), "
          f"测试段(其余) F1={f1:.4f} P={p:.4f} R={r:.4f}")


if __name__ == "__main__":
    main()
