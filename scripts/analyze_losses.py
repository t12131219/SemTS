"""预训练损失日志分析：分段均值表 + 收敛诊断（训练完成后立即用）。

用法:
    python scripts/analyze_losses.py --log experiments/pretrain_ettm1/losses.csv [--window 100]
输出: 开头/中间/结尾三段各损失均值（对比四条流收敛情况与数量级平衡），
以及 total 的下降幅度。若日志为空则提示。
"""
import argparse
import csv
import os

import numpy as np

KEYS = ["total", "temp", "inst", "time", "amp", "phase", "lag", "dir", "recon", "qpred"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"[analyze_losses] 日志不存在: {args.log}")
        return
    rows = []
    with open(args.log, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(r[k]) for k in KEYS if k in r})
    if not rows:
        print("[analyze_losses] 日志为空")
        return

    n = len(rows)
    w = min(args.window, n // 3) if n >= 3 else n
    segs = {
        "开头": rows[:w],
        "中间": rows[n // 2 - w // 2: n // 2 + w // 2],
        "结尾": rows[-w:],
    }
    print(f"[analyze_losses] 共 {n} 步，分段均值（窗口 {w}）：")
    print("".join([f"{'loss':>10}"] + [f"{name:>14}" for name in segs]))
    for k in KEYS:
        if k not in rows[0]:
            continue
        vals = "".join(f"{np.mean([r[k] for r in seg]):>14.4f}" for seg in segs.values())
        print(f"{k:>10}{vals}")

    total_head = np.mean([r["total"] for r in rows[:w]])
    total_tail = np.mean([r["total"] for r in rows[-w:]])
    if total_head > 0:
        drop = 1 - total_tail / total_head
        print(f"\n[analyze_losses] total: {total_head:.4f} → {total_tail:.4f} "
              f"(下降 {drop:.1%})")
    # 各流数量级平衡提示（论文权重调试用）
    tail = {k: np.mean([r[k] for r in rows[-w:]]) for k in KEYS if k in rows[0] and k != "total"}
    print("[analyze_losses] 结尾各流数量级（调 w1..w4 权重参考）:")
    for k, v in sorted(tail.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>8}: {v:.4f}")


if __name__ == "__main__":
    main()
