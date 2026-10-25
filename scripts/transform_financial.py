"""金融/日频数据变换（改进方案 A4）：对数收益或一阶差分，供评估 A/B。

用法:
    python scripts/transform_financial.py --src data/shufflemtm/exchange_rate.csv \
        --dst data/shufflemtm/exchange_rate_logret.csv --mode logret
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--dst", type=str, required=True)
    ap.add_argument("--mode", type=str, default="logret", choices=["logret", "diff"])
    args = ap.parse_args()
    df = pd.read_csv(args.src)
    cols = [c for c in df.columns if c != "date"]
    vals = df[cols].values.astype(np.float64)
    if args.mode == "logret":
        trans = np.log(vals[1:] / np.maximum(vals[:-1], 1e-8))
    else:
        trans = np.diff(vals, axis=0)
    out = pd.DataFrame(trans, columns=cols)
    out.insert(0, "date", df["date"].values[1:])
    out.to_csv(args.dst, index=False)
    print(f"[transform_financial] {args.mode}: {df.shape} -> {out.shape} -> {args.dst}")


if __name__ == "__main__":
    main()
