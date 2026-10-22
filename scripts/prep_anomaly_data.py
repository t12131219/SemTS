"""异常检测数据准备：Anomaly-Transformer 预处理格式 → 本项目 eval_anomaly.py 约定。

输出（--out 目录）：
    data.csv        数值通道 CSV（首列无时间戳，纯数值；SMD 为 38 通道，PSM 为 25 通道）
    label.npy       与 CSV 行对齐的二值异常标签（训练段全 0）

用法（先 gdown 下载，见 docs/experiments_plan.md）：
    python scripts/prep_anomaly_data.py --psm data/raw/anomaly/PSM/PSM --out data/PSM
    python scripts/prep_anomaly_data.py --smd data/raw/anomaly/SMD/SMD --out data/SMD
"""
import argparse
import os

import numpy as np
import pandas as pd


def _drop_meta(df):
    return df.drop(columns=[c for c in df.columns
                            if "timestamp" in c.lower() or "date" in c.lower() or c.lower() == "label"])


def prep_psm(src, out):
    tr = pd.read_csv(os.path.join(src, "train.csv"))
    te = pd.read_csv(os.path.join(src, "test.csv"))
    lab = pd.read_csv(os.path.join(src, "test_label.csv"))
    lab_col = "label" if "label" in lab.columns else lab.columns[-1]
    # 剔除时间戳/标签等元数据列，只保留数值通道
    tr = _drop_meta(tr)
    te = _drop_meta(te)
    values = pd.concat([tr, te], axis=0, ignore_index=True)
    labels = np.concatenate(
        [np.zeros(len(tr), dtype=np.int64), lab[lab_col].to_numpy().astype(np.int64)]
    )
    os.makedirs(out, exist_ok=True)
    values.to_csv(os.path.join(out, "data.csv"), index=False)
    np.save(os.path.join(out, "label.npy"), labels)
    print(f"[prep_anomaly] PSM: data={values.shape}, anomalies={labels.sum()} "
          f"({labels.sum()/len(labels):.2%})")


def prep_smd(src, out):
    tr = np.load(os.path.join(src, "SMD_train.npy"))
    te = np.load(os.path.join(src, "SMD_test.npy"))
    lab = np.load(os.path.join(src, "SMD_test_label.npy")).astype(np.int64).ravel()
    # 兼容 (T, C) 与 (T, C, ...) 布局
    if tr.ndim == 3:
        tr = tr.reshape(tr.shape[0], -1)
        te = te.reshape(te.shape[0], -1)
    assert tr.ndim == 2 and te.ndim == 2 and len(lab) == len(te), (
        f"形状异常: train={tr.shape} test={te.shape} label={lab.shape}")
    data = np.vstack([tr, te])
    labels = np.concatenate([np.zeros(len(tr), dtype=np.int64), lab])
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(data).to_csv(os.path.join(out, "data.csv"), index=False, header=False)
    np.save(os.path.join(out, "label.npy"), labels)
    print(f"[prep_anomaly] SMD: data={data.shape}, anomalies={labels.sum()} "
          f"({labels.sum()/len(labels):.2%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psm", type=str, default=None)
    ap.add_argument("--smd", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    if args.psm:
        prep_psm(args.psm, args.out)
    elif args.smd:
        prep_smd(args.smd, args.out)
    else:
        ap.error("需要 --psm 或 --smd")


if __name__ == "__main__":
    main()
