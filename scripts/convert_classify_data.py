"""分类数据转换：SimMTM .pt 或 UEA .ts → 本项目 finetune_classify.py 的 npy 约定。

输出布局（--out_dir）：
    train.npy / val.npy / test.npy        (N, L, C) float32
    train_label.npy / val_label.npy / test_label.npy  (N,) int64

用法：
    # SimMTM 格式（figshare 下载解压后，目录含 train.pt/val.pt/test.pt）
    python scripts/convert_classify_data.py --simmtm data/raw/SleepEEG --out data/SleepEEG
    # UEA 格式（timeseriesclassification.com 下载解压后的数据集目录，含 *TRAIN.ts / *TEST.ts）
    python scripts/convert_classify_data.py --uea data/raw/BasicMotions --out data/BasicMotions

说明：SimMTM 的 .pt 为 dict{samples:(N,C,L), labels:(N,)}；本脚本统一转置为 (N,L,C)。
"""
import argparse
import glob
import os

import numpy as np


def convert_simmtm(src_dir, out_dir):
    import torch  # 懒加载：UEA 模式只需 numpy（避免 GPU 训练占用期间加载 CUDA DLL 的内存压力）
    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        p = os.path.join(src_dir, f"{split}.pt")
        if not os.path.exists(p):
            print(f"[convert] 跳过缺失的 {p}")
            continue
        d = torch.load(p, map_location="cpu")
        x = d["samples"].numpy()          # (N, C, L) 或 (N, L, C)
        y = np.asarray(d["labels"]).ravel().astype(np.int64)
        if x.ndim != 3:
            x = x.reshape(x.shape[0], 1, -1)
        # SimMTM 数据 (N, C, L) → (N, L, C)
        if x.shape[1] <= x.shape[2] and x.shape.index(min(x.shape)) != 1:
            x = x.transpose(0, 2, 1)
        np.save(os.path.join(out_dir, f"{split}.npy"), x.astype(np.float32))
        np.save(os.path.join(out_dir, f"{split}_label.npy"), y)
        print(f"[convert] {split}: x={x.shape}, y={y.shape}")


def parse_uea_file(path):
    """UEA .ts：每行 'v1,v2,...,vT:label'（标签在行尾）；多变量以 ':' 分隔维度组；
    注释行以 '#' 或 '@' 开头。"""
    xs, ys, label_map = [], [], {}
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.startswith("#") and not ln.startswith("@")]
    for ln in lines:
        rest, label = ln.rsplit(":", 1)
        dims = []
        for tok in rest.split(","):
            tok = tok.strip()
            if not tok:
                continue
            for v in tok.split(":"):
                if v:
                    dims.append(float(v))
        xs.append(dims)
        label = label.strip()
        if label not in label_map:
            label_map[label] = len(label_map)
        ys.append(label_map[label])
    return xs, ys


def convert_uea(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tr_files = glob.glob(os.path.join(src_dir, "*TRAIN.ts"))
    te_files = glob.glob(os.path.join(src_dir, "*TEST.ts"))
    if not tr_files or not te_files:
        raise FileNotFoundError(f"{src_dir} 下未找到 *TRAIN.ts/*TEST.ts")
    xs, ys = parse_uea_file(tr_files[0])
    xte, yte = parse_uea_file(te_files[0])
    L = len(xs[0])
    if any(len(v) != L for v in xs) or any(len(v) != L for v in xte):
        raise ValueError("样本长度不一致")
    x = np.asarray(xs, np.float32).reshape(len(xs), L, 1)
    x_te = np.asarray(xte, np.float32).reshape(len(xte), L, 1)
    y = np.asarray(ys, np.int64)
    y_te = np.asarray(yte, np.int64)
    # 训练集内切 20% 作验证（UEA 无 val 划分）
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(x))
    n_val = int(len(x) * 0.2)
    idx_val, idx_tr = perm[:n_val], perm[n_val:]
    np.save(os.path.join(out_dir, "train.npy"), x[idx_tr])
    np.save(os.path.join(out_dir, "train_label.npy"), y[idx_tr])
    np.save(os.path.join(out_dir, "val.npy"), x[idx_val])
    np.save(os.path.join(out_dir, "val_label.npy"), y[idx_val])
    np.save(os.path.join(out_dir, "test.npy"), x_te)
    np.save(os.path.join(out_dir, "test_label.npy"), y_te)
    print(f"[convert] UEA: train={len(idx_tr)} val={n_val} test={len(x_te)}, L={L}, classes={len(set(y))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simmtm", type=str, default=None)
    ap.add_argument("--uea", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    if args.simmtm:
        convert_simmtm(args.simmtm, args.out)
    elif args.uea:
        convert_uea(args.uea, args.out)
    else:
        ap.error("需要 --simmtm 或 --uea")


if __name__ == "__main__":
    main()
