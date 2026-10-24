"""AD 原始数据（OSF jbysn）→ COMET/ShuffleMTM 格式处理数据。

流程（与 COMET 论文一致）：
  1. 每个 preproctrialsXX.mat 的 data[0,0][2][0] = trials 列表，每个 (1280, 16)（转置后）；
  2. 每 trial 先用 StandardScaler 标准化（按通道）；
  3. 每 trial 切成 9 个半重叠 1 秒样本（256 点，步长 128）；
  4. Feature/feature_01..23.npy（N_samples, 256, 16）+ Label/label.npy（[label, id]，阳性 ID 同 COMET）。

用法: python scripts/prep_ad_data.py --src data/raw/shufflemtm/ad_raw/AD_Data --out data/shufflemtm/AD
"""
import argparse
import glob
import os

import numpy as np
import scipy.io as sio

AD_POSITIVE = [1, 3, 6, 8, 9, 11, 12, 13, 15, 17, 19, 21]
TRIAL_LEN, SAMPLE_LEN, STRIDE = 1280, 256, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="含 preproctrialsXX.mat 的目录")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "Feature"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "Label"), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.src, "*.mat")))
    assert len(files) == 23, f"期望 23 个 mat 文件，实际 {len(files)}"
    labels = np.zeros((23, 2))
    for i, path in enumerate(files):
        pid = i + 1
        mat = sio.loadmat(path)
        epochs = mat["data"][0, 0][2][0]  # trials 列表
        trials = np.stack([np.transpose(ep) for ep in epochs]).astype(np.float64)  # (N_r, 1280, 16)
        # per-trial 标准化（按通道 z-score，等价 StandardScaler）
        mu = trials.mean(axis=1, keepdims=True)
        sd = trials.std(axis=1, keepdims=True) + 1e-8
        trials = (trials - mu) / sd
        # 切 9 个半重叠样本
        segs = [trials[:, s:s + SAMPLE_LEN, :] for s in range(0, TRIAL_LEN - SAMPLE_LEN + 1, STRIDE)]
        assert len(segs) == 9
        samples = np.concatenate(segs, axis=0).astype(np.float32)  # (9N_r, 256, 16)
        np.save(os.path.join(args.out, "Feature", f"feature_{pid:02d}.npy"), samples)
        labels[i, 0] = 1 if pid in AD_POSITIVE else 0
        labels[i, 1] = pid
        print(f"feature_{pid:02d}.npy: {trials.shape[0]} trials -> {samples.shape[0]} samples, label={int(labels[i, 0])}")
    np.save(os.path.join(args.out, "Label", "label.npy"), labels)
    print(f"[prep_ad_data] done -> {args.out}（阳性 {int(labels[:, 0].sum())}/23）")


if __name__ == "__main__":
    main()
