"""PTB 原始数据（PhysioNet ptbdb 1.0.0）→ ShuffleMTM 论文口径处理数据（单心跳样本 300×15）。

流程（与 COMET 的 PTB_preprocessing.ipynb 一致，样本层级按 ShuffleMTM 论文 Table 6：
单心跳 300 点 × 15 通道，约 62370 样本）：
  1. 遍历患者目录（os.listdir 排序），.dat 记录 comments[4] 提取诊断；仅保留
     Myocardial infarction（label 1）与 Healthy control（label 0）→ 198 患者；
  2. 每 trial：每通道线性重采样 1000Hz→250Hz → StandardScaler（按通道）→
     neurokit2 R-peak 检测（ecg_clean + ecg_peaks, correct_artifacts=False）；
  3. 剔除 R 峰间隔中位数 > 300 的异常 trial；
  4. sample()：以跨通道中位 R 峰位置为中心取 ±med/2 心跳段，零填充到 300；
  5. feature_XXX.npy（N_beats, 300, 15）+ Label/label.npy（[label, id]，患者按入选顺序编号）。

用法: python scripts/prep_ptb_data.py --src data/raw/shufflemtm/ptbdb --out data/shufflemtm/PTB
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import wfdb
from scipy import interpolate
from sklearn.preprocessing import StandardScaler

import neurokit2 as nk

MAX_DURATION = 300
FS = 250


def resampling(array, freq=1000, kind="linear"):
    t = np.linspace(1, len(array), len(array))
    f = interpolate.interp1d(t, array, kind=kind)
    t_new = np.linspace(1, len(array), int(len(array) / freq * FS))
    return f(t_new)


def normalize(data):
    return StandardScaler().fit_transform(data)


def r_peaks(ecg_data):
    """返回 (intervals_median_series, df_peaks)（跨通道中位）。"""
    pos, trial_interval = [], []
    for ch in range(ecg_data.shape[1]):
        cleaned = nk.ecg_clean(ecg_data[:, ch], sampling_rate=FS, method="neurokit")
        signals, _ = nk.ecg_peaks(cleaned, sampling_rate=FS, correct_artifacts=False)
        peaks = signals[signals["ECG_R_Peaks"] == 1].index.to_list()
        pos.append(peaks)
        trial_interval.append([peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)])
    df_peaks = pd.DataFrame(pos)
    med = pd.DataFrame(trial_interval).T.median()
    return med, df_peaks


def sample_beats(ecg_data, max_duration=MAX_DURATION):
    med, df_peaks = r_peaks(ecg_data)
    trial_med = med.median()
    samples = []
    for i in range(df_peaks.shape[1]):
        rp = df_peaks.iloc[:, i].median()
        ini = ecg_data[max(0, int(rp) - int(trial_med / 2)):
                       min(int(rp) + int(trial_med / 2), ecg_data.shape[0]), :]
        left = int((max_duration - ini.shape[0]) / 2)
        beat = np.concatenate([np.zeros([left, ecg_data.shape[1]]), ini,
                               np.zeros([max_duration - left - ini.shape[0], ecg_data.shape[1]])], axis=0)
        samples.append(beat)
    return samples


def patient_label(rec):
    return wfdb.rdsamp(record_name=rec)[1]["comments"][4].split(":")[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="解压后的 ptbdb 目录（含患者子目录）")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "Feature"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "Label"), exist_ok=True)

    sub_dirs = sorted(d for d in glob.glob(os.path.join(args.src, "*")) if os.path.isdir(d))
    labels, sub_id, total = [], 1, 0
    t0 = time.time()
    for sub in sub_dirs:
        lab = None
        beats = []
        for tri in sorted(glob.glob(os.path.join(sub, "*.dat"))):
            rec = tri[:-4]
            diag = patient_label(rec)
            if diag not in ("Myocardial infarction", "Healthy control"):
                continue
            lab = 1 if diag == "Myocardial infarction" else 0
            ecg = wfdb.rdsamp(record_name=rec)[0]
            trial = np.array([resampling(ecg[:, ch]) for ch in range(ecg.shape[1])]).T
            trial = normalize(trial)
            try:
                med, _ = r_peaks(trial)
                if med.median() > MAX_DURATION:
                    continue
            except (IndexError, ValueError):
                continue
            beats.extend(sample_beats(trial))
        if lab is not None and beats:
            arr = np.stack(beats).astype(np.float32)  # (N, 300, 15)
            np.save(os.path.join(args.out, "Feature", f"feature_{sub_id:03d}.npy"), arr)
            labels.append([lab, sub_id])
            total += len(arr)
            print(f"feature_{sub_id:03d}.npy: {len(arr)} beats, label={lab} "
                  f"({time.time() - t0:.0f}s, 累计 {total})")
            sub_id += 1
    np.save(os.path.join(args.out, "Label", "label.npy"), np.array(labels))
    print(f"[prep_ptb_data] done: {sub_id - 1} 患者, {total} 样本 -> {args.out} "
          f"（论文口径 198 患者 / 62370 样本）")


if __name__ == "__main__":
    main()
