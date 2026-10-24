"""把 COMET 处理后 AD/PTB 数据转换为 finetune_classify.py 的 npy 格式。

患者独立划分与 COMET 论文一致（AD: val 17/18, test 19/20；
PTB: 负/正患者按文件顺序取后 14/7 与 42/21 为 val/test）。

用法:
    python scripts/convert_comet_classify.py --name AD --data_root data/shufflemtm/AD \
        --out data/shufflemtm/AD_classify
    python scripts/finetune_classify.py --ckpt experiments/ablation_w2/ckpt_ep9.pt \
        --data_dir data/shufflemtm/AD_classify [--fraction 0.2]
"""
import argparse
import os

import numpy as np


def load_comet(name, data_root):
    feat_dir = os.path.join(data_root, "Feature")
    label = np.load(os.path.join(data_root, "Label", "label.npy"))
    files = sorted(os.listdir(feat_dir))
    assert len(files) == len(label), f"{name}: feature 文件数 {len(files)} != label 行数 {len(label)}"
    if name == "AD":
        val_ids, test_ids = [17, 18], [19, 20]
    else:
        neg = list(np.where(label[:, 0] == 0)[0] + 1)
        pos = list(np.where(label[:, 0] == 1)[0] + 1)
        val_ids = neg[-14:-7] + pos[-42:-21]
        test_ids = neg[-7:] + pos[-21:]
    Xs, ys = {"train": [], "val": [], "test": []}, {"train": [], "val": [], "test": []}
    for j, fn in enumerate(files):
        subj = np.load(os.path.join(feat_dir, fn))
        lab = int(label[j, 0])
        split = "val" if (j + 1) in val_ids else ("test" if (j + 1) in test_ids else "train")
        for trial in subj:
            Xs[split].append(trial.astype(np.float32))
            ys[split].append(lab)
    out = {}
    for s in ("train", "val", "test"):
        out[s] = (np.stack(Xs[s]), np.array(ys[s]))
        print(f"{name} {s}: X{out[s][0].shape} y{out[s][1].shape} "
              f"pos={int((out[s][1] == 1).sum())}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, required=True, choices=["AD", "PTB"])
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load_comet(args.name, args.data_root)
    for split in ("train", "val", "test"):
        x, y = data[split]
        np.save(os.path.join(args.out, f"{split}.npy"), x)
        np.save(os.path.join(args.out, f"{split}_label.npy"), y)
    print(f"[convert_comet_classify] {args.name} -> {args.out}")


if __name__ == "__main__":
    main()
