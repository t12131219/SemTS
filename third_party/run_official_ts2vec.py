"""官方 TS2Vec（zhihanyue/ts2vec）训练 + 特征导出适配器。

统一协议：数据/切分/标准化与 SemTS 完全一致（SlidingWindowDataset），
训练用官方 fit()（n_iters 与 SemTS 10-epoch 步数对齐，默认 3240），
导出每个窗口"最后时间戳表示"（官方 encode），
评估用 scripts/finetune_from_features.py（与 SemTS 相同的岭回归协议）。

用法（在 SemTS 项目根目录）:
    python third_party/run_official_ts2vec.py --data data/ETTm1.csv --n_iters 3240
    python scripts/finetune_from_features.py --feats experiments/official_ts2vec/feats --data data/ETTm1.csv
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ts2vec"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np
import torch

from ts2vec import TS2Vec
from semts.datasets.csv_dataset import SlidingWindowDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=os.path.join("data", "ETTm1.csv"))
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--n_iters", type=int, default=3240,
                    help="迭代数（与 SemTS 10 epoch × 324 步对齐）")
    ap.add_argument("--batch_size", type=int, default=128, help="官方 ETT 协议 batch 128")
    ap.add_argument("--repr_dims", type=int, default=320, help="官方默认 320")
    ap.add_argument("--lr", type=float, default=1e-3, help="官方默认 1e-3")
    ap.add_argument("--out", type=str, default=os.path.join("experiments", "official_ts2vec"))
    ap.add_argument("--max_windows", type=int, default=None,
                    help="每 split 最多窗口数（冒烟用，控制内存；默认全部）")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    ds = SlidingWindowDataset(args.data, L=args.L)
    # 统一协议训练形态：切成 336 窗口多实例喂给官方 fit()（与官方 UCR/UEA 用法一致）。
    # 整序列 + 201 短段训练会使因果卷积表示的窗口末时间戳特征坍缩（MSE≈1.0 随机水平）
    train_series = ds.windows("train", stride=336)

    model = TS2Vec(
        input_dims=ds.n_channels,
        output_dims=args.repr_dims,
        hidden_dims=64,
        depth=10,
        device=dev,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_length=args.L,  # =336：与统一窗口协议匹配（201 短段训练会导致窗口末时间戳表示坍缩）
    )
    t0 = time.time()
    model.fit(train_series, n_iters=args.n_iters, verbose=True)
    print(f"[official_ts2vec] train done in {time.time() - t0:.0f}s")

    os.makedirs(os.path.join(args.out, "feats"), exist_ok=True)
    torch.save(
        {"model": model._net.state_dict(), "repr_dims": args.repr_dims,
         "n_channels": ds.n_channels, "L": args.L},
        os.path.join(args.out, "model.pt"),
    )

    model.net.eval()
    # 本机（4070 Laptop）CUDA 异步执行不稳定：导出阶段开启同步模式 + 小批量，避免原生崩溃
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    for split in ("train", "val", "test"):
        x = ds.windows(split, stride=1)
        if args.max_windows is not None:
            x = x[:args.max_windows]
        feats = []
        with torch.no_grad():
            for i in range(0, len(x), 16):
                xb = torch.from_numpy(x[i:i + 16].astype(np.float32))
                # 官方 encode 会把所有批次攒到最后才 cat，整段序列内存爆炸（本机页面文件小）；
                # 逐批提取末时间戳表示、立即转 numpy
                out = model._eval_with_pooling(xb, mask=None, encoding_window=None)  # (B,T,D) cpu
                feats.append(out[:, -1].numpy())
        feats = np.concatenate(feats, 0)
        np.save(os.path.join(args.out, "feats", f"{split}.npy"), feats.astype(np.float32))
        print(f"[official_ts2vec] {split} feats: {feats.shape}")

    print("[official_ts2vec] done ->", args.out)
    print("next: python scripts/finetune_from_features.py --feats "
          f"{os.path.join(args.out, 'feats')} --data {args.data}")


if __name__ == "__main__":
    main()
