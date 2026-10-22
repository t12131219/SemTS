"""官方 ShuffleMTM（cikm2025shufflemtm/ShuffleMTM）适配器：预训练（官方 run.py）+ 特征导出。

统一协议：数据切分已 patch 为 0.6/0.2/0.2；预训练用官方框架与官方 ETT 配置
（e_layers=3, d_model=32, d_ff=128, n_heads=16, patch_len=6, stride=6, lr 5e-4, seed 2024），
seq_len 对齐 336 窗口；特征导出 = 官方 backbone 每窗口最后 patch 的逐通道表示 (N, C, d)；
评估用 scripts/finetune_from_features.py（统一岭回归协议）。

用法（在 SemTS 项目根目录）:
    先由 run_official_all.bat 调官方 run.py 预训练，再执行：
    python third_party/run_official_shufflemtm.py --export --data data/ETTm1.csv
    python scripts/finetune_from_features.py --feats experiments/official_shufflemtm/feats --data data/ETTm1.csv
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHUF = os.path.join(HERE, "ShuffleMTM")
sys.path.insert(0, SHUF)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np
import torch

from semts.datasets.csv_dataset import SlidingWindowDataset


def build_model(L, C):
    """按官方 ETT 配置（etth1.sh）构造 head_type='pretrain' 的完整模型。"""
    from models.shufflemtm import Model
    patch_len, stride = 6, 6
    num_patch = (L - patch_len) // stride + 1
    return Model(
        c_in=C, target_dim=0,
        patch_len=patch_len, stride=stride, num_patch=num_patch,
        e_layers=3, d_layers=1, n_heads=16, d_model=32,
        shared_embedding=True, d_ff=128,
        dropout=0.2, head_dropout=0.2,
        act='relu', head_type='pretrain', res_attention=False,
    )


def export_features(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SlidingWindowDataset(args.data, L=args.L)
    model = build_model(args.L, ds.n_channels).to(dev)
    ckpt = torch.load(args.pretrain_ckpt, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = os.path.join(args.out, "feats")
    os.makedirs(out_dir, exist_ok=True)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 本机 CUDA 异步不稳定（见 ts2vec/CoST 适配器）

    @torch.no_grad()
    def enc(xb):
        # patchify: (B, L, C) -> (B, num_patch, C, patch_len)
        x = xb.unfold(dimension=1, size=6, step=6).to(dev)
        z = model.backbone(x)          # (B, C, d_model, num_patch)
        return z[:, :, :, -1]          # (B, C, d_model) 最后 patch

    for split in ("train", "val", "test"):
        x = ds.windows(split, stride=1)
        if args.max_windows is not None:
            x = x[:args.max_windows]
        feats = []
        for i in range(0, len(x), 32):
            xb = torch.from_numpy(x[i:i + 32]).float()
            feats.append(enc(xb).cpu().numpy())
        feats = np.concatenate(feats, 0)
        np.save(os.path.join(out_dir, f"{split}.npy"), feats.astype(np.float32))
        print(f"[official_shufflemtm] {split} feats: {feats.shape}")
    print("[official_shufflemtm] done ->", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=os.path.join("data", "ETTm1.csv"))
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--pretrain_ckpt", type=str, default=None)
    ap.add_argument("--max_windows", type=int, default=None,
                    help="每 split 最多窗口数（冒烟用，控制内存；默认全部）")
    ap.add_argument("--out", type=str, default=os.path.join("experiments", "official_shufflemtm"))
    args = ap.parse_args()

    if not args.export:
        print("用法：先官方预训练（见 run_official_all.bat），再 --export 导出特征")
        return
    if args.pretrain_ckpt is None:
        # 官方预训练输出：outputs/pretrain_checkpoints/ETTm1/<setting>/2024/ckpt_best.pth
        cands = sorted(glob.glob(os.path.join(
            SHUF, "outputs", "pretrain_checkpoints", "ETTm1", "*", "2024", "ckpt_best.pth")))
        if not cands:
            raise FileNotFoundError("未找到官方预训练 ckpt_best.pth（先跑 run_official_all.bat 的预训练步骤）")
        args.pretrain_ckpt = cands[-1]
        print(f"[official_shufflemtm] use ckpt: {args.pretrain_ckpt}")
    export_features(args)
    print("next: python scripts/finetune_from_features.py --feats "
          f"{os.path.join(args.out, 'feats')} --data {args.data}")


if __name__ == "__main__":
    main()
