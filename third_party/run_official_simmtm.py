"""官方 SimMTM（thuml/SimMTM）适配器：预训练（官方框架 run.py）+ 特征导出 + 官方微调。

流程（在 SimMTM_Forecasting 目录内执行）：
  1. 预训练：官方 run.py（数据切分已按我们的 0.6/0.2/0.2 修改）
  2. 特征导出：本脚本 --export，用官方预训练 ckpt 提取每窗口 (N, C, d) 逐通道最后时间戳表示
  3. 评估：scripts/finetune_from_features.py（统一岭回归协议）
  4. 官方口径微调结果：官方 run.py --task_name finetune（其自报 MSE 也一并记录）

用法:
    python run_official_simmtm.py --export --data_path ../data/ETTm1.csv --pretrain_ckpt <ckpt>
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "SimMTM", "SimMTM_Forecasting")
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np
import torch

from semts.datasets.csv_dataset import SlidingWindowDataset


def build_configs(args):
    """构造官方 Model 所需的 configs 命名空间（与官方 ETTm1.sh 预训练配置一致）。"""
    cfg = argparse.Namespace()
    cfg.task_name = "pretrain"
    cfg.pred_len = 96
    cfg.seq_len = args.L
    cfg.label_len = 48
    cfg.output_attention = False
    cfg.d_model = 32       # 官方 ETT 配置（512 会导致池化头 59GB 权重）
    cfg.embed = "timeF"
    cfg.freq = "h"
    cfg.dropout = 0.0      # 官方 finetune 脚本显式 --dropout 0
    cfg.factor = 1
    cfg.n_heads = 8
    cfg.d_ff = 64          # 官方 ETT 配置
    cfg.activation = "gelu"
    cfg.e_layers = 2
    cfg.head_dropout = 0.0
    return cfg


def export_features(args):
    from models.SimMTM import Model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_configs(args)
    model = Model(cfg).to(dev)
    ckpt = torch.load(args.pretrain_ckpt, map_location=dev)
    model.load_state_dict(ckpt)
    model.eval()

    ds = SlidingWindowDataset(args.data, L=args.L)
    out_dir = os.path.join(args.out, "feats")
    os.makedirs(out_dir, exist_ok=True)

    @torch.no_grad()
    def enc(xb):
        """xb: (B, L, C) → (B, C, d_model)（官方 forecast 的归一化+嵌入+编码，止于编码器）。"""
        xb = xb.to(dev)
        bs, seq_len, n_vars = xb.shape
        means = xb.mean(1, keepdim=True)
        xb = xb - means
        stdev = torch.sqrt(torch.var(xb, dim=1, keepdim=True, unbiased=False) + 1e-5)
        xb = xb / stdev
        xb = xb.permute(0, 2, 1).reshape(-1, seq_len, 1)
        enc_out = model.enc_embedding(xb)
        enc_out, _ = model.encoder(enc_out)
        enc_out = enc_out.reshape(bs, n_vars, seq_len, -1)
        return enc_out[:, :, -1, :]  # (B, C, d_model) 最后时间戳

    for split in ("train", "val", "test"):
        x = ds.windows(split, stride=1)
        feats = []
        for i in range(0, len(x), 64):
            xb = torch.from_numpy(x[i:i + 64]).float()
            feats.append(enc(xb).cpu().numpy())
        feats = np.concatenate(feats, 0)
        np.save(os.path.join(out_dir, f"{split}.npy"), feats.astype(np.float32))
        print(f"[official_simmtm] {split} feats: {feats.shape}")
    print("[official_simmtm] done ->", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=os.path.join("..", "data", "ETTm1.csv"))
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--export", action="store_true", help="导出特征（需要 --pretrain_ckpt）")
    ap.add_argument("--pretrain_ckpt", type=str, default=None)
    ap.add_argument("--out", type=str, default=os.path.join("..", "experiments", "official_simmtm"))
    args = ap.parse_args()

    if args.export:
        if args.pretrain_ckpt is None:
            # 官方预训练输出默认位置
            args.pretrain_ckpt = os.path.join(
                SIM, "outputs", "pretrain_checkpoints", "ETTm1", "ckpt_best.pth"
            )
        export_features(args)
        print("next: python scripts/finetune_from_features.py --feats "
              f"{os.path.join(args.out, 'feats')} --data data/ETTm1.csv")
    else:
        print("用法：先官方预训练（见 run_official_all.bat），再 --export 导出特征")


if __name__ == "__main__":
    main()
