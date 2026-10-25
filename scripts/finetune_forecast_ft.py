"""预测微调协议（改进方案 A2）：冻结 LP + 全量 FT + 通道共享线性头。

文献依据：arXiv:2605.19462 —— SSL 在预测任务上的收益主要体现于微调协议，
线性探针不能代表微调表现；本脚本与 ShuffleMTM 官方协议对齐
（freeze 线性探针 n 步 + 全量微调 2n 步）。

模型：冻结/微调 SemTS 编码器 → 每通道末 patch 表示 (B, C, d) → 共享线性头
Linear(d, H) 逐通道预测未来 H 步（z-score 空间 MSE）。

用法:
    python scripts/finetune_forecast_ft.py --ckpt experiments/ablation_w2/ckpt_ep9.pt \
        --data data/ETTm1.csv --horizons 96 --lp 10 --ft 20
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from semts.datasets.csv_dataset import SlidingWindowDataset
from semts.model_factory import load_encoder_for_eval
from finetune_forecast import window_iter


class ForecastHead(nn.Module):
    def __init__(self, d, H):
        super().__init__()
        self.fc = nn.Linear(d, H)

    def forward(self, z):  # z: (B, C, d) 末 patch 表示
        return self.fc(z)  # (B, C, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--horizons", type=str, default="96,192,336,720")
    ap.add_argument("--lp", type=int, default=10, help="冻结线性探针 epoch 数")
    ap.add_argument("--ft", type=int, default=20, help="全量微调 epoch 数")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--revin", action="store_true",
                    help="逐窗口 RevIN（与微调协议配套；预测反归一化回原空间）")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    model = load_encoder_for_eval(args.ckpt, dev)
    d = model.cfg.d_model
    ds = SlidingWindowDataset(args.data, L=args.L)

    def loader(split, H):
        xs, ys = [], []
        for x, y in window_iter(ds, split, args.L, H, chunk=1024):
            xs.append(x)
            ys.append(y)
        return DataLoader(TensorDataset(torch.from_numpy(np.concatenate(xs)),
                                        torch.from_numpy(np.concatenate(ys))),
                          batch_size=args.batch_size, shuffle=(split == "train"))

    for H in [int(h) for h in args.horizons.split(",")]:
        t0 = time.time()
        tr = loader("train", H)
        va = loader("val", H)
        te = loader("test", H)
        head = ForecastHead(d, H).to(dev)

        def prep(xb, yb):
            """返回 (x_norm, y_perm, mu, sd)：revin 时按窗口统计量归一化。"""
            xb = xb.to(dev)
            mu = sd = None
            if args.revin:
                mu = xb.mean(1, keepdim=True)                       # (B, 1, C)
                sd = torch.sqrt(torch.var(xb, dim=1, keepdim=True, unbiased=False) + 1e-5)
                xb = (xb - mu) / sd
            y = yb.permute(0, 2, 1).to(dev)                          # (B, C, H)
            if args.revin:
                y = (y - mu.transpose(1, 2)) / sd.transpose(1, 2)
            return xb, y, mu, sd

        # 阶段 1：冻结编码器，只训头
        for p in model.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        ce = nn.MSELoss()
        best = (1e18, None)
        for ep in range(args.lp):
            head.train()
            for xb, yb in tr:
                xb, y, mu, sd = prep(xb, yb)
                z = model.encode(xb)[:, :, -1, :]      # (B, C, d)
                loss = ce(head(z), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            m = eval_mse(model, head, va, dev, args.revin)
            if m < best[0]:
                best = (m, {k: v.clone() for k, v in head.state_dict().items()})
        head.load_state_dict(best[1])
        # 阶段 2：全量微调
        for p in model.parameters():
            p.requires_grad = True
        opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr / 5)
        best = (1e18, None)
        for ep in range(args.ft):
            model.train()
            for xb, yb in tr:
                xb, y, mu, sd = prep(xb, yb)
                z = model.encode(xb)[:, :, -1, :]
                loss = ce(head(z), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            m = eval_mse(model, head, va, dev, args.revin)
            if m < best[0]:
                best = (m, {k: v.clone() for k, v in
                            list(model.state_dict().items()) + list(head.state_dict().items())})
        # 载入最佳（head 已含）
        mse, mae = eval_metrics(model, head, te, dev, args.revin)
        print(f"H={H:>4}: FT-MSE={mse:.4f} MAE={mae:.4f} (best val {best[0]:.4f}, "
              f"{time.time() - t0:.0f}s)")


@torch.no_grad()
def eval_metrics(model, head, loader, dev, revin=False):
    sse = sae = 0.0
    n = 0
    model.eval()
    head.eval()
    for xb, yb in loader:
        xb = xb.to(dev)
        mu = sd = None
        if revin:
            mu = xb.mean(1, keepdim=True)
            sd = torch.sqrt(torch.var(xb, dim=1, keepdim=True, unbiased=False) + 1e-5)
            xb = (xb - mu) / sd
        z = model.encode(xb)[:, :, -1, :]
        pred = head(z)                                     # (B, C, H) 归一化空间
        y = yb.permute(0, 2, 1).to(dev)
        if revin:                                          # 反归一化回原空间（可比口径）
            pred = pred * sd.transpose(1, 2) + mu.transpose(1, 2)
        e = pred - y
        sse += (e * e).sum().item()
        sae += e.abs().sum().item()
        n += e.numel()
    return sse / n, sae / n


def eval_mse(model, head, loader, dev, revin=False):
    mse, _ = eval_metrics(model, head, loader, dev, revin)
    return mse


if __name__ == "__main__":
    main()
