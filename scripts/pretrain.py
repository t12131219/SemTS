"""SemTS 预训练脚本。

用法（CUDA，待 conda 环境就绪）:
    python scripts/pretrain.py --data data/ETTm1.csv --epochs 10 --batch_size 128
CPU 冒烟:
    python scripts/pretrain.py --data data/ETTm1.csv --max_batches 5 --batch_size 8
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from semts.config import SemTSConfig
from semts.datasets.csv_dataset import SlidingWindowDataset
from semts.semts import SemTS

LOSS_KEYS = ["total", "temp", "inst", "time", "amp", "phase", "lag", "dir", "recon", "qpred"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/ETTm1.csv")
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--max_channels", type=int, default=None,
                    help="通道截断（大通道数据集如 Traffic=862 时建议 ≤200，S1 对比按 C^2 开销）")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_batches", type=int, default=-1, help=">0 时限制总步数（CPU 冒烟用）")
    ap.add_argument("--out", type=str, default="experiments/pretrain_ettm1")
    ap.add_argument("--tag", type=str, default="", help="追加到 out 目录名的消融标记")
    # 消融/调参 CLI（None = 用 config 默认，免改代码）
    ap.add_argument("--w1", type=float, default=None, help="S1 层级对比权重")
    ap.add_argument("--w2", type=float, default=None, help="S2 模式头权重")
    ap.add_argument("--w3", type=float, default=None, help="S3 关系头权重")
    ap.add_argument("--w4", type=float, default=None, help="S4 质量头权重")
    ap.add_argument("--lam_dir", type=float, default=None, help="S3 方向损失系数")
    ap.add_argument("--lam_q", type=float, default=None, help="S4 质量预测系数")
    ap.add_argument("--tau_corr", type=float, default=None, help="S3 滞后显著性阈值")
    ap.add_argument("--mask_ratio", type=float, default=None)
    ap.add_argument("--noise_ratio", type=float, default=None)
    ap.add_argument("--tf32", action="store_true", help="开启 TF32（Ampere+ 如 3090，提速 ~30%）")
    ap.add_argument("--grad_checkpoint", action="store_true",
                    help="编码器层激活检查点（大通道数据集省显存 ~3×，训练略慢）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.tf32 and dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[pretrain] TF32 enabled")
    print(f"[pretrain] device={dev}")

    ds = SlidingWindowDataset(args.data, L=args.L, max_channels=args.max_channels)
    x_tr = torch.from_numpy(ds.windows("train", stride=1))
    loader = DataLoader(TensorDataset(x_tr), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, num_workers=0)

    cfg = SemTSConfig()
    cfg.L = args.L  # 滞后搜索上界随实际窗口长度推导
    cfg.grad_checkpoint = args.grad_checkpoint
    for k in ("w1", "w2", "w3", "w4", "lam_dir", "lam_q", "tau_corr", "mask_ratio", "noise_ratio"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.tag:
        args.out = f"{args.out}_{args.tag}"
    model = SemTS(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] channels={ds.n_channels}, params={n_params/1e6:.2f}M, "
          f"steps/epoch={len(loader)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(loader)
    sched = None
    if total_steps > 0:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "losses.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "epoch", "lr"] + LOSS_KEYS)

    step = 0
    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {ep}", ascii=True,
                    disable=(not sys.stderr.isatty()))
        for (xb,) in pbar:
            xb = xb.to(dev)
            opt.zero_grad()
            loss, d = model.forward_losses(xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            if step % 20 == 1 or 0 < args.max_batches:
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [step, ep, round(opt.param_groups[0]["lr"], 6)]
                        + [round(d[k].item(), 5) for k in LOSS_KEYS]
                    )
            # 非交互（重定向到日志）时每 20 步输出一行纯文本进度，保证日志可见
            if step % 20 == 0 and not sys.stderr.isatty():
                print(f"[pretrain] epoch={ep} step={step}/{total_steps} "
                      f"loss={d['total'].item():.3f} lag={d['lag'].item():.3f} "
                      f"recon={d['recon'].item():.3f}", flush=True)
            pbar.set_postfix(total=f"{d['total'].item():.3f}", lag=f"{d['lag'].item():.3f}")
            if 0 < args.max_batches <= step:
                break
        torch.save({"model": model.state_dict(), "cfg": cfg},
                   os.path.join(args.out, f"ckpt_ep{ep}.pt"))
        if 0 < args.max_batches <= step:
            break
    print(f"[pretrain] done -> {log_path}")


if __name__ == "__main__":
    main()
