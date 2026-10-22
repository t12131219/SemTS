"""统一骨干基线预训练（M4：TS2Vec / CoST / SimMTM / TF-C 公平对比）。

用法:
    python scripts/pretrain_baselines.py --baseline ts2vec --data data/ETTm1.csv \
        --epochs 10 --batch_size 128
检查点: experiments/pretrain_<baseline>/ckpt_ep{epochs-1}.pt
        {"encoder": state_dict, "cfg": cfg, "baseline": name, "extra": 可选头权重}
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from semts.baselines import BASELINES, EncoderOnly, _SimMTMDecoder
from semts.config import SemTSConfig
from semts.datasets.csv_dataset import SlidingWindowDataset
from semts.stream2 import Stream2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, required=True, choices=sorted(BASELINES))
    ap.add_argument("--data", type=str, default="data/ETTm1.csv")
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--max_channels", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_batches", type=int, default=-1)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--tf32", action="store_true", help="开启 TF32（Ampere+ 如 3090，提速 ~30%）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.tf32 and dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[pretrain_baselines] TF32 enabled")
    print(f"[pretrain_baselines] baseline={args.baseline}, device={dev}")

    ds = SlidingWindowDataset(args.data, L=args.L, max_channels=args.max_channels)
    x_tr = torch.from_numpy(ds.windows("train", stride=1))
    loader = DataLoader(TensorDataset(x_tr), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, num_workers=0)

    cfg = SemTSConfig()
    model = EncoderOnly(cfg).to(dev)
    extra = {}
    if args.baseline == "cost":
        extra["stream2"] = Stream2(
            cfg.d_model, d_trend=cfg.d_trend, d_season=cfg.d_season,
            n_experts=cfg.n_experts, n_freqs=cfg.n_freqs,
            queue_size=cfg.queue_size, alpha=cfg.alpha, tau=cfg.tau, freq_tau=cfg.tau,
        ).to(dev)
    if args.baseline == "simmtm":
        extra["decoder"] = _SimMTMDecoder(cfg.d_model, cfg.patch_len).to(dev)

    params = list(model.parameters()) + [p for m in extra.values() for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(loader)
    sched = None
    if total_steps > 0:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)

    out_dir = args.out or os.path.join("experiments", f"pretrain_{args.baseline}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "losses.csv")

    step = 0
    keys = []
    header_written = False
    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {ep}", ascii=True,
                    disable=(not sys.stderr.isatty()))
        for (xb,) in pbar:
            xb = xb.to(dev)
            opt.zero_grad()
            if args.baseline == "cost":
                loss, d = BASELINES[args.baseline](model.encoder, xb, cfg, extra["stream2"])
            elif args.baseline == "simmtm":
                loss, d = BASELINES[args.baseline](model.encoder, extra["decoder"], xb, cfg)
            else:
                loss, d = BASELINES[args.baseline](model.encoder, xb, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            if not header_written:
                keys = sorted(d.keys())
                with open(log_path, "w", newline="") as f:
                    csv.writer(f).writerow(["step", "epoch", "lr"] + keys)
                header_written = True
            if step % 20 == 1 or 0 < args.max_batches:
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [step, ep, round(opt.param_groups[0]["lr"], 6)]
                        + [round(d[k].item(), 5) for k in keys]
                    )
            pbar.set_postfix(loss=f"{loss.item():.3f}")
            if step % 20 == 0 and not sys.stderr.isatty():
                print(f"[pretrain_baselines] {args.baseline} epoch={ep} step={step} "
                      f"loss={loss.item():.3f}", flush=True)
            if 0 < args.max_batches <= step:
                break
        torch.save(
            {"encoder": model.encoder.state_dict(), "cfg": cfg, "baseline": args.baseline,
             "extra": {k: v.state_dict() for k, v in extra.items()}},
            os.path.join(out_dir, f"ckpt_ep{ep}.pt"),
        )
        if 0 < args.max_batches <= step:
            break
    print(f"[pretrain_baselines] done -> {log_path}")


if __name__ == "__main__":
    main()
