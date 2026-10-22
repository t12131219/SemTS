#!/usr/bin/env bash
# Ablation experiments for RTX 3090 24GB / Ubuntu (batch 256 + TF32, sequential).
# Usage: bash scripts/run_ablations_3090.sh <epochs=10>
set -u
cd "$(dirname "$0")/.."
EP="${1:-10}"
LAST=$((EP - 1))

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts failed"; exit 1; }

echo "===== ablation: -S3 ====="
python scripts/pretrain.py --data data/ETTm1.csv --epochs "$EP" --batch_size 256 --tf32 --out experiments/ablation_w3 --w3 0 || exit 1
python scripts/finetune_forecast.py --ckpt "experiments/ablation_w3/ckpt_ep${LAST}.pt" --data data/ETTm1.csv --horizons 96,336 || exit 1

echo "===== ablation: -S4 ====="
python scripts/pretrain.py --data data/ETTm1.csv --epochs "$EP" --batch_size 256 --tf32 --out experiments/ablation_w4 --w4 0 || exit 1
python scripts/finetune_forecast.py --ckpt "experiments/ablation_w4/ckpt_ep${LAST}.pt" --data data/ETTm1.csv --horizons 96,336 || exit 1

echo "===== ablation: S1+S2 only ====="
python scripts/pretrain.py --data data/ETTm1.csv --epochs "$EP" --batch_size 256 --tf32 --out experiments/ablation_s12 --w3 0 --w4 0 || exit 1
python scripts/finetune_forecast.py --ckpt "experiments/ablation_s12/ckpt_ep${LAST}.pt" --data data/ETTm1.csv --horizons 96,336 || exit 1

echo "===== ABLATIONS COMPLETE ====="
