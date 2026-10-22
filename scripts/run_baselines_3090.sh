#!/usr/bin/env bash
# Unified-backbone baselines for RTX 3090 24GB / Ubuntu (batch 256 + TF32, sequential).
# Each baseline: 10-epoch pretrain -> ETTm1 linear-eval (ridge) with the SAME protocol.
# Usage: bash scripts/run_baselines_3090.sh <epochs=10>
set -u
cd "$(dirname "$0")/.."
EP="${1:-10}"
LAST=$((EP - 1))

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts failed"; exit 1; }

for B in ts2vec cost simmtm tf_c; do
    echo "===== baseline: $B (pretrain) ====="
    python scripts/pretrain_baselines.py --baseline "$B" --data data/ETTm1.csv --epochs "$EP" --batch_size 256 --tf32 || exit 1
    echo "===== baseline: $B (finetune eval) ====="
    python scripts/finetune_forecast.py --ckpt "experiments/pretrain_${B}/ckpt_ep${LAST}.pt" --data data/ETTm1.csv --horizons 96,336 || exit 1
done

echo "===== BASELINES COMPLETE ====="
