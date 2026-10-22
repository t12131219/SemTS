#!/usr/bin/env bash
# 补齐 3090 队列未完成部分（baselines.py 索引 bug 已修复）：
#   - 统一骨干 simmtm / tf_c 两个基线（10 epoch + ETTm1 评估）
#   - 关键补充消融 w2=0（S1+S3+S4）：验证 S2 是否为全量弱于 S1-only 的原因
# Usage: bash scripts/run_remaining_3090.sh [epochs=10]
set -e
cd "$(dirname "$0")/.."
EP="${1:-10}"
LAST=$((EP - 1))

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts failed"; exit 1; }

for B in simmtm tf_c; do
    echo "===== baseline: $B (pretrain) ====="
    python scripts/pretrain_baselines.py --baseline "$B" --data data/ETTm1.csv --epochs "$EP" --batch_size 256 --tf32 || exit 1
    echo "===== baseline: $B (finetune eval) ====="
    python scripts/finetune_forecast.py --ckpt "experiments/pretrain_${B}/ckpt_ep${LAST}.pt" --data data/ETTm1.csv --horizons 96,336 || exit 1
done

echo "===== ablation: -S2 (S1+S3+S4) ====="
python scripts/pretrain.py --data data/ETTm1.csv --epochs "$EP" --batch_size 256 --tf32 --out experiments/ablation_w2 --w2 0 || exit 1
python scripts/finetune_forecast.py --ckpt "experiments/ablation_w2/ckpt_ep${LAST}.pt" --data data/ETTm1.csv --horizons 96,336 || exit 1

echo "===== REMAINING RUNS COMPLETE ====="
