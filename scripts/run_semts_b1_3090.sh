#!/usr/bin/env bash
# 改进方案 B1：预训练预算对齐（10 → 100 epoch），3090 Ubuntu 执行。
# 产出 experiments/semts_b1_<dataset>/，评估用 finetune_forecast（岭）+ finetune_forecast_ft（A2 微调）。
# 显存预算同 run_semts_tables.sh（宽通道 + 梯度检查点）。
# Usage: bash scripts/run_semts_b1_3090.sh
set -e
cd "$(dirname "$0")/.."
LOG="experiments/semts_b1.log"
: > "$LOG"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    echo "===== $1 =====" | tee -a "$LOG"
    shift
    python "$@" 2>&1 | tee -a "$LOG"
    test ${PIPESTATUS[0]} -eq 0 || exit 1
}

declare -A B=([ETTh1]=256 [ETTh2]=256 [ETTm1]=256 [ETTm2]=256 [Exchange]=256
              [Weather]=128 [Electricity]=16 [Traffic]=8)

for ds in ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather Electricity Traffic; do
    csv="data/shufflemtm/$(case $ds in
        ETTh1) echo ETTh1.csv;; ETTh2) echo ETTh2.csv;; ETTm1) echo ETTm1.csv;;
        ETTm2) echo ETTm2.csv;; Exchange) echo exchange_rate.csv;; Weather) echo weather.csv;;
        Electricity) echo electricity.csv;; Traffic) echo traffic.csv;; esac)"
    case $ds in ETTh1|ETTh2|ETTm1|ETTm2) csv="data/$ds.csv";; esac
    gc=""
    case $ds in Electricity|Traffic) gc="--grad_checkpoint";; esac
    out="experiments/semts_b1_$ds"
    if [ -f "$out/ckpt_ep99.pt" ]; then
        echo "[skip] $ds (ckpt_ep99 exists)" | tee -a "$LOG"
        continue
    fi
    run "pretrain100 $ds" scripts/pretrain.py --data "$csv" --epochs 100 \
        --batch_size "${B[$ds]}" --tf32 $gc --out "$out" --w2 0
    run "ridge $ds" scripts/finetune_forecast.py --ckpt "$out/ckpt_ep99.pt" \
        --data "$csv" --horizons 96,192,336,720
    run "ft $ds" scripts/finetune_forecast_ft.py --ckpt "$out/ckpt_ep99.pt" \
        --data "$csv" --horizons 96,192,336,720 --lp 10 --ft 20 --batch_size 64
done

echo "===== SEMTS B1 DONE =====" | tee -a "$LOG"
