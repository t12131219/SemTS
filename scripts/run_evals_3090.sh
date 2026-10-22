#!/usr/bin/env bash
# SemTS full evaluation chain for RTX 3090 / Ubuntu (same protocol as run_evals.bat).
# Usage: bash scripts/run_evals_3090.sh <checkpoint_path>
set -u
cd "$(dirname "$0")/.."
if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run_evals_3090.sh experiments/pretrain_ettm1/ckpt_ep9.pt"
    exit 1
fi
CKPT="$(realpath "$1")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts failed"; exit 1; }

LOG="experiments/evals_log_3090.txt"
: > "$LOG"
TMPLOG="$(mktemp)"

run() {
    local name="$1"; shift
    echo "===== STEP: $name =====" | tee -a "$LOG"
    python "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then echo "[FAIL] STEP $name"; exit 1; fi
}

run analyze_losses scripts/analyze_losses.py --log experiments/pretrain_ettm1/losses.csv

for D in ETTm1 ETTh1 ETTh2 ETTm2; do
    run finetune_$D scripts/finetune_forecast.py --ckpt "$CKPT" --data "data/$D.csv" --horizons 96,192,336,720
done

run zeroshot scripts/eval_zeroshot.py --ckpt "$CKPT" --data_src data/ETTm1.csv --data_tgt data/ETTh1.csv --horizons 96,336
run anomaly_PSM scripts/eval_anomaly.py --ckpt "$CKPT" --data data/PSM/data.csv --labels data/PSM/label.npy --L 336
run classify_ArrowHead scripts/finetune_classify.py --ckpt "$CKPT" --data_dir data/ArrowHead

run probe_s1 scripts/probes.py --ckpt "$CKPT" --probe s1_shuffle --data data/ETTm1.csv
run probe_s2 scripts/probes.py --ckpt "$CKPT" --probe s2_pattern
run probe_s3 scripts/probes.py --ckpt "$CKPT" --probe s3_lag
run probe_s4 scripts/probes.py --ckpt "$CKPT" --probe s4_quality

echo "===== EVALS COMPLETE ====="
echo "Log: $(pwd)/$LOG"
rm -f "$TMPLOG"
