#!/usr/bin/env bash
# SemTS one-click pipeline for RTX 3090 24GB / Ubuntu (batch 256 + TF32).
# Flow: env check -> smoke test -> download ETTm1 -> CPU pretrain smoke
# -> [CUDA full pretrain -> forecasting eval].
# Usage: bash scripts/run_pipeline_3090.sh
set -u
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts failed"; exit 1; }
which python
python -V

mkdir -p experiments
LOG="experiments/pipeline_3090.log"
: > "$LOG"
TMPLOG="$(mktemp)"

run() {
    local name="$1"; shift
    echo "===== STEP: $name =====" | tee -a "$LOG"
    python "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then echo "[FAIL] STEP $name"; exit 1; fi
}

run check_env -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
run smoke_test scripts/smoke_test.py
run download_data scripts/download_data.py --out data --datasets ETTm1
run pretrain_cpu_smoke scripts/pretrain.py --data data/ETTm1.csv --max_batches 5 --batch_size 8

if ! python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "[WARN] CUDA not available, skipping full pretrain and eval (CPU smoke only)"
    exit 0
fi

run pretrain_cuda scripts/pretrain.py --data data/ETTm1.csv --epochs 10 --batch_size 256 --tf32
run finetune_forecast scripts/finetune_forecast.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt --data data/ETTm1.csv

echo "===== PIPELINE COMPLETE ====="
echo "Log: $(pwd)/$LOG"
rm -f "$TMPLOG"
