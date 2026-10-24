#!/usr/bin/env bash
# SemTS 在 ShuffleMTM 论文表格数据集上的运行（3090 Ubuntu）：
#   8 数据集（ETTh1/h2/m1/m2、Exchange、Weather、Electricity、Traffic）各预训练 10 epoch
#   + 统一岭回归评估（H=96/192/336/720），可与 ShuffleMTM 官方数字同表比较。
# 主配置 S1+S3+S4（w2=0），batch 小集 256 / 大集 64，TF32。
# Usage: bash scripts/run_semts_tables.sh
set -e
cd "$(dirname "$0")/.."
LOG="experiments/semts_tables.log"
: > "$LOG"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 缓解显存碎片

run() {
    echo "===== $1 =====" | tee -a "$LOG"
    shift
    python "$@" 2>&1 | tee -a "$LOG"
    test ${PIPESTATUS[0]} -eq 0 || exit 1
}

# 24GB 显存预算（实测校准）：双视图编码 + 4 层注意力反向激活 ≈ token 数×~4KB。
# 实测校准（24GB 预算）：双视图+反向图，有梯度检查点 ≈ 2.5MB/token、无检查点 ≈ 4.3MB/token。
#   ETT/Exchange (7-8ch) batch 256（≤2048 token）✓；Weather (21ch) batch 128（2688）✓
#   Electricity (321ch) batch 16 + cp（5136 token ≈ 13GB）✓
#   Traffic (862ch) batch 8 + cp（6896 token ≈ 17GB）✓
declare -A B=([ETTh1]=256 [ETTh2]=256 [ETTm1]=256 [ETTm2]=256 [Exchange]=256
              [Weather]=128 [Electricity]=16 [Traffic]=8)

for ds in ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather Electricity Traffic; do
    csv="data/shufflemtm/$(case $ds in
        ETTh1) echo ETTh1.csv;; ETTh2) echo ETTh2.csv;; ETTm1) echo ETTm1.csv;;
        ETTm2) echo ETTm2.csv;; Exchange) echo exchange_rate.csv;; Weather) echo weather.csv;;
        Electricity) echo electricity.csv;; Traffic) echo traffic.csv;; esac)"
    # ETT×4 使用已有 data/*.csv
    case $ds in ETTh1|ETTh2|ETTm1|ETTm2) csv="data/$ds.csv";; esac
    gc=""
    case $ds in Electricity|Traffic) gc="--grad_checkpoint";; esac
    out="experiments/semts_tables_$ds"
    if [ -f "$out/ckpt_ep9.pt" ]; then
        echo "[skip] pretrain $ds (ckpt_ep9 exists)" | tee -a "$LOG"
    else
        run "pretrain $ds" scripts/pretrain.py --data "$csv" --epochs 10 \
            --batch_size "${B[$ds]}" --tf32 $gc --out "$out" --w2 0
    fi
    run "eval $ds" scripts/finetune_forecast.py --ckpt "$out/ckpt_ep9.pt" \
        --data "$csv" --horizons 96,192,336,720
done

echo "===== SEMTS TABLES DONE =====" | tee -a "$LOG"
