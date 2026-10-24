#!/usr/bin/env bash
# SemTS 跨域预测（ShuffleMTM 论文 Table 2 的 SemTS 列）：
# 源域预训练 ckpt（④ 中已产出）冻结迁移到目标域，统一岭回归评估，4 视野平均。
# 8 对与论文一致：ETTh1→ETTh2, ETTh2→ETTh1, ETTh1→ETTm1, ETTh2→ETTm1,
#                ETTm2→ETTm1, ETTm2→ETTh1, Weather→ETTh1, Weather→ETTm1
# Usage: bash scripts/run_semts_crossdomain.sh   （在 ④ run_semts_tables.sh 之后执行）
set -e
cd "$(dirname "$0")/.."
LOG="experiments/semts_crossdomain.log"
: > "$LOG"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts"; exit 1; }

run() {
    echo "===== $1 =====" | tee -a "$LOG"
    shift
    python "$@" 2>&1 | tee -a "$LOG"
    test ${PIPESTATUS[0]} -eq 0 || exit 1
}

declare -A CSV=([ETTh1]="data/ETTh1.csv" [ETTh2]="data/ETTh2.csv" [ETTm1]="data/ETTm1.csv"
                [ETTm2]="data/ETTm2.csv" [Weather]="data/shufflemtm/weather.csv")

for pair in "ETTh1|ETTh2" "ETTh2|ETTh1" "ETTh1|ETTm1" "ETTh2|ETTm1" \
            "ETTm2|ETTm1" "ETTm2|ETTh1" "Weather|ETTh1" "Weather|ETTm1"; do
    SRC="${pair%%|*}"; TGT="${pair##*|}"
    ckpt="experiments/semts_tables_${SRC}/ckpt_ep9.pt"
    [ -f "$ckpt" ] || { echo "[FAIL] 缺少源域 ckpt: $ckpt（先跑 run_semts_tables.sh）"; exit 1; }
    run "cross ${SRC}->${TGT}" scripts/finetune_forecast.py --ckpt "$ckpt" \
        --data "${CSV[$TGT]}" --horizons 96,192,336,720
done

echo "===== SEMTS CROSS-DOMAIN DONE =====" | tee -a "$LOG"
