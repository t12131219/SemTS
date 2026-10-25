#!/usr/bin/env bash
# SemTS 改进运动一体化脚本（3090 Ubuntu，24GB）：
#   阶段 1：B1 预训练 100 epoch × 8 数据集（显存预算 + 梯度检查点 + 幂等跳过）
#   阶段 2：每数据集三协议评估（岭回归 / A2 微调 / A3b 软通道岭）
#   阶段 3：汇总对比表 docs/semts_improvement_results.md
# Usage: bash scripts/run_semts_improvement_3090.sh [--skip-pretrain]
set -e
cd "$(dirname "$0")/.."
LOG="experiments/semts_improvement.log"
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

# 数据集：csv 路径 / 预训练 batch / 是否梯度检查点
declare -A CSV=([ETTh1]="data/ETTh1.csv" [ETTh2]="data/ETTh2.csv" [ETTm1]="data/ETTm1.csv"
                [ETTm2]="data/ETTm2.csv" [Exchange]="data/shufflemtm/exchange_rate.csv"
                [Weather]="data/shufflemtm/weather.csv" [Electricity]="data/shufflemtm/electricity.csv"
                [Traffic]="data/shufflemtm/traffic.csv")
declare -A B=([ETTh1]=256 [ETTh2]=256 [ETTm1]=256 [ETTm2]=256 [Exchange]=256
              [Weather]=128 [Electricity]=16 [Traffic]=8)

if [ "$1" != "--skip-pretrain" ]; then
    echo "########## 阶段 1：B1 预训练（100 epoch）##########" | tee -a "$LOG"
    for ds in ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather Electricity Traffic; do
        gc=""
        case $ds in Electricity|Traffic) gc="--grad_checkpoint";; esac
        out="experiments/semts_b1_$ds"
        if [ -f "$out/ckpt_ep99.pt" ]; then
            echo "[skip] pretrain $ds (ckpt_ep99 exists)" | tee -a "$LOG"
            continue
        fi
        run "pretrain100 $ds" scripts/pretrain.py --data "${CSV[$ds]}" --epochs 100 \
            --batch_size "${B[$ds]}" --tf32 $gc --out "$out" --w2 0
    done
fi

echo "########## 阶段 2：三协议评估 ##########" | tee -a "$LOG"
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather Electricity Traffic; do
    for ck in "semts_b1_$ds" "semts_tables_$ds"; do
        ckpt="experiments/$ck/ckpt_ep99.pt"
        [ "$ck" = "semts_tables_$ds" ] && ckpt="experiments/$ck/ckpt_ep9.pt"
        [ -f "$ckpt" ] || continue
        tag="${ck#semts_}"
        run "ridge $tag" scripts/finetune_forecast.py --ckpt "$ckpt" \
            --data "${CSV[$ds]}" --horizons 96,192,336,720
        run "ft $tag" scripts/finetune_forecast_ft.py --ckpt "$ckpt" \
            --data "${CSV[$ds]}" --horizons 96,192,336,720 --lp 10 --ft 20 --batch_size 64
        run "softcc $tag" scripts/finetune_forecast_softcc.py --ckpt "$ckpt" \
            --data "${CSV[$ds]}" --horizons 96,192,336,720
    done
done

echo "########## 阶段 3：汇总 ##########" | tee -a "$LOG"
python scripts/aggregate_semts_improvements.py

echo "===== SEMTS IMPROVEMENT CAMPAIGN DONE =====" | tee -a "$LOG"
echo "结果：docs/semts_improvement_results.md ｜ 日志：experiments/semts_improvement.log" | tee -a "$LOG"
