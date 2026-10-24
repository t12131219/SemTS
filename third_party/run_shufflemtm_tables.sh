#!/usr/bin/env bash
# ShuffleMTM (ICLR 2025) 官方代码复现：Table 1 域内预测（8 数据集 × 4 视野）+ Table 2 跨域（8 对）。
# 官方协议：seq_len=96, patch 6/stride 6, e_layers 3, d_model 32, d_ff 128, n_heads 16,
# mask 0.4, 预训练 100 epoch lr 5e-4, 微调 freeze 10 + FT 20 epoch lr 5e-4（官方脚本默认）。
# 单次运行（论文为 5-run 平均，时间不允许时注明）；其余基线引用论文数字。
# Usage: bash third_party/run_shufflemtm_tables.sh   (在 SemTS 根目录)
set -eo pipefail   # 管道失败即停（防止 python 崩溃被 tee 吞掉）
cd "$(dirname "$0")/.."
LOG="experiments/shufflemtm_tables.log"
: > "$LOG"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts"; exit 1; }

cd third_party/ShuffleMTM
OUT=outputs
SEED=2024
HORIZONS="96 192 336 720"

# dataset|data_path|freq|batch
declare -A D=( [ETTh1]="ETTh1.csv|h|64" [ETTh2]="ETTh2.csv|h|64" [ETTm1]="ETTm1.csv|t|64" [ETTm2]="ETTm2.csv|t|64"
               [Exchange]="exchange_rate.csv|d|64" [Weather]="weather.csv|t|64"
               [Electricity]="electricity.csv|h|32" [Traffic]="traffic.csv|h|32" )

common_args="--e_layers 3 --n_heads 16 --d_model 32 --d_ff 128 --patch_len 6 --stride 6 \
--seq_len 96 --pretrain_learning_rate 5e-4 --mask_ratio 0.4 --gpu 0 --seed $SEED --num_workers 16"

setting() {  # $1=data $2=bs
    echo "${1}_sl96_el3_dl1_dm32_df128_nh16_pl6_ep100_bs${2}_lr0.0005"
}

run_step() {  # $1=标题 其余=python 参数
    echo "===== $1 =====" | tee -a "../../$LOG"
    python run.py "${@:2}" 2>&1 | tee -a "../../$LOG"
    test "${PIPESTATUS[0]}" -eq 0 || { echo "[FAIL] $1"; exit 1; }
}

root_for() {  # ETT×4 在 data/ 根；其余在 data/shufflemtm/
    case $1 in ETTh1|ETTh2|ETTm1|ETTm2) echo "../../data";; *) echo "../../data/shufflemtm";; esac
}

ckpt_of() {  # $1=data $2=bs
    echo "$OUT/pretrain_checkpoints/$1/$(setting "$1" "$2")/$SEED/ckpt_best.pth"
}

pretrain() {  # $1=data（ckpt 已存在则跳过，支持断点续跑）
    IFS='|' read -r path freq bs <<< "${D[$1]}"
    [ -f "$(ckpt_of "$1" "$bs")" ] && { echo "[skip] pretrain $1 (ckpt exists)"; return 0; }
    run_step "pretrain $1" --task_name pretrain --data "$1" --root_path "$(root_for $1)" \
        --data_path "$path" --freq "$freq" --batch_size "$bs" --train_epochs 100 $common_args
}

finetune() {  # $1=data $2=H $3=结果后缀（默认空=域内；_cross=跨域）
    IFS='|' read -r path freq bs <<< "${D[$1]}"
    suf="${3:-}"
    ft_set="$(setting "$1" "$bs")"
    [ -f "$OUT/test_results/$1/$ft_set$suf.txt" ] && \
        grep -q " -> $2 | " "$OUT/test_results/$1/$ft_set$suf.txt" && \
        { echo "[skip] finetune $1 H=$2 suffix=$suf (result exists)"; return 0; }
    run_step "finetune $1 H=$2 suffix=$suf" --task_name finetune --data "$1" --root_path "$(root_for $1)" \
        --data_path "$path" --freq "$freq" --batch_size "$bs" --train_epochs 100 \
        --pred_len "$2" --freeze_epochs 10 --finetune_epochs 20 \
        --finetune_learning_rate 5e-4 --transfer_checkpoints ckpt_best.pth \
        --result_suffix "$suf" $common_args
}

echo "########## Table 1: in-domain forecasting ##########" | tee -a "../../$LOG"
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather Electricity Traffic; do
    pretrain "$ds"
    for H in $HORIZONS; do finetune "$ds" "$H"; done
done

echo "########## Table 2: cross-domain forecasting ##########" | tee -a "../../$LOG"
# 源域 ckpt 复制到目标域的 setting 目录，再用官方 finetune 入口
cross() {  # $1=src $2=tgt
    IFS='|' read -r path freq bs <<< "${D[$2]}"
    src_set=$(setting "$1" "${D[$1]##*|}")
    tgt_set=$(setting "$2" "$bs")
    src_ckpt="$OUT/pretrain_checkpoints/$1/$src_set/$SEED/ckpt_best.pth"
    [ -f "$src_ckpt" ] || { echo "[FAIL] 源域 ckpt 缺失: $src_ckpt（$1 预训练未成功）"; exit 1; }
    mkdir -p "$OUT/pretrain_checkpoints/$2/$tgt_set/$SEED"
    cp "$src_ckpt" "$OUT/pretrain_checkpoints/$2/$tgt_set/$SEED/ckpt_best.pth"
    for H in $HORIZONS; do finetune "$2" "$H" "_cross"; done
}
cross ETTh1 ETTh2; cross ETTh2 ETTh1; cross ETTh1 ETTm1; cross ETTh2 ETTm1
cross ETTm2 ETTm1; cross ETTm2 ETTh1; cross Weather ETTh1; cross Weather ETTm1

echo "########## SHUFFLEMTM TABLES DONE ##########" | tee -a "../../$LOG"
echo "结果文件: $OUT/test_results/<dataset>/*.txt（mse/mae）" | tee -a "../../$LOG"
