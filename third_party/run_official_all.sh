#!/usr/bin/env bash
# Official-implementation comparison (Ubuntu / RTX 3090) —— 与统一骨干对比（scripts/run_baselines_3090.sh）相互独立。
#
# 内容：TS2Vec + CoST + ShuffleMTM 三个官方仓库，在 ETTm1 上用官方训练目标预训练
# （数据/切分/评估协议与 SemTS 完全一致），导出特征后用 scripts/finetune_from_features.py
# 走与 SemTS 相同的岭回归协议评估。
#
# 用法：bash third_party/run_official_all.sh [n_iters=3240]
# 前置：conda 环境 semts 已建好（见 docs/setup.md）+ 官方依赖
#       pip install -r third_party/requirements_official.txt
set -e
cd "$(dirname "$0")/.."
NIT="${1:-3240}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts failed"; exit 1; }

echo "===== official TS2Vec: pretrain + export ====="
python third_party/run_official_ts2vec.py --data data/ETTm1.csv --n_iters "$NIT"
python scripts/finetune_from_features.py --feats experiments/official_ts2vec/feats --data data/ETTm1.csv

echo "===== official CoST: pretrain + export ====="
python third_party/run_official_cost.py --data data/ETTm1.csv --n_iters "$NIT"
python scripts/finetune_from_features.py --feats experiments/official_cost/feats --data data/ETTm1.csv

echo "===== official ShuffleMTM: pretrain (their framework, official ETT config) ====="
pushd third_party/ShuffleMTM > /dev/null
python run.py --task_name pretrain --data ETTm1 --gpu 0 --seed 2024 \
    --e_layers 3 --n_heads 16 --d_model 32 --d_ff 128 --patch_len 6 --stride 6 \
    --pretrain_learning_rate 5e-4 --seq_len 336 --root_path ../../data \
    --data_path ETTm1.csv --features M --train_epochs 10 --batch_size 64 --num_workers 4
popd > /dev/null

echo "===== official ShuffleMTM: export + unified ridge ====="
python third_party/run_official_shufflemtm.py --export --data data/ETTm1.csv
python scripts/finetune_from_features.py --feats experiments/official_shufflemtm/feats --data data/ETTm1.csv

echo "===== OFFICIAL COMPARISON COMPLETE ====="
