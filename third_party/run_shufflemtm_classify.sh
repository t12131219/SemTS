#!/usr/bin/env bash
# ShuffleMTM 论文 Table 3 / Table 4 分类复现（3090 Ubuntu）：
#   ① ShuffleMTM 官方模型 + 论文 Table 7/8 配置搜索（验证集选优 + 5 次平均）
#   ② SemTS 主配置（ablation_w2）+ 线性探针（SemTS 擅长的配置与协议）
# Usage: bash third_party/run_shufflemtm_classify.sh [seeds=5]
set -eo pipefail
cd "$(dirname "$0")/.."
LOG="experiments/shufflemtm_classify.log"
: > "$LOG"
SEEDS="${1:-5}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts"; exit 1; }

echo "===== ① ShuffleMTM 论文配置搜索：AD =====" | tee -a "$LOG"
python third_party/run_shufflemtm_classify.py --name AD --data_root data/shufflemtm/AD \
    --stage all --seeds "$SEEDS" 2>&1 | tee -a "$LOG" || { echo "[FAIL] AD"; exit 1; }
echo "===== ① ShuffleMTM 论文配置搜索：PTB =====" | tee -a "$LOG"
python third_party/run_shufflemtm_classify.py --name PTB --data_root data/shufflemtm/PTB \
    --stage all --seeds "$SEEDS" 2>&1 | tee -a "$LOG" || { echo "[FAIL] PTB"; exit 1; }

echo "===== ② SemTS 探针：AD =====" | tee -a "$LOG"
python scripts/convert_comet_classify.py --name AD --data_root data/shufflemtm/AD \
    --out data/shufflemtm/AD_classify
for f in 1.0 0.2 0.1 0.05; do
    python scripts/finetune_classify.py --ckpt experiments/ablation_w2/ckpt_ep9.pt \
        --data_dir data/shufflemtm/AD_classify --fraction "$f" 2>&1 | tee -a "$LOG"
done

echo "===== ② SemTS 探针：PTB =====" | tee -a "$LOG"
python scripts/convert_comet_classify.py --name PTB --data_root data/shufflemtm/PTB \
    --out data/shufflemtm/PTB_classify
for f in 1.0 0.2 0.1 0.05; do
    python scripts/finetune_classify.py --ckpt experiments/ablation_w2/ckpt_ep9.pt \
        --data_dir data/shufflemtm/PTB_classify --fraction "$f" 2>&1 | tee -a "$LOG"
done

echo "===== CLASSIFY TABLES DONE =====" | tee -a "$LOG"
