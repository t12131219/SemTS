#!/usr/bin/env bash
# ShuffleMTM 论文 Table 3 / Table 4 分类复现（3090 Ubuntu）：
#   ① ShuffleMTM 官方模型驱动（预训练 100 ep + freeze-then-finetune，比例 100%/20%/10%/5%）
#   ② SemTS 主配置（ablation_w2）+ 线性探针（比例同上；用 SemTS 擅长的配置与协议）
# Usage: bash third_party/run_shufflemtm_classify.sh
set -e
cd "$(dirname "$0")/.."
LOG="experiments/shufflemtm_classify.log"
: > "$LOG"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts"; exit 1; }

echo "===== ① ShuffleMTM 官方模型：AD =====" | tee -a "$LOG"
python third_party/run_shufflemtm_classify.py --name AD --data_root data/shufflemtm/AD 2>&1 | tee -a "$LOG"
echo "===== ① ShuffleMTM 官方模型：PTB =====" | tee -a "$LOG"
python third_party/run_shufflemtm_classify.py --name PTB --data_root data/shufflemtm/PTB 2>&1 | tee -a "$LOG"

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
