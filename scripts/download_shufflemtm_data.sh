#!/usr/bin/env bash
# 下载并预处理 ShuffleMTM 复现所需数据集（3090 Ubuntu 上执行）。
# 网络不佳时可设置代理：export https_proxy=http://127.0.0.1:7890
# AD/PTB 原始数据来源：OSF（AD）与 PhysioNet（PTB）；figshare 原链接已失效（API 404）。
set -e
cd "$(dirname "$0")/.."
mkdir -p data/shufflemtm data/raw/shufflemtm

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts || { echo "[FAIL] conda activate semts"; exit 1; }

echo "===== 1/3 Autoformer 数据集（Exchange/Weather/Electricity/Traffic）====="
pip install gdown -q
cd data/shufflemtm
gdown --folder https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy -O . || {
    echo "[WARN] gdown 失败：请手动从 Autoformer 仓库下载 electricity.csv/exchange_rate.csv/traffic.csv/weather.csv 放入 data/shufflemtm/"
}
find . -name "*.csv" -not -path "./weather.csv" -not -path "./electricity.csv" \
    -not -path "./traffic.csv" -not -path "./exchange_rate.csv" | while read -r f; do
    mv "$f" "$(basename "$f")"
done
find . -type d -empty -delete
cd ../..

echo "===== 2/3 AD（OSF 原始数据 + 预处理）====="
cd data/raw/shufflemtm
[ -f AD_Data.tar.gz ] || curl -L -o AD_Data.tar.gz https://osf.io/download/xy8qa/
mkdir -p ad_raw
tar -xzf AD_Data.tar.gz -C ad_raw
cd ../../..
python scripts/prep_ad_data.py --src data/raw/shufflemtm/ad_raw --out data/shufflemtm/AD

echo "===== 3/3 PTB（PhysioNet 完整数据库 + 预处理，耗时较长）====="
# 注意：get-zip 打包的是 Training_PTB 子集（12 通道），不是 COMET 流程需要的
# patient001-294 完整库（15 通道）；必须递归下载文件树
cd data/raw/shufflemtm
[ -d ptbdb/1.0.0 ] || wget -r -np -nH --cut-dirs=2 -c --wait=0.5 \
    https://physionet.org/files/ptbdb/1.0.0/ -P ptbdb
cd ../../..
pip install wfdb neurokit2 -q
python scripts/prep_ptb_data.py --src data/raw/shufflemtm/ptbdb/1.0.0 --out data/shufflemtm/PTB

echo "===== DATASETS READY ====="
ls data/shufflemtm
