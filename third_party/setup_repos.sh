#!/usr/bin/env bash
# 官方实现对照：克隆官方仓库并应用统一协议补丁。
# 在 SemTS/third_party 目录下运行：bash setup_repos.sh
# 依赖安装：pip install -r third_party/requirements_official.txt
set -e
cd "$(dirname "$0")"

[ -d ts2vec ]    || git clone --depth 1 https://github.com/zhihanyue/ts2vec.git
[ -d CoST ]      || git clone --depth 1 https://github.com/salesforce/CoST.git
[ -d ShuffleMTM ] || git clone --depth 1 https://github.com/cikm2025shufflemtm/ShuffleMTM.git

# 统一协议补丁：数据切分 0.7/0.2/0.1 → 0.6/0.2/0.2（与 SemTS 一致）
sed -i 's/int(len(df_raw) \* 0\.7)/int(len(df_raw) * 0.6)/g' ShuffleMTM/data_provider/data_loader.py
sed -i 's/int(len(df_raw) \* 0\.1)/int(len(df_raw) * 0.2)/g' ShuffleMTM/data_provider/data_loader.py

echo "official repos ready"
echo "next: bash third_party/run_official_all.sh   (见 docs/setup.md 的执行顺序)"
