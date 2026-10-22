"""下载公开基准数据（ETT 系列，raw.githubusercontent 直链已验证可用）。

用法:
    python scripts/download_data.py --out data --datasets ETTm1,ETTh1
（默认下载全部 ETT；网络中断可用 --datasets 重试，残片请手动删除。）

TODO（后续补充，需要额外预处理）：
- Weather（Autoformer 标准 weather.csv，52696×21）：走 thuml 仓库 Google Drive 链接，
  或 HuggingFace `autogluon/chronos_datasets` 镜像；
- Electricity/Traffic/Exchange_rate：`github.com/laiguokun/multivariate-time-series-data`
  提供 .txt.gz 原始格式，需解压并按基准约定重采样为小时级 CSV 后放入 data/。
"""
import argparse
import os
import urllib.request

SOURCES = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--datasets", type=str, default=",".join(SOURCES.keys()),
                    help="逗号分隔，如 ETTm1,weather")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.datasets.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in SOURCES:
            print(f"[skip] 未知数据集 {name}")
            continue
        dst = os.path.join(args.out, f"{name}.csv")
        if os.path.exists(dst):
            print(f"[skip] 已存在 {dst}")
            continue
        print(f"[download] {name} -> {dst}")
        urllib.request.urlretrieve(SOURCES[name], dst)
        print(f"[done] {name}")
    print("[download_data] finished")


if __name__ == "__main__":
    main()
