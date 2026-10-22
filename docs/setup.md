# 环境搭建（PyTorch 2.6.0 + CUDA 12.4）

## Ubuntu（RTX 3090 24GB）

```bash
# 1. conda 环境
conda create -n semts python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate semts

# 2. PyTorch 2.6.0（CUDA 12.4）
conda install pytorch==2.6.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 3. 依赖
pip install numpy pandas tqdm
pip install -r third_party/requirements_official.txt   # 官方实现对照的额外依赖

# 4. 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/smoke_test.py
```

### 执行顺序（3090，全部 10 epoch 公平对照）

```bash
bash scripts/run_pipeline_3090.sh            # ① 主模型：冒烟→数据→CUDA 预训练→预测评估
bash scripts/run_ablations_3090.sh           # ② 消融：-S3 / -S4 / 仅S1+S2（统一骨干）
bash scripts/run_baselines_3090.sh           # ③ 统一骨干基线：ts2vec/cost/simmtm/tf_c
bash third_party/run_official_all.sh         # ④ 官方实现对照：TS2Vec/CoST/ShuffleMTM（独立脚本）
bash scripts/run_evals_3090.sh experiments/pretrain_ettm1/ckpt_ep9.pt   # ⑤ 全任务评估
```

注意：③ 与 ④ 是**两套独立的对比**——③ 是"同一骨干上的目标重实现"（`src/semts/baselines.py`），
④ 是"官方仓库原始实现"（`third_party/`，官方训练目标 + 官方模型，仅数据/切分/评估协议对齐）。

### 官方仓库克隆（GitHub 上传时不包含，新机器需执行）

```bash
bash third_party/setup_repos.sh   # 克隆 ts2vec/CoST/ShuffleMTM 并应用统一切分补丁
```

仓库上传说明：`.gitignore` 已排除 `data/`（数据集，按 download_data.py 重新下载）、
`experiments/`、`experiments_origin/`（检查点与日志）、`third_party/{ts2vec,CoST,ShuffleMTM}/`
（第三方代码，用 setup_repos.sh 恢复）、`__pycache__/` 与各类临时文件。

## Windows

```powershell
# 1. 创建 conda 环境（Python 3.10 或 3.11 均可，代码兼容 3.9+；本机实测为 3.10）
conda create -n semts python=3.10 -y
conda activate semts

# 2. 安装 PyTorch 2.6.0（CUDA 12.4）
conda install pytorch==2.6.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 3. 其余依赖
pip install numpy pandas tqdm

# 4. 验证 GPU 可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Ubuntu / Linux（3090 机器）

```bash
# 1. 创建环境（同上）
conda create -n semts python=3.10 -y
conda activate semts
conda install pytorch==2.6.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install numpy pandas tqdm gdown

# 2. 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
nvidia-smi

# 3. 运行（全部为 bash 脚本，从 SemTS 根目录执行）
bash scripts/run_pipeline_3090.sh            # 全量主训练 + 预测评估
bash scripts/run_ablations_3090.sh 10        # 三组消融（默认 10 epoch）
bash scripts/run_baselines_3090.sh 10        # 四个基线（默认 10 epoch）
bash scripts/run_evals_3090.sh experiments/pretrain_ettm1/ckpt_ep9.pt   # 全任务评估
```

## 快速验证（纯 CPU 即可）

```powershell
# 一键流水线（推荐）：环境检查 → 冒烟 → 下载 ETTm1 → CPU 冒烟 → CUDA 预训练 → 预测评估
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
# 只跑 CPU 冒烟（无 GPU 时）：
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -SkipCuda

# 或手动逐步：
python scripts/smoke_test.py                                                    # 冒烟测试（无数据集）
python scripts/pretrain.py --data data/ETTm1.csv --max_batches 5 --batch_size 8 # CPU 预训练冒烟
```

## 数据准备

```powershell
# 一键下载公开基准数据（ETT 系列 + Weather/Electricity/Traffic/Exchange）
python scripts/download_data.py --out data --datasets ETTm1,ETTh1,weather
```

- ETT 系列（ETTh1/h2/m1/m2）: https://github.com/zhouhaoyi/ETDataset
- Weather / Electricity / Traffic / Exchange: Autoformer 仓库 `https://github.com/thuml/Autoformer` 的 `dataset/` 目录
- 分类数据（SleepEEG/Epilepsy/FD-B/Gesture/EMG）: SimMTM 官方仓库 `https://github.com/thuml/SimMTM` 的 `datasets/` 目录，
  需转换为 `train/val/test.npy + *_label.npy` 格式（见 `scripts/finetune_classify.py` 头部说明）
- 异常检测（SMD/SWaT/PSM）: 公开镜像，后续按 `scripts/eval_anomaly.py` 头部约定补充

下载后放入 `SemTS/data/`（CSV 直接可用）。
