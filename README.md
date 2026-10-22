# SemTS

时序数据"通用语义"层面的自监督预训练模型（论文级方法，单卡可训练，PyTorch 2.6.0 + CUDA 12.4）。
主配置：**S1+S3+S4** 三条语义流 + 通道独立 patch Transformer 骨干，ETTm1 预训练后在预测/分类/异常/零样本上全面验证。

## 定位
把"通用时序语义"7 维度归纳为可消融的预训练**语义流**（主配置为 S1+S3+S4；S2 经消融判定移除，模块保留）：

| Stream | 语义 | 内容 | 来源 | 状态 |
|---|---|---|---|---|
| S1 | ①时间 + ③实体 | 层级对比（时序 + 实例，多尺度） | TS2Vec 路线 | ✅ 主配置 |
| S2 | ④模式 | 趋势/季节解耦（时域 + 频域对比） | CoST 路线 | ⚠️ 消融显示负贡献，主配置 w2=0 |
| S3 | ⑤关系（创新） | 跨通道滞后软回归 + 领先方向判别 | 新任务 | ✅ 主配置 |
| S4 | ⑦质量（创新） | 三态缺失/质量掩码重建 + 质量等级预测 | 新任务 | ✅ 主配置 |

## 核心结果（公平条件：ETTm1 预训练 10 epoch，batch 256 + TF32，RTX 3090）

**预测主结果**（MSE@H，z-score 空间，统一岭回归协议）：

| 数据集 | H=96 | H=192 | H=336 | H=720 |
|---|---|---|---|---|
| ETTm1 | **0.384** | **0.429** | **0.466** | **0.512** |
| ETTh1 | 0.520 | 0.578 | 0.623 | 0.722 |
| ETTh2 | 0.209 | 0.251 | 0.297 | 0.379 |
| ETTm2 | 0.190 | 0.253 | 0.322 | 0.404 |

**统一骨干基线**（H=96/336）：ts2vec 0.426/0.502、tf_c 0.388/0.461、simmtm 0.456/0.534、cost 1.073/1.068。
**官方实现对照**（同一协议）：ShuffleMTM 0.449、CoST 0.997、TS2Vec 1.086（其官方协议下 0.670，可复现）。
**消融证据链**：S1-only 0.426 → +S3+S4 **↓10.9%**（协同，同删退化 32%）→ +S2 退化至 0.482。
其余：异常 PSM F1=0.407、分类 ArrowHead 0.531、零样本 ETTm1→ETTh1 1.002、探针 s1=0.998/s3=0.474/s4=0.345。

📄 完整表格与逐项分析：[docs/results.md](docs/results.md) · 论文草稿：[docs/paper_draft.md](docs/paper_draft.md)

## 快速开始

```bash
# 1. 环境（Windows / Ubuntu 3090 见 docs/setup.md）
conda create -n semts python=3.10 -y && conda activate semts
conda install pytorch==2.6.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install numpy pandas tqdm
pip install -r third_party/requirements_official.txt   # 仅官方实现对照需要

# 2. 冒烟测试（纯 CPU）
python scripts/smoke_test.py

# 3. 预训练（主配置 S1+S3+S4；3090 加 --batch_size 256 --tf32）
python scripts/pretrain.py --data data/ETTm1.csv --epochs 10 --w2 0

# 4. 评估
python scripts/finetune_forecast.py --ckpt experiments/<exp>/ckpt_ep9.pt --data data/ETTm1.csv
```

**一键脚本**（10 epoch 全量公平对照，顺序执行）：

| 平台 | 主模型 | 消融 | 统一骨干基线 | 官方实现对照 | 全任务评估 |
|---|---|---|---|---|---|
| Windows | `run_pipeline.bat` | `run_ablations.bat` | `run_baselines.bat` | `third_party\run_official_all.bat` | `run_evals.bat` |
| Ubuntu/3090 | `run_pipeline_3090.sh` | `run_ablations_3090.sh` | `run_baselines_3090.sh` | `third_party/run_official_all.sh` | `run_evals_3090.sh` |

官方仓库克隆与补丁：`bash third_party/setup_repos.sh`（TS2Vec/CoST/ShuffleMTM；`third_party/` 下的官方代码不入 git）。

## 文档
- 完整设计：[docs/design.md](docs/design.md)
- 环境搭建与执行顺序：[docs/setup.md](docs/setup.md)
- 实验结果：[docs/results.md](docs/results.md)
- 论文草稿：[docs/paper_draft.md](docs/paper_draft.md)
- 调研基础：[../Atricles/Time Series Forecastinng/PreTrain/时序预训练_语义映射与研究选题.md](../Atricles/Time%20Series%20Forecastinng/PreTrain/时序预训练_语义映射与研究选题.md)

## 目录
```
src/semts/
├── backbone.py    # patch + 通道独立 Transformer 骨干
├── augs.py        # 三态掩码 / scale-shift-jitter / 窗口 / 隐向量掩码
├── losses.py      # 时序/实例/频域对比、队列 NT-Xent、软交叉熵
├── lag_labels.py  # S3 弱标签（滞后互相关）
├── stream1.py     # S1 层级对比头
├── stream2.py     # S2 模式头（TFD/SFD，主配置关闭）
├── stream3.py     # S3 关系头 ★
├── stream4.py     # S4 质量头 ★
├── semts.py       # 总装 + 总损失
├── baselines.py   # 统一骨干基线（ts2vec/cost/simmtm/tf_c）
├── model_factory.py
├── config.py      # 配置
└── datasets/      # 滑动窗口数据集
scripts/
├── smoke_test.py          # 冒烟测试（CPU）
├── run_pipeline.bat/.ps1  # 一键流水线（Windows）
├── run_pipeline_3090.sh   # 一键流水线（Ubuntu/3090）
├── run_ablations_3090.sh  # 三组消融
├── run_baselines_3090.sh  # 统一骨干基线
├── run_remaining_3090.sh  # 补齐脚本（simmtm/tf_c + -S2 消融）
├── run_evals.bat / run_evals_3090.sh  # 全任务评估
├── download_data.py       # 下载公开基准数据
├── pretrain.py            # SemTS 预训练
├── pretrain_baselines.py  # 基线预训练（统一骨干）
├── finetune_forecast.py   # 预测线性评估（岭回归闭式解）
├── finetune_from_features.py  # 预计算特征评估（官方实现统一协议）
├── finetune_classify.py   # 分类线性探针
├── eval_anomaly.py        # 异常检测（重建误差 + 阈值 F1）
├── eval_zeroshot.py       # 零样本跨域
├── probes.py              # 四组语义探针实验
└── analyze_losses.py      # 训练损失日志分析
third_party/
├── setup_repos.sh               # 克隆官方仓库 + 统一协议补丁
├── requirements_official.txt    # 官方实现对照依赖
├── run_official_all.sh/.bat     # 官方实现对照总脚本
└── run_official_{ts2vec,cost,shufflemtm}.py  # 官方实现适配器（训练+特征导出）
```

## 状态
- [x] 设计方案（docs/design.md）
- [x] 核心代码（骨干 + 四条语义流 + 冒烟测试）
- [x] 预训练 / 预测评估 / 分类评估脚本
- [x] CPU 冒烟验证 → CUDA 预训练（4070 与 3090 双机验证）
- [x] 异常检测 / 零样本 / 语义探针脚本与结果
- [x] 基线复现：统一骨干（ts2vec/cost/simmtm/tf_c）+ 官方实现（TS2Vec/CoST/ShuffleMTM 双口径）
- [x] 主结果 + 消融 + 论文草稿（docs/results.md、docs/paper_draft.md）
- [ ] 可选后续：更大规模数据集（UEA 全集、SMD/SWaT）、S2 调参再考察、事件/状态语义（⑥）
