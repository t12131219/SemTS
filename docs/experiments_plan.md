# SemTS 实验计划（执行手册）

> 本文件把所有待跑实验固化为命令矩阵与结果表骨架。执行环境恢复（DSH 重启）或用户本地运行后，
> 按序逐条执行、把结果填入对应表格，即可产出论文实验章节。
> 约定：所有命令在 `E:\PycharmProgram\SemTS` 根目录、`semts` conda 环境下运行。

## 阶段 0：冒烟与数据（必须最先全绿）

```powershell
python scripts/smoke_test.py                              # 5 项断言全 PASS
python scripts/download_data.py --out data --datasets ETTm1,ETTh1,ETTh2,ETTm2,weather,electricity
python scripts/pretrain.py --data data/ETTm1.csv --max_batches 5 --batch_size 8   # CPU 冒烟，losses.csv 出现 5 行
```

## 阶段 1：SemTS 主预训练（ETTm1）

```powershell
python scripts/pretrain.py --data data/ETTm1.csv --epochs 10 --batch_size 128
python scripts/analyze_losses.py --log experiments/pretrain_ettm1/losses.csv
```

| 检查项 | 期望 |
|---|---|
| total 下降 | 结尾均值明显低于开头（≥20%） |
| lag（S3） | 从 0 附近起步、逐步上升后回落（弱标签有效且被学习） |
| qpred（S4） | 显著高于 0.25 随机基线并下降 |
| 各流数量级 | 同一量级（否则调 w1..w4 / lam_dir / lam_q） |

## 阶段 2：预测主结果（主表 1）

对每个数据集 × 每个 ckpt 执行（ETT×4 + Weather + Electricity 子集 100 通道）：

```powershell
python scripts/finetune_forecast.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt --data data/ETTm1.csv --horizons 96,192,336,720
```

| 数据集 | MSE/MAE (H=96/192/336/720) | 备注 |
|---|---|---|
| ETTh1 | | |
| ETTh2 | | |
| ETTm1 | | |
| ETTm2 | | |
| Weather | | |
| Electricity(100ch) | | |

## 阶段 3：消融（主表 2）—— 语义流的贡献

用 CLI 覆盖权重，无需改代码（各自预训练 10 epoch，ETTm1 固定）：

```powershell
# 全量（已有）与三条消融：
python scripts/pretrain.py --data data/ETTm1.csv --epochs 10 --batch_size 128 --out experiments/ablation_w3 --w3 0
python scripts/pretrain.py --data data/ETTm1.csv --epochs 10 --batch_size 128 --out experiments/ablation_w4 --w4 0
python scripts/pretrain.py --data data/ETTm1.csv --epochs 10 --batch_size 128 --out experiments/ablation_s12 --w3 0 --w4 0
# 每个消融 ckpt 用同一命令评估：
python scripts/finetune_forecast.py --ckpt experiments/ablation_w3/ckpt_ep9.pt --data data/ETTm1.csv
```

| 配置 | ETTm1 平均 MSE | 结论 |
|---|---|---|
| 全量（w1=w2=w3=w4=1） | | 基准 |
| -S3（w3=0） | | ⑤ 关系流的贡献 |
| -S4（w4=0） | | ⑦ 质量流的贡献 |
| -S3-S4（仅主干 S1+S2） | | 创新点合计贡献 |
| 随机初始化（无预训练） | | 预训练本身的价值 |

## 阶段 4：分类 / 异常 / 零样本（主表 3/4/5）

```powershell
# 分类（数据先按 finetune_classify.py 头部约定转成 npy 格式）
python scripts/finetune_classify.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt --data_dir data/SleepEEG
# 异常（SMD/SWaT 按 eval_anomaly.py 头部约定准备 CSV + 标签 npy）
python scripts/eval_anomaly.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt --data data/SMD.csv --labels data/SMD_label.npy
# 零样本（ETTm1 → ETTh1 等跨域对）
python scripts/eval_zeroshot.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt --data_src data/ETTm1.csv --data_tgt data/ETTh1.csv
```

| 任务 | 数据集 | 指标 | 结果 |
|---|---|---|---|
| 分类 | SleepEEG / Epilepsy / FD-B / Gesture / EMG | Acc | |
| 异常 | SMD / SWaT | F1 | |
| 零样本 | ETTm1→ETTh1 / ETTm1→Weather | MSE | |

## 阶段 5：语义探针（第二层证据）

```powershell
python scripts/probes.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt --probe s1_shuffle --data data/ETTm1.csv
python scripts/probes.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt --probe s2_pattern
python scripts/probes.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt --probe s3_lag
python scripts/probes.py --ckpt experiments/pretrain_ettm1/ckpt_ep9.pt --probe s4_quality
```

| 探针 | 语义 | 期望 | 结果 |
|---|---|---|---|
| s1_shuffle | ① 顺序 | acc > 0.8 | |
| s2_pattern | ④ 趋势/季节 | 趋势表示 R²(趋势) > R²(季节)，季节反之 | |
| s3_lag | ⑤ 滞后 | acc > 0.5（随机 ≈1/7） | |
| s4_quality | ⑦ 质量 | acc > 0.6（随机 0.25） | |

## 阶段 6：统一骨干基线（对比锚点）

```powershell
python scripts/pretrain_baselines.py --baseline ts2vec --data data/ETTm1.csv --epochs 10 --batch_size 128
python scripts/pretrain_baselines.py --baseline cost   --data data/ETTm1.csv --epochs 10 --batch_size 128
python scripts/pretrain_baselines.py --baseline simmtm --data data/ETTm1.csv --epochs 10 --batch_size 128
python scripts/pretrain_baselines.py --baseline tf_c   --data data/ETTm1.csv --epochs 10 --batch_size 128
# 各基线 checkpoint 用统一评测脚本评估（待 base-line 加载器接线，见 TODO）
```

| 方法 | ETTm1 MSE | Weather MSE | 备注 |
|---|---|---|---|
| ts2vec（统一骨干） | | | 对照锚点 |
| cost（统一骨干） | | | S2 单流 |
| simmtm（统一骨干） | | | 掩码路线 |
| tf_c（统一骨干） | | | 时频路线 |
| **SemTS（全量）** | | | 本方法 |

## 阶段 7：论文产物

- 主表 1-5 + 消融表 + 探针表 + 基线表；
- 损失曲线图（analyze_losses 分段均值 + 平滑曲线）；
- 表示可视化（可选：滞后结构、缺失/质量嵌入分布）；
- 复现说明：README + 本手册 + 全部命令与随机种子。

## TODO（代码侧遗留，按执行顺序）
1. ~~给 pretrain.py 加 --w1..--w4 CLI 覆盖~~ ✅ 已完成（含 lam_dir/lam_q/tau_corr/mask/noise 与 --tag）
2. ~~基线检查点加载器~~ ✅ 已完成（src/semts/model_factory.py，全部 eval 脚本已接入）
3. 分类数据格式转换脚本（SimMTM/UEA → npy）✅ 已完成：scripts/convert_classify_data.py（支持 --simmtm 与 --uea 两种模式）；
   **数据获取（2026-10-21 调研）**：SimMTM 8 个分类数据集托管在 figshare（SleepEEG=article 19930178，
   含 train/val/test.pt，PyTorch dict{samples:(N,C,L), labels} 格式；其余见其仓库 download_datasets.sh）。
   figshare 的 ndownloader 对脚本下载返回 403（TLS 指纹防护，UA 伪装无效），**需浏览器手动下载**后放入
   data/raw/，再运行转换脚本。备选：UEA 数据集（timeseriesclassification.com，常规 HTTP 可下载）。
4. 异常数据准备脚本（SMD/SWaT → CSV + label npy）✅ 已完成：scripts/prep_anomaly_data.py（--psm / --smd 两种模式）。
   数据已就绪：data/PSM（220322×25，异常 11.07%）、data/SMD（1416825×38，异常 2.08%），
   label.npy 与行对齐（训练段全 0）。来源：Anomaly-Transformer 的 Google Drive 文件夹
   （gdown 已下载至 data/raw/anomaly/，含 MSL/PSM/SMAP/SMD 四包）。SWaT 需按官方教程另行申请。
