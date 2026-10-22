# SemTS

时序数据"通用语义"层面的自监督预训练模型（论文级方法，单卡可训练，目标环境 PyTorch 2.6.0 + CUDA 12.4）。

## 定位
把"通用时序语义"7 维度归纳为 4 条可消融的预训练**语义流**：

| Stream | 语义 | 内容 | 来源 |
|---|---|---|---|
| S1 | ①时间 + ③实体 | 层级对比（时序 + 实例，多尺度） | TS2Vec 路线 |
| S2 | ④模式 | 趋势/季节解耦（时域 + 频域对比） | CoST 路线 |
| S3 | ⑤关系（创新） | 跨通道滞后软回归 + 领先方向判别 | 新任务 |
| S4 | ⑦质量（创新） | 三态缺失/质量掩码重建 + 质量等级预测 | 新任务 |

## 快速开始
1. 环境：[docs/setup.md](docs/setup.md)（conda + PyTorch 2.6.0 CUDA 12.4）
2. 冒烟测试：`python scripts/smoke_test.py`（纯 CPU）
3. 预训练：`python scripts/pretrain.py --data data/ETTm1.csv`
4. 预测评估：`python scripts/finetune_forecast.py --ckpt <ckpt> --data data/ETTm1.csv`
5. 分类评估：`python scripts/finetune_classify.py --ckpt <ckpt> --data_dir data/SleepEEG`

## 文档
- 完整设计：[docs/design.md](docs/design.md)
- 调研基础：[../Atricles/Time Series Forecastinng/PreTrain/时序预训练_语义映射与研究选题.md](../Atricles/Time%20Series%20Forecastinng/PreTrain/时序预训练_语义映射与研究选题.md)

## 目录
```
src/semts/
├── backbone.py    # patch + 通道独立 Transformer 骨干
├── augs.py        # 三态掩码 / scale-shift-jitter / 窗口 / 隐向量掩码
├── losses.py      # 时序/实例/频域对比、队列 NT-Xent、软交叉熵
├── lag_labels.py  # S3 弱标签（滞后互相关）
├── stream1.py     # S1 层级对比头
├── stream2.py     # S2 模式头（TFD/SFD）
├── stream3.py     # S3 关系头 ★
├── stream4.py     # S4 质量头 ★
├── semts.py       # 总装 + 总损失
├── config.py      # 配置
└── datasets/      # 滑动窗口数据集
scripts/
├── smoke_test.py         # 冒烟测试（CPU）
├── run_pipeline.bat/.ps1 # 一键流水线（本地终端运行）
├── download_data.py      # 下载公开基准数据
├── pretrain.py           # SemTS 预训练
├── pretrain_baselines.py # 基线预训练（ts2vec/cost/simmtm/tf_c，统一骨干）
├── finetune_forecast.py  # 预测线性评估（岭回归闭式解）
├── finetune_classify.py  # 分类线性探针
├── eval_anomaly.py       # 异常检测（重建误差 + 阈值 F1）
├── eval_zeroshot.py      # 零样本跨域
├── probes.py             # 四组语义探针实验
└── analyze_losses.py     # 训练损失日志分析
```

## 状态
- [x] 设计方案 v0.1（docs/design.md）
- [x] M1–M2 核心代码（骨干 + 四条语义流 + 冒烟测试）
- [x] 预训练 / 预测评估 / 分类评估脚本
- [ ] 待用户 conda 环境就绪后：CPU 冒烟验证 → CUDA 预训练
- [ ] 异常检测 / 零样本 / 语义探针脚本（eval_anomaly.py、eval_zeroshot.py、probes.py）
- [ ] 基线复现（TS2Vec/CoST/SimMTM/TimesURL 统一骨干）
- [ ] 主结果 + 消融 + 论文图表
