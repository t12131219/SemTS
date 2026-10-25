# SemTS 2.0 改进方案：文献依据与优先级（v0.1）

> 背景：与 SOTA ShuffleMTM 的 32 格预测对比中 SemTS 输 24 格，差距集中在
> Exchange/Traffic/Electricity/ETTh1/ETTm1；分类与少标签全胜。
> 本方案基于 2026 年最新文献调研（见文末引用）提出分层改进，按"增益/风险比"排序。

## 0. 核心诊断（文献支撑）

1. **[arXiv 2605.19462] When Does SSL Transfer to Time-Series Tasks?**
   - SSL 在**异常检测**收益最大、**分类（微调）**收益良好，但**预测任务上线性探针几乎无收益**；
   - **线性探针表现不能预测微调表现**；加深编码器反而损害预测精度（→ SemTS 保持 4 层 ✓）；
   - 合成语料预训练常与真实语料相当。
   ⇒ **SemTS 的预测评估协议（冻结 + 岭回归）系统性低估了模型**；SOTA 基线全部走微调。
   对策：新增与基线同口径的"冻结 LP + 全量微调"预测协议（A2）。

2. **[arXiv 2609.32187] Rethinking Cross-Channel Importance**
   - 通道"相关 ≠ 有用 ≠ 被使用"；**按视野自适应选择源通道**使岭回归在 21/32 条件下提升
     （平均 +5.16%）；**有界后验支持集**改进冻结的通道独立预测器（12/16 条件）。
   ⇒ SemTS 在宽通道集（Traffic/Electricity）落后主因之一：全通道共享表示 + 无选择。
   对策：视野自适应通道选择岭回归（A3）。

3. **RevIN / SAN（非平稳归一化）**：分布漂移是长时预测核心难点；RevIN 的均值-方差归一化
   对趋势非平稳尤其有效。⇒ 评估与预训练加入逐窗口 RevIN（A1）。

4. **iTransformer（ICLR 2024）**：把变量当 token 的自注意力天然建模跨通道相关，
   在 Electricity/Traffic 上显著优于通道独立方法。⇒ 宽通道场景加"倒置通道注意力头"（C1）。

5. **SimMTM / ShuffleMTM**：掩码重建 + 序列级聚合的预训练在**微调协议**下有效；
   其官方成功还依赖 seq=96 短窗口 + RevIN + 100 epoch。⇒ 预算对齐（B1）、短窗口预训练（B2）。

## 1. 改进清单（按优先级）

### A. 评估/协议升级（成本低、风险小、直接作用于差距数字）
| ID | 改动 | 预期收益 | 状态 |
|---|---|---|---|
| A1 | finetune_forecast 加逐窗口 RevIN（编码前归一化，目标同为归一化未来） | ❌ 实测无效/有害（冻结编码器分布错位；见 A/C2 结论） | ❌ |
| A2 | 新增预测微调协议：冻结 LP 10 + 全量 FT 20（ShuffleMTM 同款），头 = 末 patch 表示线性层 | ✅ **验证有效（Exchange）**：0.164/0.326/0.645/2.499 → **0.150/0.281/0.601/1.968**（H96 −8.6%、H720 −21.2%），与 SOTA 差距大幅收窄；ETTh1/ETTm1/Traffic/Electricity 验证中 | 🔄 |
| A3 | 视野自适应通道选择岭回归（按验证 MSE 选 top-k 通道，逐视野） | ⚠️ 部分有效：PSM H96 −5.7%（0.4214）；Electricity +53%（0.364，321 通道高共线下硬选近共线源条件数爆炸）；ETTm1 无效 | ❌ 被 A3b 取代 |
| A3b ★ | 软通道选择岭回归（借鉴 DUET KDD'25 Channel-Soft-Clustering：softmax 连续权重 + 单聚合原型，特征仅 2d 维） | ✅ **PSM H96 0.4469→0.2646（−40.8%）**；**Electricity H96 0.237→0.228（−3.8%）**，零退化（修复了硬选择的 +53% 共线病）；全视野与 Traffic 验证中 | 🔄 |
| A4 | Exchange 数据用对数收益差分（金融序列标准处理） | ⚠️ 无效/不可比：H96 0.164→1.391 崩坏、H720 2.499→1.353（两口径尺度不同，且编码器在水平值上预训练存在分布错位）；Exchange 差距的主因是线性探针长视野能力，交由 A2/B1 解决 | ❌ |

### B. 预训练预算与配置（成本中，3090 跑）
| ID | 改动 | 预期收益 | 状态 |
|---|---|---|---|
| B1 | 预训练 epoch 10→100（预算对齐 SOTA） | 全面 +5~15% | ⬜ |
| B2 | 短窗口（L=96）预训练变体实验（与 ShuffleMTM 同窗口） | 待验证 | ⬜ |
| B3 | 保持深度 ≤4（文献：深度损害预测） | 维持现状 ✓ | ✅ |

### C. 架构升级（成本高，需消融验证）
| ID | 改动 | 预期收益 | 状态 |
|---|---|---|---|
| C1 | 宽通道（C>64）加倒置通道注意力头（iTransformer 式，S3 升级为全通道注意力 + 滞后可解释正则） | Traffic/Electricity +5~15% | ⬜ |
| C2 | 预训练管线内置逐窗口 RevIN（与 A1 一致化） | ❌ 实测负结果（见下） | ❌ |

**C2/A1 负结果记录（2026-10-25）**：
逐窗口 RevIN 的**全部四种组合均无效/有害**：
- 仅评估端（A1）：ETTh1 H96 −1.5%/H336 +3.2%；
- 管线内预训练+岭（C2，3ep A/B）：ETTm1 H336 0.468→0.873 严重退化；
- 微调+RevIN（A2+revin，10ep ckpt）：ETTh1 0.534/0.600/0.665/0.806，
  比纯微调（0.513/0.566/0.618/0.730）与岭（0.504/0.557/0.609/0.722）都差。
结论：编码器未在归一化空间预训练，任何后置 RevIN 均为分布错位；RevIN 仅在
"预训练与微调全程使用归一化空间 + 解码器反归一化"（ShuffleMTM 全栈）下成立，
对本项目不作采纳。

**A2 完整结论（2026-10-25，10-epoch 各数据集 ckpt，LP10+FT20）**：
- Exchange：全面有效（H96 0.164→0.150、H192 0.326→0.281、H336 0.645→0.601、
  H720 2.499→1.968）——协议性收益；
- ETTh1/ETTm1：无增益（±3% 内）——10-epoch 特征已达线性评估天花板，
  需 B1（更长预训练）提升特征本身。
| C3 | 掩码重建流（SimMTM 式序列级聚合）作为 S5 候选 | 待消融（统一骨干 simmtm 0.456 已被 SemTS 0.384 超越，优先级低） | ⬜ |

## 2. 验证协议
- 所有改动先在 4070 上以 ETTm1/ETTh1 快验（预训练 3 epoch + 岭回归/微调对比），
  确认增益后再上 3090 全量复跑；
- 对照基线固定：现行主配置（ablation_w2 ckpt）的既有数字作为 A/B 基准；
- 每项改动记录 ΔMSE（统一协议）与 ΔMSE（微调协议）两列。

## 3. 文献清单
- Major, Razmadze & Shavit, *When Does Self-Supervised Learning Transfer to Time-Series Tasks?*, arXiv:2605.19462 (2026). https://arxiv.org/abs/2605.19462
- Choi, Park & Cho, *Rethinking Cross-Channel Importance in Time-Series Forecasting*, arXiv:2609.32187 (2026). https://arxiv.org/abs/2609.32187
- Dong et al., *SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling*, NeurIPS 2023. https://papers.neurips.cc/paper_files/paper/2023/file/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Paper-Conference.pdf
- Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024. https://ise.thss.tsinghua.edu.cn/~mlong/doc/iTransformer-iclr24.pdf
- Kim et al., *Reversible Instance Normalization (RevIN)*, ICLR 2022；Liu et al., *Non-stationary Transformers*, NeurIPS 2022。
- Wang & Isola, *Understanding Contrastive Learning Requires Incorporating Loss Functions* (alignment/uniformity), ICML 2020. https://openreview.net/pdf?id=3S1kAqCiAS
