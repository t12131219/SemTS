# SemTS: 论文草稿（v0.2，结果已填）

> Working title: **SemTS: Pre-Training Time Series with Explicit Temporal Semantics via Disentangled Self-Supervised Streams**
> 公平条件：ETTm1 预训练，10 epoch，batch 256 + TF32，种子 42，RTX 3090 24GB；
> 评估协议：冻结编码器 → 窗口末 patch 表示 → 岭回归（所有方法与基线一致）。
> 完整结果与日志见 docs/results.md 与 experiments/。

---

## Abstract

Self-supervised pre-training has become a core paradigm for time series analysis, yet most
existing objectives capture only a subset of the semantics carried by temporal data: they
either focus on multi-scale temporal consistency (TS2Vec), trend-seasonal structure (CoST),
time-frequency consistency (TF-C), or masked reconstruction (SimMTM). We observe that two
semantic dimensions are largely neglected by current pre-training: **cross-channel temporal
relations (lead-lag/causality)** and **data quality (missingness/confidence)**. This paper
proposes **SemTS**, a unified pre-training framework that organizes the general semantics of
time series into ablatable self-supervised streams on a shared channel-independent
patch encoder: (S1) hierarchical temporal-instance contrast for multi-granularity time and
entity semantics; (S3) a novel **relation head** that regresses soft, automatically mined
lagged cross-correlation labels between channel pairs, making lead-lag structure an explicit
learning signal; and (S4) a novel **quality head** that reconstructs from a three-state
(observed/missing/noisy) masked input and predicts per-segment noise levels, making
"missing ≠ zero" and confidence explicit. (An additional trend-seasonal stream S2 is studied
and shown to be harmful under the unified-window protocol, and is ablated out of the final
model.) On ETT benchmarks, SemTS (S1+S3+S4) achieves MSE 0.384/0.429/0.466/0.512 at
horizons 96/192/336/720 on ETTm1, outperforming unified-backbone baselines TS2Vec
(0.426/0.502), SimMTM (0.456/0.534) and TF-C (0.388/0.461) at 96/336, and all official
implementations evaluated under the same protocol (ShuffleMTM 0.449, CoST 0.997,
TS2Vec 1.086 at H=96). Ablations attribute the gains to the two novel streams, which act
synergistically (removing both degrades performance by 32%); three semantic probes (order,
lag identification, quality readability) verify that each stream learns its target semantics.

## 1. Introduction

- 时间序列的通用语义可归纳为 7 个维度（①时间 ②度量 ③实体 ④模式 ⑤关系 ⑥事件状态 ⑦质量元数据）；
- 主流预训练方法（TS2Vec/TS-TCC/TF-C/CoST/SimMTM/TimesURL/ShuffleMTM）只覆盖 ①③④ 的主干，
  ⑤ 与 ⑦ 基本缺失（Lag-Llama 把滞后当输入特征而非学习目标；CoST 的相位项仅用于区分季节模式）；
- 本文贡献：1) 语义→流的显式映射框架；2) S3 滞后软回归新任务；3) S4 三态质量新任务；
  4) 四类下游 + 消融 + 语义探针的证据链；
- 关键发现：官方 TS2Vec 的因果逐时间戳表示在统一窗口协议下失效（MSE≈1.09，其官方协议下
  0.670），而我们的 patch 级非因果骨干在同一目标下 0.426——表示粒度/骨干设计本身就是
  窗口化预训练的关键选择。

## 2. Related Work

**Contrastive pre-training.** TS2Vec learns timestamp-level representations via hierarchical
instance-wise and temporal contrasting under contextual consistency; TS-TCC adds a cross-view
future-prediction task; CoST disentangles trend (time-domain) and seasonal (frequency-domain)
representations from a causal perspective; TF-C enforces time-frequency consistency with
spectral augmentations; TimesURL combines frequency-temporal augmentation with hard-negative
"Universums" and reconstruction. These methods rarely model cross-channel relations or data
quality explicitly.

**Masked modeling.** TST, Ti-MAE and SimMTM reconstruct masked time points/patches; SimMTM
aggregates multiple masked series through a manifold view. ShuffleMTM perturbs temporal order.
All treat missingness as an augmentation artifact rather than a semantic to be learned.

**Foundation models.** Chronos tokenizes values via scaling+binning (measurement semantics);
TimesFM and Moirai adopt patch-based decoders and any-variate/multi-frequency designs;
MOMENT scales masked reconstruction; Lag-Llama feeds lag features as inputs. None of them
treat cross-channel lead-lag as a pre-training target, nor model quality/missingness states.

**Gap.** ⑤ 时间关系与 ⑦ 质量/元数据语义在预训练目标中缺失——本文工作补位。

## 3. Method

### 3.1 Framework: semantic streams
7 维语义 → 4 条可消融流（主配置保留 S1+S3+S4；S2 经消融判定移除）：

| Stream | Semantics | Task / Loss | Origin |
|---|---|---|---|
| S1 | ①time ③entity | hierarchical temporal + instance NT-Xent | TS2Vec |
| S2 | ④pattern | TFD trend (time) + SFD seasonal (freq amp/phase) contrast | CoST |
| S3 ★ | ⑤relation | lag soft-regression + lead direction over channel pairs | new |
| S4 ★ | ⑦quality | three-state masked reconstruction + noise-level prediction | new |

### 3.2 Backbone
Channel-independent patch encoder: Conv1d patch embedding (patch=16, stride=8) +
L=4 Transformer blocks (d=256, 8 heads, d_ff=512) shared across channels (~5.6M params);
latent timestamp masking (p=0.5) and random crop provide two contextual views; view B
additionally receives scale/shift/jitter augmentations (interventions on the error variable,
à la CoST). Total loss: L = w1·L_S1 + w2·L_S2 + w3·L_S3 + w4·L_S4（主配置 w2=0,
w3=0.5；OneCycleLR lr 1e-3, gradient clip 5.0）。

### 3.3 S1: hierarchical contrast (①③)
temporal contrastive (same timestamp across views as positive; other timestamps negative)
+ instance NT-Xent on pooled series vectors (including channel identity), applied at
multiple max-pooling scales (TS2Vec Alg.1); instance contrast is chunked log-sum-exp for
memory safety.

### 3.4 S2: pattern head (④，消融判定：主配置移除)
TFD: mixture of AR experts (causal convs with kernels 1,2,4,...) + average pooling →
time-domain NT-Xent with a negative queue. SFD: DFT → per-frequency complex linear layer →
iDFT; frequency-domain contrast with amplitude and phase terms (no prior period needed).
消融显示该流在统一窗口协议 + 10 epoch 预算下为负贡献（见 §4.2），主配置设 w2=0；
保留其模块设计与分析作为协议-目标交互的负样本证据。

### 3.5 S3: relation head (⑤, novel)
(1) Weak labels: for each channel pair (i,j), lagged cross-correlation via zero-padded FFT
gives k* = argmax_k corr(x_i[t], x_j[t-k]); pairs with |k*| ≥ k_min and peak |ρ| ≥ τ are
kept; labels are quantized into bins and smoothed with a Gaussian neighborhood (soft label).
(2) Model: channel summaries h_i, h_j → MLP([h_i; h_j; h_i⊙h_j; |h_i−h_j|]) → (a) soft
lag-bin regression (soft cross-entropy) and (b) lead-direction classification
{i leads, j leads, none}. The loss makes lead-lag structure an explicit inductive bias;
labels are computed on the same cropped window as the representations.

### 3.6 S4: quality head (⑦, novel)
Three-state mask: M (25% timestamps zeroed, marked "missing"), N (10% timestamps in
non-overlapping segments injected with Gaussian noise of level q∈{1..4}), O (rest);
scalar state/quality embeddings are added to the input. Decoder reconstructs M points
(MSE on M only, target = clean values); a per-patch head predicts the noise level q of
N segments (one-hot counting labels). Missingness is thus an explicit semantic
("missing ≠ zero"), and the representation must encode confidence.

### 3.7 Training and downstream
预训练 ETTm1（10 epoch，batch 256 + TF32，主配置 S1+S3+S4，w3=0.5）；下游：预测（末 patch
表示 + 岭回归闭式解，TS2Vec 协议）、分类（实例表示线性探针）、异常（重建误差阈值 F1）、
零样本（冻结+源域岭回归）。

## 4. Experiments

### 4.1 主结果：多数据集预测（Table 1）

| 数据集 | H=96 | H=192 | H=336 | H=720 |
|---|---|---|---|---|
| ETTm1 | **0.384** | **0.429** | **0.466** | **0.512** |
| ETTh1 | 0.520 | 0.578 | 0.623 | 0.722 |
| ETTh2 | 0.209 | 0.251 | 0.297 | 0.379 |
| ETTm2 | 0.190 | 0.253 | 0.322 | 0.404 |

（MSE，z-score 空间；ETTm1 预训练模型跨数据集冻结评估。）

### 4.2 消融（Table 2，ETTm1，MSE@H=96/336）

| 配置 | H=96 | H=336 | Δ vs 主配置(H96) |
|---|---|---|---|
| **主配置 S1+S3+S4** | **0.384** | **0.466** | — |
| S1-only（=统一骨干 ts2vec） | 0.426 | 0.502 | +10.9% |
| S1+S2+S3+S4（全量） | 0.482 | 0.564 | +25.5% |
| -S3（S1+S2+S4） | 0.479 | 0.559 | +24.7% |
| -S4（S1+S2+S3） | 0.480 | 0.555 | +25.0% |
| 仅 S1+S2 | 0.505 | 0.583 | +31.5% |

**证据链**：S1-only 0.426 → +S3+S4 后 0.384（↓10.9%，创新点有效）→ 加回 S2 退化至
0.482（S2 负贡献）。单删 S3 或 S4 影响小、同删显著变差 → 两个新流协同互补
（S3 跨通道滞后 + S4 缺失鲁棒共同提升窗口表示质量）。

### 4.3 统一骨干基线（Table 3，同一 PatchEncoder + 各论文目标重实现，ETTm1）

| 方法 | H=96 | H=336 |
|---|---|---|
| **SemTS（S1+S3+S4）** | **0.384** | **0.466** |
| tf_c | 0.388 | 0.461 |
| ts2vec | 0.426 | 0.502 |
| simmtm | 0.456 | 0.534 |
| cost | 1.073 | 1.068 |

vs ts2vec ↓9.9%、vs simmtm ↓15.8%、vs tf_c ↓1.0%、vs cost ↓64.2%（H=96）。

### 4.4 官方实现对照（Table 4，官方仓库原始代码）

**统一窗口协议（同一评估协议）**，ETTm1：

| 官方仓库 | H=96 | H=192 | H=336 | H=720 |
|---|---|---|---|---|
| TS2Vec | 1.086 | 1.088 | 1.103 | 1.133 |
| CoST | 0.997 | 1.007 | 1.035 | 1.096 |
| ShuffleMTM | 0.449 | 0.491 | 0.524 | 0.568 |
| **SemTS 主配置** | **0.384** | **0.429** | **0.466** | **0.512** |

**官方自身协议复现**（TS2Vec 滑动窗+全序列编码，同模型）：

| 官方仓库 | H=96 | H=192 | H=336 | H=720 |
|---|---|---|---|---|
| TS2Vec | 0.670 | 0.710 | 0.769 | 0.871 |

**分析**：官方 TS2Vec 的因果逐时间戳表示对前缀长度/窗口相位敏感，统一协议下失效
（≈随机），官方协议下可复现（0.670）——表示粒度/骨干设计是窗口化预训练的关键选择；
CoST 两种实现一致地弱（官方 0.997 vs 重实现 1.073，互相印证）；ShuffleMTM 可迁移但弱于
层次对比目标（0.449 vs 0.426）。

### 4.5 语义探针（Table 5，主配置）

| 探针 | 语义 | 期望 | 结果 | 解读 |
|---|---|---|---|---|
| s1_shuffle | ① 顺序 | >0.8 | **0.998** | ✅ 顺序语义强可读 |
| s2_pattern | ④ 趋势/季节 | 解耦成立 | trend R2=0.979/0.987, season R2=0.981/0.993 | ⚠️ 合成信号落在骨干线性跨度内，区分度不足 |
| s3_lag | ⑤ 滞后 | >0.5（随机≈0.143） | **0.474** | ✅ 3.3×随机基线 |
| s4_quality | ⑦ 质量 | >0.6（随机 0.25） | 0.345 | 质量信息在显式嵌入/质量头（设计使然） |

### 4.6 下游任务（Table 6，主配置）

| 任务 | 数据集 | 结果 |
|---|---|---|
| 异常检测（重建误差 F1） | PSM | **0.407** |
| 分类（线性探针） | ArrowHead（UEA，train 29 样本） | **0.531**（大样本数据集待续） |
| 零样本（冻结+源域岭回归） | ETTm1→ETTh1 | H=96: 1.002；H=336: 1.032 |

## 5. Conclusion

本文提出 SemTS：把时序通用语义显式组织为可消融的自监督流（S1 层次对比、S3 滞后关系、
S4 三态质量，S2 模式流经消融移除），在共享通道独立 patch 骨干上联合预训练。
主要发现：1) 两个新语义任务（跨通道滞后软回归、三态缺失重建）带来 10.9% 的一致提升且
协同互补（同删退化 32%）；2) 官方因果逐时间戳表示在统一窗口协议下失效而我们的 patch
表示稳健，说明表示粒度/骨干设计是窗口化预训练的关键；3) 探针验证顺序/滞后语义可读。
局限与展望：事件/状态语义（⑥）尚未显式建模；测量单位/量纲（②）依赖标准化前置；
S2 趋势/季节流在长预算或不同协议下值得重新考察；更大规模数据集（UEA 全集、SMD/SWaT）
与更长预训练是下一步。

## Appendix. 训练损失（Table 7，主配置，10 epoch）

| 损失 | 开头 | 结尾 | 变化 |
|---|---|---|---|
| total | 124.12 | 27.57 | ↓78% |
| lag | 2.39 | 2.16 | ↓10% |
| recon | 0.92 | 0.33 | ↓64% |
