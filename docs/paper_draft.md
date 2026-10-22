# SemTS: 论文草稿（v0.1，结果待填）

> Working title: **SemTS: Pre-Training Time Series with Explicit Temporal Semantics via Four Disentangled Self-Supervised Streams**
> 结果占位符以 `<待填>` 标注；实验命令见 docs/experiments_plan.md。

---

## Abstract

Self-supervised pre-training has become a core paradigm for time series analysis, yet most
existing objectives capture only a subset of the semantics carried by temporal data: they
either focus on multi-scale temporal consistency (TS2Vec), trend-seasonal structure (CoST),
time-frequency consistency (TF-C), or masked reconstruction (SimMTM). We observe that two
semantic dimensions are largely neglected by current pre-training: **cross-channel temporal
relations (lead-lag/causality)** and **data quality (missingness/confidence)**. This paper
proposes **SemTS**, a unified pre-training framework that organizes the general semantics of
time series into **four ablatable self-supervised streams** on a shared channel-independent
patch encoder: (S1) hierarchical temporal-instance contrast for multi-granularity time and
entity semantics; (S2) disentangled trend-seasonal contrast in time and frequency domains for
pattern semantics; (S3) a novel **relation head** that regresses soft, automatically mined
lagged cross-correlation labels between channel pairs, making lead-lag structure an explicit
learning signal; and (S4) a novel **quality head** that reconstructs from a three-state
(observed/missing/noisy) masked input and predicts per-segment noise levels, making
"missing ≠ zero" and confidence explicit. On ETT benchmarks, SemTS achieves `<待填>` and
ablation studies attribute gains to the two novel streams; four semantic probes (order,
trend-season separation, lag identification, quality readability) verify that each stream
learns its target semantics.

## 1. Introduction

- 时间序列的通用语义可归纳为 7 个维度（①时间 ②度量 ③实体 ④模式 ⑤关系 ⑥事件状态 ⑦质量元数据）；
- 主流预训练方法（TS2Vec/TS-TCC/TF-C/CoST/SimMTM/TimesURL/ShuffleMTM）只覆盖 ①③④ 的主干，
  ⑤ 与 ⑦ 基本缺失（Lag-Llama 把滞后当输入特征而非学习目标；CoST 的相位项仅用于区分季节模式）；
- 本文贡献：1) 语义→流的显式映射框架；2) S3 滞后软回归新任务；3) S4 三态质量新任务；
  4) 四类下游 + 消融 + 语义探针的证据链。

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

### 3.1 Framework: four semantic streams
7 维语义 → 4 条可消融流（表 + 架构图占位）：
| Stream | Semantics | Task / Loss | Origin |
|---|---|---|---|
| S1 | ①time ③entity | hierarchical temporal + instance NT-Xent | TS2Vec |
| S2 | ④pattern | TFD trend (time) + SFD seasonal (freq amp/phase) contrast | CoST |
| S3 ★ | ⑤relation | lag soft-regression + lead direction over channel pairs | new |
| S4 ★ | ⑦quality | three-state masked reconstruction + noise-level prediction | new |

### 3.2 Backbone
Channel-independent patch encoder: Conv1d patch embedding (patch=16, stride=8) +
L=4 Transformer blocks (d=256, 8 heads, d_ff=512) shared across channels; latent timestamp
masking (p=0.5) and random crop provide two contextual views; view B additionally receives
scale/shift/jitter augmentations (interventions on the error variable, à la CoST).
Total loss: L = w1·L_S1 + w2·L_S2 + w3·L_S3 + w4·L_S4 (OneCycleLR, gradient clip 5.0).

### 3.3 S1: hierarchical contrast (①③)
temporal contrastive (same timestamp across views as positive; other timestamps negative)
+ instance NT-Xent on pooled series vectors (including channel identity), applied at
multiple max-pooling scales (TS2Vec Alg.1); instance contrast is chunked log-sum-exp for
memory safety.

### 3.4 S2: pattern head (④)
TFD: mixture of AR experts (causal convs with kernels 1,2,4,...) + average pooling →
time-domain NT-Xent with a negative queue. SFD: DFT → per-frequency complex linear layer →
iDFT; frequency-domain contrast with amplitude and phase terms (no prior period needed).

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
预训练 ETTm1 (10 epochs, batch 128, w3=0.5)；下游：预测（末 patch 表示 + 岭回归闭式解，
TS2Vec 协议）、分类（实例表示线性探针）、异常（重建误差阈值 F1）、零样本（冻结+源域岭回归）。

## 4. Experiments (protocol; results in docs/results.md)

- 主表 1 预测（ETT×4, H∈{96,192,336,720}）：`<待填>`
- 消融：-S3 / -S4 / 仅 S1+S2 / 随机初始化：`<待填>`
- 语义探针（s1_shuffle 乱序可分性 / s2_pattern 趋势季节解耦 R² / s3_lag 滞后 bin 分类 /
  s4_quality 质量等级线性可分）：`<待填>`
- 统一骨干基线（ts2vec/cost/simmtm/tf_c）：`<待填>`
- 分类（UEA ArrowHead 等）/ 异常（PSM, SMD）/ 零样本（ETTm1→ETTh1）：`<待填>`

## 5. Conclusion

`<待填：语义流框架总结 + 两个新任务的有效性 + 局限与展望（事件状态⑥、单位量纲②、顺序扩展）>`
