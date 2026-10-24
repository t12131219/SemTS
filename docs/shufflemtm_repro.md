# ShuffleMTM (ICLR 2025) 论文表格实验复现计划

> 范围（用户确认）：**ShuffleMTM 官方代码 + SemTS** 跑全部表格实验；其余基线
> （TimeSiam/PITS/PatchTST/SimMTM/COMET/iTransformer/Crossformer/CrossGNN/MTGNN）
> 直接引用论文已报告数字。
> 官方代码 = github.com/cikm2025shufflemtm/ShuffleMTM（GitHub 上唯一公开仓库，
> 与论文附录 B 的实现细节一致；预测任务有官方入口，分类任务需基于官方 Model 类自建驱动）。

## 1. 论文表格清单

| 表格 | 实验 | 数据集 | 协议 |
|---|---|---|---|
| Table 1 | 域内长时预测（MSE/MAE） | ETTh1/ETTh2/ETTm1/ETTm2/Exchange/Weather/Electricity/Traffic，H∈{96,192,336,720} | 各数据集自预训练 + 微调（LP 10 + FT 20 epoch） |
| Table 2 | 跨域预测（MSE/MAE） | 8 对：ETTh1→ETTh2、ETTh2→ETTh1、ETTh1→ETTm1、ETTh2→ETTm1、ETTm2→ETTm1、ETTm2→ETTh1、Weather→ETTh1、Weather→ETTm1 | 源域预训练 ckpt + 目标域微调 |
| Table 3 | 域内分类（Acc/P/R/F1） | AD（EEG 16ch×256）、PTB（ECG 15ch×300） | 自预训练 + LP+FT（20/40 或 30/60 epoch） |
| Table 4 | 少量标签分类 | AD、PTB，标签比例 {20%, 50%, 100%} | 同上，仅子集标签 |

## 2. 官方协议参数（论文附录 Table 7/8 + 官方脚本）

- 预测：seq_len=96（官方脚本未覆盖默认值），patch_len∈{6,8}、stride=patch、
  e_layers=3、d_model∈{16,32}、d_ff∈{32,64}、n_heads∈{8,16}；mask_ratio=0.4；
  预训练 100 epoch、lr∈{1e-4,5e-4}（用 5e-4）、batch 64（大集 32）；
  微调 lr 5e-4、L2、freeze(线性探针) 10 + full FT 20 epoch。
- 分类：patch_len {32,64}/{15,30}、d_model {64,128}/{64,128,256}、mask_ratio {0.4,0.8}、
  预训练 100 epoch lr {5e-4,1e-3} batch 256；微调 CE batch 128，LP/FT {20/40, 30/60}。
- 官方脚本笔误注意：etth1.sh 的 `--finetune_epoochs` 应为 `--finetune_epochs`。

## 3. 数据获取

> 注意：COMET README/TimeSiam 引用的 figshare 处理数据链接已失效（API 404）。
> 改为**原始数据 + 按 COMET 流程预处理**（脚本已在本仓库 scripts/ 下）：

| 数据集 | 来源 | 预处理脚本 | 状态 |
|---|---|---|---|
| ETT×4 | 已有 data/ETTm1.csv 等 | — | ✅ |
| Exchange/Weather/Electricity/Traffic | Autoformer Google Drive 文件夹 1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy（gdown --folder） | — | ⬜ 3090 下载 |
| AD（原始，51MB） | OSF: https://osf.io/download/xy8qa/ | scripts/prep_ad_data.py | ✅ 已处理，**5967 样本与论文 Table 6 精确一致** |
| PTB（原始，~1.7GB） | PhysioNet: https://physionet.org/files/ptbdb/1.0.0/（递归下载文件树；get-zip 是 12 通道 Training_PTB 子集，不适用） | scripts/prep_ptb_data.py（wfdb+neurokit2） | ✅ 已处理，**198 患者精确匹配**，64356 样本（论文 62370，+3.2%，neurokit 版本差异） |

预处理的 AD/PTB 与已下载的 ETT 可整体打包给 3090，省去其下载环节。

## 4. 执行顺序（3090 Ubuntu）

> **交接清单（本机已完成的工件 → 3090）**：
> 1. `git pull`（代码与脚本；含官方仓库补丁脚本）；
> 2. 拷贝 `data/shufflemtm/`（AD/PTB 已处理：AD 5967 样本、PTB 198 患者/64356 样本，
>    含 AD_classify/PTB_classify 转换产物）到 3090 同名路径（`data/` 不入 git）；
> 3. `bash third_party/setup_repos.sh`（官方仓库补丁：切分 0.6/0.2/0.2 + NumPy2 的 np.Inf 修复）。

```bash
bash scripts/download_shufflemtm_data.sh        # ① 数据（若已拷贝 data/shufflemtm 则可跳过）
bash third_party/run_shufflemtm_tables.sh       # ② ShuffleMTM 官方：8 预训练 + 32 微调 + 8 跨域
bash third_party/run_shufflemtm_classify.sh     # ③ 分类：ShuffleMTM 官方模型驱动 + SemTS 探针
bash scripts/run_semts_tables.sh                # ④ SemTS：8 数据集预训练 + 岭回归
bash scripts/run_semts_crossdomain.sh           # ④.5 SemTS 跨域（Table 2 的 SemTS 列，8 对）
python scripts/aggregate_shufflemtm_tables.py   # ⑤ 汇总四张对比表（docs/shufflemtm_tables_results.md）
```

> **⚠️ ② 重跑前必须清空旧的官方结果文件**（早期运行把跨域结果追加混入了域内文件）：
> `rm -rf third_party/ShuffleMTM/outputs/test_results`
> 修复后的脚本用 `_cross.txt` 后缀分离两类结果，且支持幂等续跑。

> 协议说明（用户确认）：**SemTS 侧使用其擅长的配置**（S1+S3+S4、L=336、batch 256+TF32、
> 10 epoch、统一岭回归/线性探针），**不模仿 ShuffleMTM 的 100-epoch 预训练+全量微调协议**；
> 官方 ShuffleMTM 侧保持其官方协议以维持"官方实现"口径。

## 5. 时间估算（单 3090）

- ShuffleMTM 预训练 100 epoch × 8 数据集：小集 ~2h、Electricity ~5-8h、Traffic ~8-15h
  → 合计 **~30-45h**；微调 40 次 ~8h；分类（2 数据集 × 3 标签比例 + 全量）~8h。
- SemTS：8 预训练（10 ep）+ 评估 ~12h；分类探针 ~1h。
- **总计约 60-75 GPU 小时 ≈ 2.5-3 天连续运行**（5-run 平均在论文中要求，这里每格跑 1 次，
  时间不允许时注明为单次结果）。
