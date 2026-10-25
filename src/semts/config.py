"""SemTS 配置。"""
from dataclasses import dataclass


@dataclass
class SemTSConfig:
    # ---- 数据 ----
    L: int = 336                 # 预训练窗口长度（用于推导滞后搜索上界 K_eff = min(lag_K, L-1)）

    # ---- 骨干（通道独立 patch Transformer）----
    d_model: int = 256
    patch_len: int = 16
    stride: int = 8
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1

    # ---- S2 模式头 ----
    d_trend: int = 64
    d_season: int = 64
    n_experts: int = 6          # 混合 AR 专家数（卷积核 1,2,4,...,2^(n-1)）
    n_freqs: int = 64           # 频域线性层最大频率数（实际 F=P//2+1，自动截断）
    queue_size: int = 256       # 时域对比负样本队列
    alpha: float = 1.0          # 频域损失权重（幅值/相位）

    # ---- S3 关系头 ----
    n_bins: int = 12            # 兼容保留；实际由 SemTS 内部按 2*lag_K//bin_width 派生
    lag_K: int = 48             # 滞后搜索范围 [-K, K]
    tau_corr: float = 0.5       # 显著性阈值（峰值相关系数）
    k_min: int = 3              # 最小显著滞后
    bin_width: int = 8          # 滞后分桶宽度
    max_pairs: int = 512        # 每步最多通道对数（大通道数据集随机子采样，控制 S3 开销）

    # ---- S4 质量头 ----
    n_q: int = 4                # 噪声质量等级数
    mask_ratio: float = 0.25    # 掩码缺失比例（M）
    noise_ratio: float = 0.10   # 噪声注入比例（N）
    lam_q: float = 0.2          # 质量等级预测损失权重

    # ---- 视图构造 ----
    crop_min_frac: float = 0.6  # 随机窗口最小长度比例（batch 共享几何）
    p_mask: float = 0.5         # 隐向量时间戳掩码概率（TS2Vec 式）
    revin: bool = False         # 预训练管线内置逐窗口 RevIN（改进方案 C2：编码器学习
                                # 对窗口尺度/平移不变；评估时需同样开启 --revin）

    # ---- 损失权重 ----
    tau: float = 0.1            # 对比温度（S1/S2 时域）
    freq_tau: float = 0.07      # 频域对比温度（S2 幅值/相位，CoST 原值）
    lam_dir: float = 0.5        # 领先方向损失权重（S3 内部）
    w1: float = 1.0             # S1 层级对比
    w2: float = 1.0             # S2 模式
    w3: float = 0.5             # S3 关系
    w4: float = 1.0             # S4 质量
