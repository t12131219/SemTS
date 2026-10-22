"""S3 弱标签生成：滞后互相关 → 跨通道 lead-lag 标签（⑤ 时间关系语义）。

约定（FFT 线性互相关惯例）：
    cc[m] = Σ_t xi[t] * xj[t - m]
即相关峰位置 m* < 0 表示 **i 领先 j**（x_j 滞后 x_i），m* > 0 表示 j 领先 i。
下游 DirectionHead 标签：0=i 领先 j，1=j 领先 i，2=无显著关系（|rho|<tau 或 |k|<k_min）。

FFT 索引映射（零填充至 2L 的 irfft 输出）：
    cc[q] = Σ_t xi[t]·xj[(t-q) mod 2L]
正滞后 m ∈ [0, K] 位于索引 [0, K]；负滞后 m ∈ [-K, -1] 位于尾部索引 [2L-K, 2L-1]。
"""
import torch


def lagged_cross_correlation(xi, xj, K):
    """xi, xj: (N, L)。零填充 FFT 计算滞后 m ∈ [-K, K] 的线性互相关。

    返回 (k_star, rho)：k_star ∈ [-K, K] 为相关峰位置（上述惯例），
    rho 为峰值归一化相关系数（近似 Pearson）。"""
    N, L = xi.shape
    n2 = 2 * L
    X = torch.fft.rfft(xi, n=n2)
    Y = torch.fft.rfft(xj, n=n2)
    cc = torch.fft.irfft(X * Y.conj(), n=n2)  # (N, 2L)，cc[q]=Σ_t xi[t]xj[(t-q) mod 2L]
    norm = torch.sqrt((xi ** 2).sum(1) * (xj ** 2).sum(1)).clamp_min(1e-8)
    cc = cc / norm.unsqueeze(1)
    K_eff = min(K, L - 1)
    idx_pos = torch.arange(0, K_eff + 1, device=xi.device)    # 滞后 0..K
    idx_neg = torch.arange(n2 - K_eff, n2, device=xi.device)  # 滞后 -K..-1（尾部）
    idx = torch.cat([idx_pos, idx_neg])
    krel = torch.cat([idx_pos, idx_neg - n2])
    c = cc[:, idx]
    kpos = c.argmax(dim=1)
    k_star = krel[kpos]
    rho = c.gather(1, kpos.unsqueeze(1)).squeeze(1)
    return k_star, rho


def _soft_bin_label(k_star, bin_centers, bin_width):
    """以 k* 为中心的高斯邻域软标签（sigma = 0.5 bin），按行归一化。"""
    dist = (bin_centers.unsqueeze(0) - k_star.float().unsqueeze(1)) / bin_width
    soft = torch.exp(-(dist ** 2) / (2 * 0.5 ** 2))
    return soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-8)


def compute_lag_labels(x, K, tau_corr=0.5, k_min=3, bin_width=8, max_pairs=None):
    """x: (B, L, C) 干净输入（z-score 后）。返回通道对标签列表，每项 dict：

    i, j      通道索引
    k_star    (B,) 相关峰位置
    rho       (B,) 峰值相关系数
    soft      (B, n_bins) 高斯邻域软标签（和为 1）
    dir       (B,) 0=i 领先 j，1=j 领先 i，2=无显著（不显著 = 低相关或小滞后）
    valid     (B,) bool：|k*|>=k_min 且 |rho|>=tau_corr（滞后回归有效样本）
    """
    B, L, C = x.shape
    n_bins = 2 * K // bin_width
    bin_centers = (
        torch.arange(n_bins, device=x.device).float() * bin_width - K + bin_width / 2.0
    )
    pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]
    if max_pairs is not None and len(pairs) > max_pairs:
        # 用 torch.randperm（受 manual_seed 控制）保证子采样可复现
        perm = torch.randperm(len(pairs), device=x.device)[:max_pairs].tolist()
        pairs = [pairs[i] for i in perm]

    labels = []
    for i, j in pairs:
        ks, rho = lagged_cross_correlation(x[:, :, i], x[:, :, j], K)
        valid = (rho.abs() >= tau_corr) & (ks.abs() >= k_min)
        soft = _soft_bin_label(ks, bin_centers, bin_width)
        # 方向：仅对"显著"对给 0/1，其余归入"无显著"类（与设计 §3.4 一致）
        dir_ = torch.where(
            valid,
            torch.where(ks < 0, torch.zeros_like(ks), torch.ones_like(ks)),
            2 * torch.ones_like(ks),
        )
        labels.append({"i": i, "j": j, "k_star": ks, "rho": rho,
                       "soft": soft, "dir": dir_, "valid": valid})
    return labels
