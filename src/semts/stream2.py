"""S2：模式头（④ 趋势/季节解耦）。CoST 路线的移植与简化。

- TFD 趋势解耦器：混合自回归专家（因果卷积核 1,2,4,...,2^(n-1)）→ 趋势表示，
  时域对比（NT-Xent + 负样本队列）。
- SFD 季节解耦器：DFT → 逐频率复值线性层（频内交互）→ iDFT → 季节表示，
  频域对比（幅值项 + 相位项，无需人工设定周期）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import NTXentQueue, freq_amp_loss, freq_phase_loss


class TrendDisentangler(nn.Module):
    """混合 AR 专家：核 2^i 的因果卷积 + 平均池化（自适应多尺度 lookback）。

    因果性通过 F.pad 左侧补 (k-1) 个 0 实现（Conv1d 的 padding 不支持二元组）。"""

    def __init__(self, d_model, d_trend, n_experts):
        super().__init__()
        self.kernels = [2 ** i for i in range(n_experts)]
        self.experts = nn.ModuleList(
            [nn.Conv1d(d_model, d_trend, kernel_size=k) for k in self.kernels]
        )

    def forward(self, z):
        """z: (B, C, P, d) → 趋势表示 (B, C, P, d_trend)。"""
        B, C, P, D = z.shape
        zf = z.reshape(B * C, P, D).transpose(1, 2)  # (B*C, D, P)
        outs = []
        for ex, k in zip(self.experts, self.kernels):
            x_p = F.pad(zf, (k - 1, 0))  # 左侧补 0 → 输出长度仍为 P（因果）
            outs.append(ex(x_p))
        out = torch.stack(outs, dim=0).mean(0)  # (B*C, dT, P)
        return out.transpose(1, 2).reshape(B, C, P, -1)


class SeasonalDisentangler(nn.Module):
    """DFT → 逐频率复值线性层（频内交互）→ iDFT。"""

    def __init__(self, d_model, d_season, n_freqs):
        super().__init__()
        self.d_season = d_season
        self.n_freqs = n_freqs
        self.Ar = nn.Parameter(torch.randn(n_freqs, d_season, d_model) * 0.02)
        self.Ai = nn.Parameter(torch.randn(n_freqs, d_season, d_model) * 0.02)
        self.br = nn.Parameter(torch.zeros(n_freqs, d_season))
        self.bi = nn.Parameter(torch.zeros(n_freqs, d_season))

    def transform(self, z):
        """z: (B, C, P, d) → 频域季节表示 F: (B, C, F, d_season) complex，F = min(P//2+1, n_freqs)。

        复值线性层用实部/虚部分解实现：y = A·z + b，其中
        yr = Ar·zr - Ai·zi + br，yi = Ar·zi + Ai·zr + bi。"""
        F = torch.fft.rfft(z, dim=2)  # (B, C, F, d)
        nf = min(F.size(2), self.n_freqs)
        zr, zi = F[:, :, :nf].real, F[:, :, :nf].imag
        Ar, Ai = self.Ar[:nf], self.Ai[:nf]
        br, bi = self.br[:nf], self.bi[:nf]
        yr = torch.einsum("fok,bcfk->bcfo", Ar, zr) - torch.einsum("fok,bcfk->bcfo", Ai, zi) + br[None, None]
        yi = torch.einsum("fok,bcfk->bcfo", Ar, zi) + torch.einsum("fok,bcfk->bcfo", Ai, zr) + bi[None, None]
        return torch.complex(yr, yi)

    def forward(self, z, n_time):
        """z: (B, C, P, d) → 季节表示 (B, C, P, d_season)（iDFT 回时域）。"""
        F = self.transform(z)
        return torch.fft.irfft(F, n=n_time, dim=2)


class Stream2(nn.Module):
    """趋势（时域对比）+ 季节（频域幅值/相位对比）。"""

    def __init__(self, d_model, d_trend=64, d_season=64, n_experts=6, n_freqs=64,
                 queue_size=256, alpha=1.0, tau=0.07, freq_tau=None):
        super().__init__()
        self.alpha = alpha
        self.d_season = d_season
        self.freq_tau = tau if freq_tau is None else freq_tau
        self.tfd = TrendDisentangler(d_model, d_trend, n_experts)
        self.sfd = SeasonalDisentangler(d_model, d_season, n_freqs)
        self.queue = NTXentQueue(d_trend, queue_size=queue_size, tau=tau)

    def forward(self, z_a, z_b):
        """z_a: 三态掩码视图，z_b: scale/shift/jitter 增强视图，(B, C, T, d)，同一窗口。

        返回 (l_time, l_amp, l_phase)。v0.1 简化：时域对比用序列级池化 + 队列，
        key 无动量编码器（CoST 原文为 MoCo）。"""
        B, C, T, d = z_a.shape
        t_a = self.tfd(z_a).mean(2).reshape(B * C, -1)  # 序列级趋势向量 (B*C, dT)
        t_b = self.tfd(z_b).mean(2).reshape(B * C, -1)
        l_time = self.queue(t_a, t_b)
        f_a = self.sfd.transform(z_a).reshape(B * C, -1, self.d_season)  # (B*C, F, dS)
        f_b = self.sfd.transform(z_b).reshape(B * C, -1, self.d_season)
        l_amp = freq_amp_loss(f_a, f_b, tau=self.freq_tau)
        l_phase = freq_phase_loss(f_a, f_b, tau=self.freq_tau)
        return l_time, l_amp, l_phase
