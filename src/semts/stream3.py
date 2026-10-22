"""S3：关系头（⑤ 跨通道滞后/因果）★创新点。

与现有工作的区别：
- Lag-Llama 把滞后当"输入特征"；CoST 的相位项只用于区分季节模式；
- 本模块首次把"跨通道 lead-lag 结构"本身作为显式预训练目标，
  使表示内部编码时序关系信息。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lag_labels import compute_lag_labels
from .losses import soft_cross_entropy


class RelationHead(nn.Module):
    """通道对 (i, j) 的表示 → 滞后 bin 软回归 + 领先方向分类。"""

    def __init__(self, d_model, n_bins, hidden=256):
        super().__init__()
        self.n_bins = n_bins
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_bins + 3),
        )

    def forward(self, hi, hj):
        """hi, hj: (B, d)。返回 (lag_logits (B, n_bins), dir_logits (B, 3))。"""
        feat = torch.cat([hi, hj, hi * hj, (hi - hj).abs()], dim=-1)
        out = self.mlp(feat)
        return out[:, : self.n_bins], out[:, self.n_bins:]


def stream3_loss(head, z, x_clean, K=48, tau_corr=0.5, k_min=3, bin_width=8,
                 max_pairs=None):
    """z: (B, C, T, d) 干净/三态掩码视图；x_clean: (B, L, C) 原始输入（生成弱标签）。

    返回 (l_lag, l_dir)：
    - l_lag：仅对显著通道对（|k*|>=k_min 且 |rho|>=tau_corr）计算滞后软回归；
    - l_dir：对所有通道对计算方向分类（含"无显著"类，保证负样本信息）。
    """
    B, C, T, d = z.shape
    h = z.mean(dim=2)  # (B, C, d) 通道摘要
    labels = compute_lag_labels(x_clean, K, tau_corr, k_min, bin_width,
                                max_pairs=max_pairs)

    l_lag_total = torch.zeros((), device=z.device)
    l_dir_total = torch.zeros((), device=z.device)
    n_valid_pairs = 0
    for lab in labels:
        lag_logits, dir_logits = head(h[:, lab["i"]], h[:, lab["j"]])
        if lab["valid"].any():
            l_lag_total = l_lag_total + soft_cross_entropy(lag_logits, lab["soft"], lab["valid"])
            n_valid_pairs += 1
        l_dir_total = l_dir_total + F.cross_entropy(dir_logits, lab["dir"])
    l_lag = l_lag_total / max(n_valid_pairs, 1)
    l_dir = l_dir_total / max(len(labels), 1)
    return l_lag, l_dir
