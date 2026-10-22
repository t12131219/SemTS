"""数据增强与视图构造（四个语义流共用）。"""
from typing import Optional

import torch


def three_state_mask(x, mask_ratio=0.25, noise_ratio=0.10, n_q=4,
                     sigma_scale=(0.05, 0.1, 0.2, 0.4), seg_len=8):
    """三态质量掩码（S4 输入，⑦ 缺失/置信度语义）。

    state: 0=观测 O，1=掩码缺失 M（置 0），2=噪声注入 N（等级 q）。
    q: N 时间点的质量等级 0..n_q-1，其余 -1。
    顺序：先放 N 段、后放 M 掩码（M 覆盖 N，避免重叠歧义），N 与 M 重叠处 q 置 -1。
    返回 (x_aug, state, q)。缺失点置 0 但带状态嵌入，显式区分"缺失 ≠ 0"。
    """
    B, L, C = x.shape
    if n_q > len(sigma_scale):
        sigma_scale = tuple(sigma_scale) + tuple(
            sigma_scale[-1] * (2 ** (i + 1)) for i in range(n_q - len(sigma_scale))
        )
    state = torch.zeros(B, L, C, dtype=torch.long, device=x.device)
    q = torch.full((B, L, C), -1, dtype=torch.long, device=x.device)

    # 1) 噪声段 N（先放；段在每样本内互不重叠，避免 scatter 写入竞争）
    if noise_ratio > 0:
        n_seg = max(1, int(round(noise_ratio * L / seg_len)))
        n_seg = min(n_seg, max(1, L // seg_len))
        bin_len = L // n_seg
        off_max = max(1, bin_len - seg_len + 1)
        starts = (
            torch.arange(n_seg, device=x.device).view(1, -1) * bin_len
            + torch.randint(0, off_max, (B, n_seg), device=x.device)
        )  # (B, n_seg)：各段落在自己的 bin 内 → 段间不重叠
        qs = torch.randint(0, n_q, (B, n_seg), device=x.device)
        seg_idx = torch.arange(seg_len, device=x.device).view(1, -1)  # (1, seg_len)
        for s in range(n_seg):
            t_idx = (starts[:, s].view(B, 1) + seg_idx).clamp(max=L - 1)  # (B, seg_len)
            i3 = t_idx.unsqueeze(-1).expand(B, seg_len, C)  # (B, seg_len, C)
            state.scatter_(1, i3, torch.full((B, seg_len, C), 2, dtype=torch.long, device=x.device))
            q.scatter_(1, i3, qs[:, s].view(B, 1, 1).expand(B, seg_len, C))

    # 2) 掩码缺失 M（后放，覆盖 N）
    if mask_ratio > 0:
        m = torch.rand(B, L, 1, device=x.device) < mask_ratio
        state.masked_fill_(m.expand(-1, -1, C), 1)
        q.masked_fill_(m.expand(-1, -1, C), -1)  # M 覆盖 N 后清理 q

    x_aug = x.clone()
    x_aug[state == 1] = 0.0
    std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
    for ql in range(n_q):
        sel = (state == 2) & (q == ql)
        if sel.any():
            noise = torch.randn_like(x_aug)[sel] * sigma_scale[ql] * std.expand_as(x_aug)[sel]
            x_aug[sel] = x_aug[sel] + noise
    return x_aug, state, q


def scale_shift_jitter(x, scale_sigma=0.2, shift_sigma=0.2, jitter_sigma=0.2, jitter_p=0.2):
    """CoST 式增强：scale / shift / jitter —— 模拟对误差变量 E 的干预（④ 模式语义）。"""
    B, L, C = x.shape
    out = x.clone()
    std = x.std(dim=1, keepdim=True).clamp_min(1e-5)
    out = out * (1.0 + torch.randn(B, 1, C, device=x.device) * scale_sigma)
    out = out + torch.randn(B, 1, C, device=x.device) * shift_sigma * std
    jmask = torch.rand(B, L, C, device=x.device) < jitter_p
    out[jmask] = out[jmask] + torch.randn(B, L, C, device=x.device)[jmask] * jitter_sigma * std.expand_as(out)[jmask]
    return out


def random_crop(P_full: int, min_frac: float = 0.6,
                rng: Optional[torch.Generator] = None):
    """随机选择一个 patch 窗口（每个 batch 共享同一几何，两视图共用保证对齐）。

    返回 (s, p)：窗口 [s, s+p)。v0.1 简化：两视图共用同一窗口（TS2Vec 式双窗口裁剪留待 v0.2）。
    """
    lo = max(2, int(P_full * min_frac))
    hi = P_full
    if hi <= lo:
        return 0, hi
    p = int(torch.randint(lo, hi + 1, (1,), generator=rng).item())
    s = int(torch.randint(0, hi - p + 1, (1,), generator=rng).item())
    return s, p


def patch_to_timestamps(s: int, p: int, patch_len: int, stride: int):
    """patch 窗口 [s, s+p) 对应的输入时间戳范围 [t0, t1)。"""
    t0 = s * stride
    t1 = s * stride + (p - 1) * stride + patch_len
    return t0, t1


def latent_keep_masks(B: int, C: int, P: int, p_mask: float, device):
    """TS2Vec 式隐向量时间戳掩码：keep ∈ (B*C, P)，以概率 p_mask 置 0。"""
    return (torch.rand(B * C, P, device=device) > p_mask).float()


def add_state_quality_emb(x, state, q, es, eq):
    """把三态缺失状态与质量等级以标量嵌入加回输入。es: (3,)，eq: (n_q,)。"""
    return x + es[state] + eq[q.clamp(min=0)] * (state == 2).float()
