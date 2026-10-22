"""S4：质量头（⑦ 缺失/置信度）★创新点。

三态掩码（观测 O / 缺失 M / 噪声 N，N 分 n_q 个质量等级）+ 状态/质量嵌入 →
任务 1：重建被 M 掩码的时间点（MSE 仅算 M 点，MAE 式，目标为干净原始值）；
任务 2：预测 N 段的噪声质量等级（分类，只对 N 点 one-hot 计数取 argmax）。
学习信号：显式区分"缺失 ≠ 0、低质量 ≠ 正常"。

输入约定：x_win / state_win / q_win 均为 (B, C, Lw) 布局（semts.py 中已转置）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityHead(nn.Module):
    def __init__(self, d_model, patch_len, n_q=4):
        super().__init__()
        self.patch_len = patch_len
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, patch_len)
        )
        self.q_head = nn.Linear(d_model, n_q)

    def forward(self, z):
        """z: (B, C, P, d) → vals: (B, C, P, patch_len) 逐点重建，
        qlogits: (B, C, P, n_q) 逐 patch 质量等级 logits。"""
        vals = self.decoder(z)
        qlogits = self.q_head(z)
        return vals, qlogits


def reconstruction_loss(vals, x_win, state_win, stride, patch_len):
    """仅在被掩码（state==1）的时间点上计算 MSE（目标为干净原始值）。

    vals: (B, C, P, patch_len)；x_win/state_win: (B, C, Lw)。
    patch p 覆盖窗口时间戳 [p*stride, p*stride+patch_len)。"""
    B, C, P, pl = vals.shape
    Lw = x_win.size(2)
    off = torch.arange(pl, device=vals.device).view(1, -1)
    ts = torch.arange(P, device=vals.device).view(P, 1) * stride + off  # (P, pl)
    msel = state_win == 1
    loss = torch.zeros((), device=vals.device)
    cnt = 0
    for p in range(P):
        t = ts[p]
        v = t < Lw
        if not v.any():
            continue
        tt = t[v]
        pred = vals[:, :, p, v]
        tgt = x_win[:, :, tt]
        m = msel[:, :, tt]
        loss = loss + ((pred - tgt) ** 2 * m).sum()
        cnt = cnt + m.sum()
    return loss / cnt.clamp_min(1)


def patch_quality_labels(q_win, state_win, stride, patch_len, n_q):
    """每 patch 的质量等级标签：只统计该 patch 覆盖的 N 时间点，one-hot 计数取 argmax；
    无 N 点则 -1（忽略）。q_win/state_win: (B, C, Lw) → labels: (B, C, P)。"""
    B, C, Lw = q_win.shape
    P = (Lw - patch_len) // stride + 1
    labels = torch.full((B, C, P), -1, dtype=torch.long, device=q_win.device)
    off = torch.arange(patch_len, device=q_win.device)
    for p in range(P):
        t = p * stride + off
        t = t[t < Lw]
        q_patch = q_win[:, :, t]            # (B, C, k)
        s_patch = state_win[:, :, t] == 2   # (B, C, k)
        cnt = torch.zeros(B, C, n_q, device=q_win.device)
        for ql in range(n_q):
            cnt[..., ql] = ((q_patch == ql) & s_patch).sum(-1)
        best = cnt.argmax(-1)               # (B, C)
        has = s_patch.any(-1)               # (B, C)
        labels[:, :, p] = torch.where(has, best, labels[:, :, p])
    return labels


def stream4_loss(vals, qlogits, x_win, state_win, q_win, stride, patch_len, n_q):
    """返回 (l_recon, l_qpred)。组合权重（lam_q）在 SemTS 总装中处理。"""
    l_recon = reconstruction_loss(vals, x_win, state_win, stride, patch_len)
    plabels = patch_quality_labels(q_win, state_win, stride, patch_len, n_q)
    valid = plabels >= 0
    if valid.any():
        l_qpred = F.cross_entropy(qlogits[valid], plabels[valid])
    else:
        l_qpred = torch.zeros((), device=vals.device)
    return l_recon, l_qpred
