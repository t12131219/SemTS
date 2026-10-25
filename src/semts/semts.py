"""SemTS 总装：骨干 + 四条语义流 + 总损失。

每条流对应一个可消融的语义维度（消融实验 = 置零对应权重 w1..w4，该流整体跳过计算）：
  S1 w1 —— ① 时间（多粒度）+ ③ 实体
  S2 w2 —— ④ 模式（趋势/季节）
  S3 w3 —— ⑤ 关系（跨通道滞后/因果）★创新
  S4 w4 —— ⑦ 质量（缺失/置信度）★创新
"""
import torch
import torch.nn as nn

from .augs import (
    add_state_quality_emb,
    latent_keep_masks,
    patch_to_timestamps,
    random_crop,
    scale_shift_jitter,
    three_state_mask,
)
from .backbone import PatchEncoder
from .stream1 import stream1_loss
from .stream2 import Stream2
from .stream3 import RelationHead, stream3_loss
from .stream4 import QualityHead, stream4_loss


class SemTS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(
            d_model=cfg.d_model,
            patch_len=cfg.patch_len,
            stride=cfg.stride,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            grad_checkpoint=getattr(cfg, "grad_checkpoint", False),
        )
        self.stream2 = Stream2(
            cfg.d_model,
            d_trend=cfg.d_trend,
            d_season=cfg.d_season,
            n_experts=cfg.n_experts,
            n_freqs=cfg.n_freqs,
            queue_size=cfg.queue_size,
            alpha=cfg.alpha,
            tau=cfg.tau,
            freq_tau=cfg.freq_tau,
        )
        # 滞后搜索上界受窗口长度限制；n_bins 由 K_eff 派生，保证与 compute_lag_labels 一致
        self.lag_K_eff = min(cfg.lag_K, cfg.L - 1)
        n_bins = 2 * self.lag_K_eff // cfg.bin_width
        self.n_bins = n_bins
        self.stream3_head = RelationHead(cfg.d_model, n_bins)
        self.stream4_head = QualityHead(cfg.d_model, cfg.patch_len, n_q=cfg.n_q)
        # 三态/质量标量嵌入（加回输入，让模型区分 O/M/N 与质量等级）
        self.emb_state = nn.Parameter(torch.zeros(3))
        self.emb_qual = nn.Parameter(torch.zeros(cfg.n_q))

    def forward_losses(self, x):
        """x: (B, L, C) 干净输入。返回 (total, dict)（dict 用于日志与消融）。"""
        cfg = self.cfg
        B, L, C = x.shape
        if cfg.w1 <= 0 and cfg.w2 <= 0 and cfg.w3 <= 0 and cfg.w4 <= 0:
            raise ValueError("至少启用一条语义流（w1..w4 不能全为 0）")
        P_full = self.encoder.n_patches(L)

        # 1) 三态质量掩码（S4 输入）→ 加状态/质量嵌入 → 增强视图（S2/S1-B）
        x_aug, state, q = three_state_mask(
            x, mask_ratio=cfg.mask_ratio, noise_ratio=cfg.noise_ratio, n_q=cfg.n_q
        )
        x_emb = add_state_quality_emb(x_aug, state, q, self.emb_state, self.emb_qual)
        need_b = (cfg.w1 > 0) or (cfg.w2 > 0)
        x_b = scale_shift_jitter(x_emb) if need_b else None

        # 2) 随机窗口（batch 共享几何；v0.1 两视图共用同一窗口保证时间轴对齐）
        s, p = random_crop(P_full, min_frac=cfg.crop_min_frac)
        t0, t1 = patch_to_timestamps(s, p, cfg.patch_len, cfg.stride)
        win_a = x_emb[:, t0:t1]
        win_b = x_b[:, t0:t1] if need_b else None
        # 预训练管线内置 RevIN（C2）：窗口统计量取自干净输入，输入与 S4 重建目标
        # 同时归一化，使编码器学习对逐窗口尺度/平移不变
        if getattr(cfg, "revin", False):
            xw = x[:, t0:t1]                                  # (B, Lw, C) 干净窗口
            mu = xw.mean(1, keepdim=True)
            sd = torch.sqrt(torch.var(xw, dim=1, keepdim=True, unbiased=False) + 1e-5)
            win_a = (win_a - mu) / sd
            if win_b is not None:
                win_b = (win_b - mu) / sd
        # S4 重建真值必须是干净输入（M 点已置 0+嵌入，目标应为其原始值）
        win_clean = x[:, t0:t1].transpose(1, 2)      # (B, C, Lw)
        if getattr(cfg, "revin", False):
            win_clean = ((win_clean.transpose(1, 2) - mu) / sd).transpose(1, 2)
        state_win = state[:, t0:t1].transpose(1, 2)  # (B, C, Lw)
        q_win = q[:, t0:t1].transpose(1, 2)          # (B, C, Lw)

        # 3) 编码视图（隐向量时间戳掩码独立采样）
        z_a = self.encoder(win_a, latent_keep_masks(B, C, p, cfg.p_mask, x.device))
        z_b = None
        if need_b:
            z_b = self.encoder(win_b, latent_keep_masks(B, C, p, cfg.p_mask, x.device))

        # 4) 四条语义流（w=0 时整条流跳过：消融即开关，同时节省计算）
        out = {}
        total = None

        def add(key, loss, w):
            nonlocal total
            out[key] = loss.detach()
            total = (w * loss) if total is None else total + w * loss

        if cfg.w1 > 0:
            l_temp, l_inst = stream1_loss(z_a, z_b, tau=cfg.tau)
            add("temp", l_temp, cfg.w1)
            add("inst", l_inst, cfg.w1)
        if cfg.w2 > 0:
            l_time, l_amp, l_phase = self.stream2(z_a, z_b)
            add("time", l_time, cfg.w2)
            add("amp", 0.5 * cfg.alpha * l_amp, cfg.w2)
            add("phase", 0.5 * cfg.alpha * l_phase, cfg.w2)
        if cfg.w3 > 0:
            # 弱标签与表示使用同一窗口（x 窗口内干净值），避免"全序列标签 vs 裁剪窗口表示"的错位
            l_lag, l_dir = stream3_loss(
                self.stream3_head, z_a, x[:, t0:t1],
                K=self.lag_K_eff, tau_corr=cfg.tau_corr, k_min=cfg.k_min,
                bin_width=cfg.bin_width, max_pairs=cfg.max_pairs,
            )
            add("lag", l_lag, cfg.w3)
            add("dir", cfg.lam_dir * l_dir, cfg.w3)
        if cfg.w4 > 0:
            vals, qlogits = self.stream4_head(z_a)
            l_recon, l_qpred = stream4_loss(
                vals, qlogits, win_clean, state_win, q_win,
                cfg.stride, cfg.patch_len, cfg.n_q,
            )
            add("recon", l_recon, cfg.w4)
            add("qpred", cfg.lam_q * l_qpred, cfg.w4)

        for k in ("temp", "inst", "time", "amp", "phase", "lag", "dir", "recon", "qpred"):
            if k not in out:
                out[k] = torch.zeros((), device=x.device)
        out["total"] = total.detach()
        return total, out

    @torch.no_grad()
    def encode(self, x, keep_mask=None):
        """下游任务用：x: (B, L, C) → z: (B, C, P, d)。"""
        return self.encoder(x, keep_mask)
