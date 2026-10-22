"""统一骨干基线：同一 PatchEncoder 骨干上的主流预训练目标（M4 公平对比用）。

- ts2vec: TS2Vec —— 层级时序对比 + 实例 NT-Xent（等价于 SemTS 的 S1 流）
- cost:   CoST —— 趋势/季节解耦（时域队列 + 频域幅值/相位对比）（等价于 SemTS 的 S2 流）
- simmtm: SimMTM —— 多掩码序列流形重建（简化：输入级掩码 + 序列级相似度加权聚合 + 流形约束）
- tf_c:   TF-C —— 时域/频域表示一致性（简化：共享编码器 + 时/频/跨域三个 NT-Xent）

与 SemTS 完全相同的数据管线、视图增强、骨干与下游评估协议，
保证"语义流增量 vs 基线"的对比干净（消融语义 = 从 SemTS 减去对应流即可，基线 = 对照锚点）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .augs import scale_shift_jitter
from .backbone import PatchEncoder
from .losses import hierarchical_temporal_contrast, instance_ntxent
from .stream2 import Stream2


def _two_views(encoder, x, cfg):
    """两上下文视图（与 SemTS 一致：视图 B 带输入增强，隐向量掩码独立采样）。"""
    B, L, C = x.shape
    P = encoder.n_patches(L)
    x_b = scale_shift_jitter(x)
    ma = (torch.rand(B * C, P, device=x.device) > cfg.p_mask).float()
    mb = (torch.rand(B * C, P, device=x.device) > cfg.p_mask).float()
    return encoder(x, ma), encoder(x_b, mb)


def ts2vec_loss(encoder, x, cfg):
    """TS2Vec 基线（统一骨干版）。"""
    za, zb = _two_views(encoder, x, cfg)
    B, C, T, d = za.shape
    l_temp = hierarchical_temporal_contrast(
        za.reshape(B * C, T, d), zb.reshape(B * C, T, d), tau=cfg.tau
    )
    l_inst = instance_ntxent(
        za.reshape(B * C, T, d).mean(1), zb.reshape(B * C, T, d).mean(1), tau=cfg.tau
    )
    return l_temp + l_inst, {"temp": l_temp, "inst": l_inst}


def cost_loss(encoder, x, cfg, stream2):
    """CoST 基线（统一骨干版，复用 SemTS 的 S2 实现）。"""
    za, zb = _two_views(encoder, x, cfg)
    l_time, l_amp, l_phase = stream2(za, zb)
    total = l_time + 0.5 * cfg.alpha * (l_amp + l_phase)
    return total, {"time": l_time, "amp": l_amp, "phase": l_phase}


class _SimMTMDecoder(nn.Module):
    def __init__(self, d_model, patch_len):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, patch_len)
        )

    def forward(self, z):
        return self.mlp(z)  # (..., P, d) -> (..., P, patch_len)


def simmtm_loss(encoder, decoder, x, cfg, M=4, lam=1.0):
    """SimMTM 基线（统一骨干版简化）。

    - M 个输入级掩码序列（25% 时间点置 0）+ 原序列；
    - 序列级相似度（归一化内积）加权聚合点级表示；
    - 重建：聚合表示解码后与原始 patch 目标 MSE（全部时间点）；
    - 流形约束：原序列表示与"掩码序列均值表示"的实例 NT-Xent（简化）。
    """
    B, L, C = x.shape
    P = encoder.n_patches(L)
    mask_ratio = 0.25
    xs = [x]
    for _ in range(M):
        m = torch.rand(B, L, 1, device=x.device) < mask_ratio
        xs.append(x * (~m).float())
    xf = torch.stack(xs, dim=1).reshape(B * (M + 1), L, C)  # (B*(M+1), L, C)
    z = encoder(xf)                                         # (B*(M+1), C, P, d)
    z = z.reshape(B, M + 1, C, P, -1)                       # (B, k, C, P, d)
    # 序列级表示与相似度（k 维 = M+1 个视图）
    s = z.mean(dim=(2, 3))                                  # (B, M+1, d)
    sn = F.normalize(s, dim=-1)
    sim = (sn @ sn.transpose(1, 2)) / cfg.tau               # (B, M+1, M+1)
    w = sim.softmax(dim=2)
    # 加权聚合点级表示：zagg[b,m] = Σ_k w[b,m,k]·z[b,k]
    zagg = torch.einsum("bmk,bkcpd->bmcpd", w, z)           # (B, M+1, C, P, d)
    # 重建（patch 目标与 stream4 相同的 ts 映射，ts 最大 = L-1 不越界）
    off = torch.arange(cfg.patch_len, device=x.device).view(1, -1)
    ts = torch.arange(P, device=x.device).view(P, 1) * cfg.stride + off  # (P, patch_len)
    # x 为 (B, L, C)：先转置到 (B, C, L) 再按时间维索引（否则网格命中通道维）
    tgt = x.permute(0, 2, 1)[:, :, ts]                      # (B, C, P, patch_len)
    pred = decoder(zagg.reshape(B * (M + 1), C, P, -1))     # (B*(M+1), C, P, patch_len)
    pred = pred.reshape(B, M + 1, C, P, cfg.patch_len)
    l_recon = ((pred - tgt.unsqueeze(1)) ** 2).mean()
    # 流形约束（简化）
    sm = sn[:, 1:].mean(1)                                  # (B, d)
    l_constraint = instance_ntxent(sn[:, 0], sm, tau=cfg.tau)
    return l_recon + lam * l_constraint, {"recon": l_recon, "constraint": l_constraint}


def tf_c_loss(encoder, x, cfg):
    """TF-C 基线（统一骨干版简化）：时域池化向量与频域幅值池化向量的跨域一致性。"""
    za, zb = _two_views(encoder, x, cfg)
    t_a = za.mean(dim=(1, 2))  # (B, d) 时域
    t_b = zb.mean(dim=(1, 2))
    f_a = torch.fft.rfft(za, dim=2).abs().mean(dim=(1, 2))  # (B, d) 频域幅值
    f_b = torch.fft.rfft(zb, dim=2).abs().mean(dim=(1, 2))
    l_t = instance_ntxent(t_a, t_b, tau=cfg.tau)  # 时域实例判别
    l_f = instance_ntxent(f_a, f_b, tau=cfg.tau)  # 频域实例判别
    l_c = instance_ntxent(t_a, f_a, tau=cfg.tau)  # 时-频一致性（同实例跨域为正）
    return l_t + l_f + l_c, {"time_dom": l_t, "freq_dom": l_f, "consist": l_c}


class EncoderOnly(nn.Module):
    """仅含编码器的下游评估包装（encode 接口与 SemTS 一致，供统一评测脚本使用）。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(
            d_model=cfg.d_model, patch_len=cfg.patch_len, stride=cfg.stride,
            n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_ff=cfg.d_ff, dropout=cfg.dropout,
        )

    @torch.no_grad()
    def encode(self, x, keep_mask=None):
        return self.encoder(x, keep_mask)


BASELINES = {
    "ts2vec": ts2vec_loss,
    "cost": cost_loss,
    "simmtm": simmtm_loss,
    "tf_c": tf_c_loss,
}
