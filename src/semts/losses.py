"""对比损失、软标签损失与负样本队列。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_contrastive(z1, z2, tau=0.1):
    """TS2Vec 式时序对比（原文式 1）：同一时间戳跨视图为正，其余时间戳为负。

    z1, z2: (N, T, D)，时间轴已对齐（两视图共用同一窗口）。
    负样本 = 另一视图的全部时间戳 + 本视图除自身外的全部时间戳。
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    T = z1.size(1)
    pos = (z1 * z2).sum(-1) / tau  # (N, T)
    sim12 = torch.bmm(z1, z2.transpose(1, 2)) / tau  # (N, T, T)
    sim11 = torch.bmm(z1, z1.transpose(1, 2)) / tau
    eye = torch.eye(T, device=z1.device, dtype=z1.dtype)
    sim11 = sim11.masked_fill(eye.bool().unsqueeze(0), float("-inf"))
    denom = torch.logsumexp(torch.cat([sim12, sim11], dim=-1), dim=-1)  # (N, T)
    return (denom - pos).mean()


def instance_ntxent(v1, v2, tau=0.1, chunk=4096):
    """序列级实例对比（③ 实体语义）：v1, v2: (N, D)。

    正对 = 同一样本两视图；负 = 其余样本的两视图（SimCLR 式，负样本用 2N 池）。
    v0.1 简化：与 TS2Vec 逐时间戳的实例对比不同，这里用序列级池化向量，
    计算量从 O(N²T) 降为 O(N²)，N = B*C（含通道身份判别）。
    """
    v1 = F.normalize(v1, dim=-1)
    v2 = F.normalize(v2, dim=-1)
    N = v1.size(0)
    pos = (v1 * v2).sum(-1) / tau  # (N,)
    eye = torch.eye(N, device=v1.device, dtype=torch.bool)
    lse = torch.full((N,), float("-inf"), device=v1.device)
    for j in range(0, N, chunk):
        s2 = v1 @ v2[j:j + chunk].t() / tau  # (N, c)
        s1 = v1 @ v1[j:j + chunk].t() / tau
        s1 = s1.masked_fill(eye[:, j:j + chunk], float("-inf"))
        lse = torch.logaddexp(lse, torch.cat([s2, s1], dim=-1).logsumexp(dim=-1))
    return (lse - pos).mean()


def hierarchical_temporal_contrast(z1, z2, tau=0.1):
    """TS2Vec 式分层对比（原文 Algorithm 1）：沿时间轴 maxpool(k=2) 逐层累加，
    覆盖 ① 时间语义的多粒度（时间戳 → 子段 → 实例）。"""
    total = temporal_contrastive(z1, z2, tau)
    d = 1
    while z1.size(1) > 1:
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        total = total + temporal_contrastive(z1, z2, tau)
        d += 1
    return total / d


class NTXentQueue(nn.Module):
    """带队列的 NT-Xent（S2 时域对比用，CoST MoCo 的简化：
    v0.1 不使用动量编码器，key 直接来自增强视图的同一编码器）。"""

    def __init__(self, dim, queue_size=256, tau=0.07):
        super().__init__()
        self.tau = tau
        self.queue_size = queue_size
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, dim), dim=-1))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, k):
        k = F.normalize(k.detach(), dim=-1)
        n = k.size(0)
        qs = self.queue_size
        if n >= qs:
            self.queue.copy_(k[-qs:])
            return
        ptr = int(self.ptr)
        if ptr + n > qs:
            self.queue[ptr:] = k[: qs - ptr]
            self.queue[: ptr + n - qs] = k[qs - ptr:]
        else:
            self.queue[ptr: ptr + n] = k
        self.ptr[0] = (ptr + n) % qs

    def forward(self, q, k):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        pos = (q * k).sum(-1, keepdim=True)
        # 用队列快照（clone）计算负样本：enqueue 的 in-place 更新不会破坏本次计算图
        neg = q @ self.queue.clone().t()
        logits = torch.cat([pos, neg], dim=-1) / self.tau
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)
        self.enqueue(k)
        return loss


def freq_amp_loss(F1, F2, tau=0.07):
    """CoST 式频域幅值对比：正对 = 同一样本跨视图，负 = 其他样本，逐频率平均。
    F1, F2: (N, F, D) complex。对角线（正样本自身）从负样本池中屏蔽。"""
    a1 = F.normalize(F1.abs(), dim=-1)
    a2 = F.normalize(F2.abs(), dim=-1)
    N = a1.size(0)
    pos = (a1 * a2).sum(-1)  # (N, F)
    sim = torch.einsum("nfd,mgd->nfm", a1, a2)  # (N, F, N)
    eye = torch.eye(N, device=a1.device, dtype=torch.bool).unsqueeze(1)
    sim = sim.masked_fill(eye, float("-inf"))
    return (torch.logsumexp(sim / tau, dim=-1) - pos / tau).mean()


def freq_phase_loss(F1, F2, tau=0.07):
    """CoST 式频域相位对比。对角线（正样本自身）从负样本池中屏蔽。"""
    p1 = F.normalize(F1.angle(), dim=-1)
    p2 = F.normalize(F2.angle(), dim=-1)
    N = p1.size(0)
    pos = (p1 * p2).sum(-1)
    sim = torch.einsum("nfd,mgd->nfm", p1, p2)
    eye = torch.eye(N, device=p1.device, dtype=torch.bool).unsqueeze(1)
    sim = sim.masked_fill(eye, float("-inf"))
    return (torch.logsumexp(sim / tau, dim=-1) - pos / tau).mean()


def soft_cross_entropy(logits, soft_targets, mask=None):
    """软交叉熵（S3 滞后软回归用）：logits (N, K)，soft_targets (N, K) 和为 1。"""
    logp = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * logp).sum(-1)
    if mask is not None:
        m = mask.float()
        return (loss * m).sum() / m.sum().clamp_min(1.0)
    return loss.mean()
