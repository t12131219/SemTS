"""骨干编码器：patch embedding + 通道独立 Transformer。

输入 x: (B, L, C) → 输出 z: (B, C, P, d)，P = (L - patch_len) // stride + 1。
所有通道共享同一套权重（channel-independent），从而天然支持任意通道数，
支撑 ③ 实体语义与跨域迁移（不同通道数的数据集可复用同一骨干）。
"""
import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(-2)]


class TransformerBlock(nn.Module):
    """Pre-norm Transformer 块（TS-TCC 同款稳定性设计）。"""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEncoder(nn.Module):
    """通道独立 patch 编码器。"""

    def __init__(self, d_model=256, patch_len=16, stride=8, n_layers=4, n_heads=8,
                 d_ff=512, dropout=0.1):
        super().__init__()
        assert d_model % (2 * n_heads) == 0, \
            "d_model 必须被 2*n_heads 整除（保证 d_model 为偶数：PositionalEncoding 的奇偶切片要求）"
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.input_conv = nn.Conv1d(1, d_model, kernel_size=patch_len, stride=stride)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def n_patches(self, L: int) -> int:
        return max(1, (L - self.patch_len) // self.stride + 1)

    def forward(self, x: torch.Tensor, keep_mask: Optional[torch.Tensor] = None):
        """x: (B, L, C) → z: (B, C, P, d)。

        keep_mask: (B*C, P)，1=保留，0=置零（TS2Vec 式隐向量时间戳掩码，
        掩码打在隐向量而非原始值上，规避时序数值无界/量纲问题）。
        """
        B, L, C = x.shape
        if L < self.patch_len:
            raise ValueError(f"输入长度 {L} 小于 patch_len {self.patch_len}，无法构造 patch")
        z = self.input_conv(x.permute(0, 2, 1).reshape(B * C, 1, L))  # (B*C, d, P)
        z = z.transpose(1, 2)  # (B*C, P, d)
        if keep_mask is not None:
            z = z * keep_mask.unsqueeze(-1)
        z = self.pos(z)
        for blk in self.blocks:
            z = blk(z)
        return z.reshape(B, C, z.size(1), self.d_model)
