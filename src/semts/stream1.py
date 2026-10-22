"""S1：层级对比头（① 时间多粒度 + ③ 实体判别）。"""
from .losses import hierarchical_temporal_contrast, instance_ntxent


def stream1_loss(z_a, z_b, tau=0.1):
    """z_a, z_b: (B, C, T, d)，同一几何窗口的两个上下文视图
    （视图差异 = 输入增强 + 独立采样的隐向量掩码）。

    时序对比：把 (b, c) 视作独立序列（N = B*C），逐时间戳跨视图对比；
    实例对比：序列级池化向量的 NT-Xent（含通道身份判别，覆盖 ③）。
    返回 (l_temp, l_inst)。"""
    B, C, T, d = z_a.shape
    za = z_a.reshape(B * C, T, d)
    zb = z_b.reshape(B * C, T, d)
    l_temp = hierarchical_temporal_contrast(za, zb, tau)
    l_inst = instance_ntxent(za.mean(1), zb.mean(1), tau)
    return l_temp, l_inst
