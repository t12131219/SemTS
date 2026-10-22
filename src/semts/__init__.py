"""SemTS：时序数据"通用语义"层面的自监督预训练模型。

四条可消融的预训练语义流：
  S1 层级对比头 —— ① 时间（多粒度）+ ③ 实体（实例判别）
  S2 模式头     —— ④ 趋势/季节解耦（时域 + 频域对比）
  S3 关系头     —— ⑤ 跨通道滞后/因果（软回归弱标签）★创新
  S4 质量头     —— ⑦ 三态缺失/质量感知重建 ★创新
"""
from .config import SemTSConfig
from .semts import SemTS

__all__ = ["SemTS", "SemTSConfig"]
