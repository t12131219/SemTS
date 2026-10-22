"""检查点统一加载：兼容 SemTS 与统一骨干基线两种格式，返回带 .encode/.cfg 的评估包装。

SemTS 检查点:   {"model": state_dict(全模型), "cfg": cfg}
基线检查点:     {"encoder": state_dict(仅编码器), "cfg": cfg, "baseline": 名称, "extra": ...}
"""
import torch

from .baselines import EncoderOnly
from .semts import SemTS


def load_encoder_for_eval(ckpt_path, dev="cpu"):
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ckpt["cfg"]
    if "model" in ckpt and "baseline" not in ckpt:
        # SemTS 全模型检查点
        model = SemTS(cfg).to(dev)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model
    # 基线检查点：仅编码器
    model = EncoderOnly(cfg).to(dev)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.eval()
    return model
