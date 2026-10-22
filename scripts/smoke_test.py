"""最小闭环冒烟测试（纯 CPU，无需数据集）：验证各模块形状/数值与前向反向。

用法: python scripts/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

from semts.augs import scale_shift_jitter, three_state_mask
from semts.config import SemTSConfig
from semts.lag_labels import lagged_cross_correlation
from semts.semts import SemTS


def test_three_state():
    x = torch.randn(8, 64, 2)
    xa, state, q = three_state_mask(x, mask_ratio=0.25, noise_ratio=0.1)
    assert xa.shape == x.shape and state.shape == x.shape and q.shape == x.shape
    assert (xa[state == 1] == 0).all(), "M 点应置 0"
    assert (state == 2).float().mean() > 0, "应有噪声段"
    assert (q[state == 2] >= 0).all() and (q[state != 2] == -1).all()
    print("[PASS] three_state_mask")


def test_scale_shift_jitter():
    x = torch.randn(4, 64, 2)
    y = scale_shift_jitter(x)
    assert torch.isfinite(y).all() and y.shape == x.shape
    print("[PASS] scale_shift_jitter")


def test_lag_labels():
    torch.manual_seed(0)
    B, L, lag = 16, 256, 5
    x1 = torch.randn(B, L)
    x2 = torch.zeros(B, L)
    x2[:, lag:] = x1[:, :-lag]  # x2[t] = x1[t-5]：x1 领先 x2 5 步
    x2[:, :lag] = torch.randn(B, lag) * 0.1
    ks, rho = lagged_cross_correlation(x1, x2, K=20)
    err = (ks + lag).abs().float()  # m* 应 ≈ -5（负号 = i 领先 j）；float 化以支持 mean()
    assert err.mean() < 1.0, f"滞后识别失败: k*={ks[:5].tolist()}"
    assert rho.mean() > 0.5, f"相关系数过低: {rho.mean().item():.3f}"
    print(f"[PASS] lag_labels: k*={ks[0].item()}, rho={rho[0].item():.3f}")


def test_semts_forward_backward():
    torch.manual_seed(0)
    cfg = SemTSConfig(d_model=32, d_ff=64, n_layers=2, n_heads=4,
                      d_trend=16, d_season=16, n_freqs=32, n_bins=12)
    model = SemTS(cfg)
    x = torch.randn(4, 96, 3) * 2 + torch.arange(96).float().view(1, 96, 1) * 0.05
    loss, d = model.forward_losses(x)
    assert torch.isfinite(loss), "loss 非有限"
    print("losses:", {k: round(v.item(), 4) for k, v in d.items()})
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "梯度异常"
    print("[PASS] SemTS forward/backward")


def test_ablation_weights():
    """验证四条流可独立关闭（消融即置零权重）。"""
    cfg = SemTSConfig(d_model=32, d_ff=64, n_layers=2, n_heads=4,
                      d_trend=16, d_season=16, n_freqs=32, n_bins=12,
                      w1=1.0, w2=0.0, w3=0.0, w4=0.0)
    model = SemTS(cfg)
    x = torch.randn(4, 96, 3)
    loss, d = model.forward_losses(x)
    assert torch.isfinite(loss)
    loss.backward()
    print("[PASS] ablation weights (w2=w3=w4=0)")


if __name__ == "__main__":
    torch.manual_seed(0)  # 固定种子，消除概率性断言 flake
    test_three_state()
    test_scale_shift_jitter()
    test_lag_labels()
    test_semts_forward_backward()
    test_ablation_weights()
    print("ALL SMOKE TESTS PASSED")
