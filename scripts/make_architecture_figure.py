"""绘制 SemTS 架构图（发表级 PNG + SVG）。

用法: python scripts/make_architecture_figure.py
输出: docs/figures/semts_architecture.png / .svg
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 配色（打印友好、色盲安全）----
C_BACK = "#EEF2F7"
C_ENC = "#DCE7F4"
C_STREAM = "#FDF3E3"
C_NEW = "#FBE3E3"
C_DOWN = "#E6F2E6"
C_TEXT = "#1F2933"
C_EDGE = "#4B5563"
C_S1 = "#3B6EA5"
C_S2 = "#8A8F98"
C_S3 = "#C0504D"
C_S4 = "#7B5EA7"
C_DOWN_C = "#3F7D4E"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 8

fig, ax = plt.subplots(figsize=(14, 8.2), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")


def box(x, y, w, h, title, body="", fc=C_BACK, ec=C_EDGE, tc=C_TEXT,
        title_color=None, bold=True, fs_title=8.5, fs_body=7.2, lw=1.1, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if body:
        ax.text(x + w / 2, y + h - 0.9, title, ha="center", va="top",
                fontsize=fs_title, fontweight="bold" if bold else "normal",
                color=title_color or tc, zorder=z + 1)
        ax.text(x + w / 2, y + h - 2.35, body, ha="center", va="top",
                fontsize=fs_body, color=tc, linespacing=1.35, zorder=z + 1)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs_title, fontweight="bold" if bold else "normal",
                color=title_color or tc, zorder=z + 1)


def arrow(x1, y1, x2, y2, color=C_EDGE, lw=1.3, ls="-", rad=0.0, z=1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


# ===== 1. 输入 =====
box(1.5, 26.5, 15, 12, r"Input window  $x \in \mathbb{R}^{C \times L}$",
    "L = 336 steps, C channels\nchannel-independent treatment\n(z-score normalized)",
    fc="#F4F6F8", fs_body=6.8)

# 小波形示意
import numpy as np
rng = np.random.default_rng(0)
t = np.linspace(0, 4 * np.pi, 200)
for k, (yy, ph, f) in enumerate([(32.6, 0.0, 1.0), (30.0, 1.2, 1.6), (27.4, 2.1, 2.4)]):
    w = np.sin(f * t + ph) * 0.55 + 0.18 * rng.normal(size=t.size)
    ax.plot(3.2 + np.linspace(0, 11.2, t.size), yy + w, color="#5B6B7C", lw=0.7, zorder=3)

# ===== 2. 骨干 =====
box(20, 26.5, 14, 12, "Patch Embedding",
    "Conv1d, kernel p = 16\nstride s = 8 → P patches\n(P = ⌊(L−p)/s⌋+1 = 41)",
    fc=C_ENC)
box(36.5, 26.5, 15, 12, "Shared Transformer Encoder",
    "4 blocks · d_model = 256\n8 heads · d_ff = 512\n≈ 5.6 M parameters",
    fc=C_ENC)
box(54, 26.5, 10.5, 12, r"$z \in \mathbb{R}^{C \times P \times d}$",
    "patch-level\nrepresentations\n(last patch → probe)",
    fc="#EDE7F6", fs_body=6.9)

# 两个视图
box(36.5, 41.5, 15, 6.5, "View A: mask p=0.5 + random crop", "",
    fc="#FFFFFF", ls="--", fs_title=7.6, bold=False)
box(36.5, 17.5, 15, 6.5, "View B: + scale / shift / jitter", "",
    fc="#FFFFFF", ls="--", fs_title=7.6, bold=False)
arrow(44, 41.3, 44, 38.8, color="#6B7A8C", ls="--", lw=1.0)
arrow(44, 17.7, 44, 26.3, color="#6B7A8C", ls="--", lw=1.0)

# ===== 3. 四条语义流 =====
SX, SW, SH = 67.5, 30, 7.2
streams = [
    ("S1 · ① time + ③ entity", "Hierarchical contrastive learning\nmulti-scale temporal NT-Xent + instance NT-Xent", C_S1, C_STREAM, 40.2),
    ("S2 · ④ pattern  (ablated: w2 = 0)", "TFD: causal trend experts + queue NT-Xent\nSFD: DFT → complex linear → iDFT amp/phase contrast", C_S2, C_STREAM, 32.2),
    ("S3 · ⑤ relation  (novel)", "Soft lag-bin regression + lead-direction CE\nlabels: FFT lagged cross-correlation k*, ρ", C_S3, C_NEW, 24.2),
    ("S4 · ⑦ quality  (novel)", "Three-state mask O / M(q) / N(noise q∈1..4)\nreconstruct M + per-patch noise-level prediction", C_S4, C_NEW, 16.2),
]
for title, body, tcol, fc, yy in streams:
    box(SX, yy, SW, SH, title, body, fc=fc, tc=C_TEXT, title_color=tcol)

# 输入 → 骨干
arrow(16.8, 32.5, 19.8, 32.5)
arrow(34.2, 32.5, 36.3, 32.5)
arrow(51.7, 32.5, 53.8, 32.5)
# z → 各流
for _, _, _, _, yy in streams:
    arrow(64.7, 32.5, SX - 0.3, yy + SH / 2, rad=0.0)
# 原始 x → S3 弱标签（走下方窄带，避开两个 View 框）
arrow(16.8, 27.0, SX - 0.3, 24.2 + SH - 0.6, color=C_S3, ls="--", lw=1.1, rad=-0.12)

# ===== 4. 总损失 =====
box(20, 5.5, 46, 6.5, "Total loss",
    "L = w1·L_S1 + w2·L_S2 + w3·L_S3 + w4·L_S4        (main config: w2 = 0, w3 = 0.5)",
    fc="#FFF8E1", fs_body=7.0)
# 收集总线：各流垂直下引到 y=13.8 的横向汇流线，再一条箭头进损失框
BUS_Y = 13.8
drop_xs = [74.0, 80.0, 86.0, 92.0]
ax.plot([72.0, 94.0], [BUS_Y, BUS_Y], color=C_EDGE, lw=1.4, solid_capstyle="round", zorder=1)
for (_, _, tcol, _, yy), dx in zip(streams, drop_xs):
    arrow(dx, yy - 0.2, dx, BUS_Y + 0.3, color=tcol, lw=1.1)
arrow(72.0, BUS_Y, 62.5, 12.3, color=C_EDGE, lw=1.4)

# ===== 5. 下游 =====
box(70, 5.5, 28, 6.5, "Frozen encoder → linear probes",
    "forecasting (ridge) · classification · anomaly · zero-shot",
    fc=C_DOWN, title_color=C_DOWN_C, fs_body=6.9)
# z → 下游（冻结编码器路径，沿损失框上方走线）
ax.plot([57.0, 57.0], [26.3, 12.45], color=C_DOWN_C, lw=1.1, ls="--", zorder=1)
ax.plot([57.0, 69.6], [12.45, 12.45], color=C_DOWN_C, lw=1.1, ls="--", zorder=1)
arrow(69.6, 12.45, 70.2, 10.5, color=C_DOWN_C, lw=1.1, ls="--")

# 图例
ax.text(1.5, 59.5, "SemTS: self-supervised pre-training with disentangled temporal semantics",
        fontsize=11, fontweight="bold", color=C_TEXT)
ax.text(1.5, 57.2,
        "four ablatable semantic streams on a shared channel-independent patch Transformer;\n"
        "main configuration keeps S1 (time/entity) + S3 (cross-channel lead-lag) + S4 (quality), "
        "S2 (pattern) ablated out",
        fontsize=7.4, color="#4B5563", linespacing=1.3)

fig.tight_layout()
for ext in ("png", "svg"):
    path = os.path.join(OUT_DIR, f"semts_architecture.{ext}")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print("[figure]", path)
plt.close(fig)
