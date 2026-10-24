"""绘制 SemTS 详细版架构图（发表级 PNG + SVG）。

v2 布局（含总线走线的干净版本）：
- 左边缘：损失总线（四流 → 总损失）
- 中间列缝：z 分线（patch 表示 → 四流）
- 右边缘：冻结总线（表示 → 下游探针）
- 每个流带任务示意插图；底部为损失/下游/7 维语义图例

用法: python scripts/make_architecture_figure_detailed.py
输出: docs/figures/semts_architecture_detailed.png / .svg
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

TXT = "#1F2933"
EDGE = "#4B5563"
C_S1, F_S1 = "#3B6EA5", "#E8F0F9"
C_S2, F_S2 = "#8A8F98", "#F1F2F4"
C_S3, F_S3 = "#C0504D", "#FBE9E9"
C_S4, F_S4 = "#7B5EA7", "#F0EAF7"
F_ENC = "#E3ECF6"
F_LOSS = "#FFF6E0"
F_DOWN = "#E9F3EA"
C_FROZEN = "#3F7D4E"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 8

fig = plt.figure(figsize=(17.5, 14.6), dpi=170)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 102)
ax.set_ylim(0, 100)
ax.axis("off")


def rbox(x, y, w, h, fc="white", ec=EDGE, lw=1.1, ls="-", z=2, r=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.22,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def label(x, y, s, fs=8, color=TXT, bold=False, ha="center", va="center", z=6, ls_=1.3, rot=0):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=z,
            fontweight="bold" if bold else "normal", linespacing=ls_, rotation=rot)


def arrow(x1, y1, x2, y2, color=EDGE, lw=1.3, ls="-", rad=0.0, z=1, ms=12, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


def line(xs, ys, color=EDGE, lw=1.2, ls="-", z=1):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")


def inset(x, y, w, h):
    return fig.add_axes([x / 100, y / 100, w / 100, h / 100])


rng = np.random.default_rng(7)

# ================= 标题 =================
label(2.0, 97.4, "SemTS: self-supervised pre-training with disentangled temporal semantics",
      fs=15, bold=True, ha="left")
label(2.0, 95.0,
      "shared channel-independent patch Transformer + four ablatable semantic streams;   "
      "main configuration: S1 (time/entity) + S3 (cross-channel lead-lag) + S4 (quality),  S2 (pattern) ablated out",
      fs=8.6, color="#4B5563", ha="left")

# ================= Row 1: 输入 → 骨干 =================
rbox(2.5, 78.5, 18.0, 14.5, fc="#F5F7F9")
label(11.5, 91.6, "Input window", fs=9.5, bold=True)
label(11.5, 89.6, r"$x \in \mathbb{R}^{C \times L}$   (C channels, L = 336)", fs=8)
label(11.5, 87.9, "z-score normalized · each channel encoded independently", fs=6.9, color="#6B7A8C")
t = np.linspace(0, 6 * np.pi, 240)
for k, (yy, ph, f, col) in enumerate([(84.2, 0.0, 1.0, "#2E5F8A"), (82.2, 1.3, 1.7, "#8A5A2E"), (80.2, 2.2, 2.6, "#5A8A4A")]):
    w = 0.34 * np.sin(f * t + ph) + 0.06 * rng.normal(size=t.size)
    ax.plot(3.6 + np.linspace(0, 15.6, t.size), yy + w, color=col, lw=0.9, zorder=4)

# 两个视图
rbox(22.5, 78.5, 16.0, 14.5, fc="#F5F7F9")
label(30.5, 91.6, "Two contextual views", fs=9.5, bold=True)
label(24.4, 89.3, "A: mask p=0.5 + crop", fs=7.2, ha="left", color="#33414F")
p_keep = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
for i, kp in enumerate(p_keep):
    ax.add_patch(Rectangle((23.8 + i * 0.90, 86.5), 0.78, 1.7,
                           fc="#BFD3E6" if kp else "#FFFFFF", ec="#8FAAC4", lw=0.5, zorder=4))
label(32.2, 85.2, "kept / masked patches", fs=6.4, color="#6B7A8C")
label(24.4, 83.4, "B: A + scale / shift / jitter", fs=7.2, ha="left", color="#33414F")
p_keep2 = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1])
for i, kp in enumerate(p_keep2):
    ax.add_patch(Rectangle((23.8 + i * 0.90, 80.6), 0.78, 1.7,
                           fc="#F3D9A4" if kp else "#FFFFFF", ec="#C9A961", lw=0.5, zorder=4))
label(32.2, 79.3, "augmented (error-variable interventions)", fs=6.2, color="#6B7A8C")

# patch embedding
rbox(40.5, 78.5, 15.0, 14.5, fc=F_ENC)
label(48.0, 91.6, "Patch Embedding", fs=9.5, bold=True)
label(48.0, 89.6, "Conv1d  kernel p = 16, stride s = 8", fs=7.3)
label(48.0, 88.2, r"$\rightarrow\ P = \lfloor (L-p)/s \rfloor + 1 = 41$ patches", fs=7.3)
for i, g in enumerate(np.linspace(0.35, 0.95, 41)):
    ax.add_patch(Rectangle((41.6 + i * 0.315, 84.6), 0.28, 2.0,
                           fc=plt.cm.Blues(g), ec="white", lw=0.25, zorder=4))
label(48.0, 83.4, "patch tokens  (per channel, weights shared)", fs=6.5, color="#4B5563")

# Transformer block 细节
rbox(57.5, 78.5, 20.0, 14.5, fc=F_ENC)
label(67.5, 91.6, "Shared Transformer Encoder  ×4", fs=9.5, bold=True)
bx, by, bw, bh = 58.6, 80.6, 10.6, 9.2
ax.add_patch(Rectangle((bx, by), bw, bh, fc="white", ec="#8FAAC4", lw=0.9, zorder=3))
label(bx + bw / 2, by + bh - 0.75, "Transformer block", fs=6.5, color="#33506B")
steps = ["LN", "MHSA (8 heads)", "LN", "FFN (d_ff=512)"]
yy = by + 6.4
for i, s in enumerate(steps):
    hh = 1.05 if i % 2 == 0 else 1.25
    ax.add_patch(FancyBboxPatch((bx + 0.9, yy), 4.4, hh, boxstyle="round,pad=0.05,rounding_size=0.25",
                                fc="#DCE7F4" if i % 2 == 0 else "#C4D6EA", ec="#6C87A5", lw=0.6, zorder=4))
    label(bx + 3.1, yy + hh / 2, s, fs=5.3, color="#26384B")
    if i < 3:
        arrow(bx + 3.1, yy, bx + 3.1, yy - 1.0, lw=0.7, ms=6)
    yy -= 1.4
arrow(bx + 5.6, by + 7.3, bx + 5.6, by + 4.6, rad=-0.55, lw=0.8, ms=7, ls="--")
arrow(bx + 5.6, by + 4.6, bx + 5.6, by + 1.9, rad=-0.55, lw=0.8, ms=7, ls="--")
label(bx + 8.1, by + 5.7, "+ residual", fs=5.2, color="#6B7A8C")
arrow(bx + 3.1, by - 0.2, bx + 3.1, by - 1.2, lw=0.8, ms=7)
label(70.6, 88.4, "d_model = 256", fs=6.7, ha="left")
label(70.6, 87.0, "8 heads · d_ff = 512", fs=6.7, ha="left")
label(70.6, 85.6, "≈ 5.6 M params", fs=6.7, ha="left")
label(70.6, 83.8, "weight-shared", fs=6.7, ha="left", color=C_S1, bold=True)
label(70.6, 82.5, "across channels", fs=6.7, ha="left", color=C_S1)
label(70.6, 80.9, "latent mask p=0.5", fs=6.2, ha="left", color="#6B7A8C")

# z
rbox(79.5, 78.5, 18.5, 14.5, fc="#EDE7F6")
label(88.75, 91.6, "Patch representations", fs=9.5, bold=True)
label(88.75, 89.6, r"$z \in \mathbb{R}^{C \times P \times d}$", fs=8.2)
axg = inset(81.0, 82.4, 15.5, 5.9)
axg.imshow(rng.uniform(0.15, 0.95, size=(7, 22)), aspect="auto", cmap="Purples", interpolation="nearest")
axg.set_xticks([])
axg.set_yticks([])
for sp in axg.spines.values():
    sp.set_edgecolor("#9C8AB8")
    sp.set_linewidth(0.7)
label(88.75, 80.9, "C channels × P patches × d", fs=6.4, color="#4B5563")

# 行内箭头
arrow(20.7, 85.7, 22.3, 85.7)
arrow(38.7, 85.7, 40.3, 85.7)
arrow(55.7, 85.7, 57.3, 85.7)
arrow(77.7, 85.7, 79.3, 85.7)

# ================= Row 2: S1 / S2 =================
# ---- S1 ----
rbox(2.5, 54.5, 47.2, 21.0, fc=F_S1, ec=C_S1, lw=1.3)
label(4.4, 73.8, "S1 · ① time + ③ entity — hierarchical contrastive learning", fs=9.6, bold=True, ha="left", color=C_S1)
label(4.4, 71.9, "multi-scale temporal contrast (same timestamp across views = positive)  +  instance NT-Xent", fs=6.9,
      ha="left", color="#33414F")
sm = np.full((10, 10), 0.10)
for i in range(10):
    for j in range(10):
        sm[i, j] += 0.78 * np.exp(-abs(i - j) / 1.7)
np.fill_diagonal(sm, 1.0)
axm = inset(4.8, 56.8, 9.0, 9.0)
axm.imshow(sm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
for i in range(10):
    axm.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=C_S3, lw=0.9))
axm.set_xticks([])
axm.set_yticks([])
axm.set_title("view A ↔ B cosine", fontsize=5.8, pad=1.5, color="#33506B")
for k, (pw, yy) in enumerate([(8.6, 58.0), (6.2, 60.6), (3.8, 63.2)]):
    ax.add_patch(FancyBboxPatch((21.5, yy), pw, 2.0, boxstyle="round,pad=0.08,rounding_size=0.3",
                                fc="#BFD3E6", ec="#6C87A5", lw=0.6, zorder=4))
    label(21.5 + pw / 2, yy + 1.0, f"max-pool P/{2 ** k}", fs=5.3, color="#26384B")
label(25.8, 65.9, "hierarchical pooling over time", fs=6.2, color="#4B5563")
label(36.0, 62.0, r"$\mathcal{L}_{S1} = \mathcal{L}_{temp} + \lambda\,\mathcal{L}_{inst}$", fs=9.0, color=C_S1)
label(36.0, 59.2, "chunked log-sum-exp over negatives", fs=6.3, color="#4B5563")
label(36.0, 57.2, "instance loss uses series + channel identity", fs=6.3, color="#4B5563")
label(36.0, 55.4, "queue-free; positives drawn per view", fs=6.3, color="#4B5563")

# ---- S2 ----
rbox(52.7, 54.5, 47.0, 21.0, fc=F_S2, ec=C_S2, lw=1.2, ls="--")
label(54.6, 73.8, "S2 · ④ pattern — trend/seasonal disentangling  (ABLATED: w2 = 0)", fs=9.4, bold=True,
      ha="left", color=C_S2)
label(54.6, 71.9, "TFD: causal trend experts → time-domain NT-Xent with queue        SFD: DFT → ℂ-linear → iDFT → amp/phase",
      fs=6.9, ha="left", color="#4B5563")
label(56.8, 69.8, "TFD (trend)", fs=6.8, ha="left", color=C_S2, bold=True)
for k, ks in enumerate([1, 2, 4]):
    ax.add_patch(FancyBboxPatch((56.6, 64.2 - k * 2.6), 1.6, 1.9, boxstyle="round,pad=0.06,rounding_size=0.25",
                                fc="#D9DCE1", ec=C_S2, lw=0.6, zorder=4))
    label(57.4, 65.15 - k * 2.6, f"k={ks}", fs=5.3, color="#33414F")
label(59.6, 65.6, "causal conv experts\n(AR mixture, avg.)", fs=5.7, ha="left", color="#4B5563")
label(59.6, 61.6, "time-domain NT-Xent\n+ negative queue", fs=5.7, ha="left", color="#4B5563")
label(71.4, 69.8, "SFD (seasonal)", fs=6.8, ha="left", color=C_S2, bold=True)
freq = np.abs(np.sin(np.linspace(0.4, 9, 24))) * 0.8 + 0.1
for i, v in enumerate(freq):
    ax.add_patch(Rectangle((71.2 + i * 0.19, 66.4), 0.15, v * 2.2, fc="#9AA3AE", ec="none", zorder=4))
ax.add_patch(FancyBboxPatch((76.3, 65.4), 2.4, 3.2, boxstyle="round,pad=0.06,rounding_size=0.25",
                            fc="#D9DCE1", ec=C_S2, lw=0.6, zorder=4))
label(77.5, 67.0, "ℂ-linear\nper freq", fs=4.8, color="#26384B")
arrow(74.9, 67.2, 76.1, 67.2, lw=0.7, ms=6)
label(79.4, 68.0, "iDFT", fs=5.5, color="#26384B")
label(79.4, 66.0, "amp + phase\ncontrast", fs=5.4, ha="left", color="#4B5563")
label(89.6, 62.2, r"$\mathcal{L}_{S2} = \mathcal{L}_{time} + \frac{\alpha}{2}(\mathcal{L}_{amp}+\mathcal{L}_{phase})$",
      fs=8.2, color=C_S2)
label(89.6, 59.4, "no prior period needed;\ntrend/seasonal disentangled", fs=6.2, color="#4B5563")
label(89.6, 56.4, "harmful under windowed protocol → w2 = 0\n(see §4.2)", fs=6.2, color="#7A4536")

# ================= Row 3: S3 / S4 =================
# ---- S3 ----
rbox(2.5, 31.5, 47.2, 21.0, fc=F_S3, ec=C_S3, lw=1.3)
label(4.4, 50.8, "S3 · ⑤ relation — cross-channel lead-lag (novel)", fs=9.6, bold=True, ha="left", color=C_S3)
label(4.4, 48.9, "weak labels from FFT lagged cross-correlation of raw x;  soft lag-bin regression + lead direction",
      fs=6.9, ha="left", color="#33414F")
ks = np.arange(-48, 49)
rho = 0.18 + 0.72 * np.exp(-((ks - 3.0) ** 2) / (2 * 3.2 ** 2))
axr = inset(4.8, 33.6, 11.6, 11.2)
axr.plot(ks, rho, color=C_S3, lw=1.2)
axr.axvline(3, color="#7A4536", lw=0.8, ls="--")
axr.axhline(0.5, color="#B0B7C0", lw=0.6, ls=":")
axr.annotate("k* = +3", xy=(3, 0.9), xytext=(12, 0.76), fontsize=5.6, color="#7A4536",
             arrowprops=dict(arrowstyle="->", lw=0.6, color="#7A4536"))
axr.set_title(r"$\rho(k)=\mathrm{corr}(x_i[t],\,x_j[t-k])$", fontsize=5.6, pad=1.5, color="#5A2E2E")
axr.set_xticks([-48, 0, 48])
axr.tick_params(labelsize=5, colors="#6B7A8C")
for sp in axr.spines.values():
    sp.set_edgecolor("#C7A7A7")
    sp.set_linewidth(0.6)
axs = inset(17.6, 33.6, 8.4, 11.2)
bins = np.arange(-6, 7)
soft = 0.15 + 0.85 * np.exp(-((bins - 3) ** 2) / (2 * 1.4 ** 2))
axs.bar(bins, soft, color="#E0A6A4", edgecolor=C_S3, lw=0.5, width=0.85)
axs.set_title("soft label over bins", fontsize=5.6, pad=1.5, color="#5A2E2E")
axs.set_xticks([-6, 0, 6])
axs.tick_params(labelsize=5, colors="#6B7A8C")
for sp in axs.spines.values():
    sp.set_edgecolor("#C7A7A7")
    sp.set_linewidth(0.6)
ax.add_patch(FancyBboxPatch((27.8, 40.2), 7.6, 5.0, boxstyle="round,pad=0.08,rounding_size=0.3",
                            fc="white", ec=C_S3, lw=0.8, zorder=4))
label(31.6, 43.9, "pair MLP", fs=6.4, bold=True, color="#5A2E2E")
label(31.6, 42.0, r"$[h_i; h_j; h_i\!\odot\!h_j; |h_i{-}h_j|]$", fs=5.6, color="#33414F")
label(31.6, 40.9, "channel summaries $h_i, h_j$", fs=5.2, color="#6B7A8C")
arrow(26.6, 41.6, 27.6, 42.2, lw=0.9, ms=7)
ax.add_patch(FancyBboxPatch((36.6, 44.6), 12.4, 1.9, boxstyle="round,pad=0.05,rounding_size=0.25",
                            fc="#F6D5D4", ec=C_S3, lw=0.6, zorder=4))
label(42.8, 45.55, "soft lag-bin regression (CE)", fs=6.0, color="#5A2E2E")
ax.add_patch(FancyBboxPatch((36.6, 41.8), 12.4, 1.9, boxstyle="round,pad=0.05,rounding_size=0.25",
                            fc="#F6D5D4", ec=C_S3, lw=0.6, zorder=4))
label(42.8, 42.75, "lead direction {i→j, j→i, none}", fs=6.0, color="#5A2E2E")
arrow(35.6, 43.4, 36.4, 45.3, lw=0.8, ms=7)
arrow(35.6, 42.6, 36.4, 42.6, lw=0.8, ms=7)
label(42.8, 39.4, r"$\mathcal{L}_{S3} = \mathcal{L}_{lag} + 0.5\,\mathcal{L}_{dir}$", fs=8.6, color=C_S3)
label(42.8, 36.8, "labels recomputed per crop (no leakage)", fs=6.1, color="#4B5563")
label(42.8, 34.9, "k_min = 3, |ρ| ≥ 0.5, lag range K = 48", fs=6.1, color="#4B5563")

# ---- S4 ----
rbox(52.7, 31.5, 47.0, 21.0, fc=F_S4, ec=C_S4, lw=1.3)
label(54.6, 50.8, "S4 · ⑦ quality — three-state missingness & noise (novel)", fs=9.6, bold=True, ha="left", color=C_S4)
label(54.6, 48.9, "state embeddings per timestamp;  reconstruct M from clean targets;  predict noise level q per patch",
      fs=6.9, ha="left", color="#33414F")
seg_w, seg_y = 1.0, 43.2
states = "OOMMOONNOOOMMNNNOOOO"
for i, s in enumerate(states):
    xx = 54.9 + i * seg_w
    if s == "O":
        ax.add_patch(Rectangle((xx, seg_y), seg_w * 0.94, 2.3, fc="white", ec="#A9B4C0", lw=0.5, zorder=4))
    elif s == "M":
        ax.add_patch(Rectangle((xx, seg_y), seg_w * 0.94, 2.3, fc="#D5D9DE", ec="#8E9AA6", lw=0.5, zorder=4))
    else:
        ax.add_patch(Rectangle((xx, seg_y), seg_w * 0.94, 2.3, fc="#F0C9C7", ec=C_S4, lw=0.5, zorder=4))
        zig = np.linspace(-1, 1, 14)
        ax.plot(xx + 0.5 + 0.30 * zig, seg_y + 1.15 + 0.45 * np.sin(zig * 7), color=C_S4, lw=0.6, zorder=5)
label(55.6, 46.6, "O observed", fs=5.8, ha="left", color="#4B5563")
label(61.2, 46.6, "M missing (zeroed)", fs=5.8, ha="left", color="#4B5563")
label(68.8, 46.6, "N noise (q ∈ {1..4})", fs=5.8, ha="left", color=C_S4)
ax.add_patch(FancyBboxPatch((54.9, 38.6), 8.8, 3.0, boxstyle="round,pad=0.06,rounding_size=0.3",
                            fc="white", ec=C_S4, lw=0.8, zorder=4))
label(59.3, 40.7, "lightweight decoder", fs=6.0, color="#4A3566")
label(59.3, 39.5, "→ reconstruct M points", fs=5.9, color="#4A3566")
arrow(59.3, 43.0, 59.3, 41.8, lw=0.9, ms=7)
label(65.6, 40.1, r"$\mathcal{L}_{rec} = \|\hat{x}_{M} - x_{M}\|^2$", fs=8.0, color=C_S4)
label(65.6, 37.8, "clean targets (missing ≠ zero)", fs=6.0, color="#4B5563")
axb = inset(76.8, 36.6, 9.4, 8.8)
q_true = [0.9, 0.35, 0.7, 0.2]
q_pred = [0.82, 0.42, 0.64, 0.28]
xbar = np.arange(4)
axb.bar(xbar - 0.19, q_true, width=0.36, color="#D5D9DE", edgecolor="#8E9AA6", lw=0.5, label="true")
axb.bar(xbar + 0.19, q_pred, width=0.36, color="#B9A0D8", edgecolor=C_S4, lw=0.5, label="pred")
axb.set_xticks(xbar)
axb.set_xticklabels(["q=1", "q=2", "q=3", "q=4"], fontsize=5.2)
axb.tick_params(labelsize=5, colors="#6B7A8C")
axb.set_title("per-patch noise level", fontsize=5.8, pad=1.5, color="#4A3566")
axb.legend(fontsize=4.5, frameon=False, loc="upper right")
for sp in axb.spines.values():
    sp.set_edgecolor("#B9A0D8")
    sp.set_linewidth(0.6)
label(88.0, 45.2, r"$\mathcal{L}_{S4} = \mathcal{L}_{rec} + 0.2\,\mathcal{L}_{q}$", fs=8.4, ha="left", color=C_S4)
label(88.0, 42.4, "one-hot counting labels\nfrom the mask (q ∈ {1..4})", fs=5.8, ha="left", color="#4B5563")
label(88.0, 39.0, "state embeddings make\nobserved / missing / noisy\ndistinct tokens", fs=5.8, ha="left", color="#4B5563")

# ================= 总线走线 =================
# 左边缘：损失总线
line([1.2, 1.2], [75.0, 27.6], color=EDGE, lw=1.3)
for yy, col in [(65.0, C_S1), (65.0, None), (42.0, C_S3), (42.0, None)]:
    pass
line([2.5, 1.2], [65.0, 65.0], color=C_S1, lw=1.1)   # S1 → bus
line([52.7, 49.9, 1.2], [65.0, 65.0, 65.0], color=C_S2, lw=1.0)  # S2 → bus（贴 S1 底）
line([2.5, 1.2], [42.0, 42.0], color=C_S3, lw=1.1)   # S3 → bus
line([52.7, 49.9, 1.2], [42.0, 42.0, 42.0], color=C_S4, lw=1.0)  # S4 → bus
arrow(1.2, 27.6, 2.3, 20.0, lw=1.3, ms=13)          # bus → 总损失

# 中间列缝：z 分线
line([88.75, 88.75], [78.3, 77.3], color=EDGE, lw=1.3)
line([88.75, 50.9], [77.3, 77.3], color=EDGE, lw=1.3)      # 上横线（S1/S2 上方）
line([50.9, 50.9], [77.3, 53.4], color=EDGE, lw=1.3)       # 竖线（列缝）
line([50.9, 26.1], [77.3, 77.3], color=EDGE, lw=0.9)       # 已覆盖；分支：
arrow(26.1, 77.3, 26.1, 75.7, color=C_S1, lw=1.2, ms=11)
arrow(73.5, 77.3, 73.5, 75.7, color=C_S2, lw=1.2, ms=11)
line([50.9, 50.9], [53.4, 53.4], color=EDGE, lw=1.1)
line([50.9, 26.1], [53.4, 53.4], color=C_S3, lw=1.1)
arrow(26.1, 53.4, 26.1, 52.9, color=C_S3, lw=1.2, ms=11)
line([50.9, 73.5], [53.4, 53.4], color=C_S4, lw=1.1)
arrow(73.5, 53.4, 73.5, 52.9, color=C_S4, lw=1.2, ms=11)

# 右边缘：冻结总线
line([98.0, 100.6], [86.0, 86.0], color=C_FROZEN, lw=1.2, ls="--")
line([100.6, 100.6], [86.0, 27.6], color=C_FROZEN, lw=1.2, ls="--")
line([100.6, 57.0], [27.6, 27.6], color=C_FROZEN, lw=1.2, ls="--")
arrow(57.0, 27.6, 57.0, 26.7, color=C_FROZEN, lw=1.2, ms=11, ls="--")
label(100.2, 55.0, "frozen (no grad)", fs=6.2, color=C_FROZEN, ha="left", rot=90)

# ================= Row 4: 损失 + 下游 + 图例 =================
rbox(2.5, 9.5, 38.0, 17.0, fc=F_LOSS)
label(21.5, 24.6, "Total pre-training objective", fs=9.6, bold=True)
label(21.5, 21.6, r"$\mathcal{L} = w_1\mathcal{L}_{S1} + w_2\mathcal{L}_{S2} + w_3\mathcal{L}_{S3} + w_4\mathcal{L}_{S4}$", fs=10.5)
label(21.5, 18.7, "main config:  w1 = 1,  w2 = 0,  w3 = 0.5,  w4 = 1", fs=7.6, color="#6B4E1F", bold=True)
label(21.5, 16.1, "OneCycleLR (lr 1e-3) · grad clip 5.0 · 10 epochs on ETTm1 · batch 256 + TF32", fs=6.6, color="#4B5563")
label(21.5, 13.9, "two views per batch (mask/crop ± scale/shift/jitter), one shared encoder", fs=6.6, color="#4B5563")
label(21.5, 11.6, "stream weights ablated in §4.2 (S2 removed by evidence)", fs=6.4, color="#7A4536")

rbox(42.5, 9.5, 27.0, 17.0, fc=F_DOWN)
label(56.0, 24.6, "Frozen encoder → downstream probes", fs=9.0, bold=True)
label(56.0, 23.3, "(no fine-tuning of the encoder)", fs=6.2, color="#4B5563")
probes = [
    ("Forecasting", "ridge on last patch\nETTm1/ETTh1/ETTh2/ETTm2", 42.9, 16.9),
    ("Classification", "linear probe\nUEA ArrowHead", 56.1, 16.9),
    ("Anomaly", "recon-error threshold\nPSM / SMD", 42.9, 11.9),
    ("Zero-shot", "cross-domain ridge\nm→h transfer", 56.1, 11.9),
]
for name, desc, px, py in probes:
    ax.add_patch(FancyBboxPatch((px, py), 12.4, 4.6, boxstyle="round,pad=0.06,rounding_size=0.3",
                                fc="white", ec="#6E9A76", lw=0.8, zorder=4))
    label(px + 6.2, py + 3.5, name, fs=6.8, bold=True, color="#2E5E38")
    label(px + 6.2, py + 1.7, desc, fs=5.7, color="#4B5563")

rbox(71.0, 9.5, 28.5, 17.0, fc="#F7F8FA")
label(72.6, 24.6, "Semantic dimensions covered", fs=9.0, bold=True, ha="left")
dims = [
    ("① time", "S1", C_S1), ("② measurement", "—", "#9AA3AE"),
    ("③ entity", "S1", C_S1), ("④ pattern", "S2*", C_S2),
    ("⑤ relation", "S3", C_S3), ("⑥ event/state", "—", "#9AA3AE"),
    ("⑦ quality", "S4", C_S4),
]
for i, (dname, stream, col) in enumerate(dims):
    x0 = 72.6 + (i % 2) * 13.6
    y0 = 22.5 - (i // 2) * 2.9
    ax.add_patch(FancyBboxPatch((x0, y0 - 1.1), 2.2, 2.2, boxstyle="round,pad=0.05,rounding_size=0.3",
                                fc=col, ec="white", lw=0.6, zorder=4))
    label(x0 + 3.0, y0, dname, fs=6.5, ha="left", color="#33414F")
    label(x0 + 11.0, y0, stream, fs=6.5, ha="left", color=col, bold=True)
label(72.6, 11.3, "② and ⑥ not modelled (future work);   *S2 ablated in the main", fs=5.9, ha="left", color="#6B7A8C")
label(72.6, 10.1, "configuration (loss weights);   main model = S1 + S3 + S4", fs=5.9, ha="left", color="#6B7A8C")

fig.savefig(os.path.join(OUT_DIR, "semts_architecture_detailed.png"), dpi=170,
            bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT_DIR, "semts_architecture_detailed.svg"),
            bbox_inches="tight", facecolor="white")
print("[figure] docs/figures/semts_architecture_detailed.png / .svg")
plt.close(fig)
