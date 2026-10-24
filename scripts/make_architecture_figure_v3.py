"""绘制 SemTS 架构图 v3（参考 TS2Vec / CoST / SimMTM / TF-C / ShuffleMTM 论文图风格）。

风格要点（对齐参考图）：
- 浅色粉彩填充盒 + 细深色边框（patch 方块、损失公式框、探针框）
- 粗浅灰箭头表示主数据流；红/蓝虚线表示正/负样本关系；绿虚线表示冻结路径；紫箭头表示实例/对比方向
- 向量表示为"小方块条带"（stacked strips），patch 用彩色方块行
- 虚线分组框 + 左上标题；右侧每流一个损失公式框；底部图例

用法: python scripts/make_architecture_figure_v3.py
输出: docs/figures/semts_architecture_v3.png / .svg
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 参考图风格配色 ----
INK = "#2B2B2B"
GRAY_FLOW = "#C3C8CE"
PASTEL = {
    "green": "#D9F0DA", "green_e": "#7FBF84",
    "blue": "#CFE6F5", "blue_e": "#6FA8CC",
    "pink": "#F9DCE7", "pink_e": "#D98CAB",
    "yellow": "#FCEBC2", "yellow_e": "#D9B24C",
    "purple": "#E7E0F6", "purple_e": "#9A86C8",
    "orange": "#FBE3CC", "orange_e": "#D9915B",
    "gray": "#EDEFF2", "gray_e": "#9AA4AE",
}
RED, BLUE_, GREEN, PURPLE = "#D9534F", "#3B7DD8", "#4CAF50", "#8E5EA2"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 8

fig = plt.figure(figsize=(17.0, 11.6), dpi=170)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)
ax.axis("off")


def rbox(x, y, w, h, fc="white", ec=INK, lw=1.0, ls="-", z=2, r=0.6, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.18,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha))


def label(x, y, s, fs=8, color=INK, bold=False, ha="center", va="center", z=7, rot=0, ls_=1.25):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=z,
            fontweight="bold" if bold else "normal", rotation=rot, linespacing=ls_)


def flow(x1, y1, x2, y2, color=GRAY_FLOW, lw=3.0, ls="-", rad=0.0, ms=15, z=1):
    """主数据流：粗灰箭头。"""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


def rel(x1, y1, x2, y2, color=RED, lw=1.2, ls="--", rad=0.0, ms=9, z=5, style="-|>"):
    """关系箭头：正/负样本、冻结、对比方向。"""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


def dgroup(x, y, w, h, title, fs=8.6):
    rbox(x, y, w, h, fc="none", ec="#6B7480", lw=1.1, ls=(0, (5, 3)), z=1)
    label(x + 1.0, y + h - 1.1, title, fs=fs, bold=True, ha="left", color="#2F3A45")


def inset(x, y, w, h):
    # 主画布 ylim 为 0..70（xlim 0..100），换算到 figure 分数
    return fig.add_axes([x / 100, y / 70, w / 100, h / 70])


def strip_column(x, y, n_rows=5, w=0.85, h=0.42, gap=0.10, colors=None):
    """一列向量条带（TF-C 风格）。"""
    colors = colors or [PASTEL["blue"], PASTEL["blue"], PASTEL["pink"], PASTEL["pink"], PASTEL["gray"]]
    for i in range(n_rows):
        ax.add_patch(Rectangle((x, y + i * (h + gap)), w, h, fc=colors[i % len(colors)],
                               ec="#7E8792", lw=0.45, zorder=5))


def patch_row(x, y, n, w=1.30, h=1.55, gap=0.10, state=None, base=None):
    """patch 方块行（ShuffleMTM 风格）：state 为 0/1 列表，1=保留（彩色），0=掩码（灰虚线）。"""
    base = base or [PASTEL["green"], PASTEL["blue"], PASTEL["pink"], PASTEL["yellow"]]
    for i in range(n):
        keep = True if state is None else bool(state[i])
        fc = base[i % len(base)] if keep else "#E3E6EA"
        ec = PASTEL["green_e"] if keep else "#B7BEC6"
        ax.add_patch(Rectangle((x + i * (w + gap), y), w, h, fc=fc, ec=ec, lw=0.6,
                               ls="-" if keep else (0, (2, 1.5)), zorder=5))


rng = np.random.default_rng(3)

# ======================= 标题 =======================
label(2.0, 68.3, "SemTS: self-supervised pre-training with disentangled temporal semantics",
      fs=14.5, bold=True, ha="left")
label(2.0, 66.5,
      "dual-view contrast on a shared channel-independent patch Transformer;  four ablatable semantic streams —"
      "  main model = S1 (time/entity) + S3 (lead-lag) + S4 (quality),   S2 (pattern) ablated (w2 = 0)",
      fs=8.2, color="#4B5563", ha="left")

# ======================= 左：输入 + 两视图 =======================
dgroup(1.5, 33.0, 25.0, 31.5, "Input & two contextual views")

# 输入窗口卡
rbox(2.6, 53.2, 22.8, 9.4, fc=PASTEL["gray"], ec="#9AA4AE")
label(14.0, 61.6, "Multi-channel window", fs=9.0, bold=True)
label(14.0, 60.2, r"$x \in \mathbb{R}^{C \times L}$,  L = 336,  z-score", fs=7.6)
t = np.linspace(0, 6 * np.pi, 240)
for k, (yy, ph, f, col) in enumerate([(58.2, 0.0, 1.0, "#4E7CA6"), (56.4, 1.3, 1.7, "#A67B4E"), (54.6, 2.2, 2.6, "#6C9E6C")]):
    ax.plot(3.8 + np.linspace(0, 20.4, t.size), yy + 0.30 * np.sin(f * t + ph) + 0.05 * rng.normal(size=t.size),
            color=col, lw=0.9, zorder=6)

label(3.0, 51.8, "View A:  mask p = 0.5 + random crop", fs=7.2, ha="left", color="#33414F")
patch_row(4.2, 49.2, 12, state=[1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
label(3.0, 47.9, "View B:  A + scale / shift / jitter", fs=7.2, ha="left", color="#33414F")
patch_row(4.2, 45.3, 12, state=[1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
          base=[PASTEL["yellow"], PASTEL["orange"], PASTEL["yellow"], PASTEL["orange"]])
# patch masking 上箭头（ShuffleMTM 风格）
rel(3.0, 45.0, 3.0, 48.9, color="#6B7480", lw=1.0, ls="-", ms=9)
label(2.35, 47.0, "patch\nmasking", fs=6.2, color="#4B5563", rot=90)

# 两视图 → 编码器
flow(24.3, 46.0, 29.6, 57.4, rad=-0.06)
flow(24.3, 50.0, 29.6, 57.0, rad=-0.03)
label(27.7, 43.0, "two views,\nno labels", fs=5.4, color="#4B5563", rot=0)

# ======================= 中：编码器 =======================
dgroup(28.6, 33.0, 28.2, 31.5, "Encoder — shared across channels")
rbox(29.9, 53.4, 25.4, 8.0, fc=PASTEL["gray"], ec="#7E8792", lw=1.2)
label(42.6, 60.4, "Patch-based Transformer Encoder  × 4", fs=9.2, bold=True)
label(42.6, 58.8, r"Conv1d patch $p{=}16,\ s{=}8 \Rightarrow P{=}41$", fs=7.4)
label(42.6, 57.2, "d_model 256 · 8 heads · d_ff 512 · ≈5.6 M", fs=7.4)
label(42.6, 55.4, "weights shared across C channels · latent mask p=0.5", fs=6.8, color="#4B5563")

# 缩放块：Transformer block 内部（ShuffleMTM 底部风格）
rbox(30.2, 34.0, 24.6, 16.8, fc="#F7F8FA", ec="#9AA4AE", lw=1.0)
label(42.5, 49.6, "Transformer block (zoomed)", fs=7.8, bold=True, color="#2F3A45")
chain = [
    ("Projection + Pos. Embedding", PASTEL["blue"], PASTEL["blue_e"]),
    ("Multi-Head Attention (8h)", PASTEL["green"], PASTEL["green_e"]),
    ("Add & Norm", PASTEL["orange"], PASTEL["orange_e"]),
    ("Feed Forward (d_ff=512)", PASTEL["green"], PASTEL["green_e"]),
    ("Add & Norm", PASTEL["orange"], PASTEL["orange_e"]),
]
yy = 45.4
for i, (name, fc, ec) in enumerate(chain):
    ax.add_patch(FancyBboxPatch((34.2, yy), 13.2, 2.0, boxstyle="round,pad=0.06,rounding_size=0.3",
                                fc=fc, ec=ec, lw=0.7, zorder=5))
    label(40.8, yy + 1.0, name, fs=6.4, color="#26323C")
    if i < 4:
        rel(40.8, yy, 40.8, yy - 1.05, color="#4B5563", lw=1.0, ls="-", ms=8)
    yy -= 2.3
label(49.6, 42.0, "× N blocks", fs=6.4, color="#4B5563", ha="left")
# weight sharing 虚线（ShuffleMTM 的 weight sharing 双箭头）
rel(36.6, 52.9, 36.6, 53.2, color="#6B7480", lw=1.0, ls="--", ms=8)
rel(37.4, 53.2, 37.4, 52.9, color="#6B7480", lw=1.0, ls="--", ms=8)
label(37.0, 52.0, "weight sharing", fs=6.0, color="#4B5563")

# ======================= z 表示 =======================
flow(55.5, 57.4, 57.8, 55.2)
for k, xx in enumerate([58.0, 59.1, 60.2]):
    strip_column(xx, 44.0, n_rows=6, colors=[PASTEL["blue"], PASTEL["blue"], PASTEL["pink"],
                                             PASTEL["pink"], PASTEL["yellow"], PASTEL["yellow"]])
label(60.6, 42.6, r"$z \in \mathbb{R}^{C \times P \times d}$   (patch reps)", fs=8.0, ha="left")
label(60.6, 41.2, "last patch → linear probe", fs=6.4, ha="left", color="#4B5563")

# ======================= 右：四条流 =======================
SX, SW = 67.6, 29.4
stream_specs = [
    ("S1 · ① time + ③ entity — hierarchical contrast", PASTEL["blue"], PASTEL["blue_e"], 54.0, 10.6),
    ("S2 · ④ pattern — trend/seasonal (ABLATED: w2 = 0)", PASTEL["gray"], "#9AA4AE", 41.6, 10.6),
    ("S3 · ⑤ relation — cross-channel lead-lag (novel)", PASTEL["pink"], PASTEL["pink_e"], 28.6, 11.2),
    ("S4 · ⑦ quality — three-state missingness (novel)", PASTEL["purple"], PASTEL["purple_e"], 15.6, 11.2),
]

# --- S1 ---
x0, y0, h0 = SX, 54.0, 10.6
rbox(x0, y0, SW, h0, fc=PASTEL["blue"], ec=PASTEL["blue_e"], lw=1.2)
label(x0 + 1.0, y0 + h0 - 1.1, stream_specs[0][0], fs=7.8, bold=True, ha="left", color="#2F4F73")
sm = np.full((6, 6), 0.12)
for i in range(6):
    for j in range(6):
        sm[i, j] += 0.78 * np.exp(-abs(i - j) / 1.5)
np.fill_diagonal(sm, 1.0)
axm = inset(x0 + 1.4, y0 + 1.3, 7.4, 6.4)
axm.imshow(sm, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
for i in range(6):
    axm.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=RED, lw=0.8))
axm.set_xticks([])
axm.set_yticks([])
axm.set_title("A↔B cosine", fontsize=5.6, pad=1.2, color="#2F4F73")
# 正/负对示意（红色虚线=正样本，蓝色虚线=负样本）
rel(x0 + 9.8, y0 + 7.0, x0 + 11.6, y0 + 7.0, color=RED, lw=1.4, ls="--", rad=0.35, ms=9)
label(x0 + 12.4, y0 + 7.0, "pos", fs=5.6, color=RED, ha="left")
rel(x0 + 9.8, y0 + 5.4, x0 + 11.6, y0 + 5.4, color=BLUE_, lw=1.4, ls="--", rad=-0.35, ms=9)
label(x0 + 12.4, y0 + 5.4, "neg", fs=5.6, color=BLUE_, ha="left")
label(x0 + 9.8, y0 + 3.6, "multi-scale max-pool", fs=5.8, ha="left", color="#4B5563")
label(x0 + 9.8, y0 + 2.4, "instance = series + channel id", fs=5.8, ha="left", color="#4B5563")
label(x0 + 9.8, y0 + 1.2, "chunked log-sum-exp", fs=5.6, ha="left", color="#6B7A8C")
rbox(x0 + 19.2, y0 + 3.0, 9.4, 4.6, fc="white", ec=PASTEL["blue_e"], lw=0.9)
label(x0 + 23.9, y0 + 6.1, r"$\mathcal{L}_{S1}$", fs=8.0, color="#2F4F73")
label(x0 + 23.9, y0 + 4.2, r"$=\mathcal{L}_{temp}+\lambda\mathcal{L}_{inst}$", fs=6.2, color="#2F4F73")

# --- S2 ---
x0, y0, h0 = SX, 41.6, 10.6
rbox(x0, y0, SW, h0, fc=PASTEL["gray"], ec="#9AA4AE", lw=1.0, ls=(0, (4, 2)))
label(x0 + 1.0, y0 + h0 - 1.1, stream_specs[1][0], fs=7.8, bold=True, ha="left", color="#5B6570")
for k, ks in enumerate([1, 2, 4]):
    ax.add_patch(FancyBboxPatch((x0 + 1.4, y0 + 6.4 - k * 1.9), 2.0, 1.4,
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                fc="#DCE0E5", ec="#9AA4AE", lw=0.6, zorder=5))
    label(x0 + 2.4, y0 + 7.1 - k * 1.9, f"k={ks}", fs=5.4, color="#3F4A54")
label(x0 + 4.2, y0 + 6.6, "TFD: causal conv experts", fs=6.2, ha="left", color="#4B5563")
label(x0 + 4.2, y0 + 4.2, "→ time NT-Xent + queue", fs=6.2, ha="left", color="#4B5563")
freq = np.abs(np.sin(np.linspace(0.4, 9, 18))) * 0.8 + 0.1
for i, v in enumerate(freq):
    ax.add_patch(Rectangle((x0 + 4.4 + i * 0.22, y0 + 1.2), 0.16, v * 1.8, fc="#AEB6BF", ec="none", zorder=5))
label(x0 + 8.8, y0 + 4.4, "SFD: DFT → ℂ-linear → iDFT", fs=5.8, ha="left", color="#4B5563")
label(x0 + 8.8, y0 + 3.4, "amp + phase contrast", fs=5.8, ha="left", color="#4B5563")
rbox(x0 + 19.2, y0 + 2.8, 9.4, 4.6, fc="white", ec="#9AA4AE", lw=0.9, ls=(0, (4, 2)))
label(x0 + 23.9, y0 + 6.1, r"$\mathcal{L}_{S2}$", fs=8.0, color="#5B6570")
label(x0 + 23.9, y0 + 4.6, r"$=\mathcal{L}_{time}$", fs=6.2, color="#5B6570")
label(x0 + 23.9, y0 + 3.6, r"$+\frac{\alpha}{2}(\mathcal{L}_{amp}+\mathcal{L}_{phase})$", fs=5.6, color="#5B6570")

# --- S3 ---
x0, y0, h0 = SX, 28.6, 11.2
rbox(x0, y0, SW, h0, fc=PASTEL["pink"], ec=PASTEL["pink_e"], lw=1.2)
label(x0 + 1.0, y0 + h0 - 1.1, stream_specs[2][0], fs=7.8, bold=True, ha="left", color="#6E2B44")
ks = np.arange(-48, 49)
rho = 0.18 + 0.72 * np.exp(-((ks - 3.0) ** 2) / (2 * 3.2 ** 2))
axr = inset(x0 + 1.3, y0 + 1.4, 7.4, 7.2)
axr.plot(ks, rho, color=RED, lw=1.2)
axr.axvline(3, color="#7A3034", lw=0.8, ls="--")
axr.annotate("k*=+3", xy=(3, 0.88), xytext=(11, 0.62), fontsize=5.2, color="#7A3034",
             arrowprops=dict(arrowstyle="->", lw=0.5, color="#7A3034"))
axr.set_title(r"$\rho(k)$", fontsize=5.6, pad=1.0, color="#6E2B44")
axr.set_xticks([-48, 0, 48])
axr.tick_params(labelsize=4.4, colors="#8A6A6C", length=1.5)
for sp in axr.spines.values():
    sp.set_edgecolor("#CBA9AB")
    sp.set_linewidth(0.6)
axs = inset(x0 + 9.4, y0 + 1.4, 6.2, 7.2)
bins = np.arange(-6, 7)
soft = 0.15 + 0.85 * np.exp(-((bins - 3) ** 2) / (2 * 1.4 ** 2))
axs.bar(bins, soft, color="#E7A8A6", edgecolor=RED, lw=0.45, width=0.85)
axs.set_title("soft label", fontsize=5.6, pad=1.0, color="#6E2B44")
axs.set_xticks([-6, 0, 6])
axs.tick_params(labelsize=4.4, colors="#8A6A6C", length=1.5)
for sp in axs.spines.values():
    sp.set_edgecolor("#CBA9AB")
    sp.set_linewidth(0.6)
ax.add_patch(FancyBboxPatch((x0 + 16.2, y0 + 5.8), 9.4, 3.0, boxstyle="round,pad=0.05,rounding_size=0.25",
                            fc="white", ec=PASTEL["pink_e"], lw=0.7, zorder=5))
label(x0 + 20.9, y0 + 8.15, r"$[h_i;h_j;h_i\!\odot\!h_j;|h_i{-}h_j|]$", fs=5.2, color="#6E2B44")
label(x0 + 20.9, y0 + 6.7, "pair MLP", fs=6.0, color="#6E2B44")
label(x0 + 16.4, y0 + 4.6, "→ soft lag-bin CE  +  lead direction", fs=5.8, ha="left", color="#4B5563")
label(x0 + 16.4, y0 + 3.2, "labels: FFT lagged x-corr, k_min=3", fs=5.4, ha="left", color="#6B7A8C")
rbox(x0 + 19.2, y0 + 1.2, 9.4, 2.2, fc="white", ec=PASTEL["pink_e"], lw=0.9)
label(x0 + 23.9, y0 + 2.3, r"$\mathcal{L}_{S3}=\mathcal{L}_{lag}+0.5\mathcal{L}_{dir}$", fs=6.2, color="#6E2B44")

# --- S4 ---
x0, y0, h0 = SX, 15.6, 11.2
rbox(x0, y0, SW, h0, fc=PASTEL["purple"], ec=PASTEL["purple_e"], lw=1.2)
label(x0 + 1.0, y0 + h0 - 1.1, stream_specs[3][0], fs=7.8, bold=True, ha="left", color="#4A3566")
states = "OOMMOONNOOOMMNNNOOOO"
sw_, sy_ = 0.66, y0 + 6.9
for i, s in enumerate(states):
    xx = x0 + 1.4 + i * sw_
    if s == "O":
        ax.add_patch(Rectangle((xx, sy_), sw_ * 0.9, 1.6, fc="white", ec="#9AA4AE", lw=0.5, zorder=5))
    elif s == "M":
        ax.add_patch(Rectangle((xx, sy_), sw_ * 0.9, 1.6, fc="#D5D9DE", ec="#8E9AA6", lw=0.5, zorder=5))
    else:
        ax.add_patch(Rectangle((xx, sy_), sw_ * 0.9, 1.6, fc="#F0C9C7", ec=PURPLE, lw=0.5, zorder=5))
        zig = np.linspace(-1, 1, 12)
        ax.plot(xx + 0.30 + 0.20 * zig, sy_ + 0.8 + 0.30 * np.sin(zig * 7), color=PURPLE, lw=0.5, zorder=6)
label(x0 + 1.4, y0 + 6.1, "O observed   M missing   N noise(q)", fs=5.4, ha="left", color="#4B5563")
ax.add_patch(FancyBboxPatch((x0 + 1.4, y0 + 1.4), 6.4, 2.8, boxstyle="round,pad=0.05,rounding_size=0.25",
                            fc="white", ec=PASTEL["purple_e"], lw=0.7, zorder=5))
label(x0 + 4.6, y0 + 3.3, "decoder → M", fs=5.8, color="#4A3566")
label(x0 + 4.6, y0 + 2.2, "clean targets", fs=5.4, color="#4A3566")
axb = inset(x0 + 8.8, y0 + 1.2, 6.2, 4.4)
q_true = [0.9, 0.35, 0.7, 0.2]
q_pred = [0.82, 0.42, 0.64, 0.28]
xb = np.arange(4)
axb.bar(xb - 0.19, q_true, width=0.34, color="#D5D9DE", edgecolor="#8E9AA6", lw=0.45)
axb.bar(xb + 0.19, q_pred, width=0.34, color="#B9A0D8", edgecolor=PURPLE, lw=0.45)
axb.set_xticks(xb)
axb.set_xticklabels(["1", "2", "3", "4"], fontsize=4.6)
axb.tick_params(labelsize=4.2, colors="#7A6A8C", length=1.5)
axb.set_title("noise level q", fontsize=5.4, pad=1.0, color="#4A3566")
for sp in axb.spines.values():
    sp.set_edgecolor("#B9A0D8")
    sp.set_linewidth(0.6)
label(x0 + 15.6, y0 + 5.6, "state embeddings make\nO / M / N distinct tokens", fs=5.6, ha="left", color="#4B5563")
rbox(x0 + 19.2, y0 + 1.2, 9.4, 2.6, fc="white", ec=PASTEL["purple_e"], lw=0.9)
label(x0 + 23.9, y0 + 3.0, r"$\mathcal{L}_{S4}=\mathcal{L}_{rec}+0.2\mathcal{L}_{q}$", fs=6.2, color="#4A3566")

# z → 四流
for yy in (59.3, 46.9, 34.2, 21.2):
    flow(60.6, 46.2, 67.2, yy, rad=0.0 if yy < 46 else -0.06)

# 总损失箱
rbox(SX, 7.6, SW, 5.6, fc=PASTEL["yellow"], ec=PASTEL["yellow_e"], lw=1.2)
label(SX + SW / 2, 11.4, r"$\mathcal{L} = w_1\mathcal{L}_{S1} + w_2\mathcal{L}_{S2} + w_3\mathcal{L}_{S3} + w_4\mathcal{L}_{S4}$",
      fs=8.4, color="#6B4E1F")
label(SX + SW / 2, 9.1, "main config:  w1 = 1,  w2 = 0,  w3 = 0.5,  w4 = 1", fs=6.6, color="#6B4E1F", bold=True)
# 损失总线（右缘）
ax.plot([98.2, 98.2], [59.5, 12.6], color=PASTEL["yellow_e"], lw=1.2, zorder=1)
for yy in (58.6, 45.4, 31.2, 17.6):
    ax.plot([96.4, 98.2], [yy, yy], color=PASTEL["yellow_e"], lw=1.0, zorder=1)
flow(98.2, 12.6, 96.6, 11.6, color=PASTEL["yellow_e"], lw=1.4, ms=11)

# ======================= 底部：下游 + 图例 =======================
dgroup(1.5, 9.0, 64.5, 20.0, "Frozen encoder → downstream probes (no fine-tuning)")
probes = [
    ("Forecasting", "ridge on last patch\nETTm1 / ETTh1 / ETTh2 / ETTm2", 3.0, 18.0),
    ("Classification", "linear probe\nUEA ArrowHead", 19.0, 18.0),
    ("Anomaly", "recon-error F1\nPSM / SMD", 35.0, 18.0),
    ("Zero-shot", "cross-domain ridge\nETTm1 → ETTh1", 51.0, 18.0),
]
for name, desc, px, py in probes:
    rbox(px, py, 14.0, 6.4, fc=PASTEL["green"], ec=PASTEL["green_e"], lw=0.9)
    label(px + 7.0, py + 4.7, name, fs=6.8, bold=True, color="#2E5E38")
    label(px + 7.0, py + 2.4, desc, fs=5.8, color="#3F4A54")
# 冻结虚线（z → 下游）
rel(60.0, 41.0, 60.0, 29.6, color=GREEN, lw=1.3, ls="--", ms=10)
label(61.0, 35.2, "frozen (no grad)", fs=6.0, color=GREEN, ha="left")

# 图例
rbox(1.5, 1.2, 64.5, 6.6, fc="#F7F8FA", ec="#9AA4AE", lw=0.9)
leg_items = [
    ("positive pair", RED, "--"), ("negative pair", BLUE_, "--"),
    ("contrast direction", GREEN, "--"), ("instance / lag", PURPLE, "-"),
    ("main data flow", GRAY_FLOW, "-"),
]
for i, (name, col, ls) in enumerate(leg_items):
    xx = 3.0 + i * 12.2
    rel(xx, 6.0, xx + 3.2, 6.0, color=col, lw=1.6, ls=ls, ms=9)
    label(xx + 3.6, 6.0, name, fs=5.8, ha="left", color="#3F4A54")
dims = ["① time", "② measurement", "③ entity", "④ pattern", "⑤ relation", "⑥ event/state", "⑦ quality"]
cover = [PASTEL["blue_e"], "#9AA4AE", PASTEL["blue_e"], "#9AA4AE", PASTEL["pink_e"], "#9AA4AE", PASTEL["purple_e"]]
label(3.0, 2.6, "Semantics:", fs=6.0, ha="left", color="#4B5563")
for i, (d, c) in enumerate(zip(dims, cover)):
    xx = 12.0 + i * 7.6
    ax.add_patch(FancyBboxPatch((xx, 2.0), 1.4, 1.4, boxstyle="round,pad=0.03,rounding_size=0.2",
                                fc=c, ec="white", lw=0.5, zorder=5))
    label(xx + 1.5, 2.7, d, fs=5.4, ha="left", color="#3F4A54")

fig.savefig(os.path.join(OUT_DIR, "semts_architecture.png"), dpi=170,
            bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT_DIR, "semts_architecture.svg"), bbox_inches="tight", facecolor="white")
print("[figure] docs/figures/semts_architecture.png / .svg (v3 style)")
plt.close(fig)
