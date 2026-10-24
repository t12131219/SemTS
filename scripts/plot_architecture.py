"""SemTS 架构图（PNG + PDF 矢量版）。用法：python scripts/plot_architecture.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(16, 11), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

C = {
    "view": "#D6EAF8", "enc": "#AED6F1", "enc_inner": "#EBF5FB",
    "s1": "#D5F5E3", "s2": "#FADBD8", "s3": "#FCF3CF", "s4": "#E8DAEF",
    "loss": "#FDEBD0", "down": "#EAECEE", "text": "#1B2631",
}


def box(x0, y0, w, h, title, sub, fc, title_zh=None, ls="-", lw=1.6, fs=11, tfs=13):
    p = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.6",
                       fc=fc, ec="#2C3E50", lw=lw, linestyle=ls)
    ax.add_patch(p)
    ax.text(x0 + w / 2, y0 + h / 2, title, ha="center", va="center",
            fontsize=tfs, fontweight="bold", color=C["text"])
    if sub:
        ax.text(x0 + w / 2, y0 + h / 2 - 4.5, sub, ha="center", va="center",
                fontsize=fs, color="#34495E")


def arrow(x0, y0, x1, y1, style="-", color="#2C3E50", lw=1.6, rad=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                        linestyle=style, color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


# ===== 标题 =====
ax.text(50, 97.5, "SemTS 架构：显式语义流式时序自监督预训练", ha="center",
        fontsize=17, fontweight="bold", color=C["text"])

# ===== 输入 =====
box(2, 78, 15, 12, "输入窗口\nx ∈ R^(L×C)", "多变量时序（z-score）", C["view"])
ax.text(9.5, 76.2, "L=336, C=通道数", ha="center", fontsize=9, color="#566573")

# ===== 视图构造 =====
box(2, 58, 15, 12, "视图 A", "随机裁剪 + 隐向量掩码 (p=0.5)", C["view"], fs=9.5)
box(2, 40, 15, 12, "视图 B", "A + 尺度/平移/抖动\n+ 三态掩码 O/M/N", C["view"], fs=9.5)
arrow(9.5, 78, 9.5, 70.4)
arrow(9.5, 58, 9.5, 52.4)

# ===== 共享骨干 =====
box(26, 42, 30, 46, "", None, C["enc"])
ax.text(41, 84, "共享 Patch 编码器（通道独立）", ha="center", fontsize=13.5,
        fontweight="bold", color=C["text"])
box(29, 68, 24, 12, "Patch Embedding", "Conv1d（patch=16, stride=8）", C["enc_inner"], fs=10)
box(29, 50, 24, 13, "Transformer Encoder", "L=4 层，d=256，8 头，d_ff=512", C["enc_inner"], fs=10)
box(29, 42, 24, 6, "", None, C["enc_inner"])
ax.text(41, 44.6, "z：逐通道 patch 表示（通道间权重共享，≈5.6M 参数）", ha="center", fontsize=9.5,
        color="#1B2631")
arrow(41, 68, 41, 63.4)
arrow(17.2, 64, 28.7, 74, rad=-0.15)
arrow(17.2, 46, 28.7, 56, rad=0.15)

# ===== 四条语义流 =====
streams = [
    ("S1  层级对比（①时间 ③实体）", "时序 NT-Xent + 实例 NT-Xent（多尺度）", "L_S1", C["s1"], 66, "-", 1.0),
    ("S2  模式头（④，主配置 w2=0）", "TFD 趋势 + SFD 季节（时/频域对比）", "L_S2", C["s2"], 48, "--", 0.0),
    ("S3  关系头 ★（⑤滞后/因果）", "FFT 滞后互相关弱标签 → 软滞后回归\n+ 领先方向判别", "L_S3", C["s3"], 30, "-", 0.5),
    ("S4  质量头 ★（⑦质量/缺失）", "三态掩码重建（M 点干净目标）\n+ 逐 patch 噪声等级预测 q", "L_S4", C["s4"], 12, "-", 1.0),
]
for i, (title, sub, loss, fc, y, ls, w) in enumerate(streams):
    box(64, y, 34, 15, title, sub, fc, fs=9.5, tfs=11.5, ls=ls)
    ax.text(96.3, y + 2.2, loss, ha="right", va="center", fontsize=11,
            fontweight="bold", color="#922B21")
    arrow(56.2, 42 + 3, 63.7, y + 7.5)

# ===== 总损失 =====
box(64, 2, 34, 10, "总损失 L = w1·L_S1 + w2·L_S2 + w3·L_S3 + w4·L_S4",
    "主配置 w2=0, w3=0.5 · OneCycleLR(1e-3) · 梯度裁剪 5.0", C["loss"], fs=10, tfs=11)
arrow(81, 12, 81, 12.4)

# ===== 下游任务 =====
box(2, 2, 52, 9, "下游任务（冻结编码器）", None, C["down"])
ax.text(28, 4.8, "预测：末 patch 表示 + 岭回归   ·   分类：线性探针   ·   异常：重建误差 F1   ·   零样本：冻结 + 源域岭回归",
        ha="center", fontsize=10, color=C["text"])
arrow(26, 46, 24, 11.2, style=(0, (5, 4)), color="#7F8C8D", rad=0.35)

# ===== 图例/注记 =====
ax.text(50, 90.5, "★ = 本文创新任务", ha="center", fontsize=10.5, color="#922B21")

import os
os.makedirs("docs/figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"docs/figures/semts_architecture.{ext}", bbox_inches="tight",
                facecolor="white")
print("saved docs/figures/semts_architecture.png / .pdf")
