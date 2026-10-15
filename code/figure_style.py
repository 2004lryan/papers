"""
figure_style.py: Nature 子刊级科研制图统一风格（/scientific-figure-making + /nature-figure 规范）

运行方式:
    作为模块导入使用，不单独运行
    其他实验脚本通过 `from figure_style import apply_nature_style, WONG, NF_FIGSIZES` 调用

按项目约定第三章约定，本文件作为"工具/库类"模块，豁免数字前缀。
按第八章 Item 1: Nature Machine Intelligence / Nature Food 视觉标准。

主要导出:
    WONG          — Wong (2011 Nat Methods) 8 色色盲安全调色板（dict + list 双形式）
    NF_FIGSIZES   — Nature 子刊常用 figure 尺寸（单栏 / 双栏 / 全幅）
    apply_nature_style() — 一次性应用全局 matplotlib rcParams
    add_panel_label(ax, label) — 添加面板字母标签（a/b/c …，左上 bold）
    save_nature(fig, file_stem, index, dpi=600, formats=('pdf','svg')) — 双格式 + 双语保存
    style_axes(ax) — top/right spine off + tick formatter + grid
    color_palette(n, family='wong') — 取 n 色

NF 关键约束（强制）:
    - 字体: sans-serif (Helvetica → Arial fallback)
    - 字号: 7 pt body / 8 pt axis label / 9 pt panel title
    - 线宽: 0.5 pt spine / 0.75 pt axis / 1.0 pt data line
    - top + right spine: 默认隐藏
    - 误差条: capsize=0, elinewidth=0.75
    - legend: frameon=False
    - 颜色: Wong 8 色优先；连续场景用 viridis / cividis；禁止红绿对比
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ─── Wong 调色板（Nat Methods 2011 标准，色盲安全） ──────────────────────────
WONG = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "red":     "#D55E00",
    "pink":    "#CC79A7",
}

# 按通用 contrast 排序的 list 形式
WONG_LIST = [
    "#0072B2",  # blue（主线 / 我方方法）
    "#D55E00",  # red（强调 / 关键基线）
    "#009E73",  # green（积极 / 改进）
    "#E69F00",  # orange（次要类别）
    "#CC79A7",  # pink（第三类别）
    "#56B4E9",  # skyblue（辅助）
    "#F0E442",  # yellow（高亮）
    "#000000",  # black（基线 / 参考）
]

# 项目级 baseline 配色（保持跨图一致性）
PROJECT_PALETTE = {
    "MechFoFR_full":   "#0072B2",  # 我方主方法
    "MechFoFR_lite":   "#56B4E9",
    "FoFR":            "#009E73",
    "Ridge":           "#E69F00",
    "Mode":            "#CC79A7",
    "Random":          "#000000",
    "PINN":            "#D55E00",
    "Foundation":      "#F0E442",
    "Conformal":       "#0072B2",
    "Point":           "#D55E00",
    "AbstainReroute":  "#009E73",
}

# Region 配色（CO₂eq 区域图）
REGION_PALETTE = {
    "CN (China)":  "#D55E00",
    "IN (India)":  "#E69F00",
    "BR (Brazil)": "#009E73",
    "EU (EU-27)":  "#0072B2",
    "US (USA)":    "#CC79A7",
}

# ─── 全稿统一“实体→颜色”规范（9 图共用唯一真源；色盲安全） ──────────────────
# 同一语义实体在任意图中必须同色；蓝/橙为主 2 路对比；禁止红绿单独区分两组。
ENTITY_COLORS = {
    # 我方/可恢复/界-认证 → 蓝
    "functional":   WONG["blue"],     # 函数型(本文) / 本文方法
    "ours":         WONG["blue"],
    "recoverable":  WONG["blue"],     # 可恢复 / go region 主色
    "certified":    WONG["blue"],     # κ_K-界认证
    # 离散插值 → 橙
    "discrete":     WONG["orange"],   # 离散插值
    "unrecoverable": WONG["orange"],  # 不可恢复 / no-go region 主色
    # 深度 1D-CNN → 天蓝
    "deepcnn":      WONG["skyblue"],
    # FoodSpec-FM / 谱 Transformer / Foundation → 粉
    "foundation":   WONG["pink"],
    "transformer":  WONG["pink"],
    # 监督式 PDS / 参考基线 → 黑
    "pds":          WONG["black"],
    "reference":    WONG["black"],
    # 启发式（条件数/K/SNR/Kennard-Stone 等）→ 绿（不与红同时作两组唯一区分）
    "heuristic":    WONG["green"],
}

# 连续场（heatmap / K 梯度 / 密度）唯一色图：cividis（最色盲安全）
CONTINUOUS_CMAP = "cividis"

# 区域填充：不可恢复/no-go = 橙淡彩；可恢复/go = 蓝淡彩（与 fig1 既有填充一致）
REGION_FILL = {
    "go":   WONG["blue"],     # 配 alpha≈0.07–0.10
    "nogo": WONG["orange"],
}

# ─── Nature 子刊 figure 尺寸（Nat Food / NMI / Nat Plants 实际投稿规格） ────
# 单位: inches；Nature single column = 89 mm = 3.50 in；double = 183 mm = 7.20 in
NF_FIGSIZES = {
    "single_short":   (3.5, 2.5),    # 单栏短：横向 89 mm × 高 64 mm
    "single_medium":  (3.5, 3.0),    # 单栏中：89 × 76
    "single_tall":    (3.5, 4.5),    # 单栏高：89 × 114
    "double_short":   (7.2, 3.0),    # 双栏短：183 × 76
    "double_medium":  (7.2, 4.5),    # 双栏中：183 × 114
    "double_tall":    (7.2, 6.0),    # 双栏高：183 × 152
    "full_page":      (7.2, 9.0),    # 全页：183 × 229
    "panel_2x2":      (7.2, 6.0),    # 2×2 多面板
    "panel_2x3":      (7.2, 4.5),    # 2×3 多面板
    "panel_1x3":      (7.2, 2.6),    # 1×3 横向
}

# ─── matplotlib 全局 rcParams（一次性应用） ──────────────────────────────────
_NF_RCPARAMS = {
    # 字体
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            7.0,
    "axes.labelsize":       8.0,
    "axes.titlesize":       9.0,
    "xtick.labelsize":      7.0,
    "ytick.labelsize":      7.0,
    "legend.fontsize":      7.0,
    "figure.titlesize":     9.0,

    # 线宽
    "axes.linewidth":       0.75,
    "lines.linewidth":      1.0,
    "lines.markersize":     4.0,
    "patch.linewidth":      0.5,
    "grid.linewidth":       0.4,
    "xtick.major.width":    0.75,
    "ytick.major.width":    0.75,
    "xtick.minor.width":    0.5,
    "ytick.minor.width":    0.5,

    # tick 长度
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "xtick.minor.size":     1.5,
    "ytick.minor.size":     1.5,
    "xtick.direction":      "out",
    "ytick.direction":      "out",

    # spine
    "axes.spines.top":      False,
    "axes.spines.right":    False,

    # legend
    "legend.frameon":       False,
    "legend.handlelength":  1.5,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "legend.borderpad":     0.3,
    "legend.labelspacing":  0.3,

    # 误差条
    "errorbar.capsize":     0.0,  # NF 风格无 cap

    # 网格
    "axes.grid":            False,  # 默认 off，需要时 ax.grid(alpha=0.25)
    "grid.alpha":           0.25,
    "grid.linestyle":       "--",

    # 保存
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "savefig.transparent":  False,

    # PDF
    "pdf.fonttype":         42,   # TrueType embedded (不要 Type 3)
    "ps.fonttype":          42,

    # 数学
    "mathtext.fontset":     "dejavusans",

    # 防中文负号乱码（与 setup_chinese_font 协作）
    "axes.unicode_minus":   False,
}


def apply_nature_style(also_chinese: bool = True) -> None:
    """一次性应用 Nature 子刊级 matplotlib 风格。

    Args:
        also_chinese: 若 True，额外设置中文字体回退链（避免双语图标题乱码）
    """
    matplotlib.rcParams.update(_NF_RCPARAMS)
    if also_chinese:
        # 在英文字体优先链后追加中文 fallback；与 export_utils.setup_chinese_font 兼容
        cn_fonts = ["SimHei", "Arial Unicode MS", "Microsoft YaHei", "PingFang SC"]
        cur = list(matplotlib.rcParams["font.sans-serif"])
        for f in cn_fonts:
            if f not in cur:
                cur.append(f)
        matplotlib.rcParams["font.sans-serif"] = cur


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.02,
                    fontsize: float = 10.0, weight: str = "bold") -> None:
    """添加 NF 风格 panel 字母 (a/b/c …) 到坐标轴左上角。

    Args:
        ax: matplotlib Axes
        label: 'a' / 'b' / 'A' / 'B' …
        x, y: 相对坐标 (axes fraction)
        fontsize: 默认 10 pt（panel label 比 axis label 大一档）
        weight: 'bold'（NF 标准）
    """
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize, weight=weight,
            ha="left", va="bottom")


def style_axes(ax: plt.Axes, grid: bool = False, grid_axis: str = "y",
               minor_ticks: bool = False) -> None:
    """单 axes 一键 NF 化（top/right off + 可选 grid + 可选 minor ticks）。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis=grid_axis, alpha=0.25, linewidth=0.4, linestyle="--", zorder=0)
    if minor_ticks:
        ax.minorticks_on()


def color_palette(n: int, family: str = "wong") -> list[str]:
    """取 n 色调色板。

    family ∈ {'wong', 'viridis', 'cividis', 'project'}
    """
    if family == "wong":
        if n <= len(WONG_LIST):
            return WONG_LIST[:n]
        # 超过 8 类，回退到 tab20
        cmap = matplotlib.colormaps["tab20"]
        return [matplotlib.colors.to_hex(cmap(i / 20)) for i in range(n)]
    if family in ("viridis", "cividis"):
        cmap = matplotlib.colormaps[family]
        return [matplotlib.colors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
    if family == "project":
        return list(PROJECT_PALETTE.values())[:n]
    raise ValueError(f"Unknown palette family: {family}")


def save_nature(fig: plt.Figure, file_stem: str, index: int, lang: str,
                output_dir: str | Path = "04outputs",
                formats: Sequence[str] = ("pdf",), dpi: int = 600) -> dict[str, str]:
    """以 Nature 风格保存 figure（PDF 矢量为主，SVG 可选）。

    与 export_utils.figure_output_paths 互补：本函数 NF-style 直存，
    名称仍遵循项目约定第四章 Item 3 中英双版本规范。
    """
    import string
    letter = string.ascii_lowercase[index]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for fmt in formats:
        if lang == "zh":
            name = f"{file_stem}-图{letter}.{fmt}"
        else:
            name = f"{file_stem}-fig-{letter}.{fmt}"
        path = output_dir / name
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        saved[fmt] = str(path)
    return saved


def significance_marker(p: float) -> str:
    """NF / NeurIPS 标准: ns / * / ** / ***。"""
    if p is None or np.isnan(p):
        return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ─── 自动应用 ───────────────────────────────────────────────────────────────
# 任何脚本只要 `from figure_style import ...` 即触发；避免重复调用
apply_nature_style(also_chinese=True)
