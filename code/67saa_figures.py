"""67saa_figures.py: Spectrochimica Acta Part A 投稿稿的正文图 1–4 与补充材料图 S1–S3（中英双语）。

    图 1  研究设计：苹果采集域（年份 × 仪器 × 测量方式 × 每果光谱数 × 各产地果数）、留出域（同仪器两年设计与
          五个公开数据集）、各暴露臂的训练池与选检查点所用的池（单折示意）
    图 2  跨年域偏移与 2025 年内各产地差异（SNV 后均值 ± 1 SD）
    图 3  六个编码器族的 S_src−R：主口径 18 个留出域逐域值，与各族混合效应均值（S_src−R、S_half−R^h）
    图 4  方法选择：苹果两年设计逐 K 的各方法；各留出域逐 K 排第一的方法（暴露改变第一时标出）；按组件分的泄漏
    图 S1 源质心融合权重 α 的扫描：α × 留出折的宏 F1 热力图，α=0 与 α=0.4 的逐折对比
    图 S2 苹果一维卷积编码器各臂的逐折宏 F1 与六个对照差值（含只经检查点选择暴露的 R_ckpt−R）
    图 S3 同一批单面情节上两种编码器的原型网络与源年份选参的 SVM、PLS-DA；固定设置下迁移基线与集成的增量；
          逐折原型网络（编码器 R）减较强的调参基线

数据来源（本脚本不训练、不产生新实验数字，所有数值从结果文件读取）:
    图 1  冻结数据 03data/processed/all_9elements.csv（各域果数、光谱数、面位数）
    图 2  同上
    图 3、图 4  04outputs/75_leakage_generality_summary.xlsx（表 a、c、d）；正式工作簿缺失时退用 pilot 汇总，
          图内标注「预览，非投稿数字」
    图 S1 04outputs/formal_seeds/17_alpha_sensitivity__seed*.xlsx（5 个正式种子，K=5）
    图 S2 04outputs/formal_seeds/64_leakage__<design>__canonical__seed<S>.xlsx（5 个正式种子，CUDA）；
          正式结果不全时退用同名 __pilot 文件里的 5 个正式种子，图内标注「预览，非投稿数字」
    图 S3 (a)(c) 04outputs/72_kscan_arms.xlsx（64 号 Formal 编码器 R、S_pool 与 71 号源年份选参的 SVM、PLS-DA，
          同一批情节）；最近质心取 42 号九路线三折均值
          (b) 04outputs/42_ensemble_kshot.xlsx（表 S6 同源）、
              04outputs/65_domain_adaptation_baselines.xlsx（池化/CORAL/TCA）

运行方式（本机，纯 CPU）:
    cd <项目根>
    PYTHONPATH=<figure QA helpers> python 02code/67saa_figures.py

输出文件:
    04outputs/67saa_figures-fig-{a..g}.pdf / -图{a..g}.pdf           — 图 1–4 与图 S1–S3 英文 / 中文
    06doc/01manuscript/figs/Figure_{1..4}.pdf、图{1..4}.pdf          — 稿件用副本
    06doc/01manuscript/figs/Figure_S{1..3}.pdf、图S{1..3}.pdf        — 补充材料用副本
    04outputs/67saa_figures.xlsx                                     — 各图底层数据
    04outputs/67saa_figures-fig-*.alignment.json                     — 多面板对齐闸报告
"""

from __future__ import annotations

import argparse
import glob
import importlib
import itertools
import os
import shutil
import sys
from functools import partial
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import stats

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import export_utils as eu  # noqa: E402

NDArray = npt.NDArray[Any]
MM = 1 / 25.4
FIG_W = 174.5 * MM  # 稿件版心宽 494.5 pt（Elsevier.cls）：图按 1:1 排入，图内字号即印刷字号
FS_SUB = 7.4  # 含上下标的文字：mathtext 上下标约为父字号的 0.7，7.4 磅时为 5.18 磅；图宽比版心多出几磅时仍不低于 5 磅
FIG_DIR = os.path.join(eu.DOC_DIR, "01manuscript", "figs")
STEM = "67saa_figures"

# Wong 色盲安全色；产地为无序类别
ORIGIN_COLOR = {"XJ": "#0072B2", "SD": "#D55E00", "SX": "#009E73", "GS": "#CC79A7"}
ARM_STYLE: dict[str, dict[str, Any]] = {
    "R": {"color": "#7F7F7F", "marker": "o", "mfc": "white"},
    "S_size": {"color": "#56B4E9", "marker": "s", "mfc": "#56B4E9"},
    "S_shuf": {"color": "#E69F00", "marker": "v", "mfc": "#E69F00"},
    "S_src": {"color": "#0072B2", "marker": "D", "mfc": "#0072B2"},
    "S_pool": {"color": "#003F66", "marker": "^", "mfc": "white"},
    "S_half": {"color": "#009E73", "marker": "P", "mfc": "#009E73"},
    "R_ckpt": {"color": "#D55E00", "marker": "h", "mfc": "white"},
}
ARM_ORDER = ["R", "R_ckpt", "S_size", "S_shuf", "S_src", "S_pool"]
# 后两项只在目标年的留出半数 T_B 上评估：S_half 与同一批果实上的 R、S_src（R_B、S_src_B）相比
CONTRASTS = [
    ("S_src", "R"),
    ("S_src", "S_shuf"),
    ("R_ckpt", "R"),
    ("S_pool", "S_src"),
    ("S_half", "R_B"),
    ("S_src_B", "S_half"),
]
CONTRAST_COLOR = {
    ("S_src", "R"): "#0072B2",
    ("S_src", "S_shuf"): "#E69F00",
    ("R_ckpt", "R"): "#D55E00",
    ("S_pool", "S_src"): "#7F7F7F",
    ("S_half", "R_B"): "#009E73",
    ("S_src_B", "S_half"): "#CC79A7",
}

TXT: dict[str, dict[str, str]] = {
    "en": {
        "wl": "Wavelength (nm)",
        "snv": "SNV intensity",
        "year": "Acquisition year",
        "origin25": "Declared origin (2025)",
        "a_spec": "Cross-year domain shift",
        "b_spec": "Within-2025 origin differences",
        "XJ": "Xinjiang",
        "SD": "Shandong",
        "SX": "Shaanxi",
        "GS": "Gansu",
        "f1": "Macro-F1",
        "delta": "Contrast in macro-F1",
        "three": "Three-year design",
        "two": "Same-instrument two-year design",
        "chance": "chance",
        "K": "Reference fruits per origin, K",
        "ens_inc": "Ensemble minus strongest member",
        "pn_svm": "ProtoNet minus SVM",
        "mean3": "Three-fold mean",
        "preview": "PRELIMINARY: local MPS preview, not for submission",
        "a_des": "Acquisition domains",
        "b_des": "Evaluation designs",
        "c_des": "Encoder arms (one fold)",
        "inst_old": "Fibre-optic\nspectrometer\ntransmission\n1 spectrum per fruit",
        "inst_new": "Push-broom imager\nreflectance\n3–5 facets per fruit",
        "train": "Training pool",
        "ckpt": "Checkpoint on",
        "src": "S",
        "tgt": "T",
        "sub": "subsampled to |S|",
        "shuf": "T labels shuffled",
        "fruits": "fruits",
        "spectra": "spectra",
        "key": (
            "S source years; T target year;\n"
            "narrow blocks: pool subsampled to the size of S;\n"
            "T\u2032 (orange): origin labels permuted;\n"
            "A, B: random halves of T, B (open) evaluated only"
        ),
        "t_arms": "Encoder arms",
        "t_alpha_a": "Macro-F1 over \u03b1 and fold (\u2605 best \u03b1 per fold)",
        "t_alpha_b": "Cost of fusion: \u03b1 = 0.4 against \u03b1 = 0",
        "alpha_x": "Source-centroid fusion weight, \u03b1",
        "alpha_f1": "Macro-F1 (K = 5)",
        "alpha0": "\u03b1 = 0 (no fusion)",
        "alpha04": "\u03b1 = 0.4 (ProtoNet-C)",
        "t_con": "Leakage contrasts",
        "t_routes": "Routes on the same episodes",
        "t_ens": "Transfer and ensembles, fixed settings",
        "t_fold": "Network minus best tuned baseline",
        "diff": "Difference in macro-F1",
        "b_des2": "Held-out domains",
        "c_des2": "Exposure arms (one fold)",
        "fam_line": "Six encoder families: CNN, MLP, ResNet, Transformer, PLS, PCA",
        "cnn_only": "CNN, apple",
        "pub_head": "Public data sets (split, domains)",
        "pub_rows": (
            "Maize (instrument, 3)\nTablets (instrument, 2)\nMango (season, 4)\n"
            "Kiwifruit (season, one instrument, 3)\nIndian Pines (quadrant, 4)"
        ),
        "three_note": "Three-year apple design: Supplementary Material",
        "t_fam_a": "Exposure contrast by held-out domain",
        "t_fam_b": "Mean over 18 held-out domains",
        "rep": "S$_\\mathrm{src}$ − R",
        "half": "S$_\\mathrm{half}$ − R$^\\mathrm{h}$",
        "t_ch_a": "Apple, same instrument",
        "t_ch_b": "First-ranked method",
        "t_ch_c": "Leakage by component",
        "free": "leakage-free",
        "exposed": "exposed",
        "enc_leak": "Encoder",
        "svm_leak": "SVM hyperparameters",
        "pls_leak": "PLS-DA hyperparameters",
        "K2": "Reference samples per class, K",
        "flip_key": "Cell colour: leakage-free first; letters: exposed first where it differs",
    },
    "zh": {
        "wl": "波长 (nm)",
        "snv": "SNV 强度",
        "year": "采集年份",
        "origin25": "声明产地（2025）",
        "a_spec": "跨年域偏移",
        "b_spec": "2025 年内各产地差异",
        "XJ": "新疆",
        "SD": "山东",
        "SX": "陕西",
        "GS": "甘肃",
        "f1": "宏 F1",
        "delta": "宏 F1 差值",
        "three": "三年设计",
        "two": "同仪器两年设计",
        "chance": "随机水平",
        "K": "每产地参考果数 K",
        "ens_inc": "集成减最强成员",
        "pn_svm": "原型网络减 SVM",
        "mean3": "三折均值",
        "preview": "预览：本机 MPS 结果，非投稿数字",
        "a_des": "采集域",
        "b_des": "评估设计",
        "c_des": "编码器臂（单折示意）",
        "inst_old": "光纤光谱仪\n透射\n每果 1 条光谱",
        "inst_new": "推扫式成像仪\n反射\n每果 3–5 个面",
        "train": "训练池",
        "ckpt": "选检查点",
        "src": "S",
        "tgt": "T",
        "sub": "抽样至 |S|",
        "shuf": "T 标签置乱",
        "fruits": "果",
        "spectra": "条光谱",
        "key": "S 源年份；T 目标年；窄块：训练池抽样至与 S 等大；\nT\u2032（橙色）：目标年产地标签置乱；\n"
        "A、B：目标年随机分成的两半，B（空心）只用于评估",
        "t_arms": "编码器各臂",
        "t_alpha_a": "\u03b1 × 留出折的宏 F1（\u2605 为每折最优 \u03b1）",
        "t_alpha_b": "融合的代价：\u03b1 = 0.4 相对 \u03b1 = 0",
        "alpha_x": "源质心融合权重 \u03b1",
        "alpha_f1": "宏 F1（K = 5）",
        "alpha0": "\u03b1 = 0（不融合）",
        "alpha04": "\u03b1 = 0.4（ProtoNet-C）",
        "t_con": "泄漏对照差值",
        "t_routes": "同一批情节上的各路线",
        "t_ens": "固定设置下的迁移与集成",
        "t_fold": "原型网络减最佳调参基线",
        "diff": "宏 F1 差值",
        "b_des2": "留出域",
        "c_des2": "暴露臂（单折示意）",
        "fam_line": "六个编码器族：CNN、MLP、ResNet、Transformer、PLS、PCA",
        "cnn_only": "仅 CNN，苹果",
        "pub_head": "公开数据集（划分，留出域数）",
        "pub_rows": (
            "玉米（仪器，3）\n药片（仪器，2）\n芒果（采收季，4）\n"
            "猕猴桃（采收季，同一仪器，3）\nIndian Pines（象限，4）"
        ),
        "three_note": "苹果三年设计：见补充材料",
        "t_fam_a": "逐留出域的暴露对照",
        "t_fam_b": "18 个留出域上的均值",
        "rep": "S$_\\mathrm{src}$ − R",
        "half": "S$_\\mathrm{half}$ − R$^\\mathrm{h}$",
        "t_ch_a": "苹果，同一仪器",
        "t_ch_b": "排第一的方法",
        "t_ch_c": "各组件的泄漏",
        "free": "无泄漏",
        "exposed": "暴露",
        "enc_leak": "编码器",
        "svm_leak": "SVM 超参数",
        "pls_leak": "PLS-DA 超参数",
        "K2": "每类参考样本数 K",
        "flip_key": "底色：无泄漏第一；字母：与之不同时的暴露第一",
    },
}
ROUTE_LABEL = {
    "en": {
        "VanillaProtoNet": "Prototypical network",
        "SVM-Kshot": "SVM",
        "PLS-DA-Kshot": "PLS-DA",
        "NearestCentroid": "Nearest centroid",
        "Pooled-SVM": "Pooled-SVM",
        "CORAL-SVM": "CORAL-SVM",
        "TCA-SVM": "TCA-SVM",
        "Ens-WAvg": "Weighted averaging",
        "Ens-Vote": "Majority voting",
        "Ens-Stack": "Stacking",
        "R": "ProtoNet, encoder R",
        "S_pool": "ProtoNet, pooled encoder",
        "SVM(源年份选参)": "SVM, source-tuned",
        "PLS-DA(源年份选参)": "PLS-DA, source-tuned",
        "Pooled-SVM_d": "Pooled-SVM",
        "CORAL-SVM_d": "CORAL-SVM",
        "TCA-SVM_d": "TCA-SVM",
        "Ens-WAvg_d": "Weighted avg.",
        "Ens-Vote_d": "Voting",
        "Ens-Stack_d": "Stacking",
        "minus_svm": "Minus fixed SVM",
        "minus_best": "Minus best member",
    },
    "zh": {
        "VanillaProtoNet": "原型网络",
        "SVM-Kshot": "SVM",
        "PLS-DA-Kshot": "PLS-DA",
        "NearestCentroid": "最近质心",
        "Pooled-SVM": "池化 SVM",
        "CORAL-SVM": "CORAL-SVM",
        "TCA-SVM": "TCA-SVM",
        "Ens-WAvg": "加权平均",
        "Ens-Vote": "多数投票",
        "Ens-Stack": "堆叠",
        "R": "原型网络（编码器 R）",
        "S_pool": "原型网络（合并池编码器）",
        "SVM(源年份选参)": "SVM（源年份选参）",
        "PLS-DA(源年份选参)": "PLS-DA（源年份选参）",
        "Pooled-SVM_d": "池化 SVM",
        "CORAL-SVM_d": "CORAL-SVM",
        "TCA-SVM_d": "TCA-SVM",
        "minus_svm": "减固定 SVM",
        "minus_best": "减最强成员",
        "Ens-WAvg_d": "加权平均",
        "Ens-Vote_d": "多数投票",
        "Ens-Stack_d": "堆叠",
    },
}
ROUTE_LABEL_CHOICE = {
    "en": {
        "原型网络(R)": "ProtoNet, encoder R",
        "原型网络(S_src)": "ProtoNet, exposed encoder",  # 图例字号 5.6 磅，不用下标（下标会低于 5 磅）
        "SVM(源域选参)": "SVM, source-selected",
        "SVM(留出域选参)": "SVM, held-out-selected",
        "PLS-DA(源域选参)": "PLS-DA, source-selected",
        "PLS-DA(留出域选参)": "PLS-DA, held-out-selected",
        "最近质心": "Nearest centroid",
    },
    "zh": {
        "原型网络(R)": "原型网络（编码器 R）",
        "原型网络(S_src)": "原型网络（目标暴露编码器）",
        "SVM(源域选参)": "SVM（源域选参）",
        "SVM(留出域选参)": "SVM（留出域选参）",
        "PLS-DA(源域选参)": "PLS-DA（源域选参）",
        "PLS-DA(留出域选参)": "PLS-DA（留出域选参）",
        "最近质心": "最近质心",
    },
}
ARM_LABEL = {
    "R": "R",
    "S_size": "S$_\\mathrm{size}$",
    "S_shuf": "S$_\\mathrm{shuf}$",
    "S_src": "S$_\\mathrm{src}$",
    "S_pool": "S$_\\mathrm{pool}$",
    "S_half": "S$_\\mathrm{half}$",
    "R_B": "R$^\\mathrm{h}$",
    "S_src_B": "S$_\\mathrm{src}^\\mathrm{h}$",
    "R_ckpt": "R$_\\mathrm{ckpt}$",
}


# 编码器族与数据集（图 3、图 4；与 75、77 号同名）
FAM_ORDER = ["cnn", "mlp", "resnet", "transformer", "pls", "pca"]
FAM_STYLE: dict[str, dict[str, Any]] = {
    "cnn": {"color": "#0072B2", "marker": "D", "mfc": "#0072B2"},
    "mlp": {"color": "#E69F00", "marker": "o", "mfc": "#E69F00"},
    "resnet": {"color": "#009E73", "marker": "s", "mfc": "#009E73"},
    "transformer": {"color": "#CC79A7", "marker": "^", "mfc": "#CC79A7"},
    "pls": {"color": "#56B4E9", "marker": "v", "mfc": "white"},
    "pca": {"color": "#7F7F7F", "marker": "o", "mfc": "white"},
}
FAM_LABEL = {"cnn": "CNN", "mlp": "MLP", "resnet": "ResNet", "transformer": "Transformer", "pls": "PLS", "pca": "PCA"}
GEN_SETS = ["apple_two_year", "corn_instr", "tablet_instr", "mango_season", "kiwifruit_season", "indianpines_spatial"]
GEN_SET_LABEL = {
    "en": {
        "apple_two_year": "Apple",
        "corn_instr": "Maize",
        "tablet_instr": "Tablets",
        "mango_season": "Mango",
        "kiwifruit_season": "Kiwifruit",
        "indianpines_spatial": "Indian Pines",
    },
    "zh": {
        "apple_two_year": "苹果",
        "corn_instr": "玉米",
        "tablet_instr": "药片",
        "mango_season": "芒果",
        "kiwifruit_season": "猕猴桃",
        "indianpines_spatial": "Indian Pines",
    },
}
# 图 4 的路线：无泄漏实心、暴露空心虚线；同一方法同色
CHOICE_ROUTES = [
    ("原型网络(R)", "#0072B2", "-", "o", True),
    ("原型网络(S_src)", "#0072B2", "--", "o", False),
    ("SVM(源域选参)", "#D55E00", "-", "s", True),
    ("SVM(留出域选参)", "#D55E00", "--", "s", False),
    ("PLS-DA(源域选参)", "#009E73", "-", "^", True),
    ("PLS-DA(留出域选参)", "#009E73", "--", "^", False),
    ("最近质心", "#7F7F7F", "-", "v", True),
]
CHOICE_FAMILY_COLOR = {"PN": "#9ECAE1", "SVM": "#FDD0A2", "PLS": "#A1D99B", "NC": "#D9D9D9"}
CHOICE_ABBR = {
    "原型网络(R)": "PN",
    "原型网络(S_src)": "PN",
    "SVM(源域选参)": "SVM",
    "SVM(留出域选参)": "SVM",
    "PLS-DA(源域选参)": "PLS",
    "PLS-DA(留出域选参)": "PLS",
    "最近质心": "NC",
}
CHOICE_K = [1, 5, 10, 20]


def _held_short(ds: str, held: str) -> str:
    if ds.startswith("apple") or ds == "kiwifruit_season":
        return held[1:]
    if ds == "corn_instr":
        return held.replace("spec", "")
    if ds == "tablet_instr":
        return f"I{held[-1]}"
    return held


def style() -> None:
    eu.setup_chinese_font()
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_title(ax: Any, letter: str, text: str) -> None:
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=7, fontweight="bold", pad=4)


def alignment_gate(fig: Figure, name: str, exemptions: list[dict[str, Any]] | None = None, **groups: Any) -> None:
    """出图时的面板对齐闸；groups 透传 axes / panel_ids / row_groups（色条等非面板轴须显式声明）。"""
    try:
        apa = importlib.import_module("audit_panel_alignment")
    except ImportError as exc:  # 运行方式里写明了 PYTHONPATH；缺它就拒绝出图
        raise SystemExit("缺 audit_panel_alignment：按脚本头的运行方式设 PYTHONPATH") from exc
    apa.require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(eu.OUTPUT_DIR, f"{STEM}-{name}.alignment.json"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
        exemptions=exemptions or [],
        **groups,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 数据
# ═══════════════════════════════════════════════════════════════════════════════
def domain_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["年份", "产地"]).agg(光谱数=("苹果编号", "size"), 果数=("苹果编号", "nunique")).reset_index()
    return g


def spectra_stats(df: pd.DataFrame, wl_cols: list[str]) -> tuple[NDArray, dict[str, Any]]:
    wl = np.array([float(c.replace("nm", "")) for c in wl_cols])
    out: dict[str, Any] = {}
    for y in (2018, 2019, 2025):
        x = df.loc[df["年份"] == y, wl_cols].to_numpy(dtype=float)
        out[f"year_{y}"] = (x.mean(0), x.std(0), len(x))
    for r in ("XJ", "SD", "GS"):
        x = df.loc[(df["年份"] == 2025) & (df["产地"] == r), wl_cols].to_numpy(dtype=float)
        out[f"o25_{r}"] = (x.mean(0), x.std(0), len(x))
    return wl, out


def load_leakage() -> tuple[pd.DataFrame, str]:
    base = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
    seeds = set(eu.RECOMMENDED_SEEDS)
    frames: list[pd.DataFrame] = []
    tag = "formal"
    for design in ("three_year", "two_year"):
        formal = [
            f
            for f in glob.glob(os.path.join(base, f"64_leakage__{design}__canonical__seed*.xlsx"))
            if "__pilot" not in f
        ]
        got = {int(os.path.basename(f).split("seed")[1].split(".")[0]) for f in formal}
        if got >= seeds:
            files = formal
        else:
            tag = "preview"
            files = [
                f
                for f in glob.glob(os.path.join(base, f"64_leakage__{design}__canonical__seed*__pilot.xlsx"))
                if int(os.path.basename(f).split("seed")[1].split("__")[0]) in seeds
            ]
        if not files:  # 正式种子的预览也还没有：退用 pilot 种子，同样标「预览」
            tag = "preview"
            files = glob.glob(os.path.join(base, f"64_leakage__{design}__canonical__seed*__pilot.xlsx"))
        for f in files:
            frames.append(pd.read_excel(f, sheet_name=0))
    if not frames:
        raise FileNotFoundError("没有 64 号脚本的逐种子结果")
    res = pd.concat(frames, ignore_index=True)
    res = res[(res["变体"] == "canonical") & (res["src_alpha"] == 0.0)]
    return res, tag


def load_methods() -> dict[str, pd.DataFrame]:
    x42 = pd.ExcelFile(os.path.join(eu.OUTPUT_DIR, "42_ensemble_kshot.xlsx"))
    mean42 = x42.parse(next(s for s in x42.sheet_names if "三折均值" in s)).set_index("K")
    sd42 = x42.parse(next(s for s in x42.sheet_names if "跨种子" in s)).set_index("K")
    mean42 = mean42.rename(columns={c: "VanillaProtoNet" for c in mean42.columns if "ProtoNet" in c})
    x65 = pd.ExcelFile(os.path.join(eu.OUTPUT_DIR, "65_domain_adaptation_baselines.xlsx"))
    mean65 = x65.parse(next(s for s in x65.sheet_names if "三折均值" in s)).set_index("K")
    sd65 = x65.parse(next(s for s in x65.sheet_names if "跨种子" in s)).set_index("K")
    means = pd.concat([mean42, mean65[["Pooled-SVM", "CORAL-SVM", "TCA-SVM"]]], axis=1)
    sds = pd.concat([sd42, sd65], axis=1)
    # 72 号聚合：协议对齐的编码器 R、S_pool 与只在源年份选参的 SVM、PLS-DA，同一批单面情节
    x72 = pd.ExcelFile(os.path.join(eu.OUTPUT_DIR, "72_kscan_arms.xlsx"))
    k72 = x72.parse(next(s for s in x72.sheet_names if "三折均值" in s)).set_index("K")
    f72 = x72.parse(next(s for s in x72.sheet_names if "逐折均值" in s))
    return {"mean": means, "sd": sds, "k72": k72, "f72": f72}


# ═══════════════════════════════════════════════════════════════════════════════
# 图 1：研究设计
# ═══════════════════════════════════════════════════════════════════════════════
def _box(ax: Any, x: float, y: float, w: float, h: float, fc: str, ec: str = "none", **kw: Any) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=0.6, **kw))


def fig_design(lang: str, dom: pd.DataFrame) -> Figure:
    t = TXT[lang]
    fig = plt.figure(figsize=(FIG_W, 64 * MM))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.5, 0.95, 1.1], left=0.01, right=0.99, top=0.88, bottom=0.03, wspace=0.06
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # (a) 采集域
    ax = axes[0]
    panel_title(ax, "a", t["a_des"])
    rows = {2018: 0.70, 2019: 0.45, 2025: 0.08}
    for y, y0 in rows.items():
        h = 0.22 if y < 2025 else 0.26
        ax.text(0.0, y0 + h / 2, str(y), ha="left", va="center", fontsize=7, fontweight="bold")
        inst = t["inst_old"] if y < 2025 else t["inst_new"]
        _box(ax, 0.115, y0, 0.29, h, fc="#F2F2F2")
        ax.text(0.26, y0 + h / 2, inst, ha="center", va="center", fontsize=5.8)
        sub = dom[dom["年份"] == y].set_index("产地")
        order = [o for o in ("XJ", "SD", "SX", "GS") if o in sub.index]
        for i, o in enumerate(order):
            x0 = 0.42 + i * 0.194
            _box(ax, x0, y0, 0.184, h, fc=ORIGIN_COLOR[o], alpha=0.18, ec=ORIGIN_COLOR[o])
            n_f, n_s = int(sub.loc[o, "果数"]), int(sub.loc[o, "光谱数"])
            lab = f"{t[o]}\n{n_f} {t['fruits']}"
            if y == 2025:
                lab += f"\n{n_s} {t['spectra']}"
            ax.text(x0 + 0.092, y0 + h / 2, lab, ha="center", va="center", fontsize=5.8)
    ax.plot([0.115, 1.0], [0.39, 0.39], color="#BBBBBB", lw=0.5, ls=(0, (2, 2)))

    # (b) 留出域：苹果同仪器两年设计与五个公开数据集；三年设计见补充材料
    ax = axes[1]
    panel_title(ax, "b", t["b_des2"])

    def grid(y_top: float, cols: list[int], title: str) -> None:
        cw, ch = 0.22, 0.105
        x0 = 0.36
        ax.text(0.0, y_top + 0.075, title, ha="left", va="bottom", fontsize=6.3)
        for j, yr in enumerate(cols):
            ax.text(x0 + j * cw + cw / 2, y_top, str(yr), ha="center", va="bottom", fontsize=5.8, color="#555555")
        for i, fy in enumerate(cols):
            yy = y_top - (i + 1) * ch - 0.01
            ax.text(x0 - 0.02, yy + ch / 2, f"LOEO-{fy}", ha="right", va="center", fontsize=5.8)
            for j, yr in enumerate(cols):
                fc, lab, tc = ("#0072B2", t["tgt"], "white") if yr == fy else ("#D9D9D9", t["src"], "black")
                ax.add_patch(
                    Rectangle((x0 + j * cw + 0.006, yy + 0.006), cw - 0.012, ch - 0.012, fc=fc, ec="#999999", lw=0.4)
                )
                ax.text(x0 + j * cw + cw / 2, yy + ch / 2, lab, ha="center", va="center", fontsize=6, color=tc)

    grid(0.74, [2018, 2019], t["two"])
    ax.text(0.0, 0.44, t["pub_head"], ha="left", va="bottom", fontsize=6.3)
    ax.text(0.0, 0.41, t["pub_rows"], ha="left", va="top", fontsize=5.8, linespacing=1.35)
    ax.text(0.0, 0.0, t["three_note"], ha="left", va="bottom", fontsize=5.6, color="#555555")

    # (c) 编码器臂：训练池与选检查点所用的池；S_size 两块按比例缩到与 R 等宽，S_shuf 的 T 块用橙色（与图 3 同色）
    ax = axes[2]
    panel_title(ax, "c", t["c_des2"])
    ax.text(0.355, 0.845, t["train"], ha="center", va="bottom", fontsize=6.3)
    ax.text(0.83, 0.845, t["ckpt"], ha="center", va="bottom", fontsize=6.3)
    rh, pad = 0.085, 0.012  # 行高、块的上下留白
    x0, ws, wt = 0.165, 0.25, 0.25  # 块的左端（留出臂名的宽度）、源域块与目标年块宽度
    for i, arm in enumerate(["R", "R_ckpt", "S_shuf", "S_half", "S_src", "S_size", "S_pool"]):
        yy = 0.83 - (i + 1) * rh
        mark = "\u2020" if arm in ("S_size", "S_pool", "R_ckpt") else ""
        ax.text(0.0, yy + rh / 2, ARM_LABEL[arm] + mark, ha="left", va="center", fontsize=FS_SUB)
        if arm == "S_size":
            s_w, t_w = ws * ws / (ws + wt), ws * wt / (ws + wt)
        else:
            s_w, t_w = ws, wt
        ax.add_patch(Rectangle((x0, yy + pad), s_w, rh - 2 * pad, fc="#D9D9D9", ec="#999999", lw=0.4))
        ax.text(x0 + s_w / 2, yy + rh / 2, t["src"], ha="center", va="center", fontsize=6)
        if arm == "S_half":
            x_t = x0 + s_w + 0.01
            half = (wt - 0.01) / 2
            ax.add_patch(Rectangle((x_t, yy + pad), half, rh - 2 * pad, fc="#0072B2", ec="#0072B2", lw=0.4))
            ax.text(
                x_t + half / 2, yy + rh / 2, "T$_\\mathrm{A}$", ha="center", va="center", fontsize=FS_SUB, color="white"
            )
            xb = x_t + half + 0.01
            ax.add_patch(Rectangle((xb, yy + pad), half, rh - 2 * pad, fc="white", ec="#0072B2", lw=0.6, ls="--"))
            ax.text(
                xb + half / 2,
                yy + rh / 2,
                "T$_\\mathrm{B}$",
                ha="center",
                va="center",
                fontsize=FS_SUB,
                color="#0072B2",
            )
        elif arm not in ("R", "R_ckpt"):
            x_t = x0 + s_w + 0.01
            shuf = arm == "S_shuf"
            fc = ARM_STYLE["S_shuf"]["color"] if shuf else "#0072B2"
            ax.add_patch(Rectangle((x_t, yy + pad), t_w, rh - 2 * pad, fc=fc, ec=fc, lw=0.4))
            ax.text(
                x_t + t_w / 2,
                yy + rh / 2,
                t["tgt"] + ("\u2032" if shuf else ""),
                ha="center",
                va="center",
                fontsize=6,
                color="black" if shuf else "white",
            )
        if arm != "R_ckpt":  # R_ckpt 只在目标年情节上选检查点
            ax.add_patch(Rectangle((0.74, yy + pad), 0.085, rh - 2 * pad, fc="#D9D9D9", ec="#999999", lw=0.4))
            ax.text(0.7825, yy + rh / 2, t["src"], ha="center", va="center", fontsize=6)
        if arm in ("S_pool", "R_ckpt"):
            ax.add_patch(Rectangle((0.835, yy + pad), 0.085, rh - 2 * pad, fc="#0072B2", ec="#0072B2", lw=0.4))
            ax.text(0.8775, yy + rh / 2, t["tgt"], ha="center", va="center", fontsize=6, color="white")
    ax.text(
        0.0,
        0.0,
        t["key"] + "\n\u2020 " + t["cnn_only"] + "\n" + t["fam_line"],
        ha="left",
        va="bottom",
        fontsize=5.2,
        color="#444444",
    )
    alignment_gate(
        fig,
        "fig-a" if lang == "en" else "图a",
        [{"panels": ["a", "b", "c"], "checks": ["panel-width"], "reason": "schematic panels sized to content"}],
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 图 2：光谱
# ═══════════════════════════════════════════════════════════════════════════════
def fig_spectra(lang: str, wl: NDArray, st: dict[str, Any]) -> Figure:
    t = TXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 62 * MM), sharex=True)
    year_style = {2018: ("#9ECAE1", "-"), 2019: ("#4292C6", "--"), 2025: ("#08306B", "-.")}
    ax = axes[0]
    for y, (c, ls) in year_style.items():
        m, s, n = st[f"year_{y}"]
        ax.fill_between(wl, m - s, m + s, color=c, alpha=0.18, lw=0)
        ax.plot(wl, m, color=c, ls=ls, lw=1.1, label=f"{y} (n = {n})")
    ax.legend(title=t["year"], loc="upper right", title_fontsize=6.5, handlelength=2.6)
    panel_title(ax, "a", t["a_spec"])
    ax = axes[1]
    for r, ls in (("XJ", "-"), ("SD", "--"), ("GS", ":")):
        m, s, n = st[f"o25_{r}"]
        ax.fill_between(wl, m - s, m + s, color=ORIGIN_COLOR[r], alpha=0.12, lw=0)
        ax.plot(wl, m, color=ORIGIN_COLOR[r], ls=ls, lw=1.1, label=f"{t[r]} (n = {n})")
    ax.legend(title=t["origin25"], loc="lower center", title_fontsize=6.5, handlelength=2.6)
    panel_title(ax, "b", t["b_spec"])
    for ax in axes:
        ax.set_xlabel(t["wl"])
        ax.set_ylabel(t["snv"])
        ax.set_xlim(585, 1006)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.17, wspace=0.22)
    alignment_gate(fig, "fig-b" if lang == "en" else "图b")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 图 3：协议对齐泄漏
# ═══════════════════════════════════════════════════════════════════════════════
def leakage_frames(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [("three_year", 2018), ("three_year", 2019), ("three_year", 2025), ("two_year", 2018), ("two_year", 2019)]
    arm_rows: list[dict[str, Any]] = []
    con_rows: list[dict[str, Any]] = []
    for gi, (design, fold) in enumerate(groups):
        sub = res[(res["设计"] == design) & (res["目标年份"] == fold)]
        piv = sub.pivot_table(index="种子", columns="臂", values="F1均值")
        for arm in ARM_ORDER:
            if arm in piv:
                vals = piv[arm].dropna()
                arm_rows.append(
                    {"组": gi, "设计": design, "折": fold, "臂": arm, "均值": float(vals.mean()), "种子值": list(vals)}
                )
        for a, b in CONTRASTS:
            if a in piv and b in piv:
                d = (piv[a] - piv[b]).dropna().to_numpy()
                if len(d) > 1:
                    half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
                else:
                    half = float("nan")
                con_rows.append(
                    {
                        "组": gi,
                        "设计": design,
                        "折": fold,
                        "对照": f"{a}-{b}",
                        "均值": float(d.mean()),
                        "半宽": half,
                        "种子数": len(d),
                        "为正种子数": int((d > 0).sum()),
                    }
                )
    return pd.DataFrame(arm_rows), pd.DataFrame(con_rows)


def fig_leakage(lang: str, arms: pd.DataFrame, cons: pd.DataFrame, tag: str, gate: str = "f") -> Figure:
    t = TXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 68 * MM), gridspec_kw={"width_ratios": [1.35, 1.0]})
    xpos = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.4, 4: 4.4}
    ticklab = ["LOEO-2018", "LOEO-2019", "LOEO-2025", "LOEO-2018", "LOEO-2019"]
    ax = axes[0]
    offs = np.linspace(-0.3, 0.3, len(ARM_ORDER))
    rng = np.random.default_rng(0)
    for k, arm in enumerate(ARM_ORDER):
        sub = arms[arms["臂"] == arm]
        if sub.empty:
            continue
        stl = ARM_STYLE[arm]
        xs = [xpos[g] + offs[k] for g in sub["组"]]
        for x, sv in zip(xs, sub["种子值"], strict=True):
            ax.scatter(x + rng.uniform(-0.035, 0.035, len(sv)), sv, s=3, color=stl["color"], alpha=0.45, lw=0, zorder=2)
        ax.plot(
            xs,
            sub["均值"],
            ls="none",
            marker=stl["marker"],
            ms=4.2,
            mfc=stl["mfc"],
            mec=stl["color"],
            mew=0.8,
            label=ARM_LABEL[arm],
            zorder=3,
        )
    ax.axhline(1 / 3, color="#999999", lw=0.6, ls=":")
    ax.text(4.9, 1 / 3 + 0.01, t["chance"], ha="right", va="bottom", fontsize=5.8, color="#777777")
    ax.set_ylim(0.28, 1.08)
    ax.set_ylabel(t["f1"])
    # 图例只占左边三年设计那一段，不跨过 x=2.7 的分隔线
    ax.legend(
        loc="lower left", ncol=3, handletextpad=0.2, columnspacing=0.8, bbox_to_anchor=(0.0, 0.1), fontsize=FS_SUB
    )
    panel_title(ax, "a", t["t_arms"])
    ax = axes[1]
    offc = dict(zip(CONTRASTS, np.linspace(-0.3, 0.3, len(CONTRASTS)), strict=True))
    for (a, b), off in offc.items():
        sub = cons[cons["对照"] == f"{a}-{b}"]
        if sub.empty:
            continue
        xs = [xpos[g] + off for g in sub["组"]]
        lab = f"{ARM_LABEL[a]} − {ARM_LABEL[b]}"
        ax.errorbar(
            xs,
            sub["均值"],
            yerr=sub["半宽"],
            fmt="o",
            ms=3.6,
            color=CONTRAST_COLOR[(a, b)],
            elinewidth=0.8,
            capsize=1.6,
            label=lab,
        )
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_ylabel(t["diff"])
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -0.05), hi + 1.45 * (hi - lo))
    # 图例放左上，单栏，不跨过 x=2.7 的分隔线；上方留白足够六个对照的图例不压数据
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, 0.9), ncol=1, handletextpad=0.2, labelspacing=0.35, fontsize=FS_SUB
    )
    panel_title(ax, "b", t["t_con"])
    for ax in axes:
        ax.set_xticks([xpos[g] for g in range(5)])
        ax.set_xticklabels(ticklab, rotation=30, ha="right")
        ax.set_xlim(-0.55, 4.95)
        ax.axvline(2.7, color="#CCCCCC", lw=0.5)
        ytop = ax.get_ylim()[1]
        ax.text(1.0, ytop, t["three"], ha="center", va="top", fontsize=6, color="#444444")
        ax.text(
            3.9,
            ytop,
            t["two"].replace(" two-year", "\ntwo-year") if lang == "en" else t["two"],
            ha="center",
            va="top",
            fontsize=6,
            color="#444444",
        )
    if tag != "formal":
        fig.text(
            0.5, 0.5, t["preview"], ha="center", va="center", fontsize=11, color="#D55E00", alpha=0.35, rotation=12
        )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.25, wspace=0.22)
    alignment_gate(
        fig,
        f"fig-{gate}" if lang == "en" else f"图{gate}",
        [{"panels": ["a", "b"], "checks": ["panel-width"], "reason": "hero panel a wider than contrast panel b"}],
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 图 4：无泄漏方法比较
# ═══════════════════════════════════════════════════════════════════════════════
TUNED = ("SVM(源年份选参)", "PLS-DA(源年份选参)")


def crossover(k72: pd.DataFrame) -> tuple[float, float] | None:
    """三折均值上原型网络（编码器 R）最后领先与首次落后于最佳调参基线的两个 K；没有交叉返回 None。"""
    gap = k72["R_均值"] - k72[[f"{m}_均值" for m in TUNED]].max(axis=1)
    ks = gap.index.to_list()
    for a, b in itertools.pairwise(ks):
        if gap[a] > 0 >= gap[b]:
            return float(a), float(b)
    return None


def fig_methods(lang: str, md: dict[str, pd.DataFrame], gate: str = "g") -> Figure:
    t = TXT[lang]
    lab = ROUTE_LABEL[lang]
    means, k72, f72 = md["mean"], md["k72"], md["f72"]
    ks = k72.index.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 64 * MM))
    # (a) 同一批情节：两种编码器的原型网络、只在源年份选参的 SVM 与 PLS-DA；最近质心没有超参数
    ax = axes[0]
    routes = [
        ("S_pool", "#0072B2", "--", "o", "white"),
        ("R", "#0072B2", "-", "o", "#0072B2"),
        (TUNED[0], "#D55E00", "-", "s", "#D55E00"),
        (TUNED[1], "#009E73", "-", "^", "#009E73"),
    ]
    for r, c, ls, mk, mfc in routes:
        ax.errorbar(
            ks,
            k72[f"{r}_均值"],
            yerr=k72[f"{r}_种子SD"],
            color=c,
            ls=ls,
            lw=1.3,
            marker=mk,
            ms=3.0,
            mfc=mfc,
            elinewidth=0.6,
            capsize=1.2,
            label=lab[r],
        )
    nc = means.loc[means.index.isin(ks), "NearestCentroid"]
    ax.plot(nc.index, nc, color="#A6A6A6", lw=0.9, marker="v", ms=2.8, mfc="white", label=lab["NearestCentroid"])
    ax.set_ylabel(t["f1"])
    ax.set_ylim(0.30, ax.get_ylim()[1])  # 曲线最低约 0.46，下方留给图例，图例不压数据
    ax.legend(loc="lower right", handlelength=2.2, frameon=False)
    panel_title(ax, "a", t["t_routes"])
    # (b) 固定设置下的增量：迁移基线减同设置 SVM、集成减其最强成员
    ax = axes[1]
    kb = means.index.to_numpy()
    members = means[["SVM-Kshot", "PLS-DA-Kshot", "RandomForest"]].max(axis=1)
    for r, c, ls, mk in (
        ("Pooled-SVM", "#009E73", ":", "x"),
        ("CORAL-SVM", "#009E73", "--", "D"),
        ("TCA-SVM", "#009E73", "-.", "P"),
    ):
        ax.plot(
            kb,
            means[r] - means["SVM-Kshot"],
            color=c,
            ls=ls,
            marker=mk,
            ms=2.8,
            lw=1.0,
            mfc="white",
            label=lab[r + "_d"],
        )
    for r, c, ls, mk in (
        ("Ens-WAvg", "#CC79A7", "-", "o"),
        ("Ens-Vote", "#CC79A7", "--", "s"),
        ("Ens-Stack", "#CC79A7", "-.", "^"),
    ):
        ax.plot(kb, means[r] - members, color=c, ls=ls, marker=mk, ms=2.8, lw=1.0, label=lab[r + "_d"])
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_ylabel(t["diff"])
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.55 * (hi - lo), hi)
    hs, ls_ = ax.get_legend_handles_labels()
    kw = {"handlelength": 1.5, "handletextpad": 0.4, "fontsize": 6.0, "title_fontsize": 6.0, "frameon": False}
    leg = ax.legend(hs[:3], ls_[:3], title=lab["minus_svm"], loc="lower left", **kw)
    ax.add_artist(leg)
    ax.legend(hs[3:], ls_[3:], title=lab["minus_best"], loc="lower right", **kw)
    panel_title(ax, "b", t["t_ens"])
    # (c) 逐折：原型网络（编码器 R）减该折该 K 下两种调参基线中较强的一个
    ax = axes[2]
    fold_style = {2018: ("#BDBDBD", ":", "s"), 2019: ("#8C8C8C", "--", "^"), 2025: ("#404040", "-.", "v")}
    fw = f72.set_index(["臂", "K"])
    for fy, (c, ls, mk) in fold_style.items():
        best = np.maximum(fw.loc[TUNED[0], fy].to_numpy(), fw.loc[TUNED[1], fy].to_numpy())
        ax.plot(ks, fw.loc["R", fy].to_numpy() - best, color=c, ls=ls, lw=1.0, marker=mk, ms=2.8, label=f"LOEO-{fy}")
    mean_gap = k72["R_均值"] - k72[[f"{m}_均值" for m in TUNED]].max(axis=1)
    ax.plot(ks, mean_gap, color="#0072B2", lw=1.6, marker="o", ms=3.0, label=t["mean3"])
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_ylabel(t["diff"])
    ax.legend(loc="lower left", frameon=True, framealpha=1.0, edgecolor="none")
    panel_title(ax, "c", t["t_fold"])
    span = crossover(k72)
    for i, ax in enumerate(axes):
        kk = kb if i == 1 else ks
        ax.set_xscale("log")
        ax.set_xticks(list(kk))
        ax.set_xticklabels([str(int(k)) for k in kk])
        ax.minorticks_off()
        ax.set_xlabel(t["K"])
        if span is not None and i == 2:
            ax.axvspan(*span, color="#EEEEEE", zorder=0, lw=0)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.17, wspace=0.34)
    alignment_gate(fig, f"fig-{gate}" if lang == "en" else f"图{gate}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 图 3、图 4：编码器族 × 留出域的泄漏与方法选择（75 号汇总）
# ═══════════════════════════════════════════════════════════════════════════════
def load_generality() -> tuple[dict[str, pd.DataFrame], str]:
    """75 号汇总工作簿；正式工作簿不在或有缺表时退用 pilot 汇总，图内标「预览」。"""
    formal = os.path.join(eu.OUTPUT_DIR, "75_leakage_generality_summary.xlsx")
    pilot = os.path.join(eu.OUTPUT_DIR, "75_leakage_generality_summary__pilot.xlsx")
    tag = "formal"
    path = formal
    if os.path.exists(formal):
        xl = pd.ExcelFile(formal)
        if not xl.parse(xl.sheet_names[5]).dropna(how="all").empty:
            tag, path = "preview", pilot
    else:
        tag, path = "preview", pilot
    xl = pd.ExcelFile(path)
    keys = ["a", "b", "c", "d", "e", "f"]
    return {k: xl.parse(sh) for k, sh in zip(keys, xl.sheet_names, strict=False)}, tag


def family_interval(c: pd.DataFrame, fam: str, contrast: str) -> tuple[float, float, float, str] | None:
    """主口径某族某对照的估计与 95% 区间：混合效应截距及其区间；拟合不可用时取折级均值与簇自助区间。

    返回 (估计, 下限, 上限, 来源)；该族该对照无汇总行时返回 None。图 3(b) 与图形摘要共用。
    """
    if c.empty:
        return None
    main_scope = c[c["口径"].astype(str).str.startswith("主口径")]
    r = main_scope[(main_scope["编码器"] == fam) & (main_scope["对照"] == contrast)]
    if r.empty:
        return None
    row = r.iloc[0]
    est = row.get("混合效应截距", np.nan)
    if pd.notna(est):
        return float(est), float(row["截距95%下限"]), float(row["截距95%上限"]), "混合效应截距"
    return float(row["折级均值"]), float(row["簇自助95%下限"]), float(row["簇自助95%上限"]), "折级均值+簇自助"


def _domain_positions(a: pd.DataFrame) -> tuple[dict[tuple[str, str], float], list[tuple[str, float]]]:
    """主口径 18 个留出域的横坐标（数据集之间留空）与各数据集的中点。"""
    pos: dict[tuple[str, str], float] = {}
    mids: list[tuple[str, float]] = []
    x = 0.0
    for ds in GEN_SETS:
        helds = sorted(a.loc[a["dataset"] == ds, "held"].astype(str).unique())
        if not helds:
            continue
        start = x
        for h in helds:
            pos[(ds, h)] = x
            x += 1.0
        mids.append((ds, (start + x - 1.0) / 2))
        x += 0.7
    return pos, mids


def fig_families(lang: str, g: dict[str, pd.DataFrame], tag: str) -> Figure:
    t = TXT[lang]
    a, c = g["a"], g["c"]
    a = a[a["dataset"].isin(GEN_SETS)].copy()
    a["held"] = a["held"].astype(str)
    fig = plt.figure(figsize=(FIG_W, 70 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.0], left=0.065, right=0.99, top=0.9, bottom=0.2, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    pos, mids = _domain_positions(a)
    offs = dict(zip(FAM_ORDER, np.linspace(-0.3, 0.3, len(FAM_ORDER)), strict=True))
    for fam in FAM_ORDER:
        sub = a[(a["编码器"] == fam) & a["Δ_rep"].notna()]
        if sub.empty:
            continue
        stl = FAM_STYLE[fam]
        xs = [pos[(d, h)] + offs[fam] for d, h in zip(sub["dataset"], sub["held"], strict=True)]
        ax.plot(
            xs,
            sub["Δ_rep"],
            ls="none",
            marker=stl["marker"],
            ms=3.4,
            mfc=stl["mfc"],
            mec=stl["color"],
            mew=0.8,
            label=FAM_LABEL[fam],
            zorder=3,
        )
    ax.axhline(0, color="#999999", lw=0.6, zorder=1)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([_held_short(d, h) for d, h in pos], fontsize=5.8, rotation=90)
    seps = [(pos[d0] + pos[d1]) / 2 for (d0, _), (d1, _) in itertools.pairwise(pos.items()) if d0[0] != d1[0]]
    for ds, xm in mids:
        ax.text(
            xm, -0.2, GEN_SET_LABEL[lang][ds], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.3
        )
    ax.set_xlim(min(pos.values()) - 0.6, max(pos.values()) + 0.6)
    ax.set_ylabel(t["rep"], fontsize=FS_SUB)
    # 上方留出一条给图例，图例整个落在数据之上；数据集分隔线在图例横向范围内只画到图例下沿，不穿过图例文字
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    leg = ax.legend(
        loc="upper right",
        ncol=3,
        handletextpad=0.2,
        columnspacing=0.8,
        fontsize=6.3,
        borderaxespad=0.2,
    )
    fig.canvas.draw()
    lb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    for xs_ in seps:
        xa = ax.transAxes.inverted().transform(ax.transData.transform((xs_, 0.0)))[0]
        top = lb.y0 - 0.01 if lb.x0 - 0.01 <= xa <= lb.x1 + 0.01 else 1.0
        ax.axvline(xs_, ymax=top, color="#DDDDDD", lw=0.5, zorder=0)
    panel_title(ax, "a", t["t_fam_a"])
    # (b) 各族在主口径 18 个留出域上的混合效应均值与 95% 区间（拟合不可用时取留出域均值与簇自助区间）
    ax = fig.add_subplot(gs[0, 1])
    for j, (contrast, key, filled) in enumerate((("Δ_rep", "rep", True), ("Δ_half", "half", False))):
        for i, fam in enumerate(FAM_ORDER):
            iv = family_interval(c, fam, contrast)
            if iv is None:
                continue
            est, lo, hi, _ = iv
            y = i + (-0.17 if j == 0 else 0.17)
            stl = FAM_STYLE[fam]
            ax.errorbar(
                [est],
                [y],
                xerr=[[est - lo], [hi - est]],
                fmt=stl["marker"],
                ms=3.4,
                color=stl["color"],
                mfc=stl["color"] if filled else "white",
                elinewidth=0.8,
                capsize=1.4,
                label=t[key] if i == 0 else None,
            )
    ax.axvline(0, color="#999999", lw=0.6)
    ax.set_yticks(range(len(FAM_ORDER)))
    ax.set_yticklabels([FAM_LABEL[f] for f in FAM_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel(t["diff"])
    ax.legend(loc="lower right", handletextpad=0.3, fontsize=FS_SUB)
    panel_title(ax, "b", t["t_fam_b"])
    if tag != "formal":
        fig.text(
            0.5, 0.5, t["preview"], ha="center", va="center", fontsize=11, color="#D55E00", alpha=0.35, rotation=12
        )
    alignment_gate(
        fig,
        "fig-c" if lang == "en" else "图c",
        [{"panels": ["a", "b"], "checks": ["panel-width"], "reason": "domain panel a wider than summary panel b"}],
    )
    return fig


def fig_choice(lang: str, g: dict[str, pd.DataFrame], tag: str) -> Figure:
    t = TXT[lang]
    d = g["d"].copy()
    d["held"] = d["held"].astype(str)
    fig = plt.figure(figsize=(FIG_W, 72 * MM))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.0, 0.95, 1.0], left=0.065, right=0.99, top=0.9, bottom=0.17, wspace=0.42
    )
    # (a) 苹果同仪器两年设计：两折均值
    ax = fig.add_subplot(gs[0, 0])
    ap = d[d["dataset"] == "apple_two_year"]
    for route, col, ls, mk, filled in CHOICE_ROUTES:
        if ap.empty or route not in ap:
            continue
        m = ap.groupby("K")[route].mean().reindex(CHOICE_K)
        ax.plot(
            m.index,
            m.to_numpy(),
            color=col,
            ls=ls,
            lw=1.1,
            marker=mk,
            ms=3.0,
            mfc=col if filled else "white",
            label=ROUTE_LABEL_CHOICE[lang][route],
        )
    ax.set_xscale("log")
    ax.set_xticks(CHOICE_K)
    ax.set_xticklabels([str(k) for k in CHOICE_K])
    ax.minorticks_off()
    ax.set_xlabel(t["K2"])
    ax.set_ylabel(t["f1"])
    # 图例单列放在右下：两列会比本面板宽、伸出左轴压住刻度；下方留出的空白只放图例，刻度只标曲线所在范围
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.45 * (hi - lo), hi)
    ax.set_yticks(np.arange(np.ceil(lo * 10) / 10, hi + 1e-9, 0.1))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(
            loc="lower right",
            ncol=1,
            handlelength=2.0,
            handletextpad=0.3,
            labelspacing=0.3,
            borderaxespad=0.2,
            frameon=False,
            fontsize=5.6,
        )
    panel_title(ax, "a", t["t_ch_a"])
    # (b) 逐留出域逐 K：底色为无泄漏第一；与之不同时写上暴露第一
    ax = fig.add_subplot(gs[0, 1])
    rows = [(ds, h) for ds in GEN_SETS for h in sorted(d.loc[d["dataset"] == ds, "held"].unique())]
    for i, (ds, h) in enumerate(rows):
        for j, k in enumerate(CHOICE_K):
            r = d[(d["dataset"] == ds) & (d["held"] == h) & (d["K"] == k)]
            if r.empty:
                ax.add_patch(Rectangle((j, i), 1, 1, fc="white", ec="#CCCCCC", lw=0.3))
                continue
            row = r.iloc[0]
            free, leak = CHOICE_ABBR[str(row["无泄漏第一"])], CHOICE_ABBR[str(row["泄漏第一"])]
            ax.add_patch(Rectangle((j, i), 1, 1, fc=CHOICE_FAMILY_COLOR[free], ec="white", lw=0.6))
            if bool(row["翻转"]):
                ax.text(j + 0.5, i + 0.5, leak, ha="center", va="center", fontsize=5.4, fontweight="bold")
    ax.set_xlim(0, len(CHOICE_K))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.5 for j in range(len(CHOICE_K))])
    ax.set_xticklabels([str(k) for k in CHOICE_K])
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([f"{GEN_SET_LABEL[lang][ds]} {_held_short(ds, h)}" for ds, h in rows], fontsize=5.4)
    ax.tick_params(length=0)
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(t["K2"])
    handles = [Rectangle((0, 0), 1, 1, fc=CHOICE_FAMILY_COLOR[f], ec="none") for f in CHOICE_FAMILY_COLOR]
    ax.legend(
        handles,
        list(CHOICE_FAMILY_COLOR),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        handlelength=1.0,
        columnspacing=0.6,
        fontsize=5.8,
    )
    panel_title(ax, "b", t["t_ch_b"])
    # (c) 各组件的泄漏（主口径 18 个留出域，横线为均值）：
    #     编码器（原型网络 S_src − R）、SVM 与 PLS-DA 超参数（留出域选参 − 源域选参）
    ax = fig.add_subplot(gs[0, 2])
    comps = [
        ("编码器泄漏", t["enc_leak"], "#0072B2"),
        ("SVM超参数泄漏", t["svm_leak"], "#D55E00"),
        ("PLS-DA超参数泄漏", t["pls_leak"], "#009E73"),
    ]
    rng = np.random.default_rng(0)
    main = d[d["dataset"].isin(GEN_SETS)]
    for j, k in enumerate(CHOICE_K):
        sub = main[main["K"] == k]
        for q, (col, lab, colr) in enumerate(comps):
            x0 = j + (q - 1) * 0.25
            v = sub[col].dropna().to_numpy()
            if v.size == 0:
                continue
            ax.scatter(x0 + rng.uniform(-0.05, 0.05, v.size), v, s=4, color=colr, alpha=0.55, lw=0, zorder=2)
            ax.plot([x0 - 0.1, x0 + 0.1], [v.mean()] * 2, color=colr, lw=1.4, zorder=3, label=lab if j == 0 else None)
    ax.axhline(0, color="#999999", lw=0.6, zorder=1)
    ax.set_xticks(range(len(CHOICE_K)))
    ax.set_xticklabels([str(k) for k in CHOICE_K])
    ax.set_xlabel(t["K2"])
    ax.set_ylabel(t["diff"])
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))  # 上方留出图例的位置，图例不压散点
    ax.legend(loc="upper right", handlelength=1.2, fontsize=5.8, borderaxespad=0.2)
    panel_title(ax, "c", t["t_ch_c"])
    if tag != "formal":
        fig.text(
            0.5, 0.5, t["preview"], ha="center", va="center", fontsize=11, color="#D55E00", alpha=0.35, rotation=12
        )
    alignment_gate(
        fig,
        "fig-d" if lang == "en" else "图d",
        [{"panels": ["a", "b", "c"], "checks": ["panel-width"], "reason": "matrix panel b sized to its grid"}],
    )
    return fig


def load_alpha() -> pd.DataFrame:
    """17 号融合权重扫描的正式五种子结果（K=5）：每折每个 α 的宏 F1 种子均值与种子间标准差。"""
    base = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
    frames = []
    for s in eu.RECOMMENDED_SEEDS:
        d = pd.read_excel(os.path.join(base, f"17_alpha_sensitivity__seed{s}.xlsx"), sheet_name=0)
        d["种子"] = s
        frames.append(d)
    panel = pd.concat(frames, ignore_index=True)
    ycol = next(c for c in panel.columns if "年份" in str(c))
    acol = next(c for c in panel.columns if "alpha" in str(c).lower())
    fcol = next(c for c in panel.columns if "F1均值" in str(c))
    kcol = next(c for c in panel.columns if str(c).strip() == "K")
    g = panel[panel[kcol] == 5].groupby([ycol, acol])[fcol]
    out = pd.DataFrame({"F1均值": g.mean(), "跨种子std": g.std(ddof=1), "种子数": g.size()}).reset_index()
    return out.rename(columns={ycol: "折", acol: "alpha"})


def fig_alpha(lang: str, al: pd.DataFrame) -> Figure:
    """图 S1：(a) 融合权重 α × 留出折的宏 F1 热力图，★ 为每折最优 α；(b) α=0 与 α=0.4 的逐折对比。"""
    t = TXT[lang]
    folds = [2018, 2019, 2025]
    alphas = np.array(sorted(al["alpha"].unique()), dtype=float)
    grid = np.vstack(
        [al[al["折"] == f].set_index("alpha")["F1均值"].reindex(alphas).to_numpy(dtype=float) for f in folds]
    )
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 60 * MM), gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    # extent 按格边界给：★ 画在真实 α 处，与色块对齐
    half = (float(alphas[-1]) - float(alphas[0])) / (2 * (len(alphas) - 1))
    im = ax.imshow(
        grid,
        aspect="auto",
        cmap="cividis",
        origin="upper",
        extent=(float(alphas[0]) - half, float(alphas[-1]) + half, len(folds) - 0.5, -0.5),
    )
    for i in range(len(folds)):
        j = int(np.argmax(grid[i]))
        ax.plot(alphas[j], i, "*", ms=8, color="white", mec="black", mew=0.5, zorder=3, clip_on=False)
    ax.axvline(0.4, ls="--", lw=0.9, color="#D55E00", zorder=2)
    ax.text(0.43, -0.34, "\u03b1 = 0.4", ha="left", va="center", color="#D55E00", fontsize=6.5)  # 标在虚线右侧，不压线
    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([f"LOEO-{f}" for f in folds])
    ax.set_xlabel(t["alpha_x"])
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label(t["alpha_f1"])
    cb.ax.tick_params(labelsize=6.5, width=0.6, length=2.5)
    plt.setp(cb.outline, linewidth=0.6)
    panel_title(ax, "a", t["t_alpha_a"])
    ax = axes[1]
    i0 = int(np.argmin(np.abs(alphas - 0.0)))
    i4 = int(np.argmin(np.abs(alphas - 0.4)))
    ys = np.arange(len(folds))[::-1]
    for y, i in zip(ys, range(len(folds)), strict=True):
        v0, v4 = float(grid[i, i0]), float(grid[i, i4])
        ax.hlines(y, min(v0, v4), max(v0, v4), color="#BBBBBB", lw=2.4, zorder=1)
        ax.plot(v0, y, "o", ms=5, color="#0072B2", mec="black", mew=0.4, zorder=3)
        ax.plot(v4, y, "^", ms=5, color="#56B4E9", mec="black", mew=0.4, zorder=3)
        d = v4 - v0
        txt = "\u2248 0" if abs(d) < 5e-4 else f"{d:+.3f}".replace("-", "\u2212")
        ax.text((v0 + v4) / 2, y + 0.2, txt, ha="center", va="bottom", fontsize=6.5, color="#444444")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"LOEO-{f}" for f in folds])
    ax.set_ylim(-0.75, len(folds) - 0.35)
    ax.set_xlabel(t["alpha_f1"])
    handles = [
        Line2D([], [], marker="o", ls="none", ms=5, color="#0072B2", mec="black", mew=0.4, label=t["alpha0"]),
        Line2D([], [], marker="^", ls="none", ms=5, color="#56B4E9", mec="black", mew=0.4, label=t["alpha04"]),
    ]
    ax.legend(handles=handles, loc="upper left", handletextpad=0.3)  # 左上无数据，不压哑铃线
    panel_title(ax, "b", t["t_alpha_b"])
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.2, wspace=0.5)
    alignment_gate(
        fig,
        "fig-e" if lang == "en" else "图e",
        [{"panels": ["a", "b"], "checks": ["panel-width"], "reason": "heat map with colour bar beside dumbbells"}],
        axes=[axes[0], axes[1]],
        panel_ids=["a", "b"],
        row_groups=[["a", "b"]],
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="67saa_figures：SAA 正文图 1–4")
    parser.add_argument("--no_copy", action="store_true", help="不复制到稿件 figs/")
    args = parser.parse_args()
    logger = eu.get_logger(STEM)
    style()
    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    dom = domain_table(df)
    wl, st = spectra_stats(df, wl_cols)
    res, tag = load_leakage()
    arms, cons = leakage_frames(res)
    md = load_methods()
    al = load_alpha()
    logger.log(f"图 3 数据口径：{tag}（{sorted(res['种子'].unique().tolist())}）")

    g, gtag = load_generality()
    logger.log(f"图 3、图 4 数据口径：{gtag}")

    def plot(i: int, lang: str) -> Figure:
        style()  # save_figure_bilingual 先设中文字体，这里再压一遍本图字号与线宽
        if i == 0:
            return fig_design(lang, dom)
        if i == 1:
            return fig_spectra(lang, wl, st)
        if i == 2:
            return fig_families(lang, g, gtag)
        return fig_choice(lang, g, gtag)

    for i in range(4):
        zh, en = eu.save_figure_bilingual(partial(plot, i), STEM, i, dpi=600)
        logger.log(f"图 {i + 1} → {en} / {zh}")
        if not args.no_copy:
            if i in (2, 3) and gtag != "formal":
                logger.log(f"  图 {i + 1} 为预览口径：仍复制到稿件目录，图内有「预览」水印；正式结果到位后重跑本脚本")
            shutil.copyfile(en, os.path.join(FIG_DIR, f"Figure_{i + 1}.pdf"))
            shutil.copyfile(zh, os.path.join(FIG_DIR, f"图{i + 1}.pdf"))

    def plot_si(i: int, lang: str) -> Figure:
        style()
        if i == 4:
            return fig_alpha(lang, al)
        if i == 5:
            return fig_leakage(lang, arms, cons, tag, "f")
        return fig_methods(lang, md, "g")

    for i, name in ((4, "S1"), (5, "S2"), (6, "S3")):
        zh, en = eu.save_figure_bilingual(partial(plot_si, i), STEM, i, dpi=600)
        logger.log(f"图 {name} → {en} / {zh}")
        if not args.no_copy:
            if name == "S2" and tag != "formal":
                logger.log("  图 S2 为预览口径：图内有「预览」水印；正式结果到位后重跑本脚本")
            shutil.copyfile(en, os.path.join(FIG_DIR, f"Figure_{name}.pdf"))
            shutil.copyfile(zh, os.path.join(FIG_DIR, f"图{name}.pdf"))

    sheets: dict[object, object] = {
        eu.table_sheet_name(0, "图1各域计数"): dom,
        eu.table_sheet_name(1, "图3各臂逐折"): arms.assign(种子值=arms["种子值"].astype(str)),
        eu.table_sheet_name(2, "图3对照差值"): cons,
        eu.table_sheet_name(3, "图4b固定设置三折均值"): md["mean"].reset_index(),
        eu.table_sheet_name(4, "图4b固定设置跨种子SD"): md["sd"].reset_index(),
        eu.table_sheet_name(5, "图4ac同情节三折均值"): md["k72"].reset_index(),
        eu.table_sheet_name(6, "图4c逐折均值"): md["f72"],
        eu.table_sheet_name(7, "图S1融合权重扫描"): al,
        eu.table_sheet_name(8, "图3逐留出域对照"): g["a"],
        eu.table_sheet_name(9, "图3跨数据集推断"): g["c"],
        eu.table_sheet_name(10, "图4逐域排序"): g["d"],
        "说明": pd.DataFrame([{"图S2口径": tag, "图3图4口径": gtag}]),
    }
    eu.write_script_workbook(__file__, sheets)
    logger.log(f"底层数据 → {os.path.join(eu.OUTPUT_DIR, STEM + '.xlsx')}")
    logger.close()


if __name__ == "__main__":
    main()
