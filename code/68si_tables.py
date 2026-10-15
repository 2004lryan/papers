"""68si_tables.py: 补充材料表 S9–S13 与正文表 1 的表体、正文 3.2 与 3.5 节的数字宏，全部从结果工作簿读取，写回生成块。

    表 S9   无泄漏路线的宏 F1 与各路线所用信息
            原型网络、只看支持集的 SVM ← 20 号 Formal 归档（经 42 号 load_formal_panel）
            池化 SVM / CORAL-SVM / TCA-SVM ← 04outputs/65_domain_adaptation_baselines.xlsx 表a（逐种子逐折）
            仅源域训练的 1D-CNN / 1D-ResNet ← formal_seeds/22_cnn_resnet_baseline__seed*.xlsx
    表 S10  光谱质控与 Savitzky-Golay 宽度 ← 04outputs/66_sg_window_sensitivity.xlsx（表c、表e、表f）
            (a) 保留光谱数与域距离；(b) 光谱空间三条路线的三折宏 F1（4 位小数，工作簿即 4 位）
    表 S11  协议对齐编码器臂的种子级明细 ← 04outputs/64_leakage_controls.xlsx「对照汇总」
            正式汇总缺失或种子不全时，退用 formal_seeds/64_leakage__*__pilot.xlsx 中的正式种子，
            数值一律包 \\pend{}，表示预览、非投稿数字
    正文表 1（同装置两年设计）、表 S15（三年设计）与数字宏 \\Lk…  ← 与表 S11 同一份 64 号逐种子结果
            （各臂逐折均值、各对照逐折均值与逐格正负）；
            捆绑对照的占比取表 S12 的「两臂均不融合」差值
    表 S12  早先的捆绑对照（Formal 5 种子）
            共享编码器 α=0.4 ← formal_seeds/06_loeo_kshot__seed*.xlsx；α=0 ← formal_seeds/06_alpha0.0__seed*.xlsx
            折限定 α=0.4 / α=0 ← formal_seeds/12_loeo_ablate_alpha__seed*.xlsx 的配置 A_uniform_0.4 / C_no_fusion
    表 S13  逐年波段归因 ← 04outputs/70_band_attribution.xlsx（表a 宏 F1、表b 分块重要性）：
            每条件每年的宏 F1、按块中心波长分五区的正重要性份额、下降最大的块
    数字宏 \\Ba…（正文 3.5 节与补充材料 S2.7 共用）← 表 S13 同一份数；70 号不分预览与正式，不包 \\pend{}
    补充材料里的 \\Lk… 与正文同一份宏（生成块 LK 同时写入两份文件）
    数字宏 \\Ex…（补充材料 S1.6 与 S8）← 04outputs/24_crossdataset_leakage.xlsx（及 __cap0、__buf10_indianpines_spatial
            两份敏感性）、25_mixed_effects_leakage.xlsx、31_leakage_decomposition.xlsx；24 号为 CUDA 五种子结果、
            25 号苹果 Δ 取自 64 号 Formal 汇总时才不包 \\pend{}
    表 S9 前四行、表 S14 与数字宏 \\Mc… ← formal_seeds/72_kscan__seed*.xlsx（64 号 Formal 编码器 R、S_src、S_pool
            在单面情节上的 K 扫描、72_kscan_arms.xlsx 的折级检验）、04outputs/71_source_only_tuning.xlsx（源年份选参的
            SVM、PLS-DA，同一批情节）、73_grid_edge_check.xlsx（外沿格外推）与 42_ensemble_kshot.xlsx（集成三折均值）；
            正文 3.4 节的定性表述在 macros_mc 里逐条断言

均值口径：先对每个种子求三折均值，再对种子求均值（与 42/64 号脚本相同）；小数位在最后一步一次取舍，
不对已取舍的数再取舍。

运行方式:
    cd <项目根>
    python 02code/68si_tables.py            # 重写 SI 中 %% <gen:KEY> … %% </gen:KEY> 之间的表体
    python 02code/68si_tables.py --check    # 只核对，表体与结果文件不一致时退出码 1

输出文件:
    06doc/01manuscript/Supplementary_en.tex（及带生成块的 Supplementary_zh.tex）— 原地替换生成块，靠 git 留底
    06doc/01manuscript/Elsevier_en.tex（及带生成块的 Elsevier_zh.tex）— 生成块 T1（表 1 表体；表 2 由 77 号生成）
            与 LK、BA、EX（数字宏）
    04outputs/68si_tables.xlsx — 各表数值
    05logs/68si_tables_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import ast
import glob
import importlib
import os
import re
import sys
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import export_utils as eu  # noqa: E402

SEEDS = [int(s) for s in eu.RECOMMENDED_SEEDS]
FOLDS = [2018, 2019, 2025]
K_S9 = [1, 5, 10, 20, 50]
K_S10 = [1, 5, 50]
FS = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
MS_DIR = os.path.join(eu.BASE_DIR, "06doc", "01manuscript")
TEX = {"en": os.path.join(MS_DIR, "Supplementary_en.tex"), "zh": os.path.join(MS_DIR, "Supplementary_zh.tex")}
MAIN = {"en": os.path.join(MS_DIR, "Elsevier_en.tex"), "zh": os.path.join(MS_DIR, "Elsevier_zh.tex")}

LABELS: dict[str, dict[str, str]] = {
    "en": {
        "VanillaProtoNet": "Prototypical network (encoder R)",
        "SVM-Kshot": "Support vector machine",
        "Pooled-SVM": "Pooled-SVM",
        "CORAL-SVM": "CORAL-SVM",
        "TCA-SVM": "TCA-SVM",
        "CNN1D": "1D-CNN, source only",
        "ResNet1D": "1D-ResNet, source only",
        "nosupport": "no support set",
        "all3": "Three-fold mean",
        "f2025": "LOEO-2025 fold",
        "strict": "strict",
        "broad": "broad",
        "none": "none",
        "three_year": "Three-year, all spectra",
        "two_year": "Two-year, all spectra",
        "Q0": "Three-year, strict QC",
        "QB": "Three-year, broad QC",
        "Q1": "Three-year, equal width 25 nm",
        "Q3": "Three-year, no smoothing",
        "shared": "Shared encoder",
        "fold": "Fold-restricted encoder",
        "diff": "Difference",
        "reg04": "Fusion weight 0.4 in both arms",
        "reg00": "No fusion in either arm",
        "PN-R": "Prototypical network, encoder R",
        "PN-Spool": r"Prototypical network, pooled encoder S\textsubscript{pool}",
        "SVM-tuned": "Support vector machine, source-tuned",
        "PLS-tuned": "PLS-DA, source-tuned",
        "SVM-fixed": "Support vector machine, fixed setting",
    },
    "zh": {
        "VanillaProtoNet": "原型网络（编码器 R）",
        "SVM-Kshot": "支持向量机",
        "Pooled-SVM": "池化 SVM",
        "CORAL-SVM": "CORAL-SVM",
        "TCA-SVM": "TCA-SVM",
        "CNN1D": "1D-CNN（仅源域）",
        "ResNet1D": "1D-ResNet（仅源域）",
        "nosupport": "不用支持集",
        "all3": "三折均值",
        "f2025": "LOEO-2025 折",
        "strict": "严格",
        "broad": "宽松",
        "none": "无",
        "three_year": "三年设计，全部光谱",
        "two_year": "两年设计，全部光谱",
        "Q0": "三年设计，严格质控",
        "QB": "三年设计，宽松质控",
        "Q1": "三年设计，等宽 25 nm",
        "Q3": "三年设计，不平滑",
        "shared": "共享编码器",
        "fold": "折限定编码器",
        "diff": "差值",
        "reg04": "两臂融合权重均为 0.4",
        "reg00": "两臂均不融合",
        "PN-R": "原型网络（编码器 R）",
        "PN-Spool": r"原型网络（合并池编码器 S\textsubscript{pool}）",
        "SVM-tuned": "支持向量机（源年份选参）",
        "PLS-tuned": "PLS-DA（源年份选参）",
        "SVM-fixed": "支持向量机（固定设置）",
    },
}
INFO = {
    "VanillaProtoNet": "S, K",
    "PN-R": "S, K",
    "PN-Spool": "S, T, K",
    "SVM-tuned": "K, (S)",
    "PLS-tuned": "K, (S)",
    "SVM-Kshot": "K",
    "Pooled-SVM": "S, K",
    "CORAL-SVM": "S, U, K",
    "TCA-SVM": "S, U, K",
    "CNN1D": "S",
    "ResNet1D": "S",
}
CONTRAST_TEX = {
    "Δ_rep": r"S\textsubscript{src}$-$R",
    "Δ_size": r"S\textsubscript{size}$-$R",
    "Δ_lab": r"S\textsubscript{src}$-$S\textsubscript{shuf}",
    "Δ_shuf": r"S\textsubscript{shuf}$-$R",
    "Δ_ckpt": r"S\textsubscript{pool}$-$S\textsubscript{src}",
    "Δ_ckpt_R": r"R\textsubscript{ckpt}$-$R",
    "Δ_all": r"S\textsubscript{pool}$-$R",
    "Δ_half": r"S\textsubscript{half}$-$R\textsuperscript{h}",
    "Δ_seen": r"S\textsubscript{src}\textsuperscript{h}$-$S\textsubscript{half}",
    "Δ_rep_B": r"S\textsubscript{src}\textsuperscript{h}$-$R\textsuperscript{h}",
}
CONTRASTS = {
    "Δ_rep": ("S_src", "R"),
    "Δ_size": ("S_size", "R"),
    "Δ_lab": ("S_src", "S_shuf"),
    "Δ_shuf": ("S_shuf", "R"),
    "Δ_ckpt": ("S_pool", "S_src"),
    "Δ_ckpt_R": ("R_ckpt", "R"),
    "Δ_all": ("S_pool", "R"),
    "Δ_half": ("S_half", "R_B"),
    "Δ_seen": ("S_src_B", "S_half"),
    "Δ_rep_B": ("S_src_B", "R_B"),
}
T1_ARMS = {
    "en": [
        ("R", "R, source only"),
        ("R_ckpt", r"R\textsubscript{ckpt}, held-out checkpoint"),
        ("S_size", r"S\textsubscript{size}, equal size"),
        ("S_shuf", r"S\textsubscript{shuf}, shuffled labels"),
        ("S_src", r"S\textsubscript{src}, target exposure"),
        ("S_pool", r"S\textsubscript{pool}, pooled"),
    ],
    "zh": [
        ("R", "R，仅源域"),
        ("R_ckpt", r"R\textsubscript{ckpt}，留出年选检查点"),
        ("S_size", r"S\textsubscript{size}，等样本量"),
        ("S_shuf", r"S\textsubscript{shuf}，标签置乱"),
        ("S_src", r"S\textsubscript{src}，目标暴露"),
        ("S_pool", r"S\textsubscript{pool}，合并池"),
    ],
}
T1_CONTRASTS = ["Δ_rep", "Δ_lab", "Δ_ckpt", "Δ_ckpt_R", "Δ_half", "Δ_seen"]


def num(v: float, nd: int = 3) -> str:
    """LaTeX 数字：负号用数学减号。"""
    s = f"{v:.{nd}f}"
    return f"$-${s[1:]}" if s.startswith("-") and float(s) != 0 else s.lstrip("-")


def seed_fold_mean(df: pd.DataFrame, key: list[str], val: str) -> pd.Series:
    """先对每个种子求所选折的均值，再对种子求均值。df 需含 种子 列。"""
    return df.groupby([*key, "种子"])[val].mean().groupby(key).mean()


def t_interval(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    half = float(stats.t.ppf(0.975, arr.size - 1)) * float(arr.std(ddof=1)) / float(np.sqrt(arr.size))
    return float(arr.mean()) - half, float(arr.mean()) + half


# ── 表 S9 ────────────────────────────────────────────────────────────────────
def data_s9(logger: eu.Logger) -> pd.DataFrame:
    m42 = importlib.import_module("42_ensemble_kshot")
    panel = m42.load_formal_panel()
    if panel is None:
        raise FileNotFoundError("缺 20 号 Formal 归档")
    panel = panel[panel["方法"].isin(["VanillaProtoNet", "SVM-Kshot"])][["方法", "目标年份", "K", "种子", "F1均值"]]
    da = pd.read_excel(os.path.join(eu.OUTPUT_DIR, "65_domain_adaptation_baselines.xlsx"), sheet_name=0)
    da = da[da["方法"].isin(["Pooled-SVM", "CORAL-SVM", "TCA-SVM"])][["方法", "目标年份", "K", "种子", "F1均值"]]
    cnn = []
    for f in sorted(glob.glob(os.path.join(FS, "22_cnn_resnet_baseline__seed*.xlsx"))):
        d = pd.read_excel(f, sheet_name=0)
        seed = int(os.path.basename(f).split("seed")[-1].split(".")[0])
        cnn.append(
            d.rename(columns={"TargetYear": "目标年份", "Method": "方法", "F1_mean": "F1均值"}).assign(种子=seed, K=0)[
                ["方法", "目标年份", "K", "种子", "F1均值"]
            ]
        )
    # 同一批情节上的四条路线：72 号的编码器 R、S_pool，71 号源年份选参的 SVM、PLS-DA
    k72 = pd.concat(
        [pd.read_excel(os.path.join(FS, f"72_kscan__seed{s}.xlsx"), sheet_name=0) for s in SEEDS], ignore_index=True
    )
    k72 = k72[k72["臂"].isin(["R", "S_pool"])].assign(方法=lambda d: d["臂"].map({"R": "PN-R", "S_pool": "PN-Spool"}))
    t71 = pd.read_excel(os.path.join(eu.OUTPUT_DIR, "71_source_only_tuning.xlsx"), sheet_name=0)
    t71 = t71.assign(
        方法=t71["方法"].map({"SVM-Kshot": "SVM-tuned", "PLS-DA-Kshot": "PLS-tuned"}), F1均值=t71["目标宏F1(所选)"]
    )
    cols = ["方法", "目标年份", "K", "种子", "F1均值"]
    long = pd.concat([panel, da, *cnn, k72[cols], t71[cols]], ignore_index=True)
    for m, g in long.groupby("方法"):
        if sorted(g["种子"].unique().tolist()) != sorted(SEEDS):
            raise ValueError(f"{m} 的种子集合不是 Formal 5 种子")
    rows = []
    for (m, k), g in long.groupby(["方法", "K"]):
        by_fold = {f"LOEO-{y}": float(g[g["目标年份"] == y].groupby("种子")["F1均值"].mean().mean()) for y in FOLDS}
        mean3 = float(seed_fold_mean(g.assign(k=0), ["k"], "F1均值").iloc[0])
        rows.append({"方法": m, "K": k, "三折均值": mean3, **by_fold})
    out = pd.DataFrame(rows)
    logger.log("[S9]\n" + out.round(4).to_string(index=False))
    return out


def tex_s9(d: pd.DataFrame, lang: str) -> str:
    lab = LABELS[lang]
    order = [
        "PN-R",
        "SVM-tuned",
        "PLS-tuned",
        "SVM-Kshot",
        "Pooled-SVM",
        "CORAL-SVM",
        "TCA-SVM",
        "CNN1D",
        "ResNet1D",
        "PN-Spool",
    ]
    lines: list[str] = []
    for block, col in (("all3", "三折均值"), ("f2025", "LOEO-2025")):
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{{lab[block]}}} \\")
        for m in order:
            g = d[d["方法"] == m].set_index("K")
            if m in ("CNN1D", "ResNet1D"):
                cells = rf"\multicolumn{{5}}{{c}}{{{num(g.loc[0, col])} ({lab['nosupport']})}}"
            else:
                cells = " & ".join(num(g.loc[k, col]) for k in K_S9)
            if m == "PN-Spool":
                lines.append(r"\addlinespace[1pt]")  # 唯一不是无泄漏的一行，与上面各行隔开
            name = lab["SVM-fixed"] if m == "SVM-Kshot" else lab[m]
            lines.append(rf"\quad {name} & {INFO[m]} & {cells} \\")
        if block == "all3":
            lines.append(r"\addlinespace")
    return "\n".join(lines)


# ── 表 S10 ───────────────────────────────────────────────────────────────────
def data_s10(logger: eu.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.ExcelFile(os.path.join(eu.OUTPUT_DIR, "66_sg_window_sensitivity.xlsx"))
    defs = x.parse(next(s for s in x.sheet_names if "变体定义" in s)).set_index("变体")
    gap = x.parse(next(s for s in x.sheet_names if "域差距" in s)).set_index("变体")
    ks = x.parse(next(s for s in x.sheet_names if "KShot三折均值" in s))
    methods = ["SVM-Kshot", "PLS-DA-Kshot", "NearestCentroid"]
    order = ["V0", "Q0", "QB", "Q1", "Q2", "Q3"]
    ref = {"V0": None, "Q0": "V0", "QB": "V0", "Q1": "Q0", "Q2": "Q0", "Q3": "Q0"}
    a_rows, b_rows = [], []
    for v in order:
        dist_cols = {
            "历史-2025": "历史年-2025年份距离_平均",
            "2018-2019": "2018-2019年份距离",
            "同年产地": "同年产地均值距离_平均",
            "比_历史-2025": "跨年(历史→2025)/同年产地 距离比",
            "比_2018-2019": "跨年(2018↔2019)/同年产地 距离比",
        }
        qc = defs.loc[v, "质控档"]
        a_rows.append(
            {
                "变体": v,
                "质控": "" if pd.isna(qc) else qc,
                "SG_历史_nm": defs.loc[v, "SG跨度_2018_nm"],
                "SG_2025_nm": defs.loc[v, "SG跨度_2025_nm"],
                "保留光谱数": defs.loc[v, "保留光谱数"],
                **{k: float(gap.loc[v, c]) for k, c in dist_cols.items()},
            }
        )
        sub = ks[ks["变体"] == v].set_index("K")
        row: dict[str, Any] = {"变体": v}
        for m in methods:
            for k in K_S10:
                row[f"{m}@{k}"] = float(sub.loc[k, m])
        if ref[v]:
            rs = ks[ks["变体"] == ref[v]].set_index("K")
            row["最大|差|"] = float(max(abs(sub.loc[k, m] - rs.loc[k, m]) for m in methods for k in sub.index))
            row["参照"] = ref[v]
        b_rows.append(row)
    a, b = pd.DataFrame(a_rows), pd.DataFrame(b_rows)
    v0 = a.set_index("变体").loc["V0"]
    for c in ("比_历史-2025", "比_2018-2019"):
        logger.log(f"[S10] {c} 相对 V0 的最大|差| = {float((a[c] - v0[c]).abs().max()):.4f}")
    logger.log("[S10a]\n" + a.round(4).to_string(index=False))
    logger.log("[S10b]\n" + b.round(4).to_string(index=False))
    return a, b


def tex_s10a(a: pd.DataFrame, lang: str) -> str:
    lab = LABELS[lang]
    n_all = int(pd.read_csv(os.path.join(eu.DATA_DIR, "processed", "all_9elements.csv"), usecols=[0]).shape[0])
    cur = a.set_index("变体").loc["Q0"]  # V0 与 Q0 同一套现行平滑，只差质控
    lines: list[str] = []
    for _, r in a.iterrows():
        qc = {"严格": lab["strict"], "宽松": lab["broad"]}.get(r["质控"], lab["none"])
        src = cur if r["变体"] == "V0" else r
        sg_h = lab["none"] if src["SG_历史_nm"] == 0 else f"{src['SG_历史_nm']:.1f}"
        sg_n = lab["none"] if src["SG_2025_nm"] == 0 else f"{src['SG_2025_nm']:.1f}"
        n = n_all if r["变体"] == "V0" else int(r["保留光谱数"])
        lines.append(
            f"{r['变体']} & {qc} & {sg_h} & {sg_n} & {n} & {r['历史-2025']:.2f} & {r['2018-2019']:.2f} & "
            f"{r['同年产地']:.2f} & {r['比_历史-2025']:.4f} & {r['比_2018-2019']:.4f} \\\\"
        )
    return "\n".join(lines)


def tex_s10b(b: pd.DataFrame, lang: str) -> str:
    lines: list[str] = []
    for _, r in b.iterrows():
        vals = " & ".join(
            f"{r[f'{m}@{k}']:.4f}" for m in ("SVM-Kshot", "PLS-DA-Kshot", "NearestCentroid") for k in K_S10
        )
        mx = "--" if pd.isna(r.get("最大|差|")) else f"{r['最大|差|']:.4f} ({r['参照']})"
        lines.append(f"{r['变体']} & {vals} & {mx} \\\\")
    return "\n".join(lines)


# ── 表 S11 ───────────────────────────────────────────────────────────────────
def load_64(logger: eu.Logger) -> tuple[pd.DataFrame | None, bool]:
    """64 号逐种子结果：正式汇总含全部正式种子时用之，否则退用正式种子的 pilot 逐种子表（预览）。"""
    agg = os.path.join(eu.OUTPUT_DIR, "64_leakage_controls.xlsx")
    res: pd.DataFrame | None = None
    formal = False
    if os.path.exists(agg):
        t = pd.read_excel(agg, sheet_name=1)
        if set(t["种子"].unique().tolist()) >= set(SEEDS):
            res, formal = t, True
    if res is None:
        files = [
            f
            for f in glob.glob(os.path.join(FS, "64_leakage__*__pilot.xlsx"))
            if any(f"__seed{s}__" in f for s in SEEDS)
        ]
        res = pd.concat([pd.read_excel(f, sheet_name=0) for f in sorted(files)], ignore_index=True) if files else None
        logger.log(f"[64·预览] 正式汇总缺失，退用 {len(files)} 个正式种子的 pilot 逐种子表")
    if res is not None:
        res = res[res["src_alpha"] == 0]
    return res, formal


def data_s11(res: pd.DataFrame | None, formal: bool, logger: eu.Logger) -> pd.DataFrame:
    """长表（设计, 变体, 对照, 逐折均值, 均值, 种子 t 区间, 为正种子数, 种子数, 为正格数, 格数）。"""
    rows: list[dict[str, Any]] = []
    if res is None:
        return pd.DataFrame(rows)
    for (design, variant), sub in res.groupby(["设计", "变体"]):
        folds = sorted(sub["目标年份"].unique().tolist())
        for name, (a, b) in CONTRASTS.items():
            seeds = sorted(set(sub[sub["臂"] == a]["种子"]) & set(sub[sub["臂"] == b]["种子"]))
            if not seeds:
                continue

            def f1(arm: str, fold: int, seed: int, _sub: pd.DataFrame = sub) -> float:
                m = _sub[(_sub["臂"] == arm) & (_sub["目标年份"] == fold) & (_sub["种子"] == seed)]
                return float(m["F1均值"].iloc[0])

            def has(arm: str, fold: int, seed: int, _sub: pd.DataFrame = sub) -> bool:
                m = _sub[(_sub["臂"] == arm) & (_sub["目标年份"] == fold) & (_sub["种子"] == seed)]
                return len(m) == 1

            ok = [s for s in seeds if all(has(arm, f, s) for arm in (a, b) for f in folds)]
            per_seed = {s: [f1(a, f, s) - f1(b, f, s) for f in folds] for s in ok}
            mean_by_seed = [float(np.mean(v)) for v in per_seed.values()]
            lo, hi = t_interval(mean_by_seed) if len(ok) > 1 else (np.nan, np.nan)
            row: dict[str, Any] = {
                "设计": design,
                "变体": variant,
                "对照": name,
                "种子数": len(ok),
                "均值": float(np.mean(mean_by_seed)),
                "区间下": lo,
                "区间上": hi,
                "为正种子数": int(sum(v > 0 for v in mean_by_seed)),
                "为正格数": int(sum(x > 0 for v in per_seed.values() for x in v)),
                "格数": int(sum(len(v) for v in per_seed.values())),
            }
            for i, f in enumerate(folds):
                row[f"LOEO-{f}"] = float(np.mean([v[i] for v in per_seed.values()]))
            rows.append(row)
    out = pd.DataFrame(rows)
    logger.log(f"[S11] 正式={formal}\n" + out.round(4).to_string(index=False))
    return out


def data_t1(res: pd.DataFrame | None, logger: eu.Logger) -> pd.DataFrame:
    """各臂逐折均值与三折均值（先对种子求折均值；均值口径同 S11：先种子内三折均值，再对种子求均值）。

    只取全部折都有结果的种子；R_B、S_src_B、S_half 只在 B 半上评估，不进表 1 的臂行。
    """
    rows: list[dict[str, Any]] = []
    if res is None:
        return pd.DataFrame(rows)
    sub0 = res[res["变体"] == "canonical"]
    for (design, arm), sub in sub0.groupby(["设计", "臂"]):
        piv = sub.pivot_table(index="种子", columns="目标年份", values="F1均值").dropna()
        if piv.empty:
            continue
        row: dict[str, Any] = {"设计": design, "臂": arm, "种子数": len(piv), "均值": float(piv.mean(axis=1).mean())}
        for f in piv.columns:
            row[f"LOEO-{int(f)}"] = float(piv[f].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    logger.log("[表1] 各臂\n" + out.round(4).to_string(index=False))
    return out


def _pick(df: pd.DataFrame, **kv: Any) -> pd.Series | None:
    if df.empty:
        return None
    m = df
    for k, v in kv.items():
        m = m[m[k] == v]
    return m.iloc[0] if len(m) == 1 else None


def tex_t1(t1: pd.DataFrame, s11: pd.DataFrame, formal: bool, lang: str, design: str) -> str:
    """一种设计的五臂与五项对照：正文表 1 为同装置两年设计（两折），表 S15 为三年设计（三折）。"""

    def w(x: str) -> str:
        return x if formal else rf"\pend{{{x}}}"

    folds = [2018, 2019] if design == "two_year" else FOLDS
    entries: list[tuple[str, pd.Series | None]] = [
        (label, _pick(t1, 设计=design, 臂=arm)) for arm, label in T1_ARMS[lang]
    ]
    entries += [(CONTRAST_TEX[name], _pick(s11, 设计=design, 变体="canonical", 对照=name)) for name in T1_CONTRASTS]
    lines: list[str] = []
    for label, r in entries:
        cells = []
        for f in folds:
            if r is None or pd.isna(r.get(f"LOEO-{f}", np.nan)):
                cells.append(w("--"))
            else:
                cells.append(w(num(float(r[f"LOEO-{f}"]))))
        cells.append(w("--") if r is None else w(num(float(r["均值"]))))
        lines.append(rf"{label} & {' & '.join(cells)} \\")
    return "\n".join(lines)


def macros_lk(t1: pd.DataFrame, s11: pd.DataFrame, s12: pd.DataFrame, formal: bool) -> str:
    """正文 3.2 节、摘要与结论引用的数字宏。

    预览时凡取自 64 号结果的值都包 \\pend{}；捆绑对照的差值取自表 S12 的 Formal 归档，不包。
    宏名全部预先声明，缺数据的给「--」，正文因此在任何阶段都能编译。
    """

    def con(design: str, name: str, variant: str = "canonical") -> pd.Series | None:
        return _pick(s11, 设计=design, 变体=variant, 对照=name)

    folds_of = {"three_year": FOLDS, "two_year": [2018, 2019]}
    con_keys = {
        "Δ_rep": "Rep",
        "Δ_lab": "Lab",
        "Δ_shuf": "Shuf",
        "Δ_ckpt": "Ckpt",
        "Δ_ckpt_R": "CkptR",
        "Δ_half": "Half",
        "Δ_seen": "Seen",
        "Δ_rep_B": "RepH",
    }
    expected: list[str] = []
    for tag, n_f in (("Th", 3), ("Tw", 2)):
        for key in con_keys.values():
            expected += [f"Lk{tag}{key}{sfx}" for sfx in ("", "S", "MinS", "MaxS", "Cells", *"ABC"[:n_f])]
        for key in ("R", "Src"):
            expected += [
                f"Lk{tag}{key}{sfx}" for sfx in ("MinS", "MaxS", *"ABC"[:n_f], *(x + "S" for x in "ABC"[:n_f]))
            ]
        expected.append(f"Lk{tag}HalfShare")
    for key in ("Ckpt", "CkptR", "Shuf", "Half", "Seen"):
        expected += [f"Lk{key}Min", f"Lk{key}Max"]
    expected += ["LkSizeMax", "LkQcMaxDev", "LkQcMaxDevFold", "LkQcN", "LkBundle", "LkBundleShare"]

    m: dict[str, str] = {}
    for design, tag in (("three_year", "Th"), ("two_year", "Tw")):
        folds = folds_of[design]
        for name, key in con_keys.items():
            r = con(design, name)
            if r is None:
                continue
            m[f"Lk{tag}{key}"] = num(float(r["均值"]))
            m[f"Lk{tag}{key}S"] = num(float(r["均值"]), 2)
            fv = [float(r[f"LOEO-{f}"]) for f in folds]
            for letter, v in zip("ABC", fv, strict=False):
                m[f"Lk{tag}{key}{letter}"] = num(v)
            m[f"Lk{tag}{key}MinS"] = num(min(fv), 2)
            m[f"Lk{tag}{key}MaxS"] = num(max(fv), 2)
            m[f"Lk{tag}{key}Cells"] = f"{int(r['为正格数'])} of {int(r['格数'])}"
        for arm, key in (("R", "R"), ("S_src", "Src")):
            a = _pick(t1, 设计=design, 臂=arm)
            if a is not None:
                fv = [float(a[f"LOEO-{f}"]) for f in folds]
                m[f"Lk{tag}{key}MinS"] = num(min(fv), 2)
                m[f"Lk{tag}{key}MaxS"] = num(max(fv), 2)
                for letter, v in zip("ABC", fv, strict=False):
                    m[f"Lk{tag}{key}{letter}"] = num(v)
                    m[f"Lk{tag}{key}{letter}S"] = num(v, 2)
        half, rep_h = con(design, "Δ_half"), con(design, "Δ_rep_B")
        if half is not None and rep_h is not None and float(rep_h["均值"]) != 0:
            m[f"Lk{tag}HalfShare"] = f"{100 * float(half['均值']) / float(rep_h['均值']):.0f}\\%"
    # 跨两种设计、全部折：等样本量与 S_src 的最大差；检查点、置乱、半数暴露各对照的逐折范围
    size_gap: list[float] = []
    for design, folds in folds_of.items():
        a, b = _pick(t1, 设计=design, 臂="S_size"), _pick(t1, 设计=design, 臂="S_src")
        if a is not None and b is not None:
            size_gap += [abs(float(a[f"LOEO-{f}"]) - float(b[f"LOEO-{f}"])) for f in folds]
    if size_gap:
        m["LkSizeMax"] = num(max(size_gap))
    for name, key in (
        ("Δ_ckpt", "Ckpt"),
        ("Δ_ckpt_R", "CkptR"),
        ("Δ_shuf", "Shuf"),
        ("Δ_half", "Half"),
        ("Δ_seen", "Seen"),
    ):
        vals = [
            float(r[f"LOEO-{f}"])
            for design, folds in folds_of.items()
            if (r := con(design, name)) is not None
            for f in folds
        ]
        if vals:
            m[f"Lk{key}Min"] = num(min(vals))
            m[f"Lk{key}Max"] = num(max(vals))
    base = con("three_year", "Δ_rep")
    devs = [
        abs(float(r["均值"]) - float(base["均值"]))
        for v in ("Q0", "QB", "Q1", "Q3")
        if base is not None and (r := con("three_year", "Δ_rep", v)) is not None
    ]
    if devs:
        m["LkQcMaxDev"] = num(max(devs))
        m["LkQcN"] = str(len(devs))
    # 同一组质控与平滑口径的逐折偏差（三折均值的偏差见 LkQcMaxDev）
    fold_devs = [
        abs(float(r[f"LOEO-{f}"]) - float(base[f"LOEO-{f}"]))
        for v in ("Q0", "QB", "Q1", "Q3")
        if base is not None and (r := con("three_year", "Δ_rep", v)) is not None
        for f in folds_of["three_year"]
    ]
    if fold_devs:
        m["LkQcMaxDevFold"] = num(max(fold_devs))
    bundle = _pick(s12, α="0.0", 臂="diff")
    if bundle is not None:
        m["LkBundle"] = num(float(bundle["均值"]))
        if base is not None:
            m["LkBundleShare"] = f"{100 * float(base['均值']) / float(bundle['均值']):.0f}\\%"
    unknown = set(m) - set(expected)
    if unknown:
        raise ValueError(f"未声明的宏：{sorted(unknown)}")
    always_formal = {"LkBundle"}

    def w(key: str, x: str) -> str:
        return x if formal or key in always_formal else rf"\pend{{{x}}}"

    lines = [rf"\newcommand{{\{k}}}{{{w(k, m.get(k, '--'))}}}" for k in expected]
    return "\n".join(lines)


def tex_s11(d: pd.DataFrame, formal: bool, lang: str) -> str:
    lab = LABELS[lang]

    def w(s: str) -> str:
        return s if formal else rf"\pend{{{s}}}"

    lines: list[str] = []
    blocks = [("three_year", "canonical", list(CONTRASTS)), ("two_year", "canonical", list(CONTRASTS))]
    blocks += [("three_year", v, ["Δ_rep"]) for v in ("Q0", "QB", "Q1", "Q3")]
    for design, variant, names in blocks:
        title = lab[design] if variant == "canonical" else lab[variant]
        if variant == "canonical" or variant == "Q0":
            if lines:
                lines.append(r"\addlinespace")
            if variant == "canonical":
                lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{{title}}} \\")
        for name in names:
            sub = d[(d["设计"] == design) & (d["变体"] == variant) & (d["对照"] == name)] if len(d) else d
            label = CONTRAST_TEX[name] if variant == "canonical" else rf"{title}: {CONTRAST_TEX[name]}"
            if len(sub) != 1:
                cells = " & ".join([w("--")] * 6)
            else:
                r = sub.iloc[0]
                fold_cells = [
                    w(num(r[f"LOEO-{f}"])) if f"LOEO-{f}" in r and not pd.isna(r[f"LOEO-{f}"]) else "--" for f in FOLDS
                ]
                sep = " to " if lang == "en" else "～"  # 中文稿区间一律写「a～b」
                ci = "--" if pd.isna(r["区间下"]) else f"{num(r['区间下'])}{sep}{num(r['区间上'])}"
                pos = f"{int(r['为正种子数'])}/{int(r['种子数'])}"
                cells = " & ".join([*fold_cells, w(num(r["均值"])), w(ci), w(pos)])
            lines.append(rf"\quad {label} & {cells} \\")
    return "\n".join(lines)


# ── 表 S12 ───────────────────────────────────────────────────────────────────
def data_s12(logger: eu.Logger) -> pd.DataFrame:
    def read(pfx: str, config: str | None = None) -> pd.DataFrame:
        frames = []
        for s in SEEDS:
            d = pd.read_excel(os.path.join(FS, f"{pfx}__seed{s}.xlsx"), sheet_name=0)
            if config:
                d = d[d["配置"] == config]
            frames.append(d[d["K"] == 5].assign(种子=s)[["目标年份", "种子", "F1均值"]])
        return pd.concat(frames, ignore_index=True)

    src = {
        ("0.4", "shared"): read("06_loeo_kshot"),
        ("0.0", "shared"): read("06_alpha0.0"),
        ("0.4", "fold"): read("12_loeo_ablate_alpha", "A_uniform_0.4"),
        ("0.0", "fold"): read("12_loeo_ablate_alpha", "C_no_fusion"),
    }
    rows = []
    for (alpha, arm), d in src.items():
        row: dict[str, Any] = {"α": alpha, "臂": arm}
        for f in FOLDS:
            row[f"LOEO-{f}"] = float(d[d["目标年份"] == f]["F1均值"].mean())
        row["均值"] = float(d.groupby("种子")["F1均值"].mean().mean())
        rows.append(row)
    for alpha in ("0.4", "0.0"):
        sh = next(r for r in rows if r["α"] == alpha and r["臂"] == "shared")
        fo = next(r for r in rows if r["α"] == alpha and r["臂"] == "fold")
        dif = {"α": alpha, "臂": "diff", **{c: sh[c] - fo[c] for c in ("LOEO-2018", "LOEO-2019", "LOEO-2025", "均值")}}
        m_sh = src[(alpha, "shared")].groupby("种子")["F1均值"].mean()
        m_fo = src[(alpha, "fold")].groupby("种子")["F1均值"].mean()
        dif["跨种子SD"] = float((m_sh - m_fo).std(ddof=1))
        rows.append(dif)
    out = pd.DataFrame(rows)
    logger.log("[S12]\n" + out.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    return out


def tex_s12(d: pd.DataFrame, lang: str) -> str:
    lab = LABELS[lang]
    lines: list[str] = []
    for alpha, reg in (("0.4", "reg04"), ("0.0", "reg00")):
        if lines:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{{lab[reg]}}} \\")
        for arm in ("shared", "fold", "diff"):
            r = d[(d["α"] == alpha) & (d["臂"] == arm)].iloc[0]
            cells = " & ".join(num(r[c]) for c in ("LOEO-2018", "LOEO-2019", "LOEO-2025", "均值"))
            lines.append(rf"\quad {lab[arm]} & {cells} \\")
    return "\n".join(lines)


# ── 表 S13 与数字宏 \Ba…：逐年波段归因 ───────────────────────────────────────
BA_CONDS = ("canonical", "Q0", "Q3", "canonical_trim")
BA_REGIONS = ((591.0, 606.0), (606.0, 700.0), (700.0, 780.0), (780.0, 900.0), (900.0, 1002.0))
BA_LABELS = {
    "en": {
        "canonical": "All spectra, current smoothing",
        "Q0": "Strict quality control, current smoothing",
        "Q3": "Strict quality control, no smoothing",
        "canonical_trim": "All spectra, 606 to 986 nm only",
    },
    "zh": {
        "canonical": "全部光谱，现行平滑",
        "Q0": "严格质控，现行平滑",
        "Q3": "严格质控，不平滑",
        "canonical_trim": "全部光谱，仅 606～986 nm",
    },
}


def _region(lo: float, hi: float) -> str:
    return f"{lo:.0f}-{hi - 1:.0f}"


def data_s13(logger: eu.Logger) -> pd.DataFrame:
    """70 号：每条件每年的宏 F1（种子×折）、按块中心波长分区的正重要性份额、下降最大的块。"""
    x = pd.ExcelFile(os.path.join(eu.OUTPUT_DIR, "70_band_attribution.xlsx"))
    f1 = x.parse(next(s for s in x.sheet_names if "宏F1" in s))
    imp = x.parse(next(s for s in x.sheet_names if "分块重要性" in s))
    rows: list[dict[str, Any]] = []
    for cond in BA_CONDS:
        for year in FOLDS:
            f = f1[(f1["条件"] == cond) & (f1["年份"] == year)]["宏F1"]
            g = imp[(imp["条件"] == cond) & (imp["年份"] == year)]
            pos = g["宏F1下降"].clip(lower=0)
            top = g.loc[g["宏F1下降"].idxmax()]
            row: dict[str, Any] = {
                "条件": cond,
                "年份": year,
                "宏F1均值": float(f.mean()),
                "宏F1_SD": float(f.std(ddof=1)),
                "种子×折": len(f),
                "块数": len(g),
            }
            for lo, hi in BA_REGIONS:
                inside = (g["中心nm"] >= lo) & (g["中心nm"] < hi)
                row[_region(lo, hi)] = float(pos[inside].sum() / pos.sum()) if inside.any() else float("nan")
            row.update(
                {
                    "最大块起nm": float(top["起nm"]),
                    "最大块止nm": float(top["止nm"]),
                    "最大块中心nm": float(top["中心nm"]),
                    "最大块下降": float(top["宏F1下降"]),
                    "最大块SD": float(top["SD"]),
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    logger.log("[S13]\n" + out.round(4).to_string(index=False))
    return out


def tex_s13(d: pd.DataFrame, lang: str) -> str:
    lines: list[str] = []
    for cond in BA_CONDS:
        if lines:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{{BA_LABELS[lang][cond]}}} \\")
        for _, r in d[d["条件"] == cond].iterrows():
            shares = " & ".join(
                "--" if pd.isna(r[_region(lo, hi)]) else f"{100 * r[_region(lo, hi)]:.0f}" for lo, hi in BA_REGIONS
            )
            top = f"{r['最大块起nm']:.0f}--{r['最大块止nm']:.0f}, {num(r['最大块下降'])}"
            lines.append(
                rf"\quad {int(r['年份'])} & {num(r['宏F1均值'])} $\pm$ {num(r['宏F1_SD'])} & {shares} & {top} \\"
            )
    return "\n".join(lines)


def macros_ba(d: pd.DataFrame) -> str:
    """正文 3.5 节与补充材料 S2.7 引用的归因数字；70 号结果不分预览与正式，不包 \\pend。"""

    def cell(cond: str, year: int) -> pd.Series:
        return d[(d["条件"] == cond) & (d["年份"] == year)].iloc[0]

    pig, far = _region(606.0, 700.0), _region(900.0, 1002.0)
    edge = _region(591.0, 606.0)
    # 正文 3.5 节与 S2.7、S6.5 的文字依赖下面三条；任何一条不成立都要先改写文字
    for cond in ("canonical", "Q0", "Q3"):
        if cell(cond, 2025)["最大块中心nm"] >= 606.0:
            raise ValueError(f"{cond} 2025 年下降最大的块不在窗口短波端，正文 3.5 节的表述须改写")
    if not all(cell("canonical_trim", y)[pig] > cell("canonical", y)[pig] for y in FOLDS):
        raise ValueError("去掉两端后 606–699 nm 的份额并非每年都上升，S2.7 的表述须改写")
    if cell("canonical_trim", 2018)["最大块起nm"] >= 616.0:
        raise ValueError("去掉两端后 2018 年下降最大的块不在新的短波端，S6.5 的表述须改写")
    m = {
        "BaFA": num(cell("canonical", 2018)["宏F1均值"]),
        "BaFB": num(cell("canonical", 2019)["宏F1均值"]),
        "BaFC": num(cell("canonical", 2025)["宏F1均值"]),
        "BaPigB": f"{100 * cell('canonical', 2019)[pig]:.0f}\\%",
        "BaPigC": f"{100 * cell('canonical', 2025)[pig]:.0f}\\%",
        "BaMaxA": num(cell("canonical", 2018)["最大块下降"]),
        "BaEdgeLo": f"{cell('canonical', 2025)['最大块起nm']:.0f}",
        "BaEdgeHi": f"{cell('canonical', 2025)['最大块止nm']:.0f}",
        "BaEdgeQ": num(cell("Q0", 2025)["最大块下降"]),
        "BaEdgeNoSG": num(cell("Q3", 2025)["最大块下降"]),
        "BaEdgeShareC": f"{100 * cell('canonical', 2025)[edge]:.0f}\\%",
        "BaTrimMax": num(
            max(abs(cell("canonical_trim", y)["宏F1均值"] - cell("canonical", y)["宏F1均值"]) for y in FOLDS)
        ),
        "BaLongMax": f"{100 * max(cell('canonical', y)[far] for y in FOLDS):.0f}\\%",
    }
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items())


# ── 数字宏 \Ex…：四个公开数据集上一维卷积编码器的 S_src − R（24、25、31 号）──────────
def _sheet(x: pd.ExcelFile, key: str) -> pd.DataFrame:
    return x.parse(next(s for s in x.sheet_names if key in s))


def data_ex(logger: eu.Logger) -> dict[str, Any]:
    """24 号主结果与两项敏感性、25 号混合效应、31 号监督对照；正式与否分别判定。

    24 号正式 = compute 表记有 GPU 时长（CUDA）且长格式的种子恰为五个正式种子；
    25 号正式 = 其「苹果来源」表标 formal（苹果 Δ 取自 64 号 Formal 汇总）且所用 24 号结果为正式。
    """
    x = pd.ExcelFile(os.path.join(eu.OUTPUT_DIR, "24_crossdataset_leakage.xlsx"))
    long, folds, comp = _sheet(x, "长格式"), _sheet(x, "折级"), _sheet(x, "compute")
    formal24 = float(comp["gpu_hours"].iloc[0]) > 0 and sorted(long["seed"].unique()) == sorted(SEEDS)
    per_ds = long.groupby("dataset").agg(delta=("delta", "mean"), pos=("delta", lambda v: float((v > 0).mean())))
    fold_pos = folds.assign(pos=folds["delta_mean"] > 0).groupby("dataset")["pos"].agg(["sum", "count"])
    sens: dict[str, float | None] = {}
    for tag in ("cap0", "buf10_indianpines_spatial"):
        p = os.path.join(eu.OUTPUT_DIR, f"24_crossdataset_leakage__{tag}.xlsx")
        if os.path.exists(p):
            b = _sheet(pd.ExcelFile(p), "长格式")
            sens[tag] = float(b[b["dataset"] == "indianpines_spatial"]["delta"].mean())
        else:
            sens[tag] = None
    mixed: dict[str, float] | None = None
    formal25 = False
    for name in ("25_mixed_effects_leakage.xlsx", "25_mixed_effects_leakage__preview.xlsx"):
        p = os.path.join(eu.OUTPUT_DIR, name)
        if not os.path.exists(p):
            continue
        y = pd.ExcelFile(p)
        if not any("苹果来源" in s for s in y.sheet_names):
            continue  # 早先按捆绑对照算的版本没有这张表，不再采用
        me = _sheet(y, "混合效应汇总").iloc[0]
        mixed = {"est": float(me["估计"]), "lo": float(me["CI下"]), "hi": float(me["CI上"])}
        formal25 = formal24 and (_sheet(y, "苹果来源")["来源"] == "formal").all()
        break
    x31 = pd.ExcelFile(os.path.join(eu.OUTPUT_DIR, "31_leakage_decomposition.xlsx"))
    sup_by = _sheet(x31, "数据集汇总").set_index("dataset")["delta_sup_mean"]
    sup_folds = _sheet(x31, "折级")  # 31 号监督对照（同一对比、同五个种子的第二次运行）的逐折值
    out = {
        "per_ds": per_ds,
        "folds": folds,
        "fold_pos": fold_pos,
        "sens": sens,
        "mixed": mixed,
        "sup": sup_by,
        "sup_folds": sup_folds,
        "formal24": formal24,
        "formal25": formal25,
    }
    logger.log(
        f"[EX] 24 号{'正式' if formal24 else '预览'}；25 号{'正式' if formal25 else '预览'}\n"
        + per_ds.round(4).to_string()
        + f"\n  敏感性 {sens}；混合效应 {mixed}"
    )
    return out


def _wrap(formal: bool, x: str) -> str:
    return x if formal else rf"\pend{{{x}}}"


def macros_ex(ex: dict[str, Any]) -> str:
    """正文 3.3 节、摘要与补充材料 S1.6 引用的公开数据集数字。"""
    per_ds, folds, fp = ex["per_ds"], ex["folds"], ex["fold_pos"]
    least = per_ds["pos"].idxmin()
    if least != "corn_instr":
        raise ValueError(f"种子×折为正比例最低的是 {least} 而非玉米，正文 3.3 节「最不一致的是玉米」须改写")
    maize = folds[folds["dataset"] == "corn_instr"]
    w24 = partial(_wrap, ex["formal24"])
    w25 = partial(_wrap, ex["formal25"])
    m = {
        "ExMaize": w24(num(per_ds.loc["corn_instr", "delta"])),
        "ExTablet": w24(num(per_ds.loc["tablet_instr", "delta"])),
        "ExMango": w24(num(per_ds.loc["mango_season", "delta"])),
        "ExIP": w24(num(per_ds.loc["indianpines_spatial", "delta"])),
        "ExMinS": w24(num(float(per_ds["delta"].min()), 2)),
        "ExMaxS": w24(num(float(per_ds["delta"].max()), 2)),
        "ExFoldsPos": w24(f"{int(fp['sum'].sum())} of {int(fp['count'].sum())}"),
        "ExMaizeMin": w24(num(float(maize["delta_mean"].min()))),
        "ExMaizeMax": w24(num(float(maize["delta_mean"].max()))),
        "ExMaizeSD": w24(num(float(maize["delta_std"].max()), 2)),
        "ExSupMaxDiff": w24(num(float(max(abs(per_ds.loc[d, "delta"] - ex["sup"].loc[d]) for d in per_ds.index)))),
    }
    # 同一对比的第二次运行（31 号，与无标签暴露臂一起训练）：摘要与 3.3 节按两次运行报范围
    sup, sup_folds = ex["sup"].reindex(per_ds.index), ex["sup_folds"]
    if not ((per_ds["delta"] > 0).all() and (sup > 0).all()):
        raise ValueError("两次运行中有数据集的对比不为正，摘要「在 4 个公开数据集上重现」须改写")
    others = [d for d in per_ds.index if d != "corn_instr"]
    m["ExSupMaize"] = w24(num(float(sup.loc["corn_instr"])))
    m["ExSupFoldsPos"] = w24(f"{int((sup_folds['delta_sup'] > 0).sum())} of {len(sup_folds)}")
    m["ExSupOtherMaxDiff"] = w24(num(float(max(abs(per_ds.loc[d, "delta"] - sup.loc[d]) for d in others))))
    m["ExBothMinS"] = w24(num(float(min(per_ds["delta"].min(), sup.min())), 2))
    m["ExBothMaxS"] = w24(num(float(max(per_ds["delta"].max(), sup.max())), 2))
    for tag, key in (("cap0", "ExIPCapFree"), ("buf10_indianpines_spatial", "ExIPBuffer")):
        v = ex["sens"][tag]
        m[key] = "--" if v is None else w24(num(v))
    mixed = ex["mixed"]
    for k, sub in (("ExMixed", "est"), ("ExMixedLo", "lo"), ("ExMixedHi", "hi")):
        m[k] = "--" if mixed is None else w25(num(mixed[sub]))
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items())


# ── 表 S14 与数字宏 Mc…：源年份选参与同一批情节上的方法比较 ─────────────────────
def _pow10(v: float) -> str:
    e = round(np.log10(v))
    return {0: "1", 1: "10"}.get(e, f"10^{{{e}}}")


def _svm_choice(param: str, lang: str) -> str:
    val = ast.literal_eval(param)
    if val[0] == "linear":
        return ("线性核" if lang == "zh" else "linear") + rf", $C={_pow10(val[1])}$"
    kern = "RBF 核" if lang == "zh" else "RBF"
    return rf"{kern}, $C={_pow10(val[1])}$, $\gamma_s\times{_pow10(val[2])}$"


def data_s14(logger: eu.Logger) -> pd.DataFrame:
    """71 号：每个 K 下两种基线最常被选中的超参数、其占 15 个种子×折的次数、落在网格外沿的次数与宏 F1。"""
    m71 = importlib.import_module("71_source_only_tuning")
    a = pd.read_excel(os.path.join(eu.OUTPUT_DIR, "71_source_only_tuning.xlsx"), sheet_name=0)
    rows: list[dict[str, Any]] = []
    for k in K_S9:
        row: dict[str, Any] = {"K": k}
        for meth, key, grid in (("SVM-Kshot", "SVM", m71.SVM_GRID), ("PLS-DA-Kshot", "PLS", m71.PLS_GRID)):
            sub = a[(a["K"] == k) & (a["方法"] == meth)]
            vc = sub["所选超参数"].value_counts()
            pos = {str(p): i for i, p in enumerate(grid)}
            top = min(vc.index, key=lambda v: (-int(vc[v]), pos[v]))  # 次数并列时取候选序列里靠前的
            row[f"{key}_选中"] = str(top)
            row[f"{key}_次数"] = int(vc[top])
            row[f"{key}_格数"] = len(sub)
            row[f"{key}_外沿"] = int(sub["所选超参数"].map(m71.on_edge).sum())
            if key == "SVM":
                row["SVM_角"] = int((sub["所选超参数"] == str(("rbf", m71.C_GRID[-1], m71.G_GRID[0]))).sum())
            for col, tag in (("目标宏F1(所选)", "所选"), ("目标宏F1(固定)", "固定")):
                row[f"{key}_{tag}"] = float(sub.groupby("种子")[col].mean().mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    logger.log("[S14]\n" + out.round(4).to_string(index=False))
    return out


def tex_s14(d: pd.DataFrame, lang: str) -> str:
    lines = []
    for _, r in d.iterrows():
        svm = _svm_choice(r["SVM_选中"], lang)
        lv = int(r["PLS_选中"])
        lines.append(
            f"{int(r['K'])} & {svm} & {r['SVM_次数']}/{r['SVM_格数']} & {num(r['SVM_所选'])} & {num(r['SVM_固定'])}"
            f" & {lv} & {r['PLS_次数']}/{r['PLS_格数']} & {num(r['PLS_所选'])} & {num(r['PLS_固定'])} " + r"\\"
        )
    return "\n".join(lines)


def data_mc(logger: eu.Logger) -> dict[str, Any]:
    """同一批单面情节上的逐种子三折均值：72 号的 R、S_src、S_pool 与 71 号源年份选参的 SVM、PLS-DA。"""
    k72 = pd.concat(
        [pd.read_excel(os.path.join(FS, f"72_kscan__seed{s}.xlsx"), sheet_name=0) for s in SEEDS], ignore_index=True
    )
    t71 = pd.read_excel(os.path.join(eu.OUTPUT_DIR, "71_source_only_tuning.xlsx"), sheet_name=0)
    t71 = t71.assign(臂=t71["方法"].map({"SVM-Kshot": "SVM", "PLS-DA-Kshot": "PLS"}), F1均值=t71["目标宏F1(所选)"])
    fixed = t71.assign(臂=t71["臂"] + "_fixed", F1均值=t71["目标宏F1(固定)"])
    cols = ["臂", "种子", "目标年份", "K", "F1均值"]
    long = pd.concat([k72[cols], t71[cols], fixed[cols]], ignore_index=True)
    per_seed = long.groupby(["臂", "K", "种子"])["F1均值"].mean().unstack("臂")
    per_fold = long.groupby(["臂", "K", "目标年份"])["F1均值"].mean().unstack("臂")
    cell = long.groupby(["臂", "K", "种子", "目标年份"])["F1均值"].mean().unstack("臂")
    x72 = pd.ExcelFile(os.path.join(eu.OUTPUT_DIR, "72_kscan_arms.xlsx"))
    tests = x72.parse(next(s for s in x72.sheet_names if "折级检验" in s))
    x42 = pd.ExcelFile(os.path.join(eu.OUTPUT_DIR, "42_ensemble_kshot.xlsx"))
    ens = x42.parse(next(s for s in x42.sheet_names if "三折均值汇总" in s)).set_index("K")
    panel = importlib.import_module("42_ensemble_kshot").load_formal_panel()
    pn20 = panel[panel["方法"] == "VanillaProtoNet"].groupby(["K", "种子"])["F1均值"].mean().groupby("K").mean()
    logger.log("[MC] 逐种子三折均值的种子均值\n" + per_seed.groupby("K").mean().round(4).to_string())
    x73 = pd.ExcelFile(os.path.join(eu.OUTPUT_DIR, "73_grid_edge_check.xlsx"))
    edge = x73.parse(next(s for s in x73.sheet_names if "按方法与K汇总" in s))
    return {
        "per_seed": per_seed,
        "per_fold": per_fold,
        "cell": cell,
        "tests": tests,
        "ens": ens,
        "edge": edge,
        "pn20": pn20,
    }


def macros_mc(mc: dict[str, Any], s14: pd.DataFrame, s9: pd.DataFrame) -> str:
    """正文 2.6、3.4 节，摘要与结论 (2) 的方法比较数字；下面各条断言对应正文的定性表述，不成立即须改写正文。"""
    ps, pf = mc["per_seed"], mc["per_fold"]
    mean = ps.groupby("K").mean()
    best_s = ps[["SVM", "PLS"]].max(axis=1)
    best_f = pf[["SVM", "PLS"]].max(axis=1)
    gap_s = (ps["R"] - best_s).unstack("种子")  # 行 K，列种子
    gap_f = (pf["R"] - best_f).unstack("目标年份")
    pool_gap = mean["S_pool"] - mean[["SVM", "PLS"]].max(axis=1)
    late = [k for k in K_S9 if k >= 10]
    if not (gap_s.loc[1] > 0).all():
        raise ValueError("K=1 时原型网络并非在每个种子都领先")
    tie = max(abs(mean.loc[5, a] - mean.loc[5, b]) for a, b in (("R", "SVM"), ("R", "PLS"), ("SVM", "PLS")))
    if tie > 0.02:
        raise ValueError(f"K=5 时三条路线相差 {tie:.3f}，正文「相差无几」须改写")
    cell = mc["cell"]
    gap_c = cell["R"] - cell[["SVM", "PLS"]].max(axis=1)  # 索引 (K, 种子, 目标年份)
    late_c = gap_c[gap_c.index.get_level_values("K") >= 10]
    if not (gap_s.loc[late] < 0).all().all() or not (gap_f.loc[late] < 0).all().all() or not (late_c < 0).all():
        raise ValueError("K≥10 时原型网络并非在每个种子×折都落后于较强的调参基线")
    if not ((pool_gap.loc[[1, 5, 10, 20]] > 0).all() and pool_gap.loc[50] < 0):
        raise ValueError("合并池编码器的原型网络并非恰好在 K≤20 领先、K=50 落后")
    # 正文 3.4 节：合并池编码器相对编码器 R 的抬升，取表 S9 同一口径（两行三折均值之差），读者可由表核对
    leak = mean["S_pool"] - mean["R"]
    if float(leak.max() - leak.min()) >= 0.05:
        raise ValueError("S_pool−R 随 K 变化超过 0.05，正文「整条曲线平移」须改写")
    # 固定设置下的迁移基线与集成：任何 K 都不超过当时领先的路线；CORAL/TCA 在 K≤10 低于 R、K≥20 低于固定设置 SVM
    s9m = s9.pivot_table(index="K", columns="方法", values="三折均值")
    ens = mc["ens"]
    lead = mean[["R", "SVM", "PLS"]].max(axis=1)
    for k in K_S9:
        others = [float(s9m.loc[k, c]) for c in ("Pooled-SVM", "CORAL-SVM", "TCA-SVM")]
        others += [float(ens.loc[k, c]) for c in ("RandomForest", "Ens-Vote", "Ens-WAvg", "Ens-Stack")]
        if max(others) >= float(lead.loc[k]):
            raise ValueError(f"K={k} 时有迁移基线或集成超过领先路线")
    if not all(s9m.loc[k, c] < s9m.loc[k, "SVM-Kshot"] for k in K_S9 for c in ("Pooled-SVM", "CORAL-SVM", "TCA-SVM")):
        raise ValueError("固定设置下有迁移基线在某个 K 不低于同设置 SVM，正文「源年份进分类器无益」须改写")
    for c in ("CORAL-SVM", "TCA-SVM"):
        if not all(s9m.loc[k, c] < mean.loc[k, "R"] for k in K_S9 if k <= 10):
            raise ValueError(f"{c} 在 K≤10 并非都低于编码器 R 的原型网络")
        if not all(s9m.loc[k, c] < s9m.loc[k, "SVM-Kshot"] for k in K_S9 if k >= 20):
            raise ValueError(f"{c} 在 K≥20 并非都低于固定设置的 SVM")
    tests = mc["tests"]
    tests = tests[tests["对比"].str.startswith("R ")]
    m: dict[str, str] = {}
    for letter, k in zip("ABCDE", K_S9, strict=True):
        for arm, key in (("R", "R"), ("SVM", "Svm"), ("PLS", "Pls"), ("S_pool", "Pool")):
            m[f"Mc{key}{letter}"] = num(float(mean.loc[k, arm]))
    m["McSvmFixE"] = num(float(mean.loc[50, "SVM_fixed"]))
    m["McPlsFixE"] = num(float(mean.loc[50, "PLS_fixed"]))
    # 领先与落后的幅度取三折均值之差（较强者按三折均值定），与图 4(c) 的三折均值曲线同一口径
    gap_m = mean["R"] - mean[["SVM", "PLS"]].max(axis=1)
    m["McLeadA"] = num(float(gap_m.loc[1]))
    m["McTieB"] = num(float(tie))
    for letter, k in (("C", 10), ("D", 20), ("E", 50)):
        m[f"McTrail{letter}"] = num(float(-gap_m.loc[k]))
    m["McTrailN"] = str(int(late_c.size))
    # 按时间先后排列的 LOEO-2025 折单列（K≤5 领先、K≥10 落后）；LOEO-2018 在 K=5 已落后
    if not (all(gap_f.loc[k, 2025] > 0 for k in K_S9 if k <= 5) and all(gap_f.loc[k, 2025] < 0 for k in late)):
        raise ValueError("LOEO-2025 折并非 K≤5 领先、K≥10 落后，正文须改写")
    if not gap_f.loc[5, 2018] < 0:
        raise ValueError("LOEO-2018 折在 K=5 并未落后，正文须改写")
    m["McTfRB"] = num(float(pf.loc[(5, 2025), "R"]))
    m["McTfBestB"] = num(float(best_f.loc[(5, 2025)]))
    m["McTfRC"] = num(float(pf.loc[(10, 2025), "R"]))
    m["McTfBestC"] = num(float(best_f.loc[(10, 2025)]))
    m["McPoolLeadC"] = num(float(pool_gap.loc[10]))
    m["McPoolLeadD"] = num(float(pool_gap.loc[20]))
    m["McPoolTrailE"] = num(float(-pool_gap.loc[50]))
    m["McLeakMin"] = num(float(leak.min()))
    m["McLeakMax"] = num(float(leak.max()))
    m["McPanelDiff"] = num(float((mean["R"] - mc["pn20"].reindex(mean.index)).abs().max()))
    m["McSvmGainE"] = num(float(mean.loc[50, "SVM"] - mean.loc[50, "SVM_fixed"]))
    m["McPlsGainE"] = num(float(mean.loc[50, "PLS"] - mean.loc[50, "PLS_fixed"]))
    m["McSvmEdge"] = str(int(s14["SVM_外沿"].sum()))
    m["McPlsEdge"] = str(int(s14["PLS_外沿"].sum()))
    m["McCells"] = str(int(s14["SVM_格数"].sum()))
    m["McEdgeCorner"] = str(int(s14["SVM_角"].sum()))
    m["McTests"] = str(len(tests))
    m["McSigN"] = str(int((tests["p_holm"] < 0.05).sum()))
    m["McPHolmMin"] = f"{float(tests['p_holm'].min()):.2f}"
    tune_a = max(abs(mean.loc[1, "SVM"] - mean.loc[1, "SVM_fixed"]), abs(mean.loc[1, "PLS"] - mean.loc[1, "PLS_fixed"]))
    if tune_a >= 0.01 or not all(mean.loc[k, a] > mean.loc[k, f"{a}_fixed"] for k in late for a in ("SVM", "PLS")):
        raise ValueError("选参在 K=1 的改变不小于 0.01，或在 K≥10 并非同时提高两种基线")
    m["McTuneA"] = num(float(tune_a))
    edge = mc["edge"]
    m["McEdgeCells"] = str(int(edge["外沿格数"].sum()))
    m["McEdgeSwitch"] = str(int(edge["换用格数"].sum()))
    m["McEdgeMaxDiff"] = num(float(edge["变化"].abs().max()))
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items())


# ── 写回 ─────────────────────────────────────────────────────────────────────
def _fill(body: str, m: re.Match[str]) -> str:
    return m.group(1) + body + "\n" + m.group(3)


def replace_blocks(path: str, blocks: dict[str, str], check: bool, logger: eu.Logger) -> bool:
    """替换 %% <gen:KEY> 与 %% </gen:KEY> 之间的内容；check 模式只比较。

    返回是否一致：文件不存在、或缺少要写的某个生成块，都算不一致（check 模式据此以非零状态退出）。
    """
    if not os.path.exists(path):
        logger.log(f"  {path} 不存在")
        return False
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    new = text
    missing: list[str] = []
    for key, body in blocks.items():
        pat = re.compile(rf"(%% <gen:{key}>\n)(.*?)(%% </gen:{key}>)", re.S)
        if not pat.search(new):
            logger.log(f"  {os.path.basename(path)} 无生成块 {key}，跳过")
            missing.append(key)
            continue
        new = pat.sub(partial(_fill, body), new)
    same = new == text and not missing
    if check:
        logger.log(f"  {os.path.basename(path)}: {'一致' if same else '与结果文件不一致'}")
    elif new != text:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
        logger.log(f"  已写回 {os.path.basename(path)}")
    else:
        logger.log(f"  {os.path.basename(path)} 无变化")
    return same


def main() -> None:
    parser = argparse.ArgumentParser(description="补充材料表 S9–S12 的生成块")
    parser.add_argument("--check", action="store_true", help="只核对，不写回")
    args = parser.parse_args()
    logger = eu.get_logger("68si_tables")

    s9 = data_s9(logger)
    s10a, s10b = data_s10(logger)
    res64, formal = load_64(logger)
    s11 = data_s11(res64, formal, logger)
    t1 = data_t1(res64, logger)
    s12 = data_s12(logger)
    s13 = data_s13(logger)
    ba = macros_ba(s13)
    lk = macros_lk(t1, s11, s12, formal)
    ex = data_ex(logger)
    exm = macros_ex(ex)
    s14 = data_s14(logger)
    mc = data_mc(logger)
    mcm = macros_mc(mc, s14, s9)

    ok = True
    for lang, path in TEX.items():
        blocks = {
            "S9": tex_s9(s9, lang),
            "S10A": tex_s10a(s10a, lang),
            "S10B": tex_s10b(s10b, lang),
            "S11": tex_s11(s11, formal, lang),
            "S12": tex_s12(s12, lang),
            "S13": tex_s13(s13, lang),
            "S14": tex_s14(s14, lang),
            "S15": tex_t1(t1, s11, formal, lang, "three_year"),
            "LK": lk,
            "BA": ba,
            "EX": exm,
            "MC": mcm,
        }
        ok &= replace_blocks(path, blocks, args.check, logger)
    for lang, path in MAIN.items():
        blocks = {
            "T1": tex_t1(t1, s11, formal, lang, "two_year"),
            "LK": lk,
            "BA": ba,
            "EX": exm,
            "MC": mcm,
        }
        ok &= replace_blocks(path, blocks, args.check, logger)

    if not args.check:
        out = eu.write_script_workbook(
            __file__,
            {
                eu.table_sheet_name(1, "S9路线与信息"): s9,
                eu.table_sheet_name(2, "S10a质控与域距离"): s10a,
                eu.table_sheet_name(3, "S10b光谱空间路线"): s10b,
                eu.table_sheet_name(4, "S11编码器臂" + ("" if formal else "_预览")): s11,
                eu.table_sheet_name(5, "S12捆绑对照"): s12,
                eu.table_sheet_name(6, "表1各臂" + ("" if formal else "_预览")): t1,
                eu.table_sheet_name(7, "S13逐年波段归因"): s13,
                eu.table_sheet_name(8, "公开数据集" + ("" if ex["formal24"] else "_预览")): ex["per_ds"].reset_index(),
                eu.table_sheet_name(9, "S14源年份选参"): s14,
                eu.table_sheet_name(10, "方法比较逐种子"): mc["per_seed"].reset_index(),
            },
        )
        logger.log(f"已写 {out}")
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
