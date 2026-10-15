"""77_generality_tables.py: 正文表 2、补充材料表 S16–S18 与数字宏 \\Me…，全部从 75 号汇总工作簿读取，写回生成块。

    正文表 2   各数据集 × 编码器族的 S_src−R（该数据集各留出域的种子均值再取均值；苹果取两年设计），
               以及各族跨数据集推断（混合效应截距、95% 区间、为正留出域数）
    表 S16     逐留出域 × 编码器族的 S_src−R 与 S_half−R^h（主口径 18 个留出域，另列苹果三年设计 3 个）
    表 S17     方法选择：逐留出域逐 K 的无泄漏第一与泄漏第一，翻转加粗
    表 S18     逐数据集逐 K 的超参数泄漏、编码器泄漏与最佳值抬升（留出域均值）
    表 S19     苹果两种设计上在目标年选参相对源年份选参的抬升（76 号，K = 1–50；76 号为纯 CPU 正式结果，
               五个正式种子齐全且自检复现 71 号时不包 \\pend{}）
    数字宏 \\Me…  正文 3.2–3.4 节、摘要与结论；中英文稿共用同一份宏
    定性断言   check_claims 逐条核对正文的定性表述；正式结果下任何一条不成立或无法核对即报错，须改写正文

口径与 75 号相同：对照在种子内配对作差、再对种子求均值；跨数据集推断取 75 号「表c」主口径，
表 S16 末段另列 75 号的两个敏感性口径（去掉玉米与药片、去掉 Indian Pines）。苹果 CNN 的检查点单独暴露臂
R_ckpt 的对照读 64 号汇总（04outputs/64_leakage_controls.xlsx），只用于定性断言；其数字宏由 68 号写出。
表 2 与表 S16 中数据集方差估计为 0 的混合效应拟合加上标 a（阈值见 ZERO_VAR），两表表注与正文 2.9 节说明其含义；
正式结果下两表若已没有这类拟合即报错，须同时删去表注与 2.9 节的说明。
75 号工作簿不是正式结果（--stage pilot，或正式工作簿「缺表」非空）时，数值一律包 \\pend{}，表示预览、非投稿数字。
宏名全部预先声明，缺数据的给「--」，稿件因此在任何阶段都能编译。

运行方式:
    cd <项目根>
    python 02code/77_generality_tables.py                 # 读正式汇总，写回生成块
    python 02code/77_generality_tables.py --stage pilot   # 读 pilot 汇总（数值包 \\pend{}）
    python 02code/77_generality_tables.py --check         # 只核对，生成块与结果文件不一致时退出码 1

输出文件:
    06doc/01manuscript/Elsevier_en.tex、Elsevier_zh.tex — 生成块 T2（表 2 表体）与 ME（数字宏）
    06doc/01manuscript/Supplementary_en.tex、Supplementary_zh.tex — 生成块 S16–S19 与 ME
    04outputs/77_generality_tables.xlsx — 各表数值
    05logs/77_generality_tables_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Any

import numpy as np
import pandas as pd

import export_utils as eu

m68: Any = importlib.import_module("68si_tables")
num = m68.num

FAMILIES = ["cnn", "mlp", "resnet", "transformer", "pls", "pca"]
FAM_KEY = {"cnn": "Cnn", "mlp": "Mlp", "resnet": "Res", "transformer": "Tfm", "pls": "Pls", "pca": "Pca"}
FAM_TEX = {"cnn": "CNN", "mlp": "MLP", "resnet": "ResNet", "transformer": "Transformer", "pls": "PLS", "pca": "PCA"}
MAIN_SETS = [
    "apple_two_year",
    "corn_instr",
    "tablet_instr",
    "mango_season",
    "kiwifruit_season",
    "indianpines_spatial",
]
DS_TEX = {
    "en": {
        "apple_two_year": "Apple, two years",
        "apple_three_year": "Apple, three years",
        "corn_instr": "Maize",
        "tablet_instr": "Tablets",
        "mango_season": "Mango",
        "kiwifruit_season": "Kiwifruit",
        "indianpines_spatial": "Indian Pines",
    },
    "zh": {
        "apple_two_year": "苹果，两年设计",
        "apple_three_year": "苹果，三年设计",
        "corn_instr": "玉米",
        "tablet_instr": "药片",
        "mango_season": "芒果",
        "kiwifruit_season": "猕猴桃",
        "indianpines_spatial": "Indian Pines",
    },
}
ROW_TEX = {
    "en": {
        "mixed": "Mixed-effects mean",
        "lo": "95\\% interval, lower",
        "hi": "95\\% interval, upper",
        "pos": "Positive domains",
    },
    "zh": {"mixed": "混合效应均值", "lo": "95\\% 区间下限", "hi": "95\\% 区间上限", "pos": "为正的留出域"},
}
MAIN_SCOPE = "主口径（苹果两年设计）"
# 敏感性口径：宏名前缀 → 75 号「表c」的口径名
SENS_SCOPES = {"Nc": "敏感性：主口径去掉玉米与药片", "Ni": "敏感性：主口径去掉 Indian Pines"}
SENS_TEX = {
    "en": {
        "head": r"S\textsubscript{src}$-$R in two sensitivity analyses",
        "Nc": "Without maize and tablets",
        "Ni": "Without Indian Pines",
        "mixed": "Mixed-effects mean",
        "lo": "95\\% lower bound",
        "hi": "95\\% upper bound",
    },
    "zh": {
        "head": r"两项敏感性分析中的 S\textsubscript{src}$-$R",
        "Nc": "去掉玉米与药片",
        "Ni": "去掉 Indian Pines",
        "mixed": "混合效应均值",
        "lo": "95\\% 下限",
        "hi": "95\\% 上限",
    },
}
# 数据集方差估计为 0 的混合效应拟合（< 1e-6，即数据集间标准差 < 0.001 宏 F1）：此时均值与 95% 区间就是逐域普通均值
# 及其正态近似区间。statsmodels 的「boundary」告警按方差绝对值 < 0.01 触发，对宏 F1 的对照几乎每次都报，不能用作标记依据；
# 表 2 与表 S16 按下面的阈值给这类拟合加上标 a，正文 2.9 节与两表的表注说明其含义。
ZERO_VAR = 1e-6
ZMARK = r"\textsuperscript{a}"
CKEY = {"Δ_rep": "Rep", "Δ_half": "Half", "Δ_lab": "Lab", "Δ_shuf": "Shuf", "Δ_seen": "Seen"}
INFER = ("Δ_rep", "Δ_half", "Δ_lab")
K_KEY = {1: "A", 5: "B", 10: "C", 20: "D"}
K76 = {1: "A", 5: "B", 10: "C", 20: "D", 50: "E"}
DESIGN_KEY = {"two_year": "Tw", "three_year": "Th"}
METH_KEY = {"SVM-Kshot": "Svm", "PLS-DA-Kshot": "Pls"}
SEEDS = [int(x) for x in eu.RECOMMENDED_SEEDS]
ROUTE_KEY = {
    "原型网络(R)": "PnR",
    "原型网络(S_src)": "PnS",
    "SVM(源域选参)": "SvmS",
    "SVM(留出域选参)": "SvmT",
    "PLS-DA(源域选参)": "PlsS",
    "PLS-DA(留出域选参)": "PlsT",
    "最近质心": "Nc",
}
ROUTE_ABBR = {
    "原型网络(R)": "PN",
    "原型网络(S_src)": "PN",
    "SVM(源域选参)": "SVM",
    "SVM(留出域选参)": "SVM",
    "PLS-DA(源域选参)": "PLS",
    "PLS-DA(留出域选参)": "PLS",
    "最近质心": "NC",
}


def _held_tex(ds: str, held: str) -> str:
    if ds.startswith("apple"):
        return f"LOEO-{held[1:]}"
    if ds == "corn_instr":
        return held.replace("spec", "")
    if ds == "tablet_instr":
        return f"Instrument {held[-1]}"
    if ds == "mango_season":
        return f"Season {held[1:]}"
    if ds == "kiwifruit_season":
        return held[1:]
    return held


def _pos_text(n: int, total: int, lang: str) -> str:
    return f"{n}/{total}" if lang == "zh" else f"{n} of {total}"


class Wrap:
    """预览时把数值包进 \\pend{}。"""

    def __init__(self, formal: bool) -> None:
        self.formal = formal

    def __call__(self, x: str) -> str:
        return x if self.formal or x == "--" else rf"\pend{{{x}}}"


def load(stage: str, logger: eu.Logger) -> tuple[dict[str, pd.DataFrame], bool]:
    """读 75 号汇总工作簿的六张表；返回 (表, 是否正式)。"""
    name = "75_leakage_generality_summary.xlsx" if stage == "formal" else "75_leakage_generality_summary__pilot.xlsx"
    path = os.path.join(eu.OUTPUT_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺 75 号汇总：{path}")
    xl = pd.ExcelFile(path)
    keys = ["a", "b", "c", "d", "e", "f"]
    tabs = {k: xl.parse(sh) for k, sh in zip(keys, xl.sheet_names, strict=False)}
    missing = tabs["f"].dropna(how="all") if "f" in tabs else pd.DataFrame()
    formal = stage == "formal" and missing.empty
    logger.log(f"读 {name}：{'正式' if formal else '预览'}（缺表 {len(missing)} 个）")
    return tabs, formal


# ── 表 2 与 S16 ──────────────────────────────────────────────────────────────
def ds_family_mean(a: pd.DataFrame, contrast: str) -> pd.DataFrame:
    """数据集 × 编码器族的对照均值（留出域均值）。"""
    if a.empty or contrast not in a:
        return pd.DataFrame()
    return a.groupby(["dataset", "编码器"])[contrast].mean().unstack("编码器")


def _infer_row(c: pd.DataFrame, fam: str, contrast: str, scope: str = MAIN_SCOPE) -> pd.Series | None:
    if c.empty:
        return None
    m = c[(c["口径"] == scope) & (c["编码器"] == fam) & (c["对照"] == contrast)]
    return m.iloc[0] if len(m) == 1 else None


def _zmark(r: pd.Series | None) -> str:
    """数据集方差估计为 0 的拟合返回上标 a，否则返回空串。"""
    return ZMARK if r is not None and float(r.get("数据集方差", np.nan)) < ZERO_VAR else ""


def tex_t2(a: pd.DataFrame, c: pd.DataFrame, w: Wrap, lang: str) -> str:
    means = ds_family_mean(a, "Δ_rep")
    n_held = a[a["编码器"] == "cnn"].groupby("dataset")["held"].nunique() if not a.empty else pd.Series(dtype=int)
    lines: list[str] = []
    for ds in MAIN_SETS:
        cells = []
        for fam in FAMILIES:
            v = means.loc[ds, fam] if (not means.empty and ds in means.index and fam in means) else np.nan
            cells.append(w(num(float(v))) if pd.notna(v) else w("--"))
        nh = int(n_held.get(ds, 0))
        lines.append(rf"{DS_TEX[lang][ds]} & {nh if nh else '--'} & {' & '.join(cells)} \\")
    lines.append(r"\midrule")
    for key, col in (("mixed", "混合效应截距"), ("lo", "截距95%下限"), ("hi", "截距95%上限")):
        cells = []
        for fam in FAMILIES:
            r = _infer_row(c, fam, "Δ_rep")
            v = float(r[col]) if r is not None and pd.notna(r.get(col, np.nan)) else np.nan
            mark = _zmark(r) if key == "mixed" else ""
            cells.append(w(num(v)) + mark if np.isfinite(v) else w("--"))
        lines.append(rf"{ROW_TEX[lang][key]} & & {' & '.join(cells)} \\")
    cells = []
    for fam in FAMILIES:
        r = _infer_row(c, fam, "Δ_rep")
        cells.append(w(str(r["为正留出域"])) if r is not None else w("--"))
    lines.append(rf"{ROW_TEX[lang]['pos']} & & {' & '.join(cells)} \\")
    return "\n".join(lines)


CON_TEX = {
    "Δ_rep": r"S\textsubscript{src}$-$R",
    "Δ_lab": r"S\textsubscript{src}$-$S\textsubscript{shuf}",
    "Δ_shuf": r"S\textsubscript{shuf}$-$R",
    "Δ_half": r"S\textsubscript{half}$-$R\textsuperscript{h}",
    "Δ_seen": r"S\textsubscript{src}\textsuperscript{h}$-$S\textsubscript{half}",
}
SUM_TEX = {
    "en": {
        "head": "Summary over the 18 held-out domains of the main analysis",
        "mixed": "Mixed-effects mean",
        "mean": "Mean",
        "pos": "Positive domains",
    },
    "zh": {"head": "主分析 18 个留出域的汇总", "mixed": "混合效应均值", "mean": "均值", "pos": "为正的留出域"},
}


def family_stat(main_a: pd.DataFrame, c: pd.DataFrame, fam: str, contrast: str) -> tuple[float, str]:
    """主口径某族某对照的汇总值：Δ_rep、Δ_half、Δ_lab 取混合效应截距，Δ_shuf、Δ_seen 取留出域均值。

    返回 (值, 为正留出域「n/总数」)；无数据时值为 nan、计数为空串。
    """
    sub = main_a[main_a["编码器"] == fam] if not main_a.empty else main_a
    v = sub[contrast].dropna() if (not sub.empty and contrast in sub) else pd.Series(dtype=float)
    pos = f"{int((v > 0).sum())}/{len(v)}" if not v.empty else ""
    if contrast in INFER:
        r = _infer_row(c, fam, contrast)
        b = float(r.get("混合效应截距", np.nan)) if r is not None else np.nan
        return b, pos
    return (float(v.mean()) if not v.empty else np.nan), pos


def tex_s16(a: pd.DataFrame, c: pd.DataFrame, w: Wrap, lang: str) -> str:
    """逐留出域：上半 S_src−R，下半 S_half−R^h（每半先主口径 18 个留出域，再苹果三年设计）；末段为主口径各族汇总。"""
    head = {"Δ_rep": CON_TEX["Δ_rep"], "Δ_half": CON_TEX["Δ_half"]}
    lines: list[str] = []
    for contrast in ("Δ_rep", "Δ_half"):
        if lines:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{{head[contrast]}}} \\")
        for ds in [*MAIN_SETS, "apple_three_year"]:
            sub = a[a["dataset"] == ds] if not a.empty else a
            if sub.empty:
                continue
            for held in sorted(sub["held"].unique(), key=str):
                cells = []
                for fam in FAMILIES:
                    m = sub[(sub["held"] == held) & (sub["编码器"] == fam)]
                    ok = len(m) == 1 and contrast in m and pd.notna(m[contrast].iloc[0])
                    v = float(m[contrast].iloc[0]) if ok else np.nan
                    cells.append(w(num(v)) if np.isfinite(v) else "--")
                lines.append(rf"\quad {DS_TEX[lang][ds]} & {_held_tex(ds, str(held))} & {' & '.join(cells)} \\")
    main_a = a[a["dataset"].isin(MAIN_SETS)] if not a.empty else a
    lines += [r"\addlinespace", rf"\multicolumn{{8}}{{@{{}}l}}{{{SUM_TEX[lang]['head']}}} \\"]
    for contrast in ("Δ_rep", "Δ_lab", "Δ_shuf", "Δ_half", "Δ_seen"):
        stats = [family_stat(main_a, c, fam, contrast) for fam in FAMILIES]
        kind = SUM_TEX[lang]["mixed" if contrast in INFER else "mean"]
        marks = [_zmark(_infer_row(c, fam, contrast)) if contrast in INFER else "" for fam in FAMILIES]
        vals = [w(num(v)) + m if np.isfinite(v) else "--" for (v, _), m in zip(stats, marks, strict=True)]
        pos = [w(pc.replace("/", " of ") if lang == "en" else pc) if pc else "--" for _, pc in stats]
        lines.append(rf"\quad {CON_TEX[contrast]} & {kind} & {' & '.join(vals)} \\")
        lines.append(rf" & {SUM_TEX[lang]['pos']} & {' & '.join(pos)} \\")
    st = SENS_TEX[lang]
    lines += [r"\addlinespace", rf"\multicolumn{{8}}{{@{{}}l}}{{{st['head']}}} \\"]
    for sk, scope in SENS_SCOPES.items():
        rows = [_infer_row(c, fam, "Δ_rep", scope) for fam in FAMILIES]
        for kind, col in (("mixed", "混合效应截距"), ("lo", "截距95%下限"), ("hi", "截距95%上限")):
            cells = [w(num(float(r[col]))) + (_zmark(r) if kind == "mixed" else "") if r is not None else "--" for r in rows]
            label = rf"\quad {st[sk]}" if kind == "mixed" else ""
            lines.append(rf"{label} & {st[kind]} & {' & '.join(cells)} \\")
    return "\n".join(lines)


# ── 表 S17、S18 ──────────────────────────────────────────────────────────────
def tex_s17(d: pd.DataFrame, w: Wrap, lang: str) -> str:
    """逐留出域逐 K：无泄漏第一→泄漏第一，翻转加粗。"""
    lines: list[str] = []
    if d.empty:
        return r"\multicolumn{6}{@{}l}{--} \\"
    for ds in MAIN_SETS:
        sub = d[d["dataset"] == ds]
        for held in sorted(sub["held"].unique(), key=str):
            cells = []
            for k in K_KEY:
                m = sub[(sub["held"] == held) & (sub["K"] == k)]
                if len(m) != 1:
                    cells.append("--")
                    continue
                r = m.iloc[0]
                txt = rf"{ROUTE_ABBR[str(r['无泄漏第一'])]}$\to${ROUTE_ABBR[str(r['泄漏第一'])]}"
                cells.append(w(rf"\textbf{{{txt}}}" if bool(r["翻转"]) else txt))
            lines.append(rf"{DS_TEX[lang][ds]} & {_held_tex(ds, str(held))} & {' & '.join(cells)} \\")
    return "\n".join(lines)


def data_s18(d: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if d.empty:
        return pd.DataFrame(rows)
    for ds in MAIN_SETS:
        for k in K_KEY:
            s = d[(d["dataset"] == ds) & (d["K"] == k)]
            if s.empty:
                continue
            rows.append(
                {
                    "dataset": ds,
                    "K": k,
                    "留出域数": len(s),
                    "SVM超参数泄漏": float(s["SVM超参数泄漏"].mean()),
                    "PLS-DA超参数泄漏": float(s["PLS-DA超参数泄漏"].mean()),
                    "编码器泄漏": float(s["编码器泄漏"].mean()),
                    "最佳值抬升": float(s["最佳值抬升"].mean()),
                    "翻转数": int(s["翻转"].sum()),
                }
            )
    return pd.DataFrame(rows)


def tex_s18(s18: pd.DataFrame, w: Wrap, lang: str) -> str:
    if s18.empty:
        return r"\multicolumn{7}{@{}l}{--} \\"
    lines: list[str] = []
    for ds in MAIN_SETS:
        sub = s18[s18["dataset"] == ds]
        for i, (_, r) in enumerate(sub.iterrows()):
            name = DS_TEX[lang][ds] if i == 0 else ""
            vals = [w(num(float(r[c]))) for c in ("SVM超参数泄漏", "PLS-DA超参数泄漏", "编码器泄漏", "最佳值抬升")]
            flips = w(f"{int(r['翻转数'])}/{int(r['留出域数'])}")
            lines.append(rf"{name} & {int(r['K'])} & {' & '.join(vals)} & {flips} \\")
    return "\n".join(lines)


def data_s19(logger: eu.Logger) -> tuple[pd.DataFrame, bool]:
    """76 号苹果两种设计的超参数泄漏：由逐种子逐折表重算（先对种子求各折均值，再对种子求均值），不用已取舍的汇总。

    正式 = 五个正式种子齐全，且三年设计的源年份选参逐格复现 71 号（76 号自检表）。
    """
    path = os.path.join(eu.OUTPUT_DIR, "76_target_tuning_leakage.xlsx")
    if not os.path.exists(path):
        logger.log("  缺 76 号汇总，表 S19 与 \\MeTw…、\\MeTh… 宏给「--」")
        return pd.DataFrame(), False
    xl = pd.ExcelFile(path)
    long = xl.parse(next(n for n in xl.sheet_names if "逐种子" in n))
    chk = xl.parse(next(n for n in xl.sheet_names if "自检" in n)).iloc[0]
    seeds_ok = sorted(int(x) for x in long["种子"].unique()) == sorted(SEEDS)
    self_ok = int(chk["所选超参数一致"]) == int(chk["比对格数"]) and float(chk["目标宏F1最大差"]) <= 5e-4
    rows: list[dict[str, Any]] = []
    for (design, meth, k), sub in long.groupby(["设计", "方法", "K"]):
        per_seed = sub.groupby("种子")[["目标宏F1(源年份选参)", "目标宏F1(目标年选参)"]].mean()
        leak = per_seed["目标宏F1(目标年选参)"] - per_seed["目标宏F1(源年份选参)"]
        rows.append(
            {
                "设计": str(design),
                "方法": str(meth),
                "K": int(k),
                "源年份选参": float(per_seed["目标宏F1(源年份选参)"].mean()),
                "目标年选参": float(per_seed["目标宏F1(目标年选参)"].mean()),
                "超参数泄漏": float(leak.mean()),
                "为正种子数": int((leak > 0).sum()),
                "种子数": len(leak),
            }
        )
    logger.log(f"  76 号：种子{'齐全' if seeds_ok else '不全'}，自检{'通过' if self_ok else '未通过'}")
    return pd.DataFrame(rows), seeds_ok and self_ok


def tex_s19(s19: pd.DataFrame, w: Wrap, lang: str) -> str:
    if s19.empty:
        return r"\multicolumn{7}{@{}l}{--} \\"
    design_tex = {
        "en": {"two_year": "Two-year", "three_year": "Three-year"},
        "zh": {"two_year": "两年设计", "three_year": "三年设计"},
    }
    meth_tex = {"SVM-Kshot": "SVM", "PLS-DA-Kshot": "PLS-DA"}
    lines: list[str] = []
    for design in ("two_year", "three_year"):
        for meth in ("SVM-Kshot", "PLS-DA-Kshot"):
            sub = s19[(s19["设计"] == design) & (s19["方法"] == meth)].sort_values("K")
            for i, (_, r) in enumerate(sub.iterrows()):
                d_txt = design_tex[lang][design] if i == 0 and meth == "SVM-Kshot" else ""
                m_txt = meth_tex[meth] if i == 0 else ""
                vals = [w(num(float(r[c]))) for c in ("源年份选参", "目标年选参", "超参数泄漏")]
                pos = w(f"{int(r['为正种子数'])}/{int(r['种子数'])}")
                lines.append(rf"{d_txt} & {m_txt} & {int(r['K'])} & {' & '.join(vals)} & {pos} \\")
            if not (design == "three_year" and meth == "PLS-DA-Kshot"):
                lines.append(r"\addlinespace")
    return "\n".join(lines)


# ── 数字宏 ───────────────────────────────────────────────────────────────────
def macros_me(t: dict[str, pd.DataFrame], formal: bool, s19: pd.DataFrame, formal76: bool) -> tuple[str, pd.DataFrame]:
    """正文引用的数字宏。每个宏先声明为「--」，有数据再覆盖；返回 (宏块, 宏值表)。"""
    w = Wrap(formal)
    a, c, d, e = (t.get(k, pd.DataFrame()) for k in "acde")
    vals: dict[str, str] = {}
    main_a = a[a["dataset"].isin(MAIN_SETS)] if not a.empty else a

    for fam in FAMILIES:
        fk = FAM_KEY[fam]
        for contrast, ck in CKEY.items():
            base = f"Me{ck}{fk}"
            for suffix in ("", "Lo", "Hi", "BLo", "BHi", "Pos", "Ds", "Min", "Max"):
                vals[base + suffix] = "--"
            sub = main_a[main_a["编码器"] == fam] if not main_a.empty else main_a
            v = sub[contrast].dropna() if (not sub.empty and contrast in sub) else pd.Series(dtype=float)
            if not v.empty:
                vals[base + "Min"] = w(num(float(v.min())))
                vals[base + "Max"] = w(num(float(v.max())))
                vals[base + "Pos"] = w(f"{int((v > 0).sum())} of {len(v)}")
                if contrast not in INFER:
                    vals[base] = w(num(float(v.mean())))
            if contrast in INFER:
                r = _infer_row(c, fam, contrast)
                if r is not None:
                    b = float(r.get("混合效应截距", np.nan))
                    if np.isfinite(b):
                        vals[base] = w(num(b))
                        vals[base + "Lo"] = w(num(float(r["截距95%下限"])))
                        vals[base + "Hi"] = w(num(float(r["截距95%上限"])))
                    else:
                        vals[base] = w(num(float(r["折级均值"])))
                    vals[base + "BLo"] = w(num(float(r["簇自助95%下限"])))
                    vals[base + "BHi"] = w(num(float(r["簇自助95%上限"])))
                    pos_d = str(r["为正数据集"]).split("/")
                    vals[base + "Ds"] = w(f"{pos_d[0]} of {pos_d[1]}")
        # 苹果两年设计：两折均值
        ap = a[(a["dataset"] == "apple_two_year") & (a["编码器"] == fam)] if not a.empty else a
        for contrast, ck in CKEY.items():
            key = f"MeAp{ck}{fk}"
            vals[key] = "--"
            if not ap.empty and contrast in ap and ap[contrast].notna().all() and len(ap) == 2:
                vals[key] = w(num(float(ap[contrast].mean())))

    # 置乱标签：四个深度族中 S_shuf−R 为正的留出域数的最大值（3.3 节「任一深度族为正的留出域至多 … 个」）
    vals["MeShufDeepPosMax"] = "--"
    pos_deep: list[tuple[int, int]] = []
    for fam in DEEP:
        sub = main_a[main_a["编码器"] == fam] if not main_a.empty else main_a
        v = sub["Δ_shuf"].dropna() if (not sub.empty and "Δ_shuf" in sub) else pd.Series(dtype=float)
        if not v.empty:
            pos_deep.append((int((v > 0).sum()), len(v)))
    if len(pos_deep) == len(DEEP) and len({n for _, n in pos_deep}) == 1:
        vals["MeShufDeepPosMax"] = w(f"{max(k for k, _ in pos_deep)} of {pos_deep[0][1]}")

    # B3：主口径逐 K
    for k, kk in K_KEY.items():
        for suffix in ("Flip", "Lift", "HpSvm", "HpSvmPos", "HpPls", "HpPlsPos", "Enc", "EncPos"):
            vals[f"Me{suffix}{kk}"] = "--"
        s = e[(e["口径"] == MAIN_SCOPE) & (e["K"] == k)] if not e.empty else e
        if len(s) == 1:
            r = s.iloc[0]
            n = int(r["留出域数"])
            vals[f"MeFlip{kk}"] = w(f"{int(r['翻转数'])} of {n}")
            vals[f"MeLift{kk}"] = w(num(float(r["最佳值抬升均值"])))
            vals[f"MeHpSvm{kk}"] = w(num(float(r["SVM超参数泄漏均值"])))
            vals[f"MeHpSvmPos{kk}"] = w(str(r["SVM超参数泄漏为正"]).replace("/", " of "))
            vals[f"MeHpPls{kk}"] = w(num(float(r["PLS-DA超参数泄漏均值"])))
            vals[f"MeHpPlsPos{kk}"] = w(str(r["PLS-DA超参数泄漏为正"]).replace("/", " of "))
            vals[f"MeEnc{kk}"] = w(num(float(r["编码器泄漏均值"])))
            vals[f"MeEncPos{kk}"] = w(str(r["编码器泄漏为正"]).replace("/", " of "))
    vals["MeFlipAll"] = "--"
    main_e = e[e["口径"] == MAIN_SCOPE] if not e.empty else e
    if not main_e.empty:
        vals["MeFlipAll"] = w(f"{int(main_e['翻转数'].sum())} of {int(main_e['留出域数'].sum())}")
    # 翻转的来源：暴露面板排第一的是暴露编码器的原型网络，还是留出域选参的 SVM / PLS-DA
    vals["MeFlipEnc"] = vals["MeFlipHp"] = vals["MeFlipIp"] = "--"
    if len(main_e) == len(K_KEY):
        vals["MeFlipEnc"] = w(str(int(main_e["翻转后第一为原型网络(S_src)"].sum())))
        vals["MeFlipHp"] = w(str(int(main_e["翻转后第一为留出域选参"].sum())))
    if not d.empty:
        ip = d[(d["dataset"] == "indianpines_spatial") & d["翻转"].astype(bool)]
        vals["MeFlipIp"] = w(str(len(ip)))

    # 两项敏感性分析：S_src−R 的混合效应推断与逐 K 翻转
    for sk, scope in SENS_SCOPES.items():
        vals[f"Me{sk}N"] = vals[f"Me{sk}Sets"] = vals[f"Me{sk}FlipAll"] = "--"
        for fam in FAMILIES:
            base = f"Me{sk}Rep{FAM_KEY[fam]}"
            for suffix in ("", "Lo", "Hi", "BLo", "BHi", "Ds"):
                vals[base + suffix] = "--"
            r = _infer_row(c, fam, "Δ_rep", scope)
            if r is None:
                continue
            vals[f"Me{sk}N"] = w(str(int(r["留出域数"])))
            vals[f"Me{sk}Sets"] = w(str(int(r["数据集数"])))
            b = float(r.get("混合效应截距", np.nan))
            if np.isfinite(b):
                vals[base] = w(num(b))
                vals[base + "Lo"] = w(num(float(r["截距95%下限"])))
                vals[base + "Hi"] = w(num(float(r["截距95%上限"])))
            vals[base + "BLo"] = w(num(float(r["簇自助95%下限"])))
            vals[base + "BHi"] = w(num(float(r["簇自助95%上限"])))
            pos_d = str(r["为正数据集"]).split("/")
            vals[base + "Ds"] = w(f"{pos_d[0]} of {pos_d[1]}")
        se = e[e["口径"] == scope] if not e.empty else e
        if len(se) == len(K_KEY):
            vals[f"Me{sk}FlipAll"] = w(f"{int(se['翻转数'].sum())} of {int(se['留出域数'].sum())}")

    # 苹果两年设计的路线（两折均值）
    ap_d = d[d["dataset"] == "apple_two_year"] if not d.empty else d
    for k, kk in K_KEY.items():
        for route, rk in ROUTE_KEY.items():
            key = f"MeAp{rk}{kk}"
            vals[key] = "--"
            s = ap_d[ap_d["K"] == k] if not ap_d.empty else ap_d
            if len(s) == 2 and route in s:
                vals[key] = w(num(float(s[route].mean())))

    # 苹果两种设计的超参数泄漏（76 号，K = 1, 5, 10, 20, 50）
    w76 = Wrap(formal76)
    for design, dk in DESIGN_KEY.items():
        for meth, mk in METH_KEY.items():
            for k, kk in K76.items():
                key = f"Me{dk}Hp{mk}{kk}"
                vals[key] = "--"
                vals[key + "Pos"] = "--"
                if s19.empty:
                    continue
                r = s19[(s19["设计"] == design) & (s19["方法"] == meth) & (s19["K"] == k)]
                if len(r) == 1:
                    vals[key] = w76(num(float(r["超参数泄漏"].iloc[0])))
                    vals[key + "Pos"] = w76(f"{int(r['为正种子数'].iloc[0])} of {int(r['种子数'].iloc[0])}")

    vals["MeApHpSvmMax"] = "--"
    vals["MeTwHpPlsLateMin"] = "--"
    vals["MeTwHpPlsLateMax"] = "--"
    if not s19.empty:
        svm = s19[s19["方法"] == "SVM-Kshot"]["超参数泄漏"]
        if not svm.empty:
            vals["MeApHpSvmMax"] = w76(num(float(svm.max())))
        late = s19[(s19["设计"] == "two_year") & (s19["方法"] == "PLS-DA-Kshot") & (s19["K"] >= 10)]["超参数泄漏"]
        if not late.empty:
            vals["MeTwHpPlsLateMin"] = w76(num(float(late.min())))
            vals["MeTwHpPlsLateMax"] = w76(num(float(late.max())))

    block = "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in vals.items())
    return block, pd.DataFrame({"宏": list(vals), "值": list(vals.values())})


DEEP = ["cnn", "mlp", "resnet", "transformer"]
LABEL_FAMS = ["cnn", "mlp", "resnet", "transformer", "pls"]


def _lower(c: pd.DataFrame, fam: str, contrast: str, scope: str = MAIN_SCOPE) -> float:
    """混合效应截距的 95% 下限（默认主口径）；拟合不可用时取簇自助下限。"""
    r = _infer_row(c, fam, contrast, scope)
    if r is None:
        return np.nan
    lo = float(r.get("截距95%下限", np.nan))
    return lo if np.isfinite(lo) else float(r.get("簇自助95%下限", np.nan))


def _upper(c: pd.DataFrame, fam: str, contrast: str, scope: str = MAIN_SCOPE) -> float:
    """混合效应截距的 95% 上限（默认主口径）；拟合不可用时取簇自助上限。"""
    r = _infer_row(c, fam, contrast, scope)
    if r is None:
        return np.nan
    hi = float(r.get("截距95%上限", np.nan))
    return hi if np.isfinite(hi) else float(r.get("簇自助95%上限", np.nan))


def ckpt_cells() -> dict[str, np.ndarray]:
    """苹果 CNN 的 R_ckpt−R：64 号正式汇总「逐种子结果」里每个（折, 种子）格的差值，按设计分组。"""
    path = os.path.join(eu.OUTPUT_DIR, "64_leakage_controls.xlsx")
    out: dict[str, np.ndarray] = {}
    if not os.path.exists(path):
        return out
    t = pd.read_excel(path, sheet_name=1)
    t = t[(t["src_alpha"] == 0) & (t["变体"] == "canonical") & t["臂"].isin(["R", "R_ckpt"])]
    for design, sub in t.groupby("设计"):
        p = sub.pivot_table(index=["目标年份", "种子"], columns="臂", values="F1均值")
        if {"R", "R_ckpt"} <= set(p.columns):
            out[str(design)] = (p["R_ckpt"] - p["R"]).dropna().to_numpy(dtype=float)
    return out


def check_claims(t: dict[str, pd.DataFrame], s19: pd.DataFrame) -> pd.DataFrame:
    """正文 3.2–3.4 节、3.6 节、摘要与结论的定性表述，逐条对结果核对；不成立即须改写正文。

    返回逐条结果表：成立 = True / False；缺数据无法核对时为 None。
    """
    a, c, e = (t.get(k, pd.DataFrame()) for k in "ace")
    main_a = a[a["dataset"].isin(MAIN_SETS)] if not a.empty else a
    st = {(f, k): family_stat(main_a, c, f, k)[0] for f in FAMILIES for k in CKEY}
    rows: list[dict[str, Any]] = []

    def claim(cid: str, text: str, vals: list[float], ok: Any) -> None:
        known = all(np.isfinite(v) for v in vals)
        rows.append({"编号": cid, "正文表述": text, "成立": bool(ok) if known else None})

    rep = [st[f, "Δ_rep"] for f in FAMILIES]
    lo_rep = [_lower(c, f, "Δ_rep") for f in DEEP]
    claim("C1", "3.2 节、结论 (1)：四个深度族 S_src−R 的 95% 区间都不含 0", lo_rep, all(v > 0 for v in lo_rep))
    lin, deep = [st["pls", "Δ_rep"], st["pca", "Δ_rep"]], [st[f, "Δ_rep"] for f in DEEP]
    claim("C2", "3.2、3.6 节：两个线性投影的 S_src−R 都低于任一深度族", rep, max(lin) < min(deep))
    claim("C3", "3.2 节：无监督 PCA 的 S_src−R 低于有监督 PLS", lin, lin[1] < lin[0])
    lin_ci = [g(c, f, "Δ_rep") for f in ("pls", "pca") for g in (_lower, _upper)]
    claim(
        "C12",
        "摘要、3.2 节、结论 (1)：两个线性投影 S_src−R 的 95% 区间都含 0（「虚高很小或没有」）",
        lin_ci,
        all(lo <= 0 <= hi for lo, hi in zip(lin_ci[::2], lin_ci[1::2], strict=True)),
    )
    lab = [st[f, "Δ_lab"] for f in LABEL_FAMS]
    shuf = [st[f, "Δ_shuf"] for f in LABEL_FAMS]
    claim(
        "C4",
        "3.3 节、结论 (2)：标签承载虚高——各用标签的族 S_src−S_shuf 为正，S_shuf−R 不超过其一半或 0.02",
        lab + shuf,
        all(x > 0 and (abs(y) <= 0.5 * x or abs(y) <= 0.02) for x, y in zip(lab, shuf, strict=True)),
    )
    big = [f for f in FAMILIES if np.isfinite(st[f, "Δ_rep"]) and st[f, "Δ_rep"] >= 0.02]
    half = [st[f, "Δ_half"] for f in big]
    seen = [st[f, "Δ_seen"] for f in big]
    claim(
        "C5",
        "3.3 节、结论 (2)：S_src−R 不小于 0.02 的族，S_half−R^h 不小于 S_src^h−S_half",
        half + seen,
        all(x >= y for x, y in zip(half, seen, strict=True)),
    )
    lo_half = [_lower(c, f, "Δ_half") for f in DEEP]
    claim("C6", "3.3 节：四个深度族 S_half−R^h 的 95% 区间都不含 0", lo_half, all(v > 0 for v in lo_half))
    if s19.empty:
        claim("C7", "3.4 节：苹果两种设计全部 K 的 SVM 超参数泄漏不超过 0.02", [np.nan], False)
        claim("C8", "3.4 节：PLS-DA 在 K=5 两种设计的超参数泄漏为正且五个种子全正", [np.nan], False)
        claim("C9", "3.4 节：两年设计 K=10、20、50 的 PLS-DA 超参数泄漏都为正", [np.nan], False)
    else:
        svm = s19[s19["方法"] == "SVM-Kshot"]["超参数泄漏"].to_numpy(dtype=float)
        claim("C7", "3.4 节：苹果两种设计全部 K 的 SVM 超参数泄漏不超过 0.02", list(svm), float(svm.max()) <= 0.02)
        k5 = s19[(s19["方法"] == "PLS-DA-Kshot") & (s19["K"] == 5)]
        claim(
            "C8",
            "3.4 节：PLS-DA 在 K=5 两种设计的超参数泄漏为正且五个种子全正",
            list(k5["超参数泄漏"].astype(float)),
            len(k5) == 2 and bool((k5["超参数泄漏"] > 0).all()) and bool((k5["为正种子数"] == k5["种子数"]).all()),
        )
        late = s19[(s19["设计"] == "two_year") & (s19["方法"] == "PLS-DA-Kshot") & (s19["K"] >= 10)]
        vals = list(late["超参数泄漏"].astype(float))
        claim("C9", "3.4 节：两年设计 K=10、20、50 的 PLS-DA 超参数泄漏都为正", vals, len(vals) == 3 and min(vals) > 0)
    main_e = e[e["口径"] == MAIN_SCOPE] if not e.empty else e
    flips = [float(main_e["翻转数"].sum())] if len(main_e) == len(K_KEY) else [np.nan]
    claim("C10", "3.4 节、摘要与结论 (3)：暴露改变了排第一的方法（翻转总数大于 0）", flips, flips[0] > 0)
    lift = list(main_e["最佳值抬升均值"].astype(float)) if len(main_e) == len(K_KEY) else [np.nan]
    claim("C11", "3.4 节：每个 K 上暴露面板的最佳值（留出域均值）都高于无泄漏面板", lift, all(v > 0 for v in lift))
    ck = ckpt_cells()
    tw, th = ck.get("two_year", np.array([np.nan])), ck.get("three_year", np.array([np.nan]))
    claim(
        "C13",
        "摘要、3.2 节：苹果两年设计 R_ckpt−R 在 2 折 × 5 种子共 10 格全为正，且均值大于三年设计",
        [*tw, *th],
        len(tw) == 10 and bool((tw > 0).all()) and float(tw.mean()) > float(th.mean()),
    )
    nc, ni = SENS_SCOPES["Nc"], SENS_SCOPES["Ni"]
    lo_nc = [_lower(c, f, "Δ_rep", nc) for f in DEEP]
    claim("C14", "3.2 节：去掉玉米与药片后，四个深度族 S_src−R 的 95% 区间都不含 0", lo_nc, all(v > 0 for v in lo_nc))
    lo_ni = [_lower(c, f, "Δ_rep", ni) for f in DEEP]
    claim(
        "C15",
        "3.2 节：去掉 Indian Pines 后，ResNet 的 S_src−R 95% 区间含 0，CNN、MLP、Transformer 不含 0",
        lo_ni,
        lo_ni[2] <= 0 and all(lo_ni[i] > 0 for i in (0, 1, 3)),
    )
    pls_nc, pca_b = _lower(c, "pls", "Δ_rep", nc), [g(c, "pca", "Δ_rep", s) for s in (nc, ni) for g in (_lower, _upper)]
    claim(
        "C16",
        "3.2 节：去掉玉米与药片后 PLS 的 S_src−R 95% 区间在 0 以上；PCA 在两项敏感性分析中区间都含 0",
        [pls_nc, *pca_b],
        pls_nc > 0 and all(lo <= 0 <= hi for lo, hi in zip(pca_b[::2], pca_b[1::2], strict=True)),
    )
    if len(main_e) == len(K_KEY):
        enc = float(main_e["翻转后第一为原型网络(S_src)"].sum())
        hp = float(main_e["翻转后第一为留出域选参"].sum())
        n_flip = float(main_e["翻转数"].sum())
    else:
        enc = hp = n_flip = np.nan
    claim(
        "C17",
        "摘要、3.4 节、结论 (3)：每一次翻转都是暴露编码器的原型网络排到第一，留出域选参的 SVM/PLS-DA 一次也没有",
        [enc, hp, n_flip],
        enc == n_flip and hp == 0,
    )
    ni_e = e[e["口径"] == ni] if not e.empty else e
    ni_flips = [float(ni_e["翻转数"].sum())] if len(ni_e) == len(K_KEY) else [np.nan]
    claim("C18", "摘要、3.4 节：去掉 Indian Pines 后仍有翻转", ni_flips, ni_flips[0] > 0)
    shuf_pos: list[float] = []
    for f in DEEP:
        sub = main_a[main_a["编码器"] == f] if not main_a.empty else main_a
        v = sub["Δ_shuf"].dropna() if (not sub.empty and "Δ_shuf" in sub) else pd.Series(dtype=float)
        shuf_pos.append(float((v > 0).mean()) if not v.empty else np.nan)
    rep_lab = [st[f, "Δ_rep"] for f in LABEL_FAMS]
    claim(
        "C19",
        "摘要、3.3 节：置乱标签消除增益——用标签的五个族 S_shuf−R 的留出域均值都不大于 0，四个深度族为正的留出域都不到一半，"
        "且 S_src−S_shuf 都大于 S_src−R",
        shuf + shuf_pos + lab + rep_lab,
        all(y <= 0 for y in shuf)
        and all(q < 0.5 for q in shuf_pos)
        and all(x > r for x, r in zip(lab, rep_lab, strict=True)),
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="77_generality_tables：多编码器泄漏与方法选择的表与宏")
    parser.add_argument("--stage", default="formal", choices=["formal", "pilot"])
    parser.add_argument("--check", action="store_true", help="只核对，不写回")
    args = parser.parse_args()
    logger = eu.get_logger("77_generality_tables")
    t, formal = load(args.stage, logger)
    w = Wrap(formal)
    a = t.get("a", pd.DataFrame())
    c = t.get("c", pd.DataFrame())
    d = t.get("d", pd.DataFrame())
    s19, formal76 = data_s19(logger)
    me, me_df = macros_me(t, formal, s19, formal76)
    s18 = data_s18(d)
    claims = check_claims(t, s19)
    for _, r in claims.iterrows():
        state = {True: "成立", False: "不成立", None: "缺数据"}[r["成立"]]
        logger.log(f"  断言 {r['编号']}：{state}　{r['正文表述']}")
    if formal and not claims["成立"].eq(True).all():
        bad = claims[~claims["成立"].eq(True)]
        raise ValueError(f"正式结果下有定性表述不成立或无法核对，须改写正文：{list(bad['编号'])}")
    ok = True
    for lang, path in m68.MAIN.items():
        t2 = tex_t2(a, c, w, lang)
        if formal and ZMARK not in t2:
            raise ValueError("表 2 没有数据集方差估计为 0 的拟合，表注 a 已无对象：须删去表 2 的表注 a 与 2.9 节的相应说明")
        ok &= m68.replace_blocks(path, {"T2": t2, "ME": me}, args.check, logger)
    for lang, path in m68.TEX.items():
        s16 = tex_s16(a, c, w, lang)
        if formal and ZMARK not in s16:
            raise ValueError("表 S16 没有数据集方差估计为 0 的拟合，表注 a 已无对象：须删去表 S16 的表注 a")
        blocks = {
            "S16": s16,
            "S17": tex_s17(d, w, lang),
            "S18": tex_s18(s18, w, lang),
            "S19": tex_s19(s19, Wrap(formal76), lang),
            "ME": me,
        }
        ok &= m68.replace_blocks(path, blocks, args.check, logger)
    if not args.check:
        out = eu.write_script_workbook(
            __file__,
            {
                eu.table_sheet_name(0, "表2数据集×编码器"): ds_family_mean(a, "Δ_rep").reset_index(),
                eu.table_sheet_name(1, "S18超参数与编码器泄漏"): s18,
                eu.table_sheet_name(2, "S19苹果超参数泄漏"): s19,
                eu.table_sheet_name(3, "数字宏" + ("" if formal else "_预览")): me_df,
                eu.table_sheet_name(4, "定性断言"): claims,
            },
        )
        logger.log(f"已写 {out}")
    logger.close()
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
