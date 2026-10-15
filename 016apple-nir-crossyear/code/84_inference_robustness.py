"""84_inference_robustness.py: 多编码器泄漏对照与方法翻转计数的推断稳健性（补充材料 S9.8）。

为什么:
    75 号按编码器族给了跨数据集的混合效应截距、簇自助区间与符号检验，但稿件里簇自助只报了一维卷积，
    符号检验只报计数，没有逐个去掉数据集的重算；方法翻转也只报了格数。本脚本只读已有的逐种子结果表
    （不训练、不重评估），补下面五项稳健性分析，口径全部沿用 75、77 号的现有定义。

范围:
    主口径（75 号 SCOPES）：苹果两年设计 2 个留出年份 + 五个公开数据集 16 个留出域，共 6 个数据集簇、
    18 个留出域；六个编码器族；三个对照（75 号 INFER_CONTRASTS）：Δ_rep = S_src − R，
    Δ_half = S_half − R_B（R 在同一 B 半上的值），Δ_lab = S_src − S_shuf。某族没有的对照
    （pca 不跑 S_shuf，没有 Δ_lab）与 75 号 inference_table 一样跳过。
    折级值同 75 号 fold_contrasts：种子内配对作差，再对种子求均值。

分析内容:
    表a 留一数据集
        逐个去掉一个数据集，对余下的留出域直接调 75 号 infer（混合效应 Δ ~ 1 + (1|数据集) 的 REML 截距、
        数据集簇自助、两级符号检验）。估计值与区间按 77 号的取法：混合效应截距及其 95% 区间，拟合不可用时
        取折级均值与簇自助区间。每组首行是不去掉任何数据集的参照，并列出留一估计的最小与最大、留一后区间
        不含 0 的次数、去掉后区间结论改变的数据集、估计值变化最大的数据集。簇自助由 infer 自己播种
        （用它的默认种子），--n_boot 为 2000 时参照行与 75 号「表c」主口径相同。
    表b 符号翻转检验
        把每个数据集的域均值（该数据集各留出域折级值的均值）整体乘以 ±1，n 个数据集共 2^n 种
        （6 个时 64 种，含观测本身），全部穷举；统计量 = 各数据集均值的均值，
        单侧 p = 统计量不小于观测值的翻转数 / 2^n。
    表c 逐留出域
        每个（数据集, 留出域, 编码器族, 对照）：种子内配对差的均值、跨种子 95% t 区间
        （均值 ± t(0.975, n−1)·SD/√n，74 号 _t_interval）、为正种子数；另给每个（族, 对照）下
        区间不含 0 的留出域数 / 总数。
    表d 翻转计数的区间
        格 = 留出域 × K（K 取 1、5、10、20，七条路线齐全的格）。翻转按 75 号 ranking_table：无泄漏面板与
        泄漏面板各自排第一的路线属于不同方法（原型网络、SVM、PLS-DA、最近质心；同一方法只换了编码器或
        选参方式的不算）。观测计数按 75 号 ranking_summary 逐 K 相加（77 号翻转总数宏的取法）。
        自助以数据集为簇：有放回地抽与簇数相同个数的数据集，抽中数据集的格整体进入，
        翻转比例 = 翻转格数之和 / 格数之和，取 2.5% 与 97.5% 百分位。另给去掉 Indian Pines 的口径
        （75 号 SCOPES 里的同名敏感性口径）。
    表e1、表e2 留一种子
        64 号评估的第 i 次重采样用种子「正式种子 + i」（i = 0…49），正式种子 2023 与 2024 因此有 49 个
        重采样种子相同：在全部苹果上评估的臂，这 49 次的支持集与查询集相同（两个种子的训练互相独立；
        B 半上的三臂因各种子的半分割不同，抽到的果不同）。公开数据集一侧每个种子各自播种，不重合。
        表e1 逐个去掉一个正式种子：读入的长表按种子过滤后经 75 号 fold_contrasts 重算折级值（余下
        4 个种子），再按表a 参照行的口径（主口径、75 号 infer）重算；每组首行是 5 个种子的参照（即表a
        的参照行），并列出留一估计的最小与最大、区间不含 0 的次数、去掉后区间结论改变的种子。
        表e2 只看苹果两年设计的 2 个留出域：全部种子、去掉 2024 号、去掉 2023 号种子时的种子均值与
        种子 t 区间（去掉后余 4 个种子），以及区间是否不含 0。

先天限制:
    只有 6 个数据集簇：符号翻转检验的单侧 p 不可能小于 1/64 ≈ 0.016；簇自助与留一后的混合效应
    （余 5 个簇）都很粗，区间只能当量级看；翻转集中在少数数据集时，翻转比例的区间主要取决于这些簇
    被抽中的次数。
    逐情节数值没有落盘：64 号只存每个（臂, 留出年份, 种子）50 次重采样的均值与情节 SD（保留 4 位小数），
    74 号只存均值。逐留出域因此只有 5 个种子可用，t 区间衡量的是种子间（训练、半分割、支持集抽样）的
    稳定性，不是对情节或样本的推断；苹果的种子差值带 1e-4 的舍入粒度。留一种子后每个留出域只剩 4 个种子。

自检:
    与 75 号已有的汇总工作簿对账（没有该工作簿时跳过，compute 表记「未比对」）：表a 的参照行对其
    「表c」主口径，去掉 Indian Pines 的留一行对其同名敏感性口径，表d 的翻转计数对其「表e」；
    容差 5e-4，簇自助区间只在 --n_boot 为 2000 时比对。不一致时逐处写进日志，compute 表记「未通过」，
    结果表照写，脚本以非零状态退出；用表之前先看 compute 表的「自检」一栏。留一种子没有 75 号的参照值，不在其内。
    另有两处内部对账，不一致即中止：表c 的种子均值与为正种子数对 fold_contrasts，
    表d 的观测计数对逐格表的直接计数。
    实际读到的主口径数据集数与留出域数记在 compute 表；不是 6 个数据集、18 个留出域时日志给 [WARN]。

复用:
    75 号 load_arms、fold_contrasts、infer、load_routes、ranking_table、ranking_summary，
    以及常量 SCOPES、CONTRASTS、INFER_CONTRASTS、B3_K；74 号 _t_interval；
    export_utils 的 get_logger（Logger）、log_experiment_header、track_compute、
    write_script_workbook、table_sheet_name。

输入（经 75 号的读表函数读取，文件名、工作表与列名以其为准）:
    04outputs/formal_seeds/64_leakage__<设计>__canonical[__<编码器>]__seed<S>.xlsx  苹果各臂宏 F1
    04outputs/formal_seeds/74_multi_encoder__<编码器>__seed<S>.xlsx  公开数据集各臂宏 F1（cnn 另有路线面板）
    04outputs/formal_seeds/72_kscan__[two_year__]seed<S>.xlsx、76_target_tuning__seed<S>.xlsx、
    20_unified_singleface_kshot__seed<S>.xlsx  苹果的方法路线
    04outputs/75_leakage_generality_summary.xlsx  只用于自检，可以没有

运行方式:
    cd <项目根>
    python 02code/84_inference_robustness.py                                # 自助 2000 次
    python 02code/84_inference_robustness.py --n_boot 2000 --seed 20060515  # --seed 只管表d 的簇自助

输出文件:
    04outputs/84_inference_robustness.xlsx — 表a 留一数据集；表b 符号翻转检验；表c 逐留出域；
                                             表d 翻转计数的区间；表e1 留一种子（主口径）；
                                             表e2 留一种子（苹果逐域）；compute
    05logs/84_inference_robustness_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import os
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd

import encoders as enc_lib
import export_utils as eu

m74: Any = importlib.import_module("74_multi_encoder_leakage")
m75: Any = importlib.import_module("75_leakage_generality_summary")

Fits = dict[tuple[str, str], dict[str, Any]]  # （编码器族, 对照）→ 75 号 infer 的结果

MAIN_SCOPE = "主口径（苹果两年设计）"  # 75 号 SCOPES 的键（77 号 MAIN_SCOPE 同值）
NO_IP_SCOPE = "敏感性：主口径去掉 Indian Pines"  # 75 号 SCOPES 的键（77 号 SENS_SCOPES["Ni"] 同值）
CONTRASTS: tuple[str, ...] = tuple(m75.INFER_CONTRASTS)  # Δ_rep、Δ_half、Δ_lab
MAIN_HELD = 18  # 主口径的留出域数（75 号文件头的口径说明）；实际读到的数目不符时日志给 [WARN]
REF_LABEL = "无（全部数据集）"  # 表a 参照行的「去掉的数据集」
REF_SEED_LABEL = "无（全部种子）"  # 表e1 参照行的「去掉的种子」
APPLE_SET = "apple_two_year"  # 苹果两年设计在 75 号里的数据集名（APPLE_DESIGNS 的键）
# 64 号 evaluate 的第 i 次重采样用种子「正式种子 + i」（i < 50）：下面两个正式种子有 49 个重采样种子相同，
# 表e2 看去掉其中一个之后苹果两个留出域的种子区间
OVERLAP_SEEDS = (2024, 2023)
SHEET_E1 = "表e1：留一种子（主口径）"
SHEET_E2 = "表e2：留一种子（苹果逐域）"
TIE_TOL = 1e-12  # 符号翻转统计量与观测值比较时的浮点容差
# 自检读的 75 号汇总工作簿：文件名与表名取自 75 号 main 的写表代码
SUMMARY_75 = os.path.join(eu.OUTPUT_DIR, "75_leakage_generality_summary.xlsx")
SHEET_75_INFER = eu.table_sheet_name(2, "跨数据集推断")
SHEET_75_RANK = eu.table_sheet_name(4, "B3汇总")
CHECK_COLS = ("留出域数", "数据集数", "折级均值", "混合效应截距", "截距95%下限", "截距95%上限", "数据集均值的均值")
CHECK_BOOT_COLS = ("簇自助95%下限", "簇自助95%上限")
CHECK_TEXT_COLS = ("为正留出域", "为正数据集")
CHECK_TOL = 5e-4
INFER_N_BOOT = 2000  # 75 号 infer 的默认自助次数；75 号汇总按它算，自助次数不同时簇自助区间不可比


# ═══════════════════════════════════════════════════════════════════════════════
# 一、参照拟合与留一块（表a、表e1 共用）；表a：留一数据集
# ═══════════════════════════════════════════════════════════════════════════════
def _num(x: Any) -> float:
    """转成浮点；缺失或非数值记 nan。"""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def contrast_text(name: str) -> str:
    """对照的写法，如「S_src − R」；两臂的名字取 75 号 CONTRASTS。"""
    arm_a, arm_b = m75.CONTRASTS[name]
    return f"{arm_a} − {arm_b}"


def headline(res: dict[str, Any]) -> dict[str, Any]:
    """从 infer 的结果里取估计值与区间，取法同 77 号（macros_me、_lower、_upper）。

    估计值取混合效应截距，拟合不可用时取折级均值；区间取截距的 95% 区间，不可用时取簇自助区间。
    """
    b = _num(res.get("混合效应截距"))
    lo, hi = _num(res.get("截距95%下限")), _num(res.get("截距95%上限"))
    b_lo, b_hi = _num(res.get("簇自助95%下限")), _num(res.get("簇自助95%上限"))
    if np.isfinite(b) and np.isfinite(lo) and np.isfinite(hi):
        source = "混合效应截距与其95%区间"
    elif np.isfinite(b):
        source = "混合效应截距；区间取簇自助"
    else:
        source = "折级均值；区间取簇自助"
    lo = lo if np.isfinite(lo) else b_lo
    hi = hi if np.isfinite(hi) else b_hi
    return {
        "估计值": b if np.isfinite(b) else _num(res.get("折级均值")),
        "区间下限": lo,
        "区间上限": hi,
        "区间不含0": bool(lo > 0 or hi < 0),
        "簇自助区间不含0": bool(b_lo > 0 or b_hi < 0),
        "估计与区间取自": source,
    }


def loo_summary(ref: dict[str, Any], loo: list[dict[str, Any]], unit: str) -> dict[str, Any]:
    """留一汇总：估计值的最小与最大、区间不含 0 的次数、去掉后区间结论改变的是哪几个、估计值变化最大的是哪个。"""
    if not loo:
        return {}
    key = f"去掉的{unit}"
    est = np.array([r["估计值"] for r in loo], dtype=np.float64)
    chg = np.abs(np.array([r["相对全量的变化"] for r in loo], dtype=np.float64))
    top = loo[int(np.nanargmax(chg))] if np.isfinite(chg).any() else None
    moved = [str(r[key]) for r in loo if bool(r["区间不含0"]) != bool(ref["区间不含0"])]
    return {
        "留一估计最小": float(np.nanmin(est)),
        "留一估计最大": float(np.nanmax(est)),
        "留一后区间不含0": f"{sum(bool(r['区间不含0']) for r in loo)}/{len(loo)}",
        "留一后簇自助区间不含0": f"{sum(bool(r['簇自助区间不含0']) for r in loo)}/{len(loo)}",
        f"去掉后区间结论改变的{unit}": "; ".join(moved),
        f"变化最大的{unit}": "" if top is None else str(top[key]),
        "最大变化": float("nan") if top is None else float(top["相对全量的变化"]),
    }


def reference_fits(folds_main: pd.DataFrame, n_boot: int, logger: eu.Logger) -> Fits:
    """各（编码器族, 对照）在主口径全部数据集、全部种子上的 infer 结果：表a 与表e1 的参照行。"""
    fits: Fits = {}
    for enc in enc_lib.ENCODERS:
        e = folds_main[folds_main["编码器"] == enc]
        for name in CONTRASTS:
            if name not in e or e[name].dropna().empty:  # 该族没有这一对照：与 75 号 inference_table 一样跳过
                logger.log(f"  {enc} × {name}：没有这一对照，跳过")
                continue
            fits[(enc, name)] = m75.infer(e[name], e["dataset"], n_boot=n_boot)
    return fits


def leave_out_block(
    enc: str,
    name: str,
    full: dict[str, Any],
    variants: list[tuple[str, pd.DataFrame]],
    unit: str,
    ref_label: str,
    n_boot: int,
    logger: eu.Logger,
) -> list[dict[str, Any]]:
    """一个（编码器族, 对照）的留一块：首行是什么都不去掉的参照并带留一汇总，其后每行去掉一个数据集或种子。

    variants 的每一项是（被去掉的那一个的名字, 去掉之后该族的折级表）；unit 取「数据集」或「种子」。
    """
    key = f"去掉的{unit}"
    base: dict[str, Any] = {"编码器": enc, "对照": name, "对照定义": contrast_text(name)}
    h_full = headline(full)
    loo: list[dict[str, Any]] = []
    for label, sub in variants:
        if name not in sub or sub[name].dropna().empty:
            logger.log(f"  {enc} × {name}：去掉{unit} {label} 后没有数据，不出这一行")
            continue
        res: dict[str, Any] = m75.infer(sub[name], sub["dataset"], n_boot=n_boot)
        h = headline(res)
        change = float(h["估计值"]) - float(h_full["估计值"])
        loo.append({**base, key: label, **h, "相对全量的变化": change, **res})
    summ = loo_summary(h_full, loo, unit)
    ref = {**base, key: ref_label, **h_full, "相对全量的变化": float("nan")}
    est_min, est_max = _num(summ.get("留一估计最小")), _num(summ.get("留一估计最大"))
    logger.log(
        f"  {enc} × {name}：全量 {h_full['估计值']:+.4f}"
        f"（{h_full['区间下限']:+.4f}, {h_full['区间上限']:+.4f}），"
        f"去掉一个{unit} {est_min:+.4f} 至 {est_max:+.4f}，"
        f"区间不含 0 的次数 {summ.get('留一后区间不含0', '--')}"
    )
    return [{**ref, **summ, **full}, *loo]


def leave_one_dataset_out(
    folds_main: pd.DataFrame, sets: list[str], fits: Fits, n_boot: int, logger: eu.Logger
) -> pd.DataFrame:
    """表a：逐个去掉一个数据集后的总体估计与区间；每组首行是不去掉任何数据集的参照，并带留一汇总。"""
    rows: list[dict[str, Any]] = []
    for (enc, name), full in fits.items():
        e = folds_main[folds_main["编码器"] == enc]
        have = set(e.dropna(subset=[name])["dataset"])
        for ds in [d for d in sets if d not in have]:
            logger.log(f"  {enc} × {name}：{ds} 没有这一对照的留出域，不做留一")
        variants = [(ds, e[e["dataset"] != ds]) for ds in sets if ds in have]
        rows += leave_out_block(enc, name, full, variants, "数据集", REF_LABEL, n_boot, logger)
    return pd.DataFrame(rows)


def summary_view(tab: pd.DataFrame, unit: str, ref_label: str) -> str:
    """留一表的参照行（每组一行）连同留一汇总，写日志用。"""
    cols = [
        "编码器",
        "对照",
        "估计值",
        "区间下限",
        "区间上限",
        "区间不含0",
        "留一估计最小",
        "留一估计最大",
        "留一后区间不含0",
        f"去掉后区间结论改变的{unit}",
        f"变化最大的{unit}",
        "最大变化",
    ]
    ref = tab[tab[f"去掉的{unit}"] == ref_label]
    return str(ref[[c for c in cols if c in ref]].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# 二、表b：数据集簇级符号翻转检验
# ═══════════════════════════════════════════════════════════════════════════════
def sign_flip_table(folds_main: pd.DataFrame, sets: list[str]) -> pd.DataFrame:
    """表b：把每个数据集的域均值整体翻转符号，穷举全部 2^n 种；统计量为各数据集均值的均值，单侧。"""
    rows: list[dict[str, Any]] = []
    for enc in enc_lib.ENCODERS:
        e = folds_main[folds_main["编码器"] == enc]
        for name in CONTRASTS:
            if name not in e or e[name].dropna().empty:
                continue
            ds_mean = e.dropna(subset=[name]).groupby("dataset")[name].mean()  # 与 infer 里的数据集均值相同
            order = [d for d in sets if d in ds_mean.index]
            d_mean = ds_mean.loc[order].to_numpy(dtype=np.float64)
            n = len(order)
            # 每行一种翻转；首行全为 +1，即观测本身
            signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)), dtype=np.float64)
            stat = signs @ d_mean / n
            obs = float(stat[0])
            n_all, n_ge = len(stat), int((stat >= obs - TIE_TOL).sum())
            rows.append(
                {
                    "编码器": enc,
                    "对照": name,
                    "对照定义": contrast_text(name),
                    "数据集数": n,
                    "数据集均值的均值": obs,
                    "为正数据集": f"{int((d_mean > 0).sum())}/{n}",
                    "翻转总数": n_all,
                    "不小于观测值的翻转数": n_ge,
                    "单侧p": n_ge / n_all,
                    "p下限": 1.0 / n_all,
                    "各数据集均值": "; ".join(f"{k}={v:+.4f}" for k, v in zip(order, d_mean, strict=True)),
                    "表注": (
                        f"按数据集整体翻转符号，共 2^{n} = {n_all} 种（含观测本身），统计量为各数据集均值的均值；"
                        f"单侧 p 的下限是 1/{n_all} = {1.0 / n_all:.4f}"
                    ),
                }
            )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 三、表c：逐留出域的种子区间
# ═══════════════════════════════════════════════════════════════════════════════
def paired_diffs(sub: pd.DataFrame, name: str) -> dict[int, float]:
    """一个（数据集, 留出域, 编码器族）里某对照逐种子的配对差：同一种子的两臂作差（同 75 号 fold_contrasts）。"""
    f1 = {(str(a), int(s)): float(v) for a, s, v in zip(sub["臂"], sub["种子"], sub["F1"], strict=True)}
    arm_a, arm_b = m75.CONTRASTS[name]
    seeds = sorted({s for _, s in f1})
    return {s: f1[(arm_a, s)] - f1[(arm_b, s)] for s in seeds if (arm_a, s) in f1 and (arm_b, s) in f1}


def seed_stats(dl: list[float], prefix: str = "") -> dict[str, Any]:
    """逐种子配对差的种子数、均值、跨种子 95% t 区间（74 号 _t_interval）、为正种子数与区间是否不含 0。"""
    lo, hi = (float(v) for v in m74._t_interval(dl))
    return {
        f"{prefix}种子数": len(dl),
        f"{prefix}种子均值": float(np.mean(dl)) if dl else float("nan"),
        f"{prefix}种子t区间下限": lo,
        f"{prefix}种子t区间上限": hi,
        f"{prefix}为正种子数": sum(d > 0 for d in dl),
        f"{prefix}区间不含0": bool(lo > 0 or hi < 0),
    }


def per_domain_table(arms: pd.DataFrame, folds: pd.DataFrame, sets: list[str]) -> pd.DataFrame:
    """表c：逐（数据集, 留出域, 编码器族, 对照）的种子均值、种子 t 区间与为正种子数。"""
    fold_ref = {(str(r["dataset"]), str(r["held"]), str(r["编码器"])): r for _, r in folds.iterrows()}
    rows: list[dict[str, Any]] = []
    for ds in sets:
        a_ds = arms[arms["dataset"] == ds]
        for held in sorted(a_ds["held"].unique(), key=str):
            for enc in enc_lib.ENCODERS:
                sub = a_ds[(a_ds["held"] == held) & (a_ds["编码器"] == enc)]
                if sub.empty:
                    continue
                want = fold_ref[(ds, str(held), enc)]
                for name in CONTRASTS:
                    dl = list(paired_diffs(sub, name).values())
                    if not dl:
                        continue
                    st = seed_stats(dl)
                    # 与 75 号 fold_contrasts 的折级值对账
                    mean, ref_mean = float(st["种子均值"]), _num(want.get(name))
                    same = (np.isnan(mean) and np.isnan(ref_mean)) or abs(mean - ref_mean) <= 1e-9
                    if not same or st["为正种子数"] != _num(want.get(f"{name}_为正种子数")):
                        raise RuntimeError(f"{ds}/{held}/{enc}/{name}：种子配对差与 75 号 fold_contrasts 不一致")
                    ids = {"dataset": ds, "held": str(held), "编码器": enc, "对照": name}
                    rows.append({**ids, "对照定义": contrast_text(name), **st})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 每个（族, 对照）下区间不含 0 的留出域数 / 总数，写在该组的每一行；总数只计种子均值有值的留出域（同 infer）
    count: dict[tuple[str, str], tuple[str, str]] = {}
    for (enc, name), g in out.groupby(["编码器", "对照"], sort=False):
        ok = g[g["种子均值"].notna()]
        n_out, n_up = int(ok["区间不含0"].sum()), int((ok["种子t区间下限"] > 0).sum())
        count[(str(enc), str(name))] = (f"{n_out}/{len(ok)}", f"{n_up}/{len(ok)}")
    keys = list(zip(out["编码器"], out["对照"], strict=True))
    out["本族本对照区间不含0的留出域"] = [count[k][0] for k in keys]
    out["其中区间在0以上"] = [count[k][1] for k in keys]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 四、表d：方法翻转比例的数据集簇自助
# ═══════════════════════════════════════════════════════════════════════════════
def flip_ratio_table(rank: pd.DataFrame, rank_sum: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """表d：翻转比例（翻转格数 / 总格数）的数据集簇自助百分位区间；主口径与去掉 Indian Pines 两个口径。"""
    k_text = "、".join(str(k) for k in m75.B3_K)
    rows: list[dict[str, Any]] = []
    for stream, scope in enumerate((MAIN_SCOPE, NO_IP_SCOPE), start=1):
        sets = list(m75.SCOPES[scope])
        cells = rank[rank["dataset"].isin(sets)]
        have = set(cells["dataset"])
        order = [d for d in sets if d in have]
        n_flip = np.array([int(cells.loc[cells["dataset"] == d, "翻转"].sum()) for d in order], dtype=np.float64)
        n_cell = np.array([int((cells["dataset"] == d).sum()) for d in order], dtype=np.float64)
        # 观测计数取 75 号 ranking_summary 逐 K 相加（77 号翻转总数宏的取法），并与逐格表的直接计数对账
        s = rank_sum[rank_sum["口径"] == scope]
        flips, total = int(s["翻转数"].sum()), int(s["留出域数"].sum())
        if total == 0 or (flips, total) != (int(n_flip.sum()), int(n_cell.sum())):
            raise RuntimeError(f"{scope}：没有可用的格，或翻转计数 {flips}/{total} 与逐格表的直接计数不一致")
        rng = np.random.default_rng([seed, 84, stream])
        pick = rng.integers(0, len(order), size=(n_boot, len(order)))  # 每行一次自助：有放回抽中的数据集
        ratio = n_flip[pick].sum(axis=1) / n_cell[pick].sum(axis=1)
        lo, hi = (float(v) for v in np.percentile(ratio, [2.5, 97.5]))
        rows.append(
            {
                "口径": scope,
                "数据集簇数": len(order),
                "总格数": total,
                "翻转数": flips,
                "翻转比例": flips / total,
                "自助次数": n_boot,
                "随机数流": f"default_rng([{seed}, 84, {stream}])",
                "比例95%下限": lo,
                "比例95%上限": hi,
                "各数据集翻转数/格数": "; ".join(
                    f"{d}={int(f)}/{int(c)}" for d, f, c in zip(order, n_flip, n_cell, strict=True)
                ),
                "表注": (
                    f"格 = 留出域 × K（K 取 {k_text}，七条路线齐全的格）；"
                    "翻转 = 无泄漏面板与泄漏面板排第一的方法不同；"
                    "自助以数据集为簇、簇内的格整体进出，区间取翻转比例的 2.5% 与 97.5% 百分位；"
                    f"簇只有 {len(order)} 个，区间很粗"
                ),
            }
        )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 五、表e1、表e2：留一种子
# ═══════════════════════════════════════════════════════════════════════════════
def leave_one_seed_out(
    arms: pd.DataFrame, sets: list[str], seeds: list[int], fits: Fits, n_boot: int, logger: eu.Logger
) -> pd.DataFrame:
    """表e1：逐个去掉一个正式种子后主口径的总体估计与区间；每组首行是不去掉任何种子的参照（即表a 的参照行）。"""
    # 读入后的长表按种子过滤，再经 75 号 fold_contrasts 重算折级值（余下的种子内配对作差、求均值）
    dropped: dict[int, pd.DataFrame] = {}
    for s in seeds:
        folds_s = m75.fold_contrasts(arms[arms["种子"] != s])
        dropped[s] = folds_s[folds_s["dataset"].isin(sets)]
    rows: list[dict[str, Any]] = []
    for (enc, name), full in fits.items():
        variants = [(str(s), d[d["编码器"] == enc]) for s, d in dropped.items()]
        rows += leave_out_block(enc, name, full, variants, "种子", REF_SEED_LABEL, n_boot, logger)
    return pd.DataFrame(rows)


def apple_seed_table(arms: pd.DataFrame) -> pd.DataFrame:
    """表e2：苹果两年设计各留出域在全部种子、去掉 OVERLAP_SEEDS 里的一个种子时的种子均值与种子 t 区间。"""
    a_ds = arms[arms["dataset"] == APPLE_SET]
    rows: list[dict[str, Any]] = []
    for held in sorted(a_ds["held"].unique(), key=str):
        for enc in enc_lib.ENCODERS:
            sub = a_ds[(a_ds["held"] == held) & (a_ds["编码器"] == enc)]
            if sub.empty:
                continue
            for name in CONTRASTS:
                diff = paired_diffs(sub, name)
                if not diff:
                    continue
                row: dict[str, Any] = {"dataset": APPLE_SET, "held": str(held), "编码器": enc, "对照": name}
                row["对照定义"] = contrast_text(name)
                row.update(seed_stats(list(diff.values()), "全部种子_"))
                moved: list[str] = []
                for s in OVERLAP_SEEDS:
                    tag = f"去掉{s}"
                    row.update(seed_stats([v for k, v in diff.items() if k != s], f"{tag}_"))
                    if row[f"{tag}_区间不含0"] != row["全部种子_区间不含0"]:
                        moved.append(tag)
                row["区间结论改变于"] = "; ".join(moved)
                rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 六、自检与主流程
# ═══════════════════════════════════════════════════════════════════════════════
def selfcheck(tab_a: pd.DataFrame, tab_d: pd.DataFrame, ip_set: str, n_boot: int, logger: eu.Logger) -> tuple[str, int]:
    """与 75 号已有的汇总工作簿对账；返回（写进 compute 表的状态, 不一致的处数），逐处不一致写进日志。"""
    if not os.path.exists(SUMMARY_75):
        logger.log(f"[自检] 没有 {SUMMARY_75}，未与 75 号汇总对账")
        return "未比对（没有 75 号汇总工作簿）", 0
    xl = pd.ExcelFile(SUMMARY_75)
    if SHEET_75_INFER not in xl.sheet_names or SHEET_75_RANK not in xl.sheet_names:
        logger.log(f"[自检] {SUMMARY_75} 里没有「{SHEET_75_INFER}」或「{SHEET_75_RANK}」，未对账")
        return "未比对（75 号汇总工作簿缺表）", 0
    infer_75, rank_75 = xl.parse(SHEET_75_INFER), xl.parse(SHEET_75_RANK)
    if "口径" not in infer_75 or "口径" not in rank_75:
        logger.log(f"[自检] {SUMMARY_75} 的推断表或排序汇总表是空的，未对账")
        return "未比对（75 号汇总工作簿的表是空的）", 0
    cols = [*CHECK_COLS, *(CHECK_BOOT_COLS if n_boot == INFER_N_BOOT else ())]
    bad: list[str] = []
    n_cmp = 0
    # 参照行对主口径；去掉 Indian Pines 的留一行对同名敏感性口径
    for scope, label in ((MAIN_SCOPE, REF_LABEL), (NO_IP_SCOPE, ip_set)):
        mine = tab_a[tab_a["去掉的数据集"] == label]
        theirs = infer_75[infer_75["口径"] == scope]
        # 行数只对参照行核：被去掉的数据集本来就没有某个对照时，本脚本不出那一行留一，75 号的敏感性口径仍有一行
        if label == REF_LABEL and len(mine) != len(theirs):
            bad.append(f"{scope}：本脚本 {len(mine)} 行，75 号汇总 {len(theirs)} 行")
        for _, r in mine.iterrows():
            tag = f"{scope} / {r['编码器']} / {r['对照']}"
            m = theirs[(theirs["编码器"] == r["编码器"]) & (theirs["对照"] == r["对照"])]
            if len(m) != 1:
                bad.append(f"{tag}：75 号汇总里有 {len(m)} 行")
                continue
            t = m.iloc[0]
            for col in cols:
                x, y = _num(r.get(col)), _num(t.get(col))
                n_cmp += 1
                if not ((np.isnan(x) and np.isnan(y)) or abs(x - y) <= CHECK_TOL):
                    bad.append(f"{tag} / {col}：本脚本 {x:.6f}，75 号汇总 {y:.6f}")
            for col in CHECK_TEXT_COLS:
                n_cmp += 1
                if str(r.get(col)) != str(t.get(col)):
                    bad.append(f"{tag} / {col}：本脚本 {r.get(col)}，75 号汇总 {t.get(col)}")
    for _, r in tab_d.iterrows():
        s = rank_75[rank_75["口径"] == r["口径"]]
        got = (int(s["翻转数"].sum()), int(s["留出域数"].sum()))
        n_cmp += 1
        if got != (int(r["翻转数"]), int(r["总格数"])):
            bad.append(f"{r['口径']} / 翻转计数：本脚本 {r['翻转数']}/{r['总格数']}，75 号汇总 {got[0]}/{got[1]}")
    if bad:
        for line in bad:
            logger.log(f"[自检] 不一致　{line}")
        return f"未通过（对 75 号汇总 {n_cmp} 项，{len(bad)} 处不一致，见日志）", len(bad)
    logger.log(f"[自检] 与 75 号汇总工作簿对账通过：{n_cmp} 项，容差 {CHECK_TOL}")
    return f"通过（对 75 号汇总 {n_cmp} 项，容差 {CHECK_TOL}）", 0


def main() -> None:
    parser = argparse.ArgumentParser(description="84_inference_robustness：泄漏对照与方法翻转计数的推断稳健性")
    parser.add_argument("--n_boot", type=int, default=2000, help="簇自助次数（表a、表e1 传给 infer，表d 自用）")
    parser.add_argument(
        "--seed", type=int, default=20060515, help="表d 簇自助的种子；表a、表e1 的簇自助由 75 号 infer 自己播种"
    )
    args = parser.parse_args()
    if args.n_boot < 1 or args.seed < 0:
        parser.error("--n_boot 须为正整数，--seed 须为非负整数")
    unknown = [s for s in (MAIN_SCOPE, NO_IP_SCOPE) if s not in m75.SCOPES]
    if unknown:
        raise SystemExit(f"75 号 SCOPES 里没有口径 {unknown}：口径名已改，须同步本脚本的常量")
    sets = [str(d) for d in m75.SCOPES[MAIN_SCOPE]]
    dropped = [d for d in sets if d not in m75.SCOPES[NO_IP_SCOPE]]
    if len(dropped) != 1:
        raise SystemExit(f"「{NO_IP_SCOPE}」相对主口径应恰好少一个数据集，实际少 {dropped}")
    seeds = [int(s) for s in eu.RECOMMENDED_SEEDS]
    if APPLE_SET not in sets or any(s not in seeds for s in OVERLAP_SEEDS):
        raise SystemExit(f"{APPLE_SET} 不在主口径里，或 {OVERLAP_SEEDS} 不全是正式种子：须同步本脚本的常量")

    logger = eu.get_logger("84_inference_robustness")
    eu.log_experiment_header(
        logger,
        {
            "任务": "泄漏对照与方法翻转计数的推断稳健性",
            "内容": "留一数据集、符号翻转检验、逐留出域区间、翻转比例的簇自助、留一种子",
            "口径": f"{MAIN_SCOPE}：{sets}",
            "编码器族": list(enc_lib.ENCODERS),
            "对照": {c: contrast_text(c) for c in CONTRASTS},
            "正式种子": seeds,
            "留一种子": f"逐个去掉一个正式种子；{APPLE_SET} 逐域另看去掉 {OVERLAP_SEEDS} 之一",
            "自助次数": args.n_boot,
            "自助的随机数": f"表d 用 default_rng([{args.seed}, 84, 流编号])；表a、表e1 由 75 号 infer 自己播种",
            "统计库版本": {p: metadata.version(p) for p in ("pandas", "scipy", "statsmodels")},
        },
    )
    with eu.track_compute("推断稳健性", logger, None) as budget:
        arms, miss_a = m75.load_arms("formal", seeds)
        routes, miss_r = m75.load_routes("formal", seeds)
        missing = [*miss_a, *miss_r]
        if missing:
            logger.log(f"缺 {len(missing)} 个逐种子结果表：" + "; ".join(str(x) for x in missing))
            raise SystemExit("逐种子结果表不全，不出稳健性表")
        folds = m75.fold_contrasts(arms)
        folds_main = folds[folds["dataset"].isin(sets)]
        n_held = folds_main.groupby("dataset")["held"].nunique()
        n_sets, n_dom = len(n_held), int(n_held.sum())
        logger.log(f"主口径：{n_sets} 个数据集、{n_dom} 个留出域（{n_held.to_dict()}）")
        if n_sets != len(sets) or n_dom != MAIN_HELD:
            logger.log(f"  [WARN] 主口径应为 {len(sets)} 个数据集、{MAIN_HELD} 个留出域，先核对逐种子结果表")
        fits = reference_fits(folds_main, args.n_boot, logger)
        logger.log("一、留一数据集")
        tab_a = leave_one_dataset_out(folds_main, sets, fits, args.n_boot, logger)
        logger.log("二、符号翻转检验")
        tab_b = sign_flip_table(folds_main, sets)
        logger.log("三、逐留出域")
        tab_c = per_domain_table(arms, folds, sets)
        logger.log("四、翻转计数的区间")
        rank = m75.ranking_table(routes)
        if rank.empty:
            raise SystemExit("没有七条路线齐全的格，算不了翻转计数")
        tab_d = flip_ratio_table(rank, m75.ranking_summary(rank), args.n_boot, args.seed)
        logger.log("五、留一种子")
        tab_e1 = leave_one_seed_out(arms, sets, seeds, fits, args.n_boot, logger)
        tab_e2 = apple_seed_table(arms)
        check, n_bad = selfcheck(tab_a, tab_d, dropped[0], args.n_boot, logger)

    logger.log("表a 留一数据集（参照行）\n" + summary_view(tab_a, "数据集", REF_LABEL))
    logger.log("表b 符号翻转检验\n" + tab_b.drop(columns=["各数据集均值", "表注"]).to_string(index=False))
    n_short = int((tab_c["种子数"] < len(seeds)).sum())
    if n_short:
        logger.log(f"表c 有 {n_short} 行可配对的种子少于 {len(seeds)} 个，t 区间按实际种子数算")
    count_cols = ["编码器", "对照", "本族本对照区间不含0的留出域", "其中区间在0以上"]
    per = tab_c.drop_duplicates(["编码器", "对照"])[count_cols]
    logger.log("表c 各（族, 对照）下种子 t 区间不含 0 的留出域数 / 总数\n" + per.to_string(index=False))
    logger.log("表d 翻转计数的区间\n" + tab_d.drop(columns=["表注"]).to_string(index=False))
    logger.log("表e1 留一种子（参照行）\n" + summary_view(tab_e1, "种子", REF_SEED_LABEL))
    e2_cols = ["held", "编码器", "对照", "全部种子_种子均值", "全部种子_区间不含0"]
    for s in OVERLAP_SEEDS:
        e2_cols += [f"去掉{s}_种子均值", f"去掉{s}_种子t区间下限", f"去掉{s}_种子t区间上限", f"去掉{s}_区间不含0"]
    e2_view = tab_e2[[c for c in [*e2_cols, "区间结论改变于"] if c in tab_e2]]
    logger.log(f"表e2 留一种子（{APPLE_SET} 逐域）\n" + e2_view.to_string(index=False))
    out = eu.write_script_workbook(
        __file__,
        {
            eu.table_sheet_name(0, "留一数据集"): tab_a,
            eu.table_sheet_name(1, "符号翻转检验"): tab_b,
            eu.table_sheet_name(2, "逐留出域"): tab_c,
            eu.table_sheet_name(3, "翻转计数的区间"): tab_d,
            SHEET_E1: tab_e1,
            SHEET_E2: tab_e2,
            "compute": pd.DataFrame(
                [
                    {
                        "口径": MAIN_SCOPE,
                        "数据集数": n_sets,
                        "留出域数": n_dom,
                        "正式种子": ",".join(str(s) for s in seeds),
                        "留一种子": f"逐个去掉 {len(seeds)} 个正式种子；苹果逐域去掉 {OVERLAP_SEEDS} 之一",
                        "n_boot": args.n_boot,
                        "seed": args.seed,
                        "自检": check,
                        **budget,
                    }
                ]
            ),
        },
    )
    logger.log(f"结果表 → {out}（自检：{check}）")
    logger.close()
    if n_bad:
        raise SystemExit(
            f"自检未通过：与 75 号汇总工作簿有 {n_bad} 处不一致（见日志）。结果表已写出，compute 表记了「未通过」；"
            "先核对逐种子结果表与该汇总是否同一批结果、统计库版本是否相同，再用这份结果"
        )


if __name__ == "__main__":
    main()
