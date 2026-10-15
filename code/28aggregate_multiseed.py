"""
28aggregate_multiseed.py: Formal 多种子结果聚合（§5.3 / §7.2），把 5 种子 per-seed xlsx 聚合为
正式 mean/std/95%CI 表，回填稿件数字。

读取 04outputs/formal_seeds/<脚本stem>__seed<S>.xlsx（由 run_evals.sh 逐种子归档），
对核心指标在 5 个模型种子上聚合：
  - 评估泄漏 Δ@K=5（苹果）= 共享协议(06) − 折内受限(10)，逐折 + 总体；
  - K 曲线：折内受限(10) vs 共享(06) macro-F1 随 K；
  - src_alpha 消融(12)：配置 A/B/C 的 F1@K5；折内 per-class(19)：逐产地 F1；
  - 外部 OliveOil(25)：共享 vs 折内受限 Δ；LLH-GBID(20)：各方法 GBIDS 与主排名；
  - 负控(24)：label permutation p 值 + rand/id-probe 真实 F1；校准(26)：ECE + 温度 T；
  - 可解释(27)：top 波段与化学归属；面均值 2×2(15)：查询/支持侧效应；
  - 方差分解(10)：编码器层 vs 情节层 SD（替代 23）。
统计口径：n_primary=K_outer=3（折），n_seeds=5；报告 mean±std，CI 为种子级 percentile bootstrap。

运行方式:
    cd <project root>
    python 02code/28aggregate_multiseed.py

输出文件:
    04outputs/28aggregate_multiseed.xlsx  — 各核心指标的多种子聚合表
    05logs/28aggregate_multiseed_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

import export_utils as eu

NDArray = npt.NDArray[Any]
FORMAL_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
SEEDS: tuple[int, ...] = eu.RECOMMENDED_SEEDS


def _load(stem: str, seed: int, sheet: str | int = 0) -> pd.DataFrame | None:
    """读取某脚本某种子的归档 xlsx 指定 sheet；缺失返回 None。"""
    path = os.path.join(FORMAL_DIR, f"{stem}__seed{seed}.xlsx")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except (ValueError, KeyError):
        return None


def _agg(values: list[float]) -> tuple[float, float, float, float, int]:
    """返回 (mean, std, ci_lo, ci_hi, n)；CI 为种子级 percentile bootstrap(1000)。"""
    arr = np.asarray([v for v in values if v == v], dtype=float)  # 去 NaN
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        rng = np.random.default_rng(2024)
        boot = np.asarray([rng.choice(arr, size=n, replace=True).mean() for _ in range(1000)])
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    else:
        lo = hi = mean
    return (mean, std, lo, hi, n)


def _sheet_a(stem: str, seed: int) -> pd.DataFrame | None:
    """读"表a"（10/06 的首个 sheet 名以'表a'起头）。"""
    df = _load(stem, seed, 0)
    return df


def agg_leakage_and_curve(logger: eu.Logger) -> dict[str, pd.DataFrame]:
    """泄漏 Δ@K5（逐折+总体）+ K 曲线（折内受限 vs 共享）。"""
    ks = [1, 3, 5, 10, 20, 50]
    # 收集 per-seed 的 (fold,K)->F1，restricted=10, shared=06
    rest: dict[tuple[int, int], list[float]] = {}
    shar: dict[tuple[int, int], list[float]] = {}
    for s in SEEDS:
        for stem, store in (("10_loeo_year_foldwise", rest), ("06_loeo_kshot", shar)):
            df = _sheet_a(stem, s)
            if df is None or "F1均值" not in df.columns:
                continue
            for _, r in df.iterrows():
                try:
                    fold = int(r["目标年份"])
                    k = int(r["K"])
                except (ValueError, TypeError):
                    continue
                store.setdefault((fold, k), []).append(float(r["F1均值"]))
    folds = sorted({f for f, _ in rest} | {f for f, _ in shar})

    # K 曲线（折平均后跨种子聚合）—— 先对每种子算折均值，再跨种子
    curve_rows: list[dict[str, Any]] = []
    for k in ks:
        # 每种子的折均值
        def fold_mean(store: dict[tuple[int, int], list[float]], kk: int, si: int) -> float:
            vals = [store[(f, kk)][si] for f in folds if (f, kk) in store and len(store[(f, kk)]) > si]
            return float(np.mean(vals)) if vals else float("nan")

        nseed = len(SEEDS)
        r_seed = [fold_mean(rest, k, i) for i in range(nseed)]
        s_seed = [fold_mean(shar, k, i) for i in range(nseed)]
        d_seed = [s - r for s, r in zip(s_seed, r_seed, strict=False) if s == s and r == r]
        rm = _agg(r_seed)
        sm = _agg(s_seed)
        dm = _agg(d_seed)
        curve_rows.append(
            {
                "K": k,
                "折内受限F1均值": round(rm[0], 4),
                "折内受限std": round(rm[1], 4),
                "共享F1均值": round(sm[0], 4),
                "共享std": round(sm[1], 4),
                "泄漏Δ均值": round(dm[0], 4),
                "泄漏Δstd": round(dm[1], 4),
                "Δ_95CI": f"[{dm[2]:.4f}, {dm[3]:.4f}]",
                "n_seeds": dm[4],
            }
        )
        logger.log(f"  K={k}: 折内={rm[0]:.4f}±{rm[1]:.4f} 共享={sm[0]:.4f} Δ={dm[0]:.4f}±{dm[1]:.4f}")

    # 泄漏 Δ@K5 逐折
    leak_rows: list[dict[str, Any]] = []
    for f in folds:
        if (f, 5) in rest and (f, 5) in shar:
            r5 = rest[(f, 5)]
            s5 = shar[(f, 5)]
            d5 = [s - r for s, r in zip(s5, r5, strict=False)]
            dm = _agg(d5)
            leak_rows.append(
                {
                    "折(目标年份)": f,
                    "折内受限F1@K5": round(float(np.mean(r5)), 4),
                    "共享F1@K5": round(float(np.mean(s5)), 4),
                    "泄漏Δ@K5均值": round(dm[0], 4),
                    "Δstd": round(dm[1], 4),
                    "Δ_95CI": f"[{dm[2]:.4f}, {dm[3]:.4f}]",
                    "n_seeds": dm[4],
                }
            )
    return {"泄漏Δ_K5逐折": pd.DataFrame(leak_rows), "K曲线_折内vs共享": pd.DataFrame(curve_rows)}


def agg_external(logger: eu.Logger) -> pd.DataFrame:
    """OliveOil 外部验证：共享 vs 折内受限 的 macro-F1 与 Δ。"""
    by_proto: dict[str, list[float]] = {}
    for s in SEEDS:
        df = _load("25external", s, 0)
        if df is None or "协议" not in df.columns:
            continue
        for _, r in df.iterrows():
            by_proto.setdefault(str(r["协议"]), []).append(float(r["Macro-F1均值"]))
    rows: list[dict[str, Any]] = []
    for proto, vals in by_proto.items():
        m = _agg(vals)
        rows.append(
            {
                "协议": proto,
                "Macro-F1均值": round(m[0], 4),
                "std": round(m[1], 4),
                "95CI": f"[{m[2]:.4f},{m[3]:.4f}]",
                "n_seeds": m[4],
            }
        )
    logger.log(f"  OliveOil 协议数={len(rows)}")
    return pd.DataFrame(rows)


def agg_gbid(logger: eu.Logger) -> pd.DataFrame:
    """LLH-GBID(20 表b)：各方法 GBIDS 与主排名跨种子聚合。"""
    gbids: dict[str, list[float]] = {}
    ranks: dict[str, list[float]] = {}
    for s in SEEDS:
        df = _load("20_unified_singleface_kshot", s, "表b：GBID综合结果表")
        if df is None or "方法" not in df.columns:
            continue
        for _, r in df.iterrows():
            m = str(r["方法"])
            gbids.setdefault(m, []).append(float(r["GBIDS"]))
            ranks.setdefault(m, []).append(float(r["主排名"]))
    rows: list[dict[str, Any]] = []
    for m in gbids:
        gm = _agg(gbids[m])
        rm = _agg(ranks[m])
        rows.append(
            {
                "方法": m,
                "GBIDS均值": round(gm[0], 2),
                "GBIDSstd": round(gm[1], 2),
                "平均主排名": round(rm[0], 2),
                "n_seeds": gm[4],
            }
        )
    rows.sort(key=lambda d: d["平均主排名"])
    logger.log(f"  GBID 方法数={len(rows)}")
    return pd.DataFrame(rows)


def agg_negctrl(logger: eu.Logger) -> pd.DataFrame:
    """负控(24)：label_permutation p 值 + 三种负控真实 F1 跨种子。

    label_permutation 以置换 p 值证伪；random_feature / id_only_probe 以真实 F1
    退化至随机水平(3 类≈0.33)证伪，无 p 值。故同时聚合真实 F1 与 p 值。
    """
    p_by: dict[tuple[Any, str], list[float]] = {}
    f1_by: dict[tuple[Any, str], list[float]] = {}
    for s in SEEDS:
        df = _load("24negctrl", s, 0)
        if df is None or "负控" not in df.columns:
            continue
        for _, r in df.iterrows():
            key = (r["fold"], str(r["负控"]))
            if "真实F1" in df.columns and r["真实F1"] == r["真实F1"]:
                f1_by.setdefault(key, []).append(float(r["真实F1"]))
            if "p_value" in df.columns and r["p_value"] == r["p_value"]:
                p_by.setdefault(key, []).append(float(r["p_value"]))
    rows: list[dict[str, Any]] = []
    for key in sorted(set(p_by) | set(f1_by), key=lambda t: (str(t[0]), t[1])):
        fold, ctrl = key
        fm = _agg(f1_by.get(key, []))
        pm = _agg(p_by.get(key, []))
        has_p = pm[4] > 0
        rows.append(
            {
                "fold": fold,
                "负控": ctrl,
                "真实F1均值": round(fm[0], 4),
                "F1std": round(fm[1], 4),
                "p均值": float(f"{pm[0]:.6f}") if has_p else float("nan"),
                "判定": (("YES(p<0.001)" if pm[0] < 0.001 else "NO") if has_p else "退化至随机✓"),
                "n_seeds": max(fm[4], pm[4]),
            }
        )
    return pd.DataFrame(rows)


def agg_calibration(logger: eu.Logger) -> pd.DataFrame:
    """校准(26)：ECE / 校准后 ECE / 温度缩放 T 跨种子。"""
    by: dict[Any, list[float]] = {}
    byc: dict[Any, list[float]] = {}
    byt: dict[Any, list[float]] = {}
    for s in SEEDS:
        df = _load("26calibration", s, 0)
        if df is None or "ECE" not in df.columns:
            continue
        for _, r in df.iterrows():
            by.setdefault(r["fold"], []).append(float(r["ECE"]))
            if "校准后ECE" in df.columns:
                byc.setdefault(r["fold"], []).append(float(r["校准后ECE"]))
            if "温度缩放T" in df.columns and r["温度缩放T"] == r["温度缩放T"]:
                byt.setdefault(r["fold"], []).append(float(r["温度缩放T"]))
    rows: list[dict[str, Any]] = []
    for fold, vals in by.items():
        m = _agg(vals)
        mc = _agg(byc.get(fold, []))
        mt = _agg(byt.get(fold, []))
        rows.append(
            {
                "fold": fold,
                "ECE均值": round(m[0], 4),
                "ECEstd": round(m[1], 4),
                "校准后ECE均值": round(mc[0], 4),
                "温度T均值": round(mt[0], 3) if mt[4] > 0 else float("nan"),
                "n_seeds": m[4],
            }
        )
    return pd.DataFrame(rows)


def agg_alpha(logger: eu.Logger) -> pd.DataFrame:
    """src_alpha 消融(12 表a)：配置 A/B/C 在 K=5 的三折均值 F1 跨种子。

    每种子内先对目标折求均值，再跨种子聚合，保持 n_primary=K_outer 口径。
    """
    by: dict[str, list[float]] = {}
    for s in SEEDS:
        df = _load("12_loeo_ablate_alpha", s, 0)
        if df is None or "配置" not in df.columns or "K" not in df.columns:
            continue
        sub = df[df["K"] == 5]
        for cfg, g in sub.groupby("配置"):
            by.setdefault(str(cfg), []).append(float(g["F1均值"].mean()))
    rows: list[dict[str, Any]] = []
    for cfg in sorted(by):
        m = _agg(by[cfg])
        rows.append(
            {
                "配置": cfg,
                "F1@K5均值": round(m[0], 4),
                "std": round(m[1], 4),
                "95CI": f"[{m[2]:.4f},{m[3]:.4f}]",
                "n_seeds": m[4],
            }
        )
    logger.log(f"  alpha 配置数={len(rows)}")
    return pd.DataFrame(rows)


def agg_perclass(logger: eu.Logger) -> pd.DataFrame:
    """折内 per-class F1(19 表a)：K=5 各目标年份逐产地 F1 跨种子。"""
    macro: dict[Any, list[float]] = {}
    per: dict[tuple[Any, str], list[float]] = {}
    prov = ("SD", "SX", "XJ", "GS")
    for s in SEEDS:
        df = _load("19_perclass_f1_foldwise", s, 0)
        if df is None or "宏F1均值" not in df.columns:
            continue
        sub = df[df["K"] == 5]
        for _, r in sub.iterrows():
            yr = r["目标年份"]
            macro.setdefault(yr, []).append(float(r["宏F1均值"]))
            for p in prov:
                col = f"F1_{p}_均值"
                if col in df.columns and r[col] == r[col]:
                    per.setdefault((yr, p), []).append(float(r[col]))
    rows: list[dict[str, Any]] = []
    for yr in sorted(macro, key=str):
        mm = _agg(macro[yr])
        row: dict[str, Any] = {
            "目标年份": yr,
            "宏F1均值": round(mm[0], 4),
            "宏F1std": round(mm[1], 4),
        }
        for p in prov:
            pm = _agg(per.get((yr, p), []))
            row[f"F1_{p}"] = round(pm[0], 4) if pm[4] > 0 else float("nan")
        row["n_seeds"] = mm[4]
        rows.append(row)
    logger.log(f"  per-class 年份数={len(rows)}")
    return pd.DataFrame(rows)


def agg_interpret(logger: eu.Logger) -> pd.DataFrame:
    """可解释(27 表a)：跨种子统计 top 波段的波长范围与主导化学归属。"""
    waves: list[float] = []
    chem: dict[str, int] = {}
    n_seed = 0
    for s in SEEDS:
        df = _load("27interpret", s, 0)
        if df is None or "波长nm" not in df.columns:
            continue
        n_seed += 1
        waves.extend(float(w) for w in df["波长nm"])
        for c in df["化学归属"]:
            chem[str(c)] = chem.get(str(c), 0) + 1
    if not waves:
        return pd.DataFrame([])
    arr = np.asarray(waves)
    top_chem = max(chem, key=lambda k: chem[k])
    rows = [
        {
            "top波段波长范围nm": f"{arr.min():.1f}-{arr.max():.1f}",
            "中位波长nm": round(float(np.median(arr)), 1),
            "主导化学归属": top_chem,
            "主导占比": f"{chem[top_chem]}/{len(waves)}",
            "n_seeds": n_seed,
        }
    ]
    logger.log(f"  interpret n_seeds={n_seed}")
    return pd.DataFrame(rows)


def agg_face(logger: eu.Logger) -> pd.DataFrame:
    """面均值 2x2 消融(15 效应分解K5)：各条件 F1 跨种子（若已归档）。"""
    by: dict[str, list[float]] = {}
    col_f1: Any = None
    cond_col: Any = None
    for s in SEEDS:
        df = _load("15_face_ablation_50reps", s, "效应分解K5")
        if df is None or df.empty:
            continue
        if cond_col is None:
            cond_col = df.columns[0]
            for c in df.columns:
                if "F1" in str(c) and ("均值" in str(c) or "mean" in str(c).lower()):
                    col_f1 = c
                    break
            if col_f1 is None:
                col_f1 = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        for _, r in df.iterrows():
            by.setdefault(str(r[cond_col]), []).append(float(r[col_f1]))
    rows: list[dict[str, Any]] = []
    for cond in by:
        m = _agg(by[cond])
        rows.append(
            {
                "条件": cond,
                "F1均值": round(m[0], 4),
                "std": round(m[1], 4),
                "n_seeds": m[4],
            }
        )
    if rows:
        logger.log(f"  face 条件数={len(rows)}")
    return pd.DataFrame(rows)


def agg_variance(logger: eu.Logger) -> pd.DataFrame:
    """折内受限(脚本10, vanilla ProtoNet α=0)的编码器层 vs 情节层方差分解(替代 23)。

    编码器层 SD = 5 种子折均值的标准差；情节层 SD = 各种子内情节 std(F1标准差)的均值；
    编码器跨度 = 5 种子折均值的极差。刻画报告 std 中编码器随机性 vs 情节采样的占比。
    """
    ks = [1, 3, 5, 10, 20, 50]
    fold_mean_by_k: dict[int, list[float]] = {}
    epi_sd_by_k: dict[int, list[float]] = {}
    for s in SEEDS:
        df = _load("10_loeo_year_foldwise", s, 0)
        if df is None or "F1均值" not in df.columns:
            continue
        for k in ks:
            sub = df[df["K"] == k]
            if sub.empty:
                continue
            fold_mean_by_k.setdefault(k, []).append(float(sub["F1均值"].mean()))
            if "F1标准差" in df.columns:
                epi_sd_by_k.setdefault(k, []).append(float(sub["F1标准差"].mean()))
    rows: list[dict[str, Any]] = []
    for k in ks:
        fm = fold_mean_by_k.get(k, [])
        if not fm:
            continue
        enc_sd = float(np.std(fm, ddof=1)) if len(fm) > 1 else 0.0
        epi_sd = float(np.mean(epi_sd_by_k.get(k, [0.0])))
        ratio = f"1:{epi_sd / enc_sd:.1f}" if enc_sd > 0 else "NA"
        rows.append(
            {
                "K": k,
                "三折均值": round(float(np.mean(fm)), 4),
                "编码器层SD": round(enc_sd, 4),
                "情节层SD": round(epi_sd, 4),
                "比例": ratio,
                "编码器跨度": round(float(max(fm) - min(fm)), 4),
                "n_seeds": len(fm),
            }
        )
    logger.log(f"  variance K数={len(rows)}")
    return pd.DataFrame(rows)


def agg_kshot_methods(logger: eu.Logger) -> pd.DataFrame:
    """K-shot 五方法对比(11 表a)：各 (K, 方法) 三折均值 F1 跨种子 → tab:kshot。"""
    name_map = {
        "OurProtoNet": "ProtoNet-C",
        "SVM-Kshot": "SVM",
        "VanillaProtoNet": "原始ProtoNet",
        "PLS-DA-Kshot": "PLS-DA",
        "NearestCentroid": "最近质心",
    }
    ks = [1, 3, 5, 10, 20, 50]
    by: dict[tuple[int, str], list[float]] = {}
    for s in SEEDS:
        df = _load("11_kshot_baselines", s, 0)
        if df is None or "方法" not in df.columns:
            continue
        for k in ks:
            for meth, g in df[df["K"] == k].groupby("方法"):
                by.setdefault((k, str(meth)), []).append(float(g["F1均值"].mean()))
    rows: list[dict[str, Any]] = []
    for k in ks:
        row: dict[str, Any] = {"K": k}
        for raw, disp in name_map.items():
            m = _agg(by.get((k, raw), []))
            row[disp] = round(m[0], 4) if m[4] > 0 else float("nan")
        row["n_seeds"] = max((_agg(by.get((k, r), []))[4] for r in name_map), default=0)
        rows.append(row)
    logger.log(f"  kshot方法 K数={len(rows)}")
    return pd.DataFrame(rows)


def plot_leakage(logger: eu.Logger) -> None:
    """核心评估泄漏对比图（主图，dot plot 非柱状，§8.1）：共享(06) vs 折限定(10) 逐折 K5。"""
    eu.setup_chinese_font()
    folds = [2018, 2019, 2025]
    sh: dict[int, tuple[float, float, float, float, int]] = {}
    rr: dict[int, tuple[float, float, float, float, int]] = {}
    for f in folds:
        sv: list[float] = []
        rv: list[float] = []
        for s in SEEDS:
            d6 = _load("06_loeo_kshot", s, 0)
            d10 = _load("10_loeo_year_foldwise", s, 0)
            if d6 is not None and "F1均值" in d6.columns:
                v = d6[(d6["目标年份"] == f) & (d6["K"] == 5)]["F1均值"]
                if len(v):
                    sv.append(float(v.mean()))
            if d10 is not None and "F1均值" in d10.columns:
                v = d10[(d10["目标年份"] == f) & (d10["K"] == 5)]["F1均值"]
                if len(v):
                    rv.append(float(v.mean()))
        sh[f] = _agg(sv)
        rr[f] = _agg(rv)

    def plot_fn(lang: str) -> Any:
        import matplotlib.pyplot as plt

        zh = lang == "zh"
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        x = list(range(len(folds)))
        ax.errorbar(
            x,
            [sh[f][0] for f in folds],
            yerr=[sh[f][1] for f in folds],
            fmt="o",
            color="#D55E00",
            ms=11,
            capsize=5,
            lw=2,
            label="共享编码器（泄漏）" if zh else "Shared encoder (leaky)",
        )
        ax.errorbar(
            x,
            [rr[f][0] for f in folds],
            yerr=[rr[f][1] for f in folds],
            fmt="s",
            color="#0072B2",
            ms=11,
            capsize=5,
            lw=2,
            label="折限定编码器（无泄漏）" if zh else "Fold-restricted (clean)",
        )
        for i, f in enumerate(folds):
            d = sh[f][0] - rr[f][0]
            ax.annotate(
                f"$\\Delta$={d:.3f}",
                (i, (sh[f][0] + rr[f][0]) / 2),
                ha="left",
                va="center",
                fontsize=10,
                color="#333333",
                xytext=(14, 0),
                textcoords="offset points",
            )
        ax.axhline(1 / 3, ls=":", color="gray", lw=1, label="随机基线" if zh else "Chance")
        ax.set_xticks(x)
        ax.set_xticklabels([f"留出 {f}" if zh else f"LOEO-{f}" for f in folds])
        ax.set_ylabel("Macro-F1（3 类产地）" if zh else "Macro-F1 (3 origins)")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            "评估泄漏：共享 vs 折限定编码器（K=5，5 种子）"
            if zh
            else "Evaluation leakage: shared vs fold-restricted (K=5, 5 seeds)"
        )
        ax.legend(loc="center right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    p_zh, p_en = eu.save_figure_bilingual(plot_fn, "28aggregate_multiseed", 0)
    logger.log(f"泄漏对比图 → {os.path.basename(p_zh)} / {os.path.basename(p_en)}")


def plot_kshot(logger: eu.Logger) -> None:
    """K-shot 5 方法对比图（fig:kshot，折线非柱状）：ProtoNet-C(组合口径) + 4 基线。

    ProtoNet-C = 2018/2019 折 vanilla 单面 + 2025 折多面(S_multi_Q_multi)，与 tab:kshot 一致；
    其余 4 方法取自 11_kshot_baselines。
    """
    eu.setup_chinese_font()
    ks = [1, 3, 5, 10, 20, 50]
    d11 = [_load("11_kshot_baselines", s, 0) for s in SEEDS]
    d15 = [_load("15_face_ablation_50reps", s, "表a：面位消融结果") for s in SEEDS]
    methods = {
        "SVM": "SVM-Kshot",
        "原始ProtoNet": "VanillaProtoNet",
        "PLS-DA": "PLS-DA-Kshot",
        "最近质心": "NearestCentroid",
    }
    curves: dict[str, list[float]] = {}
    for disp, raw in methods.items():
        seq: list[float] = []
        for k in ks:
            vals = [
                float(df[(df["K"] == k) & (df["方法"] == raw)]["F1均值"].mean())
                for df in d11
                if df is not None and "方法" in df.columns
            ]
            seq.append(_agg([v for v in vals if v == v])[0])
        curves[disp] = seq
    pc: list[float] = []
    for k in ks:
        vals = []
        for df11, df15 in zip(d11, d15, strict=False):
            if df11 is None or df15 is None:
                continue
            v18 = df11[(df11["K"] == k) & (df11["方法"] == "VanillaProtoNet") & (df11["目标年份"] == 2018)][
                "F1均值"
            ].mean()
            v19 = df11[(df11["K"] == k) & (df11["方法"] == "VanillaProtoNet") & (df11["目标年份"] == 2019)][
                "F1均值"
            ].mean()
            m25 = df15[(df15["配置"] == "S_multi_Q_multi") & (df15["K"] == k)]["F1均值"].mean()
            vals.append((v18 + v19 + m25) / 3)
        pc.append(_agg([v for v in vals if v == v])[0])
    curves["ProtoNet-C"] = pc

    def plot_fn(lang: str) -> Any:
        import matplotlib.pyplot as plt

        zh = lang == "zh"
        colors = {
            "ProtoNet-C": "#D55E00",
            "SVM": "#009E73",
            "原始ProtoNet": "#56B4E9",
            "PLS-DA": "#E69F00",
            "最近质心": "#999999",
        }
        en = {
            "ProtoNet-C": "ProtoNet-C",
            "SVM": "SVM",
            "原始ProtoNet": "Vanilla ProtoNet",
            "PLS-DA": "PLS-DA",
            "最近质心": "Nearest centroid",
        }
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for meth in ["ProtoNet-C", "SVM", "原始ProtoNet", "PLS-DA", "最近质心"]:
            ax.plot(ks, curves[meth], marker="o", color=colors[meth], lw=2, label=meth if zh else en[meth])
        ax.axhline(1 / 3, ls=":", color="gray", lw=1, label="随机基线" if zh else "Chance")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("K-shot（每产地支持苹果数）" if zh else "K-shot (support apples per origin)")
        ax.set_ylabel("Macro-F1（三折均值）" if zh else "Macro-F1 (3-fold mean)")
        ax.set_ylim(0, 1.0)
        ax.set_title(
            "LOEO-Year K-shot 5 方法对比（折限定编码器，5 种子）"
            if zh
            else "LOEO-Year K-shot: 5 methods (fold-restricted, 5 seeds)"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    p_zh, p_en = eu.save_figure_bilingual(plot_fn, "28aggregate_multiseed", 1)
    logger.log(f"K-shot 对比图 → {os.path.basename(p_zh)} / {os.path.basename(p_en)}")


def plot_face(logger: eu.Logger) -> None:
    """面均值 2x2 K 曲线图（fig:face，折线+CI带）：4 策略 + SVM，LOEO-2025，脚本 15 五种子。

    与 tab:face 同源（15_face_ablation 表a），确保图表数值一致。
    """
    eu.setup_chinese_font()
    ks = [1, 3, 5, 10, 20, 50]
    d15 = [_load("15_face_ablation_50reps", s, "表a：面位消融结果") for s in SEEDS]
    cfgs = [
        "S_multi_Q_multi",
        "S_single_Q_multi",
        "S_multi_Q_single",
        "S_single_Q_single",
        "SVM-Kshot",
    ]
    mean_c: dict[str, list[float]] = {}
    std_c: dict[str, list[float]] = {}
    for c in cfgs:
        ms: list[float] = []
        ss: list[float] = []
        for k in ks:
            vals = [
                float(df[(df["配置"] == c) & (df["K"] == k)]["F1均值"].mean())
                for df in d15
                if df is not None and "配置" in df.columns
            ]
            a = _agg([v for v in vals if v == v])
            ms.append(a[0])
            ss.append(a[1])
        mean_c[c] = ms
        std_c[c] = ss

    def plot_fn(lang: str) -> Any:
        import matplotlib.pyplot as plt

        zh = lang == "zh"
        colors = {
            "S_multi_Q_multi": "#D55E00",
            "S_single_Q_multi": "#CC79A7",
            "S_multi_Q_single": "#E69F00",
            "S_single_Q_single": "#009E73",
            "SVM-Kshot": "#56B4E9",
        }
        zh_lab = {
            "S_multi_Q_multi": "支持全面位+查询全面位",
            "S_single_Q_multi": "支持单面位+查询全面位",
            "S_multi_Q_single": "支持全面位+查询单面位",
            "S_single_Q_single": "支持单面位+查询单面位（≈2018/2019）",
            "SVM-Kshot": "SVM（全面位均值光谱）",
        }
        en_lab = {
            "S_multi_Q_multi": "Support-multi / Query-multi",
            "S_single_Q_multi": "Support-single / Query-multi",
            "S_multi_Q_single": "Support-multi / Query-single",
            "S_single_Q_single": "Support-single / Query-single",
            "SVM-Kshot": "SVM (multi-face mean)",
        }
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        for c in cfgs:
            m = np.array(mean_c[c])
            e = np.array(std_c[c])
            lbl = zh_lab[c] if zh else en_lab[c]
            ax.plot(ks, m, marker="o", color=colors[c], lw=2, label=lbl)
            ax.fill_between(ks, m - e, m + e, color=colors[c], alpha=0.15)
        ax.axhline(1 / 3, ls=":", color="gray", lw=1, label="随机基线" if zh else "Chance")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("K-shot（每产地支持苹果数）" if zh else "K-shot (support apples per origin)")
        ax.set_ylabel("Macro-F1（LOEO-2025）" if zh else "Macro-F1 (LOEO-2025)")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            "LOEO-2025 面位数对 ProtoNet 的贡献（5 种子）"
            if zh
            else "LOEO-2025 face-averaging contribution to ProtoNet (5 seeds)"
        )
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    p_zh, p_en = eu.save_figure_bilingual(plot_fn, "28aggregate_multiseed", 2)
    logger.log(f"面均值 K 曲线图 → {os.path.basename(p_zh)} / {os.path.basename(p_en)}")


def plot_alpha(logger: eu.Logger) -> None:
    """α 灵敏度曲线（fig:alpha_sensitivity）：3 折 K=5 macro-F1 vs α，17 脚本 5 种子聚合。

    读 17_alpha_sensitivity sheet0（α敏感性扫描：目标年份/K/alpha/F1均值）；17 未归档则跳过。
    """
    d17 = [_load("17_alpha_sensitivity", s, 0) for s in SEEDS]
    valid = [d for d in d17 if d is not None and "alpha" in d.columns and "F1均值" in d.columns]
    if not valid:
        logger.log("  [跳过] 17_alpha 未归档，fig:alpha_sensitivity 暂用 pilot 图")
        return
    eu.setup_chinese_font()
    folds = [2018, 2019, 2025]
    alphas = sorted({float(a) for a in valid[0]["alpha"]})
    curves: dict[int, tuple[list[float], list[float]]] = {}
    for yr in folds:
        ms: list[float] = []
        es: list[float] = []
        for a in alphas:
            vals: list[float] = []
            for d in valid:
                sub = d[(d["目标年份"] == yr) & (d["K"] == 5) & ((d["alpha"] - a).abs() < 1e-6)]
                if len(sub):
                    vals.append(float(sub["F1均值"].mean()))
            agg = _agg([v for v in vals if v == v])
            ms.append(agg[0])
            es.append(agg[1])
        curves[yr] = (ms, es)

    def plot_fn(lang: str) -> Any:
        import matplotlib.pyplot as plt

        zh = lang == "zh"
        colors = {2018: "#0072B2", 2019: "#009E73", 2025: "#D55E00"}
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        for yr in folds:
            m = np.array(curves[yr][0])
            e = np.array(curves[yr][1])
            ax.plot(alphas, m, marker="o", ms=4, color=colors[yr], lw=2, label=f"留出 {yr}" if zh else f"LOEO-{yr}")
            ax.fill_between(alphas, m - e, m + e, color=colors[yr], alpha=0.15)
        ax.axvline(0.4, ls="--", color="gray", lw=1, alpha=0.7, label="α=0.4 参考" if zh else "α=0.4 ref")
        ax.set_xlabel("源质心融合权重 α" if zh else "Source-prototype fusion weight α")
        ax.set_ylabel("Macro-F1（K=5，三折）" if zh else "Macro-F1 (K=5, per fold)")
        ax.set_title(
            "源质心融合权重 α 灵敏度（K=5，5 种子 CUDA）"
            if zh
            else "Source-fusion weight α sensitivity (K=5, 5 seeds CUDA)"
        )
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    p_zh, p_en = eu.save_figure_bilingual(plot_fn, "28aggregate_multiseed", 3)
    logger.log(f"α 灵敏度图 → {os.path.basename(p_zh)} / {os.path.basename(p_en)}")


def main() -> None:
    logger = eu.get_logger("28aggregate_multiseed")
    eu.log_experiment_header(
        logger, {"任务": "Formal 多种子聚合(§5.3/§7.2)", "种子": str(SEEDS), "归档目录": FORMAL_DIR}
    )
    if not os.path.isdir(FORMAL_DIR):
        logger.log(f"[错误] 未找到归档目录 {FORMAL_DIR}；请先完成 run_evals.sh 并回传 formal_seeds/。")
        logger.close()
        return

    sheets: dict[Any, Any] = {}
    idx = 0
    logger.log("== 泄漏 Δ + K 曲线 ==")
    for name, df in agg_leakage_and_curve(logger).items():
        sheets[idx] = (name, df)
        idx += 1
    logger.log("== 外部 OliveOil ==")
    sheets[idx] = ("外部OliveOil", agg_external(logger))
    idx += 1
    logger.log("== LLH-GBID ==")
    sheets[idx] = ("GBID汇总", agg_gbid(logger))
    idx += 1
    logger.log("== 负控 ==")
    sheets[idx] = ("负控", agg_negctrl(logger))
    idx += 1
    logger.log("== 校准 ==")
    sheets[idx] = ("校准ECE", agg_calibration(logger))
    idx += 1
    logger.log("== src_alpha 消融 ==")
    sheets[idx] = ("alpha消融", agg_alpha(logger))
    idx += 1
    logger.log("== 折内 per-class ==")
    sheets[idx] = ("per-class", agg_perclass(logger))
    idx += 1
    logger.log("== 可解释波段 ==")
    sheets[idx] = ("可解释波段", agg_interpret(logger))
    idx += 1
    logger.log("== 面均值 2x2 ==")
    face_df = agg_face(logger)
    if not face_df.empty:
        sheets[idx] = ("面均值2x2", face_df)
        idx += 1
    logger.log("== 方差分解 ==")
    sheets[idx] = ("方差分解", agg_variance(logger))
    idx += 1
    logger.log("== K-shot 五方法 ==")
    sheets[idx] = ("kshot五方法", agg_kshot_methods(logger))
    idx += 1

    xlsx = eu.write_script_workbook(__file__, sheets)
    logger.log(f"聚合结果 → {xlsx}")
    logger.log("== 泄漏对比图（主图）==")
    plot_leakage(logger)
    logger.log("== K-shot 对比图（主图）==")
    plot_kshot(logger)
    logger.log("== 面均值 K 曲线图 ==")
    plot_face(logger)
    logger.log("== α 灵敏度图 ==")
    plot_alpha(logger)
    logger.close()


if __name__ == "__main__":
    main()
