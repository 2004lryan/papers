"""75_leakage_generality_summary.py: 泄漏跨编码器族与数据集的汇总（B1、B2）与泄漏对方法选择的影响（B3）。

为什么:
    改定位后的主线是「可学习组件见过留出域带标签的数据，交叉域结果就被抬高」。本脚本把 64 号（苹果）与 74 号
    （五个公开数据集）的逐种子结果合成「编码器族 × 数据集 × 留出域」的对照表，并按编码器族做跨数据集推断；
    再把苹果（72、76、20 号）与公开数据集（74 号路线面板）的方法比较合成同一口径，统计泄漏评估与无泄漏评估下
    排第一的方法是否不同，以及超参数在留出域上选定带来的抬升。

口径:
    对照在种子内配对作差、再对种子求均值：Δ_rep = S_src − R，Δ_lab = S_src − S_shuf，Δ_half = S_half − R_B（B 半），
    Δ_seen = S_src_B − S_half，Δ_rep_B = S_src_B − R_B。
    主口径：苹果取同仪器两年设计（2 个留出年份）+ 五个公开数据集（16 个留出域）= 6 个数据集、18 个留出域；
    补充口径：苹果换成三年设计（3 个留出年份）。两种苹果设计共用 2018、2019 两个目标年，不同时计入。
    敏感性口径：主口径去掉玉米与药片（这两个数据集的各域是同一批实物在不同仪器上的光谱，R 的训练池里已有
    被评估实物在其他仪器上的光谱与标签）；主口径去掉 Indian Pines（遥感成像的地物分类，不是分子光谱）。
    推断按编码器族分别做（折级值 = 种子均值）：混合效应 Δ ~ 1 + (1|数据集)（REML）；簇自助（簇 = 数据集，2000 次）；
    折级与数据集级单侧二项符号检验。与 25 号同一口径。
    B3：每个留出域、每个 K 按种子均值排序。无泄漏路线 = {原型网络(R)、SVM(源域选参)、PLS-DA(源域选参)、最近质心}，
    泄漏路线 = {原型网络(S_src)、SVM(留出域选参)、PLS-DA(留出域选参)、最近质心}；两边排第一的方法不同记为翻转。
    苹果的四类路线在同一批单面情节上评估（20、71、72、76 号共用种子 S+i 的情节）；公开数据集的七条路线在
    74 号面板的同一批情节上评估。

运行方式:
    cd <项目根>
    python 02code/75_leakage_generality_summary.py                 # 正式结果（5 个正式种子）
    python 02code/75_leakage_generality_summary.py --stage pilot   # 只读 pilot 结果（种子 2004），检查流程

输出文件:
    04outputs/75_leakage_generality_summary.xlsx（--stage pilot 时文件名加 __pilot）
    05logs/75_leakage_generality_summary_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import importlib
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

import encoders as enc_lib
import export_utils as eu

m74 = importlib.import_module("74_multi_encoder_leakage")

SEED_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
APPLE_DESIGNS = {"apple_two_year": "two_year", "apple_three_year": "three_year"}
PUBLIC_SETS = list(m74.DATASETS)
SHARED_SAMPLE_SETS = ("corn_instr", "tablet_instr")  # 各域是同一批实物在不同仪器上的光谱
SCOPES = {
    "主口径（苹果两年设计）": ["apple_two_year", *PUBLIC_SETS],
    "补充口径（苹果三年设计）": ["apple_three_year", *PUBLIC_SETS],
    "敏感性：主口径去掉玉米与药片": ["apple_two_year", *[d for d in PUBLIC_SETS if d not in SHARED_SAMPLE_SETS]],
    "敏感性：主口径去掉 Indian Pines": ["apple_two_year", *[d for d in PUBLIC_SETS if d != "indianpines_spatial"]],
}
CONTRASTS: dict[str, tuple[str, str]] = m74.CONTRASTS
INFER_CONTRASTS = ("Δ_rep", "Δ_half", "Δ_lab")
FAMILY = {
    "原型网络(R)": "原型网络",
    "原型网络(S_src)": "原型网络",
    "SVM(源域选参)": "SVM",
    "SVM(留出域选参)": "SVM",
    "PLS-DA(源域选参)": "PLS-DA",
    "PLS-DA(留出域选参)": "PLS-DA",
    "最近质心": "最近质心",
}
ROUTES_FREE = m74.ROUTES_FREE
ROUTES_LEAK = m74.ROUTES_LEAK
B3_K = (1, 5, 10, 20)
MIN_DATASETS_MIXED = 3  # 随机截距至少要 3 个簇才可估
FIT_METHODS = ("lbfgs", "bfgs", "powell", "nm")  # REML 拟合依次试这四种优化器，取对数似然最高者


# ═══════════════════════════════════════════════════════════════════════════════
# 一、读入逐种子结果
# ═══════════════════════════════════════════════════════════════════════════════
def _suffix(stage: str) -> str:
    return "" if stage == "formal" else "__pilot"


def load_arms(stage: str, seeds: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """苹果（64 号）与公开数据集（74 号）的逐臂宏 F1，统一为 (dataset, held, 编码器, 种子, 臂, F1)。"""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for ds, design in APPLE_DESIGNS.items():
        for enc in enc_lib.ENCODERS:
            seg = "" if enc == "cnn" else f"__{enc}"
            for s in seeds:
                name = f"64_leakage__{design}__canonical{seg}__seed{s}{_suffix(stage)}.xlsx"
                path = os.path.join(SEED_DIR, name)
                if not os.path.exists(path):
                    missing.append(name)
                    continue
                d = pd.read_excel(path, sheet_name=0)
                d = d[d["src_alpha"] == 0.0]
                for _, r in d.iterrows():
                    rows.append(
                        {
                            "dataset": ds,
                            "held": f"Y{int(r['目标年份'])}",
                            "编码器": enc,
                            "种子": s,
                            "臂": str(r["臂"]),
                            "F1": float(r["F1均值"]),
                        }
                    )
    for enc in enc_lib.ENCODERS:
        for s in seeds:
            name = f"74_multi_encoder__{enc}__seed{s}{_suffix(stage)}.xlsx"
            path = os.path.join(SEED_DIR, name)
            if not os.path.exists(path):
                missing.append(name)
                continue
            d = pd.read_excel(path, sheet_name=0)
            rows += [
                {
                    "dataset": str(r["dataset"]),
                    "held": str(r["held"]),
                    "编码器": enc,
                    "种子": s,
                    "臂": str(r["臂"]),
                    "F1": float(r["F1"]),
                }
                for _, r in d.iterrows()
            ]
    return pd.DataFrame(rows), missing


def fold_contrasts(arms: pd.DataFrame) -> pd.DataFrame:
    """逐 (数据集, 留出域, 编码器) 的各臂种子均值与对照（种子内配对作差后求均值、为正种子数）。"""
    out: list[dict[str, Any]] = []
    for (ds, held, enc), sub in arms.groupby(["dataset", "held", "编码器"], sort=False):
        f1 = {(a, int(s)): float(v) for a, s, v in zip(sub["臂"], sub["种子"], sub["F1"], strict=True)}
        seeds = sorted({s for _, s in f1})
        row: dict[str, Any] = {"dataset": ds, "held": held, "编码器": enc, "种子数": len(seeds)}
        for arm in sorted({a for a, _ in f1}):
            row[arm] = float(np.mean([f1[(arm, s)] for s in seeds if (arm, s) in f1]))
        for name, (a, b) in CONTRASTS.items():
            dl = [f1[(a, s)] - f1[(b, s)] for s in seeds if (a, s) in f1 and (b, s) in f1]
            if dl:
                row[name] = float(np.mean(dl))
                row[f"{name}_为正种子数"] = int(sum(d > 0 for d in dl))
        out.append(row)
    return pd.DataFrame(out)


# ═══════════════════════════════════════════════════════════════════════════════
# 二、按编码器族的跨数据集推断（与 25 号同口径）
# ═══════════════════════════════════════════════════════════════════════════════
def _fit_note(messages: list[str], converged: bool, se: float) -> str:
    """把拟合时的告警归成几类写进结果表：数据集方差落在边界、Hessian 非正定、未收敛、SE 无效。"""
    notes: list[str] = []
    if not converged or any("failed to converge" in m or "did not converge" in m for m in messages):
        notes.append("未完全收敛")
    if any("boundary" in m for m in messages):
        notes.append("数据集方差在边界（≈0）")
    if any("Hessian" in m for m in messages):
        notes.append("Hessian 非正定")
    if not np.isfinite(se):
        notes.append("截距 SE 无效")
    return "；".join(notes) if notes else "正常"


def infer(values: pd.Series, groups: pd.Series, n_boot: int = 2000, seed: int = 20060515) -> dict[str, Any]:
    """混合效应截距、簇自助区间与两级符号检验；values 为折级值，groups 为数据集。"""
    df = pd.DataFrame({"y": values.to_numpy(dtype=np.float64), "g": groups.to_numpy()}).dropna()
    res: dict[str, Any] = {"留出域数": len(df), "数据集数": df["g"].nunique(), "折级均值": float(df["y"].mean())}
    if res["数据集数"] < MIN_DATASETS_MIXED:
        res["混合效应"] = f"数据集少于 {MIN_DATASETS_MIXED} 个，不拟合"
    else:
        # 数据集方差落在边界附近时，单一优化器可能停在非最优点（实测 lbfgs 对 PCA 的 Δ_rep 停在截距 0、REML 对数似然
        # 66.76，而 bfgs/powell/nm 都到 −0.00085、67.11）：四种优化器各拟合一次，取 REML 对数似然最高者
        model = MixedLM.from_formula("y ~ 1", groups="g", data=df)
        fits: list[tuple[str, Any, list[str]]] = []
        for method in FIT_METHODS:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    f = model.fit(reml=True, method=method)
                if np.isfinite(f.llf):
                    fits.append((method, f, [str(w.message) for w in caught]))
            except Exception:  # 该优化器失败时看其余优化器
                continue
        if fits:
            method, fit, msgs = max(fits, key=lambda t: t[1].llf)
            b, se = float(fit.fe_params["Intercept"]), float(fit.bse_fe["Intercept"])
            res.update(
                {
                    "混合效应截距": b,
                    "截距SE": se,
                    "截距95%下限": b - 1.96 * se,
                    "截距95%上限": b + 1.96 * se,
                    "截距p": float(fit.pvalues["Intercept"]),
                    "数据集方差": float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else 0.0,
                    "混合效应": _fit_note(msgs, bool(fit.converged), se),
                    "优化器": method,
                    "REML对数似然": float(fit.llf),
                }
            )
        else:  # 奇异或不收敛时只报簇自助
            res["混合效应"] = "未收敛：四种优化器均失败"
    rng = np.random.default_rng(seed)
    ds_mean = df.groupby("g")["y"].mean()
    names = ds_mean.index.to_numpy()
    boots = [float(ds_mean.loc[rng.choice(names, size=len(names), replace=True)].mean()) for _ in range(n_boot)]
    res["数据集均值的均值"] = float(ds_mean.mean())
    res["簇自助95%下限"], res["簇自助95%上限"] = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    n_pos, n_tot = int((df["y"] > 0).sum()), len(df)
    res["为正留出域"] = f"{n_pos}/{n_tot}"
    res["折级符号检验p"] = float(stats.binomtest(n_pos, n_tot, 0.5, alternative="greater").pvalue)
    d_pos, d_tot = int((ds_mean > 0).sum()), len(ds_mean)
    res["为正数据集"] = f"{d_pos}/{d_tot}"
    res["数据集级符号检验p"] = float(stats.binomtest(d_pos, d_tot, 0.5, alternative="greater").pvalue)
    return res


def inference_table(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, sets in SCOPES.items():
        sub = folds[folds["dataset"].isin(sets)]
        for enc in enc_lib.ENCODERS:
            e = sub[sub["编码器"] == enc]
            for name in INFER_CONTRASTS:
                if name not in e or e[name].dropna().empty:
                    continue
                rows.append({"口径": scope, "编码器": enc, "对照": name, **infer(e[name], e["dataset"])})
    return pd.DataFrame(rows)


def by_dataset_table(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    order = ["apple_two_year", "apple_three_year", *PUBLIC_SETS]
    for ds in order:
        for enc in enc_lib.ENCODERS:
            sub = folds[(folds["dataset"] == ds) & (folds["编码器"] == enc)]
            if sub.empty:
                continue
            row: dict[str, Any] = {"dataset": ds, "编码器": enc, "留出域数": len(sub)}
            for arm in ("R", "S_src"):
                if arm in sub:
                    row[f"{arm}均值"] = round(float(sub[arm].mean()), 4)
            for name in CONTRASTS:
                if name in sub and not sub[name].dropna().empty:
                    v = sub[name].dropna()
                    row[f"{name}均值"] = round(float(v.mean()), 4)
                    row[f"{name}最小"] = round(float(v.min()), 4)
                    row[f"{name}最大"] = round(float(v.max()), 4)
                    row[f"{name}为正留出域"] = f"{int((v > 0).sum())}/{len(v)}"
            rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 三、B3：泄漏是否改变排第一的方法
# ═══════════════════════════════════════════════════════════════════════════════
def load_routes(stage: str, seeds: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """统一为 (dataset, held, 种子, K, 路线, F1)。"""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    sfx = _suffix(stage)
    for ds, design in APPLE_DESIGNS.items():
        seg = "" if design == "three_year" else f"{design}__"
        for s in seeds:
            name = f"72_kscan__{seg}seed{s}{sfx}.xlsx"
            path = os.path.join(SEED_DIR, name)
            if not os.path.exists(path):
                missing.append(name)
            else:
                d = pd.read_excel(path, sheet_name=0)
                for _, r in d[d["臂"].isin(["R", "S_src"])].iterrows():
                    rows.append(
                        {
                            "dataset": ds,
                            "held": f"Y{int(r['目标年份'])}",
                            "种子": s,
                            "K": int(r["K"]),
                            "路线": f"原型网络({r['臂']})",
                            "F1": float(r["F1均值"]),
                        }
                    )
            name = f"76_target_tuning__seed{s}{sfx}.xlsx"
            path = os.path.join(SEED_DIR, name)
            if not os.path.exists(path):
                missing.append(name)
            else:
                d = pd.read_excel(path, sheet_name=0)
                d = d[d["设计"] == design]
                for _, r in d.iterrows():
                    meth = "SVM" if str(r["方法"]).startswith("SVM") else "PLS-DA"
                    for col, tag in (("目标宏F1(源年份选参)", "源域选参"), ("目标宏F1(目标年选参)", "留出域选参")):
                        rows.append(
                            {
                                "dataset": ds,
                                "held": f"Y{int(r['目标年份'])}",
                                "种子": s,
                                "K": int(r["K"]),
                                "路线": f"{meth}({tag})",
                                "F1": float(r[col]),
                            }
                        )
            # 最近质心不训练、只看目标年：取 20 号正式面板（与设计无关）
            path = os.path.join(SEED_DIR, f"20_unified_singleface_kshot__seed{s}.xlsx")
            if not os.path.exists(path):
                missing.append(os.path.basename(path))
            else:
                d = pd.read_excel(path, sheet_name=0)
                folds = [2018, 2019] if design == "two_year" else [2018, 2019, 2025]
                d = d[(d["方法"] == "NearestCentroid") & d["目标年份"].isin(folds)]
                rows += [
                    {
                        "dataset": ds,
                        "held": f"Y{int(r['目标年份'])}",
                        "种子": s,
                        "K": int(r["K"]),
                        "路线": "最近质心",
                        "F1": float(r["F1均值"]),
                    }
                    for _, r in d.iterrows()
                ]
    for s in seeds:
        name = f"74_multi_encoder__cnn__seed{s}{sfx}.xlsx"
        path = os.path.join(SEED_DIR, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        xl = pd.ExcelFile(path)
        sheet = next((n for n in xl.sheet_names if "路线面板" in n), None)
        if sheet is None:
            missing.append(f"{name}（无路线面板）")
            continue
        d = xl.parse(sheet)
        rows += [
            {
                "dataset": str(r["dataset"]),
                "held": str(r["held"]),
                "种子": s,
                "K": int(r["K"]),
                "路线": str(r["路线"]),
                "F1": float(r["F1"]),
            }
            for _, r in d.iterrows()
        ]
    return pd.DataFrame(rows), missing


def ranking_table(routes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (ds, held, k), sub in routes[routes["K"].isin(B3_K)].groupby(["dataset", "held", "K"], sort=False):
        mean = sub.groupby("路线")["F1"].mean()
        n_seed = sub.groupby("路线")["种子"].nunique()
        if not all(r in mean for r in (*ROUTES_FREE, *ROUTES_LEAK)):
            continue
        free = sorted(ROUTES_FREE, key=lambda r: -float(mean[r]))
        leak = sorted(ROUTES_LEAK, key=lambda r: -float(mean[r]))
        row: dict[str, Any] = {
            "dataset": ds,
            "held": held,
            "K": k,
            "种子数": int(n_seed.min()),
            **{r: round(float(mean[r]), 4) for r in dict.fromkeys((*ROUTES_FREE, *ROUTES_LEAK))},
            "无泄漏第一": free[0],
            "泄漏第一": leak[0],
            "翻转": FAMILY[free[0]] != FAMILY[leak[0]],
            "最佳值抬升": round(float(mean[leak[0]] - mean[free[0]]), 4),
            "SVM超参数泄漏": round(float(mean["SVM(留出域选参)"] - mean["SVM(源域选参)"]), 4),
            "PLS-DA超参数泄漏": round(float(mean["PLS-DA(留出域选参)"] - mean["PLS-DA(源域选参)"]), 4),
            "编码器泄漏": round(float(mean["原型网络(S_src)"] - mean["原型网络(R)"]), 4),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_summary(rank: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, sets in SCOPES.items():
        sub = rank[rank["dataset"].isin(sets)]
        for k in B3_K:
            s = sub[sub["K"] == k]
            if s.empty:
                continue
            flips = s[s["翻转"]]
            row: dict[str, Any] = {
                "口径": scope,
                "K": k,
                "留出域数": len(s),
                "翻转数": len(flips),
                # 翻转后排第一的是哪一类泄漏路线：暴露编码器的原型网络，或在留出域选参的 SVM / PLS-DA
                "翻转后第一为原型网络(S_src)": int((flips["泄漏第一"] == "原型网络(S_src)").sum()),
                "翻转后第一为留出域选参": int(flips["泄漏第一"].str.endswith("(留出域选参)").sum()),
            }
            for col in ("最佳值抬升", "SVM超参数泄漏", "PLS-DA超参数泄漏", "编码器泄漏"):
                row[f"{col}均值"] = round(float(s[col].mean()), 4)
                row[f"{col}为正"] = f"{int((s[col] > 0).sum())}/{len(s)}"
            row["翻转的留出域"] = "; ".join(
                f"{d}/{h}" for d, h in zip(s.loc[s["翻转"], "dataset"], s.loc[s["翻转"], "held"], strict=True)
            )
            rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 四、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="75_leakage_generality_summary：多编码器泄漏与方法选择的汇总")
    parser.add_argument("--stage", default="formal", choices=["formal", "pilot"])
    parser.add_argument("--allow_partial", action="store_true", help="允许缺格（只用于查看进度，不作正式结果）")
    args = parser.parse_args()
    logger = eu.get_logger("75_leakage_generality_summary")
    seeds = [int(s) for s in eu.RECOMMENDED_SEEDS] if args.stage == "formal" else [int(eu.PILOT_SEED)]
    arms, miss_a = load_arms(args.stage, seeds)
    routes, miss_r = load_routes(args.stage, seeds)
    missing = miss_a + miss_r
    if missing:
        logger.log(
            f"缺 {len(missing)} 个逐种子结果表：" + "; ".join(missing[:30]) + (" …" if len(missing) > 30 else "")
        )
        if not args.allow_partial:
            raise SystemExit("结果不全；如只看进度，加 --allow_partial")
    folds = fold_contrasts(arms)
    by_ds = by_dataset_table(folds)
    inf = inference_table(folds)
    rank = ranking_table(routes) if not routes.empty else pd.DataFrame()
    rank_sum = ranking_summary(rank) if not rank.empty else pd.DataFrame()
    logger.log("按数据集 × 编码器族\n" + by_ds.to_string(index=False))
    logger.log("跨数据集推断\n" + inf.to_string(index=False))
    if not rank_sum.empty:
        logger.log("B3 排序翻转\n" + rank_sum.to_string(index=False))
    out = os.path.join(eu.OUTPUT_DIR, f"75_leakage_generality_summary{_suffix(args.stage)}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        folds.to_excel(writer, sheet_name=eu.table_sheet_name(0, "逐留出域对照"), index=False)
        by_ds.to_excel(writer, sheet_name=eu.table_sheet_name(1, "按数据集汇总"), index=False)
        inf.to_excel(writer, sheet_name=eu.table_sheet_name(2, "跨数据集推断"), index=False)
        rank.to_excel(writer, sheet_name=eu.table_sheet_name(3, "B3逐域排序"), index=False)
        rank_sum.to_excel(writer, sheet_name=eu.table_sheet_name(4, "B3汇总"), index=False)
        pd.DataFrame({"缺表": missing}).to_excel(writer, sheet_name=eu.table_sheet_name(5, "缺表"), index=False)
    logger.log(f"汇总表 → {out}")
    logger.close()


if __name__ == "__main__":
    main()
