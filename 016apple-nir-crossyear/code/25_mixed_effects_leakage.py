"""25_mixed_effects_leakage.py: 跨数据集泄漏的混合效应统计（回应 n=3 伪重复）。

读 24_crossdataset_leakage.xlsx 长格式，把所有(数据集×留一域×种子)折的 Delta 汇总，
并入苹果基准的逐折 Δ（64 号协议对齐对照的 S_src − R，与外部数据集同一对比口径），做：
(1) 混合效应 Delta ~ 1 + (1|数据集)：固定截距=总体泄漏量，数据集随机效应吸收异质；
(2) Delta ~ z(MMD^2) 混合效应回归（数据集随机截距）：检验"漂移越大泄漏越大"跨数据集；
(3) cluster-bootstrap（簇=数据集，≥1000）给固定截距 95% CI；
(4) 符号检验：Delta>0 折占比 + 二项检验 p。

运行方式:
    cd <project root>
    python 02code/25_mixed_effects_leakage.py
    # 敏感性：用 24 号脚本某个带后缀工作簿（如 cap0、buf10-indianpines_spatial）里的数据集替换主结果中的同名数据集
    python 02code/25_mixed_effects_leakage.py --variant buf10_indianpines_spatial

苹果逐折 Δ 的来源:
    04outputs/64_leakage_controls.xlsx 的「对照汇总」（three_year / canonical / α=0，Formal 5 种子）。
    该表不存在或种子不全时，退用 formal_seeds/ 下的 pilot 逐种子表，输出文件名加 __preview，
    不覆盖正式结果。

输出文件:
    04outputs/25_mixed_effects_leakage.xlsx  — 混合效应系数/回归/bootstrap/符号检验（含「苹果来源」表）
    04outputs/25_mixed_effects_leakage__preview.xlsx — 苹果 Δ 取自 pilot 时的预览
    05logs/25_mixed_effects_leakage_YY-MM-DD_HHMMSS.log — 运行日志
"""

from __future__ import annotations

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

import export_utils as eu

warnings.filterwarnings("ignore")

SRC_XLSX = os.path.join(eu.OUTPUT_DIR, "24_crossdataset_leakage.xlsx")
COLS = ["dataset", "drift_axis", "modality", "held", "seed", "shared", "fold_restricted", "delta", "mmd2"]


LEAK64_XLSX = os.path.join(eu.OUTPUT_DIR, "64_leakage_controls.xlsx")


# 苹果逐折 Δ 与外部数据集同一口径：编码器训练池含 / 不含目标年带标签光谱（S_src − R），
# 其余设置全部对齐（64 号脚本）。OliveOil 为 IID 序号划分的记忆化上限，只进稳健性口径。
def _apple(held: str, delta: float) -> dict[str, object]:
    return {"dataset": "apple_year", "drift_axis": "年份", "modality": "NIR点谱",
            "held": held, "delta": delta, "mmd2": np.nan}


def apple_folds(logger: eu.Logger) -> tuple[list[dict[str, object]], str]:
    """苹果三折的 Δ_rep（S_src − R，three_year / canonical / α=0，跨种子均值）及其来源标记。"""
    if os.path.exists(LEAK64_XLSX):
        t = pd.read_excel(LEAK64_XLSX, sheet_name=0)
        sub = t[(t["设计"] == "three_year") & (t["变体"] == "canonical") & (t["src_alpha"] == 0)
                & (t["目标年份"].astype(str) != "均值")]
        if len(sub) == 3 and (sub["种子数"] == len(eu.RECOMMENDED_SEEDS)).all():
            rows = [_apple(str(int(r["目标年份"])), float(r["Δ_rep"])) for _, r in sub.iterrows()]
            logger.log(f"[苹果] 取自 {os.path.basename(LEAK64_XLSX)}（Formal 5 种子）")
            return rows, "formal"
    files = sorted(glob.glob(os.path.join(
        eu.OUTPUT_DIR, "formal_seeds", "64_leakage__three_year__canonical__seed*__pilot.xlsx")))
    formal_seed = [f for f in files if any(f"__seed{s}__" in f for s in eu.RECOMMENDED_SEEDS)]
    files = formal_seed or files
    if not files:
        raise FileNotFoundError("缺 64 号脚本的苹果对照结果（正式汇总与 pilot 逐种子表都没有）")
    d = pd.concat([pd.read_excel(f, sheet_name=1) for f in files], ignore_index=True)
    d = d[d["src_alpha"] == 0]
    per_fold = d.groupby("目标年份")["Δ_rep"].mean()
    logger.log(f"[苹果·预览] 正式汇总缺失或种子不全，退用 {len(files)} 个 pilot 逐种子表")
    return [_apple(str(int(k)), float(v)) for k, v in per_fold.items()], f"pilot({len(files)} seeds)"


OLIVE_FOLDS = [{"dataset": "oliveoil_country", "drift_axis": "国别", "modality": "FT-IR",
                "held": "all", "delta": 0.178, "mmd2": np.nan}]


def _read_long(path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = next(s for s in xl.sheet_names if "长格式" in s)
    out: pd.DataFrame = xl.parse(sheet)[COLS]
    return out


def load_long(
    apple: list[dict[str, object]], include_olive: bool = False, variant: str | None = None
) -> pd.DataFrame:
    r"""读 24 长格式记录并\emph{聚合到折级均值}，并入苹果（及可选 OliveOil）折级 Δ。

    关键：24 长格式是\emph{种子级}（每折 5 个种子行），而苹果/OliveOil 注入的是\emph{折级}均值 Δ。
    若直接混合会使观测单位不一致、种子级重复主导混合效应截距。故先把 24 的种子级记录按折
    (dataset,held) 聚合为折级均值（每折 1 行、跨种子平均），再与苹果/OliveOil 折级 Δ 合并，
    每折等权，与 sign_test/cluster_bootstrap 的折级口径一致。

    include_olive=False（默认，本文主口径）：仅并入苹果，得\textbf{5 个真实域划分数据集、16 折}
        （苹果+玉米+药片+芒果+Indian Pines）。橄榄油为 UCR 官方 IID 序号划分（无域漂移，共享臂
        记忆化使宏 F1 恒为 1.000，见正文 2.5 节），\textbf{不计入域漂移证据链}，故默认剔除。
    include_olive=True（稳健性对照）：额外并入橄榄油折，得 6 数据集、17 折，用于说明"即便并入
        记忆化上限，总体泄漏结论仍不变"。
    """
    base_seed = _read_long(SRC_XLSX)
    if variant:
        var_seed = _read_long(os.path.join(eu.OUTPUT_DIR, f"24_crossdataset_leakage__{variant}.xlsx"))
        base_seed = pd.concat(
            [base_seed[~base_seed["dataset"].isin(var_seed["dataset"].unique())], var_seed], ignore_index=True
        )
    base: pd.DataFrame = base_seed.groupby(
        ["dataset", "drift_axis", "modality", "held"], as_index=False
    ).agg(
        seed=("seed", "first"), shared=("shared", "mean"),
        fold_restricted=("fold_restricted", "mean"), delta=("delta", "mean"), mmd2=("mmd2", "mean"),
    )[COLS]
    extra = pd.DataFrame(apple + OLIVE_FOLDS if include_olive else apple)
    for c in ["seed", "shared", "fold_restricted"]:
        extra[c] = np.nan
    df: pd.DataFrame = pd.concat([base, extra[COLS]], ignore_index=True)
    return df


def fit_intercept_model(df: pd.DataFrame, logger: eu.Logger) -> dict[str, float]:
    """混合效应 Delta ~ 1 + (1|dataset)：估计总体泄漏截距与随机效应方差。"""
    md = MixedLM.from_formula("delta ~ 1", groups="dataset", data=df)
    res = md.fit(reml=True, method="lbfgs")
    intercept = float(res.fe_params["Intercept"])
    se = float(res.bse_fe["Intercept"])
    ci_lo, ci_hi = intercept - 1.96 * se, intercept + 1.96 * se
    group_var = float(res.cov_re.iloc[0, 0]) if res.cov_re.size else 0.0
    logger.log(
        f"[混合效应·截距] 总体泄漏 Delta={intercept:.4f} (SE={se:.4f}, "
        f"95%CI[{ci_lo:.4f},{ci_hi:.4f}]), 数据集随机效应方差={group_var:.5f}"
    )
    return {
        "intercept": intercept, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "group_var": group_var, "p": float(res.pvalues["Intercept"]),
    }


def fit_drift_model(df: pd.DataFrame, logger: eu.Logger) -> dict[str, float]:
    """混合效应 Delta ~ z(MMD^2) + (1|dataset)：漂移-泄漏关系（MMD 数据集内标准化消尺度）。"""
    d = df.dropna(subset=["mmd2"]).copy()  # 仅含 MMD 的数据集(苹果/olive 无逐折 MMD，跳过)
    d["z_mmd"] = d.groupby("dataset")["mmd2"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
    res = MixedLM.from_formula("delta ~ z_mmd", groups="dataset", data=d).fit(reml=True, method="lbfgs")
    slope = float(res.fe_params["z_mmd"])
    se = float(res.bse_fe["z_mmd"])
    lo, hi = slope - 1.96 * se, slope + 1.96 * se
    p = float(res.pvalues["z_mmd"])
    logger.log(f"[混合效应·漂移] Delta~z(MMD): 斜率={slope:+.4f} (SE={se:.4f}, 95%CI[{lo:+.4f},{hi:+.4f}], p={p:.4f})")
    return {"slope": slope, "se": se, "ci_lo": lo, "ci_hi": hi, "p": p}


def cluster_bootstrap(df: pd.DataFrame, n_boot: int, seed: int, logger: eu.Logger) -> dict[str, float]:
    """簇=数据集的 cluster-bootstrap，给总体泄漏截距 95% CI（折级均值再按数据集聚合）。"""
    rng = np.random.default_rng(seed)
    fold_means = df.groupby(["dataset", "held"])["delta"].mean().reset_index()
    datasets = fold_means["dataset"].unique()
    boot: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(datasets, size=len(datasets), replace=True)
        vals = [fold_means[fold_means["dataset"] == ds]["delta"].mean() for ds in pick]
        boot.append(float(np.mean(vals)))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    logger.log(f"[cluster-bootstrap] 数据集级泄漏均值 95%CI=[{lo:.4f},{hi:.4f}] ({n_boot}次, 簇=数据集)")
    return {"boot_mean": float(np.mean(boot)), "ci_lo": lo, "ci_hi": hi}


def sign_test(df: pd.DataFrame, logger: eu.Logger) -> dict[str, float]:
    """折级 Delta>0 占比 + 二项检验（H0: p=0.5）。"""
    fold_means = df.groupby(["dataset", "held"])["delta"].mean()
    n_pos = int((fold_means > 0).sum())
    n_tot = len(fold_means)
    p_binom = float(stats.binomtest(n_pos, n_tot, 0.5, alternative="greater").pvalue)
    # 更保守：数据集级(独立单位)符号检验
    ds_means = df.groupby("dataset")["delta"].mean()
    ds_pos = int((ds_means > 0).sum())
    ds_tot = len(ds_means)
    p_ds = float(stats.binomtest(ds_pos, ds_tot, 0.5, alternative="greater").pvalue)
    logger.log(f"[符号检验·折级] Delta>0 折={n_pos}/{n_tot}={n_pos / n_tot:.0%}, 二项 p={p_binom:.2e}")
    logger.log(f"[符号检验·数据集级] Delta>0 数据集={ds_pos}/{ds_tot}, 二项 p={p_ds:.3f}")
    return {"n_pos": n_pos, "n_tot": n_tot, "pos_frac": n_pos / n_tot, "p_binom": p_binom,
            "ds_pos": ds_pos, "ds_tot": ds_tot, "p_ds": p_ds}


def main() -> None:
    parser = argparse.ArgumentParser(description="跨数据集泄漏的混合效应统计")
    parser.add_argument("--variant", default="", help="24 号脚本敏感性工作簿的后缀；空 = 主结果")
    args = parser.parse_args()
    logger = eu.get_logger("25_mixed_effects_leakage")
    if not os.path.exists(SRC_XLSX):
        logger.log(f"缺 {SRC_XLSX}，请先运行 24_crossdataset_leakage.py")
        raise SystemExit(1)
    if args.variant:
        logger.log(f"[敏感性] 用 24_crossdataset_leakage__{args.variant}.xlsx 中的数据集替换主结果中的同名数据集")
    apple, apple_src = apple_folds(logger)
    df = load_long(apple, include_olive=False, variant=args.variant or None)  # 主口径：5 真实域数据集
    df_all = load_long(apple, include_olive=True, variant=args.variant or None)  # 稳健性：6 数据集（并入橄榄油）
    n_folds = df.groupby(["dataset", "held"]).ngroups
    logger.log(f"[主口径·真实域] {len(df)} 折级记录，{df['dataset'].nunique()} 数据集，{n_folds} 折（不含橄榄油）")

    inter = fit_intercept_model(df, logger)
    drift = fit_drift_model(df, logger)
    boot = cluster_bootstrap(df, 2000, 20060515, logger)
    sign = sign_test(df, logger)

    # 稳健性对照：并入橄榄油（记忆化上限）后的 6 数据集总体泄漏截距 + 符号检验
    logger.log("[稳健性·含橄榄油] 6 数据集（额外并入 IID 划分的橄榄油记忆化上限 0.178）：")
    inter_all = fit_intercept_model(df_all, logger)
    sign_all = sign_test(df_all, logger)

    per_ds = df.groupby(["dataset", "drift_axis", "modality"]).agg(
        n_folds=("held", "nunique"), delta_mean=("delta", "mean"), delta_std=("delta", "std"),
    ).reset_index()

    me_rows = [
        {"模型": "Delta~1+(1|数据集) 主口径5真实域", "估计": inter["intercept"], "SE": inter["se"],
         "CI下": inter["ci_lo"], "CI上": inter["ci_hi"], "p": inter["p"]},
        {"模型": "Delta~z(MMD)斜率", "估计": drift["slope"], "SE": drift["se"],
         "CI下": drift["ci_lo"], "CI上": drift["ci_hi"], "p": drift["p"]},
        {"模型": "cluster-bootstrap截距 5真实域", "估计": boot["boot_mean"], "SE": np.nan,
         "CI下": boot["ci_lo"], "CI上": boot["ci_hi"], "p": np.nan},
        {"模型": "Delta~1+(1|数据集) 稳健性6含橄榄油", "估计": inter_all["intercept"], "SE": inter_all["se"],
         "CI下": inter_all["ci_lo"], "CI上": inter_all["ci_hi"], "p": inter_all["p"]},
    ]
    sign_primary = {"口径": "5真实域(不含橄榄油)", **sign}
    sign_robust = {"口径": "6含橄榄油(稳健性)", **sign_all}
    sheets: dict[object, object] = {
        eu.table_sheet_name(1, "混合效应汇总"): pd.DataFrame(me_rows),
        eu.table_sheet_name(2, "符号检验"): pd.DataFrame([sign_primary, sign_robust]),
        eu.table_sheet_name(3, "各数据集"): per_ds,
        eu.table_sheet_name(4, "苹果来源"): pd.DataFrame(
            [{"折": a["held"], "Δ_rep": a["delta"], "来源": apple_src} for a in apple]),
    }
    stem = os.path.splitext(os.path.basename(__file__))[0]
    tags = [t for t in (args.variant, "" if apple_src == "formal" else "preview") if t]
    out = eu.write_script_workbook(f"{stem}__{'__'.join(tags)}.py" if tags else __file__, sheets)
    logger.log(f"已写 {out}")


if __name__ == "__main__":
    main()
