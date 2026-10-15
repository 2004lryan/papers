"""
55_trend_collinearity_check.py: common-trend specification (S16) — collinearity diagnostics and cross-implementation

Equation (4) of the main text adds the exposure-specific linear trend t×exposure alongside the electrification
interaction lnNEV(c,t)×exposure, and the interaction then drops to zero. The new energy vehicle (NEV) stock rises
almost monotonically over time, so the first question readers will ask is whether the interaction vanishes because
the two regressors are collinear and cancel each other out.
Answering this question takes three things; this script provides all three, with numbers that can be checked:
    1. Collinearity diagnostics: after partialling out the station and city-by-year fixed effects, the correlation
       between the two regressors, the variance inflation factor, and how much of the interaction's variance and of
       its standard deviation is orthogonal to the trend.
    2. Cross-implementation: the same estimator is computed once with each of three algorithms,
       and the coefficients must agree, ruling out "the zero is a numerical problem of one particular software package".
           A  linearmodels.PanelOLS (the same implementation as in script 48)
           B  hand-written alternating projections partialling out both sets of fixed effects + least squares
              + city-clustered standard errors
           C  Frisch–Waugh: first orthogonalise the interaction with respect to the trend term and the fixed effects,
              then regress on the orthogonal part alone
       A and B are independent of each other; C uses only the orthogonal part of B's partialled-out data, so it checks
       the algebra rather than the software.
    3. Year-dummy specification: the linear trend is replaced by a full set of year dummies × exposure (no trend shape
       imposed), estimated once for NO₂ and once for the NO₂ peak-to-trough ratio, with the same implementation as the
       other columns of Table 2 (linearmodels, clustered by city).

The sample is built step by step as in the script-48 baseline regression (NO₂ valid days ≥274 → city-years with ≥2
stations → non-missing regressors), and the sample size is asserted to equal that of script 48; on a mismatch the
script exits with an error and outputs no numbers.

Usage:
    python code/55_trend_collinearity_check.py

Output files:
    outputs/55_趋势共线性与交叉实现.xlsx (trend collinearity and cross-implementation)
        — sheets 01_01_共线性诊断 / 02_02_交叉实现 / 03_03_标准误放大 / 04_04_年份哑变量规格
          (collinearity diagnostics / cross-implementation / standard-error inflation / year-dummy specification;
          the index prefix appears twice because the titles passed to write_excel_workbook already carry one)
    logs/55_trend_collinearity_check.log

Column names and log messages label the baseline and common-trend specifications `式 (2)` and `式 (3)`; in the
manuscript they are equations (3) and (4).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from linearmodels.panel import PanelOLS

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
if _SPEC is None or _SPEC.loader is None:
    raise SystemExit("01_export_utils.py 无法加载")
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

Arr = npt.NDArray[np.float64]
STA, CITY, TIME = "站点编号", "所属城市", "年份"
MIN_DAYS = 274
# Sample size of the script-48 baseline regression: station-years, stations, cities
EXPECTED = (7325, 1251, 278)
X_NEV, X_TREND, Y = "NEVxExpo", "tx暴露", "ln_NO2"


def demean(z: Arr, groups: list[npt.NDArray[np.intp]], tol: float = 1e-13, max_iter: int = 20000) -> Arr:
    """Alternating projections: subtract each group's mean in turn until convergence, giving the residuals after
    partialling out several sets of fixed effects."""
    out = z.copy()
    counts = [np.bincount(g).astype(float) for g in groups]
    for _ in range(max_iter):
        prev = out.copy()
        for g, n in zip(groups, counts, strict=True):
            for j in range(out.shape[1]):
                out[:, j] -= (np.bincount(g, weights=out[:, j]) / n)[g]
        if float(np.max(np.abs(out - prev))) < tol:
            return out
    raise SystemExit("交替投影未收敛")


def cluster_se(xm: Arr, u: Arr, g: npt.NDArray[np.intp]) -> Arr:
    """City-clustered sandwich standard errors, with the same small-sample factor as the t statistic used in the
    wild bootstrap of script 48.

    The degrees-of-freedom adjustment subtracts only the number of regressors, not the number of partialled-out fixed
    effects, so these standard errors are smaller than the analytic ones from linearmodels; the main text reports the
    linearmodels one.
    """
    bread = np.linalg.inv(xm.T @ xm)
    k = xm.shape[1]
    meat = np.zeros((k, k))
    for gid in np.unique(g):
        s = xm[g == gid].T @ u[g == gid]
        meat += np.outer(s, s)
    n_cl, n = len(np.unique(g)), len(u)
    v = bread @ meat @ bread * (n_cl / (n_cl - 1)) * ((n - 1) / (n - k))
    return np.sqrt(np.diag(v))


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("55_trend_collinearity_check")
    p = pd.read_csv(export_utils.DATA_DIR / "站点年面板-2015_2024.csv")
    p.columns = [c.lstrip("﻿") for c in p.columns]

    v = p["NO2_年均"].where((p["NO2_年均"] > 0) & (p["NO2_有效日"] >= MIN_DAYS))
    p[Y] = np.log(v)
    p["城市年"] = p[CITY].astype(str) + "_" + p[TIME].astype(str)
    # Exposure centring and the time origin both use the full panel, as in script 48; both change only terms that
    # the station and city-by-year fixed effects absorb
    expo = p["建成区占比_500m"] - p["建成区占比_500m"].mean()
    p["lnNEV"] = np.log(p["NEV保有量_辆"].where(p["NEV保有量_辆"] > 0))
    p[X_NEV] = p["lnNEV"] * expo
    p[X_TREND] = (p[TIME] - p[TIME].min()) * expo
    ratio = p["NO2_峰谷比"].where((p["NO2_峰谷比"] > 0) & (p["NO2_有效日"] >= MIN_DAYS))
    p["ln_NO2峰谷比"] = np.log(ratio)
    valid = p.dropna(subset=[Y])
    p["城内站数"] = p["城市年"].map(valid.groupby("城市年")[STA].nunique())
    d = p[p["城内站数"] >= 2].dropna(subset=[Y, X_NEV, X_TREND]).reset_index(drop=True)

    got = (len(d), d[STA].nunique(), d[CITY].nunique())
    logger.info("回归样本：站点年 %s，站点 %s，城市 %s（48 号基准为 %s）", *got, EXPECTED)
    if got != EXPECTED:
        raise SystemExit(f"样本与 48 号基准回归不一致：得到 {got}，应为 {EXPECTED}。不输出任何数。")

    g_sta = pd.factorize(d[STA])[0]
    g_ct = pd.factorize(d["城市年"])[0]
    g_city = pd.factorize(d[CITY])[0]
    z = demean(d[[Y, X_NEV, X_TREND]].to_numpy(float), [g_sta, g_ct])
    y, x_nev, x_trend = z[:, 0], z[:, 1], z[:, 2]

    # —— Collinearity diagnostics ——
    r = float(np.corrcoef(x_nev, x_trend)[0, 1])
    coll = pd.DataFrame([{
        "偏出固定效应后两回归量的相关 r": round(r, 4),
        "方差膨胀因子 VIF = 1/(1−r²)": round(1 / (1 - r * r), 2),
        "交互项方差中与趋势正交的比例 1−r²": round(1 - r * r, 4),
        "交互项标准差中与趋势正交的比例 √(1−r²)": round(float(np.sqrt(1 - r * r)), 4),
    }])
    logger.info("偏出固定效应后 r=%.4f，VIF=%.2f，正交方差占比 %.4f，正交标准差占比 %.4f",
                r, 1 / (1 - r * r), 1 - r * r, np.sqrt(1 - r * r))

    # —— Cross-implementation ——
    pdf = d.set_index([STA, TIME])
    res_a = PanelOLS(pdf[Y], pdf[[X_NEV, X_TREND]], entity_effects=True, other_effects=pdf[["城市年"]],
                     drop_absorbed=True, check_rank=False).fit(cov_type="clustered", clusters=pdf[[CITY]])
    res_base = PanelOLS(pdf[Y], pdf[[X_NEV]], entity_effects=True, other_effects=pdf[["城市年"]],
                        drop_absorbed=True, check_rank=False).fit(cov_type="clustered", clusters=pdf[[CITY]])

    xb = np.column_stack([x_nev, x_trend])
    b_b, *_ = np.linalg.lstsq(xb, y, rcond=None)
    se_b = cluster_se(xb, y - xb @ b_b, g_city)

    # Frisch–Waugh: orthogonalise the interaction with respect to the trend term and use only the orthogonal part
    x_perp = x_nev - x_trend * float(x_trend @ x_nev / (x_trend @ x_trend))
    b_c = float(x_perp @ y / (x_perp @ x_perp))

    rows = [
        {"实现": "A linearmodels.PanelOLS", "β_NEV×暴露": float(res_a.params[X_NEV]),
         "SE_NEV×暴露": float(res_a.std_errors[X_NEV]), "β_t×暴露": float(res_a.params[X_TREND])},
        {"实现": "B 交替投影 + 最小二乘 + 城市聚类", "β_NEV×暴露": float(b_b[0]),
         "SE_NEV×暴露": float(se_b[0]), "β_t×暴露": float(b_b[1])},
        {"实现": "C Frisch–Waugh 正交部分单独回归", "β_NEV×暴露": b_c,
         "SE_NEV×暴露": float("nan"), "β_t×暴露": float("nan")},
    ]
    impl = pd.DataFrame(rows)
    spread = float(impl["β_NEV×暴露"].max() - impl["β_NEV×暴露"].min())
    impl["与 A 的系数差"] = impl["β_NEV×暴露"] - rows[0]["β_NEV×暴露"]
    for rw in rows:
        logger.info("  %-24s β_NEV×暴露=%.10f  SE=%.5f  β_t×暴露=%.6f", rw["实现"], rw["β_NEV×暴露"],
                    rw["SE_NEV×暴露"], rw["β_t×暴露"])
    logger.info("三种实现的 β_NEV×暴露 极差 %.2e", spread)

    se_base = float(res_base.std_errors[X_NEV])
    inflate = pd.DataFrame([{
        "基准式 (2) SE": round(se_base, 5), "共同趋势式 (3) SE": round(float(res_a.std_errors[X_NEV]), 5),
        "SE 放大倍数": round(float(res_a.std_errors[X_NEV]) / se_base, 2),
        "式 (3) 的最小可检测效应 2.8·SE": round(2.8 * float(res_a.std_errors[X_NEV]), 4),
    }])
    logger.info("SE：式 (2) %.5f → 式 (3) %.5f（×%.2f）；式 (3) 的 MDE=%.4f", se_base,
                res_a.std_errors[X_NEV], float(res_a.std_errors[X_NEV]) / se_base,
                2.8 * float(res_a.std_errors[X_NEV]))

    # —— Year-dummy specification: the window is the years with vehicle-stock data; the first year is the base ——
    base = p[p["城内站数"] >= 2].dropna(subset=["lnNEV"]).copy()
    yd_rows = []
    for yname in (Y, "ln_NO2峰谷比"):
        e = base.dropna(subset=[yname]).reset_index(drop=True)
        cols = [X_NEV]
        for yr in sorted(e[TIME].unique())[1:]:
            e[f"y{yr}x暴露"] = (e[TIME] == yr).astype(float) * (e["建成区占比_500m"] - p["建成区占比_500m"].mean())
            cols.append(f"y{yr}x暴露")
        ep = e.set_index([STA, TIME])
        ry = PanelOLS(ep[yname], ep[cols], entity_effects=True, other_effects=ep[["城市年"]],
                      drop_absorbed=True, check_rank=False).fit(cov_type="clustered", clusters=ep[[CITY]])
        yd_rows.append({"结果变量": yname, "N": int(ry.nobs), "β_NEV×暴露": round(float(ry.params[X_NEV]), 6),
                        "SE": round(float(ry.std_errors[X_NEV]), 5), "p": round(float(ry.pvalues[X_NEV]), 4)})
        logger.info("年份哑变量规格 %s：β_NEV×暴露=%.6f（SE %.5f，p=%.4f），N=%s", yname,
                    ry.params[X_NEV], ry.std_errors[X_NEV], ry.pvalues[X_NEV], int(ry.nobs))

    out = export_utils.write_excel_workbook("55_趋势共线性与交叉实现",
                                            [("01_共线性诊断", coll), ("02_交叉实现", impl),
                                             ("03_标准误放大", inflate), ("04_年份哑变量规格", pd.DataFrame(yd_rows))])
    logger.info("已写出 %s；日志 %s", out, log_path)


if __name__ == "__main__":
    main()
