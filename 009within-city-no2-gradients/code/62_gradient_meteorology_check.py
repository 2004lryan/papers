"""
62_gradient_meteorology_check.py: does the narrowing of the within-city gradient come from interannual variation
in meteorology?

Why it is needed:
    The annual gradient of script 51 is the slope "within one year, ln Y on exposure, with city fixed effects". City
    fixed effects (city-by-year fixed effects in the pooled model) absorb the concentration level common to the whole
    city, so a multiplicative dilution that is uniform across the city cancels out of the log gradient; but if
    meteorology acts differently on high-exposure and low-exposure stations (e.g. wind speed and mixing conditions
    dilute more strongly at stations near sources), it changes the gradient itself. If the 2015–2024 narrowing came
    partly from meteorology, the main text's attribution of the narrowing to the emission side would not hold.

    This script writes the annual gradients of script 51 as a pooled model (city-by-year fixed effects + exposure ×
    year), first reproduces script 51 cell by cell, then adds "exposure × city-year meteorology" interactions to see
    whether the path of the exposure × year coefficients and the size of the 2015→2024 narrowing change:
        M0  exposure × year (as in script 51, with the sample limited to cities with meteorology data)
        M1  M0 + exposure × city-year meteorological levels (each of the five variables standardised)
        M2  M0 + exposure × city-year meteorological anomaly (difference from the city's own ten-year mean)
            + exposure × city ten-year mean
    The meteorological variables are the city annual values computed by script 29 from NASA POWER daily values at
    the city coordinates (temperature, precipitation, wind speed, relative humidity, pressure).
    The 2015–2024 trends of the five meteorological variables, averaged with equal weights over the cities of the NO₂
    sample, are also given: if meteorology itself has no monotonic trend, it cannot explain a monotonic narrowing.

    Uncertainty comes from a city cluster bootstrap (999 draws, the same as the other bootstraps in the manuscript):
    under city-by-year fixed effects the cities are mutually uncorrelated, each city contributes to the normal
    equations only through its own X'X and X'y, and a city drawn k times contributes k times.

Usage:
    python code/62_gradient_meteorology_check.py

Output files:
    outputs/62_梯度气象核验.xlsx (gradient meteorology check)
    logs/62_gradient_meteorology_check.log
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats


def _load(name: str, file: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    if spec is None or spec.loader is None:
        raise SystemExit(f"{file} 无法加载")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export_utils = _load("export_utils", "01_export_utils.py")
m51 = _load("m51", "51_station_timing_and_power.py")   # sample, series and gradient function all from script 51

STA, CITY, TIME, EXPO = m51.STA, m51.CITY, m51.TIME, m51.EXPO
YEARS = list(range(2015, 2025))
B = 999
SEED = 20260925
# segments used in the main text and in script 61
SEGS = [(2015, 2017), (2017, 2020), (2020, 2024), (2017, 2024)]
MET = ["年平均气温_摄氏度", "年降水量_毫米", "年平均风速_米每秒", "年平均相对湿度_百分比", "年平均气压_百帕"]
# the city annual values of script 29 use short names; autonomous prefectures appear there as "<place> Prefecture",
# so each one is mapped individually by its short place name
PREFECTURE_ALIAS = {
    "伊犁哈萨克自治州": "伊犁哈萨克州", "大理白族自治州": "大理州", "巴音郭楞蒙古自治州": "巴音郭楞州",
    "德宏傣族景颇族自治州": "德宏州", "怒江傈僳族自治州": "怒江州", "文山壮族苗族自治州": "文山州",
    "昌吉回族自治州": "昌吉州", "楚雄彝族自治州": "楚雄州", "湘西土家族苗族自治州": "湘西州",
    "甘南藏族自治州": "甘南州", "红河哈尼族彝族自治州": "红河州", "西双版纳傣族自治州": "西双版纳州",
    "迪庆藏族自治州": "迪庆州",
}
Arr = npt.NDArray[np.float64]


def met_name(city: str, known: set[str]) -> str | None:
    """Full city name in the panel → its name in the city annual values of script 29; None if there is no match."""
    if city in known:
        return city
    if city in PREFECTURE_ALIAS:
        return PREFECTURE_ALIAS[city] if PREFECTURE_ALIAS[city] in known else None
    for suf in ("地区", "市", "盟"):
        if city.endswith(suf) and city[: -len(suf)] in known:
            return city[: -len(suf)]
    return None


def design(g: pd.DataFrame, model: str) -> tuple[Arr, Arr, list[str]]:
    """Design matrix and dependent variable after city-year demeaning (the dependent variable column is always _y)."""
    e = g[EXPO].to_numpy(float)
    cols: dict[str, Arr] = {f"暴露×{yr}": e * (g[TIME].to_numpy() == yr) for yr in YEARS}
    if model == "M1":
        for v in MET:
            w = g[v].to_numpy(float)
            cols[f"暴露×{v}"] = e * (w - w.mean()) / w.std()
    elif model == "M2":
        for v in MET:
            anom = g[f"{v}_距平"].to_numpy(float)
            clim = g[f"{v}_城市均值"].to_numpy(float)
            cols[f"暴露×{v}_距平"] = e * (anom - anom.mean()) / anom.std()
            cols[f"暴露×{v}_城市均值"] = e * (clim - clim.mean()) / clim.std()
    x = pd.DataFrame(cols, index=g.index)
    x["_y"] = g["_y"].to_numpy(float)
    x = x - x.groupby(g["城市年"].to_numpy()).transform("mean")
    names = [c for c in x.columns if c != "_y"]
    return x[names].to_numpy(float), x["_y"].to_numpy(float), names


def city_blocks(g: pd.DataFrame, xm: Arr, y: Arr) -> tuple[list[Arr], list[Arr]]:
    """Each city's contribution X'X and X'y to the normal equations (cities are mutually uncorrelated under
    city-by-year fixed effects)."""
    gid = pd.factorize(g[CITY])[0]
    xtx = [xm[gid == k].T @ xm[gid == k] for k in range(gid.max() + 1)]
    xty = [xm[gid == k].T @ y[gid == k] for k in range(gid.max() + 1)]
    return xtx, xty


def fit(g: pd.DataFrame, model: str) -> tuple[pd.DataFrame, list[Arr], list[Arr], Arr]:
    """Coefficients, city-clustered standard errors (with the same G/(G−1) correction as script 51), each city's
    normal-equation blocks and the covariance matrix."""
    xm, y, names = design(g, model)
    xtx, xty = city_blocks(g, xm, y)
    a_inv = np.linalg.pinv(sum(xtx))
    b = a_inv @ sum(xty)
    u = y - xm @ b
    gid = pd.factorize(g[CITY])[0]
    meat = sum(np.outer(xm[gid == k].T @ u[gid == k], xm[gid == k].T @ u[gid == k]) for k in range(gid.max() + 1))
    n_clusters = gid.max() + 1
    cov = a_inv @ np.atleast_2d(meat) @ a_inv * n_clusters / (n_clusters - 1)
    se = np.sqrt(np.diag(cov))
    z = b / se
    out = pd.DataFrame({"项": names, "系数": b, "城市聚类SE": se, "p": 2 * stats.norm.sf(np.abs(z))})
    return out, xtx, xty, cov


def wald(est: pd.DataFrame, cov: Arr, terms: list[str]) -> tuple[float, int, float]:
    """Cluster-robust Wald test that a set of coefficients is jointly zero (χ² approximation): returns the statistic,
    degrees of freedom and p."""
    idx = [i for i, n in enumerate(est["项"]) if str(n) in terms]
    b = est["系数"].to_numpy(float)[idx]
    w = float(b @ np.linalg.pinv(cov[np.ix_(idx, idx)]) @ b)
    return w, len(idx), float(stats.chi2.sf(w, len(idx)))


def seg_rate(coef: Arr, names: list[str], a: int, c: int) -> float:
    """Mean annual change over a segment (gradient units per year), defined as in the main text and script 61."""
    return float((coef[names.index(f"暴露×{c}")] - coef[names.index(f"暴露×{a}")]) / (c - a))


def narrowing(coef: Arr, names: list[str]) -> float:
    """Relative change (%) of the gradient from 2015 to 2024; negative values mean narrowing."""
    b0, b1 = coef[names.index("暴露×2015")], coef[names.index("暴露×2024")]
    return float(100 * (b1 - b0) / b0)


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("62_gradient_meteorology_check")
    d = m51.load()
    met = pd.read_excel(export_utils.OUTPUT_DIR / "29_气象构建过程.xlsx", sheet_name="02_城市年值")
    known = set(met["城市"])
    cities = sorted(d[CITY].unique())
    align = pd.DataFrame({"面板城市": cities, "气象城市": [met_name(c, known) for c in cities]})
    d["气象城市"] = d[CITY].map(dict(zip(align["面板城市"], align["气象城市"], strict=True)))
    clim = met.groupby("城市")[MET].mean().add_suffix("_城市均值")
    met = met.join(clim, on="城市")
    for v in MET:
        met[f"{v}_距平"] = met[v] - met[f"{v}_城市均值"]
    keep = ["城市", "年份", *MET, *[f"{v}_城市均值" for v in MET], *[f"{v}_距平" for v in MET]]
    dm = d.merge(met[keep].rename(columns={"城市": "气象城市"}), on=["气象城市", TIME], how="left")
    unmatched = align[align["气象城市"].isna()]["面板城市"].tolist()
    logger.info("样本 %s 站-年、城市 %s；无气象的城市 %s 个：%s", len(d), len(cities), len(unmatched), unmatched)

    rng = np.random.default_rng(SEED)
    path_rows: list[dict[str, object]] = []
    narrow_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    repro_rows: list[dict[str, object]] = []
    wald_rows: list[dict[str, object]] = []
    seg_rows: list[dict[str, object]] = []
    for tag, ycol in m51.SERIES:
        # ① reproduce the annual gradients of script 51 cell by cell on its full sample
        full = d.dropna(subset=[ycol, EXPO]).copy()
        full["_y"] = full[ycol]
        est0, _, _, _ = fit(full, "M0")
        for yr in YEARS:
            b51 = m51.gradient(d[d[TIME] == yr], ycol)[0]
            b62 = float(est0.loc[est0["项"] == f"暴露×{yr}", "系数"].iloc[0])
            repro_rows.append({"序列": tag, "年份": yr, "51 号梯度": b51, "本脚本 M0（全样本）": b62, "差": b62 - b51})
            if abs(b62 - b51) > 1e-10:
                raise SystemExit(f"{tag} {yr} 未复现 51 号梯度：{b62} vs {b51}")

        # ② compare M0, M1 and M2 on one and the same sample with meteorology data
        g = dm.dropna(subset=[ycol, EXPO, *MET]).copy()
        g["_y"] = g[ycol]
        n_city = g[CITY].nunique()
        counts = rng.multinomial(n_city, np.full(n_city, 1 / n_city), size=B)
        draws: dict[str, Arr] = {}
        point: dict[str, float] = {}
        for model in ["M0", "M1", "M2"]:
            est, xtx, xty, cov = fit(g, model)
            names = est["项"].tolist()
            coef = est["系数"].to_numpy(float)
            blocks = {"M1": {"气象水平": [f"暴露×{v}" for v in MET]},
                      "M2": {"气象距平": [f"暴露×{v}_距平" for v in MET],
                             "城市气候均值": [f"暴露×{v}_城市均值" for v in MET]}}.get(model, {})
            for lab, terms in blocks.items():
                w, k, pw = wald(est, cov, terms)
                wald_rows.append({"序列": tag, "模型": model, "交互项组": lab, "Wald χ²": w, "自由度": k, "p": pw})
            for _, r in est.iterrows():
                if str(r["项"]).startswith("暴露×") and str(r["项"])[3:].isdigit():
                    path_rows.append({"序列": tag, "模型": model, "年份": int(str(r["项"])[3:]),
                                      "城内梯度": r["系数"], "城市聚类SE": r["城市聚类SE"], "N": len(g)})
                else:
                    coef_rows.append({"序列": tag, "模型": model, "交互项": r["项"], "系数": r["系数"],
                                      "城市聚类SE": r["城市聚类SE"], "p": r["p"]})
            xtx_arr, xty_arr = np.stack(xtx), np.stack(xty)
            boot = np.empty(B)
            seg_boot = np.empty((B, len(SEGS)))
            for i in range(B):
                bb = np.linalg.solve(np.tensordot(counts[i], xtx_arr, axes=1), counts[i] @ xty_arr)
                boot[i] = narrowing(bb, names)
                seg_boot[i] = [seg_rate(bb, names, a, c) for a, c in SEGS]
            for j, (a, c) in enumerate(SEGS):
                slo, shi = np.percentile(seg_boot[:, j], [2.5, 97.5])
                seg_rows.append({"序列": tag, "模型": model, "分段": f"{a}–{c}",
                                 "平均年变化（梯度单位/年）": seg_rate(coef, names, a, c),
                                 "bootstrap 95% 下限": float(slo), "bootstrap 95% 上限": float(shi)})
            draws[model] = boot
            point[model] = narrowing(coef, names)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            narrow_rows.append({"序列": tag, "模型": model, "N（站-年）": len(g), "城市": n_city,
                                "2015 梯度": coef[names.index("暴露×2015")],
                                "2024 梯度": coef[names.index("暴露×2024")],
                                "2015→2024 相对变化%": narrowing(coef, names),
                                "bootstrap 95% 下限": float(lo), "bootstrap 95% 上限": float(hi)})
        for model in ["M1", "M2"]:
            diff = draws[model] - draws["M0"]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            narrow_rows.append({"序列": tag, "模型": f"{model} − M0",
                                "2015→2024 相对变化%": point[model] - point["M0"],
                                "bootstrap 95% 下限": float(lo), "bootstrap 95% 上限": float(hi)})

    # ③ ten-year trend of the meteorology itself: equal-weight annual mean over the main-sample cities (NO₂ series)
    no2_cities = dm.dropna(subset=["ln_NO2", EXPO, *MET])["气象城市"].unique()
    mt = met[met["城市"].isin(no2_cities)].groupby("年份")[MET].mean()
    trend_rows: list[dict[str, object]] = []
    for v in MET:
        x = mt.index.to_numpy(float)
        yv = mt[v].to_numpy(float)
        ols = stats.linregress(x, yv)
        ts = stats.theilslopes(yv, x)
        trend_rows.append({"变量": v, "城市数": len(no2_cities), "2015 均值": yv[0], "2024 均值": yv[-1],
                           "OLS 斜率/十年": 10 * ols.slope, "OLS p": ols.pvalue,
                           "Theil–Sen 斜率/十年": 10 * ts.slope,
                           "Theil–Sen 95% 下限/十年": 10 * ts.low_slope, "Theil–Sen 95% 上限/十年": 10 * ts.high_slope})

    narrow = pd.DataFrame(narrow_rows)
    for _, r in narrow.iterrows():
        logger.info("%s %s 2015→2024 相对变化 %.2f%%（95%% %.2f, %.2f）", r["序列"], r["模型"],
                    r["2015→2024 相对变化%"], r["bootstrap 95% 下限"], r["bootstrap 95% 上限"])
    for r in trend_rows:
        logger.info("%s OLS 斜率/十年 %.4f（p=%.3f）", r["变量"], r["OLS 斜率/十年"], r["OLS p"])
    for r in wald_rows:
        logger.info("%s %s %s 联合检验 χ²=%.2f（df=%s）p=%.3f", r["序列"], r["模型"], r["交互项组"],
                    r["Wald χ²"], r["自由度"], r["p"])
    for r in seg_rows:
        if r["序列"] == "NO2":
            logger.info("NO2 %s %s 平均年变化 %.4f（95%% %.4f, %.4f）", r["模型"], r["分段"],
                        r["平均年变化（梯度单位/年）"], r["bootstrap 95% 下限"], r["bootstrap 95% 上限"])
    out = export_utils.write_excel_workbook("62_梯度气象核验", [
        ("收窄幅度", narrow),
        ("分段速率", pd.DataFrame(seg_rows)),
        ("联合检验", pd.DataFrame(wald_rows)),
        ("逐年梯度", pd.DataFrame(path_rows)),
        ("气象交互系数", pd.DataFrame(coef_rows)),
        ("气象十年趋势", pd.DataFrame(trend_rows)),
        ("51号复现", pd.DataFrame(repro_rows)),
        ("城市名称对齐", align),
    ])
    print(f"✓ 梯度气象核验完成\n  输出: {out}\n  日志: {log_path}")


if __name__ == "__main__":
    main()
