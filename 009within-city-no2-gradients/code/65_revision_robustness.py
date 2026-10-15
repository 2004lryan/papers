"""
65_revision_robustness.py: five further robustness checks that can be answered with the data already at hand

What it checks:
    This script answers five questions item by item: whether the 500 m built-up fraction behaves as traffic
    exposure, given that the SO₂ gradient narrows faster and the SO₂ electrification interaction is also larger; what
    the standard error and bootstrap p of the trend term are in the regressions with the trend; whether the results
    hold under alternative definitions of the fleet data; whether the city cross-sectional correlation of Figure S2
    survives a control for the base-year gradient (mean reversion); and whether the fast segment 2017–2020 rests on
    the pandemic year. Blocks ① and ④ state how their results are to be read; the repository holds no dated record
    that these rules preceded the runs, and the manuscript reports the results whichever way they fall.

    ① Traffic fingerprint of the built-up axis (same-city, same-year cross-section, 2015–2024 pooled):
       for NO₂, SO₂, CO and PM2.5 separately, regress the ln annual mean concentration, the ln commuting-peak to
       overnight ratio (commuting peaks at hours 7–9 and 17–20 ÷ overnight hours 1–4) and the weekday−weekend
       relative difference, as well as ln annual mean O₃ and ln Ox, on the built-up fraction, absorbing city-by-year
       fixed effects and clustering by city; the slope differences NO₂ − SO₂, NO₂ − CO and NO₂ − PM2.5 (ln annual
       mean and ln commuting-peak to overnight ratio) get intervals from a whole-city bootstrap (999 draws).
       Interpretation: if the slope of the NO₂ commuting-peak to overnight ratio is positive (p<0.05) and larger
       than that of SO₂ (the interval of the difference excludes 0), and the O₃ slope is negative (p<0.05,
       titration), then the built-up axis carries a traffic-type diurnal cycle and a titration signature for NO₂
       that SO₂ does not have; otherwise the built-up axis can only be called a degree of urban coreness, and the
       paper no longer calls it traffic exposure anywhere.
    ② Alternative definitions of the treatment: provincial NEV vehicle stock (Ministry of Public Security
       registration basis, differing from the national total by ≤1%), provincial NEV share of civilian vehicles
       (in levels), dropping the city with an inferred value (Mudanjiang), and dropping 2023 (transcribed from
       screenshots); each runs Eq. (3), Eq. (4) and the year-indicator specification, and provincial treatments are
       clustered by province. For each definition the script gives the coefficient implied by "the narrowing is
       entirely due to the fleet", the power against it, and the minimum detectable effect under the year-indicator
       specification. The implied coefficient is computed on the definition's own regression sample: the difference
       between the within-city gradients of the first and last years, divided by the difference in the mean of the
       treatment under the same weights as the gradient (within-city sum of squared exposure deviations) (the last
       year of the sample without 2023 is 2022).
    ③ Inference under collinearity: standard errors and wild cluster bootstrap p values of the two interaction terms
       of Eq. (4); the bootstrap p of the year-indicator specification; and the coefficient interval converted into
       the share of the narrowing that the fleet can explain.
       Full attribution (coefficient equal to the implied coefficient) also gets a restricted wild cluster bootstrap
       p: the provincial definitions have only 31 clusters, so the analytic p may be too small.
    ④ Controls for the Figure S2 cross-section: city gradient change (2017→2023) on the log growth of the fleet,
       controlling in turn for the 2017 city gradient, then also for priority region and the log number of stations,
       then also for the log 2017 fleet; also the Spearman correlation
       and a run without Mudanjiang. Interpretation: if after the controls the coefficient on fleet growth is still
       not negative and is significant, then "cities whose fleets grew faster narrowed less" is not produced by mean
       reversion.
    ⑤ Year-by-year decomposition of the fast segment 2017–2020 and the pandemic year: each year's decline as a share
       of 2017→2020, the mean annual rates over 2017→2019, 2017→2020 and 2019→2024, and the completion of the
       narrowing by 2019 (before the pandemic) and by 2020, against the completion of the fleet increment on three
       scales: levels, the logarithm used in the regressions, and the NEV share of civilian vehicles.

Usage:
    python code/65_revision_robustness.py

Output files:
    outputs/65_终审意见核验.xlsx (workbook of the five robustness checks)
    logs/65_revision_robustness.log
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats


def _load(name: str, file: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    if spec is None or spec.loader is None:
        raise SystemExit(f"{file} 无法加载")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export_utils = _load("export_utils", "01_export_utils.py")
m48 = _load("m48", "48_run_station_panel.py")   # regression: PanelOLS, two-way demeaning, clustered t, as in script 48
m51 = _load("m51", "51_station_timing_and_power.py")   # priority-region city list and annual gradients from script 51

STA, CITY, TIME, PROV = m48.STA, m48.CITY, m48.TIME, "省级行政区"
EXPO_RAW = "建成区占比_500m"
MIN_DAYS = m48.MIN_DAYS_MAIN
RNG = np.random.default_rng(20260925)
RNG_GAP = np.random.default_rng(20260926)   # own stream for the full-attribution test; other bootstrap draws unaffected
B_BOOT = 999
SPECIES = ["NO2", "SO2", "CO", "PM2.5"]
INFERRED_CITY = "牡丹江市"   # the only city whose 2023 value was inferred from the provincial residual (script 40)

# Gate: the panel and regressions of this script must reproduce digit for digit the Table 2 numbers of the main text
# stored in the workbooks of scripts 48/51; otherwise the script stops and writes no workbook
GATE = {"式(3) β": (-0.04952, 5), "式(3) SE": (0.01366, 5),
        "式(4) β": (0.00000, 5), "式(4) SE": (0.04228, 5), "式(4) 趋势 β": (-0.03138, 5),
        "年份指示 β": (-0.001438, 6), "年份指示 SE": (0.0461, 4),
        "日变化加趋势 β": (0.0147, 4), "日变化加趋势 SE": (0.0256, 4),
        # column 6 of Table 2 in the main text: the workbook of script 48 stores the trend term as −0.03585 (full
        # precision −0.035849, printed as −0.0358); the gate checks the five-decimal value of the workbook
        "日变化加趋势 趋势 β": (-0.03585, 5)}


def build_panel() -> pd.DataFrame:
    """Same definitions as main() of script 48: 274-day threshold, city-years with ≥2 NO₂ stations in the city;
    adds the fingerprint variables and the provincial treatments."""
    p = pd.read_csv(export_utils.DATA_DIR / "站点年面板-2015_2024.csv")
    p.columns = [str(c).lstrip("﻿") for c in p.columns]
    p["城市年"] = p[CITY].astype(str) + "_" + p[TIME].astype(str)
    for sp in SPECIES:
        ok = p[f"{sp}_有效日"] >= MIN_DAYS
        p[f"ln_{sp}"] = np.log(p[f"{sp}_年均"].where((p[f"{sp}_年均"] > 0) & ok))
        p[f"ln_{sp}_峰谷比"] = np.log(p[f"{sp}_峰谷比"].where((p[f"{sp}_峰谷比"] > 0) & ok))
        p[f"{sp}_周末效应"] = p[f"{sp}_周末效应"].where(ok)
    o3_ok = p["O3_年均_有效日"] >= MIN_DAYS
    p["ln_O3年均"] = np.log(p["O3_年均"].where((p["O3_年均"] > 0) & o3_ok))
    p["ln_Ox"] = np.log(p["Ox_年均_ppb"].where((p["Ox_年均_ppb"] > 0) & o3_ok & (p["NO2_有效日"] >= MIN_DAYS)))

    # exposure is centred on the full-panel mean as in script 48; the product of the centring constant and the
    # treatment varies by city-year and is absorbed by the city-by-year fixed effects
    p["Expo"] = p[EXPO_RAW] - p[EXPO_RAW].mean()
    p["lnNEV"] = np.log(p["NEV保有量_辆"].where(p["NEV保有量_辆"] > 0))
    p["NEVxExpo"] = p["lnNEV"] * p["Expo"]
    p["t"] = p[TIME] - p[TIME].min()
    p["tx暴露"] = p["t"] * p["Expo"]

    # provincial treatment: vehicle stock on the Ministry of Public Security registration basis and provincial
    # civilian vehicle ownership (10,000 vehicles)
    pn = pd.read_csv(export_utils.DATA_DIR / "省级新能源汽车保有量-公安部口径-2017_2023.csv")
    pn.columns = [str(c).lstrip("﻿") for c in pn.columns]
    pc = pd.read_csv(export_utils.DATA_DIR / "建模面板-2017_2023.csv")
    pc.columns = [str(c).lstrip("﻿") for c in pc.columns]
    prov = pn.merge(pc[[PROV, TIME, "民用汽车拥有量"]], on=[PROV, TIME], how="left")
    prov["省NEV占比%"] = 100 * prov["新能源汽车保有量_万辆"] / prov["民用汽车拥有量"]
    p = p.merge(prov[[PROV, TIME, "新能源汽车保有量_辆", "省NEV占比%"]], on=[PROV, TIME], how="left")
    p["ln省NEV"] = np.log(p["新能源汽车保有量_辆"].where(p["新能源汽车保有量_辆"] > 0))
    p["省NEVxExpo"] = p["ln省NEV"] * p["Expo"]
    p["省占比xExpo"] = p["省NEV占比%"] * p["Expo"]

    valid = p.dropna(subset=["ln_NO2"])
    p["城内站数"] = p["城市年"].map(valid.groupby("城市年")[STA].nunique())
    return p[p["城内站数"] >= 2].copy()


def year_dummies(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Year indicators × exposure (base year: the first year covered by the treatment), the same specification as ④
    of script 51."""
    d = d.copy()
    cols = []
    for yv in sorted(d[TIME].unique())[1:]:
        c = f"y{yv}xExpo"
        d[c] = (d[TIME] == yv).astype(float) * d["Expo"]
        cols.append(c)
    return d, cols


def fit(d: pd.DataFrame, y: str, xs: list[str], cluster: str = CITY) -> tuple[Any, pd.DataFrame]:
    """Station fixed effects + city-by-year fixed effects, clustered by `cluster` (same implementation as fit in
    script 48)."""
    dd = d.dropna(subset=[y, *xs]).copy()
    pdf = dd.set_index([STA, TIME])
    res = PanelOLS(pdf[y], pdf[xs], entity_effects=True, other_effects=pdf[["城市年"]],
                   drop_absorbed=True, check_rank=False).fit(cov_type="clustered", clusters=pdf[[cluster]])
    return res, dd


def wild_boot_p(d: pd.DataFrame, y: str, xs: list[str], target: str, cluster: str = CITY,
                rng: np.random.Generator = RNG) -> float:
    """Restricted wild cluster bootstrap (Rademacher, 999 draws), the same algorithm as script 48, with a
    changeable cluster variable."""
    dd = d.dropna(subset=[y, *xs]).reset_index(drop=True)
    z = m48.demean_two(dd, [y, *xs], STA, "城市年")
    yv, X = z[y].to_numpy(float), z[xs].to_numpy(float)
    g = pd.factorize(dd[cluster])[0]
    j = xs.index(target)
    t_obs = m48.cluster_t(yv, X, g, j)
    keep = [k for k in range(X.shape[1]) if k != j]
    Xr = X[:, keep]
    br = np.linalg.pinv(Xr.T @ Xr) @ Xr.T @ yv
    ur, fitted = yv - Xr @ br, Xr @ br
    G = int(g.max()) + 1
    cnt = sum(abs(m48.cluster_t(fitted + rng.choice([-1.0, 1.0], size=G)[g] * ur, X, g, j)) >= abs(t_obs)
              for _ in range(B_BOOT))
    return float((cnt + 1) / (B_BOOT + 1))


def within_slope(d: pd.DataFrame, y: str, x: str = EXPO_RAW) -> dict[str, Any]:
    """Same-city, same-year cross-sectional slope: y on x absorbing city-by-year fixed effects, clustered by city."""
    dd = d.dropna(subset=[y, x]).copy()
    pdf = dd.set_index([STA, TIME])
    res = PanelOLS(pdf[y], pdf[[x]], other_effects=pdf[["城市年"]], drop_absorbed=True,
                   check_rank=False).fit(cov_type="clustered", clusters=pdf[[CITY]])
    return {"斜率": float(res.params[x]), "SE": float(res.std_errors[x]), "p": float(res.pvalues[x]),
            "站点年": int(res.nobs), "站点": dd[STA].nunique(), "城市": dd[CITY].nunique()}


def city_sums(d: pd.DataFrame, y: str, x: str = EXPO_RAW) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-city sums Σx̃ỹ and Σx̃² (x̃, ỹ demeaned within city-year); the whole-city bootstrap then only needs to
    resample city weights."""
    z = d[[y, x]].astype(float)
    z = z - z.groupby(d["城市年"].to_numpy()).transform("mean")
    gid = pd.factorize(d[CITY])[0]
    sxy = np.bincount(gid, weights=(z[x] * z[y]).to_numpy(float)).astype(np.float64)
    sxx = np.bincount(gid, weights=(z[x] * z[x]).to_numpy(float)).astype(np.float64)
    return sxy, sxx


def slope_diff_boot(d: pd.DataFrame, ya: str, yb: str) -> dict[str, float]:
    """Difference between the cross-sectional slopes of two outcomes on the same station-years, with a whole-city
    bootstrap 95% interval."""
    dd = d.dropna(subset=[ya, yb, EXPO_RAW])
    ax, axx = city_sums(dd, ya)
    bx, bxx = city_sums(dd, yb)
    est = ax.sum() / axx.sum() - bx.sum() / bxx.sum()
    G = len(ax)
    draws = np.empty(B_BOOT)
    for k in range(B_BOOT):
        w = np.bincount(RNG.integers(0, G, G), minlength=G).astype(float)
        draws[k] = (w @ ax) / (w @ axx) - (w @ bx) / (w @ bxx)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"差值": float(est), "区间下限": float(lo), "区间上限": float(hi), "站点年": len(dd),
            "城市": int(G)}


def fingerprint(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every species is regressed on the NO₂ analysis sample (station-years with valid NO₂), with the species' own
    274-day threshold applied on top; this is the definition of the P1 comparison in script 47, and the slope of
    annual mean SO₂ must reproduce its 0.2891 (10600 station-years)."""
    d = d.dropna(subset=["ln_NO2"])
    rows = []
    for sp in SPECIES:
        for kind, y in [("ln 年均浓度", f"ln_{sp}"), ("ln 峰谷比", f"ln_{sp}_峰谷比"),
                        ("工作日−周末相对差", f"{sp}_周末效应")]:
            if d[y].notna().sum() < 500:
                continue
            rows.append({"物种": sp, "量": kind, **within_slope(d, y)})
    for sp, y in [("O3", "ln_O3年均"), ("Ox", "ln_Ox")]:
        rows.append({"物种": sp, "量": "ln 年均浓度", **within_slope(d, y)})
    fp = pd.DataFrame(rows)
    diffs = []
    for kind, suf in [("ln 年均浓度", ""), ("ln 峰谷比", "_峰谷比")]:
        for other in ["SO2", "CO", "PM2.5"]:
            r = slope_diff_boot(d, f"ln_NO2{suf}", f"ln_{other}{suf}")
            diffs.append({"量": kind, "对照": f"NO2 − {other}", **r})
    return fp, pd.DataFrame(diffs)


def treatment_block(d: pd.DataFrame) -> pd.DataFrame:
    """Eq. (3), Eq. (4) and the year-indicator specification under each treatment definition.

    The implied coefficient of full attribution is computed on each definition's own regression sample. Under full
    attribution the gradient of city c changes by β·ΔT_c, and the annual gradient is the mean of the city gradients
    weighted by the within-city sum of squared exposure deviations, so the difference between the first-year and
    last-year gradients is exactly β times the difference in the mean of the treatment under the same weights; the
    latter is the within-city slope of "treatment × exposure" on exposure (computed with the gradient function of
    script 51 on the same stations). The last year of the sample without 2023 is 2022.
    """
    specs = [("城市 ln 保有量（正文口径）", "NEVxExpo", d, CITY),
             ("城市 ln 保有量，剔除牡丹江", "NEVxExpo", d[d[CITY] != INFERRED_CITY], CITY),
             ("城市 ln 保有量，剔除 2023 年", "NEVxExpo", d[d[TIME] != 2023], CITY),
             ("省级 ln 保有量（公安部口径）", "省NEVxExpo", d, PROV),
             ("省级新能源车占民用汽车比例（%，水平值）", "省占比xExpo", d, PROV)]
    rows = []
    for label, tx, sub, cl in specs:
        # the three specifications of a definition use the same station-years: treatment and cluster both non-missing
        base = sub.dropna(subset=["ln_NO2", tx, cl]).copy()
        base, ycols = year_dummies(base[base[TIME].between(2017, 2023)])
        r3, _ = fit(base, "ln_NO2", [tx], cluster=cl)
        r4, _ = fit(base, "ln_NO2", [tx, "tx暴露"], cluster=cl)
        ry, _ = fit(base, "ln_NO2", [tx, *ycols], cluster=cl)
        b_y, se_y = float(ry.params[tx]), float(ry.std_errors[tx])
        end = int(base[TIME].max())
        first, last = base[base[TIME] == 2017], base[base[TIME] == end]
        d_grad = m51.gradient(last, "ln_NO2")[0] - m51.gradient(first, "ln_NO2")[0]
        d_treat = m51.gradient(last, tx)[0] - m51.gradient(first, tx)[0]
        implied = d_grad / d_treat
        z_gap = abs(b_y - implied) / se_y
        z_imp = abs(implied) / se_y
        # restricted wild cluster bootstrap for full attribution (β equal to the implied coefficient): replace y by
        # y − implied coefficient × treatment and test a zero treatment coefficient
        gap = base.assign(_y_gap=base["ln_NO2"] - implied * base[tx])
        p_gap_wild = wild_boot_p(gap, "_y_gap", [tx, *ycols], tx, cluster=cl, rng=RNG_GAP)
        rows.append({
            "处理口径": label, "聚类": "城市" if cl == CITY else "省", "站点年": int(r3.nobs),
            "城市": base[CITY].nunique(), "聚类数": base[cl].nunique(), "隐含系数所用区间": f"2017→{end}",
            "样本梯度变化": d_grad, "处理增幅（梯度权重）": d_treat,
            "式3 β": float(r3.params[tx]), "式3 SE": float(r3.std_errors[tx]), "式3 p": float(r3.pvalues[tx]),
            "式3 wild p": wild_boot_p(base, "ln_NO2", [tx], tx, cluster=cl),
            "式4 β": float(r4.params[tx]), "式4 SE": float(r4.std_errors[tx]), "式4 p": float(r4.pvalues[tx]),
            "式4 wild p": wild_boot_p(base, "ln_NO2", [tx, "tx暴露"], tx, cluster=cl),
            "式4 趋势 β": float(r4.params["tx暴露"]), "式4 趋势 SE": float(r4.std_errors["tx暴露"]),
            "式4 趋势 p": float(r4.pvalues["tx暴露"]),
            "式4 趋势 wild p": wild_boot_p(base, "ln_NO2", [tx, "tx暴露"], "tx暴露", cluster=cl),
            "年份指示 β": b_y, "年份指示 SE": se_y, "年份指示 p": float(ry.pvalues[tx]),
            "年份指示 wild p": wild_boot_p(base, "ln_NO2", [tx, *ycols], tx, cluster=cl),
            "年份指示 95%区间": f"[{b_y - 1.96 * se_y:.4f}, {b_y + 1.96 * se_y:.4f}]",
            "MDE(2.8·SE)": 2.8 * se_y, "全额归因隐含系数": implied,
            "对隐含系数的功效": float(stats.norm.cdf(z_imp - 1.96) + stats.norm.cdf(-z_imp - 1.96)),
            "距隐含系数 SE 倍数": z_gap, "该距离双侧 p": float(2 * stats.norm.sf(z_gap)),
            "该距离 wild p": p_gap_wild,
            # both ends of the interval as the share of the narrowing the fleet can explain (implied coefficient = 100%)
            "可解释份额区间%": f"[{100 * min((b_y - 1.96 * se_y) / implied, (b_y + 1.96 * se_y) / implied):.0f}, "
                               f"{100 * max((b_y - 1.96 * se_y) / implied, (b_y + 1.96 * se_y) / implied):.0f}]",
        })
    return pd.DataFrame(rows)


def cross_section(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the 160-city cross-section of ③ in script 51, then add controls."""
    rows = []
    sub = d[(d[TIME] >= 2017) & (d[TIME] <= 2023)].dropna(subset=["ln_NO2", EXPO_RAW])
    for city, g in sub.groupby(CITY):
        yrs = sorted(g[TIME].unique())
        if len(yrs) < 5 or yrs[0] != 2017 or yrs[-1] != 2023:
            continue
        grad = {}
        for yr in (2017, 2023):
            gg = g[g[TIME] == yr]
            if gg[STA].nunique() < 3 or gg[EXPO_RAW].std() < 1e-6:
                continue
            A = np.column_stack([np.ones(len(gg)), gg[EXPO_RAW].to_numpy(float)])
            grad[yr] = float(np.linalg.lstsq(A, gg["ln_NO2"].to_numpy(float), rcond=None)[0][1])
        if len(grad) == 2:
            rows.append({"城市": city, "梯度2017": grad[2017], "梯度变化": grad[2023] - grad[2017],
                         "站点数": g[STA].nunique(), "重点区域": float(g["重点区域"].max())})
    cs = pd.DataFrame(rows)
    c = pd.read_csv(export_utils.DATA_DIR / "城市面板-2017_2023.csv")
    c.columns = [str(x).lstrip("﻿") for x in c.columns]
    cn = c[c["NEV保有量_辆"] > 0].pivot_table(index="城市", columns=TIME, values="NEV保有量_辆")
    gr = pd.DataFrame({"城市名": cn.index, "NEV对数增幅": np.log(cn[2023] / cn[2017]).to_numpy(),
                       "lnNEV2017": np.log(cn[2017]).to_numpy()})
    cs["城市名"] = cs["城市"].astype(str).str.replace(r"(市|地区|自治州|盟)$", "", regex=True)
    m = cs.merge(gr, on="城市名", how="inner").dropna(subset=["梯度变化", "NEV对数增幅"])

    m = m.assign(ln站点数=np.log(m["站点数"]))
    if len(m) != 160 or round(float(np.corrcoef(m["梯度变化"], m["NEV对数增幅"])[0, 1]), 3) != 0.234:
        raise SystemExit("横截面未复现 51 号的 160 城与 r=+0.234。不输出任何数。")
    out = []
    for label, mm in [("全部", m), ("剔除牡丹江", m[m["城市"] != INFERRED_CITY])]:
        r, p_r = stats.pearsonr(mm["梯度变化"], mm["NEV对数增幅"])
        rho, p_rho = stats.spearmanr(mm["梯度变化"], mm["NEV对数增幅"])
        for ctrl in [[], ["梯度2017"], ["梯度2017", "重点区域", "ln站点数"],
                     ["梯度2017", "重点区域", "ln站点数", "lnNEV2017"]]:
            X = np.column_stack([np.ones(len(mm)), mm["NEV对数增幅"].to_numpy(float),
                                 *[mm[k].to_numpy(float) for k in ctrl]])
            yv = mm["梯度变化"].to_numpy(float)
            b = np.linalg.lstsq(X, yv, rcond=None)[0]
            u = yv - X @ b
            n, k = X.shape
            XtX_i = np.linalg.inv(X.T @ X)
            V = XtX_i @ (X.T * u ** 2) @ X @ XtX_i * n / (n - k)       # HC1
            se = float(np.sqrt(V[1, 1]))
            out.append({"样本": label, "城市数": n, "控制": "、".join(ctrl) or "无",
                        "Pearson r": float(r), "Pearson p": float(p_r),
                        "Spearman ρ": float(rho), "Spearman p": float(p_rho),
                        "车队增幅系数": float(b[1]), "HC1 SE": se,
                        "p": float(2 * stats.t.sf(abs(b[1] / se), n - k))})
    return m, pd.DataFrame(out)


def fast_segment(d: pd.DataFrame, nev: pd.Series, car: pd.Series) -> pd.DataFrame:
    """Year-by-year decomposition of the fast segment, against the completion of the fleet increment on three scales:
    levels, the logarithm used in the regressions, and the share of civilian vehicles."""
    g = {int(yr): m51.gradient(gg, "ln_NO2")[0] for yr, gg in d.groupby(TIME)}
    if (round(g[2017], 4), round(g[2020], 4), round(g[2024], 4)) != (0.6941, 0.5017, 0.4054):
        raise SystemExit("逐年梯度未复现 51 号路径。不输出任何数。")
    tot = g[2020] - g[2017]
    rows = [{"项": f"{a}→{b} 下降", "值": g[b] - g[a], "占 2017→2020 的份额%": 100 * (g[b] - g[a]) / tot}
            for a, b in [(2017, 2018), (2018, 2019), (2019, 2020)]]
    rows += [{"项": "2017→2019 年均变化", "值": (g[2019] - g[2017]) / 2, "占 2017→2020 的份额%": np.nan},
             {"项": "2017→2020 年均变化", "值": (g[2020] - g[2017]) / 3, "占 2017→2020 的份额%": np.nan},
             {"项": "2019→2024 年均变化", "值": (g[2024] - g[2019]) / 5, "占 2017→2020 的份额%": np.nan}]
    share = 100 * nev / car
    scales = {"车队增量（水平值）": nev, "车队增量（对数）": np.log(nev), "新能源车占比增量": share}
    full = g[2023] - g[2017]
    for cut in (2019, 2020):
        rows.append({"项": f"截至 {cut} 年梯度收窄完成度%（2017→2023 为 100）", "值": 100 * (g[cut] - g[2017]) / full,
                     "占 2017→2020 的份额%": np.nan})
        for name, s in scales.items():
            rows.append({"项": f"截至 {cut} 年{name}完成度%（2017→2023 为 100）",
                         "值": 100 * (s[cut] - s[2017]) / (s[2023] - s[2017]), "占 2017→2020 的份额%": np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("65_revision_robustness")
    d = build_panel()
    # priority regions follow the city-list definition of script 51
    d = d.merge(m51.load()[[STA, TIME, "重点区域"]], on=[STA, TIME], how="left")
    logger.info("面板 %s 站-年，站点 %s，城市 %s", len(d), d[STA].nunique(), d[CITY].nunique())

    # ── Gate: reproduce Table 2 of the main text ──
    nv = d[d[TIME].between(2017, 2023)]
    r3, _ = fit(nv, "ln_NO2", ["NEVxExpo"])
    r4, _ = fit(nv, "ln_NO2", ["NEVxExpo", "tx暴露"])
    yd, ycols = year_dummies(nv.dropna(subset=["ln_NO2", "NEVxExpo"]))
    ry, _ = fit(yd, "ln_NO2", ["NEVxExpo", *ycols])
    r6, _ = fit(nv, "ln_NO2_峰谷比", ["NEVxExpo", "tx暴露"])
    got = {"式(3) β": r3.params["NEVxExpo"], "式(3) SE": r3.std_errors["NEVxExpo"],
           "日变化加趋势 β": r6.params["NEVxExpo"], "日变化加趋势 SE": r6.std_errors["NEVxExpo"],
           "日变化加趋势 趋势 β": r6.params["tx暴露"],
           "式(4) β": r4.params["NEVxExpo"], "式(4) SE": r4.std_errors["NEVxExpo"],
           "式(4) 趋势 β": r4.params["tx暴露"],
           "年份指示 β": ry.params["NEVxExpo"], "年份指示 SE": ry.std_errors["NEVxExpo"]}
    gate = pd.DataFrame([{"项": k, "参照": v, "本脚本": round(float(got[k]), nd) + 0.0,
                          "一致": round(float(got[k]), nd) + 0.0 == v} for k, (v, nd) in GATE.items()])
    logger.info("闸：\n%s", gate.to_string(index=False))
    if not gate["一致"].all():
        raise SystemExit("未能复现正文表 2 的数字。不输出任何数。")

    # column 6 of Table 2 (commuting-peak to overnight ratio with trend): SE and bootstrap p of both interaction terms
    t6 = pd.DataFrame([{
        "设定": "表 2 第 6 列：ln NO₂ 峰谷比，式 (4)", "站点年": int(r6.nobs),
        "电动化 β": float(r6.params["NEVxExpo"]), "电动化 SE": float(r6.std_errors["NEVxExpo"]),
        "电动化 p": float(r6.pvalues["NEVxExpo"]),
        "电动化 wild p": wild_boot_p(nv, "ln_NO2_峰谷比", ["NEVxExpo", "tx暴露"], "NEVxExpo"),
        "趋势 β": float(r6.params["tx暴露"]), "趋势 SE": float(r6.std_errors["tx暴露"]),
        "趋势 p": float(r6.pvalues["tx暴露"]),
        "趋势 wild p": wild_boot_p(nv, "ln_NO2_峰谷比", ["NEVxExpo", "tx暴露"], "tx暴露")}])
    logger.info("表 2 第 6 列：\n%s", t6.to_string(index=False))

    # ── ① Traffic fingerprint of the built-up axis ──
    fp, diffs = fingerprint(d)
    logger.info("① 指纹：\n%s\n%s", fp.to_string(index=False), diffs.to_string(index=False))
    so2 = fp[(fp["物种"] == "SO2") & (fp["量"] == "ln 年均浓度")].iloc[0]
    no2 = fp[(fp["物种"] == "NO2") & (fp["量"] == "ln 年均浓度")].iloc[0]
    if (round(float(so2["斜率"]), 4), int(so2["站点年"]), round(float(no2["斜率"]), 4)) != (0.2891, 10600, 0.53):
        raise SystemExit("指纹未复现 47 号 P1 的 SO₂ 0.2891（10600 站点年）与 NO₂ 0.5300。不输出任何数。")
    no2_pt = fp[(fp["物种"] == "NO2") & (fp["量"] == "ln 峰谷比")].iloc[0]
    so2_diff = diffs[(diffs["量"] == "ln 峰谷比") & (diffs["对照"] == "NO2 − SO2")].iloc[0]
    o3 = fp[fp["物种"] == "O3"].iloc[0]
    traffic_sig = bool(no2_pt["斜率"] > 0 and no2_pt["p"] < 0.05 and so2_diff["区间下限"] > 0
                       and o3["斜率"] < 0 and o3["p"] < 0.05)
    # "预设判读" in the message below refers to the interpretation rule of block ① in the docstring
    verdict = ("建成区轴对 NO₂ 带有交通型日变化与滴定特征，SO₂ 没有" if traffic_sig else
               "未满足预设判读：建成区轴只能称为城市核心程度")
    logger.info("① 判读：%s", verdict)

    # ── ②③ Treatment definitions and inference ──
    pn = pd.read_csv(export_utils.DATA_DIR / "省级新能源汽车保有量-公安部口径-2017_2023.csv")
    pn.columns = [str(c).lstrip("﻿") for c in pn.columns]
    pc = pd.read_csv(export_utils.DATA_DIR / "建模面板-2017_2023.csv")
    pc.columns = [str(c).lstrip("﻿") for c in pc.columns]
    nat_nev = pn.groupby(TIME)["新能源汽车保有量_万辆"].sum()
    nat_car = pc.groupby(TIME)["民用汽车拥有量"].sum()
    g17 = m51.gradient(d[d[TIME] == 2017], "ln_NO2")[0]
    nat: dict[int, dict[str, float]] = {}
    for end in (2022, 2023):
        g_end = m51.gradient(d[d[TIME] == end], "ln_NO2")[0]
        nat[end] = {"梯度变化": g_end - g17, "ln保有量": float(np.log(nat_nev[end] / nat_nev[2017])),
                    "占比%": float(100 * (nat_nev[end] / nat_car[end] - nat_nev[2017] / nat_car[2017]))}
        logger.info("全国 2017→%s：梯度变化 %.4f，ln 保有量增幅 %.3f，占比增幅 %.2f 个百分点",
                    end, nat[end]["梯度变化"], nat[end]["ln保有量"], nat[end]["占比%"])
    tr = treatment_block(d)
    # standard errors etc. are logged to 7 decimals, so four-decimal rounding can be checked directly against the log
    logger.info("②③ 处理口径：\n%s", tr.to_string(index=False, float_format="{:.7f}".format))

    # ── ④ Cross-section ──
    cs, csr = cross_section(d)
    logger.info("④ 横截面：\n%s", csr.to_string(index=False))

    # ── ⑤ Year-by-year decomposition of the fast segment ──
    seg = fast_segment(d, nat_nev, nat_car)
    logger.info("⑤ 快段：\n%s", seg.to_string(index=False))

    labels = [("梯度变化", "梯度变化"), ("ln 保有量增幅", "ln保有量"), ("新能源车占比增幅（百分点）", "占比%")]
    summ = pd.DataFrame([{"项": "① 判读", "值": verdict}] + [
        {"项": f"全国 2017→{end} {name}", "值": f"{nat[end][key]:.4f}"}
        for end in (2022, 2023) for name, key in labels])
    out = export_utils.write_excel_workbook("65_终审意见核验", [
        ("摘要", summ), ("闸", gate), ("建成区轴指纹", fp), ("指纹斜率差", diffs), ("处理口径", tr),
        ("表2第6列", t6),
        ("横截面加控制", csr), ("横截面_城市", cs), ("快段逐年分解", seg)])
    logger.info("已写出 %s；日志 %s", out, log_path)
    print(f"✓ 完成：{out}\n  ① {verdict}")


if __name__ == "__main__":
    main()
