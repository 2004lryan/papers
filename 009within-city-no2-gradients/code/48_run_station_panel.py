"""
48_run_station_panel.py: station-level electrification regression (equation (3) of the main text; rows "M9…" of the
workbook) + robustness checks S1, S2, S4–S11 and S13–S16 of Supplementary Table S2

Model:
    ln Y(s,c,t) = β·[lnNEV(c,t) × Exposure(s)] + γ·[Controls(c,t) × Exposure(s)] + α_s + δ_(c,t) + ε
                  (the γ term enters only checks S4a and S4c–S4e; the main-table rows "M9a"/"M9c" have no controls)

    α_s      station fixed effects
    δ_(c,t)  **city-by-year fixed effects** — absorb all time-varying confounding at the city level (regional haze
             abatement, meteorology, economy, industrial structure, the pandemic); this is the fundamental difference
             between this design and province- or city-level panels. NEV's main effect is absorbed by them; the model
             identifies only the interaction: within the same city and year, how the difference between stations with
             high and low traffic exposure changes with electrification.
    SE clustered by city; the main table additionally reports a city-level wild cluster bootstrap.

Prerequisite: P1 of script 47 must pass (a within-city NO₂ gradient related to built-up exposure does exist);
otherwise the coefficients of this script have no interpretation.

Usage:
    python code/48_run_station_panel.py

Output: outputs/48_站点面板回归.xlsx (station panel regressions); logs/48_run_station_panel.log
"""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from linearmodels.panel import PanelOLS

if TYPE_CHECKING:
    from linearmodels.panel.results import PanelEffectsResults

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260902)
B_BOOT = 999   # Number of replications stated in the manuscript. The wild bootstrap takes B such
               # that (B+1)α is an integer; 999 gives 50 tail draws at the two-sided 5% level. The implementation must
               # match what the manuscript states.

STA, CITY, TIME = "站点编号", "所属城市", "年份"
# Validity threshold for annual means (script 46 does not filter, it only provides valid-day counts): 274 days ≈75%
# in the main specification; robustness check S13 uses 324 days (GB 3095-2012, Table 4)
MIN_DAYS_MAIN, MIN_DAYS_STRICT = 274, 324
DAYCOL = {"NO2_年均": "NO2_有效日", "PM2.5_年均": "PM2.5_有效日", "PM10_年均": "PM10_有效日",
          "SO2_年均": "SO2_有效日", "CO_年均": "CO_有效日", "O3_8h_90分位": "O3_有效日"}
# Outcome variables. O3_8h follows the assessment standard (annual 90th percentile of the daily maximum 8-h mean);
# annual-mean O3 and Ox are added by script 50 for the **titration consistency test** —
# a real decline in traffic NOx raises O₃ in the urban core (weaker NO titration), while Ox ≡ O₃+NO₂ is conserved
# under titration. A "general improvement of the urban core" of the SO₂ kind cannot produce this chemical signature,
# so the test can separate the two.
OUTCOMES = {"NO2": "ln_NO2", "PM2.5": "ln_PM2.5", "PM10": "ln_PM10", "SO2": "ln_SO2",
            "O3_8h": "ln_O3", "O3年均": "ln_O3年均", "Ox": "ln_Ox"}
# D2 mechanism variables (sheet 02_D2机制_交通时段; Table S5, panel B, of the supplement): the **traffic-hour
# component** of NO₂, i.e. the commuting-peak / overnight-trough ratio and the weekday–weekend effect. These two
# quantities discriminate traffic sources better than concentration levels do — industry and heating have neither a
# weekday/weekend rhythm that follows commuting nor commuting peaks.
# P2 of script 47 already shows that the within-city SO₂ gradient narrows at least as much as that of NO₂ (PM2.5
# narrows less), so the "narrowing of the concentration-level gradient" may be a general improvement of the urban
# core rather than a change in traffic sources; D2 is the test that can separate the two.
MECH_SPECIES = ["NO2", "SO2", "PM2.5", "CO"]
MAIN = "NEVxExpo"


def fit(
    d: pd.DataFrame, y: str, xs: list[str], cluster: str = "both"
) -> tuple[PanelEffectsResults, pd.DataFrame]:
    dd = d.dropna(subset=[y, *xs]).copy()
    pdf = dd.set_index([STA, TIME])
    mod = PanelOLS(pdf[y], pdf[xs], entity_effects=True, other_effects=pdf[["城市年"]],
                   drop_absorbed=True, check_rank=False)
    if cluster == "both":
        res = mod.fit(cov_type="clustered", clusters=dd.set_index([STA, TIME])[[CITY]])
    elif cluster == "station":
        res = mod.fit(cov_type="clustered", cluster_entity=True)
    else:
        res = mod.fit(cov_type="robust")
    return res, dd


def detrend_within(d: pd.DataFrame, cols: list[str], g: str, t: str) -> pd.DataFrame | None:
    """Within each group g, regress cols linearly on t (with an intercept) and take the residuals — absorbing group
    fixed effects and group-specific trends at the same time."""
    out = {}
    for c in cols:
        def _res(x: pd.DataFrame, c: str = c) -> pd.Series:
            if len(x) < 3:
                return pd.Series(np.nan, index=x.index)
            A = np.column_stack([np.ones(len(x)), x[t].to_numpy(float)])
            v = x[c].to_numpy(float)
            b, *_ = np.linalg.lstsq(A, v, rcond=None)
            return pd.Series(v - A @ b, index=x.index)
        out[f"{c}_dt"] = d.groupby(g, group_keys=False)[[c, t]].apply(_res)
    r = pd.DataFrame(out)
    return None if r.isna().all().all() else r


def demean_two(d: pd.DataFrame, cols: list[str], g1: str, g2: str, iters: int = 200) -> pd.DataFrame:
    z = d[cols].astype(float).copy()
    for _ in range(iters):
        prev = z.values.copy()
        z = z - z.groupby(d[g1]).transform("mean")
        z = z - z.groupby(d[g2]).transform("mean")
        if np.max(np.abs(z.values - prev)) < 1e-9:
            break
    return z


def cluster_t(y: npt.NDArray[np.float64], X: npt.NDArray[np.float64], g: npt.NDArray[np.intp], j: int) -> float:
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gid in np.unique(g):
        m = g == gid
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G, N, K = len(np.unique(g)), len(y), X.shape[1]
    V = XtX @ meat @ XtX * (G / (G - 1)) * ((N - 1) / (N - K))
    return float(b[j] / np.sqrt(V[j, j]))


def wild_boot_p(d: pd.DataFrame, y: str, xs: list[str], target: str, B: int = B_BOOT) -> float:
    dd = d.dropna(subset=[y, *xs]).reset_index(drop=True)
    z = demean_two(dd, [y, *xs], STA, "城市年")
    yv, X = z[y].values, z[xs].values
    g = pd.factorize(dd[CITY])[0]
    j = xs.index(target)
    t_obs = cluster_t(yv, X, g, j)
    keep = [k for k in range(X.shape[1]) if k != j]
    Xr = X[:, keep]
    br = np.linalg.pinv(Xr.T @ Xr) @ Xr.T @ yv
    ur, fitted = yv - Xr @ br, Xr @ br
    G = g.max() + 1
    cnt = sum(abs(cluster_t(fitted + RNG.choice([-1.0, 1.0], size=G)[g] * ur, X, g, j)) >= abs(t_obs)
              for _ in range(B))
    return (cnt + 1) / (B + 1)


def balanced(d: pd.DataFrame, y: str) -> pd.DataFrame:
    """Check S5 of Supplementary Table S2, "keep only stations complete in all seven years": stations whose outcome and
    treatment variables have values in every year of the regression window.

    The regression window is set by the years with vehicle-stock data (2017–2023, 7 years). Now that the panel is
    extended to 2015–2024, stations can no longer be filtered by "number of years in which the station appears == 7":
    that would pick stations that appear in exactly seven years and exclude stations present in all ten.
    """
    use = d.dropna(subset=[y, MAIN])
    n_years = use[TIME].nunique()
    full = use.groupby(STA)[TIME].nunique()
    return use[use[STA].isin(full.index[full == n_years])]


def row(
    res: PanelEffectsResults, d: pd.DataFrame, label: str, tag: str, target: str = MAIN,
    boot: bool = False, y: str | None = None, xs: list[str] | None = None,
) -> dict[str, Any]:
    out = {"规格": label, "结果变量": tag, "N": int(res.nobs), "站点数": d[STA].nunique(),
           "城市数": d[CITY].nunique(), "组内R2": round(float(res.rsquared_within), 5)}
    for name in dict.fromkeys([target, "机动化xExpo", "领先NEVxExpo"]):
        if name in res.params.index:
            k = "β_交互" if name == target else f"β_{name}"
            out[k] = round(float(res.params[name]), 5)
            out[f"{k}_SE"] = round(float(res.std_errors[name]), 5)
            out[f"{k}_p"] = round(float(res.pvalues[name]), 5)
    if boot and y and xs:
        out["β_交互_城市wild_boot_p"] = round(wild_boot_p(d, y, xs, target), 4)
    return out


def main_df_rows_titration(main_rows: list[dict[str, Any]]) -> str | None:
    """Read the titration signature from the main-table coefficients for NO₂ / annual-mean O₃ / Ox side by side."""
    g = {r["结果变量"]: r for r in main_rows if r.get("规格", "").startswith("M9c")}
    need = ["NO2", "O3年均", "Ox"]
    if not all(k in g for k in need):
        return None
    b = {k: g[k].get("β_交互") for k in need}
    pv: dict[str, Any] = {k: g[k].get("β_交互_p") for k in need}   # value is None when the entry is missing
    if b["NO2"] is None or b["O3年均"] is None:
        return None
    sig: Callable[[str], bool] = lambda k: (pv[k] is not None and pv[k] < 0.05)  # noqa: E731
    if b["NO2"] < 0 and sig("NO2") and b["O3年均"] > 0 and sig("O3年均"):
        tail = ("Ox 不显著 → 纯滴定重分配，与 NOx 排放真实下降一致"
                if not sig("Ox") else
                f"Ox 亦显著（β={b['Ox']}）→ 光化学生成本身也在减少")
        return f"NO₂ 降（β={b['NO2']}）且 O₃ 升（β={b['O3年均']}）：符合滴定签名；{tail}"
    return (f"未出现滴定签名：β_NO₂={b['NO2']}（p={pv['NO2']}），"
            f"β_O₃年均={b['O3年均']}（p={pv['O3年均']}）")


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("48_run_station_panel")
    raw = pd.read_csv(export_utils.DATA_DIR / "站点年面板-2015_2024.csv")

    def prep(src: pd.DataFrame, min_days: int) -> pd.DataFrame:
        p = src.copy()
        p["城市年"] = p[CITY].astype(str) + "_" + p[TIME].astype(str)
        for tag, col in [("NO2", "NO2_年均"), ("PM2.5", "PM2.5_年均"), ("PM10", "PM10_年均"),
                         ("SO2", "SO2_年均"), ("O3_8h", "O3_8h_90分位"),
                         ("O3年均", "O3_年均"), ("Ox", "Ox_年均_ppb")]:
            if col not in p.columns:
                continue
            v = p[col].where(p[col] > 0)
            dcol = DAYCOL.get(col)
            if col in ("O3_年均", "Ox_年均_ppb") and "O3_年均_有效日" in p.columns:
                v = v.where(p["O3_年均_有效日"] >= min_days)
            if col == "Ox_年均_ppb" and "NO2_有效日" in p.columns:
                v = v.where(p["NO2_有效日"] >= min_days)
            if dcol in p.columns:
                v = v.where(p[dcol] >= min_days)
            p[OUTCOMES[tag]] = np.log(v)
        return p

    p = prep(raw, MIN_DAYS_MAIN)
    mech_cols: dict[str, str] = {}
    for sp in MECH_SPECIES:
        dcol = f"{sp}_有效日"
        ok = p[dcol] >= MIN_DAYS_MAIN if dcol in p.columns else True
        if f"{sp}_峰谷比" in p.columns:
            v = p[f"{sp}_峰谷比"]
            p[f"ln_{sp}_峰谷比"] = np.log(v.where((v > 0) & ok))
            mech_cols[f"{sp}峰谷比"] = f"ln_{sp}_峰谷比"
        if f"{sp}_周末效应" in p.columns:
            p[f"{sp}_周末效应"] = p[f"{sp}_周末效应"].where(ok)
            mech_cols[f"{sp}周末效应"] = f"{sp}_周末效应"

    # S4 motorisation confounding: city-level motor-vehicle totals are not available, so the **provincial** civilian
    # vehicle stock is used instead (the gasoline-consumption variant S4b is not run; see the S4 block below).
    # City-by-year fixed effects are finer than province-by-year ones and already absorb the main effect of this
    # variable; only its interaction with station exposure is identified here —
    # i.e. "where motorisation is faster, do stations with high traffic exposure deteriorate relatively faster", which
    # is exactly the main threat to this design.
    # City-level covariates (script 49, Major Cities Annual Data, covering only 36 major cities → subsample robustness)
    cpath = export_utils.DATA_DIR / "国家统计局-城市级协变量-2017_2023.csv"
    if cpath.exists():
        cc = pd.read_csv(cpath)
        cc = cc.rename(columns={"城市": "城市名"}).drop(columns=["城市代码"], errors="ignore")
        cc["城市名"] = cc["城市名"].astype(str).str.replace(r"(市|地区)$", "", regex=True)
        p = p.merge(cc, on=["城市名", "年份"], how="left")
        for c in ["城市地区生产总值", "城市第二产业增加值", "城市第三产业增加值",
                  "城市社会消费品零售总额", "城市货物运输量"]:
            if c in p.columns:
                p[f"ln_{c}"] = np.log(p[c].where(p[c] > 0))
        n36 = int(p["道路交通等效声级"].notna().sum()) if "道路交通等效声级" in p.columns else 0
        logger.info("已并入城市级协变量：36 城子样本，道路交通声级非空 %s 站-年", n36)

    prov = pd.read_csv(export_utils.DATA_DIR / "建模面板-2017_2023.csv")
    prov.columns = [c.lstrip("\ufeff") for c in prov.columns]
    keep = [c for c in ["民用汽车拥有量"] if c in prov.columns]
    if keep:
        p = p.merge(prov[["省级行政区", "年份", *keep]], on=["省级行政区", "年份"], how="left")
        for c in keep:
            p[f"ln_{c}"] = np.log(p[c].where(p[c] > 0))
    p["lnNEV"] = np.log(p["NEV保有量_辆"].where(p["NEV保有量_辆"] > 0))

    # Exposure: centred; three measures
    for src, name in [("建成区占比_500m", "Expo"), ("建成区占比_1000m", "Expo1000"),
                      ("建成区占比_城内百分位", "ExpoPct")]:
        if src in p.columns:
            p[name] = p[src] - p[src].mean()
    # Base-period exposure (S11): each station is fixed at its 2017 value. Exposure comes from a single epoch of the
    # built-up layer and does not change over time, so in effect this keeps only the stations with a 2017 record
    p["ExpoBase"] = p[STA].map(p[p[TIME] == 2017].set_index(STA)["建成区占比_500m"])
    p["ExpoBase"] = p["ExpoBase"] - p["ExpoBase"].mean()

    p["NEVxExpo"] = p["lnNEV"] * p["Expo"]
    # Exposure does not vary over time, while lnNEV rises monotonically in almost every city — after two-way
    # demeaning, their interaction is very close to an "exposure-specific linear time trend". S15/S16 separate the two:
    #   S15 replaces the treatment with a pure time trend t×exposure — if it alone reproduces the coefficient, NEV is
    #       only a proxy for time
    #   S16 includes both t×exposure and NEV×exposure — once the common trend is absorbed, all that remains is "do the
    #       high-exposure stations of cities where NEV grows faster narrow faster", which is exactly what the
    #       electrification test asks
    p["t"] = p[TIME] - p[TIME].min()
    p["tx暴露"] = p["t"] * p["Expo"]
    p = p.sort_values([STA, TIME]).reset_index(drop=True)
    # The S9 lead term (Supplementary Table S2) is the vehicle stock of the station's city in year t+1,
    # lnNEV(c,t+1), aligned by city-year: taking the station's next row would pick up t+2 when the station misses a
    # year, and would wrongly drop the lead term when the station has no record in t+1
    nev_ct = p.dropna(subset=["lnNEV"]).groupby([CITY, TIME])["lnNEV"].agg(["first", "nunique"])
    if int(nev_ct["nunique"].max()) != 1:
        raise SystemExit("lnNEV 在同一城市-年内不唯一，领先项无法按城市-年对齐")
    lead = nev_ct["first"].rename("lnNEV_lead1").reset_index()
    lead[TIME] = lead[TIME] - 1
    p = p.merge(lead, on=[CITY, TIME], how="left")
    p["领先NEVxExpo"] = p["lnNEV_lead1"] * p["Expo"]

    # Keep only city-years with ≥2 stations in the city
    valid = p.dropna(subset=["ln_NO2"])
    p["城内站数"] = p["城市年"].map(valid.groupby("城市年")[STA].nunique())
    d0 = p[p["城内站数"] >= 2].copy()
    logger.info("面板 %s 行 → 城内 ≥2 站样本 %s 行，站点 %s，城市 %s",
                len(p), len(d0), d0[STA].nunique(), d0[CITY].nunique())

    main_rows, rob_rows, mech_rows = [], [], []
    for tag, y in OUTCOMES.items():
        if y not in d0.columns or d0[y].notna().sum() < 500:
            continue
        specs = [("M9a 仅交互", [MAIN]), ("M9c 主表", [MAIN])]
        for label, xs in specs:
            res, dd = fit(d0, y, xs)
            main_rows.append(row(res, dd, label, tag, boot=label.startswith("M9c"), y=y, xs=xs))
            logger.info("[%s] %-12s β=%.5f (p=%.4g) N=%s", tag, label, res.params[MAIN], res.pvalues[MAIN], res.nobs)

    # ── D2 mechanism: traffic-hour component ──
    for tag, y in mech_cols.items():
        if y not in d0.columns or d0[y].notna().sum() < 500:
            logger.info("[D2] %s 数据不足，跳过", tag)
            continue
        is_signal = tag.startswith("NO2")
        res, dd = fit(d0, y, [MAIN])
        r = row(res, dd, "M9 机制" if is_signal else "M9 安慰剂", tag, boot=True, y=y, xs=[MAIN])
        r["角色"] = "交通信号" if is_signal else "安慰剂（无通勤节律）"
        mech_rows.append(r)
        logger.info("[D2] %-14s %-4s β=%.5f (p=%.4g) boot_p=%s N=%s", tag,
                    "信号" if is_signal else "安慰剂", res.params[MAIN], res.pvalues[MAIN],
                    r.get("β_交互_城市wild_boot_p"), res.nobs)
        if is_signal:
            for label, sub in [("S6 剔除2020", d0[d0[TIME] != 2020]),
                               ("S10 城内≥3站", d0[d0["城内站数"] >= 3]),
                               ("S5 七年齐全站", balanced(d0, y))]:
                rs, ds = fit(sub, y, [MAIN])
                rr = row(rs, ds, label, tag)
                rr["角色"] = "交通信号"
                mech_rows.append(rr)
            # Separate the time trend on the mechanism variables as well
            if "tx暴露" in d0.columns:
                rs, ds = fit(d0, y, ["tx暴露"])
                rr = row(rs, ds, "S15 处理换纯时间趋势 t×暴露", tag, target="tx暴露")
                rr["角色"] = "交通信号"
                mech_rows.append(rr)
                rs, ds = fit(d0, y, [MAIN, "tx暴露"])
                rr = row(rs, ds, "S16 同时放 t×暴露 与 NEV×暴露", tag)
                if "tx暴露" in rs.params.index:
                    rr["β_t×暴露"] = round(float(rs.params["tx暴露"]), 5)
                    rr["β_t×暴露_p"] = round(float(rs.pvalues["tx暴露"]), 5)
                rr["角色"] = "交通信号"
                mech_rows.append(rr)
                logger.info("[D2] %s S16 控住共同趋势后 β_NEV=%.5f (p=%.4g)", tag,
                            rs.params[MAIN], rs.pvalues[MAIN])

    # Titration consistency: put the three coefficients for NO₂ / annual-mean O₃ / Ox side by side and check whether
    # they match the chemical signature of a "real NOx decline"
    tit = main_df_rows_titration(main_rows)
    if tit is not None:
        logger.info("滴定一致性：%s", tit)

    # Robustness checks only for the main outcome NO2 and the placebo SO2
    for tag in ["NO2", "SO2"]:
        y = OUTCOMES[tag]
        if y not in d0.columns:
            continue
        for label, col in [("S1 暴露1000m", "Expo1000"), ("S2 暴露城内百分位", "ExpoPct"),
                           ("S11 暴露基期固定", "ExpoBase")]:
            if col not in d0.columns:
                continue
            d0["_x"] = d0["lnNEV"] * d0[col]
            res, dd = fit(d0, y, ["_x"])
            # row() already writes the target's coefficient/SE/p under the common `β_交互*` (interaction)
            # columns, so no renaming is needed
            rob_rows.append(row(res, dd, label, tag, target="_x"))
        for label, sub in [("S5 七年齐全站", balanced(d0, y)),
                           ("S6 剔除2020", d0[d0[TIME] != 2020]),
                           ("S10 城内≥3站", d0[d0["城内站数"] >= 3])]:
            res, dd = fit(sub, y, [MAIN])
            rob_rows.append(row(res, dd, label, tag))
        res, dd = fit(d0, y, [MAIN, "领先NEVxExpo"])
        rob_rows.append(row(res, dd, "S9 加领先项", tag))

        # S4 motorisation confounding (provincial proxy)
        # A second proxy, S4b (provincial gasoline consumption × exposure), is not run: the National Bureau of
        # Statistics stopped publishing provincial gasoline consumption from 2020 on, leaving only the three years
        # 2017–2019, which are almost collinear with the city-by-year fixed effects (a trial run gave a negative
        # within R² and diverging coefficients).
        for label, src in [("S4a 加省汽车保有量x暴露", "ln_民用汽车拥有量")]:
            if src not in d0.columns or d0[src].notna().sum() < 500:
                continue
            d0["机动化xExpo"] = d0[src] * d0["Expo"]
            res, dd = fit(d0, y, [MAIN, "机动化xExpo"])
            rob_rows.append(row(res, dd, label, tag))

        # S7 station-specific linear trends: first detrend on t within each station (with an intercept, which is
        # equivalent to also absorbing station fixed effects), then include only city-by-year fixed effects
        sub = d0.dropna(subset=[y, MAIN]).copy()
        sub["_t"] = sub[TIME] - sub[TIME].min()
        det = detrend_within(sub, [y, MAIN], STA, "_t")
        if det is not None:
            sub[[f"{y}_dt", f"{MAIN}_dt"]] = det
            dd2 = sub.set_index([STA, TIME])
            r7 = PanelOLS(dd2[f"{y}_dt"], dd2[[f"{MAIN}_dt"]], other_effects=dd2[["城市年"]],
                          drop_absorbed=True, check_rank=False).fit(
                cov_type="clustered", clusters=dd2[[CITY]])
            rob_rows.append(row(r7, sub, "S7 站点特定线性趋势", tag, target=f"{MAIN}_dt"))

        # S8 first differences: differencing within stations removes station fixed effects; city-by-year fixed
        # effects are still included
        fd = d0.sort_values([STA, TIME]).copy()
        # Use only differences between adjacent years: when a station misses a year, two consecutive rows span two
        # years and cannot be treated as a one-year change
        step = fd.groupby(STA)[TIME].diff()
        for c in [y, MAIN]:
            fd[f"d_{c}"] = fd.groupby(STA)[c].diff().where(step == 1)
        fd = fd.dropna(subset=[f"d_{y}", f"d_{MAIN}"])
        if len(fd) >= 500:
            dd3 = fd.set_index([STA, TIME])
            r8 = PanelOLS(dd3[f"d_{y}"], dd3[[f"d_{MAIN}"]], other_effects=dd3[["城市年"]],
                          drop_absorbed=True, check_rank=False).fit(
                cov_type="clustered", clusters=dd3[[CITY]])
            rob_rows.append(row(r8, fd, "S8 一阶差分", tag, target=f"d_{MAIN}"))

        # S4c–S4e city-level confounding (subsample of 36 major cities): measured traffic intensity (road-traffic
        # noise), industrial structure (log secondary-industry value added), level of development (log gross regional
        # product)
        for label, src in [("S4c 加城市道路交通声级x暴露", "道路交通等效声级"),
                           ("S4d 加城市第二产业x暴露", "ln_城市第二产业增加值"),
                           ("S4e 加城市GDPx暴露", "ln_城市地区生产总值")]:
            if src not in d0.columns or d0[src].notna().sum() < 300:
                continue
            d0["机动化xExpo"] = d0[src] * d0["Expo"]
            res, dd = fit(d0.dropna(subset=[src]), y, [MAIN, "机动化xExpo"])
            r = row(res, dd, label, tag)
            r["样本"] = "36主要城市"
            rob_rows.append(r)

        # S14 falsification: replace the treatment variable NEV with proxies for the level of development and rerun
        # the same design. If a development proxy alone reproduces the NEV coefficient, NEV is merely a stand-in for
        # the level of development.
        for label, src in [("S14a 处理换城市GDP", "ln_城市地区生产总值"),
                           ("S14b 处理换城市零售额", "ln_城市社会消费品零售总额"),
                           ("S14c 处理换城市第二产业", "ln_城市第二产业增加值")]:
            if src not in d0.columns or d0[src].notna().sum() < 300:
                continue
            sub = d0.dropna(subset=[src]).copy()
            sub["_x"] = sub[src] * sub["Expo"]
            res, dd = fit(sub, y, ["_x"])
            r = row(res, dd, label, tag, target="_x")
            r["样本"] = "36主要城市"
            rob_rows.append(r)
            # NEV's own coefficient on the same subsample, as a comparable benchmark
            res0, dd0 = fit(sub, y, [MAIN])
            r0 = row(res0, dd0, f"{label[:4]}对照 同子样本NEV", tag)
            r0["样本"] = "36主要城市"
            rob_rows.append(r0)

        # S15/S16 time trend vs. NEV
        if "tx暴露" in d0.columns:
            res, dd = fit(d0, y, ["tx暴露"])
            r = row(res, dd, "S15 处理换纯时间趋势 t×暴露", tag, target="tx暴露")
            rob_rows.append(r)
            res, dd = fit(d0, y, [MAIN, "tx暴露"])
            r = row(res, dd, "S16 同时放 t×暴露 与 NEV×暴露", tag)
            if "tx暴露" in res.params.index:
                r["β_t×暴露"] = round(float(res.params["tx暴露"]), 5)
                r["β_t×暴露_p"] = round(float(res.pvalues["tx暴露"]), 5)
            rob_rows.append(r)
            logger.info("[%s] S16 控住共同趋势后 β_NEV=%.5f (p=%.4g)", tag,
                        res.params[MAIN], res.pvalues[MAIN])

        logger.info("[%s] 稳健性完成（含 36 城子样本 S4c–S4e 与证伪 S14–S16）", tag)

    # S13 validity threshold switched to the national standard GB 3095-2012 (324 days) — the panel shrinks and 2018
    # loses half of its observations; check whether the conclusions flip
    q = prep(raw, MIN_DAYS_STRICT)
    q["Expo"] = q["建成区占比_500m"] - q["建成区占比_500m"].mean()
    q["lnNEV"] = np.log(q["NEV保有量_辆"].where(q["NEV保有量_辆"] > 0))
    q["NEVxExpo"] = q["lnNEV"] * q["Expo"]
    q["城内站数"] = q["城市年"].map(q.dropna(subset=["ln_NO2"]).groupby("城市年")[STA].nunique())
    dq = q[q["城内站数"] >= 2].copy()
    for tag in ["NO2", "SO2"]:
        if OUTCOMES[tag] in dq.columns and dq[OUTCOMES[tag]].notna().sum() >= 500:
            res, dd = fit(dq, OUTCOMES[tag], [MAIN])
            rob_rows.append(row(res, dd, f"S13 门槛国标{MIN_DAYS_STRICT}天", tag))
    logger.info("S13 门槛敏感性完成（主口径 N=%s 站-年 vs 国标 N=%s）",
                int(d0["ln_NO2"].notna().sum()), int(dq["ln_NO2"].notna().sum()))

    main_df, rob_df = pd.DataFrame(main_rows), pd.DataFrame(rob_rows)
    mech_df = pd.DataFrame(mech_rows)
    export_utils.write_excel_workbook("48_站点面板回归",
                                      [("主表", main_df), ("D2机制_交通时段", mech_df), ("稳健性", rob_df)])

    print("✓ 站点面板回归完成\n")
    print(main_df.to_string(index=False))
    print("\n── D2 机制（交通时段成分）──")
    print(mech_df.to_string(index=False) if not mech_df.empty else "（无）")
    print()
    print(rob_df.to_string(index=False))
    print(f"\n  输出: {export_utils.OUTPUT_DIR / '48_站点面板回归.xlsx'}\n  日志: {log_path}")


if __name__ == "__main__":
    main()
