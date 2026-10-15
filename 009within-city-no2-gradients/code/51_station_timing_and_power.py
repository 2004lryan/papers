"""
51_station_timing_and_power.py: narrowing path of the within-city pollution gradient, timing check against
electrification, and detectability assessment

Robustness check S16 of script 48 (Table S2 of the Supplementary Material) gives a zero coefficient: once an
"exposure-specific common time trend" is controlled for, the NEV interaction goes to zero
(ln NO₂: β goes from −0.0495 to −0.0000003; with year dummies × exposure instead it is still −0.0014, p=0.98).
A zero coefficient on its own is not enough for a conclusion; two questions must be answered:

    ① **Does the timing line up?** If the narrowing of the gradient happened **before** the surge in electrification,
       then "electrification caused the narrowing" already fails on timing, and this judgement does not depend on any
       regression specification.
    ② **Is the zero coefficient "truly nothing" or "undetectable"?** Report the minimum detectable effect (MDE)
       and compare it with the coefficient implied by "the narrowing is caused entirely by NEVs". Without this step,
       a null result is not publishable.

This script also sets three comparison series side by side: SO₂ (industrial tracer), PM2.5, and the NO₂
peak-to-trough ratio (traffic time-of-day component), as well as a comparison of the narrowing paths inside and
outside the priority regions of the Blue Sky Protection Campaign. The latter is an alternative explanation that had
already come up in this project's earlier provincial- and city-level analyses, and it has to be tested head-on at
the station level as well.
Priority regions are assigned by the cities listed in the Action Plan (see KEY_CITIES); for how this differs from an
assignment by province, see script 60.

Usage:
    python code/51_station_timing_and_power.py

Output files:
    outputs/51_梯度路径与可检测性.xlsx (gradient path and detectability)
    logs/51_station_timing_and_power.log
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

STA, CITY, TIME = "站点编号", "所属城市", "年份"
EXPO = "建成区占比_500m"
MIN_DAYS = 274
# The three priority regions of the Blue Sky Protection Campaign: Three-Year Action Plan for Winning the Blue Sky
# Defence War (State Council document Guofa [2018] No. 22), Section I(3), "Scope of priority regions".
# Beijing–Tianjin–Hebei (BTH) and its surroundings and the Fenwei Plain are listed city by city, while the Yangtze
# River Delta is included as whole provinces (municipalities);
# in the five provinces of Hebei, Shanxi, Shandong, Henan and Shaanxi only the listed cities belong to the priority
# regions, so the assignment cannot be made by province.
KEY_CITIES = ["北京市", "天津市",
              "石家庄市", "唐山市", "邯郸市", "邢台市", "保定市", "沧州市", "廊坊市", "衡水市",
              "太原市", "阳泉市", "长治市", "晋城市",
              "济南市", "淄博市", "济宁市", "德州市", "聊城市", "滨州市", "菏泽市",
              "郑州市", "开封市", "安阳市", "鹤壁市", "新乡市", "焦作市", "濮阳市",     # BTH and surrounding areas
              "晋中市", "运城市", "临汾市", "吕梁市", "洛阳市", "三门峡市",
              "西安市", "铜川市", "宝鸡市", "咸阳市", "渭南市"]                       # Fenwei Plain
KEY_PROVINCES = ["上海市", "江苏省", "浙江省", "安徽省"]                               # Yangtze River Delta
SERIES = [("NO2", "ln_NO2"), ("NO2峰谷比", "ln_峰谷比"), ("SO2", "ln_SO2"), ("PM2.5", "ln_PM25")]


def load() -> pd.DataFrame:
    p = pd.read_csv(export_utils.DATA_DIR / "站点年面板-2015_2024.csv")
    p.columns = [c.lstrip("﻿") for c in p.columns]
    # Priority-region membership is decided by the city a station belongs to, so it covers all years. The panel
    # carries the province (`省级行政区`, joined by script 46 from the city panel) only in 2017–2023; the province
    # needed to include the Yangtze River Delta by province is filled in from the same city's records in those years;
    # the cities that never carry a province are all autonomous prefectures, prefectures (diqu), leagues and the like,
    # none of which lie in the four Yangtze River Delta provinces and municipalities.
    absent = sorted(set(KEY_CITIES) - set(p[CITY]))
    if absent:
        raise SystemExit(f"重点区域名单中的城市不在面板里：{absent}")
    known = p.dropna(subset=["省级行政区"])
    if known.groupby(CITY)["省级行政区"].nunique().max() != 1:
        raise SystemExit("面板中有城市对应多个省级行政区，无法按城市补齐省份")
    city_prov = known.groupby(CITY)["省级行政区"].first()
    p["重点区域"] = (p[CITY].isin(KEY_CITIES) | p[CITY].map(city_prov).isin(KEY_PROVINCES)).astype(float)
    p = p[p["NO2_有效日"] >= MIN_DAYS].copy()
    p["城市年"] = p[CITY].astype(str) + "_" + p[TIME].astype(str)
    # Each pollutant is filtered by **its own** valid-day threshold (same rule as apply_threshold in script 47).
    # Filtering on NO₂ valid days alone would let in station-years with too few PM2.5 / SO₂ valid days.
    for src, dst, dcol in [("NO2_年均", "ln_NO2", "NO2_有效日"),
                           ("SO2_年均", "ln_SO2", "SO2_有效日"),
                           ("PM2.5_年均", "ln_PM25", "PM2.5_有效日"),
                           ("NO2_峰谷比", "ln_峰谷比", "NO2_有效日")]:
        v = p[src].where(p[src] > 0)
        if dcol in p.columns:
            v = v.where(p[dcol] >= MIN_DAYS)
        p[dst] = np.log(v)
    n = p.dropna(subset=["ln_NO2"]).groupby("城市年")[STA].nunique()
    p["城内站数"] = p["城市年"].map(n)
    return p[p["城内站数"] >= 2].copy()


def gradient(df: pd.DataFrame, ycol: str) -> tuple[float, float, int]:
    """Within-city gradient: ln Y ~ exposure + city fixed effects; SE clustered by city."""
    g = df.dropna(subset=[ycol, EXPO])
    if g[CITY].nunique() < 10:
        return np.nan, np.nan, len(g)
    z = g[[ycol, EXPO]].astype(float)
    z = z - z.groupby(g[CITY].values).transform("mean")
    X, y = z[[EXPO]].values, z[ycol].values
    XtX_i = np.linalg.pinv(X.T @ X)
    b = XtX_i @ X.T @ y
    u = y - X @ b
    gid = pd.factorize(g[CITY])[0]
    meat = sum(np.outer(X[gid == k].T @ u[gid == k], X[gid == k].T @ u[gid == k])
               for k in np.unique(gid))
    G, N = len(np.unique(gid)), len(y)
    V = XtX_i @ np.atleast_2d(meat) @ XtX_i * (G / (G - 1)) * ((N - 1) / (N - 1))
    return float(b[0]), float(np.sqrt(V[0, 0])), len(g)


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("51_station_timing_and_power")
    d = load()
    logger.info("样本 %s 站-年，站点 %s，城市 %s", len(d), d[STA].nunique(), d[CITY].nunique())

    # ── ① Year-by-year gradient path (four series + split by priority region) ──
    rows = []
    for tag, ycol in SERIES:
        if ycol not in d.columns:
            continue
        for yr, g in d.groupby(TIME):
            b, se, n = gradient(g, ycol)
            rows.append({"序列": tag, "年份": int(yr), "城内梯度": round(b, 4),
                         "SE": round(se, 4), "N": n, "样本": "全部"})
        for key, lab in [(1.0, "重点区域"), (0.0, "非重点区域")]:
            for yr, g in d[d["重点区域"] == key].groupby(TIME):
                b, se, n = gradient(g, ycol)
                rows.append({"序列": tag, "年份": int(yr), "城内梯度": round(b, 4),
                             "SE": round(se, 4), "N": n, "样本": lab})
    path = pd.DataFrame(rows)

    # ── ② Timing check against the national NEV path ──
    c = pd.read_csv(export_utils.DATA_DIR / "城市面板-2017_2023.csv")
    c.columns = [x.lstrip("﻿") for x in c.columns]
    # The national NEV series uses the **31-province total of the provincial data on the Ministry of Public Security
    # registration basis**, not the sum of the city panel: the city panel contains an "unallocated within the province
    # in that year" residual; its 2017 total is only 112 × 10⁴ vehicles, 27% below the provincial total of
    # 154 × 10⁴ vehicles, and the understatement shrinks year by year (below 1% by 2022), so using it directly would
    # overstate the 2017–2023 log growth (2.90 vs 2.58).
    prov = pd.read_csv(export_utils.DATA_DIR / "省级新能源汽车保有量-公安部口径-2017_2023.csv")
    prov.columns = [x.lstrip("﻿") for x in prov.columns]
    nev = prov.groupby(TIME)["新能源汽车保有量_辆"].sum()
    g0 = path[(path["序列"] == "NO2") & (path["样本"] == "全部")].set_index("年份")["城内梯度"]
    tim = pd.DataFrame({"NO₂城内梯度": g0, "全国NEV保有量_万辆": (nev / 1e4).round(0)}).dropna()
    tim["梯度累计收窄%"] = (100 * (tim["NO₂城内梯度"] - tim["NO₂城内梯度"].iloc[0])
                            / tim["NO₂城内梯度"].iloc[0]).round(1)
    tim["NEV累计增量占比%"] = (100 * (tim["全国NEV保有量_万辆"] - tim["全国NEV保有量_万辆"].iloc[0])
                              / (tim["全国NEV保有量_万辆"].iloc[-1] - tim["全国NEV保有量_万辆"].iloc[0])).round(0)
    tim["梯度收窄完成度%"] = (100 * (tim["NO₂城内梯度"] - tim["NO₂城内梯度"].iloc[0])
                             / (tim["NO₂城内梯度"].iloc[-1] - tim["NO₂城内梯度"].iloc[0])).round(0)
    mism = []
    for cut in tim.index[1:-1]:
        mism.append({"截至年份": int(cut),
                     "梯度收窄完成度%": float(tim.loc[cut, "梯度收窄完成度%"]),
                     "NEV增量完成度%": float(tim.loc[cut, "NEV累计增量占比%"]),
                     "错配（梯度领先 pp）": round(float(tim.loc[cut, "梯度收窄完成度%"]
                                                      - tim.loc[cut, "NEV累计增量占比%"]), 0)})
    mis = pd.DataFrame(mism)
    # Annual increments use the unrounded 31-province total: rounding to 10⁴ vehicles pulls r from 0.747 to 0.745,
    # which then no longer matches the point estimate of the bootstrap in script 61
    dg, dn = tim["NO₂城内梯度"].diff().dropna(), (nev / 1e4).loc[tim.index].diff().dropna()
    r_year = float(np.corrcoef(dg, dn)[0, 1])
    logger.info("时序：截至 2020 梯度收窄完成 %.0f%%，NEV 增量只完成 %.0f%%",
                tim.loc[2020, "梯度收窄完成度%"], tim.loc[2020, "NEV累计增量占比%"])
    logger.info("年度变化相关 r=%.3f（电动化驱动收窄应为显著负相关）", r_year)

    # ── ②b Pre-electrification baseline: narrowing speed in 2015–2017 vs after 2017 ──
    # In 2015–2016 NEVs were below 1% of the fleet (2017, Ministry of Public Security registration basis:
    # 154 × 10⁴ ÷ 2.17 × 10⁸ vehicles = 0.71%), so the narrowing in those two years cannot have been caused by
    # electrification. If the **average annual narrowing speed** of the pre-electrification baseline is comparable to
    # that afterwards, then "narrowing preceded electrification" no longer depends on the identifying power of the
    # regression, nor on the detectability assessment.
    pre_rows = []
    for tag, ycol in SERIES:
        if ycol not in d.columns:
            continue
        gs = {}
        for yr in sorted(d[TIME].unique()):
            bq, _, _ = gradient(d[d[TIME] == yr], ycol)
            if not np.isnan(bq):
                gs[int(yr)] = bq
        if not gs:
            continue
        ys = sorted(gs)
        seg = [(f"电动化前 {ys[0]}→2017", ys[0], 2017), (f"电动化期 2017→{ys[-1]}", 2017, ys[-1])]
        for lab, a0, a1 in seg:
            if a0 not in gs or a1 not in gs or a1 <= a0:
                continue
            pre_rows.append({"序列": tag, "区间": lab, "起始梯度": round(gs[a0], 4),
                             "结束梯度": round(gs[a1], 4),
                             "年均变化": round((gs[a1] - gs[a0]) / (a1 - a0), 5),
                             "年均相对变化%": round(100 * (gs[a1] - gs[a0]) / gs[a0] / (a1 - a0), 2)})
    pre = pd.DataFrame(pre_rows)
    if not pre.empty:
        for tag, g in pre.groupby("序列", sort=False):
            if len(g) == 2:
                r0, r1 = g["年均相对变化%"].iloc[0], g["年均相对变化%"].iloc[1]
                logger.info("前基线对比 %s：电动化前年均 %+.2f%%／年，电动化期 %+.2f%%／年", tag, r0, r1)

    # ── ③ Cross-section: change in each city's gradient vs that city's NEV growth ──
    # The window must be aligned with the years covered by the NEV data (2017–2023); otherwise the 2015→2024 gradient
    # change would be compared with the 2017→2023 NEV growth, and each end would add a stretch of change unrelated to
    # NEVs.
    nev_y0, nev_y1 = int(nev.index.min()), int(nev.index.max())
    cs_rows = []
    for city, g in d[(d[TIME] >= nev_y0) & (d[TIME] <= nev_y1)].dropna(
            subset=["ln_NO2", EXPO]).groupby(CITY):
        yrs = sorted(g[TIME].unique())
        if len(yrs) < 5 or yrs[0] != nev_y0 or yrs[-1] != nev_y1:
            continue
        grad_by_year = {}
        for yr in (yrs[0], yrs[-1]):
            gg = g[g[TIME] == yr]
            if gg[STA].nunique() < 3 or gg[EXPO].std() < 1e-6:
                continue
            A = np.column_stack([np.ones(len(gg)), gg[EXPO].to_numpy(float)])
            grad_by_year[yr] = float(np.linalg.lstsq(A, gg["ln_NO2"].to_numpy(float), rcond=None)[0][1])
        if len(grad_by_year) == 2:
            cs_rows.append({"城市": city, "梯度变化": grad_by_year[yrs[-1]] - grad_by_year[yrs[0]],
                            "首年": yrs[0], "末年": yrs[-1], "站点数": g[STA].nunique()})
    cs = pd.DataFrame(cs_rows)
    cn = c[c["NEV保有量_辆"] > 0].pivot_table(index="城市", columns=TIME, values="NEV保有量_辆")
    gr = np.log(cn[nev_y1] / cn[nev_y0]).rename("NEV对数增幅").reset_index()
    cs["城市名"] = cs["城市"].astype(str).str.replace(r"(市|地区|自治州|盟)$", "", regex=True)
    m = cs.merge(gr.rename(columns={"城市": "城市名"}), on="城市名", how="inner").dropna(
        subset=["梯度变化", "NEV对数增幅"])
    r_cs, p_cs = (np.nan, np.nan)
    if len(m) >= 20:
        r_cs = float(np.corrcoef(m["梯度变化"], m["NEV对数增幅"])[0, 1])
        tt = r_cs * np.sqrt((len(m) - 2) / (1 - r_cs ** 2))
        p_cs = float(2 * (1 - stats.t.cdf(abs(tt), len(m) - 2)))
        logger.info("横截面 %s 城（窗口 %s–%s，与 NEV 覆盖对齐）：梯度变化 vs NEV 增幅 r=%+.3f (p=%.3g)",
                    len(m), nev_y0, nev_y1, r_cs, p_cs)

    # ── ④ Detectability assessment ──
    # Year-dummy specification: the full set of year dummies × exposure + NEV × exposure, with no shape restriction on
    # the trend.
    # It is not the same specification as robustness check S16 in script 48 (Table S2): S16 includes a linear trend
    # t × exposure, so the β and SE of the two are not expected to be equal (S16 in script 48 gives 0.0000/0.0423,
    # this one gives -0.0014/0.0461).
    # Table 2 of the main text lists them as two separate columns, "plus trend" and "year dummies"; the output labels
    # here must match that, otherwise anyone checking would think the same specification produced two numbers.
    dd = d.dropna(subset=["ln_NO2", EXPO, "NEV保有量_辆"]).reset_index(drop=True)
    dd = dd[dd["NEV保有量_辆"] > 0].copy()
    dd["Expo_c"] = dd[EXPO] - dd[EXPO].mean()
    dd["NEVxExpo"] = np.log(dd["NEV保有量_辆"]) * dd["Expo_c"]
    ycols = []
    for yv in sorted(dd[TIME].unique())[1:]:
        cc = f"y{yv}xExpo"
        dd[cc] = (dd[TIME] == yv).astype(float) * dd["Expo_c"]
        ycols.append(cc)
    # Same implementation as the other columns of Table 2 in script 48 (linearmodels, clustered by city, degrees of
    # freedom counting the absorbed fixed effects).
    # A hand-written implementation deducts only the number of regressors from the degrees of freedom, which makes the
    # standard errors about 30% smaller; the two conventions must not be mixed in one table.
    pdf = dd.set_index([STA, TIME])
    res = PanelOLS(pdf["ln_NO2"], pdf[["NEVxExpo", *ycols]], entity_effects=True,
                   other_effects=pdf[["城市年"]], drop_absorbed=True, check_rank=False).fit(
        cov_type="clustered", clusters=pdf[[CITY]])
    b_nev, se_nev = float(res.params["NEVxExpo"]), float(res.std_errors["NEVxExpo"])
    mde = 2.8 * se_nev                       # 80% power, 5% two-sided
    # Coefficient implied by "the narrowing is caused entirely by NEVs". The treatment variable is each city's own ln
    # vehicle stock: under full attribution the gradient of city c changes by β·Δln V_c, and the yearly gradient is a
    # weighted mean of the city gradients with the within-city sum of squared exposure deviations as weights, so the
    # difference between the gradients in the first and last years of the regression sample is exactly β times the
    # difference in the means of ln V_c under the same weights. The latter is the within-city slope of NEVxExpo on
    # exposure.
    y0n, y1n = int(dd[TIME].min()), int(dd[TIME].max())
    d_grad = gradient(dd[dd[TIME] == y1n], "ln_NO2")[0] - gradient(dd[dd[TIME] == y0n], "ln_NO2")[0]
    d_lnnev = gradient(dd[dd[TIME] == y1n], "NEVxExpo")[0] - gradient(dd[dd[TIME] == y0n], "NEVxExpo")[0]
    implied = d_grad / d_lnnev
    # The p value is output here as well: the main text reports it
    p_nev = float(res.pvalues["NEVxExpo"])
    # Supplementary Material S6 also reports: the power against the implied coefficient, how many standard errors the
    # estimate lies from the implied coefficient and the two-sided p of that distance, and the 95% interval
    z_imp = abs(implied) / se_nev
    power = float(stats.norm.cdf(z_imp - 1.96) + stats.norm.cdf(-z_imp - 1.96))
    z_gap = abs(b_nev - implied) / se_nev
    pw = pd.DataFrame([{
        "设定": "年份哑变量×暴露 + NEV×暴露（正文表 2 第 4 列）",
        "β_NEV": round(b_nev, 6), "SE": round(se_nev, 5), "p": round(p_nev, 4),
        "最小可检测效应 MDE(=2.8·SE)": round(mde, 5),
        "样本城市 lnNEV 增幅（梯度权重均值）": round(d_lnnev, 3),
        "样本内梯度变化（首末年）": round(d_grad, 4),
        "「收窄全由NEV造成」隐含系数": round(implied, 5),
        "隐含系数 / MDE": round(abs(implied) / mde, 2),
        "对隐含系数的功效（5% 双侧）": round(power, 3),
        "估计值距隐含系数（SE 倍数）": round(z_gap, 2),
        "该距离的双侧 p": round(float(2 * stats.norm.sf(z_gap)), 3),
        "β_NEV 的 95% 区间": f"[{b_nev - 1.96 * se_nev:.4f}, {b_nev + 1.96 * se_nev:.4f}]",
        "判定": ("有识别力：隐含系数大于 MDE，若 NEV 造成收窄本设计能看到，"
                 "而实测系数为零" if abs(implied) > mde else
                 "识别力不足：隐含系数小于 MDE，零系数不能排除 NEV 的作用")}])
    logger.info("可检测性：β=%.6f (SE %.5f, p=%.4f)，MDE=%.5f，隐含系数=%.5f → %s",
                b_nev, se_nev, p_nev, mde, implied, pw["判定"].iloc[0])

    # Why the denominator of the implied coefficient exceeds the national ln growth (Supplementary Material S6): the
    # national total and the total over the sample cities are both "the log of a total", dominated by the cities that
    # already had large fleets in 2017; the regression uses each city's own ln growth, and cities starting from a small
    # base grow more.
    first, last = dd[dd[TIME] == y0n], dd[dd[TIME] == y1n]
    wy = dd.groupby([CITY, TIME])["NEV保有量_辆"].first().unstack()[[y0n, y1n]].dropna()
    dl = np.log(wy[y1n] / wy[y0n])
    small = wy[y0n] < 1000
    dec = pd.DataFrame([
        {"项": f"{y0n} 年回归样本：城市数 / 站点数", "值": f"{first[CITY].nunique()} / {first[STA].nunique()}"},
        {"项": f"{y1n} 年回归样本：城市数 / 站点数", "值": f"{last[CITY].nunique()} / {last[STA].nunique()}"},
        {"项": f"{y0n} 年梯度", "值": round(gradient(first, "ln_NO2")[0], 4)},
        {"项": f"{y1n} 年梯度", "值": round(gradient(last, "ln_NO2")[0], 4)},
        {"项": f"{y0n} 年 ln 保有量（梯度权重均值）", "值": round(gradient(first, "NEVxExpo")[0], 4)},
        {"项": f"{y1n} 年 ln 保有量（梯度权重均值）", "值": round(gradient(last, "NEVxExpo")[0], 4)},
        {"项": "两年均在样本的城市数", "值": len(wy)},
        {"项": "城市 ln 增幅：下四分位数", "值": round(float(dl.quantile(0.25)), 3)},
        {"项": "城市 ln 增幅：中位数", "值": round(float(dl.median()), 3)},
        {"项": "城市 ln 增幅：上四分位数", "值": round(float(dl.quantile(0.75)), 3)},
        {"项": "城市 ln 增幅：城市等权均值", "值": round(float(dl.mean()), 3)},
        {"项": f"{y0n} 年不足 1000 辆的城市数", "值": int(small.sum())},
        {"项": f"{y0n} 年不足 1000 辆的城市：ln 增幅均值", "值": round(float(dl[small].mean()), 3)},
        {"项": f"{y0n} 年 1000 辆及以上的城市：ln 增幅均值", "值": round(float(dl[~small].mean()), 3)},
        {"项": "两年均在样本的城市合计：ln 增幅", "值": round(float(np.log(wy[y1n].sum() / wy[y0n].sum())), 3)},
        {"项": "全国（31 省合计）：ln 增幅", "值": round(float(np.log(nev[y1n] / nev[y0n])), 3)},
    ])
    logger.info("基准分解：\n%s", dec.to_string(index=False))

    summ = pd.DataFrame([
        {"项": "梯度收窄完成度 @2019", "值": f"{tim.loc[2019, '梯度收窄完成度%']:.0f}%"},
        {"项": "NEV 增量完成度 @2019", "值": f"{tim.loc[2019, 'NEV累计增量占比%']:.0f}%"},
        {"项": "梯度收窄完成度 @2020", "值": f"{tim.loc[2020, '梯度收窄完成度%']:.0f}%"},
        {"项": "NEV 增量完成度 @2020", "值": f"{tim.loc[2020, 'NEV累计增量占比%']:.0f}%"},
        {"项": "年度变化相关 r（梯度变化 vs NEV 增量）", "值": f"{r_year:+.3f}"},
        {"项": f"横截面相关 r（{len(m)} 城）", "值": f"{r_cs:+.3f}（p={p_cs:.3g}）"},
        {"项": "年份哑变量设定 β_NEV", "值": f"{b_nev:+.6f}（SE {se_nev:.5f}, p {p_nev:.4f}）"},
        {"项": "MDE vs 隐含系数", "值": f"{mde:.5f} vs {implied:.5f}"},
    ])

    export_utils.write_excel_workbook("51_梯度路径与可检测性", [
        ("摘要", summ), ("逐年梯度路径", path), ("时序核对", tim.reset_index()),
        ("时序错配", mis), ("电动化前基线对比", pre), ("横截面_城市", m), ("可检测性", pw), ("基准分解", dec)])

    print("✓ 梯度路径与可检测性完成\n")
    print(tim.to_string())
    print(f"\n  截至 2020：梯度收窄完成 {tim.loc[2020, '梯度收窄完成度%']:.0f}%，"
          f"NEV 增量只完成 {tim.loc[2020, 'NEV累计增量占比%']:.0f}%")
    print(f"  年度变化相关 r = {r_year:+.3f}；横截面 {len(m)} 城 r = {r_cs:+.3f}（p={p_cs:.3g}）")
    if not pre.empty:
        print("\n  电动化前基线 vs 电动化期（年均相对变化 %／年）：")
        print(pre.pivot_table(index="序列", columns="区间", values="年均相对变化%",
                              sort=False).to_string())
    print(f"\n  年份哑变量设定 β_NEV = {b_nev:+.6f}（SE {se_nev:.5f}, p {p_nev:.4f}）；MDE {mde:.5f}，"
          f"「收窄全由 NEV 造成」隐含 {implied:.5f}")
    print(f"  → {pw['判定'].iloc[0]}")
    print(f"\n  输出: {export_utils.OUTPUT_DIR / '51_梯度路径与可检测性.xlsx'}\n  日志: {log_path}")


if __name__ == "__main__":
    main()
