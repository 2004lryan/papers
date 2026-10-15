"""
47_station_pretest.py: pre-tests P1–P5 for the station-level design

**P1 is the prerequisite of the main regression (script 48).** All the identifying power of the station design comes
from "differences between stations with different exposure within the same city"; if NO₂ does not vary with built-up
exposure within cities at all, the exposure measure is invalid and no coefficient of the main regression can be
interpreted. This script only reports the verdict; it does not stop later scripts from running.

P1 Within-city spatial gradient: ln NO₂ ~ built-up exposure + city-by-year fixed effects. The coefficient must be
   significantly positive.
P2 Time shape of the gradient: estimate the P1 coefficient year by year for NO₂, and for SO₂ and PM2.5 as
   comparisons, and see whether the gradient narrows over time (a necessary condition for the hypothesis that
   electrification narrowed the gradient; descriptive); P2b compares the first-to-last-year change across the three.
P3 Sample selection: whether stations that report NO₂ but are absent from the station register, and so never enter
   the panel, differ in mean NO₂ from registered stations, in 2017 and in 2023 (Welch t test); and, inside the panel,
   whether stations valid in 2017 but not in 2023 differ from retained stations in base-period NO₂ and exposure.
P4 Exposure validity: correlation between the built-up fraction and the OSM weighted road density; computed only
   when at least 5 stations have a positive road density (only 4 do in the panel used here, so P4 is skipped).
P5 Differences in NEV trajectories between cities: residual variation of ln NEV after city and year fixed
   effects, the precondition for check S16 (Table S2 of the supplement) to have identifying power.

Usage:
    python code/47_station_pretest.py

Output: outputs/47_站点前置检验.xlsx (station pre-tests); logs/47_station_pretest.log
"""

from __future__ import annotations

import importlib.util
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats

if TYPE_CHECKING:
    from linearmodels.panel.results import PanelEffectsResults

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)
warnings.filterwarnings("ignore")

STA, CITY, TIME = "站点编号", "所属城市", "年份"
EXPO = "建成区占比_500m"
# The validity threshold for annual means is applied here (script 46 only outputs the number of valid days and does
# not filter).
# Main threshold 274 days ≈ 75% (the US EPA requires 75% of hours for the NO₂ annual mean); robustness threshold
# 324 days = the annual-mean threshold in Table 4 of GB 3095-2012 (about 90%).
# In 2018 the national-standard threshold falls exactly at the median of the valid-day distribution (323 days), so
# using it alone would cut half of that year's stations.
MIN_DAYS_MAIN, MIN_DAYS_STRICT = 274, 324


def apply_threshold(p: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """Set to missing the annual means of station-years below the valid-day threshold (rows are kept, so that the
    sample sizes under the two thresholds can be compared)."""
    q = p.copy()
    for c, dcol in [("NO2_年均", "NO2_有效日"), ("PM2.5_年均", "PM2.5_有效日"),
                    ("PM10_年均", "PM10_有效日"), ("SO2_年均", "SO2_有效日"),
                    ("CO_年均", "CO_有效日"), ("O3_8h_90分位", "O3_有效日")]:
        if c in q.columns and dcol in q.columns:
            q.loc[q[dcol] < min_days, c] = np.nan
    return q


def fit_cityyear(
    d: pd.DataFrame, y: str, xs: list[str], cluster_col: str = CITY
) -> tuple[PanelEffectsResults, pd.DataFrame]:
    """City-by-year fixed effects (no station fixed effects: P1 is precisely about the cross-sectional differences
    between stations)."""
    dd = d.dropna(subset=[y, *xs]).copy()
    dd["_i"] = np.arange(len(dd))
    pdf = dd.set_index(["_i", TIME])
    res = PanelOLS(pdf[y], pdf[xs], other_effects=pdf[["城市年"]],
                   drop_absorbed=True, check_rank=False).fit(
        cov_type="clustered", clusters=pdf[[cluster_col]])
    return res, dd


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("47_station_pretest")
    raw = pd.read_csv(export_utils.DATA_DIR / "站点年面板-2015_2024.csv")
    logger.info("面板原始 %s 行；按年有效日中位：%s", len(raw),
                raw.groupby(TIME)["NO2_有效日"].median().round(0).to_dict())
    p = apply_threshold(raw, MIN_DAYS_MAIN)
    p["城市年"] = p[CITY].astype(str) + "_" + p[TIME].astype(str)
    p["ln_NO2"] = np.log(p["NO2_年均"].where(p["NO2_年均"] > 0))
    p["ln_SO2"] = np.log(p["SO2_年均"].where(p["SO2_年均"] > 0))
    p["ln_PM25"] = np.log(p["PM2.5_年均"].where(p["PM2.5_年均"] > 0))
    # Keep only city-years with ≥2 stations in the city; only then do city-by-year fixed effects leave a within-city
    # contrast
    p["城内站数"] = p["城市年"].map(p.dropna(subset=["ln_NO2"]).groupby("城市年")[STA].nunique())
    d = p[(p["城内站数"] >= 2) & p["ln_NO2"].notna() & p[EXPO].notna()].copy()
    logger.info("面板 %s 行；城内 ≥2 站的样本 %s 行，站点 %s，城市 %s",
                len(p), len(d), d[STA].nunique(), d[CITY].nunique())

    sheets, out_lines = [], []

    # ── P1 within-city spatial gradient ──
    res, dd = fit_cityyear(d, "ln_NO2", [EXPO])
    b, se, pv = float(res.params[EXPO]), float(res.std_errors[EXPO]), float(res.pvalues[EXPO])
    p1_ok = (b > 0) and (pv < 0.05)
    logger.info("P1 ln NO2 ~ 建成区暴露 + 城市×年FE：β=%.4f (SE %.4f, p=%.4g)，N=%s → %s",
                b, se, pv, int(res.nobs), "通过" if p1_ok else "不通过")
    out_lines.append(f"P1 城市内梯度：β={b:.4f}（SE {se:.4f}, p={pv:.3g}），N={int(res.nobs)}，"
                     f"判定 {'通过' if p1_ok else '不通过'}")
    # Magnitude: how much NO2 differs between exposure at p10 and at p90.
    # The dependent variable is in logs, so the percentage difference must be exp(βΔ)-1, not 100βΔ directly: the
    # latter is a log-point difference, and with a Δ this large the two differ by more than one percentage point
    # (15.1 vs 16.3), and readers read the figure as a concentration ratio.
    lo, hi = d[EXPO].quantile(.1), d[EXPO].quantile(.9)
    pct = 100 * (math.exp(b * (hi - lo)) - 1)
    out_lines.append(f"   暴露 p10→p90（{lo:.3f}→{hi:.3f}）对应 NO₂ 相差 {pct:.1f}%"
                     f"（对数点差 {100*b*(hi-lo):.1f}%）")
    p1 = pd.DataFrame([{"检验": "P1 城市内空间梯度", "系数": round(b, 4), "SE": round(se, 4), "p": round(pv, 5),
                        "N": int(res.nobs), "站点数": dd[STA].nunique(), "城市数": dd[CITY].nunique(),
                        # The main text reports both quantiles and their difference; they are output here as
                        # well, for checking the 16.3% conversion
                        "暴露 p10": round(lo, 4), "暴露 p90": round(hi, 4), "暴露 p10→p90 差": round(hi - lo, 4),
                        "p10→p90 的 NO₂ 差异%": round(pct, 1),
                        "同上_对数点差%": round(100 * b * (hi - lo), 1),
                        "判定": "通过" if p1_ok else "不通过"}])
    # Comparison: the same design with SO2 (an industrial tracer); its gradient should be much weaker
    res_s, _ = fit_cityyear(d[d["ln_SO2"].notna()], "ln_SO2", [EXPO])
    bs, ps = float(res_s.params[EXPO]), float(res_s.pvalues[EXPO])
    p1 = pd.concat([p1, pd.DataFrame([{"检验": "P1 对照 SO₂", "系数": round(bs, 4),
                                       "SE": round(float(res_s.std_errors[EXPO]), 4), "p": round(ps, 5),
                                       "N": int(res_s.nobs), "判定": "参考"}])], ignore_index=True)
    out_lines.append(f"   对照 SO₂：β={bs:.4f}（p={ps:.3g}）；NO₂/SO₂ 梯度比 {b/bs:.2f}" if bs else "")
    # Sensitivity of P1 to the threshold: re-estimate with the GB 3095-2012 national-standard threshold; the verdict
    # should not flip
    q = apply_threshold(raw, MIN_DAYS_STRICT)
    q["城市年"] = q[CITY].astype(str) + "_" + q[TIME].astype(str)
    q["ln_NO2"] = np.log(q["NO2_年均"].where(q["NO2_年均"] > 0))
    q["城内站数"] = q["城市年"].map(q.dropna(subset=["ln_NO2"]).groupby("城市年")[STA].nunique())
    dq = q[(q["城内站数"] >= 2) & q["ln_NO2"].notna() & q[EXPO].notna()]
    res_q, _ = fit_cityyear(dq, "ln_NO2", [EXPO])
    bq, pq = float(res_q.params[EXPO]), float(res_q.pvalues[EXPO])
    p1 = pd.concat([p1, pd.DataFrame([{"检验": f"P1 门槛敏感性（国标 {MIN_DAYS_STRICT} 天）",
                                       "系数": round(bq, 4), "SE": round(float(res_q.std_errors[EXPO]), 4),
                                       "p": round(pq, 5), "N": int(res_q.nobs), "站点数": dq[STA].nunique(),
                                       "判定": "同向显著" if (bq > 0 and pq < .05) == p1_ok else "与主口径不一致"}])],
                   ignore_index=True)
    out_lines.append(f"   门槛敏感性（国标 {MIN_DAYS_STRICT} 天，N={int(res_q.nobs)}）：β={bq:.4f}（p={pq:.3g}）")
    logger.info("P1 门槛敏感性：主口径 %s 天 β=%.4f (N=%s) vs 国标 %s 天 β=%.4f (N=%s)",
                MIN_DAYS_MAIN, b, int(res.nobs), MIN_DAYS_STRICT, bq, int(res_q.nobs))
    sheets.append(("P1_城市内梯度", p1))

    # ── P2 time shape of the gradient (NO₂ main; SO₂ / PM2.5 as comparisons) ──
    # Key comparison: if the within-city gradients of SO₂ and PM2.5 narrow **in the same proportion**, the narrowing
    # of NO₂ is only a "general improvement of the city core", not a change in traffic sources, and a traffic
    # (electrification) reading of the narrowing does not hold.
    rows = []
    for tag, ycol in [("NO2", "ln_NO2"), ("SO2", "ln_SO2"), ("PM2.5", "ln_PM25")]:
        if ycol not in d.columns:
            continue
        for yr, g in d[d[ycol].notna()].groupby(TIME):
            if g[CITY].nunique() < 30:
                continue
            r, _ = fit_cityyear(g, ycol, [EXPO])
            rows.append({"污染物": tag, "年份": yr, "梯度系数": round(float(r.params[EXPO]), 4),
                         "SE": round(float(r.std_errors[EXPO]), 4), "p": round(float(r.pvalues[EXPO]), 5),
                         "N": int(r.nobs), "站点数": g[STA].nunique(), "城市数": g[CITY].nunique()})
    p2 = pd.DataFrame(rows)
    if not p2.empty:
        summ = []
        for tag, gg in p2.groupby("污染物", sort=False):
            gg = gg.sort_values("年份")
            b0, b1 = gg["梯度系数"].iloc[0], gg["梯度系数"].iloc[-1]
            rel = (b1 - b0) / b0 if b0 else np.nan
            summ.append({"污染物": tag, "首年": int(gg["年份"].iloc[0]), "首年梯度": b0,
                         "末年": int(gg["年份"].iloc[-1]), "末年梯度": b1,
                         "绝对变化": round(b1 - b0, 4), "相对变化%": round(100 * rel, 1)})
            out_lines.append(f"P2 {tag} 城内梯度：{int(gg['年份'].iloc[0])} {b0:.4f} → "
                             f"{int(gg['年份'].iloc[-1])} {b1:.4f}（{100*rel:+.1f}%，"
                             f"{'收窄' if b1 < b0 else '扩大'}）")
            logger.info("P2 %s 梯度 %.4f → %.4f（%+.1f%%）", tag, b0, b1, 100 * rel)
        sheets.append(("P2b_梯度收窄对照", pd.DataFrame(summ)))
        no2 = next((x for x in summ if x["污染物"] == "NO2"), None)
        others = [x for x in summ if x["污染物"] != "NO2"]
        if no2 and others:
            # Narrowing is recorded as a negative value, so "the comparison that narrowed most" is the **smallest**.
            # Criterion: the traffic-source interpretation is supported only if NO₂ narrowed more than **every**
            # comparison pollutant; as soon as one comparison narrowed at least as much as NO₂, the narrowing may be
            # only a general improvement of the city core.
            strongest = min(x["相对变化%"] for x in others)
            who = min(others, key=lambda x: x["相对变化%"])["污染物"]
            verdict = ("NO₂ 收窄幅度大于所有对照污染物，与交通来源解释一致"
                       if no2["相对变化%"] < strongest else
                       f"对照污染物收窄不亚于 NO₂（{who} {strongest:+.1f}% vs NO₂ "
                       f"{no2['相对变化%']:+.1f}%）—— 收窄可能只是城市核心普遍改善，"
                       "D1 的机制解释存疑")
            out_lines.append(f"   → {verdict}")
            logger.info("P2 判读：%s", verdict)
    sheets.append(("P2_梯度时间形状", p2))

    # ── P3 sample selection: stations that cannot enter the panel vs stations that can ──
    # The panel is built by an **inner join** with the station register from script 43, and the register covers only
    # stations that are still being published. The real attrition is therefore not inside the panel (almost every
    # station valid in 2017 within the panel survives to 2023) but the **whole group of stations that never enter the
    # panel**: they have observations in the hourly data but no coordinates and no city attribution.
    # This group can only be compared using the raw hourly data, and only on concentration levels (no coordinates
    # means no exposure).
    p3_rows: list[dict[str, Any]] = []
    meta_ids = set()
    mpath = export_utils.DATA_DIR / "国控站点元数据.csv"
    if mpath.exists():
        meta_ids = set(pd.read_csv(mpath)["站点编号"].astype(str))
    for yr in (2017, 2023):
        src = export_utils.DATA_DIR / f"站点小时空气质量-{yr}.parquet"
        if not src.exists() or not meta_ids:
            continue
        h = pd.read_parquet(src, columns=["站点编号", "日期", "指标", "值"])
        h = h[h["指标"].astype(str) == "NO2"]
        st = (h.groupby(h["站点编号"].astype(str))["值"]
              .agg(NO2均值="mean", 小时数="count").reset_index()
              .rename(columns={"站点编号": "站点"}))
        st = st[st["小时数"] >= 24 * MIN_DAYS_MAIN]
        st["进入面板"] = st["站点"].isin(meta_ids)
        g = st.groupby("进入面板")["NO2均值"]
        row: dict[str, float] = {"年份": yr, "报数站点": len(st),
               "进入面板": int((st["进入面板"]).sum()),
               "未进面板": int((~st["进入面板"]).sum())}
        if st["进入面板"].nunique() == 2:
            a = st[st["进入面板"]]["NO2均值"]
            b = st[~st["进入面板"]]["NO2均值"]
            t = stats.ttest_ind(a, b, equal_var=False)
            row |= {"进入面板_NO2均值": round(float(a.mean()), 2),
                    "未进面板_NO2均值": round(float(b.mean()), 2),
                    "差(µg/m³)": round(float(a.mean() - b.mean()), 2),
                    "t": round(float(t.statistic), 2), "p": round(float(t.pvalue), 5)}
        p3_rows.append(row)
        del h, st
    p3 = pd.DataFrame(p3_rows)
    if not p3.empty:
        for _, r in p3.iterrows():
            if "t" in r and pd.notna(r.get("t")):
                out_lines.append(f"P3 {int(r['年份'])} 年：报数站 {int(r['报数站点'])}，"
                                 f"其中 {int(r['未进面板'])} 站因无元数据未进面板；"
                                 f"两组基期 NO₂ 差 {r['差(µg/m³)']:+.2f} µg/m³"
                                 f"（t={r['t']:.2f}, p={r['p']:.3g}）")
            else:
                out_lines.append(f"P3 {int(r['年份'])} 年：报数站 {int(r['报数站点'])}，"
                                 f"未进面板 {int(r['未进面板'])} 站")
            logger.info("P3 %s", out_lines[-1])
    # Entry and exit inside the panel (for comparison: the panel is an inner join with the register and is nearly
    # balanced inside)
    last = set(p[(p[TIME] == 2023) & p["ln_NO2"].notna()][STA])
    base = p[(p[TIME] == 2017) & p["ln_NO2"].notna()].copy()
    base["留存到2023"] = base[STA].isin(last)
    p3b = (base.groupby("留存到2023").agg(站点数=(STA, "nunique"),
                                          基期NO2均值=("NO2_年均", "mean"),
                                          基期暴露均值=(EXPO, "mean")).round(3).reset_index())
    out_lines.append(f"   面板内部：2017 年有效站 {int(base[STA].nunique())}，"
                     f"其中留存到 2023 的 {int(base['留存到2023'].sum())} 站"
                     f"（面板按元数据内连接构建，内部近似平衡，流失全部发生在入面板这一步）")
    sheets.append(("P3_样本选择", p3))
    sheets.append(("P3b_面板内部进出", p3b))

    # ── P4 exposure validity ──
    if "加权路网密度_500m" in p.columns:
        q = p.drop_duplicates(STA)[[STA, EXPO, "加权路网密度_500m", CITY]].dropna()
        q = q[q["加权路网密度_500m"] > 0]
        if len(q) >= 5:
            r = q[[EXPO, "加权路网密度_500m"]].corr().iloc[0, 1]
            out_lines.append(f"P4 建成区 vs 路网密度：n={len(q)}，r={r:.3f}"
                             f"{'（样本太小，须扩样后重判）' if len(q) < 50 else ''}")
            sheets.append(("P4_暴露效度", pd.DataFrame([{"n": len(q), "相关系数": round(r, 3),
                                                        "备注": "OSM 在中国覆盖不均，仅覆盖良好的站参与"}])))

    # ── P5 differences in NEV trajectories between cities (precondition for S16 to have identifying power) ──
    # Check S16 (Table S2 of the supplement; equation (4) of the main text) absorbs the "exposure-specific common
    # time trend" and keeps only the dimension "cities where NEVs grow faster". If all cities had nearly the same NEV
    # trajectory, lnNEV would have no residual variation after removing city and year fixed effects, and S16 could
    # not be identified.
    cp = export_utils.DATA_DIR / "城市面板-2017_2023.csv"
    if cp.exists():
        cdf = pd.read_csv(cp)
        cdf.columns = [c.lstrip("\ufeff") for c in cdf.columns]
        cdf = cdf[cdf["NEV保有量_辆"] > 0].dropna(subset=["NEV保有量_辆"]).copy()
        cdf["lnNEV"] = np.log(cdf["NEV保有量_辆"])
        z = cdf[["lnNEV"]].copy()
        for _ in range(80):
            prev = z.values.copy()
            z = z - z.groupby(cdf["城市"].values).transform("mean")
            z = z - z.groupby(cdf["年份"].values).transform("mean")
            if np.max(np.abs(z.values - prev)) < 1e-10:
                break
        tot, res_sd = float(cdf["lnNEV"].std()), float(z["lnNEV"].std())
        share = res_sd / tot if tot else np.nan
        w = cdf.pivot_table(index="城市", columns="年份", values="lnNEV")
        gr = (w[w.columns.max()] - w[w.columns.min()]).dropna()
        p5_ok = share >= 0.05
        p5 = pd.DataFrame([{"检验": "P5 城际 NEV 轨迹差异", "lnNEV总标准差": round(tot, 3),
                            "去城市+年FE后残差标准差": round(res_sd, 3),
                            "残差占总变异%": round(100 * share, 1),
                            "首末年增幅中位_对数": round(float(gr.median()), 3),
                            "增幅城际标准差": round(float(gr.std()), 3),
                            "城市数": int(cdf["城市"].nunique()),
                            "判定": "S16 有识别力" if p5_ok else "S16 无识别力，直接判 D0"}])
        sheets.append(("P5_NEV轨迹差异", p5))
        out_lines.append(f"P5 城际 NEV 轨迹差异：去掉城市与年固定效应后残差标准差 {res_sd:.3f}"
                         f"（占总变异 {100*share:.1f}%），首末年增幅城际标准差 {gr.std():.3f} → "
                         f"{'S16 有识别力' if p5_ok else 'S16 无识别力'}")
        logger.info("P5 %s", out_lines[-1])

    export_utils.write_excel_workbook("47_站点前置检验", sheets)
    print("✓ 站点前置检验完成\n")
    for line in out_lines:
        if line:
            print("  " + line)
    print("\n  " + ("P1 通过，可进入主回归（48 号）" if p1_ok
                    else "✗ P1 不通过：暴露口径无法识别城市内交通梯度，按计划 D0 处理"))
    print(f"  输出: {export_utils.OUTPUT_DIR / '47_站点前置检验.xlsx'}\n  日志: {log_path}")


if __name__ == "__main__":
    main()
