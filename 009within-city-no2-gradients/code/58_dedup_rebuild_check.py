"""
58_dedup_rebuild_check.py: rebuild the affected station-years after removing duplicate hourly records, and
re-estimate all result rows of Table 2 and Table S5

Why it is needed:
    Script 57 found that in 2019–2022 some days of 12 station-years were repeated in full (the two records have
    identical values). Scripts 46 and 50 count records, so 302 of these "station-day × pollutant" cells had fewer
    than 20 valid hours but were counted as valid days because their records were doubled;
    6 of these 12 station-years belong to stations outside the station register, which are not in the panel in the
    first place, so they are only logged, not rebuilt;
    script 46 rolls the O₃ daily maximum 8-hour window over adjacent records, so on the repeated days the window
    covers only half as many clock hours.
    This script rebuilds the station-years that are in the panel from the deduplicated hourly data, substitutes
    them into a copy of the panel (nothing is written back to data/), and then re-estimates every column of
    Table 2 and all result rows of Table S5 with the same implementation as script 48, reporting the change in
    each row's coefficient.

Two gates (if either one fails, the script exits before writing the workbook and names the failed comparison):
    1. Rebuilding these station-years from the hourly data without deduplication, using this script's
       re-implementation, must reproduce the current panel values column by column
       (script 46: annual means, valid days, O₃ 8h, diurnal structure; script 50: O₃ annual mean and Ox);
    2. The rows re-estimated on the original panel must agree item by item, within the precision stored, with β,
       SE, p and N in the workbooks of script 48 (the main specification, equation (3) of the manuscript, and the
       exposure-trend specification, equation (4)) and script 55 (year indicators). In the 规格 (specification)
       column the trend rows are labelled `式(3)`; they are equation (4) of the manuscript.

Usage:
    python code/58_dedup_rebuild_check.py

Output files:
    outputs/58_去重重建核验.xlsx (deduplication rebuild check) — sheets "01_01_受影响站点年" (affected station-years)
        / "02_02_重建复现核对" (rebuild reproduction check) / "03_03_系数对照" (coefficient comparison)
    logs/58_dedup_rebuild_check.log
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _load(name: str, file: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    if spec is None or spec.loader is None:
        raise SystemExit(f"{file} 无法加载")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export_utils = _load("export_utils", "01_export_utils.py")
m46 = _load("m46", "46_build_station_panel.py")      # constants only: the same thresholds and hour windows as script 46
m48 = _load("m48", "48_run_station_panel.py")        # regressions via fit() of script 48, same code as the table rows

STA, CITY, TIME = "站点编号", "所属城市", "年份"
MIN_DAYS = 274
MOLAR_VOL, MW_O3, MW_NO2 = 24.45, 48.00, 46.01        # same as script 50
MAIN = "NEVxExpo"
COLS = ["日期", "小时", STA, "指标", "值"]


def hourly_for(year: int, stations: set[str]) -> pd.DataFrame:
    """Read one year of hourly data row group by row group, keeping only the given stations."""
    # pyarrow's ParquetFile has no type annotations; the call is allowed as an unannotated third-party call
    pf = pq.ParquetFile(export_utils.DATA_DIR / f"站点小时空气质量-{year}.parquet")  # type: ignore[no-untyped-call]
    parts = []
    for i in range(pf.metadata.num_row_groups):
        g = pf.read_row_group(i, columns=COLS).to_pandas()  # type: ignore[no-untyped-call]
        g = g[g[STA].astype(str).isin(stations)]
        if len(g):
            parts.append(g.assign(**{STA: g[STA].astype(str), "指标": g["指标"].astype(str)}))
    return pd.concat(parts, ignore_index=True)


def rebuild(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Rebuild one year of station-level indicators, written the same way as daily_from_hourly() and the annual
    aggregation of script 46 and the O₃ annual mean of script 50."""
    sub = df[df["指标"].isin(m46.GASES)]
    daily = sub.groupby([STA, "日期", "指标"])["值"].agg(["mean", "count"]).reset_index()
    daily = daily[daily["count"] >= m46.MIN_HOURS_DAY]
    daily = daily.pivot_table(index=[STA, "日期"], columns="指标", values="mean").reset_index()
    daily.columns.name = None
    o3 = df[df["指标"] == "O3"].sort_values([STA, "日期", "小时"]).copy()
    o3["滑动8h"] = (o3.groupby([STA, "日期"], sort=False)["值"].rolling(8, min_periods=6).mean().to_numpy())
    o3d = o3.groupby([STA, "日期"])["滑动8h"].agg(["max", "count"]).reset_index()
    o3d = o3d[o3d["count"] >= m46.MIN_8H_DAY][[STA, "日期", "max"]].rename(columns={"max": "O3_日最大8h"})
    daily = daily.merge(o3d, on=[STA, "日期"], how="outer")
    parts = []
    for sp in m46.DIURNAL_SPECIES:
        z = df[df["指标"] == sp]
        parts.append(z[z["小时"].isin(m46.PEAK_HOURS)].groupby([STA, "日期"])["值"].mean().rename(f"{sp}_高峰均值"))
        parts.append(z[z["小时"].isin(m46.NIGHT_HOURS)].groupby([STA, "日期"])["值"].mean().rename(f"{sp}_夜间均值"))
    diurnal = pd.concat(parts, axis=1).reset_index()
    daily["日期"] = daily["日期"].astype(str)
    diurnal["日期"] = diurnal["日期"].astype(str)
    d = daily.merge(diurnal, on=[STA, "日期"], how="left")
    d["是周末"] = pd.to_datetime(d["日期"], format="%Y%m%d").dt.weekday >= 5
    agg: dict[str, tuple[str, object]] = {}
    for c in m46.GASES:
        agg[f"{c}_年均"] = (c, "mean")
        agg[f"{c}_有效日"] = (c, "count")
    agg["O3_8h_90分位"] = ("O3_日最大8h", lambda s: s.quantile(0.90))
    agg["O3_有效日"] = ("O3_日最大8h", "count")
    y = d.groupby(STA).agg(**agg).reset_index()
    for sp in m46.DIURNAL_SPECIES:
        wk = (d.groupby([STA, "是周末"])[sp].mean().unstack()
              .rename(columns={False: f"{sp}_工作日", True: f"{sp}_周末"}))
        y = y.merge(wk.reset_index(), on=STA, how="left")
        y[f"{sp}_周末效应"] = (y[f"{sp}_工作日"] - y[f"{sp}_周末"]) / y[f"{sp}_周末"]
        dz = d.groupby(STA)[[f"{sp}_高峰均值", f"{sp}_夜间均值"]].mean().reset_index()
        y = y.merge(dz, on=STA, how="left")
        y[f"{sp}_峰谷比"] = y[f"{sp}_高峰均值"] / y[f"{sp}_夜间均值"]
    # Script 50: annual mean of the daily O₃ means (days with ≥20 records), plus molar mixing ratios and Ox
    og = df[df["指标"] == "O3"].groupby([STA, "日期"])["值"].agg(["mean", "count"])
    og = og[og["count"] >= m46.MIN_HOURS_DAY].reset_index()
    o3y = og.groupby(STA)["mean"].agg(O3_年均="mean", O3_年均_有效日="count").reset_index()
    y = y.merge(o3y, on=STA, how="left")
    y["O3_年均_ppb"] = y["O3_年均"] * MOLAR_VOL / MW_O3
    y["NO2_年均_ppb"] = y["NO2_年均"] * MOLAR_VOL / MW_NO2
    y["Ox_年均_ppb"] = y["O3_年均_ppb"] + y["NO2_年均_ppb"]
    return y.assign(**{TIME: year})


def regressions(p: pd.DataFrame) -> pd.DataFrame:
    """Re-estimate each row, written the same way as prep() and the main and mechanism tables of script 48
    (analytic SEs only; no bootstrap)."""
    p = p.copy()
    p["城市年"] = p[CITY].astype(str) + "_" + p[TIME].astype(str)
    outs = {"NO2": "NO2_年均", "PM2.5": "PM2.5_年均", "PM10": "PM10_年均", "SO2": "SO2_年均",
            "O3_8h": "O3_8h_90分位", "O3年均": "O3_年均", "Ox": "Ox_年均_ppb"}
    for tag, col in outs.items():
        v = p[col].where(p[col] > 0)
        if col in ("O3_年均", "Ox_年均_ppb"):
            v = v.where(p["O3_年均_有效日"] >= MIN_DAYS)
        if col == "Ox_年均_ppb":
            v = v.where(p["NO2_有效日"] >= MIN_DAYS)
        dcol = m48.DAYCOL.get(col)
        if dcol is not None:
            v = v.where(p[dcol] >= MIN_DAYS)
        p[f"y_{tag}"] = np.log(v)
    for sp in m48.MECH_SPECIES:
        ok = p[f"{sp}_有效日"] >= MIN_DAYS
        v = p[f"{sp}_峰谷比"]
        p[f"y_{sp}峰谷比"] = np.log(v.where((v > 0) & ok))
        p[f"y_{sp}周末效应"] = p[f"{sp}_周末效应"].where(ok)
    p["Expo"] = p["建成区占比_500m"] - p["建成区占比_500m"].mean()
    p["lnNEV"] = np.log(p["NEV保有量_辆"].where(p["NEV保有量_辆"] > 0))
    p[MAIN] = p["lnNEV"] * p["Expo"]
    p["tx暴露"] = (p[TIME] - p[TIME].min()) * p["Expo"]
    valid = p.dropna(subset=["y_NO2"])
    p["城内站数"] = p["城市年"].map(valid.groupby("城市年")[STA].nunique())
    d0 = p[p["城内站数"] >= 2]

    def row(spec: str, y: str, xs: list[str], d: pd.DataFrame) -> dict[str, object]:
        res, _ = m48.fit(d, y, xs)
        out: dict[str, object] = {"规格": spec, "结果变量": y[2:], "N": int(res.nobs),
                                  "β_交互": round(float(res.params[MAIN]), 5),
                                  "SE": round(float(res.std_errors[MAIN]), 5), "p": round(float(res.pvalues[MAIN]), 5)}
        if "tx暴露" in xs:                                  # Table 2 columns (3) and (6) also report the trend term
            out["β_趋势"] = round(float(res.params["tx暴露"]), 5)
            out["p_趋势"] = round(float(res.pvalues["tx暴露"]), 5)
        return out

    rows = [row("主设定", y, [MAIN], d0) for y in [c for c in p.columns if c.startswith("y_")]]
    # Columns (3) and (6) of Table 2: equation (4), which adds an exposure-specific linear trend
    rows += [row("式(3)", y, [MAIN, "tx暴露"], d0) for y in ("y_NO2", "y_NO2峰谷比")]
    # Column (4) of Table 2 and the peak-to-night-ratio variant in the main text: year indicators × exposure
    # (first year as the base), written the same way as script 55
    base = d0.dropna(subset=["lnNEV"])
    for y in ("y_NO2", "y_NO2峰谷比"):
        e = base.dropna(subset=[y]).copy()
        xs = [MAIN]
        for yr in sorted(e[TIME].unique())[1:]:
            e[f"y{yr}x暴露"] = (e[TIME] == yr).astype(float) * e["Expo"]
            xs.append(f"y{yr}x暴露")
        rows.append(row("年份哑变量", y, xs, e))
    return pd.DataFrame(rows)


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("58_dedup_rebuild_check")
    det = pd.read_excel(export_utils.OUTPUT_DIR / "57_小时数据重复记录核验.xlsx", sheet_name="02_02_受影响站日明细")
    det[STA] = det[STA].astype(str)
    affected = det.groupby([TIME, STA]).agg(受影响站日指标=("指标", "size"), 跨过门槛=("跨过门槛", "sum")).reset_index()
    logger.info("受影响站点年 %s 个：%s", len(affected), affected.to_dict("records"))

    panel = pd.read_csv(export_utils.DATA_DIR / "站点年面板-2015_2024.csv")
    panel.columns = [c.lstrip("﻿") for c in panel.columns]
    panel[STA] = panel[STA].astype(str)
    # Station-years not in the panel (stations outside the station register) enter no result; they are only logged
    keys = pd.MultiIndex.from_frame(affected[[STA, TIME]])
    affected["在面板中"] = keys.isin(panel.set_index([STA, TIME]).index)
    logger.info("其中在面板中的站点年 %s 个；不在面板中的 %s 个：%s", int(affected["在面板中"].sum()),
                int((~affected["在面板中"]).sum()), affected.loc[~affected["在面板中"], [TIME, STA]].to_dict("records"))

    raw_parts, dedup_parts = [], []
    for yr, grp in affected[affected["在面板中"]].groupby(TIME):
        df = hourly_for(int(yr), set(grp[STA]))
        raw_parts.append(rebuild(df, int(yr)))
        dedup_parts.append(rebuild(df.drop_duplicates([STA, "日期", "小时", "指标"]), int(yr)))
    raw, dedup = pd.concat(raw_parts, ignore_index=True), pd.concat(dedup_parts, ignore_index=True)
    cols = [c for c in raw.columns if c not in (STA, TIME)]

    # Gate 1: the rebuild without deduplication must reproduce the current panel values
    cur = panel.set_index([STA, TIME]).loc[raw.set_index([STA, TIME]).index, cols]
    rb = raw.set_index([STA, TIME])[cols]
    diff = (rb - cur).abs().max()
    nan_mismatch = int((rb.isna() != cur.isna()).sum().sum())
    check = pd.DataFrame({"列": cols, "最大绝对差": [float(diff[c]) for c in cols]})
    worst = float(np.nanmax(check["最大绝对差"].to_numpy())) if len(check) else 0.0
    logger.info("闸 1：重建复现面板现值，最大绝对差 %.3g，缺失位置不一致 %s 处", worst, nan_mismatch)
    if nan_mismatch or not worst < 1e-8:
        raise SystemExit(f"闸 1 不过：最大绝对差 {worst}，缺失位置不一致 {nan_mismatch} 处。不输出任何数。")

    # Gate 2: re-estimates on the original panel must agree with the workbooks of scripts 48 and 55
    base = regressions(panel)
    ref_main = pd.read_excel(export_utils.OUTPUT_DIR / "48_站点面板回归.xlsx", sheet_name="01_主表")
    ref_mech = pd.read_excel(export_utils.OUTPUT_DIR / "48_站点面板回归.xlsx", sheet_name="02_D2机制_交通时段")
    ref = pd.concat([ref_main[ref_main["规格"] == "M9c 主表"],
                     ref_mech[ref_mech["规格"].isin(["M9 机制", "M9 安慰剂"])]], ignore_index=True)
    ref = ref.rename(columns={"β_交互_SE": "SE", "β_交互_p": "p"})[["结果变量", "N", "β_交互", "SE", "p"]]
    ref["规格"] = "主设定"
    rob = pd.read_excel(export_utils.OUTPUT_DIR / "48_站点面板回归.xlsx", sheet_name="03_稳健性")
    s16 = pd.concat([rob[(rob["规格"].str.startswith("S16")) & (rob["结果变量"] == "NO2")],
                     ref_mech[(ref_mech["规格"].str.startswith("S16")) & (ref_mech["结果变量"] == "NO2峰谷比")]])
    s16 = s16.rename(columns={"β_交互_SE": "SE", "β_交互_p": "p"})[["结果变量", "N", "β_交互", "SE", "p"]]
    s16["规格"] = "式(3)"
    yd = pd.read_excel(export_utils.OUTPUT_DIR / "55_趋势共线性与交叉实现.xlsx", sheet_name="04_04_年份哑变量规格")
    yd = yd.rename(columns={"β_NEV×暴露": "β_交互"}).assign(
        结果变量=yd["结果变量"].map({"ln_NO2": "NO2", "ln_NO2峰谷比": "NO2峰谷比"}), 规格="年份哑变量")
    ref = pd.concat([ref, s16, yd[["规格", "结果变量", "N", "β_交互", "SE", "p"]]], ignore_index=True)
    m = base.merge(ref, on=["规格", "结果变量"], suffixes=("", "_48"))
    # This script and script 48 round β, SE and p to 5 decimals; script 55 rounds β to 6, SE to 5 and p to 4. Each
    # tolerance is half a unit in the coarsest decimal kept (the 5th for β and SE, the 4th for p) plus a small margin
    tol = {"β_交互": 5.1e-6, "SE": 5.1e-6, "p": 5.1e-5}
    off = np.zeros(len(m), dtype=bool)
    for c, t in tol.items():
        off |= (m[c] - m[f"{c}_48"]).abs().to_numpy() > t
    bad = m[off | (m["N"] != m["N_48"]).to_numpy()]
    logger.info("闸 2：与 48／55 号工作簿对上 %s 行，不一致 %s 行", len(m), len(bad))
    if len(m) != len(ref) or len(bad):
        raise SystemExit(f"闸 2 不过：\n{bad.to_string()}\n不输出任何数。")

    # Copy of the panel after deduplication
    pd2 = panel.set_index([STA, TIME])
    pd2.loc[dedup.set_index([STA, TIME]).index, cols] = dedup.set_index([STA, TIME])[cols]
    new = regressions(pd2.reset_index())
    comp = base.merge(new, on=["规格", "结果变量"], suffixes=("_原", "_去重"))
    comp["Δβ"] = (comp["β_交互_去重"] - comp["β_交互_原"]).round(5)
    comp["Δβ/SE"] = (comp["Δβ"] / comp["SE_原"]).round(3)
    comp["Δβ_趋势"] = (comp["β_趋势_去重"] - comp["β_趋势_原"]).round(5)
    for r in comp.to_dict("records"):
        logger.info("%s", r)
    logger.info("最大 |Δβ/SE| = %.3f", float(comp["Δβ/SE"].abs().max()))
    out = export_utils.write_excel_workbook("58_去重重建核验",
                                            [("01_受影响站点年", affected), ("02_重建复现核对", check),
                                             ("03_系数对照", comp)])
    logger.info("已写出 %s；日志 %s", out, log_path)


if __name__ == "__main__":
    main()
