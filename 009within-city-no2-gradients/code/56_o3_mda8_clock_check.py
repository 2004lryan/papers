"""
56_o3_mda8_clock_check.py: O₃ daily maximum 8-hour running mean recomputed on clock-hour windows, to check the
O₃ (8 h) row of Table S5 of the Supplementary Material

Why it is needed:
    Clause 3.9 of GB 3095-2012 defines the 8-hour average as "the arithmetic mean of the average concentrations of
    8 consecutive hours", and its Table 4 requires at least 6 hourly values in every 8 hours. Script 46 implements
    it as a rolling mean over 8 adjacent hourly records within each station × day; hours missing from the hourly
    data have no record row, so on a day with missing hours a window spans more than 8 clock hours, and the "at
    least 6 hours per window" threshold counts records rather than clock hours and never screens out such a window.
    The difference affects only the 8-hour O₃ measure (annual 90th percentile). The annual mean O₃ and Ox used in
    the titration test come from the daily means of script 50 and are not affected.

    This script reads the raw hourly data, the station annual panel and the workbook of script 48 (nothing is
    written back to data/), in two steps:
        1. With the current O₃ 8h column of script 46 in the panel, reproduce the O₃_8h row of the main table of
           script 48; continue only if coefficient, standard error, p and N each agree with the workbook of script
           48, otherwise exit without writing the workbook;
        2. switch to the O₃ 8h column on the clock-hour definition and re-estimate with the same sample rules and
           the same implementation (fit and wild_boot_p of script 48).

Clock-hour definition:
    Each station-day is laid out as 24 cells for hours 0–23 (empty when missing; several records for the same
    clock hour are averaged, and the number of such keys is recorded for each year);
    the 17 8-hour windows lying entirely within the day (ending at hours 7–23) are used,
    and a window is valid only with ≥ 6 valid hours in it; a day is a valid day only with ≥ 14 valid windows
    (the same threshold as MIN_8H_DAY in script 46),
    the daily value is the maximum over the valid windows; the annual value is the 90th percentile of the daily
    values of the valid days (pandas quantile with linear interpolation, as in script 46).
    The threshold on the annual number of valid days is the same as in the main definition of script 48 (274 days).

Usage:
    python code/56_o3_mda8_clock_check.py

Output files:
    outputs/56_臭氧8小时钟点窗口核验.xlsx (ozone 8-hour clock-window check)
        year-by-year comparison of the two definitions / regression comparison
    logs/56_o3_mda8_clock_check.log
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
# regression and wild bootstrap call the functions of script 48 directly, which guarantees the same implementation
# as for the other rows of the table
m48 = _load("m48", "48_run_station_panel.py")

STA, CITY, TIME = "站点编号", "所属城市", "年份"
MIN_DAYS = 274
YEARS = range(2017, 2024)            # regression window for which vehicle stock is available
ENDS = np.arange(7, 24)              # end hours of the 8-hour windows lying entirely within the calendar day
MIN_IN_WIN, MIN_WIN_DAY = 6, 14
MAIN = "NEVxExpo"


def o3_hourly(year: int) -> tuple[pd.DataFrame, list[str]]:
    """Read one year of hourly data row group by row group, keeping only the O₃ rows; station codes are replaced by
    integer codes consistent over the whole year, and dates by integers.

    Reading the whole year at once and then filtering O₃ would first load the five columns of all pollutants into
    memory (about 3 GB), which pages repeatedly on a machine with 8 GB of memory; the data are therefore read and
    filtered by row group (about 1 million rows each), keeping only the four numeric columns for O₃.
    """
    # pyarrow's ParquetFile has no type annotations; these two calls are exempted as untyped third-party calls
    pf = pq.ParquetFile(export_utils.DATA_DIR / f"站点小时空气质量-{year}.parquet")  # type: ignore[no-untyped-call]
    sta_id: dict[str, int] = {}
    parts = []
    for i in range(pf.metadata.num_row_groups):
        g = pf.read_row_group(i, columns=["日期", "小时", STA, "指标", "值"]).to_pandas()  # type: ignore[no-untyped-call]
        g = g[g["指标"].astype(str) == "O3"]
        cats = [str(c) for c in g[STA].cat.categories]
        glob = np.array([sta_id.setdefault(c, len(sta_id)) for c in cats], dtype=np.int32)
        parts.append(pd.DataFrame({"sid": glob[g[STA].cat.codes.to_numpy()],
                                   "日期": g["日期"].astype(np.int32).to_numpy(),
                                   "小时": g["小时"].astype(np.int8).to_numpy(),
                                   "值": g["值"].to_numpy(float)}))
    names = list(sta_id)
    return pd.concat(parts, ignore_index=True), names


def clock_o3_8h(year: int) -> tuple[pd.DataFrame, int]:
    """Station-level annual 90th percentile of O₃ 8h and number of valid days for one year (clock-hour definition),
    and the number of keys with more than one record for the same clock hour."""
    t, names = o3_hourly(year)
    hk = ["sid", "日期", "小时"]
    dup = t.duplicated(hk, keep=False)
    n_dup = int(t[dup].groupby(hk).ngroups) if dup.any() else 0
    if n_dup:
        # several records for one station-day and clock hour: average them, so each clock hour fills one cell
        t = t.groupby(hk, as_index=False)["值"].mean()
    grp = t.groupby(["sid", "日期"], sort=True)
    key = grp.ngroup().to_numpy()
    keys = grp.size().reset_index()[["sid", "日期"]]
    keys[STA] = np.array(names, dtype=object)[keys["sid"].to_numpy()]
    arr = np.full((len(keys), 24), np.nan)
    arr[key, t["小时"].to_numpy().astype(int)] = t["值"].to_numpy(float)
    ok = ~np.isnan(arr)
    cs = np.concatenate([np.zeros((len(keys), 1)), np.where(ok, arr, 0.0).cumsum(1)], 1)
    cc = np.concatenate([np.zeros((len(keys), 1)), ok.cumsum(1)], 1)
    s, c = cs[:, ENDS + 1] - cs[:, ENDS - 7], cc[:, ENDS + 1] - cc[:, ENDS - 7]
    avg = np.where(c >= MIN_IN_WIN, s / np.maximum(c, 1), np.nan)
    n_valid = (~np.isnan(avg)).sum(1)
    daily_max = np.where(np.isnan(avg), -np.inf, avg).max(1)
    keys["日值"] = np.where(n_valid >= MIN_WIN_DAY, daily_max, np.nan)
    d = keys.dropna(subset=["日值"])
    y = d.groupby(STA)["日值"].agg(O3_8h_90_钟点=lambda v: v.quantile(0.90), O3_有效日_钟点="count")
    return y.reset_index().assign(**{TIME: year}), n_dup


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("56_o3_mda8_clock_check")
    p = pd.read_csv(export_utils.DATA_DIR / "站点年面板-2015_2024.csv")
    p.columns = [c.lstrip("﻿") for c in p.columns]
    p[STA] = p[STA].astype(str)

    per_year = {yr: clock_o3_8h(yr) for yr in YEARS}
    clock = pd.concat([v[0] for v in per_year.values()], ignore_index=True)
    p = p.merge(clock, on=[STA, TIME], how="left")

    # —— Year-by-year comparison of the two definitions (stations in the panel) ——
    comp = []
    for yr in YEARS:
        e = p[p[TIME] == yr]
        both = e.dropna(subset=["O3_8h_90分位", "O3_8h_90_钟点"])
        ratio = both["O3_8h_90_钟点"] / both["O3_8h_90分位"]
        rel = (ratio - 1).abs()
        comp.append({"年份": yr, "O3同钟点重复记录键数": per_year[yr][1], "面板站点": len(e), "两口径皆有值": len(both),
                     "相关系数": round(float(both["O3_8h_90分位"].corr(both["O3_8h_90_钟点"])), 5),
                     "相对差中位数": round(float(rel.median()), 5), "相对差P99": round(float(rel.quantile(0.99)), 4),
                     "对数差均值(钟点−现行)": round(float(np.log(ratio).mean()), 5),
                     "有效日中位_现行": float(e["O3_有效日"].median()),
                     "有效日中位_钟点": float(e["O3_有效日_钟点"].median()),
                     "达274天_现行": int((e["O3_有效日"] >= MIN_DAYS).sum()),
                     "达274天_钟点": int((e["O3_有效日_钟点"] >= MIN_DAYS).sum())})
        logger.info("%s", comp[-1])

    # —— Sample construction identical to script 48 ——
    p["城市年"] = p[CITY].astype(str) + "_" + p[TIME].astype(str)
    no2 = p["NO2_年均"].where((p["NO2_年均"] > 0) & (p["NO2_有效日"] >= MIN_DAYS))
    p["ln_NO2"] = np.log(no2)
    for out, col, dcol in [("ln_O3", "O3_8h_90分位", "O3_有效日"),
                           ("ln_O3_钟点", "O3_8h_90_钟点", "O3_有效日_钟点")]:
        p[out] = np.log(p[col].where((p[col] > 0) & (p[dcol] >= MIN_DAYS)))
    p["Expo"] = p["建成区占比_500m"] - p["建成区占比_500m"].mean()
    p["lnNEV"] = np.log(p["NEV保有量_辆"].where(p["NEV保有量_辆"] > 0))
    p[MAIN] = p["lnNEV"] * p["Expo"]
    valid = p.dropna(subset=["ln_NO2"])
    p["城内站数"] = p["城市年"].map(valid.groupby("城市年")[STA].nunique())
    d0 = p[p["城内站数"] >= 2].copy()

    ref = pd.read_excel(export_utils.OUTPUT_DIR / "48_站点面板回归.xlsx", sheet_name="01_主表")
    ref = ref[(ref["规格"] == "M9c 主表") & (ref["结果变量"] == "O3_8h")].iloc[0]

    rows = []
    for label, y in [("现行（46 号，相邻 8 条记录）", "ln_O3"), ("钟点窗口（GB 3095-2012 第 3.9 条）", "ln_O3_钟点")]:
        res, dd = m48.fit(d0, y, [MAIN])
        rows.append({"口径": label, "N": int(res.nobs), "站点数": dd[STA].nunique(), "城市数": dd[CITY].nunique(),
                     "β_交互": round(float(res.params[MAIN]), 5), "SE": round(float(res.std_errors[MAIN]), 5),
                     "p": round(float(res.pvalues[MAIN]), 5),
                     "城市wild_boot_p": round(float(m48.wild_boot_p(dd, y, [MAIN], MAIN)), 4)})
        logger.info("%s", rows[-1])
        if y == "ln_O3":
            got = (rows[0]["N"], rows[0]["β_交互"], rows[0]["SE"], rows[0]["p"])
            want = (int(ref["N"]), float(ref["β_交互"]), float(ref["β_交互_SE"]), float(ref["β_交互_p"]))
            if got != want:
                raise SystemExit(f"未能复现 48 号 O3_8h 一行：得到 {got}，48 号为 {want}。不输出任何数。")
            logger.info("已复现 48 号主表 O3_8h 一行：%s", got)

    out = export_utils.write_excel_workbook("56_臭氧8小时钟点窗口核验",
                                            [("01_口径对照", pd.DataFrame(comp)), ("02_回归对照", pd.DataFrame(rows))])
    logger.info("已写出 %s；日志 %s", out, log_path)


if __name__ == "__main__":
    main()
