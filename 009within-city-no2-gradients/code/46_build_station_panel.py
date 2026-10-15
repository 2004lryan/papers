"""
46_build_station_panel.py: station annual panel (2015–2024 by default) — hourly observations → station × year,
joined with exposure and city NEV (2017–2023 only)

Pipeline:
    hourly station data from script 42 (7.82 × 10⁸ records in 2015–2024) → station × day → station × year
    + station register from script 43 (coordinates, city of the station)
    + built-up exposure from script 45 (main measure) / road-network exposure from script 44 (optional cross-check;
      only a 12-station pilot exists, see data/DATA.md)
    + city NEV vehicle stock and 省级行政区 (province) from the city panel of script 37 (2017–2023 rows only)

Validity thresholds:
    Daily: a day is a valid day only if it has ≥ 20 valid hours (O₃ uses the daily maximum 8-hour moving average,
    with ≥ 14 8-h values).
    Annual: **this script does not filter**; it only outputs the number of valid days of each station-year in the
    `*_有效日` (valid days) columns, and the threshold is applied by the analysis scripts.
    Reason: the 324-day annual threshold of GB 3095-2012 Table 4 falls in 2018 exactly at the median of the
    valid-day distribution (323), so it would cut half of that year's stations at one stroke and leave the panel
    severely unbalanced; keeping the threshold for the analysis stage allows switching between the main rule of 75%
    (274 days; the US EPA requires 75% of hours for the NO₂ annual mean) and the robustness rule of 90% (324 days,
    GB 3095-2012).
    No imputation in any case.

Core output variables:
    NO2_年均, PM2.5_年均, PM10_年均, SO2_年均, CO_年均 (annual means), O3_8h_90分位 (90th percentile of the daily
    maximum 8-h O₃)
    and the **time-of-day structure measures**: weekday−weekend relative difference, commuting-peak / overnight-trough
    ratio.
    For NO₂ this is the traffic signal (mechanism evidence; Section 2.7 of the main text); **the same is computed for
    SO₂ as a placebo**: industrial coal burning and heating have no diurnal rhythm that follows commuting, so if the
    SO₂ peak-to-trough ratio also changed with electrification, what is being captured is not the traffic component.
    PM2.5 and CO serve as references.

Usage:
    python code/46_build_station_panel.py --start-year 2015 --end-year 2024

Output files:
    data/站点年面板-<start year>_<end year>.csv (station annual panel)
    outputs/46_站点面板构建-<start year>_<end year>.xlsx (station panel construction)
        — panel / "按年覆盖" (coverage by year) / "有效性损失" (validity losses)
    logs/46_build_station_panel.log
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import logging

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

# Default 2015–2024: 2015–2016 predate any meaningful electrification (NEVs below 1% of the fleet) and are used to
# judge whether the narrowing of the within-city gradient **had already begun before electrification**; this judgement
# does not depend on any regression specification.
# City NEV data cover only 2017–2023; NEV is left empty in the rows for the early years and 2024, which enter only the
# descriptive gradient path.
YEAR_START, YEAR_END = 2015, 2024
MIN_HOURS_DAY = 20      # threshold on the number of valid hours in a day
MIN_8H_DAY = 14         # number of 8-h values required for the O₃ daily maximum 8-h value
# The annual validity threshold is **not applied in this script**: it only records the number of valid days of each
# station-year, and scripts 47/48 filter as needed.
# Reason: 324 days (GB 3095-2012 Table 4, about 90%) sits in 2018 exactly at the median of the valid-day distribution
# (323), cutting half of the stations at one stroke and leaving the panel severely unbalanced. Deferring the threshold
# to the analysis stage allows switching between
#   the main rule of 75% (the US EPA completeness level for the NO₂ annual mean, counted in hours) and the robustness
#   rule of 90% (GB 3095-2012)
# without rerunning this script (about 35 minutes per run).
MIN_DAYS_MAIN = 274     # main rule: about 75%
MIN_DAYS_STRICT = 324   # robustness: GB 3095-2012 Table 4 annual threshold (monthly requirement not imposed)
PEAK_HOURS = {7, 8, 9, 17, 18, 19, 20}
NIGHT_HOURS = {1, 2, 3, 4}
GASES = ["NO2", "PM2.5", "PM10", "SO2", "CO"]
# The time-of-day structure is computed for all of these: NO₂ is the main signal and SO₂ is the **placebo** —
# industrial coal burning and heating have no morning and evening peaks that follow commuting, so if the SO₂
# peak-to-trough ratio also changed with electrification, what is captured is not traffic. PM2.5 / CO serve as
# references.
DIURNAL_SPECIES = ["NO2", "SO2", "PM2.5", "CO"]


def daily_from_hourly(year: int, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (station × day table of daily means and the O₃ daily maximum 8-h value, station × day table of peak and
    overnight means of NO2, SO2, PM2.5 and CO)."""
    src = export_utils.DATA_DIR / f"站点小时空气质量-{year}.parquet"
    df = pd.read_parquet(src)
    df["站点编号"] = df["站点编号"].astype(str)
    df["指标"] = df["指标"].astype(str)

    # —— Daily means (the five pollutants other than O₃) ——
    sub = df[df["指标"].isin(GASES)]
    g = sub.groupby(["站点编号", "日期", "指标"])["值"]
    daily = g.agg(["mean", "count"]).reset_index()
    daily = daily[daily["count"] >= MIN_HOURS_DAY]
    daily = daily.pivot_table(index=["站点编号", "日期"], columns="指标", values="mean").reset_index()
    daily.columns.name = None

    # —— O₃: daily maximum 8-hour moving average ——
    o3 = df[df["指标"] == "O3"].sort_values(["站点编号", "日期", "小时"]).copy()
    if len(o3):
        # Moving average over 8 adjacent hourly records (at least 6) within the calendar day, then the maximum of the
        # day. Unlike the clock-hour windows of GB 3095-2012, a window spans more than 8 clock hours when hours are
        # missing; script 56 recomputes the measure on clock-hour windows.
        # groupby(...).rolling returns rows in the same order as the already sorted source table, so the result can be
        # written back by position.
        o3["滑动8h"] = (o3.groupby(["站点编号", "日期"], sort=False)["值"]
                        .rolling(8, min_periods=6).mean().to_numpy())
        o3d = o3.groupby(["站点编号", "日期"])["滑动8h"].agg(["max", "count"]).reset_index()
        o3d = o3d[o3d["count"] >= MIN_8H_DAY][["站点编号", "日期", "max"]].rename(columns={"max": "O3_日最大8h"})
        daily = daily.merge(o3d, on=["站点编号", "日期"], how="outer")

    # —— Time-of-day structure (traffic signal + placebo) ——
    parts = []
    for sp in DIURNAL_SPECIES:
        z = df[df["指标"] == sp]
        if not len(z):
            continue
        parts.append(z[z["小时"].isin(PEAK_HOURS)].groupby(["站点编号", "日期"])["值"].mean()
                     .rename(f"{sp}_高峰均值"))
        parts.append(z[z["小时"].isin(NIGHT_HOURS)].groupby(["站点编号", "日期"])["值"].mean()
                     .rename(f"{sp}_夜间均值"))
    diurnal = pd.concat(parts, axis=1).reset_index() if parts else pd.DataFrame(
        columns=["站点编号", "日期"])

    logger.info("  %s：小时记录 %s → 有效站-日 %s", year, len(df), len(daily))
    del df, sub, g
    return daily, diurnal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=YEAR_START)
    ap.add_argument("--end-year", type=int, default=YEAR_END)
    args = ap.parse_args()
    years = range(args.start_year, args.end_year + 1)

    logger, log_path = export_utils.configure_file_logger("46_build_station_panel")
    meta = pd.read_csv(export_utils.DATA_DIR / "国控站点元数据.csv")
    meta["站点编号"] = meta["站点编号"].astype(str)

    year_rows, loss_rows = [], []
    for year in years:
        src = export_utils.DATA_DIR / f"站点小时空气质量-{year}.parquet"
        if not src.exists():
            logger.warning("  %s 缺 parquet，跳过", year)
            continue
        daily, diurnal = daily_from_hourly(year, logger)
        daily["日期"] = daily["日期"].astype(str)
        diurnal["日期"] = diurnal["日期"].astype(str)
        d = daily.merge(diurnal, on=["站点编号", "日期"], how="left")
        d["星期"] = pd.to_datetime(d["日期"], format="%Y%m%d").dt.weekday
        d["是周末"] = d["星期"] >= 5

        agg: dict[str, tuple[str, Any]] = {}
        for c in GASES:
            if c in d.columns:
                agg[f"{c}_年均"] = (c, "mean")
                agg[f"{c}_有效日"] = (c, "count")
        if "O3_日最大8h" in d.columns:
            agg["O3_8h_90分位"] = ("O3_日最大8h", lambda s: s.quantile(0.90))
            agg["O3_有效日"] = ("O3_日最大8h", "count")
        y = d.groupby("站点编号").agg(**agg).reset_index()

        # Traffic signal (NO₂) and placebo (SO₂ / PM2.5 / CO): weekday − weekend, peak − overnight
        for sp in DIURNAL_SPECIES:
            if sp in d.columns:
                wk = (d.groupby(["站点编号", "是周末"])[sp].mean().unstack()
                      .rename(columns={False: f"{sp}_工作日", True: f"{sp}_周末"}))
                y = y.merge(wk.reset_index(), on="站点编号", how="left")
                if f"{sp}_工作日" in y.columns and f"{sp}_周末" in y.columns:
                    y[f"{sp}_周末效应"] = (y[f"{sp}_工作日"] - y[f"{sp}_周末"]) / y[f"{sp}_周末"]
            cols = [f"{sp}_高峰均值", f"{sp}_夜间均值"]
            if all(c in d.columns for c in cols):
                dz = d.groupby("站点编号")[cols].mean().reset_index()
                y = y.merge(dz, on="站点编号", how="left")
                y[f"{sp}_峰谷比"] = y[f"{sp}_高峰均值"] / y[f"{sp}_夜间均值"]

        y["年份"] = year
        y["当年数据天数"] = d["日期"].nunique()
        n_raw = len(y)
        # No values are blanked out; only compliance under the two thresholds is recorded, for the analysis stage to
        # choose from
        rec: dict[str, float] = {"年份": year, "站点数": n_raw, "当年数据天数": int(y["当年数据天数"].iloc[0])}
        for c in [*GASES, "O3_8h"]:
            col, dcol = (f"{c}_年均", f"{c}_有效日") if c != "O3_8h" else ("O3_8h_90分位", "O3_有效日")
            if col in y.columns and dcol in y.columns:
                rec[f"{c}_达标_75%"] = int((y[dcol] >= MIN_DAYS_MAIN).sum())
                rec[f"{c}_达标_国标"] = int((y[dcol] >= MIN_DAYS_STRICT).sum())
                rec[f"{c}_有效日中位"] = float(y[dcol].median())
        loss_rows.append(rec)
        year_rows.append(y)
        logger.info("  %s：站点 %s，NO2 达标 75%%门槛 %s / 国标门槛 %s（有效日中位 %.0f）", year, n_raw,
                    rec.get("NO2_达标_75%"), rec.get("NO2_达标_国标"), rec.get("NO2_有效日中位", 0))
        del daily, diurnal, d, y

    panel = pd.concat(year_rows, ignore_index=True)
    panel = panel.merge(meta[["站点编号", "站点名称", "所属城市", "城市代码", "纬度", "经度"]],
                        on="站点编号", how="inner")   # stations absent from the station register are excluded
    logger.info("并入元数据后 %s 行，站点 %s，城市 %s",
                len(panel), panel["站点编号"].nunique(), panel["所属城市"].nunique())

    # Exposure: built-up area (main measure) and road network (cross-validation)
    b = export_utils.DATA_DIR / "站点建成区暴露.csv"
    if b.exists():
        bd = pd.read_csv(b)
        cols = ["站点编号", "建成区占比_500m", "建成区占比_1000m", "建成区占比_城内百分位"]
        panel = panel.merge(bd[[c for c in cols if c in bd.columns]], on="站点编号", how="left")
        logger.info("已并入建成区暴露，非空 %s / %s", int(panel["建成区占比_500m"].notna().sum()), len(panel))
    o = export_utils.DATA_DIR / "站点交通暴露.csv"
    if o.exists():
        od = pd.read_csv(o)
        panel = panel.merge(od[["站点编号", "加权路网密度_500m"]], on="站点编号", how="left")

    # City NEV (city panel from script 37)
    c = export_utils.DATA_DIR / "城市面板-2017_2023.csv"
    if c.exists():
        cd = pd.read_csv(c)[["城市", "省级行政区", "年份", "NEV保有量_辆", "ln_NEV", "本省当年未分配占比"]]
        panel["城市名"] = panel["所属城市"].str.replace(
            r"(市|地区|自治州|盟|自治县|林区)$", "", regex=True)
        panel = panel.merge(cd.rename(columns={"城市": "城市名"}), on=["城市名", "年份"], how="left")
        hit = panel["NEV保有量_辆"].notna().mean()
        logger.info("已并入城市 NEV，匹配率 %.1f%%", 100 * hit)

    panel = panel.sort_values(["所属城市", "站点编号", "年份"]).reset_index(drop=True)
    out_csv = export_utils.DATA_DIR / f"站点年面板-{args.start_year}_{args.end_year}.csv"
    panel.to_csv(out_csv, index=False, encoding="utf-8-sig")

    ok = panel[panel["NO2_有效日"] >= MIN_DAYS_MAIN].dropna(subset=["NO2_年均"])
    n_per_city = ok.groupby(["所属城市", "年份"])["站点编号"].nunique()
    cover = (ok.groupby("年份").agg(站点数=("站点编号", "nunique"), 城市数=("所属城市", "nunique"),
                                    观测数=("站点编号", "size")).reset_index())
    cover["城市≥2站"] = [int((n_per_city.loc[:, y] >= 2).sum()) for y in cover["年份"]]
    cover["≥2站城市内的站点数"] = [int(n_per_city.loc[:, y][n_per_city.loc[:, y] >= 2].sum()) for y in cover["年份"]]

    export_utils.write_excel_workbook(f"46_站点面板构建-{args.start_year}_{args.end_year}", [
        ("站点年面板", panel), ("按年覆盖", cover), ("有效性损失", pd.DataFrame(loss_rows))])

    print(f"✓ 站点年面板 {panel.shape[0]} 行 × {panel.shape[1]} 列 → {out_csv}")
    print(cover.to_string(index=False))
    if "建成区占比_500m" in panel.columns:
        print(f"\n建成区暴露非空 {int(panel['建成区占比_500m'].notna().sum())} / {len(panel)}")
    if "NEV保有量_辆" in panel.columns:
        print(f"城市 NEV 匹配 {int(panel['NEV保有量_辆'].notna().sum())} / {len(panel)}")
    print(f"  日志: {log_path}")


if __name__ == "__main__":
    main()
