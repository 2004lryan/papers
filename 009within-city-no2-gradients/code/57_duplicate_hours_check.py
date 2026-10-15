"""
57_duplicate_hours_check.py: check of how duplicate records in the hourly data affect daily values and valid days

Why this is needed:
    While script 56 was arranging hourly O₃ values by clock hour, it found that the raw hourly data for 2019–2022
    contain cases where "the same station, the same day, the same clock hour and the same pollutant" have two records.
    Scripts 46 and 50 count by record: the daily mean is the mean of all records of the day, and the number of valid
    hours is the number of records.
    Duplicate records can therefore (a) change the daily mean (when only some of the day's clock hours are
    duplicated), (b) turn a day that actually has fewer than 20 valid hours into a valid day.
    This script finds all duplicate keys year by year, classifies the type of duplication by
    "station × day × pollutant", counts how many cases there are of each of the two situations above, and also checks
    whether the two duplicate records have the same value. It only reads the raw hourly data and writes nothing back
    to data/.

Criterion:
    If every duplication is "all clock hours of the same station-day and pollutant repeated once as a whole, with
    identical values in the two records", the daily means do not change; if, in addition, no day crosses the 20-hour
    threshold because of duplication, the valid days do not change either, so the annual means and valid-day counts
    built from daily means by scripts 46/50, and all regressions built on them, are identical digit for digit to those
    after deduplication. This does not cover the O₃ 8-hour measure of script 46 (value and valid days): its window
    rolls over records, so a repeated day changes it (see script 58).

Usage:
    python code/57_duplicate_hours_check.py

Output files:
    outputs/57_小时数据重复记录核验.xlsx (duplicate-record check of the hourly data)
        — sheets "01_01_逐年汇总" (yearly summary) / "02_02_受影响站日明细" (details of affected station-days)
    logs/57_duplicate_hours_check.log
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
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

YEARS = range(2015, 2025)
MIN_HOURS_DAY = 20                      # daily valid-hour threshold of scripts 46/50
COLS = ["日期", "小时", "站点编号", "指标", "值"]
I64 = npt.NDArray[np.int64]


class Encoder:
    """Map the station and pollutant categories of each row group to integer codes that are consistent across the
    whole year."""

    def __init__(self) -> None:
        self.sta: dict[str, int] = {}
        self.pol: dict[str, int] = {}

    def keys(self, g: pd.DataFrame) -> tuple[I64, I64]:
        s = np.array([self.sta.setdefault(str(c), len(self.sta)) for c in g["站点编号"].cat.categories],
                     dtype=np.int64)[g["站点编号"].cat.codes.to_numpy()]
        p = np.array([self.pol.setdefault(str(c), len(self.pol)) for c in g["指标"].cat.categories],
                     dtype=np.int64)[g["指标"].cat.codes.to_numpy()]
        d = g["日期"].astype(np.int64).to_numpy()
        day = ((s * 400 + (d // 100 % 100) * 31 + d % 100) * 16 + p)       # station × day × pollutant
        return day * 24 + g["小时"].to_numpy().astype(np.int64), day


def row_groups(year: int) -> tuple[int, pq.ParquetFile]:
    # pyarrow's ParquetFile has no type annotations; the call is allowed as an unannotated third-party call
    pf = pq.ParquetFile(export_utils.DATA_DIR / f"站点小时空气质量-{year}.parquet")  # type: ignore[no-untyped-call]
    return int(pf.metadata.num_row_groups), pf


def check_year(year: int) -> tuple[dict[str, object], pd.DataFrame]:
    n_groups, pf = row_groups(year)
    enc = Encoder()
    parts = []
    for i in range(n_groups):
        g = pf.read_row_group(i, columns=COLS).to_pandas()  # type: ignore[no-untyped-call]
        parts.append(enc.keys(g)[0])
    k = np.concatenate(parts)
    n_rec = len(k)
    k.sort()
    dup_keys = np.unique(k[1:][k[1:] == k[:-1]])
    del k, parts
    summary: dict[str, object] = {"年份": year, "记录数": n_rec, "重复键": len(dup_keys)}
    if not len(dup_keys):
        return summary | {"受影响站日×指标": 0, "整日整体重复": 0, "部分钟点重复": 0, "两条取值不同的键": 0,
                          "日均值改变的站日×指标": 0, "因重复跨过20小时门槛": 0}, pd.DataFrame()
    dup_days = np.unique(dup_keys // 24)
    # Second pass: read all records of the affected "station × day × pollutant" cells only
    rows = []
    for i in range(n_groups):
        g = pf.read_row_group(i, columns=COLS).to_pandas()  # type: ignore[no-untyped-call]
        hk, day = enc.keys(g)
        m = np.isin(day, dup_days)
        if m.any():
            rows.append(pd.DataFrame({"day": day[m], "hk": hk[m], "值": g["值"].to_numpy(float)[m]}))
    r = pd.concat(rows, ignore_index=True)
    per_key = r.groupby("hk")["值"].agg(["size", "nunique"])
    r["小时"] = r["hk"] % 24
    daily = r.groupby("day").agg(记录数=("值", "size"), 钟点数=("小时", "nunique"), 均值_按记录=("值", "mean"))
    uniq = r.drop_duplicates("hk")
    daily["均值_去重"] = uniq.groupby("day")["值"].mean()
    daily["整日整体重复"] = daily["记录数"] == 2 * daily["钟点数"]
    daily["日均值差"] = daily["均值_按记录"] - daily["均值_去重"]
    daily["跨过门槛"] = (daily["钟点数"] < MIN_HOURS_DAY) & (daily["记录数"] >= MIN_HOURS_DAY)
    pols = {v: k_ for k_, v in enc.pol.items()}
    stas = {v: k_ for k_, v in enc.sta.items()}
    idx = daily.index.to_numpy()
    daily.insert(0, "指标", [pols[int(x)] for x in idx % 16])
    md = [(int(x) // 16) % 400 for x in idx]                # month×31+day, converted back to yyyymmdd
    daily.insert(0, "日期", [year * 10000 + (c - 1) // 31 * 100 + c - (c - 1) // 31 * 31 for c in md])
    daily.insert(0, "站点编号", [stas[int(x)] for x in idx // 16 // 400])
    summary |= {"受影响站日×指标": len(daily), "整日整体重复": int(daily["整日整体重复"].sum()),
                "部分钟点重复": int((~daily["整日整体重复"]).sum()),
                "两条取值不同的键": int((per_key["nunique"] > 1).sum()),
                "日均值改变的站日×指标": int((daily["日均值差"].abs() > 1e-9).sum()),
                "因重复跨过20小时门槛": int(daily["跨过门槛"].sum())}
    return summary, daily.reset_index(drop=True).assign(年份=year)


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("57_duplicate_hours_check")
    rows, details = [], []
    for yr in YEARS:
        s, d = check_year(yr)
        rows.append(s)
        if len(d):
            details.append(d)
        logger.info("%s", s)
    summ = pd.DataFrame(rows)
    det = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    clean = int(summ["部分钟点重复"].sum()) == 0 and int(summ["两条取值不同的键"].sum()) == 0 \
        and int(summ["因重复跨过20小时门槛"].sum()) == 0
    logger.info("判据：全部重复为整日整体重复且取值相同、无一天因重复跨过 20 小时门槛 —— %s",
                "成立，日均值与有效日与去重后逐位相同" if clean else "不成立，见明细")
    out = export_utils.write_excel_workbook("57_小时数据重复记录核验",
                                            [("01_逐年汇总", summ), ("02_受影响站日明细", det)])
    logger.info("已写出 %s；日志 %s", out, log_path)


if __name__ == "__main__":
    main()
