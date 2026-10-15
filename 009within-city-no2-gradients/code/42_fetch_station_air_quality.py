"""
42_fetch_station_air_quality.py: hourly air quality at the national monitoring stations (default 2017–2023; the
paper uses 2015–2024)

Why station level:
    In the provincial and city-level annual panel regressions of earlier project stages (not part of this release),
    the NEV signal is drowned out by city-level confounding: the provincial interaction was shown to be a disguised
    trend of the Blue Sky Protection Campaign priority regions, since it vanished once priority-region-by-year fixed
    effects were added, and the city panel flips sign because it lacks the total vehicle fleet as a denominator.
    At station level **city-by-year fixed effects** can be included, absorbing all city-level confounding as a whole
    (regional haze control, meteorology, economy, industry, the pandemic), so that identification uses only the
    differences "between stations with different traffic exposure within the same city in the same year".
    Empirical basis: on 2023-03-15 the daily mean NO₂ at Beijing's 12 national monitoring stations ranged from 2.2 to
    11.3 μg/m³, a between-station range of 146% of the mean, far larger than the year-to-year change between cities
    (about 10%). There is ample identifying variation.

Data source:
    https://quotsoft.net/air/data/china_sites_YYYYMMDD.csv
    columns = date, hour, type, <station codes…>; 2,026 national monitoring stations, 15 types, hourly.

Disk strategy:
    The raw daily files are about 2 MB each. After download **only the types needed for the analysis are kept**
    (NO2/CO/O3/PM2.5/PM10/SO2), converted to a long table and written to one parquet file per year (about 1.5 GB for
    2015–2024 together); each raw daily file is deleted once parsed unless --keep-raw is given.

Usage:
    python code/42_fetch_station_air_quality.py --start 20150101 --end 20241231
    (Without --start/--end only 2017–2023 is fetched, whereas the station panel of script 46 reads 2015–2024.
    Daily files are cached in data/raw/site_cache, or in the directory given by --cache-dir or the environment
    variable SITE_CACHE. --workers sets the number of parallel downloads (default 6); --keep-raw keeps the daily
    files after parsing; --only-missing fetches only the dates missing from an existing parquet file and merges
    them in, to recover days that failed under rate limiting.)

Output files:
    data/站点小时空气质量-YYYY.parquet (station hourly air quality)  — one per year: station × date × hour × pollutant
    outputs/42_站点数据覆盖.xlsx (station data coverage)             — sheet 01_按年覆盖: days, stations, records,
        failed and missing days for each year fetched in this run (each run overwrites the workbook)
    logs/42_fetch_station_air_quality.log
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

BASE_URL = "https://quotsoft.net/air/data/china_sites_{d}.csv"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
KEEP_TYPES = {"NO2", "CO", "O3", "PM2.5", "PM10", "SO2"}


def daterange(start: str, end: str) -> list[str]:
    d0 = dt.date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    d1 = dt.date(int(end[:4]), int(end[4:6]), int(end[6:8]))
    return [(d0 + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((d1 - d0).days + 1)]


def download(day: str, cache: Path) -> tuple[str, str]:
    path = cache / f"china_sites_{day}.csv"
    if path.exists() and path.stat().st_size > 50000:
        return day, "cached"
    try:
        code = subprocess.run(
            ["curl", "-s", "-o", str(path), "-w", "%{http_code}", "--max-time", "90",
             "-A", UA, BASE_URL.format(d=day)],
            capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception:
        return day, "failed"
    if code == "200" and path.exists() and path.stat().st_size > 50000:
        return day, "ok"
    if path.exists():
        path.unlink()
    return day, "missing" if code == "404" else "failed"


def parse_day(day: str, cache: Path) -> pd.DataFrame | None:
    """Reshape one day's wide table into a long table of station × hour × pollutant, keeping only KEEP_TYPES."""
    path = cache / f"china_sites_{day}.csv"
    if not path.exists():
        return None
    recs: list[tuple[str, int, str, str, float]] = []
    with path.open(encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return None
        if not header or header[0] != "date":
            return None
        sites = header[3:]
        for row in reader:
            if len(row) < 4 or row[2] not in KEEP_TYPES:
                continue
            hour, kind = row[1], row[2]
            for i, raw in enumerate(row[3:len(sites) + 3]):
                if raw:
                    try:
                        recs.append((day, int(hour), sites[i], kind, float(raw)))
                    except ValueError:
                        pass
    if not recs:
        return None
    return pd.DataFrame(recs, columns=["日期", "小时", "站点编号", "指标", "值"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20170101")
    ap.add_argument("--end", default="20231231")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache-dir", default=os.environ.get(
        "SITE_CACHE", "data/raw/site_cache"))
    ap.add_argument("--keep-raw", action="store_true", help="解析后保留原始日文件（默认删除以省磁盘）")
    ap.add_argument("--only-missing", action="store_true",
                    help="只补已落盘 parquet 里缺的日期，补完与原数据合并（源站限速导致的 failed 用它回收）")
    args = ap.parse_args()

    logger, log_path = export_utils.configure_file_logger("42_fetch_station_air_quality")
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    days = daterange(args.start, args.end)
    logger.info("站点级抓取 %s → %s，共 %s 天，并发 %s；缓存 %s", args.start, args.end, len(days), args.workers, cache)

    cover_rows: list[dict[str, Any]] = []
    for year in sorted({d[:4] for d in days}):
        ydays = [d for d in days if d.startswith(year)]
        out_path = export_utils.DATA_DIR / f"站点小时空气质量-{year}.parquet"
        prev = None
        if args.only_missing and out_path.exists():
            prev = pd.read_parquet(out_path)
            have = set(prev["日期"].astype(str).unique())
            ydays = [d for d in ydays if d not in have]
            logger.info("  %s 已有 %s 天，待补 %s 天", year, len(have), len(ydays))
            if not ydays:
                continue
        status: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (day, st) in enumerate(pool.map(lambda d: download(d, cache), ydays), 1):
                status[day] = st
                if i % 100 == 0:
                    logger.info("  %s 下载 %s/%s（ok=%s cached=%s missing=%s failed=%s）", year, i, len(ydays),
                                *[sum(v == k for v in status.values()) for k in ("ok", "cached", "missing", "failed")])

        frames = []
        for day in ydays:
            df = parse_day(day, cache)
            if df is not None:
                frames.append(df)
            if not args.keep_raw:
                p = cache / f"china_sites_{day}.csv"
                if p.exists():
                    p.unlink()
        if not frames:
            logger.warning("  %s 无数据", year)
            continue
        ydf = pd.concat(frames, ignore_index=True)
        if prev is not None:
            ydf = pd.concat([prev.astype({"站点编号": str, "指标": str}), ydf], ignore_index=True)
            logger.info("  %s 合并已有数据，总记录 %s", year, len(ydf))
        ydf["站点编号"] = ydf["站点编号"].astype("category")
        ydf["指标"] = ydf["指标"].astype("category")
        out = out_path
        ydf.to_parquet(out, index=False, compression="zstd")
        n_site = ydf["站点编号"].nunique()
        cover_rows.append({"年份": int(year), "天数": len(set(ydf["日期"])), "站点数": n_site,
                           "记录数": len(ydf), "NO2记录": int((ydf["指标"] == "NO2").sum()),
                           "文件MB": round(out.stat().st_size / 1e6, 1),
                           "下载失败天": sum(v == "failed" for v in status.values()),
                           "源站无此日": sum(v == "missing" for v in status.values())})
        logger.info("  %s 完成：%s 天 × %s 站，记录 %s，%.1f MB → %s",
                    year, cover_rows[-1]["天数"], n_site, len(ydf), cover_rows[-1]["文件MB"], out.name)
        del ydf, frames

    cover = pd.DataFrame(cover_rows)
    export_utils.write_excel_workbook("42_站点数据覆盖", [("按年覆盖", cover)])
    logger.info("全部完成\n%s", cover.to_string(index=False))
    print("✓ 站点级抓取完成")
    print(cover.to_string(index=False))
    print(f"  日志: {log_path}")


if __name__ == "__main__":
    main()
