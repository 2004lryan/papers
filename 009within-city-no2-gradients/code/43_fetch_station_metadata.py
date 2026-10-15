"""
43_fetch_station_metadata.py: station register of the national monitoring stations (code / name / city / coordinates)

Source:
    China National Environmental Monitoring Centre, "National Urban Air Quality Real-time Publishing Platform"
    POST https://air.cnemc.cn:18007/HourChangesPublish/GetAllAQIPublishLive
    Returns the real-time observations of all national monitoring stations currently publishing; among the returned
    fields, the seven fields StationCode / PositionName / Area / CityCode / ProvinceId / Latitude / Longitude make
    up the station register.
    The endpoint has no public documentation, and the field names are taken from the response body itself.

Why it is needed:
    The station-level hourly data fetched by script 42 identify each station only by its code (e.g. 1001A) and carry
    no location. The station-level identification design uses "station traffic exposure" as its dimension of
    heterogeneity and therefore needs coordinates first; the city each station belongs to is also used to align
    the station panel with the city-level NEV vehicle stock.

Limitation (stated in docs/station-register-datasheet.md):
    The endpoint gives the stations publishing **at present** (about 1,600). Stations closed or renamed during
    2015–2024 are not among them, and their historical observations cannot enter the station panel for lack of
    coordinates. This script counts them for each year from 2017 to 2023 (sheet 02_与数据匹配率), and script 46
    drops them when it builds the panel.

Usage:
    python code/43_fetch_station_metadata.py

Output files:
    data/国控站点元数据.csv (national monitoring station register)
        station code / name / city / province ID / city code / latitude and longitude
    outputs/43_站点元数据核验.xlsx (station register checks)
        full register, match rate against the data of script 42, station counts by city, coordinate anomalies
    logs/43_fetch_station_metadata.log
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

URL = "https://air.cnemc.cn:18007/HourChangesPublish/GetAllAQIPublishLive"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
# latitude and longitude range of mainland China, for the coordinate plausibility check
LAT_RANGE, LON_RANGE = (3.0, 54.0), (73.0, 136.0)


def fetch(tries: int = 4) -> list[dict[str, Any]]:
    for _attempt in range(1, tries + 1):
        try:
            out = subprocess.run(
                ["curl", "-sk", "-X", "POST", "--max-time", "90", URL,
                 "-H", f"User-Agent: {UA}", "-H", "X-Requested-With: XMLHttpRequest",
                 "-H", "Origin: https://air.cnemc.cn:18007",
                 "-H", "Referer: https://air.cnemc.cn:18007/",
                 "-H", "Content-Length: 0", "--data", ""],
                capture_output=True, text=True, timeout=120).stdout
            data = json.loads(out)
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    raise SystemExit("✗ 站点元数据接口取数失败")


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("43_fetch_station_metadata")
    raw = fetch()
    logger.info("接口返回 %s 条站点记录", len(raw))

    df = pd.DataFrame([{
        "站点编号": r.get("StationCode"),
        "站点名称": r.get("PositionName"),
        "所属城市": r.get("Area"),
        "城市代码": r.get("CityCode"),
        "省份ID": r.get("ProvinceId"),
        "纬度": pd.to_numeric(r.get("Latitude"), errors="coerce"),
        "经度": pd.to_numeric(r.get("Longitude"), errors="coerce"),
    } for r in raw]).drop_duplicates("站点编号").reset_index(drop=True)

    bad_coord = df[~(df["纬度"].between(*LAT_RANGE) & df["经度"].between(*LON_RANGE))]
    logger.info("站点 %s 个，覆盖城市 %s 个；坐标越界或缺失 %s 个",
                len(df), df["所属城市"].nunique(), len(bad_coord))

    # match rate against each year from 2017 to 2023 already written to disk by script 42
    match_rows = []
    for year in range(2017, 2024):
        p = export_utils.DATA_DIR / f"站点小时空气质量-{year}.parquet"
        if not p.exists():
            continue
        codes = set(pd.read_parquet(p, columns=["站点编号"])["站点编号"].astype(str).unique())
        hit = codes & set(df["站点编号"])
        match_rows.append({"年份": year, "数据中站点数": len(codes), "有元数据": len(hit),
                           "匹配率": round(len(hit) / len(codes), 4),
                           "无坐标站点数": len(codes - set(df["站点编号"]))})
        logger.info("  %s：数据站点 %s，有元数据 %s（%.1f%%）", year, len(codes), len(hit), 100 * len(hit) / len(codes))
    match = pd.DataFrame(match_rows)

    by_city = (df.groupby("所属城市").agg(站点数=("站点编号", "count")).reset_index()
               .sort_values("站点数", ascending=False))

    out_csv = export_utils.DATA_DIR / "国控站点元数据.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    export_utils.write_excel_workbook("43_站点元数据核验", [
        ("站点元数据", df), ("与数据匹配率", match), ("按城市站点数", by_city), ("坐标异常", bad_coord)])

    print(f"✓ 站点元数据 {len(df)} 个站，覆盖 {df['所属城市'].nunique()} 个城市 → {out_csv}")
    if len(match):
        print(match.to_string(index=False))
    print(f"\n每城站点数：中位 {by_city['站点数'].median():.0f}，"
          f"最多 {by_city.iloc[0]['所属城市']} {by_city.iloc[0]['站点数']} 个，"
          f"仅 1 个站的城市 {int((by_city['站点数'] == 1).sum())} 个")
    print(f"  日志: {log_path}")


if __name__ == "__main__":
    main()
