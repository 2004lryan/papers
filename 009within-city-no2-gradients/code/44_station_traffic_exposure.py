"""
44_station_traffic_exposure.py: station traffic-exposure measures (OpenStreetMap road network)

Why it is needed:
    In the station-level identification design, the treatment-intensity dimension is "how close a station is to
    traffic sources". Once city-by-year fixed effects absorb everything at the city level, identification rests on
    "stations with high vs. low traffic exposure within the same city".
    This dimension must come from **external data** and cannot be constructed from NO₂ itself — defining treatment
    intensity with the outcome variable induces regression to the mean, and no negative coefficient could then be
    distinguished from "high-concentration stations would have fallen more anyway".

Measures:
    For each station, the Overpass API returns every road (way) that comes within a 500 m / 1000 m radius, with its
    full geometry; the full length of each returned way, not clipped to the buffer, is weighted by road class:
        motorway/trunk = 1.0, primary = 0.7, secondary = 0.4, tertiary = 0.2
    Weighted length ÷ buffer area = weighted road density (km/km²); because lengths are not clipped, this overstates
    the road length inside the buffer. The weights are set by the order of magnitude of traffic volumes; they are
    not estimated but chosen by the researchers, and sensitivity is checked with two alternative measures,
    "unweighted total length" and "major-road length only".

    Two quantities that do not depend on the weights are also given: the distance to the nearest vertex of a major
    road (motorway/trunk/primary), and a proxy for the number of road intersections (vertex positions that occur at
    least three times across the returned road geometries, which are not clipped to the buffer either).

Candidate measures (all three are computed). The study ran this script only as a 12-station pilot; open road data
proved too sparse, so the road-network check S3 was not run (Supplementary Section S2) and the analysis uses the
built-up exposure of script 45:
    E1 weighted road density (500 m)
    E2 weighted road density (1000 m)
    E3 distance to the nearest motorway/trunk/primary

Rate limiting and retries:
    Public Overpass instances are rate-limited; by default the script pauses 1.2 s after each successful query and
    makes at most three attempts per query, rotating across three mirrors and waiting 3 s, 6 s and 9 s after the
    first, second and third failed attempts.
    With two queries per station, the pauses alone come to about 64 minutes for the 1,601 stations, plus query time.
    Results are written to disk periodically and at the end, so a run can be resumed incrementally (stations already
    done are skipped).

Usage:
    python code/44_station_traffic_exposure.py

Output files:
    data/站点交通暴露.csv (station traffic exposure) — `站点编号`, `站点名称`, `所属城市`, `纬度`, `经度` and, per
        radius (suffixes _500m, _1000m): `加权路长_km`, `总路长_km`, `主干道长_km`, `加权路网密度`,
        `总路网密度`, `最近主干道距离_m`, `交叉口数`
    outputs/44_交通暴露分布.xlsx (written only when more than 20 stations have values)
        — sheets 01_站点暴露 (station values), 02_城市内离散度 (within-city dispersion), 03_口径间相关
        (correlations among the density, distance and intersection columns)
    logs/44_station_traffic_exposure.log
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

MIRRORS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.osm.ch/api/interpreter"]
WEIGHTS = {"motorway": 1.0, "motorway_link": 1.0, "trunk": 1.0, "trunk_link": 1.0,
           "primary": 0.7, "primary_link": 0.7, "secondary": 0.4, "secondary_link": 0.4,
           "tertiary": 0.2, "tertiary_link": 0.2}
MAJOR = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two points (metres)."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def query(lat: float, lon: float, radius: int, mirror_idx: int, tries: int = 3) -> dict[str, Any] | None:
    q = (f'[out:json][timeout:60];'
         f'way(around:{radius},{lat},{lon})[highway~"^(motorway|trunk|primary|secondary|tertiary)(_link)?$"];'
         f'out geom;')
    for t in range(tries):
        url = MIRRORS[(mirror_idx + t) % len(MIRRORS)]
        try:
            out = subprocess.run(["curl", "-s", "--max-time", "90", "-X", "POST", url, "--data-urlencode", f"data={q}"],
                                 capture_output=True, text=True, timeout=120).stdout
            if out.lstrip().startswith("{"):
                return cast("dict[str, Any]", json.loads(out))
        except Exception:
            pass
        time.sleep(3 * (t + 1))
    return None


def summarize(payload: dict[str, Any], lat: float, lon: float, radius: int) -> dict[str, Any]:
    """Reduce the road-segment geometry returned by Overpass to length and distance indicators."""
    w_len = raw_len = major_len = 0.0
    nearest_major = float("inf")
    nodes: dict[tuple[float, float], int] = {}
    for el in (payload.get("elements") or []):
        geom = el.get("geometry") or []
        hw = (el.get("tags") or {}).get("highway", "")
        w = WEIGHTS.get(hw, 0.0)
        seg = 0.0
        for a, b in itertools.pairwise(geom):
            seg += haversine(a["lat"], a["lon"], b["lat"], b["lon"])
        raw_len += seg
        w_len += seg * w
        if hw in MAJOR:
            major_len += seg
            for p in geom:
                nearest_major = min(nearest_major, haversine(lat, lon, p["lat"], p["lon"]))
        for p in geom:                       # count coinciding geometry points → proxy for intersections
            k = (round(p["lat"], 5), round(p["lon"], 5))
            nodes[k] = nodes.get(k, 0) + 1
    area = math.pi * (radius / 1000.0) ** 2  # km²
    return {"加权路长_km": w_len / 1000, "总路长_km": raw_len / 1000, "主干道长_km": major_len / 1000,
            "加权路网密度": (w_len / 1000) / area, "总路网密度": (raw_len / 1000) / area,
            "最近主干道距离_m": None if nearest_major == float("inf") else round(nearest_major, 1),
            "交叉口数": sum(1 for v in nodes.values() if v >= 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=1.2, help="每次查询后的间隔秒数")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个站（0=全部），用于试跑")
    args = ap.parse_args()

    logger, log_path = export_utils.configure_file_logger("44_station_traffic_exposure")
    meta = pd.read_csv(export_utils.DATA_DIR / "国控站点元数据.csv")
    out_path = export_utils.DATA_DIR / "站点交通暴露.csv"

    done: dict[str, dict[str, Any]] = {}
    if out_path.exists():                     # incremental resume
        prev = pd.read_csv(out_path)
        done = {r["站点编号"]: r.to_dict() for _, r in prev.iterrows()}
        logger.info("已有 %s 个站的暴露值，续跑剩余", len(done))

    todo = meta[~meta["站点编号"].isin(done)].reset_index(drop=True)
    if args.limit:
        todo = todo.head(args.limit)
    logger.info("待计算 %s 个站；Overpass 镜像 %s 个，间隔 %.1fs", len(todo), len(MIRRORS), args.sleep)

    rows = list(done.values())
    fails = 0
    for i, r in todo.iterrows():
        lat, lon = float(r["纬度"]), float(r["经度"])
        rec = {"站点编号": r["站点编号"], "站点名称": r["站点名称"], "所属城市": r["所属城市"],
               "纬度": lat, "经度": lon}
        ok = True
        for radius, tag in [(500, "500m"), (1000, "1000m")]:
            payload = query(lat, lon, radius, mirror_idx=i % len(MIRRORS))
            if payload is None:
                ok = False
                break
            for k, v in summarize(payload, lat, lon, radius).items():
                rec[f"{k}_{tag}"] = v
            time.sleep(args.sleep)
        if not ok:
            fails += 1
            logger.warning("  %s（%s）查询失败，跳过", r["站点编号"], r["站点名称"])
            continue
        rows.append(rec)
        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info("  进度 %s/%s，失败 %s，已落盘", i + 1, len(todo), fails)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("完成：%s 个站有暴露值，失败 %s", len(df), fails)

    if len(df) > 20:
        # Within-city dispersion — is there enough identifying variation?
        g = df.groupby("所属城市")["加权路网密度_500m"]
        disp = pd.DataFrame({"站点数": g.size(), "城内均值": g.mean(), "城内标准差": g.std(),
                             "城内极差": g.max() - g.min()}).query("站点数 >= 2")
        corr = df[[c for c in df.columns
                   if c.startswith(("加权路网密度", "总路网密度", "最近主干道距离", "交叉口数"))]].corr()
        export_utils.write_excel_workbook("44_交通暴露分布", [
            ("站点暴露", df), ("城市内离散度", disp.reset_index()), ("口径间相关", corr.reset_index())])
        print(f"✓ 交通暴露：{len(df)} 站")
        print(f"  加权路网密度(500m)：中位 {df['加权路网密度_500m'].median():.2f}，"
              f"p10–p90 {df['加权路网密度_500m'].quantile(.1):.2f}–{df['加权路网密度_500m'].quantile(.9):.2f} km/km²")
        print(f"  城市内离散：{len(disp)} 个城市有 ≥2 站，城内标准差中位 {disp['城内标准差'].median():.2f}，"
              f"城内极差中位 {disp['城内极差'].median():.2f}")
        print(f"  最近主干道距离：中位 {df['最近主干道距离_m_500m'].median():.0f} m")
    print(f"  日志: {log_path}")


if __name__ == "__main__":
    main()
