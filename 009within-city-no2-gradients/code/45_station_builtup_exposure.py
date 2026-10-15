"""
45_station_builtup_exposure.py: station built-up exposure measure (GHSL built-up surface raster, 100 m)

Why OSM (script 44) was replaced:
    Script 44 takes the road network from Overpass, and testing revealed two fatal problems:
    ① **Uneven coverage of China**: in a pilot run on 12 western stations, 7 had no mapped road of tertiary class
       or above within 500 m (8 had no motorway, trunk or primary road; the closing console summary of this script
       quotes that count). This is not because there are really no roads, but because OSM is almost blank for
       China's small and medium-sized cities and its western regions. This missingness correlates with how
       developed a city is and would mix "whether OSM has mapped the roads" into "whether traffic exposure is
       high", a systematic measurement error.
    ② **Speed**: a public Overpass instance takes about 50 seconds per station, so 1,601 stations need more than
       20 hours.
    Script 44 stopped after this twelve-station pilot and is not an exposure measure in the analysis.

This script uses the GHSL (Global Human Settlement Layer, JRC R2023A) built-up area raster instead:
    GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100 — a 100 m grid giving the built-up area of each cell (m², at most
    10000).
    It is a globally uniform remote-sensing retrieval, so **coverage and data quality do not vary with country or
    city tier**, which is exactly where OSM falls short.

Exposure measures (three; B1 is the main measure, fixed before any station-level regression was run, and B2 and B3
are the robustness checks S1 and S2 of Supplementary Table S2):
    B1 built-up fraction (built-up area within the 500 m buffer ÷ buffer area)
    B2 built-up fraction (1000 m)
    B3 rank of the station's built-up fraction within its own city (within-city percentile) — for comparisons
       within a city, removing differences in city size

Why built-up area can proxy traffic exposure: NO₂ has a lifetime of only a few hours and steep spatial gradients,
and the dominant NO₂ source in urban cores is road traffic. Built-up density is a direct measure of position in the
urban core, and it is independent of NO₂ concentrations themselves (remote-sensing retrieval, not environmental
monitoring).
Limitation: it also proxies population and commercial-activity density and is not pure traffic volume. The planned
cross-check against the road-network measure of script 44 did not go beyond its twelve-station pilot; when
data/站点交通暴露.csv is present, the script correlates the two on the pilot stations with roads within 500 m.

Usage:
    python code/45_station_builtup_exposure.py
    (GHSL tiles are downloaded to data/raw/ghsl on first use; they are not stored in the repository, see BASE for
    their address.)

Output files:
    data/站点建成区暴露.csv (station built-up exposure)             — stations × three measures + raw built-up area
    outputs/45_建成区暴露分布.xlsx (built-up exposure distribution) — sheets 01_站点建成区暴露 (station table),
        02_城市内离散度 (within-city dispersion) and, only when data/站点交通暴露.csv gives at least 5 stations
        with roads within 500 m, 03_与路网口径相关 (correlation with the road-network measure of script 44)
    logs/45_station_builtup_exposure.log
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

if TYPE_CHECKING:
    import logging

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

PRODUCT = "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100"
BASE = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/"
        f"{PRODUCT}/V1-0/tiles/{PRODUCT}_V1_0_R{{r}}_C{{c}}.zip")
CACHE = Path("data/raw/ghsl")   # GHSL tiles, downloaded on first use (relative to the working directory)
RADII = [500, 1000]
CELL = 100.0          # raster cell size (m)
CELL_AREA = CELL ** 2


def tile_of(x: float, y: float) -> tuple[int, int]:
    """Mollweide coordinates → GHSL R2023A tile row and column."""
    return int((9_000_000 - y) // 1_000_000) + 1, int((x + 18_041_000) // 1_000_000) + 1


def ensure_tile(r: int, c: int, logger: logging.Logger) -> Path | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tif = CACHE / f"{PRODUCT}_V1_0_R{r}_C{c}.tif"
    if tif.exists():
        return tif
    zp = CACHE / f"R{r}_C{c}.zip"
    if not zp.exists() or zp.stat().st_size < 100000:
        url = BASE.format(r=r, c=c)
        logger.info("  下载瓦片 R%s_C%s …", r, c)
        code = subprocess.run(["curl", "-s", "-o", str(zp), "-w", "%{http_code}", "--max-time", "900", url],
                              capture_output=True, text=True, timeout=1200).stdout.strip()
        if code != "200" or not zp.exists() or zp.stat().st_size < 100000:
            logger.warning("  瓦片 R%s_C%s 下载失败（HTTP %s）", r, c, code)
            if zp.exists():
                zp.unlink()
            return None
    try:
        with zipfile.ZipFile(zp) as z:
            name = next(n for n in z.namelist() if n.endswith(".tif"))
            with z.open(name) as src, tif.open("wb") as dst:
                dst.write(src.read())
    except Exception as exc:
        logger.warning("  瓦片 R%s_C%s 解压失败：%s", r, c, exc)
        return None
    zp.unlink(missing_ok=True)
    return tif


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("45_station_builtup_exposure")
    meta = pd.read_csv(export_utils.DATA_DIR / "国控站点元数据.csv")
    tr = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    xs, ys = tr.transform(meta["经度"].values, meta["纬度"].values)
    meta["x"], meta["y"] = xs, ys
    meta["瓦片"] = [tile_of(x, y) for x, y in zip(xs, ys, strict=True)]
    logger.info("站点 %s 个，需要瓦片 %s 个", len(meta), meta["瓦片"].nunique())

    recs: list[dict[str, Any]] = []
    for (r, c), grp in meta.groupby("瓦片"):
        tif = ensure_tile(r, c, logger)
        if tif is None:
            logger.warning("  瓦片 R%s_C%s 不可用，%s 个站跳过", r, c, len(grp))
            continue
        with rasterio.open(tif) as ds:
            for _, s in grp.iterrows():
                row, col = ds.index(s["x"], s["y"])
                rec = {"站点编号": s["站点编号"], "站点名称": s["站点名称"], "所属城市": s["所属城市"],
                       "纬度": s["纬度"], "经度": s["经度"], "瓦片": f"R{r}_C{c}"}
                for radius in RADII:
                    k = math.ceil(radius / CELL)
                    r0, c0 = row - k, col - k
                    win = Window(c0, r0, 2 * k + 1, 2 * k + 1)
                    try:
                        arr = ds.read(1, window=win, boundless=True, fill_value=0).astype("float64")
                    except Exception:
                        arr = np.zeros((2 * k + 1, 2 * k + 1))
                    # Circular mask: count only the cells that fall within the radius
                    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
                    mask = (yy ** 2 + xx ** 2) * CELL_AREA <= radius ** 2
                    built = float(np.nansum(np.where(mask, arr, 0.0)))
                    n_cell = int(mask.sum())
                    rec[f"建成区面积_m2_{radius}m"] = round(built, 1)
                    rec[f"建成区占比_{radius}m"] = round(built / (n_cell * CELL_AREA), 5) if n_cell else np.nan
                recs.append(rec)
        logger.info("  瓦片 R%s_C%s 完成，%s 个站", r, c, len(grp))

    df = pd.DataFrame(recs)
    if df.empty:
        raise SystemExit("✗ 无任何站点取到建成区值")
    # Within-city percentile (the measure for comparisons within a city)
    df["建成区占比_城内百分位"] = (df.groupby("所属城市")["建成区占比_500m"]
                                    .rank(pct=True, method="average").round(4))
    n_city = df.groupby("所属城市")["站点编号"].transform("size")
    df.loc[n_city < 2, "建成区占比_城内百分位"] = np.nan   # single-station cities allow no within-city comparison

    out_csv = export_utils.DATA_DIR / "站点建成区暴露.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    g = df.groupby("所属城市")["建成区占比_500m"]
    disp = (pd.DataFrame({"站点数": g.size(), "城内均值": g.mean().round(3),
                          "城内标准差": g.std().round(3), "城内极差": (g.max() - g.min()).round(3)})
            .query("站点数 >= 2").reset_index())
    sheets = [("站点建成区暴露", df), ("城市内离散度", disp)]

    # Cross-validation against the road-network measure of script 44 (if available)
    osm_path = export_utils.DATA_DIR / "站点交通暴露.csv"
    if osm_path.exists():
        osm = pd.read_csv(osm_path)
        j = df.merge(osm[["站点编号", "加权路网密度_500m", "总路网密度_500m"]], on="站点编号", how="inner")
        j = j[j["总路网密度_500m"] > 0]
        if len(j) >= 5:
            cc = j[["建成区占比_500m", "加权路网密度_500m", "总路网密度_500m"]].corr().round(3)
            sheets.append(("与路网口径相关", cc.reset_index()))
            logger.info("与 44 号路网口径交叉验证：n=%s，建成区占比 vs 加权路网密度 r=%.3f",
                        len(j), cc.loc["建成区占比_500m", "加权路网密度_500m"])

    export_utils.write_excel_workbook("45_建成区暴露分布", sheets)
    logger.info("完成：%s 个站，%s 个城市有 ≥2 站", len(df), len(disp))

    print(f"✓ 建成区暴露 {len(df)} 站 → {out_csv}")
    print(f"  建成区占比(500m)：中位 {df['建成区占比_500m'].median():.3f}，"
          f"p10–p90 {df['建成区占比_500m'].quantile(.1):.3f}–{df['建成区占比_500m'].quantile(.9):.3f}")
    print(f"  城市内离散：{len(disp)} 城有 ≥2 站，城内标准差中位 {disp['城内标准差'].median():.3f}，"
          f"城内极差中位 {disp['城内极差'].median():.3f}")
    print(f"  全为零的站：{int((df['建成区占比_500m'] == 0).sum())}（OSM 口径下曾有 8/12 为零）")
    print(f"  日志: {log_path}")


if __name__ == "__main__":
    main()
