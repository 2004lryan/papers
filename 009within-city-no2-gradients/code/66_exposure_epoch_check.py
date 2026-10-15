"""
66_exposure_epoch_check.py: does fixing the exposure at the GHSL 2020 epoch affect the narrowing path of the
within-city gradient?

Why it is needed:
    Exposure in the main text is the built-up fraction within a 500 m buffer around each station, taken from the 2020
    epoch of GHSL R2023A and held fixed for the whole decade (script 45).
    If built-up expansion in 2015–2020 was concentrated around stations that were originally little built up, fixing
    the 2020 epoch makes the exposure ranking of the early years depart from the actual situation of those years, and
    the gradient path picks up an epoch error. Re-estimating with the 2015 epoch tests this directly.

    This script takes the 2015 epoch of the same product (GHS_BUILT_S_E2015_GLOBE_R2023A_54009_100, on the same tiles
    as the 2020 epoch), computes the 500 m built-up fraction with the same station projection, tile lookup, window and
    circular mask as script 45, and then reports:
      ① the station-level change in the built-up fraction from 2015 to 2020, and how the change relates to the 2020
         level;
      ② the year-by-year within-city gradients, the 2015→2024 narrowing and the segment rates under three exposure
         definitions: the 2020 epoch (main-text definition), the 2015 epoch, and time-varying (2015 epoch in 2015,
         linear interpolation between the two epochs in 2016–2019, 2020 epoch from 2020 on);
      ③ paired intervals for the difference from the main-text definition: city cluster bootstrap with 999
         replications, the three definitions sharing the same set of city resampling weights.
    The estimator is the same as M0 of script 62 and as script 63 (city-by-year fixed effects + exposure × year, SEs
    clustered by city); the functions of script 62 are called directly.
    Epochs from 2025 on are GHSL extrapolations and are not used.

Decision rule (written with the script; the repository holds no dated record that it preceded the run):
    If, under both the 2015-epoch and the time-varying definitions, the 2015→2024 narrowing of the NO₂ gradient and
    its 2017–2020 decline both keep their direction, and the paired intervals for the difference from the main-text
    definition contain 0, then fixing the 2020 epoch is not the source of the narrowing, and the limitations section of
    the main text is rewritten accordingly; if any condition is not met, the main text reports the affected numbers
    instead and states their direction.

Usage:
    python code/66_exposure_epoch_check.py
    (Run after script 63: its workbook outputs/63_梯度平衡面板核验.xlsx is read to check that the main-text
    definition reproduces the full-sample NO₂ narrowing. The tile cache directory defaults to data/raw/ghsl2015
    and can be set with the environment variable GHSL_CACHE; the tiles are not stored in the repository, see BASE
    for their address.)

Output files:
    outputs/66_暴露时相核验.xlsx (exposure epoch check)
    logs/66_exposure_epoch_check.log
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window


def _load(name: str, file: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    if spec is None or spec.loader is None:
        raise SystemExit(f"{file} 无法加载")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export_utils = _load("export_utils", "01_export_utils.py")
m45 = _load("m45", "45_station_builtup_exposure.py")   # tile lookup and download reused from script 45
m51 = _load("m51", "51_station_timing_and_power.py")   # sample and series
m62 = _load("m62", "62_gradient_meteorology_check.py")  # estimator, segment rates and narrowing

# The download functions of script 45 read module-level constants, so switching these to the 2015 epoch lets them be
# reused unchanged
PRODUCT = "GHS_BUILT_S_E2015_GLOBE_R2023A_54009_100"
BASE = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/"
        f"{PRODUCT}/V1-0/tiles/{PRODUCT}_V1_0_R{{r}}_C{{c}}.zip")
CACHE = Path(os.environ.get("GHSL_CACHE", "data/raw/ghsl2015"))
vars(m45).update(PRODUCT=PRODUCT, BASE=BASE, CACHE=CACHE)

STA, CITY, TIME, EXPO = m51.STA, m51.CITY, m51.TIME, m51.EXPO
RADIUS = 500
YEARS = m62.YEARS
SEGS = m62.SEGS
B = 999
SEED = 20260926
Arr = npt.NDArray[np.float64]


def builtup_2015(logger: logging.Logger) -> pd.DataFrame:
    """500 m built-up fraction for the 2015 epoch; window, circular mask and edge handling are line-for-line the same
    as in script 45."""
    meta = pd.read_csv(export_utils.DATA_DIR / "国控站点元数据.csv")
    meta.columns = [str(c).lstrip("﻿") for c in meta.columns]
    tr = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    xs, ys = tr.transform(meta["经度"].values, meta["纬度"].values)
    meta["x"], meta["y"] = xs, ys
    meta["瓦片"] = [m45.tile_of(x, y) for x, y in zip(xs, ys, strict=True)]
    logger.info("站点 %s 个，需要瓦片 %s 个", len(meta), meta["瓦片"].nunique())
    k = math.ceil(RADIUS / m45.CELL)
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    mask = (yy ** 2 + xx ** 2) * m45.CELL_AREA <= RADIUS ** 2
    n_cell = int(mask.sum())
    recs = []
    for (r, c), grp in meta.groupby("瓦片"):
        tif = m45.ensure_tile(r, c, logger)
        if tif is None:
            raise SystemExit(f"瓦片 R{r}_C{c} 不可用，不输出任何数")
        with rasterio.open(tif) as ds:
            for _, s in grp.iterrows():
                row, col = ds.index(s["x"], s["y"])
                win = Window(col - k, row - k, 2 * k + 1, 2 * k + 1)
                try:
                    arr = ds.read(1, window=win, boundless=True, fill_value=0).astype("float64")
                except Exception:
                    arr = np.zeros((2 * k + 1, 2 * k + 1))
                built = float(np.nansum(np.where(mask, arr, 0.0)))
                recs.append({STA: s[STA], "瓦片": f"R{r}_C{c}", "建成区面积_m2_500m_2015": round(built, 1),
                             "建成区占比_500m_2015": round(built / (n_cell * m45.CELL_AREA), 5)})
        logger.info("  瓦片 R%s_C%s 完成，%s 个站", r, c, len(grp))
    return pd.DataFrame(recs)


def exposure_change(ex: pd.DataFrame) -> pd.DataFrame:
    """Station-level change between the two epochs: means, shares rising and falling, correlation between the epochs,
    slope of the change on the 2020 level, and stability of the within-city ranking."""
    e15, e20 = ex["建成区占比_500m_2015"], ex["建成区占比_500m_2020"]
    dlt = e20 - e15
    slope = float(np.polyfit(e20, dlt, 1)[0])
    # Within-city ranking: correlation between the within-city percentiles computed separately for each epoch
    r15 = ex.groupby(CITY)["建成区占比_500m_2015"].rank(pct=True)
    r20 = ex.groupby(CITY)["建成区占比_500m_2020"].rank(pct=True)
    multi = ex.groupby(CITY)[STA].transform("nunique") >= 2
    return pd.DataFrame([
        {"项": "站点数", "值": len(ex)},
        {"项": "2015 期均值", "值": float(e15.mean())},
        {"项": "2020 期均值", "值": float(e20.mean())},
        {"项": "2015→2020 变化均值", "值": float(dlt.mean())},
        {"项": "2015→2020 变化中位数", "值": float(dlt.median())},
        {"项": "上升的站点占比%", "值": float(100 * (dlt > 0).mean())},
        {"项": "下降的站点占比%", "值": float(100 * (dlt < 0).mean())},
        {"项": "两期 Pearson 相关", "值": float(np.corrcoef(e15, e20)[0, 1])},
        {"项": "变化对 2020 期水平的斜率", "值": slope},
        {"项": "城内百分位两期相关（同城 ≥2 站）", "值": float(np.corrcoef(r15[multi], r20[multi])[0, 1])},
    ])


def boot_stats(xtx: list[Arr], xty: list[Arr], names: list[str], w: npt.NDArray[np.int64]) -> list[float]:
    """Narrowing and segment rates under one set of city resampling weights (as in script 63)."""
    bb = np.linalg.solve(np.tensordot(w, np.stack(xtx), axes=1), w @ np.stack(xty))
    return [m62.narrowing(bb, names), *[m62.seg_rate(bb, names, a, c) for a, c in SEGS]]


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("66_exposure_epoch_check")
    e15 = builtup_2015(logger)
    d = m51.load()
    ex = d.groupby(STA).agg({CITY: "first", EXPO: "first"}).reset_index()
    ex = ex.merge(e15, on=STA, how="left").rename(columns={EXPO: "建成区占比_500m_2020"})
    if ex["建成区占比_500m_2015"].isna().any():
        raise SystemExit(f"{int(ex['建成区占比_500m_2015'].isna().sum())} 个面板站点缺 2015 期暴露，不输出任何数")
    chg = exposure_change(ex)
    logger.info("站点暴露两期对比：\n%s", chg.to_string(index=False))

    d = d.merge(ex[[STA, "建成区占比_500m_2015"]], on=STA, how="left")
    w = ((d[TIME] - 2015) / 5).clip(0, 1)
    exposures = {"2020 期（正文口径）": d[EXPO],
                 "2015 期": d["建成区占比_500m_2015"],
                 "随时间变化（2015–2020 插值）": (1 - w) * d["建成区占比_500m_2015"] + w * d[EXPO]}
    rng = np.random.default_rng(SEED)
    narrow_rows: list[dict[str, object]] = []
    seg_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    for tag, ycol in m51.SERIES:
        full = d.dropna(subset=[ycol, EXPO]).copy()
        full["_y"] = full[ycol]
        cities = list(pd.factorize(full[CITY])[1])
        counts = rng.multinomial(len(cities), np.full(len(cities), 1 / len(cities)), size=B)
        draws: dict[str, Arr] = {}
        point: dict[str, list[float]] = {}
        for lab, e in exposures.items():
            g = full.copy()
            g[EXPO] = e.loc[g.index].to_numpy(float)   # the design matrix of script 62 reads the EXPO column
            est, xtx, xty, _ = m62.fit(g, "M0")
            names = est["项"].tolist()
            coef = est["系数"].to_numpy(float)
            draws[lab] = np.array([boot_stats(xtx, xty, names, counts[i]) for i in range(B)])
            point[lab] = [m62.narrowing(coef, names), *[m62.seg_rate(coef, names, a, c) for a, c in SEGS]]
            lo, hi = np.percentile(draws[lab][:, 0], [2.5, 97.5])
            narrow_rows.append({"序列": tag, "暴露口径": lab, "N（站-年）": len(g),
                                "2015 梯度": coef[names.index("暴露×2015")],
                                "2024 梯度": coef[names.index("暴露×2024")],
                                "2015→2024 相对变化%": point[lab][0],
                                "bootstrap 95% 下限": float(lo), "bootstrap 95% 上限": float(hi)})
            for j, (a, c) in enumerate(SEGS, start=1):
                slo, shi = np.percentile(draws[lab][:, j], [2.5, 97.5])
                seg_rows.append({"序列": tag, "暴露口径": lab, "分段": f"{a}–{c}",
                                 "平均年变化（梯度单位/年）": point[lab][j],
                                 "bootstrap 95% 下限": float(slo), "bootstrap 95% 上限": float(shi)})
            for _, r in est.iterrows():
                path_rows.append({"序列": tag, "暴露口径": lab, "年份": int(str(r["项"])[3:]),
                                  "城内梯度": r["系数"], "城市聚类SE": r["城市聚类SE"]})
        for lab in list(exposures)[1:]:
            diff = draws[lab] - draws["2020 期（正文口径）"]
            lo, hi = np.percentile(diff[:, 0], [2.5, 97.5])
            narrow_rows.append({"序列": tag, "暴露口径": f"{lab} − 正文口径",
                                "2015→2024 相对变化%": point[lab][0] - point["2020 期（正文口径）"][0],
                                "bootstrap 95% 下限": float(lo), "bootstrap 95% 上限": float(hi)})
            for j, (a, c) in enumerate(SEGS, start=1):
                slo, shi = np.percentile(diff[:, j], [2.5, 97.5])
                seg_rows.append({"序列": tag, "暴露口径": f"{lab} − 正文口径", "分段": f"{a}–{c}",
                                 "平均年变化（梯度单位/年）": point[lab][j] - point["2020 期（正文口径）"][j],
                                 "bootstrap 95% 下限": float(slo), "bootstrap 95% 上限": float(shi)})

    narrow, segs = pd.DataFrame(narrow_rows), pd.DataFrame(seg_rows)
    # Gate: the main-text definition must reproduce the full-sample NO₂ narrowing of script 63 (same estimator,
    # same sample)
    ref = pd.read_excel(export_utils.OUTPUT_DIR / "63_梯度平衡面板核验.xlsx", sheet_name="01_收窄幅度")
    ref_no2 = float(ref[(ref["序列"] == "NO2") & (ref["样本"] == "全样本")]["2015→2024 相对变化%"].iloc[0])
    got = float(narrow[(narrow["序列"] == "NO2") & (narrow["暴露口径"] == "2020 期（正文口径）")]
                ["2015→2024 相对变化%"].iloc[0])
    logger.info("闸：正文口径 NO₂ 收窄 %.4f%%，63 号全样本 %.4f%%", got, ref_no2)
    if round(got, 6) != round(ref_no2, 6):
        raise SystemExit("正文口径未复现 63 号的 NO₂ 收窄幅度，不输出任何数")

    no2 = segs[(segs["序列"] == "NO2")]
    ok = True
    for lab in list(exposures)[1:]:
        n_pt = narrow[(narrow["序列"] == "NO2") & (narrow["暴露口径"] == lab)].iloc[0]
        n_df = narrow[(narrow["序列"] == "NO2") & (narrow["暴露口径"] == f"{lab} − 正文口径")].iloc[0]
        s_pt = no2[(no2["暴露口径"] == lab) & (no2["分段"] == "2017–2020")].iloc[0]
        s_df = no2[(no2["暴露口径"] == f"{lab} − 正文口径") & (no2["分段"] == "2017–2020")].iloc[0]
        cond = (n_pt["2015→2024 相对变化%"] < 0 and s_pt["平均年变化（梯度单位/年）"] < 0
                and n_df["bootstrap 95% 下限"] <= 0 <= n_df["bootstrap 95% 上限"]
                and s_df["bootstrap 95% 下限"] <= 0 <= s_df["bootstrap 95% 上限"])
        ok = ok and bool(cond)
        logger.info("NO₂ %s：收窄 %.2f%%（与正文口径之差 %.2f，95%% %.2f, %.2f）；2017–2020 年均 %.4f"
                    "（差 %.4f，95%% %.4f, %.4f）", lab, n_pt["2015→2024 相对变化%"], n_df["2015→2024 相对变化%"],
                    n_df["bootstrap 95% 下限"], n_df["bootstrap 95% 上限"], s_pt["平均年变化（梯度单位/年）"],
                    s_df["平均年变化（梯度单位/年）"], s_df["bootstrap 95% 下限"], s_df["bootstrap 95% 上限"])
    # "预设判读" in the message below refers to the decision rule in the docstring; see its note on timing
    verdict = ("固定 2020 期不是收窄的来源：两种替代口径下收窄与 2017–2020 年下降均保持方向，与正文口径之差的区间含 0"
               if ok else "未满足预设判读：正文须改报受影响的数字")
    logger.info("判读：%s", verdict)
    out = export_utils.write_excel_workbook("66_暴露时相核验", [
        ("判读", pd.DataFrame([{"判读": verdict}])),
        ("站点暴露两期对比", chg),
        ("站点暴露", ex),
        ("收窄幅度", narrow),
        ("分段速率", segs),
        ("逐年梯度", pd.DataFrame(path_rows)),
    ])
    print(f"✓ 暴露时相核验完成：{verdict}\n  输出: {out}\n  日志: {log_path}")


if __name__ == "__main__":
    main()
