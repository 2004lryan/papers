"""
63_gradient_balanced_panel_check.py: does the narrowing of the within-city gradient come from changes in station
composition?

Why it is needed:
    The annual gradients of script 51 are estimated each year from all stations that qualify in that year. The
    number of stations in the panel is not fixed: the NO₂ series has 967 station-years in 2020 and 1278 in 2021,
    so more than three hundred stations entered within one year. If the new stations fall systematically at one
    end of the exposure distribution, or lie in cities whose gradient was flat to begin with, part of the decline
    in the annual gradient may come from "a different set of stations" rather than from a shrinking gap between
    the same stations.

    This script keeps only the stations that qualify in every one of the ten years (2015–2024) and, on this fixed
    station set, re-estimates the annual gradients, the size of the narrowing from 2015 to 2024 and the segment
    rates (2015–2017, 2017–2020 and 2020–2024 as in the main text, plus 2017–2024), comparing each of them with the
    full sample. If the two agree, the narrowing is not caused by compositional change. For NO₂ it also gives station
    counts and mean exposure grouped by year of entry, to show how large the compositional change itself is.

    The estimator is exactly M0 of script 62 (city-by-year fixed effects + exposure × year, city-clustered SE),
    and the functions of script 62 are called directly; uncertainty comes from a city cluster bootstrap (999
    replications). The full sample and the fixed station set use **the same** city resampling weights, so the
    interval of their difference is a paired interval.

Usage:
    python code/63_gradient_balanced_panel_check.py

Output files:
    outputs/63_梯度平衡面板核验.xlsx (gradient balanced-panel check)
    logs/63_gradient_balanced_panel_check.log
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pandas as pd


def _load(name: str, file: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    if spec is None or spec.loader is None:
        raise SystemExit(f"{file} 无法加载")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export_utils = _load("export_utils", "01_export_utils.py")
m51 = _load("m51", "51_station_timing_and_power.py")   # sample and series
m62 = _load("m62", "62_gradient_meteorology_check.py")  # definitions of the estimator, segment rates and narrowing size

STA, CITY, TIME, EXPO = m51.STA, m51.CITY, m51.TIME, m51.EXPO
YEARS = m62.YEARS
SEGS = m62.SEGS
B = 999
SEED = 20260925
Arr = npt.NDArray[np.float64]


def balanced(g: pd.DataFrame) -> pd.DataFrame:
    """Stations that qualify in every one of the ten years; a city with fewer than 2 such stations cannot give a
    within-city gradient and is dropped as well."""
    yrs = g.groupby(STA)[TIME].nunique()
    keep = g[g[STA].isin(yrs[yrs == len(YEARS)].index)]
    per_city = keep.groupby(CITY)[STA].nunique()
    return keep[keep[CITY].isin(per_city[per_city >= 2].index)].copy()


def boot_stats(xtx: list[Arr], xty: list[Arr], names: list[str], w: npt.NDArray[np.int64]) -> list[float]:
    """Size of the narrowing and each segment rate under one set of city resampling weights."""
    bb = np.linalg.solve(np.tensordot(w, np.stack(xtx), axes=1), w @ np.stack(xty))
    return [m62.narrowing(bb, names), *[m62.seg_rate(bb, names, a, c) for a, c in SEGS]]


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("63_gradient_balanced_panel_check")
    d = m51.load()
    rng = np.random.default_rng(SEED)
    narrow_rows: list[dict[str, object]] = []
    seg_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    size_rows: list[dict[str, object]] = []
    for tag, ycol in m51.SERIES:
        full = d.dropna(subset=[ycol, EXPO]).copy()
        full["_y"] = full[ycol]
        bal = balanced(full)
        samples = {"全样本": full, "十年齐全站点": bal}
        cities = list(pd.factorize(full[CITY])[1])
        counts = rng.multinomial(len(cities), np.full(len(cities), 1 / len(cities)), size=B)
        draws: dict[str, Arr] = {}
        point: dict[str, list[float]] = {}
        for lab, g in samples.items():
            est, xtx, xty, _ = m62.fit(g, "M0")
            names = est["项"].tolist()
            coef = est["系数"].to_numpy(float)
            # Positions of this sample's cities within the full-sample city resampling weights: the two samples
            # share the same draw, which is what makes their difference paired
            order = [cities.index(c) for c in pd.factorize(g[CITY])[1]]
            draws[lab] = np.array([boot_stats(xtx, xty, names, counts[i, order]) for i in range(B)])
            point[lab] = [m62.narrowing(coef, names), *[m62.seg_rate(coef, names, a, c) for a, c in SEGS]]
            lo, hi = np.percentile(draws[lab][:, 0], [2.5, 97.5])
            narrow_rows.append({"序列": tag, "样本": lab, "站点": g[STA].nunique(), "城市": g[CITY].nunique(),
                                "N（站-年）": len(g), "2015 梯度": coef[names.index("暴露×2015")],
                                "2024 梯度": coef[names.index("暴露×2024")],
                                "2015→2024 相对变化%": point[lab][0],
                                "bootstrap 95% 下限": float(lo), "bootstrap 95% 上限": float(hi)})
            for j, (a, c) in enumerate(SEGS, start=1):
                slo, shi = np.percentile(draws[lab][:, j], [2.5, 97.5])
                seg_rows.append({"序列": tag, "样本": lab, "分段": f"{a}–{c}",
                                 "平均年变化（梯度单位/年）": point[lab][j],
                                 "bootstrap 95% 下限": float(slo), "bootstrap 95% 上限": float(shi)})
            for _, r in est.iterrows():
                path_rows.append({"序列": tag, "样本": lab, "年份": int(str(r["项"])[3:]),
                                  "城内梯度": r["系数"], "城市聚类SE": r["城市聚类SE"],
                                  "当年站点数": int((g[TIME] == int(str(r["项"])[3:])).sum())})
        diff = draws["十年齐全站点"] - draws["全样本"]
        lo, hi = np.percentile(diff[:, 0], [2.5, 97.5])
        narrow_rows.append({"序列": tag, "样本": "十年齐全站点 − 全样本",
                            "2015→2024 相对变化%": point["十年齐全站点"][0] - point["全样本"][0],
                            "bootstrap 95% 下限": float(lo), "bootstrap 95% 上限": float(hi)})
        for j, (a, c) in enumerate(SEGS, start=1):
            slo, shi = np.percentile(diff[:, j], [2.5, 97.5])
            seg_rows.append({"序列": tag, "样本": "十年齐全站点 − 全样本", "分段": f"{a}–{c}",
                             "平均年变化（梯度单位/年）": point["十年齐全站点"][j] - point["全样本"][j],
                             "bootstrap 95% 下限": float(slo), "bootstrap 95% 上限": float(shi)})
        if tag == "NO2":
            # How large the compositional change itself is: stations grouped by the year they first qualify
            first = full.groupby(STA)[TIME].min().rename("首次合格年份")
            ex = full.groupby(STA)[EXPO].first()
            ent = pd.concat([first, ex], axis=1)
            for yr, grp in ent.groupby("首次合格年份"):
                size_rows.append({"首次合格年份": int(yr), "站点数": len(grp),
                                  "建成区占比均值": float(grp[EXPO].mean()),
                                  "建成区占比中位数": float(grp[EXPO].median()),
                                  "其中十年齐全": int(grp.index.isin(bal[STA]).sum())})

    narrow = pd.DataFrame(narrow_rows)
    for _, r in narrow.iterrows():
        logger.info("%s %s 2015→2024 相对变化 %.2f%%（95%% %.2f, %.2f）", r["序列"], r["样本"],
                    r["2015→2024 相对变化%"], r["bootstrap 95% 下限"], r["bootstrap 95% 上限"])
    for r in seg_rows:
        if r["序列"] == "NO2":
            logger.info("NO2 %s %s 平均年变化 %.4f（95%% %.4f, %.4f）", r["样本"], r["分段"],
                        r["平均年变化（梯度单位/年）"], r["bootstrap 95% 下限"], r["bootstrap 95% 上限"])
    for r in size_rows:
        logger.info("NO2 首次合格 %s 年：%s 站，建成区占比均值 %.4f，其中十年齐全 %s 站", r["首次合格年份"],
                    r["站点数"], r["建成区占比均值"], r["其中十年齐全"])
    out = export_utils.write_excel_workbook("63_梯度平衡面板核验", [
        ("收窄幅度", narrow),
        ("分段速率", pd.DataFrame(seg_rows)),
        ("逐年梯度", pd.DataFrame(path_rows)),
        ("站点进入年份", pd.DataFrame(size_rows)),
    ])
    print(f"✓ 梯度平衡面板核验完成\n  输出: {out}\n  日志: {log_path}")


if __name__ == "__main__":
    main()
