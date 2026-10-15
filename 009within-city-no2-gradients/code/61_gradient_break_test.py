"""
61_gradient_break_test.py: city cluster bootstrap of differences between segment rates of the within-city gradient

Why it is needed:
    Section 3.2 of the main text and Section S3 of the Supplementary Material describe the gradient series by their
    mean annual change in each segment; Table S3 does so for all four series (NO₂, NO₂ commuting-peak to overnight
    ratio, SO₂, PM2.5), and its note gives e.g. a mean annual relative change of −0.78% for NO₂ over 2015–2017 and
    −5.94% over 2017–2024, and −11.43% in both segments for SO₂.
    These are point estimates computed from the end points and carry no sampling uncertainty of their own, which is
    not enough to judge whether the NO₂ acceleration exceeds sampling noise, or how large a change the "same in both
    segments" of SO₂ and PM2.5 can rule out.
    This script runs a city cluster bootstrap: each draw samples cities with replacement (a city drawn more than
    once counts as distinct clusters), re-estimates the four series year by year with the same gradient estimator as
    script 51 (within one year, ln Y on exposure, with city fixed effects), recomputes the segment rates and their
    differences as defined in the main text, and gives 95% percentile intervals and two-sided bootstrap p values.

    The same resampling also gives: the change in level of the gradient over each segment; the completion of the NO₂
    narrowing used in the timing argument of the main text (Section 3.4) and the annual-change correlation of
    Table S4; the sensitivity
    of completion by 2020 to the end year (2022, 2023, 2024), with the corresponding completion of the fleet on the
    two scales of vehicle counts and logarithms; and, inside and outside the priority regions (city-list definition
    of script 51), the relative change over 2017–2020 and 2020–2023 for NO₂, its diurnal ratio and SO₂, and the
    inside-minus-outside difference.

    With city fixed effects a city contributes to the slope only through its own demeaned cross-product sum Sxy and
    sum of squared deviations Sxx, and a city drawn k times contributes k times, so the slope of each draw equals
    Σk·Sxy / Σk·Sxx and the data need not be reassembled.
    Sample, series and gradient definition are all taken from script 51; the point estimates first reproduce the
    annual gradients of script 51 cell by cell, and the script stops at any mismatch.

Usage:
    python code/61_gradient_break_test.py

Output files:
    outputs/61_梯度分段速率检验.xlsx (tests of the gradient segment rates)
    logs/61_gradient_break_test.log
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
m51 = _load("m51", "51_station_timing_and_power.py")   # sample, series and gradient function all from script 51
m52 = _load("m52", "52_station_figures.py")            # national NEV series (2024: Ministry of Public Security total)

CITY, TIME, EXPO = m51.CITY, m51.TIME, m51.EXPO
YEARS = list(range(2015, 2025))
B = 999                      # same number of replications as the other bootstraps in the manuscript
SEED = 20260922
# segments compared in Table S3; NO₂ also has the 2017–2020 and 2020–2024 segments used in the main text
SEGMENTS = [(2015, 2017), (2017, 2024)]
NO2_EXTRA = [(2017, 2020), (2020, 2024)]
Arr = npt.NDArray[np.float64]


def city_sums(d: pd.DataFrame, ycol: str) -> tuple[Arr, Arr]:
    """Demeaned cross-product sum Sxy and sum of squared deviations Sxx for every year and city
    (rows: cities; columns: years)."""
    cities = pd.Index(sorted(d[CITY].unique()))
    sxy = np.zeros((len(cities), len(YEARS)))
    sxx = np.zeros((len(cities), len(YEARS)))
    for j, yr in enumerate(YEARS):
        g = d[d[TIME] == yr].dropna(subset=[ycol, EXPO])
        z = g[[ycol, EXPO]].astype(float)
        z = z - z.groupby(g[CITY].values).transform("mean")
        prod = (z[ycol] * z[EXPO]).groupby(g[CITY].values).sum()
        sq = (z[EXPO] ** 2).groupby(g[CITY].values).sum()
        idx = cities.get_indexer(prod.index)
        sxy[idx, j] = prod.values
        sxx[idx, j] = sq.values
    return sxy, sxx


def rates(g: Arr, a: int, c: int) -> tuple[Arr, Arr]:
    """Mean annual change over a segment (units per year) and mean annual relative change (%/year, relative to the
    start of the segment); the last axis of g is the year."""
    ga, gc = g[..., YEARS.index(a)], g[..., YEARS.index(c)]
    absr = (gc - ga) / (c - a)
    return absr, 100 * absr / ga


def summarise(point: float, draws: Arr) -> dict[str, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((1 + np.sum(draws <= 0)) / (B + 1), (1 + np.sum(draws >= 0)) / (B + 1))
    return {"点估计": point, "bootstrap SE": float(np.std(draws, ddof=1)),
            "95% 区间下限": float(lo), "95% 区间上限": float(hi), "双侧 bootstrap p": min(float(p), 1.0)}


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("61_gradient_break_test")
    d = m51.load()
    cities = sorted(d[CITY].unique())
    logger.info("样本 %s 站-年，城市 %s；bootstrap %s 次，种子 %s", len(d), len(cities), B, SEED)

    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(len(cities), np.full(len(cities), 1 / len(cities)), size=B)   # B × cities

    # national NEV: sum of the 31 provinces on the Ministry of Public Security registration basis (same source as
    # script 51), annual increments over 2017–2023
    prov = pd.read_csv(export_utils.DATA_DIR / "省级新能源汽车保有量-公安部口径-2017_2023.csv")
    prov.columns = [x.lstrip("﻿") for x in prov.columns]
    nev_inc = prov.groupby(TIME)["新能源汽车保有量_辆"].sum().sort_index().diff().dropna()

    repro: list[dict[str, object]] = []
    seg_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []
    comp_rows: list[dict[str, object]] = []
    end_rows: list[dict[str, object]] = []
    corr_row: dict[str, object] = {}
    for tag, ycol in m51.SERIES:
        sxy, sxx = city_sums(d, ycol)
        g_hat = sxy.sum(0) / sxx.sum(0)
        for j, yr in enumerate(YEARS):
            b51 = m51.gradient(d[d[TIME] == yr], ycol)[0]
            repro.append({"序列": tag, "年份": yr, "51 号梯度": b51, "本脚本梯度": g_hat[j],
                          "差": g_hat[j] - b51})
            if abs(g_hat[j] - b51) > 1e-10:
                raise SystemExit(f"{tag} {yr} 未复现 51 号梯度：{g_hat[j]} vs {b51}")
        g_bs = (counts @ sxy) / (counts @ sxx)                                            # B × years
        segs = SEGMENTS + (NO2_EXTRA if tag == "NO2" else [])
        est_pt: dict[tuple[int, int], tuple[float, float]] = {}
        est_bs: dict[tuple[int, int], tuple[Arr, Arr]] = {}
        for a, c in segs:
            ab, rel = rates(g_hat, a, c)
            ab_b, rel_b = rates(g_bs, a, c)
            est_pt[(a, c)] = (float(ab), float(rel))
            est_bs[(a, c)] = (ab_b, rel_b)
            for unit, pt, dr in [("单位/年", float(ab), ab_b), ("%/年", float(rel), rel_b)]:
                seg_rows.append({"序列": tag, "分段": f"{a}–{c}", "口径": unit, **summarise(pt, dr)})
        comps = [((2015, 2017), (2017, 2024))] + ([((2017, 2020), (2020, 2024))] if tag == "NO2" else [])
        for s1, s2 in comps:
            for unit, k in [("单位/年", 0), ("%/年（百分点）", 1)]:
                pt = est_pt[s2][k] - est_pt[s1][k]
                dr = est_bs[s2][k] - est_bs[s1][k]
                row = {"序列": tag, "对比": f"{s2[0]}–{s2[1]} 减 {s1[0]}–{s1[1]}", "口径": unit,
                       **summarise(pt, dr)}
                diff_rows.append(row)
                logger.info("%s %s（%s）：%.4f，95%% 区间 [%.4f, %.4f]，p=%.3f", tag, row["对比"], unit,
                            pt, row["95% 区间下限"], row["95% 区间上限"], row["双侧 bootstrap p"])

        # level change per segment (last-year minus first-year gradient): whether the gradient really changed within it
        for a, c in [(2015, 2017), (2017, 2020), (2020, 2024), (2017, 2024), (2015, 2024)]:
            ia, ic = YEARS.index(a), YEARS.index(c)
            level_rows.append({"序列": tag, "分段": f"{a}–{c}",
                               **summarise(float(g_hat[ic] - g_hat[ia]), g_bs[:, ic] - g_bs[:, ia])})
        if tag == "NO2":
            # timing argument of main-text §3.4 and SI S4: share (%) of the 2017–2023 narrowing completed by each year
            i17, i23 = YEARS.index(2017), YEARS.index(2023)
            for yr in range(2018, 2023):
                i = YEARS.index(yr)
                pt = 100 * (g_hat[i17] - g_hat[i]) / (g_hat[i17] - g_hat[i23])
                dr = 100 * (g_bs[:, i17] - g_bs[:, i]) / (g_bs[:, i17] - g_bs[:, i23])
                comp_rows.append({"截至年份": yr, **summarise(float(pt), dr)})
            # end-point sensitivity: the 2023 gradient rose slightly from 2022; with the end point moved to 2022 or
            # to the last panel year 2024, the completion by 2020 of the narrowing and of the fleet (national series,
            # not resampled)
            nev_all = m52.nev_series()
            i20 = YEARS.index(2020)
            for end in (2022, 2023, 2024):
                ie = YEARS.index(end)
                pt = 100 * (g_hat[i17] - g_hat[i20]) / (g_hat[i17] - g_hat[ie])
                dr = 100 * (g_bs[:, i17] - g_bs[:, i20]) / (g_bs[:, i17] - g_bs[:, ie])
                end_rows.append({"终点年份": end, "截至年份": 2020, **summarise(float(pt), dr),
                                 "车队完成度%（辆数）": 100 * float((nev_all[2020] - nev_all[2017])
                                                             / (nev_all[end] - nev_all[2017])),
                                 "车队完成度%（对数）": 100 * float(np.log(nev_all[2020] / nev_all[2017])
                                                             / np.log(nev_all[end] / nev_all[2017]))})
            # correlation of annual changes with annual national NEV increments (national registrations, not resampled)
            i17b = YEARS.index(2017)
            dn = nev_inc.values
            dg_hat = np.diff(g_hat[i17b:i23 + 1])
            dg_bs = np.diff(g_bs[:, i17b:i23 + 1], axis=1)
            r_bs = np.array([np.corrcoef(x, dn)[0, 1] for x in dg_bs])
            corr_row = {"量": "逐年梯度变化与全国 NEV 年增量的相关 r（2018–2023，6 对）",
                        **summarise(float(np.corrcoef(dg_hat, dn)[0, 1]), r_bs)}

    # Inside and outside the priority regions (city-list definition of script 51): relative change in the two
    # segments and the inside-minus-outside difference.
    # Region is a city attribute, so a resampled city carries its own group, and within-group slopes again follow
    # from Σk·Sxy / Σk·Sxx.
    region_rows: list[dict[str, object]] = []
    in_region = d.groupby(CITY)["重点区域"].first().reindex(cities).to_numpy() == 1.0
    for tag, ycol in m51.SERIES[:3]:
        sxy, sxx = city_sums(d, ycol)
        chg: dict[str, tuple[float, Arr]] = {}
        for lab, mask in [("重点区域", in_region), ("非重点区域", ~in_region)]:
            g_hat = sxy[mask].sum(0) / sxx[mask].sum(0)
            b51 = [m51.gradient(d[(d[TIME] == yr) & (d["重点区域"] == float(lab == "重点区域"))], ycol)[0]
                   for yr in YEARS]
            if np.max(np.abs(g_hat - np.array(b51))) > 1e-10:
                raise SystemExit(f"{tag} {lab} 未复现 51 号分组梯度")
            g_bs = (counts[:, mask] @ sxy[mask]) / (counts[:, mask] @ sxx[mask])
            for a, c in [(2017, 2020), (2020, 2023)]:
                ia, ic = YEARS.index(a), YEARS.index(c)
                pt = 100 * (g_hat[ic] / g_hat[ia] - 1)
                dr = 100 * (g_bs[:, ic] / g_bs[:, ia] - 1)
                chg[f"{lab} {a}–{c}"] = (float(pt), dr)
                region_rows.append({"序列": tag, "组别": lab, "分段": f"{a}–{c}", "口径": "相对变化%",
                                    **summarise(float(pt), dr)})
        for a, c in [(2017, 2020), (2020, 2023)]:
            p_in, d_in = chg[f"重点区域 {a}–{c}"]
            p_out, d_out = chg[f"非重点区域 {a}–{c}"]
            region_rows.append({"序列": tag, "组别": "区内减区外", "分段": f"{a}–{c}", "口径": "百分点",
                                **summarise(p_in - p_out, d_in - d_out)})

    out = export_utils.write_excel_workbook("61_梯度分段速率检验", [
        ("逐年梯度复现", pd.DataFrame(repro)), ("分段速率", pd.DataFrame(seg_rows)),
        ("速率之差", pd.DataFrame(diff_rows)), ("分段水平变化", pd.DataFrame(level_rows)),
        ("收窄完成度", pd.DataFrame(comp_rows)), ("年度相关", pd.DataFrame([corr_row])),
        ("重点区域内外", pd.DataFrame(region_rows)), ("完成度终点敏感性", pd.DataFrame(end_rows))])
    logger.info("输出 %s；日志 %s", out, log_path)


if __name__ == "__main__":
    main()
