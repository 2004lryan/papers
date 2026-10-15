"""
67_gradient_differential_check.py: annual path of the within-city gradient of NO₂ minus that of a control species

Why it is needed:
    Section 3.2 of the main text and Table S3 of the Supplementary Material report the within-city gradient paths of
    NO₂ and SO₂ separately: SO₂ narrows by 84.6%, faster than the 42.5% of NO₂, from which the narrowing is judged
    not to be specific to traffic. The two paths each use their own sample and cannot answer "how does the part of
    the NO₂ gradient in excess of SO₂ change". That requires estimating, on one and the same set of station-years,
    the annual gradient of ln NO₂ − ln SO₂ (the log of the NO₂ to SO₂ concentration ratio) on exposure.
    Differencing the separate NO₂ and SO₂ paths of Table S3 suggests that this difference widened from about 0.05
    to about 0.30; this script gives the formal estimate and interval, with PM2.5 as a second control.

    The estimator is the same as M0 of script 62 and as script 63 (city-by-year fixed effects + exposure × year,
    city-clustered SE), and the functions of script 62 are called directly; city cluster bootstrap with 999 draws.
    The sample is the station-years on which NO₂ and the control species both meet their own 274-day threshold.
    The separate paths of NO₂ and of the control species are also given on the same sample: the difference path
    must equal the difference of the two in every year (linearity of OLS under one design matrix); this identity
    serves as a gate, and if it fails no number is output.

Interpretation rule (written with the script; the repository holds no dated record that it preceded the run):
    If the interval of the 2015→2024 change in the difference gradient (gradient units) excludes 0 and is positive,
    the main text says "relative to this control species, the NO₂ gradient widened"; if it excludes 0 and is
    negative, it says "also narrowing relative to this control species"; if it contains 0, it says "no detectable
    change relative to this control species".

Usage:
    python code/67_gradient_differential_check.py

Output files:
    outputs/67_梯度差值路径.xlsx (gradient difference paths)
    logs/67_gradient_differential_check.log
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
m51 = _load("m51", "51_station_timing_and_power.py")   # sample
m62 = _load("m62", "62_gradient_meteorology_check.py")  # estimator and segment rates

CITY, TIME, EXPO = m51.CITY, m51.TIME, m51.EXPO
SEGS = m62.SEGS
CONTROLS = [("SO2", "ln_SO2"), ("PM2.5", "ln_PM25")]
B = 999
SEED = 20260927
Arr = npt.NDArray[np.float64]


def change_stats(b: Arr, names: list[str]) -> list[float]:
    """2015→2024 change of the gradient (gradient units) and the mean annual change in each segment."""
    return [float(b[names.index("暴露×2024")] - b[names.index("暴露×2015")]),
            *[m62.seg_rate(b, names, a, c) for a, c in SEGS]]


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("67_gradient_differential_check")
    d = m51.load()
    rng = np.random.default_rng(SEED)
    chg_rows: list[dict[str, object]] = []
    seg_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    for ctl, ccol in CONTROLS:
        s = d.dropna(subset=["ln_NO2", ccol, EXPO]).copy()
        cities = list(pd.factorize(s[CITY])[1])
        counts = rng.multinomial(len(cities), np.full(len(cities), 1 / len(cities)), size=B)
        series = {f"NO2 − {ctl}": s["ln_NO2"] - s[ccol], "NO2（同一样本）": s["ln_NO2"],
                  f"{ctl}（同一样本）": s[ccol]}
        coefs: dict[str, Arr] = {}
        for lab, y in series.items():
            g = s.copy()
            g["_y"] = y.to_numpy(float)
            est, xtx, xty, _ = m62.fit(g, "M0")
            names = est["项"].tolist()
            coef = est["系数"].to_numpy(float)
            coefs[lab] = coef
            draws = np.array([change_stats(np.linalg.solve(np.tensordot(counts[i], np.stack(xtx), axes=1),
                                                           counts[i] @ np.stack(xty)), names) for i in range(B)])
            pt = change_stats(coef, names)
            lo, hi = np.percentile(draws[:, 0], [2.5, 97.5])
            # base-year difference is near 0, so a relative change is meaningless; reported for single species only
            rel = np.nan if lab.startswith("NO2 −") else m62.narrowing(coef, names)
            chg_rows.append({"对照": ctl, "序列": lab, "N（站-年）": len(g), "城市": g[CITY].nunique(),
                             "2015 梯度": coef[names.index("暴露×2015")], "2024 梯度": coef[names.index("暴露×2024")],
                             "2015→2024 变化（梯度单位）": pt[0],
                             "bootstrap 95% 下限": float(lo), "bootstrap 95% 上限": float(hi),
                             "2015→2024 相对变化%": rel})
            for j, (a, c) in enumerate(SEGS, start=1):
                slo, shi = np.percentile(draws[:, j], [2.5, 97.5])
                seg_rows.append({"对照": ctl, "序列": lab, "分段": f"{a}–{c}", "平均年变化（梯度单位/年）": pt[j],
                                 "bootstrap 95% 下限": float(slo), "bootstrap 95% 上限": float(shi)})
            for _, r in est.iterrows():
                yr = int(str(r["项"])[3:])
                path_rows.append({"对照": ctl, "序列": lab, "年份": yr, "城内梯度": r["系数"],
                                  "城市聚类SE": r["城市聚类SE"], "当年站点数": int((g[TIME] == yr).sum())})
        gap = np.max(np.abs(coefs[f"NO2 − {ctl}"] - (coefs["NO2（同一样本）"] - coefs[f"{ctl}（同一样本）"])))
        logger.info("闸（%s）：差值路径与两条路径之差的最大偏差 %.2e", ctl, gap)
        if gap > 1e-10:
            raise SystemExit("差值路径不等于两条路径之差，不输出任何数")

    chg = pd.DataFrame(chg_rows)
    verdicts = []
    for ctl, _ in CONTROLS:
        r = chg[chg["序列"] == f"NO2 − {ctl}"].iloc[0]
        lo, hi, v = r["bootstrap 95% 下限"], r["bootstrap 95% 上限"], r["2015→2024 变化（梯度单位）"]
        text = (f"相对 {ctl}，NO₂ 的梯度扩大" if lo > 0 else f"相对 {ctl} 也在收窄" if hi < 0
                else f"相对 {ctl} 没有可检出的变化")
        verdicts.append({"对照": ctl, "2015 差值梯度": r["2015 梯度"], "2024 差值梯度": r["2024 梯度"],
                         "变化": v, "95% 下限": lo, "95% 上限": hi, "判读": text})
        logger.info("NO₂ − %s：2015 %.4f → 2024 %.4f，变化 %.4f（95%% %.4f, %.4f）→ %s", ctl, r["2015 梯度"],
                    r["2024 梯度"], v, lo, hi, text)
    for r in seg_rows:
        logger.info("%s %s 平均年变化 %.4f（95%% %.4f, %.4f）", r["序列"], r["分段"], r["平均年变化（梯度单位/年）"],
                    r["bootstrap 95% 下限"], r["bootstrap 95% 上限"])
    out = export_utils.write_excel_workbook("67_梯度差值路径", [
        ("判读", pd.DataFrame(verdicts)),
        ("十年变化", chg),
        ("分段速率", pd.DataFrame(seg_rows)),
        ("逐年梯度", pd.DataFrame(path_rows)),
    ])
    print(f"✓ 梯度差值路径完成\n  输出: {out}\n  日志: {log_path}")


if __name__ == "__main__":
    main()
