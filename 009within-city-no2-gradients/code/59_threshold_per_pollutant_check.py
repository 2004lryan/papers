"""
59_threshold_per_pollutant_check.py: valid-day threshold applied per pollutant vs. applying the NO₂ threshold

Why it is needed:
    Supplementary Section S1 states that the threshold is applied separately to each pollutant and gives an example:
    if the NO₂ threshold were applied to PM₂.₅, station-years with incomplete PM₂.₅ series would enter the sample and
    the 2024 PM₂.₅ within-city gradient would change accordingly, while the NO₂ results would stay the same.
    This script computes that comparison with the same sample and the same gradient function as script 51, and
    reports by year the PM₂.₅ and SO₂ gradients and sample sizes under both approaches, together with the NO₂ gradient
    (identical under both). It reads the panel and outputs/51_梯度路径与可检测性.xlsx (sheet 02_逐年梯度路径), whose
    own-threshold gradients it must reproduce, and writes nothing back to data/.

    Script 51 first selects station-years with NO₂ valid days ≥274 (the ≥2 stations per city requirement is counted
    on NO₂), then takes each pollutant's values subject to that pollutant's own valid-day threshold; "applying the NO₂
    threshold" means that the pollutant's own threshold is no longer imposed within this sample.

Usage:
    python code/59_threshold_per_pollutant_check.py

Output files:
    outputs/59_分污染物门槛对照.xlsx (per-pollutant threshold comparison)
    logs/59_threshold_per_pollutant_check.log
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


def _load(name: str, file: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    if spec is None or spec.loader is None:
        raise SystemExit(f"{file} 无法加载")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export_utils = _load("export_utils", "01_export_utils.py")
m51 = _load("m51", "51_station_timing_and_power.py")   # both the sample and the gradient function come from script 51

TIME = "年份"
PAIRS = [("PM2.5", "PM2.5_年均", "ln_PM25"), ("SO2", "SO2_年均", "ln_SO2")]


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("59_threshold_per_pollutant_check")
    d = m51.load()
    ref = pd.read_excel(export_utils.OUTPUT_DIR / "51_梯度路径与可检测性.xlsx", sheet_name="02_逐年梯度路径")
    ref = ref[ref["样本"] == "全部"].set_index(["序列", "年份"])["城内梯度"]
    rows = []
    for yr, g in d.groupby(TIME):
        b_no2, _, n_no2 = m51.gradient(g, "ln_NO2")
        row: dict[str, object] = {"年份": int(yr), "NO2梯度": round(b_no2, 4), "NO2_N": n_no2}
        for tag, src, own in PAIRS:
            g = g.assign(**{f"{own}_套用NO2门槛": np.log(g[src].where(g[src] > 0))})
            b_own, _, n_own = m51.gradient(g, own)
            b_alt, _, n_alt = m51.gradient(g, f"{own}_套用NO2门槛")
            # the own-threshold column must reproduce the script-51 workbook
            if round(b_own, 4) != float(ref[(tag, int(yr))]):
                raise SystemExit(f"{yr} 年 {tag} 梯度 {b_own:.4f} 与 51 号 {ref[(tag, int(yr))]} 不符。不输出任何数。")
            row |= {f"{tag}梯度_自身门槛": round(b_own, 4), f"{tag}_N_自身门槛": n_own,
                    f"{tag}梯度_套用NO2门槛": round(b_alt, 4), f"{tag}_N_套用NO2门槛": n_alt}
        rows.append(row)
        logger.info("%s", row)
    out = export_utils.write_excel_workbook("59_分污染物门槛对照", [("逐年对照", pd.DataFrame(rows))])
    logger.info("已写出 %s；日志 %s", out, log_path)


if __name__ == "__main__":
    main()
