"""
50_add_o3_annual_mean.py: add the O₃ annual mean and Ox (odd oxygen) to the station annual panel, for a titration
consistency test

Why it is needed:
    Trial runs of script 48 show that, in concentration **levels**, the interaction coefficient of SO₂ has the
    same sign as that of NO₂ and is stronger. This means that the confounder "city cores improve as development
    rises" is not removed by the city-by-year fixed effects, so NO₂ levels alone cannot be attributed to traffic.

    Atmospheric chemistry offers a criterion that SO₂ cannot mimic: **NO titration**. NO concentrations are high
    in city cores, and O₃ is consumed there by titration with NO (NO + O₃ → NO₂ + O₂), so O₃ in the core is lower
    than in the suburbs. Once traffic NOx really falls, titration weakens and **O₃ in the core rises**. Cutting
    industrial SO₂ does not produce this effect.

    A further step separates "mere redistribution by titration" from "genuinely less photochemical production":
        Ox ≡ O₃ + NO₂ (odd oxygen, molar mixing ratio) is conserved in the titration reaction.
        If the interaction lowers NO₂ and raises O₃ while **Ox is unchanged** → pure redistribution by titration,
        and NOx emissions really fell;
        if Ox falls as well → photochemical production itself has decreased.
    Both support "NOx emissions fell", and neither can be produced by SO₂.

    Script 46 only computes the 90th percentile of the O₃ daily maximum 8h (the metric of the assessment
    standard), which cannot give Ox. This script adds the O₃ **annual mean**, combines it into Ox by molar mixing
    ratio, and writes the result directly back to the station annual panel.

Unit conversion (25 °C, 101.325 kPa):
    ppb = µg/m³ × 24.45 / molecular weight; O₃ 48.00 g/mol, NO₂ 46.01 g/mol

Usage:
    python code/50_add_o3_annual_mean.py

Output: updates data/站点年面板-2015_2024.csv (station annual panel) in place, adding the columns O3_年均
     (O₃ annual mean), O3_年均_有效日 (valid days of the O₃ annual mean), Ox_年均_ppb (Ox annual mean, ppb),
     NO2_年均_ppb and O3_年均_ppb (NO₂ and O₃ annual means, ppb); logs/50_add_o3_annual_mean.log
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

YEARS = range(2015, 2025)
MIN_HOURS_DAY = 20
MOLAR_VOL = 24.45          # L/mol, 25 °C 101.325 kPa
MW = {"O3": 48.00, "NO2": 46.01}


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("50_add_o3_annual_mean")
    panel_path = export_utils.DATA_DIR / "站点年面板-2015_2024.csv"
    panel = pd.read_csv(panel_path)
    panel.columns = [c.lstrip("﻿") for c in panel.columns]
    panel["站点编号"] = panel["站点编号"].astype(str)
    logger.info("读入面板 %s 行 × %s 列", *panel.shape)

    rows = []
    for year in YEARS:
        src = export_utils.DATA_DIR / f"站点小时空气质量-{year}.parquet"
        if not src.exists():
            logger.warning("  %s 缺 parquet，跳过", year)
            continue
        df = pd.read_parquet(src, columns=["站点编号", "日期", "小时", "指标", "值"])
        o3 = df[df["指标"].astype(str) == "O3"]
        g = o3.groupby([o3["站点编号"].astype(str), "日期"])["值"].agg(["mean", "count"])
        g = g[g["count"] >= MIN_HOURS_DAY].reset_index()
        y = (g.groupby("站点编号")["mean"].agg(O3_年均="mean", O3_年均_有效日="count")
             .reset_index())
        y["年份"] = year
        rows.append(y)
        logger.info("  %s：O₃ 有效站-日 %s，站点 %s，年均有效日中位 %.0f",
                    year, len(g), len(y), y["O3_年均_有效日"].median())
        del df, o3, g, y

    if not rows:
        raise SystemExit("✗ 未读到任何 parquet")
    o3y = pd.concat(rows, ignore_index=True)
    o3y["站点编号"] = o3y["站点编号"].astype(str)

    panel = panel.drop(columns=[c for c in ["O3_年均", "O3_年均_有效日", "O3_年均_ppb",
                                            "NO2_年均_ppb", "Ox_年均_ppb"] if c in panel.columns])
    panel = panel.merge(o3y, on=["站点编号", "年份"], how="left")

    # Molar mixing ratios and Ox
    panel["O3_年均_ppb"] = panel["O3_年均"] * MOLAR_VOL / MW["O3"]
    panel["NO2_年均_ppb"] = panel["NO2_年均"] * MOLAR_VOL / MW["NO2"]
    panel["Ox_年均_ppb"] = panel["O3_年均_ppb"] + panel["NO2_年均_ppb"]

    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    hit = panel["O3_年均"].notna().mean()
    logger.info("已写回面板：O₃ 年均非空 %.1f%%，Ox 非空 %.1f%%",
                100 * hit, 100 * panel["Ox_年均_ppb"].notna().mean())

    print(f"✓ 面板已补 O₃ 年均与 Ox → {panel_path}")
    print(f"  O₃ 年均非空 {int(panel['O3_年均'].notna().sum())} / {len(panel)}（{100*hit:.1f}%）")
    print(f"  O₃ 年均 中位 {panel['O3_年均'].median():.1f} µg/m³；"
          f"Ox 中位 {panel['Ox_年均_ppb'].median():.1f} ppb")
    print(f"  日志: {log_path}")


if __name__ == "__main__":
    main()
