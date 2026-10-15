"""
54_cluster_size_summary.py: size distribution of the city clusters in the regression sample

Script 48 clusters its standard errors by city and re-checks the p values with a wild cluster bootstrap. The main text
has to explain why this re-check is needed: cluster-robust inference over-rejects when cluster sizes are very unequal
(MacKinnon & Webb 2017), and the number of national monitoring stations per city in this panel varies widely. That
"varies widely" must be backed by a number that can be checked; this script provides that number.

The sample is built step by step as in script 48 (NO₂ valid days ≥274 → city-years with ≥2 stations in the city →
non-missing regression variables), and the script asserts that the sample size equals that of the baseline regression
of script 48; if they do not match, it exits with an error and outputs no distribution.

Usage:
    python code/54_cluster_size_summary.py

Output files:
    outputs/54_回归样本簇规模.xlsx (cluster sizes in the regression sample)  — sheets 01_01_汇总 (summary) /
        02_02_逐城市 (by city)
    logs/54_cluster_size_summary.log
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
if _SPEC is None or _SPEC.loader is None:
    raise SystemExit("01_export_utils.py 无法加载")
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

STA, CITY, TIME = "站点编号", "所属城市", "年份"
MIN_DAYS = 274
# Sample size of the baseline regression of script 48 (Table 2, column 1): station-years, stations, cities
EXPECTED = (7325, 1251, 278)


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("54_cluster_size_summary")
    p = pd.read_csv(export_utils.DATA_DIR / "站点年面板-2015_2024.csv")
    p.columns = [c.lstrip("﻿") for c in p.columns]

    v = p["NO2_年均"].where((p["NO2_年均"] > 0) & (p["NO2_有效日"] >= MIN_DAYS))
    p["ln_NO2"] = np.log(v)
    p["城市年"] = p[CITY].astype(str) + "_" + p[TIME].astype(str)
    valid = p.dropna(subset=["ln_NO2"])
    p["城内站数"] = p["城市年"].map(valid.groupby("城市年")[STA].nunique())
    d = p[p["城内站数"] >= 2].copy()
    d["lnNEV"] = np.log(d["NEV保有量_辆"].where(d["NEV保有量_辆"] > 0))
    d["NEVxExpo"] = d["lnNEV"] * d["建成区占比_500m"]
    d = d.dropna(subset=["ln_NO2", "NEVxExpo"])

    got = (len(d), d[STA].nunique(), d[CITY].nunique())
    logger.info("回归样本：站点年 %s，站点 %s，城市 %s（48 号基准为 %s）", *got, EXPECTED)
    if got != EXPECTED:
        raise SystemExit(f"样本与 48 号基准回归不一致：得到 {got}，应为 {EXPECTED}。不输出分布。")

    per_city = (d.groupby(CITY).agg(站点数=(STA, "nunique"), 站点年数=(STA, "size"))
                .sort_values("站点年数", ascending=False).reset_index())
    rows = []
    for col in ("站点数", "站点年数"):
        s = per_city[col]
        rows.append({"量": col, "簇数": len(s), "最小": int(s.min()), "P10": float(s.quantile(0.10)),
                     "P25": float(s.quantile(0.25)), "中位": float(s.median()), "P75": float(s.quantile(0.75)),
                     "P90": float(s.quantile(0.90)), "最大": int(s.max()),
                     "最大/最小": round(float(s.max() / s.min()), 1),
                     "规模最大的 10 簇占全部观测的比例": round(float(s.nlargest(10).sum() / s.sum()), 4)})
        logger.info("  每簇%s：最小 %s，中位 %s，最大 %s；最大的 10 簇占 %.1f%%", col, int(s.min()),
                    s.median(), int(s.max()), 100 * s.nlargest(10).sum() / s.sum())
    out = export_utils.write_excel_workbook("54_回归样本簇规模",
                                            [("01_汇总", pd.DataFrame(rows)), ("02_逐城市", per_city)])
    logger.info("已写出 %s；日志 %s", out, log_path)


if __name__ == "__main__":
    main()
