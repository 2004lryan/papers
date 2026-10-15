"""
60_priority_region_city_check.py: priority regions of the Blue Sky Protection Campaign assigned by the city list vs
assigned by province

Why this is needed:
    The three regions defined in Section I(3), "Scope of priority regions", of the Three-Year Action Plan for Winning
    the Blue Sky Defence War (State Council document Guofa [2018] No. 22) contain only the listed cities in the five
    provinces of Hebei, Shanxi, Shandong, Henan and Shaanxi. Script 51 assigns priority regions by this city list;
    this script instead assigns them by province (the 11 provinces and municipalities that the three regions touch,
    KEY_PROVINCES_11, counted as priority regions as a whole),
    re-estimates the within-city gradient of the two groups year by year and compares them,
    to show the effect of the assignment rule itself on the regional comparison:
      ① assignment by province can only be used for 2017–2023, the years in which the station panel carries the
        province (`省级行政区`, joined by script 46 from the city panel);
      ② switching to the city list on the same station-years shows the effect of the rule alone;
      ③ the city list, with the Yangtze River Delta provinces filled in for all years from each city's 2017–2023
        records, covers all station-years of 2015–2024 (i.e. the rule of script 51).
    The gradient function and the sample are both those of script 51; the panel is only read, and nothing is written
    back to data/. Gate: the within-city gradients of the city-list rows on all station-years must equal sheet
    02_逐年梯度路径 of outputs/51_梯度路径与可检测性.xlsx exactly (run script 51 first); otherwise the script exits
    without output.

Usage:
    python code/60_priority_region_city_check.py

Output files:
    outputs/60_重点区域城市口径核验.xlsx (priority-region city-list check)
    logs/60_priority_region_city_check.log
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
m51 = _load("m51", "51_station_timing_and_power.py")   # sample, city list and gradient function all from script 51

CITY, TIME, PROV = m51.CITY, m51.TIME, "省级行政区"
# Assignment by province: the 11 provinces (municipalities) involved in the three priority regions count as priority
# regions as a whole
KEY_PROVINCES_11 = ["北京市", "天津市", "河北省", "山西省", "山东省", "河南省",
                    "陕西省", "上海市", "江苏省", "浙江省", "安徽省"]
SEGMENTS = [(2015, 2017), (2017, 2020), (2020, 2023), (2023, 2024)]


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("60_priority_region_city_check")
    d = m51.load()
    d["重点区域_按省"] = np.where(d[PROV].isna(), np.nan, d[PROV].isin(KEY_PROVINCES_11))

    schemes = [("按省（2017–2023）", "重点区域_按省", d[d[PROV].notna()]),
               ("城市名单·同一批站点年", "重点区域", d[d[PROV].notna()]),
               ("城市名单·全部站点年（51 号）", "重点区域", d)]
    rows = []
    for tag, ycol in m51.SERIES:
        for scheme, col, sub in schemes:
            for key, lab in [(1.0, "重点区域"), (0.0, "非重点区域")]:
                for yr, g in sub[sub[col] == key].groupby(TIME):
                    b, se, n = m51.gradient(g, ycol)
                    rows.append({"口径": scheme, "序列": tag, "样本": lab, "年份": int(yr),
                                 "城内梯度": round(b, 4), "SE": round(se, 4), "N": n,
                                 "城市数": g.dropna(subset=[ycol, m51.EXPO])[CITY].nunique()})
    path = pd.DataFrame(rows)

    # Gate: the city-list scheme on all station-years must reproduce the subgroup rows of the script 51 workbook cell
    # by cell
    ref = pd.read_excel(export_utils.OUTPUT_DIR / "51_梯度路径与可检测性.xlsx", sheet_name="02_逐年梯度路径")
    ref = ref[ref["样本"] != "全部"].set_index(["序列", "样本", "年份"])["城内梯度"].sort_index()
    mine = (path[path["口径"] == "城市名单·全部站点年（51 号）"]
            .set_index(["序列", "样本", "年份"])["城内梯度"].sort_index())
    if not mine.equals(ref):
        raise SystemExit("城市名单口径与 51 号工作簿不符。不输出任何数。")

    seg = []
    for (scheme, tag, lab), g in path.groupby(["口径", "序列", "样本"], sort=False):
        s = g.set_index("年份")["城内梯度"]
        row: dict[str, object] = {"口径": scheme, "序列": tag, "样本": lab}
        for a, b in SEGMENTS:
            if a in s.index and b in s.index:
                row[f"{a}→{b} 相对变化%"] = round(100 * (s[b] / s[a] - 1), 1)
        seg.append(row)
    seg_df = pd.DataFrame(seg)

    cities = (d.groupby(CITY)
              .agg(省=(PROV, "first"), 按省=("重点区域_按省", "max"), 按城市名单=("重点区域", "max"),
                   站点年数=(m51.STA, "size"))
              .reset_index())
    cities["两种口径不一致"] = (cities["按省"] == 1) & (cities["按城市名单"] == 0)
    moved = cities[cities["两种口径不一致"]]
    logger.info("按省划入重点区域、按城市名单不属于重点区域的城市 %s 个，站点年 %s：%s",
                len(moved), int(moved["站点年数"].sum()), "、".join(moved[CITY]))
    for r in seg_df.itertuples(index=False):
        logger.info("%s", dict(zip(seg_df.columns, r, strict=True)))

    out = export_utils.write_excel_workbook("60_重点区域城市口径核验", [
        ("分段变化", seg_df), ("逐年梯度", path),
        ("城市归属", cities.sort_values(["两种口径不一致", "省", CITY], ascending=[False, True, True]))])
    logger.info("已写出 %s；日志 %s", out, log_path)


if __name__ == "__main__":
    main()
