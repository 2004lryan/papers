"""
37_build_city_panel.py: city-level panel (2017–2023) — city NEV stock on the Ministry of Public Security registration
basis × city annual air quality × city annual meteorology

Why a city panel:
    In a provincial panel (an earlier stage of this project, not part of this release), the negative "penetration ×
    thermal power share" interaction disappears once Blue Sky Protection Campaign priority-region-by-year fixed
    effects are added, which shows that provincial identification is contaminated by regional haze-control trends.
    The city panel uses **city fixed effects + province-by-year fixed effects**, absorbing all province-year policies
    and trends as a whole, and identifies only from differences in electrification between cities within the same
    province. N rises from 217 province-years to about 2,300 city-years.

Inputs:
    data/raw/公安部口径-新能源汽车保有量-2017_2023/2017…2022各省市*.xlsx
        (folder: NEV stock on the Ministry of Public Security registration basis; files: by province and city)
        — city-level NEV (including the "Other" rows)
    2023各省市新能源汽车保有量（截图识别）.csv in the same folder
        (2023 NEV stock by province and city, transcribed from screenshots)
        — the 2023 values, transcribed from screenshots by script 40 and checked against the provincial totals
    outputs/26_省级空气质量构建过程.xlsx (provincial air-quality build steps), sheet "03_城市年评价值"
        (city annual assessment values) — city annual PM2.5/PM10/O3/NO2/SO2/CO, rows with `纳入省级聚合` true only
        (built by earlier-stage preprocessing that is not part of this release; see data/DATA.md)
    outputs/29_气象构建过程.xlsx (meteorology build steps), sheet "02_城市年值" (city annual values)
        — city annual meteorology (intermediate table of script 29)
    (the province of each city, `省级行政区`, comes from the air-quality table above)

City name alignment:
    The NEV tables use short names (Linxia, Ili, Hinggan), while the standard names in the air-quality data carry
    suffixes (Linxia Zhou, Ili Kazakh Zhou, Hinggan Meng).
    Both sides are normalised by the same rules (norm(): drop a trailing city / prefecture / autonomous prefecture /
    autonomous county / forest district / urban district / new district suffix, then ethnic-group words, then a
    trailing prefecture (zhou) or league (meng) suffix), plus an explicit alias table for typos and special cases.
    Cities that cannot be matched are **listed one by one** in the build report instead of being dropped silently.

The "Other" rows:
    The part of a province's stock that is not assigned to a specific city (rows labelled "Other" or "directly
    administered"). The city panel does not use it, but records its share by province-year in the column
    `本省当年未分配占比` and the sheet 04_未分配占比, so that province-years with a high Other share can be excluded in
    robustness checks.

Usage:
    python code/37_build_city_panel.py

Output files:
    data/城市面板-2017_2023.csv (city panel)
    outputs/37_城市面板构建.xlsx (city panel construction)
        — sheets 01_城市面板 (panel), 02_按年覆盖 (coverage by year), 03_未匹配城市 (unmatched cities),
          04_未分配占比 (Other share by province-year)
    logs/37_build_city_panel.log
"""

from __future__ import annotations

import glob
import importlib.util
import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

RAW = export_utils.DATA_DIR / "raw" / "公安部口径-新能源汽车保有量-2017_2023"
YEARS = range(2017, 2024)

ETHNIC = tuple(sorted(
    ("哈萨克", "柯尔克孜", "蒙古族", "蒙古", "藏族羌族", "藏族", "回族", "彝族", "白族", "傣族景颇族", "傣族",
     "傈僳族", "哈尼族彝族", "壮族苗族", "苗族侗族", "布依族苗族", "土家族苗族",
     "朝鲜族", "羌族", "景颇族", "侗族", "苗族"),
    key=len, reverse=True))   # longest words first, so that "Yi" is not replaced before "Hani and Yi"
# Only typos and genuine special cases; differences in the autonomous prefecture / league / prefecture suffixes are
# handled uniformly by norm()
ALIAS = {"巴摩淖尔": "巴彦淖尔", "超关": "韶关", "静州": "滁州"}


def norm(name: str) -> str:
    s = str(name).strip().replace(" ", "")
    s = re.sub(r"(市|地区|自治州|自治县|林区|市区|新区)$", "", s)
    for e in ETHNIC:
        s = s.replace(e, "")
    s = re.sub(r"(州|盟)$", "", s)
    return s


def load_nev_city(year: int) -> pd.DataFrame:
    if year == 2023:   # table transcribed from screenshots (script 40); matches every provincial total exactly
        d = pd.read_csv(RAW / "2023各省市新能源汽车保有量（截图识别）.csv")[["序号", "省份", "城市", "保有量"]]
        d.columns = ["a", "省", "市", "保有量"]
    else:
        f = glob.glob(str(RAW / f"{year}各省市*.xlsx"))[0]
        xl = pd.ExcelFile(f)
        sheet = next(s for s in xl.sheet_names if "341" in s or "工作表" in s)
        d = xl.parse(sheet).iloc[:, :4]
        d.columns = ["a", "省", "市", "保有量"]
    d = d[pd.to_numeric(d["保有量"], errors="coerce").notna()].copy()
    d["保有量"] = d["保有量"].astype(float)
    d["年份"] = year
    d["省简称"] = d["省"].astype(str).str.replace(r"(省|市|壮族自治区|回族自治区|维吾尔自治区|自治区)$", "", regex=True)
    d["市"] = d["市"].astype(str).str.strip()
    d["未分配"] = d["市"].str.contains("其他|直辖")
    return d[["年份", "省简称", "市", "保有量", "未分配"]]


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("37_build_city_panel")

    # ── City annual air-quality values (intermediate table from earlier-stage preprocessing, see data/DATA.md) ──
    aq = pd.read_excel(export_utils.OUTPUT_DIR / "26_省级空气质量构建过程.xlsx", sheet_name="03_城市年评价值")
    aq = aq[aq["纳入省级聚合"] == True].copy()  # noqa: E712
    logger.info("空气质量城市年值 %s 行，城市 %s 个", len(aq), aq["城市"].nunique())

    # ── City annual meteorology values (intermediate table of script 29) ──
    met = pd.read_excel(export_utils.OUTPUT_DIR / "29_气象构建过程.xlsx", sheet_name="02_城市年值")
    logger.info("气象城市年值 %s 行，城市 %s 个", len(met), met["城市"].nunique())

    # ── Standard-name index: normalised name → standard name (one national index; duplicate keys are only logged) ──
    std = aq[["城市", "省级行政区"]].drop_duplicates()
    std["key"] = std["城市"].map(norm)
    dup = std[std.duplicated("key", keep=False)]
    if len(dup):
        logger.warning("标准名归一化后重名：%s", dup.to_dict("records"))
    key2std = dict(zip(std["key"], std["城市"], strict=True))
    std_prov = dict(zip(std["城市"], std["省级行政区"], strict=True))

    # ── NEV city table ──
    nev_all = pd.concat([load_nev_city(y) for y in YEARS], ignore_index=True)
    other = (nev_all.groupby(["年份", "省简称"])
             .apply(lambda g: pd.Series({"省内合计_辆": g["保有量"].sum(),
                                         "未分配_辆": g.loc[g["未分配"], "保有量"].sum()}))
             .reset_index())
    other["未分配占比"] = (other["未分配_辆"] / other["省内合计_辆"]).round(4)

    nev = nev_all[~nev_all["未分配"]].copy()
    nev["raw"] = nev["市"].map(lambda s: ALIAS.get(s, s))
    nev["key"] = nev["raw"].map(lambda s: norm(cast(str, ALIAS.get(s, s))))
    nev["标准名"] = nev["key"].map(key2std)
    # Second attempt: the alias itself may already be a standard name
    miss = nev["标准名"].isna()
    nev.loc[miss, "标准名"] = nev.loc[miss, "raw"].where(nev.loc[miss, "raw"].isin(key2std.values()))
    matched = nev[nev["标准名"].notna()].copy()
    unmatched = (nev[nev["标准名"].isna()].groupby(["省简称", "市"])
                 .agg(年份数=("年份", "nunique"), 保有量均值=("保有量", "mean")).reset_index()
                 .sort_values("保有量均值", ascending=False))
    logger.info("NEV 具名城市-年 %s 条，匹配到标准名 %s 条（%.1f%%）；未匹配城市 %s 个",
                len(nev), len(matched), 100 * len(matched) / len(nev), len(unmatched))

    # If one standard name comes from several rows in the same year (very rare, e.g. an alias collision), sum them
    matched = (matched.groupby(["标准名", "年份"], as_index=False)["保有量"].sum()
               .rename(columns={"标准名": "城市", "保有量": "NEV保有量_辆"}))
    matched["省级行政区"] = matched["城市"].map(std_prov)

    # ── Merge ──
    aq_cols = ["城市", "省级行政区", "年份", "PM25年均", "PM25有效天", "PM25年均_有效年", "PM10年均", "PM10有效天",
               "PM10年均_有效年", "O3_8h_90分位", "O3有效天", "O3_8h_90分位_有效年"]
    aq_cols += [c for c in aq.columns if c.startswith(("NO2", "SO2", "CO"))]
    panel = matched.merge(aq[aq_cols], on=["城市", "省级行政区", "年份"], how="left")
    met_cols = ["城市", "省级行政区", "年份", "年平均气温_摄氏度", "年降水量_毫米", "年平均风速_米每秒",
                "年平均相对湿度_百分比", "年平均气压_百帕", "有效天数"]
    panel = panel.merge(met[met_cols].rename(columns={"有效天数": "气象有效天数"}),
                        on=["城市", "省级行政区", "年份"], how="left")
    prov_short = panel["省级行政区"].str.replace(r"(省|市|壮族自治区|回族自治区|维吾尔自治区|自治区)$", "", regex=True)
    panel = panel.merge(other[["年份", "省简称", "未分配占比"]].rename(columns={"未分配占比": "本省当年未分配占比"}),
                        left_on=["年份", prov_short], right_on=["年份", "省简称"], how="left").drop(columns=["省简称"])
    panel["ln_NEV"] = np.log(panel["NEV保有量_辆"].where(panel["NEV保有量_辆"] > 0))
    panel = panel.sort_values(["省级行政区", "城市", "年份"]).reset_index(drop=True)

    cover = (panel.groupby("年份")
             .agg(城市数=("城市", "nunique"), NEV非空=("NEV保有量_辆", "count"),
                  PM25有效=("PM25年均_有效年", "sum"), 气象非空=("年平均气温_摄氏度", "count"),
                  未分配占比_均值=("本省当年未分配占比", "mean"))
             .reset_index())
    balanced = panel.groupby("城市")["年份"].nunique()
    n_full = int((balanced == len(YEARS)).sum())

    out_csv = export_utils.DATA_DIR / "城市面板-2017_2023.csv"
    panel.to_csv(out_csv, index=False, encoding="utf-8-sig")
    export_utils.write_excel_workbook("37_城市面板构建", [
        ("城市面板", panel), ("按年覆盖", cover), ("未匹配城市", unmatched), ("未分配占比", other)])

    logger.info("城市面板 %s 行 × %s 列；城市 %s 个，其中各年齐全 %s 个", *panel.shape, panel["城市"].nunique(), n_full)
    print(f"✓ 城市面板 {panel.shape[0]} 行 × {panel.shape[1]} 列；"
          f"城市 {panel['城市'].nunique()} 个，各年齐全 {n_full} 个")
    print(cover.round(3).to_string(index=False))
    print(f"\n未匹配城市 {len(unmatched)} 个（按保有量均值降序，前 15）：")
    print(unmatched.head(15).to_string(index=False))
    print(f"\n  数据: {out_csv}\n  日志: {log_path}")


if __name__ == "__main__":
    main()
