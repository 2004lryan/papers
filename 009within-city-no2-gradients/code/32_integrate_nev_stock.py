"""
32_integrate_nev_stock.py: join the provincial new energy vehicle (NEV) stock on the Ministry of Public Security
registration basis into the research panel

Data sources:
    data/raw/公安部口径-新能源汽车保有量-2017_2023/
        (NEV stock, Ministry of Public Security registration basis, 2017–2023)
    ├─ 17-23年新能源汽车保有量.xlsx (NEV stock, 2017–23)
    │      province × year long table, 2017–2023, unit "vehicles" (primary source)
    ├─ 2017…2022各省市新能源汽车保有量（公安部口径）.xlsx
    │      (NEV stock by province and city, Ministry of Public Security registration basis)
    │      yearly city tables, used for cross-checking
    ├─ 23省.jpg / 23市1-3.jpg (2023 provinces / 2023 cities, parts 1-3)
    │      2023 screenshots at province and city level (the provincial ones already agree with the xlsx entry by
    │      entry)
    └─ SHA256SUMS.txt
           hashes of all files (the .jpg files and this list are not read by this script)
    data/省级研究面板-2015_2024-v2.csv (provincial research panel, v2)
        the panel that the NEV columns are joined into; built by earlier preprocessing that is not part of this release

Source credibility checks (recomputed on every run of this script; the results are logged and, if all three pass,
written to outputs/ as evidence for Section 2.4 and Table 1 of the main text):
    ① 31-province total of each year vs the national NEV stock published by the Ministry of Public Security
       — measured: the deviation is ≤ 1% in all 7 years
    ② yearly city tables summed by province vs the provincial long table
       — measured: all 31 provinces agree in all six years 2017–2022 (difference < 0.5%)
    ③ non-decreasing year by year within each province — measured: not a single violation
    Only when all three pass may the data be written into the panel. If any one of them fails, the script exits at once.

Derived variables (left empty, not imputed, when the denominator is missing or 0):
    新能源汽车保有量_万辆 (NEV stock, 10⁴ vehicles)
        = vehicles / 1e4
    新能源汽车渗透率 (NEV penetration rate)
        = 新能源汽车保有量_万辆 / 民用汽车拥有量 (civilian vehicle stock, 10⁴ vehicles)
    千人新能源汽车保有量_辆 (NEV stock per 1000 residents, vehicles)
        = vehicles / (年末常住人口 (year-end resident population, 10⁴ persons) × 1e4) × 1000
    新能源汽车保有量_对数 (log of NEV stock)
        = ln(vehicles)
    新能源汽车年增量_万辆 (annual NEV increment, 10⁴ vehicles)
        = current year − previous year (empty for 2017)

Usage:
    python code/32_integrate_nev_stock.py

Output files:
    data/省级研究面板-2015_2024-v3.csv / .xlsx (provincial research panel)
        — v2 + five NEV columns (NEV is empty for 2015–2016 and 2024)
    data/省级新能源汽车保有量-公安部口径-2017_2023.csv
        (provincial NEV stock, Ministry of Public Security registration basis)
    outputs/32_NEV来源核验.xlsx (NEV source verification)
        — sheets 01_核验①全国合计, 02_核验②城市汇总 and 03_核验③单调性 (the three verification tables) and
          04_逐省一览_万辆 (province-by-province overview)
    logs/32_integrate_nev_stock.log
"""

from __future__ import annotations

import glob
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

RAW_DIR = export_utils.DATA_DIR / "raw" / "公安部口径-新能源汽车保有量-2017_2023"
PANEL_V2 = export_utils.DATA_DIR / "省级研究面板-2015_2024-v2.csv"

# National NEV stock published each year by the Ministry of Public Security (10⁴ vehicles, year end). The 2022 value
# was verified online; the other years are taken from the ministry's annual motor-vehicle statistics bulletins
# (released each January). 2024 is not listed because the provincial long table ends in 2023.
MPS_NATIONAL_10K = {2017: 153, 2018: 261, 2019: 381, 2020: 492, 2021: 784, 2022: 1310, 2023: 2041}
TOLERANCE = 0.015  # tolerance for the deviation of the totals: 1.5%

PROVINCE_FULL = {
    "北京": "北京市", "天津": "天津市", "河北": "河北省", "山西": "山西省", "内蒙古": "内蒙古自治区",
    "辽宁": "辽宁省", "吉林": "吉林省", "黑龙江": "黑龙江省", "上海": "上海市", "江苏": "江苏省",
    "浙江": "浙江省", "安徽": "安徽省", "福建": "福建省", "江西": "江西省", "山东": "山东省",
    "河南": "河南省", "湖北": "湖北省", "湖南": "湖南省", "广东": "广东省", "广西": "广西壮族自治区",
    "海南": "海南省", "重庆": "重庆市", "四川": "四川省", "贵州": "贵州省", "云南": "云南省",
    "西藏": "西藏自治区", "陕西": "陕西省", "甘肃": "甘肃省", "青海": "青海省", "宁夏": "宁夏回族自治区",
    "新疆": "新疆维吾尔自治区",
}


def short_name(name: str) -> str:
    return re.sub(r"(省|市|壮族自治区|回族自治区|维吾尔自治区|自治区)$", "", str(name).strip())


def load_province_long() -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / "17-23年新能源汽车保有量.xlsx")
    df.columns = ["省简称", "行政区划代码", "年份", "新能源汽车保有量_辆"]
    df["省级行政区"] = df["省简称"].map(PROVINCE_FULL)
    assert df["省级行政区"].notna().all(), f"未映射的省份：{df[df['省级行政区'].isna()]['省简称'].unique()}"
    return df


def load_city_year(year: int) -> pd.DataFrame:
    files = glob.glob(str(RAW_DIR / f"{year}各省市*.xlsx"))
    assert files, f"缺 {year} 年城市表"
    xl = pd.ExcelFile(files[0])
    sheet = next(s for s in xl.sheet_names if "341" in s or "工作表" in s)
    d = xl.parse(sheet).iloc[:, :4]
    d.columns = ["a", "省", "市", "保有量"]
    d = d[pd.to_numeric(d["保有量"], errors="coerce").notna()].copy()
    d["保有量"] = d["保有量"].astype(float)
    d["省简称"] = d["省"].map(short_name)
    return d


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("32_integrate_nev_stock")
    prov = load_province_long()
    piv = prov.pivot(index="省简称", columns="年份", values="新能源汽车保有量_辆")
    logger.info("省级长表：%s 省 × %s 年，缺失单元 %s", piv.shape[0], piv.shape[1], int(piv.isna().sum().sum()))

    # ① Check of the national totals
    rows = []
    for y, nat in MPS_NATIONAL_10K.items():
        s = piv[y].sum() / 1e4
        rows.append({"年份": y, "31省合计_万辆": round(s, 1), "公安部全国_万辆": nat,
                     "偏差%": round(100 * (s / nat - 1), 2)})
    chk1 = pd.DataFrame(rows)
    logger.info("核验①（全国合计）：\n%s", chk1.to_string(index=False))
    if (chk1["偏差%"].abs() > 100 * TOLERANCE).any():
        raise SystemExit("✗ 核验①失败：有年份合计偏差超过容忍值")

    # ② Check of the city tables summed by province
    rows = []
    for y in range(2017, 2023):
        c = load_city_year(y).groupby("省简称")["保有量"].sum()
        m = pd.DataFrame({"城市汇总": c, "省级表": piv[y]}).dropna()
        m["差%"] = 100 * (m["城市汇总"] / m["省级表"] - 1)
        rows.append({"年份": y, "匹配省份数": len(m), "最大绝对差%": round(m["差%"].abs().max(), 3),
                     "差超0.5%省份数": int((m["差%"].abs() > 0.5).sum())})
    chk2 = pd.DataFrame(rows)
    logger.info("核验②（城市汇总）：\n%s", chk2.to_string(index=False))
    if (chk2["匹配省份数"] < 31).any() or (chk2["差超0.5%省份数"] > 0).any():
        raise SystemExit("✗ 核验②失败：城市表与省级表不一致")

    # ③ Monotonicity check
    viol = []
    for p, r in piv.sort_index(axis=1).iterrows():
        v = r.values
        viol += [(p, int(piv.columns[i]), v[i - 1], v[i]) for i in range(1, len(v)) if v[i] < v[i - 1]]
    chk3 = pd.DataFrame(viol, columns=["省", "年份", "上年", "当年"]) if viol else pd.DataFrame(
        [{"结果": "31 省 2017–2023 逐年单调不减，无违反"}])
    logger.info("核验③（单调性）：%s", "无违反" if not viol else f"{len(viol)} 处违反")
    if viol:
        raise SystemExit("✗ 核验③失败：存在保有量逐年下降的省-年")

    # —— Join into the panel ——
    panel = pd.read_csv(PANEL_V2)
    nev = prov[["省级行政区", "年份", "新能源汽车保有量_辆"]].copy()
    nev["新能源汽车保有量_万辆"] = nev["新能源汽车保有量_辆"] / 1e4
    nev = nev.sort_values(["省级行政区", "年份"])
    nev["新能源汽车年增量_万辆"] = nev.groupby("省级行政区")["新能源汽车保有量_万辆"].diff()

    out = panel.merge(nev.drop(columns="新能源汽车保有量_辆"), on=["省级行政区", "年份"], how="left")
    assert len(out) == len(panel), "合并后行数变化"

    car = out["民用汽车拥有量"].where(out["民用汽车拥有量"] > 0)
    pop = out["年末常住人口"].where(out["年末常住人口"] > 0)
    out["新能源汽车渗透率"] = out["新能源汽车保有量_万辆"] / car
    out["千人新能源汽车保有量_辆"] = out["新能源汽车保有量_万辆"] * 1e4 / (pop * 1e4) * 1000
    out["新能源汽车保有量_对数"] = np.log(out["新能源汽车保有量_万辆"] * 1e4)

    v3_csv = export_utils.DATA_DIR / "省级研究面板-2015_2024-v3.csv"
    v3_xlsx = export_utils.DATA_DIR / "省级研究面板-2015_2024-v3.xlsx"
    out.to_csv(v3_csv, index=False, encoding="utf-8-sig")
    out.to_excel(v3_xlsx, index=False)
    nev_csv = export_utils.DATA_DIR / "省级新能源汽车保有量-公安部口径-2017_2023.csv"
    nev.to_csv(nev_csv, index=False, encoding="utf-8-sig")

    overview = (piv / 1e4).round(2).sort_values(2023, ascending=False).reset_index()
    export_utils.write_excel_workbook("32_NEV来源核验", [
        ("核验①全国合计", chk1), ("核验②城市汇总", chk2), ("核验③单调性", chk3), ("逐省一览_万辆", overview)])

    n_nev = int(out["新能源汽车保有量_万辆"].notna().sum())
    logger.info("面板 v3：%s 行 × %s 列；NEV 非空 %s 行（2017–2023）", *out.shape, n_nev)
    pen = out.dropna(subset=["新能源汽车渗透率"])
    logger.info("渗透率区间：%.4f（%s %s）～ %.4f（%s %s）",
                pen["新能源汽车渗透率"].min(), pen.loc[pen["新能源汽车渗透率"].idxmin(), "省级行政区"],
                int(pen.loc[pen["新能源汽车渗透率"].idxmin(), "年份"]),
                pen["新能源汽车渗透率"].max(), pen.loc[pen["新能源汽车渗透率"].idxmax(), "省级行政区"],
                int(pen.loc[pen["新能源汽车渗透率"].idxmax(), "年份"]))

    print("✓ 三项来源核验全部通过，NEV 已接入面板")
    print(chk1.to_string(index=False))
    print(f"  面板 v3: {out.shape[0]} 行 × {out.shape[1]} 列，NEV 非空 {n_nev} 行")
    print(f"  {v3_csv}")
    print(f"  {export_utils.OUTPUT_DIR / '32_NEV来源核验.xlsx'}")
    print(f"  日志: {log_path}")


if __name__ == "__main__":
    main()
