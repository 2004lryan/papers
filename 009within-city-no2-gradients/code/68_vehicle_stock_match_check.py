"""
68_vehicle_stock_match_check.py: station-years without a city vehicle stock, the cause, and re-estimates once restored

Why it is needed:
    Script 46 removes the administrative suffix (市, 地区, 自治州, 盟 and others) from the city name in the station
    register and then merges the vehicle stock of the city panel of script 37 on city name and year. The city panel
    writes autonomous prefectures, leagues and regions under their full name or under a short form ending in 州, so
    the shortened name finds no match; the station-years of these cities carry no vehicle stock and are absent from
    the regression sample of script 48. Section 2.3 of the main text and Section S1 of the Supplementary Material
    report this part of the gap between the regression sample and the descriptive sample.
    This script answers two questions:
    (1) within the 2017 to 2023 part of the descriptive sample, how many station-years, stations and cities carry no
        vehicle stock, and which case each city falls under;
    (2) once the names are matched as the city panel writes them, how much larger the sample is and whether the NO₂
        estimates of column 1 (equation 3) and column 3 (equation 4) of Table 2 of the main text change.

    The estimator is the fit function of script 48 (station and city-by-year fixed effects, standard errors clustered
    on city), and the sample is built line by line as in script 48. The script only reads data/; the restored stock
    is held in memory and the station panel is not rewritten.

Gate (if it fails no number is output):
    The two columns re-estimated under the current matching must equal the coefficients and sample sizes in the
    archived workbook of script 48 digit for digit.

Usage:
    python code/68_vehicle_stock_match_check.py

Output files:
    outputs/68_保有量匹配核对.xlsx (vehicle stock matching check)
    logs/68_vehicle_stock_match_check.log
"""

from __future__ import annotations

import importlib.util
import re
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
m48 = _load("m48", "48_run_station_panel.py")   # estimator and completeness threshold

STA, CITY, TIME, MAIN = m48.STA, m48.CITY, m48.TIME, m48.MAIN
STOCK, TREND = "NEV保有量_辆", "tx暴露"
SUFFIX = r"(市|地区|自治州|盟|自治县|林区)$"   # administrative suffixes removed by script 46
YEARS = (2017, 2023)                            # years covered by the vehicle stock series
SPECS = [("表2第1列 式(3)", [MAIN]), ("表2第3列 式(4)", [MAIN, TREND])]
SAME, SHORT, ABSENT, NOYEAR = "城市面板写全称", "城市面板写简称", "城市面板无此城市", "名称对上但该年无值"


def panel_name(city: str, names: set[str]) -> str | None:
    """Name under which the city panel writes a city of the station register: the shortened name, the full name and
    the short form ending in 州 are tried in turn; None if none is found."""
    short = re.sub(SUFFIX, "", city)
    if short in names:
        return short
    if city in names:
        return city
    if city.endswith("自治州"):
        cand = [n for n in names if n.endswith("州") and len(n) > 2 and city.startswith(n[:-1])]
        if len(cand) == 1:
            return cand[0]
        if len(cand) > 1:
            raise SystemExit(f"{city} 在城市面板里有多个候选写法：{cand}")
    return None


def sample(raw: pd.DataFrame, stock: pd.Series) -> pd.DataFrame:
    """Sample as in script 48: log annual mean NO₂ under the 274-day threshold, centred exposure, and only city-years
    with at least two qualifying stations."""
    p = raw.copy()
    p[STOCK] = stock.to_numpy()
    p["城市年"] = p[CITY].astype(str) + "_" + p[TIME].astype(str)
    v = p["NO2_年均"].where(p["NO2_年均"] > 0).where(p["NO2_有效日"] >= m48.MIN_DAYS_MAIN)
    p["ln_NO2"] = np.log(v)
    p["lnNEV"] = np.log(p[STOCK].where(p[STOCK] > 0))
    p["Expo"] = p["建成区占比_500m"] - p["建成区占比_500m"].mean()
    p[MAIN] = p["lnNEV"] * p["Expo"]
    p[TREND] = (p[TIME] - p[TIME].min()) * p["Expo"]
    n_sta = p.dropna(subset=["ln_NO2"]).groupby("城市年")[STA].nunique()
    return p[p["城市年"].map(n_sta) >= 2].copy()


def estimates(d0: pd.DataFrame, match: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, xs in SPECS:
        res, dd = m48.fit(d0, "ln_NO2", xs)
        r: dict[str, object] = {"匹配": match, "规格": label, "N": int(res.nobs), "站点数": int(dd[STA].nunique()),
                                "城市数": int(dd[CITY].nunique()), "β_交互": round(float(res.params[MAIN]), 5),
                                "β_交互_SE": round(float(res.std_errors[MAIN]), 5),
                                "β_交互_p": round(float(res.pvalues[MAIN]), 5)}
        if TREND in xs:
            r |= {"β_t×暴露": round(float(res.params[TREND]), 5),
                  "β_t×暴露_SE": round(float(res.std_errors[TREND]), 5),
                  "β_t×暴露_p": round(float(res.pvalues[TREND]), 5)}
        rows.append(r)
    return rows


def archived() -> list[tuple[int, float, float | None]]:
    """N, interaction and trend coefficient of the two columns in the archived workbook of script 48 (column 1 has no
    trend term)."""
    book = pd.ExcelFile(export_utils.table_output_path("48_站点面板回归", suffix="xlsx"))
    main = book.parse("01_主表")
    rob = book.parse("03_稳健性")
    a = main[main["规格"].str.startswith("M9c") & (main["结果变量"] == "NO2")].iloc[0]
    b = rob[rob["规格"].str.startswith("S16") & (rob["结果变量"] == "NO2")].iloc[0]
    return [(int(a["N"]), float(a["β_交互"]), None), (int(b["N"]), float(b["β_交互"]), float(b["β_t×暴露"]))]


def main() -> None:
    logger, log_path = export_utils.configure_file_logger("68_vehicle_stock_match_check")
    raw = pd.read_csv(export_utils.DATA_DIR / "站点年面板-2015_2024.csv")
    city = pd.read_csv(export_utils.DATA_DIR / "城市面板-2017_2023.csv")[["城市", TIME, STOCK]]
    names = set(city["城市"].astype(str))

    # current matching: the stock merged into the station panel by script 46; gate
    cur = sample(raw, raw[STOCK])
    est_cur = estimates(cur, "现行（46 号去后缀名）")
    for r, (n, beta, trend) in zip(est_cur, archived(), strict=True):
        same = r["N"] == n and r["β_交互"] == round(beta, 5) and (trend is None or r["β_t×暴露"] == round(trend, 5))
        if not same:
            raise SystemExit(f"闸未过：{r['规格']} 重估 {r} 与 48 号归档 N={n} β={beta} 趋势={trend} 不同")
    logger.info("闸通过：现行匹配下两列与 48 号归档工作簿逐位相同")

    # restored: stock looked up under the name the city panel uses
    key = raw[CITY].map(lambda c: panel_name(str(c), names))
    looked = pd.DataFrame({"城市": key, TIME: raw[TIME]}).merge(city, on=["城市", TIME], how="left")[STOCK]
    fixed_stock = raw[STOCK].where(raw[STOCK].notna(), looked.to_numpy())
    fix = sample(raw, fixed_stock)
    est_fix = estimates(fix, "补回（城市面板里的写法）")

    # counts: the 2017 to 2023 part of the descriptive sample
    in_span = cur[TIME].between(*YEARS) & cur["ln_NO2"].notna()
    desc = cur[in_span]
    lost = desc[desc[STOCK].isna()]
    fix_desc = fix[fix[TIME].between(*YEARS) & fix["ln_NO2"].notna()]
    if len(fix_desc) != len(desc) or not (fix_desc.index == desc.index).all():
        raise SystemExit("补回前后描述性样本的行不一致")
    back = fix_desc.loc[lost.index, STOCK].notna()
    city_rows: list[dict[str, object]] = []
    for c, g in lost.groupby(CITY):
        name = panel_name(str(c), names)
        if name is None:
            kind = ABSENT
        elif name == re.sub(SUFFIX, "", str(c)):
            kind = NOYEAR
        else:
            kind = SAME if name == c else SHORT
        city_rows.append({"所属城市": c, "去后缀名": re.sub(SUFFIX, "", str(c)), "城市面板写法": name or "",
                          "情形": kind, "无保有量站点年份": len(g), "站点数": int(g[STA].nunique()),
                          "补回站点年份": int(back.loc[g.index].sum()),
                          "补回后仍无": int((~back.loc[g.index]).sum())})
    cities = pd.DataFrame(city_rows)
    kinds = cities["情形"].value_counts()
    restored = lost[back.to_numpy()]
    left = cities[cities["补回后仍无"] > 0]
    reg_cur, reg_fix = cur.dropna(subset=["ln_NO2", MAIN]), fix.dropna(subset=["ln_NO2", MAIN])
    summary = pd.DataFrame([
        ("描述性样本 2017–2023 站点年份", len(desc)),
        ("其中无保有量：站点年份", len(lost)), ("其中无保有量：站点", int(lost[STA].nunique())),
        ("其中无保有量：城市", int(lost[CITY].nunique())),
        (f"城市数：{SAME}", int(kinds.get(SAME, 0))), (f"城市数：{SHORT}", int(kinds.get(SHORT, 0))),
        (f"城市数：{ABSENT}", int(kinds.get(ABSENT, 0))), (f"城市数：{NOYEAR}", int(kinds.get(NOYEAR, 0))),
        ("补回：站点年份", len(restored)), ("补回：站点", int(restored[STA].nunique())),
        ("补回：城市", int(restored[CITY].nunique())),
        ("补回后仍无保有量：站点年份", int(cities["补回后仍无"].sum())),
        (f"补回后仍无：{ABSENT}的站点年份", int(left.loc[left["情形"] == ABSENT, "补回后仍无"].sum())),
        ("补回后仍无：名称已对上、城市面板该年无值的站点年份",
         int(left.loc[left["情形"] != ABSENT, "补回后仍无"].sum())),
        ("补回后仍无：名称已对上、城市面板该年无值的城市", int((left["情形"] != ABSENT).sum())),
        ("回归样本（现行）：站点年份", len(reg_cur)), ("回归样本（现行）：站点", int(reg_cur[STA].nunique())),
        ("回归样本（现行）：城市", int(reg_cur[CITY].nunique())),
        ("回归样本（补回）：站点年份", len(reg_fix)), ("回归样本（补回）：站点", int(reg_fix[STA].nunique())),
        ("回归样本（补回）：城市", int(reg_fix[CITY].nunique())),
    ], columns=["项目", "数值"])
    est = pd.DataFrame(est_cur + est_fix)
    for _, s in summary.iterrows():
        logger.info("%s = %s", s["项目"], s["数值"])
    for _, e in est.iterrows():
        logger.info("%s", {k: v for k, v in e.items() if pd.notna(v)})
    out = export_utils.write_excel_workbook("68_保有量匹配核对", [("汇总", summary), ("逐城市", cities), ("重估", est)])
    logger.info("已写出 %s；日志 %s", out, log_path)
    print(summary.to_string(index=False))
    print(est.to_string(index=False))


if __name__ == "__main__":
    main()
