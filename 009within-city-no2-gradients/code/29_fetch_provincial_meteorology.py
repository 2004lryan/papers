"""
29_fetch_provincial_meteorology.py: provincial annual meteorological control variables
(temperature / precipitation / wind speed / humidity / pressure)

Why this is needed:
    Pollutant concentrations are strongly affected by meteorological conditions; without controlling for meteorology,
    its year-to-year fluctuations are attributed to policy and to the composition of the vehicle fleet.
    The National Bureau of Statistics provincial annual database has no meteorological indicators (all 3115 indicators
    were checked), so they have to be supplied from an external source.

Data sources:
    NASA POWER Daily Point API (downscaled product of the MERRA-2 reanalysis, NASA Langley Research Center)
      https://power.larc.nasa.gov/api/temporal/daily/point
    City coordinates: Open-Meteo Geocoding API (built on GeoNames) + the manual coordinate table in this file
      https://geocoding-api.open-meteo.com/v1/search
    Both are free, need no key, and allow academic use.

    Why the Open-Meteo ERA5 archive API is not used for the meteorological values: its free quota is weighted by
    "number of variables × number of locations × number of days", and 337 cities × 5 variables × 10 years far exceeds
    the daily quota. In testing, requests are answered with "Minutely API request limit exceeded", and this error is
    a JSON payload with HTTP 200, so it shows up silently as "fewer cities retrieved". NASA POWER, by contrast,
    returns the full 10-year series for a single point in one request (tested: 3653 days, about 5.8 seconds).

Aggregation (**kept consistent** with the aggregation of the provincial air-quality series, so that the two can be
paired in the same regression):
    first compute city annual values → then take the arithmetic mean over the province's **cities at prefecture level
    and above**.
    City annual values: temperature / wind speed / humidity / pressure are annual means of the daily values;
    precipitation is the annual sum of daily precipitation.

Coordinate check:
    A geocoding candidate is accepted only if its admin1 (province-level name) matches the province that this
    project's "城市-省份对照表" (city-province lookup table) gives for the city; a city with no matching candidate is
    marked as not found in the 校验 (check) column of the `03_坐标校验` (coordinate check) sheet (the query names
    tried are only logged) and **excluded** from aggregation, so that a coordinate lying in a neighbouring province
    never stands in for this province's meteorology. Cities in COORD_OVERRIDE bypass this check.

Usage:
    python code/29_fetch_provincial_meteorology.py --start-year 2015 --end-year 2024

Output files:
    data/省级年度气象-2015_2024.csv (provincial annual meteorology)
        — province × year × 5 meteorological variables
    data/城市坐标-地级及以上.csv (city coordinates, prefecture level and above)
        — geocoding results and check status; written only when absent or with --regeocode, otherwise read and reused
    outputs/29_气象构建过程.xlsx (meteorology construction steps)
        — sheets 01_省级年度气象 (provincial results), 02_城市年值 (city annual values),
          03_坐标校验 (coordinate check)
    logs/29_fetch_provincial_meteorology.log
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import pandas as pd

if TYPE_CHECKING:
    import logging

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
# NASA POWER daily parameters: air temperature (℃) / precipitation (mm·d⁻¹) / 10 m wind speed (m·s⁻¹) /
# 2 m relative humidity (%) / surface pressure (kPa)
POWER_VARS = ["T2M", "PRECTOTCORR", "WS10M", "RH2M", "PS"]
POWER_FILL = -999.0  # POWER fill value for missing data


def curl_json(url: str, tries: int = 4) -> Any:
    """Python's SSL stack handshook unreliably on the network path used when the data were retrieved, so all requests
    go through curl.

    Non-ASCII characters in the URL are percent-encoded first: the nginx in front of Open-Meteo answers unencoded
    Chinese query strings with 403 straight away, and without encoding the call keeps retrying until it times out.
    """
    url = quote(url, safe=":/?&=,%.-_~")
    for i in range(tries):
        p = subprocess.run(["curl", "-s", "--max-time", "90", url], capture_output=True, text=True)
        if p.stdout.strip():
            try:
                return json.loads(p.stdout)
            except json.JSONDecodeError:
                pass
        time.sleep(3 * (i + 1))
    return None


def norm_prov(name: str) -> str:
    """Reduce Chinese province-level names such as Henan Province, Xinjiang Uygur Autonomous Region or Guangxi Zhuang
    Autonomous Region to short names that can be compared."""
    s = re.sub(r"(维吾尔|壮族|回族|自治区|省|市)$", "", name)
    s = re.sub(r"(维吾尔|壮族|回族)", "", s)
    return s.replace("自治区", "").replace("省", "").replace("市", "")


# Seats of prefecture-level divisions: autonomous prefectures / leagues / prefectures (diqu) are mostly listed in
# GeoNames under their **seat city**, so a search by the prefecture name does not find the division itself (e.g. the
# seat of Nujiang Prefecture is Lushui, and the seat of Xilin Gol League is Xilinhot).
SEAT: dict[str, str] = {
    "延边朝鲜族自治州": "延吉", "大兴安岭地区": "加格达奇",
    "恩施土家族苗族自治州": "恩施", "湘西州": "吉首",
    "阿坝藏族羌族自治州": "马尔康", "甘孜藏族自治州": "康定", "凉山彝族自治州": "西昌",
    "黔西南布依族苗族自治州": "兴义", "黔东南苗族侗族自治州": "凯里", "黔南布依族苗族自治州": "都匀",
    "楚雄州": "楚雄", "红河州": "蒙自", "文山州": "文山", "西双版纳州": "景洪",
    "大理州": "大理", "德宏州": "芒市", "怒江州": "泸水", "迪庆州": "香格里拉",
    "阿里地区": "狮泉河",
    "临夏回族自治州": "临夏", "甘南州": "合作",
    "海北藏族自治州": "海晏", "黄南藏族自治州": "同仁", "海南藏族自治州": "共和",
    "果洛藏族自治州": "玛沁", "玉树藏族自治州": "玉树", "海西蒙古族藏族自治州": "德令哈",
    "昌吉州": "昌吉", "博尔塔拉蒙古自治州": "博乐", "巴音郭楞州": "库尔勒",
    "克孜勒苏柯尔克孜自治州": "阿图什", "伊犁哈萨克州": "伊宁",
    "阿克苏地区": "阿克苏", "塔城地区": "塔城", "阿勒泰地区": "阿勒泰",
    "喀什地区": "喀什", "和田地区": "和田",
    "兴安盟": "乌兰浩特", "锡林郭勒盟": "锡林浩特", "阿拉善盟": "巴彦浩特",
}

# GeoNames admin1 English name → province-level name used in this project
ADMIN1_EN: dict[str, str] = {
    "beijing": "北京市", "tianjin": "天津市", "shanghai": "上海市", "chongqing": "重庆市",
    "hebei": "河北省", "shanxi": "山西省", "inner mongolia": "内蒙古自治区",
    "liaoning": "辽宁省", "jilin": "吉林省", "heilongjiang": "黑龙江省",
    "jiangsu": "江苏省", "zhejiang": "浙江省", "anhui": "安徽省", "fujian": "福建省",
    "jiangxi": "江西省", "shandong": "山东省", "henan": "河南省", "hubei": "湖北省",
    "hunan": "湖南省", "guangdong": "广东省", "guangxi": "广西壮族自治区", "hainan": "海南省",
    "sichuan": "四川省", "guizhou": "贵州省", "yunnan": "云南省", "tibet": "西藏自治区",
    "shaanxi": "陕西省", "gansu": "甘肃省", "qinghai": "青海省",
    "ningxia": "宁夏回族自治区", "xinjiang": "新疆维吾尔自治区",
}


# Cities that geocoding cannot resolve get their coordinates directly (autonomous prefectures / leagues use the
# administrative seat).
# GeoNames' Chinese index lacks these entries, and the pinyin does not match either (Changzhi → lazy_pinyin gives
# Zhangzhi; Lüliang → Lvliang, which differs from the spelling in the database). Rather than keep piling up query
# names, this table is hard-coded so that it can be checked entry by entry.
# Two decimal places are enough: the NASA POWER meteorology grid (MERRA-2, 0.5° × 0.625°) spans roughly 50–70 km,
# and two decimals (about 1 km) are far finer than the grid.
COORD_OVERRIDE: dict[str, tuple[float, float]] = {
    "成都": (30.66, 104.06), "自贡": (29.34, 104.78), "泸州": (28.87, 105.44),
    "绵阳": (31.47, 104.68), "广元": (32.44, 105.84), "广安": (30.46, 106.63),
    "达州": (31.21, 107.47),
    "阿坝藏族羌族自治州": (31.90, 102.22),   # seat: Barkam
    "凉山彝族自治州": (27.89, 102.27),       # seat: Xichang
    "宿州": (33.65, 116.98), "马鞍山": (31.67, 118.51),
    "德州": (37.44, 116.36), "泰安": (36.19, 117.09), "滨州": (37.38, 117.97),
    "吕梁": (37.52, 111.13), "长治": (36.19, 113.12),
    "梧州": (23.48, 111.28), "贺州": (24.40, 111.57),
    "常州": (31.81, 119.97), "扬州": (32.39, 119.42), "泰州": (32.46, 119.92),
    "抚州": (27.98, 116.36),
    "湖州": (30.89, 120.09), "衢州": (28.94, 118.87),
    "三沙": (16.83, 112.34),                 # seat: Yongxing Island, Xisha
    "儋州": (19.52, 109.58),
    "鄂州": (30.40, 114.89),
    "山南": (29.24, 91.77),                  # seat: Naidong
    "黔南布依族苗族自治州": (26.26, 107.52),  # seat: Duyun
    "锦州": (41.10, 121.13),
    "海北藏族自治州": (36.90, 100.99),        # seat: Haiyan
    "海南藏族自治州": (36.28, 100.62),        # seat: Gonghe
    "阿拉善盟": (38.83, 105.67),              # seat: Bayanhot
}


def pinyin_of(name: str) -> str:
    from pypinyin import lazy_pinyin
    return "".join(lazy_pinyin(name)).capitalize()


def query_names(city: str) -> list[str]:
    """Candidate query names for one city, from most to least reliable.

    GeoNames covers Chinese names only partially: a form without the generic suffix, such as Changchun, often hits a
    village of the same name, and Chinese names such as Lincang City or Tongliao City are not found at all, whereas the
    corresponding pinyin names (Lincang, Tongliao) are all there.
    Autonomous prefectures / leagues / prefectures (diqu) are listed in the database under their seat city, so a search
    by the prefecture name does not find the division itself.
    Hence the order: Chinese name with "city" (shi) appended when it has no generic suffix → Chinese name as given →
    pinyin of that name → name without its generic suffix and its pinyin → seat name in Chinese (with and without
    "shi") → seat name in pinyin.
    """
    names: list[str] = []
    if not city.endswith(("州", "地区", "盟", "市")):
        names.append(city + "市")
    names.append(city)
    # The pinyin of the full name must be tried first: for prefecture-level cities such as Changzhou, Luzhou and Ezhou,
    # whose **own name already ends in the suffix "zhou"**, stripping the suffix first and then converting to pinyin
    # gives Chang instead of Changzhou, which is never found.
    names.append(pinyin_of(city))

    base = city
    for suffix in ("自治州", "地区", "盟", "州", "市"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base and base != city:
        names += [base, pinyin_of(base)]

    seat = SEAT.get(city)
    if seat:
        names += [seat + "市", seat, pinyin_of(seat)]

    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def score_candidate(cand: dict[str, Any], prov: str) -> int:
    """The province must match (admin1 is accepted in Chinese or English); candidates are then ranked by
    administrative-centre level and population."""
    if cand.get("country_code") != "CN":
        return -1
    raw = (cand.get("admin1") or "").strip()
    mapped = ADMIN1_EN.get(raw.lower())
    if mapped:
        if mapped != prov:
            return -1
    else:
        a1 = norm_prov(raw)
        if not a1 or not (a1 in norm_prov(prov) or norm_prov(prov) in a1):
            return -1
    score = {"PPLC": 400, "PPLA": 300, "PPLA2": 200, "PPLA3": 120, "PPL": 60}.get(
        cand.get("feature_code") or "", 0)
    pop = cand.get("population") or 0
    return score + min(int(pop) // 10000, 200)


def geocode_one(city: str, prov: str) -> dict[str, Any]:
    if city in COORD_OVERRIDE:
        lat, lon = COORD_OVERRIDE[city]
        return {"城市": city, "省级行政区": prov, "检索名": "人工指定", "纬度": lat, "经度": lon,
                "地理编码名": city, "地理编码省": prov, "校验": "通过（人工指定坐标）", "试过": ""}
    best, best_score, used = None, -1, ""
    for qname in query_names(city):
        lang = "en" if qname.isascii() else "zh"
        data = curl_json(f"{GEO_URL}?name={qname}&count=10&language={lang}&format=json", tries=3)
        for cand in (data or {}).get("results", []) or []:
            sc = score_candidate(cand, prov)
            if sc > best_score:
                best, best_score, used = cand, sc, qname
        if best is not None:
            break
    if best is None:
        return {"城市": city, "省级行政区": prov, "检索名": "", "纬度": None, "经度": None,
                "地理编码名": "", "地理编码省": "", "校验": "未找到",
                "试过": "/".join(query_names(city))}
    return {"城市": city, "省级行政区": prov, "检索名": used,
            "纬度": best["latitude"], "经度": best["longitude"],
            "地理编码名": best.get("name", ""), "地理编码省": best.get("admin1", ""),
            "校验": "通过", "试过": ""}


def geocode(cities: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Concurrent geocoding. Run serially, a city that cannot be found goes through every candidate query name, each
    with up to three tries and a linear back-off (3, 6, 9 s) after failed requests, and 337 cities take more than two
    hours; concurrency cuts the wall-clock time to a dozen or so minutes."""
    pairs = [(r["标准名"], r["省级行政区"]) for _, r in cities.iterrows()]
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for res in pool.map(lambda t: geocode_one(*t), pairs):
            rows.append(res)
            done += 1
            if res["校验"] == "未找到":
                logger.warning("  未找到坐标: %s（%s）—— 试过 %s",
                               res["城市"], res["省级行政区"], res["试过"])
            if done % 50 == 0:
                logger.info("  地理编码进度 %s/%s", done, len(pairs))
    return pd.DataFrame(rows).drop(columns=["试过"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--regeocode", action="store_true", help="忽略已有坐标表，重新地理编码")
    args = ap.parse_args()

    logger, log_path = export_utils.configure_file_logger("29_fetch_provincial_meteorology")
    data_dir = export_utils.DATA_DIR

    mapping = pd.read_csv(data_dir / "城市-省份对照表.csv")
    pref = (mapping[mapping["纳入省级聚合"]][["标准名", "省级行政区"]]
            .drop_duplicates().sort_values(["省级行政区", "标准名"]).reset_index(drop=True))
    logger.info("待取气象的地级及以上城市 %s 个，覆盖 %s 个省级行政区",
                len(pref), pref["省级行政区"].nunique())

    coord_path = data_dir / "城市坐标-地级及以上.csv"
    if coord_path.exists() and not args.regeocode:
        coords = pd.read_csv(coord_path)
        logger.info("复用已有坐标表 %s（%s 行）", coord_path.name, len(coords))
    else:
        logger.info("开始地理编码 …")
        coords = geocode(pref, logger)
        coords.to_csv(coord_path, index=False, encoding="utf-8-sig")
    ok_coords = coords[coords["校验"].str.startswith("通过") & coords["纬度"].notna()].reset_index(drop=True)
    logger.info("坐标校验：通过 %s / %s；未找到 %s（未找到的城市不参与本省聚合）",
                len(ok_coords), len(coords), int((coords["校验"] == "未找到").sum()))
    thin = (ok_coords.groupby("省级行政区").size()
            / coords.groupby("省级行政区").size()).sort_values()
    for prov, ratio in thin[thin < 0.8].items():
        logger.warning("  %s 仅 %.0f%% 的地级及以上城市取到坐标", prov, 100 * ratio)

    city_year_rows = []
    # ── Fetch NASA POWER daily values city by city (one request covers all years) ────────────────
    def fetch_city(rec: tuple[int, pd.Series]) -> list[dict[str, Any]]:
        _, c = rec
        url = (f"{POWER_URL}?parameters={','.join(POWER_VARS)}&community=AG"
               f"&longitude={c['经度']:.4f}&latitude={c['纬度']:.4f}"
               f"&start={args.start_year}0101&end={args.end_year}1231&format=JSON")
        data = curl_json(url, tries=4)
        param = ((data or {}).get("properties") or {}).get("parameter") or {}
        if not param.get("T2M"):
            return []
        df = pd.DataFrame({k: pd.Series(v) for k, v in param.items()})
        df = df.replace(POWER_FILL, pd.NA)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        out = []
        for year, g in df.groupby(df.index.year):
            out.append({
                "城市": c["城市"], "省级行政区": c["省级行政区"], "年份": int(year),
                "年平均气温_摄氏度": round(pd.to_numeric(g["T2M"]).mean(), 2),
                "年降水量_毫米": round(pd.to_numeric(g["PRECTOTCORR"]).sum(), 1),
                "年平均风速_米每秒": round(pd.to_numeric(g["WS10M"]).mean(), 3),
                "年平均相对湿度_百分比": round(pd.to_numeric(g["RH2M"]).mean(), 1),
                "年平均气压_百帕": round(pd.to_numeric(g["PS"]).mean() * 10, 1),  # kPa → hPa
                "有效天数": int(pd.to_numeric(g["T2M"]).notna().sum()),
            })
        return out

    failed_cities: list[str] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, res in enumerate(pool.map(fetch_city, list(ok_coords.iterrows())), 1):
            if res:
                city_year_rows.extend(res)
            else:
                failed_cities.append(ok_coords.iloc[i - 1]["城市"])
            if i % 50 == 0:
                logger.info("  取数进度 %s/%s，累计城市-年记录 %s", i, len(ok_coords), len(city_year_rows))
    if failed_cities:
        logger.warning("取数失败 %s 个城市：%s", len(failed_cities), failed_cities[:30])

    city_year = pd.DataFrame(city_year_rows)
    if city_year.empty:
        raise SystemExit("✗ 未取到任何气象数据")

    metrics = ["年平均气温_摄氏度", "年降水量_毫米", "年平均风速_米每秒",
               "年平均相对湿度_百分比", "年平均气压_百帕"]
    prov_year = (city_year.groupby(["省级行政区", "年份"])
                 .agg(**{m: (m, "mean") for m in metrics},
                      参与城市数=("城市", "nunique"))
                 .round(3).reset_index())

    out_csv = data_dir / "省级年度气象-2015_2024.csv"
    prov_year.to_csv(out_csv, index=False, encoding="utf-8-sig")

    export_utils.write_excel_workbook("29_气象构建过程",
                                      [("省级年度气象", prov_year), ("城市年值", city_year),
                                       ("坐标校验", coords)])

    logger.info("完成：省级气象面板 %s 行（%s 省 × %s 年），参与城市 %s 个",
                len(prov_year), prov_year["省级行政区"].nunique(),
                prov_year["年份"].nunique(), city_year["城市"].nunique())
    logger.info("  - %s", out_csv)
    print(f"✓ 省级年度气象：{len(prov_year)} 行 / {prov_year['省级行政区'].nunique()} 省 / "
          f"{prov_year['年份'].min()}-{prov_year['年份'].max()}，参与城市 {city_year['城市'].nunique()} 个")
    print(prov_year.groupby("年份")[metrics].mean().round(2).to_string())
    print(f"  数据: {out_csv}")
    print(f"  日志: {log_path}")


if __name__ == "__main__":
    main()
