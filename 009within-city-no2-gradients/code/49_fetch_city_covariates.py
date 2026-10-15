"""
49_fetch_city_covariates.py: fetch city-level covariates from "Major Cities Annual Data" (2017–2023)

Why it is needed:
    The largest threat to the station-level design is that the NEV stock is highly correlated with a city's
    level of development, so "the cores of developed cities improve faster" could be misread as "electrification makes
    traffic stations improve faster".
    Check S4a in script 48 uses only **provincial** civilian vehicle ownership as a proxy (city-level totals of motor
    vehicles are not available).
    This script adds three groups of city-level variables for checks S4c–S4e and S14a–S14c of script 48, upgrading
    the control for this threat from a provincial proxy to city-level measurements.
    **Coverage**: the database includes only 36 major cities (4 municipalities directly under the central government
    + provincial capitals + cities specifically designated in the state plan), not all prefecture-level cities, so
    these variables can serve only **subsample robustness** checks and cannot enter the main table — but these 36
    cities are exactly the ones with the densest stations.

    ① Road-traffic and ambient-noise equivalent sound levels, dB(A); the road-traffic level is a **measured urban
       road-traffic intensity** (check S4c). Noise is determined by traffic volume and vehicle-type mix; unlike
       motor-vehicle registrations, it measures "how many vehicles are actually running on the roads", which is
       exactly what the confounder control needs to capture.
    ② Gross regional product, value added of the secondary and of the tertiary industry, registered population at
       year end — level of development and industrial structure. Script 48 uses gross regional product and
       secondary-industry value added as controls (checks S4e, S4d) and as treatments in the falsification test that
       "replaces NEV with a development proxy" (S14a, S14c): if a development proxy alone reproduces the NEV
       coefficient, NEV is merely a stand-in for the level of development.
    ③ Freight and passenger volumes and total retail sales of consumer goods — freight (diesel vehicles) and the
       intensity of commercial activity; retail sales enter the falsification test as S14b.

All indicator codes below were copied from the NBS catalogue tree of the Major Cities Annual Data database rather than
found by keyword search, because fuzzy keyword matching can pick indicators that turn out to be entirely empty.

The rootId is not in the tree, so the script first intercepts the table request sent by the page itself to discover
it; the city list is read from the region catalogue of each indicator category, and the indicators are then fetched
one by one with the same payload format.

Usage:
    python code/49_fetch_city_covariates.py --start-year 2017 --end-year 2023

Output files:
    data/国家统计局-城市级协变量-2017_2023.csv (NBS city-level covariates, 2017–2023)
        — wide table, one row per city and year
    outputs/49_城市级协变量.xlsx (city-level covariates)  — sheets 01_宽表 / 02_长表 / 03_覆盖率 (wide table / long
        table / coverage)
    logs/49_fetch_city_covariates.log
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from playwright.sync_api import Page, sync_playwright

if TYPE_CHECKING:
    import logging

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

CITY_URL = "https://data.stats.gov.cn/dg/website/page.html#/pc/national/mainYearData"
TABLE_ENDPOINT = "/dg/website/publicrelease/web/external/stream/esData"
MIN_COVERAGE = 0.10

TARGETS: list[dict[str, str]] = [
    {"短名": "道路交通等效声级", "code": "a504d94f5c7c4f128ebc039fb3ad0944",
     "cid": "a929d2a830aa4ddb9917f42bb1c7c25e",
     "path": "主要城市年度数据 > 噪声监测 > 道路交通等效声级dB (A)"},
    {"短名": "环境噪声等效声级", "code": "7547bac794b04488ad14f1517c100fef",
     "cid": "a929d2a830aa4ddb9917f42bb1c7c25e",
     "path": "主要城市年度数据 > 噪声监测 > 环境噪声等效声级dB (A)"},
    {"短名": "城市地区生产总值", "code": "35613910b9ad47e19cf79ae041ccb7ed",
     "cid": "75a85abe58c749e0ad1d85be343e9056",
     "path": "主要城市年度数据 > 国民经济核算 > 地区生产总值 (当年价格) (亿元)"},
    {"短名": "城市第二产业增加值", "code": "368fe576d65c455a8013770b9bf757b8",
     "cid": "75a85abe58c749e0ad1d85be343e9056",
     "path": "主要城市年度数据 > 国民经济核算 > 第二产业增加值 (亿元)"},
    {"短名": "城市第三产业增加值", "code": "e7a0226e5029477bbe9044fb071b16ea",
     "cid": "75a85abe58c749e0ad1d85be343e9056",
     "path": "主要城市年度数据 > 国民经济核算 > 第三产业增加值 (亿元)"},
    {"短名": "城市年末户籍人口", "code": "3825c2d7a32b4cf6b3d72fc5ed9e62c6",
     "cid": "b13e73d4a4734a46bf358aa1baff2649",
     "path": "主要城市年度数据 > 人口和就业 > 年末户籍人口 (万人)"},
    {"短名": "城市货物运输量", "code": "f1bacc7f53cf4d27b021f33f241f2d8d",
     "cid": "21edb86e322a4125a7eff2dbf25d4ca6",
     "path": "主要城市年度数据 > 运输和邮电 > 货物运输量 (万吨)"},
    {"短名": "城市旅客运输量", "code": "d80f8c7cd07747949ef6282cecc62b90",
     "cid": "21edb86e322a4125a7eff2dbf25d4ca6",
     "path": "主要城市年度数据 > 运输和邮电 > 旅客运输量 (万人)"},
    {"短名": "城市社会消费品零售总额", "code": "06eac78eee1c4eac9fcaeb0e8b5b108d",
     "cid": "e4703d312cd44d498c168a10d2337f4a",
     "path": "主要城市年度数据 > 贸易经济 > 社会消费品零售总额 (亿元)"},
]

_SNIFF = """() => {
    if (window.__cap) return;
    window.__cap = [];
    const of = window.fetch;
    window.fetch = function (...args) {
        try {
            const u = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url) || '';
            if (u.indexOf('esData') >= 0 && args[1] && args[1].body) {
                window.__cap.push(String(args[1].body));
            }
        } catch (e) {}
        return of.apply(this, args);
    };
}"""

_GET = """async (url) => {
    const r = await fetch(url, {credentials: 'include'});
    return await r.text();
}"""

_POST = """async ([endpoint, payload]) => {
    const r = await fetch(endpoint, {
        method: 'POST', credentials: 'include',
        headers: {'Content-Type': 'application/json;charset=UTF-8'},
        body: JSON.stringify(payload)});
    return await r.text();
}"""


def get_json(page: Page, url: str) -> Any:
    txt = page.evaluate(_GET, url)
    if txt.lstrip().startswith("<"):
        raise RuntimeError(f"接口返回 HTML 而不是 JSON：{url}")
    return json.loads(txt)


def sniff_root(page: Page, logger: logging.Logger) -> str:
    """Take the rootId from the table request sent by the page itself (the page selects only one city by default, so
    its das cannot be used)."""
    page.add_init_script(_SNIFF)
    page.goto(CITY_URL, wait_until="domcontentloaded", timeout=90_000)
    page.evaluate(_SNIFF)
    for _ in range(60):
        page.wait_for_timeout(1000)
        for body in page.evaluate("() => window.__cap || []"):
            try:
                root = (json.loads(body) or {}).get("rootId") or ""
            except Exception:
                continue
            if root:
                logger.info("探到 rootId=%s", root)
                return root
    raise RuntimeError("未能探到 rootId——站点结构可能又变了")


def city_list(page: Page, indicator_cid: str, logger: logging.Logger) -> list[dict[str, Any]]:
    """Get the region catalogue of an indicator category, then **all cities** under that catalogue.
    By default the page ticks only one city (Beijing), so using the das from the page's payload directly would fetch
    only one row."""
    tree = get_json(page, "/dg/website/publicrelease/web/external/"
                          f"getDaCatalogTreeByIndicatorCid?indicatorCid={indicator_cid}")
    # The node primary key is _id / publicrelease_web_dacatalog_id, not id
    ids, stack = [], list(tree.get("data") or [])
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            for k in ("publicrelease_web_dacatalog_id", "_id", "id"):
                if n.get(k):
                    ids.append(n[k])
                    break
            stack.extend(n.get("children") or [])
    best: list[dict[str, Any]] = []
    for did in dict.fromkeys(ids):
        try:
            data = get_json(page, "/dg/website/publicrelease/web/external/"
                                  f"getDasByDaCatalogId?daCid={did}").get("data") or []
        except Exception:
            continue
        if len(data) > len(best):
            best = data
    if not best:
        raise RuntimeError(f"指标分类 {indicator_cid} 下取不到城市列表")
    logger.info("  地区目录 %s 个候选 → 取到 %s 个城市（示例 %s）", len(ids), len(best),
                [x.get("show_name") for x in best[:4]])
    return [{"text": x["show_name"], "value": x["name_value"]} for x in best]


def post_json(page: Page, payload: dict[str, Any]) -> Any:
    txt = page.evaluate(_POST, [TABLE_ENDPOINT, payload])
    if txt.lstrip().startswith("<"):
        raise RuntimeError("表格接口返回 HTML，而不是 JSON")
    return json.loads(txt)


def fetch_one(
    page: Page, target: dict[str, str], das: list[dict[str, Any]], root: str, year_range: str
) -> pd.DataFrame:
    res = post_json(page, {"cid": target["cid"], "indicatorIds": [target["code"]],
                           "daCatalogId": "", "das": das, "showType": "3",
                           "dts": [year_range], "rootId": root})
    recs = []
    for yr in (res.get("data") or []):
        year = int(str(yr.get("code", "")).replace("YY", ""))
        for v in (yr.get("values") or []):
            recs.append({"年份": year, "城市代码": v.get("areaCode", ""), "城市": v.get("area", ""),
                         "指标": target["短名"], "指标编码": target["code"],
                         "单位": v.get("du_name") or "",
                         "数值": pd.to_numeric(v.get("value"), errors="coerce"),
                         "出处": target["path"]})
    return pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2017)
    ap.add_argument("--end-year", type=int, default=2023)
    args = ap.parse_args()

    logger, log_path = export_utils.configure_file_logger("49_fetch_city_covariates")
    kw, exe = export_utils.build_playwright_launch_kwargs(headless=True)
    yr = f"{args.start_year}YY-{args.end_year}YY"
    logger.info("抓主要城市年度数据；年份 %s，指标 %s 个；浏览器 %s", yr, len(TARGETS), exe)

    frames, cov = [], []
    with sync_playwright() as pw:
        br = pw.chromium.launch(**kw)
        page = br.new_page()
        root = sniff_root(page, logger)
        das_cache: dict[str, list[dict[str, Any]]] = {}
        n_years = args.end_year - args.start_year + 1
        for i, t in enumerate(TARGETS, 1):
            try:
                if t["cid"] not in das_cache:
                    das_cache[t["cid"]] = city_list(page, t["cid"], logger)
                das = das_cache[t["cid"]]
                n_cell = len(das) * n_years
                df = fetch_one(page, t, das, root, yr)
            except Exception as exc:
                logger.warning("[%s/%s] %s 抓取失败：%s", i, len(TARGETS), t["短名"], exc)
                cov.append({"指标": t["短名"], "行数": 0, "非空": 0, "非空率": 0.0,
                            "判定": "抓取失败", "出处": t["path"]})
                continue
            nn = int(df["数值"].notna().sum())
            rate = nn / n_cell if n_cell else 0.0
            ok = rate >= MIN_COVERAGE
            cov.append({"指标": t["短名"], "行数": len(df), "非空": nn, "非空率": round(rate, 4),
                        "判定": "可用" if ok else "覆盖不足/官方无数据", "出处": t["path"]})
            logger.info("[%s/%s] %-12s 行 %-5s 非空 %-5s (%.1f%%) → %s",
                        i, len(TARGETS), t["短名"], len(df), nn, 100 * rate,
                        "可用" if ok else "覆盖不足")
            if ok:
                frames.append(df)
            time.sleep(1.2)
        br.close()

    if not frames:
        raise SystemExit("✗ 没有任何指标达到覆盖率门槛")
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["城市", "城市代码", "年份"], columns="指标",
                            values="数值").reset_index()
    wide.columns.name = None
    out = export_utils.DATA_DIR / f"国家统计局-城市级协变量-{args.start_year}_{args.end_year}.csv"
    wide.to_csv(out, index=False, encoding="utf-8-sig")
    cov_df = pd.DataFrame(cov)
    export_utils.write_excel_workbook("49_城市级协变量",
                                      [("宽表", wide), ("长表", long), ("覆盖率", cov_df)])

    print(f"✓ 城市级协变量 {wide.shape[0]} 行 × {wide.shape[1]} 列 → {out}")
    print(f"  城市 {wide['城市'].nunique()} 个，年份 {wide['年份'].min()}–{wide['年份'].max()}")
    print(cov_df.to_string(index=False))
    print(f"  日志: {log_path}")


if __name__ == "__main__":
    main()
