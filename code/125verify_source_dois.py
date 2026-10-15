#!/usr/bin/env python3
"""
125 — 源数据 DOI 可解析性核验（CILS Option C 合规的证据）

背景
----
本刊 Option C 要求「把研究数据存入相关仓库 + 在文中引用并链接」。本研究未采集新数据，
三个真实体系的数据均由**其原作者**存入公开仓库，本文的义务因而是「引用并链接」。
稿件的 Data & Code Availability 把这一点作为合规依据，故这三个 DOI 是该声明的唯一支点——
它们必须真实注册且可解析，不能只凭稿件里写着就算数。本脚本把该核验固化为可复算的产物。

为何分两个注册机构查
--------------------
Mendeley Data 的数据集 DOI 由 **DataCite** 注册，Crossref 查它必然 404——这不是失效。
期刊论文 DOI 由 **Crossref** 注册。故须按类型分流，否则会把正常记录误判为坏 DOI。
另注：MDPI 站点对 curl 直连返回 403（反爬），也**不能**据此判定 DOI 失效；
判据一律取注册机构 API 的元数据，不取 doi.org 的 HTTP 状态码。

运行: python3 02code/125verify_source_dois.py
输出: 04outputs/125source_doi_verification.json
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "arrhenius-recoverability-audit/1.0 (mailto:panlinryan@gmail.com)"}

SOURCES: list[dict[str, Any]] = [
    {
        "id": "hass_avocado",
        "doi": "10.17632/3xd9n945v8.1",
        "registry": "datacite",
        "role": "真实体系 3 / 原始 xlsx 直接读取",
        "expect_type": "Dataset",
        "expect_title_contains": "Avocado Ripening",
    },
    {
        "id": "apple_firmness",
        "doi": "10.3390/foods12112113",
        "registry": "crossref",
        "role": "真实体系 1 / 表 A1 提取",
        "expect_type": "journal-article",
        "expect_title_contains": "Fresh Food Quality",
    },
    {
        "id": "food_ammonia_tbars",
        "doi": "10.3390/foods11142004",
        "registry": "crossref",
        "role": "真实体系 2 / 表 3 提取",
        "expect_type": "journal-article",
        "expect_title_contains": "Long-Life Food",
    },
]


def _get(url: str) -> Any:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def query(doi: str, registry: str) -> dict[str, Any]:
    try:
        if registry == "crossref":
            m = _get(f"https://api.crossref.org/works/{doi}")["message"]
            return {
                "ok": True,
                "title": (m.get("title") or ["?"])[0],
                "container": (m.get("container-title") or [""])[0],
                "year": m.get("issued", {}).get("date-parts", [["?"]])[0][0],
                "type": m.get("type"),
                "license": [x.get("URL") for x in m.get("license", [])] or None,
            }
        a = _get(f"https://api.datacite.org/dois/{doi}")["data"]["attributes"]
        return {
            "ok": True,
            "title": a["titles"][0]["title"],
            "container": a.get("publisher"),
            "year": a.get("publicationYear"),
            "type": a.get("types", {}).get("resourceTypeGeneral"),
            "license": [x.get("rightsIdentifier") for x in a.get("rightsList", [])
                        if x.get("rightsIdentifier")] or None,
        }
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    rows: list[dict[str, Any]] = []
    for s in SOURCES:
        r = query(s["doi"], s["registry"])
        title_ok = bool(r.get("ok")) and s["expect_title_contains"].lower() in str(r.get("title", "")).lower()
        type_ok = bool(r.get("ok")) and str(r.get("type")) == s["expect_type"]
        rows.append({**{k: s[k] for k in ("id", "doi", "registry", "role")},
                     **r, "title_matches_expected": title_ok, "type_matches_expected": type_ok,
                     "verdict": "PASS" if (title_ok and type_ok) else "FAIL"})

    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    out = {
        "script": Path(__file__).name,
        "generated": stamp,
        "policy": "CILS Option C —— 存入相关仓库 + 在文中引用并链接；本文未采集新数据，"
                  "三体系均由原作者存入公开仓库，本文履行『引用并链接』。",
        "n_sources": len(rows), "n_pass": n_pass,
        "all_pass": n_pass == len(rows),
        "caveat": "本核验只证明 DOI 已注册、元数据与稿件用途相符；不证明数据内容正确"
                  "——内容正确性由脚本 55 的逐值复核与脚本 120 的 SHA-256 登记承担。",
        "sources": rows,
    }
    (ROOT / "04outputs/125source_doi_verification.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in rows:
        mark = "✅" if r["verdict"] == "PASS" else "❌"
        print(f"  {mark} {r['doi']:<24} [{r['registry']}] {str(r.get('title'))[:60]}")
        print(f"      {r.get('container')} · {r.get('year')} · {r.get('type')} · 许可 {r.get('license')}")
    print(f"\n{n_pass}/{len(rows)} 通过 -> {'过' if out['all_pass'] else '不过'}")


if __name__ == "__main__":
    main()
