#!/usr/bin/env python3
r"""
143 — RYA-579：台账侧三件收据，全部机械算出，不接受转述。

A. a9930fd7d82b（oracle vs plug-in Fisher，现为 OPEN + UNRESOLVED_DISAPPEARANCE）
   §L.17.2 已判不成立，依据是「稿面事先声明该组半合成实验刻意用 oracle 层」。
   本节把那句话在【本轮新冻结版】的两稿里重新定位并落哈希，让该判定带收据，
   不再以「消失」的形态挂在台账上。判定权仍在 adjudicator，本脚本只提供证据。

B. fingerprint 把绝对路径算进义务身份 -> 同一 finding 每换一次沙箱被开成新条。
   判据写死在这里，不用形容词，且分两步（第一步会误伤，故必须有第二步）：
     第一步（候选）：同 skill + 同 pattern_id + 描述里出现同一个 DOI，且分布在 >= 2 个不同沙箱目录下。
     第二步（判真）：候选组内各条所争的「值对」（年份 / 条目号，剥掉 DOI 与 URL 后抽出）必须完全一致。
       实测第一步会把「同一篇文献的两个不同缺陷」也圈进来（kurniawan：条目号 vs 作者后缀），
       那不是路径导致的重复，不得计入。第二步按所争值对来判，通过才算真重复组。
   无 DOI 的条目改用标题相似度 >= 0.75 且跨沙箱。

C. 六条 OPEN 的本轮处置对照表（只登记事实，不改台账文件）。

输出 04outputs/143rya579_ledger_receipts.json
`refine-logs/FORENSICS_*.json` 只读，本脚本一个字节都不写它们。
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "06doc" / "01manuscript"
LEDGER = ROOT / "refine-logs" / "FORENSICS_obligations.json"

# 本轮六条 OPEN 的处置（RYA-579）
DISPOSITION = {
    "4c047ea6ba6833ec1a73f7dd": ("改稿", "LAN 临界档非局部位移：下界改走精确高斯 KL，三档统一，全程不援引 lem:laqf"),
    "b7d6ca641c1be58d30acda57": ("改稿", "实证范围：讨论节收窄，写明另两个体系各自贡献什么、为何不构成设计法则证据"),
    "f22b4a386e47e1e9c1c07f09": ("判不成立 + 改根因", "CV 比：稿面口径为 5 种子均值，reviewer 比的是单个最窄设计对；本轮补产物指向"),
    "a3d46e9f14929a94bbede04f": ("判不成立 + 改根因", "Spearman：稿面为 5 种子均值 -0.7759，reviewer 比的是原生单次抽样 -0.8294；本轮补产物指向"),
    "7d5c0b20273ce31b85e09c59": ("挂起待 ryan 签", "donoho2009 年份：三轮判假阳性，稿面不改；waive 需真人 --approver，执行方不得自签"),
    "a9930fd7d82bdc86fe7b8adb": ("判不成立 + 补收据", "oracle vs plug-in：稿面事先声明刻意用 oracle 层，见本收据 A 节"),
}

ORACLE_PROBE = {
    "Elsevier_zh.tex": "本组半合成实验用的是 oracle 层",
    "Elsevier_en.tex": "uses the oracle layer",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sandbox(item) -> str:
    ev = item["finding_snapshot"].get("evidence") or [{}]
    f = ev[0].get("location", {}).get("file", "")
    m = re.search(r"aris_forensics/([^/]+)/", f)
    return m.group(1) if m else "(root)"


# 台账的 skill 字段记的是取证各专项的模块名；收据里只写中性的专项名，按模块名首段映射。
# 首段不在表内就报错，不猜。
TRACK = {"citation": "文献", "experiment": "实验", "proof": "推导", "consistency": "一致性"}


def track(skill: str) -> str:
    head = skill.split("-", 1)[0]
    if head not in TRACK:
        raise KeyError(f"台账里出现未登记的专项：{skill}")
    return TRACK[head]


def disputed_values(item):
    """抽出这条 finding 所争的『值对』：先剥掉 DOI 与 URL（它们自带数字，会污染），
    再取描述里出现的 4 位年份与 5--7 位条目号。选这个而不是关键词，是因为关键词会被
    『title, authors, journal, volume, issue, pages 都对得上』这类陪衬句带偏——
    实测四条 donoho 的关键词集合互不相同，而它们所争的值对完全一样（2008 vs 2009）。"""
    d = item["finding_snapshot"].get("description", "")
    d = re.sub(r"10\.\d{4,9}/[^\s,;)\"]+", "", d)
    d = re.sub(r"https?://\S+", "", d)
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", d))
    nums = set(re.findall(r"\b\d{5,7}\b", d))
    return tuple(sorted(years | nums))


def dois(item):
    d = item["finding_snapshot"].get("description", "")
    return tuple(sorted({x.rstrip(".") for x in re.findall(r"10\.\d{4,9}/[^\s,;)\"]+", d)}))


def receipt_A():
    out = {}
    for name, probe in ORACLE_PROBE.items():
        p = MS / name
        lines = p.read_text(encoding="utf-8").split("\n")
        hit = [(i + 1, l) for i, l in enumerate(lines) if probe in l]
        assert len(hit) == 1, f"{name}: 探针命中 {len(hit)} 次，收据不成立"
        ln, text = hit[0]
        out[name] = {
            "sha256": sha256(p),
            "line": ln,
            "span_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "span_head": text[:220],
            "刻意声明": ("这是刻意的" in text) or ("That is deliberate" in text),
        }
    return {
        "obligation_id": "a9930fd7d82bdc86fe7b8adb",
        "台账状态": "OPEN / UNRESOLVED_DISAPPEARANCE",
        "本轮处置": "补带哈希的收据，正式登记『判不成立』；不自行改台账状态",
        "判据": "稿面在报出该组半合成结果之前即声明使用 oracle 层 Ψ(θ₀)，且说明理由（受检对象是定理的相律）",
        "定位": out,
        "AGREES": all(v["刻意声明"] for v in out.values()),
    }


def receipt_B(items):
    by_doi = defaultdict(list)
    for it in items:
        d = dois(it)
        if d:
            by_doi[(it["skill"], it["pattern_id"], d)].append(it)
    groups = []
    rejected = []
    for k, v in by_doi.items():
        if len({sandbox(x) for x in v}) < 2:
            continue
        fields = {disputed_values(x) for x in v}
        entry = {
                    "判据": "同 skill + 同 pattern_id + 同 DOI + 跨沙箱 + 所争值对完全一致",
                    "skill": track(k[0]),
                    "pattern_id": k[1],
                    "doi": list(k[2]),
                    "所争值对": sorted({f for fs in fields for f in fs}),
                    "members": [
                        {"obligation_id": x["obligation_id"], "status": x["status"],
                         "severity": x["severity"], "sandbox": sandbox(x),
                         "title": x["finding_snapshot"]["title"]}
                        for x in sorted(v, key=sandbox)
                    ],
        }
        if len(fields) == 1:
            groups.append(entry)
        else:
            entry["判据"] = "第一步候选通过、第二步被否：组内各条所争值对不同，属同一文献的不同缺陷"
            entry["各条所争值对"] = {x["obligation_id"]: list(disputed_values(x)) for x in v}
            rejected.append(entry)
    # 无 DOI 的：跨沙箱标题相似度
    nod = [it for it in items if not dois(it)]
    seen = set()
    for a in range(len(nod)):
        for b in range(a + 1, len(nod)):
            A, B = nod[a], nod[b]
            if (A["skill"], A["pattern_id"]) != (B["skill"], B["pattern_id"]):
                continue
            if sandbox(A) == sandbox(B):
                continue
            r = difflib.SequenceMatcher(
                None, A["finding_snapshot"]["title"], B["finding_snapshot"]["title"]
            ).ratio()
            if r >= 0.75 and (A["obligation_id"], B["obligation_id"]) not in seen:
                seen.add((A["obligation_id"], B["obligation_id"]))
                groups.append(
                    {
                        "判据": "同 skill + 同 pattern_id + 标题相似度>=0.75 + 跨沙箱",
                        "similarity": round(r, 3),
                        "members": [
                            {"obligation_id": x["obligation_id"], "status": x["status"],
                             "severity": x["severity"], "sandbox": sandbox(x),
                             "title": x["finding_snapshot"]["title"]}
                            for x in (A, B)
                        ],
                    }
                )
    near_miss = []
    for k, v in by_doi.items():
        if len({sandbox(x) for x in v}) >= 2:
            continue
        if len(v) > 1:
            near_miss.append({"doi": list(k[2]), "n": len(v), "reason": "同沙箱，不属路径重复"})
    return {
        "机制": "取证门禁的 fingerprint 计算把 evidence.location.file 的绝对路径算进义务身份",
        "复现条件": "同一份稿件在不同轮次的取证沙箱目录下重跑取证，"
                    "同一条 finding 会因路径不同而算出不同 fingerprint，被开成新的 obligation_id；"
                    "旧条目则以 UNRESOLVED_DISAPPEARANCE 的形态留在台账上",
        "本轮不修上游": True,
        "重复组": groups,
        "候选但被第二步否掉（同文献不同缺陷，非路径重复）": rejected,
        "同沙箱多条（非路径重复）": near_miss,
    }


def main() -> int:
    led = json.loads(LEDGER.read_text())
    items = led["obligations"]
    A = receipt_A()
    B = receipt_B(items)
    opens = [i for i in items if i["status"] == "OPEN"]
    C = [
        {
            "obligation_id": i["obligation_id"],
            "severity": i["severity"],
            "skill": track(i["skill"]),
            "title": i["finding_snapshot"]["title"],
            "本轮处置": DISPOSITION.get(i["obligation_id"], ("未登记", ""))[0],
            "说明": DISPOSITION.get(i["obligation_id"], ("", "未登记"))[1],
        }
        for i in opens
    ]
    rec = {
        "任务": "RYA-579",
        "台账文件": str(LEDGER.relative_to(ROOT)),
        "台账文件_sha256": sha256(LEDGER),
        "台账计数": {"total": len(items),
                     **{s: sum(1 for i in items if i["status"] == s) for s in ("RESOLVED", "WAIVED", "OPEN")}},
        "A_a9930fd7收据": A,
        "B_fingerprint重复组": B,
        "C_六条OPEN本轮处置": C,
    }
    assert len(C) == 6 and all(c["本轮处置"] != "未登记" for c in C), "OPEN 条目与处置表对不上"
    dst = ROOT / "04outputs" / "143rya579_ledger_receipts.json"
    dst.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print("A a9930fd7 收据 AGREES =", A["AGREES"], "|", {k: v["line"] for k, v in A["定位"].items()})
    print("B 重复组 %d 组：" % len(B["重复组"]))
    for g in B["重复组"]:
        print("   ", g.get("doi", g.get("similarity")), "->",
              [(m["obligation_id"][:8], m["status"], m["sandbox"]) for m in g["members"]])
    print("B 候选被否 %d 组：" % len(B["候选但被第二步否掉（同文献不同缺陷，非路径重复）"]))
    for g in B["候选但被第二步否掉（同文献不同缺陷，非路径重复）"]:
        print("   ", g["doi"], "->", g["各条所争值对"])
    print("C 六条 OPEN 处置已登记")
    print("->", dst.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
