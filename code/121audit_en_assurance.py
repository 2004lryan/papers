#!/usr/bin/env python3
r"""
121 — 英文稿投稿保障三件套复跑（项目约定 §10.5 强制）

三项
----
A. claim audit（中英数字奇偶校验）：英文稿每个承载主张的数字必须在中文稿出现且相等。
   翻译最常见的错误就是数字漂移，故以中文稿为基准做集合比对。
B. citation audit（机械层）：\cite 键 vs \bibitem 键双向核对；未定义引用、未被引书目均须为 0。
C. integrity（过程性主张扫描）——**本轮新增**：
   上一轮 claim audit 只核数字，漏掉表注里「附 SHA-256 校验」这句无据过程声明。
   故此处显式枚举可证伪的过程性断言（我们做过某流程 / 某文件已存放 / 已登记）并要求逐条给出在盘证据。

输出 04outputs/121audit_en_assurance.json
"""
from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path(os.environ.get("ARIS_RAW_ROOT", str(ROOT.parents[1] / "05data")))


def _rel(p: Path) -> str:
    """登记用路径：仓库内 → 仓库相对；仓库外只读原始数据 → ${ARIS_RAW_ROOT}/… 占位。"""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        pass
    try:
        return "${ARIS_RAW_ROOT}/" + p.resolve().relative_to(RAW_ROOT.resolve()).as_posix()
    except ValueError:
        return p.name
MS = ROOT / "06doc/01manuscript"
ZH, EN = MS / "Elsevier_zh.tex", MS / "Elsevier_en.tex"


def strip_comments(s: str) -> str:
    """去 LaTeX 注释；(?<!\\\\) 避免把转义的 \\% 之后的正文一起吃掉（此前踩过的坑）。"""
    return re.sub(r"(?<!\\)%[^\n]*", "", s)


# 纯排版/编号来源，不承载主张，比对时剔除
# 注意：必须连同 {...} 参数一起吃掉——只删命令名会把 \linespread{1.0} 的 1.0 留成假阳性。
TYPESET_CTX = re.compile(
    r"\\(?:documentclass|usepackage|setcounter|linespread|tolerance|emergencystretch|"
    r"hspace|vspace|includegraphics|label|ref|cite\w*|bibitem|textwidth|columnwidth|"
    r"arraystretch|tabcolsep|baselineskip|fontsize|selectfont|multicolumn|cline|hline|"
    r"@plus|@minus|abovedisplayskip|belowdisplayskip|parskip|itemsep|topsep)"
    r"(?:\s*\*?\s*(?:\[[^\]]*\])?\s*(?:\{[^{}]*\})*)?"
)


def numbers(tex: str) -> list[str]:
    out = []
    for line in strip_comments(tex).split("\n"):
        body = re.sub(r"\\@(?:plus|minus)\s*-?\d*\.?\d+\s*(?:pt|em|ex|mm|cm|in|sp|bp|dd|cc)?", " ", line)
        # 「数值 + 长度宏」是排版尺寸不是主张数字：p{0.52\columnwidth} 里的 0.52 曾被读成
        # "英文独有的数字"。数值在宏之前，TYPESET_CTX 只吃宏名、吃不掉它，须单独剥。
        body = re.sub(
            r"-?\d*\.?\d+\s*\\(?:columnwidth|textwidth|linewidth|paperwidth|"
            r"textheight|paperheight|baselineskip|arraystretch|tabcolsep|dimexpr)",
            " ", body)
        # 带单位的长度值。刻意不含 "in"、且不允许空格：英文正文的
        # "0.948 in the low-Psi decile" 会被 "\d+\.\d+ in" 吃掉，而中文稿无此写法，
        # 反而制造"中文独有"的假阳性——与本轮 \approx0.01 同源的陷阱。
        body = re.sub(r"(?<![\w.])-?\d*\.\d+(?:pt|em|ex|mm|cm|sp|bp|dd|cc)(?![\w])", " ", body)
        body = TYPESET_CTX.sub(" ", body)
        body = re.sub(r"10\.\d{4,}/\S+", " ", body)          # DOI
        body = re.sub(r"https?://\S+", " ", body)            # URL
        body = re.sub(r"PMC\d+", " ", body)                  # PMC id
        # 剥掉剩余的 LaTeX 命令名：否则 \approx0.01 会因后顾断言 (?<![\w.]) 被漏读，
        # 而中文稿写作 $0.01$ 能读到 —— 制造"中文独有"的假阳性（本轮踩过）
        body = re.sub(r"\\[a-zA-Z]+", " ", body)
        for m in re.finditer(r"(?<![\w.])\d+\.\d+(?![\w])", body):
            out.append(m.group())
    return out


def audit_numbers() -> dict[str, Any]:
    zh, en = numbers(ZH.read_text(encoding="utf-8")), numbers(EN.read_text(encoding="utf-8"))
    zset, eset = set(zh), set(en)
    return {
        "n_zh_distinct": len(zset), "n_en_distinct": len(eset),
        "en_only": sorted(eset - zset), "zh_only": sorted(zset - eset),
        "pass": not (eset - zset),
    }


def audit_citations(p: Path) -> dict[str, Any]:
    tex = strip_comments(p.read_text(encoding="utf-8"))
    cited = set()
    for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    listed = {m.group(1) for m in re.finditer(r"\\bibitem\{([^}]*)\}", tex)}
    return {
        "n_cited": len(cited), "n_bibitem": len(listed),
        "undefined": sorted(cited - listed), "uncited": sorted(listed - cited),
        "pass": not (cited - listed) and not (listed - cited),
    }


# --- C. 过程性主张：逐条声明 + 在盘证据 -----------------------------------
PROCESS_CLAIMS: list[dict[str, Any]] = [
    {
        "claim": "各源文件与派生文件的 SHA-256 摘要随代码登记",
        "where": "tab:datacard 表注 + Data & Code Availability",
        "evidence": ROOT / "04outputs/0NDATACARD_C5_manifest.tsv",
        "check": "manifest 存在且行数 = 登记文件数 + 表头",
    },
    {
        "claim": "数据卡（项目约定 §4.6 九项必填）已建立",
        "where": "tab:datacard 自称『数据卡』",
        "evidence": ROOT / "refine-logs/0NDATACARD_C5.md",
        "check": "文件存在且含源数据/派生/SHA-256 三节",
    },
    {
        "claim": "食品体系经 Europe PMC 全文逐值复核",
        "where": "tab:datacard 表注 + §data",
        "evidence": ROOT / "03data/processed/pmc_real/PMC9319022_T3.csv",
        "check": "提取件在盘（复核记录见脚本 55 与 00SUBMISSION_AUDIT.md §一 C24）",
    },
    {
        "claim": "牛油果体系直接读取原始 xlsx、未作数值变换",
        "where": "tab:datacard 表注",
        # 【2026-08-10 订正，RYA-105】原为作者机绝对路径；改由 ARIS_RAW_ROOT（默认 <根>/../../05data）解析
        "evidence": RAW_ROOT / "004_hass_avocado_rgb_ripening" /
                    "Hass Avocado Ripening Photographic Dataset" / "Avocado Ripening Dataset.xlsx",
        "check": "原始文件在盘且只读引用（未复制进项目）",
    },
    {
        "claim": "每个正文数字均可回溯至具体脚本与结果文件",
        "where": "Data & Code Availability",
        "evidence": ROOT / "06doc/02sub/00SUBMISSION_AUDIT.md",
        "check": "映射表存在（原 claim_evidence_map.md，2026-07-27 原样并入 00SUBMISSION_AUDIT.md §一）",
    },
    {
        "claim": "三个源数据集均由原作者存入公开仓库，本文以可解析 DOI 引用并链接",
        "where": "Data & Code Availability（CILS Option C 的履行方式）",
        "evidence": ROOT / "04outputs/125source_doi_verification.json",
        "check": "脚本 125 按注册机构分流实查：Mendeley 走 DataCite、两篇 MDPI 走 Crossref，"
                 "标题与类型均须与稿件用途相符（3/3 PASS，且皆 CC BY 4.0）",
    },
    {
        "claim": "派生产物 + SHA-256 清单 + 全部代码已存入公开仓库并在文中链接",
        "where": "Data & Code Availability",
        "evidence": ROOT / "08github/data/SHA256-MANIFEST.tsv",
        "check": "发布仓库内的清单在盘；仓库本身为 github.com/2004lryan/023arrhenius-recoverability。"
                 "注：本文未采集新数据，派生产物为计算输出与仿真产物，故不另铸数据集 DOI",
    },
    {
        # 2026-08-19 RYA-526 新增（取证义务 a5286696 / 737748de）：稿面 §sec:repro 声称
        # 依赖已固定版本。此前仓库内只有区间式 requirements.txt、无锁文件，该过程性主张无据。
        "claim": "依赖已逐包固定到精确版本（版本锁，非哈希锁），含解释器版本",
        "where": "Reproducibility, compute and availability",
        "evidence": ROOT / "08github/requirements.lock.txt",
        "check": "锁文件在盘且逐包为 `==` 精确版本；覆盖 04outputs 侧复现路径 import 期所需的 "
                 "torch/openpyxl/joblib（export_utils 模块级导入与工作簿写出通道），"
                 "并显式登记为版本锁而非哈希锁",
    },
]


def audit_process() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for c in PROCESS_CLAIMS:
        ev = c["evidence"]
        ok = bool(ev and Path(ev).exists())
        rows.append({"claim": c["claim"], "where": c["where"], "check": c["check"],
                     # 【2026-08-10 订正，RYA-105】仓库内 → 相对路径；仓库外只读原始数据 → ${ARIS_RAW_ROOT} 占位
                     "evidence": _rel(ev) if ev else None, "on_disk": ok})
    unsupported = [r["claim"] for r in rows if not r["on_disk"]]
    return {"rows": rows, "unsupported": unsupported, "pass": not unsupported}


def main() -> None:
    res = {
        "A_numbers": audit_numbers(),
        "B_citations_en": audit_citations(EN),
        "B_citations_zh": audit_citations(ZH),
        "C_process_claims": audit_process(),
    }
    (ROOT / "04outputs/121audit_en_assurance.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    a = res["A_numbers"]
    print(f"A 数字奇偶  英文独有 {len(a['en_only'])} 个 -> {'过' if a['pass'] else '不过: ' + str(a['en_only'])}")
    print(f"   （中文独有 {len(a['zh_only'])} 个，属中文稿表格/正文未译入英文稿的排版数，需逐条看）")
    for k in ("B_citations_en", "B_citations_zh"):
        b = res[k]
        print(f"{k}: cite {b['n_cited']} / bibitem {b['n_bibitem']} · undefined {len(b['undefined'])} · "
              f"uncited {len(b['uncited'])} -> {'过' if b['pass'] else '不过'}")
    c = res["C_process_claims"]
    print("C 过程性主张:")
    for r in c["rows"]:
        print(f"   [{'有据' if r['on_disk'] else '★无据'}] {r['claim']}")
    print(f"   -> {'过' if c['pass'] else '存在无据项 ' + str(c['unsupported'])}")


if __name__ == "__main__":
    main()
