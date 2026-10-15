#!/usr/bin/env python3
r"""
141 — RYA-579：把「哪些数字随种子变、哪些不变」机械算成一张可复核的收据。

为什么要有这一件
----------------
前四轮取证里，每一轮全新的敌意 reviewer 都会打开稿面所引的 54c5_realkinetics.json /
-designsweep.csv，发现里面的数字与稿面标注「5 种子均值」的数字对不上，然后开一条
critical。判定不成立是对的，但只判不修，下一轮还会再来一条。

根因不在 reviewer，在稿面：54c5_* 是【原生单次抽样】的运行产物（脚本每次运行只初始化
一个 RNG，本身不作多种子聚合），5 种子聚合另存于 105_seeds/ 与 105multiseed_realdata.json。

本脚本给出该分界的机械证据：把原生产物与 5 个逐种子产物逐键/逐列对齐，
判定每一个量是「设计侧（逐字节相同）」还是「自助派生（随种子变动）」。
稿面 §统计检验与误差报告 的产物指向约定即引用本产物。

输出 04outputs/141rya579_seedscope_receipt.json
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"
SEEDS = ROOT / "04outputs" / "105_seeds"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def leaves(d, prefix=""):
    if isinstance(d, dict):
        for k, v in sorted(d.items()):
            yield from leaves(v, f"{prefix}.{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, d


def partition_json():
    files = [OUT / "54c5_realkinetics.json"] + sorted(SEEDS.glob("54c5_realkinetics-seed*.json"))
    tables = [dict(leaves(json.loads(p.read_text()))) for p in files]
    common = set(tables[0])
    for t in tables[1:]:
        common &= set(t)
    invariant, varying = [], []
    for k in sorted(common):
        (invariant if all(t[k] == tables[0][k] for t in tables) else varying).append(k)
    return files, sorted(common), invariant, varying


def partition_csv():
    files = [OUT / "54c5_realkinetics-designsweep.csv"] + sorted(
        SEEDS.glob("54c5_realkinetics-designsweep-seed*.csv")
    )
    tables = [list(csv.DictReader(p.open())) for p in files]
    n = len(tables[0])
    assert all(len(t) == n for t in tables), "逐种子扫描表行数不一致，无法逐行对齐"
    cols = list(tables[0][0].keys())
    invariant, varying = [], []
    for c in cols:
        same = all(t[i][c] == tables[0][i][c] for t in tables for i in range(n))
        (invariant if same else varying).append(c)
    return files, n, cols, invariant, varying


def main() -> int:
    jf, jcommon, jinv, jvar = partition_json()
    cf, nrows, ccols, cinv, cvar = partition_csv()

    subtrees = sorted({k.split(".")[1] for k in jvar if k.count(".") >= 2})

    rec = {
        "任务": "RYA-579",
        "用途": "支撑稿面 §统计检验与误差报告 的『多种子聚合值的产物指向』约定",
        "口径": "原生单次抽样产物 vs 5 个逐种子产物逐键/逐列对齐；相同=设计侧，不同=自助派生",
        "seeds": [p.name.split("seed")[-1].split(".")[0] for p in jf[1:]],
        "json": {
            "files": [str(p.relative_to(ROOT)) for p in jf],
            "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in jf},
            "n_common_leaves": len(jcommon),
            "n_seed_invariant": len(jinv),
            "n_seed_varying": len(jvar),
            "seed_varying_keys": jvar,
            "seed_varying_subtrees": subtrees,
        },
        "csv": {
            "files": [str(p.relative_to(ROOT)) for p in cf],
            "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in cf},
            "n_rows": nrows,
            "columns": ccols,
            "seed_invariant_columns": cinv,
            "seed_varying_columns": cvar,
        },
        "结论": (
            f"设计侧逐字节相同：CSV {len(cinv)} 列、JSON {len(jinv)}/{len(jcommon)} 个数值键；"
            f"随种子变动：CSV {len(cvar)} 列、JSON {len(jvar)} 个键，全部落在 "
            + "、".join(subtrees)
            + " 子树下。"
        ),
    }
    dst = OUT / "141rya579_seedscope_receipt.json"
    dst.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(rec["结论"])
    print("CSV 不变列:", cinv)
    print("CSV 变动列:", cvar)
    print("JSON 变动键:", len(jvar), "子树:", subtrees)
    print("->", dst.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
