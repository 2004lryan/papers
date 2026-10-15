"""94_mango_data_facts.py: 芒果公开数据表的几项计数——补充材料 S9.9、S9.10 节引用的取样点、采收批与波长数。

为什么:
    S9.9 节写「5641 对相邻行的七个元数据字段与干物质值都相同，按同一取样点的两次扫描处理」，S9.10 节写
    「112 个采收批没有一个同时出现在两个产区」「用全部 306 个波长」。这几个数是数据表本身的性质，归纳值写在
    90、91 号的文件头里；91 号运行时只按大小核对取样点的个数（不符只告警），结果文件里没有记下它们。
    本脚本在全表上直接数出这些量，与稿件和文件头所写的值逐项核对，有一项不符即中止、不写结果表。

计数（全表，不抽样）:
    波长      表头里列名为纯数字的列（取列同 crossdataset_lib.load_mango 与 90 号）：个数、首末波长、相邻间隔
    取样点    Season、Region、Date、Type、Cultivar、Pop、Temp、DM 这 8 列全同的行归为一组（90 号 POINT_KEYS，
              即七个元数据字段加干物质）：组数，两行一组、一行一组、多于两行的组数，两行一组里两行在表内相邻的组数
    相邻同值  表内相邻两行干物质值相同的对数；这些行对与两行一组的取样点是否为同一批行
    采收批    Pop 列的取值个数、各采收季的个数；出现在两个及以上采收季、两个产区的采收批个数
    品种      品种个数、只在一个产区的品种个数、两个产区都有的品种

输入:
    <data_root>/041_mango_anderson_crossseason/NAnderson2020MendeleyMangoNIRData.csv（读入时核对 SHA-256，
    登记值取 90 号；元数据列按字符串读入，同 90 号）

输出:
    04outputs/94_mango_data_facts.xlsx
        表a 计数：项、数出的值、登记值、依据、是否一致
        compute：数据文件、SHA-256、行数、核对结果
    05logs/94_mango_data_facts_YY-MM-DD_HHMMSS.log

运行方式:
    cd <项目根>
    python 02code/94_mango_data_facts.py --data_root <公开数据根目录>

复用:
    90 号         MANGO_CSV、MANGO_SHA256、META_STR_COLS、POINT_KEYS、REGIONS、SHARED_CULTIVARS
    export_utils  compute_sha256、get_logger、log_experiment_header、track_compute、write_script_workbook
"""

from __future__ import annotations

import argparse
import importlib
import os
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

import export_utils as eu

m90: Any = importlib.import_module("90_mango_region_task")

POINT_KEYS: list[str] = list(m90.POINT_KEYS)
SHARED = ",".join(sorted(m90.SHARED_CULTIVARS))
# (项, 登记值, 依据)：稿件里写到的值，以及 90、91 号文件头里的归纳值
EXPECTED: tuple[tuple[str, Any, str], ...] = (
    ("行数", 11691, "91 号文件头"),
    ("波长个数", 306, "补充材料 S9.10 节"),
    ("首个波长（nm）", 285, "90 号文件头"),
    ("末个波长（nm）", 1200, "90 号文件头"),
    ("相邻波长间隔（nm）", "3", "90 号文件头"),
    ("取样点组数", 6050, "91 号文件头"),
    ("两行一组的取样点", 5641, "补充材料 S9.9 节"),
    ("一行一组的取样点", 409, "90、91 号文件头"),
    ("多于两行的取样点", 0, "91 号文件头"),
    ("两行一组且两行在表内相邻", 5641, "补充材料 S9.9 节"),
    ("表内相邻两行干物质相同的对数", 5641, "91 号文件头"),
    ("相邻同干物质的行对与两行一组的取样点相同", True, "91 号文件头"),
    ("采收批个数", 112, "补充材料 S9.10 节"),
    ("各采收季的采收批个数", "25,19,50,18", "91 号文件头"),
    ("出现在两个及以上采收季的采收批", 0, "91 号文件头"),
    ("出现在两个产区的采收批", 0, "补充材料 S9.10 节"),
    ("品种个数", 10, "补充材料 S9.10 节"),
    ("只在一个产区的品种个数", 6, "补充材料 S9.10 节"),
    ("两个产区都有的品种", SHARED, "补充材料 S9.10 节"),
)


def count_facts(df: pd.DataFrame) -> dict[str, Any]:
    """在全表上数出 EXPECTED 里的各项，键同 EXPECTED 的项名。"""
    out: dict[str, Any] = {"行数": len(df)}
    wl = [c for c in df.columns if c.isdigit()]
    steps = sorted({int(b) - int(a) for a, b in pairwise(wl)})
    out["波长个数"] = len(wl)
    out["首个波长（nm）"] = int(wl[0])
    out["末个波长（nm）"] = int(wl[-1])
    out["相邻波长间隔（nm）"] = ",".join(str(s) for s in steps)

    group = df.groupby(POINT_KEYS, sort=False, dropna=False).ngroup().to_numpy()
    size = np.bincount(group)
    out["取样点组数"] = len(size)
    out["两行一组的取样点"] = int((size == 2).sum())
    out["一行一组的取样点"] = int((size == 1).sum())
    out["多于两行的取样点"] = int((size > 2).sum())
    pos = np.arange(len(df))
    pairs_point: set[tuple[int, int]] = set()
    for g in np.flatnonzero(size == 2):
        i, j = pos[group == g]
        pairs_point.add((int(i), int(j)))
    out["两行一组且两行在表内相邻"] = sum(j - i == 1 for i, j in pairs_point)
    dm = df["DM"].to_numpy(dtype=float)
    pairs_dm = {(int(i), int(i) + 1) for i in np.flatnonzero(dm[:-1] == dm[1:])}
    out["表内相邻两行干物质相同的对数"] = len(pairs_dm)
    out["相邻同干物质的行对与两行一组的取样点相同"] = pairs_dm == pairs_point

    pops = df.groupby("Pop")
    out["采收批个数"] = int(df["Pop"].nunique())
    per_season = df.groupby("Season")["Pop"].nunique()
    out["各采收季的采收批个数"] = ",".join(str(int(per_season[s])) for s in sorted(per_season.index))
    out["出现在两个及以上采收季的采收批"] = int((pops["Season"].nunique() > 1).sum())
    out["出现在两个产区的采收批"] = int((pops["Region"].nunique() > 1).sum())

    by_region = [set(df.loc[df["Region"] == r, "Cultivar"].tolist()) for r in m90.REGIONS]
    every = set().union(*by_region)
    out["品种个数"] = len(every)
    out["只在一个产区的品种个数"] = sum(sum(c in s for s in by_region) == 1 for c in every)
    out["两个产区都有的品种"] = ",".join(sorted(set.intersection(*by_region)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="94_mango_data_facts：芒果公开数据表的取样点、采收批与波长计数")
    parser.add_argument("--data_root", default="05data")
    args = parser.parse_args()
    path = os.path.join(args.data_root, m90.MANGO_CSV)

    logger = eu.get_logger("94_mango_data_facts")
    eu.log_experiment_header(logger, {"任务": "芒果公开数据表的计数", "数据文件": path, "取样点的键": POINT_KEYS})
    with eu.track_compute("芒果数据计数", logger, None) as budget:
        sha = eu.compute_sha256(path)
        if sha != m90.MANGO_SHA256:
            raise SystemExit(f"芒果数据文件的 SHA-256 与数据卡登记值不符：{path}\n  实际值={sha}")
        df = pd.read_csv(path, dtype=dict.fromkeys(m90.META_STR_COLS, str))
        found = count_facts(df)
    rows = [
        {"项": item, "数出的值": found[item], "登记值": want, "依据": basis, "一致": str(found[item]) == str(want)}
        for item, want, basis in EXPECTED
    ]
    tab = pd.DataFrame(rows)
    logger.log("表a 计数\n" + tab.to_string(index=False))
    bad = tab[~tab["一致"]]
    if len(bad):
        logger.close()
        raise SystemExit(f"{len(bad)} 项与登记值不符，不写结果表：\n" + bad.to_string(index=False))
    out = eu.write_script_workbook(
        __file__,
        {
            0: ("计数", tab),
            "compute": pd.DataFrame(
                [{"数据文件": m90.MANGO_CSV, "SHA-256": sha, "行数": len(df), "核对": "全部一致", **budget}]
            ),
        },
    )
    logger.log(f"结果表 → {out}")
    logger.close()


if __name__ == "__main__":
    main()
