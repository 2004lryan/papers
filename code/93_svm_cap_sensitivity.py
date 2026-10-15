"""93_svm_cap_sensitivity.py: 83 号池化 SVM 迭代上限的灵敏度——同一个正式种子在两个上限下的逐格对照。

为什么:
    83 号的池化 SVM（StandardScaler + RBF 核 SVC）把 SMO 迭代上限设为 10^6 次，正式结果里有一部分留出域
    出现停在上限的拟合。停在上限的解与收敛的解可能不同，表 S22 的暴露增量与对等暴露下的排名可能因此带偏。
    本脚本读同一个正式种子在更高上限下的重跑（83 号原样运行，只改 --svm_max_iter），逐格对照池化 SVM 两臂
    的宏 F1、暴露增量与所选超参数。上限只作用于池化 SVM；其余路线（池化 PLS-DA、只用支持样本的参照、
    最近质心）用的是同一批情节，应当逐位相同，用来核对两次运行除上限外的设置一致。

输入:
    04outputs/formal_seeds/83_symmetric_exposure__seed<S>.xlsx   正式表（83 号缺省上限 10^6）
    04outputs/83_svmcap<C>__seed<S>.xlsx                          同一种子、上限 C 的重跑。83 号把结果写进
        formal_seeds/，在独立副本里跑完后改成这个名字放进 04outputs/：留在 formal_seeds/ 会被 83 号
        --aggregate 当成同一种子的又一张表读入（数值不同即中止）。

输出:
    04outputs/93_svm_cap_sensitivity.xlsx
        表a 逐留出域：两个上限下池化 SVM 无暴露臂、暴露臂的 F1 与暴露增量，增量之差，所选超参数是否相同，
            停在上限的拟合数（选参、评估）；其余路线逐格的最大绝对差
        表b 汇总：主口径（苹果两年设计 + 公开数据集）与全部留出域，各给增量之差的最大绝对值与均值、
            两个上限下的增量均值、各上限下有拟合停在上限的留出域数、所选超参数改变的留出域数、
            其余路线的最大绝对差
    05logs/93_svm_cap_sensitivity_YY-MM-DD_HHMMSS.log

自检（不过即中止，不写结果表）:
    两张表记的上限分别是 10^6 与 --cap；两次运行已评的留出域集合相同，每格各路线的情节数相同；
    其余路线逐格之差不超过 1e-12（否则两次运行的情节或数据不同，对照没有意义）。

运行方式:
    cd <项目根>
    python 02code/93_svm_cap_sensitivity.py [--seed 20060515] [--cap 10000000]
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Any

import numpy as np
import pandas as pd

import export_utils as eu

m77: Any = importlib.import_module("77_generality_tables")
m83: Any = importlib.import_module("83_symmetric_exposure")

MAIN_SETS: list[str] = list(m77.MAIN_SETS)
BASE_CAP = 1_000_000  # 83 号 --svm_max_iter 的缺省值，正式表用的就是它
SVM = "SVM"
R_FREE: str = m83.route_pool(SVM, m83.ARM_FREE)
R_EXPO: str = m83.route_pool(SVM, m83.ARM_EXPO)
CUT_COLS = ("停在上限的拟合数(选参)", "停在上限的拟合数(评估)")
KEYS = ["dataset", "held"]
TOL_OTHER = 1e-12
SCOPE_MAIN = "主口径（苹果两年设计 + 公开数据集）"
SCOPE_ALL = "全部留出域（另含苹果三年设计）"


def load_seed_table(path: str, cap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读 83 号的一张逐种子表：逐格结果（第 1 张表）与已评留出域的信息（第 2 张表），核对其中记的上限。"""
    if not os.path.exists(path):
        raise SystemExit(f"没有 {path}")
    res = pd.read_excel(path, sheet_name=0)
    info = pd.read_excel(path, sheet_name=1)
    info = info[info["状态"] == m83.STATUS_OK].reset_index(drop=True)
    caps = sorted({int(c) for c in info["SVM迭代上限"]})
    if caps != [cap]:
        raise SystemExit(f"{os.path.basename(path)} 记的 SVM 迭代上限是 {caps}，应为 {cap}")
    keep = info[KEYS].astype(str).apply(tuple, axis=1)
    res = res[res[KEYS].astype(str).apply(tuple, axis=1).isin(set(keep))]
    return res.reset_index(drop=True), info


def svm_row(res: pd.DataFrame, info: pd.DataFrame, ds: str, held: str) -> dict[str, Any]:
    """一个留出域上池化 SVM 两臂的 F1、增量、所选超参数与停在上限的拟合数。"""
    sub = res[(res["dataset"].astype(str) == ds) & (res["held"].astype(str) == held)]
    free = sub[sub["路线"] == R_FREE]
    expo = sub[sub["路线"] == R_EXPO]
    if len(free) != 1 or len(expo) != 1:
        raise SystemExit(f"{ds}/{held}：池化 SVM 两臂应各有一行，实际 {len(free)}、{len(expo)}")
    f, e = float(free["F1"].iloc[0]), float(expo["F1"].iloc[0])
    inf = info[(info["dataset"].astype(str) == ds) & (info["held"].astype(str) == held)].iloc[0]
    return {
        "无暴露": f,
        "暴露": e,
        "增量": e - f,
        "所选超参数": str(free["所选超参数"].iloc[0]),
        **{c: int(inf[c]) for c in CUT_COLS},
    }


def other_routes_gap(lo: pd.DataFrame, hi: pd.DataFrame) -> pd.DataFrame:
    """上限不作用的路线：逐格 |F1 之差|，并核对情节数相同。"""
    cols = [*KEYS, "路线", "F1", "情节数"]
    a = lo[~lo["路线"].isin([R_FREE, R_EXPO])][cols]
    b = hi[~hi["路线"].isin([R_FREE, R_EXPO])][cols]
    m = a.merge(b, on=[*KEYS, "路线"], suffixes=("_lo", "_hi"), how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        raise SystemExit("两次运行的路线或留出域不同：\n" + m[m["_merge"] != "both"][[*KEYS, "路线"]].to_string())
    if (m["情节数_lo"] != m["情节数_hi"]).any():
        raise SystemExit("两次运行同一格的情节数不同")
    m["F1差"] = (m["F1_hi"] - m["F1_lo"]).abs()
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="93_svm_cap_sensitivity：83 号池化 SVM 迭代上限的灵敏度")
    parser.add_argument("--seed", type=int, default=int(eu.RECOMMENDED_SEEDS[0]), help="重跑所用的正式种子")
    parser.add_argument("--cap", type=int, default=10_000_000, help="重跑的 --svm_max_iter")
    args = parser.parse_args()
    if args.seed not in eu.RECOMMENDED_SEEDS or args.cap <= BASE_CAP:
        parser.error(f"--seed 须为正式种子 {eu.RECOMMENDED_SEEDS} 之一，--cap 须大于 {BASE_CAP}")

    lo_path = os.path.join(m83.SEED_DIR, f"{m83.OUT_STEM}__seed{args.seed}.xlsx")
    hi_path = os.path.join(eu.OUTPUT_DIR, f"83_svmcap{args.cap}__seed{args.seed}.xlsx")
    logger = eu.get_logger("93_svm_cap_sensitivity")
    eu.log_experiment_header(
        logger,
        {
            "任务": "83 号池化 SVM 迭代上限的灵敏度",
            "种子": args.seed,
            "上限": f"{BASE_CAP} 对 {args.cap}",
            "正式表": lo_path,
            "重跑表": hi_path,
            "主口径": MAIN_SETS,
        },
    )
    lo_res, lo_info = load_seed_table(lo_path, BASE_CAP)
    hi_res, hi_info = load_seed_table(hi_path, args.cap)
    doms_lo = set(lo_info[KEYS].astype(str).apply(tuple, axis=1))
    doms_hi = set(hi_info[KEYS].astype(str).apply(tuple, axis=1))
    if doms_lo != doms_hi:
        only_lo, only_hi = sorted(doms_lo - doms_hi), sorted(doms_hi - doms_lo)
        raise SystemExit(f"两次运行已评的留出域不同：只在正式表 {only_lo}，只在重跑 {only_hi}")
    gap = other_routes_gap(lo_res, hi_res)
    worst = float(gap["F1差"].max())
    logger.log(f"上限不作用的路线 {len(gap)} 格，|F1 之差| 最大 {worst:.3g}")
    if worst > TOL_OTHER:
        raise SystemExit(f"上限不作用的路线有格不同（最大差 {worst:.3g}）：两次运行的情节或数据不一致，对照无意义")

    rows: list[dict[str, Any]] = []
    for ds, held in sorted(doms_lo):
        a = svm_row(lo_res, lo_info, ds, held)
        b = svm_row(hi_res, hi_info, ds, held)
        g = gap[(gap["dataset"].astype(str) == ds) & (gap["held"].astype(str) == held)]
        rows.append(
            {
                "dataset": ds,
                "held": held,
                "主口径": ds in MAIN_SETS,
                **{f"{k}_{BASE_CAP}": v for k, v in a.items()},
                **{f"{k}_{args.cap}": v for k, v in b.items()},
                "增量差": b["增量"] - a["增量"],
                "无暴露差": b["无暴露"] - a["无暴露"],
                "暴露差": b["暴露"] - a["暴露"],
                "所选超参数相同": a["所选超参数"] == b["所选超参数"],
                "其余路线最大差": float(g["F1差"].max()),
            }
        )
    tab_a = pd.DataFrame(rows)

    def summary(scope: str, sub: pd.DataFrame) -> dict[str, Any]:
        def capped(cap: int) -> int:
            return int(((sub[f"{CUT_COLS[0]}_{cap}"] > 0) | (sub[f"{CUT_COLS[1]}_{cap}"] > 0)).sum())

        return {
            "口径": scope,
            "种子": args.seed,
            "留出域数": len(sub),
            "增量差_最大绝对值": float(sub["增量差"].abs().max()),
            "增量差_均值": float(sub["增量差"].mean()),
            "无暴露差_最大绝对值": float(sub["无暴露差"].abs().max()),
            "暴露差_最大绝对值": float(sub["暴露差"].abs().max()),
            f"增量均值_{BASE_CAP}": float(sub[f"增量_{BASE_CAP}"].mean()),
            f"增量均值_{args.cap}": float(sub[f"增量_{args.cap}"].mean()),
            f"有拟合停在上限的留出域_{BASE_CAP}": capped(BASE_CAP),
            f"有拟合停在上限的留出域_{args.cap}": capped(args.cap),
            "所选超参数改变的留出域": int((~sub["所选超参数相同"]).sum()),
            "其余路线最大差": float(sub["其余路线最大差"].max()),
        }

    tab_b = pd.DataFrame([summary(SCOPE_MAIN, tab_a[tab_a["主口径"]]), summary(SCOPE_ALL, tab_a)])
    view = [
        *KEYS,
        f"增量_{BASE_CAP}",
        f"增量_{args.cap}",
        "增量差",
        "所选超参数相同",
        f"{CUT_COLS[1]}_{BASE_CAP}",
        f"{CUT_COLS[1]}_{args.cap}",
    ]
    logger.log("表a 逐留出域\n" + tab_a[view].to_string(index=False))
    logger.log("表b 汇总\n" + tab_b.T.to_string())
    out = eu.write_script_workbook(
        __file__,
        {
            0: ("逐留出域", tab_a),
            1: ("汇总", tab_b),
            "compute": pd.DataFrame(
                [
                    {
                        "种子": args.seed,
                        "上限": f"{BASE_CAP},{args.cap}",
                        "正式表": os.path.basename(lo_path),
                        "重跑表": os.path.basename(hi_path),
                        "其余路线格数": len(gap),
                        "其余路线最大差": worst,
                        "自检": "通过",
                    }
                ]
            ),
        },
    )
    logger.log(f"结果表 → {out}")
    logger.close()
    if not np.isfinite(tab_b["增量差_最大绝对值"]).all():
        raise SystemExit("汇总里有缺值：先核对两张逐种子表")


if __name__ == "__main__":
    main()
