"""
73_grid_edge_check.py: 71 号选参落在网格外沿时，把网格再向外延伸两个数量级，看选参目标与目标年结果会不会变。

为什么：
    71 号 SVM 有相当一部分所选超参数落在 RBF 网格的角上（C 最大、核宽最小）。沿 C·核宽 不变的方向，
    RBF 核 SVM 趋于线性核 SVM，而线性核已在候选里；但这是极限论证，不能替代实测。
    这里对每个落在外沿的格，沿它所在的外沿方向外推 1–2 个数量级，在同一批源年份伪目标情节上重算选参目标；
    外推候选胜出时换用它，再在目标年同一批 50 个情节上重算宏 F1。若三折均值几乎不变，网格对结论就够宽。

设计：
    格           71 号表 a 中所选超参数落在网格外沿的格（种子 × 折 × K × 方法）
    外推候选     线性核 C 在两端：C÷10、C÷100 或 C×10、C×100
                 RBF 核 C 在两端：同上；核宽倍数在两端：同上；C 最大且核宽最小（角）时再加沿脊线的
                 (C×10, 核宽÷10)、(C×100, 核宽÷100)
                 PLS-DA 取到 24：潜变量数 32、48（受 3K−1 截断，去重）
    选参         与 71 号相同：两源年份各 20 个伪目标情节（种子 S+100000+j）宏 F1 取均值；外推候选须严格
                 高于原所选值才换
    评估         换用外推候选的格，在目标年 50 个情节（种子 S+i）上重算宏 F1；未换的格沿用 71 号的值
    自检         重算的原所选值选参目标须与 71 号表 a 记录值一致（≤ 1e-4，表 a 保留 4 位小数）

运行方式:
    cd <项目根>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/73_grid_edge_check.py

输出文件:
    04outputs/73_grid_edge_check.xlsx — 表a 逐格外推结果；表b 按方法与 K 汇总（换用格数、三折均值变化）；compute
    05logs/73_grid_edge_check_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import ast
import importlib
import os
from typing import Any

import numpy as np
import pandas as pd

import export_utils as eu
from protonet import build_year_region_data

m20 = importlib.import_module("20_unified_singleface_kshot")
m71: Any = importlib.import_module("71_source_only_tuning")


def extensions(kind: str, param: Any, k: int) -> list[Any]:
    """沿所选值所在外沿方向外推 1–2 个数量级的候选。"""
    if kind == "pls":
        cap = 3 * k - 1
        return sorted({min(cap, v) for v in (32, 48)} - {int(param)})
    out: list[Any] = []
    if param[0] == "linear":
        c = param[1]
        if c == m71.LIN_C_GRID[0]:
            out += [("linear", c / 10), ("linear", c / 100)]
        if c == m71.LIN_C_GRID[-1]:
            out += [("linear", c * 10), ("linear", c * 100)]
        return out
    _, c, g = param
    lo_c, hi_c = c == m71.C_GRID[0], c == m71.C_GRID[-1]
    lo_g, hi_g = g == m71.G_GRID[0], g == m71.G_GRID[-1]
    if hi_c:
        out += [("rbf", c * 10, g), ("rbf", c * 100, g)]
    if lo_c:
        out += [("rbf", c / 10, g), ("rbf", c / 100, g)]
    if lo_g:
        out += [("rbf", c, g / 10), ("rbf", c, g / 100)]
    if hi_g:
        out += [("rbf", c, g * 10), ("rbf", c, g * 100)]
    if hi_c and lo_g:
        out += [("rbf", c * 10, g / 10), ("rbf", c * 100, g / 100)]
    return out


def objective(year_data: Any, src_years: list[int], k: int, seed: int, kind: str, param: Any) -> float:
    """与 71 号相同的选参目标：两源年份各 20 个伪目标情节的宏 F1 均值。"""
    return float(
        np.mean(
            [
                m71.score(year_data, src, k, seed + m71.TUNE_OFFSET + j, kind, param)
                for src in src_years
                for j in range(m71.TUNE_EPISODES)
            ]
        )
    )


def main() -> None:
    logger = eu.get_logger("73_grid_edge_check")
    eu.log_experiment_header(
        logger,
        {
            "任务": "71 号选参落在网格外沿的格：外推两个数量级后重选、重评",
            "输入": "04outputs/71_source_only_tuning.xlsx 表a",
            "目标年评估重采样": 50,
        },
    )
    t71 = pd.read_excel(os.path.join(eu.OUTPUT_DIR, "71_source_only_tuning.xlsx"), sheet_name=0)
    df = eu.load_all_elements_df(verify_sha256=True)
    year_data = build_year_region_data(df, eu.get_spectrum_columns(df))
    edge = t71[t71["所选超参数"].map(m71.on_edge)].reset_index(drop=True)
    n_svm = int((edge["方法"] == "SVM-Kshot").sum())
    logger.log(f"外沿格：{len(edge)}（SVM {n_svm}，PLS-DA {len(edge) - n_svm}）")
    rows: list[dict[str, Any]] = []
    with eu.track_compute("网格外推", logger, None) as budget:
        for _, r in edge.iterrows():
            kind = "svm" if r["方法"] == "SVM-Kshot" else "pls"
            seed, tgt, k = int(r["种子"]), int(r["目标年份"]), int(r["K"])
            src = m20.FOLD_CFG[tgt]["src_years"]
            chosen = ast.literal_eval(r["所选超参数"])
            base = objective(year_data, src, k, seed, kind, chosen)
            cands = extensions(kind, chosen, k)
            ext = {p: objective(year_data, src, k, seed, kind, p) for p in cands}
            best, best_val = chosen, base
            for p in cands:
                if ext[p] > best_val:
                    best, best_val = p, ext[p]
            if best != chosen:
                f_new = float(np.mean([m71.score(year_data, tgt, k, seed + i, kind, best) for i in range(50)]))
            else:
                f_new = float(r["目标宏F1(所选)"])
            rows.append(
                {
                    "种子": seed,
                    "目标年份": tgt,
                    "K": k,
                    "方法": r["方法"],
                    "原所选": str(chosen),
                    "原选参目标(71号记录)": float(r["源年份宏F1(所选)"]),
                    "原选参目标(重算)": round(base, 4),
                    "外推候选数": len(cands),
                    "外推最高选参目标": round(max(ext.values()), 4) if ext else np.nan,
                    "外推后所选": str(best),
                    "是否换用": best != chosen,
                    "目标宏F1(原所选)": float(r["目标宏F1(所选)"]),
                    "目标宏F1(外推后)": f_new,
                }
            )
            logger.log(f"  种子 {seed} 折 {tgt} K={k} {r['方法']}：{chosen} → {best}")
    res = pd.DataFrame(rows)
    dev = float((res["原选参目标(重算)"] - res["原选参目标(71号记录)"]).abs().max())
    logger.log(f"自检：重算原所选值的选参目标与 71 号记录的最大差 {dev:.2e}")
    if dev > 1e-4:
        raise SystemExit("重算的选参目标与 71 号不一致，外推结果不可用")
    # 三折均值：未落在外沿的格沿用 71 号值，外沿格用外推后的值
    full = t71.copy()
    key = ["种子", "目标年份", "K", "方法"]
    full = full.merge(res[[*key, "目标宏F1(外推后)"]], on=key, how="left")
    full["目标宏F1(外推后)"] = full["目标宏F1(外推后)"].fillna(full["目标宏F1(所选)"])
    summ: list[dict[str, Any]] = []
    for (meth, k), sub in full.groupby(["方法", "K"]):
        before = float(sub.groupby("种子")["目标宏F1(所选)"].mean().mean())
        after = float(sub.groupby("种子")["目标宏F1(外推后)"].mean().mean())
        e = res[(res["方法"] == meth) & (res["K"] == k)]
        summ.append(
            {
                "方法": meth,
                "K": k,
                "外沿格数": len(e),
                "换用格数": int(e["是否换用"].sum()),
                "三折均值(71号)": round(before, 4),
                "三折均值(外推后)": round(after, 4),
                "变化": round(after - before, 4),
            }
        )
    summ_df = pd.DataFrame(summ)
    logger.log("\n" + summ_df.to_string(index=False))
    out = eu.write_script_workbook(
        __file__,
        {0: ("逐格外推", res), 1: ("按方法与K汇总", summ_df), "compute": pd.DataFrame([budget])},
    )
    logger.log(f"结果表 → {out}")
    logger.close()


if __name__ == "__main__":
    main()
