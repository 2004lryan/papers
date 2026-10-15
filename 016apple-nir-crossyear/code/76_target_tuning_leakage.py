"""76_target_tuning_leakage.py: 在目标年上为 K-shot SVM 与 PLS-DA 选超参数——超参数选择的评估泄漏（苹果两种设计）。

为什么:
    71 号只在源年份上选参（把每个源年份当伪目标，无泄漏）。更常见的做法是在包含留出年份的数据上交叉验证，
    选潜变量数或 C 与核宽，留出年份带标签的光谱因此参与了选参。本脚本在与 71 号逐位相同的评估情节上，
    把选参的伪目标换成目标年本身：选参情节抽自目标年同一批果实（种子偏移 TUNE_OFFSET，与评估情节的
    种子不重叠，但被评估的果实会出现在选参情节里），与源年份选参按情节配对比较。
    并补同仪器两年设计的源年份选参：LOEO-2018 只用 2019 选参，LOEO-2019 只用 2018 选参。

设计:
    折     三年设计 LOEO-2018/2019/2025（源年份同 71 号）；两年设计 LOEO-2018（源 2019）、LOEO-2019（源 2018）
    K      1, 5, 10, 20, 50（与 71 号相同）
    候选   与 71 号相同（SVM 线性核与 RBF 核网格；PLS-DA 潜变量数 1–24，受 3K−1 截断）
    选参   源年份：每个源年份 20 个伪目标情节；目标年：目标年上 20 个选参情节；取宏 F1 最高者，并列取靠前者
    评估   目标年上 71 号同一组 50 个情节（正式种子 S 的第 i 次为 S+i）
    目标年上的评估情节只由目标年、K 与种子决定，与设计无关，故目标年选参在两种设计下相同，只算一次
    自检   三年设计的源年份选参逐格复现 71 号正式结果（所选超参数一致，目标宏 F1 差 ≤ 5e-4）

运行方式:
    cd <项目根>
    python 02code/76_target_tuning_leakage.py --seed 20060515      # 每个正式种子一次（纯 CPU）
    python 02code/76_target_tuning_leakage.py --aggregate            # 五个种子都有之后汇总并自检

输出文件:
    04outputs/formal_seeds/76_target_tuning__seed<S>.xlsx — 逐设计逐折逐 K 的所选超参数与目标宏 F1
    04outputs/76_target_tuning_leakage.xlsx — 汇总：长表、按 K 的三折/两折均值、超参数泄漏、自检
    05logs/76_target_tuning_leakage_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
from typing import Any

import numpy as np
import pandas as pd

import export_utils as eu
from protonet import build_year_region_data

m20 = importlib.import_module("20_unified_singleface_kshot")
m71 = importlib.import_module("71_source_only_tuning")

DESIGN_SRC: dict[str, dict[int, list[int]]] = {
    "three_year": {tgt: list(cfg["src_years"]) for tgt, cfg in m20.FOLD_CFG.items()},
    "two_year": {2018: [2019], 2019: [2018]},
}
KINDS: dict[str, tuple[str, tuple[Any, ...]]] = {
    "SVM-Kshot": ("svm", m71.SVM_GRID),
    "PLS-DA-Kshot": ("pls", m71.PLS_GRID),
}
SEED_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
REF_71 = os.path.join(eu.OUTPUT_DIR, "71_source_only_tuning.xlsx")


def tune(year_data: Any, years: list[int], k: int, seed: int, kind: str, grid: tuple[Any, ...]) -> dict[Any, float]:
    """各候选在给定年份的选参情节上的宏 F1 均值（年份 × 20 个情节）。"""
    return {
        p: float(
            np.mean(
                [
                    m71.score(year_data, y, k, seed + m71.TUNE_OFFSET + j, kind, p)
                    for y in years
                    for j in range(m71.TUNE_EPISODES)
                ]
            )
        )
        for p in grid
    }


def target_f1(
    year_data: Any, tgt: int, k: int, eval_seeds: list[int], kind: str, param: Any, cache: dict[str, float]
) -> float:
    """所选超参数在目标年评估情节上的宏 F1 均值（同一 (折, K, 方法) 内按超参数缓存）。"""
    key = str(param)
    if key not in cache:
        cache[key] = float(np.mean([m71.score(year_data, tgt, k, s, kind, param) for s in eval_seeds]))
    return cache[key]


def run_seed(seed: int, reps: int, logger: eu.Logger) -> pd.DataFrame:
    df = eu.load_all_elements_df(verify_sha256=True)
    year_data = build_year_region_data(df, eu.get_spectrum_columns(df))
    rows: list[dict[str, Any]] = []
    targets = sorted({t for d in DESIGN_SRC.values() for t in d})
    for tgt in targets:
        for k in m71.K_LIST:
            eval_seeds = [seed + i for i in range(reps)]
            for meth, (kind, grid) in KINDS.items():
                cache: dict[str, float] = {}
                t_tgt = tune(year_data, [tgt], k, seed, kind, grid)
                best_tgt = m71.pick_best(grid, t_tgt)
                for design, src_map in DESIGN_SRC.items():
                    if tgt not in src_map:
                        continue
                    t_src = tune(year_data, src_map[tgt], k, seed, kind, grid)
                    best_src = m71.pick_best(grid, t_src)
                    rows.append(
                        {
                            "种子": seed,
                            "设计": design,
                            "目标年份": tgt,
                            "K": k,
                            "方法": meth,
                            "源年份所选": str(best_src),
                            "源年份选参宏F1": round(t_src[best_src], 4),
                            "目标年所选": str(best_tgt),
                            "目标年选参宏F1": round(t_tgt[best_tgt], 4),
                            "目标宏F1(源年份选参)": target_f1(year_data, tgt, k, eval_seeds, kind, best_src, cache),
                            "目标宏F1(目标年选参)": target_f1(year_data, tgt, k, eval_seeds, kind, best_tgt, cache),
                        }
                    )
            logger.log(f"  种子 {seed} 留出 {tgt} K={k} 完成")
    return pd.DataFrame(rows)


def aggregate(logger: eu.Logger) -> None:
    files = sorted(f for f in glob.glob(os.path.join(SEED_DIR, "76_target_tuning__seed*.xlsx")) if "__pilot" not in f)
    if not files:
        raise FileNotFoundError("没有 76 号的逐种子结果")
    res = pd.concat([pd.read_excel(f, sheet_name=0) for f in files], ignore_index=True)
    seeds = sorted(res["种子"].unique().tolist())
    if set(seeds) != set(eu.RECOMMENDED_SEEDS):
        logger.log(f"  [WARN] 种子集合 {seeds} 不是 Formal 5 种子")
    res["超参数泄漏"] = res["目标宏F1(目标年选参)"] - res["目标宏F1(源年份选参)"]

    # 自检：三年设计的源年份选参 vs 71 号正式结果
    ref = pd.read_excel(REF_71, sheet_name=0)
    mine = res[res["设计"] == "three_year"].merge(
        ref[["种子", "目标年份", "K", "方法", "所选超参数", "目标宏F1(所选)"]], on=["种子", "目标年份", "K", "方法"]
    )
    same_param = int((mine["源年份所选"] == mine["所选超参数"]).sum())
    diff = (mine["目标宏F1(源年份选参)"] - mine["目标宏F1(所选)"]).abs()
    chk = pd.DataFrame(
        [
            {
                "比对格数": len(mine),
                "所选超参数一致": same_param,
                "目标宏F1最大差": float(diff.max()),
                "PLS-DA 最大差": float(diff[mine["方法"] == "PLS-DA-Kshot"].max()),
            }
        ]
    )
    logger.log(f"自检（三年设计源年份选参 vs 71 号）：\n{chk.to_string(index=False)}")

    summ: list[dict[str, Any]] = []
    for (design, meth, k), sub in res.groupby(["设计", "方法", "K"]):
        per_seed = sub.groupby("种子")[["目标宏F1(源年份选参)", "目标宏F1(目标年选参)", "超参数泄漏"]].mean()
        leak = per_seed["超参数泄漏"]
        summ.append(
            {
                "设计": design,
                "方法": meth,
                "K": k,
                "源年份选参": round(float(per_seed["目标宏F1(源年份选参)"].mean()), 4),
                "目标年选参": round(float(per_seed["目标宏F1(目标年选参)"].mean()), 4),
                "超参数泄漏": round(float(leak.mean()), 4),
                "超参数泄漏为正种子数": int((leak > 0).sum()),
                "逐折超参数泄漏": "; ".join(
                    f"{t}:{v:+.4f}" for t, v in sub.groupby("目标年份")["超参数泄漏"].mean().items()
                ),
            }
        )
    summ_df = pd.DataFrame(summ)
    logger.log("\n" + summ_df.to_string(index=False))
    out = os.path.join(eu.OUTPUT_DIR, "76_target_tuning_leakage.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summ_df.to_excel(writer, sheet_name=eu.table_sheet_name(0, "按K汇总"), index=False)
        res.to_excel(writer, sheet_name=eu.table_sheet_name(1, "逐种子逐折逐K"), index=False)
        chk.to_excel(writer, sheet_name=eu.table_sheet_name(2, "自检"), index=False)
    logger.log(f"汇总表 → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="76_target_tuning_leakage：目标年选参的评估泄漏")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    logger = eu.get_logger("76_target_tuning_leakage")
    if args.aggregate:
        aggregate(logger)
        logger.close()
        return
    formal = args.seed in [int(s) for s in eu.RECOMMENDED_SEEDS]
    eu.log_experiment_header(
        logger,
        {
            "任务": "SVM 与 PLS-DA 在目标年上选参（泄漏）与在源年份上选参（无泄漏）的配对比较",
            "设计": {d: dict(m) for d, m in DESIGN_SRC.items()},
            "K": m71.K_LIST,
            "每年份选参情节": m71.TUNE_EPISODES,
            "评估重采样": args.reps,
            "种子": args.seed,
            "阶段": "formal（纯 CPU，sklearn 确定性）" if formal else "pilot",
        },
    )
    with eu.track_compute("目标年选参", logger, None) as budget:
        res = run_seed(args.seed, args.reps, logger)
    os.makedirs(SEED_DIR, exist_ok=True)
    suffix = "" if formal else "__pilot"
    out = os.path.join(SEED_DIR, f"76_target_tuning__seed{args.seed}{suffix}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        res.to_excel(writer, sheet_name=eu.table_sheet_name(0, "逐设计逐折逐K"), index=False)
        pd.DataFrame([{"seed": args.seed, **budget}]).to_excel(writer, sheet_name="compute", index=False)
    logger.log(f"结果表 → {out}")
    logger.close()


if __name__ == "__main__":
    main()
