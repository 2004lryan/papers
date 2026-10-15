"""
70_band_attribution.py: 逐采集年的判别波段归因，并复核对分析窗口两端与平滑的依赖。

为什么要逐年做：
    产地类别与采集年、仪器部分重合（甘肃只在 2025 年的成像反射数据里，陕西只在 2018/2019 年的光纤透射
    数据里）。三年合在一起训练的模型可以借仪器与测量方式的差异区分产地，其重要波段因而混有仪器差异。
    每个采集年内只有一种仪器、一种测量方式，产地之间的差异不再与仪器重合。
为什么要复核窗口两端：
    平滑在裁剪到 590–1001 nm 之后进行，15 点 Savitzky-Golay 窗口在成像仪网格上约 25 nm，窗口两端各约
    12.6 nm 由边界多项式给出；硅探测器在 1000 nm 附近响应下降。重要性若集中在两端，须排除这些处理与
    仪器因素。

设计：
    年份     2018、2019（光纤透射，每果一条谱）与 2025（成像反射，每果 3–5 个面）各自单独建模；
             类别为该年的申报产地
    划分     按苹果编号分组、按产地分层的 5 折（StratifiedGroupKFold，打乱），5 个正式种子；
             同一果实的各个面始终落在同一侧
    模型     随机森林（300 棵树，类别平衡权重），在训练折上拟合，只在留出折上评估与归因
    重要性   留出折上的分块置换重要性：每块 5 个相邻波段（成像仪网格上约 9 nm），块内各列用同一置换
             一起打乱，重复 3 次，记宏 F1 的下降量
    条件     canonical      冻结数据（全部光谱，现行 15 点 SG）
             Q0 / Q3        严格档质控后，现行 SG / 不平滑（66 号脚本写出的变体，两者只差平滑）
             canonical_trim 冻结数据去掉两端各 15 nm（只留 606–986 nm）
    边缘占比  正重要性落在 591–605 nm 与 986–1001 nm 两段的份额

运行方式:
    cd <项目根>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/70_band_attribution.py

输出文件:
    04outputs/70_band_attribution.xlsx — 表a 逐年逐条件宏 F1（种子×折）；表b 逐年逐条件分块重要性；
                                        表c 边缘占比与前五块；compute
    05logs/70_band_attribution_YY-MM-DD_HHMMSS.log

说明:
    随机森林与置换重要性只用 CPU；本脚本不训练编码器，结论只关于光谱空间里哪些波段区分申报产地。
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import export_utils as eu

NDArray = npt.NDArray[Any]
YEARS = (2018, 2019, 2025)
BLOCK = 5
EDGE_LO, EDGE_HI = 605.0, 986.0  # 591–605 nm 与 986–1001 nm 记为窗口两端
TRIM = (606.0, 986.0)


def load_condition(cond: str, logger: eu.Logger) -> pd.DataFrame:
    if cond in ("canonical", "canonical_trim"):
        return eu.load_all_elements_df(verify_sha256=True)
    m64 = importlib.import_module("64_leakage_controls")
    return m64.load_data(cond, logger)


def block_importance(
    rf: RandomForestClassifier, x: NDArray, y: NDArray, blocks: list[NDArray], rng: np.random.Generator, reps: int
) -> tuple[float, NDArray]:
    base = float(f1_score(y, rf.predict(x), average="macro"))
    drops = np.zeros(len(blocks))
    for b, cols in enumerate(blocks):
        acc = []
        for _ in range(reps):
            xp = x.copy()
            xp[:, cols] = x[rng.permutation(len(x))][:, cols]
            acc.append(base - float(f1_score(y, rf.predict(xp), average="macro")))
        drops[b] = float(np.mean(acc))
    return base, drops


def main() -> None:
    parser = argparse.ArgumentParser(description="70_band_attribution：逐年判别波段与窗口两端复核")
    parser.add_argument("--conditions", default="canonical,Q0,Q3,canonical_trim")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()

    logger = eu.get_logger("70_band_attribution")
    seeds = [int(s) for s in eu.RECOMMENDED_SEEDS]
    conds = [c for c in args.conditions.split(",") if c]
    eu.log_experiment_header(
        logger,
        {
            "任务": "逐年判别波段归因 + 窗口两端/平滑复核",
            "条件": conds,
            "种子": seeds,
            "折数": args.folds,
            "树数": args.n_estimators,
            "分块": f"{BLOCK} 个相邻波段",
            "置换重复": args.reps,
        },
    )
    f1_rows: list[dict[str, Any]] = []
    imp_rows: list[dict[str, Any]] = []
    with eu.track_compute("逐年波段归因", logger, None) as budget:
        for cond in conds:
            df = load_condition(cond, logger)
            wl_cols = eu.get_spectrum_columns(df)
            wl = eu.get_spectrum_wavelengths(df, wl_cols)
            keep: NDArray = np.ones(len(wl), dtype=bool)
            if cond == "canonical_trim":
                keep = (wl >= TRIM[0]) & (wl <= TRIM[1])
            cols = [c for c, k in zip(wl_cols, keep, strict=True) if k]
            wl_k = wl[keep]
            blocks: list[NDArray] = [np.arange(i, min(i + BLOCK, len(cols))) for i in range(0, len(cols), BLOCK)]
            for year in YEARS:
                sub = df[df["年份"] == year]
                x = sub[cols].to_numpy(dtype=np.float32)
                y = sub["产地"].astype(str).to_numpy()
                g = sub["苹果编号"].astype(str).to_numpy()
                drops_all: list[NDArray] = []
                for seed in seeds:
                    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
                    rng = np.random.default_rng([seed, year, 70])
                    for k, (tr, te) in enumerate(cv.split(x, y, g)):
                        rf = RandomForestClassifier(
                            n_estimators=args.n_estimators,
                            class_weight="balanced",
                            random_state=seed + k,
                            n_jobs=args.n_jobs,
                        )
                        rf.fit(x[tr], y[tr])
                        base, drops = block_importance(rf, x[te], y[te], blocks, rng, args.reps)
                        drops_all.append(drops)
                        f1_rows.append({"条件": cond, "年份": year, "种子": seed, "折": k, "宏F1": round(base, 4)})
                mean_drop = np.mean(drops_all, axis=0)
                sd_drop = np.std(drops_all, axis=0, ddof=1)
                for b, idx in enumerate(blocks):
                    imp_rows.append(
                        {
                            "条件": cond,
                            "年份": year,
                            "块": b,
                            "起nm": round(float(wl_k[idx[0]]), 2),
                            "止nm": round(float(wl_k[idx[-1]]), 2),
                            "中心nm": round(float(wl_k[idx].mean()), 2),
                            "宏F1下降": round(float(mean_drop[b]), 5),
                            "SD": round(float(sd_drop[b]), 5),
                        }
                    )
                f1y = [r["宏F1"] for r in f1_rows if r["条件"] == cond and r["年份"] == year]
                logger.log(
                    f"  [{cond} {year}] 宏F1={np.mean(f1y):.4f}±{np.std(f1y, ddof=1):.4f}（{len(f1y)} 个种子×折）"
                )

    f1_df = pd.DataFrame(f1_rows)
    imp_df = pd.DataFrame(imp_rows)
    summ: list[dict[str, Any]] = []
    for (cond, year), sub in imp_df.groupby(["条件", "年份"]):
        pos = sub["宏F1下降"].clip(lower=0)
        edge = (sub["止nm"] <= EDGE_LO) | (sub["起nm"] >= EDGE_HI)
        top = sub.sort_values("宏F1下降", ascending=False).head(5)
        f1y = f1_df[(f1_df["条件"] == cond) & (f1_df["年份"] == year)]["宏F1"]
        summ.append(
            {
                "条件": cond,
                "年份": year,
                "宏F1均值": round(float(f1y.mean()), 4),
                "宏F1_SD": round(float(f1y.std(ddof=1)), 4),
                "边缘块数": int(edge.sum()),
                "边缘正重要性占比": round(float(pos[edge].sum() / pos.sum()), 4) if pos.sum() > 0 else float("nan"),
                "边缘波段占比": round(float(edge.mean()), 4),
                "前五块中心nm": ", ".join(f"{v:.0f}" for v in top["中心nm"]),
                "前五块下降": ", ".join(f"{v:.4f}" for v in top["宏F1下降"]),
            }
        )
    summ_df = pd.DataFrame(summ)
    logger.log("\n" + summ_df.to_string(index=False))
    out = eu.write_script_workbook(
        __file__,
        {
            0: ("逐年逐条件宏F1", f1_df),
            1: ("逐年逐条件分块重要性", imp_df),
            2: ("边缘占比与前五块", summ_df),
            "compute": pd.DataFrame([budget]),
        },
    )
    logger.log(f"结果表 → {out}")
    logger.close()


if __name__ == "__main__":
    main()
