"""
71_source_only_tuning.py: 只在源年份上为 K-shot SVM 与 PLS-DA 选超参数，复核无泄漏方法比较的先后次序。

为什么：
    方法比较里 SVM（C=10、核宽按 scale 规则）与 PLS-DA（潜变量数取类别数）的超参数在各情节间固定。
    在目标年上调参会把目标信息带进来，但固定值也可能让某个方法欠调。能不泄漏地调参的只有源年份：
    把每个源年份当作「伪目标年」，按同一 K-shot 协议抽情节，选出宏 F1 最高的超参数，再冻结到目标年。

设计：
    折           LOEO-2018 / 2019 / 2025，源年份与 20 号脚本 FOLD_CFG 相同
    K            1, 5, 10, 20, 50
    候选         SVM 线性核：C ∈ {10^-4, 10^-3, …, 10^3}（十倍步长）
                 SVM RBF 核：C ∈ {0.1, 1, …, 10^6} × 核宽 ∈ {0.001, 0.01, 0.1, 1, 10} × scale 规则值
                 RBF 核在核宽→0、C→∞ 时趋于线性核，两类核都放进候选，所选值才能落在网格内部；
                 线性核排在前面，源年份宏 F1 并列时取线性核
                 PLS-DA：潜变量数 ∈ {1, 2, 3, 5, 8, 12, 16, 24}（受 3K−1 截断）
                 所选值落在网格外沿（C 或核宽取到本核网格的两端、PLS-DA 取到 24）的格数写入表 b，
                 用来判断网格是否够宽
    选参         每个源年份抽 20 个伪目标情节（种子与评估情节不重叠），两源年份宏 F1 取均值，取最高者；
                 并列时取候选序列里靠前（更简单）的一个
    评估         目标年上用 20 号脚本 Formal 面板的同一组情节：正式种子 S 的第 i 次重采样种子为 S+i，
                 i = 0..49；逐情节与固定超参数配对比较
    自检         固定超参数（C=10、scale 核宽；潜变量数 3）时复算 20 号 Formal 归档：PLS-DA 逐格差 ≤ 5e-5
                 （归档保留 4 位小数）；SVM 按五种子均值比，差 ≤ 5e-4——libsvm 在 Linux 与 macOS 上对并列
                 决策值的取舍不同，逐格可差到 1e-3 量级，与 65 号脚本自检的容差相同

运行方式:
    cd <项目根>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/71_source_only_tuning.py

输出文件:
    04outputs/71_source_only_tuning.xlsx — 表a 逐种子逐折逐 K 的所选超参数与宏 F1；表b 三折均值汇总
                                          （与原型网络 R 的先后次序）；表c 自检；compute
    05logs/71_source_only_tuning_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import ast
import importlib
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import export_utils as eu
from protonet import build_year_region_data

NDArray = npt.NDArray[Any]
m20 = importlib.import_module("20_unified_singleface_kshot")
K_LIST = (1, 5, 10, 20, 50)
LIN_C_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3)
C_GRID = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
G_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)  # 乘在 scale 规则值 1/(n_features·Var(X)) 上
SVM_GRID: tuple[tuple[Any, ...], ...] = (
    *(("linear", c) for c in LIN_C_GRID),
    *(("rbf", c, g) for c in C_GRID for g in G_GRID),
)
SVM_FIXED = ("rbf", 10.0, 1.0)  # 20 号 Formal 面板的固定设置
PLS_GRID = (1, 2, 3, 5, 8, 12, 16, 24)
TUNE_EPISODES = 20
TUNE_OFFSET = 100_000  # 伪目标情节的种子偏移，与评估情节 S+i（i<50）不重叠


def episode(year_data: Any, year: int, k: int, seed: int) -> tuple[list[str], NDArray, NDArray, NDArray, list[str]]:
    """与 20 号 _eval_sklearn 同序抽样：先支持集各果一面，再查询集各果一面。"""
    rng = np.random.default_rng(seed)
    regions, support, query = m20.build_support_query(year_data, year, k, rng)
    idx = {r: i for i, r in enumerate(regions)}
    x_tr, y_tr = [], []
    for reg in regions:
        for aid in support[reg]:
            x_tr.append(m20._one_face(year_data[year][reg][aid], rng))
            y_tr.append(idx[reg])
    x_q, y_t = [], []
    for aid, true_reg in query:
        x_q.append(m20._one_face(year_data[year][true_reg][aid], rng))
        y_t.append(true_reg)
    return regions, m20.snv(np.stack(x_tr)), np.array(y_tr), m20.snv(np.stack(x_q)), y_t


def fit_predict(kind: str, param: Any, regions: list[str], x_tr: NDArray, y_tr: NDArray, x_q: NDArray) -> list[str]:
    if kind == "svm":
        scaler = StandardScaler().fit(x_tr)
        xs = scaler.transform(x_tr)
        if param[0] == "linear":
            svm = SVC(kernel="linear", C=param[1]).fit(xs, y_tr)
        else:
            _, c, g = param
            # gamma="scale" 即 1/(n_features·Var(xs))；g 为其倍数，g=1 时与 20 号的 Pipeline 数值相同
            gamma: Any = "scale"
            if g != 1.0:
                var = float(xs.var())
                gamma = g / (xs.shape[1] * var) if var > 0 else g / xs.shape[1]
            svm = SVC(kernel="rbf", C=c, gamma=gamma).fit(xs, y_tr)
        return [regions[int(i)] for i in svm.predict(scaler.transform(x_q))]
    n_comp = max(1, min(len(x_tr) - 1, x_tr.shape[1], int(param)))
    y_oh = np.zeros((len(x_tr), len(regions)))
    y_oh[np.arange(len(x_tr)), y_tr] = 1.0
    pls = PLSRegression(n_components=n_comp, max_iter=500)
    pls.fit(x_tr, y_oh)
    return [regions[int(s.argmax())] for s in pls.predict(x_q)]


def score(year_data: Any, year: int, k: int, seed: int, kind: str, param: Any) -> float:
    """与 20 号相同：只有一类或拟合失败时退回最近质心。"""
    regions, x_tr, y_tr, x_q, y_t = episode(year_data, year, k, seed)
    if len(np.unique(y_tr)) < 2:
        return float(m20.eval_nearest_centroid(year_data, year, k, seed)[0])
    try:
        y_p = fit_predict(kind, param, regions, x_tr, y_tr, x_q)
    except Exception:
        return float(m20.eval_nearest_centroid(year_data, year, k, seed)[0])
    return float(m20._macro_and_perclass(y_t, y_p, regions)[0])


def on_edge(param: str) -> bool:
    """所选超参数是否取到网格外沿：C 或核宽取到本核网格的两端，PLS-DA 取到最大潜变量数。"""
    val = ast.literal_eval(param)  # 表 a 里写的是 str(tuple) 或 str(int)
    if isinstance(val, tuple) and val[0] == "linear":
        return val[1] in (LIN_C_GRID[0], LIN_C_GRID[-1])
    if isinstance(val, tuple):
        _, c, g = val
        return c in (C_GRID[0], C_GRID[-1]) or g in (G_GRID[0], G_GRID[-1])
    return bool(val == PLS_GRID[-1])


def pick_best(grid: tuple[Any, ...], tune: dict[Any, float]) -> Any:
    """源年份宏 F1 最高的候选；并列时保留候选序列里靠前（更简单）的一个。"""
    best = grid[0]
    for p in grid[1:]:
        if tune[p] > tune[best]:
            best = p
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="71_source_only_tuning：源年份内选超参数后的无泄漏方法比较")
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()
    logger = eu.get_logger("71_source_only_tuning")
    seeds = [int(s) for s in eu.RECOMMENDED_SEEDS]
    eu.log_experiment_header(
        logger,
        {
            "任务": "源年份内选超参数（SVM 的 C、PLS-DA 潜变量数），冻结后在目标年评估",
            "K": K_LIST,
            "SVM 线性核候选 C": LIN_C_GRID,
            "SVM RBF 核候选 C": C_GRID,
            "SVM RBF 核候选核宽倍数": G_GRID,
            "PLS-DA 候选潜变量数": PLS_GRID,
            "每源年份伪目标情节": TUNE_EPISODES,
            "目标年评估重采样": args.reps,
            "正式种子": seeds,
        },
    )
    df = eu.load_all_elements_df(verify_sha256=True)
    year_data = build_year_region_data(df, eu.get_spectrum_columns(df))
    formal = importlib.import_module("42_ensemble_kshot").load_formal_panel()
    if formal is None:
        raise FileNotFoundError("缺 20 号 Formal 面板（formal_seeds/20_unified_singleface_kshot__seed*.xlsx）")
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    with eu.track_compute("源年份选参", logger, None) as budget:
        for seed in seeds:
            for tgt, cfg in sorted(m20.FOLD_CFG.items()):
                for k in K_LIST:
                    grids: tuple[tuple[str, tuple[Any, ...], Any], ...] = (
                        ("svm", SVM_GRID, SVM_FIXED),
                        ("pls", PLS_GRID, 3),
                    )
                    for kind, grid, fixed in grids:
                        tune = {
                            p: float(
                                np.mean(
                                    [
                                        score(year_data, src, k, seed + TUNE_OFFSET + j, kind, p)
                                        for src in cfg["src_years"]
                                        for j in range(TUNE_EPISODES)
                                    ]
                                )
                            )
                            for p in grid
                        }
                        best = pick_best(grid, tune)
                        eval_seeds = [seed + i for i in range(args.reps)]
                        f_best = [score(year_data, tgt, k, s, kind, best) for s in eval_seeds]
                        f_fixed = [score(year_data, tgt, k, s, kind, fixed) for s in eval_seeds]
                        rows.append(
                            {
                                "种子": seed,
                                "目标年份": tgt,
                                "K": k,
                                "方法": "SVM-Kshot" if kind == "svm" else "PLS-DA-Kshot",
                                "所选超参数": str(best),
                                "源年份宏F1(所选)": round(tune[best], 4),
                                "目标宏F1(所选)": float(np.mean(f_best)),
                                "目标宏F1(固定)": float(np.mean(f_fixed)),
                            }
                        )
                        ref = formal[
                            (formal["种子"] == seed)
                            & (formal["目标年份"] == tgt)
                            & (formal["K"] == k)
                            & (formal["方法"] == rows[-1]["方法"])
                        ]
                        if len(ref) == 1:
                            checks.append(
                                {
                                    "种子": seed,
                                    "目标年份": tgt,
                                    "K": k,
                                    "方法": rows[-1]["方法"],
                                    "本脚本(固定)": float(np.mean(f_fixed)),
                                    "20号Formal": float(ref["F1均值"].iloc[0]),
                                }
                            )
                logger.log(f"  种子 {seed} 折 {tgt} 完成")
    res = pd.DataFrame(rows)
    chk = pd.DataFrame(checks)
    chk["差"] = (chk["本脚本(固定)"] - chk["20号Formal"]).abs()
    pls_max = float(chk[chk["方法"] == "PLS-DA-Kshot"]["差"].max())
    svm_chk = chk[chk["方法"] == "SVM-Kshot"].groupby(["目标年份", "K"])[["本脚本(固定)", "20号Formal"]].mean()
    svm_mean = float((svm_chk["本脚本(固定)"] - svm_chk["20号Formal"]).abs().max())
    logger.log(f"自检：{len(chk)} 格；PLS-DA 逐格最大差 {pls_max:.2e}；SVM 五种子均值最大差 {svm_mean:.2e}")
    # 三折均值：先种子内三折均值，再对种子求均值；原型网络取 20 号 Formal 面板的 VanillaProtoNet
    pn = formal[formal["方法"] == "VanillaProtoNet"]
    summ: list[dict[str, Any]] = []
    for k in K_LIST:
        row: dict[str, Any] = {"K": k}
        for meth in ("SVM-Kshot", "PLS-DA-Kshot"):
            sub = res[(res["K"] == k) & (res["方法"] == meth)]
            for col, tag in (("目标宏F1(所选)", "所选"), ("目标宏F1(固定)", "固定")):
                row[f"{meth}_{tag}"] = round(float(sub.groupby("种子")[col].mean().mean()), 4)
            row[f"{meth}_所选超参数"] = "; ".join(
                f"{v}×{n}" for v, n in sub["所选超参数"].value_counts().sort_index().items()
            )
            row[f"{meth}_落在网格外沿"] = int(sub["所选超参数"].map(on_edge).sum())
        pk = pn[pn["K"] == k]
        row["原型网络R"] = round(float(pk.groupby("种子")["F1均值"].mean().mean()), 4)
        row["原型网络−SVM(所选)"] = round(row["原型网络R"] - row["SVM-Kshot_所选"], 4)
        row["原型网络−SVM(固定)"] = round(row["原型网络R"] - row["SVM-Kshot_固定"], 4)
        summ.append(row)
    summ_df = pd.DataFrame(summ)
    logger.log("\n" + summ_df.to_string(index=False))
    out = eu.write_script_workbook(
        __file__,
        {
            0: ("逐种子逐折逐K", res),
            1: ("三折均值汇总", summ_df),
            2: ("自检", chk),
            "compute": pd.DataFrame([budget]),
        },
    )
    logger.log(f"结果表 → {out}")
    logger.close()
    if pls_max > 5.0001e-5 or svm_mean > 5e-4:
        raise SystemExit("自检失败：固定超参数下与 20 号 Formal 归档不一致")


if __name__ == "__main__":
    main()
