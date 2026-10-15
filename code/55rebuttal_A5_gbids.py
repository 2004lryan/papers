# -*- coding: utf-8 -*-
"""
55rebuttal_A5_gbids.py: 在 47 正式结果的 25 个外层折上重算附录 A5 的 LLH-GBID 综合评价指数。

目的：按协议 B9 锁定的四指标效用化、两级等权、(q_B, q_W)=(2, 2) 与 u_R²≥0.5 软否决，
对六个方法逐折计算 GBID/GBIDS，并按 B10.2 用 47 原文的修正重复 k 折 t 检验与 Holm 校正作 15 对比较。

输入：由 results47.resolve() 选中的 47 正式结果标记，以及该标记记录的六法 45 汇总与逐样本预测 CSV。
脚本不读取 05data；每个种子的完整真值向量由其 5 个外层测试折的预测文件按 idx 还原，训练折真值为补集。

输出（原子替换）：
  04outputs/55rebuttal_A5_folds.csv   — 每方法 × 种子 × 外层折的指标、效用、GBID/GBIDS、Pass
  04outputs/55rebuttal_A5_summary.csv — 25 折汇总、平均排名、第一名频率与 q 敏感性排名
  04outputs/55rebuttal_A5_pairs.csv   — 六法 15 对修正重复 k 折检验与 Holm 结论
  04outputs/55rebuttal_A5_diag.csv    — 四项指标的截断前上下截断比例
  05logs/55rebuttal_A5_gbids_<yy-mm-dd_HHMMSS>.log — 完整运行日志

口径：R² 为效益型 I=1、L=0；MAE/P90 为成本型 I=0、L=同折训练均值预测基准；bias 为目标型
T=0、tau±=0.25·s_y_train、delta±=0。W=四维等权 1/4，每维单指标 v=1；GBIDS=100(1−GBID)。

运行方式：
    cd <项目根目录>
    python3 02code/55rebuttal_A5_gbids.py

只读 02code 与 04outputs；除上述四个 CSV 与日志外不写任何文件。不并行，不使用 GPU。
"""

from __future__ import annotations

import itertools
import ast
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

sys.path.insert(0, "02code")
from results47 import Source47, fold_metrics45, resolve  # noqa: E402
from results54 import load47  # noqa: E402

ALPHA_TAU = 0.25
MAIN_Q = (2, 2)
Q_PAIRS = ((2, 2), (3, 2), (2, 3), (3, 3))
METRICS = ("R2", "MAE", "P90", "BIAS", "base_MAE", "base_P90", "s_y_train")
UTILITY_NAMES = ("R2", "MAE", "P90", "BIAS")
W = np.full(4, 0.25)
V = [np.ones(1) for _ in range(4)]
DIM_GROUPS = [np.array([i], dtype=int) for i in range(4)]
PAIR_COLUMNS = [
    "family", "pair", "n_pairs", "diff", "se_corr", "t", "df", "p", "ci_lo", "ci_hi", "a_wins",
    "shapiro_p", "wilcoxon_p", "holm_p", "family_size", "结论",
]
OUTPUT_NAMES = {
    "folds": "55rebuttal_A5_folds.csv",
    "summary": "55rebuttal_A5_summary.csv",
    "pairs": "55rebuttal_A5_pairs.csv",
    "diag": "55rebuttal_A5_diag.csv",
}


def load_gbid_functions(script: str = "02code/export_utils.py") -> tuple[Callable, Callable, Callable]:
    """从 export_utils.py 原文抽取三项锁定实现，避免导入该通用模块的绘图依赖与缓存副作用。"""
    with open(script, encoding="utf-8") as source_file:
        tree = ast.parse(source_file.read(), filename=script)
    wanted = {"compute_utility", "compute_target_utility", "compute_gbid"}
    functions: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    constants: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_GBID_EPS" for target in node.targets
        ):
            constants.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_GBID_EPS":
            constants.append(node)
    if {node.name for node in functions if isinstance(node, ast.FunctionDef)} != wanted or len(constants) != 1:
        raise ValueError(f"{script} 里没有唯一的 _GBID_EPS 与三项 GBID 函数")
    namespace: dict[str, Any] = {
        "__name__": "_gbid_from_export_utils",
        "np": np,
        "Any": Any,
        "Sequence": Sequence,
        "FloatArray": Any,
        "IntArray": Any,
    }
    body = constants + functions
    exec(compile(ast.Module(body=body, type_ignores=[]), script, "exec"), namespace)
    return namespace["compute_utility"], namespace["compute_target_utility"], namespace["compute_gbid"]


compute_utility, compute_target_utility, compute_gbid = load_gbid_functions()


class Tee:
    """把 stdout/stderr 同时写到终端与日志。"""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


# 样品数取自 47 原文的 N（协议 B10.10 数据清洗后的样品数），与 47 的覆盖核对同一口径
N_SAMPLES = int(load47(funcs=(), consts=("K", "R", "N"), base={})["N"])


def load_47_tools() -> tuple[list[str], int, float, Callable, Callable]:
    """从 47 原文抽取方法顺序、J、RHO、corrected_test 与 holm。"""
    ns = load47(
        funcs=("corrected_test", "holm"),
        consts=("K", "R", "N_TE", "N_TR", "ORDER", "J", "RHO"),
        base={"np": np, "stats": stats},
    )
    return list(ns["ORDER"]), int(ns["J"]), float(ns["RHO"]), ns["corrected_test"], ns["holm"]


def _input_paths(source: Source47, order: list[str]) -> dict[str, tuple[str, str, str]]:
    """从 47 标记记录的 inputs45 定位六法汇总、预测文件与 basis，不写死具体文件名。"""
    paths = [str(path) for path in source.marker.get("inputs45", [])]
    found: dict[str, tuple[str, str, str]] = {}
    for method in order:
        label = method.replace("-", "")
        expected_basis = "bspline" if method in ("FPLR", "FPCR") else "none"
        stem = f"45rebuttal_ncv_{expected_basis}_{label}"
        summaries = [p for p in paths if Path(p).name == f"{stem}.csv"]
        predictions = [p for p in paths if Path(p).name == f"{stem}_pred.csv"]
        if len(summaries) != 1 or len(predictions) != 1:
            raise SystemExit(f"47 标记中 {method} 的主分析汇总/预测文件不是各恰好一个：{summaries}, {predictions}")
        summary = summaries[0]
        basis = Path(summary).stem.split("_")[-2]
        if basis != expected_basis:
            raise SystemExit(f"47 标记中 {method} 的 basis={basis}，预期 {expected_basis}")
        found[method] = (summary, predictions[0], basis)
    return found


def audit_and_load_metrics(source: Source47, order: list[str]) -> pd.DataFrame:
    """由逐样本预测按 45 原文重算七项 A5 所需指标，并与 45 汇总逐项绝对误差 1e-9 对账。"""
    metric_fn = fold_metrics45()
    input_paths = _input_paths(source, order)
    rows: list[dict] = []
    problems: list[str] = []
    truth_by_seed: dict[int, pd.Series] = {}
    for method in order:
        summary_path, prediction_path, basis = input_paths[method]
        summary = pd.read_csv(summary_path)
        predictions = pd.read_csv(prediction_path)
        if summary.duplicated(["seed", "fold"]).any():
            problems.append(f"{method} 汇总存在重复（seed, fold）")
        if predictions.duplicated(["seed", "fold", "idx"]).any():
            problems.append(f"{method} 预测存在重复（seed, fold, idx）")
        summary_indexed = summary.set_index(["seed", "fold"])
        for seed, seed_predictions in predictions.groupby("seed"):
            seed_int = int(seed)
            truth = seed_predictions.sort_values("idx").set_index("idx").y.astype(float)
            if truth.index.has_duplicates or len(truth) != N_SAMPLES or set(truth.index) != set(range(N_SAMPLES)):
                problems.append(f"{method} seed={seed_int} 的 5 折测试集未按 idx 恰好覆盖 {N_SAMPLES} 个样品")
                continue
            if seed_int in truth_by_seed:
                aligned = truth_by_seed[seed_int].loc[truth.index].to_numpy()
                if not np.array_equal(aligned, truth.to_numpy()):
                    problems.append(f"{method} seed={seed_int} 的真值向量与其他方法不一致")
                    continue
            else:
                truth_by_seed[seed_int] = truth
            for fold, fold_predictions in seed_predictions.groupby("fold"):
                fold_int = int(fold)
                key = (seed_int, fold_int)
                if key not in summary_indexed.index:
                    problems.append(f"{method} seed={seed_int} fold={fold_int} 缺汇总行")
                    continue
                test = fold_predictions.sort_values("idx")
                test_idx = test.idx.astype(int).to_numpy()
                y_test = test.y.astype(float).to_numpy()
                y_train = truth.drop(index=test_idx).to_numpy()
                recalculated = metric_fn(y_train, y_test, test.pred.astype(float).to_numpy())
                recorded = summary_indexed.loc[key]
                for metric in METRICS:
                    delta = abs(float(recalculated[metric]) - float(recorded[metric]))
                    if delta > 1e-9:
                        problems.append(
                            f"{method} seed={seed_int} fold={fold_int} {metric} 不一致："
                            f"重算={recalculated[metric]:.15g}，汇总={float(recorded[metric]):.15g}，|差|={delta:.3e}"
                        )
                rows.append(
                    {
                        "basis": basis,
                        "method": method,
                        "seed": seed_int,
                        "fold": fold_int,
                        **{metric: float(recalculated[metric]) for metric in METRICS},
                    }
                )
    if problems:
        raise SystemExit("45 汇总与逐样本预测对账未通过，不写产物：\n  " + "\n  ".join(problems))
    metrics = pd.DataFrame(rows)
    expected = {(method, seed, fold) for method in order for seed in metrics.seed.unique() for fold in range(5)}
    actual = set(metrics[["method", "seed", "fold"]].itertuples(index=False, name=None))
    if len(metrics.seed.unique()) != 5 or actual != expected or len(metrics) != len(order) * 25:
        raise SystemExit(f"六法覆盖不是 5 种子×5 折：行数={len(metrics)}，缺失={sorted(expected - actual)[:10]}")
    return metrics.sort_values(["seed", "fold", "method"]).reset_index(drop=True)


def _utilities(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """对同一（种子, 折）的各方法按 B9 §9.1 效用化。"""
    for column in ("base_MAE", "base_P90", "s_y_train"):
        values = group[column].to_numpy(dtype=float)
        if float(np.max(values) - np.min(values)) > 1e-9:
            raise ValueError(f"同一外层折各方法的 {column} 不一致：{values.tolist()}")
    base_mae = float(group.base_MAE.iloc[0])
    base_p90 = float(group.base_P90.iloc[0])
    tau = ALPHA_TAU * float(group.s_y_train.iloc[0])
    converted = (
        compute_utility(group.R2.to_numpy(), I=1.0, L=0.0, direction="benefit"),
        compute_utility(group.MAE.to_numpy(), I=0.0, L=base_mae, direction="cost"),
        compute_utility(group.P90.to_numpy(), I=0.0, L=base_p90, direction="cost"),
        compute_target_utility(group.BIAS.to_numpy(), T=0.0, tau_minus=tau, tau_plus=tau),
    )
    u = np.column_stack([item["u"] for item in converted])
    u_tilde = np.column_stack([item["u_tilde"] for item in converted])
    return u, u_tilde


def score_folds(metrics: pd.DataFrame, order: list[str], q_pairs=Q_PAIRS) -> pd.DataFrame:
    """逐折计算效用、GBID/GBIDS、Pass 与四组 q 参数下的折内名次。"""
    rows: list[dict] = []
    for (seed, fold), raw_group in metrics.groupby(["seed", "fold"], sort=True):
        group = raw_group.set_index("method").loc[order].reset_index()
        u, u_tilde = _utilities(group)
        q_results = {
            (q_b, q_w): compute_gbid(u, W, V, DIM_GROUPS, q_B=float(q_b), q_W=float(q_w))
            for q_b, q_w in q_pairs
        }
        ranks = {
            key: rankdata(result["gbid"], method="min").astype(int)
            for key, result in q_results.items()
        }
        main = q_results[MAIN_Q]
        for index, _method in enumerate(order):
            row = group.iloc[index].to_dict()
            row.update({"seed": int(seed), "fold": int(fold)})
            for metric_index, metric in enumerate(UTILITY_NAMES):
                row[f"u_{metric}"] = float(u[index, metric_index])
                row[f"u_tilde_{metric}"] = float(u_tilde[index, metric_index])
            row.update(
                {
                    "u_bar": float(main["u_bar"][index]),
                    "U_D1": float(main["U_d"][index, 0]),
                    "U_D2": float(main["U_d"][index, 1]),
                    "U_D3": float(main["U_d"][index, 2]),
                    "U_D4": float(main["U_d"][index, 3]),
                    "C_B": float(main["C_B"][index]),
                    "C_W": float(main["C_W"][index]),
                    "GBID": float(main["gbid"][index]),
                    "GBIDS": float(main["gbids"][index]),
                    "Pass": int(u[index, 0] >= 0.5),
                }
            )
            for key, values in ranks.items():
                row[f"rank_qB{key[0]}_qW{key[1]}"] = int(values[index])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["seed", "fold", "method"]).reset_index(drop=True)


def summarize_folds(folds: pd.DataFrame, order: list[str], q_pairs=Q_PAIRS) -> pd.DataFrame:
    """汇总 25 折 GBIDS、Pass、平均排名、第一名频率与 q 敏感性。"""
    rows: list[dict] = []
    for method in order:
        group = folds[folds.method == method]
        if len(group) != 25:
            raise ValueError(f"{method} 的逐折行数 {len(group)} ≠ 25")
        row = {
            "method": method,
            "n_folds": len(group),
            "GBIDS_mean": float(group.GBIDS.mean()),
            "GBIDS_sd": float(group.GBIDS.std(ddof=1)),
            "GBIDS_min": float(group.GBIDS.min()),
            "Pass_n": int(group.Pass.sum()),
            "avg_rank": float(group["rank_qB2_qW2"].mean()),
            "first_freq": float((group["rank_qB2_qW2"] == 1).mean()),
        }
        for q_b, q_w in q_pairs:
            row[f"avg_rank_qB{q_b}_qW{q_w}"] = float(group[f"rank_qB{q_b}_qW{q_w}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(
    folds: pd.DataFrame,
    order: list[str],
    corrected_test: Callable,
    holm: Callable,
) -> pd.DataFrame:
    """按 47 ORDER 生成 15 对 GBIDS 比较，并使用 47 原文的检验与 Holm。"""
    tests: list[dict] = []
    for method_a, method_b in itertools.combinations(order, 2):
        a = folds[folds.method == method_a].set_index(["seed", "fold"]).GBIDS.sort_index()
        b = folds[folds.method == method_b].set_index(["seed", "fold"]).GBIDS.sort_index()
        tests.append(
            {
                "family": "A5 GBIDS",
                "pair": f"{method_a} − {method_b}",
                **corrected_test(a, b),
            }
        )
    pairs = pd.DataFrame(tests)
    if len(pairs) != 15:
        raise ValueError(f"六法两两比较应为 15 对，实际 {len(pairs)}")
    pairs["holm_p"] = holm(pairs.p.to_numpy())
    pairs["family_size"] = 15
    pairs["结论"] = np.where(pairs.holm_p < 0.05, "差异显著", "未检出差异")
    return pairs[PAIR_COLUMNS]


def clipping_diagnostics(folds: pd.DataFrame) -> pd.DataFrame:
    """汇总四项效用的截断前上下截断比例。"""
    rows = []
    for metric in UTILITY_NAMES:
        values = folds[f"u_tilde_{metric}"].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "n": len(values),
                "lower_clip_ratio": float((values <= 0).mean()),
                "upper_clip_ratio": float((values >= 1).mean()),
                "u_tilde_min": float(values.min()),
                "u_tilde_max": float(values.max()),
                "over_50pct": bool((values <= 0).mean() > 0.5 or (values >= 1).mean() > 0.5),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(outputs: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, str]:
    """把四张表以临时文件 + os.replace 原子写出；测试也通过本函数生成夹具。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for key, filename in OUTPUT_NAMES.items():
        path = directory / filename
        temporary = directory / f".{filename}.{os.getpid()}.tmp"
        outputs[key].to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
        written[key] = str(path)
    return written


def run() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    source = resolve()
    if not source.final:
        raise SystemExit("A5 只接受 47 正式结果；当前 resolve() 取到的是部分结果")
    order, j, rho, corrected_test, holm = load_47_tools()
    if j != 25 or not np.isclose(rho, 0.25):
        raise SystemExit(f"47 原文的 J={j}, RHO={rho}，与 B10.2 锁定口径不符")
    metrics = audit_and_load_metrics(source, order)
    folds = score_folds(metrics, order)
    summary = summarize_folds(folds, order)
    pairs = paired_comparisons(folds, order, corrected_test, holm)
    diag = clipping_diagnostics(folds)
    outputs = {"folds": folds, "summary": summary, "pairs": pairs, "diag": diag}
    written = write_outputs(outputs, "04outputs")
    print("【45 逐折对账】六法 × 25 折的七项 A5 指标全部满足 |差|≤1e−9")
    print("\n【A5 GBIDS 汇总】")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n【A5 GBIDS 配对比较】")
    print(pairs.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\n【截断前诊断】")
    print(diag.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n已写：" + "、".join(written.values()))
    return outputs, written


def main() -> None:
    Path("05logs").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%y-%m-%d_%H%M%S")
    log_path = Path("05logs") / f"55rebuttal_A5_gbids_{timestamp}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"开始：{datetime.now().isoformat(timespec='seconds')}")
            run()
            print(f"完成：{datetime.now().isoformat(timespec='seconds')}")
            print(f"日志：{log_path}")


if __name__ == "__main__":
    main()
