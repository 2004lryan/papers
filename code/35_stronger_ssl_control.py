"""35_stronger_ssl_control.py: 更强自监督目标下的 Δ_unsup 复核（回应审稿人 #4）。

31 的判决"Δ_unsup≈0（无标签目标暴露不构成大泄漏）"仅在\emph{掩码光谱重构}这一自监督目标下成立；
审稿人可质疑该结论是否只因该 SSL 目标弱/不适配分类。本脚本用一个\textbf{更强的 target-aware
无监督目标——SimCLR 式对比学习}（谱增强=波段掩码+高斯噪声+随机缩放，NT-Xent 损失，编码器绝不接触
标签）重做无监督臂：U_src2（对比自监督仅源 x）vs U_tgt2（对比自监督源+目标 x），
Δ_unsup^ctr = U_tgt2 − U_src2。

若 Δ_unsup^ctr 仍小、CI 掠过 0，则"无标签目标暴露泄漏小"的结论对 SSL 目标选择稳健，
不是掩码重构目标弱的假象；若显著增大，则如实报告"更强 SSL 下无标签暴露泄漏增大"这一开放结论。
覆盖 31 中 Δ_unsup 唯一为负的 corn 与最正的 Indian Pines 两端最敏感数据集，兼跑 tablet/mango 全 4 集。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE python 02code/35_stronger_ssl_control.py --device auto --stage pilot
    # Formal（CUDA 5 种子）:
    python 02code/35_stronger_ssl_control.py --device cuda --stage formal \
        --seeds 20060515,20041210,19810915,2023,2024

输出文件:
    04outputs/35_stronger_ssl_control-图a.pdf / -fig-a.pdf  — 掩码 vs 对比 SSL 的 Δ_unsup 对比(含95%CI)
    04outputs/35_stronger_ssl_control.xlsx  — 折级/总体 Δ_unsup^ctr + 与掩码臂对照 + 算力
    05logs/35_stronger_ssl_control_YY-MM-DD_HHMMSS.log — 运行日志
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 后台无显示环境安全出图
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

import crossdataset_lib as cdl
import export_utils as eu

warnings.filterwarnings("ignore")

NDArray = npt.NDArray[Any]


def run_fold_contrastive(
    doms: dict[str, tuple[NDArray, NDArray]],
    is_reg: bool,
    held: str,
    device: Any,
    ssm_epochs: int,
    reps: int,
    k: int,
    seed: int,
) -> dict[str, Any] | None:
    """单折对比自监督无监督臂：U_src2/U_tgt2/Δ_unsup^ctr（编码器绝不接触标签）。"""
    fa = cdl.prepare_fold(doms, is_reg, held, k, preprocess="snv")
    if fa is None:
        return None
    enc_usrc = cdl.contrastive_ssm_pretrain(fa.xs, fa.in_dim, device, ssm_epochs, seed)
    enc_utgt = cdl.contrastive_ssm_pretrain(fa.xall, fa.in_dim, device, ssm_epochs, seed)
    f_usrc = cdl.eval_kshot(enc_usrc, fa.xh, fa.yh, fa.classes, device, k, reps, seed)
    f_utgt = cdl.eval_kshot(enc_utgt, fa.xh, fa.yh, fa.classes, device, k, reps, seed)
    return {
        "held": held,
        "u_src_ctr": f_usrc,
        "u_tgt_ctr": f_utgt,
        "delta_unsup_ctr": f_utgt - f_usrc,
        "mmd2": fa.mmd2,
    }


def _make_plot(ds_summ: pd.DataFrame, overall: dict[str, float]) -> Any:
    """返回 plot_fn(lang)：逐数据集对比 SSL 的 Δ_unsup^ctr 点图 + 总体混合效应截距(含95%CI)。"""

    txt = {
        "zh": {
            "title": "更强自监督（对比）目标下的无监督暴露 Δ_unsup",
            "xlabel": "对比自监督泄漏 Δ_unsup^ctr（留出域宏 F1 高估）",
            "overall": "总体(混合效应)",
        },
        "en": {
            "title": "Unsupervised exposure Δ_unsup under a stronger (contrastive) SSL objective",
            "xlabel": "Contrastive-SSL leakage Δ_unsup^ctr (held-domain macro-F1 inflation)",
            "overall": "Overall (mixed-effects)",
        },
    }

    def plot_fn(lang: str) -> Any:
        t = txt[lang]
        rows = [*ds_summ["dataset"], t["overall"]]
        y = np.arange(len(rows))
        vals = [*ds_summ["delta_unsup_ctr_mean"], overall["intercept"]]
        fig, ax = plt.subplots(figsize=(7.0, 0.6 * len(rows) + 1.3))
        ax.axvline(0.0, color="#999999", lw=1, ls="--")
        ax.scatter(vals, y, color="#0072B2", s=55, zorder=3)
        ax.plot(
            [overall["ci_low"], overall["ci_high"]],
            [len(rows) - 1, len(rows) - 1],
            color="#0072B2", lw=2, zorder=2,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(rows)
        ax.set_xlabel(t["xlabel"])
        ax.set_title(t["title"])
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    return plot_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="更强(对比)SSL 下的 Δ_unsup 复核")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    parser.add_argument("--seeds", default="2004", help="逗号分隔种子；formal 用 5 固定种子")
    parser.add_argument("--data_root", default="05data")
    parser.add_argument("--ssm_epochs", type=int, default=200)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    device = eu.get_device(args.device)
    logger = eu.get_logger("35_stronger_ssl_control")
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.stage == "formal" and seeds != list(eu.RECOMMENDED_SEEDS):
        logger.log(f"[WARN] formal 阶段种子应为 {eu.RECOMMENDED_SEEDS}，当前 {seeds}")
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "更强(对比)SSL 下 Δ_unsup 复核",
            "阶段": args.stage,
            "设备": str(device),
            "种子": seeds,
            "SSL": "SimCLR对比(NT-Xent+谱增强)",
            "数据集": list(cdl.DATASETS),
        },
    )
    run = eu.init_wandb(__file__, seeds[0], eu.PROJECT_CODE, args.stage, vars(args))

    long_rows: list[dict[str, Any]] = []
    with eu.track_compute("更强SSL对照", logger, run) as budget:
        for seed in seeds:
            for dsname, meta, doms, is_reg in cdl.iter_datasets(args.data_root, seed):
                for held in list(doms.keys()):
                    rec = run_fold_contrastive(doms, is_reg, held, device, args.ssm_epochs, args.reps, args.k, seed)
                    if rec is None:
                        continue
                    rec.update({"dataset": dsname, "drift_axis": meta["drift"], "seed": seed})
                    long_rows.append(rec)
                    logger.log(
                        f"  [{dsname} seed{seed}] 留出{rec['held']}: "
                        f"U_src2={rec['u_src_ctr']:.4f} U_tgt2={rec['u_tgt_ctr']:.4f} "
                        f"Δ_unsup^ctr={rec['delta_unsup_ctr']:+.4f}"
                    )

    long_df = pd.DataFrame(long_rows)
    fold_df = (
        long_df.groupby(["dataset", "drift_axis", "held"])
        .agg(
            delta_unsup_ctr=("delta_unsup_ctr", "mean"),
            u_src_ctr=("u_src_ctr", "mean"),
            u_tgt_ctr=("u_tgt_ctr", "mean"),
            mmd2=("mmd2", "mean"),
        )
        .reset_index()
    )
    ds_summ = (
        fold_df.groupby(["dataset", "drift_axis"])
        .agg(
            n_folds=("held", "nunique"),
            delta_unsup_ctr_mean=("delta_unsup_ctr", "mean"),
            pos_frac=("delta_unsup_ctr", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    overall = cdl.robust_intercept_ci(fold_df["delta_unsup_ctr"].to_numpy(), fold_df["dataset"].to_numpy())
    n_folds = len(fold_df)
    pos = int((fold_df["delta_unsup_ctr"] > 0).sum())
    logger.log(
        f"\n★ 对比 SSL Δ_unsup 汇总（{n_folds} 折级）:\n"
        f"  Δ_unsup^ctr 折正 {pos}/{n_folds}，混合效应截距={overall['intercept']:+.4f} "
        f"[{overall['ci_low']:+.4f},{overall['ci_high']:+.4f}] ({overall['method']})"
    )
    cross_zero = overall["ci_low"] <= 0
    logger.log(
        "★ 复核判决: "
        + (
            "Δ_unsup^ctr CI 仍掠过 0 → '无标签暴露泄漏小'对 SSL 目标稳健（掩码/对比一致）"
            if cross_zero
            else f"Δ_unsup^ctr CI 下界>0（={overall['ci_low']:+.4f}）→ 更强 SSL 下无标签暴露泄漏增大，如实报告"
        )
    )

    overall_df = pd.DataFrame(
        [{"分量": "Δ_unsup^ctr(对比SSL无标签目标暴露)", **overall, "折正比例": pos / max(n_folds, 1)}]
    )
    zh_path, _ = eu.save_figure_bilingual(_make_plot(ds_summ, overall), "35_stronger_ssl_control", 0)
    logger.log(f"已写图 {zh_path}")

    sheets: dict[object, object] = {
        eu.table_sheet_name(1, "数据集汇总"): ds_summ,
        eu.table_sheet_name(2, "折级Δunsup对比"): fold_df,
        eu.table_sheet_name(3, "混合效应总体"): overall_df,
        eu.table_sheet_name(4, "长格式"): long_df,
        eu.table_sheet_name(5, "compute"): pd.DataFrame([dict(budget)]),
    }
    eu.write_script_workbook(__file__, sheets)
    logger.log("已写 04outputs/35_stronger_ssl_control.xlsx")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
