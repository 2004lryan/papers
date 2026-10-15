"""31_leakage_decomposition.py: 泄漏的监督/无监督分解（Δ_unsup vs Δ_sup，承重实验）。

回应审稿意见一致指出的头号问题——"共享编码器用目标域标签(xall,yall)元训练=
label-aware，审稿人视作教科书 label leakage"。本脚本在同一留一域折上对照三种表示学习协议，
把泄漏 Δ 拆成监督上界与真正隐蔽的无监督分量：

监督臂（隔离 Δ_sup）：
  E0 有监督折限定 : 编码器仅在源域 (x,y) 有监督元训练。合法基线。
  E2 label-aware  : 编码器在源+目标域 (x,y) 有监督元训练(复现现有共享协议)。
无监督臂（隔离 Δ_unsup，两臂同为掩码自监督、仅差是否含目标域 x）：
  U_src 自监督折限定 : 掩码自监督重构仅在源域 x 上预训练(不碰任何标签)。
  U_tgt 自监督目标暴露 : 掩码自监督重构在源+目标域 x 上预训练(不碰任何标签)。

  Δ_sup   = E2 − E0     (目标标签进入有监督表示学习 = 泄漏上界，transfer 的既知代价)
  Δ_unsup = U_tgt − U_src  (「同一自监督目标下」仅加无标签目标 x 就产生的高估
                            = 干净隔离的"隐蔽泄漏"关键证据；不与"有监督→自监督"目标变化混淆)

评估：留出域 K=5 ProtoNet(α=0) 折限定原型，逐折宏 F1。以数据集为随机效应的混合效应模型
(折级观测单位)分别给出 Δ_unsup、Δ_sup 总体截距 + 95%CI。预注册判决见 23NMI_PROGRAM_PROTOCOL.md。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE python 02code/31_leakage_decomposition.py --device auto --stage pilot
    # Formal（CUDA 5 种子）:
    python 02code/31_leakage_decomposition.py --device cuda --stage formal \
        --seeds 20060515,20041210,19810915,2023,2024

输出文件:
    04outputs/31_leakage_decomposition-图a.pdf / -fig-a.pdf  — Δ_unsup vs Δ_sup 逐数据集点图(含95%CI)
    04outputs/31_leakage_decomposition.xlsx  — 折级/数据集级/长格式 Δ分解 + 混合效应 + 算力
    05logs/31_leakage_decomposition_YY-MM-DD_HHMMSS.log — 运行日志
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

# 掩码自监督原语（SpectralDecoder / span_mask / masked_ssm_pretrain）由 crossdataset_lib 提供，
# 与 32_dg_benchmark_leakage 共用同一实现（单一真值源）。


# 单折监督/无监督四协议分解由 cdl.run_fold_decomp 提供（与 32 共用；preprocess=snv 谱域口径）。


# ─────────────────────────── 混合效应汇总 ───────────────────────────
def _mixed_intercept(fold_df: pd.DataFrame, col: str) -> dict[str, Any]:
    """总体截距 + 95%CI（MixedLM 优先，奇异则簇自助回退，见 cdl.robust_intercept_ci）。"""
    sub = fold_df[["dataset", col]].dropna()
    return cdl.robust_intercept_ci(sub[col].to_numpy(), sub["dataset"].to_numpy())


def _make_plot(ds_summ: pd.DataFrame, overall: dict[str, dict[str, float]]) -> Any:
    """返回 plot_fn(lang)：逐数据集 Δ_unsup vs Δ_sup 点图 + 总体混合效应截距(含95%CI)。"""

    txt = {
        "zh": {
            "title": "泄漏分解：无监督暴露 Δ_unsup 与 label-aware Δ_sup",
            "xlabel": "泄漏 Δ（留出域宏 F1 高估）",
            "unsup": "Δ_unsup（仅无标签目标 x）",
            "sup": "Δ_sup（目标 x+y，上界）",
            "overall": "总体(混合效应)",
        },
        "en": {
            "title": "Leakage decomposition: unsupervised Δ_unsup vs label-aware Δ_sup",
            "xlabel": "Leakage Δ (held-domain macro-F1 inflation)",
            "unsup": "Δ_unsup (unlabeled target x only)",
            "sup": "Δ_sup (target x+y, upper bound)",
            "overall": "Overall (mixed-effects)",
        },
    }

    def plot_fn(lang: str) -> Any:
        t = txt[lang]
        rows = [*ds_summ["dataset"], t["overall"]]
        y = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(rows) + 1.4))
        du = [*ds_summ["delta_unsup_mean"], overall["unsup"]["intercept"]]
        ds = [*ds_summ["delta_sup_mean"], overall["sup"]["intercept"]]
        ax.axvline(0.0, color="#999999", lw=1, ls="--")
        ax.scatter(ds, y + 0.12, color="#D55E00", s=55, label=t["sup"], zorder=3)
        ax.scatter(du, y - 0.12, color="#0072B2", s=55, label=t["unsup"], zorder=3)
        # 总体 CI 横条
        ax.plot(
            [overall["sup"]["ci_low"], overall["sup"]["ci_high"]],
            [len(rows) - 1 + 0.12, len(rows) - 1 + 0.12],
            color="#D55E00", lw=2, zorder=2,
        )
        ax.plot(
            [overall["unsup"]["ci_low"], overall["unsup"]["ci_high"]],
            [len(rows) - 1 - 0.12, len(rows) - 1 - 0.12],
            color="#0072B2", lw=2, zorder=2,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(rows)
        ax.set_xlabel(t["xlabel"])
        ax.set_title(t["title"])
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    return plot_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="泄漏监督/无监督分解 Δ_unsup vs Δ_sup")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    parser.add_argument("--seeds", default="2004", help="逗号分隔种子；formal 用 5 固定种子")
    parser.add_argument("--data_root", default="05data")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--ssm_epochs", type=int, default=200)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    device = eu.get_device(args.device)
    logger = eu.get_logger("31_leakage_decomposition")
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.stage == "formal" and seeds != list(eu.RECOMMENDED_SEEDS):
        logger.log(f"[WARN] formal 阶段种子应为 {eu.RECOMMENDED_SEEDS}，当前 {seeds}")
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "泄漏监督/无监督分解",
            "阶段": args.stage,
            "设备": str(device),
            "种子": seeds,
            "数据集": list(cdl.DATASETS),
            "协议": "E0折限定/E1无标签自监督/E2 label-aware",
        },
    )
    run = eu.init_wandb(__file__, seeds[0], eu.PROJECT_CODE, args.stage, vars(args))

    long_rows: list[dict[str, Any]] = []
    with eu.track_compute("泄漏分解", logger, run) as budget:
        for seed in seeds:
            for dsname, meta, doms, is_reg in cdl.iter_datasets(args.data_root, seed):
                for held in list(doms.keys()):
                    rec = cdl.run_fold_decomp(
                        doms, is_reg, held, device, args.episodes, args.ssm_epochs, args.reps, args.k, seed
                    )
                    if rec is None:
                        continue
                    rec.update(
                        {"dataset": dsname, "drift_axis": meta["drift"], "modality": meta["modality"], "seed": seed}
                    )
                    long_rows.append(rec)
                    logger.log(
                        f"  [{dsname} seed{seed}] 留出{rec['held']}: "
                        f"E0={rec['e0_sup_fold_restricted']:.4f} E2={rec['e2_label_aware']:.4f} "
                        f"U_src={rec['u_src_selfsup_fold_restricted']:.4f} U_tgt={rec['u_tgt_selfsup_exposure']:.4f} "
                        f"Δ_sup={rec['delta_sup']:+.4f} Δ_unsup={rec['delta_unsup']:+.4f}"
                    )

    long_df = pd.DataFrame(long_rows)
    # 折级：先按 (dataset,held) 跨种子聚合为折级均值（混合效应观测单位）
    fold_df = (
        long_df.groupby(["dataset", "drift_axis", "modality", "held"])
        .agg(
            delta_unsup=("delta_unsup", "mean"),
            delta_sup=("delta_sup", "mean"),
            e0_sup=("e0_sup_fold_restricted", "mean"),
            e2_label_aware=("e2_label_aware", "mean"),
            u_src=("u_src_selfsup_fold_restricted", "mean"),
            u_tgt=("u_tgt_selfsup_exposure", "mean"),
            mmd2=("mmd2", "mean"),
        )
        .reset_index()
    )
    ds_summ = (
        fold_df.groupby(["dataset", "drift_axis", "modality"])
        .agg(
            n_folds=("held", "nunique"),
            delta_unsup_mean=("delta_unsup", "mean"),
            delta_sup_mean=("delta_sup", "mean"),
            unsup_pos_frac=("delta_unsup", lambda s: float((s > 0).mean())),
            sup_pos_frac=("delta_sup", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    overall = {"unsup": _mixed_intercept(fold_df, "delta_unsup"), "sup": _mixed_intercept(fold_df, "delta_sup")}
    n_folds = len(fold_df)
    unsup_pos = int((fold_df["delta_unsup"] > 0).sum())
    sup_pos = int((fold_df["delta_sup"] > 0).sum())
    logger.log(
        f"\n★ 分解汇总（{n_folds} 折级）:\n"
        f"  Δ_unsup 折正 {unsup_pos}/{n_folds}，混合效应截距={overall['unsup']['intercept']:+.4f} "
        f"[{overall['unsup']['ci_low']:+.4f},{overall['unsup']['ci_high']:+.4f}]\n"
        f"  Δ_sup   折正 {sup_pos}/{n_folds}，混合效应截距={overall['sup']['intercept']:+.4f} "
        f"[{overall['sup']['ci_low']:+.4f},{overall['sup']['ci_high']:+.4f}]"
    )
    ci_low = overall["unsup"]["ci_low"]
    ds_pos = int((ds_summ["delta_unsup_mean"] > 0).sum())
    pass_verdict = ci_low > 0 and ds_pos >= max(1, len(ds_summ) - 1)
    verdict = (
        f"Δ_unsup CI 下界={ci_low:+.4f}，数据集级正 {ds_pos}/{len(ds_summ)}；"
        + ("判决=隐蔽泄漏成立(保强主张冲NMI)" if pass_verdict else "判决=收窄主张(投NC)/待CUDA全量")
    )
    logger.log(f"★ 预注册判决口径: {verdict}")

    overall_df = pd.DataFrame(
        [
            {"分量": "Δ_unsup(无标签目标暴露)", **overall["unsup"], "折正比例": unsup_pos / max(n_folds, 1)},
            {"分量": "Δ_sup(label-aware上界)", **overall["sup"], "折正比例": sup_pos / max(n_folds, 1)},
        ]
    )
    zh_path, _ = eu.save_figure_bilingual(_make_plot(ds_summ, overall), "31_leakage_decomposition", 0)
    logger.log(f"已写图 {zh_path}")

    sheets: dict[object, object] = {
        eu.table_sheet_name(1, "数据集汇总"): ds_summ,
        eu.table_sheet_name(2, "折级Δ分解"): fold_df,
        eu.table_sheet_name(3, "混合效应总体"): overall_df,
        eu.table_sheet_name(4, "长格式"): long_df,
        eu.table_sheet_name(5, "compute"): pd.DataFrame([dict(budget)]),
    }
    eu.write_script_workbook(__file__, sheets)
    logger.log("已写 04outputs/31_leakage_decomposition.xlsx")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
