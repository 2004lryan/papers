"""
17_alpha_sensitivity.py: ProtoNet-C 源域融合权重 α 敏感性曲线（单方法连续扫描）。

在三个 LOEO-Year 折上对 src_alpha ∈ [0, 1] 做约 21 点连续扫描（默认 0.00, 0.05, …, 1.00），
使用 10 的折内编码器、多面位均值苹果特征，每点多种子取均值，刻画 α 对 Macro-F1 的影响曲线。
属"单方法连续扫描"，非多模型多指标比较，不产出 LLH-GBID（仅敏感性折线）。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/17_alpha_sensitivity.py --device auto --seeds 50

输出文件:
    04outputs/17_alpha_sensitivity-图a.pdf / -fig-a.pdf  — K=5 三折 α 敏感性曲线（主图）
    04outputs/17_alpha_sensitivity-图b.pdf / -fig-b.pdf  — 三折 × 多 K 值 α 敏感性面板
    04outputs/17_alpha_sensitivity.xlsx                   — 完整扫描结果 + 最优 α 汇总 + 算力
    05logs/17_alpha_sensitivity_YY-MM-DD_HHMMSS.log      — 运行日志

说明:
    依赖 10 的 fold 编码器（--encoder_seed，默认 PILOT_SEED）。置信带按 ±1.96·SD/√n 近似。
    MPS 下为 Pilot（<MPS_PILOT>）。
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.metrics import f1_score

import export_utils as eu
from protonet import ProtoEncoder, apple_feature, build_year_region_data

NDArray = npt.NDArray[Any]

FOLD_CFG: dict[int, dict[str, Any]] = {
    2018: {"src_years": [2019, 2025]},
    2019: {"src_years": [2018, 2025]},
    2025: {"src_years": [2018, 2019]},
}
FOLD_COLORS = {2018: "#0072B2", 2019: "#009E73", 2025: "#D55E00"}
FOLD_MARKERS = {2018: "o", 2019: "s", 2025: "^"}
WONG = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999999"]


def load_fold_encoder(tgt_year: int, encoder_seed: int, n_bands: int, device: torch.device) -> ProtoEncoder | None:
    """加载 10 产出的 fold 编码器（新格式 ckpt["state_dict"]/["feat_dim"]）；缺失返回 None。"""
    run_name = f"10_loeo_year_foldwise_seed{encoder_seed}_fold{tgt_year}"
    ckpt_path = os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = ProtoEncoder(n_bands, feat_dim=int(ckpt.get("feat_dim", 128)))
    enc.load_state_dict(ckpt["state_dict"])
    enc.to(device)
    enc.eval()
    return enc


def eval_protonet_alpha(
    encoder: ProtoEncoder,
    year_data: Any,
    tgt_year: int,
    src_years: list[int],
    k_shot: int,
    seed: int,
    src_alpha: float,
    device: torch.device,
) -> float:
    """折内编码器对目标年份的 K-Shot Macro-F1（多面位均值苹果特征 + 源域同类原型融合 src_alpha）。"""
    rng = np.random.default_rng(seed)
    tgt_regions = sorted(year_data[tgt_year].keys())
    support_ids: dict[str, list[Any]] = {}
    for reg in tgt_regions:
        apples = list(year_data[tgt_year][reg].keys())
        k = min(k_shot, max(1, len(apples) - 1))
        support_ids[reg] = list(rng.choice(apples, size=k, replace=False))
    sup_set = {aid for aids in support_ids.values() for aid in aids}
    query_items = [(aid, reg) for reg in tgt_regions for aid in year_data[tgt_year][reg] if aid not in sup_set]
    if not query_items:
        query_items = [(aid, reg) for reg in tgt_regions for aid in year_data[tgt_year][reg]]

    tgt_protos: dict[str, Any] = {}
    for reg in tgt_regions:
        feats = [apple_feature(encoder, year_data[tgt_year][reg][aid], device) for aid in support_ids[reg]]
        p = np.mean(feats, axis=0)
        tgt_protos[reg] = p / (float(np.linalg.norm(p)) + 1e-8)

    src_protos: dict[str, list[Any]] = {}
    if src_alpha > 0:
        for src_yr in src_years:
            for reg in year_data.get(src_yr, {}):
                if reg not in tgt_regions:
                    continue
                feats = [apple_feature(encoder, faces, device) for faces in year_data[src_yr][reg].values()]
                if feats:
                    p = np.mean(feats, axis=0)
                    src_protos.setdefault(reg, []).append(p / (float(np.linalg.norm(p)) + 1e-8))

    final_protos: dict[str, Any] = {}
    for reg in tgt_regions:
        if reg in src_protos and src_alpha > 0:
            src_avg = np.mean(src_protos[reg], axis=0)
            src_avg = src_avg / (float(np.linalg.norm(src_avg)) + 1e-8)
            fused = (1 - src_alpha) * tgt_protos[reg] + src_alpha * src_avg
            final_protos[reg] = fused / (float(np.linalg.norm(fused)) + 1e-8)
        else:
            final_protos[reg] = tgt_protos[reg]

    proto_mat = np.stack([final_protos[r] for r in tgt_regions])
    y_true: list[str] = []
    y_pred: list[str] = []
    for aid, true_reg in query_items:
        feat = apple_feature(encoder, year_data[tgt_year][true_reg][aid], device).reshape(1, -1)
        sim = (feat @ proto_mat.T)[0]
        y_true.append(true_reg)
        y_pred.append(tgt_regions[int(sim.argmax())])
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def summarize_best_alpha(res_df: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """各折各 K 的最优 α 与 α=0 / α=0.4 参考值汇总。"""
    rows: list[dict[str, Any]] = []
    for yr in (2018, 2019, 2025):
        for k in k_values:
            sub = res_df[(res_df["目标年份"] == yr) & (res_df["K"] == k)]
            if sub.empty:
                continue
            best = sub.loc[sub["F1均值"].idxmax()]
            a0 = sub[sub["alpha"] == 0.0]["F1均值"]
            a04 = sub[sub["alpha"] == 0.4]["F1均值"]
            rows.append(
                {
                    "目标年份": yr,
                    "K": k,
                    "最优alpha": float(best["alpha"]),
                    "最优F1": round(float(best["F1均值"]), 4),
                    "F1(alpha=0)": round(float(a0.iloc[0]), 4) if not a0.empty else float("nan"),
                    "F1(alpha=0.4)": round(float(a04.iloc[0]), 4) if not a04.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="17_alpha_sensitivity：α 敏感性曲线（单方法扫描）")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k_list", default="1,3,5,10", help="K-shot 值列表，逗号分隔")
    parser.add_argument("--alpha_steps", type=int, default=21, help="α 扫描点数（含端点），21 → 0.00,0.05,…,1.00")
    parser.add_argument("--seeds", type=int, default=50, help="目标域 support 重采样次数")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--encoder_seed", type=int, default=eu.PILOT_SEED, help="定位 10 的 fold 编码器种子")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_list.split(",")]
    alpha_list = [round(float(v), 2) for v in np.linspace(0, 1, args.alpha_steps)]
    rep_seeds = [args.seed + i for i in range(args.seeds)]
    device = eu.get_device(args.device)
    logger = eu.get_logger("17_alpha_sensitivity")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "ProtoNet-C α 敏感性扫描（单方法）",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "reps": args.seeds,
            "K_list": k_values,
            "alpha_steps": args.alpha_steps,
            "alpha范围": f"{alpha_list[0]} → {alpha_list[-1]}（{len(alpha_list)} 点）",
            "encoder_seed": args.encoder_seed,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    year_data = build_year_region_data(df, wl_cols)

    fold_encoders: dict[int, ProtoEncoder] = {}
    for tgt_year in (2018, 2019, 2025):
        enc = load_fold_encoder(tgt_year, args.encoder_seed, n_bands, device)
        if enc is None:
            logger.log(f"  fold {tgt_year} 编码器缺失（先跑 10_loeo_year_foldwise），跳过该折")
            continue
        fold_encoders[tgt_year] = enc
        logger.log(f"  fold {tgt_year} 编码器：OK")

    all_rows: list[dict[str, Any]] = []
    with eu.track_compute("α 敏感性扫描", logger, run) as budget:
        for tgt_year, fold in sorted(FOLD_CFG.items()):
            if tgt_year not in fold_encoders:
                continue
            encoder = fold_encoders[tgt_year]
            tgt_regs = sorted(year_data[tgt_year].keys())
            logger.log(f"\n【LOEO-{tgt_year}】目标产地={tgt_regs}  源年份={fold['src_years']}")
            for k in k_values:
                logger.log(f"  K={k}  扫描 α …")
                for alpha in alpha_list:
                    f1s = [
                        eval_protonet_alpha(encoder, year_data, tgt_year, fold["src_years"], k, s, alpha, device)
                        for s in rep_seeds
                    ]
                    all_rows.append(
                        {
                            "目标年份": tgt_year,
                            "源年份": "+".join(str(y) for y in fold["src_years"]),
                            "K": k,
                            "alpha": alpha,
                            "F1均值": round(float(np.mean(f1s)), 4),
                            "F1标准差": round(float(np.std(f1s)), 4),
                            "种子数": len(rep_seeds),
                        }
                    )
                    logger.log(f"    α={alpha:.2f} K={k:2d} F1={float(np.mean(f1s)):.4f}±{float(np.std(f1s)):.4f}")

    res_df = pd.DataFrame(all_rows)
    best_df = summarize_best_alpha(res_df, k_values)
    logger.log("\n── 各折最优 α 汇总 ──")
    logger.log(best_df.to_string(index=False))

    n_rep = max(len(rep_seeds), 1)
    txt = {
        "zh": {
            "x": "源域融合权重 α",
            "y": "Macro-F1",
            "ta": "ProtoNet-C α 敏感性分析（K=5，折内编码器）",
            "tb": "ProtoNet-C α 敏感性分析（各 K 值，折内编码器）",
            "a04": "论文选取 α=0.40",
            "a0": "纯 K-Shot（α=0）",
            "rand": "随机基线",
            "lo": "LOEO-",
        },
        "en": {
            "x": "Source fusion weight α",
            "y": "Macro-F1",
            "ta": "ProtoNet-C α sensitivity (K=5, fold encoder)",
            "tb": "ProtoNet-C α sensitivity (per K, fold encoder)",
            "a04": "Paper choice α=0.40",
            "a0": "Pure K-shot (α=0)",
            "rand": "Chance",
            "lo": "LOEO-",
        },
    }

    def plot_a(lang: str) -> Any:
        t = txt[lang]
        fig, ax = plt.subplots(figsize=(8, 5))
        k5_df = res_df[res_df["K"] == 5]
        for yr in (2018, 2019, 2025):
            sub = k5_df[k5_df["目标年份"] == yr].sort_values("alpha")
            if sub.empty:
                continue
            ax.plot(
                sub["alpha"],
                sub["F1均值"],
                color=FOLD_COLORS[yr],
                marker=FOLD_MARKERS[yr],
                markersize=5,
                linewidth=1.8,
                label=f"{t['lo']}{yr}",
            )
            ci = 1.96 * sub["F1标准差"] / np.sqrt(n_rep)
            ax.fill_between(sub["alpha"], sub["F1均值"] - ci, sub["F1均值"] + ci, alpha=0.12, color=FOLD_COLORS[yr])
        ax.axvline(0.4, color="#E69F00", linestyle="--", linewidth=1.4, label=t["a04"], zorder=5)
        ax.axvline(0.0, color="#999999", linestyle=":", linewidth=1.2, label=t["a0"], zorder=4)
        ax.axhline(1 / 3, color="#999999", linestyle=":", linewidth=1.0, alpha=0.6, label=t["rand"])
        ax.set_xlabel(t["x"])
        ax.set_ylabel(t["y"])
        ax.set_title(t["ta"])
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(0, 1.05, 0.1))
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    def plot_b(lang: str) -> Any:
        t = txt[lang]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
        for ax_idx, yr in enumerate((2018, 2019, 2025)):
            ax = axes[ax_idx]
            sub = res_df[res_df["目标年份"] == yr]
            for ki, k in enumerate(k_values):
                sub_k = sub[sub["K"] == k].sort_values("alpha")
                if sub_k.empty:
                    continue
                color = WONG[ki % len(WONG)]
                ax.plot(
                    sub_k["alpha"],
                    sub_k["F1均值"],
                    color=color,
                    marker="o",
                    markersize=4,
                    linewidth=1.6,
                    label=f"K={k}",
                )
                ci = 1.96 * sub_k["F1标准差"] / np.sqrt(n_rep)
                ax.fill_between(sub_k["alpha"], sub_k["F1均值"] - ci, sub_k["F1均值"] + ci, alpha=0.10, color=color)
            ax.axvline(0.4, color="#999999", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.axhline(1 / 3, color="#999999", linestyle=":", linewidth=1.0, alpha=0.6)
            ax.set_title(f"{t['lo']}{yr}", fontsize=12)
            ax.set_xlabel(t["x"])
            if ax_idx == 0:
                ax.set_ylabel(t["y"])
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9, loc="best")
        fig.suptitle(t["tb"], fontsize=13)
        fig.tight_layout()
        return fig

    for idx, fn in enumerate((plot_a, plot_b)):
        zh, en = eu.save_figure_bilingual(fn, "17_alpha_sensitivity", idx)
        logger.log(f"图{chr(97 + idx)} → {zh} / {en}")

    budget_df = pd.DataFrame(
        [{"reps": args.seeds, "alpha_steps": args.alpha_steps, "encoder_seed": args.encoder_seed, **budget}]
    )
    xlsx = eu.write_script_workbook(
        __file__,
        {0: ("α敏感性扫描", res_df), 1: ("各折最优α汇总", best_df), "compute": budget_df},
    )
    logger.log(f"结果（敏感性扫描+最优α汇总）→ {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
