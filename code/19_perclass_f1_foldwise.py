"""
19_perclass_f1_foldwise.py: 折内编码器下各产地 per-class F1/Recall 分析。

用 10 产出的折内编码器（无训练污染）重验证各产地相对识别难度，并给出三折 per-class
F1/Recall 表，供论文 per-class 声明使用。单模型（ProtoNet）评估，无多模型对比，
故不产出 LLH-GBID（§13）。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/19_perclass_f1_foldwise.py --device auto --seeds 50

输出文件:
    04outputs/19_perclass_f1_foldwise-图a.pdf / -fig-a.pdf  — 三折各产地 Recall 热力图（K×产地）
    04outputs/19_perclass_f1_foldwise-图b.pdf / -fig-b.pdf  — K=5 各产地 per-class F1（附录 Supplementary）
    04outputs/19_perclass_f1_foldwise.xlsx                   — 三折 per-class F1/Recall + 算力
    05logs/19_perclass_f1_foldwise_YY-MM-DD_HHMMSS.log       — 运行日志

说明:
    依赖 10 的 fold 编码器（--encoder_seed 指定，默认 PILOT_SEED）：
    07checkpoints/10_loeo_year_foldwise_seed<encoder_seed>_fold<year>/best.pt。MPS 下为 Pilot（<MPS_PILOT>）。
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

# 三折源年份配置（仅用于记录）
FOLD_CFG: dict[int, dict[str, Any]] = {
    2018: {"src_years": [2019, 2025]},
    2019: {"src_years": [2018, 2025]},
    2025: {"src_years": [2018, 2019]},
}
FOLD_YEARS = (2018, 2019, 2025)
ALL_REGIONS = ["XJ", "SD", "SX", "GS"]
REGION_CN = {"XJ": "新疆(XJ)", "SD": "山东(SD)", "SX": "陕西(SX)", "GS": "甘肃(GS)"}
REGION_EN = {"XJ": "Xinjiang", "SD": "Shandong", "SX": "Shaanxi", "GS": "Gansu"}
# 产地配色（Wong 安全色，禁止红绿组合）
REGION_COLORS = {"XJ": "#0072B2", "SD": "#E69F00", "SX": "#009E73", "GS": "#D55E00"}


def load_fold_encoders(
    encoder_seed: int, n_bands: int, device: torch.device, logger: eu.Logger, fallback_feat_dim: int
) -> dict[int, ProtoEncoder]:
    """加载 10 产出的各折编码器（新格式 ckpt['state_dict']/['feat_dim']）。"""
    encoders: dict[int, ProtoEncoder] = {}
    for yr in FOLD_YEARS:
        run_name = f"10_loeo_year_foldwise_seed{encoder_seed}_fold{yr}"
        ckpt_path = os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt")
        if not os.path.exists(ckpt_path):
            logger.log(f"  警告：{ckpt_path} 不存在，跳过 LOEO-{yr}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        enc = ProtoEncoder(n_bands, feat_dim=int(ckpt.get("feat_dim", fallback_feat_dim)))
        enc.load_state_dict(ckpt["state_dict"])
        enc.to(device)
        enc.eval()
        encoders[yr] = enc
        logger.log(f"  LOEO-{yr} 折编码器加载 {ckpt_path}（feat_dim={ckpt.get('feat_dim')}）")
    return encoders


def eval_protonet_perclass(
    encoder: ProtoEncoder, year_data: Any, tgt_year: int, k_shot: int, seed: int, device: torch.device
) -> tuple[float, dict[str, float], dict[str, float]]:
    """折内编码器 ProtoNet K-shot（多面位均值特征）；返回 (宏F1, 各产地Recall, 各产地F1)。"""
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

    tgt_protos: dict[str, NDArray] = {}
    for reg in tgt_regions:
        feats = [apple_feature(encoder, year_data[tgt_year][reg][aid], device) for aid in support_ids[reg]]
        p = np.mean(feats, axis=0)
        tgt_protos[reg] = p / (float(np.linalg.norm(p)) + 1e-8)
    proto_mat = np.stack([tgt_protos[r] for r in tgt_regions])

    y_true: list[str] = []
    y_pred: list[str] = []
    for aid, true_reg in query_items:
        feat = apple_feature(encoder, year_data[tgt_year][true_reg][aid], device).reshape(1, -1)
        y_true.append(true_reg)
        y_pred.append(tgt_regions[int((proto_mat @ feat.T).flatten().argmax())])

    region_recall: dict[str, float] = {}
    for reg in tgt_regions:
        idxs = [i for i, r in enumerate(y_true) if r == reg]
        if not idxs:
            region_recall[reg] = float("nan")
        else:
            region_recall[reg] = sum(1 for i in idxs if y_pred[i] == reg) / len(idxs)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_f1 = {
        reg: float(f1_score(y_true, y_pred, labels=[reg], average="macro", zero_division=0)) for reg in tgt_regions
    }
    return macro_f1, region_recall, per_class_f1


def _build_txt() -> dict[str, dict[str, str]]:
    return {
        "zh": {
            "cbar": "Recall",
            "fold_x": "产地",
            "fold_y": "K-Shot",
            "ta_sup": "折内编码器下各产地 Recall 热力图（无训练污染）",
            "tb_y": "Per-Class F1（K=5）",
            "tb_sup": "K=5 各折各产地 Per-Class F1（折内编码器，无训练污染）（附录 Supplementary）",
            "macro": "宏F1",
        },
        "en": {
            "cbar": "Recall",
            "fold_x": "Origin",
            "fold_y": "K-shot",
            "ta_sup": "Per-origin recall heatmap (fold-restricted encoder, leakage-free)",
            "tb_y": "Per-class F1 (K=5)",
            "tb_sup": "K=5 per-origin per-class F1 (fold-restricted encoder, leakage-free) (Supplementary)",
            "macro": "Macro-F1",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="19_perclass_f1_foldwise：折内编码器 per-class F1/Recall")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k_list", default="1,3,5,10,20,50")
    parser.add_argument("--seeds", type=int, default=50, help="评估重复次数（目标域 support 重采样）")
    parser.add_argument("--feat_dim", type=int, default=128, help="编码器 feat_dim 回退值（新格式 ckpt 内已含）")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--encoder_seed", type=int, default=eu.PILOT_SEED, help="定位 10 的 fold 编码器种子")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_list.split(",")]
    rep_seeds = [args.seed + i for i in range(args.seeds)]
    device = eu.get_device(args.device)
    logger = eu.get_logger("19_perclass_f1_foldwise")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "折内编码器 per-class F1/Recall",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "reps": args.seeds,
            "K_list": k_values,
            "encoder_seed": args.encoder_seed,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    year_data = build_year_region_data(df, wl_cols)

    fold_encoders = load_fold_encoders(args.encoder_seed, n_bands, device, logger, args.feat_dim)
    if not fold_encoders:
        logger.log("错误：未找到任何折内编码器，请先运行 10_loeo_year_foldwise.py")
        logger.close()
        raise SystemExit(1)

    all_rows: list[dict[str, Any]] = []
    with eu.track_compute("per-class 评估", logger, run) as budget:
        for tgt_year in FOLD_YEARS:
            if tgt_year not in fold_encoders:
                logger.log(f"\nLOEO-{tgt_year} 编码器缺失，跳过")
                continue
            enc = fold_encoders[tgt_year]
            tgt_regions = sorted(year_data[tgt_year].keys())
            logger.log(f"\n【LOEO-{tgt_year}】 目标产地={tgt_regions} 源年份={FOLD_CFG[tgt_year]['src_years']}")
            for k in k_values:
                macro_list: list[float] = []
                recall_lists: dict[str, list[float]] = {r: [] for r in tgt_regions}
                f1_lists: dict[str, list[float]] = {r: [] for r in tgt_regions}
                for s in rep_seeds:
                    macro_f1, region_recall, per_class_f1 = eval_protonet_perclass(
                        enc, year_data, tgt_year, k, s, device
                    )
                    macro_list.append(macro_f1)
                    for r in tgt_regions:
                        recall_lists[r].append(region_recall.get(r, float("nan")))
                        f1_lists[r].append(per_class_f1.get(r, float("nan")))
                row: dict[str, Any] = {
                    "目标年份": tgt_year,
                    "源年份": "+".join(str(y) for y in FOLD_CFG[tgt_year]["src_years"]),
                    "目标产地": "/".join(tgt_regions),
                    "K": k,
                    "宏F1均值": round(float(np.mean(macro_list)), 4),
                    "宏F1标准差": round(float(np.std(macro_list)), 4),
                }
                for r in tgt_regions:
                    row[f"Recall_{r}_均值"] = round(float(np.nanmean(recall_lists[r])), 4)
                    row[f"Recall_{r}_std"] = round(float(np.nanstd(recall_lists[r])), 4)
                    row[f"F1_{r}_均值"] = round(float(np.nanmean(f1_lists[r])), 4)
                    row[f"F1_{r}_std"] = round(float(np.nanstd(f1_lists[r])), 4)
                all_rows.append(row)
                recall_str = "  ".join(f"{r}={float(np.nanmean(recall_lists[r])):.3f}" for r in tgt_regions)
                logger.log(f"  K={k:3d}  宏F1={row['宏F1均值']:.4f}±{row['宏F1标准差']:.4f}  recall: {recall_str}")

    res_df = pd.DataFrame(all_rows)
    logger.log("\n各折各产地 Recall（K=5，难→易）：")
    for tgt_year in FOLD_YEARS:
        sub = res_df[(res_df["目标年份"] == tgt_year) & (res_df["K"] == 5)]
        if sub.empty:
            continue
        srow = sub.iloc[0]
        tgt_regions = sorted(year_data[tgt_year].keys())
        pairs = [(r, float(srow.get(f"Recall_{r}_均值", float("nan")))) for r in tgt_regions]
        pairs.sort(key=lambda x: x[1])
        logger.log(f"  LOEO-{tgt_year}: " + " < ".join(f"{r}({v:.3f})" for r, v in pairs))

    present_years = [yr for yr in FOLD_YEARS if yr in fold_encoders]
    txt = _build_txt()

    def plot_a(lang: str) -> Any:
        t = txt[lang]
        region_label = REGION_CN if lang == "zh" else REGION_EN
        fig, axes = plt.subplots(1, len(present_years), figsize=(4.7 * len(present_years), 4), squeeze=False)
        for ax, tgt_year in zip(axes[0], present_years, strict=True):
            sub = res_df[res_df["目标年份"] == tgt_year]
            tgt_regions = sorted(year_data[tgt_year].keys())
            recall_mat = []
            for k in k_values:
                krow = sub[sub["K"] == k]
                if krow.empty:
                    recall_mat.append([float("nan")] * len(tgt_regions))
                else:
                    krow0 = krow.iloc[0]
                    recall_mat.append([float(krow0.get(f"Recall_{r}_均值", float("nan"))) for r in tgt_regions])
            recall_arr = np.array(recall_mat)
            im = ax.imshow(recall_arr, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks(range(len(tgt_regions)))
            ax.set_xticklabels([region_label.get(r, r) for r in tgt_regions], fontsize=9)
            ax.set_yticks(range(len(k_values)))
            ax.set_yticklabels([f"K={k}" for k in k_values], fontsize=8)
            ax.set_title(f"LOEO-{tgt_year}", fontsize=10, fontweight="bold")
            ax.set_xlabel(t["fold_x"], fontsize=9)
            ax.set_ylabel(t["fold_y"], fontsize=9)
            for ri in range(recall_arr.shape[0]):
                for ci in range(recall_arr.shape[1]):
                    v = recall_arr[ri, ci]
                    if not np.isnan(v):
                        ax.text(
                            ci,
                            ri,
                            f"{v:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if v < 0.5 else "black",
                            fontweight="bold",
                        )
            plt.colorbar(im, ax=ax, shrink=0.8, label=t["cbar"])
        fig.suptitle(t["ta_sup"], fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_b(lang: str) -> Any:
        t = txt[lang]
        region_label = REGION_CN if lang == "zh" else REGION_EN
        n_fold = len(present_years)
        fig, axes = plt.subplots(1, n_fold, figsize=(4.7 * n_fold, 5), squeeze=False, sharey=True)
        for ax, tgt_year in zip(axes[0], present_years, strict=True):
            sub5 = res_df[(res_df["目标年份"] == tgt_year) & (res_df["K"] == 5)]
            if sub5.empty:
                ax.set_visible(False)
                continue
            srow = sub5.iloc[0]
            tgt_regions = sorted(year_data[tgt_year].keys())
            f1_vals = [float(srow.get(f"F1_{r}_均值", float("nan"))) for r in tgt_regions]
            f1_errs = [float(srow.get(f"F1_{r}_std", 0.0)) for r in tgt_regions]
            colors = [REGION_COLORS.get(r, "#999999") for r in tgt_regions]
            x = np.arange(len(tgt_regions))
            bars = ax.bar(x, f1_vals, 0.55, color=colors, alpha=0.9, yerr=f1_errs, capsize=5)
            for bar, v, e in zip(bars, f1_vals, f1_errs, strict=True):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + e + 0.01,
                        f"{v:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )
            macro_f1 = float(srow["宏F1均值"])
            ax.axhline(macro_f1, color="#555555", linestyle="--", linewidth=1.2, label=f"{t['macro']}={macro_f1:.3f}")
            ax.set_xticks(x)
            ax.set_xticklabels([region_label.get(r, r) for r in tgt_regions], fontsize=9)
            ax.set_ylabel(t["tb_y"], fontsize=9)
            ax.set_title(f"LOEO-{tgt_year}", fontsize=10, fontweight="bold")
            ax.set_ylim(0, 1.12)
            ax.axhline(1 / 3, color="lightgray", linestyle=":", linewidth=1.0)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(t["tb_sup"], fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    for idx, fn in enumerate((plot_a, plot_b)):
        zh, en = eu.save_figure_bilingual(fn, "19_perclass_f1_foldwise", idx)
        logger.log(f"图{chr(97 + idx)} → {zh} / {en}")

    budget_df = pd.DataFrame(
        [{"n_folds": len(present_years), "reps": args.seeds, "encoder_seed": args.encoder_seed, **budget}]
    )
    xlsx = eu.write_script_workbook(__file__, {0: ("折内per-class结果", res_df), "compute": budget_df})
    logger.log(f"结果表 → {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
