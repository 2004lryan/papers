"""06_loeo_kshot.py: LOEO-Year × K-Shot 产地分类（共享编码器，泄漏对照面）。

每轮留出整个年份（2018/2019/2025）为目标域，其余两年为源域；目标域 3 产地为真实多类
任务。用每产地 K 个标注苹果构建原型，并按 src_alpha 融合源域同类原型，评估不同 K 下性能。
2025 折中甘肃（GS）在源域不存在，是检验开放集 K-shot 适应能力的新类。本脚本为 fold-restricted
之前的早期版，载入 05 产出的「共享编码器」（在全部训练元素上预训练），是论文量化的评估泄漏
对照面；干净版见 10_loeo_year_foldwise.py。单模型单指标比较，按 §13 跳过 LLH-GBID。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/06_loeo_kshot.py --device auto

输出文件:
    04outputs/06_loeo_kshot-图a.pdf / -fig-a.pdf  — 三折 LOEO-Year K-Shot Macro-F1 曲线
    04outputs/06_loeo_kshot-图b.pdf / -fig-b.pdf  — 各年份×产地 F1 热力图（K=最大值）
    04outputs/06_loeo_kshot-图c.pdf / -fig-c.pdf  — 留出 2025 新类 GS vs 已知类 K-Shot 曲线
    04outputs/06_loeo_kshot.xlsx                   — 汇总结果 + 产地级明细 + 算力
    05logs/06_loeo_kshot_YY-MM-DD_HHMMSS.log       — 运行日志

说明:
    依赖 05 的共享编码器：07checkpoints/05_meta_train_seed<encoder_seed>/best.pt（--encoder_seed
    指定，默认 PILOT_SEED）。--seed 为评估重采样基种子。MPS 下为 Pilot（<MPS_PILOT>）。
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

import export_utils as eu
from protonet import ProtoEncoder, apple_feature, build_year_region_data

# 三折 LOEO-Year 配置：key=留出（目标）年份，value=源年份
YEAR_FOLDS: dict[int, list[int]] = {
    2018: [2019, 2025],
    2019: [2018, 2025],
    2025: [2018, 2019],
}


def load_shared_encoder(encoder_seed: int, n_bands: int, device: torch.device, logger: eu.Logger) -> ProtoEncoder:
    """加载 05 产出的共享编码器（新格式 ckpt['state_dict']/['feat_dim']），用 ProtoEncoder 重建。"""
    run_name = f"05_meta_train_seed{encoder_seed}"
    ckpt_path = os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"共享编码器不存在：{ckpt_path}\n请先运行 05_meta_train.py --seed {encoder_seed} 训练共享编码器。"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    feat_dim = int(ckpt.get("feat_dim", 128))
    enc = ProtoEncoder(n_bands, feat_dim=feat_dim)
    enc.load_state_dict(ckpt["state_dict"])
    enc.to(device)
    enc.eval()
    logger.log(
        f"共享编码器加载成功 {ckpt_path}  feat_dim={feat_dim}  训练验证Acc={float(ckpt.get('val_acc', 0.0)):.4f}"
    )
    return enc


def loeo_year_kshot_eval(
    encoder: ProtoEncoder,
    year_data: Any,
    tgt_year: int,
    src_years: list[int],
    k_shot: int,
    seed: int,
    device: torch.device,
    src_alpha: float,
) -> dict[str, Any]:
    """共享编码器对目标年份的 K-Shot 多类分类评估（目标域 K-shot 原型 + src_alpha 源域同类融合）。"""
    rng = np.random.default_rng(seed)
    tgt_regions = sorted(year_data[tgt_year].keys())
    support_ids: dict[str, list[Any]] = {}
    for reg in tgt_regions:
        apples = list(year_data[tgt_year][reg].keys())
        k = min(k_shot, max(1, len(apples) - 1))
        support_ids[reg] = list(rng.choice(apples, size=k, replace=False))
    sup_set = {aid for aids in support_ids.values() for aid in aids}
    query_items = [
        (faces, reg) for reg in tgt_regions for aid, faces in year_data[tgt_year][reg].items() if aid not in sup_set
    ]
    if not query_items:
        query_items = [(faces, reg) for reg in tgt_regions for faces in year_data[tgt_year][reg].values()]

    tgt_protos: dict[str, Any] = {}
    for reg in tgt_regions:
        feats = [apple_feature(encoder, year_data[tgt_year][reg][aid], device) for aid in support_ids[reg]]
        p = np.mean(feats, axis=0)
        tgt_protos[reg] = p / (float(np.linalg.norm(p)) + 1e-8)

    src_protos: dict[str, list[Any]] = {}
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
    for faces, true_reg in query_items:
        feat = apple_feature(encoder, faces, device).reshape(1, -1)
        sim = (feat @ proto_mat.T)[0]
        y_true.append(true_reg)
        y_pred.append(tgt_regions[int(sim.argmax())])
    per_class = f1_score(y_true, y_pred, labels=tgt_regions, average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {r: float(v) for r, v in zip(tgt_regions, per_class, strict=False)},
        "n_query": len(query_items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="06_loeo_kshot：LOEO-Year×K-Shot（共享编码器，泄漏对照面）")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k_list", default="1,3,5,10,20,50")
    parser.add_argument("--seeds", type=int, default=10, help="每个 K 的目标域 support 重采样次数")
    parser.add_argument("--src_alpha", type=float, default=0.4, help="源域原型融合权重（0=纯目标域，1=纯源域）")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--encoder_seed", type=int, default=eu.PILOT_SEED, help="定位 05 的共享编码器种子")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_list.split(",")]
    device = eu.get_device(args.device)
    logger = eu.get_logger("06_loeo_kshot")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "LOEO-Year×K-Shot（共享编码器，泄漏对照面）",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "K_list": k_values,
            "重采样次数": args.seeds,
            "src_alpha": args.src_alpha,
            "encoder_seed": args.encoder_seed,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    year_data = build_year_region_data(df, wl_cols)
    encoder = load_shared_encoder(args.encoder_seed, n_bands, device, logger)
    rep_seeds = [args.seed + i for i in range(args.seeds)]

    all_rows: list[dict[str, Any]] = []
    cls_rows: list[dict[str, Any]] = []
    with eu.track_compute("LOEO-Year K-Shot（共享编码器）", logger, run) as budget:
        for tgt_year in sorted(YEAR_FOLDS):
            src_years = YEAR_FOLDS[tgt_year]
            tgt_regs = sorted(year_data[tgt_year].keys())
            logger.log(f"\n【留出年份 {tgt_year}】 目标产地={tgt_regs}  源年份={src_years}")
            for k in k_values:
                accs: list[float] = []
                f1s: list[float] = []
                cls_f1s: dict[str, list[float]] = {r: [] for r in tgt_regs}
                for s in rep_seeds:
                    res = loeo_year_kshot_eval(encoder, year_data, tgt_year, src_years, k, s, device, args.src_alpha)
                    accs.append(res["acc"])
                    f1s.append(res["f1"])
                    for r in tgt_regs:
                        cls_f1s[r].append(res["per_class_f1"].get(r, 0.0))
                row: dict[str, Any] = {
                    "目标年份": tgt_year,
                    "源年份": "+".join(str(y) for y in src_years),
                    "目标产地": "/".join(tgt_regs),
                    "K": k,
                    "Acc均值": round(float(np.mean(accs)), 4),
                    "Acc标准差": round(float(np.std(accs)), 4),
                    "F1均值": round(float(np.mean(f1s)), 4),
                    "F1标准差": round(float(np.std(f1s)), 4),
                }
                for r in tgt_regs:
                    row[f"F1_{r}"] = round(float(np.mean(cls_f1s[r])), 4)
                all_rows.append(row)
                for r in tgt_regs:
                    cls_rows.append(
                        {
                            "目标年份": tgt_year,
                            "K": k,
                            "产地": r,
                            "F1均值": round(float(np.mean(cls_f1s[r])), 4),
                            "F1标准差": round(float(np.std(cls_f1s[r])), 4),
                            "是否新类": (r == "GS" and tgt_year == 2025),
                        }
                    )
                logger.log(
                    f"  K={k:3d}  F1={row['F1均值']:.4f}±{row['F1标准差']:.4f}  Acc={row['Acc均值']:.4f}  "
                    + "  ".join(f"{r}:{row[f'F1_{r}']:.3f}" for r in tgt_regs)
                )

    res_df = pd.DataFrame(all_rows)
    cls_df = pd.DataFrame(cls_rows)
    k_best = max(k_values)
    mean_f1_kbest = float(res_df[res_df["K"] == k_best]["F1均值"].mean())
    logger.log(f"\n三折平均 Macro-F1（K={k_best}，共享编码器）= {mean_f1_kbest:.4f}")

    # ─── 双语图文本字典 ──────────────────────────────────────────────────────────
    fold_colors = {2018: "#0072B2", 2019: "#009E73", 2025: "#D55E00"}
    region_color = {"XJ": "#0072B2", "SD": "#009E73", "GS": "#D55E00"}
    region_ls = {"XJ": "-", "SD": "-", "GS": "--"}
    txt = {
        "zh": {
            "x": "K-Shot（每产地支持苹果数）",
            "ya": "Macro-F1（3 类产地）",
            "rand": "随机基线（1/3）",
            "ta": "LOEO-Year×K-Shot 产地分类（共享编码器，泄漏对照面）",
            "lo": "留出",
            "tb": f"LOEO-Year K={k_best} 各产地 Macro-F1（共享编码器）",
            "bx": "目标产地",
            "by": "留出年份",
            "cbar": "Macro-F1",
            "new": "（新类）",
            "yc": "Per-class F1",
            "tc": "留出 2025：新类 GS vs 已知类 K-Shot 分类能力",
            "anno": "GS 在源域不存在，\n纯靠 K-shot 原型分类",
            "rXJ": "新疆 XJ（源域有）",
            "rSD": "山东 SD（源域有）",
            "rGS": "甘肃 GS（新类，源域无）",
        },
        "en": {
            "x": "K-shot (support apples per origin)",
            "ya": "Macro-F1 (3 origins)",
            "rand": "Chance (1/3)",
            "ta": "LOEO-year × K-shot origin classification (shared encoder, leakage reference)",
            "lo": "hold-out",
            "tb": f"LOEO-year K={k_best} per-origin Macro-F1 (shared encoder)",
            "bx": "Target origin",
            "by": "Hold-out year",
            "cbar": "Macro-F1",
            "new": " (novel)",
            "yc": "Per-class F1",
            "tc": "Hold-out 2025: novel-class GS vs known classes",
            "anno": "GS absent in source;\nclassified by K-shot prototype only",
            "rXJ": "XJ (in source)",
            "rSD": "SD (in source)",
            "rGS": "GS (novel, not in source)",
        },
    }

    def plot_a(lang: str) -> Any:
        t = txt[lang]
        fig, ax = plt.subplots(figsize=(9, 5))
        for yr in (2018, 2019, 2025):
            sub = res_df[res_df["目标年份"] == yr].sort_values("K")
            ax.plot(sub["K"], sub["F1均值"], marker="o", linewidth=2, color=fold_colors[yr], label=f"{t['lo']} {yr}")
            ax.fill_between(
                sub["K"],
                sub["F1均值"] - sub["F1标准差"],
                sub["F1均值"] + sub["F1标准差"],
                alpha=0.12,
                color=fold_colors[yr],
            )
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel(t["x"])
        ax.set_ylabel(t["ya"])
        ax.set_title(t["ta"])
        ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.5, label=t["rand"])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_b(lang: str) -> Any:
        t = txt[lang]
        best = cls_df[cls_df["K"] == k_best]
        pivot = best.pivot(index="目标年份", columns="产地", values="F1均值").reindex([2018, 2019, 2025])
        cols = sorted(pivot.columns.tolist())
        pivot = pivot[cols]
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.9, label=t["cbar"])
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(y) for y in pivot.index.tolist()])
        ax.set_xlabel(t["bx"])
        ax.set_ylabel(t["by"])
        ax.set_title(t["tb"])
        for i in range(len(pivot.index)):
            for j in range(len(cols)):
                v = pivot.values[i, j]
                if np.isnan(v):
                    continue
                anno = f"{v:.3f}"
                if int(pivot.index[i]) == 2025 and cols[j] == "GS":
                    anno += t["new"]
                ax.text(
                    j,
                    i,
                    anno,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white" if v < 0.5 else "black",
                )
        fig.tight_layout()
        return fig

    def plot_c(lang: str) -> Any:
        t = txt[lang]
        sub_2025 = cls_df[cls_df["目标年份"] == 2025].sort_values("K")
        labels = {"XJ": t["rXJ"], "SD": t["rSD"], "GS": t["rGS"]}
        fig, ax = plt.subplots(figsize=(8, 5))
        for reg in sorted(sub_2025["产地"].unique().tolist()):
            sub = sub_2025[sub_2025["产地"] == reg].sort_values("K")
            ax.plot(
                sub["K"],
                sub["F1均值"],
                marker="o",
                linewidth=2,
                linestyle=region_ls.get(reg, "-"),
                color=region_color.get(reg, "#999999"),
                label=labels.get(reg, reg),
            )
            ax.fill_between(
                sub["K"],
                sub["F1均值"] - sub["F1标准差"],
                sub["F1均值"] + sub["F1标准差"],
                alpha=0.12,
                color=region_color.get(reg, "#999999"),
            )
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel(t["x"])
        ax.set_ylabel(t["yc"])
        ax.set_title(t["tc"])
        ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.5, label=t["rand"])
        ax.set_ylim(0, 1.05)
        gs_sub = sub_2025[sub_2025["产地"] == "GS"].sort_values("K")
        if not gs_sub.empty:
            y0 = float(gs_sub["F1均值"].iloc[0])
            ax.annotate(
                t["anno"],
                xy=(k_values[0], y0),
                xytext=(k_values[min(1, len(k_values) - 1)] * 1.3, y0 - 0.15),
                arrowprops={"arrowstyle": "->", "color": region_color["GS"]},
                fontsize=9,
                color=region_color["GS"],
            )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    for idx, fn in enumerate((plot_a, plot_b, plot_c)):
        zh, en = eu.save_figure_bilingual(fn, "06_loeo_kshot", idx)
        logger.log(f"图{chr(97 + idx)} → {zh} / {en}")

    budget_df = pd.DataFrame(
        [
            {
                "mean_f1_kbest": mean_f1_kbest,
                "k_best": k_best,
                "reps": args.seeds,
                "encoder_seed": args.encoder_seed,
                **budget,
            }
        ]
    )
    xlsx = eu.write_script_workbook(
        __file__,
        {0: ("LOEO-Year K-Shot结果", res_df), 1: ("产地级F1明细", cls_df), "compute": budget_df},
    )
    logger.log(f"结果表 → {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
