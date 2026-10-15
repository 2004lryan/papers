"""
10_loeo_year_foldwise.py: 折内编码器重训 LOEO-Year K-Shot（fold-restricted，官方干净协议）。

对每个 LOEO-Year 折单独重训编码器，目标年份数据完全排除在训练之外，消除"全数据共享
编码器"的评估泄漏（见 05_meta_train 的共享编码器对照）。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/10_loeo_year_foldwise.py --device auto --episodes 2000

输出文件:
    04outputs/10_loeo_year_foldwise-图a.pdf / -fig-a.pdf  — 三折 K-Shot F1 曲线（折内编码器）
    04outputs/10_loeo_year_foldwise-图b.pdf / -fig-b.pdf  — 折内 vs 共享编码器 F1（泄漏量，K=5）
    04outputs/10_loeo_year_foldwise.xlsx                   — 完整数值 + 算力
    07checkpoints/10_loeo_year_foldwise_seed<seed>_fold<year>/best.pt — 各折专属编码器（含模型卡）
    05logs/10_loeo_year_foldwise_YY-MM-DD_HHMMSS.log       — 运行日志

说明:
    --seed 为单次运行的模型种子；Formal 阶段须以 RECOMMENDED_SEEDS 各跑一次并在下游聚合。
    每个 K 下的 eval_reps 为目标域 support 重采样（episode 级），n_primary=K_outer=3 折。
    MPS 下为 Pilot（<MPS_PILOT>）。
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

import export_utils as eu
from protonet import EpisodeSampler, ProtoEncoder, apple_feature, build_apple_data, build_year_region_data, proto_loss

# 三折配置：key=留出年份，value=源元素与源年份
FOLD_CFG: dict[int, dict[str, Any]] = {
    2018: {"src_elems": ["E4", "E5", "E6", "E7", "E8", "E9"], "src_years": [2019, 2025]},
    2019: {"src_elems": ["E1", "E2", "E3", "E7", "E8", "E9"], "src_years": [2018, 2025]},
    2025: {"src_elems": ["E1", "E2", "E3", "E4", "E5", "E6"], "src_years": [2018, 2019]},
}


def train_fold_encoder(
    fold_year: int,
    src_elems: list[str],
    apple_data: dict[str, dict[Any, dict[str, Any]]],
    n_bands: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: eu.Logger,
    run: Any,
) -> ProtoEncoder:
    """在仅含源域元素的数据上训练 ProtoNet 编码器（目标年份隔离），落盘并返回最优编码器。"""
    run_name = f"10_loeo_year_foldwise_seed{args.seed}_fold{fold_year}"
    ckpt_path = os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        enc = ProtoEncoder(n_bands, feat_dim=int(ckpt.get("feat_dim", args.feat_dim)))
        enc.load_state_dict(ckpt["state_dict"])
        enc.to(device)
        enc.eval()
        logger.log(f"  [折{fold_year}] 复用已存编码器 {ckpt_path}（验证Acc={ckpt.get('val_acc', 0):.4f}）")
        return enc

    logger.log(f"\n{'=' * 56}\n【折内训练】留出 {fold_year}  源域元素={src_elems}")
    sampler = EpisodeSampler(src_elems, apple_data, args.n_way, args.k_support, args.q_query)
    encoder = ProtoEncoder(n_bands, feat_dim=args.feat_dim)
    encoder.to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.episodes, eta_min=1e-5)
    best_val_acc = 0.0
    no_improve = 0

    for ep in range(1, args.episodes + 1):
        encoder.train()
        sup_x, sup_y, qry_x, qry_y, nw = sampler.sample()
        optimizer.zero_grad()
        loss, _ = proto_loss(encoder, sup_x, sup_y, qry_x, qry_y, nw, device)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        if ep % args.val_every == 0:
            encoder.eval()
            v_accs: list[float] = []
            with torch.no_grad():
                for _ in range(30):
                    sv_x, sv_y, qv_x, qv_y, vnw = sampler.sample()
                    _, va = proto_loss(encoder, sv_x, sv_y, qv_x, qv_y, vnw, device)
                    v_accs.append(va)
            vm_acc = float(np.mean(v_accs))
            logger.log(f"  ep={ep:5d}/{args.episodes}  折内验证Acc={vm_acc:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")
            if vm_acc > best_val_acc:
                best_val_acc, no_improve = vm_acc, 0
                eu.save_best_checkpoint(
                    encoder,
                    run_name,
                    {
                        "stage": args.stage,
                        "training_hardware": device.type,
                        "deterministic_guarantee": "full" if device.type == "cuda" else "partial(MPS pilot)",
                        "architecture": "ProtoEncoder(1D-CNN+MLP)",
                        "task": f"fold-restricted 编码器（留出 {fold_year}）",
                        "fold_year": fold_year,
                        "src_elems": ",".join(src_elems),
                        "val_acc": round(best_val_acc, 4),
                    },
                    extra_state={
                        "n_bands": n_bands,
                        "feat_dim": args.feat_dim,
                        "fold_year": fold_year,
                        "val_acc": best_val_acc,
                    },
                )
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    logger.log(f"  早停：{no_improve} 次无提升，共 {ep} episodes")
                    break

    if run is not None:
        run.log({f"fold{fold_year}/best_val_acc": best_val_acc})
    logger.log(f"  [折{fold_year}] 训练结束 最优折内验证Acc={best_val_acc:.4f} → {ckpt_path}")
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(best["state_dict"])
    encoder.eval()
    return encoder


def kshot_eval_fold(
    encoder: ProtoEncoder,
    year_data: Any,
    tgt_year: int,
    src_years: list[int],
    k_shot: int,
    seed: int,
    device: torch.device,
    src_alpha: float,
) -> dict[str, Any]:
    """折内编码器对目标年份的 K-Shot 多类分类评估（含源域同类原型融合 src_alpha）。"""
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
    parser = argparse.ArgumentParser(description="10_loeo_year_foldwise：fold-restricted LOEO-Year K-Shot")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--n_way", type=int, default=3)
    parser.add_argument("--k_support", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--k_list", default="1,3,5,10,20,50")
    parser.add_argument("--eval_reps", type=int, default=15, help="K-shot 目标域 support 重采样次数（episode 级）")
    parser.add_argument("--src_alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_list.split(",")]
    device = eu.get_device(args.device)
    logger = eu.get_logger("10_loeo_year_foldwise")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "fold-restricted LOEO-Year",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "episodes/折": args.episodes,
            "K_list": k_values,
            "eval_reps": args.eval_reps,
            "src_alpha": args.src_alpha,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    apple_data = build_apple_data(df, wl_cols)
    year_data = build_year_region_data(df, wl_cols)
    rep_seeds = [args.seed + i for i in range(args.eval_reps)]

    all_rows: list[dict[str, Any]] = []
    with eu.track_compute("LOEO-Year 折内", logger, run) as budget:
        for tgt_year, cfg in sorted(FOLD_CFG.items()):
            fold_encoder = train_fold_encoder(
                tgt_year, cfg["src_elems"], apple_data, n_bands, args, device, logger, run
            )
            tgt_regs = sorted(year_data[tgt_year].keys())
            logger.log(f"\n评估 LOEO-{tgt_year}（目标产地={tgt_regs}）...")
            for k in k_values:
                accs: list[float] = []
                f1s: list[float] = []
                cls_f1s: dict[str, list[float]] = {r: [] for r in tgt_regs}
                for s in rep_seeds:
                    res = kshot_eval_fold(
                        fold_encoder, year_data, tgt_year, cfg["src_years"], k, s, device, args.src_alpha
                    )
                    accs.append(res["acc"])
                    f1s.append(res["f1"])
                    for r in tgt_regs:
                        cls_f1s[r].append(res["per_class_f1"].get(r, 0.0))
                row: dict[str, Any] = {
                    "目标年份": tgt_year,
                    "源年份": "+".join(str(y) for y in cfg["src_years"]),
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
                logger.log(f"  K={k:3d}  F1={row['F1均值']:.4f}±{row['F1标准差']:.4f}  Acc={row['Acc均值']:.4f}")

    res_df = pd.DataFrame(all_rows)
    mean_f1_k5 = float(res_df[res_df["K"] == 5]["F1均值"].mean())
    logger.log(f"\n三折平均 Macro-F1（K=5，fold-restricted）= {mean_f1_k5:.4f}")

    # 共享编码器对照（读 06 的 xlsx 头牌结果表），用于可视化泄漏量；缺失则跳过
    shared_xlsx = os.path.join(eu.OUTPUT_DIR, "06_loeo_kshot.xlsx")
    shared_df = None
    if os.path.exists(shared_xlsx):
        try:
            shared_df = pd.read_excel(shared_xlsx, sheet_name=0)
        except Exception:
            shared_df = None

    fold_colors = {2018: "#0072B2", 2019: "#009E73", 2025: "#D55E00"}
    txt = {
        "zh": {
            "x": "K-Shot（每产地支持苹果数）",
            "y": "Macro-F1（3 类产地）",
            "rand": "随机基线",
            "ta": "LOEO-Year K-Shot（折内重训编码器，无泄漏）",
            "lo": "留出",
            "tb": "折内 vs 共享编码器（K=5，泄漏量）",
            "fr": "折内重训",
            "sh": "共享编码器(有泄漏)",
        },
        "en": {
            "x": "K-shot (support apples per origin)",
            "y": "Macro-F1 (3 origins)",
            "rand": "Chance",
            "ta": "LOEO-Year K-shot (fold-restricted encoder, leakage-free)",
            "lo": "hold-out",
            "tb": "Fold-restricted vs shared encoder (K=5, leakage)",
            "fr": "Fold-restricted",
            "sh": "Shared (leaky)",
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
        ax.set_ylabel(t["y"])
        ax.set_title(t["ta"])
        ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.5, label=t["rand"])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_b(lang: str) -> Any:
        t = txt[lang]
        fig, ax = plt.subplots(figsize=(9, 5))
        new_sub = res_df[res_df["K"] == 5].sort_values("目标年份")
        x = np.arange(len(new_sub))
        ax.bar(
            x - 0.2, new_sub["F1均值"], width=0.35, color="#0072B2", label=t["fr"], yerr=new_sub["F1标准差"], capsize=4
        )
        if shared_df is not None and "K" in shared_df.columns and "F1均值" in shared_df.columns:
            old_sub = shared_df[shared_df["K"] == 5].sort_values("目标年份")
            if len(old_sub) == len(new_sub):
                ax.bar(x + 0.2, old_sub["F1均值"], width=0.35, color="#D55E00", label=t["sh"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t['lo']} {yr}" for yr in new_sub["目标年份"].tolist()])
        ax.set_ylabel(t["y"])
        ax.set_title(t["tb"])
        ax.set_ylim(0, 1.05)
        ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    for idx, fn in enumerate((plot_a, plot_b)):
        zh, en = eu.save_figure_bilingual(fn, "10_loeo_year_foldwise", idx)
        logger.log(f"图{chr(97 + idx)} → {zh} / {en}")

    budget_df = pd.DataFrame([{"mean_f1_k5": mean_f1_k5, "n_folds": 3, "eval_reps": args.eval_reps, **budget}])
    xlsx = eu.write_script_workbook(__file__, {0: ("折内K-Shot结果", res_df), "compute": budget_df})
    logger.log(f"结果表 → {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
