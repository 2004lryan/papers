"""
05_meta_train.py: ProtoNet 元学习预训练（共享编码器，泄漏对照基线）。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/05_meta_train.py --device auto --episodes 3000

输出文件:
    04outputs/05_meta_train-图a.pdf / -fig-a.pdf  — 训练/验证 损失与准确率曲线
    04outputs/05_meta_train.xlsx                   — 每轮验证指标 + 算力
    07checkpoints/05_meta_train_seed<seed>/best.pt — 最优编码器（含模型卡 + SHA-256）
    05logs/05_meta_train_YY-MM-DD_HHMMSS.log       — 运行日志

说明:
    本脚本产出"共享编码器"（在全部训练元素上预训练），是论文中量化的评估泄漏对照面；
    干净的 fold-restricted 编码器见 10_loeo_year_foldwise.py。MPS 下为 Pilot（<MPS_PILOT>）。
    ProtoEncoder / proto_loss / EpisodeSampler 等共享组件来自 protonet.py。
"""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import export_utils as eu
from protonet import ELEM_INFO, EpisodeSampler, ProtoEncoder, build_apple_data, proto_loss

# 训练/验证元素（同分布验证；E9 甘肃新类留作下游少样本评估）
TRAIN_ELEMS: list[str] = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]


def main() -> None:
    parser = argparse.ArgumentParser(description="05_meta_train：ProtoNet 元学习预训练")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--n_way", type=int, default=3)
    parser.add_argument("--k_support", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    device = eu.get_device(args.device)
    logger = eu.get_logger("05_meta_train")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "ProtoNet 元预训练",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "episodes": args.episodes,
            "n_way": args.n_way,
            "k_support": args.k_support,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    apple_data = build_apple_data(df, wl_cols)
    for eid in TRAIN_ELEMS:
        logger.log(f"  {eid} {ELEM_INFO[eid]}: {len(apple_data[eid])} 苹果")

    sampler = EpisodeSampler(TRAIN_ELEMS, apple_data, args.n_way, args.k_support, args.q_query)
    encoder = ProtoEncoder(n_bands, feat_dim=args.feat_dim)
    encoder.to(device)
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.log(f"编码器参数量：{n_params:,}  产地={sampler.all_regions}")
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.episodes, eta_min=1e-5)

    run_name = f"05_meta_train_seed{args.seed}"
    train_losses: list[float] = []
    train_accs: list[float] = []
    val_episodes: list[int] = []
    val_losses: list[float] = []
    val_accs: list[float] = []
    best_val_acc = 0.0
    best_episode = 0
    no_improve = 0

    with eu.track_compute("元训练", logger, run) as budget:
        for ep in range(1, args.episodes + 1):
            encoder.train()
            sup_x, sup_y, qry_x, qry_y, nw = sampler.sample()
            optimizer.zero_grad()
            loss, acc = proto_loss(encoder, sup_x, sup_y, qry_x, qry_y, nw, device)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))
            train_accs.append(acc)

            if ep % args.val_every == 0:
                encoder.eval()
                vl_list: list[float] = []
                va_list: list[float] = []
                with torch.no_grad():
                    for _ in range(50):
                        sv_x, sv_y, qv_x, qv_y, vnw = sampler.sample()
                        vl, va = proto_loss(encoder, sv_x, sv_y, qv_x, qv_y, vnw, device)
                        vl_list.append(float(vl.item()))
                        va_list.append(va)
                vm_loss, vm_acc = float(np.mean(vl_list)), float(np.mean(va_list))
                val_episodes.append(ep)
                val_losses.append(vm_loss)
                val_accs.append(vm_acc)
                tr_acc = float(np.mean(train_accs[-args.val_every :]))
                tr_los = float(np.mean(train_losses[-args.val_every :]))
                logger.log(
                    f"  Episode {ep:5d}/{args.episodes}  训练损失={tr_los:.4f} 训练Acc={tr_acc:.4f}  "
                    f"验证损失={vm_loss:.4f} 验证Acc={vm_acc:.4f}  lr={scheduler.get_last_lr()[0]:.6f}"
                )
                if run is not None:
                    run.log(
                        {"train/loss": tr_los, "train/acc": tr_acc, "val/loss": vm_loss, "val/acc": vm_acc},
                        step=ep,
                    )
                if vm_acc > best_val_acc:
                    best_val_acc, best_episode, no_improve = vm_acc, ep, 0
                    eu.save_best_checkpoint(
                        encoder,
                        run_name,
                        {
                            "stage": args.stage,
                            "training_hardware": device.type,
                            "deterministic_guarantee": "full" if device.type == "cuda" else "partial(MPS pilot)",
                            "architecture": "ProtoEncoder(1D-CNN+MLP)",
                            "params": n_params,
                            "task": "ProtoNet 共享编码器（泄漏对照基线）",
                            "val_acc": round(best_val_acc, 4),
                            "best_episode": best_episode,
                        },
                        extra_state={
                            "n_bands": n_bands,
                            "feat_dim": args.feat_dim,
                            "val_acc": best_val_acc,
                            "args": vars(args),
                        },
                    )
                    logger.log(f"    ★ 新最优 验证Acc={best_val_acc:.4f}，checkpoint 已落盘（含模型卡）")
                else:
                    no_improve += 1
                    if no_improve >= args.patience:
                        logger.log(f"  早停：连续 {no_improve} 次验证无提升")
                        break

    logger.log(f"训练结束  最优验证Acc={best_val_acc:.4f} @ep{best_episode}  → 07checkpoints/{run_name}/best.pt")

    # ─── 双语损失/准确率曲线 ─────────────────────────────────────────────────
    window = args.val_every
    n_pts = max(len(train_losses) // window, 1)
    sm_ep = [i * window for i in range(1, n_pts + 1)]
    sm_loss = [float(np.mean(train_losses[i * window : (i + 1) * window])) for i in range(n_pts)]
    sm_acc = [float(np.mean(train_accs[i * window : (i + 1) * window])) for i in range(n_pts)]
    txt = {
        "zh": {
            "loss": "损失",
            "acc": "准确率",
            "ep": "Episode",
            "trl": "训练损失",
            "vl": "验证损失",
            "tra": "训练Acc",
            "va": "验证Acc",
            "rand": "随机基准",
            "sup": "ProtoNet 元训练过程",
            "lt": "训练/验证损失",
            "at": "训练/验证准确率",
        },
        "en": {
            "loss": "Loss",
            "acc": "Accuracy",
            "ep": "Episode",
            "trl": "Train loss",
            "vl": "Val loss",
            "tra": "Train acc",
            "va": "Val acc",
            "rand": "Chance",
            "sup": "ProtoNet meta-training",
            "lt": "Train/val loss",
            "at": "Train/val accuracy",
        },
    }

    def plot_curve(lang: str) -> Any:
        t = txt[lang]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(sm_ep, sm_loss, label=t["trl"], color="#0072B2", linewidth=1.5)
        if val_episodes:
            axes[0].plot(
                val_episodes,
                val_losses,
                label=t["vl"],
                color="#D55E00",
                linewidth=1.5,
                linestyle="--",
                marker="o",
                markersize=4,
            )
        axes[0].set_xlabel(t["ep"])
        axes[0].set_ylabel(t["loss"])
        axes[0].set_title(t["lt"])
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(sm_ep, sm_acc, label=t["tra"], color="#009E73", linewidth=1.5)
        if val_episodes:
            axes[1].plot(
                val_episodes,
                val_accs,
                label=t["va"],
                color="#CC79A7",
                linewidth=1.5,
                linestyle="--",
                marker="o",
                markersize=4,
            )
        axes[1].axhline(1 / 3, color="gray", linestyle=":", alpha=0.6, label=t["rand"])
        axes[1].set_xlabel(t["ep"])
        axes[1].set_ylabel(t["acc"])
        axes[1].set_title(t["at"])
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.suptitle(t["sup"], fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    zh, en = eu.save_figure_bilingual(plot_curve, "05_meta_train", 0)
    logger.log(f"曲线 → {zh} / {en}")

    val_df = pd.DataFrame({"episode": val_episodes, "val_loss": val_losses, "val_acc": val_accs})
    budget_df = pd.DataFrame(
        [{"best_val_acc": best_val_acc, "best_episode": best_episode, "params": n_params, **budget}]
    )
    xlsx = eu.write_script_workbook(__file__, {0: ("验证指标", val_df), "compute": budget_df})
    logger.log(f"指标表 → {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
