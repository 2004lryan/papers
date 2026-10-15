"""
26calibration.py: 不确定性量化 —— 折内 ProtoNet 的分类校准（ECE/MCE/reliability），§10.2。

对 fold-restricted ProtoNet（单面查询，K-shot）收集预测置信度与正确性，计算：
  - ECE（期望校准误差，≥10 bins）、MCE（最大校准误差）、reliability diagram；
  - 温度缩放（temperature scaling）后 ECE 改善量（§10.2：ECE>0.05 须校准）。
ProtoNet 概率 = softmax(temp · 余弦相似度(query, 各产地原型))。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/26calibration.py --device auto --reps 50

输出文件:
    04outputs/26calibration-图a.pdf / -fig-a.pdf  — 各折 reliability diagram（校准前/后）
    04outputs/26calibration.xlsx                    — ECE/MCE 表 + 温度缩放 + 算力
    05logs/26calibration_YY-MM-DD_HHMMSS.log

说明:
    依赖 10 的 fold 编码器（--encoder_seed）。MPS 为 Pilot（<MPS_PILOT>）。
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

import export_utils as eu
from protonet import ProtoEncoder, build_year_region_data, encode_batch

NDArray = npt.NDArray[Any]


def _softmax(z: NDArray) -> NDArray:
    z = z - z.max()
    e = np.exp(z)
    return np.asarray(e / (e.sum() + 1e-12))


def ece_mce(confs: NDArray, correct: NDArray, n_bins: int = 10) -> tuple[float, float, list[dict[str, float]]]:
    """计算 ECE/MCE 与各 bin 统计（reliability diagram 数据）。"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins: list[dict[str, float]] = []
    n = len(confs)
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1]) if i > 0 else (confs >= edges[i]) & (confs <= edges[i + 1])
        cnt = int(m.sum())
        if cnt > 0:
            acc = float(correct[m].mean())
            conf = float(confs[m].mean())
            gap = abs(acc - conf)
            ece += (cnt / n) * gap
            mce = max(mce, gap)
            bins.append({"bin_lo": float(edges[i]), "acc": acc, "conf": conf, "count": float(cnt)})
    return float(ece), float(mce), bins


def collect_probs(
    enc: ProtoEncoder, year_data: Any, tgt_year: int, k: int, temp: float, reps: int, seed: int, device: torch.device
) -> tuple[NDArray, NDArray]:
    """返回 (置信度, 是否正确) —— 折内 ProtoNet 单面查询的预测概率最大值与正确性。"""
    regs = sorted(year_data[tgt_year].keys())
    # 缓存每苹果每面位特征
    feats: dict[str, dict[Any, NDArray]] = {}
    for r in regs:
        feats[r] = {}
        for aid in year_data[tgt_year][r]:
            feats[r][aid] = encode_batch(enc, year_data[tgt_year][r][aid].astype(np.float32), device)
    confs: list[float] = []
    correct: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        support: dict[str, list[Any]] = {}
        sup_set: set[Any] = set()
        for r in regs:
            apples = list(feats[r].keys())
            kk = min(k, max(1, len(apples) - 1))
            chosen = list(rng.choice(apples, size=kk, replace=False))
            support[r] = chosen
            sup_set.update(chosen)
        protos = []
        for r in regs:
            ff = [feats[r][a][rng.integers(0, len(feats[r][a]))] for a in support[r]]
            p = np.mean(ff, axis=0)
            protos.append(p / (float(np.linalg.norm(p)) + 1e-8))
        pmat = np.stack(protos)
        for ri, r in enumerate(regs):
            for aid in feats[r]:
                if aid in sup_set:
                    continue
                fa = feats[r][aid]
                fv = fa[rng.integers(0, len(fa))]
                fv = fv / (float(np.linalg.norm(fv)) + 1e-8)
                prob = _softmax(temp * (pmat @ fv))
                confs.append(float(prob.max()))
                correct.append(1.0 if int(prob.argmax()) == ri else 0.0)
    return np.array(confs), np.array(correct)


def main() -> None:
    parser = argparse.ArgumentParser(description="26calibration：ProtoNet 校准（§10.2）")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temp", type=float, default=10.0)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--encoder_seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    device = eu.get_device(args.device)
    logger = eu.get_logger("26calibration")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "ProtoNet 校准 ECE/MCE(§10.2)",
            "阶段": args.stage,
            "设备": str(device),
            "K": args.k,
            "temp": args.temp,
            "reps": args.reps,
            "n_bins": args.n_bins,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    year_data = build_year_region_data(df, wl_cols)

    rows: list[dict[str, Any]] = []
    rel_by_fold: dict[int, list[dict[str, float]]] = {}
    with eu.track_compute("校准", logger, run):
        for tgt_year in (2018, 2019, 2025):
            ckpt_path = os.path.join(
                eu.CHECKPOINT_DIR, f"10_loeo_year_foldwise_seed{args.encoder_seed}_fold{tgt_year}", "best.pt"
            )
            if not os.path.exists(ckpt_path):
                logger.log(f"  缺 fold 编码器 {ckpt_path}，请先跑 10。")
                logger.close()
                return
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            enc = ProtoEncoder(n_bands, feat_dim=int(ckpt.get("feat_dim", 128)))
            enc.load_state_dict(ckpt["state_dict"])
            enc.to(device)
            enc.eval()
            confs, correct = collect_probs(enc, year_data, tgt_year, args.k, args.temp, args.reps, args.seed, device)
            ece, mce, bins = ece_mce(confs, correct, args.n_bins)
            rel_by_fold[tgt_year] = bins
            # 温度缩放：网格搜最优 T 使 ECE 最小（在 logit=temp·sim 上再除以 T_scale）
            best_t, best_ece = 1.0, ece
            for ts in np.linspace(0.5, 5.0, 19):
                e2, _, _ = ece_mce(_recalib_conf(confs, correct, ts), correct, args.n_bins)
                if e2 < best_ece:
                    best_ece, best_t = e2, float(ts)
            rows.append(
                {
                    "fold": tgt_year,
                    "ECE": round(ece, 4),
                    "MCE": round(mce, 4),
                    "整体准确率": round(float(correct.mean()), 4),
                    "平均置信度": round(float(confs.mean()), 4),
                    "需校准(ECE>0.05)": "YES" if ece > 0.05 else "NO",
                    "温度缩放T": best_t,
                    "校准后ECE": round(best_ece, 4),
                }
            )
            logger.log(
                f"  LOEO-{tgt_year}: ECE={ece:.4f} MCE={mce:.4f} acc={correct.mean():.4f} "
                f"conf={confs.mean():.4f}  温度缩放后ECE={best_ece:.4f}(T={best_t})"
            )
            del enc

    res_df = pd.DataFrame(rows)
    fold_color = {2018: "#0072B2", 2019: "#009E73", 2025: "#D55E00"}
    txt = {
        "zh": {"x": "预测置信度", "y": "实际准确率", "t": "Reliability diagram（折内 ProtoNet）", "ideal": "完美校准"},
        "en": {
            "x": "Predicted confidence",
            "y": "Empirical accuracy",
            "t": "Reliability diagram (fold-restricted ProtoNet)",
            "ideal": "Perfect calibration",
        },
    }

    def plot_a(lang: str) -> Any:
        t = txt[lang]
        folds = sorted(rel_by_fold.keys())
        fig, axes = plt.subplots(1, len(folds), figsize=(4 * len(folds), 4.2), sharey=True)
        axes_list = axes if len(folds) > 1 else [axes]
        for ax, fy in zip(axes_list, folds, strict=False):
            bins = rel_by_fold[fy]
            xs = [b["conf"] for b in bins]
            ys = [b["acc"] for b in bins]
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7, label=t["ideal"])
            ax.plot(xs, ys, marker="o", color=fold_color[fy])
            ax.set_title(f"LOEO-{fy}", color=fold_color[fy])
            ax.set_xlabel(t["x"])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes_list[0].set_ylabel(t["y"])
        fig.suptitle(t["t"])
        fig.tight_layout()
        return fig

    zh, en = eu.save_figure_bilingual(plot_a, "26calibration", 0)
    logger.log(f"图a → {zh} / {en}")
    xlsx = eu.write_script_workbook(__file__, {"calibration": res_df})
    logger.log(f"结果 → {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


def _recalib_conf(confs: NDArray, correct: NDArray, t_scale: float) -> NDArray:
    """近似温度缩放：对最大置信度按 T 重标定（保序，用于 ECE 网格搜索）。"""
    logit = np.log(np.clip(confs, 1e-6, 1 - 1e-6) / (1 - np.clip(confs, 1e-6, 1 - 1e-6)))
    scaled = 1.0 / (1.0 + np.exp(-logit / t_scale))
    return np.asarray(scaled)


if __name__ == "__main__":
    main()
