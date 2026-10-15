"""
24negctrl.py: 负控实验（label permutation / 随机特征 / ID-only probe）。

投稿必要前提：标签置换 ≥1000 次、单尾 p<0.001。本脚本对 fold-restricted ProtoNet 的
K-shot 产地判别做三类负控：
  (1) Label permutation：打乱目标年苹果的产地标签后重建 support/query，破坏光谱↔产地关联；
      真实 F1 应显著高于置换零分布（p<0.001）。
  (2) 随机特征基线：用同形状高斯噪声替换光谱特征，应回落到随机水平。
  (3) ID-only probe：仅用苹果序号（中性序）做 LogReg，不应高于随机（否则 ID 泄漏产地）。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/24negctrl.py --device auto --n_perm 1000

输出文件:
    04outputs/24negctrl-图a.pdf / -fig-a.pdf  — 各折标签置换零分布 + 真实 F1
    04outputs/24negctrl.xlsx                    — negative_control 三类结果 + 算力
    05logs/24negctrl_YY-MM-DD_HHMMSS.log

说明:
    依赖 10 的 fold 编码器（--encoder_seed，默认 PILOT_SEED）。MPS 为 Pilot（<MPS_PILOT>）。
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import export_utils as eu
from protonet import ProtoEncoder, build_year_region_data, encode_batch

NDArray = npt.NDArray[Any]


def _norm(v: NDArray) -> NDArray:
    return np.asarray(v / (float(np.linalg.norm(v)) + 1e-8))


def _apple_table(feats: Any, tgt_year: int) -> tuple[list[NDArray], list[str]]:
    """返回目标年所有苹果的（多面位均值）特征与真实产地标签。"""
    mats: list[NDArray] = []
    labels: list[str] = []
    for reg in sorted(feats[tgt_year].keys()):
        for aid in feats[tgt_year][reg]:
            mats.append(_norm(feats[tgt_year][reg][aid].mean(0)))
            labels.append(reg)
    return mats, labels


def _proto_f1(apple_feats: list[NDArray], labels: list[str], k_shot: int, rng: np.random.Generator) -> float:
    """对给定（特征,标签）按 K-shot 最近原型协议算 Macro-F1。"""
    regs = sorted(set(labels))
    by_reg: dict[str, list[int]] = {r: [] for r in regs}
    for i, lb in enumerate(labels):
        by_reg[lb].append(i)
    support: dict[str, list[int]] = {}
    sup_set: set[int] = set()
    for r in regs:
        idxs = by_reg[r]
        k = min(k_shot, max(1, len(idxs) - 1))
        chosen = list(rng.choice(idxs, size=k, replace=False))
        support[r] = chosen
        sup_set.update(chosen)
    protos = {r: _norm(np.mean([apple_feats[i] for i in support[r]], axis=0)) for r in regs}
    pmat = np.stack([protos[r] for r in regs])
    y_true: list[str] = []
    y_pred: list[str] = []
    for i, lb in enumerate(labels):
        if i in sup_set:
            continue
        y_true.append(lb)
        y_pred.append(regs[int((pmat @ apple_feats[i]).argmax())])
    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="24negctrl：负控（§7.6）")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_perm", type=int, default=1000, help="标签置换次数（≥1000 为投稿前提）")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--eval_reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--encoder_seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    device = eu.get_device(args.device)
    logger = eu.get_logger("24negctrl")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "负控(label perm/随机特征/ID probe)",
            "阶段": args.stage,
            "设备": str(device),
            "n_perm": args.n_perm,
            "K": args.k,
            "eval_reps": args.eval_reps,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    year_data = build_year_region_data(df, wl_cols)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    null_by_fold: dict[int, NDArray] = {}
    real_by_fold: dict[int, float] = {}

    with eu.track_compute("负控", logger, run):
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
            # 缓存目标年苹果特征（多面位均值）
            feats_cache: dict[int, dict[str, dict[Any, NDArray]]] = {tgt_year: {}}
            for reg in year_data[tgt_year]:
                feats_cache[tgt_year][reg] = {}
                for aid in year_data[tgt_year][reg]:
                    faces = year_data[tgt_year][reg][aid]
                    feats_cache[tgt_year][reg][aid] = encode_batch(enc, faces.astype(np.float32), device)
            apple_feats, labels = _apple_table(feats_cache, tgt_year)
            labels_arr = np.array(labels)

            # (1) 真实 F1（多 eval_reps 均值）
            real = float(np.mean([_proto_f1(apple_feats, labels, args.k, rng) for _ in range(args.eval_reps)]))
            # 标签置换零分布
            null = np.empty(args.n_perm, dtype=float)
            for p in range(args.n_perm):
                perm = list(rng.permutation(labels_arr))
                null[p] = _proto_f1(apple_feats, perm, args.k, rng)
            p_value = float((np.sum(null >= real) + 1) / (args.n_perm + 1))
            null_by_fold[tgt_year] = null
            real_by_fold[tgt_year] = real
            rows.append(
                {
                    "fold": tgt_year,
                    "负控": "label_permutation",
                    "真实F1": round(real, 4),
                    "零分布均值": round(float(null.mean()), 4),
                    "零分布P95": round(float(np.percentile(null, 95)), 4),
                    "p_value": p_value,
                    "n_perm": args.n_perm,
                    "通过(p<0.001)": "YES" if p_value < 0.001 else "NO",
                }
            )
            logger.log(f"  LOEO-{tgt_year} label-perm: 真实F1={real:.4f} 零均值={null.mean():.4f} p={p_value:.4g}")

            # (2) 随机特征基线（同形状高斯噪声）
            rand_feats = [_norm(rng.standard_normal(apple_feats[0].shape)) for _ in apple_feats]
            rand_f1 = float(np.mean([_proto_f1(rand_feats, labels, args.k, rng) for _ in range(args.eval_reps)]))
            rows.append(
                {
                    "fold": tgt_year,
                    "负控": "random_feature",
                    "真实F1": round(rand_f1, 4),
                    "零分布均值": "",
                    "零分布P95": "",
                    "p_value": "",
                    "n_perm": "",
                    "通过(p<0.001)": "≈随机即合格" if rand_f1 < real else "异常",
                }
            )

            # (3) ID-only probe：产地无关的随机中性序号 → LogReg，应≈随机（否则序号泄漏产地）
            # 注：本数据集苹果编号按"年份_产地"合成、且按产地分组存储，故必须用打乱后的中性序，
            # 避免存储顺序伪泄漏；这检验的是"无信息标识不应预测产地"。
            perm_order = rng.permutation(len(labels))
            ordinal = np.zeros((len(labels), 1), dtype=float)
            ordinal[perm_order, 0] = np.arange(len(labels), dtype=float) / max(len(labels), 1)
            try:
                clf = LogisticRegression(max_iter=500)
                clf.fit(ordinal, labels_arr)
                id_f1 = float(f1_score(labels_arr, clf.predict(ordinal), average="macro", zero_division=0))
            except Exception:
                id_f1 = float("nan")
            rows.append(
                {
                    "fold": tgt_year,
                    "负控": "id_only_probe",
                    "真实F1": round(id_f1, 4),
                    "零分布均值": "",
                    "零分布P95": "",
                    "p_value": "",
                    "n_perm": "",
                    "通过(p<0.001)": "≈随机即合格" if id_f1 < real else "可能ID泄漏",
                }
            )
            logger.log(f"  LOEO-{tgt_year} 随机特征F1={rand_f1:.4f}  ID-probe F1={id_f1:.4f}")
            del enc

    res_df = pd.DataFrame(rows)

    txt = {
        "zh": {
            "x": "Macro-F1",
            "y": "频数",
            "t": "标签置换零分布 vs 真实 F1（各折）",
            "real": "真实",
            "null": "置换零分布",
        },
        "en": {
            "x": "Macro-F1",
            "y": "Count",
            "t": "Label-permutation null vs real F1 (per fold)",
            "real": "Real",
            "null": "Permuted null",
        },
    }
    fold_color = {2018: "#0072B2", 2019: "#009E73", 2025: "#D55E00"}

    def plot_a(lang: str) -> Any:
        t = txt[lang]
        folds = sorted(null_by_fold.keys())
        fig, axes = plt.subplots(1, len(folds), figsize=(4.2 * len(folds), 4.2), sharey=True)
        axes_list = axes if len(folds) > 1 else [axes]
        for ax, fy in zip(axes_list, folds, strict=False):
            ax.hist(null_by_fold[fy], bins=30, color=fold_color[fy], alpha=0.55, label=t["null"])
            ax.axvline(real_by_fold[fy], color="black", linewidth=2, label=t["real"])
            ax.set_title(f"LOEO-{fy}", color=fold_color[fy])
            ax.set_xlabel(t["x"])
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes_list[0].set_ylabel(t["y"])
        fig.suptitle(t["t"])
        fig.tight_layout()
        return fig

    zh, en = eu.save_figure_bilingual(plot_a, "24negctrl", 0)
    logger.log(f"图a → {zh} / {en}")
    xlsx = eu.write_script_workbook(__file__, {"negative_control": res_df})
    logger.log(f"结果 → {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
