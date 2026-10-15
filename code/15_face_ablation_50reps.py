"""
15_face_ablation_50reps.py: LOEO-2025 面位均值消融（5 配置 × 50-rep + Bootstrap CI + LLH-GBID 三表）。

2×2 析因（支持侧 × 查询侧）面位均值消融 + SVM 对照，仅对 LOEO-2025 折有意义
（2018/2019 每苹果仅 1 面位）。本脚本为「同折同条件 ≥2 配置多指标比较」，
直接产出 LLH-GBID 三表（综合结果表 / 原始指标表 / 敏感性总表）。

5 配置:
    S_multi_Q_multi / S_multi_Q_single / S_single_Q_multi / S_single_Q_single + SVM-Kshot(对照)

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/15_face_ablation_50reps.py --device auto --seeds 50

输出文件:
    04outputs/15_face_ablation_50reps-图a.pdf / -fig-a.pdf  — 5 配置 K-Shot F1 曲线（Bootstrap 95% CI 带）
    04outputs/15_face_ablation_50reps-图b.pdf / -fig-b.pdf  — K=5 森林点图（95% CI + 主效应标注）
    04outputs/15_face_ablation_50reps-图c.pdf / -fig-c.pdf  — LLH-GBID 综合得分点图（含准入状态）
    04outputs/15_face_ablation_50reps.xlsx                   — 消融结果 + 效应分解 + GBID 三表 + 算力
    05logs/15_face_ablation_50reps_YY-MM-DD_HHMMSS.log

说明:
    依赖 10 的 fold-2025 编码器（07checkpoints/10_loeo_year_foldwise_seed{encoder_seed}_fold2025/best.pt）。
    GBID 设定锁定于 01STATISTICAL_PROTOCOL.md；基准/相关性/重采样三项敏感性留待 CUDA 全量复跑补齐。
    MPS 为 Pilot（<MPS_PILOT>）。
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import export_utils as eu
from protonet import ProtoEncoder, apple_feature, build_year_region_data, encode_batch

NDArray = npt.NDArray[Any]

TGT_YEAR = 2025
ALL_REGIONS = ["XJ", "SD", "SX", "GS"]
# (名称, 支持侧全面位均值?, 查询侧全面位均值?, 中文描述)
FACE_CONFIGS: list[tuple[str, bool, bool, str]] = [
    ("S_multi_Q_multi", True, True, "支持全面位+查询全面位（主方法）"),
    ("S_multi_Q_single", True, False, "支持全面位+查询单面位"),
    ("S_single_Q_multi", False, True, "支持单面位+查询全面位"),
    ("S_single_Q_single", False, False, "支持单面位+查询单面位（模拟单面场景）"),
]
CONFIG_NAMES = [c[0] for c in FACE_CONFIGS] + ["SVM-Kshot"]
# LLH-GBID 设定（与 01STATISTICAL_PROTOCOL.md 锁定一致）
GBID_DIMS = [[0, 1], [2], [3]]  # D1 判别力 / D2 最弱类 / D3 稳定性
GBID_W = [1 / 3, 1 / 3, 1 / 3]
GBID_V = [[0.5, 0.5], [1.0], [1.0]]
F1_BASELINE = 1 / 3  # 3 类随机无技能
SD_BASELINE = 0.10
UMIN = 0.30


def snv(x: NDArray) -> NDArray:
    """逐样本标准正态变量变换（SNV，2D：按波段轴）。"""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return np.asarray((x - mean) / std)


def _macro_and_perclass(y_true: list[str], y_pred: list[str], regions: list[str]) -> tuple[float, dict[str, float]]:
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per = f1_score(y_true, y_pred, labels=regions, average=None, zero_division=0)
    return macro, {r: float(v) for r, v in zip(regions, per, strict=False)}


def _feat_single(encoder: ProtoEncoder, faces_x: NDArray, device: torch.device, rng: np.random.Generator) -> NDArray:
    """随机单面位特征（L2 归一化）。"""
    idx = int(rng.integers(0, len(faces_x)))
    f = encode_batch(encoder, faces_x[idx : idx + 1], device)[0]
    return np.asarray(f / (float(np.linalg.norm(f)) + 1e-8))


def eval_protonet_facecfg(
    encoder: ProtoEncoder,
    year_data: Any,
    tgt_year: int,
    k_shot: int,
    seed: int,
    sup_multi: bool,
    qry_multi: bool,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """单折 ProtoNet 面位组合评估（支持/查询侧各可选全面位均值或随机单面位）。"""
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
        feats = [
            apple_feature(encoder, year_data[tgt_year][reg][aid], device)
            if sup_multi
            else _feat_single(encoder, year_data[tgt_year][reg][aid], device, rng)
            for aid in support_ids[reg]
        ]
        p = np.mean(feats, axis=0)
        tgt_protos[reg] = p / (float(np.linalg.norm(p)) + 1e-8)
    proto_mat = np.stack([tgt_protos[r] for r in tgt_regions])

    y_true: list[str] = []
    y_pred: list[str] = []
    for aid, true_reg in query_items:
        faces = year_data[tgt_year][true_reg][aid]
        feat = apple_feature(encoder, faces, device) if qry_multi else _feat_single(encoder, faces, device, rng)
        y_true.append(true_reg)
        y_pred.append(tgt_regions[int((proto_mat @ feat).argmax())])
    return _macro_and_perclass(y_true, y_pred, tgt_regions)


def eval_svm(year_data: Any, tgt_year: int, k_shot: int, seed: int) -> tuple[float, dict[str, float]]:
    """SVM 对照：全面位均值光谱 + SNV + StandardScaler + LinearSVC(C=0.1)。"""
    rng = np.random.default_rng(seed)
    tgt_regions = sorted(year_data[tgt_year].keys())
    reg_to_idx = {r: i for i, r in enumerate(tgt_regions)}
    support_ids: dict[str, list[Any]] = {}
    for reg in tgt_regions:
        apples = list(year_data[tgt_year][reg].keys())
        k = min(k_shot, max(1, len(apples) - 1))
        support_ids[reg] = list(rng.choice(apples, size=k, replace=False))
    sup_set = {aid for aids in support_ids.values() for aid in aids}
    query_items = [(aid, reg) for reg in tgt_regions for aid in year_data[tgt_year][reg] if aid not in sup_set]
    if not query_items:
        query_items = [(aid, reg) for reg in tgt_regions for aid in year_data[tgt_year][reg]]

    x_tr: list[NDArray] = []
    y_tr: list[int] = []
    for reg in tgt_regions:
        for aid in support_ids[reg]:
            x_tr.append(year_data[tgt_year][reg][aid].mean(axis=0))
            y_tr.append(reg_to_idx[reg])
    x_train = snv(np.stack(x_tr))
    y_train = np.array(y_tr)
    if len(np.unique(y_train)) < 2:
        return 0.0, dict.fromkeys(tgt_regions, 0.0)
    try:
        clf = Pipeline([("sc", StandardScaler()), ("svm", LinearSVC(C=0.1, max_iter=2000, dual="auto"))])
        clf.fit(x_train, y_train)
        y_true = [true_reg for _, true_reg in query_items]
        y_pred = [
            tgt_regions[int(clf.predict(snv(year_data[tgt_year][true_reg][aid].mean(axis=0).reshape(1, -1)))[0])]
            for aid, true_reg in query_items
        ]
    except Exception:
        return 0.0, dict.fromkeys(tgt_regions, 0.0)
    return _macro_and_perclass(y_true, y_pred, tgt_regions)


def bootstrap_ci(
    samples: NDArray, n_boot: int = 1000, alpha: float = 0.05, rng_seed: int = 0
) -> tuple[float, float, float]:
    """i.i.d. 重采样 95% CI（描述性 episode 抽样方差，非方法比较推断）。返回 (均值, 下界, 上界)。"""
    arr = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(rng_seed)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    means = boot.mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def load_fold2025_encoder(encoder_seed: int, n_bands: int, device: torch.device) -> ProtoEncoder:
    """加载 10 产出的 fold-2025 编码器（新格式 state_dict/feat_dim）；缺失则抛错。"""
    run_name = f"10_loeo_year_foldwise_seed{encoder_seed}_fold{TGT_YEAR}"
    ckpt_path = os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"缺少 fold-2025 编码器 {ckpt_path}，请先运行 10_loeo_year_foldwise.py。")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = ProtoEncoder(n_bands, feat_dim=int(ckpt.get("feat_dim", 128)))
    enc.load_state_dict(ckpt["state_dict"])
    enc.to(device)
    enc.eval()
    return enc


def compute_gbid_tables(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 §15.8 产出综合结果表 / 原始指标表 / 敏感性总表（实体=面位配置）。"""
    pc_cols = [f"F1类_{r}" for r in ALL_REGIONS if f"F1类_{r}" in panel_df.columns]
    metrics: dict[str, dict[str, float]] = {}
    for m in CONFIG_NAMES:
        sub = panel_df[panel_df["配置"] == m]
        k5 = sub[sub["K"] == 5]
        k20 = sub[sub["K"] == 20]
        f1_k5 = float(k5["F1均值"].mean()) if not k5.empty else 0.0
        f1_k20 = float(k20["F1均值"].mean()) if not k20.empty else f1_k5
        sd_k5 = float(k5["F1标准差"].mean()) if not k5.empty else 0.0
        pc_vals = k5[pc_cols].to_numpy() if pc_cols and not k5.empty else np.array([[f1_k5]])
        minpc = float(np.nanmin(pc_vals)) if np.isfinite(pc_vals).any() else f1_k5
        metrics[m] = {"F1@K5": f1_k5, "F1@K20": f1_k20, "minPC@K5": minpc, "SD@K5": sd_k5}

    u_by_model: dict[str, list[float]] = {}
    gbid_res: dict[str, dict[str, Any]] = {}
    passed: dict[str, bool] = {}
    for m in CONFIG_NAMES:
        mt = metrics[m]
        u = [
            eu.compute_utility(mt["F1@K5"], 1.0, F1_BASELINE, "benefit"),
            eu.compute_utility(mt["F1@K20"], 1.0, F1_BASELINE, "benefit"),
            eu.compute_utility(mt["minPC@K5"], 1.0, F1_BASELINE, "benefit"),
            eu.compute_utility(mt["SD@K5"], 0.0, SD_BASELINE, "cost"),
        ]
        u_by_model[m] = u
        gbid_res[m] = eu.compute_gbid(u, GBID_W, GBID_V, GBID_DIMS)
        passed[m] = u[0] >= UMIN and u[2] >= UMIN  # J_crit = {F1@K5, minPC@K5}

    order = sorted(CONFIG_NAMES, key=lambda m: (passed[m], float(gbid_res[m]["gbids"])), reverse=True)
    rank = {m: i + 1 for i, m in enumerate(order)}

    comp_rows = [
        {
            "配置": m,
            "Pass": passed[m],
            "ū": round(float(gbid_res[m]["u_bar"]), 4),
            "C_B": round(float(gbid_res[m]["C_B"]), 4),
            "C_W": round(float(gbid_res[m]["C_W"]), 4),
            "GBID": round(float(gbid_res[m]["gbid"]), 4),
            "GBIDS": round(float(gbid_res[m]["gbids"]), 2),
            "主排名": rank[m],
        }
        for m in CONFIG_NAMES
    ]
    raw_rows = [
        {
            "配置": m,
            "F1@K5": round(metrics[m]["F1@K5"], 4),
            "F1@K20": round(metrics[m]["F1@K20"], 4),
            "最弱类F1@K5": round(metrics[m]["minPC@K5"], 4),
            "F1_SD@K5": round(metrics[m]["SD@K5"], 4),
            "GBIDS": round(float(gbid_res[m]["gbids"]), 2),
        }
        for m in CONFIG_NAMES
    ]

    # q 敏感性：四配置下的平均名次
    q_gbids = {m: eu.gbid_sensitivity_q(u_by_model[m], GBID_W, GBID_V, GBID_DIMS) for m in CONFIG_NAMES}
    configs = list(q_gbids[CONFIG_NAMES[0]].keys())
    q_rank: dict[str, list[int]] = {m: [] for m in CONFIG_NAMES}
    for cfg in configs:
        cfg_order = sorted(CONFIG_NAMES, key=lambda mm: q_gbids[mm][cfg], reverse=True)
        for i, mm in enumerate(cfg_order):
            q_rank[mm].append(i + 1)
    # 权重敏感性：Dirichlet 1000 组
    w_sens = eu.gbid_sensitivity_weights(u_by_model, GBID_W, GBID_V, GBID_DIMS, n_draws=1000)
    sens_rows = [
        {
            "配置": m,
            "主排名": rank[m],
            "q敏感性均排名": round(float(np.mean(q_rank[m])), 2),
            "权重扰动第一名概率": round(w_sens[m]["first_place_prob"], 3),
            "权重扰动平均名次": round(w_sens[m]["mean_rank"], 2),
            "基准敏感性": "待CUDA全量",
            "相关性处理一致性": "待CUDA全量",
            "重采样第一名频率": "待CUDA全量",
        }
        for m in CONFIG_NAMES
    ]
    return pd.DataFrame(comp_rows), pd.DataFrame(raw_rows), pd.DataFrame(sens_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="15_face_ablation_50reps：面位均值消融 + LLH-GBID")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k_list", default="1,3,5,10,20,50")
    parser.add_argument("--seeds", type=int, default=50, help="评估重复次数（episode 重采样）")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--encoder_seed", type=int, default=eu.PILOT_SEED, help="定位 10 的 fold-2025 编码器种子")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_list.split(",")]
    rep_seeds = [args.seed + i for i in range(args.seeds)]
    device = eu.get_device(args.device)
    logger = eu.get_logger("15_face_ablation_50reps")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "LOEO-2025 面位消融 5 配置 + LLH-GBID",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "reps": args.seeds,
            "K_list": k_values,
            "n_bootstrap": args.n_bootstrap,
            "encoder_seed": args.encoder_seed,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    year_data = build_year_region_data(df, wl_cols)
    regions = sorted(year_data[TGT_YEAR].keys())
    logger.log(f"2025 目标产地={regions}  波段数={n_bands}")
    encoder = load_fold2025_encoder(args.encoder_seed, n_bands, device)
    logger.log(f"fold-2025 编码器加载完成 → 07checkpoints/10_loeo_year_foldwise_seed{args.encoder_seed}_fold{TGT_YEAR}")

    raw_f1s: dict[str, dict[int, list[float]]] = {c: {k: [] for k in k_values} for c in CONFIG_NAMES}
    raw_pc: dict[str, dict[int, dict[str, list[float]]]] = {
        c: {k: {r: [] for r in regions} for k in k_values} for c in CONFIG_NAMES
    }

    with eu.track_compute("面位消融", logger, run) as budget:
        for cfg_name, sup_multi, qry_multi, desc in FACE_CONFIGS:
            logger.log(f"\n【{cfg_name}】{desc}")
            for k in k_values:
                for s in rep_seeds:
                    macro, per = eval_protonet_facecfg(encoder, year_data, TGT_YEAR, k, s, sup_multi, qry_multi, device)
                    raw_f1s[cfg_name][k].append(macro)
                    for r in regions:
                        raw_pc[cfg_name][k][r].append(per.get(r, float("nan")))
                std_v = float(np.std(raw_f1s[cfg_name][k]))
                mean_f1, lo, hi = bootstrap_ci(np.asarray(raw_f1s[cfg_name][k], dtype=float), args.n_bootstrap)
                logger.log(f"  K={k:3d}  F1={mean_f1:.4f} 95%CI=[{lo:.4f},{hi:.4f}]  std={std_v:.4f}")
        logger.log("\n【SVM-Kshot】全面位均值光谱 + LinearSVC(C=0.1)")
        for k in k_values:
            for s in rep_seeds:
                macro, per = eval_svm(year_data, TGT_YEAR, k, s)
                raw_f1s["SVM-Kshot"][k].append(macro)
                for r in regions:
                    raw_pc["SVM-Kshot"][k][r].append(per.get(r, float("nan")))
            std_v = float(np.std(raw_f1s["SVM-Kshot"][k]))
            mean_f1, lo, hi = bootstrap_ci(np.asarray(raw_f1s["SVM-Kshot"][k], dtype=float), args.n_bootstrap)
            logger.log(f"  K={k:3d}  F1={mean_f1:.4f} 95%CI=[{lo:.4f},{hi:.4f}]  std={std_v:.4f}")

    # ─── 消融结果表 + GBID 输入面板 ───────────────────────────────────────────
    abl_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    for cfg in CONFIG_NAMES:
        for k in k_values:
            f1s = raw_f1s[cfg][k]
            mean_f1, lo, hi = bootstrap_ci(np.asarray(f1s, dtype=float), args.n_bootstrap)
            sd = float(np.std(f1s))
            abl_rows.append(
                {
                    "配置": cfg,
                    "K": k,
                    "F1均值": round(mean_f1, 4),
                    "F1标准差": round(sd, 4),
                    "CI下界95%": round(lo, 4),
                    "CI上界95%": round(hi, 4),
                    "重复次数": len(f1s),
                }
            )
            prow: dict[str, Any] = {"配置": cfg, "K": k, "F1均值": round(mean_f1, 4), "F1标准差": round(sd, 4)}
            for r in ALL_REGIONS:
                prow[f"F1类_{r}"] = round(float(np.nanmean(raw_pc[cfg][k][r])), 4) if r in regions else float("nan")
            panel_rows.append(prow)
    abl_df = pd.DataFrame(abl_rows)
    panel_df = pd.DataFrame(panel_rows)

    # ─── K=5 析因主效应分解 ──────────────────────────────────────────────────
    has_k5 = 5 in k_values
    eff_df = pd.DataFrame()
    qry_effect = sup_effect = interaction = mm_k5 = float("nan")
    qlo = qhi = slo = shi = float("nan")
    if has_k5:
        ss = float(np.mean(raw_f1s["S_single_Q_single"][5]))
        sm = float(np.mean(raw_f1s["S_single_Q_multi"][5]))
        ms = float(np.mean(raw_f1s["S_multi_Q_single"][5]))
        mm_k5 = float(np.mean(raw_f1s["S_multi_Q_multi"][5]))
        qry_effect = (sm - ss + mm_k5 - ms) / 2
        sup_effect = (ms - ss + mm_k5 - sm) / 2
        interaction = mm_k5 - ms - sm + ss
        qry_boot = (
            np.asarray(raw_f1s["S_single_Q_multi"][5], dtype=float)
            - np.asarray(raw_f1s["S_single_Q_single"][5], dtype=float)
            + np.asarray(raw_f1s["S_multi_Q_multi"][5], dtype=float)
            - np.asarray(raw_f1s["S_multi_Q_single"][5], dtype=float)
        ) / 2
        sup_boot = (
            np.asarray(raw_f1s["S_multi_Q_single"][5], dtype=float)
            - np.asarray(raw_f1s["S_single_Q_single"][5], dtype=float)
            + np.asarray(raw_f1s["S_multi_Q_multi"][5], dtype=float)
            - np.asarray(raw_f1s["S_single_Q_multi"][5], dtype=float)
        ) / 2
        _, qlo, qhi = bootstrap_ci(qry_boot, args.n_bootstrap)
        _, slo, shi = bootstrap_ci(sup_boot, args.n_bootstrap)
        logger.log("\nK=5 析因主效应分解（2×2）：")
        logger.log(f"  基线 S_single_Q_single = {ss:.4f}")
        logger.log(f"  查询侧主效应 = {qry_effect:+.4f}  95%CI=[{qlo:+.4f},{qhi:+.4f}]")
        logger.log(f"  支持侧主效应 = {sup_effect:+.4f}  95%CI=[{slo:+.4f},{shi:+.4f}]")
        logger.log(f"  交互效应     = {interaction:+.4f}")
        if slo < 0 < shi:
            logger.log("  → 支持侧增益 CI 跨零，效应功效不足")
        if qlo > 0:
            logger.log("  → 查询侧增益 CI 正值，效应稳健")
        eff_df = pd.DataFrame(
            {
                "效应": ["基线(S_single_Q_single)", "查询侧主效应", "支持侧主效应", "交互效应"],
                "值": [round(ss, 4), round(qry_effect, 4), round(sup_effect, 4), round(interaction, 4)],
                "CI下界95%": [float("nan"), round(qlo, 4), round(slo, 4), float("nan")],
                "CI上界95%": [float("nan"), round(qhi, 4), round(shi, 4), float("nan")],
            }
        )

    # ─── LLH-GBID 三表 ──────────────────────────────────────────────────────
    comp_df, raw_df, sens_df = compute_gbid_tables(panel_df)
    logger.log("\n── LLH-GBID 综合结果表（§15.8）──")
    logger.log(comp_df.to_string(index=False))
    best = comp_df.sort_values(["Pass", "GBIDS"], ascending=[False, False]).iloc[0]
    logger.log(
        eu.gbid_paper_paragraph(
            str(best["配置"]),
            {"gbids": best["GBIDS"], "u_bar": best["ū"], "C_B": best["C_B"], "C_W": best["C_W"]},
            bool(best["Pass"]),
        )
    )

    # ─── 双语图 ──────────────────────────────────────────────────────────────
    cfg_color = {
        "S_multi_Q_multi": "#0072B2",
        "S_multi_Q_single": "#E69F00",
        "S_single_Q_multi": "#56B4E9",
        "S_single_Q_single": "#009E73",
        "SVM-Kshot": "#D55E00",
    }
    cfg_style = {
        "S_multi_Q_multi": "-",
        "S_multi_Q_single": "--",
        "S_single_Q_multi": "-.",
        "S_single_Q_single": ":",
        "SVM-Kshot": "-",
    }
    cfg_label = {
        "zh": {
            "S_multi_Q_multi": "支持全+查询全",
            "S_multi_Q_single": "支持全+查询单",
            "S_single_Q_multi": "支持单+查询全",
            "S_single_Q_single": "支持单+查询单",
            "SVM-Kshot": "SVM（全面位均值，对照）",
        },
        "en": {
            "S_multi_Q_multi": "Sup multi / Qry multi",
            "S_multi_Q_single": "Sup multi / Qry single",
            "S_single_Q_multi": "Sup single / Qry multi",
            "S_single_Q_single": "Sup single / Qry single",
            "SVM-Kshot": "SVM (full-face mean, control)",
        },
    }
    txt = {
        "zh": {
            "x": "K-Shot（每产地支持苹果数）",
            "y": f"Macro-F1（LOEO-2025，{args.seeds}次重复）",
            "rand": "随机基线(1/3)",
            "ta": "面位均值消融：K-Shot 曲线（阴影=Bootstrap 95% CI）",
            "x5": "Macro-F1（K=5，LOEO-2025）",
            "tb": "面位消融 K=5 森林图（误差棒=Bootstrap 95% CI）",
            "qeff": "查询侧主效应",
            "seff": "支持侧主效应",
            "gx": "GBIDS（0—100 综合指数）",
            "tc": "LLH-GBID 综合得分（点=配置，色=准入）",
            "pass": "通过门槛",
            "fail": "未通过门槛",
        },
        "en": {
            "x": "K-shot (support apples per origin)",
            "y": f"Macro-F1 (LOEO-2025, {args.seeds} reps)",
            "rand": "Chance (1/3)",
            "ta": "Face-mean ablation: K-shot curves (band = bootstrap 95% CI)",
            "x5": "Macro-F1 (K=5, LOEO-2025)",
            "tb": "Face ablation K=5 forest plot (whisker = bootstrap 95% CI)",
            "qeff": "Query main effect",
            "seff": "Support main effect",
            "gx": "GBIDS (0-100 composite)",
            "tc": "LLH-GBID composite score (dot=config, colour=gate)",
            "pass": "Pass",
            "fail": "Fail",
        },
    }

    def plot_a(lang: str) -> Any:
        t = txt[lang]
        fig, ax = plt.subplots(figsize=(9, 5))
        for cfg in CONFIG_NAMES:
            sub = abl_df[abl_df["配置"] == cfg].sort_values("K")
            ax.plot(
                sub["K"],
                sub["F1均值"],
                marker="o",
                linewidth=2,
                color=cfg_color[cfg],
                linestyle=cfg_style[cfg],
                label=cfg_label[lang][cfg],
            )
            ax.fill_between(sub["K"], sub["CI下界95%"], sub["CI上界95%"], alpha=0.12, color=cfg_color[cfg])
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel(t["x"])
        ax.set_ylabel(t["y"])
        ax.set_title(t["ta"])
        ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.4, label=t["rand"])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_b(lang: str) -> Any:
        t = txt[lang]
        fig, ax = plt.subplots(figsize=(8, 5))
        k5map = {str(r["配置"]): r for _, r in abl_df[abl_df["K"] == 5].iterrows()}
        ypos = np.arange(len(CONFIG_NAMES))
        for yi, cfg in enumerate(CONFIG_NAMES):
            r = k5map[cfg]
            v, low, high = float(r["F1均值"]), float(r["CI下界95%"]), float(r["CI上界95%"])
            ax.errorbar(v, yi, xerr=[[v - low], [high - v]], fmt="o", color=cfg_color[cfg], capsize=4, markersize=9)
            ax.text(v, yi + 0.18, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([cfg_label[lang][c] for c in CONFIG_NAMES])
        ax.axvline(1 / 3, color="gray", linestyle=":", alpha=0.5, label=t["rand"])
        ax.set_xlabel(t["x5"])
        ax.set_title(t["tb"])
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        if has_k5:
            eff_text = (
                f"{t['qeff']} {qry_effect:+.3f} [{qlo:+.3f}, {qhi:+.3f}]\n"
                f"{t['seff']} {sup_effect:+.3f} [{slo:+.3f}, {shi:+.3f}]"
            )
            ax.text(
                0.02,
                0.98,
                eff_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
            )
        fig.tight_layout()
        return fig

    def plot_c(lang: str) -> Any:
        t = txt[lang]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ordered = comp_df.sort_values("GBIDS")
        ypos = np.arange(len(ordered))
        colors = ["#0072B2" if p else "#D55E00" for p in ordered["Pass"]]
        ax.scatter(ordered["GBIDS"], ypos, c=colors, s=90, zorder=3)
        for yi, (_, r) in zip(ypos, ordered.iterrows(), strict=False):
            ax.plot([0, r["GBIDS"]], [yi, yi], color="gray", alpha=0.3, zorder=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(ordered["配置"].tolist())
        ax.set_xlabel(t["gx"])
        ax.set_title(t["tc"])
        ax.grid(axis="x", alpha=0.3)
        ax.scatter([], [], c="#0072B2", label=t["pass"])
        ax.scatter([], [], c="#D55E00", label=t["fail"])
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        return fig

    for idx, fn in enumerate((plot_a, plot_b, plot_c)):
        zh, en = eu.save_figure_bilingual(fn, "15_face_ablation_50reps", idx)
        logger.log(f"图{chr(97 + idx)} → {zh} / {en}")

    budget_df = pd.DataFrame(
        [{"reps": args.seeds, "n_bootstrap": args.n_bootstrap, "encoder_seed": args.encoder_seed, **budget}]
    )
    sheets: dict[object, object] = {
        0: ("面位消融结果", abl_df),
        1: ("GBID综合结果表", comp_df),
        2: ("GBID原始指标表", raw_df),
        3: ("GBID敏感性总表", sens_df),
    }
    if not eff_df.empty:
        sheets["效应分解K5"] = eff_df
    sheets["compute"] = budget_df
    xlsx = eu.write_script_workbook(__file__, sheets)
    logger.log(f"结果（消融+GBID三表）→ {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
