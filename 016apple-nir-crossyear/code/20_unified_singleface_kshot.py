"""
20_unified_singleface_kshot.py: 统一单面查询协议下 5 模型 K-Shot 头牌面板 + LLH-GBID 综合评价。

所有方法、所有折一律「单面查询」（2018/2019 自然单面，2025 随机抽 1 面位），消除协议效应；
多面位均值见独立消融。本脚本为论文正文头牌面板，直接产出 LLH-GBID 三表。

5 种方法（折内编码器来自 10_loeo_year_foldwise，07checkpoints）：
    NearestCentroid / PLS-DA-Kshot / SVM-Kshot(RBF) / VanillaProtoNet(α=0) / ProtoNet-C(α=0.4)

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/20_unified_singleface_kshot.py --device auto --seeds 50

输出文件:
    04outputs/20_unified_singleface_kshot-图a.pdf / -fig-a.pdf  — 5 方法 K-Shot F1 曲线
    04outputs/20_unified_singleface_kshot-图b.pdf / -fig-b.pdf  — GBIDS 综合得分点图（含准入状态）
    04outputs/20_unified_singleface_kshot.xlsx                   — 面板结果 + GBID 三表 + 算力
    05logs/20_unified_singleface_kshot_YY-MM-DD_HHMMSS.log

说明:
    依赖 10 的 fold 编码器（--encoder_seed 指定，默认 PILOT_SEED）。MPS 为 Pilot（<MPS_PILOT>）。
    GBID 设定锁定于 01STATISTICAL_PROTOCOL.md；基准/相关性/重采样三项敏感性留待 CUDA 全量复跑补齐。
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import export_utils as eu
from protonet import ProtoEncoder, build_year_region_data, encode_batch

NDArray = npt.NDArray[Any]

METHOD_NAMES = ["NearestCentroid", "PLS-DA-Kshot", "SVM-Kshot", "VanillaProtoNet", "ProtoNet-C(a=0.4)"]
FOLD_CFG: dict[int, dict[str, Any]] = {
    2018: {"src_years": [2019, 2025]},
    2019: {"src_years": [2018, 2025]},
    2025: {"src_years": [2018, 2019]},
}
ALL_REGIONS = ["XJ", "SD", "SX", "GS"]
# LLH-GBID 设定（与 01STATISTICAL_PROTOCOL.md 锁定一致）
GBID_DIMS = [[0, 1], [2], [3]]  # D1 判别力 / D2 最弱类 / D3 稳定性
GBID_W = [1 / 3, 1 / 3, 1 / 3]
GBID_V = [[0.5, 0.5], [1.0], [1.0]]
F1_BASELINE = 1 / 3  # 3 类随机无技能
SD_BASELINE = 0.10
UMIN = 0.30


def snv(x: NDArray) -> NDArray:
    """逐样本标准正态变量变换（SNV）。"""
    if x.ndim == 1:
        return np.asarray((x - x.mean()) / (x.std() + 1e-8))
    return np.asarray((x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8))


def _macro_and_perclass(y_true: list[str], y_pred: list[str], regions: list[str]) -> tuple[float, dict[str, float]]:
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per = f1_score(y_true, y_pred, labels=regions, average=None, zero_division=0)
    return macro, {r: float(v) for r, v in zip(regions, per, strict=False)}


def build_support_query(
    year_data: Any, tgt_year: int, k_shot: int, rng: np.random.Generator
) -> tuple[list[str], dict[str, list[Any]], list[tuple[Any, str]]]:
    regions = sorted(year_data[tgt_year].keys())
    support: dict[str, list[Any]] = {}
    for reg in regions:
        apples = list(year_data[tgt_year][reg].keys())
        k = min(k_shot, max(1, len(apples) - 1))
        support[reg] = list(rng.choice(apples, size=k, replace=False))
    sup_set = {aid for aids in support.values() for aid in aids}
    query = [(aid, reg) for reg in regions for aid in year_data[tgt_year][reg] if aid not in sup_set]
    if not query:
        query = [(aid, reg) for reg in regions for aid in year_data[tgt_year][reg]]
    return regions, support, query


def _one_face(arr: NDArray, rng: np.random.Generator) -> NDArray:
    row: NDArray = arr[rng.integers(0, len(arr))]
    return row


def eval_nearest_centroid(year_data: Any, tgt_year: int, k_shot: int, seed: int) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(seed)
    regions, support, query = build_support_query(year_data, tgt_year, k_shot, rng)
    centroids = []
    for reg in regions:
        specs = [snv(_one_face(year_data[tgt_year][reg][aid], rng)) for aid in support[reg]]
        c = np.mean(specs, axis=0)
        centroids.append(c / (float(np.linalg.norm(c)) + 1e-8))
    cmat = np.stack(centroids)
    y_true: list[str] = []
    y_pred: list[str] = []
    for aid, true_reg in query:
        s = snv(_one_face(year_data[tgt_year][true_reg][aid], rng))
        s = s / (float(np.linalg.norm(s)) + 1e-8)
        y_true.append(true_reg)
        y_pred.append(regions[int((cmat @ s).argmax())])
    return _macro_and_perclass(y_true, y_pred, regions)


def _eval_sklearn(year_data: Any, tgt_year: int, k_shot: int, seed: int, kind: str) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(seed)
    regions, support, query = build_support_query(year_data, tgt_year, k_shot, rng)
    idx = {r: i for i, r in enumerate(regions)}
    x_tr = []
    y_tr = []
    for reg in regions:
        for aid in support[reg]:
            x_tr.append(_one_face(year_data[tgt_year][reg][aid], rng))
            y_tr.append(idx[reg])
    x_train = snv(np.stack(x_tr))
    y_train = np.array(y_tr)
    if len(np.unique(y_train)) < 2:
        return eval_nearest_centroid(year_data, tgt_year, k_shot, seed)
    x_q = []
    y_t: list[str] = []
    for aid, true_reg in query:
        x_q.append(_one_face(year_data[tgt_year][true_reg][aid], rng))
        y_t.append(true_reg)
    x_query = snv(np.stack(x_q))
    try:
        if kind == "pls":
            n_comp = max(1, min(len(x_train) - 1, x_train.shape[1], len(regions)))
            y_oh = np.zeros((len(x_train), len(regions)))
            y_oh[np.arange(len(x_train)), y_train] = 1.0
            pls = PLSRegression(n_components=n_comp, max_iter=500)
            pls.fit(x_train, y_oh)
            y_p = [regions[int(s.argmax())] for s in pls.predict(x_query)]
        else:
            clf = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale"))])
            clf.fit(x_train, y_train)
            y_p = [regions[int(i)] for i in clf.predict(x_query)]
    except Exception:
        return eval_nearest_centroid(year_data, tgt_year, k_shot, seed)
    return _macro_and_perclass(y_t, y_p, regions)


def _proto_eval(
    feats: Any, year_data: Any, tgt_year: int, k_shot: int, seed: int, src_years: list[int], src_alpha: float
) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(seed)
    regions, support, query = build_support_query(year_data, tgt_year, k_shot, rng)
    tgt_protos: dict[str, NDArray] = {}
    for reg in regions:
        ff = [_one_face(feats[tgt_year][reg][aid], rng) for aid in support[reg]]
        p = np.mean(ff, axis=0)
        tgt_protos[reg] = p / (float(np.linalg.norm(p)) + 1e-8)
    if src_alpha > 0:
        src_rng = np.random.default_rng(seed + 999)
        src_protos: dict[str, list[NDArray]] = {}
        for src_yr in src_years:
            for reg in feats.get(src_yr, {}):
                if reg not in regions:
                    continue
                ff = [_one_face(fa, src_rng) for fa in feats[src_yr][reg].values()]
                p = np.mean(ff, axis=0)
                src_protos.setdefault(reg, []).append(p / (float(np.linalg.norm(p)) + 1e-8))
        for reg in regions:
            if reg in src_protos:
                sa = np.mean(src_protos[reg], axis=0)
                sa = sa / (float(np.linalg.norm(sa)) + 1e-8)
                fused = (1 - src_alpha) * tgt_protos[reg] + src_alpha * sa
                tgt_protos[reg] = fused / (float(np.linalg.norm(fused)) + 1e-8)
    pmat = np.stack([tgt_protos[r] for r in regions])
    y_true: list[str] = []
    y_pred: list[str] = []
    for aid, true_reg in query:
        feat = _one_face(feats[tgt_year][true_reg][aid], rng)
        y_true.append(true_reg)
        y_pred.append(regions[int((pmat @ feat).argmax())])
    return _macro_and_perclass(y_true, y_pred, regions)


def load_fold_features(
    tgt_year: int, encoder_seed: int, year_data: Any, wl_cols: list[str], n_bands: int, device: torch.device
) -> Any:
    """加载 10 产出的 fold 编码器并预编码所有面位特征；缺失返回 None。"""
    run_name = f"10_loeo_year_foldwise_seed{encoder_seed}_fold{tgt_year}"
    ckpt_path = os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = ProtoEncoder(n_bands, feat_dim=int(ckpt.get("feat_dim", 128)))
    enc.load_state_dict(ckpt["state_dict"])
    enc.to(device)
    enc.eval()
    spec_list: list[NDArray] = []
    index_map: list[tuple[int, str, Any, int]] = []
    for year in (2018, 2019, 2025):
        for reg in year_data[year]:
            for aid in year_data[year][reg]:
                faces = year_data[year][reg][aid]
                for fi in range(len(faces)):
                    spec_list.append(faces[fi])
                    index_map.append((year, reg, aid, fi))
    f_all = encode_batch(enc, np.stack(spec_list).astype(np.float32), device, batch_size=2048)
    nested: dict[int, dict[str, dict[Any, list[Any]]]] = {y: {} for y in (2018, 2019, 2025)}
    for (year, reg, aid, fi), f in zip(index_map, f_all, strict=False):
        nested[year].setdefault(reg, {}).setdefault(aid, [None] * len(year_data[year][reg][aid]))
        nested[year][reg][aid][fi] = f
    out: dict[int, dict[str, dict[Any, NDArray]]] = {y: {} for y in (2018, 2019, 2025)}
    for year in nested:
        for reg in nested[year]:
            out[year][reg] = {aid: np.stack(v) for aid, v in nested[year][reg].items()}
    return out


def compute_gbid_tables(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 §15.8 产出综合结果表 / 原始指标表 / 敏感性总表。"""
    pc_cols = [f"F1类_{r}" for r in ALL_REGIONS if f"F1类_{r}" in panel_df.columns]
    metrics: dict[str, dict[str, float]] = {}
    for m in METHOD_NAMES:
        sub = panel_df[panel_df["方法"] == m]
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
    for m in METHOD_NAMES:
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

    order = sorted(METHOD_NAMES, key=lambda m: (passed[m], float(gbid_res[m]["gbids"])), reverse=True)
    rank = {m: i + 1 for i, m in enumerate(order)}

    comp_rows = [
        {
            "方法": m,
            "Pass": passed[m],
            "ū": round(float(gbid_res[m]["u_bar"]), 4),
            "C_B": round(float(gbid_res[m]["C_B"]), 4),
            "C_W": round(float(gbid_res[m]["C_W"]), 4),
            "GBID": round(float(gbid_res[m]["gbid"]), 4),
            "GBIDS": round(float(gbid_res[m]["gbids"]), 2),
            "主排名": rank[m],
        }
        for m in METHOD_NAMES
    ]
    raw_rows = [
        {
            "方法": m,
            "F1@K5": round(metrics[m]["F1@K5"], 4),
            "F1@K20": round(metrics[m]["F1@K20"], 4),
            "最弱类F1@K5": round(metrics[m]["minPC@K5"], 4),
            "F1_SD@K5": round(metrics[m]["SD@K5"], 4),
            "GBIDS": round(float(gbid_res[m]["gbids"]), 2),
        }
        for m in METHOD_NAMES
    ]

    # q 敏感性：四配置下的平均名次
    q_gbids = {m: eu.gbid_sensitivity_q(u_by_model[m], GBID_W, GBID_V, GBID_DIMS) for m in METHOD_NAMES}
    configs = list(q_gbids[METHOD_NAMES[0]].keys())
    q_rank: dict[str, list[int]] = {m: [] for m in METHOD_NAMES}
    for cfg in configs:
        cfg_order = sorted(METHOD_NAMES, key=lambda mm: q_gbids[mm][cfg], reverse=True)
        for i, mm in enumerate(cfg_order):
            q_rank[mm].append(i + 1)
    # 权重敏感性：Dirichlet 1000 组
    w_sens = eu.gbid_sensitivity_weights(u_by_model, GBID_W, GBID_V, GBID_DIMS, n_draws=1000)
    sens_rows = [
        {
            "方法": m,
            "主排名": rank[m],
            "q敏感性均排名": round(float(np.mean(q_rank[m])), 2),
            "权重扰动第一名概率": round(w_sens[m]["first_place_prob"], 3),
            "权重扰动平均名次": round(w_sens[m]["mean_rank"], 2),
            "基准敏感性": "待CUDA全量",
            "相关性处理一致性": "待CUDA全量",
            "重采样第一名频率": "待CUDA全量",
        }
        for m in METHOD_NAMES
    ]
    return pd.DataFrame(comp_rows), pd.DataFrame(raw_rows), pd.DataFrame(sens_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="20_unified_singleface_kshot：5 模型面板 + LLH-GBID")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k_list", default="1,3,5,10,20,50")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--src_alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--encoder_seed", type=int, default=eu.PILOT_SEED, help="定位 10 的 fold 编码器种子")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_list.split(",")]
    rep_seeds = [args.seed + i for i in range(args.seeds)]
    device = eu.get_device(args.device)
    logger = eu.get_logger("20_unified_singleface_kshot")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "统一单面查询 5 模型面板 + LLH-GBID",
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

    encoded: dict[int, Any] = {}
    for tgt_year in (2018, 2019, 2025):
        encoded[tgt_year] = load_fold_features(tgt_year, args.encoder_seed, year_data, wl_cols, n_bands, device)
        logger.log(f"  fold {tgt_year} 编码器特征：{'OK' if encoded[tgt_year] is not None else '缺失(先跑10)'}")

    all_rows: list[dict[str, Any]] = []
    with eu.track_compute("统一面板", logger, run) as budget:
        for tgt_year, cfg in sorted(FOLD_CFG.items()):
            regions = sorted(year_data[tgt_year].keys())
            feats = encoded[tgt_year]
            for k in k_values:
                acc: dict[str, list[float]] = {m: [] for m in METHOD_NAMES}
                pc_acc: dict[str, dict[str, list[float]]] = {m: {r: [] for r in regions} for m in METHOD_NAMES}
                for s in rep_seeds:
                    results = {
                        "NearestCentroid": eval_nearest_centroid(year_data, tgt_year, k, s),
                        "PLS-DA-Kshot": _eval_sklearn(year_data, tgt_year, k, s, "pls"),
                        "SVM-Kshot": _eval_sklearn(year_data, tgt_year, k, s, "svm"),
                    }
                    if feats is not None:
                        results["VanillaProtoNet"] = _proto_eval(
                            feats, year_data, tgt_year, k, s, cfg["src_years"], 0.0
                        )
                        results["ProtoNet-C(a=0.4)"] = _proto_eval(
                            feats, year_data, tgt_year, k, s, cfg["src_years"], args.src_alpha
                        )
                    for m, (macro, per) in results.items():
                        acc[m].append(macro)
                        for r in regions:
                            pc_acc[m][r].append(per.get(r, float("nan")))
                for m in METHOD_NAMES:
                    if not acc[m]:
                        continue
                    row: dict[str, Any] = {
                        "目标年份": tgt_year,
                        "源年份": "+".join(str(y) for y in cfg["src_years"]),
                        "K": k,
                        "方法": m,
                        "F1均值": round(float(np.mean(acc[m])), 4),
                        "F1标准差": round(float(np.std(acc[m])), 4),
                    }
                    for r in ALL_REGIONS:
                        row[f"F1类_{r}"] = round(float(np.nanmean(pc_acc[m][r])), 4) if r in regions else float("nan")
                    all_rows.append(row)
                logger.log(
                    f"  LOEO-{tgt_year} K={k:3d}："
                    + " ".join(f"{m[:10]}={float(np.mean(acc[m])):.3f}" for m in METHOD_NAMES if acc[m])
                )

    panel_df = pd.DataFrame(all_rows)
    comp_df, raw_df, sens_df = compute_gbid_tables(panel_df)
    logger.log("\n── LLH-GBID 综合结果表（§15.8）──")
    logger.log(comp_df.to_string(index=False))
    best = comp_df.sort_values(["Pass", "GBIDS"], ascending=[False, False]).iloc[0]
    logger.log(
        eu.gbid_paper_paragraph(
            str(best["方法"]),
            {"gbids": best["GBIDS"], "u_bar": best["ū"], "C_B": best["C_B"], "C_W": best["C_W"]},
            bool(best["Pass"]),
        )
    )

    # ─── 双语图 ──────────────────────────────────────────────────────────────
    method_color = {
        "NearestCentroid": "#999999",
        "PLS-DA-Kshot": "#E69F00",
        "SVM-Kshot": "#009E73",
        "VanillaProtoNet": "#56B4E9",
        "ProtoNet-C(a=0.4)": "#D55E00",
    }
    txt = {
        "zh": {
            "x": "K-Shot（每产地支持苹果数）",
            "y": "Macro-F1（三折均值）",
            "rand": "随机基线",
            "ta": "统一单面查询：5 方法 K-Shot 对比",
            "gx": "GBIDS（0—100 综合指数）",
            "tb": "LLH-GBID 综合得分（点=方法，色=准入）",
            "pass": "通过门槛",
            "fail": "未通过门槛",
        },
        "en": {
            "x": "K-shot (support apples per origin)",
            "y": "Macro-F1 (3-fold mean)",
            "rand": "Chance",
            "ta": "Unified single-face: 5-method K-shot",
            "gx": "GBIDS (0-100 composite)",
            "tb": "LLH-GBID composite score (dot=method, colour=gate)",
            "pass": "Pass",
            "fail": "Fail",
        },
    }

    def plot_a(lang: str) -> Any:
        t = txt[lang]
        fig, ax = plt.subplots(figsize=(9, 5))
        for m in METHOD_NAMES:
            sub = panel_df[panel_df["方法"] == m]
            if sub.empty:
                continue
            agg = sub.groupby("K")["F1均值"].mean().reset_index().sort_values("K")
            ax.plot(agg["K"], agg["F1均值"], marker="o", linewidth=2, color=method_color[m], label=m)
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel(t["x"])
        ax.set_ylabel(t["y"])
        ax.set_title(t["ta"])
        ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.4, label=t["rand"])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_b(lang: str) -> Any:
        t = txt[lang]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ordered = comp_df.sort_values("GBIDS")
        ypos = np.arange(len(ordered))
        colors = ["#0072B2" if p else "#D55E00" for p in ordered["Pass"]]
        ax.scatter(ordered["GBIDS"], ypos, c=colors, s=90, zorder=3)
        for yi, (_, r) in zip(ypos, ordered.iterrows(), strict=False):
            ax.plot([0, r["GBIDS"]], [yi, yi], color="gray", alpha=0.3, zorder=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(ordered["方法"].tolist())
        ax.set_xlabel(t["gx"])
        ax.set_title(t["tb"])
        ax.grid(axis="x", alpha=0.3)
        ax.scatter([], [], c="#0072B2", label=t["pass"])
        ax.scatter([], [], c="#D55E00", label=t["fail"])
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        return fig

    for idx, fn in enumerate((plot_a, plot_b)):
        zh, en = eu.save_figure_bilingual(fn, "20_unified_singleface_kshot", idx)
        logger.log(f"图{chr(97 + idx)} → {zh} / {en}")

    budget_df = pd.DataFrame([{"reps": args.seeds, "encoder_seed": args.encoder_seed, **budget}])
    xlsx = eu.write_script_workbook(
        __file__,
        {
            0: ("面板原始结果", panel_df),
            1: ("GBID综合结果表", comp_df),
            2: ("GBID原始指标表", raw_df),
            3: ("GBID敏感性总表", sens_df),
            "compute": budget_df,
        },
    )
    logger.log(f"结果（面板+GBID三表）→ {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
