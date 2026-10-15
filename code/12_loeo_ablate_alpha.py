"""
12_loeo_ablate_alpha.py: src_alpha 消融（3 配置）+ LLH-GBID 三表（配置即"模型"）。

对 LOEO-Year K-Shot 的源域原型融合权重 src_alpha 做 3 配置消融，定位"留出 2025"折
跨测量系统融合导致的崩塌：A 全折 α=0.4（原设置）/ B 仅 2025 折 α=0 其余 0.4（建议修复）/
C 全折 α=0（纯 K-Shot）。三配置视作"模型"，产出 LLH-GBID 三表
（仿 20_unified_singleface_kshot 的 compute_gbid_tables）。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/12_loeo_ablate_alpha.py --device auto --seeds 15

输出文件:
    04outputs/12_loeo_ablate_alpha-图a.pdf / -fig-a.pdf  — 三配置 K-Shot F1 曲线
    04outputs/12_loeo_ablate_alpha-图b.pdf / -fig-b.pdf  — GBIDS 综合得分点图（含准入状态）
    04outputs/12_loeo_ablate_alpha-图c.pdf / -fig-c.pdf  — K=5 各折三配置对比（Supplementary）
    04outputs/12_loeo_ablate_alpha.xlsx                   — 消融面板 + GBID 三表 + 算力
    05logs/12_loeo_ablate_alpha_YY-MM-DD_HHMMSS.log      — 运行日志

说明:
    依赖 10 的 fold 编码器（--encoder_seed，默认 PILOT_SEED），苹果特征=多面位均值。
    GBID 设定锁定于 01STATISTICAL_PROTOCOL.md（维度 D1 判别力/D2 最弱类/D3 稳定性，等权）。
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

CONFIG_NAMES = ["A_uniform_0.4", "B_fix_2025", "C_no_fusion"]
# 配置名 → {留出年份 → src_alpha}
ALPHA_CONFIGS: dict[str, dict[int, float]] = {
    "A_uniform_0.4": {2018: 0.4, 2019: 0.4, 2025: 0.4},
    "B_fix_2025": {2018: 0.4, 2019: 0.4, 2025: 0.0},  # 建议修复
    "C_no_fusion": {2018: 0.0, 2019: 0.0, 2025: 0.0},
}
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

CFG_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "A_uniform_0.4": "A：全折α=0.4（原设置）",
        "B_fix_2025": "B：2025折α=0其余0.4（修复）",
        "C_no_fusion": "C：全折α=0（纯K-Shot）",
    },
    "en": {
        "A_uniform_0.4": "A: α=0.4 all folds (original)",
        "B_fix_2025": "B: α=0 for 2025 only (fix)",
        "C_no_fusion": "C: α=0 all folds (pure K-shot)",
    },
}
CONFIG_COLOR = {"A_uniform_0.4": "#0072B2", "B_fix_2025": "#E69F00", "C_no_fusion": "#009E73"}


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
) -> tuple[float, dict[str, float]]:
    """折内编码器对目标年份的 K-Shot 评估（多面位均值苹果特征 + 源域同类原型融合 src_alpha）。"""
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
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=tgt_regions, average=None, zero_division=0)
    return macro, {r: float(v) for r, v in zip(tgt_regions, per_class, strict=False)}


def compute_gbid_tables(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """配置即"模型"，按 §15.8 产出综合结果表 / 原始指标表 / 敏感性总表（仿 20）。"""
    pc_cols = [f"F1_{r}" for r in ALL_REGIONS if f"F1_{r}" in panel_df.columns]
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
    q_configs = list(q_gbids[CONFIG_NAMES[0]].keys())
    q_rank: dict[str, list[int]] = {m: [] for m in CONFIG_NAMES}
    for qc in q_configs:
        qc_order = sorted(CONFIG_NAMES, key=lambda mm: q_gbids[mm][qc], reverse=True)
        for i, mm in enumerate(qc_order):
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
    parser = argparse.ArgumentParser(description="12_loeo_ablate_alpha：src_alpha 消融 3 配置 + LLH-GBID")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k_list", default="1,3,5,10,20,50")
    parser.add_argument("--seeds", type=int, default=15, help="目标域 support 重采样次数")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--encoder_seed", type=int, default=eu.PILOT_SEED, help="定位 10 的 fold 编码器种子")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_list.split(",")]
    rep_seeds = [args.seed + i for i in range(args.seeds)]
    device = eu.get_device(args.device)
    logger = eu.get_logger("12_loeo_ablate_alpha")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "src_alpha 消融（3 配置）+ LLH-GBID",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "reps": args.seeds,
            "K_list": k_values,
            "encoder_seed": args.encoder_seed,
            "配置": list(ALPHA_CONFIGS.keys()),
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
    with eu.track_compute("src_alpha 消融", logger, run) as budget:
        for cfg_name, alpha_map in ALPHA_CONFIGS.items():
            logger.log(f"\n【配置 {cfg_name}】alpha={alpha_map}")
            for tgt_year, fold in sorted(FOLD_CFG.items()):
                if tgt_year not in fold_encoders:
                    continue
                encoder = fold_encoders[tgt_year]
                tgt_regs = sorted(year_data[tgt_year].keys())
                alpha_this = alpha_map[tgt_year]
                for k in k_values:
                    f1s: list[float] = []
                    cls_f1s: dict[str, list[float]] = {r: [] for r in tgt_regs}
                    for s in rep_seeds:
                        f1, per_cls = eval_protonet_alpha(
                            encoder, year_data, tgt_year, fold["src_years"], k, s, alpha_this, device
                        )
                        f1s.append(f1)
                        for r in tgt_regs:
                            cls_f1s[r].append(per_cls.get(r, 0.0))
                    row: dict[str, Any] = {
                        "配置": cfg_name,
                        "目标年份": tgt_year,
                        "目标产地": "/".join(tgt_regs),
                        "src_alpha": alpha_this,
                        "K": k,
                        "F1均值": round(float(np.mean(f1s)), 4),
                        "F1标准差": round(float(np.std(f1s)), 4),
                    }
                    for r in tgt_regs:
                        row[f"F1_{r}"] = round(float(np.mean(cls_f1s[r])), 4)
                    all_rows.append(row)
                    logger.log(f"  留出{tgt_year} K={k:3d} α={alpha_this} F1={row['F1均值']:.4f}±{row['F1标准差']:.4f}")

    res_df = pd.DataFrame(all_rows)
    comp_df, raw_df, sens_df = compute_gbid_tables(res_df)
    logger.log("\n── LLH-GBID 综合结果表（配置即模型，§15.8）──")
    logger.log(comp_df.to_string(index=False))
    best = comp_df.sort_values(["Pass", "GBIDS"], ascending=[False, False]).iloc[0]
    logger.log(
        eu.gbid_paper_paragraph(
            str(best["配置"]),
            {"gbids": best["GBIDS"], "u_bar": best["ū"], "C_B": best["C_B"], "C_W": best["C_W"]},
            bool(best["Pass"]),
        )
    )

    txt = {
        "zh": {
            "x": "K-Shot（每产地支持苹果数）",
            "y": "三折均值 Macro-F1",
            "rand": "随机基线",
            "ta": "src_alpha 消融：三配置 K-Shot F1（LOEO-Year，折内编码器）",
            "gx": "GBIDS（0—100 综合指数）",
            "tb": "LLH-GBID 综合得分（点=配置，色=准入）",
            "pass": "通过门槛",
            "fail": "未通过门槛",
            "yc": "Macro-F1",
            "tc": "附录：src_alpha 消融 K=5 各折对比（Supplementary）",
            "lo": "留出",
        },
        "en": {
            "x": "K-shot (support apples per origin)",
            "y": "Macro-F1 (3-fold mean)",
            "rand": "Chance",
            "ta": "src_alpha ablation: 3 configs K-shot F1 (LOEO-Year, fold encoder)",
            "gx": "GBIDS (0-100 composite)",
            "tb": "LLH-GBID composite score (dot=config, colour=gate)",
            "pass": "Pass",
            "fail": "Fail",
            "yc": "Macro-F1",
            "tc": "Appendix: src_alpha ablation K=5 per-fold (Supplementary)",
            "lo": "hold-out",
        },
    }

    def plot_a(lang: str) -> Any:
        t = txt[lang]
        fig, ax = plt.subplots(figsize=(9, 5))
        for cfg_name in CONFIG_NAMES:
            sub = res_df[res_df["配置"] == cfg_name]
            if sub.empty:
                continue
            agg = sub.groupby("K")["F1均值"].mean().reset_index().sort_values("K")
            ax.plot(
                agg["K"],
                agg["F1均值"],
                marker="o",
                linewidth=2,
                color=CONFIG_COLOR[cfg_name],
                label=CFG_LABELS[lang][cfg_name],
            )
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel(t["x"])
        ax.set_ylabel(t["y"])
        ax.set_title(t["ta"])
        ax.axhline(1 / 3, color="#999999", linestyle=":", alpha=0.5, label=t["rand"])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
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
            ax.plot([0, r["GBIDS"]], [yi, yi], color="#999999", alpha=0.3, zorder=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels([CFG_LABELS[lang][c] for c in ordered["配置"].tolist()])
        ax.set_xlabel(t["gx"])
        ax.set_title(t["tb"])
        ax.grid(axis="x", alpha=0.3)
        ax.scatter([], [], c="#0072B2", label=t["pass"])
        ax.scatter([], [], c="#D55E00", label=t["fail"])
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        return fig

    def plot_c(lang: str) -> Any:
        t = txt[lang]
        years = [2018, 2019, 2025]
        sub5 = res_df[res_df["K"] == 5]
        x = np.arange(len(years))
        width = 0.26
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, cfg_name in enumerate(CONFIG_NAMES):
            vals = [
                (
                    float(sub5[(sub5["配置"] == cfg_name) & (sub5["目标年份"] == yr)]["F1均值"].iloc[0])
                    if not sub5[(sub5["配置"] == cfg_name) & (sub5["目标年份"] == yr)].empty
                    else float("nan")
                )
                for yr in years
            ]
            offset = (i - len(CONFIG_NAMES) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width, color=CONFIG_COLOR[cfg_name], label=CFG_LABELS[lang][cfg_name])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t['lo']}{yr}" for yr in years])
        ax.set_ylabel(t["yc"])
        ax.set_title(t["tc"])
        ax.set_ylim(0, 1.10)
        ax.axhline(1 / 3, color="#999999", linestyle=":", alpha=0.4)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    for idx, fn in enumerate((plot_a, plot_b, plot_c)):
        zh, en = eu.save_figure_bilingual(fn, "12_loeo_ablate_alpha", idx)
        logger.log(f"图{chr(97 + idx)} → {zh} / {en}")

    budget_df = pd.DataFrame([{"reps": args.seeds, "encoder_seed": args.encoder_seed, **budget}])
    xlsx = eu.write_script_workbook(
        __file__,
        {
            0: ("消融面板结果", res_df),
            1: ("GBID综合结果表", comp_df),
            2: ("GBID原始指标表", raw_df),
            3: ("GBID敏感性总表", sens_df),
            "compute": budget_df,
        },
    )
    logger.log(f"结果（消融面板+GBID三表）→ {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
