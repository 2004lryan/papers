"""
22_cnn_resnet_baseline.py: 1D-CNN / 1D-ResNet 折内监督学习深度基线（LOEO-Year，F1 + LLH-GBID 三表）。

补审稿人 MAJOR：主表仅有浅层 K-shot 方法，缺现代深度基线。本脚本在 LOEO-Year 折限定
（折内全部源样本带产地标签上监督训练，不接触目标年）下补两条深度基线：
    CNN1D（4 层 1D 卷积）/ ResNet1D（3 个残差块）。两者均非 ProtoNet，网络在本文件本地定义
    （不从 protonet 导入）。在目标年所有苹果上单面查询预测，报告三折 Macro-F1（对应 K=∞ 列）。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/22_cnn_resnet_baseline.py --device auto --epochs 60

输出文件:
    04outputs/22_cnn_resnet_baseline-图a.pdf / -fig-a.pdf  — 三折 CNN/ResNet Macro-F1 点图
    04outputs/22_cnn_resnet_baseline-图b.pdf / -fig-b.pdf  — GBIDS 综合得分点图（含准入）
    04outputs/22_cnn_resnet_baseline.xlsx                   — 三折明细 + GBID 三表 + 算力
    07checkpoints/22_cnn_resnet_baseline_seed<seed>_<cnn|resnet>_fold<year>/best.pt — 各折最优权重
    05logs/22_cnn_resnet_baseline_YY-MM-DD_HHMMSS.log

说明:
    2 模型多指标且 per-class F1 易得，故按 §13/§15 产出 LLH-GBID 三表：
      D1 判别力 {Macro-F1, Acc}（效益, L=1/3）/ D2 最弱类 {min per-class F1@折}（效益, L=1/3）/
      D3 稳定性 {mean SD(Macro-F1)@折}（成本, L=0.10）；J_crit={Macro-F1, 最弱类}, u_min=0.30, 主 q=2。
    每折 checkpoint 经 eu.save_best_checkpoint 落 07checkpoints（禁止裸 torch.save / 禁止读写 04outputs/*.pt）。
    每 seed 局部重播种为多种子评估实验逻辑（保留）。MPS 为 Pilot（<MPS_PILOT>）。
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

import export_utils as eu

NDArray = npt.NDArray[Any]

METHOD_NAMES: list[str] = ["CNN1D", "ResNet1D"]
FOLD_CFG: dict[int, dict[str, Any]] = {
    2018: {"src_years": [2019, 2025]},
    2019: {"src_years": [2018, 2025]},
    2025: {"src_years": [2018, 2019]},
}

# LLH-GBID 设定（§13；指标 Macro-F1/Acc/最弱类，维度自 K-shot 协议映射到监督全源）
GBID_DIMS = [[0, 1], [2], [3]]  # D1 判别力{F1,Acc} / D2 最弱类{minPC} / D3 稳定性{SD}
GBID_W = [1 / 3, 1 / 3, 1 / 3]
GBID_V = [[0.5, 0.5], [1.0], [1.0]]
F1_BASELINE = 1 / 3
SD_BASELINE = 0.10
UMIN = 0.30

TXT: dict[str, dict[str, str]] = {
    "zh": {
        "a_title": "1D 深度基线（折限定监督训练，单面查询）",
        "a_x": "Macro-F1（随机种子均值 ± SD）",
        "rand": "随机基准 (1/3)",
        "gbid_x": "GBIDS（0—100 综合指数）",
        "gbid_title": "LLH-GBID 综合得分（点=方法, 色=准入）",
        "pass": "通过门槛",
        "fail": "未通过门槛",
    },
    "en": {
        "a_title": "1D deep baselines (fold-restricted supervised, single-face query)",
        "a_x": "Macro-F1 (seed mean ± SD)",
        "rand": "Chance (1/3)",
        "gbid_x": "GBIDS (0-100 composite)",
        "gbid_title": "LLH-GBID composite score (dot=method, colour=gate)",
        "pass": "Pass",
        "fail": "Fail",
    },
}
FOLD_COLOR: dict[int, str] = {2018: "#0072B2", 2019: "#E69F00", 2025: "#009E73"}


# ═══════════════════════════════════════════════════════════════════════════════
# 网络定义（非 ProtoNet，本地定义 + 类型注解）
# ═══════════════════════════════════════════════════════════════════════════════
class CNN1D(nn.Module):
    """4 层 1D-CNN 分类器。"""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1)).squeeze(-1)
        out: torch.Tensor = self.fc(h)
        return out


class ResBlock1D(nn.Module):
    """1D 残差块（两层 3-kernel 卷积 + 跳连）。"""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return nn.functional.relu(x + h)


class ResNet1D(nn.Module):
    """3 残差块 1D-ResNet 分类器。"""

    def __init__(self, in_dim: int, num_classes: int, base_ch: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, 7, padding=3), nn.BatchNorm1d(base_ch), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.blocks = nn.Sequential(ResBlock1D(base_ch), ResBlock1D(base_ch), ResBlock1D(base_ch))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(base_ch, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x.unsqueeze(1))
        h = self.blocks(h)
        out: torch.Tensor = self.head(h)
        return out


def train_eval(
    model_factory: Callable[[int, int], nn.Module],
    x_src: NDArray,
    y_src: NDArray,
    x_tgt: NDArray,
    y_tgt: NDArray,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    n_bands: int,
    n_origin: int,
    device: torch.device,
) -> tuple[float, float, float, nn.Module]:
    """折内监督训练 + 目标年单面查询评估，返回 (Macro-F1, Acc, 最弱类 F1, 模型)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = model_factory(n_bands, n_origin).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(x_src)
    xs_t = torch.from_numpy(x_src).float()
    ys_t = torch.from_numpy(y_src).long()
    xt_t = torch.from_numpy(x_tgt).float().to(device)
    for _ep in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            sel = idx[i : i + batch_size]
            xb = xs_t[sel].to(device)
            yb = ys_t[sel].to(device)
            opt.zero_grad()
            logit = model(xb)
            loss = nn.functional.cross_entropy(logit, yb)
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        out = model(xt_t).softmax(1).cpu().numpy()
    pred = out.argmax(1)
    f1 = float(f1_score(y_tgt, pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_tgt, pred))
    labels = sorted(set(y_tgt.tolist()))
    per_cls = f1_score(y_tgt, pred, labels=labels, average=None, zero_division=0)
    min_pc = float(np.min(per_cls)) if len(per_cls) else f1
    return f1, acc, min_pc, model


def get_apple_level_x_y(
    df: pd.DataFrame, wl_cols: list[str], orig_to_idx: dict[str, int], year: int, rng: np.random.Generator
) -> tuple[NDArray, NDArray]:
    """每个苹果取一个随机面位作代表光谱（单面查询）。"""
    sub = df[df["年份"] == year]
    apple_x: list[NDArray] = []
    apple_y: list[int] = []
    for _aid, grp in sub.groupby("苹果编号"):
        row = grp.iloc[int(rng.integers(0, len(grp)))]
        apple_x.append(row[wl_cols].to_numpy().astype(np.float32))
        apple_y.append(orig_to_idx[row["产地"]])
    return np.array(apple_x, dtype=np.float32), np.array(apple_y, dtype=np.int64)


def get_full_year_x_y(
    df: pd.DataFrame, wl_cols: list[str], orig_to_idx: dict[str, int], year: int
) -> tuple[NDArray, NDArray]:
    """返回该年全部面位级训练样本。"""
    sub = df[df["年份"] == year]
    x = sub[wl_cols].to_numpy().astype(np.float32)
    y = sub["产地"].map(orig_to_idx).to_numpy().astype(np.int64)
    return x, y


def compute_gbid_tables(res_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 §15.8 产出综合结果表 / 原始指标表 / 敏感性总表。"""
    metrics: dict[str, dict[str, float]] = {}
    for m in METHOD_NAMES:
        sub = res_df[res_df["Method"] == m]
        f1 = float(sub["F1_mean"].mean()) if not sub.empty else 0.0
        acc = float(sub["Acc_mean"].mean()) if not sub.empty else 0.0
        min_pc = float(sub["minPC_mean"].min()) if not sub.empty else 0.0
        sd_f1 = float(sub["F1_std"].mean()) if not sub.empty else 0.0
        metrics[m] = {"F1": f1, "Acc": acc, "minPC": min_pc, "SD_F1": sd_f1}

    u_by_model: dict[str, list[float]] = {}
    gbid_res: dict[str, dict[str, Any]] = {}
    passed: dict[str, bool] = {}
    for m in METHOD_NAMES:
        mt = metrics[m]
        u = [
            eu.compute_utility(mt["F1"], 1.0, F1_BASELINE, "benefit"),
            eu.compute_utility(mt["Acc"], 1.0, F1_BASELINE, "benefit"),
            eu.compute_utility(mt["minPC"], 1.0, F1_BASELINE, "benefit"),
            eu.compute_utility(mt["SD_F1"], 0.0, SD_BASELINE, "cost"),
        ]
        u_by_model[m] = u
        gbid_res[m] = eu.compute_gbid(u, GBID_W, GBID_V, GBID_DIMS)
        passed[m] = u[0] >= UMIN and u[2] >= UMIN  # J_crit = {Macro-F1, 最弱类}

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
            "Macro-F1": round(metrics[m]["F1"], 4),
            "Acc": round(metrics[m]["Acc"], 4),
            "最弱类F1": round(metrics[m]["minPC"], 4),
            "F1_SD(折均)": round(metrics[m]["SD_F1"], 4),
            "GBIDS": round(float(gbid_res[m]["gbids"]), 2),
        }
        for m in METHOD_NAMES
    ]

    q_gbids = {m: eu.gbid_sensitivity_q(u_by_model[m], GBID_W, GBID_V, GBID_DIMS) for m in METHOD_NAMES}
    configs = list(q_gbids[METHOD_NAMES[0]].keys())
    q_rank: dict[str, list[int]] = {m: [] for m in METHOD_NAMES}
    for cfg in configs:
        cfg_order = sorted(METHOD_NAMES, key=lambda mm: q_gbids[mm][cfg], reverse=True)
        for i, mm in enumerate(cfg_order):
            q_rank[mm].append(i + 1)
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
    parser = argparse.ArgumentParser(description="22_cnn_resnet_baseline：1D-CNN/ResNet 折限定监督基线 + LLH-GBID")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=5, help="评估重复种子数（rep）")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    rep_seeds = [args.seed + i for i in range(args.seeds)]
    device = eu.get_device(args.device)
    logger = eu.get_logger("22_cnn_resnet_baseline")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "1D-CNN/ResNet 折限定监督基线 + LLH-GBID",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "reps": args.seeds,
            "epochs": args.epochs,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    all_origins = sorted(df["产地"].unique())
    n_origin = len(all_origins)
    orig_to_idx = {o: i for i, o in enumerate(all_origins)}
    logger.log(f"总样本：{len(df)}  波段：{n_bands}  产地（全 {n_origin} 类）：{all_origins}")

    factories: dict[str, Callable[[int, int], nn.Module]] = {"CNN1D": CNN1D, "ResNet1D": ResNet1D}
    all_rows: list[dict[str, Any]] = []
    with eu.track_compute("CNN/ResNet 折限定监督", logger, run) as budget:
        for tgt_year, cfg in sorted(FOLD_CFG.items()):
            src_years = cfg["src_years"]
            x_src_list, y_src_list = [], []
            for sy in src_years:
                xs, ys = get_full_year_x_y(df, wl_cols, orig_to_idx, sy)
                x_src_list.append(xs)
                y_src_list.append(ys)
            x_src = np.concatenate(x_src_list, axis=0)
            y_src = np.concatenate(y_src_list, axis=0)
            logger.log(f"\n【LOEO-{tgt_year}】源年份={src_years}  训练集={x_src.shape}")

            for name, factory in factories.items():
                f1_list: list[float] = []
                acc_list: list[float] = []
                minpc_list: list[float] = []
                best_f1, best_model = -1.0, None
                for s in rep_seeds:
                    rng = np.random.default_rng(s)
                    x_tgt, y_tgt = get_apple_level_x_y(df, wl_cols, orig_to_idx, tgt_year, rng)
                    f1, acc, min_pc, model = train_eval(
                        factory,
                        x_src,
                        y_src,
                        x_tgt,
                        y_tgt,
                        args.epochs,
                        args.lr,
                        args.batch,
                        s,
                        n_bands,
                        n_origin,
                        device,
                    )
                    f1_list.append(f1)
                    acc_list.append(acc)
                    minpc_list.append(min_pc)
                    if f1 > best_f1:
                        best_f1, best_model = f1, model
                m_f1, s_f1 = float(np.mean(f1_list)), float(np.std(f1_list))
                logger.log(
                    f"  {name:<10s} F1={m_f1:.4f}±{s_f1:.4f}  "
                    f"Acc={np.mean(acc_list):.4f}  最弱类={np.mean(minpc_list):.4f}"
                )
                all_rows.append(
                    {
                        "TargetYear": tgt_year,
                        "SrcYears": "+".join(str(y) for y in src_years),
                        "Method": name,
                        "F1_mean": round(m_f1, 4),
                        "F1_std": round(s_f1, 4),
                        "Acc_mean": round(float(np.mean(acc_list)), 4),
                        "minPC_mean": round(float(np.mean(minpc_list)), 4),
                    }
                )
                if best_model is not None:
                    run_name = f"22_cnn_resnet_baseline_seed{args.seed}_{name.lower()}_fold{tgt_year}"
                    eu.save_best_checkpoint(
                        best_model,
                        run_name,
                        {
                            "stage": args.stage,
                            "training_hardware": device.type,
                            "deterministic_guarantee": "full" if device.type == "cuda" else "partial(MPS pilot)",
                            "architecture": name,
                            "task": f"折限定监督深度基线（留出 {tgt_year}）",
                            "fold_year": tgt_year,
                            "val_f1": round(best_f1, 4),
                        },
                        extra_state={
                            "n_bands": n_bands,
                            "n_origin": n_origin,
                            "fold_year": tgt_year,
                            "val_f1": best_f1,
                            "model_name": name,
                        },
                    )

    res_df = pd.DataFrame(all_rows)
    comp_df, raw_df, sens_df = compute_gbid_tables(res_df)
    logger.log("\n三折均值汇总：")
    for m in METHOD_NAMES:
        sub = res_df[res_df["Method"] == m]
        logger.log(f"  {m:<10s} 三折均值 F1={sub['F1_mean'].mean():.4f}")
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
    years = [2018, 2019, 2025]

    def plot_a(lang: str) -> Any:
        t = TXT[lang]
        fig, ax = plt.subplots(figsize=(9, 5))
        ypos = np.arange(len(METHOD_NAMES))
        offsets = {2018: -0.22, 2019: 0.0, 2025: 0.22}
        for yr in years:
            xs: list[float] = []
            errs: list[float] = []
            for m in METHOD_NAMES:
                sub = res_df[(res_df["Method"] == m) & (res_df["TargetYear"] == yr)]
                xs.append(float(sub["F1_mean"].iloc[0]) if not sub.empty else float("nan"))
                errs.append(float(sub["F1_std"].iloc[0]) if not sub.empty else 0.0)
            ax.errorbar(
                xs,
                ypos + offsets[yr],
                xerr=errs,
                fmt="o",
                color=FOLD_COLOR[yr],
                capsize=4,
                markersize=8,
                label=f"LOEO-{yr}",
            )
        ax.axvline(1 / 3, color="gray", linestyle=":", alpha=0.6, label=t["rand"])
        ax.set_yticks(ypos)
        ax.set_yticklabels(METHOD_NAMES, fontsize=11)
        ax.set_xlabel(t["a_x"])
        ax.set_xlim(0, 1.0)
        ax.set_title(t["a_title"], fontsize=12)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_b(lang: str) -> Any:
        t = TXT[lang]
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        ordered = comp_df.sort_values("GBIDS")
        ypos = np.arange(len(ordered))
        colors = ["#0072B2" if p else "#D55E00" for p in ordered["Pass"]]
        ax.scatter(ordered["GBIDS"], ypos, c=colors, s=110, zorder=3)
        for yi, (_, r) in zip(ypos, ordered.iterrows(), strict=False):
            ax.plot([0, r["GBIDS"]], [yi, yi], color="gray", alpha=0.3, zorder=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(ordered["方法"].tolist())
        ax.set_xlabel(t["gbid_x"])
        ax.set_title(t["gbid_title"])
        ax.grid(axis="x", alpha=0.3)
        ax.scatter([], [], c="#0072B2", label=t["pass"])
        ax.scatter([], [], c="#D55E00", label=t["fail"])
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        return fig

    for idx, fn in enumerate((plot_a, plot_b)):
        zh, en = eu.save_figure_bilingual(fn, "22_cnn_resnet_baseline", idx)
        logger.log(f"图{chr(97 + idx)} → {zh} / {en}")

    budget_df = pd.DataFrame([{"reps": args.seeds, "epochs": args.epochs, "n_folds": len(FOLD_CFG), **budget}])
    xlsx = eu.write_script_workbook(
        __file__,
        {
            0: ("三折明细", res_df),
            1: ("GBID综合结果表", comp_df),
            2: ("GBID原始指标表", raw_df),
            3: ("GBID敏感性总表", sens_df),
            "compute": budget_df,
        },
    )
    logger.log(f"结果（三折明细 + GBID 三表）→ {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
