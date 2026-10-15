"""
23_reproducibility_multiseed.py: 复现性多种子方差分解（编码器种子 × 情节种子）+ LLH-GBID。

量化基准研究的两层方差：(1) 编码器训练种子方差；(2) 情节采样方差；(3) 方差分解；
(4) 关键结论（ProtoNet 在小 K 占优、SVM 在大 K 占优）在所有编码器种子下方向是否一致。
多模型比较→按 §15 产出 LLH-GBID 三表（稳定性维度自然取"编码器层 SD"）。

运行方式:
    cd <project root>
    # smoke:  python 02code/23_reproducibility_multiseed.py --n_enc 2 --episodes 60 --eval_seeds 3 --k_list 1,5,20
    # formal: python 02code/23_reproducibility_multiseed.py --stage formal --n_enc 5 --episodes 2000 --eval_seeds 50

输出文件:
    04outputs/23_reproducibility_multiseed-图a.pdf / -fig-a.pdf  — K=5 编码器层方差散点
    04outputs/23_reproducibility_multiseed-图b.pdf / -fig-b.pdf  — 多编码器 K-shot 曲线（种子带）
    04outputs/23_reproducibility_multiseed.xlsx                   — 原始/方差分解/方向一致/GBID 三表/算力
    07checkpoints/23_reproducibility_multiseed_seed<enc>_fold<year>/best.pt — 各折-种子编码器
    05logs/23_reproducibility_multiseed_YY-MM-DD_HHMMSS.log

说明:
    formal 阶段编码器种子取 RECOMMENDED_SEEDS（§5.3）；pilot 取 PILOT_SEED 派生。MPS 为 Pilot。
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
import torch.nn as nn
import torch.optim as optim
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.optim.lr_scheduler import CosineAnnealingLR

import export_utils as eu
from protonet import EpisodeSampler, ProtoEncoder, build_apple_data, build_year_region_data, encode_batch, proto_loss

NDArray = npt.NDArray[Any]

FOLD_CFG: dict[int, dict[str, Any]] = {
    2018: {"src_elems": ["E4", "E5", "E6", "E7", "E8", "E9"], "src_years": [2019, 2025]},
    2019: {"src_elems": ["E1", "E2", "E3", "E7", "E8", "E9"], "src_years": [2018, 2025]},
    2025: {"src_elems": ["E1", "E2", "E3", "E4", "E5", "E6"], "src_years": [2018, 2019]},
}
ALL_REGIONS = ["XJ", "SD", "SX", "GS"]
GBID_DIMS = [[0, 1], [2], [3]]
GBID_W = [1 / 3, 1 / 3, 1 / 3]
GBID_V = [[0.5, 0.5], [1.0], [1.0]]
F1_BASELINE = 1 / 3
SD_BASELINE = 0.10
UMIN = 0.30


def snv(x: NDArray) -> NDArray:
    return np.asarray((x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8))


def train_one_encoder(
    tgt_year: int,
    src_elems: list[str],
    enc_seed: int,
    apple_data: Any,
    n_bands: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: eu.Logger,
) -> ProtoEncoder:
    """按 enc_seed 训练折编码器（管线同 10），落 07checkpoints，返回编码器。"""
    run_name = f"23_reproducibility_multiseed_seed{enc_seed}_fold{tgt_year}"
    ckpt_path = os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        enc = ProtoEncoder(n_bands, feat_dim=int(ckpt.get("feat_dim", args.feat_dim)))
        enc.load_state_dict(ckpt["state_dict"])
        enc.to(device)
        enc.eval()
        logger.log(f"  [fold{tgt_year} enc_seed={enc_seed}] 复用 {ckpt_path}")
        return enc
    eu.set_random_seed(enc_seed, logger)  # 编码器随机性由 enc_seed 控制
    sampler = EpisodeSampler(src_elems, apple_data, args.n_way, args.k_support, args.q_query)
    enc = ProtoEncoder(n_bands, feat_dim=args.feat_dim)
    enc.to(device)
    opt = optim.Adam(enc.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.episodes, eta_min=1e-5)
    best_val = 0.0
    no_improve = 0
    for ep in range(1, args.episodes + 1):
        enc.train()
        sx, sy, qx, qy, nw = sampler.sample()
        opt.zero_grad()
        loss, _ = proto_loss(enc, sx, sy, qx, qy, nw, device)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(enc.parameters(), max_norm=5.0)
        opt.step()
        sched.step()
        if ep % args.val_every == 0:
            enc.eval()
            vacc: list[float] = []
            with torch.no_grad():
                for _ in range(30):
                    sv, syv, qv, qyv, nwv = sampler.sample()
                    _, va = proto_loss(enc, sv, syv, qv, qyv, nwv, device)
                    vacc.append(va)
            vm = float(np.mean(vacc))
            if vm > best_val:
                best_val, no_improve = vm, 0
                eu.save_best_checkpoint(
                    enc,
                    run_name,
                    {
                        "stage": args.stage,
                        "training_hardware": device.type,
                        "deterministic_guarantee": "full" if device.type == "cuda" else "partial(MPS pilot)",
                        "architecture": "ProtoEncoder(1D-CNN+MLP)",
                        "task": f"复现性编码器 fold{tgt_year}",
                        "fold_year": tgt_year,
                        "enc_seed": enc_seed,
                        "val_acc": round(best_val, 4),
                    },
                    extra_state={
                        "n_bands": n_bands,
                        "feat_dim": args.feat_dim,
                        "fold_year": tgt_year,
                        "enc_seed": enc_seed,
                    },
                )
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    break
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc.load_state_dict(best["state_dict"])
    enc.eval()
    logger.log(f"  [fold{tgt_year} enc_seed={enc_seed}] best_val={best_val:.4f}")
    return enc


def cache_features(enc: ProtoEncoder, year_data: Any, device: torch.device) -> Any:
    spec_list: list[NDArray] = []
    idx_map: list[tuple[int, str, Any, int]] = []
    for year in (2018, 2019, 2025):
        for reg in year_data[year]:
            for aid in year_data[year][reg]:
                faces = year_data[year][reg][aid]
                for fi in range(len(faces)):
                    spec_list.append(faces[fi])
                    idx_map.append((year, reg, aid, fi))
    f_all = encode_batch(enc, np.stack(spec_list).astype(np.float32), device, batch_size=2048)
    nested: dict[int, dict[str, dict[Any, list[Any]]]] = {y: {} for y in (2018, 2019, 2025)}
    for (year, reg, aid, fi), f in zip(idx_map, f_all, strict=False):
        nested[year].setdefault(reg, {}).setdefault(aid, [None] * len(year_data[year][reg][aid]))
        nested[year][reg][aid][fi] = f
    out: dict[int, dict[str, dict[Any, NDArray]]] = {y: {} for y in (2018, 2019, 2025)}
    for year in nested:
        for reg in nested[year]:
            out[year][reg] = {aid: np.stack(v) for aid, v in nested[year][reg].items()}
    return out


def build_sq(
    year_data: Any, tgt_year: int, k_shot: int, seed: int
) -> tuple[list[str], dict[str, list[Any]], list[tuple[Any, str]]]:
    rng = np.random.default_rng(seed)
    regs = sorted(year_data[tgt_year].keys())
    support: dict[str, list[Any]] = {}
    for reg in regs:
        apples = list(year_data[tgt_year][reg].keys())
        k = min(k_shot, max(1, len(apples) - 1))
        support[reg] = list(rng.choice(apples, size=k, replace=False))
    sup_set = {aid for aids in support.values() for aid in aids}
    query = [(aid, reg) for reg in regs for aid in year_data[tgt_year][reg] if aid not in sup_set]
    if not query:
        query = [(aid, reg) for reg in regs for aid in year_data[tgt_year][reg]]
    return regs, support, query


def _mp(y_true: list[str], y_pred: list[str], regs: list[str]) -> tuple[float, dict[str, float]]:
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per = f1_score(y_true, y_pred, labels=regs, average=None, zero_division=0)
    return macro, {r: float(v) for r, v in zip(regs, per, strict=False)}


def _norm(v: NDArray) -> NDArray:
    return np.asarray(v / (float(np.linalg.norm(v)) + 1e-8))


def eval_proto(feats: Any, year_data: Any, tgt: int, k: int, seed: int) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(seed)
    regs, support, query = build_sq(year_data, tgt, k, seed)
    protos: dict[str, NDArray] = {}
    for reg in regs:
        ff = [feats[tgt][reg][a][rng.integers(0, len(feats[tgt][reg][a]))] for a in support[reg]]
        protos[reg] = _norm(np.mean(ff, axis=0))
    pmat = np.stack([protos[r] for r in regs])
    yt = [r for _, r in query]
    yp = []
    for a, r in query:
        fa = feats[tgt][r][a]
        yp.append(regs[int((pmat @ _norm(fa[rng.integers(0, len(fa))])).argmax())])
    return _mp(yt, yp, regs)


def eval_spectral(
    year_data: Any, tgt: int, k: int, seed: int, kind: str, svm_c: float
) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(seed)
    regs, support, query = build_sq(year_data, tgt, k, seed)
    idx = {r: i for i, r in enumerate(regs)}

    def one_face(a: Any, r: str) -> NDArray:
        arr = year_data[tgt][r][a]
        row: NDArray = arr[rng.integers(0, len(arr))]
        return row

    x_tr = snv(np.stack([one_face(a, r) for r in regs for a in support[r]]))
    y_tr = np.array([idx[r] for r in regs for a in support[r]])
    yt = [r for _, r in query]
    x_q = snv(np.stack([one_face(a, r) for a, r in query]))
    if kind == "nc":
        cent = {}
        off = 0
        for r in regs:
            n_r = len(support[r])
            cent[r] = _norm(np.mean(x_tr[off : off + n_r], axis=0))
            off += n_r
        cmat = np.stack([cent[r] for r in regs])
        yp = [regs[int((cmat @ _norm(x_q[i])).argmax())] for i in range(len(x_q))]
        return _mp(yt, yp, regs)
    if len(np.unique(y_tr)) < 2:
        return eval_spectral(year_data, tgt, k, seed, "nc", svm_c)
    try:
        if kind == "svm":
            clf = Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf", C=svm_c, gamma="scale"))])
            clf.fit(x_tr, y_tr)
            yp = [regs[int(i)] for i in clf.predict(x_q)]
        else:  # pls
            n_comp = max(1, min(len(x_tr) - 1, x_tr.shape[1], len(regs)))
            y_oh = np.zeros((len(x_tr), len(regs)))
            y_oh[np.arange(len(x_tr)), y_tr] = 1.0
            pls = PLSRegression(n_components=n_comp, max_iter=500)
            pls.fit(x_tr, y_oh)
            yp = [regs[int(s.argmax())] for s in pls.predict(x_q)]
    except Exception:
        return eval_spectral(year_data, tgt, k, seed, "nc", svm_c)
    return _mp(yt, yp, regs)


def compute_gbid_tables(panel_df: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pc_cols = [f"F1类_{r}" for r in ALL_REGIONS if f"F1类_{r}" in panel_df.columns]
    u_by: dict[str, list[float]] = {}
    res: dict[str, dict[str, Any]] = {}
    passed: dict[str, bool] = {}
    metrics: dict[str, dict[str, float]] = {}
    for m in methods:
        sub = panel_df[panel_df["方法"] == m]
        k5 = sub[sub["K"] == 5]
        k20 = sub[sub["K"] == 20]
        f1_k5 = float(k5["F1均值"].mean()) if not k5.empty else 0.0
        f1_k20 = float(k20["F1均值"].mean()) if not k20.empty else f1_k5
        sd_k5 = float(k5["F1标准差"].mean()) if not k5.empty else 0.0
        pc = k5[pc_cols].to_numpy() if pc_cols and not k5.empty else np.array([[f1_k5]])
        minpc = float(np.nanmin(pc)) if np.isfinite(pc).any() else f1_k5
        metrics[m] = {"F1@K5": f1_k5, "F1@K20": f1_k20, "minPC@K5": minpc, "SD@K5": sd_k5}
        u = [
            eu.compute_utility(f1_k5, 1.0, F1_BASELINE, "benefit"),
            eu.compute_utility(f1_k20, 1.0, F1_BASELINE, "benefit"),
            eu.compute_utility(minpc, 1.0, F1_BASELINE, "benefit"),
            eu.compute_utility(sd_k5, 0.0, SD_BASELINE, "cost"),
        ]
        u_by[m] = u
        res[m] = eu.compute_gbid(u, GBID_W, GBID_V, GBID_DIMS)
        passed[m] = u[0] >= UMIN and u[2] >= UMIN
    order = sorted(methods, key=lambda m: (passed[m], float(res[m]["gbids"])), reverse=True)
    rank = {m: i + 1 for i, m in enumerate(order)}
    comp = pd.DataFrame(
        [
            {
                "方法": m,
                "Pass": passed[m],
                "ū": round(float(res[m]["u_bar"]), 4),
                "C_B": round(float(res[m]["C_B"]), 4),
                "C_W": round(float(res[m]["C_W"]), 4),
                "GBID": round(float(res[m]["gbid"]), 4),
                "GBIDS": round(float(res[m]["gbids"]), 2),
                "主排名": rank[m],
            }
            for m in methods
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "方法": m,
                "F1@K5": round(metrics[m]["F1@K5"], 4),
                "F1@K20": round(metrics[m]["F1@K20"], 4),
                "最弱类F1@K5": round(metrics[m]["minPC@K5"], 4),
                "编码器层SD@K5": round(metrics[m]["SD@K5"], 4),
                "GBIDS": round(float(res[m]["gbids"]), 2),
            }
            for m in methods
        ]
    )
    q_g = {m: eu.gbid_sensitivity_q(u_by[m], GBID_W, GBID_V, GBID_DIMS) for m in methods}
    cfgs = list(q_g[methods[0]].keys())
    q_rank: dict[str, list[int]] = {m: [] for m in methods}
    for cfg in cfgs:
        for i, mm in enumerate(sorted(methods, key=lambda x: q_g[x][cfg], reverse=True)):
            q_rank[mm].append(i + 1)
    w_sens = eu.gbid_sensitivity_weights(u_by, GBID_W, GBID_V, GBID_DIMS, n_draws=1000)
    sens = pd.DataFrame(
        [
            {
                "方法": m,
                "主排名": rank[m],
                "q敏感性均排名": round(float(np.mean(q_rank[m])), 2),
                "权重扰动第一名概率": round(w_sens[m]["first_place_prob"], 3),
                "基准敏感性": "待CUDA全量",
                "相关性处理一致性": "待CUDA全量",
                "重采样第一名频率": "待CUDA全量",
            }
            for m in methods
        ]
    )
    return comp, raw, sens


def main() -> None:
    parser = argparse.ArgumentParser(description="23_reproducibility_multiseed：两层方差 + GBID")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_enc", type=int, default=5, help="每折编码器种子数")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--n_way", type=int, default=3)
    parser.add_argument("--k_support", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_seeds", type=int, default=50)
    parser.add_argument("--k_list", default="1,3,5,10,20,50")
    parser.add_argument("--svm_c", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_list.split(",")]
    device = eu.get_device(args.device)
    logger = eu.get_logger("23_reproducibility_multiseed")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    enc_seeds = (
        list(eu.RECOMMENDED_SEEDS)[: args.n_enc]
        if args.stage == "formal"
        else [args.seed + i for i in range(args.n_enc)]
    )
    methods = ["ProtoNet", "SVM", "PLS-DA", "NearestCentroid"]
    eu.log_experiment_header(
        logger,
        {
            "任务": "复现性两层方差+GBID",
            "阶段": args.stage,
            "设备": str(device),
            "enc_seeds": enc_seeds,
            "eval_seeds": args.eval_seeds,
            "K_list": k_values,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    apple_data = build_apple_data(df, wl_cols)
    year_data = build_year_region_data(df, wl_cols)
    eval_seeds = [args.seed + i for i in range(args.eval_seeds)]

    rows: list[dict[str, Any]] = []
    enc_meta: list[dict[str, Any]] = []
    with eu.track_compute("复现性多种子", logger, run) as budget:
        for tgt in (2018, 2019, 2025):
            feats_by_enc: dict[int, Any] = {}
            for es in enc_seeds:
                enc = train_one_encoder(tgt, FOLD_CFG[tgt]["src_elems"], es, apple_data, n_bands, args, device, logger)
                feats_by_enc[es] = cache_features(enc, year_data, device)
                run_name = f"23_reproducibility_multiseed_seed{es}_fold{tgt}"
                sha = eu.compute_sha256(os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt"))
                enc_meta.append({"fold": tgt, "enc_seed": es, "sha256_16": sha[:16]})
            for es in enc_seeds:
                feats = feats_by_enc[es]
                for k in k_values:
                    for ev in eval_seeds:
                        macro_p, pc_p = eval_proto(feats, year_data, tgt, k, ev)
                        macro_s, pc_s = eval_spectral(year_data, tgt, k, ev, "svm", args.svm_c)
                        macro_l, pc_l = eval_spectral(year_data, tgt, k, ev, "pls", args.svm_c)
                        macro_n, pc_n = eval_spectral(year_data, tgt, k, ev, "nc", args.svm_c)
                        for mname, macro, pc in (
                            ("ProtoNet", macro_p, pc_p),
                            ("SVM", macro_s, pc_s),
                            ("PLS-DA", macro_l, pc_l),
                            ("NearestCentroid", macro_n, pc_n),
                        ):
                            row = {
                                "fold": tgt,
                                "enc_seed": es,
                                "eval_seed": ev,
                                "K": k,
                                "方法": mname,
                                "macro_f1": macro,
                            }
                            for r in ALL_REGIONS:
                                row[f"F1类_{r}"] = pc.get(r, float("nan"))
                            rows.append(row)
                logger.log(f"  fold{tgt} enc_seed={es} 完成 {len(k_values)} K × {len(eval_seeds)} 情节")

    raw_df = pd.DataFrame(rows)

    # 方差分解（编码器层 vs 情节层）+ panel（供 GBID）
    var_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    for k in k_values:
        for m in methods:
            sub = raw_df[(raw_df["K"] == k) & (raw_df["方法"] == m)]
            if sub.empty:
                continue
            per_enc = sub.groupby("enc_seed")["macro_f1"].mean()
            between_sd = float(per_enc.std(ddof=1)) if len(per_enc) > 1 else 0.0
            within_sd = float(sub.groupby("enc_seed")["macro_f1"].std(ddof=1).mean())
            var_rows.append(
                {
                    "K": k,
                    "方法": m,
                    "总均值": round(float(sub["macro_f1"].mean()), 4),
                    "编码器层SD": round(between_sd, 4),
                    "情节层SD": round(within_sd, 4),
                    "编码器跨度": round(float(per_enc.max() - per_enc.min()), 4),
                }
            )
            prow: dict[str, Any] = {
                "方法": m,
                "K": k,
                "F1均值": round(float(sub["macro_f1"].mean()), 4),
                "F1标准差": round(between_sd, 4),
            }  # 稳定性=编码器层SD
            for r in ALL_REGIONS:
                col = f"F1类_{r}"
                prow[col] = round(float(sub[col].mean()), 4) if col in sub and sub[col].notna().any() else float("nan")
            panel_rows.append(prow)
    var_df = pd.DataFrame(var_rows)
    panel_df = pd.DataFrame(panel_rows)
    comp_df, raw_tbl, sens_df = compute_gbid_tables(panel_df, methods)
    logger.log("\n── LLH-GBID 综合结果表（§15.8）──")
    logger.log(comp_df.to_string(index=False))

    # 方向一致性（ProtoNet vs SVM 各编码器胜负）
    dir_rows: list[dict[str, Any]] = []
    for k in k_values:
        for tgt in (2018, 2019, 2025):
            sub = raw_df[(raw_df["K"] == k) & (raw_df["fold"] == tgt)]
            p = sub[sub["方法"] == "ProtoNet"].groupby("enc_seed")["macro_f1"].mean()
            sv = sub[sub["方法"] == "SVM"].groupby("enc_seed")["macro_f1"].mean()
            pw = int((p > sv).sum())
            sw = int((sv > p).sum())
            dir_rows.append(
                {
                    "K": k,
                    "fold": tgt,
                    "n_enc": len(p),
                    "proto胜": pw,
                    "svm胜": sw,
                    "方向一致": pw == len(p) or sw == len(p),
                }
            )
    dir_df = pd.DataFrame(dir_rows)

    fold_color = {2018: "#0072B2", 2019: "#009E73", 2025: "#D55E00"}
    method_color = {"ProtoNet": "#0072B2", "SVM": "#D55E00", "PLS-DA": "#E69F00", "NearestCentroid": "#999999"}
    txt = {
        "zh": {
            "x": "编码器种子",
            "y": "宏 F1（K=5，情节均值）",
            "ta": "编码器层方差（各折）",
            "kx": "K-Shot",
            "tb": "多编码器 K-shot（细线=单种子，粗线=均值）",
            "rand": "随机基线",
        },
        "en": {
            "x": "Encoder seed",
            "y": "Macro-F1 (K=5, episode mean)",
            "ta": "Encoder-level variance (per fold)",
            "kx": "K-shot",
            "tb": "Multi-encoder K-shot (thin=seed, thick=mean)",
            "rand": "Chance",
        },
    }

    def plot_a(lang: str) -> Any:
        t = txt[lang]
        folds = [f for f in (2018, 2019, 2025) if f in raw_df["fold"].unique()]
        fig, axes = plt.subplots(1, len(folds), figsize=(4 * len(folds), 4.6), sharey=True)
        axes_list = axes if len(folds) > 1 else [axes]
        d5 = raw_df[raw_df["K"] == 5]
        for ax, tgt in zip(axes_list, folds, strict=False):
            for mi, m in enumerate(("ProtoNet", "SVM")):
                per = (
                    d5[(d5["fold"] == tgt) & (d5["方法"] == m)]
                    .groupby("enc_seed")["macro_f1"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                xp = np.arange(len(per)) + mi * 0.2
                ax.errorbar(
                    xp,
                    per["mean"],
                    yerr=per["std"].fillna(0),
                    fmt="o",
                    color=method_color[m],
                    capsize=3,
                    label=m,
                    markersize=6,
                )
            ax.set_title(f"LOEO-{tgt}", color=fold_color[tgt])
            ax.set_xlabel(t["x"])
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes_list[0].set_ylabel(t["y"])
        fig.suptitle(t["ta"])
        fig.tight_layout()
        return fig

    def plot_b(lang: str) -> Any:
        t = txt[lang]
        folds = [f for f in (2018, 2019, 2025) if f in raw_df["fold"].unique()]
        fig, axes = plt.subplots(1, len(folds), figsize=(4.5 * len(folds), 4.6), sharey=True)
        axes_list = axes if len(folds) > 1 else [axes]
        for ax, tgt in zip(axes_list, folds, strict=False):
            d = raw_df[raw_df["fold"] == tgt]
            for m in ("ProtoNet", "SVM"):
                per = d[d["方法"] == m].groupby(["enc_seed", "K"])["macro_f1"].mean().reset_index()
                for es in per["enc_seed"].unique():
                    s = per[per["enc_seed"] == es].sort_values("K")
                    ax.plot(s["K"], s["macro_f1"], color=method_color[m], alpha=0.3, linewidth=1)
                mk = per.groupby("K")["macro_f1"].mean().reset_index().sort_values("K")
                ax.plot(mk["K"], mk["macro_f1"], color=method_color[m], marker="o", linewidth=2.2, label=m)
            ax.set_xscale("log")
            ax.set_xticks(k_values)
            ax.set_xticklabels([str(k) for k in k_values])
            ax.set_xlabel(t["kx"])
            ax.set_title(f"LOEO-{tgt}", color=fold_color[tgt])
            ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.5, label=t["rand"])
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes_list[0].set_ylabel("Macro-F1")
        fig.suptitle(t["tb"])
        fig.tight_layout()
        return fig

    for idx, fn in enumerate((plot_a, plot_b)):
        zh, en = eu.save_figure_bilingual(fn, "23_reproducibility_multiseed", idx)
        logger.log(f"图{chr(97 + idx)} → {zh} / {en}")

    budget_df = pd.DataFrame([{"n_enc": len(enc_seeds), "eval_seeds": args.eval_seeds, **budget}])
    xlsx = eu.write_script_workbook(
        __file__,
        {
            0: ("原始记录", raw_df),
            1: ("方差分解", var_df),
            2: ("方向一致性", dir_df),
            3: ("编码器元数据", pd.DataFrame(enc_meta)),
            4: ("GBID综合结果表", comp_df),
            5: ("GBID原始指标表", raw_tbl),
            6: ("GBID敏感性总表", sens_df),
            "compute": budget_df,
        },
    )
    logger.log(f"结果 → {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
