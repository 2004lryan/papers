"""74_multi_encoder_leakage.py: 公开数据集上的多编码器评估泄漏对照（五个数据集、16 个留出域）。

为什么:
    24 号只用一维卷积编码器比较了共享（即 S_src）与折限定（即 R）两臂，没有标签置乱与半数暴露两臂；
    审稿意见指出泄漏只在一种编码器上测过。本脚本沿用 crossdataset_lib 的数据加载、类选择、元训练与
    K-shot 评估口径（与 24 号同一调用），对每个编码器族跑四臂：
        R       折限定  : 训练池 = 源域
        S_src   目标暴露 : 训练池 = 源域 ∪ 留出域（带标签）
        S_shuf  标签置乱 : 训练池同 S_src，但留出域样本的类别在该域内随机置换（类别大小不变）
        S_half  半数暴露 : 留出域各类样本按种子对半分成 A、B；训练池 = 源域 ∪ A（带标签），只在 B 上评估；
                          R 与 S_src 的编码器也在同一个 B 上评估（记作 R_B、S_src_B）
    并登记第五个数据集：猕猴桃（同一台仪器 KK1，2017/2018/2019 三个采收季，干物质按源季三分位分 3 类）。

编码器族（encoders.py）:
    cnn / mlp / resnet / transformer 按原型网络元训练 1000 情节（crossdataset_lib.meta_train，与 24 号相同）；
    pls / pca 在训练池上直接拟合 --lin_dim 个成分。pca 不看标签，S_shuf 与 S_src 相同，不单独跑。
    与 24 号一致，公开数据集上的编码器不落盘：每臂在固定种子下可逐位重训。

配对:
    同一折内各臂共用元训练种子（初始化相同）与评估种子（支持集逐次相同）；B 半上的三臂共用同一组类别
    （B 半里每类样本数 ≥ K+2 的类，少于 3 类时该折不评 B 半三臂）。

B3 路线面板（--route_panel，只随 cnn 运行）:
    同一折、同一批情节（与 eval_kshot 逐次相同的抽样）上评估七条路线：原型网络（编码器 R / S_src）、
    SVM 与 PLS-DA（超参数在源域选：各源域轮流作伪目标；在留出域选：留出域上的选参情节，被评估的样本会出现在其中）、
    最近质心（与 20 号相同的余弦判别）。候选网格与 71 号相同，K ∈ {1, 5, 10, 20}
    （留出域有类不足 K+2 个样本时该 K 不评）。
    面板内自检：K=5 时面板里的原型网络 R 与 B1 同进程算出的 R 必须逐位相同，否则中止。

自检（--selfcheck）:
    cnn 的 R、S_src 与 24 号正式结果（04outputs/24_crossdataset_leakage.xlsx「表d：长格式」）逐 (数据集, 留出域, 种子)
    比对；两者同一代码路径，同机同环境下应逐位一致。

运行方式:
    cd <项目根>
    KMP_DUPLICATE_LIB_OK=TRUE python 02code/74_multi_encoder_leakage.py --device auto --stage pilot --encoder mlp
    # Formal（CUDA，每次一个种子 × 一个编码器）:
    python 02code/74_multi_encoder_leakage.py --device cuda --stage formal --seed 20060515 --encoder transformer \
        --data_root /tmp/016_ext
    python 02code/74_multi_encoder_leakage.py --device cuda --stage formal --seed 20060515 --encoder cnn --route_panel \
        --data_root /tmp/016_ext
    # 自检与汇总
    python 02code/74_multi_encoder_leakage.py --selfcheck
    python 02code/74_multi_encoder_leakage.py --aggregate

输出文件:
    04outputs/formal_seeds/74_multi_encoder__<编码器>__seed<S>[__pilot].xlsx — 逐数据集逐留出域逐臂的宏 F1
                                                                        （cnn 且 --route_panel 时另有「路线面板」表）
    04outputs/74_multi_encoder_leakage.xlsx — 汇总（--aggregate）：逐折对照、按数据集汇总、长格式
    05logs/74_multi_encoder_leakage_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import f1_score

import crossdataset_lib as cdl
import encoders as enc_lib
import export_utils as eu

NDArray = npt.NDArray[Any]

DATASETS: dict[str, dict[str, Any]] = {
    **cdl.DATASETS,
    "kiwifruit_season": {"loader": cdl.load_kiwifruit, "drift": "采收季（同仪器）", "modality": "NIR点谱"},
}
ARMS = ("R", "S_src", "S_shuf", "S_half")
HALF_ARMS = {"R_B": "R", "S_src_B": "S_src"}
EVAL_ARMS = (*ARMS, *HALF_ARMS)
CONTRASTS: dict[str, tuple[str, str]] = {
    "Δ_rep": ("S_src", "R"),
    "Δ_lab": ("S_src", "S_shuf"),
    "Δ_shuf": ("S_shuf", "R"),
    "Δ_half": ("S_half", "R_B"),
    "Δ_seen": ("S_src_B", "S_half"),
    "Δ_rep_B": ("S_src_B", "R_B"),
}
REF_24 = os.path.join(eu.OUTPUT_DIR, "24_crossdataset_leakage.xlsx")
SEED_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")

# B3 路线面板：无泄漏与泄漏两种评估下的路线（最近质心不选参、不用编码器，两边共用）
m71 = importlib.import_module("71_source_only_tuning")
ROUTE_K = (1, 5, 10, 20)
ROUTES_FREE = ("原型网络(R)", "SVM(源域选参)", "PLS-DA(源域选参)", "最近质心")
ROUTES_LEAK = ("原型网络(S_src)", "SVM(留出域选参)", "PLS-DA(留出域选参)", "最近质心")
SK_KINDS: dict[str, tuple[str, tuple[Any, ...], Any]] = {
    "SVM": ("svm", m71.SVM_GRID, m71.SVM_FIXED),
    "PLS-DA": ("pls", m71.PLS_GRID, 3),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 一、折内数组与三种训练池
# ═══════════════════════════════════════════════════════════════════════════════
def fold_arrays(
    doms: dict[str, tuple[NDArray, NDArray]], is_reg: bool, held: str, k: int
) -> tuple[cdl.FoldArrays, NDArray] | None:
    """与 cdl.prepare_fold 同一口径构造本折数组，另返回 xall 各行所属的域名；类不足时返回 None。"""
    fa = cdl.prepare_fold(doms, is_reg, held, k, preprocess="snv")
    if fa is None:
        return None
    names = list(doms.keys())
    y_src_cont = np.concatenate([doms[d][1] for d in names if d != held])
    rows: list[str] = []
    for d in names:
        yl = cdl.tertile_labels(y_src_cont, doms[d][1]) if is_reg else doms[d][1].astype(int)
        rows += [d] * int(np.isin(yl, fa.classes).sum())
    dom_of_row = np.asarray(rows)
    if len(dom_of_row) != len(fa.yall) or not np.array_equal(fa.xall[dom_of_row == held], fa.xh):
        raise RuntimeError(f"留出 {held}：行归属与 prepare_fold 不一致")
    return fa, dom_of_row


def shuffled_pool(fa: cdl.FoldArrays, dom_of_row: NDArray, held: str, seed: int) -> tuple[NDArray, NDArray]:
    """S_shuf：训练池同 S_src，留出域样本的类别在该域内随机置换。"""
    y = fa.yall.copy()
    idx = np.where(dom_of_row == held)[0]
    perm = np.random.default_rng([seed, 74, 7]).permutation(len(idx))
    y[idx] = fa.yall[idx][perm]
    return fa.xall, y


def half_split(fa: cdl.FoldArrays, dom_of_row: NDArray, held: str, seed: int) -> tuple[NDArray, NDArray, NDArray]:
    """S_half：留出域各类样本按种子对半分 A、B（奇数时 B 多一个）。

    返回 (训练池 x, 训练池 y, B 半在 xh 中的下标)；训练池 = xall 去掉留出域的 B 半，行序不变。
    """
    rng = np.random.default_rng([seed, 74, 11])
    in_a = np.zeros(len(fa.yh), dtype=bool)
    for c in fa.classes:
        ids = np.where(fa.yh == c)[0]
        perm = rng.permutation(len(ids))
        in_a[ids[perm[: len(ids) // 2]]] = True
    held_rows = np.where(dom_of_row == held)[0]
    keep = np.ones(len(fa.yall), dtype=bool)
    keep[held_rows[~in_a]] = False
    return fa.xall[keep], fa.yall[keep], np.where(~in_a)[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 二、训练与评估
# ═══════════════════════════════════════════════════════════════════════════════
def obtain(
    encoder: str, x: NDArray, y: NDArray, in_dim: int, args: argparse.Namespace, device: torch.device, seed: int
) -> nn.Module:
    """深度族按原型网络元训练；线性族在训练池上拟合（pca 不看 y）。"""
    if encoder in enc_lib.LINEAR_ENCODERS:
        enc, _ = enc_lib.fit_linear_encoder(encoder, x, y if encoder == "pls" else None, args.lin_dim)
        return enc.to(device).eval()
    return cdl.meta_train(x, y, in_dim, device, args.episodes, seed, encoder=encoder)


def run_fold(
    doms: dict[str, tuple[NDArray, NDArray]],
    is_reg: bool,
    held: str,
    encoder: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    logger: eu.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """单个留出域：四臂训练 + 全部样本与 B 半上的评估，返回 (逐臂记录, B3 路线面板记录)；类不足时返回空表。"""
    got = fold_arrays(doms, is_reg, held, args.k)
    if got is None:
        logger.log(f"    留出 {held}：合法类不足 3 个，此折不评")
        return [], []
    fa, dom_of_row = got
    arms = [a for a in ARMS if not (encoder == "pca" and a == "S_shuf")]
    pools: dict[str, tuple[NDArray, NDArray]] = {"R": (fa.xs, fa.ys), "S_src": (fa.xall, fa.yall)}
    if "S_shuf" in arms:
        pools["S_shuf"] = shuffled_pool(fa, dom_of_row, held, seed)
    xa, ya, b_idx = half_split(fa, dom_of_row, held, seed)
    pools["S_half"] = (xa, ya)
    xb, yb = fa.xh[b_idx], fa.yh[b_idx]
    classes_b = [c for c in fa.classes if int((yb == c).sum()) >= args.k + 2]

    encs: dict[str, nn.Module] = {}
    for arm in arms:
        x, y = pools[arm]
        encs[arm] = obtain(encoder, x, y, fa.in_dim, args, device, seed)
    base = {
        "held": held,
        "编码器": encoder,
        "种子": seed,
        "n_classes": len(fa.classes),
        "classes": ",".join(str(c) for c in fa.classes),
        "mmd2": fa.mmd2,
    }
    rows: list[dict[str, Any]] = []
    for arm in [a for a in arms if a != "S_half"]:
        f1 = cdl.eval_kshot(encs[arm], fa.xh, fa.yh, fa.classes, device, args.k, args.reps, seed)
        n_tr = len(pools[arm][1])
        rows.append({**base, "臂": arm, "评估样本": "全部", "n_eval": len(fa.yh), "n_train": n_tr, "F1": f1})
    if len(classes_b) >= 3:
        keep = np.isin(yb, classes_b)
        for arm, src_arm in (("S_half", "S_half"), *HALF_ARMS.items()):
            f1 = cdl.eval_kshot(encs[src_arm], xb[keep], yb[keep], classes_b, device, args.k, args.reps, seed)
            rows.append(
                {
                    **base,
                    "臂": arm,
                    "评估样本": "B半",
                    "B半类别": ",".join(str(c) for c in classes_b),
                    "n_eval": int(keep.sum()),
                    "n_train": len(pools[src_arm][1]),
                    "F1": f1,
                }
            )
    else:
        logger.log(f"    留出 {held}：B 半里样本数 ≥ K+2 的类不足 3 个，不评 B 半三臂")
    msg = "  ".join(f"{r['臂']}={r['F1']:.4f}" for r in rows)
    logger.log(f"    留出 {held}（{len(fa.classes)} 类）：{msg}")
    panel: list[dict[str, Any]] = []
    if args.route_panel:
        f1_r = next(float(r["F1"]) for r in rows if r["臂"] == "R" and r["评估样本"] == "全部")
        panel = route_panel(fa, dom_of_row, held, encs["R"], encs["S_src"], f1_r, args, device, seed, logger)
    return rows, panel


# ═══════════════════════════════════════════════════════════════════════════════
# 二b、B3 路线面板（只在 cnn 运行时开启：--route_panel）
# ═══════════════════════════════════════════════════════════════════════════════
Episode = tuple[list[NDArray], NDArray]


def kshot_episodes(labels: NDArray, classes: list[int], k: int, reps: int, seed: int) -> list[Episode]:
    """与 cdl.eval_kshot 逐次相同的抽样：播种一次，每次重复逐类洗牌、取前 k 个作支持集，其余作查询集；
    某类不足 k+2 个时该次重复作废（与 eval_kshot 相同地消耗随机数）。"""
    np.random.seed(seed)
    out: list[Episode] = []
    for _ in range(reps):
        sup: list[NDArray] = []
        ok = True
        for cls in classes:
            idx = np.where(labels == cls)[0].copy()
            np.random.shuffle(idx)
            if len(idx) < k + 2:
                ok = False
                break
            sup.append(idx[:k])
        if not ok:
            continue
        used = set(np.concatenate(sup).tolist())
        q = np.array([i for i in range(len(labels)) if i not in used and labels[i] in classes], dtype=int)
        if len(q):
            out.append((sup, q))
    return out


def tune_episodes(
    labels: NDArray, classes: list[int], k: int, n: int, seed: int, tag: int
) -> tuple[list[int], list[Episode]]:
    """选参情节：独立随机流 default_rng([种子, 74, tag, j])；只用本域里样本数 ≥ k+2 的类，少于 2 类时返回空表。"""
    present = [c for c in classes if int((labels == c).sum()) >= k + 2]
    if len(present) < 2:
        return present, []
    eps: list[Episode] = []
    for j in range(n):
        rng = np.random.default_rng([seed, 74, tag, j])
        sup = [rng.permutation(np.where(labels == c)[0])[:k] for c in present]
        q = np.setdiff1d(np.where(np.isin(labels, present))[0], np.concatenate(sup))
        eps.append((sup, q))
    return present, eps


def nc_pred(x: NDArray, classes: list[int], sup: list[NDArray], q: NDArray) -> NDArray:
    """最近质心（与 20 号相同：SNV 光谱的类中心与查询各自 L2 归一化，取余弦相似度最大者）。"""
    cm = np.stack([x[s].mean(0) for s in sup])
    cm = cm / (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-8)
    xq = x[q] / (np.linalg.norm(x[q], axis=1, keepdims=True) + 1e-8)
    out: NDArray = np.asarray(classes)[(xq @ cm.T).argmax(1)]
    return out


def sk_f1(kind: str, param: Any, x: NDArray, y: NDArray, classes: list[int], ep: Episode) -> float:
    """支持集上拟合 SVM 或 PLS-DA（71 号 fit_predict），在查询集上算宏 F1；拟合失败时退回最近质心（同 71 号）。"""
    sup, q = ep
    x_tr = x[np.concatenate(sup)]
    y_tr = np.concatenate([np.full(len(s), i) for i, s in enumerate(sup)])
    try:
        pred = np.asarray(m71.fit_predict(kind, param, list(classes), x_tr, y_tr, x[q]))
    except Exception:
        pred = nc_pred(x, classes, sup, q)
    return float(f1_score(y[q], pred, average="macro", labels=classes))


def proto_f1(enc: nn.Module, x: NDArray, y: NDArray, classes: list[int], ep: Episode, device: torch.device) -> float:
    """原型网络：逐类编码支持集求原型，与 cdl.eval_kshot 的计算完全相同。"""
    sup, q = ep
    with torch.no_grad():
        protos = []
        for s in sup:
            feat = enc(torch.from_numpy(x[s]).float().to(device)).cpu().numpy().mean(0)
            protos.append(feat / (np.linalg.norm(feat) + 1e-8))
        fq = enc(torch.from_numpy(x[q]).float().to(device)).cpu().numpy()
    pred = np.asarray(classes)[(fq @ np.stack(protos).T).argmax(1)]
    return float(f1_score(y[q], pred, average="macro", labels=classes))


def select(
    kind: str, grid: tuple[Any, ...], pools: list[tuple[NDArray, NDArray, list[int], list[Episode]]]
) -> tuple[Any, float]:
    """在给定域的选参情节上取宏 F1 均值最高的候选（并列取靠前者，同 71 号 pick_best）。"""
    tune = {p: float(np.mean([sk_f1(kind, p, x, y, cls, ep) for x, y, cls, eps in pools for ep in eps])) for p in grid}
    best = m71.pick_best(grid, tune)
    return best, tune[best]


def route_panel(
    fa: cdl.FoldArrays,
    dom_of_row: NDArray,
    held: str,
    enc_r: nn.Module,
    enc_s: nn.Module,
    f1_r_k5: float,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    logger: eu.Logger,
) -> list[dict[str, Any]]:
    """同一批情节上的全部路线；超参数分别在源域（各源域轮流作伪目标）与留出域上选。"""
    enc_r.eval()
    enc_s.eval()
    dom_src = dom_of_row[dom_of_row != held]
    rows: list[dict[str, Any]] = []
    for k in [int(v) for v in args.route_k.split(",")]:
        if any(int((fa.yh == c).sum()) < k + 2 for c in fa.classes):
            logger.log(f"      K={k}：留出域有类不足 K+2 个样本，不评")
            continue
        eps = kshot_episodes(fa.yh, fa.classes, k, args.reps, seed)
        src_pools = []
        for tag, d in enumerate(sorted(set(dom_src.tolist())), start=1):
            m = dom_src == d
            present, teps = tune_episodes(fa.ys[m], fa.classes, k, args.tune_episodes, seed, tag)
            if teps:
                src_pools.append((fa.xs[m], fa.ys[m], present, teps))
        present_t, teps_t = tune_episodes(fa.yh, fa.classes, k, args.tune_episodes, seed, 0)
        tgt_pools = [(fa.xh, fa.yh, present_t, teps_t)]
        scores: dict[str, list[float]] = {r: [] for r in dict.fromkeys((*ROUTES_FREE, *ROUTES_LEAK))}
        chosen: dict[str, str] = {}
        params: dict[str, Any] = {}
        for name, (kind, grid, fixed) in SK_KINDS.items():
            if src_pools:
                best_s, tune_s = select(kind, grid, src_pools)
            else:
                best_s, tune_s = fixed, float("nan")
            best_t, tune_t = select(kind, grid, tgt_pools)
            params[f"{name}(源域选参)"] = best_s
            params[f"{name}(留出域选参)"] = best_t
            chosen[f"{name}(源域选参)"] = (
                f"{best_s}（选参宏F1 {tune_s:.4f}）" if src_pools else f"{best_s}（源域不足，取固定值）"
            )
            chosen[f"{name}(留出域选参)"] = f"{best_t}（选参宏F1 {tune_t:.4f}）"
        for ep in eps:
            scores["原型网络(R)"].append(proto_f1(enc_r, fa.xh, fa.yh, fa.classes, ep, device))
            scores["原型网络(S_src)"].append(proto_f1(enc_s, fa.xh, fa.yh, fa.classes, ep, device))
            sup, q = ep
            scores["最近质心"].append(
                float(f1_score(fa.yh[q], nc_pred(fa.xh, fa.classes, sup, q), average="macro", labels=fa.classes))
            )
            for name, (kind, _, _) in SK_KINDS.items():
                f_s = sk_f1(kind, params[f"{name}(源域选参)"], fa.xh, fa.yh, fa.classes, ep)
                same = str(params[f"{name}(留出域选参)"]) == str(params[f"{name}(源域选参)"])
                f_t = f_s if same else sk_f1(kind, params[f"{name}(留出域选参)"], fa.xh, fa.yh, fa.classes, ep)
                scores[f"{name}(源域选参)"].append(f_s)
                scores[f"{name}(留出域选参)"].append(f_t)
        if k == args.k:
            gap = abs(float(np.mean(scores["原型网络(R)"])) - f1_r_k5)
            if gap > 1e-9:
                raise RuntimeError(f"留出 {held}：面板里的原型网络 R 与 B1 的 R 不一致（差 {gap:.2e}）")
        for route, vals in scores.items():
            rows.append(
                {
                    "held": held,
                    "种子": seed,
                    "K": k,
                    "路线": route,
                    "F1": float(np.mean(vals)),
                    "情节数": len(vals),
                    "所选超参数": chosen.get(route, ""),
                    "源域选参可用": bool(src_pools),
                }
            )
        logger.log(f"      K={k}：" + "  ".join(f"{r}={np.mean(v):.3f}" for r, v in scores.items()))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 三、自检与汇总
# ═══════════════════════════════════════════════════════════════════════════════
def seed_files(stage: str = "formal") -> list[str]:
    files = sorted(glob.glob(os.path.join(SEED_DIR, "74_multi_encoder__*__seed*.xlsx")))
    return [f for f in files if ("__pilot" in f) == (stage == "pilot")]


def load_long(stage: str = "formal") -> pd.DataFrame:
    files = seed_files(stage)
    if not files:
        raise FileNotFoundError(f"没有 74 号的逐种子结果（{stage}）")
    return pd.concat([pd.read_excel(f, sheet_name=0) for f in files], ignore_index=True)


def selfcheck(logger: eu.Logger) -> None:
    """cnn 的 R、S_src 与 24 号正式长格式逐格比对。"""
    ref = pd.read_excel(REF_24, sheet_name="表d：长格式")
    mine = load_long("formal")
    mine = mine[(mine["编码器"] == "cnn") & (mine["评估样本"] == "全部")]
    diffs: list[float] = []
    missing = 0
    for _, r in ref.iterrows():
        m = mine[(mine["dataset"] == r["dataset"]) & (mine["held"] == r["held"]) & (mine["种子"] == r["seed"])]
        f = {str(a): float(v) for a, v in zip(m["臂"], m["F1"], strict=True)}
        if "R" not in f or "S_src" not in f:
            missing += 1
            continue
        diffs += [abs(f["S_src"] - float(r["shared"])), abs(f["R"] - float(r["fold_restricted"]))]
    if not diffs:
        raise RuntimeError("没有可比对的格：cnn 的正式结果还没有产出")
    worst = max(diffs)
    logger.log(f"自检：与 24 号正式结果比对 {len(diffs)} 个值，最大绝对差 {worst:.2e}，缺格 {missing}")
    if worst > 1e-6:
        raise RuntimeError(f"cnn 的 R/S_src 与 24 号不一致（最大差 {worst:.2e}）：代码路径或运行环境已变")


def _t_interval(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float("nan"), float("nan")
    half = float(stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr)))
    return float(arr.mean() - half), float(arr.mean() + half)


def aggregate(logger: eu.Logger) -> None:
    long_df = load_long("formal")
    long_df = long_df.merge(
        pd.DataFrame(
            [{"dataset": k, "drift_axis": v["drift"], "modality": v["modality"]} for k, v in DATASETS.items()]
        ),
        on="dataset",
        how="left",
    )
    seeds = sorted(long_df["种子"].unique().tolist())
    if set(seeds) != set(eu.RECOMMENDED_SEEDS):
        logger.log(f"  [WARN] 种子集合 {seeds} 不是 Formal 5 种子，汇总仅作进度查看")
    rows: list[dict[str, Any]] = []
    for (ds, held, enc), sub in long_df.groupby(["dataset", "held", "编码器"]):
        f1 = {(str(a), int(s)): float(v) for a, s, v in zip(sub["臂"], sub["种子"], sub["F1"], strict=True)}
        present = sorted({a for a, _ in f1}, key=lambda a: EVAL_ARMS.index(a))
        row: dict[str, Any] = {"dataset": ds, "held": held, "编码器": enc, "种子数": sub["种子"].nunique()}
        for arm in present:
            vals = [f1[(arm, s)] for s in seeds if (arm, s) in f1]
            row[arm] = round(float(np.mean(vals)), 4)
        for name, (a, b) in CONTRASTS.items():
            dl = [f1[(a, s)] - f1[(b, s)] for s in seeds if (a, s) in f1 and (b, s) in f1]
            if not dl:
                continue
            lo, hi = _t_interval(dl)
            row[name] = round(float(np.mean(dl)), 4)
            row[f"{name}_种子t区间"] = f"{lo:.4f}–{hi:.4f}"
            row[f"{name}_为正种子数"] = int(sum(d > 0 for d in dl))
        rows.append(row)
    fold_df = pd.DataFrame(rows)
    ds_rows: list[dict[str, Any]] = []
    for (ds, enc), sub in fold_df.groupby(["dataset", "编码器"]):
        drow: dict[str, Any] = {"dataset": ds, "编码器": enc, "留出域数": len(sub)}
        for name in CONTRASTS:
            if name in sub:
                v = sub[name].dropna()
                drow[f"{name}_均值"] = round(float(v.mean()), 4) if len(v) else float("nan")
                drow[f"{name}_为正留出域"] = f"{int((v > 0).sum())}/{len(v)}"
        ds_rows.append(drow)
    ds_df = pd.DataFrame(ds_rows)
    logger.log("\n" + ds_df.to_string(index=False))
    out = os.path.join(eu.OUTPUT_DIR, "74_multi_encoder_leakage.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        fold_df.to_excel(writer, sheet_name=eu.table_sheet_name(0, "逐折对照"), index=False)
        ds_df.to_excel(writer, sheet_name=eu.table_sheet_name(1, "按数据集汇总"), index=False)
        long_df.to_excel(writer, sheet_name=eu.table_sheet_name(2, "长格式"), index=False)
    logger.log(f"汇总表 → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 四、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="74_multi_encoder_leakage：公开数据集上的多编码器泄漏对照")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--encoder", default="cnn", choices=list(enc_lib.ENCODERS))
    parser.add_argument("--datasets", default=",".join(DATASETS), help="逗号分隔的数据集名")
    parser.add_argument("--data_root", default="05data")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--lin_dim", type=int, default=16, help="线性族（pls / pca）的成分数上限")
    parser.add_argument("--route_panel", action="store_true", help="B3 路线面板（只用于 cnn）")
    parser.add_argument("--route_k", default=",".join(str(k) for k in ROUTE_K))
    parser.add_argument("--tune_episodes", type=int, default=m71.TUNE_EPISODES, help="每个域的选参情节数")
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    logger = eu.get_logger("74_multi_encoder_leakage")
    if args.selfcheck or args.aggregate:
        if args.selfcheck:
            selfcheck(logger)
        if args.aggregate:
            aggregate(logger)
        logger.close()
        return

    if args.route_panel and args.encoder != "cnn":
        raise SystemExit("B3 路线面板只用于 cnn（与正文方法比较所用的原型网络编码器相同）")
    ds_list = [d for d in args.datasets.split(",") if d]
    unknown = sorted(set(ds_list) - set(DATASETS))
    if unknown:
        raise SystemExit(f"未知数据集：{unknown}")
    device = eu.get_device(args.device)
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "公开数据集上的多编码器泄漏对照",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "编码器": f"{args.encoder}（{enc_lib.ARCH_DESC[args.encoder]}）",
            "数据集": ds_list,
            "臂": [a for a in ARMS if not (args.encoder == "pca" and a == "S_shuf")],
            "K / 元训练情节 / 每折评估重复": f"{args.k} / {args.episodes} / {args.reps}",
            "线性族成分数上限": args.lin_dim,
            "B3 路线面板": f"K={args.route_k}，每域选参情节 {args.tune_episodes}" if args.route_panel else "不跑",
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))
    rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    with eu.track_compute("多编码器泄漏", logger, run) as budget:
        for dsname in ds_list:
            meta = DATASETS[dsname]
            loader: Callable[[str, int], cdl.LoaderResult] = meta["loader"]
            doms, is_reg = loader(args.data_root, args.seed)
            logger.log(f"\n【{dsname}】{meta['drift']}；域 {[f'{d}:{len(v[1])}' for d, v in doms.items()]}")
            for held in list(doms.keys()):
                arm_rows, fold_panel = run_fold(doms, is_reg, held, args.encoder, args, device, args.seed, logger)
                rows += [{"dataset": dsname, **r} for r in arm_rows]
                panel_rows += [{"dataset": dsname, **r} for r in fold_panel]
    res = pd.DataFrame(rows)
    suffix = "" if args.stage == "formal" else "__pilot"
    os.makedirs(SEED_DIR, exist_ok=True)
    out = os.path.join(SEED_DIR, f"74_multi_encoder__{args.encoder}__seed{args.seed}{suffix}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        res.to_excel(writer, sheet_name=eu.table_sheet_name(0, "逐臂结果"), index=False)
        if panel_rows:
            pd.DataFrame(panel_rows).to_excel(writer, sheet_name=eu.table_sheet_name(1, "路线面板"), index=False)
        pd.DataFrame([{"seed": args.seed, "stage": args.stage, "encoder": args.encoder, **budget}]).to_excel(
            writer, sheet_name="compute", index=False
        )
    logger.log(f"\n结果表 → {out}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
