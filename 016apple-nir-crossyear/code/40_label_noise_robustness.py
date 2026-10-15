"""40_label_noise_robustness.py: 训练标签噪声鲁棒性曲线（L3-2）。

要回答的问题:
    三年的产地标签全部来自商户声明，无化学金标准。审稿人质疑「标签不可靠」时，本脚本
    量化的不是「标签是对的」，而是「**标签要多不可靠，本文的结论才会被推翻**」。

    做法：把**训练标签**按翻转率 ε 随机破坏后重跑 LOEO-Year 三折 K-shot 主实验，
    观察两条核心结论各自在哪一个 ε 上失守。**查询（测试）标签始终保持声明产地不动**，
    故本曲线量化的是「结论对训练标签正确性的依赖程度」；测试标签本身也可能错这一层，
    本数据无法用任何实验消除（见报告 §2.6 边界声明）。

预设判据（**先写死再看结果**，禁止事后调整）:
    C1 主结果：三折合并宏 F1 的 cluster bootstrap 95% CI **下界 > 1/3**（优于三类随机）。
        辅助幅度判据 C1m：相对 ε=0 的宏 F1 跌幅 < 0.05。
    C2 泄漏结论（本文最核心的主张）：共享 − 折限定的**严格配对** Δ，其 95% CI 下界
        **> 0.01**（预注册的宏 F1 等价区间），且 Holm 校正后 P < 0.05。
    断点 x = 满足「**所有** ε' ≤ x 都成立」的最大 ε（前缀规则，比「最后一个成立点」保守）。

噪声模型（预设）:
    - 翻转单位 = **苹果**（apple_id），同果全部面位共享标签；面位级翻转不对应任何真实
      场景（商户按批次报产地，不会同一个果的不同面位报不同产地）。
    - 翻转目的地限定在**同年份已出现的产地**内，故每折标签空间不变（LOEO-Year 的
      「每折标签空间随年份而变」是协议前提，不能被噪声注入改掉）。
    - 每个苹果只抽一次 u~U(0,1) 与一次目的地，翻转集合按 ε **嵌套**（ε=5% ⊂ ε=10% ⊂ …），
      即共同随机数方差缩减；这让曲线的单调性反映噪声本身而非重抽样噪声。
    - 两种翻转口径：`uniform`（全类均匀翻）与 `sd_only`（只翻声明山东的果，对应
      「山东标签可能有问题」这一具体质疑）。

阶段与硬件 (CRITICAL):
    <MPS_PILOT> —— 本脚本全程 CPU，编码器为**本机重训**、固定 episode 预算、不早停，
    与稿件正式 CUDA 编码器不同源（同 29INSTRUMENT_LABEL_DECOMP.md §1.5 的复现落差）。
    其**绝对数值不得写入正文主结果**；可用的是同一口径下 ε 之间的**相对变化**。

运行方式:
    cd <project root>
    # 扫描（可分片并行、可断点续跑，每次自限时长）
    for i in 0 1 2 3 4; do KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
        python 02code/40_label_noise_robustness.py --phase sweep --shard $i --nshards 5 --max_seconds 480 & done; wait
    # 汇总出表出图
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/40_label_noise_robustness.py --phase aggregate

输出文件:
    04outputs/40_label_noise_robustness-图a.pdf / -fig-a.pdf  — 宏F1~ε（两翻转口径×三折+合并）
    04outputs/40_label_noise_robustness-图b.pdf / -fig-b.pdf  — 配对泄漏Δ~ε 与 95%CI
    04outputs/40_label_noise_robustness-图c.pdf / -fig-c.pdf  — 宏F1~ε 随 K 的变化
    04outputs/40_label_noise_robustness.xlsx                   — 表a–表f + 算力
    04outputs/40_label_noise_cache/                            — 逐单元断点缓存（可删，删后重跑）
    07checkpoints/40_label_noise_ref_eps0_fold2025/            — 参照编码器权重 + 模型卡（F1）
    05logs/40_label_noise_robustness_YY-MM-DD_HHMMSS.log       — 运行日志
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

import export_utils as eu
from protonet import EpisodeSampler, ProtoEncoder, encode_batch, proto_loss

NDArray = npt.NDArray[Any]

# ── 预设网格（改动即违反「先写死再看结果」）─────────────────────────────────────
FLIP_MODES = ("uniform", "sd_only")
EPS_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
FOLDS = (2018, 2019, 2025)
SRC_YEARS: dict[int, list[int]] = {2018: [2019, 2025], 2019: [2018, 2025], 2025: [2018, 2019]}
YEAR_CLASSES: dict[int, list[str]] = {2018: ["SD", "SX", "XJ"], 2019: ["SD", "SX", "XJ"], 2025: ["GS", "SD", "XJ"]}
ALPHAS = (0.0, 0.4)
K_LIST = (1, 3, 5, 10, 20, 50)

# ── 预设判据 ────────────────────────────────────────────────────────────────
PRIMARY_K = 5
PRIMARY_ALPHA = 0.0  # 稿件 2.4 节判定的主实验最优配置（不融合）
CHANCE = 1.0 / 3.0
EQUIV_MARGIN = 0.01  # 预注册等价区间 ε(Macro-F1)=0.01（01PROJECT_PROTOCOL 第二部分 §六）
MAG_MARGIN = 0.05  # C1m 实质性跌幅阈值
N_BOOT = 2000
ALPHA_LEVEL = 0.05

CACHE_DIR = os.path.join(eu.OUTPUT_DIR, "40_label_noise_cache")
C_FOLD = {2018: "#0072B2", 2019: "#009E73", 2025: "#D55E00"}
C_POOL = "#8E44AD"
C_MODE = {"uniform": "#0B5394", "sd_only": "#C1272D"}


# ═══════════════════════════════════════════════════════════════════════════════
# 一、数据与噪声
# ═══════════════════════════════════════════════════════════════════════════════
class Bench:
    """全脚本共用的苹果粒度索引（只读，构建一次）。"""

    def __init__(self) -> None:
        df = eu.load_all_elements_df(verify_sha256=True)
        self.wl_cols = eu.get_spectrum_columns(df)
        self.n_bands = len(self.wl_cols)
        x_face = df[self.wl_cols].to_numpy().astype(np.float32)
        aid = df["苹果编号"].to_numpy()
        uniq, inv = np.unique(aid, return_inverse=True)
        self.apple_ids: NDArray = uniq
        self.face_to_apple: NDArray = inv
        self.n_apple = len(uniq)
        self.x_face = x_face
        first = {int(i): int(np.where(inv == i)[0][0]) for i in range(self.n_apple)}
        self.year: NDArray = np.array([int(df["年份"].to_numpy()[first[i]]) for i in range(self.n_apple)])
        self.declared: NDArray = np.array([str(df["产地"].to_numpy()[first[i]]) for i in range(self.n_apple)])
        # 每果的面位光谱（EpisodeSampler 需要 (n_faces, bands)）
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(self.n_apple + 1))
        self.apple_x: list[NDArray] = [x_face[order[bounds[i] : bounds[i + 1]]] for i in range(self.n_apple)]
        self.n_face = len(x_face)

    def apple_features(self, face_feats: NDArray) -> NDArray:
        """面位特征 → 苹果特征（全面位均值后 L2 归一化），与 protonet.apple_feature 同式。"""
        dim = face_feats.shape[1]
        acc = np.zeros((self.n_apple, dim), dtype=np.float64)
        cnt = np.zeros(self.n_apple, dtype=np.float64)
        np.add.at(acc, self.face_to_apple, face_feats.astype(np.float64))
        np.add.at(cnt, self.face_to_apple, 1.0)
        mean = acc / cnt[:, None]
        return (mean / (np.linalg.norm(mean, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def make_noisy_labels(bench: Bench, mode: str, eps: float, seed: int) -> tuple[NDArray, NDArray]:
    """按预设噪声模型生成含噪训练标签；返回 (noisy_labels, flipped_mask)。

    共同随机数：u 与目的地只依赖 (mode, seed, 苹果)，不依赖 eps，故翻转集合按 eps 嵌套。
    """
    rng = np.random.default_rng([int(seed), FLIP_MODES.index(mode), 40])
    u = rng.random(bench.n_apple)
    alt = rng.integers(0, 2, bench.n_apple)
    noisy = bench.declared.copy()
    if eps <= 0:
        return noisy, np.zeros(bench.n_apple, dtype=bool)
    eligible = np.ones(bench.n_apple, dtype=bool) if mode == "uniform" else (bench.declared == "SD")
    flipped = eligible & (u < eps)
    for i in np.where(flipped)[0]:
        others = [c for c in YEAR_CLASSES[int(bench.year[i])] if c != bench.declared[i]]
        noisy[i] = others[int(alt[i]) % len(others)]
    return noisy, flipped


# ═══════════════════════════════════════════════════════════════════════════════
# 二、编码器（固定 episode 预算、不早停）
# ═══════════════════════════════════════════════════════════════════════════════
def _sampler_payload(bench: Bench, idx: NDArray, labels: NDArray) -> dict[str, dict[Any, dict[str, Any]]]:
    """把苹果子集包成 EpisodeSampler 认的 apple_data 结构（伪元素键=年份_含噪标签）。"""
    payload: dict[str, dict[Any, dict[str, Any]]] = {}
    for i in idx:
        key = f"{int(bench.year[i])}_{labels[i]}"
        payload.setdefault(key, {})[int(i)] = {"X": bench.apple_x[int(i)], "region": str(labels[i])}
    return payload


def train_encoder(
    bench: Bench,
    train_idx: NDArray,
    labels: NDArray,
    seed: int,
    episodes: int,
    device: torch.device,
) -> tuple[ProtoEncoder, float]:
    """在给定苹果子集与（含噪）标签上训练 ProtoNet 编码器；返回 (编码器, 源域验证Acc)。

    固定 episode 预算、取末次权重：早停会让「标签越噪停得越早」成为混杂，破坏 ε 之间的可比性。
    """
    eu.set_random_seed(seed)
    payload = _sampler_payload(bench, train_idx, labels)
    sampler = EpisodeSampler(list(payload.keys()), payload, 3, 5, 10)
    enc = ProtoEncoder(bench.n_bands, feat_dim=128).to(device)
    opt = optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=episodes, eta_min=1e-5)
    for _ in range(episodes):
        enc.train()
        sx, sy, qx, qy, nw = sampler.sample()
        opt.zero_grad()
        loss, _ = proto_loss(enc, sx, sy, qx, qy, nw, device)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(enc.parameters(), max_norm=5.0)
        opt.step()
        sched.step()
    enc.eval()
    accs: list[float] = []
    with torch.no_grad():
        for _ in range(30):
            sx, sy, qx, qy, nw = sampler.sample()
            _, a = proto_loss(enc, sx, sy, qx, qy, nw, device)
            accs.append(a)
    return enc, float(np.mean(accs))


# ═══════════════════════════════════════════════════════════════════════════════
# 三、K-shot 评估（两臂共用同一支持集，严格配对）
# ═══════════════════════════════════════════════════════════════════════════════
def draw_supports(bench: Bench, fold: int, noisy: NDArray, k: int, rng: np.random.Generator) -> dict[str, NDArray]:
    """按含噪标签在目标年份内抽 K-shot 支持集（苹果粒度）。两臂共用，实现严格配对。"""
    tgt = bench.year == fold
    out: dict[str, NDArray] = {}
    for g in YEAR_CLASSES[fold]:
        cand = np.where(tgt & (noisy == g))[0]
        if cand.size == 0:
            out[g] = np.array([], dtype=int)
            continue
        kk = min(k, max(1, cand.size - 1))
        out[g] = rng.choice(cand, size=kk, replace=False)
    return out


def source_protos(bench: Bench, fold: int, noisy: NDArray, feats: NDArray) -> dict[str, NDArray]:
    """源域同类原型：先按源年份各算一个再平均（与 06/10 脚本同式）。"""
    out: dict[str, NDArray] = {}
    for g in YEAR_CLASSES[fold]:
        per_year: list[NDArray] = []
        for sy in SRC_YEARS[fold]:
            sel = np.where((bench.year == sy) & (noisy == g))[0]
            if sel.size == 0:
                continue
            p = feats[sel].mean(axis=0)
            per_year.append(p / (float(np.linalg.norm(p)) + 1e-8))
        if per_year:
            p = np.mean(per_year, axis=0)
            out[g] = p / (float(np.linalg.norm(p)) + 1e-8)
    return out


def macro_f1_int(y_true: NDArray, y_pred: NDArray, n_cls: int) -> float:
    """整数标签的宏 F1（等价于 sklearn f1_score(labels=全类, average='macro', zero_division=0)）。

    自实现是为了速度：本脚本一次汇总要跑十万量级的情节，sklearn 的字符串标签路径是瓶颈。
    与 sklearn 的一致性由 `--phase selftest` 逐随机例核对。
    """
    conf = np.bincount(y_true * n_cls + y_pred, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    tp = np.diag(conf).astype(float)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


class FoldView:
    """单 (臂, 折) 的评估视图：目标年苹果特征、整数真值、源域原型，只算一次。"""

    def __init__(self, bench: Bench, fold: int, feats: NDArray, noisy: NDArray) -> None:
        self.fold = fold
        self.regs = YEAR_CLASSES[fold]
        self.tgt_idx = np.where(bench.year == fold)[0]
        self.feats_tgt = np.ascontiguousarray(feats[self.tgt_idx])
        reg_to_int = {g: i for i, g in enumerate(self.regs)}
        self.y_true = np.array([reg_to_int[str(s)] for s in bench.declared[self.tgt_idx]], dtype=np.int64)
        self.pos_of = np.full(bench.n_apple, -1, dtype=np.int64)
        self.pos_of[self.tgt_idx] = np.arange(self.tgt_idx.size)
        self.src_p = source_protos(bench, fold, noisy, feats)
        self.dim = feats.shape[1]

    def episode_f1(self, support: dict[str, NDArray], alpha: float) -> float:
        """单情节宏 F1：查询=支持集之外的全部目标年苹果，真值取**声明产地**（未翻转）。"""
        keep = np.ones(self.tgt_idx.size, dtype=bool)
        protos = np.zeros((len(self.regs), self.dim), dtype=np.float32)
        for j, g in enumerate(self.regs):
            sup = support[g]
            if sup.size:
                keep[self.pos_of[sup]] = False
                p = self.feats_tgt[self.pos_of[sup]].mean(axis=0)
                p = p / (float(np.linalg.norm(p)) + 1e-8)
            else:  # 该类被噪声掏空（实测未发生）：退化为源域原型
                p = self.src_p.get(g, np.zeros(self.dim, dtype=np.float32))
            if alpha > 0 and g in self.src_p:
                p = (1 - alpha) * p + alpha * self.src_p[g]
                p = p / (float(np.linalg.norm(p)) + 1e-8)
            protos[j] = p
        if not keep.any():
            keep[:] = True
        sim = self.feats_tgt[keep] @ protos.T
        return macro_f1_int(self.y_true[keep], sim.argmax(axis=1).astype(np.int64), len(self.regs))


# ═══════════════════════════════════════════════════════════════════════════════
# 四、扫描（断点续跑）
# ═══════════════════════════════════════════════════════════════════════════════
ARMS = ("shared", "r2018", "r2019", "r2025")


def unit_cases() -> list[tuple[str, float, int, str]]:
    """全部 (翻转口径, ε, 种子, 臂) 单元；ε=0 与口径无关，只保留 uniform 一份。

    以「一个编码器」为缓存粒度：两臂共用的支持集只由 (seed, fold, K, rep) 决定，
    与臂无关，故拆开跑完再对齐仍是**严格配对**，且断点续跑不会整块丢失。
    """
    out: list[tuple[str, float, int, str]] = []
    for seed in eu.RECOMMENDED_SEEDS:
        for mode in FLIP_MODES:
            for eps in EPS_GRID:
                if eps == 0.0 and mode != FLIP_MODES[0]:
                    continue
                for arm in ARMS:
                    out.append((mode, float(eps), int(seed), arm))
    return out


def unit_path(mode: str, eps: float, seed: int, arm: str) -> str:
    return os.path.join(CACHE_DIR, f"{mode}_eps{round(eps * 100):03d}_seed{seed}_{arm}.json")


def run_unit(
    bench: Bench, mode: str, eps: float, seed: int, arm: str, args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    """跑一个 (口径, ε, 种子, 臂) 单元：训 1 个编码器 + 评该臂负责的折。"""
    noisy, flipped = make_noisy_labels(bench, mode, eps, seed)
    if arm == "shared":
        e9 = (bench.year == 2025) & (bench.declared == "GS")  # 共享臂固定训 E1–E8（稿件 1.2）
        train_idx = np.where(~e9)[0]
        enc_seed = seed
        eval_folds = list(FOLDS)
    else:
        fold = int(arm[1:])
        train_idx = np.where(np.isin(bench.year, SRC_YEARS[fold]))[0]
        enc_seed = seed + fold
        eval_folds = [fold]
    enc, val_acc = train_encoder(bench, train_idx, noisy, enc_seed, args.episodes, device)
    feats = bench.apple_features(encode_batch(enc, bench.x_face, device))

    records: list[dict[str, Any]] = []
    for fold in eval_folds:
        view = FoldView(bench, fold, feats, noisy)
        for k in K_LIST:
            for rep in range(args.reps):
                rng = np.random.default_rng([int(seed), fold, k, rep, 41])
                sup = draw_supports(bench, fold, noisy, k, rng)
                for alpha in ALPHAS:
                    records.append(
                        {"fold": fold, "k": k, "alpha": alpha, "rep": rep, "f1": view.episode_f1(sup, alpha)}
                    )
    return {
        "mode": mode,
        "eps": eps,
        "seed": seed,
        "arm": arm,
        "episodes": args.episodes,
        "reps": args.reps,
        "n_flipped": int(flipped.sum()),
        "n_flipped_by_declared": {c: int(((bench.declared == c) & flipped).sum()) for c in ("XJ", "SD", "SX", "GS")},
        "val_acc": val_acc,
        "n_train": int(train_idx.size),
        "n_test": {str(f): int((bench.year == f).sum()) for f in eval_folds},
        "records": records,
    }


def phase_sweep(bench: Bench, args: argparse.Namespace, device: torch.device, logger: eu.Logger) -> bool:
    """跑本分片未完成的单元，超 --max_seconds 即干净退出。返回是否全部完成。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cases = unit_cases()
    mine = [c for i, c in enumerate(cases) if i % args.nshards == args.shard]
    t0 = time.time()
    for mode, eps, seed, arm in mine:
        path = unit_path(mode, eps, seed, arm)
        if os.path.exists(path):
            continue
        if time.time() - t0 > args.max_seconds:
            logger.log(f"[分片{args.shard}] 时间预算用尽，剩余单元留待下次续跑")
            break
        c0 = time.time()
        unit = run_unit(bench, mode, eps, seed, arm, args, device)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(unit, f)
        os.replace(tmp, path)  # 原子落盘：半截 json 会让后续汇总静默读到坏数据
        logger.log(
            f"[分片{args.shard}] {mode} ε={eps:.2f} seed={seed} {arm} 完成 "
            f"({time.time() - c0:.0f}s, 翻转 {unit['n_flipped']} 果, 源域验证Acc={unit['val_acc']:.3f})"
        )
    remaining = [c for c in mine if not os.path.exists(unit_path(*c))]
    logger.log(f"[分片{args.shard}] 本片 {len(mine)} 单元，尚缺 {len(remaining)}")
    return not remaining


# ═══════════════════════════════════════════════════════════════════════════════
# 五、汇总与统计
# ═══════════════════════════════════════════════════════════════════════════════
def load_records() -> pd.DataFrame:
    """读全部缓存单元 → 逐情节配对宽表；ε=0 单元同时充当两个翻转口径的共同起点。"""
    rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for fn in sorted(os.listdir(CACHE_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(CACHE_DIR, fn), encoding="utf-8") as f:
            unit = json.load(f)
        modes = list(FLIP_MODES) if unit["eps"] == 0.0 else [unit["mode"]]
        arm_col = "f1_shared" if unit["arm"] == "shared" else "f1_restricted"
        for m in modes:
            for r in unit["records"]:
                rows.append(
                    {
                        "mode": m,
                        "eps": unit["eps"],
                        "seed": unit["seed"],
                        "fold": r["fold"],
                        "k": r["k"],
                        "alpha": r["alpha"],
                        "rep": r["rep"],
                        arm_col: r["f1"],
                    }
                )
        meta.append(
            {
                "mode": unit["mode"],
                "eps": unit["eps"],
                "seed": unit["seed"],
                "arm": unit["arm"],
                "episodes": unit["episodes"],
                "reps": unit["reps"],
                "n_flipped": unit["n_flipped"],
                "val_acc": round(unit["val_acc"], 4),
                "n_train": unit["n_train"],
                **{f"flip_{k}": v for k, v in unit["n_flipped_by_declared"].items()},
                **{f"ntest_{k}": v for k, v in unit["n_test"].items()},
            }
        )
    keys = ["mode", "eps", "seed", "fold", "k", "alpha", "rep"]
    df = pd.DataFrame(rows).groupby(keys, as_index=False).first()
    missing = df["f1_shared"].isna() | df["f1_restricted"].isna()
    if missing.any():
        raise RuntimeError(f"有 {int(missing.sum())} 个情节只拿到单臂结果——配对不完整，拒绝汇总。")
    meta_df = pd.DataFrame(meta).sort_values(["mode", "eps", "seed", "arm"])
    for col in ("episodes", "reps"):
        vals = sorted(meta_df[col].unique().tolist())
        if len(vals) > 1:
            raise RuntimeError(
                f"缓存里混进了不同 {col} 的单元（{vals}）——跨预算的曲线不可比。"
                f"请清空 {CACHE_DIR} 后按同一预算重跑。"
            )
    df.attrs["meta"] = meta_df
    return df


def _fold_cluster(vals: pd.Series) -> list[int]:
    return [FOLDS.index(int(f)) for f in vals]


def summarise(df: pd.DataFrame, n_train: int, n_test: int) -> pd.DataFrame:
    """逐 (口径, ε, α, K) 出：折限定 F1 及其对随机基线的 CI、配对泄漏 Δ 及其 CI。

    统计口径严格照抄稿件 2.3 节（也是预注册 §六）：
      - **假设检验的主分析单位 = 3 个折**。折内先对 5 种子 × 重复取均值得到 3 个折级值，
        再在这 3 个值上作 Nadeau-Bengio 方差修正的配对 t 检验。**禁止**把
        3 折 × 5 种子 × 重复当独立样本——那是伪重复（预注册 §六「红线」）。
      - **置信区间 = 以折为聚类单位的 cluster bootstrap**，跑在逐情节数据上（同
        `38paired_leakage.py`），2000 次重采样。
    两者输入不同是有意的：CI 要用逐情节数据才估得动情节采样噪声，检验则必须退回 3 个折。
    """
    out: list[dict[str, Any]] = []
    for (mode, eps, alpha, k), sub in df.groupby(["mode", "eps", "alpha", "k"], sort=True):
        clusters = _fold_cluster(sub["fold"])
        ci_chance = eu.compare_methods(
            sub["f1_restricted"].tolist(), [CHANCE] * len(sub),
            clusters=clusters, n_bootstrap=N_BOOT, seed=eu.RECOMMENDED_SEEDS[0],
        )
        ci_leak = eu.compare_methods(
            sub["f1_shared"].tolist(), sub["f1_restricted"].tolist(),
            clusters=clusters, n_bootstrap=N_BOOT, seed=eu.RECOMMENDED_SEEDS[0],
        )
        fold_r = [float(sub[sub["fold"] == f]["f1_restricted"].mean()) for f in FOLDS]
        fold_s = [float(sub[sub["fold"] == f]["f1_shared"].mean()) for f in FOLDS]
        t_chance = eu.compare_methods(
            fold_r, [CHANCE] * len(FOLDS), n_train=n_train, n_test=n_test,
            n_bootstrap=N_BOOT, seed=eu.RECOMMENDED_SEEDS[0],
        )
        t_leak = eu.compare_methods(
            fold_s, fold_r, n_train=n_train, n_test=n_test,
            n_bootstrap=N_BOOT, seed=eu.RECOMMENDED_SEEDS[0],
        )
        row: dict[str, Any] = {
            "翻转口径": mode,
            "ε": eps,
            "α": alpha,
            "K": k,
            "折限定F1": ci_chance["mean_a"],
            "折限定F1_CI下界": ci_chance["ci_low"] + CHANCE,
            "折限定F1_CI上界": ci_chance["ci_high"] + CHANCE,
            "对随机基线P": t_chance["p_value"],
            "共享F1": ci_leak["mean_a"],
            "配对泄漏Δ": ci_leak["mean_diff"],
            "Δ_CI下界": ci_leak["ci_low"],
            "Δ_CI上界": ci_leak["ci_high"],
            "Δ_Cohen_d": t_leak["cohens_d"],
            "Δ_P": t_leak["p_value"],
        }
        for fold, vr, vs in zip(FOLDS, fold_r, fold_s, strict=True):
            row[f"折限定F1_{fold}"] = vr
            row[f"Δ_{fold}"] = vs - vr
        out.append(row)
    res = pd.DataFrame(out)
    # Holm 校正：每个 (口径, α, K) 家族内跨 6 个 ε
    for col_p, col_adj in (("对随机基线P", "对随机基线P_Holm"), ("Δ_P", "Δ_P_Holm")):
        res[col_adj] = np.nan
        for _, idx in res.groupby(["翻转口径", "α", "K"]).groups.items():
            sel = list(idx)
            adj = eu.apply_multi_comp_correction(res.loc[sel, col_p].tolist(), method="holm")["p_adjusted"]
            res.loc[sel, col_adj] = adj
    return res.sort_values(["翻转口径", "α", "K", "ε"]).reset_index(drop=True)


def breakpoints(summary: pd.DataFrame) -> pd.DataFrame:
    """按预设判据求断点 x（前缀规则）。"""
    out: list[dict[str, Any]] = []
    for (mode, alpha, k), sub in summary.groupby(["翻转口径", "α", "K"], sort=True):
        sub = sub.sort_values("ε")
        f1_at0 = float(sub[sub["ε"] == 0.0]["折限定F1"].iloc[0])
        rec: dict[str, Any] = {"翻转口径": mode, "α": alpha, "K": k, "ε=0基准F1": round(f1_at0, 4)}
        for tag, ok_fn in (
            ("C1_优于随机", lambda r: r["折限定F1_CI下界"] > CHANCE),
            ("C1m_跌幅<0.05", lambda r, base=f1_at0: (base - r["折限定F1"]) < MAG_MARGIN),
            (
                "C2_泄漏Δ>等价边界",
                lambda r: (r["Δ_CI下界"] > EQUIV_MARGIN) and (r["Δ_P_Holm"] < ALPHA_LEVEL),
            ),
            # 拆开报：C2 的两个分条件在 n_primary=3 下功效差别极大，合并报会看不出是谁先失守
            ("C2a_仅CI下界>0.01", lambda r: r["Δ_CI下界"] > EQUIV_MARGIN),
            ("C2b_仅Holm后P<0.05", lambda r: r["Δ_P_Holm"] < ALPHA_LEVEL),
        ):
            x = None
            for _, r in sub.iterrows():
                if ok_fn(r):
                    x = float(r["ε"])
                else:
                    break
            rec[tag] = "全网格未失守(≥0.30)" if x == max(EPS_GRID) else (f"{x:.2f}" if x is not None else "ε=0即不成立")
        out.append(rec)
    return pd.DataFrame(out)


# ═══════════════════════════════════════════════════════════════════════════════
# 六、图
# ═══════════════════════════════════════════════════════════════════════════════
TXT = {
    "zh": {
        "eps": "训练标签翻转率 ε",
        "f1": "宏 F1（折限定编码器）",
        "delta": "配对评估泄漏 Δ = 共享 − 折限定",
        "pool": "三折合并",
        "hold": "留出",
        "chance": "三类随机基线 1/3",
        "equiv": "等价边界 0.01",
        "uniform": "全类均匀翻转",
        "sd_only": "只翻声明山东",
        "ta": f"标签噪声下的主结果（K={PRIMARY_K}，α={PRIMARY_ALPHA:g}）",
        "tb": f"标签噪声下的评估泄漏（K={PRIMARY_K}，α={PRIMARY_ALPHA:g}，误差棒为95%CI）",
        "tc": "宏 F1 ~ ε 随支持样本量 K 的变化（三折合并）",
        "k": "K",
    },
    "en": {
        "eps": "Training-label flip rate ε",
        "f1": "Macro-F1 (fold-restricted encoder)",
        "delta": "Paired evaluation leakage Δ = shared − fold-restricted",
        "pool": "Pooled (3 folds)",
        "hold": "hold-out",
        "chance": "Chance (1/3)",
        "equiv": "Equivalence margin 0.01",
        "uniform": "Uniform flip (all origins)",
        "sd_only": "Flip declared-SD only",
        "ta": f"Main result under label noise (K={PRIMARY_K}, α={PRIMARY_ALPHA:g})",
        "tb": f"Evaluation leakage under label noise (K={PRIMARY_K}, α={PRIMARY_ALPHA:g}, 95% CI)",
        "tc": "Macro-F1 vs ε across support size K (pooled)",
        "k": "K",
    },
}


def make_plot_a(summary: pd.DataFrame) -> Any:
    def plot(lang: str) -> Any:
        t = TXT[lang]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
        for ax, mode in zip(axes, FLIP_MODES, strict=False):
            sub = summary[
                (summary["翻转口径"] == mode) & (summary["K"] == PRIMARY_K) & (summary["α"] == PRIMARY_ALPHA)
            ].sort_values("ε")
            x = sub["ε"] * 100
            for fold in FOLDS:
                ax.plot(x, sub[f"折限定F1_{fold}"], marker="o", ms=4, lw=1.4, color=C_FOLD[fold],
                        alpha=0.75, label=f"{t['hold']} {fold}")
            ax.plot(x, sub["折限定F1"], marker="s", ms=6, lw=2.4, color=C_POOL, label=t["pool"])
            ax.fill_between(x, sub["折限定F1_CI下界"], sub["折限定F1_CI上界"], color=C_POOL, alpha=0.15)
            ax.axhline(CHANCE, color="gray", ls=":", lw=1.2, label=t["chance"])
            ax.set_xlabel(t["eps"] + " (%)")
            ax.set_title(t[mode])
            ax.grid(alpha=0.3)
            ax.set_xticks([e * 100 for e in EPS_GRID])
        axes[0].set_ylabel(t["f1"])
        axes[0].set_ylim(0.25, 0.85)
        axes[1].legend(fontsize=8, loc="lower left", ncol=2)
        fig.suptitle(t["ta"])
        fig.tight_layout()
        return fig

    return plot


def make_plot_b(summary: pd.DataFrame) -> Any:
    def plot(lang: str) -> Any:
        t = TXT[lang]
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for mode in FLIP_MODES:
            sub = summary[
                (summary["翻转口径"] == mode) & (summary["K"] == PRIMARY_K) & (summary["α"] == PRIMARY_ALPHA)
            ].sort_values("ε")
            x = sub["ε"] * 100 + (1.0 if mode == "sd_only" else -1.0)
            ax.errorbar(
                x, sub["配对泄漏Δ"],
                yerr=[sub["配对泄漏Δ"] - sub["Δ_CI下界"], sub["Δ_CI上界"] - sub["配对泄漏Δ"]],
                marker="o", ms=6, lw=2, capsize=4, color=C_MODE[mode], label=t[mode],
            )
        ax.axhline(EQUIV_MARGIN, color="crimson", ls="--", lw=1.2, label=t["equiv"])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel(t["eps"] + " (%)")
        ax.set_ylabel(t["delta"])
        ax.set_title(t["tb"])
        ax.set_xticks([e * 100 for e in EPS_GRID])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    return plot


def make_plot_c(summary: pd.DataFrame) -> Any:
    def plot(lang: str) -> Any:
        t = TXT[lang]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
        cmap = plt.get_cmap("viridis")
        for ax, mode in zip(axes, FLIP_MODES, strict=False):
            for i, k in enumerate(K_LIST):
                sub = summary[
                    (summary["翻转口径"] == mode) & (summary["K"] == k) & (summary["α"] == PRIMARY_ALPHA)
                ].sort_values("ε")
                ax.plot(sub["ε"] * 100, sub["折限定F1"], marker="o", ms=4, lw=1.8,
                        color=cmap(i / max(1, len(K_LIST) - 1)), label=f"{t['k']}={k}")
            ax.axhline(CHANCE, color="gray", ls=":", lw=1.2)
            ax.set_xlabel(t["eps"] + " (%)")
            ax.set_title(t[mode])
            ax.set_xticks([e * 100 for e in EPS_GRID])
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(t["f1"])
        axes[1].legend(fontsize=8, ncol=2)
        fig.suptitle(t["tc"])
        fig.tight_layout()
        return fig

    return plot


# ═══════════════════════════════════════════════════════════════════════════════
# 七、入口
# ═══════════════════════════════════════════════════════════════════════════════
def phase_aggregate(bench: Bench, args: argparse.Namespace, device: torch.device, logger: eu.Logger) -> None:
    df = load_records()
    meta = df.attrs["meta"]
    if df.empty:
        raise RuntimeError(f"缓存为空：{CACHE_DIR}。请先跑 --phase sweep。")
    n_units_expected = len(unit_cases())
    n_units_have = len(meta)
    logger.log(f"缓存单元 {n_units_have}/{n_units_expected}；配对情节 {len(df)} 行")
    if n_units_have < n_units_expected:
        raise RuntimeError(f"缓存不完整（{n_units_have}/{n_units_expected}），拒绝汇总——半张网格的曲线会误导。")

    n_train = int(meta[meta["arm"] != "shared"]["n_train"].mean())
    n_test = int(np.mean([meta[f"ntest_{f}"].dropna().mean() for f in FOLDS]))
    logger.log(f"Nadeau-Bengio 修正用 n_train={n_train}（三折源域苹果均值）、n_test={n_test}（三折目标域苹果均值）")

    summary = summarise(df, n_train, n_test)
    bp = breakpoints(summary)
    logger.log("\n【断点判定】\n" + bp.to_string(index=False))
    prim = summary[(summary["K"] == PRIMARY_K) & (summary["α"] == PRIMARY_ALPHA)]
    logger.log("\n【主口径逐 ε】\n" + prim.round(4).to_string(index=False))

    # 逐折泄漏明细（表 d）
    fold_rows: list[dict[str, Any]] = []
    for (mode, eps, alpha, k, fold), sub in df.groupby(["mode", "eps", "alpha", "k", "fold"], sort=True):
        r = eu.compare_methods(
            sub["f1_shared"].tolist(), sub["f1_restricted"].tolist(),
            n_train=n_train, n_test=n_test, n_bootstrap=N_BOOT, seed=eu.RECOMMENDED_SEEDS[0],
        )
        fold_rows.append({
            "翻转口径": mode, "ε": eps, "α": alpha, "K": k, "折": fold,
            "共享F1": r["mean_a"], "折限定F1": r["mean_b"], "配对泄漏Δ": r["mean_diff"],
            "CI下界": r["ci_low"], "CI上界": r["ci_high"], "P值": r["p_value"],
        })
    fold_df = pd.DataFrame(fold_rows)

    for idx, fn in enumerate((make_plot_a(summary), make_plot_b(summary), make_plot_c(summary))):
        zh, en = eu.save_figure_bilingual(fn, "40_label_noise_robustness", idx)
        logger.log(f"图{chr(97 + idx)} → {os.path.basename(zh)} / {os.path.basename(en)}")

    # 参照编码器落盘（F1：save_checkpoint 带 SHA-256 与模型卡）
    noisy0, _ = make_noisy_labels(bench, FLIP_MODES[0], 0.0, int(eu.RECOMMENDED_SEEDS[0]))
    src_idx = np.where(np.isin(bench.year, SRC_YEARS[2025]))[0]
    ref_enc, ref_va = train_encoder(bench, src_idx, noisy0, int(eu.RECOMMENDED_SEEDS[0]) + 2025, args.episodes, device)
    ck = eu.save_checkpoint(
        ref_enc,
        "40_label_noise_ref_eps0_fold2025",
        {
            "stage": args.stage,
            "training_hardware": device.type,
            "deterministic_guarantee": "full" if device.type == "cuda" else "partial(CPU pilot)",
            "architecture": "ProtoEncoder(1D-CNN+MLP)",
            "task": "标签噪声扫描的参照编码器（ε=0、折限定、留出2025）",
            "note": "本机重训、固定episode预算不早停，与稿件正式CUDA编码器不同源；绝对数值不得写入正文主结果",
            "episodes": args.episodes,
            "seed": int(eu.RECOMMENDED_SEEDS[0]) + 2025,
            "src_val_acc": round(ref_va, 4),
        },
        extra_state={"n_bands": bench.n_bands, "feat_dim": 128, "fold_year": 2025, "val_acc": ref_va},
    )
    logger.log(f"参照编码器 → {ck['ckpt_path']}  sha256={ck['sha256'][:16]}…  源域验证Acc={ref_va:.4f}")

    prespec = pd.DataFrame([
        {"项": "翻转单位", "值": "苹果 apple_id（同果全部面位同标签）"},
        {"项": "翻转目的地", "值": "限定在同年份已出现的产地内（保持每折标签空间不变）"},
        {"项": "随机数口径", "值": "每果一次 u 与一次目的地，翻转集合按 ε 嵌套（共同随机数）"},
        {"项": "查询标签", "值": "始终为声明产地，不翻转"},
        {"项": "编码器", "值": f"固定 {args.episodes} episodes、不早停、取末次权重"},
        {"项": "主分析单位", "值": f"n_primary = 3 折；5 种子 × {args.reps} 重复仅降噪声，不作独立样本"},
        {"项": "CI", "值": f"以折为聚类单位的 cluster bootstrap，{N_BOOT} 次"},
        {"项": "检验", "值": f"Nadeau-Bengio 修正配对 t（n_train={n_train}, n_test={n_test}）+ Holm（族=6 个 ε）"},
        {"项": "C1 判据", "值": "三折合并宏 F1 的 95%CI 下界 > 1/3"},
        {"项": "C1m 判据", "值": f"相对 ε=0 的宏 F1 跌幅 < {MAG_MARGIN}"},
        {"项": "C2 判据", "值": f"配对 Δ 的 95%CI 下界 > {EQUIV_MARGIN} 且 Holm 后 P < {ALPHA_LEVEL}"},
        {"项": "断点 x", "值": "满足「所有 ε' ≤ x 均成立」的最大 ε（前缀规则）"},
    ])
    xlsx = eu.write_script_workbook(
        __file__,
        {
            0: ("预设判据", prespec),
            1: ("断点判定", bp),
            2: ("逐εα K汇总", summary.round(6)),
            3: ("逐折泄漏明细", fold_df.round(6)),
            4: ("单元元信息", meta),
            5: ("逐情节原始记录", df.round(6)),
        },
    )
    logger.log(f"结果表 → {xlsx}")


def phase_selftest(bench: Bench, logger: eu.Logger) -> None:
    """自检：① 自实现宏 F1 与 sklearn 逐例一致；② 噪声按 ε 嵌套且只落在应落的类上。"""
    rng = np.random.default_rng(20060515)
    worst = 0.0
    for _ in range(300):
        n_cls = int(rng.integers(2, 5))
        n = int(rng.integers(5, 400))
        yt = rng.integers(0, n_cls, n)
        yp = rng.integers(0, n_cls, n)
        mine = macro_f1_int(yt, yp, n_cls)
        ref = float(f1_score(yt, yp, labels=list(range(n_cls)), average="macro", zero_division=0))
        worst = max(worst, abs(mine - ref))
    logger.log(f"[自检1] 宏 F1 自实现 vs sklearn：300 例最大偏差 {worst:.3e}")
    if worst > 1e-12:
        raise RuntimeError("自实现宏 F1 与 sklearn 不一致，拒绝继续")

    for mode in FLIP_MODES:
        prev: NDArray | None = None
        for eps in EPS_GRID:
            noisy, flipped = make_noisy_labels(bench, mode, eps, int(eu.RECOMMENDED_SEEDS[0]))
            if mode == "sd_only" and (flipped & (bench.declared != "SD")).any():
                raise RuntimeError("sd_only 口径翻到了非山东的果")
            same_year = all(
                str(noisy[i]) in YEAR_CLASSES[int(bench.year[i])] for i in np.where(flipped)[0]
            )
            if not same_year:
                raise RuntimeError("翻转目的地越出了该年份的标签空间")
            if (noisy[flipped] == bench.declared[flipped]).any():
                raise RuntimeError("被标记翻转的果标签没变")
            if prev is not None and not np.all(prev[prev] <= flipped[prev]):
                raise RuntimeError("翻转集合未按 ε 嵌套")
            rate = flipped.sum() / (bench.n_apple if mode == "uniform" else int((bench.declared == "SD").sum()))
            logger.log(f"[自检2] {mode} ε={eps:.2f} 实际翻转 {flipped.sum():4d} 果，实测率 {rate:.4f}")
            prev = flipped
    logger.log("[自检] 全部通过")


def main() -> None:
    p = argparse.ArgumentParser(description="40_label_noise_robustness：训练标签噪声鲁棒性曲线（L3-2）")
    p.add_argument("--phase", default="all", choices=["sweep", "aggregate", "all", "selftest"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=2000, help="每编码器固定 episode 预算（与稿件折限定臂同值）")
    p.add_argument("--reps", type=int, default=30, help="每 (折,K) 的支持集重采样次数")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--max_seconds", type=float, default=1e9)
    p.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = p.parse_args()

    device = eu.get_device(args.device)
    logger = eu.get_logger("40_label_noise_robustness")
    eu.set_random_seed(int(eu.RECOMMENDED_SEEDS[0]), logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "训练标签噪声鲁棒性曲线（L3-2）",
            "阶段": args.stage,
            "设备": str(device),
            "翻转口径": list(FLIP_MODES),
            "ε网格": list(EPS_GRID),
            "K": list(K_LIST),
            "α": list(ALPHAS),
            "种子": list(eu.RECOMMENDED_SEEDS),
            "episodes/编码器": args.episodes,
            "重复次数": args.reps,
            "分片": f"{args.shard}/{args.nshards}",
        },
    )
    bench = Bench()
    logger.log(f"数据：{bench.n_face} 面位 / {bench.n_apple} 苹果 / {bench.n_bands} 波段")

    with eu.track_compute("标签噪声扫描", logger, None):
        if args.phase == "selftest":
            phase_selftest(bench, logger)
        if args.phase in ("sweep", "all"):
            phase_sweep(bench, args, device, logger)
        if args.phase in ("aggregate", "all"):
            phase_aggregate(bench, args, device, logger)
    logger.close()


if __name__ == "__main__":
    main()
