"""
44_budget_matched_control.py: 等预算对照——把「编码器见过目标年」与「多训 1000 情节」拆开。

背景（`06doc/02sub/00SUBMISSION_AUDIT.md` 的 major `6d976892`）:
    稿件表 tab:leakage 的头条泄漏量 0.242（α=0）／0.363（α=0.4）是一个**捆绑对照**：
    共享臂（05_meta_train）元训练 3000 情节，折限定臂（10_loeo_year_foldwise）2000 情节。
    Δ 因此同时含「编码器见过目标年」与「多训 1000 情节」两个因素，稿件逐字写着
    `no budget-matched control was run`。本脚本把预算这一维拉平，做 2×2 等预算对照。

2×2 设计（同一台服务器、同一份代码路径，四格全部同机重跑，不与历史 formal 结果混算）:

    ┌────────────┬──────────────────────┬──────────────────────────┐
    │            │ 共享臂（见过目标年） │ 折限定臂（未见目标年）   │
    ├────────────┼──────────────────────┼──────────────────────────┤
    │ 2000 情节  │ 【新】主口径·降      │ 复刻锚点（对应已发表值） │
    │ 3000 情节  │ 复刻锚点（已发表值） │ 【新】稳健性口径·升      │
    └────────────┴──────────────────────┴──────────────────────────┘

    Δ(2000) = 共享@2000 − 折限定@2000     等预算·低预算侧
    Δ(3000) = 共享@3000 − 折限定@3000     等预算·高预算侧
    Δ(捆绑) = 共享@3000 − 折限定@2000     稿件现用口径，同机复刻
    预算主效应 = 同臂内 3000 与 2000 之差（两臂各一条）

运行方式（每次一个预算 × 一个种子，Formal 阶段强制 CUDA）:
    cd <项目根>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/44_budget_matched_control.py \
        --device cuda --stage formal --episodes 2000 --seed 20060515

输出文件:
    04outputs/formal_seeds/44_budget_matched__ep<E>__seed<S>.xlsx  — K=5 结果 + 编码器训练摘要 + 算力
    07checkpoints/44_bmc_shared_ep<E>_seed<S>/best.pt              — 共享编码器（含模型卡 + SHA-256）
    07checkpoints/44_bmc_fold_ep<E>_seed<S>_fold<year>/best.pt     — 折限定编码器（含模型卡 + SHA-256）
    05logs/44_budget_matched_control_YY-MM-DD_HHMMSS.log           — 运行日志

说明:
    训练/评估的一切超参与 05_meta_train.py、10_loeo_year_foldwise.py、06_loeo_kshot.py 逐项对齐，
    **只有 --episodes 一个自由度**。两臂各自保留原脚本的 patience（共享 15／折限定 12）与验证
    episode 数（50／30）——这两处差异是既有协议的一部分，本轮不动，仅在报告中如实登记为未消除项。
    评估重复次数沿用已发表口径：共享臂 50 次（06_alpha*），折限定臂 15 次（10_/12_），二者只影响
    均值的蒙特卡洛误差、不影响期望，故不破坏跨预算可比性。
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

import export_utils as eu
from protonet import (
    ELEM_INFO,
    EpisodeSampler,
    ProtoEncoder,
    apple_feature,
    build_apple_data,
    build_year_region_data,
    proto_loss,
)

# 共享臂训练元素（与 05_meta_train.TRAIN_ELEMS 逐项一致；E9 甘肃新类留作下游评估）
SHARED_TRAIN_ELEMS: list[str] = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]

# 折限定臂三折配置（与 10_loeo_year_foldwise.FOLD_CFG 逐项一致）
FOLD_CFG: dict[int, dict[str, Any]] = {
    2018: {"src_elems": ["E4", "E5", "E6", "E7", "E8", "E9"], "src_years": [2019, 2025]},
    2019: {"src_elems": ["E1", "E2", "E3", "E7", "E8", "E9"], "src_years": [2018, 2025]},
    2025: {"src_elems": ["E1", "E2", "E3", "E4", "E5", "E6"], "src_years": [2018, 2019]},
}

ALL_REGIONS = ("XJ", "SD", "SX", "GS")


def _load_or_none(
    ckpt_path: str,
    n_bands: int,
    episodes: int,
    seed: int,
    device: torch.device,
    logger: eu.Logger,
) -> tuple[ProtoEncoder, dict[str, Any]] | None:
    """复用已存 checkpoint，但只在 (episodes, seed, n_bands) 三项全部对得上时才认。

    预算是本脚本唯一的自变量，用错预算的 checkpoint 会让整张对照表失去意义，
    故这里不做「文件存在即复用」，而是逐项核对训练配置。
    """
    if not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:  # 半截文件 / 版本不兼容 → 当作没有，重训
        logger.log(f"  [复用失败] {ckpt_path} 读取异常（{exc}），改为重训")
        return None
    meta = {
        "episodes": ckpt.get("episodes"),
        "seed": ckpt.get("seed"),
        "n_bands": ckpt.get("n_bands"),
    }
    if meta["episodes"] != episodes or meta["seed"] != seed or meta["n_bands"] != n_bands:
        logger.log(f"  [复用拒绝] {ckpt_path} 配置不符 {meta}，改为重训")
        return None
    enc = ProtoEncoder(n_bands, feat_dim=int(ckpt.get("feat_dim", 128)))
    enc.load_state_dict(ckpt["state_dict"])
    enc.to(device)
    enc.eval()
    summary = {
        "best_val_acc": float(ckpt.get("val_acc", float("nan"))),
        "best_episode": int(ckpt.get("best_episode", -1)),
        "stopped_episode": int(ckpt.get("stopped_episode", -1)),
        "reused": True,
    }
    logger.log(f"  [复用] {ckpt_path}（验证Acc={summary['best_val_acc']:.4f}）")
    return enc, summary


def train_encoder(
    run_name: str,
    train_elems: list[str],
    task_desc: str,
    apple_data: dict[str, dict[Any, dict[str, Any]]],
    n_bands: int,
    n_val_episodes: int,
    patience: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: eu.Logger,
    extra_card: dict[str, Any] | None = None,
) -> tuple[ProtoEncoder, dict[str, Any]]:
    """在给定元素集上训练 ProtoNet 编码器，落盘最优权重（含模型卡 + SHA-256）并返回训练摘要。

    训练循环与 05_meta_train / 10_loeo_year_foldwise 逐行同构：同一个 EpisodeSampler、
    同一个 Adam(lr, weight_decay=1e-4)、同一个 CosineAnnealingLR(T_max=episodes, eta_min=1e-5)、
    同一处 clip_grad_norm_(5.0)、同一套「每 val_every 轮验证、验证 Acc 提升即落盘、
    连续 patience 次不提升即早停」。两臂的唯一差别是 train_elems、n_val_episodes 与 patience。

    **每次训练前重新播种**：本脚本一个进程内要连训 4 个编码器，若沿用连续的全局 RNG 流，
    第 2 个之后的编码器起点就取决于前面那个跑了多少情节——而「跑了多少情节」正是本实验的
    自变量，会把预算效应和 RNG 位置搅在一起。重新播种使每个编码器的初始化与采样流只由
    (seed, train_elems) 决定，跨预算严格配对。
    """
    ckpt_path = os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt")
    cached = _load_or_none(ckpt_path, n_bands, args.episodes, args.seed, device, logger)
    if cached is not None:
        return cached

    logger.log(f"\n{'=' * 60}\n【训练】{task_desc}  预算={args.episodes} 情节  元素={train_elems}")
    eu.set_random_seed(args.seed)
    sampler = EpisodeSampler(train_elems, apple_data, args.n_way, args.k_support, args.q_query)
    encoder = ProtoEncoder(n_bands, feat_dim=args.feat_dim)
    encoder.to(device)
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.episodes, eta_min=1e-5)

    best_val_acc = 0.0
    best_episode = 0
    stopped_episode = 0
    no_improve = 0
    early_stopped = False
    best_card: dict[str, Any] = {}

    for ep in range(1, args.episodes + 1):
        stopped_episode = ep
        encoder.train()
        sup_x, sup_y, qry_x, qry_y, nw = sampler.sample()
        optimizer.zero_grad()
        loss, _ = proto_loss(encoder, sup_x, sup_y, qry_x, qry_y, nw, device)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        if ep % args.val_every != 0:
            continue

        encoder.eval()
        v_accs: list[float] = []
        with torch.no_grad():
            for _ in range(n_val_episodes):
                sv_x, sv_y, qv_x, qv_y, vnw = sampler.sample()
                _, va = proto_loss(encoder, sv_x, sv_y, qv_x, qv_y, vnw, device)
                v_accs.append(va)
        vm_acc = float(np.mean(v_accs))
        logger.log(f"  ep={ep:5d}/{args.episodes}  验证Acc={vm_acc:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")
        if vm_acc > best_val_acc:
            best_val_acc, best_episode, no_improve = vm_acc, ep, 0
            card = {
                "stage": args.stage,
                "training_hardware": device.type,
                "deterministic_guarantee": "full" if device.type == "cuda" else "partial(non-CUDA pilot)",
                "architecture": "ProtoEncoder(1D-CNN+MLP)",
                "params": n_params,
                "task": task_desc,
                "episodes_budget": args.episodes,
                "train_elems": ",".join(train_elems),
                "seed": args.seed,
                "val_acc": round(best_val_acc, 4),
                "best_episode": best_episode,
            }
            if extra_card:
                card.update(extra_card)
            best_card = card
            eu.save_best_checkpoint(
                encoder,
                run_name,
                card,
                extra_state={
                    "n_bands": n_bands,
                    "feat_dim": args.feat_dim,
                    "val_acc": best_val_acc,
                    "best_episode": best_episode,
                    "stopped_episode": ep,  # 落盘时点；训练收尾时按真实终止轮重写一次
                    "early_stopped": False,
                    "episodes": args.episodes,
                    "seed": args.seed,
                    "args": vars(args),
                },
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                early_stopped = True
                logger.log(f"  早停：连续 {no_improve} 次验证无提升，实际跑满 {ep} 情节")
                break

    if best_episode == 0:
        raise RuntimeError(
            f"{run_name}: 训练结束但没有任何 checkpoint 落盘（episodes={args.episodes} < val_every={args.val_every}？）"
        )

    # 收尾：把最优权重读回来，再走一次 save_best_checkpoint 落最终盘。
    # 不直接 torch.save 改写文件——那样 model_card.md 里的 SHA-256 会和文件对不上，
    # 而模型卡与权重必须一致是第十一章的硬要求。重存一次让 SHA 与卡由同一入口重算。
    best_state = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    encoder.load_state_dict(best_state)
    encoder.to(device)
    encoder.eval()
    best_card["stopped_episode"] = stopped_episode
    best_card["early_stopped"] = early_stopped
    eu.save_best_checkpoint(
        encoder,
        run_name,
        best_card,
        extra_state={
            "n_bands": n_bands,
            "feat_dim": args.feat_dim,
            "val_acc": best_val_acc,
            "best_episode": best_episode,
            "stopped_episode": stopped_episode,
            "early_stopped": early_stopped,
            "episodes": args.episodes,
            "seed": args.seed,
            "args": vars(args),
        },
    )
    sha = eu.compute_sha256(ckpt_path)
    logger.log(
        f"  完成：最优验证Acc={best_val_acc:.4f} @ep{best_episode}；"
        f"实际终止 ep{stopped_episode}（早停={early_stopped}）→ {ckpt_path}"
    )
    return encoder, {
        "best_val_acc": best_val_acc,
        "best_episode": best_episode,
        "stopped_episode": stopped_episode,
        "early_stopped": early_stopped,
        "params": n_params,
        "sha256": sha,
        "reused": False,
    }


class FeatureCache:
    """按 (年份, 产地, 苹果编号) 缓存 `apple_feature` 的结果。

    为什么需要：本脚本一个格子要跑 (2 臂 × 2 α × 3 折 × 重复次数) 个评估，而每次评估都会
    把源域两年的**全部**苹果重新编码一遍去算源域原型——源域原型根本不依赖重采样种子，
    50 次重复就是 50 次完全相同的前向。查询集苹果同理，单果特征与抽到哪个支持集无关。

    为什么安全：编码器处于 eval 模式、无 dropout、`apple_feature` 只吃该果自己的面位矩阵，
    与批次组成无关，故缓存命中返回的是逐位相同的值而非近似。`--no_cache` 保留了旁路，
    用于对拍验证（见 refine-logs 正本的等价性检查）。
    """

    def __init__(self, encoder: nn.Module, device: torch.device, enabled: bool = True) -> None:
        self.encoder = encoder
        self.device = device
        self.enabled = enabled
        self._store: dict[tuple[int, str, Any], Any] = {}
        self.hits = 0
        self.misses = 0

    def feat(self, year: int, region: str, apple_id: Any, faces: Any) -> Any:
        if not self.enabled:
            self.misses += 1
            return apple_feature(self.encoder, faces, self.device)
        key = (year, region, apple_id)
        cached = self._store.get(key)
        if cached is None:
            self.misses += 1
            cached = apple_feature(self.encoder, faces, self.device)
            self._store[key] = cached
        else:
            self.hits += 1
        return cached


def kshot_eval(
    cache: FeatureCache,
    year_data: Any,
    tgt_year: int,
    src_years: list[int],
    k_shot: int,
    seed: int,
    src_alpha: float,
) -> dict[str, Any]:
    """K-Shot 目标域评估（与 06_loeo_kshot.loeo_year_kshot_eval / 10_.kshot_eval_fold 逐行同构）。

    支持集按苹果编号抽样（果级不相交），查询集为该年份剩余苹果的全面位均值特征；
    src_alpha>0 时把目标域原型与源域同类原型按该权重融合后再做余弦最近原型判别。
    单果特征一律经 `cache.feat` 取用，与原脚本直接调 `apple_feature` 数值等价。
    """
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
        feats = [cache.feat(tgt_year, reg, aid, year_data[tgt_year][reg][aid]) for aid in support_ids[reg]]
        p = np.mean(feats, axis=0)
        tgt_protos[reg] = p / (float(np.linalg.norm(p)) + 1e-8)

    src_protos: dict[str, list[Any]] = {}
    for src_yr in src_years:
        for reg in year_data.get(src_yr, {}):
            if reg not in tgt_regions:
                continue
            feats = [cache.feat(src_yr, reg, aid, faces) for aid, faces in year_data[src_yr][reg].items()]
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
        feat = cache.feat(tgt_year, true_reg, aid, year_data[tgt_year][true_reg][aid]).reshape(1, -1)
        sim = (feat @ proto_mat.T)[0]
        y_true.append(true_reg)
        y_pred.append(tgt_regions[int(sim.argmax())])
    per_class = f1_score(y_true, y_pred, labels=tgt_regions, average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {r: float(v) for r, v in zip(tgt_regions, per_class, strict=False)},
        "n_query": len(query_items),
    }


def eval_arm(
    arm: str,
    encoders: dict[int, ProtoEncoder],
    year_data: Any,
    alphas: list[float],
    n_reps: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: eu.Logger,
) -> list[dict[str, Any]]:
    """对一条臂的三折 × 各 α 做 K-shot 评估，返回逐行结果。

    每个**编码器**一个 FeatureCache，按对象身份去重：共享臂三折是同一个编码器，共用一份缓存；
    折限定臂每折一个编码器，缓存随之分开，不会串味。
    """
    rep_seeds = [args.seed + i for i in range(n_reps)]
    by_encoder: dict[int, FeatureCache] = {}
    caches: dict[int, FeatureCache] = {}
    for yr, enc in encoders.items():
        cache = by_encoder.setdefault(id(enc), FeatureCache(enc, device, enabled=not args.no_cache))
        caches[yr] = cache
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for tgt_year, cfg in sorted(FOLD_CFG.items()):
            cache = caches[tgt_year]
            tgt_regs = sorted(year_data[tgt_year].keys())
            accs: list[float] = []
            f1s: list[float] = []
            n_query = 0
            cls_f1s: dict[str, list[float]] = {r: [] for r in tgt_regs}
            for s in rep_seeds:
                res = kshot_eval(cache, year_data, tgt_year, cfg["src_years"], args.k_shot, s, alpha)
                accs.append(res["acc"])
                f1s.append(res["f1"])
                n_query = int(res["n_query"])
                for r in tgt_regs:
                    cls_f1s[r].append(res["per_class_f1"].get(r, 0.0))
            row: dict[str, Any] = {
                "臂": arm,
                "预算": args.episodes,
                "种子": args.seed,
                "src_alpha": alpha,
                "目标年份": tgt_year,
                "目标产地": "/".join(tgt_regs),
                "K": args.k_shot,
                "重复次数": n_reps,
                "Acc均值": round(float(np.mean(accs)), 4),
                "Acc标准差": round(float(np.std(accs)), 4),
                "F1均值": round(float(np.mean(f1s)), 4),
                "F1标准差": round(float(np.std(f1s)), 4),
            }
            for r in ALL_REGIONS:
                row[f"F1_{r}"] = round(float(np.mean(cls_f1s[r])), 4) if r in cls_f1s else np.nan
            rows.append(row)
            logger.log(
                f"  [{arm} α={alpha} 留出{tgt_year}] F1={row['F1均值']:.4f}±{row['F1标准差']:.4f}  "
                f"Acc={row['Acc均值']:.4f}  n_query={n_query}"
            )
    hits = sum(c.hits for c in by_encoder.values())
    misses = sum(c.misses for c in by_encoder.values())
    logger.log(f"  [{arm}] 特征缓存：命中 {hits} 次 / 实算 {misses} 次（缓存开启={not args.no_cache}）")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 聚合：把 formal_seeds 下的逐格文件合成 2×2 汇总表
# ═══════════════════════════════════════════════════════════════════════════════

# 稿件表 tab:leakage 现用数字（FoodControl_en.tex:441-451），作为外部对照列，不参与本轮任何计算
PUBLISHED: dict[float, dict[str, Any]] = {
    0.4: {"shared": {2018: 0.929, 2019: 0.935, 2025: 0.915}, "fold": {2018: 0.678, 2019: 0.596, 2025: 0.414}},
    0.0: {"shared": {2018: 0.928, 2019: 0.929, 2025: 0.908}, "fold": {2018: 0.668, 2019: 0.702, 2025: 0.668}},
}


def _seed_t_interval(values: list[float]) -> tuple[float, float]:
    """跨种子 95% t 区间：mean ± t(0.975, n−1)·SD/√n。

    为什么不用按折聚类的自助区间：本设计只有 **3 个折**，而百分位自助在 3 个聚类下必然退化——
    三次有放回抽样全落在同一折的概率是 1/27 = 3.7%，已经大于 2.5% 的单侧尾部，于是 95% 百分位
    两端**恒等于逐折均值的极小与极大**，不比「极差」多带任何信息，却顶着「95% CI」的名字。
    本轮初版正是这样把极差标成了 CI（收口复核指出）。现在改为两条口径分开给：
    逐折极值如实叫极值；另给一条跨 5 个种子的 t 区间，它衡量的是种子重复的稳定性，
    **不是**对折的推断，两者不可混用。折层面的推断受 n=3 限制，本项目此前已多次登记该功效问题。
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < 2:
        return float(arr.mean()), float(arr.mean())
    from scipy import stats  # 仅此处用到，避免顶层引入拖慢训练路径

    half = float(stats.t.ppf(0.975, n - 1)) * float(arr.std(ddof=1)) / float(np.sqrt(n))
    return float(arr.mean()) - half, float(arr.mean()) + half


def _read_cells(logger: eu.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读入 04outputs/formal_seeds/44_budget_matched__ep*__seed*.xlsx 的全部逐格结果。"""
    import glob

    pat = os.path.join(eu.OUTPUT_DIR, "formal_seeds", "44_budget_matched__ep*__seed*.xlsx")
    files = sorted(f for f in glob.glob(pat) if "__nocache" not in f)
    if not files:
        raise FileNotFoundError(f"没有找到任何逐格结果文件：{pat}")
    res_parts: list[pd.DataFrame] = []
    trn_parts: list[pd.DataFrame] = []
    for fp in files:
        xl = pd.ExcelFile(fp)
        res_parts.append(xl.parse(0))
        trn_parts.append(xl.parse(2))
        logger.log(f"  读入 {os.path.basename(fp)}")
    res = pd.concat(res_parts, ignore_index=True)
    trn = pd.concat(trn_parts, ignore_index=True)
    logger.log(f"  共 {len(files)} 个格子，{len(res)} 行结果，{len(trn)} 行编码器摘要")
    return res, trn


def aggregate(logger: eu.Logger) -> None:
    """把逐格结果合成 2×2 汇总：等预算 Δ、捆绑 Δ、预算主效应、与稿件数字对照。"""
    res, trn = _read_cells(logger)
    budgets = sorted(res["预算"].unique().tolist())
    alphas = sorted(res["src_alpha"].unique().tolist())
    seeds = sorted(res["种子"].unique().tolist())
    folds = sorted(res["目标年份"].unique().tolist())
    logger.log(f"  预算={budgets}  α={alphas}  种子={len(seeds)} 个  折={folds}")

    def cell(arm: str, ep: int, alpha: float, fold: int, seed: int) -> float:
        m = res[
            (res["臂"] == arm)
            & (res["预算"] == ep)
            & (res["src_alpha"] == alpha)
            & (res["目标年份"] == fold)
            & (res["种子"] == seed)
        ]
        if len(m) != 1:
            raise ValueError(f"逐格结果缺失或重复：arm={arm} ep={ep} α={alpha} fold={fold} seed={seed} → {len(m)} 行")
        return float(m["F1均值"].iloc[0])

    # ── 表a：四格 × α × 折 的 5 种子均值 ────────────────────────────────────────
    rows_a: list[dict[str, Any]] = []
    for arm in ("共享", "折限定"):
        for ep in budgets:
            for alpha in alphas:
                for fold in folds:
                    vals = [cell(arm, ep, alpha, fold, s) for s in seeds]
                    rows_a.append(
                        {
                            "臂": arm,
                            "预算": ep,
                            "src_alpha": alpha,
                            "目标年份": fold,
                            "F1_5种子均值": round(float(np.mean(vals)), 4),
                            "F1_跨种子SD": round(float(np.std(vals, ddof=1)), 4),
                            "稿件已发表值": PUBLISHED[alpha]["shared" if arm == "共享" else "fold"][fold],
                        }
                    )
    df_a = pd.DataFrame(rows_a)

    # ── 表b：Δ 三口径（等预算低/等预算高/捆绑），逐折 + 三折均值 ──────────────────
    lo, hi = min(budgets), max(budgets)
    contrasts = {
        f"等预算Δ@{lo}": (lo, lo),
        f"等预算Δ@{hi}": (hi, hi),
        f"捆绑Δ(共享@{hi}−折限定@{lo})": (hi, lo),
    }
    rows_b: list[dict[str, Any]] = []
    for label, (ep_sh, ep_fr) in contrasts.items():
        for alpha in alphas:
            per_seed_mean: list[float] = []
            flat: list[float] = []
            per_fold: dict[int, list[float]] = {f: [] for f in folds}
            for s in seeds:
                ds = [cell("共享", ep_sh, alpha, f, s) - cell("折限定", ep_fr, alpha, f, s) for f in folds]
                per_seed_mean.append(float(np.mean(ds)))
                for f, d in zip(folds, ds, strict=True):
                    per_fold[f].append(d)
                    flat.append(d)
            fold_means = [float(np.mean(per_fold[f])) for f in folds]
            s_lo, s_hi = _seed_t_interval(per_seed_mean)
            row = {
                "口径": label,
                "src_alpha": alpha,
                "Δ三折均值": round(float(np.mean(per_seed_mean)), 4),
                "Δ跨种子SD": round(float(np.std(per_seed_mean, ddof=1)), 4),
                # 主分析单位是折（n=3），跨种子是重复而非独立单位；两条区间口径不同，分开给。
                "Δ_逐折极小": round(min(fold_means), 4),
                "Δ_逐折极大": round(max(fold_means), 4),
                f"Δ_跨种子95%t下界(n={len(seeds)})": round(s_lo, 4),
                f"Δ_跨种子95%t上界(n={len(seeds)})": round(s_hi, 4),
                "全部折×种子为正": bool(np.all(np.array(flat) > 0)),
                "最小单格Δ": round(float(np.min(flat)), 4),
            }
            for f in folds:
                row[f"Δ_{f}"] = round(float(np.mean(per_fold[f])), 4)
            rows_b.append(row)
    df_b = pd.DataFrame(rows_b)

    # ── 表c：预算主效应（同臂内 hi − lo），这是本轮唯一新增的自由度 ────────────────
    rows_c: list[dict[str, Any]] = []
    for arm in ("共享", "折限定"):
        for alpha in alphas:
            per_seed = [
                float(np.mean([cell(arm, hi, alpha, f, s) - cell(arm, lo, alpha, f, s) for f in folds])) for s in seeds
            ]
            rows_c.append(
                {
                    "臂": arm,
                    "src_alpha": alpha,
                    f"预算主效应({hi}−{lo})": round(float(np.mean(per_seed)), 4),
                    "跨种子SD": round(float(np.std(per_seed, ddof=1)), 4),
                }
            )
    df_c = pd.DataFrame(rows_c)

    # ── 表d：编码器是否真的用满了名义预算（早停会让「预算差」名不副实）──────────────
    trn_sum = (
        trn.groupby(["臂", "预算"])
        .agg(
            编码器数=("stopped_episode", "size"),
            实际终止情节均值=("stopped_episode", "mean"),
            实际终止情节最小=("stopped_episode", "min"),
            最优情节均值=("best_episode", "mean"),
            早停次数=("early_stopped", "sum"),
        )
        .reset_index()
    )
    for c in ("实际终止情节均值", "最优情节均值"):
        trn_sum[c] = trn_sum[c].round(1)

    logger.log(f"\n{'=' * 72}\n【2×2 汇总】")
    for _, r in df_b.iterrows():
        logger.log(
            f"  {r['口径']:28s} α={r['src_alpha']}  Δ={r['Δ三折均值']:.4f}±{r['Δ跨种子SD']:.4f}  "
            f"跨种子95%t[{r[f'Δ_跨种子95%t下界(n={len(seeds)})']:.3f},"
            f"{r[f'Δ_跨种子95%t上界(n={len(seeds)})']:.3f}]  逐折={r[f'Δ_{folds[0]}']:.3f}/"
            f"{r[f'Δ_{folds[1]}']:.3f}/{r[f'Δ_{folds[2]}']:.3f}  全正={r['全部折×种子为正']}"
        )
    logger.log("\n【预算主效应】")
    for _, r in df_c.iterrows():
        logger.log(f"  {r['臂']:6s} α={r['src_alpha']}  {hi}−{lo} = {r[f'预算主效应({hi}−{lo})']:+.4f}")
    logger.log("\n【名义预算是否用满】")
    for _, r in trn_sum.iterrows():
        logger.log(
            f"  {r['臂']:6s} 预算{r['预算']}: {int(r['编码器数'])} 个编码器，"
            f"早停 {int(r['早停次数'])} 次，实际终止情节均值={r['实际终止情节均值']}"
        )

    zh, en = eu.save_figure_bilingual(
        lambda lang: _plot_summary(lang, df_a, df_b, df_c, budgets, alphas, folds, seeds, res),
        "44_budget_matched_control",
        0,
    )
    logger.log(f"\n图a → {zh} / {en}")

    xlsx = eu.write_script_workbook(
        __file__,
        {
            0: ("四格逐折均值", df_a),
            1: ("Δ三口径", df_b),
            2: ("预算主效应", df_c),
            3: ("预算用满情况", trn_sum),
            "cells": res,
            "encoders": trn,
        },
    )
    logger.log(f"\n汇总表 → {xlsx}")


def _plot_summary(
    lang: str,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    df_c: pd.DataFrame,
    budgets: list[int],
    alphas: list[float],
    folds: list[int],
    seeds: list[int],
    res: pd.DataFrame,
) -> Any:
    """左：四格 F1 点图（臂间落差 vs 预算落差）；右：捆绑 Δ 拆成「等预算 Δ + 预算贡献」。"""
    import matplotlib.pyplot as plt

    t = {
        "zh": {
            "sup": "等预算对照：把「见过目标年」与「多训 1000 情节」拆开",
            "ta": "四格三折平均 Macro-F1（5 种子）",
            "tb": "捆绑 Δ 的分解",
            "ya": "Macro-F1（3 类产地）",
            "xb": "评估泄漏 Δ（宏 F1 之差）",
            "shared": "共享编码器（见过目标年）",
            "fold": "折限定编码器（未见目标年）",
            "eq": "等预算 Δ（同预算两臂之差）",
            "bud": "预算贡献（同臂 3000−2000）",
            "pub": "稿件现用值",
            "ep": "情节",
        },
        "en": {
            "sup": "Budget-matched control: separating target-year exposure from the extra 1000 episodes",
            "ta": "Three-fold mean macro-F1 of the four cells (5 seeds)",
            "tb": "Decomposition of the bundled leakage",
            "ya": "Macro-F1 (3 origins)",
            "xb": "Evaluation leakage $\\Delta$ (macro-F1 difference)",
            "shared": "Shared encoder (saw target year)",
            "fold": "Fold-restricted encoder (did not)",
            "eq": "Budget-matched $\\Delta$ (same budget, both arms)",
            "bud": "Budget contribution (same arm, 3000-2000)",
            "pub": "Value in the manuscript",
            "ep": "episodes",
        },
    }[lang]
    c_shared, c_fold, c_eq, c_bud = "#0072B2", "#009E73", "#0072B2", "#E69F00"
    lo, hi = min(budgets), max(budgets)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ── 左：四格点图 ─────────────────────────────────────────────────────────
    ax = axes[0]
    labels: list[str] = []
    ypos = 0.0
    yticks: list[float] = []
    for alpha in alphas:
        for arm, colour in (("共享", c_shared), ("折限定", c_fold)):
            xs: list[float] = []
            for ep in (lo, hi):
                per_seed = [float(np.mean([_cell_of(res, arm, ep, alpha, f, s) for f in folds])) for s in seeds]
                xs.append(float(np.mean(per_seed)))
                ax.scatter(
                    per_seed,
                    [ypos] * len(per_seed),
                    s=14,
                    color=colour,
                    alpha=0.28,
                    zorder=2,
                    edgecolors="none",
                )
            ax.plot(xs, [ypos, ypos], color=colour, linewidth=2.2, alpha=0.85, zorder=3)
            ax.scatter([xs[0]], [ypos], s=70, facecolors="white", edgecolors=colour, linewidths=2.0, zorder=4)
            ax.scatter([xs[1]], [ypos], s=70, color=colour, zorder=4)
            labels.append(f"α={alpha}  {t['shared'] if arm == '共享' else t['fold']}")
            yticks.append(ypos)
            ypos -= 1.0
        ypos -= 0.4
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(t["ya"])
    ax.set_title(t["ta"], fontsize=11)
    ax.axvline(1 / 3, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlim(0.25, 1.0)
    ax.grid(axis="x", alpha=0.25)
    ax.scatter([], [], s=70, facecolors="white", edgecolors="gray", linewidths=2.0, label=f"{lo} {t['ep']}")
    ax.scatter([], [], s=70, color="gray", label=f"{hi} {t['ep']}")
    ax.legend(fontsize=8.5, loc="lower left", frameon=False)

    # ── 右：捆绑 Δ = 等预算 Δ + 预算贡献 ──────────────────────────────────────
    ax = axes[1]
    ypos = 0.0
    yticks, labels = [], []
    for alpha in alphas:
        eq = float(df_b[(df_b["口径"] == f"等预算Δ@{lo}") & (df_b["src_alpha"] == alpha)]["Δ三折均值"].iloc[0])
        bundled = float(
            df_b[(df_b["口径"] == f"捆绑Δ(共享@{hi}−折限定@{lo})") & (df_b["src_alpha"] == alpha)]["Δ三折均值"].iloc[0]
        )
        ax.plot([0, eq], [ypos, ypos], color=c_eq, linewidth=9, solid_capstyle="butt", zorder=2)
        ax.plot([eq, bundled], [ypos, ypos], color=c_bud, linewidth=9, solid_capstyle="butt", zorder=2)
        pub = PUBLISHED[alpha]
        pub_delta = float(np.mean([pub["shared"][f] - pub["fold"][f] for f in folds]))
        ax.scatter([pub_delta], [ypos], marker="|", s=420, color="#333333", linewidths=2.0, zorder=5)
        ax.text(bundled + 0.008, ypos, f"{bundled:.3f}", va="center", fontsize=9, color="#333333")
        ax.text(eq / 2, ypos, f"{eq:.3f}", va="center", ha="center", fontsize=8.5, color="white", fontweight="bold")
        yticks.append(ypos)
        labels.append(f"α={alpha}")
        ypos -= 1.0
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_ylim(ypos + 0.5, 1.15)
    ax.set_xlim(0, max(0.45, ax.get_xlim()[1]))
    ax.set_xlabel(t["xb"])
    ax.set_title(t["tb"], fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.plot([], [], color=c_eq, linewidth=9, label=t["eq"])
    ax.plot([], [], color=c_bud, linewidth=9, label=t["bud"])
    ax.scatter([], [], marker="|", s=300, color="#333333", linewidths=2.0, label=t["pub"])
    ax.legend(fontsize=8.5, loc="upper right", frameon=False, ncol=1)

    fig.suptitle(t["sup"], fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    return fig


def _cell_of(res: pd.DataFrame, arm: str, ep: int, alpha: float, fold: int, seed: int) -> float:
    m = res[
        (res["臂"] == arm)
        & (res["预算"] == ep)
        & (res["src_alpha"] == alpha)
        & (res["目标年份"] == fold)
        & (res["种子"] == seed)
    ]
    return float(m["F1均值"].iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="44_budget_matched_control：等预算对照（共享 vs 折限定）")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aggregate", action="store_true", help="只聚合 formal_seeds 下已有的逐格结果，不训练")
    parser.add_argument("--episodes", type=int, help="两臂共用的元训练情节预算（本脚本唯一自变量）")
    parser.add_argument("--n_way", type=int, default=3)
    parser.add_argument("--k_support", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--shared_patience", type=int, default=15, help="共享臂早停耐心（对齐 05_meta_train）")
    parser.add_argument("--fold_patience", type=int, default=12, help="折限定臂早停耐心（对齐 10_loeo_year_foldwise）")
    parser.add_argument("--shared_val_episodes", type=int, default=50, help="对齐 05_meta_train 的验证 episode 数")
    parser.add_argument(
        "--fold_val_episodes", type=int, default=30, help="对齐 10_loeo_year_foldwise 的验证 episode 数"
    )
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--shared_reps", type=int, default=50, help="共享臂评估重复次数（对齐 06_alpha*）")
    parser.add_argument("--fold_reps", type=int, default=15, help="折限定臂评估重复次数（对齐 10_/12_）")
    parser.add_argument("--alphas", default="0.0,0.4")
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="旁路单果特征缓存，逐次重算（只用于对拍验证缓存的数值等价性，正式跑不要开）",
    )
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    if args.aggregate:
        agg_logger = eu.get_logger("44_budget_matched_control")
        aggregate(agg_logger)
        agg_logger.close()
        return
    if args.episodes is None:
        parser.error("--episodes 是必填项（除非用 --aggregate）")

    alphas = [float(a) for a in args.alphas.split(",")]
    device = eu.get_device(args.device)
    logger = eu.get_logger("44_budget_matched_control")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "等预算对照（共享 vs 折限定，2×2 之一格）",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "预算(episodes)": args.episodes,
            "alphas": alphas,
            "K": args.k_shot,
            "共享臂重复/耐心/验证episode": f"{args.shared_reps}/{args.shared_patience}/{args.shared_val_episodes}",
            "折限定臂重复/耐心/验证episode": f"{args.fold_reps}/{args.fold_patience}/{args.fold_val_episodes}",
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    apple_data = build_apple_data(df, wl_cols)
    year_data = build_year_region_data(df, wl_cols)
    for eid in SHARED_TRAIN_ELEMS:
        logger.log(f"  {eid} {ELEM_INFO[eid]}: {len(apple_data[eid])} 苹果")

    train_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    with eu.track_compute("等预算对照", logger, run) as budget:
        # ── 共享臂：一套编码器（在含目标年的 E1–E8 上训练），三折共用 ──────────────
        shared_name = f"44_bmc_shared_ep{args.episodes}_seed{args.seed}"
        shared_enc, shared_sum = train_encoder(
            shared_name,
            SHARED_TRAIN_ELEMS,
            "共享编码器（E1–E8，含目标年）",
            apple_data,
            n_bands,
            args.shared_val_episodes,
            args.shared_patience,
            args,
            device,
            logger,
        )
        train_rows.append({"臂": "共享", "预算": args.episodes, "种子": args.seed, "折": "全部", **shared_sum})

        # ── 折限定臂：每折一套编码器（目标年完全排除在训练之外）────────────────────
        fold_encs: dict[int, ProtoEncoder] = {}
        for tgt_year, cfg in sorted(FOLD_CFG.items()):
            name = f"44_bmc_fold_ep{args.episodes}_seed{args.seed}_fold{tgt_year}"
            enc, summ = train_encoder(
                name,
                cfg["src_elems"],
                f"折限定编码器（留出 {tgt_year}）",
                apple_data,
                n_bands,
                args.fold_val_episodes,
                args.fold_patience,
                args,
                device,
                logger,
                extra_card={"fold_year": tgt_year},
            )
            fold_encs[tgt_year] = enc
            train_rows.append({"臂": "折限定", "预算": args.episodes, "种子": args.seed, "折": tgt_year, **summ})

        # ── 评估：两臂 × 三折 × 各 α ────────────────────────────────────────────
        logger.log(f"\n{'=' * 60}\n【评估】K={args.k_shot}")
        result_rows += eval_arm(
            "共享", dict.fromkeys(FOLD_CFG, shared_enc), year_data, alphas, args.shared_reps, args, device, logger
        )
        result_rows += eval_arm("折限定", fold_encs, year_data, alphas, args.fold_reps, args, device, logger)

    res_df = pd.DataFrame(result_rows)
    train_df = pd.DataFrame(train_rows)

    # ── 本预算格内的 Δ（同预算两臂之差）────────────────────────────────────────
    delta_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for tgt_year in sorted(FOLD_CFG):
            sh = res_df[(res_df["臂"] == "共享") & (res_df["src_alpha"] == alpha) & (res_df["目标年份"] == tgt_year)]
            fr = res_df[(res_df["臂"] == "折限定") & (res_df["src_alpha"] == alpha) & (res_df["目标年份"] == tgt_year)]
            delta_rows.append(
                {
                    "预算": args.episodes,
                    "种子": args.seed,
                    "src_alpha": alpha,
                    "目标年份": tgt_year,
                    "共享F1": float(sh["F1均值"].iloc[0]),
                    "折限定F1": float(fr["F1均值"].iloc[0]),
                    "等预算Δ": round(float(sh["F1均值"].iloc[0]) - float(fr["F1均值"].iloc[0]), 4),
                }
            )
    delta_df = pd.DataFrame(delta_rows)
    logger.log(f"\n{'=' * 60}\n【本格等预算 Δ】预算={args.episodes} 种子={args.seed}")
    for alpha in alphas:
        sub = delta_df[delta_df["src_alpha"] == alpha]
        per_fold = "/".join(f"{v:.3f}" for v in sub["等预算Δ"])
        logger.log(f"  α={alpha}: 逐折Δ={per_fold}  三折均值={sub['等预算Δ'].mean():.4f}")

    budget_df = pd.DataFrame([{"episodes": args.episodes, "seed": args.seed, "stage": args.stage, **budget}])

    out_dir = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
    os.makedirs(out_dir, exist_ok=True)
    # 对拍跑（--no_cache）另存一个文件名，免得把正式格子覆盖掉
    suffix = "__nocache" if args.no_cache else ""
    out_path = os.path.join(out_dir, f"44_budget_matched__ep{args.episodes}__seed{args.seed}{suffix}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        res_df.to_excel(writer, sheet_name=eu.table_sheet_name(0, "K5两臂结果"), index=False)
        delta_df.to_excel(writer, sheet_name=eu.table_sheet_name(1, "等预算Δ"), index=False)
        train_df.to_excel(writer, sheet_name=eu.table_sheet_name(2, "编码器训练摘要"), index=False)
        budget_df.to_excel(writer, sheet_name="compute", index=False)
    logger.log(f"\n结果表 → {out_path}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
