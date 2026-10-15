"""
64_leakage_controls.py: 目标年暴露的评估泄漏——协议对齐对照、同装置两年对照与预处理变体复核。

背景（稿件 tab:leakage 的两处可辨识性缺口）:
    ① 已报告的共享 vs 折限定两臂，除「训练池是否含目标年」外，还差情节预算（3000 对 2000）、
       早停耐心（15 对 12）、验证情节数（50 对 30）、评估重复次数（50 对 15），且共享臂在含目标年的
       池子上选检查点；等预算对照（44）只拉平了情节预算一维。
    ② LOEO-2025 折同时改变年份、仪器（光纤透射 → 推扫高光谱反射）、测量方式、面位结构与类别集；
       LOEO-2018/2019 的源域也都含 2025 的反射数据与甘肃类，「年份」与「仪器换代」在三折里分不开。

设计：各臂的情节预算、早停耐心、验证情节数、评估重复次数、初始化种子全部相同，评估时各臂用同一组
重采样种子（支持集与查询集逐次配对）；各臂只在训练池与检查点选择上不同：
    R       折限定  : 训练池 = 源域；在源域情节上选检查点
    S_src   目标暴露 : 训练池 = 源域 ∪ 目标年（带标签）；仍只在源域情节上选检查点
    S_size  等样本量 : 训练池 = 从「源域 ∪ 目标年」各域按同一比例抽苹果，总数等于 R 的源域苹果数；源域情节选检查点
    S_pool  照常做法 : 训练池 = 源域 ∪ 目标年；在整个训练池（含目标年）的情节上选检查点
    S_shuf  标签置乱 : 训练池同 S_src，但目标年各苹果的产地标签在该年内随机置换（光谱与类别大小不变）；源域情节选检查点
    S_half  半数暴露 : 目标年各产地的苹果按种子随机对半分成 A、B；训练池 = 源域 ∪ A（带标签），源域情节选检查点，
                      只在 B 上评估。R 与 S_src 的编码器也在同一个 B 上评估（记作 R_B、S_src_B），三者共用
                      同一组重采样种子。S_src 与 S_pool 的评估苹果本身都在训练池里，S_half 的不在
    R_ckpt  留出年选检查点 : 训练池同 R（只有源域）；验证情节取自目标年，检查点与早停都按目标年情节上的准确率；
                      编码器权重从未拟合目标年数据，目标年只经由检查点选择进入
    Δ_rep  = S_src − R      目标年带标签光谱进入表征训练的效应（含训练池变大）
    Δ_size = S_size − R     同上，但训练苹果数相等
    Δ_ckpt = S_pool − S_src 检查点选择看见目标年的附加效应（叠加在已暴露的编码器上）
    Δ_ckpt_R = R_ckpt − R  只经由检查点选择与早停看见目标年的效应（编码器只见源域）
    Δ_all  = S_pool − R     协议对齐后的整体目标年暴露效应
    Δ_lab  = S_src − S_shuf 训练池相同，只差目标年标签对错：差值来自目标年标签
    Δ_shuf = S_shuf − R     目标年光谱进入训练但标签被置乱
    Δ_half = S_half − R_B   目标年带标签光谱进入训练、被评估的苹果不在其中：只经由同年其他苹果的效应
    Δ_seen = S_src_B − S_half  被评估的苹果本身也进入训练的附加效应
    Δ_rep_B = S_src_B − R_B B 半上的 Δ_rep，恒等于 Δ_half + Δ_seen
两种设计:
    three_year : 三折留一采集年（与稿件主设计相同的九个域）；源域 ∪ 目标年 = E1–E9，
                 故 S_pool 三折共用一个编码器
    two_year   : 只用 2018 与 2019（同一套光纤透射装置、同一积分时间、每果一条谱、同三类产地），
                 LOEO-2018 与 LOEO-2019 两折；两年之间的系统差异只剩采集年份（及随年份变化的采购批次）
预处理变体（--data_variant）: canonical = 冻结数据；其余由 66 号脚本写出——
    Q0 严格档质控、现行 SG；QB 宽松档质控、现行 SG；Q1 严格档质控、等宽约 25 nm；Q3 严格档质控、不平滑。
    读入冻结数据，核对行序指纹，按保留行掩码取行，再替换波段列。

运行方式（每次一个设计 × 一个变体 × 一个种子，Formal 阶段强制 CUDA）:
    cd <项目根>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/64_leakage_controls.py \
        --device cuda --stage formal --design three_year --data_variant canonical --seed 20060515
    python 02code/64_leakage_controls.py --aggregate          # 聚合 formal_seeds 下已有的全部格子
    加 --reuse_ckpt：模型卡与本次设定逐项一致、权重 SHA-256 与卡上记录相符的编码器直接载入，不重训
    （给已跑完的格子补新臂时用；评估照常全部重做）。再加 --new_arms R_ckpt：只有列出的臂允许训练，
    其余臂复用不上即报错停止，已落盘的正式编码器不会被重训覆盖

输出文件:
    04outputs/formal_seeds/64_leakage__<design>__<variant>__seed<S>.xlsx        — 各臂 K=5 结果 + Δ + 编码器摘要 + 算力
    04outputs/formal_seeds/64_leakage__<design>__<variant>__seed<S>__pilot.xlsx — pilot（不进聚合）
    04outputs/64_leakage_controls.xlsx                                          — 跨种子聚合（--aggregate）
    07checkpoints/64_<design>_<variant>_<arm>_seed<S>[_fold<Y>]/best.pt          — 各编码器（含模型卡 + SHA-256）
    05logs/64_leakage_controls_YY-MM-DD_HHMMSS.log                              — 运行日志

说明:
    训练循环与 05_meta_train / 10_loeo_year_foldwise / 44 逐行同构（同一 EpisodeSampler、Adam、余弦退火、
    梯度裁剪、每 val_every 轮验证、验证提升即落盘、连续 patience 次不升即早停），只多一个参数：验证情节
    取自哪个子池。单果特征缓存与 K-shot 评估直接调用 44 的 FeatureCache / kshot_eval。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import encoders as enc_lib
import export_utils as eu
from protonet import ELEM_INFO, EpisodeSampler, build_apple_data, build_year_region_data, proto_loss

bmc = importlib.import_module("44_budget_matched_control")

DESIGNS: dict[str, dict[int, dict[str, Any]]] = {
    "three_year": {
        2018: {"src": ["E4", "E5", "E6", "E7", "E8", "E9"], "tgt": ["E1", "E2", "E3"], "src_years": [2019, 2025]},
        2019: {"src": ["E1", "E2", "E3", "E7", "E8", "E9"], "tgt": ["E4", "E5", "E6"], "src_years": [2018, 2025]},
        2025: {"src": ["E1", "E2", "E3", "E4", "E5", "E6"], "tgt": ["E7", "E8", "E9"], "src_years": [2018, 2019]},
    },
    "two_year": {
        2018: {"src": ["E4", "E5", "E6"], "tgt": ["E1", "E2", "E3"], "src_years": [2019]},
        2019: {"src": ["E1", "E2", "E3"], "tgt": ["E4", "E5", "E6"], "src_years": [2018]},
    },
}
ARMS = ("R", "S_src", "S_size", "S_pool", "S_shuf", "S_half", "R_ckpt")
# R、S_src 的编码器在 S_half 的 B 半上的评估（随 S_half 一起产生，不单独训练）
HALF_ARMS = {"R_B": "R", "S_src_B": "S_src"}
EVAL_ARMS = (*ARMS, *HALF_ARMS)
CONTRASTS: dict[str, tuple[str, str]] = {
    "Δ_rep": ("S_src", "R"),
    "Δ_size": ("S_size", "R"),
    "Δ_ckpt": ("S_pool", "S_src"),
    "Δ_ckpt_R": ("R_ckpt", "R"),
    "Δ_all": ("S_pool", "R"),
    "Δ_lab": ("S_src", "S_shuf"),
    "Δ_shuf": ("S_shuf", "R"),
    "Δ_half": ("S_half", "R_B"),
    "Δ_seen": ("S_src_B", "S_half"),
    "Δ_rep_B": ("S_src_B", "R_B"),
}
# 下列训练设定从本版起逐项写进模型卡；此前的卡没有这几项，复用时按当时的命令行默认值比对
# （这些默认值自脚本建立起未改过）
LEGACY_CARD_PROTOCOL: dict[str, Any] = {
    "patience": 12,
    "val_episodes": 30,
    "val_every": 100,
    "lr": 0.001,
    "n_way": 3,
    "k_support": 5,
    "q_query": 10,
    "feat_dim": 128,
    "encoder": "cnn",
}
VARIANT_DIR = os.path.join(eu.PROCESSED_DIR, "variants")


# ═══════════════════════════════════════════════════════════════════════════════
# 一、数据：冻结数据或预处理变体
# ═══════════════════════════════════════════════════════════════════════════════
def load_data(variant: str, logger: eu.Logger) -> pd.DataFrame:
    df = eu.load_all_elements_df(verify_sha256=True)
    if variant == "canonical":
        return df
    path = os.path.join(VARIANT_DIR, f"all_9elements__{variant}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺预处理变体 {path}（先在本机跑 66_sg_window_sensitivity.py 并上传）")
    blob = np.load(path, allow_pickle=False)
    key = "\n".join(
        f"{a}|{b}|{c}|{d}" for a, b, c, d in df[["年份", "产地", "苹果编号", "面位"]].itertuples(index=False)
    )
    if hashlib.sha256(key.encode("utf-8")).hexdigest() != str(blob["row_fingerprint"]):
        raise RuntimeError(f"{variant} 的行序指纹与冻结数据不一致，拒绝替换")
    wl_cols = eu.get_spectrum_columns(df)
    if list(blob["wavelength_cols"]) != list(wl_cols):
        raise RuntimeError(f"{variant} 的波段列与冻结数据不一致")
    keep = np.asarray(blob["keep_mask"], dtype=bool)
    df = df.loc[keep].reset_index(drop=True).copy()
    if len(df) != len(blob["spectra"]):
        raise RuntimeError(f"{variant} 的保留行数与光谱行数不一致")
    df[wl_cols] = blob["spectra"].astype(np.float64)
    logger.log(f"  质控剔除 {int((~keep).sum())} 条饱和光谱，保留 {len(df)} 条")
    logger.log(f"  预处理变体 {variant}：{path}（SHA-256 {eu.compute_sha256(path)[:16]}…）")
    return df


def subsample_pool(
    apple_data: dict[str, dict[Any, dict[str, Any]]], elems: list[str], n_target_total: int, seed: int
) -> dict[str, dict[Any, dict[str, Any]]]:
    """从 elems 各域按同一比例无放回抽苹果，使总数等于 n_target_total（S_size 臂）。"""
    total = sum(len(apple_data[e]) for e in elems)
    rate = n_target_total / total
    rng = np.random.default_rng([seed, n_target_total, 64])
    out: dict[str, dict[Any, dict[str, Any]]] = {}
    for e in elems:
        ids = sorted(apple_data[e].keys())
        k = max(1, round(rate * len(ids)))
        keep = rng.choice(len(ids), size=k, replace=False)
        out[e] = {ids[i]: apple_data[e][ids[i]] for i in sorted(keep)}
    return out


def shuffle_target_labels(
    apple_data: dict[str, dict[Any, dict[str, Any]]], tgt_elems: list[str], seed: int
) -> dict[str, dict[Any, dict[str, Any]]]:
    """目标年各苹果的产地标签在该年内随机置换（S_shuf 臂）；光谱、苹果集合与各类苹果数不变。"""
    pairs = [(e, a) for e in tgt_elems for a in sorted(apple_data[e])]
    regions = [apple_data[e][a]["region"] for e, a in pairs]
    perm = np.random.default_rng([seed, 64, 7]).permutation(len(regions))
    out: dict[str, dict[Any, dict[str, Any]]] = {e: {} for e in tgt_elems}
    for (e, a), j in zip(pairs, perm, strict=True):
        out[e][a] = {"X": apple_data[e][a]["X"], "region": regions[j]}
    return out


def split_target_half(
    apple_data: dict[str, dict[Any, dict[str, Any]]], tgt_elems: list[str], seed: int
) -> tuple[dict[str, dict[Any, dict[str, Any]]], dict[str, set[Any]]]:
    """目标年各域（一域即一个产地）的苹果按种子随机对半分（S_half 臂）。

    返回 (A 半的 apple_data 子集, {产地: B 半的苹果编号集合})；A 进训练池，B 只用于评估；奇数个时 B 多一个。
    """
    rng = np.random.default_rng([seed, 64, 11])
    part_a: dict[str, dict[Any, dict[str, Any]]] = {}
    part_b: dict[str, set[Any]] = {}
    for e in tgt_elems:
        ids = sorted(apple_data[e])
        perm = rng.permutation(len(ids))
        n_a = len(ids) // 2
        part_a[e] = {ids[i]: apple_data[e][ids[i]] for i in sorted(perm[:n_a])}
        part_b.setdefault(str(ELEM_INFO[e][1]), set()).update(ids[i] for i in perm[n_a:])
    return part_a, part_b


# ═══════════════════════════════════════════════════════════════════════════════
# 二、训练：与 44.train_encoder 逐行同构，只多「验证情节取自哪个子池」
# ═══════════════════════════════════════════════════════════════════════════════
def train_encoder(
    run_name: str,
    train_data: dict[str, dict[Any, dict[str, Any]]],
    val_data: dict[str, dict[Any, dict[str, Any]]],
    task_desc: str,
    n_bands: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: eu.Logger,
    extra_card: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    ckpt_path = os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt")
    train_elems = sorted(train_data)
    val_elems = sorted(val_data)
    n_train_apples = sum(len(v) for v in train_data.values())
    logger.log(
        f"\n{'=' * 60}\n【训练】{task_desc}  预算={args.episodes}  训练域={train_elems}"
        f"（{n_train_apples} 苹果）  验证域={val_elems}"
    )
    eu.set_random_seed(args.seed)  # 每个编码器重新播种：初始化与采样流只由 (seed, 训练池) 决定
    sampler = EpisodeSampler(train_elems, train_data, args.n_way, args.k_support, args.q_query)
    val_sampler = EpisodeSampler(val_elems, val_data, args.n_way, args.k_support, args.q_query)
    encoder = enc_lib.build_encoder(args.encoder, n_bands, args.feat_dim).to(device)
    n_params = enc_lib.n_params(encoder)
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.episodes, eta_min=1e-5)
    best_val, best_ep, stopped, no_improve, early = 0.0, 0, 0, 0, False
    card: dict[str, Any] = {}
    for ep in range(1, args.episodes + 1):
        stopped = ep
        encoder.train()
        sx, sy, qx, qy, nw = sampler.sample()
        optimizer.zero_grad()
        loss, _ = proto_loss(encoder, sx, sy, qx, qy, nw, device)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        if ep % args.val_every != 0:
            continue
        encoder.eval()
        accs: list[float] = []
        with torch.no_grad():
            for _ in range(args.val_episodes):
                vx, vy, vqx, vqy, vnw = val_sampler.sample()
                accs.append(proto_loss(encoder, vx, vy, vqx, vqy, vnw, device)[1])
        v = float(np.mean(accs))
        if v > best_val:
            best_val, best_ep, no_improve = v, ep, 0
            card = {
                "stage": args.stage,
                "training_hardware": device.type,
                "deterministic_guarantee": "full" if device.type == "cuda" else "partial(non-CUDA pilot)",
                "architecture": enc_lib.ARCH_DESC[args.encoder],
                "params": n_params,
                "task": task_desc,
                "episodes_budget": args.episodes,
                "train_elems": ",".join(train_elems),
                "val_elems": ",".join(val_elems),
                "n_train_apples": n_train_apples,
                "seed": args.seed,
                "val_acc": round(best_val, 4),
                "best_episode": best_ep,
                **extra_card,
            }
            eu.save_best_checkpoint(
                encoder, run_name, card, extra_state={"n_bands": n_bands, "feat_dim": args.feat_dim}
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                early = True
                break
    if best_ep == 0:
        raise RuntimeError(f"{run_name}: 没有任何 checkpoint 落盘")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    encoder.load_state_dict(state)
    encoder.eval()
    card.update({"stopped_episode": stopped, "early_stopped": early})
    eu.save_best_checkpoint(encoder, run_name, card, extra_state={"n_bands": n_bands, "feat_dim": args.feat_dim})
    logger.log(f"  完成：验证Acc={best_val:.4f} @ep{best_ep}；终止 ep{stopped}（早停={early}）")
    return encoder, {
        "best_val_acc": round(best_val, 4),
        "best_episode": best_ep,
        "stopped_episode": stopped,
        "early_stopped": early,
        "n_train_apples": n_train_apples,
        "sha256": eu.compute_sha256(ckpt_path),
        "复用": False,
    }


def read_card(run_name: str) -> dict[str, str] | None:
    path = os.path.join(eu.CHECKPOINT_DIR, run_name, "model_card.md")
    if not os.path.exists(path):
        return None
    card: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("- ") and ": " in line:
                key, val = line[2:].rstrip("\n").split(": ", 1)
                card[key] = val
    return card


def obtain_encoder(
    run_name: str,
    train_data: dict[str, dict[Any, dict[str, Any]]],
    val_data: dict[str, dict[Any, dict[str, Any]]],
    task_desc: str,
    n_bands: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: eu.Logger,
    extra_card: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    """--reuse_ckpt 时先查已落盘的编码器：模型卡逐项与本次设定一致、训练已走完、权重 SHA-256 与卡上
    记录相符才载入；任何一项不符都照常训练（并覆盖旧权重）。线性族每次都在训练池上重新拟合。"""
    if args.encoder in enc_lib.LINEAR_ENCODERS:
        return fit_linear(run_name, train_data, task_desc, n_bands, args, device, logger, extra_card)
    if args.reuse_ckpt:
        card = read_card(run_name)
        ckpt = os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt")
        want: dict[str, Any] = {
            **extra_card,
            "stage": args.stage,
            "training_hardware": device.type,
            "episodes_budget": args.episodes,
            "seed": args.seed,
            "train_elems": ",".join(sorted(train_data)),
            "val_elems": ",".join(sorted(val_data)),
            "n_train_apples": sum(len(v) for v in train_data.values()),
        }
        bad = "无模型卡或权重" if card is None or not os.path.exists(ckpt) else ""
        if card is not None and not bad:
            for key, val in want.items():
                got = card.get(key, str(LEGACY_CARD_PROTOCOL[key]) if key in LEGACY_CARD_PROTOCOL else None)
                if got != str(val):
                    bad = f"{key}：卡上 {got}，本次 {val}"
                    break
            if not bad and "stopped_episode" not in card:
                bad = "训练未走完"
            if not bad and eu.compute_sha256(ckpt) != card.get("checkpoint_sha256"):
                bad = "权重 SHA-256 与模型卡不符"
        if card is not None and not bad:
            encoder = enc_lib.build_encoder(args.encoder, n_bands, args.feat_dim).to(device)
            state = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
            encoder.load_state_dict(state)
            encoder.eval()
            logger.log(f"  [复用] {run_name}：验证Acc={card['val_acc']} @ep{card['best_episode']}")
            return encoder, {
                "best_val_acc": float(card["val_acc"]),
                "best_episode": int(card["best_episode"]),
                "stopped_episode": int(card["stopped_episode"]),
                "early_stopped": card["early_stopped"] == "True",
                "n_train_apples": int(card["n_train_apples"]),
                "sha256": card["checkpoint_sha256"],
                "复用": True,
            }
        logger.log(f"  [不复用] {run_name}：{bad}")
        new_arms = [a for a in args.new_arms.split(",") if a]
        if new_arms and extra_card.get("arm") not in new_arms:
            raise RuntimeError(
                f"{run_name}：--new_arms 只允许训练 {new_arms}，该臂复用不上（{bad}），停止以免覆盖已落盘的编码器"
            )
    return train_encoder(run_name, train_data, val_data, task_desc, n_bands, args, device, logger, extra_card)


def fit_linear(
    run_name: str,
    train_data: dict[str, dict[Any, dict[str, Any]]],
    task_desc: str,
    n_bands: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: eu.Logger,
    extra_card: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    """线性族（pls / pca）：在训练池的全部光谱（2025 年每果每面位一条）上直接拟合，没有情节训练与检查点选择。

    pls 用各光谱所属苹果的产地作类别；pca 不看标签。拟合结果冻结为线性映射后与深度族同样评估。
    """
    xs: list[Any] = []
    ys: list[str] = []
    for e in sorted(train_data):
        for aid in sorted(train_data[e]):
            faces = train_data[e][aid]["X"]
            xs.append(faces)
            ys += [str(train_data[e][aid]["region"])] * len(faces)
    x = np.vstack(xs)
    y = np.asarray(ys) if args.encoder == "pls" else None
    n_train_apples = sum(len(v) for v in train_data.values())
    logger.log(
        f"\n{'=' * 60}\n【拟合 {args.encoder}】{task_desc}  训练域={sorted(train_data)}"
        f"（{n_train_apples} 苹果，{len(x)} 条光谱）  成分数上限={args.lin_dim}"
    )
    encoder, n_comp = enc_lib.fit_linear_encoder(args.encoder, x, y, args.lin_dim)
    encoder = encoder.to(device)
    encoder.eval()
    card = {
        "stage": args.stage,
        "training_hardware": device.type,
        "deterministic_guarantee": "full" if device.type == "cuda" else "partial(non-CUDA pilot)",
        "fit_backend": "numpy/sklearn on CPU (deterministic); mapping stored as buffers",
        "architecture": enc_lib.ARCH_DESC[args.encoder],
        "params": 0,
        "n_components": n_comp,
        "task": task_desc,
        "train_elems": ",".join(sorted(train_data)),
        "n_train_apples": n_train_apples,
        "n_train_spectra": len(x),
        "seed": args.seed,
        "stopped_episode": 0,
        "early_stopped": False,
        **extra_card,
    }
    ckpt = eu.save_best_checkpoint(encoder, run_name, card, extra_state={"n_bands": n_bands, "n_components": n_comp})
    logger.log(f"  完成：{n_comp} 个成分")
    return encoder, {
        "best_val_acc": float("nan"),
        "best_episode": 0,
        "stopped_episode": 0,
        "early_stopped": False,
        "n_train_apples": n_train_apples,
        "sha256": eu.compute_sha256(ckpt),
        "复用": False,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 三、评估与聚合
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(
    arm: str,
    encoders: dict[int, Any],
    folds: dict[int, dict[str, Any]],
    year_data: Any,
    args: argparse.Namespace,
    device: torch.device,
    logger: eu.Logger,
    restrict: dict[int, dict[str, set[Any]]] | None = None,
    label: str | None = None,
) -> list[dict[str, Any]]:
    """restrict 给出时，每折目标年只保留 restrict[折][产地] 里的苹果（支持集与查询集都从中抽）；源域不变。"""
    rep_seeds = [args.seed + i for i in range(args.reps)]
    name = label or arm
    by_encoder: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    for alpha in [float(a) for a in args.alphas.split(",")]:
        for tgt_year, cfg in sorted(folds.items()):
            cache = by_encoder.setdefault(id(encoders[tgt_year]), bmc.FeatureCache(encoders[tgt_year], device, True))
            yd = year_data
            if restrict is not None:
                keep = restrict[tgt_year]
                yd = {
                    **year_data,
                    tgt_year: {
                        reg: {a: x for a, x in apples.items() if a in keep[reg]}
                        for reg, apples in year_data[tgt_year].items()
                    },
                }
            f1s = [
                float(bmc.kshot_eval(cache, yd, tgt_year, cfg["src_years"], args.k_shot, s, alpha)["f1"])
                for s in rep_seeds
            ]
            rows.append(
                {
                    "设计": args.design,
                    "变体": args.data_variant,
                    "编码器": args.encoder,
                    "臂": name,
                    "评估苹果": "全部" if restrict is None else "B半",
                    "种子": args.seed,
                    "src_alpha": alpha,
                    "目标年份": tgt_year,
                    "K": args.k_shot,
                    "重复次数": args.reps,
                    "F1均值": round(float(np.mean(f1s)), 4),
                    "F1情节SD": round(float(np.std(f1s)), 4),
                }
            )
            logger.log(f"  [{name} α={alpha} 留出{tgt_year}] F1={np.mean(f1s):.4f}")
    return rows


def result_files(encoder: str = "cnn") -> list[str]:
    """64 号正式逐种子结果表；一维卷积的文件名不带编码器段，其余编码器带 `__<编码器>__`。"""
    other = [e for e in enc_lib.ENCODERS if e != encoder]
    files = []
    for f in sorted(glob.glob(os.path.join(eu.OUTPUT_DIR, "formal_seeds", "64_leakage__*.xlsx"))):
        name = os.path.basename(f)
        if "__pilot" in name or any(f"__{e}__" in name for e in other if e != "cnn"):
            continue
        if encoder != "cnn" and f"__{encoder}__" not in name:
            continue
        files.append(f)
    return files


def aggregate(logger: eu.Logger) -> None:
    """一维卷积编码器的对照汇总（64_leakage_controls.xlsx；其余编码器族的汇总见 75 号）。"""
    files = result_files("cnn")
    if not files:
        raise FileNotFoundError("没有找到 64 的逐种子结果")
    res = pd.concat([pd.read_excel(f, sheet_name=0) for f in files], ignore_index=True)
    trn = pd.concat([pd.read_excel(f, sheet_name=2) for f in files], ignore_index=True)
    if "编码器" in res.columns:
        res = res[res["编码器"].fillna("cnn") == "cnn"].drop(columns="编码器")
    rows: list[dict[str, Any]] = []
    for (design, variant, alpha), sub in res.groupby(["设计", "变体", "src_alpha"]):
        present = [a for a in EVAL_ARMS if a in set(sub["臂"])]
        seeds = sorted(sub["种子"].unique().tolist())
        folds = sorted(sub["目标年份"].unique().tolist())
        if set(seeds) != set(eu.RECOMMENDED_SEEDS):
            logger.log(f"  [WARN] {design}/{variant}/α={alpha} 的种子集合 {seeds} 不是 Formal 5 种子，仅作进度查看")

        def f1(arm: str, fold: int, seed: int, _sub: pd.DataFrame = sub) -> float:
            m = _sub[(_sub["臂"] == arm) & (_sub["目标年份"] == fold) & (_sub["种子"] == seed)]
            if len(m) != 1:
                raise ValueError(f"缺格或重复：{arm} {fold} {seed}")
            return float(m["F1均值"].iloc[0])

        for fold in [*folds, "均值"]:
            fs = folds if fold == "均值" else [fold]
            row: dict[str, Any] = {
                "设计": design,
                "变体": variant,
                "src_alpha": alpha,
                "目标年份": fold,
                "种子数": len(seeds),
            }
            for arm in present:
                row[arm] = round(float(np.mean([np.mean([f1(arm, f, s) for f in fs]) for s in seeds])), 4)
            for name, (a, b) in CONTRASTS.items():
                if a not in present or b not in present:
                    continue
                dl = [float(np.mean([f1(a, f, s) - f1(b, f, s) for f in fs])) for s in seeds]
                lo, hi = bmc._seed_t_interval(dl)
                row[name] = round(float(np.mean(dl)), 4)
                row[f"{name}_跨种子SD"] = round(float(np.std(dl, ddof=1)), 4) if len(dl) > 1 else float("nan")
                row[f"{name}_种子t区间"] = f"{lo:.4f}–{hi:.4f}"
                row[f"{name}_为正种子数"] = int(sum(d > 0 for d in dl))
            rows.append(row)
    out_df = pd.DataFrame(rows)
    logger.log("\n" + out_df.to_string(index=False))
    out = os.path.join(eu.OUTPUT_DIR, "64_leakage_controls.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name=eu.table_sheet_name(0, "对照汇总"), index=False)
        res.to_excel(writer, sheet_name=eu.table_sheet_name(1, "逐种子结果"), index=False)
        trn.to_excel(writer, sheet_name=eu.table_sheet_name(2, "编码器训练摘要"), index=False)
    logger.log(f"汇总表 → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="64_leakage_controls：协议对齐的目标年暴露对照")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--design", default="three_year", choices=sorted(DESIGNS))
    parser.add_argument("--data_variant", default="canonical", choices=["canonical", "Q0", "QB", "Q1", "Q3"])
    parser.add_argument("--encoder", default="cnn", choices=list(enc_lib.ENCODERS))
    parser.add_argument("--lin_dim", type=int, default=16, help="线性族（pls / pca）的成分数上限")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val_episodes", type=int, default=30)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--n_way", type=int, default=3)
    parser.add_argument("--k_support", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--alphas", default="0.0")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    parser.add_argument("--reuse_ckpt", action="store_true")
    parser.add_argument("--new_arms", default="", help="与 --reuse_ckpt 同用：只有这些臂允许训练，其余臂必须复用")
    args = parser.parse_args()

    if args.aggregate:
        lg = eu.get_logger("64_leakage_controls")
        aggregate(lg)
        lg.close()
        return

    arms = [a for a in args.arms.split(",") if a]
    if "S_half" in arms and not {"R", "S_src"} <= set(arms):
        raise SystemExit("S_half 须与 R、S_src 同跑：R_B、S_src_B 用这两臂的编码器在 B 半上评估")
    device = eu.get_device(args.device)
    logger = eu.get_logger("64_leakage_controls")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "协议对齐的目标年暴露对照",
            "设计": args.design,
            "预处理变体": args.data_variant,
            "编码器": f"{args.encoder}（{enc_lib.ARCH_DESC[args.encoder]}）",
            "臂": arms,
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "各臂共用 预算/耐心/验证情节/评估重复": f"{args.episodes}/{args.patience}/{args.val_episodes}/{args.reps}",
            "alphas": args.alphas,
            "K": args.k_shot,
            "复用已落盘编码器": args.reuse_ckpt,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))
    df = load_data(args.data_variant, logger)
    wl_cols = eu.get_spectrum_columns(df)
    n_bands = len(wl_cols)
    apple_data = build_apple_data(df, wl_cols)
    year_data = build_year_region_data(df, wl_cols)
    folds = DESIGNS[args.design]
    for e in sorted({e for c in folds.values() for e in c["src"] + c["tgt"]}):
        logger.log(f"  {e} {ELEM_INFO[e]}: {len(apple_data[e])} 苹果")

    enc_seg = "" if args.encoder == "cnn" else f"_{args.encoder}"
    tag = f"64_{args.design}_{args.data_variant}{enc_seg}"
    train_rows: list[dict[str, Any]] = []
    encoders: dict[str, dict[int, Any]] = {a: {} for a in arms}
    with eu.track_compute("目标年暴露对照", logger, run) as budget:
        pooled: Any = None
        half_b: dict[int, dict[str, set[Any]]] = {}
        for tgt_year, cfg in sorted(folds.items()):
            src = {e: apple_data[e] for e in cfg["src"]}
            union = {e: apple_data[e] for e in cfg["src"] + cfg["tgt"]}
            n_src = sum(len(v) for v in src.values())
            card = {
                "design": args.design,
                "data_variant": args.data_variant,
                "fold_year": tgt_year,
                **{k: getattr(args, k) for k in LEGACY_CARD_PROTOCOL},
            }
            specs: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {
                "R": (src, src, f"折限定（{cfg['src']}，留出 {tgt_year}）"),
                "S_src": (union, src, f"目标暴露·源域选检查点（留出 {tgt_year}）"),
            }
            if "S_size" in arms:
                sub = subsample_pool(apple_data, cfg["src"] + cfg["tgt"], n_src, args.seed)
                specs["S_size"] = (
                    sub,
                    {e: sub[e] for e in cfg["src"]},
                    f"等样本量目标暴露（{n_src} 苹果，留出 {tgt_year}）",
                )
            if "S_shuf" in arms:
                specs["S_shuf"] = (
                    {**src, **shuffle_target_labels(apple_data, cfg["tgt"], args.seed + tgt_year)},
                    src,
                    f"目标年标签置乱（留出 {tgt_year}）",
                )
            if "R_ckpt" in arms:
                specs["R_ckpt"] = (
                    src,
                    {e: apple_data[e] for e in cfg["tgt"]},
                    f"折限定·留出年情节选检查点与早停（留出 {tgt_year}）",
                )
            if "S_half" in arms:
                part_a, half_b[tgt_year] = split_target_half(apple_data, cfg["tgt"], args.seed + tgt_year)
                specs["S_half"] = (
                    {**src, **part_a},
                    src,
                    f"半数目标暴露（目标年 A 半 {sum(len(v) for v in part_a.values())} 苹果，留出 {tgt_year}）",
                )
            for arm in [a for a in ("R", "S_src", "S_size", "S_shuf", "S_half", "R_ckpt") if a in arms]:
                tr, va, desc = specs[arm]
                enc, summ = obtain_encoder(
                    f"{tag}_{arm}_seed{args.seed}_fold{tgt_year}",
                    tr,
                    va,
                    desc,
                    n_bands,
                    args,
                    device,
                    logger,
                    {**card, "arm": arm},
                )
                encoders[arm][tgt_year] = enc
                train_rows.append({"臂": arm, "折": tgt_year, "种子": args.seed, **summ})
            if "S_pool" in arms:
                if pooled is None or set(union) != set(pooled[1]):
                    enc, summ = obtain_encoder(
                        f"{tag}_S_pool_seed{args.seed}",
                        union,
                        union,
                        "照常做法（全池训练与选检查点）",
                        n_bands,
                        args,
                        device,
                        logger,
                        {**card, "arm": "S_pool", "fold_year": "all"},
                    )
                    pooled = (enc, union)
                    train_rows.append({"臂": "S_pool", "折": "全部", "种子": args.seed, **summ})
                encoders["S_pool"][tgt_year] = pooled[0]

        logger.log(f"\n{'=' * 60}\n【评估】K={args.k_shot}  α={args.alphas}")
        rows: list[dict[str, Any]] = []
        for arm in arms:
            restrict = half_b if arm == "S_half" else None
            rows += evaluate(arm, encoders[arm], folds, year_data, args, device, logger, restrict)
        if "S_half" in arms:
            for label, arm in HALF_ARMS.items():
                rows += evaluate(arm, encoders[arm], folds, year_data, args, device, logger, half_b, label)

    res_df = pd.DataFrame(rows)
    delta_rows: list[dict[str, Any]] = []
    for (alpha, fold), sub in res_df.groupby(["src_alpha", "目标年份"]):
        f = {r["臂"]: float(r["F1均值"]) for _, r in sub.iterrows()}
        drow: dict[str, Any] = {"src_alpha": alpha, "目标年份": fold, **{a: f[a] for a in EVAL_ARMS if a in f}}
        for name, (a, b) in CONTRASTS.items():
            if a in f and b in f:
                drow[name] = round(f[a] - f[b], 4)
        delta_rows.append(drow)
    delta_df = pd.DataFrame(delta_rows)
    logger.log("\n" + delta_df.to_string(index=False))

    suffix = "" if args.stage == "formal" else "__pilot"
    out_dir = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
    os.makedirs(out_dir, exist_ok=True)
    enc_file = "" if args.encoder == "cnn" else f"__{args.encoder}"
    out_path = os.path.join(
        out_dir, f"64_leakage__{args.design}__{args.data_variant}{enc_file}__seed{args.seed}{suffix}.xlsx"
    )
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        res_df.to_excel(writer, sheet_name=eu.table_sheet_name(0, "K5各臂结果"), index=False)
        delta_df.to_excel(writer, sheet_name=eu.table_sheet_name(1, "逐折Δ"), index=False)
        pd.DataFrame(train_rows).to_excel(writer, sheet_name=eu.table_sheet_name(2, "编码器训练摘要"), index=False)
        pd.DataFrame([{"seed": args.seed, "stage": args.stage, **budget}]).to_excel(
            writer, sheet_name="compute", index=False
        )
    logger.log(f"\n结果表 → {out_path}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
