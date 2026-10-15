"""85_checkpoint_paired.py: 同一条训练轨迹上的两种检查点选法（检查点暴露的配对对照，补充材料 S9.4）。

为什么:
    64 号的检查点臂（R_ckpt 对 R）只在一维卷积 × 苹果上跑过，而且两臂是两条训练轨迹：
    64 号的验证抽样与训练抽样共用全局随机数（protonet.EpisodeSampler 用 random 模块），
    R 的验证池是源域、R_ckpt 的验证池是留出年，池子大小不同，第 100 个情节第一次验证之后
    两臂消耗的随机数就不同，此后的训练情节不再相同。R_ckpt − R 里除了「按哪个域的验证得分
    选检查点」，还混着两条轨迹之间的差异，R_ckpt 的早停也按留出年的得分。
    本脚本每格只训一条仅源域的轨迹，在这条轨迹的快照里按不同的验证得分各选一个检查点，
    差值只来自选法。

设计（一格 = 一个留出域 × 一个编码器族 × 一个种子）:
    训练  仅源域，训满预算、不早停，优化器、学习率与情节构成同各自的正式配方：
          苹果 2000 情节（64 号 train_encoder：3 类 × 5 支持 + 10 查询，Adam 学习率 1e-3、
          权重衰减 1e-4，余弦退火到 1e-5，梯度裁剪 5，特征维 128）；
          公开数据集 1000 情节（crossdataset_lib.meta_train：全部合法类 × 5 支持 + 10 查询，
          Adam 学习率 1e-3，特征维 64）。
    快照  每 100 个情节把权重与缓冲区深拷贝到内存（不落盘），并在三个验证池上各抽 30 个
          验证情节，记查询准确率的均值（64 号选检查点用的就是这个量）：
              src    源域，即训练池（同 64 号 R 臂的验证池）
              tgt    留出域全部带标签样本（同 64 号 R_ckpt 臂的验证池）
              tgt_A  留出域的 A 半
          苹果的验证情节与 64 号同一构成（3 个产地 × 5 支持果 + 10 查询果，支持取全面位均值、
          查询取随机一面）。公开数据集的验证情节照训练情节（每类 5 支持 + 10 查询），两处不同：
          只收池内样本数 ≥ 7（即 K+2）的类；某类不足 15 个样本时查询只取支持之外剩下的样本，
          不循环取用，支持与查询不重叠。可用的类不足 2 个时该验证池不可用。
    检查点选法:
        last           末次快照，即训满预算时的权重
        src_best       源域验证得分最高的快照（并列取最早）
        tgt_best       留出域验证得分最高的快照（并列取最早）
        tgt_half_best  留出域 A 半验证得分最高的快照（并列取最早），在 B 半上评估
        src_best_on_B  src_best 的同一个快照，在 B 半上评估（半分版的对照）
    对照（种子内配对作差）:
        Δ_ckpt      = tgt_best − src_best            在留出域全体上评估，选检查点时见过被评估的样本
        Δ_ckpt_half = tgt_half_best − src_best_on_B  在 B 半上评估，选检查点时只见过同域的其他样本
    半分  留出域按 S_half 的办法分 A、B，奇数时 B 多一个：苹果以苹果编号为单位、按产地分层
          （同 64 号 split_target_half）；公开数据集逐类对半（同 74 号 half_split）。划分用本脚本
          自己的随机数流，与 64、74 号 S_half 的那一次划分不同。
    评估  K=5、每格 50 次支持集重采样的宏 F1，同一格内各检查点用同一批评估情节。
          苹果同 64 号 evaluate（44 号 kshot_eval，α=0，重采样种子 S+i）；公开数据集同 74 号
          （crossdataset_lib.eval_kshot；B 半只评其中样本数 ≥ K+2 的类，不足 3 类时不评）。

随机数（本脚本存在的理由）:
    训练情节照旧取自全局随机数（苹果 random，公开数据集 numpy.random），每条轨迹开头由
    eu.set_random_seed 播种。验证情节与 A/B 划分取自独立的随机数流
    numpy.random.default_rng([种子, 85, 格, 用途, 快照序号])，不碰全局随机数；验证时编码器
    在 eval 模式，没有随机丢弃。所以有没有验证、验证池多大，都不改变训练轨迹。
    每次快照与验证前后核对 random、numpy.random、torch（CPU 与各 CUDA 设备）四路全局
    随机数状态，有一路变了就报错中止。
    不同快照用不同的验证情节（流里带快照序号），同 64 号每次验证重新抽样的做法。

自检:
    --selfcheck（运行时）
        每格另训一条不带快照与验证的轨迹，核对末次权重与带验证的那条逐位相同，不同即中止。
        训练量加倍，pilot 时开。
    --aggregate（只对正式结果）
        公开数据集上的 last 与 74 号的 R 臂是同一配方、同一种子的仅源域训练，评估情节也相同，
        同机同环境下宏 F1 应逐位相同。逐格的差写进汇总表（表e），超过 1e-6 的格在日志里列出。
        74 号正式表若出自另一台机器，这个差反映的是机器之间的差异。苹果一侧没有可比的现成量
        （64 号 R 臂带早停，验证又与训练共用随机数）。

与 64 号 R_ckpt 臂的区别:
    一条轨迹对两条轨迹；训满预算对按验证得分早停；验证用独立随机数流对共用全局随机数；
    对照的基线是同一条轨迹上的 src_best，不是另一条轨迹的 R；另有半分版；
    范围从一维卷积 × 苹果扩到四个深度族 × 苹果 5 折与公开数据集 16 个留出域。

范围:
    编码器族  cnn / mlp / resnet / transformer（线性族没有训练轨迹，不在本脚本内）
    留出域    苹果两年设计 2 个、三年设计 3 个留出年份（64 号 DESIGNS）；
              公开数据集 16 个留出域（74 号 DATASETS）
    主口径    苹果两年设计 + 公开数据集 = 18 个留出域；苹果三年设计单列

输入:
    03data/processed/all_9elements.csv 与同目录的数据卡（苹果冻结数据，读入时核对 SHA-256）
    --data_root 下的五个公开数据集（玉米、药片、芒果、Indian Pines、猕猴桃，文件同 74 号）
    --aggregate 的核对另读 04outputs/formal_seeds/74_multi_encoder__*.xlsx（只读，可以没有）

复用:
    64_leakage_controls        DESIGNS、load_data、LEGACY_CARD_PROTOCOL（核对训练设定）；
                               训练循环照 train_encoder 写出（去掉验证落盘与早停），评估照 evaluate
    44_budget_matched_control  FeatureCache、kshot_eval
    74_multi_encoder_leakage   DATASETS、load_long（汇总时的核对）；半分照 half_split，
                               B 半的评估类照 run_fold
    crossdataset_lib           prepare_fold、ArrSampler、eval_kshot；训练循环照 meta_train 写出
    protonet                   EpisodeSampler（训练采样与验证池的索引）、proto_loss、
                               build_apple_data、build_year_region_data
    encoders                   build_encoder、DEEP_ENCODERS
    export_utils               设备、播种、Formal 的 CUDA 闸、日志、算力记录、表名

运行方式（从项目根运行；一次一个种子；按范围与编码器分文件，可并行）:
    cd <项目根>
    # pilot：情节预算与评估重采样次数各取正式值的 1/5（苹果 400、公开数据集 200 情节，评估 10 次）
    python 02code/85_checkpoint_paired.py --device cuda --stage pilot --seed 2004 --selfcheck \
        --data_root <公开数据集根目录>
    # formal（要求 CUDA）
    python 02code/85_checkpoint_paired.py --device cuda --stage formal --seed 20060515 \
        --scope public --encoders transformer --data_root <公开数据集根目录>
    python 02code/85_checkpoint_paired.py --device cuda --stage formal --seed 20060515 \
        --scope apple --encoders cnn
    # 汇总（缺省读正式结果；加 --stage pilot 读 pilot 结果）
    python 02code/85_checkpoint_paired.py --aggregate

输出文件:
    04outputs/formal_seeds/85_checkpoint_paired__<范围>__<编码器>__seed<S>[__pilot].xlsx
        表a 逐检查点选法结果：一行 = 一格 × 一种选法。列：范围、dataset（数据集）、held（留出域）、
            编码器、种子、检查点选法、评估样本、所选情节数、三个验证得分、F1（评估宏 F1）、
            情节数（评估情节数）等；dataset、held、编码器、种子、F1 的列名沿用 74、75 号
        表b 逐快照验证得分；compute
        <范围> 是 apple 或 public；只跑部分设计或数据集时带上名字（如 apple-two_year），
        不完整的表不会顶替完整的表
    04outputs/85_checkpoint_paired.xlsx（--aggregate；pilot 时文件名加 __pilot）
        表a 逐留出域（Δ 的种子均值、种子 t 区间、为正种子数）、表b 按族汇总（主口径与苹果三年设计）、
        表c 逐格、表d 长格式、表e 与 74 号 R 臂的核对、compute
    05logs/85_checkpoint_paired_YY-MM-DD_HHMMSS.log
    不写权重文件；不写 64、72、74 号的任何结果表，不读写 07checkpoints 下的权重。
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import importlib
import os
import random
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch.optim.lr_scheduler import CosineAnnealingLR

import crossdataset_lib as cdl
import encoders as enc_lib
import export_utils as eu
from protonet import ELEM_INFO, EpisodeSampler, build_apple_data, build_year_region_data, proto_loss

m64 = importlib.import_module("64_leakage_controls")
bmc = importlib.import_module("44_budget_matched_control")
m74 = importlib.import_module("74_multi_encoder_leakage")

NDArray = npt.NDArray[Any]
Episode = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
AppleData = dict[str, dict[Any, dict[str, Any]]]
YearData = dict[int, dict[str, dict[Any, NDArray]]]
RngState = tuple[Any, Any, torch.Tensor, list[torch.Tensor]]

STEM = "85_checkpoint_paired"
SEED_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
DATA_ROOT = "05data"  # 同 74 号 --data_root 的缺省值


@dataclass(frozen=True)
class Recipe:
    """仅源域原型网络元训练的一套正式设定。"""

    name: str
    episodes: int
    feat_dim: int
    lr: float
    weight_decay: float
    eta_min: float | None  # 余弦退火的终值；None 表示学习率不变
    clip_norm: float | None  # 梯度范数裁剪的上限；None 表示不裁剪


# 苹果：64 号 train_encoder 及其命令行缺省值（--episodes 2000 --lr 1e-3 --feat_dim 128；Adam 权重衰减
# 1e-4、余弦退火到 1e-5、梯度裁剪 5 写在它的训练循环里）。四个深度族共用这一套
APPLE_RECIPE = Recipe(
    name="apple", episodes=2000, feat_dim=128, lr=1e-3, weight_decay=1e-4, eta_min=1e-5, clip_norm=5.0
)
APPLE_N_WAY = 3
APPLE_K_SUPPORT = 5
APPLE_Q_QUERY = 10
# 公开数据集：crossdataset_lib.meta_train（特征维 64，Adam 学习率 1e-3，无权重衰减、无退火、无裁剪，
# ArrSampler 每类 5 支持 + 10 查询）与 74 号的 --episodes 1000。四个深度族共用这一套
PUBLIC_RECIPE = Recipe(
    name="public", episodes=1000, feat_dim=64, lr=1e-3, weight_decay=0.0, eta_min=None, clip_norm=None
)
PUBLIC_K_SUPPORT = 5
PUBLIC_Q_QUERY = 10
SNAP_EVERY = 100  # 每多少个情节存一次快照并验证（64 号 --val_every）
VAL_EPISODES = 30  # 每个验证池每次抽多少个验证情节（64 号 --val_episodes）
K_SHOT = 5  # 评估时每类的支持样本数（64 号 --k_shot、74 号 --k）
REPS = 50  # 每格评估的支持集重采样次数（64、74 号 --reps）
PILOT_DIV = 5  # pilot：情节预算与评估重采样次数各除以 5

# 独立随机数流 default_rng([种子, STREAM, 格, 用途, 序号])；86 号用同一套流（同一次划分、同一批验证情节）
STREAM = 85
TAG_SPLIT = 11
POOL_SRC, POOL_TGT, POOL_TGT_A = "src", "tgt", "tgt_A"
POOLS = (POOL_SRC, POOL_TGT, POOL_TGT_A)
VAL_TAG = {POOL_SRC: 21, POOL_TGT: 22, POOL_TGT_A: 23}
VAL_COL = {POOL_SRC: "源域验证得分", POOL_TGT: "留出域验证得分", POOL_TGT_A: "留出域A半验证得分"}
EVAL_ALL, EVAL_B = "全部", "B半"
# 检查点选法 → (按哪个验证池的得分选，None 为末次；在哪部分留出域样本上评估)
SELECTIONS: dict[str, tuple[str | None, str]] = {
    "last": (None, EVAL_ALL),
    "src_best": (POOL_SRC, EVAL_ALL),
    "tgt_best": (POOL_TGT, EVAL_ALL),
    "tgt_half_best": (POOL_TGT_A, EVAL_B),
    "src_best_on_B": (POOL_SRC, EVAL_B),
}
CONTRASTS: dict[str, tuple[str, str]] = {
    "Δ_ckpt": ("tgt_best", "src_best"),
    "Δ_ckpt_half": ("tgt_half_best", "src_best_on_B"),
}
# 苹果的设计 → 结果表里的数据集名（同 75 号 APPLE_DESIGNS）；顺序即缺省的运行顺序
APPLE_SETS = {"two_year": "apple_two_year", "three_year": "apple_three_year"}
SCOPE_SETS: dict[str, list[str]] = {
    "主口径（苹果两年设计 + 公开数据集）": ["apple_two_year", *m74.DATASETS],
    "苹果三年设计": ["apple_three_year"],
}
SCOPE_DOMAINS = {"主口径（苹果两年设计 + 公开数据集）": 18, "苹果三年设计": 3}
KEYS = ["dataset", "held", "编码器", "种子"]  # 一格的键；列名同 74、75 号
CHECK_TOL = 1e-6  # 与 74 号 R 臂核对的容差（同 74 号自检）


def check_recipe() -> None:
    """苹果配方里与 64 号重复写的几项，逐项对 64 号登记的正式训练设定；有一项不同即中止。"""
    mine = {
        "lr": APPLE_RECIPE.lr,
        "feat_dim": APPLE_RECIPE.feat_dim,
        "n_way": APPLE_N_WAY,
        "k_support": APPLE_K_SUPPORT,
        "q_query": APPLE_Q_QUERY,
        "val_every": SNAP_EVERY,
        "val_episodes": VAL_EPISODES,
    }
    theirs = m64.LEGACY_CARD_PROTOCOL
    bad = {k: (v, theirs[k]) for k, v in mine.items() if theirs[k] != v}
    if bad:
        raise RuntimeError(f"训练设定与 64 号不一致（本脚本, 64 号）：{bad}")


def stage_budget(recipe: Recipe, stage: str) -> tuple[int, int]:
    """(训练情节数, 评估重采样次数)：formal 用正式值，pilot 各取 1/5。"""
    div = 1 if stage == "formal" else PILOT_DIV
    return recipe.episodes // div, REPS // div


def last_index(episodes: int) -> int:
    """训满 episodes 个情节时末次快照的序号（从 1 起）；86 号据此取与本脚本末次快照相同的验证情节。"""
    return -(-episodes // SNAP_EVERY)


# ═══════════════════════════════════════════════════════════════════════════════
# 一、全局随机数的核对
# ═══════════════════════════════════════════════════════════════════════════════
def global_rng_state() -> RngState:
    """四路全局随机数的当前状态：random、numpy.random、torch（CPU）、torch（各 CUDA 设备）。"""
    cuda = [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), cuda


def rng_streams_moved(before: RngState, after: RngState) -> list[str]:
    """两次状态之间被消耗过的全局随机数流的名字。"""
    moved: list[str] = []
    if before[0] != after[0]:
        moved.append("random")
    np_a, np_b = before[1], after[1]
    if not (np_a[0] == np_b[0] and np.array_equal(np_a[1], np_b[1]) and tuple(np_a[2:]) == tuple(np_b[2:])):
        moved.append("numpy.random")
    if not torch.equal(before[2], after[2]):
        moved.append("torch")
    cuda_a, cuda_b = before[3], after[3]
    if len(cuda_a) != len(cuda_b) or any(not torch.equal(a, b) for a, b in zip(cuda_a, cuda_b, strict=True)):
        moved.append("torch.cuda")
    return moved


@contextlib.contextmanager
def global_rng_untouched(where: str) -> Iterator[None]:
    """核对代码块前后四路全局随机数状态相同；有一路变了即报错。

    快照与验证都放在这个块里：它们若消耗了训练所用的随机数，带验证的轨迹就不再等于不带验证的轨迹。
    torch 一侧只核对 CPU 与 CUDA 的随机数状态；MPS 没有读状态的接口，不在核对之内。
    """
    before = global_rng_state()
    yield
    moved = rng_streams_moved(before, global_rng_state())
    if moved:
        raise RuntimeError(f"{where}：快照或验证消耗了全局随机数（{'、'.join(moved)}），训练轨迹已被改变")


# ═══════════════════════════════════════════════════════════════════════════════
# 二、验证情节与 A/B 划分（随机数只取自传入的流）
# ═══════════════════════════════════════════════════════════════════════════════
class EpisodeSource(Protocol):
    """训练情节采样器：protonet.EpisodeSampler 与 crossdataset_lib.ArrSampler 都是这个形状。"""

    def sample(self) -> Episode: ...


def apple_pool(data: AppleData) -> EpisodeSampler:
    """在给定的苹果集合上按 64 号的情节构成（3 类 × 5 支持 + 10 查询）建采样池。"""
    return EpisodeSampler(sorted(data), data, APPLE_N_WAY, APPLE_K_SUPPORT, APPLE_Q_QUERY)


def draw_apple_episode(pool: EpisodeSampler, rng: np.random.Generator) -> Episode:
    """从苹果验证池抽一个情节，随机数只取自 rng。

    构成同 protonet.EpisodeSampler.sample：抽 n_way 个产地；每个产地抽 k_support + q_query 个苹果
    （不足时循环取用），前 k_support 个作支持（全面位均值谱），其余作查询（随机一个面位）。
    pool 只借用它建好的索引，不调用它的 sample()（那会消耗全局 random）。
    """
    n_way = min(pool.n_way, len(pool.all_regions))
    regions = [pool.all_regions[int(i)] for i in rng.permutation(len(pool.all_regions))[:n_way]]
    n_total = pool.k_support + pool.q_query
    sup_x: list[NDArray] = []
    sup_y: list[int] = []
    qry_x: list[NDArray] = []
    qry_y: list[int] = []
    for label, reg in enumerate(regions):
        apples = pool.region_apples[reg]
        order = rng.permutation(len(apples))
        if len(order) < n_total:
            order = np.tile(order, n_total // len(order) + 1)
        chosen = [apples[int(i)] for i in order[:n_total]]
        for eid, aid in chosen[: pool.k_support]:
            sup_x.append(pool.apple_data[eid][aid]["X"].mean(axis=0))
            sup_y.append(label)
        for eid, aid in chosen[pool.k_support :]:
            faces = pool.apple_data[eid][aid]["X"]
            qry_x.append(faces[int(rng.integers(len(faces)))])
            qry_y.append(label)
    return (
        torch.from_numpy(np.stack(sup_x)).float(),
        torch.tensor(sup_y, dtype=torch.long),
        torch.from_numpy(np.stack(qry_x)).float(),
        torch.tensor(qry_y, dtype=torch.long),
        n_way,
    )


@dataclass(frozen=True)
class ArrayPool:
    """公开数据集的一个验证池：光谱矩阵与各参与类的行号。"""

    feats: NDArray
    rows: list[NDArray]


def array_pool(feats: NDArray, labels: NDArray, classes: list[int]) -> ArrayPool | None:
    """只收池内样本数 ≥ K+2 的类（与评估、与 74 号选参情节同一门槛）；不足 2 类时该池不可用。"""
    rows = [np.where(labels == c)[0] for c in classes]
    rows = [r for r in rows if len(r) >= PUBLIC_K_SUPPORT + 2]
    return ArrayPool(feats, rows) if len(rows) >= 2 else None


def draw_array_episode(pool: ArrayPool, rng: np.random.Generator) -> Episode:
    """从公开数据集的验证池抽一个情节，随机数只取自 rng。

    构成照 crossdataset_lib.ArrSampler：池里每个参与类都出现，每类 5 个支持、至多 10 个查询。
    与训练采样器的一处不同：某类不足 15 个样本时不循环取用，查询只取支持之外剩下的样本（≥2 个），
    支持与查询不重叠。
    """
    sup_x: list[NDArray] = []
    sup_y: list[int] = []
    qry_x: list[NDArray] = []
    qry_y: list[int] = []
    for label, rows in enumerate(pool.rows):
        perm = rows[rng.permutation(len(rows))]
        sup = perm[:PUBLIC_K_SUPPORT]
        qry = perm[PUBLIC_K_SUPPORT : PUBLIC_K_SUPPORT + PUBLIC_Q_QUERY]
        sup_x.append(pool.feats[sup])
        sup_y += [label] * len(sup)
        qry_x.append(pool.feats[qry])
        qry_y += [label] * len(qry)
    return (
        torch.from_numpy(np.concatenate(sup_x)).float(),
        torch.tensor(sup_y, dtype=torch.long),
        torch.from_numpy(np.concatenate(qry_x)).float(),
        torch.tensor(qry_y, dtype=torch.long),
        len(pool.rows),
    )


def split_half_apple(
    apple_data: AppleData, tgt_elems: list[str], rng: np.random.Generator
) -> tuple[AppleData, dict[str, set[Any]]]:
    """留出年各域（一域即一个产地）的苹果随机对半分，办法同 64 号 split_target_half，随机数取自 rng。

    返回 (A 半的 apple_data 子集, {产地: B 半的苹果编号集合})；奇数个时 B 多一个。
    """
    part_a: AppleData = {}
    part_b: dict[str, set[Any]] = {}
    for e in tgt_elems:
        ids = sorted(apple_data[e])
        perm = rng.permutation(len(ids))
        n_a = len(ids) // 2
        part_a[e] = {ids[i]: apple_data[e][ids[i]] for i in sorted(perm[:n_a])}
        part_b.setdefault(str(ELEM_INFO[e][1]), set()).update(ids[i] for i in perm[n_a:])
    return part_a, part_b


def split_half_public(yh: NDArray, classes: list[int], rng: np.random.Generator) -> tuple[NDArray, NDArray]:
    """留出域各类样本随机对半分，办法同 74 号 half_split，随机数取自 rng。

    返回 (A 半, B 半) 在留出域数组里的行号；奇数个时 B 多一个。
    """
    in_a = np.zeros(len(yh), dtype=bool)
    for c in classes:
        ids = np.where(yh == c)[0]
        perm = rng.permutation(len(ids))
        in_a[ids[perm[: len(ids) // 2]]] = True
    return np.where(in_a)[0], np.where(~in_a)[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 三、格：一个留出域的训练池、验证池与评估集
# ═══════════════════════════════════════════════════════════════════════════════
class Cell(ABC):
    """一格的数据：一个留出域的仅源域训练池、三个验证池与两种评估集。"""

    def __init__(
        self, scope: str, dataset: str, held: str, in_dim: int, n_classes: int, recipe: Recipe, seed: int
    ) -> None:
        self.scope = scope
        self.dataset = dataset
        self.held = held
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.recipe = recipe
        self.seed = seed
        self.note = ""
        # 格在随机数流里的编号按名字取 CRC32：与遍历顺序、是否只跑一部分数据集都无关
        self.key = zlib.crc32(f"{dataset}|{held}".encode())

    def rng(self, tag: int, idx: int = 0) -> np.random.Generator:
        """本格的一条独立随机数流（不经过全局随机数）。"""
        return np.random.default_rng([self.seed, STREAM, self.key, tag, idx])

    @abstractmethod
    def train_sampler(self) -> EpisodeSource:
        """仅源域的训练情节采样器（与正式配方相同，用全局随机数）。"""

    @abstractmethod
    def has_pool(self, pool: str) -> bool:
        """该验证池是否可用。"""

    @abstractmethod
    def val_episode(self, pool: str, rng: np.random.Generator) -> Episode:
        """从验证池抽一个情节，随机数只取自 rng。"""

    @abstractmethod
    def evaluate(self, encoder: nn.Module, subset: str, reps: int, device: torch.device) -> tuple[float, int] | None:
        """K=5 宏 F1 的均值与有效情节数；该评估集不可用时返回 None。"""

    @abstractmethod
    def linear_xy(self) -> tuple[NDArray, NDArray]:
        """源域训练池的全部光谱与类别（线性族拟合用，86 号）。"""


class AppleCell(Cell):
    """苹果：一个设计里的一个留出年份。"""

    def __init__(
        self,
        design: str,
        tgt_year: int,
        cfg: dict[str, Any],
        apple_data: AppleData,
        year_data: YearData,
        n_bands: int,
        seed: int,
    ) -> None:
        n_regions = len(year_data[tgt_year])
        super().__init__("apple", APPLE_SETS[design], f"Y{tgt_year}", n_bands, n_regions, APPLE_RECIPE, seed)
        self.tgt_year = tgt_year
        self.src_years: list[int] = list(cfg["src_years"])
        self.src: AppleData = {e: apple_data[e] for e in cfg["src"]}
        tgt: AppleData = {e: apple_data[e] for e in cfg["tgt"]}
        part_a, part_b = split_half_apple(apple_data, list(cfg["tgt"]), self.rng(TAG_SPLIT))
        self.pools: dict[str, EpisodeSampler] = {
            POOL_SRC: apple_pool(self.src),
            POOL_TGT: apple_pool(tgt),
            POOL_TGT_A: apple_pool(part_a),
        }
        self.year_data = year_data
        # B 半的评估集：留出年每个产地只留 B 半的苹果，其余年份不动（同 64 号 evaluate 的 restrict）
        held_b: dict[str, dict[Any, NDArray]] = {}
        for reg, apples in year_data[tgt_year].items():
            held_b[reg] = {a: x for a, x in apples.items() if a in part_b[reg]}
        self.year_data_b: YearData = {**year_data, tgt_year: held_b}
        n_a = sum(len(v) for v in part_a.values())
        n_b = sum(len(v) for v in part_b.values())
        self.note = f"A 半 {n_a} 果，B 半 {n_b} 果"

    def train_sampler(self) -> EpisodeSource:
        return apple_pool(self.src)

    def has_pool(self, pool: str) -> bool:
        return pool in self.pools

    def val_episode(self, pool: str, rng: np.random.Generator) -> Episode:
        return draw_apple_episode(self.pools[pool], rng)

    def evaluate(self, encoder: nn.Module, subset: str, reps: int, device: torch.device) -> tuple[float, int] | None:
        """照 64 号 evaluate：重采样种子 S+i，α=0；支持果按苹果编号抽，查询为留出年其余全部苹果。

        与 64 号的一处不同：均值不舍入（64 号的结果表存 4 位小数）。
        """
        year_data = self.year_data if subset == EVAL_ALL else self.year_data_b
        encoder.eval()
        # 每次评估新建特征缓存：同一个模块对象会先后载入不同的快照
        cache = bmc.FeatureCache(encoder, device, True)
        f1s = [
            float(bmc.kshot_eval(cache, year_data, self.tgt_year, self.src_years, K_SHOT, self.seed + i, 0.0)["f1"])
            for i in range(reps)
        ]
        return float(np.mean(f1s)), len(f1s)

    def linear_xy(self) -> tuple[NDArray, NDArray]:
        """照 64 号 fit_linear：训练池全部光谱（每果每面位一条），类别为所属苹果的产地。"""
        xs: list[NDArray] = []
        ys: list[str] = []
        for e in sorted(self.src):
            for aid in sorted(self.src[e]):
                faces = self.src[e][aid]["X"]
                xs.append(faces)
                ys += [str(self.src[e][aid]["region"])] * len(faces)
        return np.vstack(xs), np.asarray(ys)


class PublicCell(Cell):
    """公开数据集：一个数据集里的一个留出域。"""

    def __init__(self, dataset: str, held: str, fa: cdl.FoldArrays, seed: int) -> None:
        super().__init__("public", dataset, held, fa.in_dim, len(fa.classes), PUBLIC_RECIPE, seed)
        self.fa = fa
        a_idx, b_idx = split_half_public(fa.yh, fa.classes, self.rng(TAG_SPLIT))
        self.pools: dict[str, ArrayPool | None] = {
            POOL_SRC: array_pool(fa.xs, fa.ys, fa.classes),
            POOL_TGT: array_pool(fa.xh, fa.yh, fa.classes),
            POOL_TGT_A: array_pool(fa.xh[a_idx], fa.yh[a_idx], fa.classes),
        }
        # B 半的评估集同 74 号 run_fold：只评 B 半里样本数 ≥ K+2 的类，不足 3 类时不评
        xb, yb = fa.xh[b_idx], fa.yh[b_idx]
        classes_b = [c for c in fa.classes if int((yb == c).sum()) >= K_SHOT + 2]
        self.b_eval: tuple[NDArray, NDArray, list[int]] | None = None
        if len(classes_b) >= 3:
            keep = np.isin(yb, classes_b)
            self.b_eval = (xb[keep], yb[keep], classes_b)
        self.note = f"A 半 {len(a_idx)} 条，B 半 {len(b_idx)} 条"

    def train_sampler(self) -> EpisodeSource:
        return cdl.ArrSampler(self.fa.xs, self.fa.ys, PUBLIC_K_SUPPORT, PUBLIC_Q_QUERY)

    def has_pool(self, pool: str) -> bool:
        return self.pools.get(pool) is not None

    def val_episode(self, pool: str, rng: np.random.Generator) -> Episode:
        got = self.pools[pool]
        if got is None:
            raise ValueError(f"{self.dataset} 留出 {self.held}：验证池 {pool} 不可用")
        return draw_array_episode(got, rng)

    def evaluate(self, encoder: nn.Module, subset: str, reps: int, device: torch.device) -> tuple[float, int] | None:
        """照 74 号 run_fold：crossdataset_lib.eval_kshot，评估种子 = 种子；B 半用 B 半的评估类。"""
        if subset == EVAL_ALL:
            x, y, classes = self.fa.xh, self.fa.yh, self.fa.classes
        elif self.b_eval is None:
            return None
        else:
            x, y, classes = self.b_eval
        f1 = cdl.eval_kshot(encoder, x, y, classes, device, K_SHOT, reps, self.seed)
        # eval_kshot 的一次重复只在某类不足 K+2 个样本时作废，与抽到谁无关：要么全部有效，要么一次都没有
        return f1, (reps if np.isfinite(f1) else 0)

    def linear_xy(self) -> tuple[NDArray, NDArray]:
        return self.fa.xs, self.fa.ys


def build_cells(scope: str, chosen: list[str], data_root: str, seed: int, logger: eu.Logger) -> list[Cell]:
    """建一个范围的全部格。

    苹果：chosen 是设计名，每个设计的每个留出年份一格。公开数据集：chosen 是数据集名，
    每个留出域一格（合法类不足 3 个的折跳过，同 74 号）。
    """
    cells: list[Cell] = []
    if scope == "apple":
        df = m64.load_data("canonical", logger)
        wl_cols = eu.get_spectrum_columns(df)
        apple_data = build_apple_data(df, wl_cols)
        year_data = build_year_region_data(df, wl_cols)
        for design in chosen:
            for tgt_year, cfg in sorted(m64.DESIGNS[design].items()):
                cells.append(AppleCell(design, int(tgt_year), cfg, apple_data, year_data, len(wl_cols), seed))
    else:
        for dsname in chosen:
            meta = m74.DATASETS[dsname]
            doms, is_reg = meta["loader"](data_root, seed)
            logger.log(f"\n【{dsname}】{meta['drift']}；域 {[f'{d}:{len(v[1])}' for d, v in doms.items()]}")
            for held in list(doms.keys()):
                fa = cdl.prepare_fold(doms, is_reg, held, K_SHOT)
                if fa is None:
                    logger.log(f"    留出 {held}：合法类不足 3 个，此折不评")
                    continue
                cells.append(PublicCell(dsname, str(held), fa, seed))
    for cell in cells:
        logger.log(f"  [格] {cell.dataset} 留出 {cell.held}：{cell.n_classes} 类，{cell.note}")
    return cells


# ═══════════════════════════════════════════════════════════════════════════════
# 四、训练、验证与检查点选择
# ═══════════════════════════════════════════════════════════════════════════════
def val_score(encoder: nn.Module, cell: Cell, pool: str, idx: int, device: torch.device) -> float:
    """验证得分：VAL_EPISODES 个验证情节上查询准确率的均值（64 号选检查点所用的量）。

    情节取自本格的独立随机数流（用途 = 验证池，序号 = idx），不消耗全局随机数；idx 取快照序号，
    不同快照因此用不同的情节。该池不可用时返回 nan。
    """
    if not cell.has_pool(pool):
        return float("nan")
    rng = cell.rng(VAL_TAG[pool], idx)
    encoder.eval()
    accs: list[float] = []
    with torch.no_grad():
        for _ in range(VAL_EPISODES):
            sx, sy, qx, qy, nw = cell.val_episode(pool, rng)
            accs.append(proto_loss(encoder, sx, sy, qx, qy, nw, device)[1])
    return float(np.mean(accs))


def snapshot_state(encoder: nn.Module) -> dict[str, torch.Tensor]:
    """当前权重与缓冲区的深拷贝，放在内存里（CPU），不落盘。"""
    return {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}


def train_trajectory(
    cell: Cell,
    family: str,
    lr: float,
    episodes: int,
    device: torch.device,
    on_snapshot: Callable[[int, nn.Module], None] | None = None,
) -> nn.Module:
    """仅源域训练一条轨迹到 episodes 个情节，不按验证选优、不早停；返回末次权重的编码器（eval 模式）。

    苹果照 64 号 train_encoder、公开数据集照 crossdataset_lib.meta_train 逐行写出，两者的差别都收在
    cell.recipe（特征维、权重衰减、余弦退火、梯度裁剪）与训练采样器里。
    全局随机数的消耗只有两处：sampler.sample()（苹果 random，公开数据集 numpy.random）与训练模式下
    前向里的随机丢弃（torch）。on_snapshot 每 SNAP_EVERY 个情节与最后一个情节各调用一次，调用前后
    核对全局随机数状态未变，所以给不给 on_snapshot、它里面验证多少个情节，轨迹都逐位相同。
    """
    recipe = cell.recipe
    eu.set_random_seed(cell.seed)  # 每条轨迹重新播种：初始化与训练采样流只由种子与训练池决定
    sampler = cell.train_sampler()  # 建池不消耗随机数
    encoder = enc_lib.build_encoder(family, cell.in_dim, recipe.feat_dim).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=recipe.weight_decay)
    scheduler: CosineAnnealingLR | None = None
    if recipe.eta_min is not None:
        scheduler = CosineAnnealingLR(optimizer, T_max=episodes, eta_min=recipe.eta_min)
    for ep in range(1, episodes + 1):
        encoder.train()
        sx, sy, qx, qy, nw = sampler.sample()
        optimizer.zero_grad()
        loss, _ = proto_loss(encoder, sx, sy, qx, qy, nw, device)
        loss.backward()  # type: ignore[no-untyped-call]
        if recipe.clip_norm is not None:
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=recipe.clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if on_snapshot is not None and (ep % SNAP_EVERY == 0 or ep == episodes):
            with global_rng_untouched(f"{cell.dataset} 留出 {cell.held} {family} 第 {ep} 情节"):
                on_snapshot(ep, encoder)
    encoder.eval()
    return encoder


def first_argmax(values: Sequence[float]) -> int | None:
    """最大值第一次出现的位置（并列取最早）；没有有限值时返回 None。"""
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).any():
        return None
    return int(np.nanargmax(arr))


def run_cell(
    cell: Cell, family: str, stage: str, device: torch.device, selfcheck: bool, logger: eu.Logger
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """一格：训一条仅源域的轨迹并逐快照验证，按各选法取检查点、评估。

    返回 (逐选法记录, 逐快照验证记录)。
    """
    episodes, reps = stage_budget(cell.recipe, stage)
    snaps: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []

    def on_snapshot(ep: int, enc: nn.Module) -> None:
        idx = len(snaps) + 1
        snaps.append(snapshot_state(enc))
        scores = {VAL_COL[p]: val_score(enc, cell, p, idx, device) for p in POOLS}
        curve.append({"快照序号": idx, "情节": ep, **scores})

    encoder = train_trajectory(cell, family, cell.recipe.lr, episodes, device, on_snapshot)
    if selfcheck:
        plain = train_trajectory(cell, family, cell.recipe.lr, episodes, device)
        differ = [k for k, v in plain.state_dict().items() if not torch.equal(v.detach().cpu(), snaps[-1][k])]
        if differ:
            raise RuntimeError(
                f"{cell.dataset} 留出 {cell.held} {family}：不带快照与验证重训后，"
                f"末次权重有 {len(differ)} 个张量不同（如 {differ[0]}），快照或验证改变了训练轨迹"
            )
        logger.log("      自检通过：不带快照与验证重训，末次权重逐位相同")

    ident: dict[str, Any] = {
        "范围": cell.scope,
        "dataset": cell.dataset,
        "held": cell.held,
        "编码器": family,
        "种子": cell.seed,
    }
    base: dict[str, Any] = {**ident, "n_classes": cell.n_classes, "训练情节数": episodes, "快照数": len(snaps)}
    evals: dict[tuple[int, str], tuple[float, int] | None] = {}
    rows: list[dict[str, Any]] = []
    for name, (pool, subset) in SELECTIONS.items():
        pick = len(snaps) - 1 if pool is None else first_argmax([float(c[VAL_COL[pool]]) for c in curve])
        if pick is None:
            continue
        # 同一个快照在同一评估集上只评一次：两种选法选中同一个快照时，差值恰为 0
        if (pick, subset) not in evals:
            encoder.load_state_dict(snaps[pick])
            evals[(pick, subset)] = cell.evaluate(encoder, subset, reps, device)
        got = evals[(pick, subset)]
        if got is None:
            continue
        rows.append(
            {
                **base,
                "检查点选法": name,
                "评估样本": subset,
                "所选情节数": curve[pick]["情节"],
                **{VAL_COL[p]: curve[pick][VAL_COL[p]] for p in POOLS},
                "F1": got[0],
                "情节数": got[1],
            }
        )
    shown = "  ".join(f"{r['检查点选法']}={r['F1']:.4f}@{r['所选情节数']}" for r in rows)
    logger.log(f"    [{family}] {cell.dataset} 留出 {cell.held}：{shown}")
    if not any(r["检查点选法"] == "tgt_half_best" for r in rows):
        logger.log("      半分版不评：A 半可用于验证的类不足 2 个，或 B 半里样本数 ≥ K+2 的类不足 3 个")
    return rows, [{**ident, **c} for c in curve]


# ═══════════════════════════════════════════════════════════════════════════════
# 五、结果文件与汇总（86 号共用）
# ═══════════════════════════════════════════════════════════════════════════════
def stage_suffix(stage: str) -> str:
    return "" if stage == "formal" else "__pilot"


def seed_path(stem: str, tag: str, family: str, seed: int, stage: str) -> str:
    """逐种子结果表：formal_seeds/<stem>__<范围段>__<编码器>__seed<S>[__pilot].xlsx。"""
    return os.path.join(SEED_DIR, f"{stem}__{tag}__{family}__seed{seed}{stage_suffix(stage)}.xlsx")


def summary_path(stem: str, stage: str) -> str:
    """汇总表：04outputs/<stem>[__pilot].xlsx。"""
    return os.path.join(eu.OUTPUT_DIR, f"{stem}{stage_suffix(stage)}.xlsx")


def write_workbook(path: str, tables: Sequence[tuple[str, pd.DataFrame]], compute: pd.DataFrame) -> None:
    """各表按「表a、表b…」的顺序写入，末尾写 compute 表。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for i, (title, df) in enumerate(tables):
            df.to_excel(writer, sheet_name=eu.table_sheet_name(i, title), index=False)
        compute.to_excel(writer, sheet_name="compute", index=False)


def seed_files(stem: str, stage: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(SEED_DIR, f"{stem}__*__seed*.xlsx")))
    return [f for f in files if ("__pilot" in os.path.basename(f)) == (stage == "pilot")]


def load_seed_sheet(stem: str, stage: str, sheet: int | str) -> pd.DataFrame:
    """把该阶段全部逐种子结果表的同一张工作表拼起来，另加「来源文件」一列。"""
    files = seed_files(stem, stage)
    if not files:
        raise FileNotFoundError(f"没有 {stem} 的逐种子结果（{stage}）")
    frames = []
    for f in files:
        frame = pd.read_excel(f, sheet_name=sheet)
        frame["来源文件"] = os.path.basename(f)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_unique(df: pd.DataFrame, keys: list[str], what: str) -> None:
    """同一格不得出现在两个结果表里（比如范围段不同的两次运行重叠了）；有重复即中止。"""
    dup = df[df.duplicated(keys, keep=False)]
    if len(dup):
        files = sorted({str(f) for f in dup["来源文件"]})
        raise ValueError(f"{what}：有 {len(dup)} 行的格重复出现，先清掉其中过期的结果表：{files}")


def warn_seeds(df: pd.DataFrame, stage: str, logger: eu.Logger) -> None:
    """种子集合或各格的种子数不齐时在日志里提醒（照常汇总，只作进度查看）。"""
    want = set(eu.RECOMMENDED_SEEDS) if stage == "formal" else {eu.PILOT_SEED}
    seeds = sorted(int(s) for s in df["种子"].unique())
    if set(seeds) != want:
        logger.log(f"  [WARN] 种子集合 {seeds} 不是 {stage} 应有的 {sorted(want)}，汇总仅作进度查看")
    per_fold = df.groupby(["dataset", "held", "编码器"])["种子"].nunique()
    short = int((per_fold < len(want)).sum())
    if short:
        logger.log(f"  [WARN] {short} 个（数据集, 留出域, 编码器）的种子数不足 {len(want)}")


def t_interval(values: Sequence[float]) -> tuple[float, float]:
    """种子间 95% t 区间：均值 ± t(0.975, n−1)·SD/√n（同 74 号）；不足 2 个值时返回 nan。"""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float("nan"), float("nan")
    half = float(stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr)))
    return float(arr.mean() - half), float(arr.mean() + half)


def cell_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """一行一格（数据集 × 留出域 × 编码器 × 种子）：各选法的宏 F1 与所选情节数，种子内配对作差。"""
    rows: list[dict[str, Any]] = []
    for (scope, ds, held, enc, seed), sub in long_df.groupby(["范围", "dataset", "held", "编码器", "种子"]):
        f1 = {str(s): float(v) for s, v in zip(sub["检查点选法"], sub["F1"], strict=True)}
        ep = {str(s): int(v) for s, v in zip(sub["检查点选法"], sub["所选情节数"], strict=True)}
        row: dict[str, Any] = {"范围": scope, "dataset": ds, "held": held, "编码器": enc, "种子": int(seed)}
        for sel in SELECTIONS:
            if sel in f1:
                row[f"F1_{sel}"] = f1[sel]
                row[f"情节_{sel}"] = ep[sel]
        for name, (a, b) in CONTRASTS.items():
            if a in f1 and b in f1:
                row[name] = f1[a] - f1[b]
                row[f"{name}_同选"] = ep[a] == ep[b]
        rows.append(row)
    return pd.DataFrame(rows)


def domain_table(cells: pd.DataFrame, deltas: Sequence[str], means: Sequence[str]) -> pd.DataFrame:
    """逐（数据集, 留出域, 编码器）：means 各列的种子均值；各对照的种子均值、种子 t 区间、为正种子数。

    cells 一行一格，含各对照列。有「<对照>_同选」列时，另计两种选法选中同一个对象的种子数
    （这些种子的差值恰为 0）。
    """
    rows: list[dict[str, Any]] = []
    for (ds, held, enc), sub in cells.groupby(["dataset", "held", "编码器"]):
        row: dict[str, Any] = {"dataset": ds, "held": held, "编码器": enc, "种子数": int(sub["种子"].nunique())}
        for col in means:
            if col in sub:
                row[col] = float(sub[col].mean())
        for name in deltas:
            if name not in sub:
                continue
            ok = sub[sub[name].notna()]
            diffs = [float(v) for v in ok[name]]
            if not diffs:
                continue
            lo, hi = t_interval(diffs)
            row[name] = float(np.mean(diffs))
            row[f"{name}_种子t区间"] = f"{lo:.4f}–{hi:.4f}"
            row[f"{name}_为正种子数"] = int(sum(d > 0 for d in diffs))
            row[f"{name}_种子数"] = len(diffs)
            if f"{name}_同选" in ok:
                row[f"{name}_同选种子数"] = int(ok[f"{name}_同选"].astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def family_table(dom: pd.DataFrame, deltas: Sequence[str], logger: eu.Logger) -> pd.DataFrame:
    """按口径 × 编码器族的跨留出域汇总（主口径与苹果三年设计分开）。

    各对照的均值 = 各留出域种子均值的均值；为正留出域 = 种子均值大于 0 的留出域数 / 有该对照的留出域数。
    """
    rows: list[dict[str, Any]] = []
    for scope_name, sets in SCOPE_SETS.items():
        in_scope = dom[dom["dataset"].isin(sets)]
        for enc in enc_lib.ENCODERS:
            sub = in_scope[in_scope["编码器"] == enc]
            if sub.empty:
                continue
            want = SCOPE_DOMAINS[scope_name]
            row: dict[str, Any] = {"口径": scope_name, "编码器": enc, "留出域数": len(sub), "应有留出域数": want}
            for name in deltas:
                if name not in sub:
                    continue
                vals = sub[name].dropna()
                row[f"{name}_均值"] = float(vals.mean()) if len(vals) else float("nan")
                row[f"{name}_为正留出域"] = f"{int((vals > 0).sum())}/{len(vals)}"
            rows.append(row)
            if len(sub) != want:
                logger.log(f"  [WARN] {scope_name} × {enc}：有 {len(sub)} 个留出域，应有 {want} 个")
    return pd.DataFrame(rows)


def check_against_74(long_df: pd.DataFrame, logger: eu.Logger) -> pd.DataFrame:
    """公开数据集上 last 与 74 号 R 臂的逐格核对。

    两者是同一配方、同一种子的仅源域训练，评估情节也相同；同机同环境下宏 F1 应逐位相同。
    只列出差值，不中止：74 号正式表可能出自另一台机器。没有 74 号的结果或没有重合的格时返回空表。
    """
    try:
        ref = m74.load_long("formal")
    except FileNotFoundError:
        logger.log("  [核对] 没有 74 号的正式逐种子结果，未与 74 号核对")
        return pd.DataFrame()
    ref = ref[(ref["臂"] == "R") & (ref["评估样本"] == EVAL_ALL)]
    mine = long_df[(long_df["范围"] == "public") & (long_df["检查点选法"] == "last")]
    both = mine.merge(ref, on=KEYS, suffixes=("", "_74"))
    if both.empty:
        logger.log("  [核对] 本脚本与 74 号没有重合的格，未核对")
        return pd.DataFrame()
    out = both[KEYS].copy()
    out["F1_85_last"] = both["F1"]
    out["F1_74_R"] = both["F1_74"]
    out["绝对差"] = (both["F1"] - both["F1_74"]).abs()
    out["一致"] = out["绝对差"] <= CHECK_TOL
    bad = out[~out["一致"]]
    logger.log(f"  [核对] 与 74 号 R 臂比对 {len(out)} 格，最大绝对差 {float(out['绝对差'].max()):.2e}")
    if len(bad):
        logger.log(f"  [WARN] {len(bad)} 格与 74 号 R 臂不一致（容差 {CHECK_TOL:g}），前 30 格：")
        logger.log(bad.head(30).to_string(index=False))
    return out


def aggregate(stage: str, logger: eu.Logger) -> None:
    """汇总逐种子结果：逐留出域的 Δ_ckpt 与 Δ_ckpt_half、按族的跨留出域汇总，正式结果另与 74 号核对。"""
    long_df = load_seed_sheet(STEM, stage, 0)
    check_unique(long_df, [*KEYS, "检查点选法"], "85 号逐种子结果")
    warn_seeds(long_df, stage, logger)
    cells = cell_table(long_df)
    dom = domain_table(cells, list(CONTRASTS), [f"F1_{sel}" for sel in SELECTIONS])
    fam = family_table(dom, list(CONTRASTS), logger)
    logger.log("按口径 × 编码器族\n" + fam.to_string(index=False))
    # pilot 的情节预算不同，last 与 74 号的 R 臂不是同一个量，不核对
    check = check_against_74(long_df, logger) if stage == "formal" else pd.DataFrame()
    out = summary_path(STEM, stage)
    tables = [
        ("逐留出域", dom),
        ("按族汇总", fam),
        ("逐格", cells),
        ("长格式", long_df),
        ("与74号R臂核对", check),
    ]
    write_workbook(out, tables, load_seed_sheet(STEM, stage, "compute"))
    logger.log(f"汇总表 → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 六、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def make_parser(description: str, encoders: Sequence[str]) -> argparse.ArgumentParser:
    """85、86 号共用的命令行；--stage 不给时，运行按 pilot、--aggregate 按 formal。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", default=None, choices=["pilot", "formal"])
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED, help="一次一个种子")
    parser.add_argument("--scope", default="all", choices=["apple", "public", "all"])
    parser.add_argument("--designs", default=",".join(APPLE_SETS), help="苹果的设计，逗号分隔")
    parser.add_argument("--datasets", default=",".join(m74.DATASETS), help="公开数据集，逗号分隔")
    parser.add_argument("--encoders", default=",".join(encoders), help="编码器族，逗号分隔")
    parser.add_argument("--data_root", default=DATA_ROOT, help="公开数据集根目录")
    parser.add_argument("--aggregate", action="store_true", help="汇总 formal_seeds 下已有的逐种子结果")
    return parser


def parse_subset(text: str, full: Sequence[str], what: str) -> list[str]:
    """逗号分隔的名单 → 按 full 的顺序排好的子集；有不认识的名字即退出。"""
    asked = [t for t in text.split(",") if t]
    unknown = sorted(set(asked) - set(full))
    if unknown:
        raise SystemExit(f"未知{what}：{unknown}（可选 {list(full)}）")
    return [t for t in full if t in asked]


def scope_plan(scope: str, designs: str, datasets: str) -> list[tuple[str, str, list[str]]]:
    """命令行的范围 → [(范围, 结果文件名里的范围段, 要跑的设计或数据集)]。

    跑全时范围段就是范围名；只跑一部分时带上所跑的名字，不完整的表不会顶替完整的表。
    """
    full: dict[str, list[str]] = {"apple": list(APPLE_SETS), "public": list(m74.DATASETS)}
    asked = {
        "apple": parse_subset(designs, full["apple"], "设计"),
        "public": parse_subset(datasets, full["public"], "数据集"),
    }
    plan: list[tuple[str, str, list[str]]] = []
    for name in ("apple", "public") if scope == "all" else (scope,):
        chosen = asked[name]
        if not chosen:
            raise SystemExit(f"范围 {name} 没有要跑的项")
        tag = name if chosen == full[name] else f"{name}-{'+'.join(chosen)}"
        plan.append((name, tag, chosen))
    return plan


def main() -> None:
    parser = make_parser("85_checkpoint_paired：同一条训练轨迹上的两种检查点选法", enc_lib.DEEP_ENCODERS)
    parser.add_argument("--selfcheck", action="store_true", help="每格另训一条不带快照与验证的轨迹，核对末次权重")
    args = parser.parse_args()
    args.stage = args.stage or ("formal" if args.aggregate else "pilot")
    logger = eu.get_logger(STEM)
    if args.aggregate:
        aggregate(args.stage, logger)
        logger.close()
        return

    families = parse_subset(args.encoders, enc_lib.DEEP_ENCODERS, "编码器")
    plan = scope_plan(args.scope, args.designs, args.datasets)
    check_recipe()
    device = eu.get_device(args.device)
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    ep_apple, reps = stage_budget(APPLE_RECIPE, args.stage)
    ep_public, _ = stage_budget(PUBLIC_RECIPE, args.stage)
    eu.log_experiment_header(
        logger,
        {
            "任务": "同一条训练轨迹上的两种检查点选法",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "范围": [f"{tag}：{chosen}" for _, tag, chosen in plan],
            "编码器": families,
            "训练情节（苹果 / 公开数据集）": f"{ep_apple} / {ep_public}",
            "快照间隔 / 每池验证情节": f"{SNAP_EVERY} / {VAL_EPISODES}",
            "K / 每格评估重采样": f"{K_SHOT} / {reps}",
            "检查点选法": list(SELECTIONS),
            "自检（不带快照与验证重训比对）": args.selfcheck,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))
    for scope, tag, chosen in plan:
        cells = build_cells(scope, chosen, args.data_root, args.seed, logger)
        for family in families:
            logger.log(f"\n{'=' * 60}\n【{tag} × {family}】{enc_lib.ARCH_DESC[family]}")
            rows: list[dict[str, Any]] = []
            curves: list[dict[str, Any]] = []
            with eu.track_compute(f"配对检查点 {tag} × {family}", logger, run) as budget:
                for cell in cells:
                    cell_rows, cell_curve = run_cell(cell, family, args.stage, device, args.selfcheck, logger)
                    rows += cell_rows
                    curves += cell_curve
            out = seed_path(STEM, tag, family, args.seed, args.stage)
            compute = pd.DataFrame(
                [{"seed": args.seed, "stage": args.stage, "scope": tag, "encoder": family, **budget}]
            )
            tables = [("逐检查点选法结果", pd.DataFrame(rows)), ("逐快照验证得分", pd.DataFrame(curves))]
            write_workbook(out, tables, compute)
            logger.log(f"结果表 → {out}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
