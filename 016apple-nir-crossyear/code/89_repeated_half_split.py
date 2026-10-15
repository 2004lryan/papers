"""
89_repeated_half_split.py: 重复半分割——半数暴露对照 Δ_half 与份额在划分之间的范围，及种子与划分的方差分量。

为什么:
    64 号的 S_half 臂对每个（种子, 留出年）只把留出年的苹果对半分一次（A 半进训练池，B 半只作评估），
    划分与种子绑在一起：五个正式种子就是五次不同的划分，Δ_half 的种子间波动里分不出哪部分来自划分。
    稿件里两年设计、一维卷积编码器的 Δ_half（0.222）及其占 B 半上目标暴露效应的份额（79%）都出自
    这一次划分。本脚本在同一（种子, 留出年）上把对半分重复若干次，每次各训一个 S_half 编码器，给出
    Δ_half 与份额在划分之间的范围，以及种子与划分各自的方差分量。

设计（每次运行一个种子；设计 × 编码器族依次跑，各写一张逐种子表）:
    范围      --designs（缺省只有同仪器两年设计 two_year，2 个留出年）× --encoders（缺省四个深度族
              cnn、mlp、resnet、transformer）；数据为冻结数据（canonical）
    划分      每个（种子, 留出年）做 --n_splits 次（formal 缺省 5），编号 r = 0, 1, …；都是苹果编号级、
              按产地分层的对半分，奇数个时 B 多一个
              r = 0  调 64 号 split_target_half(apple_data, 留出年各域, 种子 + 留出年)，与 64 号在同一
                     （种子, 留出年）上实际用的 A/B 划分逐苹果相同
              r ≥ 1  同一算法，随机数在 64 号的熵序列之后多加一维 r：
                     np.random.default_rng([种子 + 留出年, 64, 11, r])
              本脚本的对半分实现 half_split 与 64 号逐行同构；每次运行都用它以 [种子 + 留出年, 64, 11]
              复算 r = 0，并与 64 号函数的结果逐苹果比对，不一致即中止
    臂        S_half   训练池 = 源年份 ∪ 留出年 A_r 半（带标签），源年份情节选检查点；每个 r 各训一个
              R^h      折限定编码器 R（训练池 = 源年份）在 B_r 半上的评估
              S_src^h  目标暴露编码器 S_src（训练池 = 源年份 ∪ 留出年全部，带标签）在 B_r 半上的评估
              训练都调 64 号 train_encoder，设定取 64 号的命令行默认值：情节预算 2000、耐心 12、每 100
              情节验证 30 个情节、学习率 1e-3、每情节 3 类 × (5 支持 + 10 查询)、特征维 128
    R、S_src  每个留出年各一个，不随 r 变。先按 64 号的 run_name 在 07checkpoints/64_* 里找正式权重：
              模型卡逐项与本次设定一致、训练已走完、权重 SHA-256 与卡相符才载入（只读）；找不到或
              核对不过就按同一配方重训，新权重写 07checkpoints/89_*。本脚本不往 64_* 目录写任何东西
    评估      调 64 号 evaluate（restrict 参数）：K=5、融合权重 0，B_r 半里每个产地按苹果编号抽 5 个
              支持果，其余 B_r 半苹果作查询；formal 50 个情节，重采样种子 S+i。同一个 r 上三臂用同一批情节
    pilot     情节预算 400、评估情节 10、划分 2 次；run_name 加 _pilot，结果表名加 __pilot，不进正式汇总
    随机数    除 r ≥ 1 的划分外没有新增的抽样；训练与评估的播种沿用 64 号（每个编码器训练前
              set_random_seed(种子)，评估第 i 个情节用 default_rng(种子 + i)）

统计量（--aggregate；宏 F1 取逐种子表里的 4 位小数值，与 64 号汇总同一精度；「均值与划分间范围」表按全精度
写出，取用方显示时只舍入一次）:
    逐格        每个（留出年, 编码器族, 种子, r）：Δ_half = S_half − R^h，Δ_src^h = S_src^h − R^h，
                份额 = Δ_half / Δ_src^h；Δ_src^h 不大于 0 时份额记缺失并计数
    各折合并    每个（种子, r）先对各留出年取均值，再算 Δ 与份额。与 64 号 aggregate 的「均值」行、
                稿件里 0.222 与 79% 同口径：对照先在折上取均值、再对种子取均值，份额是两个均值之比
    均值        每个（留出年或各折合并, 编码器族）在全部种子 × 划分上的 Δ_half 均值。份额给两种：
                均值之比 = Δ_half 均值 / Δ_src^h 均值（稿件口径）；逐格均值 = 各格份额的均值（缺失格不计）
    划分间范围  先在每个 r 上对种子取均值，再取各 r 的最小与最大（Δ_half 与两种份额各一组）
    方差分量    把格值排成 种子 × 划分 的表 y[s, r]（S 个种子、R 次划分；表不完整时不估计），记 m_s、
                m_r、m 为行均值、列均值、总均值：
                    MS_种子   = R·Σ_s (m_s − m)² / (S − 1)
                    MS_划分   = S·Σ_r (m_r − m)² / (R − 1)
                    MS_残差   = Σ_sr (y[s, r] − m_s − m_r + m)² / ((S − 1)(R − 1))
                    MS_种子内 = Σ_sr (y[s, r] − m_s)² / (S(R − 1))
                交叉分解 y = μ + a_s + b_r + e_sr：种子间 (MS_种子 − MS_残差)/R，
                    划分间 (MS_划分 − MS_残差)/S，残差 MS_残差
                嵌套分解 y = μ + a_s + c_r(s)：种子间 (MS_种子 − MS_种子内)/R，种子内划分间 MS_种子内
                矩估计可以为负，照原值写出，表示该分量与 0 分不开
                读法：每个种子各有一组划分（划分的随机数含种子，r = 0 与 64 号一致），同一个 r 在不同
                种子下不是同一批苹果，r 不是各种子共有的因子水平。划分的作用看嵌套分解的「种子内
                划分间」；交叉分解的「划分间」只量各种子在同一编号上共有的偏移，期望为 0，其「残差」
                里含划分效应。两种分解并列给出，引用划分方差时取嵌套分解
    r = 0 自检  r = 0 的三臂与 64 号正式逐种子表里同一（留出年, 种子）的 S_half、R_B、S_src_B 逐格并排，
                相差超过 0.02 的格单列一表并写进日志；另核对同一（留出年, 种子, r）在各编码器族上的
                A 半指纹是否相同

运行方式:
    cd <项目根>
    # pilot（缺省种子 2004；可在非 CUDA 设备上跑，只检查流程）
    python 02code/89_repeated_half_split.py --device auto --stage pilot
    # formal（强制 CUDA，每次一个种子；四个族也可用 --encoders 拆成多个进程）
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/89_repeated_half_split.py \
        --device cuda --stage formal --seed 20060515
    # 中断后续跑：加 --reuse_ckpt，本脚本此前落盘且核对通过的 89_* 权重直接载入，不重训
    # 汇总（五个正式种子都有之后；加 --stage pilot 只读 pilot 结果，汇总表名加 __pilot）
    python 02code/89_repeated_half_split.py --aggregate

输入:
    03data/processed/all_9elements.csv 与同目录的数据卡（经 64 号 load_data 读入并核对 SHA-256）
    07checkpoints/64_<设计>_canonical[_<编码器>]_{R,S_src}_seed<S>_fold<年>/（有则只读载入）
    04outputs/formal_seeds/64_leakage__<设计>__canonical[__<编码器>]__seed<S>.xlsx（r = 0 自检用）

输出文件:
    04outputs/formal_seeds/89_repeated_half_split__<设计>__<编码器>__seed<S>[__pilot].xlsx
        — 逐划分逐臂的宏 F1（一行一格）+ 编码器来源与训练摘要 + compute
    04outputs/89_repeated_half_split.xlsx（--aggregate）
        — 逐格 Δ 与份额、各划分的种子均值、均值与划分间范围、方差分量、r = 0 对照、说明、compute
    07checkpoints/89_<设计>_canonical_<编码器>_<臂>[_h<r>]_seed<S>_fold<年>[_pilot]/best.pt — 新训的编码器
    05logs/89_repeated_half_split_YY-MM-DD_HHMMSS.log
    不写入、不覆盖 64、72 号的结果表与 07checkpoints/64_*。

复用:
    64 号  DESIGNS、LEGACY_CARD_PROTOCOL、load_data、split_target_half、train_encoder、
           obtain_encoder（复用核对）、evaluate（restrict 参数；内部调 44 号 FeatureCache、kshot_eval）
    72 号  load_encoders 的做法：把 64 号的训练函数换成拒绝训练的函数，借 obtain_encoder 核对后只读载入
    encoders.py      DEEP_ENCODERS、ARCH_DESC（build_encoder 经 64 号调用）
    protonet.py      ELEM_INFO、build_apple_data、build_year_region_data
    export_utils.py  设备与播种、Formal 的 CUDA 闸、日志、算力记录、工作表命名、一脚本一 xlsx
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn

import encoders as enc_lib
import export_utils as eu
from protonet import ELEM_INFO, build_apple_data, build_year_region_data

NDArray = npt.NDArray[Any]
Pool = dict[str, dict[Any, dict[str, Any]]]  # apple_data 的结构：{域: {苹果编号: {"X", "region"}}}
m64: Any = importlib.import_module("64_leakage_controls")
_TRAIN64 = m64.train_encoder  # 64 号的训练函数本体；本脚本只用以 89_ 打头的 run_name 调它

STEM = "89_repeated_half_split"
SEED_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
OWN_PREFIX = "89_"
VARIANT = "canonical"
# 训练设定里不在 m64.LEGACY_CARD_PROTOCOL 中的几项取 64 号的命令行默认值；pilot 约取 1/5
EPISODES, REPS, K_SHOT, N_SPLITS = 2000, 50, 5, 5
PILOT_EPISODES, PILOT_REPS, PILOT_N_SPLITS = 400, 10, 2

ARM_HALF, ARM_R, ARM_SRC = "S_half", "R^h", "S_src^h"
EVAL_ARMS = (ARM_HALF, ARM_R, ARM_SRC)
D_HALF, D_SRC, SHARE = "Δ_half", "Δ_src^h", "份额"
SHARE_RATIO, SHARE_CELL = "份额_均值之比", "份额_逐格均值"
MERGED = "各折合并"
KEYS = ["设计", "留出年", "编码器族", "种子", "划分编号"]
CELL_COLS = [*KEYS, *EVAL_ARMS, D_HALF, D_SRC, SHARE, "A半苹果数", "B半苹果数", "A半指纹"]
VAR_KEYS = ("交叉_种子间方差", "交叉_划分间方差", "交叉_残差方差", "嵌套_种子间方差", "嵌套_种子内划分间方差")

# r = 0 自检所读的 64 号逐种子表：文件名、工作表名、列名与臂名都取自 64 号 main、evaluate 的写表代码
REF64_FILE = "64_leakage__{design}__{variant}{enc_seg}__seed{seed}{suffix}.xlsx"
REF64_SHEET = eu.table_sheet_name(0, "K5各臂结果")
REF64_COLS = ("臂", "目标年份", "src_alpha", "K", "F1均值")
REF64_ARMS = {ARM_HALF: "S_half", ARM_R: "R_B", ARM_SRC: "S_src_B"}
R0_TOL = 0.02
CMP_COLS = ["设计", "留出年", "编码器族", "种子", "臂", "本脚本r0", "64号逐种子表", "差", "超过容差", "备注"]


# ═══════════════════════════════════════════════════════════════════════════════
# 一、划分：r = 0 重现 64 号的划分，r ≥ 1 在随机数的熵序列里多加一维
# ═══════════════════════════════════════════════════════════════════════════════
def half_split(apple_data: Pool, tgt_elems: list[str], entropy: list[int]) -> tuple[Pool, dict[str, set[Any]]]:
    """与 64 号 split_target_half 逐行同构的对半分，只把随机数的熵序列换成参数。

    留出年各域（一域即一个产地）的苹果编号排序后随机置换，前一半为 A、其余为 B（奇数个时 B 多一个）；
    各域依次消耗同一个随机数发生器。返回 (A 半的 apple_data 子集, {产地: B 半的苹果编号集合})。
    """
    rng = np.random.default_rng(entropy)
    part_a: Pool = {}
    part_b: dict[str, set[Any]] = {}
    for e in tgt_elems:
        ids = sorted(apple_data[e])
        perm = rng.permutation(len(ids))
        n_a = len(ids) // 2
        part_a[e] = {ids[i]: apple_data[e][ids[i]] for i in sorted(perm[:n_a])}
        part_b.setdefault(str(ELEM_INFO[e][1]), set()).update(ids[i] for i in perm[n_a:])
    return part_a, part_b


def split_for(apple_data: Pool, tgt_elems: list[str], seed_year: int, r: int) -> tuple[Pool, dict[str, set[Any]]]:
    """划分编号 r 的 (A 半, B 半)；seed_year = 种子 + 留出年，即 64 号传给 split_target_half 的种子参数。"""
    if r > 0:
        return half_split(apple_data, tgt_elems, [seed_year, 64, 11, r])
    part_a, part_b = m64.split_target_half(apple_data, tgt_elems, seed_year)
    mine_a, mine_b = half_split(apple_data, tgt_elems, [seed_year, 64, 11])
    same_a = {e: list(v) for e, v in part_a.items()} == {e: list(v) for e, v in mine_a.items()}
    if not same_a or part_b != mine_b:
        raise RuntimeError("half_split 与 64 号 split_target_half 给出的 r = 0 划分不同：两者已不同构，停止")
    return part_a, part_b


def fingerprint(part_a: Pool) -> str:
    """A 半成员的指纹：各域的苹果编号依次拼接后取 SHA-256 的前 16 位，写进结果表与 S_half 的模型卡。"""
    text = "|".join(f"{e}:{aid}" for e in sorted(part_a) for aid in part_a[e])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# 二、编码器：正式权重只读载入，载不到就按 64 号配方训练（只写 89_* 目录）
# ═══════════════════════════════════════════════════════════════════════════════
class _NotReusableError(Exception):
    """64 号 obtain_encoder 的复用核对没过（它随后要调的训练函数已换成抛出本异常的函数）。"""


def _refuse_training(run_name: str, *_: Any, **__: Any) -> Any:
    raise _NotReusableError(run_name)


@dataclass(frozen=True)
class Ctx:
    """一个（设计, 编码器族）里各次载入、训练与评估共用的环境。"""

    a64: argparse.Namespace
    n_bands: int
    device: torch.device
    logger: eu.Logger
    resume: bool

    @property
    def tail(self) -> tuple[int, argparse.Namespace, torch.device, eu.Logger]:
        """64 号 train_encoder、obtain_encoder 的第五到第八个位置参数。"""
        return self.n_bands, self.a64, self.device, self.logger


def args_for_64(args: argparse.Namespace, design: str, encoder: str) -> argparse.Namespace:
    """拼出 64 号 train_encoder、obtain_encoder、evaluate 要读的全部字段；训练设定取 64 号的命令行默认值。"""
    protocol = {k: v for k, v in m64.LEGACY_CARD_PROTOCOL.items() if k != "encoder"}
    return argparse.Namespace(
        **protocol,
        design=design,
        data_variant=VARIANT,
        encoder=encoder,
        episodes=args.episodes,
        reps=args.reps,
        k_shot=K_SHOT,
        alphas="0.0",
        seed=args.seed,
        stage=args.stage,
        reuse_ckpt=True,
        new_arms="",
    )


def lookup(
    run_name: str, train_data: Pool, val_data: Pool, card: dict[str, Any], ctx: Ctx
) -> tuple[nn.Module, dict[str, Any]] | None:
    """借 64 号 obtain_encoder 的逐项核对载入已落盘的编码器；核对不过返回 None。全程只读。

    obtain_encoder 核对不过时会转去调 m64.train_encoder 重训，并覆盖该 run_name 的权重。这里先把它换成
    只抛异常的函数（72 号 load_encoders 的做法），所以不管核对结果如何，这一步都不往 07checkpoints 里写；
    本脚本自己的训练一律直接调事先留下的原函数 _TRAIN64。
    线性族在 obtain_encoder 里不经核对直接拟合并落盘，所以这里只接受深度族。
    """
    if ctx.a64.encoder not in enc_lib.DEEP_ENCODERS:
        raise RuntimeError(f"{run_name}：只读载入只对深度族成立（线性族在 obtain_encoder 里直接拟合并落盘）")
    m64.train_encoder = _refuse_training
    try:
        enc, summ = m64.obtain_encoder(run_name, train_data, val_data, "", *ctx.tail, card)
    except _NotReusableError:
        return None
    except (KeyError, ValueError, RuntimeError, OSError) as exc:
        ctx.logger.log(f"  [不复用] {run_name}：载入出错（{type(exc).__name__}: {exc}）")
        return None
    return enc, summ


def own_encoder(
    run_name: str, train_data: Pool, val_data: Pool, task_desc: str, card: dict[str, Any], ctx: Ctx
) -> tuple[nn.Module, dict[str, Any], str]:
    """本脚本自己的编码器（run_name 以 89_ 打头）：续跑时先查此前落盘的同名权重，否则按 64 号配方训练。

    返回 (编码器, 训练摘要, 来源)。
    """
    if not run_name.startswith(OWN_PREFIX):
        raise RuntimeError(f"{run_name}：本脚本新训的权重只写 {OWN_PREFIX}* 目录")
    if ctx.resume:
        got = lookup(run_name, train_data, val_data, card, ctx)
        if got is not None:
            return got[0], got[1], "本脚本此前落盘"
    enc, summ = _TRAIN64(run_name, train_data, val_data, task_desc, *ctx.tail, card)
    return enc, summ, "本脚本训练"


def fixed_encoder(
    name64: str, name89: str, train_data: Pool, val_data: Pool, task_desc: str, card: dict[str, Any], ctx: Ctx
) -> tuple[nn.Module, dict[str, Any], str, str]:
    """不随 r 变的 R 或 S_src：先只读地找 64 号的正式权重，载不到再用本脚本的 89_* 权重（续跑载入或重训）。

    返回 (编码器, 训练摘要, 来源, 实际所用的 run_name)。
    """
    got = lookup(name64, train_data, val_data, card, ctx)
    if got is not None:
        return got[0], got[1], "64号正式权重", name64
    enc, summ, source = own_encoder(name89, train_data, val_data, task_desc, card, ctx)
    return enc, summ, source, name89


# ═══════════════════════════════════════════════════════════════════════════════
# 三、单个种子的运行
# ═══════════════════════════════════════════════════════════════════════════════
def stage_suffix(stage: str) -> str:
    return "" if stage == "formal" else "__pilot"


def run_cell(
    design: str, encoder: str, args: argparse.Namespace, apple_data: Pool, year_data: Any, ctx: Ctx
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """一个（设计, 编码器族）：各留出年先取 R、S_src，再对每个 r 训 S_half，并在 B_r 半上评估三臂。

    返回 (逐划分逐臂的结果行, 编码器来源与训练摘要行)。
    """
    logger, seed = ctx.logger, int(args.seed)
    stage_seg = "" if args.stage == "formal" else "_pilot"
    # 64 号 main 的 run_name 规则：一维卷积不带编码器段
    tag64 = f"64_{design}_{VARIANT}" + ("" if encoder == "cnn" else f"_{encoder}")
    tag89 = f"{OWN_PREFIX}{design}_{VARIANT}_{encoder}"
    env = (year_data, ctx.a64, ctx.device, logger)  # 64 号 evaluate 的第四到第七个位置参数
    rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for tgt_year, cfg in sorted(m64.DESIGNS[design].items()):
        src: Pool = {e: apple_data[e] for e in cfg["src"]}
        union: Pool = {e: apple_data[e] for e in cfg["src"] + cfg["tgt"]}
        # 模型卡字段与 64 号 main 相同：正式权重的复用核对逐项比的就是这些
        card = {
            "design": design,
            "data_variant": VARIANT,
            "fold_year": tgt_year,
            **{k: getattr(ctx.a64, k) for k in m64.LEGACY_CARD_PROTOCOL},
        }
        head = {"设计": design, "留出年": tgt_year, "编码器族": encoder, "种子": seed}
        tail = f"seed{seed}_fold{tgt_year}"
        specs: dict[str, tuple[Pool, Pool, str]] = {
            "R": (src, src, f"折限定（{cfg['src']}，留出 {tgt_year}）"),
            "S_src": (union, src, f"目标暴露·源域选检查点（留出 {tgt_year}）"),
        }
        fixed: dict[str, nn.Module] = {}
        for arm, (tr, va, desc) in specs.items():
            name64, name89 = f"{tag64}_{arm}_{tail}", f"{tag89}_{arm}_{tail}{stage_seg}"
            enc, summ, source, name = fixed_encoder(name64, name89, tr, va, desc, {**card, "arm": arm}, ctx)
            fixed[arm] = enc
            info: dict[str, Any] = {"臂": arm, "划分编号": "各划分共用", "run_name": name, "来源": source}
            train_rows.append({**head, **info, **summ})
        n_tgt = sum(len(apple_data[e]) for e in cfg["tgt"])
        one_fold = {tgt_year: cfg}
        seen: dict[str, int] = {}  # A 半指纹 → 划分编号；各次划分须互不相同
        for r in range(args.n_splits):
            part_a, part_b = split_for(apple_data, cfg["tgt"], seed + tgt_year, r)
            n_a = sum(len(v) for v in part_a.values())
            n_b = sum(len(v) for v in part_b.values())
            if n_a + n_b != n_tgt:
                raise RuntimeError(f"留出 {tgt_year} 划分 {r}：A 半 {n_a} + B 半 {n_b} 不等于留出年苹果数 {n_tgt}")
            mark = fingerprint(part_a)
            if mark in seen:
                raise RuntimeError(f"留出 {tgt_year}：划分 {r} 与划分 {seen[mark]} 的 A 半相同，随机数流没有分开")
            seen[mark] = r
            logger.log(f"\n【留出 {tgt_year}，划分 {r}】A 半 {n_a} 苹果，B 半 {n_b} 苹果，A 半指纹 {mark}")
            name = f"{tag89}_S_half_h{r}_{tail}{stage_seg}"
            desc = f"半数目标暴露，划分 {r}（目标年 A 半 {n_a} 苹果，留出 {tgt_year}）"
            half_card = {**card, "arm": "S_half", "split_r": r, "half_a_sha16": mark}
            enc_half, summ, source = own_encoder(name, {**src, **part_a}, src, desc, half_card, ctx)
            info = {"臂": "S_half", "划分编号": r, "run_name": name, "来源": source}
            train_rows.append({**head, **info, **summ})
            # B_r 半上的三臂：同一个 restrict、同一组重采样种子，支持果与查询果逐情节相同
            restrict = {tgt_year: part_b}
            for label, enc in ((ARM_HALF, enc_half), (ARM_R, fixed["R"]), (ARM_SRC, fixed["S_src"])):
                (res,) = m64.evaluate(label, {tgt_year: enc}, one_fold, *env, restrict, label)
                rows.append(
                    {
                        **head,
                        "划分编号": r,
                        "臂": label,
                        "宏F1": res["F1均值"],
                        "F1情节SD": res["F1情节SD"],
                        "情节数": res["重复次数"],
                        "K": res["K"],
                        "A半苹果数": n_a,
                        "B半苹果数": n_b,
                        "A半指纹": mark,
                        "阶段": args.stage,
                    }
                )
    return rows, train_rows


def run_seed(args: argparse.Namespace) -> None:
    device = eu.get_device(args.device)
    logger = eu.get_logger(STEM)
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    designs = [d for d in args.designs.split(",") if d]
    families = [e for e in args.encoders.split(",") if e]
    protocol = m64.LEGACY_CARD_PROTOCOL
    budget_text = f"{args.episodes}/{protocol['patience']}/{protocol['val_episodes']}/{args.reps}"
    eu.log_experiment_header(
        logger,
        {
            "任务": "重复半分割：Δ_half 与份额的划分间范围、种子与划分的方差分量",
            "设计": designs,
            "预处理变体": VARIANT,
            "编码器族": families,
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "每个（种子, 留出年）的划分次数": args.n_splits,
            "情节预算 / 耐心 / 验证情节 / 评估情节": budget_text,
            "K / 融合权重": f"{K_SHOT} / 0.0",
            "留出年带标签数据的用法": "S_half 的训练池含 A 半全部苹果；评估时 B 半每个产地 K 个支持果",
            "续跑时载入本脚本已落盘的权重": args.reuse_ckpt,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))
    df = m64.load_data(VARIANT, logger)
    wl_cols = eu.get_spectrum_columns(df)
    apple_data = build_apple_data(df, wl_cols)
    year_data = build_year_region_data(df, wl_cols)
    os.makedirs(SEED_DIR, exist_ok=True)
    for design in designs:
        for encoder in families:
            logger.log(f"\n{'#' * 78}\n【{design} / {encoder}】{enc_lib.ARCH_DESC[encoder]}")
            ctx = Ctx(args_for_64(args, design, encoder), len(wl_cols), device, logger, bool(args.reuse_ckpt))
            with eu.track_compute(f"重复半分割 {design}/{encoder}", logger, run) as budget:
                rows, train_rows = run_cell(design, encoder, args, apple_data, year_data, ctx)
            res_df, train_df = pd.DataFrame(rows), pd.DataFrame(train_rows)
            compute = {
                "seed": args.seed,
                "stage": args.stage,
                "design": design,
                "encoder": encoder,
                "n_splits": args.n_splits,
                **budget,
            }
            name = f"{STEM}__{design}__{encoder}__seed{args.seed}{stage_suffix(args.stage)}.xlsx"
            out_path = os.path.join(SEED_DIR, name)
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                res_df.to_excel(writer, sheet_name=eu.table_sheet_name(0, "逐划分逐臂结果"), index=False)
                train_df.to_excel(writer, sheet_name=eu.table_sheet_name(1, "编码器来源与训练摘要"), index=False)
                pd.DataFrame([compute]).to_excel(writer, sheet_name="compute", index=False)
            logger.log(f"结果表 → {out_path}")
            shown = cell_table(res_df)[["留出年", "划分编号", *EVAL_ARMS, D_HALF, D_SRC, SHARE]]
            logger.log(shown.round(4).to_string(index=False))
    logger.close()
    if run is not None:
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# 四、汇总：逐格 Δ 与份额、划分间范围、方差分量、r = 0 与 64 号逐种子表的对照
# ═══════════════════════════════════════════════════════════════════════════════
def seed_files(stage: str) -> list[str]:
    """本脚本的逐种子结果表；pilot 与 formal 按文件名里的 __pilot 分开。"""
    files = sorted(glob.glob(os.path.join(SEED_DIR, f"{STEM}__*__seed*.xlsx")))
    return [f for f in files if ("__pilot" in os.path.basename(f)) == (stage == "pilot")]


def load_runs(stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读入该阶段全部逐种子表，返回 (逐划分逐臂长表, 各次运行的 compute 行)。"""
    files = seed_files(stage)
    if not files:
        raise FileNotFoundError(f"没有 89 号的逐种子结果（{stage}）：{SEED_DIR}")
    res = pd.concat([pd.read_excel(f, sheet_name=0) for f in files], ignore_index=True)
    parts: list[pd.DataFrame] = []
    for f in files:
        part = pd.read_excel(f, sheet_name="compute")
        part["文件"] = os.path.basename(f)
        parts.append(part)
    return res, pd.concat(parts, ignore_index=True)


def add_contrasts(wide: pd.DataFrame) -> pd.DataFrame:
    """由三臂宏 F1 算 Δ_half、Δ_src^h 与份额；Δ_src^h 不大于 0 的格份额记缺失（nan）。"""
    out = wide.copy()
    d_half = np.round((out[ARM_HALF] - out[ARM_R]).to_numpy(dtype=float), 6)
    d_src = np.round((out[ARM_SRC] - out[ARM_R]).to_numpy(dtype=float), 6)
    share = np.full(len(out), np.nan)
    ok = d_src > 0
    share[ok] = d_half[ok] / d_src[ok]
    out[D_HALF], out[D_SRC], out[SHARE] = d_half, d_src, share
    return out


def cell_table(res: pd.DataFrame) -> pd.DataFrame:
    """逐种子长表 → 每个（设计, 留出年, 编码器族, 种子, 划分编号）一行：三臂宏 F1、Δ 与份额、A/B 半苹果数。"""
    if bool(res.duplicated([*KEYS, "臂"]).any()):
        raise ValueError("逐种子结果里有重复的格：同一（设计, 留出年, 编码器族, 种子, 划分编号, 臂）出现多次")
    wide = res.pivot(index=KEYS, columns="臂", values="宏F1").reset_index()
    wide.columns.name = None
    lacking = [a for a in EVAL_ARMS if a not in wide.columns]
    if lacking:
        raise ValueError(f"逐种子结果缺臂：{lacking}")
    wide = wide.dropna(subset=list(EVAL_ARMS))
    meta = res.drop_duplicates(KEYS)[[*KEYS, "A半苹果数", "B半苹果数", "A半指纹"]]
    return add_contrasts(wide.merge(meta, on=KEYS, how="left"))[CELL_COLS]


def merge_folds(sub: pd.DataFrame, folds: list[int]) -> pd.DataFrame:
    """各折合并的格：每个（种子, 划分编号）先对各留出年取均值（缺折的格不算），再算 Δ 与份额。

    与 64 号 aggregate 的「均值」行同口径（对照先在折上取均值、再对种子取均值）。sub 只含一个设计、
    一个编码器族；没有可合并的格时返回空表。
    """
    rows: list[dict[str, Any]] = []
    for (seed, r), blk in sub.groupby(["种子", "划分编号"]):
        if {int(y) for y in blk["留出年"]} != set(folds):
            continue
        row: dict[str, Any] = {"留出年": MERGED, "种子": int(seed), "划分编号": int(r)}
        row.update({arm: float(blk[arm].mean()) for arm in EVAL_ARMS})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["设计"], out["编码器族"] = sub["设计"].iloc[0], sub["编码器族"].iloc[0]
    return add_contrasts(out)


def per_split(block: pd.DataFrame) -> pd.DataFrame:
    """每个划分编号上对种子取均值：三臂、Δ_half、Δ_src^h，两种份额与份额缺失格数。"""
    rows: list[dict[str, Any]] = []
    for r, blk in block.groupby("划分编号"):
        d_half, d_src = float(blk[D_HALF].mean()), float(blk[D_SRC].mean())
        rows.append(
            {
                "划分编号": int(r),
                "种子数": int(blk["种子"].nunique()),
                **{arm: float(blk[arm].mean()) for arm in EVAL_ARMS},
                D_HALF: d_half,
                D_SRC: d_src,
                SHARE_RATIO: d_half / d_src if d_src > 0 else float("nan"),
                SHARE_CELL: float(blk[SHARE].mean()),
                "份额缺失格数": int(blk[SHARE].isna().sum()),
                f"{D_HALF}为正种子数": int((blk[D_HALF] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize(block: pd.DataFrame, by_split: pd.DataFrame) -> dict[str, Any]:
    """全部种子 × 划分上的均值，与划分间的范围（先在每个划分编号上对种子取均值，再取最小与最大）。"""
    d_half, d_src = float(block[D_HALF].mean()), float(block[D_SRC].mean())
    out: dict[str, Any] = {
        "种子数": int(block["种子"].nunique()),
        "划分数": int(block["划分编号"].nunique()),
        "格数": len(block),
        **{f"{arm}_均值": float(block[arm].mean()) for arm in EVAL_ARMS},
        f"{D_HALF}_均值": d_half,
        f"{D_SRC}_均值": d_src,
        f"{D_HALF}_为正格数": int((block[D_HALF] > 0).sum()),
        SHARE_RATIO: d_half / d_src if d_src > 0 else float("nan"),
        SHARE_CELL: float(block[SHARE].mean()),
        "份额缺失格数": int(block[SHARE].isna().sum()),
    }
    r0 = by_split[by_split["划分编号"] == 0]
    for col in (D_HALF, SHARE_RATIO, SHARE_CELL):
        vals = by_split[col].dropna()
        out[f"{col}_r0"] = float(r0[col].iloc[0]) if len(r0) else float("nan")
        out[f"{col}_划分间最小"] = float(vals.min()) if len(vals) else float("nan")
        out[f"{col}_划分间最大"] = float(vals.max()) if len(vals) else float("nan")
    return out


def variance_components(table: NDArray) -> dict[str, float]:
    """种子 × 划分 的完整表（行 = 种子，列 = 划分编号，每格一个值）上方差分量的矩估计。

    记 S、R 为行数、列数，m_s、m_r、m 为行均值、列均值、总均值：
        MS_种子   = R·Σ_s (m_s − m)² / (S − 1)
        MS_划分   = S·Σ_r (m_r − m)² / (R − 1)
        MS_残差   = Σ_sr (y[s, r] − m_s − m_r + m)² / ((S − 1)(R − 1))
        MS_种子内 = Σ_sr (y[s, r] − m_s)² / (S(R − 1))，其分子等于划分、残差两项的平方和相加
    交叉分解 y = μ + a_s + b_r + e_sr（种子、划分编号两个交叉的随机因子，每格一个观测）：
        种子间 (MS_种子 − MS_残差)/R，划分间 (MS_划分 − MS_残差)/S，残差 MS_残差
    嵌套分解 y = μ + a_s + c_r(s)（划分嵌在种子内）：
        种子间 (MS_种子 − MS_种子内)/R，种子内划分间 MS_种子内
    估计值可以为负，照原值返回。行或列少于 2、或表里有缺失时全部返回 nan。
    """
    if table.ndim != 2 or min(table.shape) < 2 or not bool(np.isfinite(table).all()):
        return dict.fromkeys(VAR_KEYS, float("nan"))
    n_seed, n_split = table.shape
    grand = float(table.mean())
    seed_mean = table.mean(axis=1)
    split_mean = table.mean(axis=0)
    ss_seed = n_split * float(((seed_mean - grand) ** 2).sum())
    ss_split = n_seed * float(((split_mean - grand) ** 2).sum())
    resid = table - seed_mean[:, None] - split_mean[None, :] + grand
    ss_res = float((resid**2).sum())
    ms_seed = ss_seed / (n_seed - 1)
    ms_split = ss_split / (n_split - 1)
    ms_res = ss_res / ((n_seed - 1) * (n_split - 1))
    ms_within = (ss_split + ss_res) / (n_seed * (n_split - 1))
    values = (
        (ms_seed - ms_res) / n_split,
        (ms_split - ms_res) / n_seed,
        ms_res,
        (ms_seed - ms_within) / n_split,
        ms_within,
    )
    return dict(zip(VAR_KEYS, values, strict=True))


def components(block: pd.DataFrame, col: str) -> dict[str, Any]:
    """把一个量排成 种子 × 划分编号 的表做方差分量估计；表不完整（缺格或份额缺失）时各分量为 nan。"""
    table = block.pivot(index="种子", columns="划分编号", values=col).to_numpy(dtype=float)
    return {
        "种子数": int(table.shape[0]),
        "划分数": int(table.shape[1]),
        "表完整": bool(np.isfinite(table).all()),
        **variance_components(table),
    }


def read_ref64(design: str, encoder: str, seed: int, stage: str) -> tuple[dict[tuple[int, str], float], str]:
    """64 号逐种子表里 B 半三臂的 K=5 宏 F1：{(留出年, 64 号的臂名): 值}；读不到时返回 ({}, 原因)。"""
    enc_seg = "" if encoder == "cnn" else f"__{encoder}"
    name = REF64_FILE.format(design=design, variant=VARIANT, enc_seg=enc_seg, seed=seed, suffix=stage_suffix(stage))
    path = os.path.join(SEED_DIR, name)
    if not os.path.exists(path):
        return {}, f"缺 {name}"
    try:
        table = pd.read_excel(path, sheet_name=REF64_SHEET)
    except ValueError:
        return {}, f"{name} 里没有工作表 {REF64_SHEET}"
    lacking = [c for c in REF64_COLS if c not in table.columns]
    if lacking:
        return {}, f"{name} 缺列 {lacking}"
    keep = (table["src_alpha"] == 0.0) & (table["K"] == K_SHOT) & table["臂"].isin(list(REF64_ARMS.values()))
    cols = (table[keep]["目标年份"], table[keep]["臂"], table[keep]["F1均值"])
    return {(int(y), str(a)): float(v) for y, a, v in zip(*cols, strict=True)}, ""


def compare_r0(res: pd.DataFrame, stage: str) -> pd.DataFrame:
    """r = 0 的三臂与 64 号逐种子表里同一（留出年, 种子）的 B 半三臂并排，标出相差超过容差的格。"""
    rows: list[dict[str, Any]] = []
    r0 = res[res["划分编号"] == 0]
    for (design, encoder, seed), sub in r0.groupby(["设计", "编码器族", "种子"]):
        ref, why = read_ref64(str(design), str(encoder), int(seed), stage)
        for _, row in sub.iterrows():
            year, arm, mine = int(row["留出年"]), str(row["臂"]), float(row["宏F1"])
            theirs = ref.get((year, REF64_ARMS[arm]))
            diff = float("nan") if theirs is None else round(mine - theirs, 4)
            rows.append(
                {
                    "设计": design,
                    "留出年": year,
                    "编码器族": encoder,
                    "种子": int(seed),
                    "臂": arm,
                    "本脚本r0": mine,
                    "64号逐种子表": float("nan") if theirs is None else theirs,
                    "差": diff,
                    "超过容差": theirs is not None and abs(diff) > R0_TOL,
                    "备注": "" if theirs is not None else (why or "该表里没有这一格"),
                }
            )
    return pd.DataFrame(rows, columns=CMP_COLS)


def notes_table(n_mixed: int) -> pd.DataFrame:
    """汇总表「说明」页：各量的定义与读法。"""
    notes = [
        ("臂", "S_half：训练池 = 源年份 ∪ 留出年 A 半（带标签）；R^h、S_src^h：R、S_src 在同一 B 半上的评估"),
        ("对照", "Δ_half = S_half − R^h；Δ_src^h = S_src^h − R^h；份额 = Δ_half / Δ_src^h，分母不大于 0 时缺失"),
        ("各折合并", "每个（种子, 划分编号）先对各留出年取均值，再算 Δ 与份额；同 64 号汇总的「均值」行"),
        (SHARE_RATIO, "Δ_half 的均值除以 Δ_src^h 的均值（稿件里份额的口径）"),
        (SHARE_CELL, "各格份额的均值，缺失格不计，缺失格数另列"),
        ("划分间范围", "先在每个划分编号上对种子取均值，再取各编号的最小与最大；_r0 列是划分 0 的值"),
        ("方差分量", "种子 × 划分编号 的完整表上的矩估计，公式见脚本文件头；估计值可为负，表示与 0 分不开"),
        ("嵌套分解", "划分嵌在种子内：种子间、种子内划分间。各种子各有一组划分，划分的作用看这一组"),
        ("交叉分解", "种子、划分编号作两个交叉因子：种子间、划分间、残差。同一编号在各种子下不是同一批苹果"),
        ("交叉分解（续）", "所以「划分间」只量各种子在同一编号上共有的偏移，期望为 0；「残差」里含划分效应"),
        ("r0 对照", f"划分 0 与 64 号在同一（留出年, 种子）上的划分相同；相差超过 {R0_TOL} 的格另列一表"),
        ("划分指纹", f"A 半指纹在各编码器族之间不一致的（设计, 留出年, 种子, 划分编号）组数：{n_mixed}"),
    ]
    return pd.DataFrame(notes, columns=["项", "说明"])


def aggregate(stage: str, logger: eu.Logger) -> None:
    """汇总 formal_seeds 下本脚本的逐种子表（stage 为 pilot 时只读 pilot 表，汇总表名加 __pilot）。"""
    with eu.track_compute("重复半分割汇总", logger, None) as budget:
        res, comp = load_runs(stage)
        seeds = sorted({int(s) for s in res["种子"]})
        if stage == "formal" and set(seeds) != {int(s) for s in eu.RECOMMENDED_SEEDS}:
            logger.log(f"  [WARN] 种子集合 {seeds} 不是 Formal 5 种子，汇总仅作进度查看")
        cells = cell_table(res)
        blocks: list[pd.DataFrame] = []
        split_rows: list[dict[str, Any]] = []
        sum_rows: list[dict[str, Any]] = []
        var_rows: list[dict[str, Any]] = []
        for design in sorted({str(d) for d in cells["设计"]}):
            for encoder in sorted({str(e) for e in cells["编码器族"]}):
                sub = cells[(cells["设计"] == design) & (cells["编码器族"] == encoder)]
                if sub.empty:
                    continue
                folds = sorted({int(y) for y in sub["留出年"]})
                parts: list[tuple[Any, pd.DataFrame]] = [(y, sub[sub["留出年"] == y]) for y in folds]
                merged = merge_folds(sub, folds) if len(folds) > 1 else pd.DataFrame()
                if not merged.empty:
                    parts.append((MERGED, merged))
                for label, block in parts:
                    head = {"设计": design, "留出年": label, "编码器族": encoder}
                    by_split = per_split(block)
                    split_rows += [{**head, **row} for row in by_split.to_dict("records")]
                    sum_rows.append({**head, **summarize(block, by_split)})
                    var_rows += [{**head, "量": col, **components(block, col)} for col in (D_HALF, SHARE)]
                    blocks.append(block)
        cells_out = pd.concat(blocks, ignore_index=True).reindex(columns=CELL_COLS).round(6)
        split_df = pd.DataFrame(split_rows).round(4)
        sum_df = pd.DataFrame(sum_rows)
        var_df = pd.DataFrame(var_rows).round(8)
        cmp_df = compare_r0(res, stage)
        over_df = cmp_df[cmp_df["超过容差"].astype(bool)]
        marks = res.groupby(["设计", "留出年", "种子", "划分编号"])["A半指纹"].nunique()
        n_mixed = int((marks > 1).sum())
        logger.log("各划分的种子均值\n" + split_df.to_string(index=False))
        logger.log("均值与划分间范围\n" + sum_df.to_string(index=False))
        logger.log("方差分量（矩估计）\n" + var_df.to_string(index=False))
        n_ref = int(cmp_df["64号逐种子表"].notna().sum())
        n_miss = len(cmp_df) - n_ref
        worst = float(cmp_df["差"].abs().max()) if n_ref else float("nan")
        logger.log(f"r = 0 与 64 号逐种子表：比对 {n_ref} 格，缺 {n_miss} 格，最大绝对差 {worst:.4f}")
        if len(over_df):
            logger.log(f"  [WARN] 相差超过 {R0_TOL} 的格 {len(over_df)} 个\n" + over_df.to_string(index=False))
        if stage == "pilot":
            logger.log("  pilot 的情节预算与评估情节数都缩减过，与 64 号的差值不作判据")
        if n_mixed:
            logger.log(f"  [WARN] {n_mixed} 组（设计, 留出年, 种子, 划分编号）在各编码器族上的 A 半指纹不同")
    own = pd.DataFrame([{"文件": "本次汇总", "stage": stage, **budget}])
    sheets: dict[object, object] = {
        0: ("逐格Δ与份额", cells_out),
        1: ("各划分的种子均值", split_df),
        2: ("均值与划分间范围", sum_df),
        3: ("方差分量", var_df),
        4: ("r0与64号逐种子表对照", cmp_df),
        5: ("r0相差超过容差的格", over_df),
        6: ("说明", notes_table(n_mixed)),
        "compute": pd.concat([comp, own], ignore_index=True),
    }
    out = eu.write_script_workbook(f"{STEM}{stage_suffix(stage)}.py", sheets)
    logger.log(f"汇总表 → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 五、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="89_repeated_half_split：重复半分割下的 Δ_half 与份额")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--stage",
        default=None,
        choices=["pilot", "formal"],
        help="不给时：训练与评估按 pilot，--aggregate 按 formal",
    )
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED, help="一次一个种子")
    parser.add_argument("--designs", default="two_year", help="逗号分隔的设计（64 号 DESIGNS 的键）")
    parser.add_argument("--encoders", default=",".join(enc_lib.DEEP_ENCODERS), help="逗号分隔的深度编码器族")
    parser.add_argument("--n_splits", type=int, default=None, help="划分次数；缺省 formal 5、pilot 2")
    parser.add_argument("--reuse_ckpt", action="store_true", help="续跑：载入本脚本此前落盘且核对通过的权重")
    parser.add_argument("--aggregate", action="store_true", help="汇总 formal_seeds 下已有的逐种子表")
    args = parser.parse_args()

    if args.aggregate:
        logger = eu.get_logger(STEM)
        aggregate(args.stage or "formal", logger)
        logger.close()
        return

    args.stage = args.stage or "pilot"
    formal = args.stage == "formal"
    args.episodes = EPISODES if formal else PILOT_EPISODES
    args.reps = REPS if formal else PILOT_REPS
    if args.n_splits is None:
        args.n_splits = N_SPLITS if formal else PILOT_N_SPLITS
    if args.n_splits < 1:
        raise SystemExit("--n_splits 至少为 1")
    bad_designs = sorted(set(args.designs.split(",")) - set(m64.DESIGNS) - {""})
    bad_encoders = sorted(set(args.encoders.split(",")) - set(enc_lib.DEEP_ENCODERS) - {""})
    if bad_designs or bad_encoders:
        raise SystemExit(f"未知设计 {bad_designs}，或不是深度编码器族 {bad_encoders}（线性族不在本脚本内）")
    run_seed(args)


if __name__ == "__main__":
    main()
