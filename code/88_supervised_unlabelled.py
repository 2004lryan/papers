"""
88_supervised_unlabelled.py: 监督式（非情节）训练下的留出域暴露对照，与苹果、猕猴桃上的无标签暴露对照（补充材料 S9.7）。

为什么:
    64 号（苹果）与 74 号（公开数据集）的暴露对照都按原型网络情节式训练；31、35 号的无标签暴露只在
    四个公开数据集上跑过。本脚本补两件事：抬高是否只在情节式训练下出现；无标签暴露在苹果与猕猴桃上有多大。

第一部分（--part sup）监督式训练的暴露臂:
    骨架    encoders.build_encoder 的 cnn 与 resnet，输出维数与情节式实验相同（苹果 128，公开数据集 64）。
            后接一个线性分类头，交叉熵训练；分类头的类别就是训练池里出现的类别。分类头接在骨架输出的
            L2 归一化嵌入上，其输出乘 10（情节式损失给余弦相似度乘的温度）后算交叉熵。训完去掉分类头。
    R       训练池 = 源域
    S_src   训练池 = 源域 ∪ 留出域全部带标签光谱
    S_shuf  训练池的光谱同 S_src，留出域的类别标签在该域内随机置换（各类数目不变；苹果按果置换，一果的各面位
            用同一个标签）。口径同 64 号 shuffle_target_labels、74 号 shuffled_pool
    S_half  训练池 = 源域 ∪ 留出域 A 半（带标签），只在 B 半上评估
    R_h     R 的编码器在同一 B 半上的成绩（即 R^h，64、74 号的表里记作 R_B），不另训练
    对照    Δ_rep = S_src − R；Δ_lab = S_src − S_shuf；Δ_half = S_half − R_h（名称同 64、74 号）
    A/B     苹果按苹果编号、逐产地对半分（64 号 split_target_half 的口径）；公开数据集按样本、逐类对半分
            （74 号 half_split 的口径）；奇数时 B 多一个。置乱与划分都用本脚本自己的随机数流，与 64、74 号的
            不是同一次。
    训练池  苹果照 64 号 fit_linear 整理成数组：每果的每个面位各占一行（2018、2019 年每果一条谱，2025 年
            每果至多五面），标签取所属苹果的产地。公开数据集照 74 号 fold_arrays，类别集合出自 cdl.prepare_fold。

与情节式实验的可比口径:
    相同    骨架与输出维数；同一种子下骨架的初始权重；优化器、学习率日程与梯度裁剪（苹果照 64 号
            train_encoder：Adam 1e-3、权重衰减 1e-4、余弦退火到 1e-5、范数裁剪 5；公开数据集照
            cdl.meta_train：Adam 1e-3）；梯度步数（苹果 2000，公开数据集 1000，即情节预算）；每步的样本数
            （情节式一个情节里支持加查询的样本总数，由现有采样器算出并写进日志）；K=5 最近原型评估
            （苹果 44 号 kshot_eval，公开数据集 cdl.eval_kshot；全部样本上的评估情节与 64、74 号逐次相同）。
    不同    训练目标与取样：情节式每个情节按类均衡取样、由支持样本构造原型（苹果按果抽，支持样本取该果
            各面位的均值，查询取其中一个面位）；监督式把训练池的光谱逐轮打乱后依次取小批量，对全部类别算
            交叉熵（苹果每个面位各算一条，2025 年的果在小批量里出现得更多）。另有一处：64 号的苹果编码器
            按源域验证情节选检查点并可早停，实际更新次数至多 2000；这里不做验证、不早停、取末次权重
            （公开数据集两边都取末次权重）。

第二部分（--part unlab）无标签暴露，只做 cnn:
    U_src   自监督预训练的数据 = 源域光谱
    U_tgt   自监督预训练的数据 = 源域 ∪ 留出域光谱，不用标签
    预训练  掩码重建与对比两种。猕猴桃直接调用 cdl.masked_ssm_pretrain、cdl.contrastive_ssm_pretrain；
            这两个函数把编码器写死为 ProtoEncoder(in_dim, feat_dim=64)，苹果的情节式实验是 128 维，
            苹果用本脚本的 ssl_pretrain（等价重写，差别见该函数的文档字符串）。
    之后    31、34、35 号在预训练之后不再训练，冻结的编码器直接做 K=5 最近原型评估。本脚本照此给一组
            （方式记作「掩码重建」「对比」），与现有四个公开数据集的结果同口径；另给一组在预训练权重上
            只用源域带标签数据再做情节训练的（方式后缀「+源域情节训练」：苹果按 64 号的优化器配方
            2000 情节，猕猴桃按 cdl.meta_train 1000 情节，都不做验证、取末次权重）。
    苹果    数据按年份分域整理成数组，面位处理同第一部分；评估同 64 号。

范围与配对:
    sup     苹果两种设计共 5 折（two_year 2 折、three_year 3 折）+ 公开数据集 16 个留出域
    unlab   苹果 5 折 + 猕猴桃 3 个留出域
    同一格（数据集 × 留出域 × 编码器族 × 方式）内各臂共用同一批评估情节；S_half 与 R_h 在 B 半上另成一批。

随机数:
    新增的抽样用 np.random.default_rng([种子, 88, 流编号, …])：流 1 = 监督式训练取小批量；流 2 = A/B 划分；
    流 3 = 留出域标签置乱（流 2、流 3 苹果后接留出年份，公开数据集后接「数据集/留出域」的 CRC32）。
    每个编码器训练前 eu.set_random_seed(种子)。
    ssl_pretrain 内部沿用原函数的 default_rng(种子)，输出 64 维时可与原函数逐位比对（pilot 时自检）。
    新增的算子只有线性层与二维输入的交叉熵，在 CUDA 确定性模式下都有确定实现。

运行方式:
    cd <项目根>
    # pilot（种子 2004；梯度步数、预训练轮数与评估情节数约为正式的 1/5）
    python 02code/88_supervised_unlabelled.py --part sup --stage pilot --device auto --data_root <公开数据根目录>
    python 02code/88_supervised_unlabelled.py --part unlab --stage pilot --device auto --data_root <公开数据根目录>
    python 02code/88_supervised_unlabelled.py --aggregate --stage pilot
    # formal（CUDA，每次一个种子；--scope、--encoders、--designs、--datasets 可把一个种子拆成几个进程）
    python 02code/88_supervised_unlabelled.py --part sup --stage formal --device cuda --seed 20060515 \
        --data_root <公开数据根目录>
    python 02code/88_supervised_unlabelled.py --part unlab --stage formal --device cuda --seed 20060515 \
        --data_root <公开数据根目录>
    python 02code/88_supervised_unlabelled.py --aggregate

输入:
    03data/processed/all_9elements.csv（经 64 号 load_data 读入并核对 SHA-256）；--data_root 下的公开数据集；
    --aggregate 另读 04outputs/formal_seeds/ 里 64、74 号的逐种子表（情节式训练的同一对照，只读）。

输出文件:
    04outputs/formal_seeds/88_supervised_unlabelled__<部分>__<范围>__<编码器>__seed<S>[__pilot].xlsx
        — 逐格结果（一行一格）、训练摘要、compute。只跑某个范围的一部分时，<范围> 写成 apple-two_year、
          public-corn_instr+tablet_instr 这样，与完整运行的表分开存
    04outputs/88_supervised_unlabelled.xlsx — 汇总（--aggregate；--stage pilot 时文件名加 __pilot）
    05logs/88_supervised_unlabelled_YY-MM-DD_HHMMSS.log — 运行日志
    权重不落盘（各臂在固定种子、同一环境下可重训）；不写 64、74、31、34、35 号的结果文件与 07checkpoints。

复用:
    export_utils      设备、播种、Formal 的 CUDA 闸、日志、算力记录、工作表命名、汇总工作簿
    encoders          build_encoder
    protonet          ELEM_INFO、EpisodeSampler、build_apple_data、build_year_region_data、proto_loss
    64 号             DESIGNS、load_data；训练配方、fit_linear 的数组整理、shuffle_target_labels 与
                      split_target_half 的口径照写
    44 号             FeatureCache、kshot_eval（64 号 evaluate 所调用的评估）
    74 号             DATASETS、fold_arrays；shuffled_pool 与 half_split 的口径照写
    crossdataset_lib  ArrSampler、prepare_fold、eval_kshot、masked_ssm_pretrain、contrastive_ssm_pretrain、
                      SpectralDecoder、span_mask、_ProjHead、_spectral_augment、_nt_xent
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import zlib
from dataclasses import dataclass
from typing import Any

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

NDArray = npt.NDArray[Any]
Pool = dict[str, dict[Any, dict[str, Any]]]  # 苹果训练池：{域: {苹果编号: {"X": 各面位光谱, "region": 产地}}}
Rows = list[dict[str, Any]]
CellMap = dict[tuple[str, str, str, str, str], dict[int, float]]  # (数据集, 留出域, 族, 方式, 臂) → {种子: 宏 F1}
EpiMap = dict[tuple[str, str, str, str], dict[int, float]]  # (数据集, 留出域, 族, 臂) → {种子: 宏 F1}

bmc: Any = importlib.import_module("44_budget_matched_control")
m64: Any = importlib.import_module("64_leakage_controls")
m74: Any = importlib.import_module("74_multi_encoder_leakage")

SCRIPT = "88_supervised_unlabelled"
SEED_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
K_SHOT = 5
REPS = {"formal": 50, "pilot": 10}  # 每格的评估情节数
PILOT_DIVISOR = 5  # pilot 时梯度步数与预训练轮数除以它
APPLE_VARIANT = "canonical"
APPLE_DESIGNS: tuple[str, ...] = ("two_year", "three_year")
PUBLIC_SETS: tuple[str, ...] = tuple(m74.DATASETS)
UNLAB_PUBLIC_SETS: tuple[str, ...] = ("kiwifruit_season",)
DATASET_ORDER: tuple[str, ...] = ("apple_two_year", "apple_three_year", *PUBLIC_SETS)
SUP_FAMILIES: tuple[str, ...] = ("cnn", "resnet")
UNLAB_FAMILIES: tuple[str, ...] = ("cnn",)

APPLE_N_WAY = 3  # 64 号 --n_way 的缺省值
LOGIT_SCALE = 10.0  # 分类头输出所乘的倍数，取 proto_loss 给余弦相似度乘的温度
TAIL_STEPS = 100  # 训练摘要里的损失取最后这么多步的均值
SSM_EPOCHS = 200  # 31、35 号 --ssm_epochs 的缺省值
SSL_MASK_RATIO, SSL_SPAN, SSL_TEMP, SSL_BATCH, SSL_PATIENCE = 0.30, 5, 0.5, 128, 15  # cdl 两个预训练函数的缺省值
CDL_FEAT_DIM = 64  # cdl 两个预训练函数写死的编码器输出维数
STREAM_BATCH = 1  # 随机数流：监督式训练取小批量
STREAM_HALF = 2  # 随机数流：留出域 A/B 划分
STREAM_SHUF = 3  # 随机数流：留出域标签置乱

SUP_METHOD = "监督式"
SSL_MASK = "掩码重建"
SSL_CONTRAST = "对比"
SRC_STAGE = "+源域情节训练"  # 预训练之后只用源域带标签数据再做情节训练
# (臂, 提供编码器的臂, 评估样本)：R、S_src、S_shuf 在留出域全部样本上评估，S_half 与 R_h 在 B 半上评估
SUP_EVALS = (
    ("R", "R", "全部"),
    ("S_src", "S_src", "全部"),
    ("S_shuf", "S_shuf", "全部"),
    ("S_half", "S_half", "B半"),
    ("R_h", "R", "B半"),
)
SUP_ARMS: tuple[str, ...] = tuple(arm for arm, _, _ in SUP_EVALS)
SUP_CONTRASTS: dict[str, tuple[str, str]] = {
    "Δ_rep": ("S_src", "R"),
    "Δ_lab": ("S_src", "S_shuf"),
    "Δ_half": ("S_half", "R_h"),
}

COL_METHOD = "训练或预训练方式"
CELL_COLUMNS = [
    "部分",
    "范围",
    "数据集",
    "留出域",
    "编码器族",
    COL_METHOD,
    "臂",
    "评估样本",
    "K",
    "宏F1",
    "情节数",
    "种子",
    "阶段",
]
CELL_KEYS = ["部分", "数据集", "留出域", "编码器族", COL_METHOD, "臂", "种子"]
SHEET_CELLS = eu.table_sheet_name(0, "逐格结果")
SHEET_TRAIN = eu.table_sheet_name(1, "训练摘要")
STAT_KEYS = ("", "_t区间下限", "_t区间上限", "_为正种子数", "_种子数")
# 汇总表里编码器族与方式的先后
ITEM_RANK = {
    name: i
    for i, name in enumerate(
        (*enc_lib.ENCODERS, SUP_METHOD, SSL_MASK, SSL_CONTRAST, SSL_MASK + SRC_STAGE, SSL_CONTRAST + SRC_STAGE)
    )
}

# 情节式训练的同一对照（--aggregate 并排列出）取自 64、74 号的逐种子表。文件名见 epi_file_64 / epi_file_74；
# 工作表名与列名取自写表的代码：64 号 main 写「K5各臂结果」（列由 evaluate 给出），74 号 main 写「逐臂结果」
# （列由 run_fold 给出）；75 号 load_arms 读的是同一组文件与列。
EPI_SHEET_64 = eu.table_sheet_name(0, "K5各臂结果")
EPI_SHEET_74 = eu.table_sheet_name(0, "逐臂结果")
EPI_COLS_64 = {"alpha": "src_alpha", "held": "目标年份", "arm": "臂", "f1": "F1均值"}
EPI_COLS_74 = {"dataset": "dataset", "held": "held", "arm": "臂", "f1": "F1"}
EPI_CONTRASTS: dict[str, tuple[str, str]] = {
    "Δ_rep": ("S_src", "R"),
    "Δ_lab": ("S_src", "S_shuf"),
    "Δ_half": ("S_half", "R_B"),
}
# 跨域汇总的口径：名称 → (数据集, 应有的留出域数)
SUP_SCOPES: dict[str, tuple[tuple[str, ...], int]] = {
    "主口径（苹果两年设计 + 公开数据集）": (("apple_two_year", *PUBLIC_SETS), 18),
    "苹果三年设计": (("apple_three_year",), 3),
}


@dataclass(frozen=True)
class Recipe:
    """情节式正式配方里与训练有关的各项；监督式训练与预训练之后的源域情节训练都照它取。"""

    steps: int  # 情节预算，即梯度步数
    feat_dim: int  # 骨架输出维数
    lr: float
    weight_decay: float
    eta_min: float | None  # 余弦退火的末端学习率；None = 学习率不变
    clip: float | None  # 梯度范数裁剪上限；None = 不裁剪
    k_support: int  # 情节里每类的支持样本数
    q_query: int  # 情节里每类的查询样本数


RECIPES: dict[str, Recipe] = {
    # 苹果：64 号 train_encoder 与其命令行缺省值
    "apple": Recipe(
        steps=2000, feat_dim=128, lr=1e-3, weight_decay=1e-4, eta_min=1e-5, clip=5.0, k_support=5, q_query=10
    ),
    # 公开数据集：cdl.meta_train 与 74 号 --episodes 的缺省值
    "public": Recipe(
        steps=1000, feat_dim=64, lr=1e-3, weight_decay=0.0, eta_min=None, clip=None, k_support=5, q_query=10
    ),
}


@dataclass(frozen=True)
class RunCfg:
    """一次运行里各格共用的设定。"""

    stage: str
    seed: int
    reps: int
    device: torch.device
    data_root: str


@dataclass(frozen=True)
class AppleSet:
    """苹果数据的两种索引（protonet.build_apple_data / build_year_region_data）与波段数。"""

    apple_data: Pool
    year_data: dict[int, dict[str, dict[Any, NDArray]]]
    n_bands: int


def scaled(n: int, stage: str) -> int:
    """pilot 时把预算（梯度步数、预训练轮数）减到约 1/5。"""
    return n if stage == "formal" else max(1, n // PILOT_DIVISOR)


def cell_base(part: str, scope: str, dataset: str, held: str, family: str, method: str, rc: RunCfg) -> dict[str, Any]:
    """逐格结果里一格各行共有的列。"""
    return {
        "部分": part,
        "范围": scope,
        "数据集": dataset,
        "留出域": held,
        "编码器族": family,
        COL_METHOD: method,
        "K": K_SHOT,
        "种子": rc.seed,
        "阶段": rc.stage,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 一、数据：苹果训练池的数组化、按年份分域、A/B 划分、情节的样本数
# ═══════════════════════════════════════════════════════════════════════════════
def load_apple(logger: eu.Logger) -> AppleSet:
    """冻结数据（核对 SHA-256）→ 苹果粒度索引与按年份、产地组织的评估索引，与 64 号 main 相同。"""
    df = m64.load_data(APPLE_VARIANT, logger)
    wl_cols = eu.get_spectrum_columns(df)
    apple = AppleSet(build_apple_data(df, wl_cols), build_year_region_data(df, wl_cols), len(wl_cols))
    for e in sorted(apple.apple_data):
        n_spec = sum(len(v["X"]) for v in apple.apple_data[e].values())
        logger.log(f"  {e} {ELEM_INFO[e]}: {len(apple.apple_data[e])} 苹果，{n_spec} 条光谱")
    return apple


_PUBLIC_CACHE: dict[str, cdl.LoaderResult] = {}


def load_public(dsname: str, rc: RunCfg) -> cdl.LoaderResult:
    """按 74 号 DATASETS 登记的加载器读一个公开数据集；同一进程内只读一次。"""
    if dsname not in _PUBLIC_CACHE:
        loaded: cdl.LoaderResult = m74.DATASETS[dsname]["loader"](rc.data_root, rc.seed)
        _PUBLIC_CACHE[dsname] = loaded
    return _PUBLIC_CACHE[dsname]


def pool_arrays(pool: Pool) -> tuple[NDArray, NDArray]:
    """苹果训练池 → (光谱, 产地) 数组，整理方式同 64 号 fit_linear。

    域与苹果编号各按升序，每果的每个面位各占一行（2018、2019 年每果一条谱，2025 年每果至多五面），
    标签取所属苹果的产地。
    """
    xs: list[NDArray] = []
    ys: list[str] = []
    for e in sorted(pool):
        for aid in sorted(pool[e]):
            faces = pool[e][aid]["X"]
            xs.append(faces)
            ys += [str(pool[e][aid]["region"])] * len(faces)
    return np.vstack(xs).astype(np.float32), np.asarray(ys)


def apple_year_domains(apple_data: Pool) -> dict[int, tuple[NDArray, NDArray]]:
    """苹果数据按采集年份分域：{年份: (该年全部光谱, 各行的产地)}；面位处理同 pool_arrays。"""
    years = sorted({ELEM_INFO[e][0] for e in apple_data})
    return {yr: pool_arrays({e: v for e, v in apple_data.items() if ELEM_INFO[e][0] == yr}) for yr in years}


def shuffle_apple(apple_data: Pool, tgt_elems: list[str], seed: int, tgt_year: int) -> Pool:
    """留出年各苹果的产地标签在该年内随机置换（S_shuf 臂），口径同 64 号 shuffle_target_labels。

    光谱、苹果集合与各产地的苹果数不变，一果的各面位用同一个置换后的标签。随机数流 [种子, 88, 3, 留出年份]。
    """
    pairs = [(e, a) for e in tgt_elems for a in sorted(apple_data[e])]
    regions = [apple_data[e][a]["region"] for e, a in pairs]
    perm = np.random.default_rng([seed, 88, STREAM_SHUF, tgt_year]).permutation(len(regions))
    out: Pool = {e: {} for e in tgt_elems}
    for (e, a), j in zip(pairs, perm, strict=True):
        out[e][a] = {"X": apple_data[e][a]["X"], "region": regions[j]}
    return out


def split_half_apple(
    apple_data: Pool, tgt_elems: list[str], seed: int, tgt_year: int
) -> tuple[Pool, dict[str, set[Any]]]:
    """留出年各域（一域即一个产地）的苹果按种子随机对半分，口径同 64 号 split_target_half。

    返回 (A 半的训练池子集, {产地: B 半的苹果编号集合})；奇数个时 B 多一个。随机数流 [种子, 88, 2, 留出年份]，
    同一种子、同一留出年在两种设计下是同一次划分。
    """
    rng = np.random.default_rng([seed, 88, STREAM_HALF, tgt_year])
    part_a: Pool = {}
    part_b: dict[str, set[Any]] = {}
    for e in tgt_elems:
        ids = sorted(apple_data[e])
        perm = rng.permutation(len(ids))
        n_a = len(ids) // 2
        part_a[e] = {ids[i]: apple_data[e][ids[i]] for i in sorted(perm[:n_a])}
        part_b.setdefault(str(ELEM_INFO[e][1]), set()).update(ids[i] for i in perm[n_a:])
    return part_a, part_b


def fold_stream(seed: int, stream: int, dsname: str, held: str) -> np.random.Generator:
    """公开数据集某一折的随机数流 [种子, 88, 流编号, CRC32(「数据集/留出域」)]。"""
    return np.random.default_rng([seed, 88, stream, zlib.crc32(f"{dsname}/{held}".encode())])


def shuffle_public(
    fa: cdl.FoldArrays, dom_of_row: NDArray, held: str, seed: int, dsname: str
) -> tuple[NDArray, NDArray]:
    """留出域样本的类别在该域内随机置换（S_shuf 臂），训练池的光谱同 S_src；口径同 74 号 shuffled_pool。

    各类样本数不变。随机数流 [种子, 88, 3, CRC32(「数据集/留出域」)]。
    """
    y: NDArray = fa.yall.copy()
    idx = np.where(dom_of_row == held)[0]
    perm = fold_stream(seed, STREAM_SHUF, dsname, held).permutation(len(idx))
    y[idx] = fa.yall[idx][perm]
    return fa.xall, y


def split_half_public(
    fa: cdl.FoldArrays, dom_of_row: NDArray, held: str, seed: int, dsname: str
) -> tuple[NDArray, NDArray, NDArray]:
    """留出域各类样本按种子对半分 A、B（奇数时 B 多一个），口径同 74 号 half_split。

    返回 (训练池 x, 训练池 y, B 半在 xh 中的下标)；训练池 = xall 去掉留出域的 B 半，行序不变。
    随机数流 [种子, 88, 2, CRC32(「数据集/留出域」)]。
    """
    rng = fold_stream(seed, STREAM_HALF, dsname, held)
    in_a = np.zeros(len(fa.yh), dtype=bool)
    for c in fa.classes:
        ids = np.where(fa.yh == c)[0]
        perm = rng.permutation(len(ids))
        in_a[ids[perm[: len(ids) // 2]]] = True
    held_rows = np.where(dom_of_row == held)[0]
    keep = np.ones(len(fa.yall), dtype=bool)
    keep[held_rows[~in_a]] = False
    return fa.xall[keep], fa.yall[keep], np.where(~in_a)[0]


def episode_size_apple(pool: Pool, recipe: Recipe) -> int:
    """情节式一个情节的样本总数（苹果）：按 EpisodeSampler.sample 的取法，路数 × (支持数 + 查询数)。"""
    smp = EpisodeSampler(sorted(pool), pool, APPLE_N_WAY, recipe.k_support, recipe.q_query)
    return min(smp.n_way, len(smp.all_regions)) * (smp.k_support + smp.q_query)


def episode_size_public(x: NDArray, y: NDArray, recipe: Recipe) -> int:
    """情节式一个情节的样本总数（公开数据集）：按 cdl.ArrSampler.sample 的取法，类数（至多 max_way）× (k + q)。"""
    smp = cdl.ArrSampler(x, y, recipe.k_support, recipe.q_query)
    return min(len(smp.classes), smp.max_way) * (smp.k + smp.q)


# ═══════════════════════════════════════════════════════════════════════════════
# 二、训练：监督式、预训练之后的源域情节训练、自监督预训练
# ═══════════════════════════════════════════════════════════════════════════════
class BatchStream:
    """训练池行下标的流：逐轮无放回打乱，按固定大小依次取；一轮剩下不足一批时接上下一轮的打乱结果。"""

    def __init__(self, n: int, rng: np.random.Generator) -> None:
        self.n = n
        self.rng = rng
        self.buf: NDArray = rng.permutation(n)

    def take(self, size: int) -> NDArray:
        while len(self.buf) < size:
            self.buf = np.concatenate([self.buf, self.rng.permutation(self.n)])
        out, self.buf = self.buf[:size], self.buf[size:]
        return out


def make_optimizer(
    params: list[nn.Parameter], recipe: Recipe, steps: int
) -> tuple[optim.Adam, CosineAnnealingLR | None]:
    """情节式配方的优化器与学习率日程（苹果：Adam、权重衰减、余弦退火；公开数据集：Adam、学习率不变）。"""
    optimizer = optim.Adam(params, lr=recipe.lr, weight_decay=recipe.weight_decay)
    if recipe.eta_min is None:
        return optimizer, None
    return optimizer, CosineAnnealingLR(optimizer, T_max=steps, eta_min=recipe.eta_min)


def train_supervised(
    family: str,
    x: NDArray,
    y: NDArray,
    in_dim: int,
    recipe: Recipe,
    steps: int,
    batch: int,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    """骨架加线性分类头的交叉熵训练，返回 (去掉分类头的骨架, 训练摘要)。

    步数、每步样本数、优化器与日程都取情节式配方；不做验证、不早停，取末次权重。小批量取自流 [种子, 88, 1]。
    骨架先于分类头构造，同一种子下其初始权重与情节式训练的相同。
    """
    classes = np.unique(y)
    y_idx = np.searchsorted(classes, y).astype(np.int64)
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    eu.set_random_seed(seed)
    encoder = enc_lib.build_encoder(family, in_dim, recipe.feat_dim).to(device)
    head = nn.Linear(recipe.feat_dim, len(classes)).to(device)
    params = [*encoder.parameters(), *head.parameters()]
    optimizer, scheduler = make_optimizer(params, recipe, steps)
    stream = BatchStream(len(x32), np.random.default_rng([seed, 88, STREAM_BATCH]))
    encoder.train()
    head.train()
    tail: list[float] = []
    for step in range(steps):
        idx = stream.take(batch)
        xb = torch.from_numpy(x32[idx]).to(device)
        yb = torch.from_numpy(y_idx[idx]).to(device)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(head(encoder(xb)) * LOGIT_SCALE, yb)
        loss.backward()  # type: ignore[no-untyped-call]
        if recipe.clip is not None:
            nn.utils.clip_grad_norm_(params, max_norm=recipe.clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step >= steps - TAIL_STEPS:
            tail.append(float(loss.item()))
    encoder.eval()
    return encoder, {
        "训练池光谱数": len(x32),
        "分类头类别数": len(classes),
        "梯度步数": steps,
        "小批量": batch,
        "末段平均损失": float(np.mean(tail)),
    }


def episodic_train(
    encoder: nn.Module,
    sampler: EpisodeSampler | cdl.ArrSampler,
    recipe: Recipe,
    steps: int,
    device: torch.device,
    seed: int,
) -> float:
    """在给定采样器上做原型网络情节训练，原地更新 encoder，返回末段平均损失。

    用于预训练之后只用源域带标签数据的那一步：训练循环与 64 号 train_encoder（苹果）、cdl.meta_train
    （公开数据集）相同，但从传入的权重出发，不做验证、不早停，取末次权重。
    """
    eu.set_random_seed(seed)
    params = list(encoder.parameters())
    optimizer, scheduler = make_optimizer(params, recipe, steps)
    encoder.train()
    tail: list[float] = []
    for step in range(steps):
        sup_x, sup_y, qry_x, qry_y, n_way = sampler.sample()
        optimizer.zero_grad()
        loss, _ = proto_loss(encoder, sup_x, sup_y, qry_x, qry_y, n_way, device)
        loss.backward()  # type: ignore[no-untyped-call]
        if recipe.clip is not None:
            nn.utils.clip_grad_norm_(params, max_norm=recipe.clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step >= steps - TAIL_STEPS:
            tail.append(float(loss.item()))
    encoder.eval()
    return float(np.mean(tail))


def ssl_pretrain(
    method: str, x_unlabeled: NDArray, in_dim: int, feat_dim: int, device: torch.device, epochs: int, seed: int
) -> nn.Module:
    """cdl.masked_ssm_pretrain（掩码重建）与 cdl.contrastive_ssm_pretrain（对比）的等价重写，供苹果使用。

    与原函数的差别:
      1. 编码器输出维数 feat_dim 是参数。原函数写死 ProtoEncoder(in_dim, feat_dim=64)，解码头的输入维数与
         投影头的宽度也写死 64；这里三处都取 feat_dim。
      2. 两个原函数并成一个，用 method 区分；超参数固定为原函数的缺省值（掩码比例 0.30、段长 5、对比温度 0.5、
         批大小 128、耐心 15、Adam 1e-3），31、35 号调用原函数时用的就是这些缺省值。
      3. 训练与验证子集先在 CPU 数组上按下标取好再搬到设备上；原函数用 NumPy 下标索引设备上的张量。
         取到的行相同。
    其余逐句相同：播种、default_rng(种子)、一成样本留作验证、每轮打乱、验证损失比最好值低 1e-5 以上才算改进、
    连续 15 轮不改进即停、取验证损失最低的权重。feat_dim=64 时与原函数逐位相同（check_rewrite 核对）。
    """
    eu.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    enc = enc_lib.build_encoder("cnn", in_dim, feat_dim).to(device)
    masked = method == SSL_MASK
    aux: nn.Module = (cdl.SpectralDecoder(feat_dim, in_dim) if masked else cdl._ProjHead(feat_dim)).to(device)
    opt = optim.Adam(list(enc.parameters()) + list(aux.parameters()), lr=1e-3)

    x_np = x_unlabeled.astype(np.float32)
    n = len(x_np)
    n_val = max(1 if masked else 2, int(0.1 * n))
    perm = rng.permutation(n)
    x_tr = torch.from_numpy(x_np[perm[n_val:]]).to(device)
    x_val = torch.from_numpy(x_np[perm[:n_val]]).to(device)

    def objective(xb: torch.Tensor, gen: np.random.Generator) -> torch.Tensor:
        if masked:
            m = torch.from_numpy(cdl.span_mask(len(xb), in_dim, SSL_MASK_RATIO, SSL_SPAN, gen)).to(device)
            recon = aux(enc(xb * (1.0 - m)))
            out: torch.Tensor = ((recon - xb) ** 2 * m).sum() / m.sum().clamp_min(1.0)
            return out
        v1 = aux(enc(cdl._spectral_augment(xb, in_dim, gen, device)))
        v2 = aux(enc(cdl._spectral_augment(xb, in_dim, gen, device)))
        return cdl._nt_xent(v1, v2, SSL_TEMP)

    best_val, bad = float("inf"), 0
    best_state: dict[str, Any] | None = None
    for _ in range(epochs):
        enc.train()
        aux.train()
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), SSL_BATCH):
            bidx = order[s : s + SSL_BATCH]
            if len(bidx) < 2:  # 批归一化与对比损失都要至少两个样本
                continue
            loss = objective(x_tr[torch.from_numpy(bidx).to(device)], rng)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        enc.eval()
        aux.eval()
        with torch.no_grad():
            vloss = float(objective(x_val, np.random.default_rng(seed)).item())
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in enc.state_dict().items()}
        else:
            bad += 1
            if bad >= SSL_PATIENCE:
                break
    if best_state is not None:
        enc.load_state_dict(best_state)
    enc.eval()
    return enc


def cdl_pretrain(method: str, x: NDArray, in_dim: int, device: torch.device, epochs: int, seed: int) -> nn.Module:
    """直接调用 cdl 的两个预训练函数（与 31、34、35 号同一调用；编码器输出 64 维）。"""
    if method == SSL_MASK:
        return cdl.masked_ssm_pretrain(x, in_dim, device, epochs, seed)
    return cdl.contrastive_ssm_pretrain(x, in_dim, device, epochs, seed)


def check_rewrite(x: NDArray, in_dim: int, rc: RunCfg, epochs: int, logger: eu.Logger) -> None:
    """自检：ssl_pretrain 输出 64 维时须与 cdl 的两个原函数逐位相同。

    同一输入、同一种子各训一次，比对编码器的全部权重；不同即中止。
    """
    for method in (SSL_MASK, SSL_CONTRAST):
        ref = cdl_pretrain(method, x, in_dim, rc.device, epochs, rc.seed).state_dict()
        mine = ssl_pretrain(method, x, in_dim, CDL_FEAT_DIM, rc.device, epochs, rc.seed).state_dict()
        if ref.keys() != mine.keys() or any(not torch.equal(ref[k], mine[k]) for k in ref):
            raise RuntimeError(f"自检未过：ssl_pretrain（{method}）输出 64 维时与 cdl 原函数的权重不逐位相同")
    logger.log(f"  自检通过：ssl_pretrain 输出 64 维时与 cdl 原函数逐位相同（{len(x)} 条光谱，至多 {epochs} 轮）")


# ═══════════════════════════════════════════════════════════════════════════════
# 三、评估：K=5 最近原型，口径同 64、74 号
# ═══════════════════════════════════════════════════════════════════════════════
def eval_apple(
    cache: Any,
    year_data: dict[int, dict[str, dict[Any, NDArray]]],
    tgt_year: int,
    src_years: list[int],
    reps: int,
    seed: int,
    keep: dict[str, set[Any]] | None = None,
) -> tuple[float, int]:
    """苹果的 K=5 评估，与 64 号 evaluate（融合权重 0）相同，返回 (宏 F1 的情节均值, 情节数)。

    支持集按苹果编号抽，查询为留出年其余全部苹果，重采样种子为 种子+i；keep 给出时留出年只保留
    keep[产地] 里的苹果（B 半）。cache 是 44 号的 FeatureCache，同一编码器可在全部样本与 B 半上共用。
    """
    yd = year_data
    if keep is not None:
        yd = {
            **year_data,
            tgt_year: {
                reg: {a: x for a, x in apples.items() if a in keep[reg]} for reg, apples in year_data[tgt_year].items()
            },
        }
    f1s = [float(bmc.kshot_eval(cache, yd, tgt_year, src_years, K_SHOT, seed + i, 0.0)["f1"]) for i in range(reps)]
    return float(np.mean(f1s)), len(f1s)


def eval_public(encoder: nn.Module, x: NDArray, y: NDArray, classes: list[int], rc: RunCfg) -> tuple[float, int]:
    """公开数据集的 K=5 评估：cdl.eval_kshot（与 74 号同一调用），返回 (宏 F1 的情节均值, 情节数)。

    eval_kshot 只返回均值。每次重采样是否有效只取决于各类样本数：各类都有 ≥ K+2 个样本时全部有效，否则全部作废。
    """
    n_ep = rc.reps if all(int((y == c).sum()) >= K_SHOT + 2 for c in classes) else 0
    return float(cdl.eval_kshot(encoder, x, y, classes, rc.device, K_SHOT, rc.reps, rc.seed)), n_ep


# ═══════════════════════════════════════════════════════════════════════════════
# 四、第一部分：监督式训练的暴露臂
# ═══════════════════════════════════════════════════════════════════════════════
def log_training(logger: eu.Logger, arm: str, info: dict[str, Any]) -> None:
    logger.log(
        f"    [{arm}] 训练池 {info['训练池光谱数']} 条光谱、{info['分类头类别数']} 类；"
        f"{info['梯度步数']} 步 × 每步 {info['小批量']} 条（情节式一个情节的样本数）；"
        f"末段平均损失 {info['末段平均损失']:.4f}"
    )


def sup_apple(family: str, designs: list[str], apple: AppleSet, rc: RunCfg, logger: eu.Logger) -> tuple[Rows, Rows]:
    """苹果：逐设计逐折训练 R、S_src、S_shuf、S_half 四个监督式编码器，在留出年全部苹果与 B 半上评估。"""
    recipe = RECIPES["apple"]
    steps = scaled(recipe.steps, rc.stage)
    rows: Rows = []
    train_rows: Rows = []
    for design in designs:
        for tgt_year, cfg in sorted(m64.DESIGNS[design].items()):
            src: Pool = {e: apple.apple_data[e] for e in cfg["src"]}
            part_a, part_b = split_half_apple(apple.apple_data, cfg["tgt"], rc.seed, tgt_year)
            pools: dict[str, Pool] = {
                "R": src,
                "S_src": {e: apple.apple_data[e] for e in cfg["src"] + cfg["tgt"]},
                "S_shuf": {**src, **shuffle_apple(apple.apple_data, cfg["tgt"], rc.seed, tgt_year)},
                "S_half": {**src, **part_a},
            }
            base = cell_base("sup", "apple", f"apple_{design}", f"Y{tgt_year}", family, SUP_METHOD, rc)
            n_a, n_b = sum(len(v) for v in part_a.values()), sum(len(v) for v in part_b.values())
            logger.log(f"\n【apple_{design} 留出 {tgt_year}】{family}  留出年 A 半 {n_a}、B 半 {n_b} 苹果")
            caches: dict[str, Any] = {}
            for arm, pool in pools.items():
                x, y = pool_arrays(pool)
                batch = episode_size_apple(pool, recipe)
                encoder, info = train_supervised(family, x, y, apple.n_bands, recipe, steps, batch, rc.device, rc.seed)
                caches[arm] = bmc.FeatureCache(encoder, rc.device, True)
                train_rows.append({**base, "臂": arm, **info})
                log_training(logger, arm, info)
            fold_rows: Rows = []
            for arm, src_arm, where in SUP_EVALS:
                keep = part_b if where == "B半" else None
                f1, n_ep = eval_apple(
                    caches[src_arm], apple.year_data, tgt_year, cfg["src_years"], rc.reps, rc.seed, keep
                )
                fold_rows.append({**base, "臂": arm, "评估样本": where, "宏F1": f1, "情节数": n_ep})
            logger.log("    " + "  ".join(f"{r['臂']}={r['宏F1']:.4f}" for r in fold_rows))
            rows += fold_rows
    return rows, train_rows


def sup_public(family: str, datasets: list[str], rc: RunCfg, logger: eu.Logger) -> tuple[Rows, Rows]:
    """公开数据集：逐留出域训练四个监督式编码器（R、S_src、S_shuf、S_half）；类别集合与 B 半的规则同 74 号。"""
    recipe = RECIPES["public"]
    steps = scaled(recipe.steps, rc.stage)
    rows: Rows = []
    train_rows: Rows = []
    for dsname in datasets:
        doms, is_reg = load_public(dsname, rc)
        logger.log(f"\n【{dsname}】{family}  域 {[f'{d}:{len(v[1])}' for d, v in doms.items()]}")
        for held in list(doms):
            got = m74.fold_arrays(doms, is_reg, held, K_SHOT)
            if got is None:
                logger.log(f"  留出 {held}：合法类不足 3 个，此折不评")
                continue
            fa: cdl.FoldArrays = got[0]
            dom_of_row: NDArray = got[1]
            xa, ya, b_idx = split_half_public(fa, dom_of_row, held, rc.seed, dsname)
            xb, yb = fa.xh[b_idx], fa.yh[b_idx]
            classes_b = [c for c in fa.classes if int((yb == c).sum()) >= K_SHOT + 2]
            keep_b = np.isin(yb, classes_b)
            pools: dict[str, tuple[NDArray, NDArray]] = {
                "R": (fa.xs, fa.ys),
                "S_src": (fa.xall, fa.yall),
                "S_shuf": shuffle_public(fa, dom_of_row, held, rc.seed, dsname),
            }
            logger.log(f"  留出 {held}（{len(fa.classes)} 类；B 半 {len(yb)} 个样本、可评 {len(classes_b)} 类）")
            if len(classes_b) >= 3:
                pools["S_half"] = (xa, ya)
            else:
                logger.log("    B 半里样本数 ≥ K+2 的类不足 3 个：不训 S_half，不评 B 半上的两臂")
            base = cell_base("sup", "public", dsname, held, family, SUP_METHOD, rc)
            encs: dict[str, nn.Module] = {}
            for arm, (x, y) in pools.items():
                batch = episode_size_public(x, y, recipe)
                encs[arm], info = train_supervised(family, x, y, fa.in_dim, recipe, steps, batch, rc.device, rc.seed)
                train_rows.append({**base, "臂": arm, **info})
                log_training(logger, arm, info)
            fold_rows: Rows = []
            for arm, src_arm, where in SUP_EVALS:
                if where == "全部":
                    f1, n_ep = eval_public(encs[src_arm], fa.xh, fa.yh, fa.classes, rc)
                elif "S_half" in encs:
                    f1, n_ep = eval_public(encs[src_arm], xb[keep_b], yb[keep_b], classes_b, rc)
                else:
                    continue
                fold_rows.append({**base, "臂": arm, "评估样本": where, "宏F1": f1, "情节数": n_ep})
            logger.log("    " + "  ".join(f"{r['臂']}={r['宏F1']:.4f}" for r in fold_rows))
            rows += fold_rows
    return rows, train_rows


# ═══════════════════════════════════════════════════════════════════════════════
# 五、第二部分：无标签暴露（苹果与猕猴桃）
# ═══════════════════════════════════════════════════════════════════════════════
def unlab_train_row(
    base: dict[str, Any], arm: str, n_pool: int, epochs: int, steps: int, tail: float
) -> dict[str, Any]:
    """训练摘要里无标签暴露的一行；steps 为 0 表示预训练后不再训练。"""
    return {
        **base,
        "臂": arm,
        "预训练光谱数": n_pool,
        "预训练轮数上限": epochs,
        "源域情节训练步数": steps,
        "末段平均损失": tail,
    }


def unlab_apple(
    designs: list[str], apple: AppleSet, rc: RunCfg, logger: eu.Logger, selfcheck: bool
) -> tuple[Rows, Rows]:
    """苹果：逐设计逐折，两种自监督预训练各训 U_src、U_tgt；先冻结评估，再各自只用源域带标签数据做情节训练后评估。"""
    recipe = RECIPES["apple"]
    family = UNLAB_FAMILIES[0]
    epochs = scaled(SSM_EPOCHS, rc.stage)
    steps = scaled(recipe.steps, rc.stage)
    doms = apple_year_domains(apple.apple_data)
    if selfcheck:
        check_rewrite(doms[min(doms)][0], apple.n_bands, rc, epochs, logger)
    rows: Rows = []
    train_rows: Rows = []
    for design in designs:
        for tgt_year, cfg in sorted(m64.DESIGNS[design].items()):
            src_years = sorted(cfg["src_years"])
            if {ELEM_INFO[e][0] for e in cfg["src"]} != set(src_years):
                raise RuntimeError(f"{design} 留出 {tgt_year}：源域的域清单与年份清单不一致")
            src_pool: Pool = {e: apple.apple_data[e] for e in cfg["src"]}
            pools = {
                "U_src": np.vstack([doms[yr][0] for yr in src_years]),
                "U_tgt": np.vstack([doms[yr][0] for yr in sorted([*src_years, tgt_year])]),
            }
            logger.log(
                f"\n【apple_{design} 留出 {tgt_year}】预训练光谱 U_src {len(pools['U_src'])} 条、"
                f"U_tgt {len(pools['U_tgt'])} 条（源年份 {src_years}）"
            )
            for method in (SSL_MASK, SSL_CONTRAST):
                encs = {
                    arm: ssl_pretrain(method, x, apple.n_bands, recipe.feat_dim, rc.device, epochs, rc.seed)
                    for arm, x in pools.items()
                }
                for label in (method, method + SRC_STAGE):
                    base = cell_base("unlab", "apple", f"apple_{design}", f"Y{tgt_year}", family, label, rc)
                    fold_rows: Rows = []
                    for arm, encoder in encs.items():
                        tail, n_steps = float("nan"), 0
                        if label != method:
                            sampler = EpisodeSampler(
                                sorted(src_pool), src_pool, APPLE_N_WAY, recipe.k_support, recipe.q_query
                            )
                            tail, n_steps = episodic_train(encoder, sampler, recipe, steps, rc.device, rc.seed), steps
                        cache = bmc.FeatureCache(encoder, rc.device, True)
                        f1, n_ep = eval_apple(cache, apple.year_data, tgt_year, cfg["src_years"], rc.reps, rc.seed)
                        fold_rows.append({**base, "臂": arm, "评估样本": "全部", "宏F1": f1, "情节数": n_ep})
                        train_rows.append(unlab_train_row(base, arm, len(pools[arm]), epochs, n_steps, tail))
                    logger.log(f"    {label}：" + "  ".join(f"{r['臂']}={r['宏F1']:.4f}" for r in fold_rows))
                    rows += fold_rows
    return rows, train_rows


def unlab_public(datasets: list[str], rc: RunCfg, logger: eu.Logger, selfcheck: bool) -> tuple[Rows, Rows]:
    """公开数据集（缺省只有猕猴桃）：与 34、35 号同一调用训 U_src、U_tgt 并评估，再各自做源域情节训练后评估。"""
    recipe = RECIPES["public"]
    family = UNLAB_FAMILIES[0]
    epochs = scaled(SSM_EPOCHS, rc.stage)
    steps = scaled(recipe.steps, rc.stage)
    rows: Rows = []
    train_rows: Rows = []
    for dsname in datasets:
        doms, is_reg = load_public(dsname, rc)
        logger.log(f"\n【{dsname}】域 {[f'{d}:{len(v[1])}' for d, v in doms.items()]}")
        for held in list(doms):
            fa = cdl.prepare_fold(doms, is_reg, held, K_SHOT, preprocess="snv")
            if fa is None:
                logger.log(f"  留出 {held}：合法类不足 3 个，此折不评")
                continue
            if selfcheck:
                check_rewrite(fa.xs, fa.in_dim, rc, epochs, logger)
                selfcheck = False
            pools = {"U_src": fa.xs, "U_tgt": fa.xall}
            logger.log(f"  留出 {held}：预训练光谱 U_src {len(fa.xs)} 条、U_tgt {len(fa.xall)} 条")
            for method in (SSL_MASK, SSL_CONTRAST):
                encs = {
                    "U_src": cdl_pretrain(method, pools["U_src"], fa.in_dim, rc.device, epochs, rc.seed),
                    "U_tgt": cdl_pretrain(method, pools["U_tgt"], fa.in_dim, rc.device, epochs, rc.seed),
                }
                for label in (method, method + SRC_STAGE):
                    base = cell_base("unlab", "public", dsname, held, family, label, rc)
                    fold_rows: Rows = []
                    for arm, encoder in encs.items():
                        tail, n_steps = float("nan"), 0
                        if label != method:
                            sampler = cdl.ArrSampler(fa.xs, fa.ys, recipe.k_support, recipe.q_query)
                            tail, n_steps = episodic_train(encoder, sampler, recipe, steps, rc.device, rc.seed), steps
                        f1, n_ep = eval_public(encoder, fa.xh, fa.yh, fa.classes, rc)
                        fold_rows.append({**base, "臂": arm, "评估样本": "全部", "宏F1": f1, "情节数": n_ep})
                        train_rows.append(unlab_train_row(base, arm, len(pools[arm]), epochs, n_steps, tail))
                    logger.log(f"    {label}：" + "  ".join(f"{r['臂']}={r['宏F1']:.4f}" for r in fold_rows))
                    rows += fold_rows
    return rows, train_rows


# ═══════════════════════════════════════════════════════════════════════════════
# 六、逐种子结果表与汇总
# ═══════════════════════════════════════════════════════════════════════════════
def write_seed_table(
    part: str,
    tag: str,
    family: str,
    rc: RunCfg,
    rows: Rows,
    train_rows: Rows,
    budget: dict[str, float],
    logger: eu.Logger,
) -> str:
    """逐种子结果表：逐格结果（一行一格）、训练摘要、compute。"""
    suffix = "" if rc.stage == "formal" else "__pilot"
    os.makedirs(SEED_DIR, exist_ok=True)
    out = os.path.join(SEED_DIR, f"{SCRIPT}__{part}__{tag}__{family}__seed{rc.seed}{suffix}.xlsx")
    compute = {"seed": rc.seed, "stage": rc.stage, "part": part, "scope": tag, "encoder": family, **budget}
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame(rows, columns=CELL_COLUMNS).to_excel(writer, sheet_name=SHEET_CELLS, index=False)
        pd.DataFrame(train_rows).to_excel(writer, sheet_name=SHEET_TRAIN, index=False)
        pd.DataFrame([compute]).to_excel(writer, sheet_name="compute", index=False)
    logger.log(f"结果表 → {out}（{len(rows)} 格）")
    return out


def seed_files(stage: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(SEED_DIR, f"{SCRIPT}__*__seed*.xlsx")))
    return [f for f in files if ("__pilot" in os.path.basename(f)) == (stage == "pilot")]


def load_cells(stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读入本脚本的全部逐种子表，返回 (逐格长表, 各次运行的 compute)；同一格出现在多个文件里且数值不同即报错。"""
    files = seed_files(stage)
    if not files:
        raise FileNotFoundError(f"没有 88 号的逐种子结果（{stage}）：{SEED_DIR}")
    cells: list[pd.DataFrame] = []
    compute: list[pd.DataFrame] = []
    for f in files:
        cells.append(pd.read_excel(f, sheet_name=SHEET_CELLS))
        one = pd.read_excel(f, sheet_name="compute")
        one.insert(0, "文件", os.path.basename(f))
        compute.append(one)
    long_df = pd.concat(cells, ignore_index=True)
    dup = long_df.duplicated(CELL_KEYS, keep=False)
    if bool(dup.any()):
        if int(long_df[dup].groupby(CELL_KEYS)["宏F1"].nunique(dropna=False).max()) > 1:
            raise ValueError("同一格在多个逐种子表里的数值不同；先清掉重复运行留下的表再汇总")
        long_df = long_df.drop_duplicates(CELL_KEYS).reset_index(drop=True)
    return long_df, pd.concat(compute, ignore_index=True)


def cell_map(df: pd.DataFrame) -> CellMap:
    """逐格长表 → {(数据集, 留出域, 编码器族, 方式, 臂): {种子: 宏 F1}}。"""
    out: CellMap = {}
    cols = [df[c] for c in ("数据集", "留出域", "编码器族", COL_METHOD, "臂", "种子", "宏F1")]
    for ds, held, fam, method, arm, seed, f1 in zip(*cols, strict=True):
        out.setdefault((str(ds), str(held), str(fam), str(method), str(arm)), {})[int(seed)] = float(f1)
    return out


def epi_file_64(design: str, family: str, seed: int, suffix: str) -> str:
    """64 号逐种子表的文件名（main 的写法）：cnn 不带编码器段，其余族带 `__<族>`。"""
    seg = "" if family == "cnn" else f"__{family}"
    return f"64_leakage__{design}__{APPLE_VARIANT}{seg}__seed{seed}{suffix}.xlsx"


def epi_file_74(family: str, seed: int, suffix: str) -> str:
    """74 号逐种子表的文件名（main 的写法）。"""
    return f"74_multi_encoder__{family}__seed{seed}{suffix}.xlsx"


def load_episodic(stage: str, seeds: list[int], families: list[str]) -> tuple[EpiMap, list[str]]:
    """64 号（苹果）与 74 号（公开数据集）逐种子表里情节式训练各臂的宏 F1，返回 (数值, 缺少的文件名)。"""
    suffix = "" if stage == "formal" else "__pilot"
    out: EpiMap = {}
    missing: list[str] = []
    for family in families:
        for seed in seeds:
            for design in APPLE_DESIGNS:
                name = epi_file_64(design, family, seed, suffix)
                if not os.path.exists(os.path.join(SEED_DIR, name)):
                    missing.append(name)
                    continue
                d = pd.read_excel(os.path.join(SEED_DIR, name), sheet_name=EPI_SHEET_64)
                d = d[d[EPI_COLS_64["alpha"]] == 0.0]
                cols = [d[EPI_COLS_64[c]] for c in ("held", "arm", "f1")]
                for held, arm, f1 in zip(*cols, strict=True):
                    out.setdefault((f"apple_{design}", f"Y{int(held)}", family, str(arm)), {})[seed] = float(f1)
            name = epi_file_74(family, seed, suffix)
            if not os.path.exists(os.path.join(SEED_DIR, name)):
                missing.append(name)
                continue
            d = pd.read_excel(os.path.join(SEED_DIR, name), sheet_name=EPI_SHEET_74)
            cols = [d[EPI_COLS_74[c]] for c in ("dataset", "held", "arm", "f1")]
            for ds, held, arm, f1 in zip(*cols, strict=True):
                out.setdefault((str(ds), str(held), family, str(arm)), {})[seed] = float(f1)
    return out, missing


def t_interval(values: list[float]) -> tuple[float, float]:
    """跨种子 95% t 区间（与 74 号 aggregate 相同的算式）；不足两个种子时返回 nan。"""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float("nan"), float("nan")
    half = float(stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr)))
    return float(arr.mean() - half), float(arr.mean() + half)


def seed_mean(vals: dict[int, float] | None) -> float:
    finite = [v for v in (vals or {}).values() if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def paired(name: str, a: dict[int, float] | None, b: dict[int, float] | None) -> dict[str, Any]:
    """a − b 按种子配对：均值、种子 t 区间、为正种子数、配对种子数（列名以 name 开头）；任一臂缺失时返回空字典。"""
    if not a or not b:
        return {}
    dl = [a[s] - b[s] for s in sorted(set(a) & set(b)) if np.isfinite(a[s]) and np.isfinite(b[s])]
    if not dl:
        return {}
    lo, hi = t_interval(dl)
    vals = (float(np.mean(dl)), lo, hi, int(sum(d > 0 for d in dl)), len(dl))
    return {f"{name}{k}": v for k, v in zip(STAT_KEYS, vals, strict=True)}


def stat_cols(name: str) -> list[str]:
    return [f"{name}{k}" for k in STAT_KEYS]


def ordered_groups(cells: CellMap, width: int) -> list[tuple[str, ...]]:
    """格键的前 width 项去重后排序：数据集按 DATASET_ORDER，留出域按首次出现的先后，族与方式按 ITEM_RANK。"""
    held_pos: dict[tuple[str, str], int] = {}
    for key in cells:
        held_pos.setdefault((key[0], key[1]), len(held_pos))
    ds_rank = {ds: i for i, ds in enumerate(DATASET_ORDER)}
    groups = list(dict.fromkeys(key[:width] for key in cells))
    return sorted(
        groups,
        key=lambda g: (
            ds_rank.get(g[0], len(ds_rank)),
            held_pos[(g[0], g[1])],
            *[ITEM_RANK.get(v, len(ITEM_RANK)) for v in g[2:]],
        ),
    )


def domain_summary(label: str, vals: list[float]) -> dict[str, Any]:
    """留出域级的数值（各为种子均值）汇成：留出域数、为正留出域数 / 总数、均值。"""
    return {
        f"{label}_留出域数": len(vals),
        f"{label}_为正留出域": f"{sum(v > 0 for v in vals)}/{len(vals)}",
        f"{label}_均值": float(np.mean(vals)) if vals else float("nan"),
    }


def sup_fold_rows(sup: CellMap, epi: EpiMap) -> Rows:
    """监督式各对照按 (数据集, 留出域, 族) 的种子统计，并排列出 64、74 号情节式训练的同一对照。"""
    rows: Rows = []
    for ds, held, fam in ordered_groups(sup, 3):
        arms = {a: sup.get((ds, held, fam, SUP_METHOD, a)) for a in SUP_ARMS}
        row: dict[str, Any] = {"数据集": ds, "留出域": held, "编码器族": fam}
        row.update({a: seed_mean(v) for a, v in arms.items()})
        for name, (a, b) in SUP_CONTRASTS.items():
            row.update(paired(name, arms[a], arms[b]))
            ea, eb = EPI_CONTRASTS[name]
            row.update(paired(f"情节式_{name}", epi.get((ds, held, fam, ea)), epi.get((ds, held, fam, eb))))
        rows.append(row)
    return rows


def sup_summary_rows(rows: Rows) -> Rows:
    """按族的跨域汇总：主口径（苹果两年设计 + 公开数据集）与苹果三年设计分开，监督式与情节式并排。"""
    out: Rows = []
    for scope, (sets, expected) in SUP_SCOPES.items():
        for fam in dict.fromkeys(r["编码器族"] for r in rows):
            sub = [r for r in rows if r["数据集"] in sets and r["编码器族"] == fam]
            for name in SUP_CONTRASTS:
                rec: dict[str, Any] = {"口径": scope, "应有留出域数": expected, "编码器族": fam, "对照": name}
                for label, col in (("监督式", name), ("情节式", f"情节式_{name}")):
                    rec.update(domain_summary(label, [float(r[col]) for r in sub if col in r]))
                out.append(rec)
    return out


def unlab_fold_rows(unlab: CellMap) -> Rows:
    """U_tgt − U_src 按 (数据集, 留出域, 预训练方式) 的种子统计。"""
    rows: Rows = []
    for ds, held, fam, method in ordered_groups(unlab, 4):
        src, tgt = unlab.get((ds, held, fam, method, "U_src")), unlab.get((ds, held, fam, method, "U_tgt"))
        row: dict[str, Any] = {"数据集": ds, "留出域": held, "编码器族": fam, COL_METHOD: method}
        row.update({"U_src": seed_mean(src), "U_tgt": seed_mean(tgt)})
        row.update(paired("Δ_unsup", tgt, src))
        rows.append(row)
    return rows


def unlab_summary_rows(rows: Rows) -> Rows:
    """无标签对照按 (数据集, 预训练方式) 的跨留出域汇总。"""
    out: Rows = []
    for ds, method in dict.fromkeys((r["数据集"], r[COL_METHOD]) for r in rows):
        vals = [float(r["Δ_unsup"]) for r in rows if (r["数据集"], r[COL_METHOD]) == (ds, method) and "Δ_unsup" in r]
        out.append({"数据集": ds, COL_METHOD: method, **domain_summary("Δ_unsup", vals)})
    return out


def aggregate(stage: str, logger: eu.Logger) -> None:
    """跨种子汇总：监督式的 Δ_rep、Δ_lab、Δ_half（并排列出情节式训练的同一对照）与无标签的 Δ_unsup。"""
    long_df, compute_df = load_cells(stage)
    seeds = sorted({int(s) for s in long_df["种子"].tolist()})
    if stage == "formal" and set(seeds) != set(eu.RECOMMENDED_SEEDS):
        logger.log(f"  [WARN] 种子集合 {seeds} 不是 Formal 5 种子，汇总仅作进度查看")
    sup = cell_map(long_df[long_df["部分"] == "sup"])
    unlab = cell_map(long_df[long_df["部分"] == "unlab"])
    epi: EpiMap = {}
    missing: list[str] = []
    if sup:
        epi, missing = load_episodic(stage, seeds, list(dict.fromkeys(key[2] for key in sup)))
        if missing:
            logger.log(f"  [WARN] 缺 {len(missing)} 个 64、74 号的逐种子表，情节式一侧相应留空：{missing[:10]}")
    sup_rows = sup_fold_rows(sup, epi)
    unlab_rows = unlab_fold_rows(unlab)
    sup_cols = ["数据集", "留出域", "编码器族", *SUP_ARMS]
    for name in SUP_CONTRASTS:
        sup_cols += [*stat_cols(name), *stat_cols(f"情节式_{name}")]
    unlab_cols = ["数据集", "留出域", "编码器族", COL_METHOD, "U_src", "U_tgt", *stat_cols("Δ_unsup")]
    epi_note = (
        "64 号（苹果，F1 均值存表时已舍入到 4 位）与 74 号（公开数据集）逐种子表里情节式训练的同一对照，"
        "按同一批种子配对；其 Δ_half = S_half − R_B，B 半是 64、74 号各自的划分，与本脚本的不是同一次"
    )
    notes = [
        ("Δ_rep", "S_src − R：留出域全部样本上评估，种子内配对作差"),
        ("Δ_lab", "S_src − S_shuf：两臂训练池的光谱相同，只差留出域标签是否置乱；留出域全部样本上评估"),
        ("Δ_half", "S_half − R_h：只在 B 半上评估；R_h 是 R 的编码器在同一 B 半上的成绩"),
        ("Δ_unsup", "U_tgt − U_src：同一预训练方式下只差预训练数据是否含留出域光谱"),
        ("情节式_*", epi_note),
        ("为正留出域", "留出域级的值（种子均值）大于 0 的个数 / 有该对照的留出域数"),
        ("缺少的 64、74 号逐种子表", "；".join(missing) if missing else "无"),
    ]
    tables: list[tuple[str, pd.DataFrame]] = [
        (eu.table_sheet_name(0, "监督式逐留出域对照"), pd.DataFrame(sup_rows, columns=sup_cols)),
        (eu.table_sheet_name(1, "监督式按族跨域汇总"), pd.DataFrame(sup_summary_rows(sup_rows))),
        (eu.table_sheet_name(2, "无标签逐留出域对照"), pd.DataFrame(unlab_rows, columns=unlab_cols)),
        (eu.table_sheet_name(3, "无标签按数据集汇总"), pd.DataFrame(unlab_summary_rows(unlab_rows))),
        (eu.table_sheet_name(4, "逐种子长格式"), long_df),
        (eu.table_sheet_name(5, "说明与缺表"), pd.DataFrame(notes, columns=["项", "内容"])),
        ("compute", compute_df),
    ]
    for title, table in tables[:4]:
        logger.log(f"\n{title}\n" + table.to_string(index=False))
    if stage == "formal":
        sheets: dict[object, object] = {}
        for title, table in tables:
            sheets[title] = table
        out = eu.write_script_workbook(__file__, sheets)
    else:
        out = os.path.join(eu.OUTPUT_DIR, f"{SCRIPT}__pilot.xlsx")
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for title, table in tables:
                table.to_excel(writer, sheet_name=title, index=False)
    logger.log(f"\n汇总表 → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 七、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def pick(raw: str, default: tuple[str, ...], allowed: tuple[str, ...], what: str) -> list[str]:
    """逗号分隔的命令行取值：空串取缺省值，未知名字直接退出；返回值按 allowed 的次序排列。"""
    chosen = [v for v in raw.split(",") if v] or list(default)
    unknown = sorted(set(chosen) - set(allowed))
    if unknown:
        raise SystemExit(f"未知{what}：{unknown}（可选 {list(allowed)}）")
    return [v for v in allowed if v in chosen]


def scope_tag(scope: str, chosen: list[str], full: tuple[str, ...]) -> str:
    """输出文件名里的范围段：跑的是该范围的全部缺省项时只写范围名，否则把所跑的项接在后面。"""
    return scope if chosen == list(full) else f"{scope}-{'+'.join(chosen)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="88_supervised_unlabelled：监督式训练下的暴露对照与无标签暴露对照")
    parser.add_argument("--part", choices=["sup", "unlab"], help="sup = 监督式训练的暴露臂；unlab = 无标签暴露")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED, help="一次一个种子")
    parser.add_argument("--stage", choices=["pilot", "formal"], help="缺省：运行时 pilot，--aggregate 时 formal")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data_root", default="05data")
    parser.add_argument("--scope", default="all", choices=["apple", "public", "all"])
    parser.add_argument("--designs", default=",".join(APPLE_DESIGNS), help="逗号分隔的苹果设计")
    parser.add_argument(
        "--datasets", default="", help="逗号分隔的公开数据集；缺省 sup 为 74 号登记的五个，unlab 为猕猴桃"
    )
    parser.add_argument("--encoders", default="", help="逗号分隔的编码器族；缺省 sup 为 cnn,resnet，unlab 只有 cnn")
    parser.add_argument("--aggregate", action="store_true", help="只做跨种子汇总")
    args = parser.parse_args()
    stage: str = args.stage or ("formal" if args.aggregate else "pilot")

    if args.aggregate:
        logger = eu.get_logger(SCRIPT)
        aggregate(stage, logger)
        logger.close()
        return
    if args.part is None:
        raise SystemExit("--part 必填：sup 或 unlab（只做汇总时用 --aggregate）")
    sup = args.part == "sup"
    scopes = ["apple", "public"] if args.scope == "all" else [args.scope]
    designs = pick(args.designs, APPLE_DESIGNS, APPLE_DESIGNS, "苹果设计")
    public_default = PUBLIC_SETS if sup else UNLAB_PUBLIC_SETS
    datasets = pick(args.datasets, public_default, PUBLIC_SETS, "公开数据集")
    family_default = SUP_FAMILIES if sup else UNLAB_FAMILIES
    families = pick(args.encoders, family_default, enc_lib.DEEP_ENCODERS if sup else UNLAB_FAMILIES, "编码器族")

    device = eu.get_device(args.device)
    logger = eu.get_logger(SCRIPT)
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=stage, logger=logger)
    rc = RunCfg(stage=stage, seed=args.seed, reps=REPS[stage], device=device, data_root=args.data_root)
    apple_steps, public_steps = scaled(RECIPES["apple"].steps, stage), scaled(RECIPES["public"].steps, stage)
    eu.log_experiment_header(
        logger,
        {
            "任务": "监督式训练下的暴露对照" if sup else "无标签暴露对照（苹果与公开数据集）",
            "部分": args.part,
            "阶段": stage,
            "设备": str(device),
            "种子": args.seed,
            "范围": scopes,
            "苹果设计": designs,
            "公开数据集": datasets,
            "编码器族": families,
            "臂": list(SUP_ARMS) if sup else ["U_src", "U_tgt"],
            "K / 每格评估情节": f"{K_SHOT} / {rc.reps}",
            "梯度步数（苹果 / 公开数据集）": f"{apple_steps} / {public_steps}",
            "自监督预训练轮数上限": scaled(SSM_EPOCHS, stage),
            "分类头输出所乘的倍数": LOGIT_SCALE,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, stage, {**vars(args), "stage": stage})
    selfcheck = not sup and stage == "pilot" and device.type in ("cuda", "cpu")
    for scope in scopes:
        apple = load_apple(logger) if scope == "apple" else None
        chosen, full = (designs, APPLE_DESIGNS) if scope == "apple" else (datasets, public_default)
        tag = scope_tag(scope, chosen, full)
        for family in families:
            with eu.track_compute(f"{args.part} / {tag} / {family}", logger, run) as budget:
                if sup and apple is not None:
                    rows, train_rows = sup_apple(family, designs, apple, rc, logger)
                elif sup:
                    rows, train_rows = sup_public(family, datasets, rc, logger)
                elif apple is not None:
                    rows, train_rows = unlab_apple(designs, apple, rc, logger, selfcheck)
                else:
                    rows, train_rows = unlab_public(datasets, rc, logger, selfcheck)
                selfcheck = False
            write_seed_table(args.part, tag, family, rc, rows, train_rows, dict(budget), logger)
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
