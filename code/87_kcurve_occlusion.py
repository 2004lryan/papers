"""
87_kcurve_occlusion.py: 六个编码器族的 K 曲线（补充材料 S9.5）与暴露前后编码器的波段遮蔽归因（补充材料 S9.6）。

为什么:
    多编码器的暴露对照（64 号苹果、74 号公开数据集）都只在每类 5 个参考样本上评估；判别波段的结论
    （70 号）出自光谱空间的随机森林，不涉及编码器。本脚本对六个编码器族的折限定臂 R 与目标暴露臂
    S_src 补两件事：其一（补充材料 S9.5）把同一对编码器放到 K ∈ {1, 5, 10, 20} 上评估；其二（补充材料 S9.6）逐块遮住输入光谱里的一段
    波段，比较 R 与 S_src 各自依赖哪些波段、暴露之后多依赖了哪些。

编码器（R：训练池 = 源域；S_src：训练池 = 源域 ∪ 留出域带标签光谱；六个族见 encoders.ENCODERS）:
    苹果（two_year 2 折 + three_year 3 折，冻结数据）
        深度族  先按 64 号的 run_name 规则找 07checkpoints/64_* 下的正式权重，只读载入：模型卡逐项
                与正式配方一致、训练已走完、权重 SHA-256 与卡相符才用（核对项同 64 号
                obtain_encoder）。找不到或核对不过，就按正式配方重训（64 号 train_encoder：2000 情节，
                每 100 情节在源域情节上验证，耐心 12），权重写 07checkpoints/87_*；已落盘的 87_*
                权重核对后同样复用。本脚本不写 07checkpoints/64_*
        线性族  pls / pca 在训练池全部光谱上即时拟合（encoders.fit_linear_encoder，至多 16 个成分），
                不落盘
    公开数据集（74 号登记的五个数据集，16 个留出域）
        按 74 号的配方重训：每个留出域先 R 后 S_src（74 号 obtain → crossdataset_lib.meta_train，
        1000 情节；线性族即时拟合），次序与播种同 74 号 run_fold；不落盘

S9.5 K 曲线:
    类别集合  固定为 K=5 时的集合（苹果：留出年的全部产地；公开数据集：prepare_fold 在 K=5 下选出的类）
    跳过      某个 K 下留出域有类不足 K+2 个样本（苹果按果计）时该格不评，逐种子表记「是否跳过」，
              汇总里计数（74 号路线面板的规则）
    情节      每格 50 个（pilot 10 个），同一格内 R 与 S_src 用同一批。沿用正式结果的抽样，K=5 那一档
              与 64、74 号是同一批情节
    算法      各 K 都走原评估路径，K=5 的数值即 64、74 号正式逐种子表的算法：
              苹果        逐个重采样种子（第 i 个为 S+i）调 44 号 kshot_eval，参数同 64 号 evaluate：特征经
                          FeatureCache（一果一批，各面位谱分别编码后取均值再 L2 归一化；不是 72 号的单面
                          口径，2018、2019 年每果一条谱，两种口径在这两年相同）、目标年全部苹果、设计登记
                          的源年份、α=0。kshot_eval 以 default_rng(S+i) 在各产地按苹果编号无放回抽 K 个
                          支持果，其余全部苹果作查询
              公开数据集  情节取 74 号 kshot_episodes（与 crossdataset_lib.eval_kshot 逐次相同的抽样），
                          逐情节按 eval_kshot 的分批编码（74 号 proto_f1：各类支持样本各成一批，查询样本
                          一批）；K=5 的均值与 eval_kshot 的返回值相差超过 1e-9 即中止
    本脚本没有新增的随机抽样；训练函数保持各自原有的播种方式。

S9.6 波段遮蔽:
    范围      苹果 5 折与芒果、猕猴桃的 7 个留出域（--occl_datasets）；玉米、药片、Indian Pines 不做
    分块      相邻 5 个波段一块（--block；苹果 229 个波段共 46 块，末块 4 个波段），同 70 号的分块
    遮蔽位置  编码器的输入空间，即数据加载器给出、送进编码器之前的表示：苹果是冻结数据的 229 波段
              逐面位谱，公开数据集是 prepare_fold 逐样本 SNV 之后的谱；遮蔽之后不再重做 SNV
    遮蔽取值  支持与查询光谱里该块的值换成块两端相邻波段取值之间的线性插值（横坐标取波长）；
              块在谱端时用最近的一个邻点的值填平；其余波段不动
    算法      批量算法：留出域全部样本一次编码（苹果再按果取均值、L2 归一化），在 K 曲线 K=5 那一批情节上
              算宏 F1；未遮蔽值与各块遮蔽后的值都这样算，编码器不重训
    下降量    未遮蔽 − 遮蔽后的 K=5 宏 F1，是批量算法内的差

自检:
    运行时  K=5 未遮蔽宏 F1 有两个值：K 曲线的原评估路径与波段遮蔽的批量算法。两者同一批情节、同一编码器，
            只差在分批；分批不同引起浮点末位差，偶尔翻动个别判别。差逐格记在「编码器来源与核对」表并折成
            判别数，不为零的格在日志里列出；超过 0.01 记 WARN（见于嵌入几乎没有区分度的格）。
            不做遮蔽的格不运行批量算法
    汇总时  K=5 的 R、S_src 与 S_src − R 逐（数据集, 留出域, 编码器族, 种子）对 64、74 号正式
            逐种子表，相差超过 0.02 的格全部列进「自检超差格」，正式表里找不到的格计入「缺参照」；
            另报逐位相同的格数 / 总格数（64 号的表存到 4 位小数，按同样的取舍后比）
            （--stage pilot 时对两脚本的 pilot 表；预算未必相同，超差格只供查看）

汇总（--aggregate）:
    K 曲线  各（数据集, 留出域, 编码器族, K）的 S_src − R：种子均值、种子 t 区间、为正种子数；
            按编码器族与 K 的跨域汇总（为正留出域数 / 可评留出域数、均值），主口径（苹果两年设计 +
            公开数据集，18 个留出域）与苹果三年设计分开
    遮蔽    各（数据集, 留出域, 编码器族）R 与 S_src 两条下降量曲线（种子均值）的 Spearman 秩相关；
            S_src 比 R 多依赖最多的三块（下降量之差最大）及其波长范围；苹果另给 606–699 nm 各块
            （块中心波长落在其中）的下降量占全部正下降量之和的比例，R 与 S_src 各一

运行方式:
    cd <项目根>
    # pilot（缺省种子 2004；训练情节与评估情节数为正式的 1/5，不要求 CUDA）
    python 02code/87_kcurve_occlusion.py --device auto --stage pilot --scope all --encoders cnn,pls \
        --data_root <公开数据集根目录>
    # formal（CUDA；一次一个种子，可按范围与编码器族拆成并行任务）
    python 02code/87_kcurve_occlusion.py --device cuda --stage formal --seed 20060515 --scope apple \
        --encoders cnn
    python 02code/87_kcurve_occlusion.py --device cuda --stage formal --seed 20060515 --scope public \
        --encoders transformer --data_root <公开数据集根目录>
    # 汇总与自检（缺省读正式结果；加 --stage pilot 读 pilot 结果）
    python 02code/87_kcurve_occlusion.py --aggregate
    --designs / --datasets / --k_list / --block / --occl_datasets / --skip_occlusion 取非缺省值时须同时给
    --tag <标签>：结果写进带标签的文件，不与缺省设置的结果互相覆盖

输出文件:
    04outputs/formal_seeds/87_kcurve_occlusion__<范围>__<编码器>__seed<S>[__<标签>][__pilot].xlsx
        — 表a K曲线逐格；表b 波段遮蔽逐块；表c 编码器来源与核对；说明；compute
    04outputs/87_kcurve_occlusion[__<标签>][__pilot].xlsx（--aggregate）
        — 表a K曲线逐域；表b K曲线跨域汇总；表c 自检超差格；表d 自检汇总；表e 遮蔽逐域汇总；
          表f 遮蔽多依赖前三块；表g 遮蔽逐块均值；表h 缺表；说明；compute
    07checkpoints/87_<设计>_canonical[_<编码器>]_<臂>_seed<S>_fold<年>[_pilot]/best.pt
        — 苹果一侧重训的权重（含模型卡）
    05logs/87_kcurve_occlusion_YY-MM-DD_HHMMSS.log

复用的现有函数:
    export_utils      设备、播种、CUDA 闸、日志、算力、SHA-256、工作表名、波段列与波长轴、路径常量
    64 号             DESIGNS、LEGACY_CARD_PROTOCOL、load_data、read_card、train_encoder；R 与 S_src 的
                      训练池定义、模型卡字段与复用核对项照 main / obtain_encoder 的写法在本脚本里实现
                      （原函数核对不过会重训并覆盖 64_* 权重，故不直接调用）
    44 号（经 64 号） FeatureCache、kshot_eval（苹果 K 曲线的评估）
    72 号             load_encoders 只载入、不训练的做法
    70 号、68 号      分块与块起止、中心波长的记法；606–699 nm 按块中心波长归区
    74 号             DATASETS、fold_arrays、obtain、kshot_episodes、proto_f1（公开数据集 K 曲线的评估）、
                      路线面板的跳过规则
    crossdataset_lib  各加载器、prepare_fold 与 meta_train（经 74 号）、eval_kshot（K=5 的逐位核对）
    encoders          ENCODERS、LINEAR_ENCODERS、build_encoder、fit_linear_encoder
    protonet          build_apple_data、build_year_region_data、encode_batch
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
from protonet import build_apple_data, build_year_region_data, encode_batch

NDArray = npt.NDArray[Any]
Episode = tuple[list[NDArray], NDArray]  # (各类支持单元的下标, 查询单元的下标)
AppleData = dict[str, dict[Any, dict[str, Any]]]
YearData = dict[int, dict[str, dict[Any, NDArray]]]
PublicSet = tuple[dict[str, tuple[NDArray, NDArray]], bool, NDArray | None]  # (各域数组, 是否回归, 波长轴)
RefKey = tuple[str, str, str, int, str]  # (数据集, 留出域, 编码器族, 种子, 臂)
Rows = list[dict[str, Any]]

m64: Any = importlib.import_module("64_leakage_controls")
m74: Any = importlib.import_module("74_multi_encoder_leakage")

STEM = "87_kcurve_occlusion"
VARIANT = "canonical"
ARMS = ("R", "S_src")
K_LIST = (1, 5, 10, 20)
K_REF = 5  # 定类别集合、做遮蔽与自检的 K，即 64、74 号正式结果的 K
BLOCK = 5  # 遮蔽分块的波段数，与 70 号相同
REPS = 50  # 每格情节数
APPLE_EPISODES = 2000  # 64 号正式配方的情节预算；其余训练设定取 m64.LEGACY_CARD_PROTOCOL
PUBLIC_EPISODES = 1000  # 74 号正式配方的元训练情节数
LIN_DIM = 16  # 线性族成分数上限，64、74 号相同
PILOT_DIV = 5  # pilot 时训练情节与评估情节数除以它
SCOPES = ("apple", "public")
APPLE_DESIGNS = {"apple_two_year": "two_year", "apple_three_year": "three_year"}  # 数据集名同 75 号
OCCL_DEFAULT = ("apple", "mango_season", "kiwifruit_season")
DATA_ROOT = "05data"  # 与 74 号的缺省值相同
PATH_TOL = 0.01  # 原评估路径与批量算法的 K=5 宏 F1 之差的告警线；差一律只记录（分批不同引起的浮点末位差）
EXACT_TOL = 1e-9  # 公开数据集：逐情节原评估路径的 K=5 均值与 cdl.eval_kshot 返回值之差的上限
SELFCHECK_TOL = 0.02  # 汇总自检：与 64、74 号正式逐种子表之差的上限
PIGMENT_BAND = (606.0, 700.0)  # 606–699 nm，按块中心波长归区（同 68 号表 S13 的分区口径）
MAIN_SCOPE = "主口径（苹果两年设计 + 公开数据集）"
THREE_SCOPE = "苹果三年设计"
EXPECTED_HELD = {MAIN_SCOPE: 18, THREE_SCOPE: 3}
GUARDED = ("designs", "datasets", "k_list", "block", "occl_datasets", "skip_occlusion")  # 非缺省时须带 --tag
SEED_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")

SRC_64 = "64号正式权重"
SRC_87_SAVED = "87号已落盘权重"
SRC_87_TRAIN = "87号重训"
SRC_74_TRAIN = "74号配方重训"
SRC_FIT = "即时拟合"

# 逐种子工作簿的工作表与列
SHEET_K = eu.table_sheet_name(0, "K曲线逐格")
SHEET_O = eu.table_sheet_name(1, "波段遮蔽逐块")
SHEET_E = eu.table_sheet_name(2, "编码器来源与核对")
ID_COLS = ["范围", "数据集", "留出域", "编码器族", "臂", "种子"]
K_COLS = [*ID_COLS, "K", "宏F1", "F1情节SD", "情节数", "是否跳过", "跳过原因"]
K_COLS += ["类别数", "最小类样本数", "编码器来源"]
O_COLS = [*ID_COLS, "块号", "块起波段", "块止波段", "块起nm", "块止nm", "块中心nm"]
O_COLS += ["K", "情节数", "未遮蔽宏F1", "遮蔽后宏F1", "下降量"]

# 汇总工作簿各表的列
FOLD_ID = ["口径", "范围", "数据集", "留出域", "编码器族", "K", "种子数", "跳过种子数"]
FOLD_VAL = ["R", "S_src", "Δ_rep", "Δ_rep_种子SD", "Δ_rep_种子t下限", "Δ_rep_种子t上限"]
FOLD_VAL += ["Δ_rep_为正种子数"]
FLAG_COLS = ["范围", "数据集", "留出域", "编码器族", "种子", "量", "本脚本", "正式表", "差", "编码器来源"]
OCC_ID = ["范围", "数据集", "留出域", "编码器族"]
OCC_SUM_COLS = [*OCC_ID, "块数", "种子数", "未遮蔽R", "未遮蔽S_src", "秩相关_种子均值曲线"]
OCC_SUM_COLS += ["秩相关_逐种子均值", "606–699nm正下降占比_R", "606–699nm正下降占比_S_src"]
BLOCK_COLS = ["块号", "块起波段", "块止波段", "块起nm", "块止nm", "块中心nm"]

# 自检参照：64、74 号正式逐种子表。文件名、工作表次序与列名取自两脚本写表的代码
# （64 号 main 末尾与 evaluate；74 号 main 末尾与 run_fold），并与盘上的正式表核对过
REF64_FILE = "64_leakage__{design}__{variant}{enc_seg}__seed{seed}{suffix}.xlsx"  # enc_seg：cnn 为空
REF74_FILE = "74_multi_encoder__{encoder}__seed{seed}{suffix}.xlsx"
REF_SHEET = 0  # 第一张表：64 号「表a：K5各臂结果」，74 号「表a：逐臂结果」
REF64_COL = {"arm": "臂", "part": "评估苹果", "alpha": "src_alpha", "fold": "目标年份", "k": "K", "f1": "F1均值"}
REF74_COL = {"arm": "臂", "part": "评估样本", "dataset": "dataset", "held": "held", "f1": "F1"}
REF_PART_ALL = "全部"  # 两张表里「在全部评估样本上」的取值
REF64_DECIMALS = 4  # 64 号 evaluate 写表时 F1均值 取到的小数位；74 号写表时不取舍

NOTES: tuple[tuple[str, str], ...] = (
    (
        "K 曲线",
        "类别集合固定为 K=5 时的集合；留出域有类不足 K+2 个样本（苹果按果计）的格跳过；"
        "每格的宏 F1 是各情节的均值，同一格内 R 与 S_src 用同一批情节",
    ),
    (
        "K 曲线的算法",
        "各 K 都走原评估路径：苹果逐个重采样种子调 44 号 kshot_eval（参数同 64 号 evaluate，α=0）；"
        "公开数据集逐情节按 crossdataset_lib.eval_kshot 的分批编码（各类支持样本各成一批，查询样本一批）。"
        "K=5 的数值即 64、74 号正式逐种子表的算法",
    ),
    (
        "情节（苹果）",
        "第 i 个情节 default_rng(种子+i)：各产地按苹果编号无放回抽 K 个支持果，其余全部苹果作查询；"
        "单果特征为各面位谱分别编码后取均值再 L2 归一化（64 号 K=5 评估的口径）",
    ),
    (
        "情节（公开数据集）",
        "74 号 kshot_episodes，与 crossdataset_lib.eval_kshot 逐次相同："
        "np.random.seed(种子) 后逐类洗牌，前 K 个作支持，其余作查询",
    ),
    (
        "遮蔽位置",
        "编码器的输入空间，即数据加载器给出、送进编码器之前的表示：苹果是冻结数据的 229 波段逐面位谱，"
        "公开数据集是 prepare_fold 逐样本 SNV 之后的谱；遮蔽之后不再重做 SNV",
    ),
    (
        "遮蔽取值",
        "支持与查询光谱里该块的值换成块两端相邻波段取值之间的线性插值"
        "（横坐标取波长，没有波长轴的数据集取波段序号）；"
        "块在谱端时用最近的一个邻点的值填平；其余波段不动",
    ),
    (
        "分块与波长",
        "相邻若干个波段一块，末块不足时取剩余波段；块号与波段序号都从 0 起；"
        "块起nm、块止nm 是块内首末波段的波长，块中心nm 是块内各波段波长的均值",
    ),
    (
        "下降量",
        "未遮蔽 − 遮蔽后的 K=5 宏 F1；两者都用批量算法（留出域全部样本一次编码），"
        "在 K 曲线 K=5 那一批情节上算，下降量是这一算法内的差",
    ),
    (
        "两条路径",
        "K 曲线用原评估路径，波段遮蔽用批量算法；两者在 K=5 未遮蔽宏 F1 上的差来自分批不同引起的浮点末位差"
        f"（偶尔翻动个别判别），逐格记在「编码器来源与核对」表并折成判别数，超过 {PATH_TOL} 记 WARN"
        "（见于嵌入几乎没有区分度、大量查询处在平局边缘的格）；不做遮蔽的格不运行批量算法",
    ),
)


@dataclass
class Plan:
    """一次运行的设定（由命令行与阶段确定）。"""

    seed: int
    stage: str
    reps: int
    apple_episodes: int
    public_episodes: int
    k_values: list[int]
    block: int
    occl: set[str]
    designs: list[str]
    datasets: list[str]


@dataclass
class Cell:
    """一个留出域的评估对象。

    x 是编码器输入空间的光谱（每行一条）；owner 给出每行所属评估单元的下标，None 表示一行即一个单元
    （公开数据集），苹果一侧一果多面位；y 是各单元的类别；wl 是波长轴（nm），没有时为 None。
    """

    x: NDArray
    owner: NDArray | None
    y: NDArray
    classes: list[int]
    wl: NDArray | None


# ═══════════════════════════════════════════════════════════════════════════════
# 一、遮蔽，以及批量算法的特征与情节
# ═══════════════════════════════════════════════════════════════════════════════
def make_blocks(n_bands: int, block: int) -> list[tuple[int, int]]:
    """相邻 block 个波段一块（同 70 号），返回各块首末波段的下标（含末端）；末块不足时取剩余波段。"""
    if block < 1 or n_bands <= block:
        raise ValueError(f"分块无效：{n_bands} 个波段，每块 {block} 个")
    return [(i, min(i + block, n_bands) - 1) for i in range(0, n_bands, block)]


def occlude(x: NDArray, lo: int, hi: int, pos: NDArray) -> NDArray:
    """把第 lo..hi 列换成块两端相邻波段取值之间的线性插值（横坐标 pos），返回新数组，其余列不动。

    块在谱端、只有一侧有邻点时，用这个邻点的值填平。
    """
    out: NDArray = x.copy()
    last = x.shape[1] - 1
    if lo > 0 and hi < last:
        t = (pos[lo : hi + 1] - pos[lo - 1]) / (pos[hi + 1] - pos[lo - 1])
        left, right = x[:, lo - 1 : lo], x[:, hi + 1 : hi + 2]
        out[:, lo : hi + 1] = left + (right - left) * t[None, :]
    elif hi < last:
        out[:, lo : hi + 1] = x[:, hi + 1 : hi + 2]
    elif lo > 0:
        out[:, lo : hi + 1] = x[:, lo - 1 : lo]
    else:
        raise ValueError("整条光谱只有一块，没有可用的邻点")
    return out


def unit_features(enc: nn.Module, x: NDArray, owner: NDArray | None, device: torch.device) -> NDArray:
    """各评估单元的特征（批量算法）：全部光谱按固定批大小一次编码完，不按情节分批。

    owner 为 None 时一行即一个单元，直接取编码器输出（公开数据集）；否则同一单元各行的特征取均值后再
    L2 归一化（苹果一果多面位，算式同 protonet.apple_feature）。与原评估路径只差在分批。
    """
    enc.eval()
    feat: NDArray = encode_batch(enc, np.ascontiguousarray(x, dtype=np.float32), device, batch_size=2048)
    if owner is None:
        return feat
    n_unit = int(owner.max()) + 1
    acc = np.zeros((n_unit, feat.shape[1]), dtype=np.float32)
    np.add.at(acc, owner, feat)
    mean = acc / np.bincount(owner, minlength=n_unit).astype(np.float32)[:, None]
    out: NDArray = mean / (np.linalg.norm(mean, axis=1, keepdims=True) + 1e-8)
    return out


def episode_f1(feat: NDArray, y: NDArray, classes: list[int], ep: Episode) -> float:
    """批量算法下一个情节的宏 F1：各类支持单元的特征取均值、L2 归一化作原型，查询单元判给内积最大的类。

    判别规则同 44 号 kshot_eval（src_alpha=0）与 crossdataset_lib.eval_kshot，特征取自 unit_features。
    """
    sup, q = ep
    protos: list[NDArray] = []
    for s in sup:
        p = feat[s].mean(0)
        protos.append(p / (float(np.linalg.norm(p)) + 1e-8))
    pred = np.asarray(classes)[(feat[q] @ np.stack(protos).T).argmax(1)]
    return float(f1_score(y[q], pred, average="macro", labels=classes, zero_division=0))


def apple_cell(year_data: YearData, year: int, wl: NDArray) -> tuple[Cell, list[str], dict[tuple[str, Any], int]]:
    """留出年的评估对象：单元 = 苹果，行 = 面位谱；产地按名称排序，类别即产地的序号（同 44 号 kshot_eval）。

    另返回排序后的产地名与 (产地, 苹果编号) → 单元下标的索引。
    """
    regions = sorted(year_data[year])
    rows: list[NDArray] = []
    owner: list[int] = []
    labels: list[int] = []
    index: dict[tuple[str, Any], int] = {}
    for ri, reg in enumerate(regions):
        for aid, faces in year_data[year][reg].items():
            index[(reg, aid)] = len(labels)
            owner += [len(labels)] * len(faces)
            labels.append(ri)
            rows.append(faces)
    y = np.asarray(labels, dtype=np.int64)
    cell = Cell(np.vstack(rows), np.asarray(owner, dtype=np.int64), y, list(range(len(regions))), wl)
    return cell, regions, index


def apple_episodes(
    tgt_data: dict[str, dict[Any, NDArray]],
    regions: list[str],
    index: dict[tuple[str, Any], int],
    k: int,
    reps: int,
    seed: int,
) -> list[Episode]:
    """与 64 号评估（44 号 kshot_eval）逐情节相同的抽样，以单元下标给出，供批量算法用。

    第 i 个情节 default_rng(seed + i)：各产地按排序后的次序、从苹果编号里无放回抽 k 个支持果，其余苹果
    作查询；编号出现在任一产地支持集里的苹果不进查询（同 44 号）。调用前须保证各产地至少 k+2 个苹果。
    """
    eps: list[Episode] = []
    for i in range(reps):
        rng = np.random.default_rng(seed + i)
        sup: list[NDArray] = []
        taken: set[Any] = set()
        for reg in regions:
            ids = list(rng.choice(list(tgt_data[reg]), size=k, replace=False))
            sup.append(np.asarray([index[(reg, a)] for a in ids], dtype=np.int64))
            taken.update(ids)
        rest = [index[(reg, a)] for reg in regions for a in tgt_data[reg] if a not in taken]
        eps.append((sup, np.asarray(rest, dtype=np.int64)))
    return eps


def public_wavelengths(dsname: str, data_root: str) -> NDArray | None:
    """芒果、猕猴桃的波长轴（nm）：只读表头，取列的办法同 crossdataset_lib 的两个加载器。

    其余数据集的加载器不给波长，返回 None（遮蔽表里只记波段序号）。
    """
    if dsname == "mango_season":
        path = f"{data_root}/041_mango_anderson_crossseason/NAnderson2020MendeleyMangoNIRData.csv"
        wl = [float(c) for c in pd.read_csv(path, nrows=0).columns if str(c).isdigit()]
    elif dsname == "kiwifruit_season":
        path = f"{data_root}/012_kiwifruit_nir_drymatter/kiwifruit_dat.csv"
        cols = [str(c) for c in pd.read_csv(path, nrows=0).columns]
        wl = [float(c[1:]) for c in cols if c.startswith("X") and int(c[1:]) <= 1065]
    else:
        return None
    out: NDArray = np.asarray(wl, dtype=np.float64)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 二、苹果一侧的编码器：只读载入正式权重，或按正式配方重训
# ═══════════════════════════════════════════════════════════════════════════════
def load_saved(
    run_name: str,
    want: dict[str, Any],
    enc_name: str,
    n_bands: int,
    feat_dim: int,
    device: torch.device,
) -> tuple[nn.Module | None, str, dict[str, str]]:
    """只读核对并载入已落盘的编码器，返回 (编码器, 未载入的原因, 模型卡)。

    核对项同 64 号 obtain_encoder：模型卡逐项与本次设定一致、训练已走完、权重 SHA-256 与卡上记录相符。
    任何一项不符即返回 (None, 原因, {})。本函数不训练、不写任何文件。
    """
    card: dict[str, str] | None = m64.read_card(run_name)
    ckpt = os.path.join(eu.CHECKPOINT_DIR, run_name, "best.pt")
    if card is None or not os.path.exists(ckpt):
        return None, "无模型卡或权重", {}
    legacy: dict[str, Any] = m64.LEGACY_CARD_PROTOCOL
    for key, val in want.items():
        got = card.get(key, str(legacy[key]) if key in legacy else None)
        if got != str(val):
            return None, f"{key}：卡上 {got}，本次 {val}", {}
    if "stopped_episode" not in card:
        return None, "训练未走完", {}
    if eu.compute_sha256(ckpt) != card.get("checkpoint_sha256"):
        return None, "权重 SHA-256 与模型卡不符", {}
    encoder = enc_lib.build_encoder(enc_name, n_bands, feat_dim).to(device)
    encoder.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
    encoder.eval()
    return encoder, "", card


def fit_linear_apple(enc_name: str, train_data: AppleData, device: torch.device) -> tuple[nn.Module, int, int]:
    """线性族在训练池全部光谱上即时拟合，返回 (编码器, 成分数, 光谱条数)。

    拼池的次序与标签同 64 号 fit_linear：域名排序、域内苹果编号排序，2025 年每果每面位一条；
    pls 以各光谱所属苹果的产地作类别，pca 不看标签。不落盘。
    """
    xs: list[NDArray] = []
    ys: list[str] = []
    for elem in sorted(train_data):
        for aid in sorted(train_data[elem]):
            faces = train_data[elem][aid]["X"]
            xs.append(faces)
            ys += [str(train_data[elem][aid]["region"])] * len(faces)
    x = np.vstack(xs)
    y = np.asarray(ys) if enc_name == "pls" else None
    enc, n_comp = enc_lib.fit_linear_encoder(enc_name, x, y, LIN_DIM)
    return enc.to(device).eval(), n_comp, len(x)


def obtain_apple(
    arm: str,
    design: str,
    year: int,
    spec: tuple[AppleData, AppleData, str],
    n_bands: int,
    a64: argparse.Namespace,
    device: torch.device,
    logger: eu.Logger,
) -> tuple[nn.Module, dict[str, Any]]:
    """取得一个苹果编码器，返回 (编码器, 来源与训练摘要)；spec = (训练池, 验证池, 任务说明)。

    线性族即时拟合。深度族依次只读核对 64 号正式权重与本脚本已落盘的 87_* 权重，核对通过即载入；
    都不可用时按正式配方重训（64 号 train_encoder，播种方式不变），权重写 07checkpoints/87_*。
    """
    train_data, val_data, desc = spec
    n_train = sum(len(v) for v in train_data.values())
    info: dict[str, Any] = {"训练池规模": n_train}
    if a64.encoder in enc_lib.LINEAR_ENCODERS:
        enc_lin, n_comp, n_spec = fit_linear_apple(a64.encoder, train_data, device)
        logger.log(f"  [即时拟合] {a64.encoder} {arm} 留出 {year}：{n_spec} 条光谱，{n_comp} 个成分")
        info.update({"来源": SRC_FIT, "成分数": n_comp})
        return enc_lin, info
    # 模型卡字段与 64 号 main 写进 extra_card 的相同；want 是 64 号 obtain_encoder 的核对项
    card: dict[str, Any] = {
        "design": design,
        "data_variant": VARIANT,
        "fold_year": year,
        **{key: getattr(a64, key) for key in m64.LEGACY_CARD_PROTOCOL},
        "arm": arm,
    }
    want: dict[str, Any] = {
        **card,
        "stage": a64.stage,
        "training_hardware": device.type,
        "episodes_budget": a64.episodes,
        "seed": a64.seed,
        "train_elems": ",".join(sorted(train_data)),
        "val_elems": ",".join(sorted(val_data)),
        "n_train_apples": n_train,
    }
    enc_seg = "" if a64.encoder == "cnn" else f"_{a64.encoder}"
    stem = f"{design}_{VARIANT}{enc_seg}_{arm}_seed{a64.seed}_fold{year}"
    own = f"87_{stem}" if a64.stage == "formal" else f"87_{stem}_pilot"
    for run_name, source in ((f"64_{stem}", SRC_64), (own, SRC_87_SAVED)):
        enc, why, got = load_saved(run_name, want, a64.encoder, n_bands, a64.feat_dim, device)
        if enc is None:
            logger.log(f"  [未载入] {run_name}：{why}")
            continue
        logger.log(f"  [载入] {run_name}：验证Acc={got['val_acc']} @ep{got['best_episode']}")
        info.update(
            {
                "来源": source,
                "run_name": run_name,
                "best_val_acc": float(got["val_acc"]),
                "best_episode": int(got["best_episode"]),
                "stopped_episode": int(got["stopped_episode"]),
                "early_stopped": got["early_stopped"] == "True",
                "sha256": got["checkpoint_sha256"],
            }
        )
        return enc, info
    if not own.startswith("87_"):
        raise RuntimeError(f"{own}：本脚本只写以 87_ 打头的权重目录")
    trained, summ = m64.train_encoder(own, train_data, val_data, desc, n_bands, a64, device, logger, card)
    info.update({"来源": SRC_87_TRAIN, "run_name": own})
    info.update({key: summ[key] for key in ("best_val_acc", "best_episode", "stopped_episode", "early_stopped")})
    info["sha256"] = summ["sha256"]
    return trained, info


# ═══════════════════════════════════════════════════════════════════════════════
# 三、单个留出域的评估：K 曲线（原评估路径）与波段遮蔽（批量算法）
# ═══════════════════════════════════════════════════════════════════════════════
def skipped_rows(base: dict[str, Any], k_values: list[int], why: str) -> Rows:
    """整个留出域不评时的 K 曲线行：各 K、两臂都记为跳过。"""
    rows: Rows = []
    for k in k_values:
        for arm in ARMS:
            rows.append(
                {
                    **base,
                    "臂": arm,
                    "K": k,
                    "宏F1": float("nan"),
                    "F1情节SD": float("nan"),
                    "情节数": 0,
                    "是否跳过": True,
                    "跳过原因": why,
                    "编码器来源": "",
                }
            )
    return rows


def apple_curve_f1(
    cache: Any, year_data: YearData, year: int, src_years: list[int], k: int, rep_seeds: list[int]
) -> list[float]:
    """苹果的原评估路径：逐个重采样种子调 44 号 kshot_eval，返回各情节的宏 F1。

    参数取法同 64 号 evaluate：特征经 FeatureCache（一果一批）、目标年取全部苹果、源年份取设计登记的
    src_years、α=0（源年份原型不参与判别）。K=K_REF 时就是 64 号正式结果的算法。
    """
    return [float(m64.bmc.kshot_eval(cache, year_data, year, src_years, k, s, 0.0)["f1"]) for s in rep_seeds]


def public_curve_f1(enc: nn.Module, cell: Cell, eps: list[Episode], device: torch.device) -> list[float]:
    """公开数据集的原评估路径：逐情节按 crossdataset_lib.eval_kshot 的分批编码，返回各情节的宏 F1。

    用 74 号 proto_f1：各类支持样本按抽到的次序各成一批、取均值再 L2 归一化作原型，查询样本一批，
    与 eval_kshot 逐句相同。情节由调用方取自 74 号 kshot_episodes（与 eval_kshot 同一随机数流）。
    """
    enc.eval()
    return [float(m74.proto_f1(enc, cell.x, cell.y, cell.classes, ep, device)) for ep in eps]


def curve_rows(
    base: dict[str, Any],
    curve: dict[str, dict[int, list[float]]],
    skip_why: dict[int, str],
    k_values: list[int],
    sources: dict[str, str],
    logger: eu.Logger,
) -> Rows:
    """K 曲线行：各臂各 K 取原评估路径逐情节宏 F1 的均值与 SD；没有情节的 K 记为跳过。"""
    rows: Rows = []
    notes: list[str] = []
    for k in k_values:
        vals: list[str] = []
        for arm in ARMS:
            f1s = curve[arm].get(k, [])
            mean = float(np.mean(f1s)) if f1s else float("nan")
            rows.append(
                {
                    **base,
                    "臂": arm,
                    "K": k,
                    "宏F1": mean,
                    "F1情节SD": float(np.std(f1s)) if f1s else float("nan"),
                    "情节数": len(f1s),
                    "是否跳过": not f1s,
                    "跳过原因": "" if f1s else skip_why.get(k, "没有有效情节"),
                    "编码器来源": sources[arm],
                }
            )
            vals.append(f"{arm}={mean:.4f}" if f1s else f"{arm}=跳过")
        notes.append(f"K={k} " + " ".join(vals))
    logger.log(f"    {base['数据集']} 留出 {base['留出域']}：" + " | ".join(notes))
    return rows


def occlusion_rows(
    cell: Cell,
    encs: dict[str, nn.Module],
    eps_ref: list[Episode],
    occl_block: int,
    device: torch.device,
    base: dict[str, Any],
    logger: eu.Logger,
) -> tuple[Rows, dict[str, float]]:
    """批量算法：留出域全部样本一次编码，在 K=K_REF 那一批情节上算宏 F1；返回 (遮蔽行, 各臂未遮蔽值)。

    未遮蔽值与各块遮蔽后的值出自同一算法、同一批情节，下降量是这一算法内的差。occl_block > 0 时逐块
    遮蔽：作用在 cell.x（编码器输入）上，支持与查询光谱一同遮蔽，编码器不重训。没有情节或这一格不做
    遮蔽（occl_block < 1）时两项皆空，批量算法不运行。
    """
    rows_o: Rows = []
    clean: dict[str, float] = {}
    if not eps_ref or occl_block < 1:
        return rows_o, clean
    for arm in ARMS:
        feat = unit_features(encs[arm], cell.x, cell.owner, device)
        clean[arm] = float(np.mean([episode_f1(feat, cell.y, cell.classes, ep) for ep in eps_ref]))
    n_bands = int(cell.x.shape[1])
    pos: NDArray = cell.wl if cell.wl is not None else np.arange(n_bands, dtype=np.float64)
    if len(pos) != n_bands or bool(np.any(np.diff(pos) <= 0)):
        raise RuntimeError(f"{base['数据集']}：波长轴与光谱列数不符，或不是严格递增")
    blocks = make_blocks(n_bands, occl_block)
    for b, (lo, hi) in enumerate(blocks):
        x_occ = occlude(cell.x, lo, hi, pos)
        nm = [float("nan"), float("nan"), float("nan")]
        if cell.wl is not None:
            nm = [round(float(v), 2) for v in (cell.wl[lo], cell.wl[hi], cell.wl[lo : hi + 1].mean())]
        for arm in ARMS:
            feat_occ = unit_features(encs[arm], x_occ, cell.owner, device)
            occ = float(np.mean([episode_f1(feat_occ, cell.y, cell.classes, ep) for ep in eps_ref]))
            rows_o.append(
                {
                    **base,
                    "臂": arm,
                    "块号": b,
                    "块起波段": lo,
                    "块止波段": hi,
                    "块起nm": nm[0],
                    "块止nm": nm[1],
                    "块中心nm": nm[2],
                    "K": K_REF,
                    "情节数": len(eps_ref),
                    "未遮蔽宏F1": clean[arm],
                    "遮蔽后宏F1": occ,
                    "下降量": clean[arm] - occ,
                }
            )
    for arm in ARMS:
        drops = [float(r["下降量"]) for r in rows_o if r["臂"] == arm]
        top = int(np.argmax(drops))
        logger.log(f"      遮蔽 {len(blocks)} 块 {arm}：最大下降 {drops[top]:.4f}（块 {top}）")
    return rows_o, clean


def exact_check(where: str, mine: float, ref: float) -> None:
    """公开数据集：逐情节原评估路径的 K=K_REF 均值须等于 crossdataset_lib.eval_kshot 的返回值。

    两者是同一批情节、同一种分批，相差超过 EXACT_TOL 说明情节或分批没有对上，中止。
    """
    gap = abs(mine - ref)
    if np.isnan(gap) or gap > EXACT_TOL:
        raise RuntimeError(
            f"{where}：K={K_REF} 宏 F1 逐情节路径 {mine:.12f}、eval_kshot {ref:.12f}，"
            f"相差超过 {EXACT_TOL}，情节或分批与原评估函数没有对上"
        )


def path_gap(where: str, batch: float, orig: float, n_dec: int, logger: eu.Logger) -> dict[str, Any]:
    """K=K_REF 未遮蔽宏 F1 在两条路径上的差，返回「编码器来源与核对」表的几列。

    batch 出自批量算法（波段遮蔽用），orig 出自原评估路径（K 曲线用）；两者同一批情节、同一编码器，只差
    在分批，分批不同引起的浮点末位差偶尔翻动个别判别。差一律只记录，并按 n_dec（这一格的判别总数）折成
    判别数。超过 PATH_TOL 时记 WARN：实测出现在嵌入几乎没有区分度的格（宏 F1 接近随机水平，大量查询处在
    平局边缘，末位差就能翻动成批判别），这样的格里遮蔽的下降量本身也不稳，读表时对照这一列。
    """
    gap = abs(batch - orig)
    if np.isnan(gap):
        raise RuntimeError(f"{where}：K={K_REF} 宏 F1 批量算法 {batch}、原评估函数 {orig}，有一侧不是数")
    flips = round(gap * n_dec)
    if gap > PATH_TOL:
        logger.log(
            f"      [WARN][两路径] {where}：批量算法 {batch:.6f}、原评估函数 {orig:.6f}，"
            f"差 {gap:.2e} 超过 {PATH_TOL}，约合 {flips}/{n_dec} 个判别"
        )
    elif gap > 0:
        logger.log(f"      [两路径] {where}：批量算法与原评估函数差 {gap:.2e}，约合 {flips}/{n_dec} 个判别")
    return {
        "K5宏F1_批量算法": batch,
        "K5宏F1_原评估函数": orig,
        "两路径之差": gap,
        "K5判别总数": n_dec,
        "两路径之差折合判别数": flips,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 四、两种范围的主循环与逐种子表
# ═══════════════════════════════════════════════════════════════════════════════
def load_apple(logger: eu.Logger) -> tuple[AppleData, YearData, NDArray]:
    """冻结数据 → (apple_data, year_data, 波长轴)，读法同 64 号 main 与 70 号。"""
    df = m64.load_data(VARIANT, logger)
    wl_cols = eu.get_spectrum_columns(df)
    wl: NDArray = eu.get_spectrum_wavelengths(df, wl_cols).astype(np.float64)
    return build_apple_data(df, wl_cols), build_year_region_data(df, wl_cols), wl


def run_apple(
    enc_name: str,
    apple: tuple[AppleData, YearData, NDArray],
    plan: Plan,
    device: torch.device,
    logger: eu.Logger,
) -> tuple[Rows, Rows, Rows]:
    """苹果一侧：各设计各折取得 R、S_src 编码器；K 曲线逐个重采样种子调 44 号 kshot_eval，波段遮蔽用批量算法。"""
    apple_data, year_data, wl = apple
    n_bands = len(wl)
    a64 = argparse.Namespace(
        seed=plan.seed,
        stage=plan.stage,
        encoder=enc_name,
        episodes=plan.apple_episodes,
        **{key: val for key, val in m64.LEGACY_CARD_PROTOCOL.items() if key != "encoder"},
    )
    occl_block = plan.block if "apple" in plan.occl else 0
    rep_seeds = [plan.seed + i for i in range(plan.reps)]  # 重采样种子同 64 号 evaluate
    rows_k: Rows = []
    rows_o: Rows = []
    rows_e: Rows = []
    for design in plan.designs:
        logger.log(f"\n【apple_{design}】{enc_name}")
        for tgt, cfg in sorted(m64.DESIGNS[design].items()):
            year = int(tgt)
            src: AppleData = {e: apple_data[e] for e in cfg["src"]}
            union: AppleData = {e: apple_data[e] for e in cfg["src"] + cfg["tgt"]}
            # 训练池与验证池同 64 号 main 的 specs：R =（源域, 源域），S_src =（源域 ∪ 留出年, 源域）
            specs: dict[str, tuple[AppleData, AppleData, str]] = {
                "R": (src, src, f"折限定（{cfg['src']}，留出 {year}）"),
                "S_src": (union, src, f"目标暴露·源域选检查点（留出 {year}）"),
            }
            encs: dict[str, nn.Module] = {}
            info: dict[str, dict[str, Any]] = {}
            for arm in ARMS:  # 次序同 64 号 main：先 R 后 S_src；重训时 train_encoder 开头各自重新播种
                encs[arm], info[arm] = obtain_apple(arm, design, year, specs[arm], n_bands, a64, device, logger)
            cell, regions, index = apple_cell(year_data, year, wl)
            n_min = min(int((cell.y == c).sum()) for c in cell.classes)
            skip_why = {k: f"留出年有产地不足 {k + 2} 个苹果" for k in plan.k_values if n_min < k + 2}
            # K 曲线：原评估路径。每个编码器一个特征缓存，各 K、各重采样种子共用；参数同 64 号 evaluate
            caches = {arm: m64.bmc.FeatureCache(encs[arm], device, True) for arm in ARMS}
            curve: dict[str, dict[int, list[float]]] = {}
            for arm in ARMS:
                curve[arm] = {
                    k: apple_curve_f1(caches[arm], year_data, year, cfg["src_years"], k, rep_seeds)
                    for k in plan.k_values
                    if k not in skip_why
                }
            base: dict[str, Any] = {
                "范围": "apple",
                "数据集": f"apple_{design}",
                "留出域": f"Y{year}",
                "编码器族": enc_name,
                "种子": plan.seed,
                "类别数": len(cell.classes),
                "最小类样本数": n_min,
            }
            sources = {arm: str(info[arm]["来源"]) for arm in ARMS}
            rows_k += curve_rows(base, curve, skip_why, plan.k_values, sources, logger)
            # 波段遮蔽：批量算法；情节按 kshot_eval 的抽法重现 K=K_REF 那一批
            eps_ref: list[Episode] = []
            if K_REF not in skip_why:
                eps_ref = apple_episodes(year_data[year], regions, index, K_REF, plan.reps, plan.seed)
            part_o, clean = occlusion_rows(cell, encs, eps_ref, occl_block, device, base, logger)
            rows_o += part_o
            n_dec = sum(len(q) for _, q in eps_ref)
            for arm in ARMS:
                row_e: dict[str, Any] = {**base, "臂": arm, **info[arm]}
                if arm in clean:
                    where = f"apple_{design} 留出 {year} {enc_name} {arm}"
                    orig = float(np.mean(curve[arm][K_REF]))
                    row_e.update(path_gap(where, clean[arm], orig, n_dec, logger))
                elif curve[arm].get(K_REF):
                    row_e["K5宏F1_原评估函数"] = float(np.mean(curve[arm][K_REF]))
                rows_e.append(row_e)
    return rows_k, rows_o, rows_e


def load_public(plan: Plan, data_root: str, logger: eu.Logger) -> dict[str, PublicSet]:
    """按 74 号的登记加载公开数据集（同一加载器、同一种子），并取各数据集的波长轴。"""
    out: dict[str, PublicSet] = {}
    for dsname in plan.datasets:
        meta = m74.DATASETS[dsname]
        loader: Callable[[str, int], cdl.LoaderResult] = meta["loader"]
        doms, is_reg = loader(data_root, plan.seed)
        wl = public_wavelengths(dsname, data_root)
        out[dsname] = (doms, is_reg, wl)
        axis = "无（遮蔽表只记波段序号）" if wl is None else f"{wl[0]:.0f}–{wl[-1]:.0f} nm，{len(wl)} 个波段"
        logger.log(f"【{dsname}】{meta['drift']}；域 {[f'{d}:{len(v[1])}' for d, v in doms.items()]}；波长轴 {axis}")
    return out


def run_public(
    enc_name: str,
    public: dict[str, PublicSet],
    plan: Plan,
    device: torch.device,
    logger: eu.Logger,
) -> tuple[Rows, Rows, Rows]:
    """公开数据集一侧：每个留出域按 74 号的配方先训 R、再训 S_src（不落盘），评估 K 曲线与波段遮蔽。

    类别集合由 74 号 fold_arrays 在 K=K_REF 下选定。K 曲线逐情节按 crossdataset_lib.eval_kshot 的分批
    编码，K=K_REF 的均值须与 eval_kshot 的返回值相同；波段遮蔽用批量算法。
    """
    a74 = argparse.Namespace(lin_dim=LIN_DIM, episodes=plan.public_episodes)
    source = SRC_FIT if enc_name in enc_lib.LINEAR_ENCODERS else SRC_74_TRAIN
    rows_k: Rows = []
    rows_o: Rows = []
    rows_e: Rows = []
    for dsname in plan.datasets:
        doms, is_reg, wl = public[dsname]
        occl_block = plan.block if dsname in plan.occl else 0
        logger.log(f"\n【{dsname}】{enc_name}")
        for held in doms:
            base: dict[str, Any] = {
                "范围": "public",
                "数据集": dsname,
                "留出域": held,
                "编码器族": enc_name,
                "种子": plan.seed,
            }
            got = m74.fold_arrays(doms, is_reg, held, K_REF)
            if got is None:
                why = f"K={K_REF} 时合法类不足 3 个"
                logger.log(f"    {dsname} 留出 {held}：{why}，此留出域不评")
                rows_k += skipped_rows(base, plan.k_values, why)
                continue
            fa: cdl.FoldArrays = got[0]
            if wl is not None and len(wl) != fa.in_dim:
                raise RuntimeError(f"{dsname}：波长轴 {len(wl)} 个点，光谱 {fa.in_dim} 个波段")
            pools: dict[str, tuple[NDArray, NDArray]] = {"R": (fa.xs, fa.ys), "S_src": (fa.xall, fa.yall)}
            encs: dict[str, nn.Module] = {}
            for arm in ARMS:  # 次序同 74 号 run_fold：先 R 后 S_src；meta_train 开头各自重新播种
                encs[arm] = m74.obtain(enc_name, pools[arm][0], pools[arm][1], fa.in_dim, a74, device, plan.seed)
            classes = list(fa.classes)
            cell = Cell(fa.xh, None, fa.yh, classes, wl)
            n_min = min(int((fa.yh == c).sum()) for c in classes)
            eps_by_k: dict[int, list[Episode]] = {}
            skip_why: dict[int, str] = {}
            for k in plan.k_values:
                if n_min < k + 2:
                    skip_why[k] = f"留出域有类不足 {k + 2} 个样本"
                else:
                    eps_by_k[k] = m74.kshot_episodes(fa.yh, classes, k, plan.reps, plan.seed)
            # K 曲线：原评估路径，逐情节按 eval_kshot 的分批编码
            curve: dict[str, dict[int, list[float]]] = {}
            for arm in ARMS:
                curve[arm] = {k: public_curve_f1(encs[arm], cell, eps, device) for k, eps in eps_by_k.items()}
            base.update({"类别数": len(classes), "最小类样本数": n_min})
            sources = dict.fromkeys(ARMS, source)
            rows_k += curve_rows(base, curve, skip_why, plan.k_values, sources, logger)
            # 波段遮蔽：批量算法，用 K=K_REF 那一批情节
            eps_ref = eps_by_k.get(K_REF, [])
            part_o, clean = occlusion_rows(cell, encs, eps_ref, occl_block, device, base, logger)
            rows_o += part_o
            n_dec = sum(len(q) for _, q in eps_ref)
            for arm in ARMS:
                row_e: dict[str, Any] = {**base, "臂": arm, "来源": source, "训练池规模": len(pools[arm][1])}
                if curve[arm].get(K_REF):
                    where = f"{dsname} 留出 {held} {enc_name} {arm}"
                    ref = float(cdl.eval_kshot(encs[arm], fa.xh, fa.yh, classes, device, K_REF, plan.reps, plan.seed))
                    exact_check(where, float(np.mean(curve[arm][K_REF])), ref)
                    if arm in clean:
                        row_e.update(path_gap(where, clean[arm], ref, n_dec, logger))
                    else:
                        row_e["K5宏F1_原评估函数"] = ref
                rows_e.append(row_e)
    return rows_k, rows_o, rows_e


def notes_table(extra: dict[str, Any]) -> pd.DataFrame:
    """工作簿「说明」表：评估与遮蔽的口径，后接本次运行或汇总的设定。"""
    rows = [{"项": key, "内容": text} for key, text in NOTES]
    rows += [{"项": key, "内容": str(val)} for key, val in extra.items()]
    return pd.DataFrame(rows)


def seed_file(scope: str, encoder: str, seed: int, stage: str, tag_seg: str) -> str:
    """逐种子工作簿的路径：按范围与编码器族分文件；非缺省设置带标签段，pilot 带 __pilot。"""
    suffix = "" if stage == "formal" else "__pilot"
    return os.path.join(SEED_DIR, f"{STEM}__{scope}__{encoder}__seed{seed}{tag_seg}{suffix}.xlsx")


def save_seed_tables(
    path: str,
    tables: tuple[Rows, Rows, Rows],
    compute: dict[str, Any],
    notes: pd.DataFrame,
    logger: eu.Logger,
) -> None:
    """写一个（范围, 编码器族, 种子）的逐种子工作簿；遮蔽表没有内容时也保留表头。"""
    rows_k, rows_o, rows_e = tables
    os.makedirs(SEED_DIR, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows_k, columns=K_COLS).to_excel(writer, sheet_name=SHEET_K, index=False)
        pd.DataFrame(rows_o, columns=O_COLS).to_excel(writer, sheet_name=SHEET_O, index=False)
        pd.DataFrame(rows_e).to_excel(writer, sheet_name=SHEET_E, index=False)
        notes.to_excel(writer, sheet_name="说明", index=False)
        pd.DataFrame([compute]).to_excel(writer, sheet_name="compute", index=False)
    n_skip = sum(1 for r in rows_k if r["是否跳过"]) // len(ARMS)
    logger.log(f"结果表 → {path}")
    logger.log(f"  K 曲线 {len(rows_k)} 行（跳过 {n_skip} 格），遮蔽 {len(rows_o)} 行")


# ═══════════════════════════════════════════════════════════════════════════════
# 五、汇总与自检（--aggregate）
# ═══════════════════════════════════════════════════════════════════════════════
def _t_interval(values: list[float]) -> tuple[float, float]:
    """跨种子 95% t 区间：均值 ± t(0.975, n−1)·SD/√n；少于 2 个种子时为 nan。"""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float("nan"), float("nan")
    half = float(stats.t.ppf(0.975, len(arr) - 1)) * float(arr.std(ddof=1)) / float(np.sqrt(len(arr)))
    return float(arr.mean()) - half, float(arr.mean()) + half


def _spearman(a: NDArray, b: NDArray) -> float:
    """两条下降量曲线的 Spearman 秩相关；不足 3 块或有一条曲线为常数时记 nan。"""
    if len(a) < 3 or float(np.ptp(a)) == 0.0 or float(np.ptp(b)) == 0.0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def load_seed_tables(
    stage: str, tag_seg: str, seeds: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """读入逐种子工作簿，返回 (K 曲线长表, 遮蔽长表, compute, 缺的文件名)。"""
    parts_k: list[pd.DataFrame] = []
    parts_o: list[pd.DataFrame] = []
    parts_c: list[pd.DataFrame] = []
    missing: list[str] = []
    for scope in SCOPES:
        for enc in enc_lib.ENCODERS:
            for seed in seeds:
                path = seed_file(scope, enc, seed, stage, tag_seg)
                if not os.path.exists(path):
                    missing.append(os.path.basename(path))
                    continue
                book = pd.read_excel(path, sheet_name=None)
                parts_k.append(book[SHEET_K])
                occ_part = book.get(SHEET_O)
                if occ_part is not None and len(occ_part):
                    parts_o.append(occ_part)
                if "compute" in book:
                    parts_c.append(book["compute"])
    if not parts_k:
        raise FileNotFoundError(f"没有找到 87 号的逐种子结果（{stage}{tag_seg}）：{SEED_DIR}")
    kdf = pd.concat(parts_k, ignore_index=True)
    odf = pd.concat(parts_o, ignore_index=True) if parts_o else pd.DataFrame(columns=O_COLS)
    cdf = pd.concat(parts_c, ignore_index=True) if parts_c else pd.DataFrame()
    return kdf, odf, cdf, missing


def kcurve_tables(kdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K 曲线汇总，返回 (逐域表, 跨域汇总)。

    逐域表：各（数据集, 留出域, 编码器族, K）在种子内配对作差的 S_src − R（记作 Δ_rep，同 64、74 号）。
    跨域汇总：按口径 × 编码器族 × K，对逐域的种子均值计为正留出域数与均值；被跳过的留出域单独计数。
    """
    order_ds = {name: i for i, name in enumerate([*APPLE_DESIGNS, *m74.DATASETS])}
    order_enc = {name: i for i, name in enumerate(enc_lib.ENCODERS)}
    rows: Rows = []
    keys = ["范围", "数据集", "留出域", "编码器族", "K"]
    for (scope, ds, held, enc, k), sub in kdf.groupby(keys, sort=False):
        ok = sub[~sub["是否跳过"].astype(bool)]
        f1 = {(str(a), int(s)): float(v) for a, s, v in zip(ok["臂"], ok["种子"], ok["宏F1"], strict=True)}
        seeds = sorted({int(s) for s in sub["种子"]})
        paired = [s for s in seeds if ("R", s) in f1 and ("S_src", s) in f1]
        row: dict[str, Any] = {
            "口径": THREE_SCOPE if ds == "apple_three_year" else MAIN_SCOPE,
            "范围": scope,
            "数据集": ds,
            "留出域": held,
            "编码器族": enc,
            "K": int(k),
            "种子数": len(paired),
            "跳过种子数": len(seeds) - len(paired),
            "_ds": order_ds.get(str(ds), len(order_ds)),
            "_enc": order_enc.get(str(enc), len(order_enc)),
        }
        if paired:
            deltas = [f1[("S_src", s)] - f1[("R", s)] for s in paired]
            lo, hi = _t_interval(deltas)
            row.update(
                {
                    "R": float(np.mean([f1[("R", s)] for s in paired])),
                    "S_src": float(np.mean([f1[("S_src", s)] for s in paired])),
                    "Δ_rep": float(np.mean(deltas)),
                    "Δ_rep_种子SD": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan"),
                    "Δ_rep_种子t下限": lo,
                    "Δ_rep_种子t上限": hi,
                    "Δ_rep_为正种子数": int(sum(d > 0 for d in deltas)),
                }
            )
        rows.append(row)
    fold_df = pd.DataFrame(rows, columns=[*FOLD_ID, *FOLD_VAL, "_ds", "_enc"])
    fold_df = fold_df.sort_values(["_ds", "留出域", "_enc", "K"]).drop(columns=["_ds", "_enc"])
    fold_df = fold_df.reset_index(drop=True)

    sum_rows: Rows = []
    for (label, enc, k), sub in fold_df.groupby(["口径", "编码器族", "K"], sort=False):
        vals = sub["Δ_rep"].dropna()
        n_pos = int((vals > 0).sum())
        sum_rows.append(
            {
                "口径": label,
                "编码器族": enc,
                "K": int(k),
                "应有留出域数": EXPECTED_HELD.get(str(label)),
                "留出域数": len(sub),
                "跳过留出域数": len(sub) - len(vals),
                "可评留出域数": len(vals),
                "为正留出域数": n_pos,
                "为正/可评": f"{n_pos}/{len(vals)}",
                "Δ_rep均值": float(vals.mean()) if len(vals) else float("nan"),
                "Δ_rep最小": float(vals.min()) if len(vals) else float("nan"),
                "Δ_rep最大": float(vals.max()) if len(vals) else float("nan"),
                "R均值": float(sub["R"].mean()),
                "S_src均值": float(sub["S_src"].mean()),
            }
        )
    return fold_df, pd.DataFrame(sum_rows)


def load_reference(stage: str, seeds: list[int]) -> tuple[dict[RefKey, float], list[str]]:
    """64、74 号逐种子表里 K=K_REF、全部评估样本上的 R 与 S_src，返回 (参照值, 缺的文件名)。"""
    suffix = "" if stage == "formal" else "__pilot"
    ref: dict[RefKey, float] = {}
    missing: list[str] = []
    for enc in enc_lib.ENCODERS:
        enc_seg = "" if enc == "cnn" else f"__{enc}"
        for seed in seeds:
            for dsname, design in APPLE_DESIGNS.items():
                name = REF64_FILE.format(design=design, variant=VARIANT, enc_seg=enc_seg, seed=seed, suffix=suffix)
                path = os.path.join(SEED_DIR, name)
                if not os.path.exists(path):
                    missing.append(name)
                    continue
                d64 = pd.read_excel(path, sheet_name=REF_SHEET)
                d64 = d64[d64[REF64_COL["arm"]].isin(ARMS)]
                if REF64_COL["part"] in d64.columns:
                    d64 = d64[d64[REF64_COL["part"]] == REF_PART_ALL]
                if REF64_COL["alpha"] in d64.columns:
                    d64 = d64[d64[REF64_COL["alpha"]] == 0.0]
                if REF64_COL["k"] in d64.columns:
                    d64 = d64[d64[REF64_COL["k"]] == K_REF]
                for _, r in d64.iterrows():
                    held = f"Y{int(r[REF64_COL['fold']])}"
                    ref[(dsname, held, enc, seed, str(r[REF64_COL["arm"]]))] = float(r[REF64_COL["f1"]])
            name = REF74_FILE.format(encoder=enc, seed=seed, suffix=suffix)
            path = os.path.join(SEED_DIR, name)
            if not os.path.exists(path):
                missing.append(name)
                continue
            d74 = pd.read_excel(path, sheet_name=REF_SHEET)
            d74 = d74[d74[REF74_COL["arm"]].isin(ARMS) & (d74[REF74_COL["part"]] == REF_PART_ALL)]
            for _, r in d74.iterrows():
                dsname, held = str(r[REF74_COL["dataset"]]), str(r[REF74_COL["held"]])
                ref[(dsname, held, enc, seed, str(r[REF74_COL["arm"]]))] = float(r[REF74_COL["f1"]])
    return ref, missing


def _same_digits(dsname: str, ours: float, theirs: float) -> bool:
    """与正式表的存数逐位相同：74 号写表时不取舍，直接比；64 号的表存到 REF64_DECIMALS 位小数，取舍后再比。"""
    if dsname in APPLE_DESIGNS:
        return round(ours, REF64_DECIMALS) == theirs
    return ours == theirs


def selfcheck_tables(kdf: pd.DataFrame, ref: dict[RefKey, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K=K_REF 的 R、S_src 与 S_src − R 对正式逐种子表，返回 (超差格, 按范围 × 数据集 × 编码器族的汇总)。

    超差 = 绝对差大于 SELFCHECK_TOL；正式表里找不到 R 或 S_src 的格计入「缺参照格数」，不参与比对。
    汇总另计逐位相同的格：一格（数据集, 留出域, 编码器族, 种子）的 R 与 S_src 都与正式表的存数相同才算。
    """
    k5 = kdf[(kdf["K"] == K_REF) & ~kdf["是否跳过"].astype(bool)]
    flagged: Rows = []
    stat: dict[tuple[str, str, str], dict[str, Any]] = {}
    keys = ["范围", "数据集", "留出域", "编码器族", "种子"]
    for (scope, ds, held, enc, seed), sub in k5.groupby(keys, sort=False):
        mine = {str(a): float(v) for a, v in zip(sub["臂"], sub["宏F1"], strict=True)}
        group = (str(scope), str(ds), str(enc))
        if group not in stat:
            stat[group] = {
                "比对值个数": 0,
                "超差个数": 0,
                "最大绝对差": 0.0,
                "缺参照格数": 0,
                "比对格数": 0,
                "逐位相同格数": 0,
            }
        r_mine, s_mine = mine.get("R"), mine.get("S_src")
        r_ref = ref.get((str(ds), str(held), str(enc), int(seed), "R"))
        s_ref = ref.get((str(ds), str(held), str(enc), int(seed), "S_src"))
        if r_mine is None or s_mine is None or r_ref is None or s_ref is None:
            stat[group]["缺参照格数"] += 1
            continue
        stat[group]["比对格数"] += 1
        if _same_digits(str(ds), r_mine, r_ref) and _same_digits(str(ds), s_mine, s_ref):
            stat[group]["逐位相同格数"] += 1
        source = "；".join(dict.fromkeys(str(v) for v in sub["编码器来源"]))
        pairs = (("R", r_mine, r_ref), ("S_src", s_mine, s_ref), ("S_src−R", s_mine - r_mine, s_ref - r_ref))
        for name, ours, theirs in pairs:
            diff = ours - theirs
            stat[group]["比对值个数"] += 1
            stat[group]["最大绝对差"] = max(float(stat[group]["最大绝对差"]), abs(diff))
            if abs(diff) > SELFCHECK_TOL:
                stat[group]["超差个数"] += 1
                flagged.append(
                    {
                        "范围": scope,
                        "数据集": ds,
                        "留出域": held,
                        "编码器族": enc,
                        "种子": int(seed),
                        "量": name,
                        "本脚本": ours,
                        "正式表": theirs,
                        "差": diff,
                        "编码器来源": source,
                    }
                )
    check_rows: Rows = []
    for g, val in stat.items():
        ratio = f"{val['逐位相同格数']}/{val['比对格数']}"
        check_rows.append({"范围": g[0], "数据集": g[1], "编码器族": g[2], **val, "逐位相同格数/总格数": ratio})
    return pd.DataFrame(flagged, columns=FLAG_COLS), pd.DataFrame(check_rows)


def occlusion_tables(odf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """遮蔽汇总，返回 (逐域汇总, S_src 比 R 多依赖最多的三块, 逐块的种子均值曲线)。

    逐（数据集, 留出域, 编码器族）：先对种子求各块下降量的均值，得 R 与 S_src 两条曲线；秩相关在这两条
    均值曲线上算，另给逐种子秩相关的均值；「多依赖」按 S_src 下降量 − R 下降量从大到小取前三块。
    苹果另给块中心波长落在 PIGMENT_BAND 内各块的正下降量占全部正下降量之和的比例。
    """
    summ: Rows = []
    top: Rows = []
    curve: Rows = []
    for (scope, ds, held, enc), sub in odf.groupby(OCC_ID, sort=False):
        drops: dict[str, dict[int, NDArray]] = {"R": {}, "S_src": {}}
        clean: dict[str, dict[int, float]] = {"R": {}, "S_src": {}}
        for (arm, seed), part in sub.groupby(["臂", "种子"], sort=True):
            ordered = part.sort_values("块号")
            drops[str(arm)][int(seed)] = ordered["下降量"].to_numpy(dtype=np.float64)
            clean[str(arm)][int(seed)] = float(ordered["未遮蔽宏F1"].iloc[0])
        seeds = sorted(set(drops["R"]) & set(drops["S_src"]))
        if not seeds:
            continue
        meta = sub.drop_duplicates("块号").sort_values("块号").reset_index(drop=True)
        n_block = len(meta)
        if any(len(drops[arm][s]) != n_block for arm in ARMS for s in seeds):
            raise ValueError(f"{ds}/{held}/{enc}：各种子、各臂的块数不一致")
        mean_r = np.mean([drops["R"][s] for s in seeds], axis=0)
        mean_s = np.mean([drops["S_src"][s] for s in seeds], axis=0)
        sd_r = np.full(n_block, np.nan)
        sd_s = np.full(n_block, np.nan)
        if len(seeds) > 1:
            sd_r = np.std([drops["R"][s] for s in seeds], axis=0, ddof=1)
            sd_s = np.std([drops["S_src"][s] for s in seeds], axis=0, ddof=1)
        diff = mean_s - mean_r
        by_seed = [_spearman(drops["R"][s], drops["S_src"][s]) for s in seeds]
        valid = [v for v in by_seed if not np.isnan(v)]
        ident = {"范围": scope, "数据集": ds, "留出域": held, "编码器族": enc}
        row: dict[str, Any] = {
            **ident,
            "块数": n_block,
            "种子数": len(seeds),
            "未遮蔽R": float(np.mean([clean["R"][s] for s in seeds])),
            "未遮蔽S_src": float(np.mean([clean["S_src"][s] for s in seeds])),
            "秩相关_种子均值曲线": _spearman(mean_r, mean_s),
            "秩相关_逐种子均值": float(np.mean(valid)) if valid else float("nan"),
        }
        if ds in APPLE_DESIGNS:
            centre = meta["块中心nm"].to_numpy(dtype=np.float64)
            inside = (centre >= PIGMENT_BAND[0]) & (centre < PIGMENT_BAND[1])
            for arm, mean in (("R", mean_r), ("S_src", mean_s)):
                gain = np.clip(mean, 0.0, None)
                share = float(gain[inside].sum() / gain.sum()) if float(gain.sum()) > 0 else float("nan")
                row[f"606–699nm正下降占比_{arm}"] = share
        summ.append(row)
        for rank, b in enumerate(np.argsort(-diff, kind="stable")[:3], start=1):
            i = int(b)
            block = {col: meta[col].iloc[i] for col in BLOCK_COLS}
            numbers = {"R下降量": float(mean_r[i]), "S_src下降量": float(mean_s[i]), "下降量之差": float(diff[i])}
            top.append({**ident, "名次": rank, **block, **numbers})
        for i in range(n_block):
            curve.append(
                {
                    **ident,
                    **{col: meta[col].iloc[i] for col in BLOCK_COLS},
                    "种子数": len(seeds),
                    "R下降量": float(mean_r[i]),
                    "R下降量_种子SD": float(sd_r[i]),
                    "S_src下降量": float(mean_s[i]),
                    "S_src下降量_种子SD": float(sd_s[i]),
                    "下降量之差": float(diff[i]),
                }
            )
    return pd.DataFrame(summ, columns=OCC_SUM_COLS), pd.DataFrame(top), pd.DataFrame(curve)


def aggregate(stage: str, tag_seg: str, logger: eu.Logger) -> None:
    """汇总逐种子工作簿并做自检，写 04outputs/87_kcurve_occlusion[__<标签>][__pilot].xlsx。"""
    seeds = [int(s) for s in eu.RECOMMENDED_SEEDS] if stage == "formal" else [int(eu.PILOT_SEED)]
    kdf, odf, cdf, missing = load_seed_tables(stage, tag_seg, seeds)
    if missing:
        head = "; ".join(missing[:20]) + (" …" if len(missing) > 20 else "")
        logger.log(f"[WARN] 缺 {len(missing)} 个逐种子工作簿，汇总只作进度查看：{head}")
    fold_df, sum_df = kcurve_tables(kdf)
    ref, ref_missing = load_reference(stage, seeds)
    flag_df, check_df = selfcheck_tables(kdf, ref)
    occ_sum, occ_top, occ_curve = occlusion_tables(odf)

    logger.log("K 曲线跨域汇总\n" + sum_df.to_string(index=False))
    n_cmp = int(check_df["比对值个数"].sum()) if len(check_df) else 0
    n_noref = int(check_df["缺参照格数"].sum()) if len(check_df) else 0
    n_cell = int(check_df["比对格数"].sum()) if len(check_df) else 0
    n_same = int(check_df["逐位相同格数"].sum()) if len(check_df) else 0
    logger.log(
        f"[自检] 与 64、74 号逐种子表（{stage}）比对 {n_cmp} 个值："
        f"相差超过 {SELFCHECK_TOL} 的 {len(flag_df)} 个，缺参照 {n_noref} 格；逐位相同 {n_same}/{n_cell} 格"
    )
    if n_cmp == 0:
        logger.log(f"[自检][WARN] 没有可比对的格（缺 {len(ref_missing)} 个参照表），自检未执行")
    if n_same < n_cell:
        logger.log(f"[自检][WARN] {n_cell - n_same} 格与参照表不逐位相同，各组的计数与最大绝对差见「自检汇总」表")
    if len(flag_df):
        logger.log("[自检][WARN] 超差格（已写入汇总工作簿）\n" + flag_df.to_string(index=False))
    if len(occ_sum):
        logger.log("遮蔽逐域汇总\n" + occ_sum.to_string(index=False))

    suffix = "" if stage == "formal" else "__pilot"
    extra: dict[str, Any] = {
        "汇总的阶段与种子": f"{stage}；{seeds}",
        "Δ_rep": "S_src − R，在种子内配对作差；种子 t 区间为跨种子 95% t 区间",
        "口径": f"{MAIN_SCOPE}与{THREE_SCOPE}分开汇总；两种苹果设计共用 2018、2019 两个留出年，不同时计入",
        "自检": f"K={K_REF} 的 R、S_src 与 S_src − R 对 64、74 号逐种子表，绝对差大于 {SELFCHECK_TOL} 记为超差",
        "逐位相同": f"R 与 S_src 都与正式表的存数相同才计入；64 号的表存 {REF64_DECIMALS} 位小数，取舍后再比",
        "遮蔽的秩相关": "R 与 S_src 两条下降量曲线（先对种子求均值）的 Spearman 秩相关",
        "606–699 nm 占比": "苹果：块中心波长落在 606–699 nm 的各块正下降量之和 ÷ 全部块正下降量之和",
    }
    if stage != "formal":
        caveat = "pilot 的训练与评估情节数取正式的 1/5，64、74 号的 pilot 表未必是同一预算，超差格只供查看"
        extra["pilot 的自检"] = caveat
        logger.log(f"[自检] {caveat}")
    notes = notes_table(extra)
    out = os.path.join(eu.OUTPUT_DIR, f"{STEM}{tag_seg}{suffix}.xlsx")
    lacking = pd.DataFrame({"缺表": [*missing, *ref_missing]})
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        fold_df.to_excel(writer, sheet_name=eu.table_sheet_name(0, "K曲线逐域"), index=False)
        sum_df.to_excel(writer, sheet_name=eu.table_sheet_name(1, "K曲线跨域汇总"), index=False)
        flag_df.to_excel(writer, sheet_name=eu.table_sheet_name(2, "自检超差格"), index=False)
        check_df.to_excel(writer, sheet_name=eu.table_sheet_name(3, "自检汇总"), index=False)
        occ_sum.to_excel(writer, sheet_name=eu.table_sheet_name(4, "遮蔽逐域汇总"), index=False)
        occ_top.to_excel(writer, sheet_name=eu.table_sheet_name(5, "遮蔽多依赖前三块"), index=False)
        occ_curve.to_excel(writer, sheet_name=eu.table_sheet_name(6, "遮蔽逐块均值"), index=False)
        lacking.to_excel(writer, sheet_name=eu.table_sheet_name(7, "缺表"), index=False)
        notes.to_excel(writer, sheet_name="说明", index=False)
        cdf.to_excel(writer, sheet_name="compute", index=False)
    logger.log(f"汇总表 → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 六、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def _split(text: str, allowed: Sequence[str], what: str) -> list[str]:
    """把逗号分隔的参数拆成去重的列表，并核对取值。"""
    items = list(dict.fromkeys(v for v in text.split(",") if v))
    unknown = sorted(set(items) - set(allowed))
    if unknown or not items:
        raise SystemExit(f"{what} 取值无效：{unknown or text!r}；可选 {list(allowed)}")
    return items


def tag_segment(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """输出文件名里的标签段。会改变结果内容的参数取非缺省值时必须给 --tag，以免覆盖缺省设置的结果。"""
    if args.tag and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.tag):
        raise SystemExit("--tag 只能由字母、数字、下划线与连字符组成")
    changed = [key for key in GUARDED if getattr(args, key) != parser.get_default(key)]
    if changed and not args.tag and not args.aggregate:
        raise SystemExit(f"{changed} 取了非缺省值：须加 --tag <标签>，以免覆盖缺省设置的结果文件")
    return f"__{args.tag}" if args.tag else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="87_kcurve_occlusion：六个编码器族的 K 曲线与波段遮蔽归因")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", default=None, choices=["pilot", "formal"], help="缺省：运行 pilot，汇总 formal")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED, help="一次一个种子")
    parser.add_argument("--scope", default="all", choices=[*SCOPES, "all"])
    parser.add_argument("--designs", default=",".join(APPLE_DESIGNS.values()), help="苹果的设计，逗号分隔")
    parser.add_argument("--datasets", default=",".join(m74.DATASETS), help="公开数据集，逗号分隔")
    parser.add_argument("--encoders", default=",".join(enc_lib.ENCODERS), help="编码器族，逗号分隔")
    parser.add_argument("--data_root", default=DATA_ROOT, help="公开数据集根目录")
    parser.add_argument("--k_list", default=",".join(str(k) for k in K_LIST), help="K 曲线的各档，须含 5")
    parser.add_argument("--block", type=int, default=BLOCK, help="遮蔽分块的波段数")
    parser.add_argument("--occl_datasets", default=",".join(OCCL_DEFAULT), help="做遮蔽的数据集；apple 指苹果")
    parser.add_argument("--skip_occlusion", action="store_true", help="只做 K 曲线")
    parser.add_argument("--tag", default="", help="输出文件名里的标签段；非缺省设置时必须给")
    parser.add_argument("--aggregate", action="store_true", help="汇总逐种子工作簿并做自检")
    args = parser.parse_args()
    stage: str = args.stage or ("formal" if args.aggregate else "pilot")
    args.stage = stage
    tag_seg = tag_segment(args, parser)

    if args.aggregate:
        agg_logger = eu.get_logger(STEM)
        aggregate(stage, tag_seg, agg_logger)
        agg_logger.close()
        return

    scopes = list(SCOPES) if args.scope == "all" else [str(args.scope)]
    enc_list = _split(args.encoders, enc_lib.ENCODERS, "--encoders")
    k_values = sorted({int(v) for v in args.k_list.split(",") if v})
    if K_REF not in k_values or k_values[0] < 1:
        raise SystemExit(f"--k_list 须为正整数且含 {K_REF}（类别集合、遮蔽与自检都以 K={K_REF} 为准）")
    occl: set[str] = set()
    if not args.skip_occlusion:
        occl = set(_split(args.occl_datasets, ["apple", *m74.DATASETS], "--occl_datasets"))
    div = 1 if stage == "formal" else PILOT_DIV
    plan = Plan(
        seed=int(args.seed),
        stage=stage,
        reps=REPS // div,
        apple_episodes=APPLE_EPISODES // div,
        public_episodes=PUBLIC_EPISODES // div,
        k_values=k_values,
        block=int(args.block),
        occl=occl,
        designs=_split(args.designs, sorted(m64.DESIGNS), "--designs"),
        datasets=_split(args.datasets, list(m74.DATASETS), "--datasets"),
    )

    device = eu.get_device(args.device)
    logger = eu.get_logger(STEM)
    eu.set_random_seed(plan.seed, logger)
    eu.enforce_publication_grade_device(device, stage=stage, logger=logger)
    settings: dict[str, Any] = {
        "阶段": stage,
        "种子": plan.seed,
        "K": plan.k_values,
        "每格情节数": plan.reps,
        "训练情节（苹果 / 公开数据集）": f"{plan.apple_episodes} / {plan.public_episodes}",
        "波段遮蔽": f"每块 {plan.block} 个波段；数据集 {sorted(plan.occl)}" if plan.occl else "不做",
    }
    eu.log_experiment_header(
        logger,
        {
            "任务": "六个编码器族的 K 曲线与波段遮蔽归因",
            "设备": str(device),
            "范围": scopes,
            "编码器族": enc_list,
            "臂": ARMS,
            "苹果设计": plan.designs,
            "公开数据集": plan.datasets,
            **settings,
            "遮蔽位置": "编码器的输入空间（数据加载器给出、送进编码器之前的表示）",
            "标签": args.tag or "无",
        },
    )
    run = eu.init_wandb(__file__, plan.seed, eu.PROJECT_CODE, stage, vars(args))
    notes = notes_table(settings)

    compute: dict[str, Any]
    if "apple" in scopes:
        apple = load_apple(logger)
        for enc_name in enc_list:
            with eu.track_compute(f"K 曲线与波段遮蔽 apple × {enc_name}", logger, run) as budget:
                tables = run_apple(enc_name, apple, plan, device, logger)
            compute = {"seed": plan.seed, "stage": stage, "scope": "apple", "encoder": enc_name, **budget}
            out_path = seed_file("apple", enc_name, plan.seed, stage, tag_seg)
            save_seed_tables(out_path, tables, compute, notes, logger)
    if "public" in scopes:
        public = load_public(plan, str(args.data_root), logger)
        for enc_name in enc_list:
            with eu.track_compute(f"K 曲线与波段遮蔽 public × {enc_name}", logger, run) as budget:
                tables = run_public(enc_name, public, plan, device, logger)
            compute = {"seed": plan.seed, "stage": stage, "scope": "public", "encoder": enc_name, **budget}
            out_path = seed_file("public", enc_name, plan.seed, stage, tag_seg)
            save_seed_tables(out_path, tables, compute, notes, logger)

    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
