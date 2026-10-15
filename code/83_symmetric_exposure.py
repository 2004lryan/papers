"""83_symmetric_exposure.py: 基线的对等暴露——池化 SVM 与 PLS-DA 接受参数级暴露后的抬高量与方法排名。

为什么:
    现有的方法排名翻转（泄漏评估与无泄漏评估下排第一的方法不同）比的是「编码器的权重见过留出域」
    对「SVM、PLS-DA 只在超参数上见过留出域」，两边的暴露级别不对等。本脚本让传统学习器也在参数上
    见到留出域带标签的光谱，量它们抬高多少，再看对等暴露下排第一的方法与无泄漏时是否相同。
    训练集里不放查询样本（放进去 RBF-SVM 会直接记住它们）：照 S_half 臂的做法，留出域的 A 半
    进训练集，只在 B 半上评估。

留出域:
    苹果两年设计 2 折、三年设计 3 折（dataset 记作 apple_two_year / apple_three_year，held 记作
    Y<年份>，同 75 号）；公开数据集 16 个留出域（74 号登记的五个数据集）。

A/B 划分（直接调用原函数、传同样的参数，与 64、74 号同一种子下 S_half 臂的划分逐样本相同）:
    苹果  m64.split_target_half(apple_data, 目标年各域, 种子 + 目标年)，苹果编号级、按产地分层
    公开  m74.half_split(fa, dom_of_row, held, 种子)，fa 取自 m74.fold_arrays(doms, is_reg, held, 5)
    折的类别集合（fold_arrays）与划分一律按 K=5 定（64、74 号正式结果的 K），与 --k 无关；
    B 半里参评的类再按 --k 筛（见下「类别」），--k 取 5 时与 74 号 B 半三臂的类别相同。

各臂（一格 = 一个留出域 × 一个种子；同一格内全部路线用同一批 B 半情节）:
    池化 SVM、池化 PLS-DA 各两臂
        无暴露          训练集 = 源域带标签光谱（每类至多 200 条，照 65 号）+ B 半里抽的 K 个支持样本
        暴露（参数级）  训练集 = 同一批源域光谱 ∪ A 半全部带标签光谱 + 同样的支持样本
    只用支持样本的参照：最近质心；支持集上拟合的 SVM 与 PLS-DA（74 号 sk_f1，内部是 71 号 fit_predict）
    情节    B 半各类抽 K 个作支持样本，B 半其余样本全部作查询样本；每格 50 个（pilot 10 个）
    类别    公开数据集取 B 半里样本数 ≥ K+2 的类，少于 3 类时该留出域不评（与 74 号的 B 半三臂相同），
            源域、A 半、支持与查询都只留这些类；苹果取目标年的产地，源域只留其中的类（照 65 号）
    苹果    支持与查询以果为单位，取该果全部面位的均值谱（与 64 号的评估单位相同；2018、2019 年
            每果一条谱）；源域与 A 半按逐面位光谱进训练集
    学习器  池化 SVM = StandardScaler + SVC，class_weight="balanced"（照 65 号：源域里没有的类只有
            K 个支持样本）；池化 PLS-DA 对类别指示矩阵回归、取得分最大的类，没有类别加权
    迭代上限  池化 SVM 的 SMO 迭代上限缺省 100 万次（--svm_max_iter，-1 为不设限）：71 号的 SVC 在 K 个
            支持样本上拟合、不设上限；池化训练集有几百到几千条，大 C 候选不设限时可能很久不收敛。
            两臂与选参用同一上限；停在上限的拟合次数逐留出域记在留出域信息里，为 0 时与不设限的解相同

超参数（只用源域数据选，池化学习器的两臂共用同一组；候选网格同 71 号，并列取候选序列里靠前者）:
    只用支持样本的参照  公开照 74 号路线面板的源域选参（各源域轮流作伪目标，每域 20 个情节，类别取 B 半参评的类）；
                        苹果照 76 号的源年份选参（每个源年份 20 个伪目标情节）
    池化学习器          各源域轮流作伪目标：训练集 = 其余源域的带标签光谱（每类至多 200 条）+ 伪目标
                        里的 K 个支持样本，在伪目标其余样本上评估；每个伪目标 5 个情节（训练集以池
                        为主，情节之间只差 K 个支持样本）。只有一个源域时（苹果两年设计、药片）没有
                        「其余源域」，把该源域按类对半分成池与伪目标
    pilot 时两类选参情节分别减到 5 个与 2 个。

随机数:
    重现划分的两处照 64、74 号原样（default_rng([种子 + 目标年, 64, 11])、default_rng([种子, 74, 11])）；
    只用支持样本的参照选参沿用 74、76 号的选参情节；其余抽样都是 default_rng([种子, 83, 流编号, …])，
    不消耗全局随机数。

汇总（--aggregate）:
    表a  池化学习器的暴露增量（暴露臂 − 无暴露臂，种子内配对）：逐留出域的种子均值、种子 t 区间、
         为正种子数；各口径的跨域汇总（留出域均值、数据集簇自助区间、为正留出域数）
    表b  逐编码器族、逐留出域的两套排名——无泄漏排名（原型网络 R^h、两个池化学习器的无暴露臂、
         三条只用支持样本的参照）与对等暴露排名（原型网络 S_half、两个池化学习器的暴露臂）——
         各自的第一名、两者是否属于同一种方法、第一名换成了谁；另列把三条参照也放进对等暴露排名
         时的第一名（参照没有可暴露的参数，数值不变）
    表c  第一名相同的留出域数 / 总数，分主口径、补充口径及各自去掉 Indian Pines
    原型网络在同一 B 半上的 R^h（臂 R_B）与 S_half 取自 64 号（苹果）与 74 号（公开）已有的逐种子表，
    是它们各自 50 次重采样的均值，与本脚本的情节不是同一批；64 号存表时舍入到 4 位小数。排名按
    种子均值，只在 K=5 上做。主文用 cnn，其余编码器族一并列出。
    划分核对：本脚本记下的 B 半样本数、B 半类别、S_half 训练池大小与留出域指纹 mmd2，同 64、74 号
    表里 S_half 臂的同名量逐格比对（计数层面；查不出两边数据行序不同）。

输入:
    03data/processed/all_9elements.csv（冻结数据，按数据卡核对 SHA-256）
    --data_root 下的五个公开数据集（同 74 号）
    汇总时另读 04outputs/formal_seeds/ 里 64、74 号的逐种子表

运行方式:
    cd <项目根>
    # pilot：种子 2004，每格 10 个情节
    python 02code/83_symmetric_exposure.py --stage pilot --data_root <公开数据集根目录> --n_jobs 8
    # formal：每个正式种子一次，每格 50 个情节（纯 CPU，不要求 CUDA）
    python 02code/83_symmetric_exposure.py --stage formal --seed 20060515 \
        --data_root <公开数据集根目录> --n_jobs 21
    # 汇总：缺省读正式种子的表；加 --stage pilot 读预跑的表
    python 02code/83_symmetric_exposure.py --aggregate

输出文件:
    04outputs/formal_seeds/83_symmetric_exposure__seed<S>[__<范围>][__pilot].xlsx
        长表（一行 = 一个留出域 × 一条路线，数值不舍入）、留出域信息、compute；
        只跑部分范围（--scope、--datasets、--k 取非缺省值）时文件名多一个范围段；
        某个留出域出错时其余照常写表，异常文本记在留出域信息里，脚本以非零状态退出
    04outputs/83_symmetric_exposure.xlsx（--stage pilot 时加 __pilot）
        表a 暴露增量、表b 逐留出域排名、表c 汇总计数、表d 逐种子长表、表e 留出域信息、
        表f 划分核对、表g 不全或不一致、compute
    05logs/83_symmetric_exposure_YY-MM-DD_HHMMSS.log

复用的现有代码（按原样调用）:
    64 号  DESIGNS、split_target_half
    74 号  DATASETS、fold_arrays（内部是 crossdataset_lib.prepare_fold）、half_split、SK_KINDS、
           tune_episodes、select、sk_f1、nc_pred
    71 号  候选网格（经 74 号 SK_KINDS）、TUNE_EPISODES、TUNE_OFFSET、score、pick_best；
           fit_predict 经 74 号 sk_f1 调用，池化学习器的 pooled_predict 照它的写法
    76 号  DESIGN_SRC（与 64 号 DESIGNS 的源年份逐项核对）；源年份选参照它的 tune
    65 号  源域每类限量的抽法（fold_assets）与池化 SVM 的 class_weight（_svm_fit_predict）：65 号的
           折表只有三年设计，fold_assets 还会顺带拟合 TCA，这里照它的写法另写
    20 号  snv
    export_utils、protonet  冻结数据读入、build_apple_data、build_year_region_data、日志与算力记录
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import time
import traceback
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import encoders as enc_lib
import export_utils as eu
from protonet import ELEM_INFO, build_apple_data, build_year_region_data

NDArray = npt.NDArray[Any]
Episode = tuple[list[NDArray], NDArray]  # (逐类的支持样本下标, 查询样本下标)，同 74 号
# 池化学习器的一个选参任务：(伪目标序号, 池 x, 池 y, 伪目标 x, 伪目标 y, 类数)，y 是 0..类数-1 的位置
TuneTask = tuple[int, NDArray, NDArray, NDArray, NDArray, int]
HyperParam = tuple[Any, float, str]  # (所选超参数, 选参情节上的宏 F1, 选参方式)
CellData = tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]  # 源域 x、y，A 半 x、y，B 半 x、y

m20 = importlib.import_module("20_unified_singleface_kshot")
m64 = importlib.import_module("64_leakage_controls")
m71 = importlib.import_module("71_source_only_tuning")
m74 = importlib.import_module("74_multi_encoder_leakage")
m76 = importlib.import_module("76_target_tuning_leakage")

OUT_STEM = "83_symmetric_exposure"
SEED_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
SPLIT_K = 5  # 64、74 号正式结果的 K：类别集合与 A/B 划分按它定，与 --k 无关
REPS = {"formal": 50, "pilot": 10}  # 每格的 B 半评估情节数
# 每个伪目标的选参情节数。只用支持样本的参照照 71、74、76 号取 20；池化学习器的训练集以源域池为主，
# 情节之间只差 K 个支持样本，情节间方差小，而每次拟合比只用支持样本的贵得多，取 5
TUNE_EPISODES = {"formal": int(m71.TUNE_EPISODES), "pilot": 5}
POOL_TUNE_EPISODES = {"formal": 5, "pilot": 2}
SRC_PER_CLASS = 200  # 源域每类至多取的光谱数（65 号 --src_per_class 的缺省值）
SVM_MAX_ITER = 1_000_000  # 池化 SVM 的 SMO 迭代上限（--svm_max_iter 的缺省值）；-1 为不设限，同 71 号的 SVC
KINDS: dict[str, tuple[str, tuple[Any, ...], Any]] = m74.SK_KINDS  # 学习器名 → (类型, 候选网格, 固定值)
APPLE_DESIGNS = {"apple_two_year": "two_year", "apple_three_year": "three_year"}  # 数据集名同 75 号
APPLE_VARIANT = "canonical"
PUBLIC_SETS: list[str] = list(m74.DATASETS)
IP_SET = "indianpines_spatial"
SCOPES: dict[str, list[str]] = {
    "主口径（苹果两年设计 + 公开数据集）": ["apple_two_year", *PUBLIC_SETS],
    "主口径去掉 Indian Pines": ["apple_two_year", *[d for d in PUBLIC_SETS if d != IP_SET]],
    "补充口径（苹果三年设计 + 公开数据集）": ["apple_three_year", *PUBLIC_SETS],
    "补充口径去掉 Indian Pines": ["apple_three_year", *[d for d in PUBLIC_SETS if d != IP_SET]],
}

# 随机流：default_rng([种子, 83, 流编号, 数据类别, 留出域的编号…, 序号])
KIND_APPLE, KIND_PUBLIC = 0, 1
STREAM_SRC = 1  # 源域每类限量抽样
STREAM_EVAL = 2  # B 半评估情节
STREAM_TUNE_SPLIT = 3  # 只有一个源域时把它对半分成池与伪目标
STREAM_TUNE_POOL = 4  # 选参用的池每类限量抽样
STREAM_TUNE_EP = 5  # 池化学习器的选参情节
STREAM_BOOT = 9  # 汇总里的数据集簇自助
N_BOOT = 2000
BOOT_SEED = int(eu.RECOMMENDED_SEEDS[0])

# 路线、臂与状态的写法（逐种子表与汇总共用）
STATUS_OK, STATUS_SKIP, STATUS_FAIL = "已评", "未评", "失败"
ARM_FREE, ARM_EXPO, ARM_REF = "无暴露", "暴露", "参照"
TRAIN_FREE, TRAIN_EXPO, TRAIN_REF = "源域 + 支持样本", "源域 ∪ A 半 + 支持样本", "仅支持样本"
REF_NC = "最近质心"
PROTO = "原型网络"
PROTO_FREE, PROTO_EXPO = f"{PROTO}(R^h)", f"{PROTO}(S_half)"

# 64、74 号逐种子表的读法：文件名、工作表次序与列名都取自两个脚本写表的代码（64 号 main 末尾与
# evaluate、train_encoder 的返回值；74 号 run_fold 与 main 末尾），并与已有正式逐种子表的工作表名、
# 表头对过（64 号 cnn 一份，74 号 cnn、pls 各一份）
PROTO_K = 5  # 64 号表a 的 K 列取 5；74 号表a 没有 K 列，按它缺省的 --k 5 读
ARM_RH, ARM_SHALF = "R_B", "S_half"  # 两个脚本里 B 半上的折限定臂（即 R^h）与半数暴露臂
PROTO_ARMS = {ARM_RH: PROTO_FREE, ARM_SHALF: PROTO_EXPO}
F64_NAME = "64_leakage__{design}__{variant}{enc_seg}__seed{seed}{suffix}.xlsx"  # cnn 的 enc_seg 为空
F64_SHEET = 0  # 「表a：K5各臂结果」
C64 = {"arm": "臂", "alpha": "src_alpha", "year": "目标年份", "k": "K", "f1": "F1均值"}
F64_TRAIN_SHEET = 2  # 「表c：编码器训练摘要」
C64T = {"arm": "臂", "fold": "折", "n": "n_train_apples"}
F74_NAME = "74_multi_encoder__{enc}__seed{seed}{suffix}.xlsx"
F74_SHEET = 0  # 「表a：逐臂结果」
C74 = {
    "dataset": "dataset",
    "held": "held",
    "arm": "臂",
    "f1": "F1",
    "n_eval": "n_eval",
    "classes_b": "B半类别",
    "n_train": "n_train",
    "mmd2": "mmd2",
}
SPLIT_REF_ENCODER = "cnn"  # 划分与编码器无关，核对划分时读这一族的表
CHECK_ITEMS = ("n_eval", "B半类别", "S_half训练池", "mmd2")


def route_pool(name: str, arm: str) -> str:
    """池化学习器的路线名，如「池化SVM(无暴露)」。"""
    return f"池化{name}({arm})"


def route_ref(name: str) -> str:
    """只用支持样本的参照的路线名，如「SVM(仅支持)」。"""
    return f"{name}(仅支持)"


ROUTES_REF = (*(route_ref(n) for n in KINDS), REF_NC)
ROUTES_FREE = (PROTO_FREE, *(route_pool(n, ARM_FREE) for n in KINDS), *ROUTES_REF)  # 无泄漏排名
ROUTES_EXPO = (PROTO_EXPO, *(route_pool(n, ARM_EXPO) for n in KINDS))  # 对等暴露排名
FAMILY: dict[str, str] = {  # 路线 → 方法；两套排名的第一名是否相同按方法判
    PROTO_FREE: PROTO,
    PROTO_EXPO: PROTO,
    **{route_pool(n, arm): f"池化{n}" for n in KINDS for arm in (ARM_FREE, ARM_EXPO)},
    **{r: r for r in ROUTES_REF},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 一、通用部件：标签位置、每类限量、情节、学习器
# ═══════════════════════════════════════════════════════════════════════════════
def snv(x: NDArray) -> NDArray:
    """逐光谱 SNV（20 号的同一函数）。"""
    out: NDArray = np.asarray(m20.snv(x))
    return out


def to_positions(labels: NDArray, classes: list[Any]) -> NDArray:
    """把原始标签改写成它在 classes 里的位置；不在 classes 里的记 -1。"""
    out: NDArray = np.full(len(labels), -1, dtype=int)
    for i, c in enumerate(classes):
        out[labels == c] = i
    return out


def cap_per_class(labels: NDArray, cap: int, rng: np.random.Generator) -> NDArray:
    """各类至多留 cap 条，返回保留行的下标：超出的类无放回抽 cap 条、按原行序排列（照 65 号 fold_assets）。"""
    keep: list[NDArray] = []
    for c in np.unique(labels):
        ii = np.flatnonzero(labels == c)
        if len(ii) > cap:
            ii = np.sort(rng.choice(ii, size=cap, replace=False))
        keep.append(ii)
    out: NDArray = np.concatenate(keep) if keep else np.zeros(0, dtype=int)
    return out


def split_half(labels: NDArray, rng: np.random.Generator) -> NDArray:
    """逐类随机对半，返回布尔数组：True 的一半（每类取 n // 2 个）作池，其余作伪目标。"""
    in_pool: NDArray = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        ids = np.flatnonzero(labels == c)
        in_pool[ids[rng.permutation(len(ids))[: len(ids) // 2]]] = True
    return in_pool


def draw_episodes(labels: NDArray, n_cls: int, k: int, n: int, key: list[int]) -> list[Episode]:
    """K-shot 情节：第 j 个用 default_rng([*key, j])，逐类洗牌取前 k 个作支持样本，其余全部作查询样本。

    labels 是 0..n_cls-1 的类别位置；调用方保证每类不少于 k+2 个样本。
    """
    rows = np.arange(len(labels))
    eps: list[Episode] = []
    for j in range(n):
        rng = np.random.default_rng([*key, j])
        sup = [rng.permutation(np.flatnonzero(labels == c))[:k] for c in range(n_cls)]
        eps.append((sup, np.setdiff1d(rows, np.concatenate(sup))))
    return eps


def macro_f1(y_true: NDArray, y_pred: NDArray, n_cls: int) -> float:
    """宏 F1，类别集合固定为 0..n_cls-1（某类没有被预测到时该类记 0）。"""
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_cls)), zero_division=0))


def pooled_predict(
    kind: str, param: Any, x_tr: NDArray, y_tr: NDArray, x_q: NDArray, n_cls: int, max_iter: int
) -> tuple[NDArray, int]:
    """池化学习器在 (x_tr, y_tr) 上拟合后对 x_q 的预测（类别位置），以及这次拟合是否停在迭代上限（1 / 0）。

    写法照 71 号 fit_predict：SVM 先在训练集上拟合 StandardScaler，RBF 核宽是 scale 规则值的倍数；
    PLS-DA 对类别指示矩阵回归、取得分最大的类。SVM 另加 65 号池化 SVM 的 class_weight="balanced"；
    max_iter 是 SMO 的迭代上限（-1 为不设限），停在上限时 SVC 的 fit_status_ 为 1。PLS-DA 不受它影响。
    """
    if kind == "svm":
        scaler = StandardScaler().fit(x_tr)
        xs = scaler.transform(x_tr)
        if param[0] == "linear":
            svm = SVC(kernel="linear", C=param[1], class_weight="balanced", max_iter=max_iter)
        else:
            _, c, g = param
            gamma: Any = "scale"
            if g != 1.0:
                var = float(xs.var())
                gamma = g / (xs.shape[1] * var) if var > 0 else g / xs.shape[1]
            svm = SVC(kernel="rbf", C=c, gamma=gamma, class_weight="balanced", max_iter=max_iter)
        svm.fit(xs, y_tr)
        pred: NDArray = np.asarray(svm.predict(scaler.transform(x_q)), dtype=int)
        return pred, int(svm.fit_status_)
    n_comp = max(1, min(len(x_tr) - 1, x_tr.shape[1], int(param)))
    y_oh = np.zeros((len(x_tr), n_cls))
    y_oh[np.arange(len(x_tr)), y_tr] = 1.0
    pls = PLSRegression(n_components=n_comp, max_iter=500)
    pls.fit(x_tr, y_oh)
    out: NDArray = np.asarray(pls.predict(x_q)).argmax(axis=1)
    return out, 0


def make_task(
    idx: int, x_pool: NDArray, l_pool: NDArray, x_t: NDArray, l_t: NDArray, k: int, rng: np.random.Generator
) -> TuneTask | None:
    """把一个伪目标整理成池化学习器的选参任务。

    只留伪目标里样本数 ≥ k+2 的类（不足 2 类时返回 None，同 74 号 tune_episodes），标签改写成这些类里的
    位置；池只留这些类并按类限量。l_pool、l_t 是原始标签。
    """
    classes = [c for c in np.unique(l_t).tolist() if int((l_t == c).sum()) >= k + 2]
    if len(classes) < 2:
        return None
    y_t, y_pool = to_positions(l_t, classes), to_positions(l_pool, classes)
    x_pool, y_pool = x_pool[y_pool >= 0], y_pool[y_pool >= 0]
    kept = cap_per_class(y_pool, SRC_PER_CLASS, rng)
    return idx, x_pool[kept], y_pool[kept], x_t[y_t >= 0], y_t[y_t >= 0], len(classes)


def select_pooled(
    kind: str, grid: tuple[Any, ...], tasks: list[TuneTask], eps: list[list[Episode]], max_iter: int
) -> tuple[Any, float, int]:
    """池化学习器的源域选参：各候选在全部伪目标情节上的宏 F1 均值，取最高者（并列取候选序列里靠前者）。

    每个情节的训练集 = 该伪目标的池 + 伪目标里的 K 个支持样本，在伪目标其余样本上评估。
    返回 (所选候选, 它的选参宏 F1, 停在迭代上限的拟合次数)。
    """
    vals: dict[Any, list[float]] = {p: [] for p in grid}
    n_cut = 0
    for (_, x_pool, y_pool, x_t, y_t, n_cls), task_eps in zip(tasks, eps, strict=True):
        for sup, q in task_eps:
            si = np.concatenate(sup)
            x_tr, y_tr = np.concatenate([x_pool, x_t[si]]), np.concatenate([y_pool, y_t[si]])
            for p in grid:
                pred, cut = pooled_predict(kind, p, x_tr, y_tr, x_t[q], n_cls, max_iter)
                vals[p].append(macro_f1(y_t[q], pred, n_cls))
                n_cut += cut
    tune = {p: float(np.mean(v)) for p, v in vals.items()}
    best = m71.pick_best(grid, tune)
    return best, tune[best], n_cut


def evaluate_b_half(
    data: CellData,
    n_cls: int,
    eps: list[Episode],
    ref_params: dict[str, Any],
    pool_params: dict[str, Any],
    max_iter: int,
) -> tuple[dict[str, list[float]], int]:
    """同一批 B 半情节上的全部路线：池化学习器的两臂、只用支持样本的 SVM 与 PLS-DA、最近质心。

    返回 (各路线的逐情节宏 F1, 池化学习器停在迭代上限的拟合次数)。
    """
    x_src, y_src, x_a, y_a, x_b, y_b = data
    cls_pos = list(range(n_cls))
    scores: dict[str, list[float]] = {r: [] for r in (*ROUTES_FREE, *ROUTES_EXPO) if FAMILY[r] != PROTO}
    n_cut = 0
    for sup, q in eps:
        si = np.concatenate(sup)
        x_q, y_q = x_b[q], y_b[q]
        x_free, y_free = np.concatenate([x_src, x_b[si]]), np.concatenate([y_src, y_b[si]])
        x_expo, y_expo = np.concatenate([x_src, x_a, x_b[si]]), np.concatenate([y_src, y_a, y_b[si]])
        for name, (kind, _, _) in KINDS.items():
            p = pool_params[name]
            pred_free, cut_free = pooled_predict(kind, p, x_free, y_free, x_q, n_cls, max_iter)
            pred_expo, cut_expo = pooled_predict(kind, p, x_expo, y_expo, x_q, n_cls, max_iter)
            scores[route_pool(name, ARM_FREE)].append(macro_f1(y_q, pred_free, n_cls))
            scores[route_pool(name, ARM_EXPO)].append(macro_f1(y_q, pred_expo, n_cls))
            n_cut += cut_free + cut_expo
            # 只用支持样本：74 号 sk_f1（71 号 fit_predict；拟合失败时退回最近质心）
            f_ref = m74.sk_f1(kind, ref_params[name], x_b, y_b, cls_pos, (sup, q))
            scores[route_ref(name)].append(float(f_ref))
        pred_nc: NDArray = np.asarray(m74.nc_pred(x_b, cls_pos, sup, q))
        scores[REF_NC].append(macro_f1(y_q, pred_nc, n_cls))
    return scores, n_cut


# ═══════════════════════════════════════════════════════════════════════════════
# 二、单个留出域：重现 A/B 划分、源域选参、B 半评估
# ═══════════════════════════════════════════════════════════════════════════════
def skipped(reason: str) -> dict[str, Any]:
    """不评的留出域：没有逐格记录，只在留出域信息里记原因。"""
    return {"rows": [], "info": {"状态": STATUS_SKIP, "说明": reason}}


def base_cols(task: dict[str, Any]) -> dict[str, Any]:
    """逐格记录与留出域信息共用的标识列。"""
    return {
        "种子": int(task["seed"]),
        "阶段": str(task["stage"]),
        "dataset": str(task["dataset"]),
        "held": str(task["held"]),
        "K": int(task["k"]),
    }


def finish_cell(
    task: dict[str, Any],
    key: list[int],
    eval_key: list[int],
    classes: list[str],
    data: CellData,
    ref: dict[str, HyperParam],
    tasks: list[TuneTask],
    tune_mode: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """苹果与公开数据集共用的后半段：池化学习器选参 → 抽 B 半情节 → 评估全部路线 → 整理成逐格记录。"""
    seed, k, max_iter = int(task["seed"]), int(task["k"]), int(task["svm_max_iter"])
    _, y_src, _, y_a, _, y_b = data
    n_cls = len(classes)
    t0 = time.perf_counter()
    n_tune = int(task["n_tune_pool"])
    tune_eps = [draw_episodes(t[4], t[5], k, n_tune, [seed, 83, STREAM_TUNE_EP, *key, t[0]]) for t in tasks]
    pool: dict[str, HyperParam] = {}
    cut_tune = 0
    for name, (kind, grid, fixed) in KINDS.items():
        if tasks:
            best, val, cut = select_pooled(kind, grid, tasks, tune_eps, max_iter)
            pool[name] = (best, val, tune_mode)
            cut_tune += cut
        else:
            pool[name] = (fixed, float("nan"), "源域不足，取固定值")
    t1 = time.perf_counter()
    eps = draw_episodes(y_b, n_cls, k, int(task["reps"]), [seed, 83, STREAM_EVAL, *eval_key])
    ref_params = {n: v[0] for n, v in ref.items()}
    pool_params = {n: v[0] for n, v in pool.items()}
    scores, cut_eval = evaluate_b_half(data, n_cls, eps, ref_params, pool_params, max_iter)
    t2 = time.perf_counter()

    n_sup = k * n_cls
    n_free, n_expo = len(y_src) + n_sup, len(y_src) + len(y_a) + n_sup
    plan: list[tuple[str, str, str, str, HyperParam | None, int]] = []
    for name in KINDS:
        plan += [
            (route_pool(name, ARM_FREE), f"池化{name}", ARM_FREE, TRAIN_FREE, pool[name], n_free),
            (route_pool(name, ARM_EXPO), f"池化{name}", ARM_EXPO, TRAIN_EXPO, pool[name], n_expo),
            (route_ref(name), name, ARM_REF, TRAIN_REF, ref[name], n_sup),
        ]
    plan.append((REF_NC, REF_NC, ARM_REF, TRAIN_REF, None, n_sup))
    rows: list[dict[str, Any]] = []
    for route, learner, arm, train, hp, n_train in plan:
        vals = scores[route]
        rows.append(
            {
                **base_cols(task),
                "路线": route,
                "学习器": learner,
                "臂": arm,
                "训练集": train,
                "F1": float(np.mean(vals)),
                "F1情节SD": float(np.std(vals)),
                "情节数": len(vals),
                "所选超参数": "" if hp is None else str(hp[0]),
                "选参宏F1": float("nan") if hp is None else hp[1],
                "选参方式": "" if hp is None else hp[2],
                "n_classes": n_cls,
                "n_train": n_train,
            }
        )
    per_class = np.bincount(y_src, minlength=n_cls).tolist()
    info = {
        "状态": STATUS_OK,
        "说明": "",
        "n_classes": n_cls,
        "classes": ",".join(classes),
        "n_eval": len(y_b),
        "A半光谱数": len(y_a),
        "源域光谱数": len(y_src),
        "源域每类条数": "; ".join(f"{c}:{n}" for c, n in zip(classes, per_class, strict=True)),
        "池化选参": f"{tune_mode}（{len(tasks)} 个伪目标）" if tasks else "源域不足，取固定值",
        **extra,
        "SVM迭代上限": max_iter,
        "停在上限的拟合数(选参)": cut_tune,
        "停在上限的拟合数(评估)": cut_eval,
        "选参耗时s": round(t1 - t0, 1),
        "评估耗时s": round(t2 - t1, 1),
    }
    return {"rows": rows, "info": info}


def year_arrays(year_data: Any, year: int) -> dict[str, Any]:
    """把一个年份的 year_data 摊平成数组，都做 SNV。

    face_*  逐面位光谱（进训练集用）；apple_*  逐果的全面位均值谱（作支持与查询用）；
    face_owner 是每条面位光谱所属的果在 apple_* 里的行号。
    """
    face_x: list[NDArray] = []
    face_reg: list[str] = []
    face_owner: list[int] = []
    apple_x: list[NDArray] = []
    apple_reg: list[str] = []
    apple_id: list[Any] = []
    for reg, apples in year_data[year].items():
        for aid, faces in apples.items():
            arr = np.asarray(faces)
            face_x.append(arr)
            face_reg += [str(reg)] * len(arr)
            face_owner += [len(apple_id)] * len(arr)
            apple_x.append(arr.mean(axis=0))
            apple_reg.append(str(reg))
            apple_id.append(aid)
    return {
        "face_x": snv(np.concatenate(face_x)),
        "face_reg": np.asarray(face_reg),
        "face_owner": np.asarray(face_owner, dtype=int),
        "apple_x": snv(np.stack(apple_x)),
        "apple_reg": np.asarray(apple_reg),
        "apple_id": apple_id,
    }


def apple_cell(task: dict[str, Any]) -> dict[str, Any]:
    """苹果的一个留出年份：重现 64 号 S_half 臂的 A/B 划分，备好数组与源年份选参后交给 finish_cell。"""
    seed, k, n_tune = int(task["seed"]), int(task["k"]), int(task["n_tune"])
    design, tgt = str(task["design"]), int(task["tgt_year"])
    apple_data, year_data = task["apple_data"], task["year_data"]
    fold = m64.DESIGNS[design][tgt]
    src_years = [int(v) for v in fold["src_years"]]
    key = [KIND_APPLE, int(task["design_idx"]), tgt]

    # ── A/B 划分 ──────────────────────────────────────────
    # 64 号 main 里 S_half 臂的原句：
    #     part_a, half_b[tgt_year] = split_target_half(apple_data, cfg["tgt"], args.seed + tgt_year)
    # 这里的 apple_data 同样是冻结数据经 build_apple_data 得到的，fold 即 DESIGNS[设计][目标年]（原句里的 cfg）。
    # 函数内部：default_rng([种子 + 目标年, 64, 11])，按 cfg["tgt"] 的次序逐域对排好序的苹果编号做一次
    # permutation，前 n // 2 个进 A，其余进 B；两种设计的目标年各域相同，所以同一目标年的划分也相同。
    part_a, part_b = m64.split_target_half(apple_data, fold["tgt"], seed + tgt)
    arr = {v: year_arrays(year_data, v) for v in [tgt, *src_years]}
    tg = arr[tgt]
    regions = sorted(year_data[tgt])
    reg_of_apple = tg["apple_reg"].tolist()
    # B 半 = 目标年里编号落在 part_b[产地] 的苹果（64 号 evaluate 的 restrict 用的同一判据），其余是 A 半
    in_b: NDArray = np.asarray(
        [aid in part_b[reg] for aid, reg in zip(tg["apple_id"], reg_of_apple, strict=True)], dtype=bool
    )
    for e in fold["tgt"]:
        reg_e = str(ELEM_INFO[e][1])
        rest = {
            aid
            for aid, reg, b in zip(tg["apple_id"], reg_of_apple, in_b.tolist(), strict=True)
            if reg == reg_e and not b
        }
        if rest != set(part_a[e]):
            raise RuntimeError(f"留出 {tgt} 的 {e}：B 半的补集与划分给出的 A 半不一致")
    y_apple, y_face = to_positions(tg["apple_reg"], regions), to_positions(tg["face_reg"], regions)
    face_in_a = ~in_b[tg["face_owner"]]
    x_b, y_b = tg["apple_x"][in_b], y_apple[in_b]
    x_a, y_a = tg["face_x"][face_in_a], y_face[face_in_a]
    if int(np.bincount(y_b, minlength=len(regions)).min()) < k + 2:
        return skipped("B 半里有产地不足 K+2 个苹果")

    # ── 源域池：源年份里属于目标年类别集的逐面位光谱，每类限量（照 65 号 fold_assets）────────────
    xs_all = np.concatenate([arr[v]["face_x"] for v in src_years])
    ys_all = np.concatenate([to_positions(arr[v]["face_reg"], regions) for v in src_years])
    xs_all, ys_all = xs_all[ys_all >= 0], ys_all[ys_all >= 0]
    kept = cap_per_class(ys_all, SRC_PER_CLASS, np.random.default_rng([seed, 83, STREAM_SRC, *key]))
    x_src, y_src = xs_all[kept], ys_all[kept]

    # ── 只用支持样本的参照：源年份选参（照 76 号 tune：每个源年份 n_tune 个伪目标情节）──────────
    ref: dict[str, HyperParam] = {}
    for name, (kind, grid, _) in KINDS.items():
        tune = {
            p: float(
                np.mean(
                    [
                        m71.score(year_data, v, k, seed + m71.TUNE_OFFSET + j, kind, p)
                        for v in src_years
                        for j in range(n_tune)
                    ]
                )
            )
            for p in grid
        }
        best = m71.pick_best(grid, tune)
        ref[name] = (best, tune[best], "各源年份作伪目标（照 76 号）")

    # ── 池化学习器的选参任务：留一源年份；只有一个源年份时按苹果编号、按产地对半分成池与伪目标 ──
    tasks: list[TuneTask] = []
    for pi, v in enumerate(src_years):
        ya = arr[v]
        if len(src_years) >= 2:
            others = [o for o in src_years if o != v]
            x_pool = np.concatenate([arr[o]["face_x"] for o in others])
            l_pool = np.concatenate([arr[o]["face_reg"] for o in others])
            x_t, l_t = ya["apple_x"], ya["apple_reg"]
        else:
            in_pool = split_half(ya["apple_reg"], np.random.default_rng([seed, 83, STREAM_TUNE_SPLIT, *key, pi]))
            face_in_pool = in_pool[ya["face_owner"]]
            x_pool, l_pool = ya["face_x"][face_in_pool], ya["face_reg"][face_in_pool]
            x_t, l_t = ya["apple_x"][~in_pool], ya["apple_reg"][~in_pool]
        rng = np.random.default_rng([seed, 83, STREAM_TUNE_POOL, *key, pi])
        made = make_task(pi, x_pool, l_pool, x_t, l_t, k, rng)
        if made is not None:
            tasks.append(made)
    tune_mode = "留一源年份作伪目标" if len(src_years) >= 2 else "单一源年份内对半分成池与伪目标"

    n_src_apples = sum(len(apple_data[e]) for e in fold["src"])
    n_a_apples = sum(len(part) for part in part_a.values())
    extra = {
        "B半类别": ",".join(regions),
        "S_half训练池": n_src_apples + n_a_apples,  # 64 号 S_half 臂的训练苹果数（源域 + A 半）
        "mmd2": float("nan"),
        "A半单位数": n_a_apples,
        "B半单位数": int(in_b.sum()),
    }
    data = (x_src, y_src, x_a, y_a, x_b, y_b)
    return finish_cell(task, key, [KIND_APPLE, tgt], regions, data, ref, tasks, tune_mode, extra)


def public_cell(task: dict[str, Any]) -> dict[str, Any]:
    """公开数据集的一个留出域：重现 74 号 S_half 臂的 A/B 划分，备好数组与源域选参后交给 finish_cell。"""
    seed, k, n_tune = int(task["seed"]), int(task["k"]), int(task["n_tune"])
    doms, is_reg, held = task["doms"], bool(task["is_reg"]), str(task["held"])
    key = [KIND_PUBLIC, int(task["ds_idx"]), list(doms).index(held)]

    # ── A/B 划分 ──────────────────────────────────────────
    # 74 号 run_fold 的原句（args.k 取正式结果的 5，即 SPLIT_K；doms 同样来自 loader(data_root, 种子)）：
    #     got = fold_arrays(doms, is_reg, held, args.k)
    #     fa, dom_of_row = got
    #     xa, ya, b_idx = half_split(fa, dom_of_row, held, seed)
    #     xb, yb = fa.xh[b_idx], fa.yh[b_idx]
    #     classes_b = [c for c in fa.classes if int((yb == c).sum()) >= args.k + 2]
    #     keep = np.isin(yb, classes_b)
    # half_split 内部：default_rng([种子, 74, 11])，按 fa.classes 的次序逐类对 fa.yh 里该类的下标做一次
    # permutation，前 n // 2 个进 A；第三个返回值是 B 半在 fa.xh 里的下标。
    got = m74.fold_arrays(doms, is_reg, held, SPLIT_K)
    if got is None:
        return skipped("源域与留出域共有的合法类不足 3 个（74 号同样不评）")
    fa, dom_of_row = got
    _, _, b_idx = m74.half_split(fa, dom_of_row, held, seed)
    xb, yb = fa.xh[b_idx], fa.yh[b_idx]
    classes_b = [c for c in fa.classes if int((yb == c).sum()) >= k + 2]
    if len(classes_b) < 3:
        return skipped("B 半里样本数 ≥ K+2 的类不足 3 个（74 号同样不评 B 半）")
    keep = np.isin(yb, classes_b)
    x_b, y_b = xb[keep], to_positions(yb[keep], classes_b)
    in_a: NDArray = np.ones(len(fa.yh), dtype=bool)
    in_a[b_idx] = False
    y_h = to_positions(fa.yh, classes_b)
    x_a, y_a = fa.xh[in_a & (y_h >= 0)], y_h[in_a & (y_h >= 0)]

    # ── 源域池：源域里属于 B 半类别集的光谱，每类限量 ───────────────────────────────
    y_s = to_positions(fa.ys, classes_b)
    xs_all, ys_all = fa.xs[y_s >= 0], y_s[y_s >= 0]
    kept = cap_per_class(ys_all, SRC_PER_CLASS, np.random.default_rng([seed, 83, STREAM_SRC, *key]))
    x_src, y_src = xs_all[kept], ys_all[kept]

    # ── 只用支持样本的参照：源域选参（照 74 号 route_panel：各源域轮流作伪目标）────────────────
    dom_src = dom_of_row[dom_of_row != held]
    src_names = sorted(set(dom_src.tolist()))
    src_pools: list[tuple[NDArray, NDArray, list[Any], list[Episode]]] = []
    for tag, d in enumerate(src_names, start=1):
        in_d = dom_src == d
        present, teps = m74.tune_episodes(fa.ys[in_d], classes_b, k, n_tune, seed, tag)
        if teps:
            src_pools.append((fa.xs[in_d], fa.ys[in_d], present, teps))
    ref: dict[str, HyperParam] = {}
    for name, (kind, grid, fixed) in KINDS.items():
        if src_pools:
            best, val = m74.select(kind, grid, src_pools)
            ref[name] = (best, float(val), "各源域轮流作伪目标（照 74 号）")
        else:
            ref[name] = (fixed, float("nan"), "源域不足，取固定值")

    # ── 池化学习器的选参任务：留一源域；只有一个源域时按类对半分成池与伪目标 ─────────────────
    tasks: list[TuneTask] = []
    for pi, d in enumerate(src_names):
        in_d = (dom_src == d) & (y_s >= 0)
        if len(src_names) >= 2:
            in_rest = (dom_src != d) & (y_s >= 0)
            x_pool, l_pool, x_t, l_t = fa.xs[in_rest], fa.ys[in_rest], fa.xs[in_d], fa.ys[in_d]
        else:
            x_d, l_d = fa.xs[in_d], fa.ys[in_d]
            in_pool = split_half(l_d, np.random.default_rng([seed, 83, STREAM_TUNE_SPLIT, *key, pi]))
            x_pool, l_pool, x_t, l_t = x_d[in_pool], l_d[in_pool], x_d[~in_pool], l_d[~in_pool]
        rng = np.random.default_rng([seed, 83, STREAM_TUNE_POOL, *key, pi])
        made = make_task(pi, x_pool, l_pool, x_t, l_t, k, rng)
        if made is not None:
            tasks.append(made)
    tune_mode = "留一源域作伪目标" if len(src_names) >= 2 else "单一源域内对半分成池与伪目标"

    extra = {
        "B半类别": ",".join(str(c) for c in classes_b),  # 与 74 号表a 的写法相同
        "S_half训练池": len(fa.yall) - len(b_idx),  # 74 号 S_half 臂的 n_train
        "mmd2": float(fa.mmd2),
        "A半单位数": int(in_a.sum()),
        "B半单位数": len(b_idx),
    }
    data = (x_src, y_src, x_a, y_a, x_b, y_b)
    return finish_cell(task, key, key, [str(c) for c in classes_b], data, ref, tasks, tune_mode, extra)


def run_cell(task: dict[str, Any]) -> dict[str, Any]:
    """joblib 的工作单元 = 一个留出域。出错时把异常文本带回主进程，其余留出域照常跑完。"""
    t0 = time.perf_counter()
    try:
        out = apple_cell(task) if task["kind"] == "apple" else public_cell(task)
    except Exception:
        out = {"rows": [], "info": {"状态": STATUS_FAIL, "说明": traceback.format_exc()}}
    out["info"] = {**base_cols(task), **out["info"], "耗时s": round(time.perf_counter() - t0, 1)}
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 三、汇总：暴露增量、与原型网络的排名、划分核对
# ═══════════════════════════════════════════════════════════════════════════════
def seed_files(stage: str) -> list[str]:
    """本脚本的逐种子表（含带范围段的分批运行）；pilot 与正式分开。"""
    files = sorted(glob.glob(os.path.join(SEED_DIR, f"{OUT_STEM}__seed*.xlsx")))
    return [f for f in files if ("__pilot" in os.path.basename(f)) == (stage == "pilot")]


def load_mine(stage: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读入本脚本各逐种子表的三张表（逐格结果、留出域信息、compute），各加一列来源文件名。"""
    files = seed_files(stage)
    if not files:
        raise FileNotFoundError(f"没有 83 号的逐种子结果（{stage}）")
    parts: dict[Any, list[pd.DataFrame]] = {0: [], 1: [], "compute": []}
    for f in files:
        for sheet, bucket in parts.items():
            d = pd.read_excel(f, sheet_name=sheet)
            d["文件"] = os.path.basename(f)
            bucket.append(d)
    long = pd.concat(parts[0], ignore_index=True)
    info = pd.concat(parts[1], ignore_index=True)
    comp = pd.concat(parts["compute"], ignore_index=True)
    if long.empty:
        raise SystemExit("逐种子表里没有任何已评的格")
    # 同一格出现在多个表里（比如先单跑了一部分范围、后来又整表跑）：数值相同就只留一条，不同就停下
    keys = ["种子", "dataset", "held", "K", "路线"]
    dup = long[long.duplicated(keys, keep=False)]
    if len(dup):
        gap = dup.groupby(keys)["F1"].agg(lambda v: float(v.max() - v.min()))
        if float(gap.max()) > 1e-9:
            names = "; ".join(sorted(set(dup["文件"])))
            raise SystemExit(f"同一格在多个逐种子表里数值不同（{names}）：用现在的代码把较旧的那一批重跑一遍")
        long = long.drop_duplicates(keys).reset_index(drop=True)
    # 同一留出域在多个表里都有记录时（比如失败后只补跑了那一部分），留下已评的那一条
    info["_ord"] = (info["状态"] != STATUS_OK).astype(int)
    info = info.sort_values("_ord", kind="stable").drop_duplicates(["种子", "dataset", "held", "K"])
    return long, info.drop(columns="_ord").reset_index(drop=True), comp


def read_64(path: str, ds: str, enc: str, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """64 号的一个逐种子表 → (B 半上两臂的记录, S_half 臂的训练苹果数)；后者只在 cnn 的表里取。"""
    d = pd.read_excel(path, sheet_name=F64_SHEET)
    d = d[(d[C64["alpha"]] == 0.0) & (d[C64["k"]] == PROTO_K)]
    rows: list[dict[str, Any]] = []
    for arm, route in PROTO_ARMS.items():
        for _, r in d[d[C64["arm"]] == arm].iterrows():
            rows.append(
                {
                    "dataset": ds,
                    "held": f"Y{int(r[C64['year']])}",
                    "编码器": enc,
                    "种子": seed,
                    "路线": route,
                    "F1": float(r[C64["f1"]]),
                }
            )
    ref: list[dict[str, Any]] = []
    if enc == SPLIT_REF_ENCODER:
        t = pd.read_excel(path, sheet_name=F64_TRAIN_SHEET)
        for _, r in t[t[C64T["arm"]] == ARM_SHALF].iterrows():
            held = f"Y{int(r[C64T['fold']])}"
            ref.append({"dataset": ds, "held": held, "种子": seed, "S_half训练池": int(r[C64T["n"]])})
    return rows, ref


def read_74(path: str, enc: str, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """74 号的一个逐种子表 → (B 半上两臂的记录, S_half 臂记下的划分计数)；后者只在 cnn 的表里取。"""
    d = pd.read_excel(path, sheet_name=F74_SHEET)
    rows: list[dict[str, Any]] = []
    for arm, route in PROTO_ARMS.items():
        for _, r in d[d[C74["arm"]] == arm].iterrows():
            rows.append(
                {
                    "dataset": str(r[C74["dataset"]]),
                    "held": str(r[C74["held"]]),
                    "编码器": enc,
                    "种子": seed,
                    "路线": route,
                    "F1": float(r[C74["f1"]]),
                }
            )
    ref: list[dict[str, Any]] = []
    if enc == SPLIT_REF_ENCODER:
        for _, r in d[d[C74["arm"]] == ARM_SHALF].iterrows():
            ref.append(
                {
                    "dataset": str(r[C74["dataset"]]),
                    "held": str(r[C74["held"]]),
                    "种子": seed,
                    "n_eval": int(r[C74["n_eval"]]),
                    "B半类别": str(r[C74["classes_b"]]),
                    "S_half训练池": int(r[C74["n_train"]]),
                    "mmd2": float(r[C74["mmd2"]]),
                }
            )
    return rows, ref


def load_proto(stage: str, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """各编码器族原型网络在 B 半上的 R^h（臂 R_B）与 S_half 的逐种子宏 F1，以及核对划分用的记录。

    苹果读 64 号、公开数据集读 74 号的逐种子表（文件名、工作表与列名见文件开头的常量）。
    返回 (原型网络长表, 划分参照表, 缺的或读不出的文件)。
    """
    sfx = "" if stage == "formal" else "__pilot"
    jobs: list[tuple[str, str, str, int]] = []  # (文件名, 苹果的数据集名（74 号留空）, 编码器, 种子)
    for enc in enc_lib.ENCODERS:
        seg = "" if enc == "cnn" else f"__{enc}"
        for s in seeds:
            for ds, design in APPLE_DESIGNS.items():
                name = F64_NAME.format(design=design, variant=APPLE_VARIANT, enc_seg=seg, seed=s, suffix=sfx)
                jobs.append((name, ds, enc, s))
            jobs.append((F74_NAME.format(enc=enc, seed=s, suffix=sfx), "", enc, s))
    rows: list[dict[str, Any]] = []
    ref: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, ds, enc, s in jobs:
        path = os.path.join(SEED_DIR, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        try:
            got_rows, got_ref = read_64(path, ds, enc, s) if ds else read_74(path, enc, s)
        except (KeyError, ValueError, IndexError) as exc:
            missing.append(f"{name}（读不出：{exc!r}）")
            continue
        rows += got_rows
        ref += got_ref
    proto = pd.DataFrame(rows, columns=["dataset", "held", "编码器", "种子", "路线", "F1"]).dropna(subset=["F1"])
    return proto, pd.DataFrame(ref, columns=["dataset", "held", "种子", *CHECK_ITEMS]), missing


def same_value(item: str, a: Any, b: Any) -> bool:
    """核对一项：类别串按字符串比，其余按数值比（相对容差 1e-9）。"""
    if item == "B半类别":
        return str(a) == str(b)
    return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=0.0))


def split_check(info: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """划分的计数核对：逐 (数据集, 留出域, 种子) 把本脚本记下的量同 64、74 号 S_half 臂的同名量比对。

    「一致」列为 True / False；64、74 号的表里查不到这个留出域时记 None（无法核对）。只核对 K = SPLIT_K 的格。
    两边调用的是同一个划分函数，这几项都一致说明传进去的数据、类别集合与种子相同。
    """
    cols = ["dataset", "held", "种子"]
    mine = info[(info["状态"] == STATUS_OK) & (info["K"] == SPLIT_K)].reindex(columns=[*cols, *CHECK_ITEMS])
    if ref.empty:  # 一张 64、74 号的表都没读到：全部记作无法核对（空表的键列类型不定，不拿去合并）
        both = mine.copy()
        both["_merge"] = "left_only"
    else:
        both = mine.merge(ref, on=cols, how="outer", suffixes=("", "_ref"), indicator=True)
    rows: list[dict[str, Any]] = []
    for _, r in both.iterrows():
        base = {c: r[c] for c in cols}
        if r["_merge"] == "left_only":
            rows.append({**base, "项目": "留出域", "本脚本": "有", "64/74号": "查不到", "一致": None})
            continue
        if r["_merge"] == "right_only":
            rows.append({**base, "项目": "留出域", "本脚本": "没有已评的格", "64/74号": "有", "一致": False})
            continue
        for item in CHECK_ITEMS:
            a, b = r[item], r[f"{item}_ref"]
            if pd.isna(b):  # 64 号的表里只有训练池大小这一项
                continue
            rows.append({**base, "项目": item, "本脚本": a, "64/74号": b, "一致": same_value(item, a, b)})
    return pd.DataFrame(rows, columns=[*cols, "项目", "本脚本", "64/74号", "一致"])


def t_interval(values: list[float]) -> tuple[float, float]:
    """跨种子 95% t 区间（照 74 号 _t_interval）；不足 2 个值时返回 nan。"""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float("nan"), float("nan")
    half = float(stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr)))
    return float(arr.mean() - half), float(arr.mean() + half)


def increment_table(long: pd.DataFrame) -> pd.DataFrame:
    """表a：池化学习器的暴露增量（暴露臂 − 无暴露臂，种子内配对）——逐留出域，外加各口径的跨域汇总。

    跨域汇总的区间是数据集簇自助（簇 = 数据集，统计量是数据集均值的均值，同 75 号 infer）。
    """
    pooled = long[long["臂"].isin([ARM_FREE, ARM_EXPO])]
    learners = [f"池化{n}" for n in KINDS]
    dom_rows: list[dict[str, Any]] = []
    for (ds, held, k, learner), sub in pooled.groupby(["dataset", "held", "K", "学习器"], sort=False):
        f1 = {(str(a), int(s)): float(v) for a, s, v in zip(sub["臂"], sub["种子"], sub["F1"], strict=True)}
        seeds = sorted({s for _, s in f1 if (ARM_FREE, s) in f1 and (ARM_EXPO, s) in f1})
        if not seeds:
            continue
        free = [f1[(ARM_FREE, s)] for s in seeds]
        expo = [f1[(ARM_EXPO, s)] for s in seeds]
        gain = [e - f for e, f in zip(expo, free, strict=True)]
        lo, hi = t_interval(gain)
        dom_rows.append(
            {
                "层级": "留出域",
                "口径": "",
                "dataset": ds,
                "held": held,
                "K": int(k),
                "学习器": learner,
                "种子数": len(seeds),
                "无暴露": float(np.mean(free)),
                "暴露": float(np.mean(expo)),
                "暴露增量": float(np.mean(gain)),
                "区间下": lo,
                "区间上": hi,
                "区间口径": "种子 t 区间",
                "为正种子数": int(sum(g > 0 for g in gain)),
            }
        )
    dom = pd.DataFrame(dom_rows)
    sum_rows: list[dict[str, Any]] = []
    if dom.empty:
        return dom
    for si, (scope, sets) in enumerate(SCOPES.items()):
        in_scope = dom[dom["dataset"].isin(sets)]
        for (k, learner), sub in in_scope.groupby(["K", "学习器"], sort=False):
            vals = sub["暴露增量"].to_numpy(dtype=np.float64)
            ds_mean = sub.groupby("dataset")["暴露增量"].mean()
            names = ds_mean.index.to_numpy()
            rng = np.random.default_rng([BOOT_SEED, 83, STREAM_BOOT, si, learners.index(str(learner)), int(k)])
            boots = [float(ds_mean.loc[rng.choice(names, size=len(names), replace=True)].mean()) for _ in range(N_BOOT)]
            lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
            sum_rows.append(
                {
                    "层级": "跨域汇总",
                    "口径": scope,
                    "dataset": f"{len(names)} 个数据集",
                    "held": f"{len(vals)} 个留出域",
                    "K": int(k),
                    "学习器": learner,
                    "无暴露": float(sub["无暴露"].mean()),
                    "暴露": float(sub["暴露"].mean()),
                    "暴露增量": float(vals.mean()),
                    "区间下": lo,
                    "区间上": hi,
                    "区间口径": "数据集簇自助 95%（数据集均值的均值）",
                    "数据集均值的均值": float(ds_mean.mean()),
                    "留出域数": len(vals),
                    "为正留出域数": int((vals > 0).sum()),
                    "增量最小": float(vals.min()),
                    "增量最大": float(vals.max()),
                    "数据集数": len(names),
                    "为正数据集数": int((ds_mean > 0).sum()),
                }
            )
    return pd.concat([dom, pd.DataFrame(sum_rows)], ignore_index=True)


def first_and_margin(mean: dict[str, float], routes: tuple[str, ...]) -> tuple[str, float]:
    """均值最高的路线（并列时取 routes 里靠前者）与它领先第二名的量。"""
    order = sorted(routes, key=lambda r: -mean[r])
    return order[0], mean[order[0]] - mean[order[1]]


def ranking_table(long: pd.DataFrame, proto: pd.DataFrame) -> pd.DataFrame:
    """表b：逐编码器族、逐留出域的无泄漏排名与对等暴露排名（各路线取种子均值，只用九条路线都有的种子）。"""
    cols = ["dataset", "held", "种子", "路线", "F1"]
    mine = long[long["K"] == PROTO_K][cols]
    everything = (*ROUTES_FREE, *ROUTES_EXPO)
    rows: list[dict[str, Any]] = []
    for enc in enc_lib.ENCODERS:
        theirs = proto[proto["编码器"] == enc][cols]
        if theirs.empty:
            continue
        both = pd.concat([mine, theirs], ignore_index=True)
        for (ds, held), sub in both.groupby(["dataset", "held"], sort=False):
            f1 = {(str(r), int(s)): float(v) for r, s, v in zip(sub["路线"], sub["种子"], sub["F1"], strict=True)}
            seeds = sorted({s for _, s in f1 if all((r, s) in f1 for r in everything)})
            if not seeds:
                continue
            mean = {r: float(np.mean([f1[(r, s)] for s in seeds])) for r in everything}
            free1, gap_free = first_and_margin(mean, ROUTES_FREE)
            expo1, gap_expo = first_and_margin(mean, ROUTES_EXPO)
            wide1, _ = first_and_margin(mean, (*ROUTES_EXPO, *ROUTES_REF))
            same = FAMILY[free1] == FAMILY[expo1]
            rows.append(
                {
                    "编码器": enc,
                    "dataset": ds,
                    "held": held,
                    "K": PROTO_K,
                    "种子数": len(seeds),
                    **mean,
                    "无泄漏第一": free1,
                    "无泄漏第一领先": gap_free,
                    "对等暴露第一": expo1,
                    "对等暴露第一领先": gap_expo,
                    "第一名相同": same,
                    "第一名换成": "" if same else f"{FAMILY[free1]} → {FAMILY[expo1]}",
                    "对等暴露第一(含参照)": wide1,
                    "第一名相同(含参照)": FAMILY[free1] == FAMILY[wide1],
                }
            )
    return pd.DataFrame(rows)


def tally(routes: Any) -> str:
    """各路线排第一的次数，按次数从多到少写成一个字符串。"""
    return "; ".join(f"{r}×{int(n)}" for r, n in routes.value_counts().items())


def ranking_summary(rank: pd.DataFrame) -> pd.DataFrame:
    """表c：各口径下对等暴露第一名与无泄漏第一名属于同一种方法的留出域数 / 总数。"""
    rows: list[dict[str, Any]] = []
    if rank.empty:
        return pd.DataFrame(rows)
    for enc, by_enc in rank.groupby("编码器", sort=False):
        for scope, sets in SCOPES.items():
            s = by_enc[by_enc["dataset"].isin(sets)]
            if s.empty:
                continue
            diff = s[~s["第一名相同"].astype(bool)]
            changed = zip(diff["dataset"], diff["held"], diff["第一名换成"], strict=True)
            n_same, n_wide = int(s["第一名相同"].sum()), int(s["第一名相同(含参照)"].sum())
            rows.append(
                {
                    "编码器": enc,
                    "口径": scope,
                    "K": PROTO_K,
                    "留出域数": len(s),
                    "第一名相同数": n_same,
                    "第一名相同/总数": f"{n_same}/{len(s)}",
                    "无泄漏第一的分布": tally(s["无泄漏第一"]),
                    "对等暴露第一的分布": tally(s["对等暴露第一"]),
                    "第一名不同的留出域": "; ".join(f"{d}/{h}（{c}）" for d, h, c in changed),
                    "第一名相同数(含参照)": n_wide,
                    "第一名相同/总数(含参照)": f"{n_wide}/{len(s)}",
                }
            )
    return pd.DataFrame(rows)


def aggregate(stage: str, allow_partial: bool, logger: eu.Logger) -> None:
    """汇总各种子的表，与 64、74 号的原型网络数值排名，并核对划分。"""
    long, info, comp = load_mine(stage)
    seeds = sorted(int(s) for s in info["种子"].unique())
    problems: list[str] = []
    if stage == "formal" and set(seeds) != {int(s) for s in eu.RECOMMENDED_SEEDS}:
        problems.append(f"种子集合 {seeds} 不是正式的 5 个种子")
    for _, r in info[info["状态"] == STATUS_FAIL].iterrows():
        problems.append(f"失败的格：{r['dataset']}/{r['held']} 种子 {r['种子']}（{r['文件']}）")
    cover = info.groupby(["dataset", "held", "K"])["种子"].nunique()
    for (ds, held, k), n in cover[cover < len(seeds)].items():
        problems.append(f"缺种子：{ds}/{held} K={k} 只有 {n}/{len(seeds)} 个种子的记录")
    ok = info[info["状态"] == STATUS_OK]
    caps = sorted({int(v) for v in ok["SVM迭代上限"].dropna()})
    if len(caps) > 1:
        problems.append(f"池化 SVM 的迭代上限不一致：{caps}（各种子、各留出域要用同一个 --svm_max_iter）")
    n_cut = int(ok["停在上限的拟合数(选参)"].sum()) + int(ok["停在上限的拟合数(评估)"].sum())
    logger.log(f"池化 SVM 迭代上限 {caps}；停在上限的拟合共 {n_cut} 次（逐留出域见表e）")
    proto, ref, missing = load_proto(stage, seeds)
    problems += [f"缺 64/74 号的逐种子表：{m}" for m in missing]
    chk = split_check(info, ref)
    for _, r in chk[chk["一致"].eq(False)].iterrows():
        problems.append(
            f"划分核对不一致：{r['dataset']}/{r['held']} 种子 {r['种子']} {r['项目']}"
            f"（本脚本 {r['本脚本']}，64/74 号 {r['64/74号']}）"
        )
    for _, r in chk[chk["一致"].isna()].iterrows():
        problems.append(f"划分无法核对：64/74 号的表里查不到 {r['dataset']}/{r['held']} 种子 {r['种子']}")
    n_true, n_false = int(chk["一致"].eq(True).sum()), int(chk["一致"].eq(False).sum())
    logger.log(f"划分核对：一致 {n_true} 项，不一致 {n_false} 项，无法核对 {int(chk['一致'].isna().sum())} 项")
    inc = increment_table(long)
    rank = ranking_table(long, proto)
    rank_sum = ranking_summary(rank)
    logger.log("\n暴露增量（表a）\n" + inc.to_string(index=False))
    if not rank_sum.empty:
        logger.log("\n第一名是否相同（表c）\n" + rank_sum.to_string(index=False))
    if problems:
        shown = problems[:60]
        more = f"\n  … 另有 {len(problems) - len(shown)} 条" if len(problems) > len(shown) else ""
        logger.log(f"\n不全或不一致 {len(problems)} 条：\n  " + "\n  ".join(shown) + more)
        if stage == "formal" and not allow_partial:
            logger.close()
            raise SystemExit("结果不全或划分核对未过，没有写汇总表；只看进度时加 --allow_partial")
    sfx = "" if stage == "formal" else "__pilot"
    out = os.path.join(eu.OUTPUT_DIR, f"{OUT_STEM}{sfx}.xlsx")
    sheets = [
        ("暴露增量", inc),
        ("逐留出域排名", rank),
        ("汇总计数", rank_sum),
        ("逐种子长表", long),
        ("留出域信息", info),
        ("划分核对", chk),
        ("不全或不一致", pd.DataFrame({"条目": problems})),
    ]
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for i, (title, frame) in enumerate(sheets):
            frame.to_excel(writer, sheet_name=eu.table_sheet_name(i, title), index=False)
        comp.to_excel(writer, sheet_name="compute", index=False)
    logger.log(f"\n汇总表 → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 四、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def check_designs() -> None:
    """64 号 DESIGNS 与 76 号 DESIGN_SRC 的源年份应逐项相同（源域池取前者，源年份选参照后者的做法）。"""
    for design in APPLE_DESIGNS.values():
        for tgt, fold in m64.DESIGNS[design].items():
            if list(fold["src_years"]) != list(m76.DESIGN_SRC[design][tgt]):
                raise RuntimeError(f"{design} 留出 {tgt}：64 号与 76 号的源年份不一致")


def run_tag(scope: str, ds_list: list[str], k: int) -> str:
    """只跑部分范围时文件名里加的范围段：各批写各自的表，--aggregate 一并读入，不会互相覆盖。"""
    parts: list[str] = []
    if scope == "apple":
        parts.append("apple")
    elif set(ds_list) != set(PUBLIC_SETS):
        parts.append(f"{scope}-" + "+".join(ds_list))
    elif scope == "public":
        parts.append("public")
    if k != SPLIT_K:
        parts.append(f"k{k}")
    return "".join(f"__{p}" for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="83_symmetric_exposure：池化 SVM 与 PLS-DA 的对等暴露")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED, help="一次一个种子")
    parser.add_argument("--stage", default=None, choices=["pilot", "formal"], help="缺省：运行时 pilot，汇总时 formal")
    parser.add_argument("--data_root", default="05data", help="公开数据集根目录")
    parser.add_argument("--scope", default="all", choices=["apple", "public", "all"])
    parser.add_argument("--datasets", default=",".join(PUBLIC_SETS), help="逗号分隔的公开数据集名")
    parser.add_argument("--k", type=int, default=SPLIT_K, help="每类支持样本数")
    parser.add_argument("--n_jobs", type=int, default=4, help="joblib 并行数（按留出域）")
    parser.add_argument("--svm_max_iter", type=int, default=SVM_MAX_ITER, help="池化 SVM 的迭代上限，-1 为不设限")
    parser.add_argument("--aggregate", action="store_true", help="汇总各种子的表并与原型网络排名")
    parser.add_argument("--allow_partial", action="store_true", help="汇总时结果不全也写表（只看进度）")
    args = parser.parse_args()
    stage: str = args.stage or ("formal" if args.aggregate else "pilot")
    logger = eu.get_logger(OUT_STEM)
    if args.aggregate:
        aggregate(stage, args.allow_partial, logger)
        logger.close()
        return

    ds_list = [d for d in args.datasets.split(",") if d] if args.scope != "apple" else []
    unknown = sorted(set(ds_list) - set(PUBLIC_SETS))
    if unknown:
        raise SystemExit(f"未知数据集：{unknown}")
    if args.k < 1:
        raise SystemExit("--k 至少为 1")
    if args.svm_max_iter == 0 or args.svm_max_iter < -1:
        raise SystemExit("--svm_max_iter 取正整数，或 -1（不设限）")
    eu.log_experiment_header(
        logger,
        {
            "任务": "基线的对等暴露（池化 SVM 与 PLS-DA 的无暴露臂与参数级暴露臂）",
            "阶段": f"{stage}（纯 CPU）",
            "种子": args.seed,
            "范围": f"{args.scope}；公开数据集 {ds_list}",
            "K / 每格评估情节": f"{args.k} / {REPS[stage]}",
            "每个伪目标的选参情节（参照 / 池化学习器）": f"{TUNE_EPISODES[stage]} / {POOL_TUNE_EPISODES[stage]}",
            "源域每类上限 / 池化 SVM 迭代上限": f"{SRC_PER_CLASS} / {args.svm_max_iter}",
            "并行": args.n_jobs,
        },
    )
    if stage == "formal" and args.seed not in [int(s) for s in eu.RECOMMENDED_SEEDS]:
        logger.log(f"[WARN] 种子 {args.seed} 不在正式的 5 个种子里")
    common: dict[str, Any] = {
        "seed": args.seed,
        "stage": stage,
        "k": args.k,
        "reps": REPS[stage],
        "n_tune": TUNE_EPISODES[stage],
        "n_tune_pool": POOL_TUNE_EPISODES[stage],
        "svm_max_iter": args.svm_max_iter,
    }
    tasks: list[dict[str, Any]] = []
    if args.scope != "public":
        check_designs()
        df = eu.load_all_elements_df(verify_sha256=True)
        wl_cols = eu.get_spectrum_columns(df)
        # 与 64 号 main 相同的构造：划分函数吃的就是这个 apple_data
        apple: dict[str, Any] = {
            "apple_data": build_apple_data(df, wl_cols),
            "year_data": build_year_region_data(df, wl_cols),
        }
        for d_idx, (ds, design) in enumerate(APPLE_DESIGNS.items()):
            for tgt in sorted(m64.DESIGNS[design]):
                tasks.append(
                    {
                        **common,
                        **apple,
                        "kind": "apple",
                        "dataset": ds,
                        "held": f"Y{tgt}",
                        "design": design,
                        "design_idx": d_idx,
                        "tgt_year": int(tgt),
                    }
                )
    for dsname in ds_list:
        # 与 74 号 main 相同的读法：loader(data_root, 种子)
        doms, is_reg = m74.DATASETS[dsname]["loader"](args.data_root, args.seed)
        logger.log(f"【{dsname}】域 {[f'{d}:{len(v[1])}' for d, v in doms.items()]}")
        for held in doms:
            tasks.append(
                {
                    **common,
                    "kind": "public",
                    "dataset": dsname,
                    "held": str(held),
                    "ds_idx": PUBLIC_SETS.index(dsname),
                    "doms": doms,
                    "is_reg": is_reg,
                }
            )
    if not tasks:
        raise SystemExit("没有要跑的留出域")
    logger.log(f"共 {len(tasks)} 个留出域，n_jobs={args.n_jobs}")

    results: list[dict[str, Any]] = []
    with eu.track_compute("对等暴露", logger, None) as budget:
        try:
            pool = Parallel(n_jobs=args.n_jobs, return_as="generator")
        except TypeError:  # 旧版 joblib 没有 return_as：结果在全部留出域完成后一起返回
            pool = Parallel(n_jobs=args.n_jobs)
        for res in pool(delayed(run_cell)(t) for t in tasks):
            results.append(res)
            info = res["info"]
            cell = f"{info['dataset']}/{info['held']}"
            brief = "  ".join(f"{r['路线']}={r['F1']:.3f}" for r in res["rows"])
            logger.log(f"  [{len(results)}/{len(tasks)}] {cell}：{info['状态']}（{info['耗时s']} s）")
            logger.log(f"      {brief or info['说明']}")

    res_df = pd.DataFrame([r for res in results for r in res["rows"]])
    info_df = pd.DataFrame([res["info"] for res in results])
    suffix = "" if stage == "formal" else "__pilot"
    os.makedirs(SEED_DIR, exist_ok=True)
    out = os.path.join(SEED_DIR, f"{OUT_STEM}__seed{args.seed}{run_tag(args.scope, ds_list, args.k)}{suffix}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        res_df.to_excel(writer, sheet_name=eu.table_sheet_name(0, "逐格结果"), index=False)
        info_df.to_excel(writer, sheet_name=eu.table_sheet_name(1, "留出域信息"), index=False)
        # 本脚本只用 CPU；track_compute 在有 CUDA 的机器上会把墙钟折成 gpu_hours，这里标明设备
        compute = {
            "seed": args.seed,
            "stage": stage,
            "device": "cpu",
            "n_jobs": args.n_jobs,
            "svm_max_iter": args.svm_max_iter,
            "留出域数": len(tasks),
            **budget,
        }
        pd.DataFrame([compute]).to_excel(writer, sheet_name="compute", index=False)
    logger.log(f"\n结果表 → {out}")
    n_fail = int((info_df["状态"] == STATUS_FAIL).sum())
    if n_fail:
        logger.log(f"[FAIL] {n_fail} 个留出域失败，异常文本见上与结果表的「留出域信息」")
    logger.close()
    if n_fail:
        raise SystemExit(f"{n_fail} 个留出域失败：修好后只补跑这些范围（--scope、--datasets），或整表重跑")


if __name__ == "__main__":
    main()
