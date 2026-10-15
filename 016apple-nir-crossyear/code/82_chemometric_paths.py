"""82_chemometric_paths.py: 化学计量学流程里的留出域暴露——标度、波长选择与交叉验证重用（补充材料 S9.2）。

为什么:
    稿件量过的暴露途径是编码器权重、检查点与基线的超参数；预处理统计量与波长选择只引了前人的报告。
    本脚本不训练编码器，用两个池化学习器在苹果 5 折与公开数据集 16 个留出域上把这两条途径各量一次：
    同一批情节上只改「标度统计量在哪些光谱上拟合」或「波段在哪些带标签光谱上选」，其后的建模相同，
    两臂作差即留出域暴露带来的抬高量。交叉验证重用一条不新算，汇总时从 76 号、74 号的逐种子表里取。

学习器（训练集 = 源域池 + 留出域的 K 个支持样本；源域池 = 源域带标签光谱每类至多 --src_per_class 条）:
    Pooled-SVM     与 65 号的 Pooled-SVM 相同：StandardScaler + RBF-SVM（C=10、gamma="scale"、类别平衡权重）
    Pooled-PLS-DA  同构写法：StandardScaler + PLS 回归，响应为类别指示矩阵（各列按标准差缩放），取预测值
                   最大的类。PLS 回归不接受样本权重，类别平衡用整数倍重复少数类的行实现。潜变量数在源域池
                   内部做 5 折分层交叉验证选（候选同 71 号 PLS_GRID，宏 F1 并列取小），不含支持样本与
                   留出域；同一留出域的各 K、各路径、各臂共用。--pls_ncomp 给整数时取固定值
    源域池         苹果三年设计直接取 65 号 fold_assets 的 xs、ys（与 65 号 Pooled-SVM 是同一批光谱）；
                   苹果两年设计与公开数据集按同一写法抽取

路径与配对臂（同一格内各臂共用同一批支持集与查询集）:
    scale        标度，无标签暴露。逐波段均值与标准差（StandardScaler）
                   无暴露  在「源域池 + 支持样本」上拟合，即训练集自身（Pooled-SVM 的这一臂就是 65 号的写法）
                   暴露    在「源域池 ∪ 留出域全部光谱」上拟合，不用留出域的标签
                 其后都在「源域池 + 支持样本」上建模。RBF 核只看样本间的差，PLS 回归自身按训练集再中心化，
                 两臂的差别因此主要来自逐波段标准差
    select       波长选择，带标签暴露。按 F 统计量（sklearn 的 f_classif）取前 --band_frac 的波段
                   无暴露  在「源域池 + 支持样本」上选（逐情节）
                   暴露    在「源域池 ∪ 留出域全部带标签光谱」上选（每个留出域一次）
                 选完都在「源域池 + 支持样本」的所选波段上建模，标度在训练集内拟合
    select_half  波长选择的半分版。留出域分成 A、B 两半（苹果按苹果编号、按产地分层；公开数据集逐类、
                 按样本；奇数时 B 多一个），两臂的支持与查询都只取自 B 半
                   无暴露  在「源域池 + 支持样本」上选
                   暴露    在「源域池 ∪ A 半带标签光谱 ∪ 支持样本」上选：两臂的选择集只差 A 半，
                           查询样本都不在其中
    交叉验证重用   只在 --aggregate 里，从现有逐种子表取数：把选参情节上的最高分当成绩，对同一超参数在
                 评估情节上的成绩。苹果取 76 号的「目标年选参宏F1」对「目标宏F1(目标年选参)」；公开数据集
                 取 74 号路线面板「所选超参数」里记的选参宏 F1 对同一路线的 F1。学习器是只用支持样本的
                 SVM 与 PLS-DA；两张表里的选参分数都只存了 4 位小数。文件名、表名与列名取自写表的代码，
                 集中在常量区

范围、K 与情节:
    苹果      两年设计 LOEO-2018、2019 与三年设计 LOEO-2018、2019、2025（源年份取 76 号 DESIGN_SRC）；
              K = 1、5、10、20、50（76 号所用的 71 号 K_LIST）；情节同 65 号（42 号 build_episode，
              种子 S 的第 i 个情节用 S+i：支持与查询各果随机取一面，查询为该年其余全部苹果）
    公开      74 号 DATASETS 的五个数据集、16 个留出域；类别集合按 K=5 定（cdl.prepare_fold）；
              K = 1、5、10、20（74 号 ROUTE_K）；情节同 74 号路线面板（kshot_episodes）
    留出域（半分版为 B 半）有类不足 K+2 个样本的格不评，记入「跳过的格」
    每格 50 个情节（--stage pilot 时 10 个，是正式 50 个里的前 10 个），一次一个种子

随机数:
    新增的抽样都走独立流 np.random.default_rng([种子, 82, 流编号, …])：流 1 源域池每类限量，
    流 2 半分，流 3 选潜变量数的交叉验证划分。74 号 kshot_episodes 每次调用自行播种全局 numpy 随机数
    （与 74 号的情节逐次相同靠的就是这一点）；本脚本其余部分不读全局随机数，结果与 --n_jobs 无关

自检:
    三年设计里 Pooled-SVM 的 scale·无暴露臂与 65 号的 Pooled-SVM 是同一源域池、同一批情节、同一设置。
    种子、K、情节数都相同的格与 04outputs/65_domain_adaptation_baselines.xlsx 逐格比对（那边存 4 位小数），
    差超过 5e-4 的格在日志与「与65号比对」表里列出；没有该文件或没有可比的格时记「未比对」。
    运行前另核对 65 号 _svm_fit_predict 的估计器设置与暴露臂所用的 SVM_KW 一致，不一致即中止

复用:
    export_utils      路径常量、日志、算力记录、冻结数据的读取与 SHA-256 核对、表名
    protonet          build_year_region_data
    crossdataset_lib  prepare_fold（类别集合与逐谱 SNV）；各数据集的加载器经 74 号 DATASETS 取
    20 号 snv；42 号 build_episode（经 65 号的 M42）；65 号 fold_assets、_svm_fit_predict、FOLD_CFG；
    71 号 K_LIST、PLS_GRID、pick_best；74 号 DATASETS、ROUTE_K、kshot_episodes、_t_interval；
    76 号 DESIGN_SRC

输入:
    03data/processed/all_9elements.csv（读入时按同目录数据卡核对 SHA-256）
    --data_root 下 74 号用的五个公开数据集
    04outputs/formal_seeds/76_target_tuning__seed<S>.xlsx、74_multi_encoder__cnn__seed<S>.xlsx  只在汇总时读
    04outputs/65_domain_adaptation_baselines.xlsx  只用于自检，可以没有

运行方式:
    cd <项目根>
    # 预跑：种子 2004，每格 10 个情节
    OMP_NUM_THREADS=1 python 02code/82_chemometric_paths.py --stage pilot --seed 2004 --n_jobs 8 \
        --data_root <公开数据集根目录>
    # 正式：每个正式种子一次，每格 50 个情节
    OMP_NUM_THREADS=1 python 02code/82_chemometric_paths.py --stage formal --seed 20060515 --n_jobs 8 \
        --data_root <公开数据集根目录>
    # 汇总：缺省读正式种子的表；加 --stage pilot 读预跑的表
    python 02code/82_chemometric_paths.py --aggregate

输出文件:
    04outputs/formal_seeds/82_chemometric_paths__seed<S>[__pilot].xlsx — 逐种子表
        表a 逐格结果：长表，一行一格（范围、数据集、留出域、种子、学习器、路径、臂、K、评估样本、
             宏 F1 均值与情节 SD、情节数）
        表b 逐格配对：两臂逐情节作差的均值与情节 SD、暴露臂更高与更低的情节数、两臂所选波段的重合率
        表c 各留出域的设置；表d 跳过的格；表e 与65号比对；compute
        只跑一侧（--scope apple / public）时文件名加 __apple / __public；数据集、路径、K 等取非缺省值时
        再加 __sub-… 段，带这一段的表不进汇总
    04outputs/82_chemometric_paths.xlsx — 汇总（--aggregate；读预跑表时文件名加 __pilot）
        表a 逐留出域对照：各路径「暴露 − 无暴露」按（数据集, 留出域, 学习器, K）的种子均值、种子 t 区间、
             为正种子数
        表b 跨域汇总：按口径、路径、学习器、K 的均值与为正留出域数 / 总数
        表c 交叉验证重用（逐留出域）；表d 交叉验证重用的跨域汇总；表e 与65号比对；表f 跳过的格；
        表g 长表；compute
    05logs/82_chemometric_paths_YY-MM-DD_HHMMSS.log

说明:
    全部是 CPU 上的 sklearn / numpy 计算；--stage 只决定每格情节数与输出文件名，不要求 CUDA。
    各表数值都不舍入。只写上面两类工作簿与日志。
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import re
import sys
from collections.abc import Callable, Iterator
from importlib import metadata
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import export_utils as eu
from protonet import build_year_region_data

NDArray = npt.NDArray[Any]
Episode = tuple[NDArray, NDArray, NDArray, NDArray]  # 支持光谱、支持标签、查询光谱、查询标签（标签为类别序号）
Scores = dict[tuple[str, str, str], list[float]]  # (学习器, 路径, 臂) → 逐情节宏 F1
Pool = tuple[int, list[str], Callable[[int], Iterator[Episode]]]  # 最小类样本数、要跑的路径、按 K 给出情节

STEM = "82_chemometric_paths"
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
# 复用的脚本与模块；并行子进程里经 lib() 按名字引入
M20, M65 = "20_unified_singleface_kshot", "65_domain_adaptation_baselines"
M71, M74, M76 = "71_source_only_tuning", "74_multi_encoder_leakage", "76_target_tuning_leakage"
CDL = "crossdataset_lib"
m71: Any = importlib.import_module(M71)
m74: Any = importlib.import_module(M74)
m76: Any = importlib.import_module(M76)

# ── 范围、K 与情节数 ────────────────────────────────────────────────────────────
APPLE_SETS: dict[str, str] = {"apple_two_year": "two_year", "apple_three_year": "three_year"}  # 数据集名同 75 号
PUBLIC_SETS: tuple[str, ...] = tuple(m74.DATASETS)
DESIGN_SRC: dict[str, dict[int, list[int]]] = {
    d: {int(t): [int(y) for y in ys] for t, ys in m76.DESIGN_SRC[d].items()} for d in APPLE_SETS.values()
}
DESIGN_CODE: dict[str, int] = {"two_year": 2, "three_year": 3}
K_APPLE: tuple[int, ...] = tuple(int(k) for k in m71.K_LIST)  # 76 号用的 K
K_PUBLIC: tuple[int, ...] = tuple(int(k) for k in m74.ROUTE_K)  # 74 号路线面板的 K
K_REF = 5  # 公开数据集的类别集合按 K=5 定（74 号 --k 的缺省值）
REPS: dict[str, int] = {"formal": 50, "pilot": 10}

# ── 学习器、路径与臂 ────────────────────────────────────────────────────────────
LEARNERS: dict[str, str] = {"svm": "Pooled-SVM", "pls": "Pooled-PLS-DA"}
PATHS: dict[str, str] = {"scale": "标度", "select": "波长选择", "select_half": "波长选择(半分)"}
ARM_NONE, ARM_EXP = "无暴露", "暴露"
DELTA = "暴露−无暴露"
POOL_ALL, POOL_B = "全部", "B半"
# 暴露臂另给标度时所建 SVC 的设置：与 65 号 _svm_fit_predict 相同，运行前由 check_svm_settings 核对
SVM_KW: dict[str, Any] = {"kernel": "rbf", "C": 10.0, "gamma": "scale", "class_weight": "balanced"}
# 65 号 fold_assets 另需的四项（CORAL、TCA 的设置，取 65 号命令行缺省值）；本脚本只用它返回的 xs、ys
CFG65: dict[str, Any] = {"tca_fit": 800, "tca_dim": 30, "tca_mu": 1.0, "coral_lam": 1.0}
PLS_GRID: tuple[int, ...] = tuple(int(g) for g in m71.PLS_GRID)  # 潜变量数候选
CV_SPLITS = 5  # 选潜变量数的源域池内部交叉验证折数
BAND_FRAC, SRC_PER_CLASS = 0.2, 200  # --band_frac 与 --src_per_class 的缺省值
STREAM_SRC, STREAM_HALF, STREAM_CV = 1, 2, 3  # 随机数流编号：default_rng([种子, 82, 流编号, …])

# ── 交叉验证重用所读的现有逐种子表：文件名、表名与列名取自 76 号 run_seed、74 号 route_panel 的写表代码 ──
CV_PATH = "交叉验证重用"
CV_BASE, CV_EXP, CV_DELTA = "评估情节成绩", "选参情节最高分", "选参最高分−评估成绩"
F76_GLOB = "76_target_tuning__seed*.xlsx"  # 取第 1 张表（表a：逐设计逐折逐K）
C76_DESIGN, C76_TGT, C76_K, C76_METHOD = "设计", "目标年份", "K", "方法"
C76_TUNE = "目标年选参宏F1"  # 目标年选参情节上的最高分（存 4 位小数）
C76_EVAL = "目标宏F1(目标年选参)"  # 同一超参数在评估情节上的宏 F1
F74_GLOB = "74_multi_encoder__cnn__seed*.xlsx"  # 取表名含「路线面板」的那张表
S74_PANEL = "路线面板"
C74_DATASET, C74_HELD, C74_K, C74_ROUTE, C74_F1 = "dataset", "held", "K", "路线", "F1"
C74_CHOSEN = "所选超参数"  # 形如「('rbf', 10.0, 1.0)（选参宏F1 0.8123）」
ROUTES74: dict[str, str] = {"SVM(留出域选参)": "SVM-Kshot", "PLS-DA(留出域选参)": "PLS-DA-Kshot"}
TUNE_RE = re.compile(r"选参宏F1\s*(-?\d+(?:\.\d+)?)")

# ── 自检所读的 65 号汇总表：第 1 张表（表a：原始结果），列名取自 65 号 main ──────────────
REF_65 = os.path.join(eu.OUTPUT_DIR, "65_domain_adaptation_baselines.xlsx")
C65_TGT, C65_K, C65_SEED, C65_METHOD, C65_F1, C65_N = "目标年份", "K", "种子", "方法", "F1均值", "情节数"
M65_POOLED = "Pooled-SVM"
CHECK_TOL = 5e-4

# ── 汇总口径（同 75 号：两种苹果设计共用 2018、2019 两个留出年，不同时计入）─────────────
SCOPES: dict[str, list[str]] = {
    "主口径（苹果两年设计 + 公开数据集）": ["apple_two_year", *PUBLIC_SETS],
    "补充口径（苹果三年设计 + 公开数据集）": ["apple_three_year", *PUBLIC_SETS],
    "苹果两年设计": ["apple_two_year"],
    "苹果三年设计": ["apple_three_year"],
    "公开数据集": list(PUBLIC_SETS),
}
CELL_KEY = ["路径", "范围", "数据集", "留出域", "学习器", "K"]
SKIP_COLS = ["范围", "数据集", "留出域", "种子", "评估样本", "K", "原因"]
SEED_RE = re.compile(rf"^{re.escape(STEM)}__seed(\d+)(?:__(apple|public))?(__pilot)?\.xlsx$")


# ═══════════════════════════════════════════════════════════════════════════════
# 一、抽样、波长选择与两个学习器（并行子进程里也要用：不引用 eu 与各编号脚本的模块对象，一律经 lib() 引入）
# ═══════════════════════════════════════════════════════════════════════════════
def lib(name: str) -> Any:
    """按名字引入本目录的模块或编号脚本。先把脚本目录放进 sys.path，不依赖并行后端是否把它带给子进程。"""
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)
    return importlib.import_module(name)


def stream(seed: int, no: int, *key: int) -> np.random.Generator:
    """本脚本新增抽样的独立随机流：default_rng([种子, 82, 流编号, *键])。"""
    return np.random.default_rng([seed, 82, no, *key])


def cap_per_class(x: NDArray, y: NDArray, cap: int, rng: np.random.Generator) -> tuple[NDArray, NDArray]:
    """每类至多保留 cap 条，超出时无放回抽取（与 65 号 fold_assets 同一写法）。"""
    keep: list[NDArray] = []
    for c in np.unique(y):
        ii: NDArray = np.flatnonzero(y == c)
        if len(ii) > cap:
            ii = np.sort(rng.choice(ii, size=cap, replace=False))
        keep.append(ii)
    kept = np.concatenate(keep)
    return np.asarray(x[kept]), np.asarray(y[kept])


def n_select(n_band: int, frac: float) -> int:
    """波长选择保留的波段数：波段数 × 比例四舍五入，至少 1 个。"""
    return max(1, min(n_band, int(np.floor(frac * n_band + 0.5))))


def top_bands(x: NDArray, y: NDArray, frac: float) -> NDArray:
    """按 F 统计量（f_classif）从高到低取前 frac 比例的波段，返回升序的列下标；常数波段排在最后。"""
    f_stat = np.asarray(f_classif(np.asarray(x, dtype=np.float64), y)[0], dtype=np.float64)
    f_stat = np.where(np.isnan(f_stat), -np.inf, f_stat)
    return np.sort(np.argsort(-f_stat, kind="stable")[: n_select(int(x.shape[1]), frac)])


def balanced_rows(y: NDArray, n_class: int) -> NDArray:
    """类别平衡用的行下标：各类按「最大类样本数 / 本类样本数」四舍五入后的整数倍重复。"""
    counts = np.bincount(y, minlength=n_class)
    n_max = int(counts.max())
    parts: list[NDArray] = []
    for c in np.flatnonzero(counts):
        times = max(1, int(np.floor(n_max / counts[c] + 0.5)))
        parts.append(np.tile(np.flatnonzero(y == c), times))
    return np.concatenate(parts)


def pls_predict(z_tr: NDArray, y_tr: NDArray, z_te: NDArray, n_class: int, n_comp: int) -> NDArray:
    """已标度光谱上的 PLS-DA：类别平衡后对（各列按标准差缩放的）类别指示矩阵做 PLS 回归，取预测值最大的类。

    标度取自训练集、各类样本数相近（不触发重复）时，与 20、42、71 号的 PLSRegression(scale=True) 写法
    数值等价。潜变量数受训练样本数与波段数截断。
    """
    rows = balanced_rows(y_tr, n_class)
    y_oh = np.zeros((len(rows), n_class))
    y_oh[np.arange(len(rows)), y_tr[rows]] = 1.0
    y_sd = y_oh.std(axis=0, ddof=1)
    y_sd[y_sd == 0.0] = 1.0
    n_use = max(1, min(int(n_comp), len(y_tr) - 1, int(z_tr.shape[1])))
    pls = PLSRegression(n_components=n_use, scale=False, max_iter=500)
    pls.fit(z_tr[rows], y_oh / y_sd)
    return np.asarray((np.asarray(pls.predict(z_te)) * y_sd).argmax(axis=1), dtype=int)


def fit_predict(
    kind: str, scaler: Any, x_tr: NDArray, y_tr: NDArray, x_te: NDArray, n_class: int, n_comp: int
) -> NDArray:
    """在 (x_tr, y_tr) 上建模并预测 x_te。scaler 为 None 时标度在训练集内拟合；否则用给定的标度，其余相同。"""
    if kind == "svm":
        if scaler is None:
            pred, _ = lib(M65)._svm_fit_predict(x_tr, y_tr, x_te)
            return np.asarray(pred, dtype=int)
        svc = SVC(**SVM_KW).fit(scaler.transform(x_tr), y_tr)
        return np.asarray(svc.predict(scaler.transform(x_te)), dtype=int)
    sc = StandardScaler().fit(x_tr) if scaler is None else scaler
    return pls_predict(np.asarray(sc.transform(x_tr)), y_tr, np.asarray(sc.transform(x_te)), n_class, n_comp)


def macro_f1(y_true: NDArray, y_pred: NDArray, n_class: int) -> float:
    """宏 F1。查询集里各类都有样本，这时与 20 号、74 号各自的算法取值相同。"""
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_class)), zero_division=0))


def choose_pls_ncomp(xs: NDArray, ys: NDArray, n_class: int, rng: np.random.Generator) -> tuple[int, str]:
    """源域池内部分层交叉验证选 PLS 潜变量数：候选同 71 号 PLS_GRID，取宏 F1 最高者，并列取小。

    只用源域池，不含支持样本与留出域。源域池里不足两类时取类别数（20、42 号的固定写法）。
    返回 (所选值, 写进设置表的说明)。
    """
    present = np.unique(ys)
    if len(present) < 2:
        return n_class, "源域池不足两类，取类别数"
    y_loc = np.searchsorted(present, ys)
    fold_id = np.zeros(len(ys), dtype=int)
    for c in range(len(present)):
        ii = np.flatnonzero(y_loc == c)
        fold_id[rng.permutation(ii)] = np.arange(len(ii)) % CV_SPLITS
    scores: dict[int, list[float]] = {g: [] for g in PLS_GRID}
    for f in range(CV_SPLITS):
        tr, va = fold_id != f, fold_id == f
        if not va.any() or len(np.unique(y_loc[tr])) < len(present):
            continue
        sc = StandardScaler().fit(xs[tr])
        z_tr, z_va = np.asarray(sc.transform(xs[tr])), np.asarray(sc.transform(xs[va]))
        for g in PLS_GRID:
            pred = pls_predict(z_tr, y_loc[tr], z_va, len(present), g)
            scores[g].append(macro_f1(y_loc[va], pred, len(present)))
    tune = {g: float(np.mean(v)) if v else float("-inf") for g, v in scores.items()}
    best = int(lib(M71).pick_best(PLS_GRID, tune))
    return best, f"源域池内 {CV_SPLITS} 折宏F1：" + "; ".join(f"{g}:{tune[g]:.4f}" for g in PLS_GRID)


def check_svm_settings() -> None:
    """核对 65 号 Pooled-SVM 的写法：StandardScaler 后接 SVC，且 SVC 的全部设置与 SVC(**SVM_KW) 相同。"""
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [3.0, 3.0], [3.0, 4.0], [4.0, 3.0]])
    _, clf = lib(M65)._svm_fit_predict(x, np.array([0, 0, 0, 1, 1, 1]), x[:1])
    steps = clf.named_steps
    same = (
        list(steps) == ["scaler", "svm"]
        and steps["scaler"].get_params() == StandardScaler().get_params()
        and steps["svm"].get_params() == SVC(**SVM_KW).get_params()
    )
    if not same:
        raise RuntimeError("65 号 _svm_fit_predict 的估计器设置与本脚本的 SVM_KW 不一致：先对齐再跑")


# ═══════════════════════════════════════════════════════════════════════════════
# 二、一个留出域内的运行：各情节共用的量、一格、全部格（并行子进程里运行）
# ═══════════════════════════════════════════════════════════════════════════════
def make_fold(
    xs: NDArray, ys: NDArray, xh: NDArray, yh: NDArray, n_class: int, cfg: dict[str, Any], cv_rng: np.random.Generator
) -> dict[str, Any]:
    """一个留出域里各情节共用的量：源域池、PLS 潜变量数，以及暴露臂的标度与波段。

    暴露臂的标度与波段只依赖源域池与留出域全部光谱（xh、yh），与情节无关，每个留出域算一次。
    """
    fold: dict[str, Any] = {"xs": xs, "ys": ys, "n_class": n_class, "n_comp": n_class, "ncomp_note": "未用"}
    if "pls" in cfg["learners"]:
        if cfg["pls_ncomp"] == "cv":
            fold["n_comp"], fold["ncomp_note"] = choose_pls_ncomp(xs, ys, n_class, cv_rng)
        else:
            fold["n_comp"], fold["ncomp_note"] = int(cfg["pls_ncomp"]), "固定值"
    x_union = np.concatenate([xs, xh])
    fold["scaler_exp"] = fold["sel_exp"] = None
    if "scale" in cfg["paths"]:
        fold["scaler_exp"] = StandardScaler().fit(x_union)
    if "select" in cfg["paths"]:
        fold["sel_exp"] = top_bands(x_union, np.concatenate([ys, yh]), float(cfg["band_frac"]))
    return fold


def run_cell(
    fold: dict[str, Any], episodes: Iterator[Episode], paths: list[str], cfg: dict[str, Any]
) -> tuple[Scores, dict[str, list[float]]]:
    """一格（一个留出域 × 一个 K × 一种评估样本）：各学习器、各路径两臂在同一批情节上的逐情节宏 F1。

    另返回波长选择两臂所选波段集合的逐情节重合率（交集 / 并集）。两臂选中同一组波段时模型相同，只拟合一次。
    """
    xs, ys, n_class, n_comp = fold["xs"], fold["ys"], int(fold["n_class"]), int(fold["n_comp"])
    learners: list[str] = cfg["learners"]
    frac = float(cfg["band_frac"])
    scores: Scores = {(ln, p, a): [] for ln in learners for p in paths for a in (ARM_NONE, ARM_EXP)}
    overlap: dict[str, list[float]] = {p: [] for p in paths if p != "scale"}
    for x_sup, y_sup, x_qry, y_qry in episodes:
        x_tr = np.concatenate([xs, x_sup])
        y_tr = np.concatenate([ys, y_sup])
        for path in paths:
            if path == "scale":
                for ln in learners:
                    for arm, scaler in ((ARM_NONE, None), (ARM_EXP, fold["scaler_exp"])):
                        pred = fit_predict(ln, scaler, x_tr, y_tr, x_qry, n_class, n_comp)
                        scores[(ln, path, arm)].append(macro_f1(y_qry, pred, n_class))
                continue
            sel_none = top_bands(x_tr, y_tr, frac)
            if path == "select":
                sel_exp = fold["sel_exp"]
            else:  # 半分版：选择集 = 源域池 ∪ A 半 ∪ 本情节的支持样本
                x_fit = np.concatenate([xs, fold["xa"], x_sup])
                sel_exp = top_bands(x_fit, np.concatenate([ys, fold["ya"], y_sup]), frac)
            overlap[path].append(len(np.intersect1d(sel_none, sel_exp)) / len(np.union1d(sel_none, sel_exp)))
            same = bool(np.array_equal(sel_none, sel_exp))
            for ln in learners:
                pred = fit_predict(ln, None, x_tr[:, sel_none], y_tr, x_qry[:, sel_none], n_class, n_comp)
                f_none = f_exp = macro_f1(y_qry, pred, n_class)
                if not same:
                    pred = fit_predict(ln, None, x_tr[:, sel_exp], y_tr, x_qry[:, sel_exp], n_class, n_comp)
                    f_exp = macro_f1(y_qry, pred, n_class)
                scores[(ln, path, ARM_NONE)].append(f_none)
                scores[(ln, path, ARM_EXP)].append(f_exp)
    return scores, overlap


def cell_rows(
    base: dict[str, Any], k: int, pool: str, scores: Scores, overlap: dict[str, list[float]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把一格的逐情节宏 F1 折成长表行（一行一臂）与配对行（一行一对臂，逐情节作差）。"""
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for (ln, path, arm), vals in scores.items():
        rows.append(
            {
                **base,
                "学习器": LEARNERS[ln],
                "路径": PATHS[path],
                "臂": arm,
                "K": k,
                "评估样本": pool,
                "宏F1均值": float(np.mean(vals)),
                "宏F1情节SD": float(np.std(vals)),
                "情节数": len(vals),
            }
        )
    for ln, path in dict.fromkeys(key[:2] for key in scores):
        none = np.asarray(scores[(ln, path, ARM_NONE)], dtype=np.float64)
        diff = np.asarray(scores[(ln, path, ARM_EXP)], dtype=np.float64) - none
        pairs.append(
            {
                **base,
                "学习器": LEARNERS[ln],
                "路径": PATHS[path],
                "K": k,
                "评估样本": pool,
                ARM_NONE: float(none.mean()),
                ARM_EXP: float((none + diff).mean()),
                DELTA: float(diff.mean()),
                "差的情节SD": float(diff.std()),
                "暴露更高的情节数": int((diff > 0).sum()),
                "暴露更低的情节数": int((diff < 0).sum()),
                "情节数": len(diff),
                "两臂所选波段重合率": float(np.mean(overlap[path])) if path in overlap else float("nan"),
            }
        )
    return rows, pairs


def run_pools(
    base: dict[str, Any], fold: dict[str, Any], k_list: list[int], pools: dict[str, Pool], cfg: dict[str, Any]
) -> dict[str, Any]:
    """一个留出域的全部格：逐 K、逐评估样本（全部 / B 半）。有类不足 K+2 个样本的格不评并记下。"""
    out: dict[str, Any] = {"rows": [], "pairs": [], "skips": [], "log": []}
    for k in k_list:
        for pool, (n_min, paths, make_episodes) in pools.items():
            if not paths:
                continue
            if n_min < k + 2:
                reason = f"有类只有 {n_min} 个样本，不足 K+2"
                out["skips"].append({**base, "评估样本": pool, "K": k, "原因": reason})
                continue
            scores, overlap = run_cell(fold, make_episodes(k), paths, cfg)
            if not all(scores.values()):
                out["skips"].append({**base, "评估样本": pool, "K": k, "原因": "没有可用的情节"})
                continue
            rows, pairs = cell_rows(base, k, pool, scores, overlap)
            out["rows"] += rows
            out["pairs"] += pairs
            msg = "  ".join(f"{p['学习器']}/{p['路径']} {p[ARM_NONE]:.4f}→{p[ARM_EXP]:.4f}" for p in pairs)
            out["log"].append(f"    K={k} {pool}：{msg}")
    return out


def fold_info(
    base: dict[str, Any], names: list[str], fold: dict[str, Any], cfg: dict[str, Any], extra: dict[str, Any]
) -> dict[str, Any]:
    """一个留出域的设置记录（写进「各留出域的设置」表）。"""
    ys = fold["ys"]
    n_band = int(fold["xs"].shape[1])
    counts = np.bincount(ys, minlength=len(names))
    return {
        **base,
        "类别数": len(names),
        "类别": ",".join(names),
        "源域池条数": len(ys),
        "源域池各类条数": "; ".join(f"{n}:{int(c)}" for n, c in zip(names, counts, strict=True)),
        "波段数": n_band,
        "所选波段数": n_select(n_band, float(cfg["band_frac"])),
        "PLS潜变量数": int(fold["n_comp"]),
        "PLS潜变量数的选法": str(fold["ncomp_note"]),
        **extra,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 三、苹果的留出年（并行子进程里运行）
# ═══════════════════════════════════════════════════════════════════════════════
def apple_spectra(
    year_data: Any, years: list[int], regions: list[str], keep: dict[str, set[Any]] | None = None
) -> tuple[NDArray, NDArray]:
    """给定年份里产地属于 regions 的全部面位光谱（逐谱 SNV）与产地序号；keep 给出时每个产地只留其中的苹果。"""
    xs_parts: list[NDArray] = []
    ys_parts: list[NDArray] = []
    for year in years:
        for i, reg in enumerate(regions):
            for aid, faces in year_data[year].get(reg, {}).items():
                if keep is not None and aid not in keep[reg]:
                    continue
                xs_parts.append(np.asarray(faces))
                ys_parts.append(np.full(len(faces), i))
    return np.asarray(lib(M20).snv(np.concatenate(xs_parts))), np.concatenate(ys_parts)


def apple_half(
    year_data: Any, tgt: int, regions: list[str], rng: np.random.Generator
) -> tuple[dict[str, set[Any]], dict[str, set[Any]]]:
    """留出年各产地的苹果按编号随机对半分，返回 ({产地: A 半的苹果编号}, {产地: B 半的苹果编号})。

    口径同 64 号 split_target_half：苹果编号级、按产地分层、奇数时 B 多一个；随机数用本脚本的流 2。
    """
    part_a: dict[str, set[Any]] = {}
    part_b: dict[str, set[Any]] = {}
    for reg in regions:
        ids = sorted(year_data[tgt][reg])
        perm = rng.permutation(len(ids))
        n_a = len(ids) // 2
        part_a[reg] = {ids[i] for i in perm[:n_a]}
        part_b[reg] = {ids[i] for i in perm[n_a:]}
    return part_a, part_b


def apple_episodes(year_data: Any, tgt: int, k: int, seed: int, reps: int, regions: list[str]) -> Iterator[Episode]:
    """与 65 号逐位相同的情节：42 号 build_episode，第 i 个情节的种子为 seed + i。"""
    code = {r: i for i, r in enumerate(regions)}
    build = lib(M65).M42.build_episode
    for i in range(reps):
        regs, x_sup, y_sup, x_qry, y_true = build(year_data, tgt, k, seed + i)
        if list(regs) != regions:
            raise RuntimeError(f"留出 {tgt}：类别顺序不一致 {regs} vs {regions}")
        y_qry = np.array([code[r] for r in y_true], dtype=int)
        yield np.asarray(x_sup), np.asarray(y_sup, dtype=int), np.asarray(x_qry), y_qry


def run_apple_unit(year_data: Any, dsname: str, tgt: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """苹果的一个留出年：源域池、留出年全部光谱与半分，各 K 在全部苹果与 B 半上的两种评估。"""
    design = APPLE_SETS[dsname]
    seed, reps, cap = int(cfg["seed"]), int(cfg["reps"]), int(cfg["src_per_class"])
    key = (0, DESIGN_CODE[design], tgt)
    src_years = DESIGN_SRC[design][tgt]
    regions: list[str] = sorted(year_data[tgt].keys())
    if design == "three_year":
        # 与 65 号 Pooled-SVM 同一批源域光谱：直接取 fold_assets 的 xs、ys（它一并算出的 CORAL、TCA 资产这里不用）
        m65 = lib(M65)
        if [int(y) for y in m65.FOLD_CFG[tgt]["src_years"]] != src_years:
            raise RuntimeError(f"留出 {tgt}：65 号与 76 号的源年份不一致")
        assets = m65.fold_assets(year_data, tgt, seed, {"src_per_class": cap, **CFG65})
        if list(assets["regions"]) != regions:
            raise RuntimeError(f"留出 {tgt}：类别顺序与 65 号不一致")
        xs, ys = np.asarray(assets["xs"]), np.asarray(assets["ys"], dtype=int)
        src_note = "65 号 fold_assets"
    else:
        x_all, y_all = apple_spectra(year_data, src_years, regions)
        xs, ys = cap_per_class(x_all, y_all, cap, stream(seed, STREAM_SRC, *key))
        src_note = "本脚本随机流 1"
    xh, yh = apple_spectra(year_data, [tgt], regions)
    fold = make_fold(xs, ys, xh, yh, len(regions), cfg, stream(seed, STREAM_CV, *key))
    base = {"范围": "apple", "数据集": dsname, "留出域": f"Y{tgt}", "种子": seed}
    full_paths = [p for p in cfg["paths"] if p != "select_half"]
    n_all = min(len(year_data[tgt][r]) for r in regions)
    pools: dict[str, Pool] = {
        POOL_ALL: (n_all, full_paths, lambda k: apple_episodes(year_data, tgt, k, seed, reps, regions)),
    }
    extra: dict[str, Any] = {
        "源域": "+".join(str(y) for y in src_years),
        "源域池来源": src_note,
        "留出域光谱条数": len(yh),
        "留出域样本数": sum(len(year_data[tgt][r]) for r in regions),
    }
    if "select_half" in cfg["paths"]:
        part_a, part_b = apple_half(year_data, tgt, regions, stream(seed, STREAM_HALF, 0, tgt))
        fold["xa"], fold["ya"] = apple_spectra(year_data, [tgt], regions, part_a)
        tgt_b = {r: {a: x for a, x in year_data[tgt][r].items() if a in part_b[r]} for r in regions}
        yd_b = {**year_data, tgt: tgt_b}
        n_b = min(len(part_b[r]) for r in regions)
        pools[POOL_B] = (n_b, ["select_half"], lambda k: apple_episodes(yd_b, tgt, k, seed, reps, regions))
        extra.update({"A半光谱条数": len(fold["ya"]), "B半样本数": sum(len(part_b[r]) for r in regions)})
    out = run_pools(base, fold, [int(k) for k in cfg["k_apple"]], pools, cfg)
    out["info"] = fold_info(base, regions, fold, cfg, extra)
    head = f"【{dsname} 留出 Y{tgt}】{len(regions)} 类，源域池 {len(ys)} 条，留出年 {len(yh)} 条谱"
    out["log"].insert(0, f"{head}，PLS 潜变量数 {fold['n_comp']}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 四、公开数据集的留出域（并行子进程里运行）
# ═══════════════════════════════════════════════════════════════════════════════
def public_half(y: NDArray, n_class: int, rng: np.random.Generator) -> NDArray:
    """留出域各类样本随机对半分，返回 A 半的布尔掩码。

    口径同 74 号 half_split：逐类、样本级、奇数时 B 多一个；随机数用本脚本的流 2。
    """
    in_a = np.zeros(len(y), dtype=bool)
    for c in range(n_class):
        ids = np.flatnonzero(y == c)
        in_a[ids[rng.permutation(len(ids))[: len(ids) // 2]]] = True
    return in_a


def public_episodes(
    x: NDArray, y_idx: NDArray, y_cls: NDArray, classes: list[int], k: int, reps: int, seed: int
) -> Iterator[Episode]:
    """与 74 号路线面板逐次相同的情节（kshot_episodes：播种一次，逐类洗牌取前 k 个作支持，其余作查询）。

    y_cls 是原始类别值（传给 kshot_episodes），y_idx 是它在 classes 里的序号；支持标签按 classes 的次序编号。
    """
    for sup, q in lib(M74).kshot_episodes(y_cls, classes, k, reps, seed):
        y_sup = np.concatenate([np.full(len(s), i) for i, s in enumerate(sup)])
        yield np.asarray(x[np.concatenate(sup)]), y_sup, np.asarray(x[q]), np.asarray(y_idx[q])


def run_public_unit(
    dsname: str,
    ds_i: int,
    doms: dict[str, tuple[NDArray, NDArray]],
    is_reg: bool,
    held: str,
    held_i: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """公开数据集的一个留出域：类别集合按 K=5 定，源域池每类限量，各 K 在全部样本与 B 半上的两种评估。"""
    seed, reps, cap = int(cfg["seed"]), int(cfg["reps"]), int(cfg["src_per_class"])
    key = (1 + ds_i, held_i)
    base = {"范围": "public", "数据集": dsname, "留出域": held, "种子": seed}
    fa = lib(CDL).prepare_fold(doms, is_reg, held, K_REF, preprocess="snv")
    if fa is None:
        skip = {**base, "评估样本": POOL_ALL, "K": "全部", "原因": "合法类不足 3 个"}
        log = [f"【{dsname} 留出 {held}】合法类不足 3 个，此域不评"]
        return {"rows": [], "pairs": [], "skips": [skip], "log": log, "info": None}
    classes: list[int] = fa.classes
    xh, yh_cls = np.asarray(fa.xh), np.asarray(fa.yh)
    n_class = len(classes)
    cls_arr = np.asarray(classes)
    yh = np.searchsorted(cls_arr, yh_cls)
    xs, ys = cap_per_class(fa.xs, np.searchsorted(cls_arr, fa.ys), cap, stream(seed, STREAM_SRC, *key))
    fold = make_fold(xs, ys, xh, yh, n_class, cfg, stream(seed, STREAM_CV, *key))
    full_paths = [p for p in cfg["paths"] if p != "select_half"]
    n_all = int(np.bincount(yh, minlength=n_class).min())
    pools: dict[str, Pool] = {
        POOL_ALL: (n_all, full_paths, lambda k: public_episodes(xh, yh, yh_cls, classes, k, reps, seed)),
    }
    extra: dict[str, Any] = {
        "源域": "+".join(d for d in doms if d != held),
        "源域池来源": "本脚本随机流 1",
        "留出域光谱条数": len(yh),
        "留出域样本数": len(yh),
    }
    if "select_half" in cfg["paths"]:
        in_a = public_half(yh, n_class, stream(seed, STREAM_HALF, *key))
        fold["xa"], fold["ya"] = xh[in_a], yh[in_a]
        xb, yb, yb_cls = xh[~in_a], yh[~in_a], yh_cls[~in_a]
        n_b = int(np.bincount(yb, minlength=n_class).min())
        pools[POOL_B] = (n_b, ["select_half"], lambda k: public_episodes(xb, yb, yb_cls, classes, k, reps, seed))
        extra.update({"A半光谱条数": int(in_a.sum()), "B半样本数": int((~in_a).sum())})
    out = run_pools(base, fold, [int(k) for k in cfg["k_public"]], pools, cfg)
    out["info"] = fold_info(base, [str(c) for c in classes], fold, cfg, extra)
    head = f"【{dsname} 留出 {held}】{n_class} 类，源域池 {len(ys)} 条，留出域 {len(yh)} 条谱"
    out["log"].insert(0, f"{head}，PLS 潜变量数 {fold['n_comp']}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 五、读表、自检与汇总（只在主进程里运行）
# ═══════════════════════════════════════════════════════════════════════════════
def seed_of(path: str) -> int:
    """从逐种子表的文件名取种子。"""
    hit = re.search(r"__seed(\d+)", os.path.basename(path))
    if hit is None:
        raise ValueError(f"文件名里没有种子：{path}")
    return int(hit.group(1))


def read_sheet(path: str, title: str) -> pd.DataFrame:
    """按标题片段取工作表（表名的写法见 eu.table_sheet_name，形如「表b：路线面板」）；没有时返回空表。"""
    with pd.ExcelFile(path) as xl:
        name = next((n for n in xl.sheet_names if title in str(n)), None)
        return xl.parse(name) if name is not None else pd.DataFrame()


def write_workbook(path: str, sheets: dict[str, pd.DataFrame]) -> None:
    """把若干张表写进一个工作簿；键即表名。"""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def own_seed_files(stage: str) -> list[str]:
    """本脚本的逐种子表：同一种子全范围表在前、分范围表在后；文件名带 __sub- 段的不取。"""
    hits: list[tuple[int, int, str]] = []
    for f in glob.glob(os.path.join(SEED_DIR, f"{STEM}__seed*.xlsx")):
        m = SEED_RE.match(os.path.basename(f))
        if m is not None and (m.group(3) is not None) == (stage == "pilot"):
            hits.append((int(m.group(1)), 0 if m.group(2) is None else 1, f))
    return [f for _, _, f in sorted(hits)]


def formal_files(pattern: str) -> list[str]:
    """formal_seeds 下匹配 pattern 的正式逐种子表（文件名不含 __pilot）。"""
    files = glob.glob(os.path.join(SEED_DIR, pattern))
    return sorted(f for f in files if "__pilot" not in os.path.basename(f))


def check_against_65(long_df: pd.DataFrame, logger: eu.Logger) -> tuple[pd.DataFrame, str]:
    """三年设计 Pooled-SVM 的 scale·无暴露臂对 65 号汇总表的 Pooled-SVM，逐 (目标年份, K, 种子) 比对。

    只比种子、K、情节数都相同的格；返回 (比对表, 写进 compute 表的状态)。
    """
    if long_df.empty or not os.path.exists(REF_65):
        logger.log("[自检] 没有本脚本的结果行或没有 65 号汇总表，未比对")
        return pd.DataFrame([{"状态": "未比对"}]), "未比对"
    ref = pd.read_excel(REF_65, sheet_name=0)
    missing = [c for c in (C65_TGT, C65_K, C65_SEED, C65_METHOD, C65_F1, C65_N) if c not in ref.columns]
    if missing:
        logger.log(f"[自检] 65 号汇总表缺列 {missing}，未比对")
        return pd.DataFrame([{"状态": f"未比对：65 号汇总表缺列 {missing}"}]), "未比对（缺列）"
    theirs = {
        (int(r[C65_TGT]), int(r[C65_K]), int(r[C65_SEED])): (float(r[C65_F1]), int(r[C65_N]))
        for _, r in ref[ref[C65_METHOD] == M65_POOLED].iterrows()
    }
    mine = long_df[
        (long_df["数据集"] == "apple_three_year")
        & (long_df["学习器"] == LEARNERS["svm"])
        & (long_df["路径"] == PATHS["scale"])
        & (long_df["臂"] == ARM_NONE)
    ]
    rows: list[dict[str, Any]] = []
    for _, r in mine.iterrows():
        key = (int(str(r["留出域"]).lstrip("Y")), int(r["K"]), int(r["种子"]))
        if key in theirs and theirs[key][1] == int(r["情节数"]):
            f_mine, f_ref = float(r["宏F1均值"]), theirs[key][0]
            cell = {"目标年份": key[0], "K": key[1], "种子": key[2], "情节数": int(r["情节数"])}
            rows.append({**cell, "本脚本": f_mine, "65号": f_ref, "绝对差": abs(f_mine - f_ref)})
    if not rows:
        logger.log("[自检] 与 65 号没有种子、K、情节数都相同的格（预跑或没跑三年设计的 scale 路径时如此），未比对")
        return pd.DataFrame([{"状态": "未比对：无可比的格"}]), "未比对（无可比的格）"
    chk = pd.DataFrame(rows)
    chk["超出容差"] = chk["绝对差"] > CHECK_TOL
    n_bad, worst = int(chk["超出容差"].sum()), float(chk["绝对差"].max())
    note = f"{len(chk)} 格，最大绝对差 {worst:.2e}，超出容差 {CHECK_TOL:g} 的 {n_bad} 格"
    logger.log(f"[自检] 与 65 号 Pooled-SVM 比对：{note}")
    if n_bad:
        logger.log("  [WARN] 超出容差的格：\n" + chk[chk["超出容差"]].to_string(index=False))
    return chk, note


def checked_against_65(long_df: pd.DataFrame, logger: eu.Logger) -> tuple[pd.DataFrame, str]:
    """check_against_65 的包装：比对只是附带的核对，读表或取数出错时记日志，照常写出已算完的结果。"""
    try:
        return check_against_65(long_df, logger)
    except Exception as exc:
        logger.log(f"  [WARN] 与 65 号比对出错：{exc}")
        return pd.DataFrame([{"状态": f"比对出错：{exc}"}]), "比对出错"


def long_pairs(long_df: pd.DataFrame) -> tuple[list[dict[str, Any]], int]:
    """长表 → 逐种子的配对（base = 无暴露臂，exposed = 暴露臂）。

    同一格出现在多张逐种子表里时取先读到的；返回 (配对, 重复而未取的行数)。
    """
    cells: dict[tuple[Any, ...], dict[str, float]] = {}
    n_dup = 0
    for _, r in long_df.iterrows():
        key = (*(str(r[c]) for c in CELL_KEY[:-1]), int(r["K"]), int(r["种子"]))
        arms = cells.setdefault(key, {})
        if str(r["臂"]) in arms:
            n_dup += 1
            continue
        arms[str(r["臂"])] = float(r["宏F1均值"])
    pairs = [
        {**dict(zip([*CELL_KEY, "种子"], key, strict=True)), "base": arms[ARM_NONE], "exposed": arms[ARM_EXP]}
        for key, arms in cells.items()
        if ARM_NONE in arms and ARM_EXP in arms
    ]
    return pairs, n_dup


def cv_reuse_pairs(logger: eu.Logger) -> list[dict[str, Any]]:
    """交叉验证重用的逐种子配对：base = 所选超参数在评估情节上的成绩，exposed = 选参情节上的最高分。

    苹果读 76 号逐种子表，公开数据集读 74 号 cnn 逐种子表里的路线面板；缺列或认不出选参分数时记日志并跳过。
    """
    pairs: list[dict[str, Any]] = []
    for f in formal_files(F76_GLOB):
        d = pd.read_excel(f, sheet_name=0)
        missing = [c for c in (C76_DESIGN, C76_TGT, C76_K, C76_METHOD, C76_TUNE, C76_EVAL) if c not in d.columns]
        if missing:
            logger.log(f"  [WARN] {os.path.basename(f)} 缺列 {missing}，这张表不取")
            continue
        for _, r in d.iterrows():
            pairs.append(
                {
                    "路径": CV_PATH,
                    "范围": "apple",
                    "数据集": f"apple_{r[C76_DESIGN]}",
                    "留出域": f"Y{int(r[C76_TGT])}",
                    "学习器": str(r[C76_METHOD]),
                    "K": int(r[C76_K]),
                    "种子": seed_of(f),
                    "base": float(r[C76_EVAL]),
                    "exposed": float(r[C76_TUNE]),
                }
            )
    for f in formal_files(F74_GLOB):
        d = read_sheet(f, S74_PANEL)
        missing = [c for c in (C74_DATASET, C74_HELD, C74_K, C74_ROUTE, C74_F1, C74_CHOSEN) if c not in d.columns]
        if missing:
            logger.log(f"  [WARN] {os.path.basename(f)} 没有路线面板或缺列 {missing}，这张表不取")
            continue
        n_bad = 0
        for _, r in d[d[C74_ROUTE].isin(list(ROUTES74))].iterrows():
            hit = TUNE_RE.search(str(r[C74_CHOSEN]))
            if hit is None:
                n_bad += 1
                continue
            pairs.append(
                {
                    "路径": CV_PATH,
                    "范围": "public",
                    "数据集": str(r[C74_DATASET]),
                    "留出域": str(r[C74_HELD]),
                    "学习器": ROUTES74[str(r[C74_ROUTE])],
                    "K": int(r[C74_K]),
                    "种子": seed_of(f),
                    "base": float(r[C74_F1]),
                    "exposed": float(hit.group(1)),
                }
            )
        if n_bad:
            logger.log(f"  [WARN] {os.path.basename(f)} 有 {n_bad} 行的「{C74_CHOSEN}」里认不出选参宏 F1，未取")
    return pairs


def cell_table(pairs: list[dict[str, Any]], base_name: str, exp_name: str, delta_name: str) -> pd.DataFrame:
    """逐（路径, 范围, 数据集, 留出域, 学习器, K）：两臂的种子均值，种子内配对差的均值、t 区间与为正种子数。"""
    by_cell: dict[tuple[Any, ...], dict[int, tuple[float, float]]] = {}
    for p in pairs:
        if np.isfinite(p["base"]) and np.isfinite(p["exposed"]):
            by_cell.setdefault(tuple(p[c] for c in CELL_KEY), {})[int(p["种子"])] = (p["base"], p["exposed"])
    rows: list[dict[str, Any]] = []
    for key, by_seed in by_cell.items():
        seeds = sorted(by_seed)
        deltas = [by_seed[s][1] - by_seed[s][0] for s in seeds]
        lo, hi = (float(v) for v in m74._t_interval(deltas))
        rows.append(
            {
                **dict(zip(CELL_KEY, key, strict=True)),
                "种子数": len(seeds),
                base_name: float(np.mean([by_seed[s][0] for s in seeds])),
                exp_name: float(np.mean([by_seed[s][1] for s in seeds])),
                delta_name: float(np.mean(deltas)),
                "种子t区间下限": lo,
                "种子t区间上限": hi,
                "为正种子数": int(sum(d > 0 for d in deltas)),
            }
        )
    return pd.DataFrame(rows)


def scope_table(cells: pd.DataFrame, delta_name: str) -> pd.DataFrame:
    """按口径、路径、学习器、K 跨留出域汇总：留出域级差值（种子均值）的均值、最小、最大与为正留出域数 / 总数。"""
    rows: list[dict[str, Any]] = []
    if cells.empty:
        return pd.DataFrame(rows)
    for scope, sets in SCOPES.items():
        sub = cells[cells["数据集"].isin(sets)]
        for (path, learner, k), g in sub.groupby(["路径", "学习器", "K"]):
            v = g[delta_name].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "口径": scope,
                    "路径": path,
                    "学习器": learner,
                    "K": int(k),
                    "数据集数": int(g["数据集"].nunique()),
                    "留出域数": len(v),
                    f"{delta_name}_均值": float(v.mean()),
                    f"{delta_name}_最小": float(v.min()),
                    f"{delta_name}_最大": float(v.max()),
                    "为正留出域": f"{int((v > 0).sum())}/{len(v)}",
                }
            )
    return pd.DataFrame(rows)


def aggregate(stage: str, logger: eu.Logger) -> None:
    """汇总本脚本的逐种子表，并从 76 号、74 号的逐种子表取交叉验证重用的对照。"""
    files = own_seed_files(stage)
    if not files:
        raise FileNotFoundError(f"没有 82 号的逐种子结果（{stage}）")
    logger.log(f"读入 {len(files)} 个逐种子表：" + "; ".join(os.path.basename(f) for f in files))
    long_df = pd.concat([pd.read_excel(f, sheet_name=0) for f in files], ignore_index=True)
    if long_df.empty:
        raise SystemExit("逐种子表里没有结果行")
    seeds = sorted({int(s) for s in long_df["种子"]})
    if stage == "formal" and set(seeds) != {int(s) for s in eu.RECOMMENDED_SEEDS}:
        logger.log(f"  [WARN] 种子集合 {seeds} 不是 Formal 5 种子，汇总仅作进度查看")
    pairs, n_dup = long_pairs(long_df)
    if n_dup:
        logger.log(f"  [WARN] {n_dup} 行在多张逐种子表里重复出现，取先读到的（同一种子全范围表在前）")
    cells = cell_table(pairs, ARM_NONE, ARM_EXP, DELTA)
    summ = scope_table(cells, DELTA)
    cv_pairs = cv_reuse_pairs(logger)
    cv_cells = cell_table(cv_pairs, CV_BASE, CV_EXP, CV_DELTA)
    cv_summ = scope_table(cv_cells, CV_DELTA)
    check, check_note = checked_against_65(long_df, logger)

    skip_frames = [d for d in (read_sheet(f, "跳过的格") for f in files) if not d.empty]
    skipped: dict[tuple[str, ...], set[int]] = {}
    for d in skip_frames:
        for _, r in d.iterrows():
            skipped.setdefault(tuple(str(r[c]) for c in SKIP_COLS if c != "种子"), set()).add(int(r["种子"]))
    skip_keys = [c for c in SKIP_COLS if c != "种子"]
    skip_df = pd.DataFrame(
        [{**dict(zip(skip_keys, key, strict=True)), "种子数": len(v)} for key, v in skipped.items()],
        columns=[*skip_keys, "种子数"],
    )
    comp_frames: list[pd.DataFrame] = []
    for f in files:
        comp = read_sheet(f, "compute")
        comp["文件"] = os.path.basename(f)
        comp_frames.append(comp)
    note = {"文件": "汇总", "与65号比对": check_note, "交叉验证重用的逐种子配对数": len(cv_pairs)}
    compute = pd.concat([*comp_frames, pd.DataFrame([note])], ignore_index=True)

    logger.log(f"\n表b 跨域汇总（{DELTA}）\n" + summ.to_string(index=False))
    logger.log(f"\n表d 交叉验证重用的跨域汇总（{CV_DELTA}）\n" + cv_summ.to_string(index=False))
    if not skip_df.empty:
        logger.log("\n表f 跳过的格\n" + skip_df.to_string(index=False))
    out = os.path.join(eu.OUTPUT_DIR, f"{STEM}{'' if stage == 'formal' else '__pilot'}.xlsx")
    write_workbook(
        out,
        {
            eu.table_sheet_name(0, "逐留出域对照"): cells,
            eu.table_sheet_name(1, "跨域汇总"): summ,
            eu.table_sheet_name(2, "交叉验证重用"): cv_cells,
            eu.table_sheet_name(3, "交叉验证重用汇总"): cv_summ,
            eu.table_sheet_name(4, "与65号比对"): check,
            eu.table_sheet_name(5, "跳过的格"): skip_df,
            eu.table_sheet_name(6, "长表"): long_df,
            "compute": compute,
        },
    )
    logger.log(f"汇总表 → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 六、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def pick(text: str, allowed: list[str], flag: str, parser: argparse.ArgumentParser) -> list[str]:
    """解析逗号分隔的取值：空串取全部，结果按 allowed 的次序；有不认识的取值即报错。"""
    wanted = [t.strip() for t in text.split(",") if t.strip()]
    unknown = sorted(set(wanted) - set(allowed))
    if unknown:
        parser.error(f"{flag} 里有不认识的取值 {unknown}；可选 {allowed}")
    return [a for a in allowed if a in wanted] if wanted else list(allowed)


def run_tag(scope: str, stage: str, cfg: dict[str, Any], datasets: list[str], in_scope: list[str]) -> str:
    """输出文件名里的标签段：全范围、缺省设置时为空串，只跑一侧时为 __apple / __public。

    数据集、路径、学习器、K、情节数、波段比例、源域池上限、潜变量数任一取了非缺省值时再加 __sub-…；
    带这一段的表不进汇总，也不会盖掉缺省设置的表。
    """
    sub: list[str] = []
    if datasets != in_scope:
        sub.append("ds-" + "+".join(datasets))
    if cfg["paths"] != list(PATHS):
        sub.append("p-" + "+".join(cfg["paths"]))
    if cfg["learners"] != list(LEARNERS):
        sub.append("l-" + "+".join(cfg["learners"]))
    if tuple(cfg["k_apple"]) != K_APPLE:
        sub.append("ka" + "-".join(str(k) for k in cfg["k_apple"]))
    if tuple(cfg["k_public"]) != K_PUBLIC:
        sub.append("kp" + "-".join(str(k) for k in cfg["k_public"]))
    if cfg["reps"] != REPS[stage]:
        sub.append(f"reps{cfg['reps']}")
    if cfg["band_frac"] != BAND_FRAC:
        sub.append(f"frac{cfg['band_frac']:g}")
    if cfg["src_per_class"] != SRC_PER_CLASS:
        sub.append(f"cap{cfg['src_per_class']}")
    if cfg["pls_ncomp"] != "cv":
        sub.append(f"nc{cfg['pls_ncomp']}")
    tag = "" if scope == "all" else f"__{scope}"
    return tag + (f"__sub-{'_'.join(sub)}" if sub else "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="82_chemometric_paths：化学计量学流程里的留出域暴露")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED, help="一次一个种子")
    parser.add_argument(
        "--stage", default=None, choices=["pilot", "formal"], help="缺省：运行时 pilot，--aggregate 时 formal"
    )
    parser.add_argument("--data_root", default="05data", help="公开数据集根目录")
    parser.add_argument("--scope", default="all", choices=["apple", "public", "all"])
    parser.add_argument("--datasets", default="", help="逗号分隔的数据集名；缺省为 --scope 内的全部")
    parser.add_argument("--paths", default=",".join(PATHS), help="逗号分隔：scale、select、select_half")
    parser.add_argument("--learners", default=",".join(LEARNERS), help="逗号分隔：svm、pls")
    parser.add_argument("--reps", type=int, default=None, help="每格情节数；缺省 formal 50、pilot 10")
    parser.add_argument("--k_apple", default=",".join(str(k) for k in K_APPLE), help="苹果的 K（逗号分隔）")
    parser.add_argument("--k_public", default=",".join(str(k) for k in K_PUBLIC), help="公开数据集的 K（逗号分隔）")
    parser.add_argument("--band_frac", type=float, default=BAND_FRAC, help="波长选择保留的波段比例")
    parser.add_argument("--src_per_class", type=int, default=SRC_PER_CLASS, help="源域池每类至多条数")
    parser.add_argument("--pls_ncomp", default="cv", help="cv = 源域池内部交叉验证选；正整数 = 固定潜变量数")
    parser.add_argument("--n_jobs", type=int, default=1, help="按留出域并行的进程数（joblib）")
    parser.add_argument("--aggregate", action="store_true", help="汇总已有的逐种子表，不做新的计算")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.aggregate:
        logger = eu.get_logger(STEM)
        aggregate(args.stage or "formal", logger)
        logger.close()
        return

    stage: str = args.stage or "pilot"
    all_sets = [*APPLE_SETS, *PUBLIC_SETS]
    in_scope = [d for d in all_sets if args.scope == "all" or (d in APPLE_SETS) == (args.scope == "apple")]
    datasets = pick(args.datasets, in_scope, "--datasets", parser)
    cfg: dict[str, Any] = {
        "seed": args.seed,
        "reps": REPS[stage] if args.reps is None else args.reps,
        "paths": pick(args.paths, list(PATHS), "--paths", parser),
        "learners": pick(args.learners, list(LEARNERS), "--learners", parser),
        "k_apple": sorted({int(k) for k in args.k_apple.split(",") if k.strip()}),
        "k_public": sorted({int(k) for k in args.k_public.split(",") if k.strip()}),
        "band_frac": args.band_frac,
        "src_per_class": args.src_per_class,
        "pls_ncomp": args.pls_ncomp,
    }
    if not 0 <= args.seed < 2**32:
        parser.error("--seed 须在 0 到 2^32−1 之间（74 号 kshot_episodes 用它播种全局 numpy 随机数）")
    if not 0.0 < args.band_frac <= 1.0 or args.src_per_class < 1 or cfg["reps"] < 1 or args.n_jobs == 0:
        parser.error("--band_frac 须在 (0, 1]；--src_per_class 与每格情节数须为正整数；--n_jobs 不能为 0")
    if min([*cfg["k_apple"], *cfg["k_public"], 1]) < 1 or not cfg["k_apple"] or not cfg["k_public"]:
        parser.error("--k_apple、--k_public 须是非空的正整数列表")
    if args.pls_ncomp != "cv" and not (args.pls_ncomp.isdigit() and int(args.pls_ncomp) >= 1):
        parser.error("--pls_ncomp 取 cv 或正整数")
    suffix = "" if stage == "formal" else "__pilot"
    tag = run_tag(args.scope, stage, cfg, datasets, in_scope)
    out = os.path.join(SEED_DIR, f"{STEM}__seed{args.seed}{tag}{suffix}.xlsx")

    logger = eu.get_logger(STEM)
    eu.log_experiment_header(
        logger,
        {
            "任务": "化学计量学流程里的留出域暴露：标度与波长选择（补充材料 S9.2）",
            "阶段": f"{stage}（纯 CPU，sklearn / numpy）",
            "种子": args.seed,
            "数据集": datasets,
            "路径": [PATHS[p] for p in cfg["paths"]],
            "学习器": [LEARNERS[ln] for ln in cfg["learners"]],
            "K（苹果 / 公开数据集）": f"{cfg['k_apple']} / {cfg['k_public']}",
            "每格情节数": cfg["reps"],
            "波段比例 / 源域池每类上限 / PLS 潜变量数": f"{args.band_frac} / {args.src_per_class} / {args.pls_ncomp}",
            "苹果两种设计的源年份": DESIGN_SRC,
            "并行进程数": args.n_jobs,
            "库版本": {p: metadata.version(p) for p in ("scikit-learn", "scipy", "pandas", "joblib")},
            "输出": out,
        },
    )
    if "svm" in cfg["learners"]:
        check_svm_settings()

    jobs: list[tuple[Callable[..., dict[str, Any]], tuple[Any, ...]]] = []
    apple_sets = [d for d in datasets if d in APPLE_SETS]
    if apple_sets:
        df = eu.load_all_elements_df(verify_sha256=True)
        year_data = build_year_region_data(df, eu.get_spectrum_columns(df))
        for dsname in apple_sets:
            for tgt in sorted(DESIGN_SRC[APPLE_SETS[dsname]]):
                jobs.append((run_apple_unit, (year_data, dsname, tgt, cfg)))
    for dsname in [d for d in datasets if d not in APPLE_SETS]:
        doms, is_reg = m74.DATASETS[dsname]["loader"](args.data_root, args.seed)
        logger.log(f"【{dsname}】域 {[f'{d}:{len(v[1])}' for d, v in doms.items()]}")
        for held_i, held in enumerate(doms):
            jobs.append((run_public_unit, (dsname, PUBLIC_SETS.index(dsname), doms, is_reg, held, held_i, cfg)))
    logger.log(f"共 {len(jobs)} 个留出域，按留出域并行（n_jobs={args.n_jobs}）")

    results: list[dict[str, Any]] = []
    with eu.track_compute("化学计量学流程里的暴露", logger, None) as budget:
        if args.n_jobs == 1:
            for fn, fargs in jobs:
                results.append(fn(*fargs))
                for line in results[-1]["log"]:
                    logger.log(line)
        else:
            # 数组直接随任务传给子进程（max_nbytes=None：不走内存映射的只读视图）
            pool = Parallel(n_jobs=args.n_jobs, max_nbytes=None, verbose=5)
            results = list(pool(delayed(fn)(*fargs) for fn, fargs in jobs))
            for res in results:
                for line in res["log"]:
                    logger.log(line)

    long_df = pd.DataFrame([r for res in results for r in res["rows"]])
    pair_df = pd.DataFrame([r for res in results for r in res["pairs"]])
    info_df = pd.DataFrame([res["info"] for res in results if res["info"] is not None])
    skip_df = pd.DataFrame([r for res in results for r in res["skips"]], columns=SKIP_COLS)
    logger.log(f"\n结果 {len(long_df)} 行（一行一格一臂），跳过 {len(skip_df)} 格")
    if not pair_df.empty:
        view = pair_df.groupby(["范围", "路径", "学习器", "K"])[DELTA].agg(["mean", "count"])
        logger.log(f"各范围按路径、学习器、K：{DELTA} 的留出域均值与留出域数\n" + view.to_string())
    check_df, check_note = checked_against_65(long_df, logger)
    setting = {"seed": args.seed, "stage": stage, "scope": args.scope, "n_jobs": args.n_jobs, "reps": cfg["reps"]}
    os.makedirs(SEED_DIR, exist_ok=True)
    write_workbook(
        out,
        {
            eu.table_sheet_name(0, "逐格结果"): long_df,
            eu.table_sheet_name(1, "逐格配对"): pair_df,
            eu.table_sheet_name(2, "各留出域的设置"): info_df,
            eu.table_sheet_name(3, "跳过的格"): skip_df,
            eu.table_sheet_name(4, "与65号比对"): check_df,
            "compute": pd.DataFrame([{**setting, "与65号比对": check_note, **budget}]),
        },
    )
    logger.log(f"结果表 → {out}")
    logger.close()


if __name__ == "__main__":
    main()
