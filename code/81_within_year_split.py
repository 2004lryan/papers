"""
81_within_year_split.py: 年内随机划分与跨年留出的同口径对照（同一批光谱、同一组学习器，只换划分方式）。

为什么:
    稿件 3.6 节把已发表的同批次 7:3 随机划分准确率（94.84%，黄华等 2022）与本稿跨年留出的宏 F1 并列，
    两边的数据、学习器、指标都不同，只能声明不可相减。本脚本让同一组学习器在同一批光谱上分别接受
    年内随机划分与跨年留出两种评估，两者之间只差划分方式。

数据与学习器（A、B 两臂共用）:
    数据     冻结的 229 波段数据（eu.load_all_elements_df，核对 SHA-256）。光谱按 20、71 号的同一路径
             读入：float32，再做一次逐谱 SNV。2018、2019 年每果一条谱；2025 年每果至多 5 个面
    学习器   最近质心（20 号的余弦判别，没有超参数）、SVM（RBF 核）、PLS-DA。后两者直接调 71 号的
             fit_predict：SVM 的逐波段标度、PLS 的中心化与标度都只在拟合用的那部分光谱上估计
    选参     在拟合用的那部分光谱内部做 5 折，各候选取 5 折宏 F1 的均值，取最高者，并列时取候选序列里
             靠前的一个（71 号 pick_best）。候选沿用 71 号：SVM 取 SVM_GRID 的 RBF 部分（C 8 个 ×
             核宽 5 个；--svm_grid full 时连同线性核 8 个），PLS-DA 取 PLS_GRID（潜变量数 1–24）
    指标     准确率与宏 F1。宏 F1 按被评估年份的产地集合取平均

各臂:
    A 年内随机划分  2018、2019、2025 各年份内按产地分层随机 7:3，重复 200 次（--stage pilot 时 20 次）。
                    2025 年做两种划分：「按光谱」（同一果的各面可以分处训练与测试两侧，是被检验的
                    做法）与「按苹果编号」（同一果的各面同进同出）；内部 5 折的分组方式与外层相同。
                    2018、2019 年每果一条谱，两种划分是同一回事，只做「按光谱」
    B 跨年对照      同一组学习器在源年份全部带标签光谱上拟合（内部 5 折按苹果编号分组、按年份 × 产地
                    分层），在留出年全部光谱上评估，不用留出年的任何样本。两年设计 2018↔2019 两个方向，
                    三年设计三折，源年份取 76 号的 DESIGN_SRC。没有重复抽样，只跑一次。
                    留出年有而源年份没有的产地（三年设计留出 2025 年时的甘肃）无从判对，照样计入；
                    表 b 另列只在两边共有的产地上算的一组值
    C 重建臂        （--recon）按黄华等 2022 的文字说明重建其流程，用 2019 年原始光谱（01 号
                    load_old_year，2048 个点）：Grünwald–Letnikov 分数阶微分，阶次 0–2、步长 0.1 共 21 个；
                    每个阶次做 PCA（16 个主成分）后接 SRDA，成一个基学习器；各基学习器在训练集上的
                    预测类别组成新特征，决策树融合；7:3 随机划分重复 200 次，报训练集与测试集准确率的
                    均值与标准差。另用同一流程做两个方向的跨年：2019 年全部样本训练、2018 年原始光谱
                    测试，以及反过来。PCA、SRDA、决策树都只在训练部分拟合；分数阶微分是逐谱的固定
                    线性滤波，不在样本之间拟合任何量

重建臂里原文未给、由本脚本取定的设置（取值随结果写进表 d）:
    --recon_alpha 1.0             SRDA 的正则系数。原文未给
    SRDA 的实现                   sklearn 没有 SRDA，按其回归形式实现：主成分得分（不另做标度）对类别指示
                                  矩阵做岭回归，取响应最大的类。原文未给 SRDA 之后的判别规则
    --recon_n_pc 16               各阶次的主成分数。原文只给出 0.6 阶示例取 16，其余阶次未给，一律取 16
    --recon_tree_criterion gini   决策树的分裂准则。原文未给决策树的任何设置，这三项取 sklearn 缺省值
    --recon_tree_max_depth 0      决策树的最大深度，0 = 不限。原文未给
    --recon_tree_min_leaf 1       决策树叶子的最少样本数。原文未给。树内部打乱特征次序用的
                                  random_state 由本脚本的随机数流给出
    --recon_split stratified      7:3 划分是否按产地分层。原文只说「随机划分」，未给；取分层，此时各次
                                  划分与 A 臂 2019 年「按光谱」的划分逐次相同。random = 不分层
    --recon_stack insample        融合所用的训练集预测是样本内的（基学习器在训练集上拟合后回代）。
                                  原文未给；cv = 改用训练集内部 --recon_stack_folds（缺省 5）折的样本外预测
    --recon_bin 1                 相邻几个采样点取均值。原文写 512 个波段，盘上的原始表是 2048 个点，
                                  两者的关系原文未给；1 = 不合并，4 = 合并成 512 个点
    分数阶微分的步长与端点        步长取 1 个采样点，左端不足的项按公式自然截断，不丢端点。原文未给
    暗光谱校正                    盘上的原始表是否已做暗光谱校正无从核对，按读入的数值直接使用
    两年原始表的波长网格          运行时核对；不同时把 2018 年光谱线性插值到 2019 年的网格，写进日志与表 d

随机数:
    一律 np.random.default_rng([seed, 81, 流编号, …])，不用全局随机数。
    流 1 = 年内 7:3 划分（年份, 划分方式, 划分编号）；流 2 = 该次划分训练部分的内部 5 折；
    流 3 = B 臂源年份的内部 5 折（设计, 留出年）；流 4 = 重建臂（不分层的划分、决策树的 random_state、
    样本外堆叠的折）。每次划分各有自己的流，结果与 --n_jobs 无关；同一种子下预跑的 20 次划分是
    正式 200 次的前 20 次

复用:
    export_utils   OUTPUT_DIR、AVAILABLE_YEARS、RECOMMENDED_SEEDS、get_logger（Logger）、
                   log_experiment_header、track_compute、table_sheet_name、load_all_elements_df、
                   get_spectrum_columns
    01preprocess   load_old_year、RAW_2018、RAW_2019
    20 号          snv
    70 号          按苹果编号分组、按产地分层的划分口径（该脚本用 StratifiedGroupKFold；这里每次划分
                   要有单独的随机数流，按同一口径自己实现）
    71 号          fit_predict、pick_best、on_edge、SVM_GRID、PLS_GRID
    76 号          DESIGN_SRC

运行方式:
    cd <项目根>
    OMP_NUM_THREADS=1 python 02code/81_within_year_split.py --stage pilot --n_jobs 8 --recon     # 预跑
    OMP_NUM_THREADS=1 python 02code/81_within_year_split.py --stage formal --n_jobs 32 --recon   # 正式
    # 只改重建臂的某项设置重跑（不带 --tag 的正式工作簿只留给缺省协议）:
    OMP_NUM_THREADS=1 python 02code/81_within_year_split.py --stage formal --arms none --recon \
        --recon_alpha 0.1 --tag alpha01

输入文件:
    03data/processed/all_9elements.csv 及同目录的数据卡（读入时核对 SHA-256；A、B 两臂的数据，
        C 臂只用它核对各次划分是否与 A 臂相同）
    03data/ 顶层的 2018、2019 年原始光谱表（只有 --recon 用；路径取 01 号的 RAW_2018、RAW_2019）

输出文件:
    04outputs/81_within_year_split.xlsx（--stage pilot 时文件名加 __pilot；给了 --tag 时先加 __<tag>）
        表a 年内划分汇总：年份 × 划分方式 × 学习器的准确率与宏 F1（均值、标准差、次数）
        表b 跨年对照：设计 × 留出年 × 学习器
        表c 逐次划分的长表（不舍入；重建臂开启时其逐次结果也在里面）
        表d 重建臂（--recon 时才有）
        compute 各臂的墙钟与本次设置
    05logs/81_within_year_split_YY-MM-DD_HHMMSS.log

说明:
    全部是 CPU 上的 sklearn / numpy 计算，不训练深度模型；--stage 只决定划分次数与文件名后缀。
    标准差是各次划分之间的样本标准差（ddof=1）。各表数值都不舍入。只写上面这一个工作簿。
    执行次序是 B、C、A：费时短的两臂先跑，最费时的 A 臂在后；工作簿在全部跑完后一次写出。
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

import export_utils as eu

NDArray = npt.NDArray[Any]
m01 = importlib.import_module("01preprocess")
m20 = importlib.import_module("20_unified_singleface_kshot")
m71 = importlib.import_module("71_source_only_tuning")
m76 = importlib.import_module("76_target_tuning_leakage")

STEM = "81_within_year_split"
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
FORMAL_SPLITS, PILOT_SPLITS = 200, 20
TEST_FRAC = 0.3  # 7:3 划分里测试部分所占比例
INNER_FOLDS = 5  # 选超参数用的内部折数
LEARNERS: dict[str, str] = {"nc": "最近质心", "svm": "SVM", "pls": "PLS-DA"}
SVM_GRIDS: dict[str, tuple[Any, ...]] = {
    "rbf": tuple(p for p in m71.SVM_GRID if p[0] == "rbf"),
    "full": tuple(m71.SVM_GRID),
}
BY_SPECTRUM, BY_APPLE = "按光谱", "按苹果编号"
MODE_CODE = {BY_SPECTRUM: 0, BY_APPLE: 1}
DESIGN_CODE = {"two_year": 2, "three_year": 3}
# 随机数流编号：np.random.default_rng([seed, 81, 流编号, …])
STREAM_OUTER, STREAM_INNER, STREAM_CROSS, STREAM_RECON = 1, 2, 3, 4

# 重建臂（黄华等 2022）
RECON_YEAR, RECON_OTHER = 2019, 2018
FD_ORDERS = tuple(j / 10 for j in range(21))  # 阶次 0–2，步长 0.1
FUSED = "FD-PCA-SRDA 融合"
WITHIN_TAG, CROSS_TAG = "年内 7:3", "跨年"
# 该文报告的 200 次 7:3 划分结果（FD-PCA-SRDA 多模型融合）
PUBLISHED = {
    "训练集准确率均值": 0.9733,
    "训练集准确率SD": 0.0049,
    "测试集准确率均值": 0.9484,
    "测试集准确率SD": 0.0148,
}
WL_TOL = 1e-3  # 两年原始表的波长网格视为相同的容差（nm）

# 不带 --tag 的正式工作簿只留给缺省协议：formal 阶段下列参数取了非缺省值时必须同时给 --tag
# （划分次数、年份、臂三项在 main 里按解析后的取值另行比对）
GUARDED = ("seed", "svm_grid")
GUARDED_RECON = (
    "recon_alpha",
    "recon_n_pc",
    "recon_tree_criterion",
    "recon_tree_max_depth",
    "recon_tree_min_leaf",
    "recon_split",
    "recon_stack",
    "recon_stack_folds",
    "recon_bin",
)
LONG_COLS = (
    "臂",
    "年份",
    "划分方式",
    "划分编号",
    "学习器",
    "所选超参数",
    "内部5折宏F1",
    "训练光谱数",
    "测试光谱数",
    "准确率",
    "宏F1",
    "训练集准确率",
    "耗时s",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 一、划分、指标与学习器（并行子进程里也要用，不引用 eu 与各编号脚本的模块对象）
# ═══════════════════════════════════════════════════════════════════════════════
def tuning_lib() -> Any:
    """71 号脚本的模块对象。在调用处引入并先把脚本目录放进 sys.path，不依赖并行后端是否把它带给子进程。"""
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)
    return importlib.import_module("71_source_only_tuning")


def stratified_split(strata: NDArray, units: NDArray, test_frac: float, rng: np.random.Generator) -> NDArray:
    """按层随机取 test_frac 的划分单位作测试部分，返回逐行的布尔掩码（True = 测试）。

    units 给出每行所属的划分单位：按光谱划分时一行一个单位；按苹果编号划分时同一果的各行是一个单位，
    同进同出。每层的测试单位数取「层内单位数 × test_frac」四舍五入，至少 1 个，至少留 1 个给训练。
    """
    u_ids, first = np.unique(units, return_index=True)
    u_strata = strata[first]
    picked: list[NDArray] = []
    for s in np.unique(u_strata):
        ids = u_ids[u_strata == s]
        if len(ids) < 2:
            raise ValueError(f"层 {s} 只有 {len(ids)} 个划分单位，分不出训练与测试两部分")
        n_test = min(max(int(np.floor(len(ids) * test_frac + 0.5)), 1), len(ids) - 1)
        picked.append(ids[rng.permutation(len(ids))[:n_test]])
    return np.asarray(np.isin(units, np.concatenate(picked)))


def stratified_folds(strata: NDArray, units: NDArray, n_folds: int, rng: np.random.Generator) -> NDArray:
    """按层把划分单位随机分进 n_folds 折，返回逐行的折号。

    每层内随机排列后轮流发到各折，层与层之间接着发，各折的单位数至多差 1；同一单位的各行同折。
    """
    u_ids, first, inverse = np.unique(units, return_index=True, return_inverse=True)
    u_strata = strata[first]
    fold_of_unit = np.empty(len(u_ids), dtype=np.int64)
    dealt = 0
    for s in np.unique(u_strata):
        pos = np.flatnonzero(u_strata == s)
        if len(pos) < n_folds:
            raise ValueError(f"层 {s} 只有 {len(pos)} 个划分单位，不够分 {n_folds} 折")
        fold_of_unit[pos[rng.permutation(len(pos))]] = (dealt + np.arange(len(pos))) % n_folds
        dealt += len(pos)
    return np.asarray(fold_of_unit[inverse])


def acc_f1(y_true: NDArray, y_pred: NDArray, labels: list[Any]) -> tuple[float, float]:
    """准确率与宏 F1。宏 F1 按 labels 给定的类别集合取平均；预测成集合外类别的样本只记作其真实类别的漏判。"""
    if len(y_true) == 0:
        return float("nan"), float("nan")
    acc = float(np.mean(y_true == y_pred))
    return acc, float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def nc_predict(x_tr: NDArray, y_tr: NDArray, x_q: NDArray, n_cls: int) -> NDArray:
    """最近质心，与 20 号 eval_nearest_centroid 同一判别。

    各类训练光谱的均值与待测光谱各自 L2 归一化，取内积最大的类。
    """
    cents = np.stack([x_tr[y_tr == c].mean(axis=0) for c in range(n_cls)])
    cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)
    xq = x_q / (np.linalg.norm(x_q, axis=1, keepdims=True) + 1e-8)
    return np.asarray((xq @ cents.T).argmax(axis=1))


def cv_cell(kind: str, param: Any, fold: int, x: NDArray, y: NDArray, fold_id: NDArray, regions: list[str]) -> float:
    """一个候选在一折上的宏 F1：在其余各折上拟合（标度与中心化也只在这部分上估计），在该折上预测。"""
    va = fold_id == fold
    pred = tuning_lib().fit_predict(kind, param, regions, x[~va], y[~va], x[va])
    code = {r: i for i, r in enumerate(regions)}
    y_hat = np.array([code[p] for p in pred])
    return acc_f1(y[va], y_hat, list(range(len(regions))))[1]


def select_param(
    kind: str, grid: tuple[Any, ...], x: NDArray, y: NDArray, fold_id: NDArray, regions: list[str], n_jobs: int
) -> tuple[Any, float]:
    """内部 K 折选超参数：各候选取各折宏 F1 的均值，取最高者；并列取候选序列里靠前的一个（71 号 pick_best）。"""
    n_folds = int(fold_id.max()) + 1
    cells = [(p, f) for p in grid for f in range(n_folds)]
    vals = Parallel(n_jobs=n_jobs)(delayed(cv_cell)(kind, p, f, x, y, fold_id, regions) for p, f in cells)
    tune = {p: float(np.mean(vals[i * n_folds : (i + 1) * n_folds])) for i, p in enumerate(grid)}
    best = tuning_lib().pick_best(grid, tune)
    return best, tune[best]


def fit_learner(
    name: str,
    x_tr: NDArray,
    y_tr: NDArray,
    x_q: NDArray,
    fold_id: NDArray,
    regions: list[str],
    grids: dict[str, tuple[Any, ...]],
    n_jobs: int,
) -> tuple[NDArray, str, float]:
    """在 (x_tr, y_tr) 上拟合一个学习器并预测 x_q，返回 (预测类别码, 所选超参数, 内部折宏 F1)。

    SVM 与 PLS-DA 先按 fold_id 给出的内部折选超参数，再在全部 x_tr 上拟合；最近质心没有超参数，后两项为空。
    """
    if name == "nc":
        return nc_predict(x_tr, y_tr, x_q, len(regions)), "", float("nan")
    best, cv = select_param(name, grids[name], x_tr, y_tr, fold_id, regions, n_jobs)
    pred = tuning_lib().fit_predict(name, best, regions, x_tr, y_tr, x_q)
    code = {r: i for i, r in enumerate(regions)}
    return np.array([code[p] for p in pred]), str(best), cv


# ═══════════════════════════════════════════════════════════════════════════════
# 二、A 年内随机划分与 B 跨年对照
# ═══════════════════════════════════════════════════════════════════════════════
def within_split(
    year: int, mode: str, r: int, x: NDArray, y: NDArray, units: NDArray, regions: list[str], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """A 臂的一次年内 7:3 划分：训练部分内部选超参数、拟合，在测试部分上算准确率与宏 F1。"""
    tail = [year, MODE_CODE[mode], r]
    test = stratified_split(y, units, TEST_FRAC, np.random.default_rng([cfg["seed"], 81, STREAM_OUTER, *tail]))
    train = ~test
    rng_inner = np.random.default_rng([cfg["seed"], 81, STREAM_INNER, *tail])
    fold_id = stratified_folds(y[train], units[train], INNER_FOLDS, rng_inner)
    labels = list(range(len(regions)))
    rows: list[dict[str, Any]] = []
    for name in LEARNERS:
        t0 = time.perf_counter()
        pred, param, cv = fit_learner(name, x[train], y[train], x[test], fold_id, regions, cfg["grids"], 1)
        acc, f1 = acc_f1(y[test], pred, labels)
        rows.append(
            {
                "臂": "A",
                "年份": year,
                "划分方式": mode,
                "划分编号": r,
                "学习器": LEARNERS[name],
                "所选超参数": param,
                "内部5折宏F1": cv,
                "训练光谱数": int(train.sum()),
                "测试光谱数": int(test.sum()),
                "准确率": acc,
                "宏F1": f1,
                "耗时s": time.perf_counter() - t0,
            }
        )
    return rows


def run_within(
    data: dict[str, NDArray], years: list[int], n_splits: int, cfg: dict[str, Any], n_jobs: int, logger: eu.Logger
) -> tuple[list[dict[str, Any]], dict[int, tuple[int, int]]]:
    """A 臂：各年份内分层随机 7:3 划分 n_splits 次；每果不止一条谱的年份另做按苹果编号分组的划分。

    返回 (逐次划分的记录, {年份: (光谱数, 苹果数)})。各次划分互不依赖，按 n_jobs 并行。
    """
    rows: list[dict[str, Any]] = []
    sizes: dict[int, tuple[int, int]] = {}
    for year in years:
        sel = data["year"] == year
        x = data["x"][sel]
        names, y = np.unique(data["region"][sel], return_inverse=True)
        regions = [str(v) for v in names]
        apple = np.unique(data["apple"][sel], return_inverse=True)[1]
        n_apple = int(apple.max()) + 1
        sizes[year] = (len(y), n_apple)
        modes = [BY_SPECTRUM, BY_APPLE] if n_apple < len(y) else [BY_SPECTRUM]
        for mode in modes:
            units = apple if mode == BY_APPLE else np.arange(len(y))
            t0 = time.perf_counter()
            parts = Parallel(n_jobs=n_jobs)(
                delayed(within_split)(year, mode, r, x, y, units, regions, cfg) for r in range(n_splits)
            )
            rows += [row for part in parts for row in part]
            wall = time.perf_counter() - t0
            logger.log(
                f"  A {year} {mode}：{len(y)} 条谱、{n_apple} 果、产地 {regions}，{n_splits} 次划分，{wall:.0f} s"
            )
    return rows, sizes


def run_cross(
    data: dict[str, NDArray], years: list[int], cfg: dict[str, Any], n_jobs: int, logger: eu.Logger
) -> list[dict[str, Any]]:
    """B 臂：源年份全部带标签光谱上拟合（超参数在源年份内部选），留出年全部光谱上评估，不用留出年的样本。

    内部折按苹果编号分组、按年份 × 产地分层；候选 × 折的各次拟合按 n_jobs 并行。只做留出年在 years 里的折。
    """
    rows: list[dict[str, Any]] = []
    for design in ("two_year", "three_year"):
        for tgt, src_years in sorted(m76.DESIGN_SRC[design].items()):
            if tgt not in years:
                continue
            src = np.isin(data["year"], src_years)
            held = data["year"] == tgt
            names, y_src = np.unique(data["region"][src], return_inverse=True)
            regions = [str(v) for v in names]
            strata = np.unique(data["element"][src], return_inverse=True)[1]
            units = np.unique(data["apple"][src], return_inverse=True)[1]
            rng = np.random.default_rng([cfg["seed"], 81, STREAM_CROSS, DESIGN_CODE[design], tgt])
            fold_id = stratified_folds(strata, units, INNER_FOLDS, rng)
            y_true = data["region"][held]
            tgt_regions = sorted({str(v) for v in y_true})
            shared = [c for c in tgt_regions if c in regions]
            in_shared = np.isin(y_true, shared)
            src_txt = "+".join(str(v) for v in src_years)
            for name in LEARNERS:
                t0 = time.perf_counter()
                code, param, cv = fit_learner(
                    name, data["x"][src], y_src, data["x"][held], fold_id, regions, cfg["grids"], n_jobs
                )
                y_pred = np.asarray(regions)[code]
                acc, f1 = acc_f1(y_true, y_pred, tgt_regions)
                acc_s, f1_s = acc_f1(y_true[in_shared], y_pred[in_shared], shared)
                rows.append(
                    {
                        "设计": design,
                        "留出年": tgt,
                        "源年份": src_txt,
                        "学习器": LEARNERS[name],
                        "所选超参数": param,
                        "源年份内部5折宏F1": cv,
                        "落在网格外沿": bool(m71.on_edge(param)) if param else None,
                        "训练光谱数": int(src.sum()),
                        "评估光谱数": int(held.sum()),
                        "准确率": acc,
                        "宏F1": f1,
                        "源年份产地": ",".join(regions),
                        "留出年产地": ",".join(tgt_regions),
                        "共有产地评估光谱数": int(in_shared.sum()),
                        "共有产地准确率": acc_s,
                        "共有产地宏F1": f1_s,
                        "耗时s": time.perf_counter() - t0,
                    }
                )
                logger.log(
                    f"  B {design} 留出 {tgt}（源 {src_txt}）{LEARNERS[name]}：准确率 {acc:.4f}、宏 F1 {f1:.4f}；"
                    f"共有产地上 {acc_s:.4f}、{f1_s:.4f}；所选 {param or '无'}"
                )
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 三、C 重建臂：分数阶微分 → 逐阶次 PCA-SRDA 基学习器 → 决策树融合
# ═══════════════════════════════════════════════════════════════════════════════
def gl_weights(order: float, n: int) -> NDArray:
    """Grünwald–Letnikov 权重：w_0 = 1，w_k = w_{k−1}·(1 − (order + 1)/k)，即 (−1)^k·C(order, k)。"""
    steps = 1.0 - (order + 1.0) / np.arange(1, n)
    return np.asarray(np.cumprod(np.concatenate(([1.0], steps))))


def fd_stack(x: NDArray, orders: tuple[float, ...]) -> NDArray:
    """各阶次的分数阶微分谱，形状 (阶次数, 样本数, 点数)。步长取 1 个采样点，第 i 点对第 0…i 点加权求和。

    这是逐条光谱的固定线性滤波，不在样本之间拟合任何量：对全体光谱一次算完，与训练、测试分开各算一遍相同。
    """
    n = x.shape[1]
    lag = np.arange(n)[:, None] - np.arange(n)[None, :]
    causal, back = lag >= 0, np.clip(lag, 0, None)
    out = np.empty((len(orders), *x.shape))
    for j, order in enumerate(orders):
        op = np.where(causal, gl_weights(order, n)[back], 0.0)  # op[i, m] = w[i − m]（m ≤ i），其余为 0
        out[j] = x @ op.T
    return out


def check_fd(x: NDArray, fd: NDArray) -> None:
    """分数阶微分的自检：0 阶即原谱；1 阶、2 阶在内部点上等于一阶、二阶后向差分。不符即中止。"""
    i0, i1, i2 = FD_ORDERS.index(0.0), FD_ORDERS.index(1.0), FD_ORDERS.index(2.0)
    ok = (
        np.allclose(fd[i0], x)
        and np.allclose(fd[i1][:, 1:], np.diff(x, axis=1))
        and np.allclose(fd[i2][:, 2:], np.diff(x, n=2, axis=1))
    )
    if not ok:
        raise RuntimeError("分数阶微分自检未过：整数阶次的结果与差分不一致")


def pca_srda(
    x_fit: NDArray, y_fit: NDArray, x_new: list[NDArray], n_cls: int, n_pc: int, alpha: float
) -> list[NDArray]:
    """PCA 后接 SRDA 的一个基学习器：两步都只在 x_fit 上拟合，返回 x_new 里各矩阵的预测类别码。

    SRDA 按回归形式实现：主成分得分（不另做标度）对类别指示矩阵做岭回归（正则系数 alpha），取响应最大的类。
    PCA 用完整的奇异值分解（svd_solver="full"），不走随机化解法。
    """
    pca = PCA(n_components=min(n_pc, len(x_fit) - 1, x_fit.shape[1]), svd_solver="full").fit(x_fit)
    onehot = np.zeros((len(x_fit), n_cls))
    onehot[np.arange(len(x_fit)), y_fit] = 1.0
    reg = Ridge(alpha=alpha, solver="cholesky").fit(pca.transform(x_fit), onehot)
    return [np.asarray(reg.predict(pca.transform(m)).argmax(axis=1)) for m in x_new]


def recon_fit(
    fd_tr: NDArray,
    rows_tr: NDArray | None,
    y_tr: NDArray,
    fd_te: NDArray,
    rows_te: NDArray | None,
    n_cls: int,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, NDArray]:
    """一次完整的重建流程：逐阶次的 PCA-SRDA 基学习器 → 预测类别作新特征 → 决策树融合。

    fd_tr、fd_te 是 (阶次, 样本, 点) 的分数阶微分谱，rows_tr、rows_te 给出各自所用的行（None = 全部）。
    返回训练、测试两部分上各基学习器与融合模型的预测类别码；训练部分的预测都由拟合好的基学习器回代得到。
    rng 先给出决策树的 random_state，再（样本外堆叠时）给出训练部分的内部折。
    """
    tree_seed = int(rng.integers(0, 2**31 - 1))
    n_tr, n_ord = len(y_tr), fd_tr.shape[0]
    oof = stratified_folds(y_tr, np.arange(n_tr), cfg["stack_folds"], rng) if cfg["stack"] == "cv" else None
    base_tr = np.empty((n_tr, n_ord), dtype=np.int64)
    base_te = np.empty((fd_te.shape[1] if rows_te is None else len(rows_te), n_ord), dtype=np.int64)
    meta = np.empty_like(base_tr)
    for j in range(n_ord):
        a_tr = fd_tr[j] if rows_tr is None else fd_tr[j][rows_tr]
        a_te = fd_te[j] if rows_te is None else fd_te[j][rows_te]
        base_tr[:, j], base_te[:, j] = pca_srda(a_tr, y_tr, [a_tr, a_te], n_cls, cfg["n_pc"], cfg["alpha"])
        meta[:, j] = base_tr[:, j]
        if oof is not None:  # 融合所用的训练集特征改成内部各折的样本外预测
            for f in range(cfg["stack_folds"]):
                va = oof == f
                meta[va, j] = pca_srda(a_tr[~va], y_tr[~va], [a_tr[va]], n_cls, cfg["n_pc"], cfg["alpha"])[0]
    tree = DecisionTreeClassifier(
        criterion=cfg["tree_criterion"],
        max_depth=cfg["tree_max_depth"],
        min_samples_leaf=cfg["tree_min_leaf"],
        random_state=tree_seed,
    ).fit(meta, y_tr)
    return {
        "base_tr": base_tr,
        "base_te": base_te,
        "fused_tr": np.asarray(tree.predict(base_tr)),
        "fused_te": np.asarray(tree.predict(base_te)),
    }


def recon_rows(
    pred: dict[str, NDArray], y_tr: NDArray, y_te: NDArray, n_cls: int, head: dict[str, Any]
) -> list[dict[str, Any]]:
    """把一次重建流程的预测换成逐模型的记录：融合模型一条（排第一），各阶次的基学习器各一条。"""
    labels = list(range(n_cls))
    models = [
        (FUSED, pred["fused_tr"], pred["fused_te"]),
        *((f"基学习器 FD{o:.1f}", pred["base_tr"][:, j], pred["base_te"][:, j]) for j, o in enumerate(FD_ORDERS)),
    ]
    rows: list[dict[str, Any]] = []
    for name, p_tr, p_te in models:
        acc, f1 = acc_f1(y_te, p_te, labels)
        acc_tr = acc_f1(y_tr, p_tr, labels)[0]
        rows.append({**head, "学习器": name, "准确率": acc, "宏F1": f1, "训练集准确率": acc_tr})
    return rows


def recon_split(
    r: int, fd: NDArray, y: NDArray, strata: NDArray, n_cls: int, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """重建臂的一次年内 7:3 划分。分层时与 A 臂同年份「按光谱」的第 r 次划分用同一个随机数流。"""
    every = np.arange(len(y))
    if cfg["split"] == "stratified":
        rng = np.random.default_rng([cfg["seed"], 81, STREAM_OUTER, cfg["year"], MODE_CODE[BY_SPECTRUM], r])
        test = stratified_split(strata, every, TEST_FRAC, rng)
    else:
        rng = np.random.default_rng([cfg["seed"], 81, STREAM_RECON, 1, cfg["year"], r])
        test = stratified_split(np.zeros(len(y), dtype=np.int64), every, TEST_FRAC, rng)
    tr, te = np.flatnonzero(~test), np.flatnonzero(test)
    t0 = time.perf_counter()
    rng_fit = np.random.default_rng([cfg["seed"], 81, STREAM_RECON, 2, cfg["year"], r])
    pred = recon_fit(fd, tr, y[tr], fd, te, n_cls, cfg, rng_fit)
    head: dict[str, Any] = {
        "臂": "C",
        "年份": cfg["year"],
        "划分方式": cfg["split_name"],
        "划分编号": r,
        "所选超参数": "",
        "训练光谱数": len(tr),
        "测试光谱数": len(te),
    }
    rows = recon_rows(pred, y[tr], y[te], n_cls, head)
    rows[0]["耗时s"] = time.perf_counter() - t0  # 整个流程的耗时记在融合模型那一条上
    return rows


def load_raw(year: int, width: int, logger: eu.Logger) -> tuple[NDArray, NDArray, NDArray]:
    """读一年的原始光谱表（01 号 load_old_year），返回 (光谱, 波长, 产地)；width > 1 时相邻 width 个点取均值。"""
    path = {2018: m01.RAW_2018, 2019: m01.RAW_2019}[year]
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺 {year} 年原始光谱表（重建臂要用）：{path}")
    x_raw, wl_raw, origins, _ = m01.load_old_year(path, str(year), logger)
    x = np.asarray(x_raw, dtype=np.float64)
    wl = np.asarray(wl_raw, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"{year} 年原始光谱里有缺失或非有限值")
    if width > 1:
        keep = (x.shape[1] // width) * width
        x = np.asarray(x[:, :keep].reshape(len(x), -1, width).mean(axis=2))
        wl = np.asarray(wl[:keep].reshape(-1, width).mean(axis=1))
    return x, wl, np.asarray(origins, dtype=str)


def recon_table(within: pd.DataFrame, cross: list[dict[str, Any]], meta: dict[str, Any]) -> pd.DataFrame:
    """表 d：年内 7:3 的均值与标准差（融合模型与各阶次基学习器）、两个方向的跨年，连同本次所取的各项设置。"""
    rows: list[dict[str, Any]] = []
    for name, sub in within.groupby("学习器", sort=False):
        rows.append(
            {
                "场景": WITHIN_TAG,
                "训练年份": RECON_YEAR,
                "测试年份": RECON_YEAR,
                "学习器": name,
                "次数": len(sub),
                "训练集准确率均值": float(sub["训练集准确率"].mean()),
                "训练集准确率SD": float(sub["训练集准确率"].std(ddof=1)),
                "测试集准确率均值": float(sub["准确率"].mean()),
                "测试集准确率SD": float(sub["准确率"].std(ddof=1)),
                "测试集宏F1均值": float(sub["宏F1"].mean()),
                "测试集宏F1SD": float(sub["宏F1"].std(ddof=1)),
            }
        )
    out = pd.DataFrame([*rows, *cross])
    ref = (out["场景"] == WITHIN_TAG) & (out["学习器"] == FUSED)  # 已发表值只对应年内 7:3 的融合模型
    for key, val in PUBLISHED.items():
        out[f"已发表{key}"] = np.where(ref, val, np.nan)
    out["测试集准确率均值减已发表值"] = out["测试集准确率均值"] - out["已发表测试集准确率均值"]
    for key, val in meta.items():
        out[key] = val
    return out


def run_recon(
    args: argparse.Namespace, n_splits: int, data: dict[str, NDArray], logger: eu.Logger
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """C 臂：2019 年原始光谱上年内 7:3 划分 n_splits 次，另做 2019→2018、2018→2019 两个方向的跨年。

    返回 (逐次划分的记录, 表 d)。
    """
    x19, wl19, lab19 = load_raw(RECON_YEAR, args.recon_bin, logger)
    x18, wl18, lab18 = load_raw(RECON_OTHER, args.recon_bin, logger)
    same_grid = len(wl18) == len(wl19) and bool(np.allclose(wl18, wl19, atol=WL_TOL))
    if not same_grid:
        x18 = np.stack([np.interp(wl19, wl18, row) for row in x18])
        logger.log(f"  [注意] 两年原始表的波长网格不同，{RECON_OTHER} 年光谱已线性插值到 {RECON_YEAR} 年的网格")
    classes = sorted(set(lab19.tolist()) | set(lab18.tolist()))
    y19, y18 = np.searchsorted(classes, lab19), np.searchsorted(classes, lab18)
    strata19 = np.unique(lab19, return_inverse=True)[1]
    frozen19 = data["region"][data["year"] == RECON_YEAR]
    stratified = args.recon_split == "stratified"
    paired = stratified and len(frozen19) == len(lab19) and bool(np.all(frozen19 == lab19))
    cfg: dict[str, Any] = {
        "seed": args.seed,
        "year": RECON_YEAR,
        "split": args.recon_split,
        "split_name": BY_SPECTRUM if stratified else "按光谱（不分层）",
        "n_pc": args.recon_n_pc,
        "alpha": args.recon_alpha,
        "tree_criterion": args.recon_tree_criterion,
        "tree_max_depth": args.recon_tree_max_depth or None,
        "tree_min_leaf": args.recon_tree_min_leaf,
        "stack": args.recon_stack,
        "stack_folds": args.recon_stack_folds,
    }
    logger.log(
        f"  C {RECON_YEAR} 年原始光谱 {x19.shape[0]} 条 × {x19.shape[1]} 点，产地 {classes}；"
        f"各次划分与 A 臂 {RECON_YEAR} 年「按光谱」的划分相同：{paired}"
    )

    t0 = time.perf_counter()
    fd19 = fd_stack(x19, FD_ORDERS)
    check_fd(x19, fd19)
    parts = Parallel(n_jobs=args.n_jobs)(
        delayed(recon_split)(r, fd19, y19, strata19, len(classes), cfg) for r in range(n_splits)
    )
    long_rows = [row for part in parts for row in part]
    logger.log(f"  C 年内 7:3：{n_splits} 次划分，{time.perf_counter() - t0:.0f} s")

    fd18 = fd_stack(x18, FD_ORDERS)
    check_fd(x18, fd18)
    sets = {RECON_YEAR: (fd19, y19), RECON_OTHER: (fd18, y18)}
    cross: list[dict[str, Any]] = []
    for fit_year, test_year in ((RECON_YEAR, RECON_OTHER), (RECON_OTHER, RECON_YEAR)):
        (fd_a, y_a), (fd_b, y_b) = sets[fit_year], sets[test_year]
        rng = np.random.default_rng([args.seed, 81, STREAM_RECON, 3, fit_year])
        pred = recon_fit(fd_a, None, y_a, fd_b, None, len(classes), cfg, rng)
        for row in recon_rows(pred, y_a, y_b, len(classes), {}):
            cross.append(
                {
                    "场景": CROSS_TAG,
                    "训练年份": fit_year,
                    "测试年份": test_year,
                    "学习器": row["学习器"],
                    "次数": 1,
                    "训练集准确率均值": row["训练集准确率"],
                    "测试集准确率均值": row["准确率"],
                    "测试集宏F1均值": row["宏F1"],
                }
            )
    tree_txt = (
        f"criterion={cfg['tree_criterion']}, max_depth={cfg['tree_max_depth']}, min_samples_leaf={cfg['tree_min_leaf']}"
    )
    stack_txt = "样本内" if args.recon_stack == "insample" else f"内部 {args.recon_stack_folds} 折样本外"
    meta = {
        "SRDA正则系数": args.recon_alpha,
        "主成分数": args.recon_n_pc,
        "决策树": tree_txt,
        "划分": "按产地分层" if stratified else "不分层",
        "融合所用训练集预测": stack_txt,
        "合并采样点数": args.recon_bin,
        "光谱点数": int(x19.shape[1]),
        "两年波长网格相同": same_grid,
        "各次划分与A臂2019年按光谱相同": paired,
    }
    return long_rows, recon_table(pd.DataFrame(long_rows), cross, meta)


# ═══════════════════════════════════════════════════════════════════════════════
# 四、数据、汇总与主流程
# ═══════════════════════════════════════════════════════════════════════════════
def load_frozen(logger: eu.Logger) -> dict[str, NDArray]:
    """冻结的 229 波段数据：光谱（float32，再做一次逐谱 SNV，同 20、71 号）与年份、产地、苹果编号、元素编号。"""
    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    owners = df.groupby("苹果编号")[["年份", "产地"]].nunique()
    if int(owners.to_numpy().max()) != 1:
        raise ValueError("同一个苹果编号出现在不止一个年份或产地，按苹果编号分组的划分不成立")
    data: dict[str, NDArray] = {
        "x": np.asarray(m20.snv(df[wl_cols].to_numpy().astype(np.float32))),
        "year": df["年份"].to_numpy(dtype=np.int64),
        "region": df["产地"].to_numpy(dtype=str),
        "apple": df["苹果编号"].to_numpy(dtype=str),
        "element": df["element_id"].to_numpy(dtype=str),
    }
    logger.log(f"冻结数据：{data['x'].shape[0]} 条谱 × {data['x'].shape[1]} 波段，{len(owners)} 个苹果")
    return data


def summarise_within(long_a: pd.DataFrame, sizes: dict[int, tuple[int, int]]) -> pd.DataFrame:
    """表 a：年份 × 划分方式 × 学习器的准确率与宏 F1（各次划分的均值、样本标准差、次数）。"""
    rows: list[dict[str, Any]] = []
    for (year, mode, name), sub in long_a.groupby(["年份", "划分方式", "学习器"], sort=False):
        picked = sub["所选超参数"]
        rows.append(
            {
                "年份": int(year),
                "划分方式": mode,
                "学习器": name,
                "光谱数": sizes[int(year)][0],
                "苹果数": sizes[int(year)][1],
                "次数": len(sub),
                "准确率均值": float(sub["准确率"].mean()),
                "准确率SD": float(sub["准确率"].std(ddof=1)),
                "宏F1均值": float(sub["宏F1"].mean()),
                "宏F1SD": float(sub["宏F1"].std(ddof=1)),
                "落在网格外沿次数": int(picked.map(m71.on_edge).sum()) if bool((picked != "").all()) else None,
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。--recon_* 各项是重建臂里原文未给的设置，缺省值的来由见文件头。"""
    parser = argparse.ArgumentParser(description="81_within_year_split：年内随机划分与跨年留出的同口径对照")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    parser.add_argument("--seed", type=int, default=int(eu.RECOMMENDED_SEEDS[0]), help="划分序列的随机种子")
    parser.add_argument("--n_splits", type=int, default=None, help="年内划分的重复次数；缺省 formal 200、pilot 20")
    parser.add_argument(
        "--years",
        default=",".join(str(y) for y in eu.AVAILABLE_YEARS),
        help="A 臂做年内划分的年份，也是 B 臂要做的留出年（逗号分隔）",
    )
    parser.add_argument(
        "--arms", default="A,B", help="A = 年内划分，B = 跨年对照，none = 都不做；重建臂由 --recon 打开"
    )
    parser.add_argument(
        "--svm_grid", default="rbf", choices=["rbf", "full"], help="rbf = 71 号网格的 RBF 部分；full = 连同线性核"
    )
    parser.add_argument("--n_jobs", type=int, default=4, help="并行进程数（joblib）")
    parser.add_argument("--tag", default="", help="写进输出文件名的标签；formal 阶段改了协议参数时必须给")
    parser.add_argument("--recon", action="store_true", help="打开重建臂（要用 2018、2019 年原始光谱表）")
    parser.add_argument("--recon_alpha", type=float, default=1.0, help="SRDA（岭回归）的正则系数；原文未给")
    parser.add_argument("--recon_n_pc", type=int, default=16, help="各阶次的主成分数；原文只给出 0.6 阶取 16")
    parser.add_argument(
        "--recon_tree_criterion",
        default="gini",
        choices=["gini", "entropy", "log_loss"],
        help="决策树的分裂准则；原文未给",
    )
    parser.add_argument("--recon_tree_max_depth", type=int, default=0, help="决策树最大深度，0 = 不限；原文未给")
    parser.add_argument("--recon_tree_min_leaf", type=int, default=1, help="决策树叶子的最少样本数；原文未给")
    parser.add_argument(
        "--recon_split",
        default="stratified",
        choices=["stratified", "random"],
        help="7:3 划分是否按产地分层；原文未给",
    )
    parser.add_argument(
        "--recon_stack",
        default="insample",
        choices=["insample", "cv"],
        help="融合所用的训练集预测：insample = 样本内回代，cv = 训练集内部各折的样本外预测；原文未给",
    )
    parser.add_argument("--recon_stack_folds", type=int, default=5, help="--recon_stack cv 时的折数")
    parser.add_argument("--recon_bin", type=int, default=1, help="相邻几个采样点取均值（1 = 不合并，4 = 512 个点）")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    years = sorted({int(v) for v in args.years.split(",") if v.strip()})
    arm_text = args.arms.strip().lower()
    arms = [] if arm_text in ("", "none") else sorted({a.strip().upper() for a in args.arms.split(",") if a.strip()})
    n_splits = args.n_splits or (FORMAL_SPLITS if args.stage == "formal" else PILOT_SPLITS)
    if not years or not set(years) <= set(eu.AVAILABLE_YEARS):
        parser.error(f"--years 只能取 {eu.AVAILABLE_YEARS} 的非空子集")
    if not set(arms) <= {"A", "B"}:
        parser.error("--arms 只能取 A、B 或 none")
    if not arms and not args.recon:
        parser.error("没有要跑的臂：--arms none 时须同时给 --recon")
    if n_splits < 1 or args.recon_bin < 1 or args.recon_n_pc < 1 or args.recon_stack_folds < 2:
        parser.error("--n_splits、--recon_bin、--recon_n_pc 须 ≥ 1，--recon_stack_folds 须 ≥ 2")
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("--tag 只能含字母、数字、下划线与连字符")
    if args.recon:
        lost = [p for p in (m01.RAW_2018, m01.RAW_2019) if not os.path.exists(p)]
        if lost:
            parser.error(f"--recon 要用的原始光谱表不在：{lost}")
    # 不带 --tag 的正式工作簿只留给缺省协议
    changed = [
        k for k in (*GUARDED, *(GUARDED_RECON if args.recon else ())) if getattr(args, k) != parser.get_default(k)
    ]
    if n_splits != FORMAL_SPLITS:
        changed.append("n_splits")
    if years != sorted(eu.AVAILABLE_YEARS):
        changed.append("years")
    if arms != ["A", "B"]:
        changed.append("arms")
    if args.stage == "formal" and changed and not args.tag:
        parser.error(f"formal 阶段改了 {changed}：请同时给 --tag，不带标签的正式工作簿只留给缺省协议")
    tag = f"__{args.tag}" if args.tag else ""
    suffix = "" if args.stage == "formal" else "__pilot"
    out = os.path.join(eu.OUTPUT_DIR, f"{STEM}{tag}{suffix}.xlsx")

    logger = eu.get_logger(STEM)
    grids: dict[str, tuple[Any, ...]] = {"svm": SVM_GRIDS[args.svm_grid], "pls": tuple(m71.PLS_GRID)}
    eu.log_experiment_header(
        logger,
        {
            "任务": "年内随机 7:3 划分与跨年留出的同口径对照（补充材料 S9.1）",
            "阶段": f"{args.stage}（纯 CPU，sklearn / numpy）",
            "划分序列种子": args.seed,
            "臂": [*arms, *(["C 重建"] if args.recon else [])],
            "年份（A 的划分年份、B 的留出年）": years,
            "年内划分": f"按产地分层，测试部分占 {TEST_FRAC}，重复 {n_splits} 次",
            "选超参数": f"拟合用的那部分光谱内部 {INNER_FOLDS} 折，取宏 F1 均值最高者",
            "学习器": list(LEARNERS.values()),
            "SVM 候选": f"{args.svm_grid}（{len(grids['svm'])} 个）",
            "PLS-DA 候选潜变量数": grids["pls"],
            "跨年设计的源年份": {d: dict(m76.DESIGN_SRC[d]) for d in DESIGN_CODE},
            "重建臂设置": {k: getattr(args, k) for k in GUARDED_RECON} if args.recon else "不跑",
            "并行进程数": args.n_jobs,
            "输出": out,
        },
    )
    data = load_frozen(logger)
    cfg: dict[str, Any] = {"seed": args.seed, "grids": grids}
    setting = {
        "seed": args.seed,
        "stage": args.stage,
        "n_splits": n_splits,
        "years": ",".join(str(y) for y in years),
        "svm_grid": args.svm_grid,
        "n_jobs": args.n_jobs,
        "tag": args.tag,
    }
    long_rows: list[dict[str, Any]] = []
    compute: list[dict[str, Any]] = []
    table_a, table_b, table_d = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 先跑费时短的 B、C 两臂，出错早暴露；最费时的 A 臂放在最后。工作簿在全部跑完后一次写出
    if "B" in arms:
        with eu.track_compute("B 跨年对照", logger, None) as budget:
            table_b = pd.DataFrame(run_cross(data, years, cfg, args.n_jobs, logger))
        compute.append({"臂": "B", **setting, **budget})
    if args.recon:
        with eu.track_compute("C 重建臂", logger, None) as budget:
            rows_c, table_d = run_recon(args, n_splits, data, logger)
        long_rows += rows_c
        compute.append({"臂": "C", **setting, **budget})
        logger.log("\n" + table_d[table_d["学习器"] == FUSED].to_string(index=False))
    if "A" in arms:
        with eu.track_compute("A 年内随机划分", logger, None) as budget:
            rows_a, sizes = run_within(data, years, n_splits, cfg, args.n_jobs, logger)
        long_rows = [*rows_a, *long_rows]
        table_a = summarise_within(pd.DataFrame(rows_a), sizes)
        compute.append({"臂": "A", **setting, **budget})
        logger.log("\n" + table_a.to_string(index=False))

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        table_a.to_excel(writer, sheet_name=eu.table_sheet_name(0, "年内划分汇总"), index=False)
        table_b.to_excel(writer, sheet_name=eu.table_sheet_name(1, "跨年对照"), index=False)
        long_df = pd.DataFrame(long_rows, columns=list(LONG_COLS))
        long_df.to_excel(writer, sheet_name=eu.table_sheet_name(2, "逐次划分"), index=False)
        if args.recon:
            table_d.to_excel(writer, sheet_name=eu.table_sheet_name(3, "重建臂"), index=False)
        pd.DataFrame(compute).to_excel(writer, sheet_name="compute", index=False)
    logger.log(f"结果表 → {out}")
    logger.close()


if __name__ == "__main__":
    main()
