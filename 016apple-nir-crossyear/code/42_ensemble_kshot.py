"""42_ensemble_kshot.py: 多模型集成基线（随机森林 / 多数投票 / 加权平均 / stacking）K-Shot 面板。

补齐国家级大创 004《基于机器学习与迁移学习的苹果溯源模型研究》声明过、而稿件五方法面板
中缺席的一类方法：多模型集成。四条新增基线全部只在 SNV 预处理后的单面光谱上工作，与
20_unified_singleface_kshot.py 的三种传统方法处于同一表示层，不接触任何编码器。

**情节划分与 20 号脚本逐位一致**：本脚本以 importlib 直接加载 20 号脚本的
`build_support_query` / `_one_face` / `snv`，并以完全相同的顺序消费同一个 rng，
因此在相同 (fold, K, seed) 下得到的支持集/查询集/面位抽取与表 3 的三种传统方法完全相同。
脚本内置**自校验**：重跑 SVM / PLS-DA / NearestCentroid 三列，与
04outputs/formal_seeds/20_unified_singleface_kshot__seed<S>.xlsx 逐格比对，
超过 --selfcheck_tol 即判 FAIL 并拒绝写出结果（防止"换了个划分再来比"的伪对比）。

公平性口径（与既有五方法一致，见 TCSAE_zh.md 1.4 节）：
    - 全情节固定超参，不逐情节网格搜索；
    - 集成权重与 stacking 元学习器只用支持集自身的交叉验证拟合，**不接触查询集、不接触
      任何目标域无标签数据**，所用信息不超过 SVM/PLS-DA 已经用掉的那 K 个支持样本；
    - K=1 时支持集无法再切分，加权平均退化为等权、stacking 退化为等权软平均，如实标注。

统计口径（与 14_significance_test.py 一致）：折为主分析单位（n_primary=3）、
cluster bootstrap（2 000 次，折为聚类单位）、Nadeau-Bengio 方差修正配对 t、Holm 跨 6 个 K 校正。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
        python 02code/42_ensemble_kshot.py --seed 20060515 --seeds 50 --stage pilot

输出文件:
    04outputs/42_ensemble_kshot.xlsx    — 面板原始结果 / 自校验表 / 折级均值 / 配对检验 / 权重诊断
    04outputs/42_ensemble_kshot-图a.pdf / -fig-a.pdf — 九方法 K-Shot F1 曲线
    05logs/42_ensemble_kshot_YY-MM-DD_HHMMSS.log

设备说明:
    四条集成基线与三条传统基线均为纯 CPU sklearn 估计器，不含任何深度模型前向，
    故 `--stage pilot`（export_utils.save_sklearn_checkpoint 文档所述的纯 CPU sklearn 口径）。
    自校验通过即证明本机 CPU 数值与 Formal CUDA 归档逐格一致，集成列与表 3 同源可比。
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import export_utils as eu  # noqa: E402
from protonet import build_year_region_data  # noqa: E402

NDArray = npt.NDArray[Any]


def _load_panel20() -> Any:
    """以 importlib 加载 20 号脚本，保证情节划分函数逐字节同源（模块名以数字开头，不能 import）。

    不写入 sys.modules：子进程按 spawn 重新导入本模块时会各自重跑本函数，
    写入 sys.modules 反而会让 pickle 试图按名字 import 一个不存在的模块。
    """
    path = os.path.join(_THIS_DIR, "20_unified_singleface_kshot.py")
    spec = importlib.util.spec_from_file_location("panel20", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P20 = _load_panel20()

FOLD_CFG = P20.FOLD_CFG
ALL_REGIONS = P20.ALL_REGIONS
BASE_NAMES = ["SVM-Kshot", "PLS-DA-Kshot", "RandomForest"]
LEGACY_NAMES = ["NearestCentroid", "PLS-DA-Kshot", "SVM-Kshot"]
ENSEMBLE_NAMES = ["Ens-Vote", "Ens-WAvg", "Ens-Stack"]
NEW_NAMES = ["RandomForest", *ENSEMBLE_NAMES]
ALL_METHODS = [*LEGACY_NAMES, *NEW_NAMES]

RF_KW: dict[str, Any] = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "n_jobs": 1,
    "random_state": 0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 一、情节数据构造（与 20 号脚本 _eval_sklearn 的 rng 消费顺序逐步一致）
# ═══════════════════════════════════════════════════════════════════════════════
def build_episode(
    year_data: Any, tgt_year: int, k_shot: int, seed: int
) -> tuple[list[str], NDArray, NDArray, NDArray, list[str]]:
    """复刻 20 号脚本 `_eval_sklearn` 的数据构造：同 rng、同顺序 → 同支持集/查询集/面位。"""
    rng = np.random.default_rng(seed)
    regions, support, query = P20.build_support_query(year_data, tgt_year, k_shot, rng)
    idx = {r: i for i, r in enumerate(regions)}
    x_tr: list[NDArray] = []
    y_tr: list[int] = []
    for reg in regions:
        for aid in support[reg]:
            x_tr.append(P20._one_face(year_data[tgt_year][reg][aid], rng))
            y_tr.append(idx[reg])
    x_train = P20.snv(np.stack(x_tr))
    y_train = np.array(y_tr)
    x_q: list[NDArray] = []
    y_true: list[str] = []
    for aid, true_reg in query:
        x_q.append(P20._one_face(year_data[tgt_year][true_reg][aid], rng))
        y_true.append(true_reg)
    x_query = P20.snv(np.stack(x_q))
    return regions, x_train, y_train, x_query, y_true


# ═══════════════════════════════════════════════════════════════════════════════
# 二、基学习器与分数归一化
# ═══════════════════════════════════════════════════════════════════════════════
def _fit_predict_scores(
    kind: str, x_tr: NDArray, y_tr: NDArray, x_te: NDArray, n_class: int
) -> tuple[NDArray, NDArray]:
    """拟合基学习器，返回 ((n_te, n_class) 判别分数, (n_te,) 硬预测)。

    ⚠ 硬预测一律取估计器自身的 `predict()`，**不取分数的 argmax**：SVC 的 `predict`
    走 libsvm 的一对一投票，与 `decision_function` 的一对多变换在少数样本上并不一致；
    若改用 argmax，本脚本的 SVM 列就不再与表 3 的 SVM 列逐格相同（实测三折最大偏差
    0.0019），像素级同源性随即失守。分数只用于集成融合。
    失败时返回全零分数与全零预测（等价弃权），不中断整轮。
    """
    try:
        if kind == "svm":
            clf = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale"))])
            clf.fit(x_tr, y_tr)
            dec = np.asarray(clf.decision_function(x_te), dtype=float)
            if dec.ndim == 1:  # 二类退化
                dec = np.stack([-dec, dec], axis=1)
            out = np.zeros((len(x_te), n_class), dtype=float)
            for j, c in enumerate(clf.classes_):
                out[:, int(c)] = dec[:, j]
            return out, np.asarray(clf.predict(x_te), dtype=int)
        if kind == "pls":
            n_comp = max(1, min(len(x_tr) - 1, x_tr.shape[1], n_class))
            y_oh = np.zeros((len(x_tr), n_class))
            y_oh[np.arange(len(x_tr)), y_tr] = 1.0
            pls = PLSRegression(n_components=n_comp, max_iter=500)
            pls.fit(x_tr, y_oh)
            pred = np.asarray(pls.predict(x_te), dtype=float)
            return pred, pred.argmax(axis=1).astype(int)
        if kind == "rf":
            rf = RandomForestClassifier(**RF_KW)
            rf.fit(x_tr, y_tr)
            proba = np.asarray(rf.predict_proba(x_te), dtype=float)
            out = np.zeros((len(x_te), n_class), dtype=float)
            for j, c in enumerate(rf.classes_):
                out[:, int(c)] = proba[:, j]
            return out, np.asarray(rf.predict(x_te), dtype=int)
    except Exception:  # noqa: BLE001  基学习器在极小样本下可能不收敛；弃权而非中断整轮
        return np.zeros((len(x_te), n_class), dtype=float), np.zeros(len(x_te), dtype=int)
    raise ValueError(f"未知基学习器 {kind}")


def to_simplex(scores: NDArray) -> NDArray:
    """把量纲互不可比的判别分数映射到概率单纯形：逐样本跨类 z 标准化后 softmax。

    三种基学习器的输出量纲差异极大（SVM 判别值、PLS 回归值、RF 频率），直接平均等于
    让量纲最大的一方独裁。逐样本跨类 z 标准化对每个样本的仿射变换不变，softmax 保证
    结果落在单纯形内且各类均为正，故三方处于同一量纲。RF 的频率同样过这一步，以保证
    三方受到完全相同的处理。
    """
    s = np.asarray(scores, dtype=float)
    mu = s.mean(axis=1, keepdims=True)
    sd = s.std(axis=1, keepdims=True)
    z = (s - mu) / (sd + 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / (e.sum(axis=1, keepdims=True) + 1e-12))


# ═══════════════════════════════════════════════════════════════════════════════
# 三、集成器
# ═══════════════════════════════════════════════════════════════════════════════
def _support_cv_folds(y_tr: NDArray, k_shot: int) -> Any:
    """支持集内部交叉验证划分；K=1 无法切分时返回 None。"""
    if k_shot < 2:
        return None
    n_splits = int(min(k_shot, 5))
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)


def run_episode(
    regions: list[str], x_tr: NDArray, y_tr: NDArray, x_te: NDArray, k_shot: int
) -> tuple[dict[str, NDArray], dict[str, Any]]:
    """一轮情节：三种基学习器 + 三种集成，返回各方法的预测类别索引与权重诊断。"""
    n_class = len(regions)
    kinds = {"SVM-Kshot": "svm", "PLS-DA-Kshot": "pls", "RandomForest": "rf"}

    fitted = {m: _fit_predict_scores(kinds[m], x_tr, y_tr, x_te, n_class) for m in BASE_NAMES}
    base_prob = {m: to_simplex(fitted[m][0]) for m in BASE_NAMES}
    preds: dict[str, NDArray] = {m: fitted[m][1] for m in BASE_NAMES}

    # ── 支持集内部 CV：同时供加权平均取权重、供 stacking 造元特征 ─────────────
    cv = _support_cv_folds(y_tr, k_shot)
    oof = {m: np.full((len(x_tr), n_class), np.nan) for m in BASE_NAMES}
    cv_ok = False
    if cv is not None:
        try:
            for tr_idx, va_idx in cv.split(x_tr, y_tr):
                if len(np.unique(y_tr[tr_idx])) < 2:
                    continue
                for m in BASE_NAMES:
                    oof[m][va_idx] = to_simplex(
                        _fit_predict_scores(kinds[m], x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], n_class)[0]
                    )
            cv_ok = all(np.isfinite(oof[m]).all() for m in BASE_NAMES)
        except Exception:  # noqa: BLE001
            cv_ok = False

    # ── 多数投票：等权硬投票，平票按支持集 CV 准确率优先级裁决（无 CV 时按固定顺序）──
    votes = np.zeros((len(x_te), n_class), dtype=float)
    for m in BASE_NAMES:
        votes[np.arange(len(x_te)), preds[m]] += 1.0
    cv_acc = {m: float("nan") for m in BASE_NAMES}
    if cv_ok:
        for m in BASE_NAMES:
            cv_acc[m] = float(np.mean(oof[m].argmax(axis=1) == y_tr))
    prio_order = (
        sorted(BASE_NAMES, key=lambda m: (-cv_acc[m], BASE_NAMES.index(m))) if cv_ok else list(BASE_NAMES)
    )
    tie_break = np.zeros((len(x_te), n_class), dtype=float)
    for rank, m in enumerate(prio_order):
        tie_break[np.arange(len(x_te)), preds[m]] += (len(prio_order) - rank) * 1e-3
    preds["Ens-Vote"] = (votes + tie_break).argmax(axis=1)

    # ── 加权平均：权重 ∝ max(支持集 CV 准确率 − 1/C, 0)；K=1 或 CV 失败时等权 ──
    if cv_ok:
        raw_w = np.array([max(cv_acc[m] - 1.0 / n_class, 0.0) for m in BASE_NAMES], dtype=float)
        w = raw_w / raw_w.sum() if raw_w.sum() > 1e-12 else np.full(len(BASE_NAMES), 1.0 / len(BASE_NAMES))
        w_mode = "支持集CV"
    else:
        w = np.full(len(BASE_NAMES), 1.0 / len(BASE_NAMES))
        w_mode = "等权(K=1或CV不可用)"
    fused = np.zeros((len(x_te), n_class), dtype=float)
    for wi, m in zip(w, BASE_NAMES, strict=True):
        fused += wi * base_prob[m]
    preds["Ens-WAvg"] = fused.argmax(axis=1)

    # ── stacking：元学习器 = 多项逻辑回归，元特征 = 三基学习器的单纯形分数 ──────
    stack_mode = "退化为等权软平均(K=1或CV不可用)"
    if cv_ok:
        try:
            meta_tr = np.concatenate([oof[m] for m in BASE_NAMES], axis=1)
            meta_te = np.concatenate([base_prob[m] for m in BASE_NAMES], axis=1)
            meta = LogisticRegression(C=1.0, max_iter=1000)
            meta.fit(meta_tr, y_tr)
            proba = np.asarray(meta.predict_proba(meta_te), dtype=float)
            out = np.zeros((len(x_te), n_class), dtype=float)
            for j, c in enumerate(meta.classes_):
                out[:, int(c)] = proba[:, j]
            preds["Ens-Stack"] = out.argmax(axis=1)
            stack_mode = f"LR元学习器(cv={_support_cv_folds(y_tr, k_shot).n_splits})"
        except Exception:  # noqa: BLE001
            preds["Ens-Stack"] = fused.argmax(axis=1)
    else:
        preds["Ens-Stack"] = np.mean([base_prob[m] for m in BASE_NAMES], axis=0).argmax(axis=1)

    diag = {
        "权重模式": w_mode,
        "stacking模式": stack_mode,
        **{f"w_{m}": round(float(wi), 4) for wi, m in zip(w, BASE_NAMES, strict=True)},
        **{f"cvacc_{m}": (round(cv_acc[m], 4) if np.isfinite(cv_acc[m]) else float("nan")) for m in BASE_NAMES},
    }
    return preds, diag


def run_unit(
    year_data: Any, tgt_year: int, k_shot: int, fseed: int, n_reps: int
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]], dict[str, Any], list[str]]:
    """一个工作单元 = 一个 (折, K, 正式种子) 下的 n_reps 次情节重采样。

    单元结果只由 (tgt_year, k_shot, fseed) 决定，无共享状态，故并行与串行逐位等价。
    """
    acc: dict[str, list[float]] = {m: [] for m in ALL_METHODS}
    pc_acc: dict[str, dict[str, list[float]]] = {}
    diag: dict[str, Any] = {}
    regions: list[str] = []
    for i in range(n_reps):
        s = fseed + i
        regions, x_tr, y_tr, x_te, y_true = build_episode(year_data, tgt_year, k_shot, s)
        if not pc_acc:
            pc_acc = {m: {r: [] for r in regions} for m in ALL_METHODS}
        preds, d = run_episode(regions, x_tr, y_tr, x_te, k_shot)
        if i == 0:
            diag = d
        nc_macro, nc_per = P20.eval_nearest_centroid(year_data, tgt_year, k_shot, s)
        acc["NearestCentroid"].append(nc_macro)
        for r in regions:
            pc_acc["NearestCentroid"][r].append(nc_per.get(r, float("nan")))
        for m in [*BASE_NAMES, *ENSEMBLE_NAMES]:
            y_pred = [regions[int(j)] for j in preds[m]]
            macro, per = P20._macro_and_perclass(y_true, y_pred, regions)
            acc[m].append(macro)
            for r in regions:
                pc_acc[m][r].append(per.get(r, float("nan")))
    return acc, pc_acc, diag, regions


def fit_full_ensemble(
    regions: list[str], x_tr: NDArray, y_tr: NDArray, k_shot: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """按 `run_episode` 的同一口径重拟合一套完整集成工件，供 checkpoint 落盘。

    与 `run_episode` 的差别只在返回值：那里返回预测，这里返回**拟合好的对象本身**
    （三个基学习器 + 加权平均权重 + stacking 元学习器），使 checkpoint 能独立复现三种集成的预测。
    """
    n_class = len(regions)
    kinds = {"SVM-Kshot": "svm", "PLS-DA-Kshot": "pls", "RandomForest": "rf"}
    n_comp = max(1, min(len(x_tr) - 1, x_tr.shape[1], n_class))
    y_oh = np.zeros((len(x_tr), n_class))
    y_oh[np.arange(len(x_tr)), y_tr] = 1.0
    base = {
        "base_svm": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale"))]).fit(
            x_tr, y_tr
        ),
        "base_pls": PLSRegression(n_components=n_comp, max_iter=500).fit(x_tr, y_oh),
        "base_rf": RandomForestClassifier(**RF_KW).fit(x_tr, y_tr),
    }
    cv = _support_cv_folds(y_tr, k_shot)
    oof = {m: np.full((len(x_tr), n_class), np.nan) for m in BASE_NAMES}
    if cv is not None:
        for tr_idx, va_idx in cv.split(x_tr, y_tr):
            for m in BASE_NAMES:
                oof[m][va_idx] = to_simplex(
                    _fit_predict_scores(kinds[m], x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], n_class)[0]
                )
    cv_ok = cv is not None and all(np.isfinite(oof[m]).all() for m in BASE_NAMES)
    if cv_ok:
        cv_acc = np.array([float(np.mean(oof[m].argmax(axis=1) == y_tr)) for m in BASE_NAMES])
        raw_w = np.maximum(cv_acc - 1.0 / n_class, 0.0)
        w = raw_w / raw_w.sum() if raw_w.sum() > 1e-12 else np.full(len(BASE_NAMES), 1.0 / len(BASE_NAMES))
        meta = LogisticRegression(C=1.0, max_iter=1000).fit(
            np.concatenate([oof[m] for m in BASE_NAMES], axis=1), y_tr
        )
    else:  # K=1：支持集无法切分，退化为等权软平均且无元学习器（与主实验口径一致）
        w = np.full(len(BASE_NAMES), 1.0 / len(BASE_NAMES))
        meta = None
    bundle = {
        **base,
        "member_order": list(BASE_NAMES),
        "weights": w,
        "meta_learner": meta,
        "regions": regions,
        "score_to_simplex": "逐样本跨类 z 标准化后 softmax（见 to_simplex）",
    }
    info = {
        "n_components": n_comp,
        "weights": [round(float(v), 4) for v in w],
        "cv_folds": (cv.n_splits if cv is not None else 0),
        "meta_dim": len(BASE_NAMES) * n_class,
    }
    return bundle, info


_WORKER: dict[str, Any] = {}
CACHE_DIR = os.path.join(eu.OUTPUT_DIR, "42_ensemble_cache")


def _cache_path(tgt_year: int, k_shot: int, fseed: int, n_reps: int) -> str:
    return os.path.join(CACHE_DIR, f"unit_{tgt_year}_K{k_shot}_seed{fseed}_rep{n_reps}.joblib")


def _worker_init() -> None:
    """子进程一次性加载光谱数据（主进程已做过 SHA-256 校验，此处不重复校验）。"""
    df = eu.load_all_elements_df(verify_sha256=False)
    _WORKER["year_data"] = build_year_region_data(df, eu.get_spectrum_columns(df))


def _run_unit_task(task: tuple[int, int, int, int]) -> Any:
    """单元级磁盘缓存：单元结果只由 (折, K, 种子, 重采样次数) 决定，故缓存命中与重算等价。

    存在的意义是把一次约 20 min 的全量跑拆成可中断、可续跑的若干段——重跑不会改变任何数字。
    """
    import joblib

    tgt_year, k_shot, fseed, n_reps = task
    path = _cache_path(tgt_year, k_shot, fseed, n_reps)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:  # noqa: BLE001  缓存损坏则重算
            pass
    res = run_unit(_WORKER["year_data"], tgt_year, k_shot, fseed, n_reps)
    os.makedirs(CACHE_DIR, exist_ok=True)
    joblib.dump(res, path)
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# 四、自校验：与 Formal 归档逐格比对三条既有基线
# ═══════════════════════════════════════════════════════════════════════════════
def load_formal_panel() -> pd.DataFrame | None:
    """读取 04outputs/formal_seeds/20_unified_singleface_kshot__seed*.xlsx 的面板表。"""
    pat = os.path.join(eu.OUTPUT_DIR, "formal_seeds", "20_unified_singleface_kshot__seed*.xlsx")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    frames = []
    for f in files:
        seed = int(os.path.basename(f).split("seed")[-1].split(".")[0])
        d = pd.read_excel(f, sheet_name=0)
        d["种子"] = seed
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 五、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="42_ensemble_kshot：多模型集成 K-Shot 基线（大创 004 对齐）")
    parser.add_argument("--k_list", default="1,3,5,10,20,50")
    parser.add_argument("--seeds", type=int, default=50, help="每折每 K 的情节重采样次数")
    parser.add_argument(
        "--seed_list",
        default=",".join(str(s) for s in eu.RECOMMENDED_SEEDS),
        help="正式种子集合；逐种子跑满 --seeds 次重采样",
    )
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    parser.add_argument("--n_jobs", type=int, default=6, help="工作单元并行度；单元间无共享状态，与串行逐位等价")
    parser.add_argument("--selfcheck_tol", type=float, default=5e-4)
    parser.add_argument("--skip_selfcheck", action="store_true")
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_list.split(",")]
    formal_seeds = [int(s) for s in args.seed_list.split(",")]
    anchor_seed = formal_seeds[0]

    logger = eu.get_logger("42_ensemble_kshot")
    eu.set_random_seed(anchor_seed, logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "多模型集成 K-Shot 基线（随机森林 / 多数投票 / 加权平均 / stacking）",
            "阶段": args.stage,
            "设备": "cpu（纯 sklearn，无深度前向）",
            "正式种子": formal_seeds,
            "每折每K重采样": args.seeds,
            "K_list": k_values,
            "情节划分来源": "20_unified_singleface_kshot.build_support_query（importlib 同源）",
        },
    )
    run = eu.init_wandb(__file__, anchor_seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(df)
    year_data = build_year_region_data(df, wl_cols)

    panel_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    # per (fold, K, method) 的逐情节 F1，供折级聚合
    store: dict[tuple[int, int, str], list[float]] = {}

    # K 大的单元最慢，先派发以免尾部串行；单元间无共享状态，故并行与串行逐位等价
    units = sorted(
        [(y, k, s) for y in sorted(FOLD_CFG) for k in k_values for s in formal_seeds],
        key=lambda u: -u[1],
    )
    logger.log(f"工作单元 {len(units)} 个（折 × K × 正式种子），每单元 {args.seeds} 次情节重采样，n_jobs={args.n_jobs}")
    with eu.track_compute("集成基线面板", logger, run) as budget:
        tasks = [(y, k, s, args.seeds) for (y, k, s) in units]
        n_cached = sum(1 for t in tasks if os.path.exists(_cache_path(*t)))
        logger.log(f"    单元缓存命中 {n_cached}/{len(tasks)}（缓存目录 {CACHE_DIR}）")
        if args.n_jobs <= 1:
            _WORKER["year_data"] = year_data
            results = [_run_unit_task(t) for t in tasks]
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=ctx, initializer=_worker_init) as ex:
                results = []
                for i, r in enumerate(ex.map(_run_unit_task, tasks), start=1):
                    results.append(r)
                    if i % 10 == 0 or i == len(tasks):
                        logger.log(f"    进度 {i}/{len(tasks)} 单元")
        by_unit = dict(zip(units, results, strict=True))
        for tgt_year in sorted(FOLD_CFG):
            cfg = FOLD_CFG[tgt_year]
            for k in k_values:
                acc: dict[str, list[float]] = {m: [] for m in ALL_METHODS}
                pc_acc: dict[str, dict[str, list[float]]] = {m: {} for m in ALL_METHODS}
                for fseed in formal_seeds:  # 按正式种子顺序拼接，跨种子 SD 的切片口径依赖此顺序
                    u_acc, u_pc, u_diag, _regions = by_unit[(tgt_year, k, fseed)]
                    for m in ALL_METHODS:
                        acc[m].extend(u_acc[m])
                        for r, vals in u_pc[m].items():
                            pc_acc[m].setdefault(r, []).extend(vals)
                    diag_rows.append({"目标年份": tgt_year, "K": k, "种子": fseed, **u_diag})
                for m in ALL_METHODS:
                    store[(tgt_year, k, m)] = list(acc[m])
                    panel_rows.append(
                        {
                            "目标年份": tgt_year,
                            "源年份": "+".join(str(y) for y in cfg["src_years"]),
                            "K": k,
                            "方法": m,
                            "F1均值": round(float(np.mean(acc[m])), 4),
                            "F1标准差": round(float(np.std(acc[m])), 4),
                            "情节数": len(acc[m]),
                            **{
                                f"F1类_{r}": (
                                    round(float(np.nanmean(pc_acc[m][r])), 4) if r in pc_acc[m] else float("nan")
                                )
                                for r in ALL_REGIONS
                            },
                        }
                    )
                logger.log(
                    f"  LOEO-{tgt_year} K={k:3d}："
                    + " ".join(f"{m}={float(np.mean(acc[m])):.4f}" for m in ALL_METHODS)
                )

    panel_df = pd.DataFrame(panel_rows)

    # ── 自校验 ────────────────────────────────────────────────────────────────
    check_rows: list[dict[str, Any]] = []
    formal = None if args.skip_selfcheck else load_formal_panel()
    if formal is not None:
        ref = formal.groupby(["目标年份", "K", "方法"])["F1均值"].mean().reset_index()
        for m in LEGACY_NAMES:
            for tgt_year in (2018, 2019, 2025):
                for k in k_values:
                    r = ref[(ref["目标年份"] == tgt_year) & (ref["K"] == k) & (ref["方法"] == m)]
                    if r.empty:
                        continue
                    mine = float(np.mean(store[(tgt_year, k, m)]))
                    theirs = float(r["F1均值"].iloc[0])
                    check_rows.append(
                        {
                            "方法": m,
                            "目标年份": tgt_year,
                            "K": k,
                            "本机复跑": round(mine, 4),
                            "Formal归档": round(theirs, 4),
                            "绝对差": round(abs(mine - theirs), 5),
                        }
                    )
    check_df = pd.DataFrame(check_rows)
    if not check_df.empty:
        worst = float(check_df["绝对差"].max())
        logger.log(f"\n── 自校验：三条既有基线 vs Formal 归档，共 {len(check_df)} 格，最大绝对差 {worst:.5f} ──")
        if worst > args.selfcheck_tol:
            bad = check_df[check_df["绝对差"] > args.selfcheck_tol]
            logger.log(bad.to_string(index=False))
            logger.close()
            raise RuntimeError(
                f"自校验 FAIL：最大绝对差 {worst:.5f} > 容差 {args.selfcheck_tol}。"
                "情节划分与 Formal 归档不一致，集成列不可与表 3 同表比较，拒绝写出。"
            )
        logger.log("  自校验 PASS：本机 CPU 数值与 Formal CUDA 归档逐格一致，集成列与表 3 同源可比。")
    else:
        logger.log("\n[WARN] 未找到 Formal 归档或已跳过，自校验未执行；集成列与表 3 的同源性未经验证。")

    # ── 折级聚合 + 配对检验（n_primary=3，Holm 跨 6 个 K）────────────────────
    proto_fold: dict[tuple[int, int], float] = {}
    if formal is not None:
        pf = formal[formal["方法"] == "VanillaProtoNet"].groupby(["目标年份", "K"])["F1均值"].mean()
        proto_fold = {(int(a), int(b)): float(v) for (a, b), v in pf.items()}

    foldmean_rows: list[dict[str, Any]] = []
    for k in k_values:
        row: dict[str, Any] = {"K": k}
        for m in ALL_METHODS:
            for tgt_year in (2018, 2019, 2025):
                row[f"{m}@{tgt_year}"] = round(float(np.mean(store[(tgt_year, k, m)])), 4)
        foldmean_rows.append(row)
    foldmean_df = pd.DataFrame(foldmean_rows)

    references = ["SVM-Kshot"] + (["VanillaProtoNet"] if proto_fold else [])
    test_rows: list[dict[str, Any]] = []
    for ref_name in references:
        for m in NEW_NAMES:
            p_raw: list[float] = []
            tmp: list[dict[str, Any]] = []
            for k in k_values:
                a = [float(np.mean(store[(y, k, m)])) for y in (2018, 2019, 2025)]
                b = (
                    [proto_fold[(y, k)] for y in (2018, 2019, 2025)]
                    if ref_name == "VanillaProtoNet"
                    else [float(np.mean(store[(y, k, ref_name)])) for y in (2018, 2019, 2025)]
                )
                cmp = eu.compare_methods(a, b, n_train=100, n_test=50, n_bootstrap=2000, seed=anchor_seed)
                p_raw.append(cmp["p_value"])
                tmp.append({"K": k, **cmp})
            holm = eu.apply_multi_comp_correction(p_raw, method="holm")
            for rr, p_adj, rej in zip(tmp, holm["p_adjusted"], holm["reject"], strict=True):
                test_rows.append(
                    {
                        "比较": f"{m} vs {ref_name}",
                        "K": rr["K"],
                        "n_primary": 3,
                        "均值差(折级)": round(rr["mean_diff"], 4),
                        "CI_low": round(rr["ci_low"], 4),
                        "CI_high": round(rr["ci_high"], 4),
                        "p_raw": round(rr["p_value"], 4),
                        "p_holm": round(p_adj, 4),
                        "Cohen_d": round(rr["cohens_d"], 3),
                        "显著(Holm)": "YES" if rej >= 1.0 else "NO",
                    }
                )
    test_df = pd.DataFrame(test_rows)
    logger.log("\n── 折级配对检验（n_primary=3，Nadeau-Bengio 修正，Holm 跨 6 个 K）──")
    logger.log(test_df.to_string(index=False))

    # ── 汇总表：本脚本九列 + Formal 归档两列原型网络，供正文表 3 直接取数 ──────
    summary_rows: list[dict[str, Any]] = []
    for k in k_values:
        row = {"K": k}
        for m in ALL_METHODS:
            row[m] = round(float(np.mean([np.mean(store[(y, k, m)]) for y in (2018, 2019, 2025)])), 4)
        if proto_fold:
            row["VanillaProtoNet(Formal归档)"] = round(
                float(np.mean([proto_fold[(y, k)] for y in (2018, 2019, 2025)])), 4
            )
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    logger.log("\n── 三折均值汇总（各方法 × K）──")
    logger.log(summary_df.to_string(index=False))

    # ── 跨种子标准差（与表 3 口径一致：折内先对情节取均值 → 折间取均值 → 种子间求 SD）──
    sd_rows: list[dict[str, Any]] = []
    for k in k_values:
        row = {"K": k}
        for m in ALL_METHODS:
            per_seed = []
            for si, fseed in enumerate(formal_seeds):
                lo = si * args.seeds
                hi = lo + args.seeds
                per_seed.append(float(np.mean([np.mean(store[(y, k, m)][lo:hi]) for y in (2018, 2019, 2025)])))
            row[m] = round(float(np.std(per_seed)), 4)
        sd_rows.append(row)
    sd_df = pd.DataFrame(sd_rows)

    # ── 图 ────────────────────────────────────────────────────────────────────
    color = {
        "NearestCentroid": "#999999",
        "PLS-DA-Kshot": "#E69F00",
        "SVM-Kshot": "#009E73",
        "RandomForest": "#8C564B",
        "Ens-Vote": "#CC79A7",
        "Ens-WAvg": "#0072B2",
        "Ens-Stack": "#D55E00",
    }
    txt = {
        "zh": {"x": "K-Shot（每产地支持苹果数）", "y": "Macro-F1（三折均值）", "t": "多模型集成基线的 K-Shot 表现", "r": "随机基线"},
        "en": {
            "x": "K-shot (support apples per origin)",
            "y": "Macro-F1 (3-fold mean)",
            "t": "K-shot performance of ensemble baselines",
            "r": "Chance",
        },
    }

    def plot_a(lang: str) -> Any:
        t = txt[lang]
        fig, ax = plt.subplots(figsize=(9, 5))
        for m in ALL_METHODS:
            ys = [summary_df[summary_df["K"] == k][m].iloc[0] for k in k_values]
            style = "-" if m in NEW_NAMES else "--"
            ax.plot(k_values, ys, marker="o", linestyle=style, linewidth=2, color=color[m], label=m)
        if proto_fold:
            ys = [summary_df[summary_df["K"] == k]["VanillaProtoNet(Formal归档)"].iloc[0] for k in k_values]
            ax.plot(k_values, ys, marker="s", linestyle=":", linewidth=2, color="#56B4E9", label="VanillaProtoNet")
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel(t["x"])
        ax.set_ylabel(t["y"])
        ax.set_title(t["t"])
        ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.4, label=t["r"])
        ax.set_ylim(0.3, 0.85)
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    zh, en = eu.save_figure_bilingual(plot_a, "42_ensemble_kshot", 0)
    logger.log(f"图a → {zh} / {en}")

    # ── 最终模型 checkpoint（F1：save_checkpoint + SHA-256 + 模型卡）──────────
    # 落盘的是**可直接复现三种集成预测的完整工件**：三个基学习器 + 加权平均的权重 + stacking 元学习器。
    # 只存基学习器是不够的——那样模型卡上写的元学习器在工件里并不存在，拿到 checkpoint 也重现不了 Ens-Stack。
    regions, x_tr, y_tr, _x_te, _y_true = build_episode(year_data, 2025, 50, anchor_seed)
    final_bundle, final_meta = fit_full_ensemble(regions, x_tr, y_tr, 50)
    ck = eu.save_checkpoint(
        final_bundle,
        f"016_42_ensemble_kshot_fold2025_K50_seed{anchor_seed}",
        {
            "arch": (
                "Ensemble{SVC(rbf,C=10) + PLSRegression(n_components=%d) + RandomForest(300 trees)}"
                " → majority vote / weighted average / stacking(LogisticRegression C=1)" % final_meta["n_components"]
            ),
            "params": (
                f"支持集 {len(x_tr)} 样本 × {x_tr.shape[1]} 波段；"
                f"加权平均权重 {final_meta['weights']}（SVM/PLS-DA/RF，来自支持集 {final_meta['cv_folds']} 折 CV）；"
                f"stacking 元学习器 LogisticRegression(C=1)，元特征 {final_meta['meta_dim']} 维"
            ),
            "flops": "N/A（sklearn 估计器，非逐层前向）",
            "stage": args.stage,
            "training_hardware": "cpu",
            "deterministic_guarantee": "full（sklearn 固定 random_state，无 GPU 非确定性算子）",
            "protocol": "LOEO-2025 折，K=50，统一单面查询；情节划分同 20_unified_singleface_kshot",
            "note": (
                "权重与元学习器仅由支持集内部分层 CV 拟合，未接触查询集与目标域无标签数据；"
                "bundle 内 base_svm/base_pls/base_rf/weights/meta_learner/regions 齐备，可直接复现三种集成的预测"
            ),
        },
    )
    logger.log(
        f"最终集成模型 checkpoint → {ck['ckpt_path']}（SHA-256 {ck['sha256'][:16]}…）"
        f"，加权平均权重 {final_meta['weights']}"
    )

    budget_df = pd.DataFrame([{"正式种子数": eu.n_distinct_seeds(tuple(formal_seeds)), "每折每K重采样": args.seeds, **budget}])
    xlsx = eu.write_script_workbook(
        __file__,
        {
            0: ("面板原始结果", panel_df),
            1: ("三折均值汇总", summary_df),
            2: ("跨种子标准差", sd_df),
            3: ("自校验对账", check_df if not check_df.empty else pd.DataFrame([{"状态": "未执行"}])),
            4: ("折级均值", foldmean_df),
            5: ("折级配对检验", test_df),
            6: ("集成权重诊断", pd.DataFrame(diag_rows)),
            "compute": budget_df,
        },
    )
    logger.log(f"结果 → {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
