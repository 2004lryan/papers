"""65_domain_adaptation_baselines.py: 经典跨域基线（池化 SVM / CORAL-SVM / TCA-SVM）K-Shot 面板。

背景：方法比较此前只含只看目标年支持集的传统学习器、原型网络与多模型集成，没有覆盖跨批次、
跨年光谱研究里最常用的两种经典域适应——CORAL（二阶统计对齐）与 TCA（迁移成分分析），
而 SAA 近期的跨批次 Vis-NIR 分类研究正是以这两者为对照。本脚本在与 20/42 号脚本**逐位相同**的
情节上（直接调用 42 号脚本的 build_episode：同 rng 消费顺序 → 同支持集、同查询集、同面位抽取）
新增三条路线：

    Pooled-SVM : 源域带标签光谱 + 目标年 K 个支持样本，训练同一 RBF-SVM，不做任何对齐
    CORAL-SVM  : 先把源域光谱的均值与协方差对齐到目标年全部无标签光谱（CORAL），再同 Pooled-SVM
    TCA-SVM    : 源域与目标年无标签光谱联合做迁移成分分析（RBF 核），在隐空间里同 Pooled-SVM

所用信息（与既有路线的区别如实写进结果表）:
    - 三条路线都用源域标签（与原型网络元训练所用信息同级），目标年只用 K 个支持样本的标签；
    - CORAL 与 TCA 另外使用目标年全部光谱、但不用其标签（直推式），查询集标签从不接触；
    - 源域只保留目标年类别集内的类（如 LOEO-2018 剔除 2025 的甘肃）；目标年独有的类
      （LOEO-2025 的甘肃）只有 K 个支持样本，故这三条路线的 SVM 一律 class_weight="balanced"，
      其余超参与既有 SVM 列完全相同（StandardScaler + RBF, C=10, gamma="scale"），全情节固定；
    - 源域每类至多取 --src_per_class 条光谱、TCA 拟合样本源域与目标年各取 --tca_fit 条，
      抽样只依赖 (折, 正式种子)，与情节重采样无关。

自校验（两道，任一不过即拒绝写出；新列因此与表 S6 的九条路线处在同一套情节上）：
    ① 最近质心无随机性、无并列，以 20 号脚本的 `eval_nearest_centroid` 逐 (折, K, 种子) 复算，
       与 04outputs/formal_seeds/20_unified_singleface_kshot__seed<S>.xlsx 逐格比对，容差 --nc_tol；
    ② 只看支持集的 SVM 以 `_eval_sklearn` 复算，按 (折, K) 取 5 种子均值后比对，容差 --selfcheck_tol
       （与 42 号脚本同口径；K=1 时只有 3 个训练点，libsvm 的并列裁决在不同平台上有微小差别，
       逐种子格的偏差会略大，照录不设闸）。

运行方式:
    cd <项目根>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/65_domain_adaptation_baselines.py --workers 6

输出文件:
    04outputs/65_domain_adaptation_baselines.xlsx      — 原始结果 / 自校验 / 折级均值 / 三折均值 / 跨种子SD / 配对检验
    04outputs/65_domain_adaptation_baselines-图a.pdf / -fig-a.pdf — 新增路线与原型网络、SVM 的 K-Shot 曲线
    07checkpoints/65_da_<route>_fold2025_K50_seed20060515/best.joblib — 代表性模型（含模型卡）
    05logs/65_domain_adaptation_baselines_YY-MM-DD_HHMMSS.log

设备说明:
    全部为纯 CPU 的 sklearn/numpy 计算，按 export_utils.save_sklearn_checkpoint 文档的纯 CPU sklearn
    口径标 `--stage pilot`；5 个正式种子、每折每 K 50 次重采样，与表 S6 的集成列同一口径。
"""

from __future__ import annotations

import argparse
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
from scipy import linalg
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import export_utils as eu  # noqa: E402
from protonet import build_year_region_data  # noqa: E402

NDArray = npt.NDArray[Any]


def _load_module(alias: str, filename: str) -> Any:
    """以 importlib 按文件加载数字开头的脚本（不写入 sys.modules，子进程各自重载）。"""
    path = os.path.join(_THIS_DIR, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M42 = _load_module("panel42", "42_ensemble_kshot.py")
P20 = M42.P20
FOLD_CFG: dict[int, dict[str, Any]] = P20.FOLD_CFG
FOLDS = (2018, 2019, 2025)
NEW_METHODS = ["Pooled-SVM", "CORAL-SVM", "TCA-SVM"]
CHECK_METHOD = "SVM-Kshot"
NC_METHOD = "NearestCentroid"
ALL_METHODS = [CHECK_METHOD, NC_METHOD, *NEW_METHODS]


# ═══════════════════════════════════════════════════════════════════════════════
# 一、CORAL 与 TCA
# ═══════════════════════════════════════════════════════════════════════════════
def _mat_pow(c: NDArray, p: float) -> NDArray:
    """对称正定矩阵的实数次幂（特征分解，特征值下限 1e-10）。"""
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-10, None)
    return np.asarray((v * w**p) @ v.T)


def coral(xs: NDArray, xt: NDArray, lam: float = 1.0) -> NDArray:
    """CORAL：源域先按自身均值中心化，协方差白化后再按目标域协方差重着色，最后平移到目标域均值。

    各波段先除以两域合并标准差，使 lam=1 的正则在各波段尺度一致（Sun 等 2016 的标准化口径）。
    """
    mu_s, mu_t = xs.mean(axis=0), xt.mean(axis=0)
    scale = np.concatenate([xs, xt]).std(axis=0) + 1e-8
    zs, zt = (xs - mu_s) / scale, (xt - mu_t) / scale
    eye = np.eye(zs.shape[1])
    cs = np.cov(zs, rowvar=False) + lam * eye
    ct = np.cov(zt, rowvar=False) + lam * eye
    return np.asarray((zs @ _mat_pow(cs, -0.5) @ _mat_pow(ct, 0.5)) * scale + mu_t)


class TCAMap:
    """拟合好的 TCA 映射：z = k(x, X_fit) W。"""

    def __init__(self, x_fit: NDArray, w: NDArray, scaler: StandardScaler, gamma: float) -> None:
        self.x_fit = x_fit
        self.w = w
        self.scaler = scaler
        self.gamma = gamma

    def transform(self, x: NDArray) -> NDArray:
        return np.asarray(rbf_kernel(self.scaler.transform(x), self.x_fit, gamma=self.gamma) @ self.w)


def fit_tca(xs: NDArray, xt: NDArray, rng: np.random.Generator, n_fit: int, dim: int, mu: float) -> TCAMap:
    """迁移成分分析（Pan 等 2011）：在 RBF 核上求 max tr(WᵀKHKW) / tr(Wᵀ(KLK+μI)W) 的前 dim 个方向。

    L = eeᵀ 为两域 MMD 矩阵（e_i = 1/n_s 或 −1/n_t），H 为中心化矩阵；
    拟合样本源域与目标年各随机取至多 n_fit 条，其余样本经核函数做样本外映射。
    """
    si = rng.choice(len(xs), size=min(n_fit, len(xs)), replace=False)
    ti = rng.choice(len(xt), size=min(n_fit, len(xt)), replace=False)
    raw = np.concatenate([xs[si], xt[ti]])
    scaler = StandardScaler().fit(raw)
    x_fit = np.asarray(scaler.transform(raw))
    gamma = 1.0 / (x_fit.shape[1] * float(x_fit.var()))  # 与 sklearn gamma="scale" 同式
    k = np.asarray(rbf_kernel(x_fit, x_fit, gamma=gamma))
    ns, nt = len(si), len(ti)
    n = ns + nt
    e = np.concatenate([np.full(ns, 1.0 / ns), np.full(nt, -1.0 / nt)])
    ke = k @ e
    klk = np.outer(ke, ke)
    kc = k - k.mean(axis=0, keepdims=True)  # H K
    khk = k @ kc
    khk = 0.5 * (khk + khk.T)
    _, vecs = linalg.eigh(khk, klk + mu * np.eye(n))
    w = np.asarray(vecs[:, ::-1][:, :dim])  # 最大的 dim 个广义特征值
    return TCAMap(x_fit, w, scaler, gamma)


def _svm_fit_predict(x_tr: NDArray, y_tr: NDArray, x_te: NDArray) -> tuple[NDArray, Pipeline]:
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")),
        ]
    )
    clf.fit(x_tr, y_tr)
    return np.asarray(clf.predict(x_te), dtype=int), clf


# ═══════════════════════════════════════════════════════════════════════════════
# 二、折级资产（只依赖 (折, 正式种子)）与单元运行
# ═══════════════════════════════════════════════════════════════════════════════
_W: dict[str, Any] = {}


def _worker_init() -> None:
    """子进程一次性加载光谱数据（主进程已做过 SHA-256 校验，此处不重复校验）。"""
    df = eu.load_all_elements_df(verify_sha256=False)
    _W["year_data"] = build_year_region_data(df, eu.get_spectrum_columns(df))
    _W["assets"] = {}


def fold_assets(year_data: Any, tgt_year: int, fseed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """源域（限目标年类别集、每类限量）、目标年全部无标签光谱、CORAL 后源域、TCA 映射。"""
    regions = sorted(year_data[tgt_year].keys())
    idx = {r: i for i, r in enumerate(regions)}
    rng = np.random.default_rng([fseed, tgt_year, 65])
    xs_parts: list[NDArray] = []
    ys_parts: list[NDArray] = []
    for src_year in FOLD_CFG[tgt_year]["src_years"]:
        for reg, apples in year_data[src_year].items():
            if reg not in idx:
                continue
            for faces in apples.values():
                xs_parts.append(np.asarray(faces))
                ys_parts.append(np.full(len(faces), idx[reg]))
    xs_all = np.asarray(P20.snv(np.concatenate(xs_parts)))
    ys_all = np.concatenate(ys_parts)
    keep: list[NDArray] = []
    for c in np.unique(ys_all):
        ii = np.flatnonzero(ys_all == c)
        if len(ii) > cfg["src_per_class"]:
            ii = np.sort(rng.choice(ii, size=cfg["src_per_class"], replace=False))
        keep.append(ii)
    kept = np.concatenate(keep)
    xs, ys = xs_all[kept], ys_all[kept]
    xt = np.asarray(P20.snv(np.concatenate([np.asarray(f) for r in regions for f in year_data[tgt_year][r].values()])))
    tca = fit_tca(xs, xt, rng, cfg["tca_fit"], cfg["tca_dim"], cfg["tca_mu"])
    return {
        "regions": regions,
        "xs": xs,
        "ys": ys,
        "xs_coral": coral(xs, xt, lam=cfg["coral_lam"]),
        "tca": tca,
        "zs": tca.transform(xs),
        "n_src_per_class": {regions[int(c)]: int((ys == c).sum()) for c in np.unique(ys)},
        "n_target_unlabelled": len(xt),
    }


def run_unit(
    year_data: Any, assets: dict[str, Any], tgt_year: int, k_shot: int, fseed: int, n_reps: int
) -> dict[str, Any]:
    """一个单元 = 一个 (折, K, 正式种子) 下的 n_reps 次情节。情节与 42/20 号脚本逐位相同。"""
    regions = assets["regions"]
    macro: dict[str, list[float]] = {m: [] for m in ALL_METHODS}
    per: dict[str, dict[str, list[float]]] = {m: {r: [] for r in regions} for m in ALL_METHODS}
    for i in range(n_reps):
        s = fseed + i
        regs, x_tr, y_tr, x_te, y_true = M42.build_episode(year_data, tgt_year, k_shot, s)
        if list(regs) != list(regions):
            raise RuntimeError(f"类别顺序不一致：{regs} vs {regions}")
        preds: dict[str, NDArray] = {}
        y_pool = np.concatenate([assets["ys"], y_tr])
        preds["Pooled-SVM"], _ = _svm_fit_predict(np.concatenate([assets["xs"], x_tr]), y_pool, x_te)
        preds["CORAL-SVM"], _ = _svm_fit_predict(np.concatenate([assets["xs_coral"], x_tr]), y_pool, x_te)
        tca = assets["tca"]
        preds["TCA-SVM"], _ = _svm_fit_predict(
            np.concatenate([assets["zs"], tca.transform(x_tr)]), y_pool, tca.transform(x_te)
        )
        for m in NEW_METHODS:
            f1, pc = P20._macro_and_perclass(y_true, [regions[int(j)] for j in preds[m]], regions)
            macro[m].append(f1)
            for r in regions:
                per[m][r].append(pc.get(r, float("nan")))
        for name, (f1, pc) in (
            (CHECK_METHOD, P20._eval_sklearn(year_data, tgt_year, k_shot, s, "svm")),
            (NC_METHOD, P20.eval_nearest_centroid(year_data, tgt_year, k_shot, s)),
        ):
            macro[name].append(f1)
            for r in regions:
                per[name][r].append(pc.get(r, float("nan")))
    return {"macro": macro, "per": per, "regions": regions}


def _task(
    task: tuple[int, int, int, int, dict[str, Any]],
) -> tuple[tuple[int, int, int], dict[str, Any], dict[str, Any]]:
    tgt_year, k_shot, fseed, n_reps, cfg = task
    key = (tgt_year, fseed)
    if key not in _W["assets"]:
        _W["assets"][key] = fold_assets(_W["year_data"], tgt_year, fseed, cfg)
    assets = _W["assets"][key]
    res = run_unit(_W["year_data"], assets, tgt_year, k_shot, fseed, n_reps)
    info = {"n_src_per_class": assets["n_src_per_class"], "n_target_unlabelled": assets["n_target_unlabelled"]}
    return (tgt_year, k_shot, fseed), res, info


# ═══════════════════════════════════════════════════════════════════════════════
# 三、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="65_domain_adaptation_baselines：经典跨域基线 K-Shot 面板")
    parser.add_argument("--k_list", default="1,3,5,10,20,50")
    parser.add_argument("--seeds", type=int, default=50, help="每 (折, K, 正式种子) 的情节重采样次数")
    parser.add_argument("--seed_list", default=",".join(str(s) for s in eu.RECOMMENDED_SEEDS))
    parser.add_argument("--src_per_class", type=int, default=200)
    parser.add_argument("--tca_fit", type=int, default=800)
    parser.add_argument("--tca_dim", type=int, default=30)
    parser.add_argument("--tca_mu", type=float, default=1.0)
    parser.add_argument("--coral_lam", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--selfcheck_tol", type=float, default=5e-4, help="SVM 按 (折,K) 5 种子均值的容差")
    parser.add_argument("--nc_tol", type=float, default=1e-4, help="最近质心逐 (折,K,种子) 格的容差")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_list.split(",")]
    formal_seeds = [int(s) for s in args.seed_list.split(",")]
    anchor_seed = formal_seeds[0]
    cfg = {
        "src_per_class": args.src_per_class,
        "tca_fit": args.tca_fit,
        "tca_dim": args.tca_dim,
        "tca_mu": args.tca_mu,
        "coral_lam": args.coral_lam,
    }
    logger = eu.get_logger("65_domain_adaptation_baselines")
    eu.set_random_seed(anchor_seed, logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "经典跨域基线（Pooled-SVM / CORAL-SVM / TCA-SVM）K-Shot 面板",
            "阶段": f"{args.stage}（纯 CPU sklearn/numpy）",
            "正式种子": formal_seeds,
            "每折每K重采样": args.seeds,
            "K": k_values,
            "配置": cfg,
            "并行进程": args.workers,
        },
    )
    df = eu.load_all_elements_df(verify_sha256=True)
    year_data = build_year_region_data(df, eu.get_spectrum_columns(df))

    tasks = [(y, k, s, args.seeds, cfg) for y in FOLDS for s in formal_seeds for k in k_values]
    results: dict[tuple[int, int, int], dict[str, Any]] = {}
    infos: dict[tuple[int, int], dict[str, Any]] = {}
    with eu.track_compute("经典跨域基线面板", logger) as budget:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_worker_init) as ex:
            for n_done, (key, res, info) in enumerate(ex.map(_task, tasks), start=1):
                results[key] = res
                infos[(key[0], key[2])] = info
                if n_done % 9 == 0 or n_done == len(tasks):
                    logger.log(f"  完成 {n_done}/{len(tasks)} 个单元")

    # ── 原始结果（每单元一行：情节均值与情节 SD）────────────────────────────
    raw_rows: list[dict[str, Any]] = []
    for (y, k, s), res in sorted(results.items()):
        for m in ALL_METHODS:
            row: dict[str, Any] = {
                "目标年份": y,
                "K": k,
                "种子": s,
                "方法": m,
                "F1均值": round(float(np.mean(res["macro"][m])), 4),
                "F1情节SD": round(float(np.std(res["macro"][m])), 4),
                "情节数": len(res["macro"][m]),
            }
            for r in res["regions"]:
                row[f"F1类_{r}"] = round(float(np.nanmean(res["per"][m][r])), 4)
            raw_rows.append(row)
    raw_df = pd.DataFrame(raw_rows)

    # ── 自校验：① 最近质心逐格（严格同情节）② SVM 按 (折, K) 5 种子均值 ──────────
    formal = M42.load_formal_panel()
    if formal is None:
        logger.close()
        raise RuntimeError("自校验无法执行：未找到 20 号脚本的 Formal 归档，新列与表 S6 的同源性无从验证，拒绝写出。")

    def archived(m: str, y: int, k: int, s: int) -> float | None:
        hit = formal[(formal["方法"] == m) & (formal["目标年份"] == y) & (formal["K"] == k) & (formal["种子"] == s)]
        return float(hit["F1均值"].iloc[0]) if len(hit) == 1 else None

    check_rows: list[dict[str, Any]] = []
    for (y, k, s), res in sorted(results.items()):
        for m in (NC_METHOD, CHECK_METHOD):
            theirs = archived(m, y, k, s)
            if theirs is not None:
                mine = float(np.mean(res["macro"][m]))
                check_rows.append(
                    {
                        "口径": "逐格",
                        "方法": m,
                        "目标年份": y,
                        "K": k,
                        "种子": s,
                        "本机复算": round(mine, 5),
                        "Formal归档": round(theirs, 5),
                        "绝对差": round(abs(mine - theirs), 5),
                    }
                )
    for y in FOLDS:
        for k in k_values:
            mine_l = [float(np.mean(results[(y, k, s)]["macro"][CHECK_METHOD])) for s in formal_seeds]
            theirs_l = [archived(CHECK_METHOD, y, k, s) for s in formal_seeds]
            if all(t is not None for t in theirs_l):
                mine = float(np.mean(mine_l))
                theirs = float(np.mean([t for t in theirs_l if t is not None]))
                check_rows.append(
                    {
                        "口径": "折×K 五种子均值",
                        "方法": CHECK_METHOD,
                        "目标年份": y,
                        "K": k,
                        "种子": "5种子均值",
                        "本机复算": round(mine, 5),
                        "Formal归档": round(theirs, 5),
                        "绝对差": round(abs(mine - theirs), 5),
                    }
                )
    check_df = pd.DataFrame(check_rows)
    nc_cells = check_df[(check_df["口径"] == "逐格") & (check_df["方法"] == NC_METHOD)]
    svm_folds = check_df[check_df["口径"] == "折×K 五种子均值"]
    svm_cells = check_df[(check_df["口径"] == "逐格") & (check_df["方法"] == CHECK_METHOD)]
    expected = len(FOLDS) * len(k_values) * len(formal_seeds)
    if len(nc_cells) != expected or len(svm_folds) != len(FOLDS) * len(k_values):
        logger.close()
        raise RuntimeError(
            f"自校验覆盖不全：最近质心 {len(nc_cells)}/{expected} 格，SVM {len(svm_folds)} 个折×K。拒绝写出。"
        )
    nc_worst = float(nc_cells["绝对差"].max())
    svm_worst = float(svm_folds["绝对差"].max())
    logger.log(
        f"\n── 自校验：最近质心逐格 {len(nc_cells)} 格 max|Δ|={nc_worst:.5f}（容差 {args.nc_tol}）；"
        f"SVM 折×K 均值 {len(svm_folds)} 格 max|Δ|={svm_worst:.5f}（容差 {args.selfcheck_tol}）；"
        f"SVM 逐格 max|Δ|={float(svm_cells['绝对差'].max()):.5f}（照录）"
    )
    if nc_worst > args.nc_tol or svm_worst > args.selfcheck_tol:
        logger.log(check_df.sort_values("绝对差", ascending=False).head(10).to_string(index=False))
        logger.close()
        raise RuntimeError("自校验 FAIL：情节与表 S6 不同源，拒绝写出。")
    logger.log("  自校验 PASS：新增路线与表 S6 的九条路线处在同一套情节上。")

    # ── 参照列：Formal 归档的原型网络与只看支持集的 SVM（折级，5 种子均值）──────
    ref_fold: dict[tuple[str, int, int], float] = {}
    for name in ("VanillaProtoNet", CHECK_METHOD):
        g = formal[formal["方法"] == name].groupby(["目标年份", "K"])["F1均值"].mean()
        ref_fold.update({(name, int(a), int(b)): float(v) for (a, b), v in g.items()})

    def fold_mean(m: str, y: int, k: int, seed_slice: list[int] | None = None) -> float:
        seeds = seed_slice if seed_slice is not None else formal_seeds
        return float(np.mean([np.mean(results[(y, k, s)]["macro"][m]) for s in seeds]))

    foldmean_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    sd_rows: list[dict[str, Any]] = []
    for k in k_values:
        frow: dict[str, Any] = {"K": k}
        srow: dict[str, Any] = {"K": k}
        drow: dict[str, Any] = {"K": k}
        for m in NEW_METHODS:
            for y in FOLDS:
                frow[f"{m}@{y}"] = round(fold_mean(m, y, k), 4)
            srow[m] = round(float(np.mean([fold_mean(m, y, k) for y in FOLDS])), 4)
            per_seed = [float(np.mean([fold_mean(m, y, k, [s]) for y in FOLDS])) for s in formal_seeds]
            drow[m] = round(float(np.std(per_seed)), 4)
        for name in ("VanillaProtoNet", CHECK_METHOD):
            srow[f"{name}(Formal归档)"] = round(float(np.mean([ref_fold[(name, y, k)] for y in FOLDS])), 4)
        foldmean_rows.append(frow)
        summary_rows.append(srow)
        sd_rows.append(drow)
    foldmean_df = pd.DataFrame(foldmean_rows)
    summary_df = pd.DataFrame(summary_rows)
    sd_df = pd.DataFrame(sd_rows)
    logger.log("\n── 三折均值（新增路线 + Formal 归档参照列）──")
    logger.log(summary_df.to_string(index=False))
    logger.log("\n── 折级均值 ──")
    logger.log(foldmean_df.to_string(index=False))

    # ── 折级配对检验（n_primary=3，Nadeau-Bengio 修正，Holm 跨 6 个 K）────────────
    test_rows: list[dict[str, Any]] = []
    for ref in ("VanillaProtoNet", CHECK_METHOD):
        for m in NEW_METHODS:
            p_raw: list[float] = []
            tmp: list[dict[str, Any]] = []
            for k in k_values:
                a = [fold_mean(m, y, k) for y in FOLDS]
                b = [ref_fold[(ref, y, k)] for y in FOLDS]
                cmp = eu.compare_methods(a, b, n_train=100, n_test=50, n_bootstrap=2000, seed=anchor_seed)
                p_raw.append(cmp["p_value"])
                tmp.append({"K": k, **cmp})
            holm = eu.apply_multi_comp_correction(p_raw, method="holm")
            for rr, p_adj, rej in zip(tmp, holm["p_adjusted"], holm["reject"], strict=True):
                test_rows.append(
                    {
                        "比较": f"{m} vs {ref}",
                        "K": rr["K"],
                        "n_primary": 3,
                        "均值差(折级)": round(rr["mean_diff"], 4),
                        "折级区间下": round(rr["ci_low"], 4),
                        "折级区间上": round(rr["ci_high"], 4),
                        "p_raw": round(rr["p_value"], 4),
                        "p_holm": round(p_adj, 4),
                        "Cohen_d": round(rr["cohens_d"], 3),
                        "显著(Holm)": "YES" if rej >= 1.0 else "NO",
                    }
                )
    test_df = pd.DataFrame(test_rows)
    logger.log("\n── 折级配对检验 ──")
    logger.log(test_df.to_string(index=False))

    info_rows = [
        {
            "目标年份": y,
            "种子": s,
            "源域每类条数": str(v["n_src_per_class"]),
            "目标年无标签光谱数": v["n_target_unlabelled"],
        }
        for (y, s), v in sorted(infos.items())
    ]
    resource_df = pd.DataFrame(
        [
            {"路线": "SVM-Kshot（既有）", "源域标签": "否", "目标年无标签光谱": "否", "目标年标签": "K 个支持样本"},
            {
                "路线": "VanillaProtoNet（既有）",
                "源域标签": "是（元训练）",
                "目标年无标签光谱": "否",
                "目标年标签": "K 个支持样本",
            },
            {"路线": "Pooled-SVM", "源域标签": "是", "目标年无标签光谱": "否", "目标年标签": "K 个支持样本"},
            {
                "路线": "CORAL-SVM",
                "源域标签": "是",
                "目标年无标签光谱": "是（对齐二阶统计）",
                "目标年标签": "K 个支持样本",
            },
            {
                "路线": "TCA-SVM",
                "源域标签": "是",
                "目标年无标签光谱": "是（联合求隐空间）",
                "目标年标签": "K 个支持样本",
            },
        ]
    )

    # ── 代表性模型落盘（LOEO-2025, K=50, 首个正式种子的首个情节）────────────────
    assets = fold_assets(year_data, 2025, anchor_seed, cfg)
    regs, x_tr, y_tr, _x_te, _ = M42.build_episode(year_data, 2025, 50, anchor_seed)
    y_pool = np.concatenate([assets["ys"], y_tr])
    for route, x_src, mapper in (
        ("coral", assets["xs_coral"], None),
        ("tca", assets["zs"], assets["tca"]),
    ):
        x_sup = mapper.transform(x_tr) if mapper is not None else x_tr
        _, clf = _svm_fit_predict(np.concatenate([x_src, x_sup]), y_pool, x_sup[:1])
        eu.save_sklearn_checkpoint(
            {"svm": clf, "tca_map": mapper, "regions": regs},
            f"65_da_{route}_fold2025_K50_seed{anchor_seed}",
            {
                "arch": f"{route.upper()} + Pipeline(StandardScaler, SVC(rbf, C=10, gamma=scale, balanced))",
                "params": f"n_support_vectors={int(clf.named_steps['svm'].n_support_.sum())}",
                "flops": "N/A（sklearn 估计器，非逐层前向）",
                "stage": args.stage,
                "training_hardware": "cpu",
                "deterministic_guarantee": "full(CPU sklearn/numpy)",
                "seed": anchor_seed,
                "task": "LOEO-2025, K=50 代表性情节",
            },
            extra_state={"cfg": cfg},
        )

    budget_df = pd.DataFrame(
        [{"正式种子数": eu.n_distinct_seeds(tuple(formal_seeds)), "每折每K重采样": args.seeds, **budget}]
    )
    out_path = os.path.join(eu.OUTPUT_DIR, "65_domain_adaptation_baselines.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name=eu.table_sheet_name(0, "原始结果"), index=False)
        check_df.to_excel(writer, sheet_name=eu.table_sheet_name(1, "自校验"), index=False)
        foldmean_df.to_excel(writer, sheet_name=eu.table_sheet_name(2, "折级均值"), index=False)
        summary_df.to_excel(writer, sheet_name=eu.table_sheet_name(3, "三折均值"), index=False)
        sd_df.to_excel(writer, sheet_name=eu.table_sheet_name(4, "跨种子SD"), index=False)
        test_df.to_excel(writer, sheet_name=eu.table_sheet_name(5, "折级配对检验"), index=False)
        resource_df.to_excel(writer, sheet_name=eu.table_sheet_name(6, "各路线所用信息"), index=False)
        pd.DataFrame(info_rows).to_excel(writer, sheet_name=eu.table_sheet_name(7, "折级资产规模"), index=False)
        budget_df.to_excel(writer, sheet_name="compute", index=False)
    logger.log(f"\n结果表 → {out_path}")

    # ── 图：新增路线与参照列的 K-Shot 曲线 ─────────────────────────────────────
    colours = {"Pooled-SVM": "#999999", "CORAL-SVM": "#CC79A7", "TCA-SVM": "#E69F00"}
    txt = {
        "zh": {"x": "K-Shot（每产地支持苹果数）", "y": "Macro-F1（三折均值）", "r": "随机基线"},
        "en": {"x": "K-shot (support apples per origin)", "y": "Macro-F1 (3-fold mean)", "r": "Chance"},
    }

    def plot(lang: str) -> Any:
        t = txt[lang]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for m in NEW_METHODS:
            ax.plot(k_values, summary_df[m], marker="o", linewidth=2, color=colours[m], label=m)
        ax.plot(
            k_values,
            summary_df["VanillaProtoNet(Formal归档)"],
            marker="s",
            linestyle="--",
            linewidth=2,
            color="#0072B2",
            label="ProtoNet",
        )
        ax.plot(
            k_values,
            summary_df[f"{CHECK_METHOD}(Formal归档)"],
            marker="^",
            linestyle="--",
            linewidth=2,
            color="#009E73",
            label="SVM (support only)",
        )
        ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.5, label=t["r"])
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel(t["x"])
        ax.set_ylabel(t["y"])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", ncol=2, frameon=False)
        fig.tight_layout()
        return fig

    eu.save_figure_bilingual(plot, "65_domain_adaptation_baselines", 0)
    logger.close()


if __name__ == "__main__":
    main()
