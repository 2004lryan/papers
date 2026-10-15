"""28_g2_theory_diagnostic.py: 理论-实验衔接诊断（回应"理论未覆盖实际 regime"）。

两部分：
(A) 命题 1 充分条件在真实多类数据上的经验诊断。对每个数据集的留一域折，训练共享(W_H)与
    折限定(W_R)编码器，在\emph{目标域嵌入}中计算各类对可分性 s_{cc'}(W;T)（沿决策法向的
    信噪比），报告：①坐标支配比例 = s_{cc'}(W_H)≥s_{cc'}(W_R) 的类对占比（命题 1 条件 ii）；
    ②标准化边际相关结构变化 ||R_c(W_H)-R_c(W_R)||_F 的均值（命题 1 条件 i，越小越支持）。
(B) 有限 K 仿真：线性-高斯最近原型模型下，用\emph{有限 K 样本原型}（非总体原型）经验估计
    E[Delta]，展示 Delta>=0 在超出总体原型极限的有限 K regime 依然成立。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE python 02code/28_g2_theory_diagnostic.py --device auto

输出文件:
    04outputs/28_g2_theory_diagnostic.xlsx  — 条件诊断表 + 有限K仿真表
    05logs/28_g2_theory_diagnostic_YY-MM-DD_HHMMSS.log — 运行日志
"""

from __future__ import annotations

import argparse
import importlib
import warnings
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

import export_utils as eu

warnings.filterwarnings("ignore")

NDArray = npt.NDArray[Any]
DATA_ROOT = "05data"


def separability_and_corr(
    emb: NDArray, labels: NDArray, classes: list[int]
) -> tuple[dict[tuple[int, int], float], float]:
    """在嵌入空间计算各类对可分性 s_{cc'} 与类内标准化边际相关矩阵的平均 Frobenius 范数基。

    返回 (s_dict[(c,c')]=信噪比, R_repr) —— R_repr 为各类相关矩阵的堆叠（供两协议间比较）。
    """
    means = {c: emb[labels == c].mean(0) for c in classes}
    cov = np.cov(emb.T) + 1e-6 * np.eye(emb.shape[1])
    s_dict: dict[tuple[int, int], float] = {}
    normals: dict[tuple[int, int], NDArray] = {}
    for c, cp in combinations(classes, 2):
        diff = means[c] - means[cp]
        nrm = float(np.linalg.norm(diff))
        d = diff / (nrm + 1e-8)
        proj_var = float(d @ cov @ d)
        s_dict[(c, cp)] = nrm / (np.sqrt(proj_var) + 1e-8)
        normals[(c, cp)] = d
    return s_dict, 0.0


def corr_change(emb_h: NDArray, emb_r: NDArray, labels: NDArray, classes: list[int]) -> float:
    """命题 1 条件 i 诊断：各类标准化边际相关矩阵在两协议间的平均 Frobenius 差（相对范数）。"""
    diffs = []
    for prot_c in classes:
        others = [x for x in classes if x != prot_c]
        if len(others) < 2:
            continue
        rmats = []
        for emb in (emb_h, emb_r):
            cov = np.cov(emb.T) + 1e-6 * np.eye(emb.shape[1])
            means = {c: emb[labels == c].mean(0) for c in classes}
            normals = []
            for cp in others:
                diff = means[prot_c] - means[cp]
                normals.append(diff / (np.linalg.norm(diff) + 1e-8))
            nmat = np.stack(normals)
            gram = nmat @ cov @ nmat.T
            dsd = np.sqrt(np.diag(gram))
            rmat = gram / (np.outer(dsd, dsd) + 1e-12)
            rmats.append(rmat)
        rel = float(np.linalg.norm(rmats[0] - rmats[1]) / (np.linalg.norm(rmats[1]) + 1e-8))
        diffs.append(rel)
    return float(np.mean(diffs)) if diffs else float("nan")


def finite_k_sim(seed: int) -> pd.DataFrame:
    """(B) 线性-高斯最近原型有限 K 仿真：漂移下共享 vs 折限定，经验 E[Delta] 随 K。"""
    rng = np.random.default_rng(seed)
    dim, n_class = 20, 3
    rows = []
    for k in [1, 3, 5, 10, 20]:
        deltas = []
        for _ in range(200):
            # 源/目标类均值：目标含漂移(旋转+缩放)使源最优≠目标最优
            base = rng.normal(0, 1, (n_class, dim))
            drift = rng.normal(0, 0.6, (n_class, dim))
            mu_src, mu_tgt = base, base + drift
            # 折限定编码器=源域 LDA 方向；共享=源+目标混合方向（含目标信息=泄漏）
            w_r = np.linalg.svd(mu_src - mu_src.mean(0), full_matrices=False)[2][:n_class]
            pooled = np.vstack([mu_src, mu_tgt])
            w_h = np.linalg.svd(pooled - pooled.mean(0), full_matrices=False)[2][:n_class]
            acc = {}
            for name, w in (("R", w_r), ("H", w_h)):
                correct = 0
                total = 0
                for c in range(n_class):
                    # K 样本原型（有限 K，非总体）
                    proto_list = []
                    for cc in range(n_class):
                        sup = rng.normal(mu_tgt[cc], 1.0, (k, dim)) @ w.T
                        proto_list.append(sup.mean(0))
                    protos = np.stack(proto_list)
                    q = rng.normal(mu_tgt[c], 1.0, (50, dim)) @ w.T
                    pred = np.argmin(((q[:, None, :] - protos[None]) ** 2).sum(-1), axis=1)
                    correct += int((pred == c).sum())
                    total += len(q)
                acc[name] = correct / total
            deltas.append(acc["H"] - acc["R"])
        arr = np.array(deltas)
        rows.append({"K": k, "E_Delta": float(arr.mean()), "Delta_std": float(arr.std()),
                     "P_Delta>=0": float((arr >= 0).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="理论-实验衔接诊断")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=800)
    args = parser.parse_args()
    device = eu.get_device(args.device)
    logger = eu.get_logger("28_g2_theory_diagnostic")
    m24 = importlib.import_module("24_crossdataset_leakage")

    # (A) 真实多类数据条件诊断
    diag_rows: list[dict[str, Any]] = []
    for dsname, meta in m24.DATASETS.items():
        doms, is_reg = meta["loader"](DATA_ROOT, 2004)
        names = list(doms.keys())
        for held in names:
            src = [d for d in names if d != held]
            y_src = np.concatenate([doms[d][1] for d in src])

            def lab(v: NDArray, _ys: NDArray = y_src, _reg: bool = is_reg) -> NDArray:
                return m24.tertile_labels(_ys, v) if _reg else v.astype(int)

            ys = lab(np.concatenate([doms[d][1] for d in src]))
            yh = lab(doms[held][1])
            classes = sorted(set(np.unique(ys)) & {c for c in np.unique(yh) if (yh == c).sum() >= 8})[:6]
            if len(classes) < 3:
                continue
            in_dim = doms[names[0]][0].shape[1]

            def build(
                dom_list: list[str], _lab: Any = lab, _cls: Any = classes, _doms: Any = doms
            ) -> tuple[NDArray, NDArray]:
                feats = m24.snv(np.vstack([_doms[d][0] for d in dom_list]))
                labels = np.concatenate([_lab(_doms[d][1]) for d in dom_list])
                mask = np.isin(labels, _cls)
                return feats[mask], labels[mask]

            xs, ys2 = build(src)
            xall, yall = build(names)
            xh, yh2 = build([held])
            enc_h = m24.meta_train(xall, yall, in_dim, device, args.episodes, 2004)
            enc_r = m24.meta_train(xs, ys2, in_dim, device, args.episodes, 2004)
            with torch.no_grad():
                eh = enc_h(torch.from_numpy(xh).float().to(device)).cpu().numpy()
                er = enc_r(torch.from_numpy(xh).float().to(device)).cpu().numpy()
            s_h, _ = separability_and_corr(eh, yh2, classes)
            s_r, _ = separability_and_corr(er, yh2, classes)
            dom_frac = float(np.mean([s_h[p] >= s_r[p] for p in s_h]))
            r_chg = corr_change(eh, er, yh2, classes)
            diag_rows.append({"dataset": dsname, "held": held, "n_class": len(classes),
                              "坐标支配比例": dom_frac, "相关结构相对变化": r_chg})
            logger.log(f"  [{dsname} 留出{held}] 坐标支配={dom_frac:.2f} 相关变化={r_chg:.3f} ({len(classes)}类)")

    diag = pd.DataFrame(diag_rows)
    logger.log(f"\n[条件诊断] 坐标支配比例均值={diag['坐标支配比例'].mean():.3f}, "
               f"相关结构相对变化均值={diag['相关结构相对变化'].mean():.3f}")

    # (B) 有限 K 仿真
    sim = finite_k_sim(20060515)
    sim_msg = "; ".join(f"K={r.K}:E[Δ]={r.E_Delta:+.3f}(P≥0={r['P_Delta>=0']:.2f})" for _, r in sim.iterrows())
    logger.log("[有限K仿真] " + sim_msg)

    sheets: dict[object, object] = {
        eu.table_sheet_name(1, "命题1条件诊断"): diag,
        eu.table_sheet_name(2, "有限K仿真"): sim,
    }
    eu.write_script_workbook(__file__, sheets)
    logger.log("已写 04outputs/28_g2_theory_diagnostic.xlsx")


if __name__ == "__main__":
    main()
