"""29_g2_theory_validation.py: 命题条件的正式检验 + 真实嵌入有限K机制验证。

回应审稿 D2 的两步（把理论诊断从描述性升为正式检验、把有限K从仿真升为真实嵌入）：
(A) 逐折记录 (坐标支配比例, 相关结构变化, 实际泄漏 Δ@K5)，做：
    ①支配比例与实际 Δ 的 Spearman 相关 + 置换检验 p（1000 次打乱）；
    ②支配比例的 cluster/percentile bootstrap 95% CI；
    ③负 Δ 折是否显著低支配（Mann-Whitney）。
(B) 在\emph{真实目标域嵌入}（共享 vs 折限定编码器）上重采样有限 K 原型，K∈{1,3,5,10,20}，
    逐 K 算 Δ(K)=共享−折限定 macro-F1，检验 Δ≥0 在真实神经嵌入的有限 K regime 成立，
    并核查坐标支配失败折与 Δ<0 折是否同步。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE python 02code/29_g2_theory_validation.py --device auto

输出文件:
    04outputs/29_g2_theory_validation.xlsx  — 逐折条件-Δ表 + 正式检验 + 真实有限K表
    05logs/29_g2_theory_validation_YY-MM-DD_HHMMSS.log — 运行日志
"""

from __future__ import annotations

import argparse
import importlib
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

import export_utils as eu

warnings.filterwarnings("ignore")

NDArray = npt.NDArray[Any]
DATA_ROOT = "05data"


def kshot_f1_from_emb(
    emb: NDArray, labels: NDArray, classes: list[int], k: int, reps: int, rng: np.random.Generator
) -> float:
    """在给定嵌入上重采样 K-shot 最近原型 macro-F1（真实嵌入有限 K）。"""
    from sklearn.metrics import f1_score

    f1s: list[float] = []
    for _ in range(reps):
        proto = {}
        used: set[int] = set()
        ok = True
        for c in classes:
            idx = np.where(labels == c)[0].copy()
            rng.shuffle(idx)
            if len(idx) < k + 2:
                ok = False
                break
            sup = idx[:k]
            used.update(sup.tolist())
            p = emb[sup].mean(0)
            proto[c] = p / (np.linalg.norm(p) + 1e-8)
        if not ok:
            continue
        q = [i for i in range(len(labels)) if i not in used and labels[i] in classes]
        if not q:
            continue
        pmat = np.stack([proto[c] for c in classes])
        pred = np.array(classes)[(emb[q] @ pmat.T).argmax(1)]
        f1s.append(f1_score(np.array(labels)[q], pred, average="macro", labels=classes))
    return float(np.mean(f1s)) if f1s else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="命题条件正式检验+真实嵌入有限K")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--reps", type=int, default=40)
    args = parser.parse_args()
    device = eu.get_device(args.device)
    logger = eu.get_logger("29_g2_theory_validation")
    m24 = importlib.import_module("24_crossdataset_leakage")
    m28 = importlib.import_module("28_g2_theory_diagnostic")
    import torch

    ks = [1, 3, 5, 10, 20]
    fold_rows: list[dict[str, Any]] = []
    fink_rows: list[dict[str, Any]] = []
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

            def build(dl: list[str], _l: Any = lab, _c: Any = classes, _d: Any = doms) -> tuple[NDArray, NDArray]:
                feats = m24.snv(np.vstack([_d[x][0] for x in dl]))
                labels = np.concatenate([_l(_d[x][1]) for x in dl])
                mask = np.isin(labels, _c)
                return feats[mask], labels[mask]

            xs, ys2 = build(src)
            xall, yall = build(names)
            xh, yh2 = build([held])
            enc_h = m24.meta_train(xall, yall, in_dim, device, args.episodes, 2004)
            enc_r = m24.meta_train(xs, ys2, in_dim, device, args.episodes, 2004)
            with torch.no_grad():
                eh = enc_h(torch.from_numpy(xh).float().to(device)).cpu().numpy()
                er = enc_r(torch.from_numpy(xh).float().to(device)).cpu().numpy()
            s_h, _ = m28.separability_and_corr(eh, yh2, classes)
            s_r, _ = m28.separability_and_corr(er, yh2, classes)
            dom = float(np.mean([s_h[p] >= s_r[p] for p in s_h]))
            # 真实嵌入有限 K：逐 K Δ=共享−折限定
            delta_k = {}
            for k in ks:
                f_h = kshot_f1_from_emb(eh, yh2, classes, k, args.reps, np.random.default_rng(2004))
                f_r = kshot_f1_from_emb(er, yh2, classes, k, args.reps, np.random.default_rng(2004))
                delta_k[k] = f_h - f_r
                fink_rows.append({"dataset": dsname, "held": held, "K": k, "delta_real": f_h - f_r})
            fold_rows.append({"dataset": dsname, "held": held, "dominance": dom,
                              "delta_k5": delta_k[5], "n_class": len(classes)})
            logger.log(f"  [{dsname} {held}] 支配={dom:.2f} Δ@K5(真实嵌入)={delta_k[5]:+.3f}")

    fold = pd.DataFrame(fold_rows)
    # (A) 正式检验：支配 vs 实际Δ Spearman + 置换 p
    rho, _ = stats.spearmanr(fold["dominance"], fold["delta_k5"])
    rng = np.random.default_rng(20060515)
    perm = [stats.spearmanr(fold["dominance"], rng.permutation(fold["delta_k5"].to_numpy()))[0] for _ in range(1000)]
    p_perm = float((np.abs(perm) >= abs(rho)).mean())
    # 支配比例 bootstrap CI
    boot = [float(np.mean(rng.choice(fold["dominance"].to_numpy(), len(fold), replace=True))) for _ in range(2000)]
    dom_ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    # 负Δ折 vs 正Δ折 支配差异
    neg = fold[fold["delta_k5"] < 0]["dominance"]
    pos = fold[fold["delta_k5"] >= 0]["dominance"]
    u_p = float(stats.mannwhitneyu(neg, pos, alternative="less").pvalue) if len(neg) and len(pos) else float("nan")
    logger.log(f"[正式检验] 支配-Δ Spearman ρ={rho:+.3f} 置换p={p_perm:.3f}; "
               f"支配均值={fold['dominance'].mean():.3f} 95%CI[{dom_ci[0]:.3f},{dom_ci[1]:.3f}]; "
               f"负折支配<正折 Mann-Whitney p={u_p:.3f}")

    # (B) 真实嵌入有限 K 汇总
    finl = pd.DataFrame(fink_rows)
    fk = finl.groupby("K").agg(E_delta_real=("delta_real", "mean"),
                               P_pos=("delta_real", lambda s: float((s >= 0).mean()))).reset_index()
    fkmsg = "; ".join(f"K={int(r.K)}:E[Δ]={r.E_delta_real:+.3f}(P≥0={r.P_pos:.2f})" for _, r in fk.iterrows())
    logger.log("[真实嵌入有限K] " + fkmsg)

    tests = pd.DataFrame([{
        "支配-Δ_Spearman_rho": rho, "置换p": p_perm,
        "支配均值": fold["dominance"].mean(), "支配CI下": dom_ci[0], "支配CI上": dom_ci[1],
        "负折支配<正折_MannWhitney_p": u_p,
    }])
    sheets: dict[object, object] = {
        eu.table_sheet_name(1, "逐折条件Δ"): fold,
        eu.table_sheet_name(2, "正式检验"): tests,
        eu.table_sheet_name(3, "真实嵌入有限K"): fk,
    }
    eu.write_script_workbook(__file__, sheets)
    logger.log("已写 04outputs/29_g2_theory_validation.xlsx")


if __name__ == "__main__":
    main()
