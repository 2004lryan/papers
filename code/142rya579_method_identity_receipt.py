#!/usr/bin/env python3
r"""
142 — RYA-579：苹果两阶段管线与方法节正过程的对应关系，机械核验。

背景
----
取证 F002（HP-METHOD-DRIFT）指出：方法节写的正过程是 β=1−exp(−kt)、θ=(φ,log A)，
而苹果管线实跑的是 54c5_realkinetics.py 的 y=y0·exp(−kt)（y0 自由）+ ln k 对 1/(RT)
的线性 Arrhenius 回归（设计列 [−1/(RT),1]）。

本脚本核验三件事，用来支撑稿面方法节新增的对应关系段落：

  A 设计向量同一：φ=E_a/(R T_ref)、r=T_ref/T 时 E_a/(RT)=φ·r，故两套坐标下的
    Fisher 信息阵由固定合同变换 D=diag(1/(R T_ref),1) 相联：I^(E_a,lnA) = D·I^(φ,logA)·D。
  B λ_min 不是合同不变量，故其数值随参数化而变；但在本文所用的 26 个苹果温度子集上
    两种参数化给出的排序是否一致，须实测，不得当作数学事实。
  C 权重对应：理论侧逐观测权重 w=g²/σ²（g=∂β/∂log k）在每个温度上累加，
    恰等于 y0 已知时 ln k 估计精度 1/Var(ln k̂)；这是 delta 方法的恒等式。
    两处诚实偏差：y0 实际是估计的（自由 y0 只会抬高 Var(ln k̂)），
    σ_lnk 取自实测残差而非模型。

输出 04outputs/142rya579_method_identity_receipt.json
"""
from __future__ import annotations

import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"
SWEEP = OUT / "54c5_realkinetics-designsweep.csv"
R_GAS = 8.314


def fishers(T_C, Tref_C, sigma=1.0):
    """返回 (代码坐标 (E_a,lnA) 的 Fisher, 理论坐标 (φ,log A) 的 Fisher)。"""
    T = np.asarray(T_C, float) + 273.15
    Tref = Tref_C + 273.15
    J_code = np.stack([-1.0 / (R_GAS * T), np.ones_like(T)], axis=1) / sigma
    J_theory = np.stack([-Tref / T, np.ones_like(T)], axis=1) / sigma
    return J_code.T @ J_code, J_theory.T @ J_theory


def check_A(Tref_C=6.0):
    T_C = [2.0, 4.0, 6.0, 8.0, 10.0]
    I_c, I_t = fishers(T_C, Tref_C)
    D = np.diag([1.0 / (R_GAS * (Tref_C + 273.15)), 1.0])
    resid = float(np.linalg.norm(I_c - D @ I_t @ D))
    return {
        "T_C": T_C,
        "T_ref_C": Tref_C,
        "congruence": "I^(E_a,lnA) = D I^(phi,logA) D, D=diag(1/(R T_ref),1)",
        "frobenius_residual": resid,
        "lmin_code_coords": float(np.linalg.eigvalsh(I_c)[0]),
        "lmin_theory_coords": float(np.linalg.eigvalsh(I_t)[0]),
        "AGREES": resid < 1e-12,
    }


def check_B():
    rows = list(csv.DictReader(SWEEP.open()))
    out = {}
    for Tref_C in (0.0, 6.0, 25.0):
        lc, lt = [], []
        for r in rows:
            T_C = [float(x) for x in r["temps"].split("/")]
            I_c, I_t = fishers(T_C, Tref_C)
            lc.append(np.linalg.eigvalsh(I_c)[0])
            lt.append(np.linalg.eigvalsh(I_t)[0])
        lc, lt = np.array(lc), np.array(lt)
        disc = sum(
            1 for i, j in combinations(range(len(lc)), 2) if (lc[i] - lc[j]) * (lt[i] - lt[j]) < 0
        )
        out[f"T_ref={Tref_C}C"] = {
            "n_subsets": len(lc),
            "spearman": float(spearmanr(lc, lt).statistic),
            "discordant_pairs": disc,
            "total_pairs": len(lc) * (len(lc) - 1) // 2,
        }
    psi = np.array([float(r["Psi_n"]) for r in rows])
    lc = []
    for r in rows:
        T_C = [float(x) for x in r["temps"].split("/")]
        lc.append(np.linalg.eigvalsh(fishers(T_C, 6.0)[0])[0])
    out["paper_Psi_n_vs_code_coord_lmin_spearman"] = float(spearmanr(psi, np.array(lc)).statistic)
    out["NOTE"] = "lambda_min 不是合同不变量；此处是这批设计上的实测排序一致性，不是数学不变性"
    return out


def check_C():
    """delta 方法恒等式：sum_j w_beta,j == 1/Var(ln k_hat | y0 已知)，并给出自由 y0 的抬高倍数。"""
    k, y0, sy = 2e-3, 70.0, 0.5
    t = np.array([0.0, 12.0, 24.0, 36.0, 48.0])
    g = k * t * np.exp(-k * t)              # g = d beta / d log k
    w_sum = float((y0 ** 2) * np.sum(g ** 2) / sy ** 2)   # sum_j w_beta,j, w=g^2/sigma_beta^2, sigma_beta=sy/y0
    d_logk = -y0 * g                        # dy/d log k
    d_y0 = np.exp(-k * t)                   # dy/d y0
    F = np.array([[d_logk @ d_logk, d_logk @ d_y0], [d_logk @ d_y0, d_y0 @ d_y0]]) / sy ** 2
    inv_var_known = float(F[0, 0])
    var_free = float(np.linalg.inv(F)[0, 0])
    return {
        "示例设置": {"k": k, "y0": y0, "sigma_y": sy, "t_h": t.tolist()},
        "sum_w_beta": w_sum,
        "inv_Var_logk_y0_known": inv_var_known,
        "relative_residual": abs(inv_var_known - w_sum) / w_sum,
        "AGREES": abs(inv_var_known - w_sum) / w_sum < 1e-10,
        "free_y0_variance_inflation": var_free / (1.0 / inv_var_known),
        "NOTE": "该倍数依设计而变，仅为示例，不作为稿面数字",
    }


def main() -> int:
    rec = {
        "任务": "RYA-579",
        "用途": "支撑稿面方法节新增的『真实体系两阶段管线与正过程的对应关系』段落（取证 F002）",
        "输入": {str(SWEEP.relative_to(ROOT)): hashlib.sha256(SWEEP.read_bytes()).hexdigest()},
        "A_设计向量与合同变换": check_A(),
        "B_参数化对排序的影响": check_B(),
        "C_权重的delta方法对应": check_C(),
    }
    rec["结论"] = (
        "A 合同恒等式残差 %.3g（判定 %s）；"
        "B 26 个子集上两种参数化排序不一致对数 T_ref=0/6/25C 分别为 %d/%d/%d（共 325 对）；"
        "C delta 方法恒等式相对残差 %.3g（判定 %s）"
        % (
            rec["A_设计向量与合同变换"]["frobenius_residual"],
            "AGREES" if rec["A_设计向量与合同变换"]["AGREES"] else "FAILS",
            rec["B_参数化对排序的影响"]["T_ref=0.0C"]["discordant_pairs"],
            rec["B_参数化对排序的影响"]["T_ref=6.0C"]["discordant_pairs"],
            rec["B_参数化对排序的影响"]["T_ref=25.0C"]["discordant_pairs"],
            rec["C_权重的delta方法对应"]["relative_residual"],
            "AGREES" if rec["C_权重的delta方法对应"]["AGREES"] else "FAILS",
        )
    )
    dst = OUT / "142rya579_method_identity_receipt.json"
    dst.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(rec["结论"])
    print("->", dst.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
