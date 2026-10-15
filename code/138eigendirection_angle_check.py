#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
138eigendirection_angle_check.py — 同方差代理与异方差定义量的【最小特征方向】夹角实测

为什么需要这个脚本
------------------
取证义务 `1f9bcf45`（reviewer 原级 critical，FP-RISK 封顶为 major）指出：
`123montecarlo_criterion_study.py` 的头条速率检验把【异方差 Fisher 的特征值】(E_het)
与【同方差代理 Fisher 的最小特征方向】(criteria()["v_deg"]) 配在一起，二者来自两个不同算子。
该缺陷属实，须改为同一算子的特征值与特征方向。

本脚本是改码【之前】的独立几何预检，回答一个可证伪的问题：
    换成异方差自己的特征方向后，头条数字会不会动、动多少？

⚠️ 先订正一个被写进任务书的错误论据
------------------------------------
RYA-569 任务书给出的论据是「J=[-1/(R T_K), 1] 两列量级差约 4 个数量级
（-1/(R T_K)≈-4.2e-4 vs 1），故 I 被 I_22 支配、最小特征方向被钉在 e_1 附近、
任何正对角加权都转不动它」，并据此预测夹角中位 0.000°、最大 0.001°。

**该论据不成立，是单位错误**：4.2e-4 对应 R=8.314 J/(mol·K)，而本项目 E_a 以 kJ/mol 计、
脚本 123 用的是 R_GAS=8.314e-3 kJ/(mol·K)，故实际 1/(R T_K)≈0.384–0.471（T∈[-18,40]℃），
与第二列的 1 **同量级**（差 2.1–2.6 倍），不是 1e4。

真实几何是另一回事：x_i:=1/(R T_K,i) 只在 [0.384,0.471] 这个**窄区间**内变动，
故所有行 J_i=[-x_i,1] 近乎共线 → λmin 小、最小特征方向 v_min ⟂ 加权主方向 [-x̄_w,1]，
即 v_min ≈ [1, x̄_w]/‖·‖。**x̄_w 是 x 的加权均值，权重一变它就变**，
故异方差加权确实会转动 v_min，转动幅度以 arctan(0.471)-arctan(0.384)≈4.2° 为上限。

因此本脚本预测的不是「零偏移」，而是一个**二阶小量**：v_hom 与 v_het 都落在误差协方差
主轴附近，rmse 沿方向 v 的平方为 λ_big cos²θ+λ_small sin²θ，在主轴附近对 θ 是**二阶**平坦，
故 |Δrmse|/rmse ~ O(sin²θ/2)，远小于线性上界 sin θ。

输出
----
04outputs/138eigendirection_angle_check.json
"""
from __future__ import annotations

import importlib.util as _ilu
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs" / "138eigendirection_angle_check.json"

_spec = _ilu.spec_from_file_location("mc123", ROOT / "02code" / "123montecarlo_criterion_study.py")
mc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(mc)

SEED = 20260821
N_RANDOM = 4000


def vmin(M: np.ndarray) -> np.ndarray:
    ev, evec = np.linalg.eigh(M)
    return evec[:, int(np.argmin(ev))]


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """两条【无向】主轴的夹角（度），取 |cos| 消去特征向量的符号任意性。"""
    c = float(abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(min(max(c, -1.0), 1.0))))


def pair(T_C, n_per_T, sigma_i):
    """返回 (夹角°, CRB 下 rmse 比值)。

    CRB 比值 = sqrt(v_hetᵀ M_het⁻¹ v_het / v_homᵀ M_het⁻¹ v_hom)。
    以异方差 Fisher 的逆作为误差协方差的一阶代理（真模型即异方差），
    v_het 恰是该二次型的最大化方向，故该比值 ≥ 1，且直接预测重跑后 rmse_deg_het/rmse_deg。
    """
    T_C = np.asarray(T_C, float); n_per_T = np.asarray(n_per_T, float)
    sigma_i = np.asarray(sigma_i, float)
    M_hom = mc.fisher(T_C, n_per_T, float(np.mean(sigma_i)))
    M_het = mc.fisher_het(T_C, n_per_T, sigma_i)
    v_hom, v_het = vmin(M_hom), vmin(M_het)
    C = np.linalg.inv(M_het)
    q_het = float(v_het @ C @ v_het)
    q_hom = float(v_hom @ C @ v_hom)
    return angle_deg(v_hom, v_het), float(np.sqrt(q_het / q_hom))


def main() -> None:
    rng = np.random.default_rng(SEED)

    # --- 0. 先把任务书里的量级论据实测掉 ---
    x_lo = 1.0 / (mc.R_GAS * (40.0 + 273.15))
    x_hi = 1.0 / (mc.R_GAS * (-18.0 + 273.15))
    jac = {
        "R_GAS_used_by_123": mc.R_GAS,
        "unit": "kJ/(mol*K)（因 E_a 以 kJ/mol 计）",
        "x_eq_1_over_RT_range_over_T_-18_to_40C": [x_lo, x_hi],
        "second_column_value": 1.0,
        "column_magnitude_ratio": [1.0 / x_hi, 1.0 / x_lo],
        "task_brief_claimed_ratio": 1e4,
        "verdict": ("任务书的 4.2e-4 对应 R=8.314 J/(mol·K)，与脚本实际单位不符；"
                    "两列真实量级比只有 2.1–2.6 倍，故『方向被钉在 e_1』的论据不成立"),
        "max_possible_rotation_deg": float(np.degrees(np.arctan(x_hi) - np.arctan(x_lo))),
    }

    # --- (A) 随机设计库：用 123 自己的 sample_design，σ_i 在 1–10 倍内随机异方差 ---
    angles, ratios = [], []
    for _ in range(N_RANDOM):
        des = mc.sample_design(rng)
        n_T = len(des["T_C"])
        base = float(np.exp(rng.uniform(np.log(0.06), np.log(0.60))))   # 锚定 σ_lnk 量程
        sig = base * np.exp(rng.uniform(0.0, np.log(10.0), n_T))        # 逐温度 1–10× 异方差
        a, r = pair(des["T_C"], des["n_per_T"], sig)
        angles.append(a); ratios.append(r)
    angles = np.asarray(angles, float); ratios = np.asarray(ratios, float)

    # --- (B) 对抗极端例：刻意挑最可能把方向转开的构型 ---
    adversarial = []

    def add(name, T_C, n_per_T, sigma_i):
        sigma_i = np.asarray(sigma_i, float)
        a, r = pair(T_C, n_per_T, sigma_i)
        adversarial.append({"case": name, "T_C": np.asarray(T_C, float).tolist(),
                            "sigma_ratio_max_over_min": float(sigma_i.max() / sigma_i.min()),
                            "angle_deg": a, "crb_rmse_ratio_het_over_hom": r})

    add("3T 窄窗 2℃，异方差 1000×", [4.0, 5.0, 6.0], [10, 10, 10], [0.06, 0.6, 60.0])
    add("3T 宽窗 −18–40℃，异方差 1000×", [-18.0, 11.0, 40.0], [10, 10, 10], [0.06, 0.6, 60.0])
    add("2T，异方差 1e6", [0.0, 25.0], [10, 10], [1e-3, 1e3])
    add("4T，一个 σ→0（权重发散）", [-18.0, 0.0, 20.0, 40.0], [8, 8, 8, 8], [1e-8, 0.3, 0.3, 0.3])
    add("6T 等距，σ 单调 100×", np.linspace(-18, 40, 6), [5] * 6,
        np.exp(np.linspace(np.log(0.06), np.log(6.0), 6)))

    adv_max_a = max(a["angle_deg"] for a in adversarial)
    adv_max_r = max(a["crb_rmse_ratio_het_over_hom"] for a in adversarial)

    res = {
        "script": Path(__file__).name,
        "seed": SEED,
        "purpose": ("改码前的独立几何预检：同方差代理与异方差定义量的最小特征方向夹角，"
                    "以及它对 rmse_deg 的预期影响。用于给义务 1f9bcf45 的修复"
                    "定一个【可证伪的预注册预测】。"),
        "task_brief_mechanism_refuted": jac,
        "random_designs": {
            "n": int(N_RANDOM),
            "sigma_heteroscedasticity": "逐温度 1–10× 随机（σ 基准锚定 0.06–0.60）",
            "angle_deg_median": float(np.median(angles)),
            "angle_deg_p95": float(np.percentile(angles, 95)),
            "angle_deg_max": float(angles.max()),
            "crb_rmse_ratio_median": float(np.median(ratios)),
            "crb_rmse_ratio_p95": float(np.percentile(ratios, 95)),
            "crb_rmse_ratio_max": float(ratios.max()),
        },
        "adversarial_cases": adversarial,
        "worst_angle_deg_overall": float(max(float(angles.max()), adv_max_a)),
        "worst_crb_rmse_ratio_overall": float(max(float(ratios.max()), adv_max_r)),
        "prediction": (
            "① 夹角是 0.1°–2° 量级，不是任务书说的 1e-3°；"
            "② 但 rmse 沿方向的二次型在主轴附近二阶平坦，故 rmse_deg 的相对变化 ~ O(sin²θ/2)，"
            "由上面的 CRB 比值给出定量预测（中位与 p95 见 random_designs）；"
            "③ 因该比值 ≥1 且逐设计不同，头条 log-log 斜率可以在末位轻微移动，"
            "**不预测『逐位不变』**；预测的是移动量落在 CRB 比值所允许的范围内。"),
        "falsifier": ("若 123 全量重跑后 rmse_deg_het/rmse_deg 的中位数或 p95 明显超出本表的 CRB 预测，"
                      "则说明除方向错配外仍有别的缺陷，须停下报告而非改稿去追新数字。"),
        "honesty": ["本脚本只做几何 + CRB 一阶代理，不含仿真误差；"
                    "决定性核验由 123 全量重跑落盘的 rmse_deg 与 rmse_deg_het 两列直接对比给出。"],
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(jac, ensure_ascii=False, indent=2))
    print(json.dumps(res["random_designs"], ensure_ascii=False, indent=2))
    for a in adversarial:
        print(f"  {a['case']:<28s} σ极差 {a['sigma_ratio_max_over_min']:.1e}  "
              f"夹角 {a['angle_deg']:.3f}°  CRB比 {a['crb_rmse_ratio_het_over_hom']:.6f}")
    print(f"\n最大夹角 {res['worst_angle_deg_overall']:.3f}°  最大 CRB 比 {res['worst_crb_rmse_ratio_overall']:.6f}")
    print(f"写出 {OUT}")


if __name__ == "__main__":
    main()
