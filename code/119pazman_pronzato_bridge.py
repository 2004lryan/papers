#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
119 — 与 Pázman–Pronzato (2006) 渐近奇异设计算例的坐标对照

用途
----
支撑 Elsevier_zh.tex §相关工作「与渐近奇异设计及弱识别文献的关系」中的一句：
    「把本文的 Psi_n 施于 Pázman–Pronzato (2006) 的算例，其 n^{-1/4} 恰落在
      同一条 e_n ≍ Psi_n^{-1/2} 曲线上（数值核验：退化方向标准差 × Psi_n^{1/2}
      自 0.999 单调趋于 1.000，n 自 2e2 至 2e6）」

原文设定（Pázman A., Pronzato L., Statist. Probab. Lett. 76(11):1089-1096, 2006,
doi:10.1016/j.spl.2005.12.010，§2；HAL hal-00416062 开放全文已核验）：
    模型 eta(x, theta) = theta1*x + theta2*x^2，f(x) = (x, x^2)^T
    设计序列 n = 2m，x_{2k-1} = 1，x_{2k} = 1 - (1/k)^{1/4}，k = 1..m
    极限设计 xi* 把全部质量置于 x = 1，M(xi*) = [[1,1],[1,1]] 为秩 1，
    退化（不可估）方向为 M(xi*) 的零空间 u = (1,-1)/sqrt(2)。
    原文结论：沿该方向 LS 估计量以 n^{-1/4} 收敛（慢于 n^{-1/2}）。

诚实边界（关键，写入论文时不可省）
--------------------------------
在线性高斯模型中，沿 M 的最小特征方向有 sd = sqrt(u' M^{-1} u / n) = Psi^{-1/2}
（当 u 恰为最小特征向量时为**恒等式**）。因此本脚本的 ratio ≈ 1 是代数恒等式的
数值确认，用途是说明「两套结果处于同一坐标」，**不构成对本文定理的独立经验证据**。
本文定理的实质内容（非线性模型、三角阵列上的一致 LAQ、匹配的极小极大下界）
不由本脚本验证。

输出
----
04outputs/119pazman_pronzato_bridge.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "04outputs" / "119pazman_pronzato_bridge.json"

# PP2006 §2 的退化方向：M(xi*) = 11^T 的零空间
U_DEGENERATE = np.array([1.0, -1.0]) / np.sqrt(2.0)


def pp2006_design(m: int) -> np.ndarray:
    """构造 PP2006 §2 的设计点序列，n = 2m。"""
    k = np.arange(1, m + 1, dtype=float)
    return np.concatenate([np.ones(m), 1.0 - (1.0 / k) ** 0.25])


def diagnostics(m: int) -> dict:
    x = pp2006_design(m)
    n = x.size
    F = np.stack([x, x ** 2], axis=1)          # f(x) = (x, x^2)^T
    M = F.T @ F / n                             # 归一化单样本信息阵
    lmin = float(np.linalg.eigvalsh(M)[0])
    psi = n * lmin                              # 本文诊断量 Psi_n = n * lambda_min
    sd_u = float(np.sqrt(U_DEGENERATE @ np.linalg.inv(M) @ U_DEGENERATE / n))
    return {
        "n": int(n),
        "lambda_min_M": lmin,
        "Psi_n": psi,
        "sd_degenerate_direction": sd_u,
        "ratio_sd_times_sqrt_Psi": sd_u * np.sqrt(psi),   # 理论值 1（线性模型下为恒等式）
        "sd_over_n_minus_quarter": sd_u / n ** -0.25,     # PP2006 所述 n^{-1/4} 速率的比例常数
    }


def main() -> None:
    rows = [diagnostics(m) for m in (10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6)]
    ratios = [r["ratio_sd_times_sqrt_Psi"] for r in rows]

    result = {
        "script": Path(__file__).name,
        "purpose": "PP2006 渐近奇异设计算例与本文 Psi 参数化的坐标对照",
        "source_paper": {
            "authors": "Pázman A., Pronzato L.",
            "title": "On the irregular behavior of LS estimators for asymptotically singular designs",
            "venue": "Statistics & Probability Letters 76(11):1089-1096",
            "year": 2006,
            "doi": "10.1016/j.spl.2005.12.010",
            "open_fulltext": "https://hal.science/hal-00416062",
        },
        "rows": rows,
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "claim_in_manuscript": "退化方向标准差 × Psi_n^{1/2} 自 0.999 单调趋于 1.000（n 自 2e2 至 2e6）",
        "claim_holds": bool(min(ratios) >= 0.999 and max(ratios) <= 1.0),
        "honesty_note": (
            "线性高斯模型中沿最小特征方向 sd = Psi^{-1/2} 为恒等式；"
            "本核验说明两套结果处于同一坐标，不构成对本文定理的独立经验证据。"
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'n':>9} {'Psi_n':>12} {'sd_deg':>12} {'sd*sqrt(Psi)':>14} {'sd/n^-0.25':>12}")
    for r in rows:
        print(f"{r['n']:>9} {r['Psi_n']:>12.4f} {r['sd_degenerate_direction']:>12.4e} "
              f"{r['ratio_sd_times_sqrt_Psi']:>14.4f} {r['sd_over_n_minus_quarter']:>12.4f}")
    print(f"\nratio 区间 [{result['ratio_min']:.4f}, {result['ratio_max']:.4f}] · "
          f"落于 [0.999, 1.000]: {result['claim_holds']}")
    print(f"写出 {OUT}")


if __name__ == "__main__":
    main()
