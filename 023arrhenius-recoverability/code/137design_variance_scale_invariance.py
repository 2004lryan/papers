#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""137design_variance_scale_invariance.py — 经验设计方差相对误差的尺度不变性核验

为什么需要这个脚本
------------------
RYA-537 清偿义务 `936462b7`（随机设计涨落未进 Gaussian 余项论证）时，初稿曾写：
按总体矩读假设 A3 需要 √n·v_n→∞（即 Ψ_n≫√n），故「在临界档与不可恢复档直接失效」。
落盘前做数值自查，该断言被自己的数据推翻——Weyl 上界虽成立，但对本设计族是【松】的：
它没有计入「涨落随窗宽收缩、与 λmin 同阶变小」。真正起作用的是方差估计量的尺度不变性。

本脚本核验两件事（正文 §sec:converse 证明第 (2) 步的范围说明所依据者）：
  (a) ||Î₁ − I₁||_op 对 n 的 log-log 斜率 ≈ −1/2（Weyl 界的前提本身成立）；
  (b) 经验/总体设计方差的【相对】误差 sd(v̂_n/v_n)·√n 是与窗宽 ΔT 无关的常数，
      因而 λmin(Î₁)/λmin(I₁)→1 在三档上一律成立，不需要 √n·v_n→∞。

诚实定位
--------
这是【数值核验】，不是证明。(b) 的证明是一行尺度不变性：以 u=r/ΔT 重参数化后
v̂_n/v_n = v̂_u/v_u 与 ΔT 无关。本脚本只是把它在跨 8 个数量级的窗宽上实测一遍。
它也不主张覆盖形状随 n 退化的设计族——正文对此已明确不作主张。

输出
----
04outputs/137design_variance_scale_invariance.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs" / "137design_variance_scale_invariance.json"
SEED = 20260819
N_REP = 400


def op_norm_scaling(rng, ns, dT=1.0, reps=200):
    """(a) 固定窗宽，||Î₁ − I₁||_op 对 n 的标度。"""
    m1, m2 = dT / 2, dT ** 2 / 3
    I1 = np.array([[1.0, m1], [m1, m2]])
    out = []
    for n in ns:
        ops = []
        for _ in range(reps):
            r = rng.uniform(0, dT, n)
            X = np.stack([np.ones(n), r], 1)
            ops.append(float(np.linalg.norm(X.T @ X / n - I1, 2)))
        out.append({"n": int(n), "op_norm_mean": float(np.mean(ops))})
    sl = float(np.polyfit(np.log([o["n"] for o in out]),
                          np.log([o["op_norm_mean"] for o in out]), 1)[0])
    return {"by_n": out, "loglog_slope": sl, "theory": -0.5,
            "verdict": "matches -1/2" if abs(sl + 0.5) < 0.06 else "deviates"}


def relative_error_scale_free(rng, ns, dTs, reps=N_REP):
    """(b) sd(v̂_n/v_n)·√n 是否与窗宽无关。"""
    rows = []
    for n in ns:
        for dT in dTs:
            r = rng.uniform(0, dT, (reps, n))
            rel = r.var(1, ddof=1) / (dT ** 2 / 12)
            rows.append({"n": int(n), "dT": float(dT), "v_n": float(dT ** 2 / 12),
                         "sd_rel": float(rel.std()), "sd_rel_times_sqrt_n": float(rel.std() * np.sqrt(n))})
    vals = [x["sd_rel_times_sqrt_n"] for x in rows]
    return {"rows": rows, "spread_ratio_max_over_min": float(max(vals) / min(vals)),
            "verdict": ("scale-free" if max(vals) / min(vals) < 1.3 else "scale-dependent")}


def lmin_ratio_below_weyl_threshold(rng, ns, expo=0.95, reps=120):
    """(b') 在 √n·v_n→0 的档位上直接看 λmin(Î₁)/λmin(I₁) 是否仍 →1。"""
    rows = []
    for n in ns:
        vn = float(n ** -expo)
        dT = float(np.sqrt(12 * vn))
        I1 = np.array([[1.0, dT / 2], [dT / 2, dT ** 2 / 3]])
        l1 = float(np.linalg.eigvalsh(I1)[0])
        rat = []
        for _ in range(reps):
            r = rng.uniform(0, dT, n)
            X = np.stack([np.ones(n), r], 1)
            rat.append(float(np.linalg.eigvalsh(X.T @ X / n)[0]) / l1)
        rows.append({"n": int(n), "v_n": vn, "sqrt_n_times_v_n": float(np.sqrt(n) * vn),
                     "lmin_ratio_mean": float(np.mean(rat)), "lmin_ratio_sd": float(np.std(rat))})
    return {"rows": rows, "expo": expo,
            "verdict": ("ratio->1 even below the Weyl threshold"
                        if all(abs(x["lmin_ratio_mean"] - 1) < 0.02 for x in rows[-2:]) else "degrades")}


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {
        "script": Path(__file__).name, "seed": SEED,
        "purpose": ("核验 §sec:converse 证明第 (2) 步的范围说明：Weyl 上界要求 √n·v_n→∞，"
                    "但对本设计族该界是松的；起作用的是方差估计量的尺度不变性。"),
        "a_operator_norm_scaling": op_norm_scaling(rng, [200, 800, 3200, 12800]),
        "b_relative_error_scale_free": relative_error_scale_free(
            rng, [200, 2000, 20000], [1.0, 1e-2, 1e-4]),
        "b_prime_lmin_ratio_below_weyl_threshold": lmin_ratio_below_weyl_threshold(
            rng, [400, 4000, 40000]),
        "honesty": [
            "这是数值核验，不是证明；(b) 的证明是一行尺度不变性（u=r/ΔT 重参数化）。",
            "不覆盖形状随 n 退化的设计族——正文对此已明确不作主张。",
            "初稿曾据 Weyl 上界断言「临界档与不可恢复档直接失效」，被本脚本推翻，落盘前已改。",
        ],
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    a = res["a_operator_norm_scaling"]; b = res["b_relative_error_scale_free"]
    bp = res["b_prime_lmin_ratio_below_weyl_threshold"]
    print(f"(a) ||Î₁−I₁||_op log-log 斜率 {a['loglog_slope']:+.3f}  {a['verdict']}")
    print(f"(b) sd(v̂/v)·√n 跨窗宽极差比 {b['spread_ratio_max_over_min']:.3f}  {b['verdict']}")
    for x in bp["rows"]:
        print(f"(b') n={x['n']:6d} √n·v_n={x['sqrt_n_times_v_n']:.4f} "
              f"λmin 比值 {x['lmin_ratio_mean']:.4f}±{x['lmin_ratio_sd']:.4f}")
    print(f"(b') {bp['verdict']}")
    print(f"写出 {OUT}")


if __name__ == "__main__":
    main()
