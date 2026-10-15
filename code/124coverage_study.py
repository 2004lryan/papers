#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
124coverage_study.py — 报出的 E_a 置信区间何时是假的？Ψ 作为标定诊断

动机
----
前三轮评审的一致诊断：本文所有【真实数据支撑】的结论都是本文自认经典的
（设计处方=经典、准则=E-最优、Ψ→∞ 档=Eicker--Drygas--Wu、Krug=恒等式、准则塌缩=代数），
而真正新的部分只在它自己假设的仿真世界里被验证。

本脚本换一个问法，检验一个【经典理论不作出、且对本刊读者直接有用】的预测：

    实践者从 Arrhenius 图回归里读出的 E_a ± SE（名义 95% 置信区间），
    在 Ψ 趋于 O(1) 时会不会【失去标定】——即实际覆盖率塌到名义值以下？
    并且，Ψ 是否比 n、ΔT、R²、条件数更能预测它塌在哪里？

为何这不是循环论证
------------------
失效的是【渐近正态近似】——而仿真器并不强加这个近似：它按真实前向模型生成数据、
按实践者的真实管线反演，名义 CI 由普通最小二乘的残差标准误给出。
Cramér--Rao 只讲方差下界，【不预测区间失去标定】。故覆盖率塌陷是三分律一侧的独有预测，
不是把假设算回来。

为何对化学计量学读者有用
------------------------
窄温窗加速试验里报 "E_a = xx ± yy kJ/mol" 是本领域的日常。
本文给出的是一个【事前】判据：在做实验之前就能算的 Ψ，告诉你那个 ± 值是否可信。

诚实边界
--------
仿真研究。设计独立、真值已知、误差与区间都由实践者管线真实产生，但数据非实测。
若覆盖率不塌陷，或 Ψ 并不优于廉价替代量，本脚本照实报告——那将是又一个负面结果。

输出 04outputs/124coverage_study.json + -bins.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs" / "124coverage_study.json"
BINS_CSV = ROOT / "04outputs" / "124coverage_study-bins.csv"
DESIGN_CSV = ROOT / "04outputs" / "124coverage_study-designs.csv"

# 与脚本 123 共用设计抽样与反演内核，保证两节结果同源
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("mc123", ROOT / "02code" / "123montecarlo_criterion_study.py")
mc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(mc)

R_GAS = mc.R_GAS
N_DESIGNS = 1000
N_REPS = 400
NOMINAL = 0.95
SEED = 20260726


def simulate_with_ci(rng, T_C, n_per_T, t_grid, Ea, lnA, sigma_y, n_reps):
    """在脚本 123 的反演管线上，额外产出【实践者会报告的】名义 95% CI 并计其覆盖。

    名义 CI 的口径刻意取实践中最常见的一种：对 (x_i, log k̂_i) 做普通最小二乘，
    以残差标准误给斜率 SE，用 t_{n_T-2} 分位数张开区间。不做任何加权或修正——
    正是要检验这个默认做法何时失效。
    """
    T_K = np.asarray(T_C, float) + 273.15
    k_true = np.exp(lnA - Ea / (R_GAS * T_K))
    n_T = len(T_C)
    k_grid = np.exp(np.linspace(np.log(k_true.min()) - 4.0, np.log(k_true.max()) + 4.0, 400))

    khat = np.empty((n_reps, n_T))
    for i in range(n_T):
        beta = 1.0 - np.exp(-k_true[i] * t_grid)
        Y = beta[None, None, :] + sigma_y * rng.standard_normal((n_reps, int(n_per_T[i]), len(t_grid)))
        khat[:, i] = mc.fit_k_grid(np.tile(t_grid, int(n_per_T[i])), Y.reshape(n_reps, -1), k_grid)

    x = 1.0 / (R_GAS * T_K)
    lk = np.log(np.clip(khat, 1e-300, None))
    xm = x.mean()
    Sxx = ((x - xm) ** 2).sum()
    slope = ((lk - lk.mean(axis=1, keepdims=True)) * (x - xm)).sum(axis=1) / Sxx
    intercept = lk.mean(axis=1) - slope * xm
    Ea_hat = -slope

    dof = n_T - 2
    if dof < 1:
        return None
    fitted = intercept[:, None] + slope[:, None] * x[None, :]
    sse = ((lk - fitted) ** 2).sum(axis=1)
    s2 = sse / dof
    se_slope = np.sqrt(s2 / Sxx)                    # 斜率标准误 = E_a 的标准误
    # 三种实践中都能见到的区间口径，全部报告，避免"只测了对自己有利的一种"
    halves = {
        "t": stats.t.ppf(0.5 + NOMINAL / 2.0, dof) * se_slope,   # 教科书做法（t_{n_T-2}）
        "z": 1.959963985 * se_slope,                              # 大样本正态近似（软件默认常见）
        "z_nodf": 1.959963985 * np.sqrt((sse / n_T) / Sxx),       # 残差方差不做自由度修正
    }
    half = halves["t"]
    lo, hi = Ea_hat - half, Ea_hat + half

    ok = np.isfinite(Ea_hat) & np.isfinite(half)
    if ok.sum() < n_reps // 2:
        return None
    cover = ((lo <= Ea) & (Ea <= hi))[ok]
    cov_variants = {kk: float((((Ea_hat - hh) <= Ea) & (Ea <= (Ea_hat + hh)))[ok].mean())
                    for kk, hh in halves.items()}
    r2 = 1.0 - sse / np.maximum(((lk - lk.mean(axis=1, keepdims=True)) ** 2).sum(axis=1), 1e-300)
    return {
        "coverage": float(cover.mean()),
        "coverage_variants": cov_variants,
        "rmse_Ea": float(np.sqrt(np.mean((Ea_hat[ok] - Ea) ** 2))),
        "bias_Ea": float(np.mean(Ea_hat[ok] - Ea)),
        "median_reported_halfwidth": float(np.median(half[ok])),
        "median_true_sd": float(np.std(Ea_hat[ok])),
        "median_R2": float(np.median(r2[ok])),
        "sigma_lnk_emp": np.std(lk, axis=0, ddof=1),
        "n_ok": int(ok.sum()),
    }


def wilson(p, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(v).astype(float) for v in (x, y, z))
    Z = np.stack([rz, np.ones_like(rz)], axis=1)
    res = []
    for r in (rx, ry):
        b, *_ = np.linalg.lstsq(Z, r, rcond=None)
        res.append(r - Z @ b)
    if min(np.std(res[0]), np.std(res[1])) < 1e-12:
        return float("nan")
    return float(np.corrcoef(res[0], res[1])[0, 1])


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = []
    for d in range(N_DESIGNS):
        des = mc.sample_design(rng)
        if des["n_T"] < 3:                      # 需要 dof>=1 才有可报告的 SE
            continue
        r = simulate_with_ci(rng, des["T_C"], des["n_per_T"], des["t_grid"],
                             des["Ea"], des["lnA"], des["sigma_y"], N_REPS)
        if r is None:
            continue
        sig = np.asarray(r["sigma_lnk_emp"], float)
        if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
            continue
        cr = mc.criteria(des["T_C"], des["n_per_T"], float(np.mean(sig)))
        rows.append({
            "n_T": des["n_T"], "dT": des["dT"], "N_obs": des["N_obs"], "Ea_true": des["Ea"],
            "Psi": cr["E"], "cond": cr["cond"], "D": cr["D"],
            "coverage": r["coverage"],
            "cov_t": r["coverage_variants"]["t"], "cov_z": r["coverage_variants"]["z"],
            "cov_znodf": r["coverage_variants"]["z_nodf"],
            "rmse_Ea": r["rmse_Ea"], "bias_Ea": r["bias_Ea"],
            "reported_halfwidth": r["median_reported_halfwidth"], "true_sd": r["median_true_sd"],
            "R2": r["median_R2"], "n_ok": r["n_ok"],
            "relerr_Ea": r["rmse_Ea"] / des["Ea"],
            "SE_ratio": r["median_reported_halfwidth"] / 1.96 / max(r["median_true_sd"], 1e-300),
        })
        if (d + 1) % 200 == 0:
            print(f"  {d+1}/{N_DESIGNS} ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(DESIGN_CSV, index=False)
    print(f"有效设计 {len(df)}，用时 {time.time()-t0:.0f}s")

    # --- 覆盖率随 Ψ 的走势（按 Ψ 十分位分箱）---
    df["bin"] = pd.qcut(df["Psi"], 10, labels=False, duplicates="drop")
    bins = []
    for b, g in df.groupby("bin"):
        n_rep_tot = int(g["n_ok"].sum())
        cov = float((g["coverage"] * g["n_ok"]).sum() / max(n_rep_tot, 1))
        bins.append({
            "bin": int(b), "n_designs": int(len(g)),
            "Psi_median": float(g["Psi"].median()),
            "Psi_lo": float(g["Psi"].min()), "Psi_hi": float(g["Psi"].max()),
            "coverage": cov, "coverage_ci95": wilson(cov, n_rep_tot),
            "median_SE_ratio": float(g["SE_ratio"].median()),
            "median_R2": float(g["R2"].median()),
            "median_relerr": float(g["relerr_Ea"].median()),
            "cov_t": float((g["cov_t"] * g["n_ok"]).sum() / max(n_rep_tot, 1)),
            "cov_z": float((g["cov_z"] * g["n_ok"]).sum() / max(n_rep_tot, 1)),
            "cov_znodf": float((g["cov_znodf"] * g["n_ok"]).sum() / max(n_rep_tot, 1)),
        })
    bins_df = pd.DataFrame(bins).sort_values("Psi_median")
    bins_df.to_csv(BINS_CSV, index=False)

    lo_bin, hi_bin = bins_df.iloc[0], bins_df.iloc[-1]

    # --- Ψ 是否比廉价替代量更能预测"区间失标定"？以 |coverage−0.95| 为待预测量 ---
    miscal = (NOMINAL - df["coverage"]).clip(lower=0).to_numpy()   # 只罚反保守（覆盖不足）
    preds = {"Psi": df["Psi"].to_numpy(), "N_obs": df["N_obs"].to_numpy(),
             "dT": df["dT"].to_numpy(), "R2": df["R2"].to_numpy(),
             "cond": df["cond"].to_numpy(), "D": df["D"].to_numpy()}
    marg = {k: float(spearmanr(v, miscal)[0]) for k, v in preds.items()}
    part = {k: partial_spearman(v, miscal, df["dT"].to_numpy())
            for k, v in preds.items() if k != "dT"}

    rb = np.random.default_rng(SEED + 1)
    n = len(df)
    gap = {}
    for k in ["dT", "R2", "cond"]:
        g = []
        for _ in range(800):
            i = rb.integers(0, n, n)
            g.append(abs(spearmanr(preds["Psi"][i], miscal[i])[0])
                     - abs(spearmanr(preds[k][i], miscal[i])[0]))
        g = np.array(g)
        gap[k] = {"point": float(np.mean(g)),
                  "ci95": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
                  "P_alt_at_least_as_good": float(np.mean(g <= 0))}

    # --- 失标定的成因分解：是区间报窄了（方差低估），还是估计有偏（中心错了）？---
    df["bias_over_sd"] = df["bias_Ea"].abs() / np.maximum(df["true_sd"], 1e-300)
    lo_mask = df["Psi"] <= df["Psi"].quantile(0.10)
    hi_mask = df["Psi"] >= df["Psi"].quantile(0.90)
    decomp = {
        "low_Psi_decile": {
            "median_SE_ratio_reported_over_true": float(df.loc[lo_mask, "SE_ratio"].median()),
            "median_abs_bias_over_true_sd": float(df.loc[lo_mask, "bias_over_sd"].median()),
            "median_R2": float(df.loc[lo_mask, "R2"].median()),
        },
        "high_Psi_decile": {
            "median_SE_ratio_reported_over_true": float(df.loc[hi_mask, "SE_ratio"].median()),
            "median_abs_bias_over_true_sd": float(df.loc[hi_mask, "bias_over_sd"].median()),
            "median_R2": float(df.loc[hi_mask, "R2"].median()),
        },
        "note": ("SE_ratio = 报出的标准误 / 真实抽样标准差；<1 即区间报窄（反保守）。"
                 "bias/true_sd 衡量区间中心是否偏离真值。两者共同决定覆盖率。"),
    }

    # --- R² 是本领域最常用的"看着靠谱"指标：它在低 Ψ 处是否仍然很高（即具欺骗性）？---
    deceptive = {
        "median_R2_in_worst_coverage_decile": float(
            df.loc[df["coverage"] <= df["coverage"].quantile(0.10), "R2"].median()),
        "median_coverage_where_R2_above_0p99": float(
            df.loc[df["R2"] >= 0.99, "coverage"].median()) if (df["R2"] >= 0.99).any() else float("nan"),
        "n_designs_R2_above_0p99": int((df["R2"] >= 0.99).sum()),
        "n_of_those_with_coverage_below_0p90": int(((df["R2"] >= 0.99) & (df["coverage"] < 0.90)).sum()),
    }

    res = {
        "script": Path(__file__).name, "seed": SEED, "nominal_level": NOMINAL,
        "n_designs_valid": int(len(df)), "n_reps_per_design": N_REPS,
        "protocol": ("每个设计上按实践者的默认做法构造 E_a 的名义 95% CI"
                     "（Arrhenius 图 OLS 残差标准误 + t_{n_T-2} 分位数），"
                     "对已知真值计实际覆盖率。"),
        "why_not_circular": ("失效的是渐近正态近似，仿真器并不强加它；"
                             "Cramér--Rao 只给方差下界，不预测区间失标定。"),
        "coverage_by_Psi_decile": bins,
        "lowest_Psi_decile_coverage": float(lo_bin["coverage"]),
        "highest_Psi_decile_coverage": float(hi_bin["coverage"]),
        "spearman_Psi_vs_undercoverage": marg,
        "partial_spearman_controlling_dT": part,
        "Psi_advantage_over_alternatives": gap,
        "miscalibration_decomposition": decomp,
        "R2_is_deceptive": deceptive,
        "honesty": [
            "仿真研究：设计独立、真值已知、区间由实践者管线真实产生，但数据非实测。",
            "若覆盖率不塌陷或 Ψ 不优于廉价替代量，本脚本照实报告。",
        ],
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(f"\n=== 名义 {NOMINAL:.0%} CI 的实际覆盖率（按 Ψ 十分位）===")
    for r in bins:
        print(f"  Ψ 中位 {r['Psi_median']:10.3g}  覆盖 {r['coverage']:.3f} "
              f"CI[{r['coverage_ci95'][0]:.3f},{r['coverage_ci95'][1]:.3f}]  "
              f"| t {r['cov_t']:.3f}  z {r['cov_z']:.3f}  z_nodf {r['cov_znodf']:.3f}"
              f"  SE比 {r['median_SE_ratio']:.2f}  n={r['n_designs']}")
    print(f"\n最低 Ψ 十分位覆盖 {lo_bin['coverage']:.3f}  vs  最高 {hi_bin['coverage']:.3f}（名义 {NOMINAL}）")
    print("\n=== 谁更能预测「覆盖不足」（Spearman，越大越强）===")
    for k, v in sorted(marg.items(), key=lambda kv: -abs(kv[1])):
        pv = part.get(k)
        print(f"  {k:>7s} 边际 {v:+.3f}" + (f"   偏(控ΔT) {pv:+.3f}" if pv is not None else ""))
    print("\n=== Ψ 相对各替代量的优势（bootstrap）===")
    for k, v in gap.items():
        print(f"  vs {k:>6s}: {v['point']:+.3f} CI{[round(x,3) for x in v['ci95']]} "
              f"P(替代量不劣)={v['P_alt_at_least_as_good']:.1%}")
    print("\n=== 失标定成因分解 ===")
    for k in ("low_Psi_decile", "high_Psi_decile"):
        d = decomp[k]
        print(f"  {k}: SE比(报/真) {d['median_SE_ratio_reported_over_true']:.3f}  "
              f"|偏差|/真SD {d['median_abs_bias_over_true_sd']:.3f}  R² {d['median_R2']:.3f}")
    print("\n=== R² 的欺骗性 ===")
    print(f"  覆盖率最差十分位的中位 R² = {deceptive['median_R2_in_worst_coverage_decile']:.3f}")
    print(f"  R²≥0.99 的设计 {deceptive['n_designs_R2_above_0p99']} 个，"
          f"其中覆盖率<0.90 的有 {deceptive['n_of_those_with_coverage_below_0p90']} 个")
    print(f"\n写出 {OUT}")


if __name__ == "__main__":
    main()
