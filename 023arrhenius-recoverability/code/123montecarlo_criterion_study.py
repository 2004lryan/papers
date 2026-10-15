#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
123montecarlo_criterion_study.py — 独立设计库上的 Monte-Carlo 准则研究（真值估计误差）

为什么需要这个脚本
------------------
2026-07-25 第二轮外审（5.93/10）的一致诊断：诚实过滤后，本文没有一条**既新又已证实**
的经验主张。具体地，脚本 122 的准则对决被其自身数据推翻——

  · "E/A/c 秩等价"是**算术恒等**（A/E ∈ [1.999961, 1.999991]），非实证发现；
  · "唯 D 排错"**未达统计可辨**（bootstrap CI 含零，120 对中仅 6 对不同）；
  · 根因是 16 个子集**相互嵌套**，有效自由度远小于 16。

本脚本按外审的一致建议重做：**数百个相互独立的 Arrhenius 设计** + **模拟真值估计误差**
（而非 bootstrap 噪声传播），并报告**决策相关量**（按 D 选设计相对按 E 选，真实 E_a 误差
上升多少百分比）而非两个相关系数之差。同一次运行同时扫 n 与 ΔT 两轴，
以闭合正文 §scope 的 L4/L5（"无单一体系同时覆盖 Ψ=n·(ΔT)² 两轴且带真统计误差"）。

诚实定位
--------
这是**模拟**研究：设计是独立抽取的，真值是已知的，误差是真的估计误差（多次噪声重复的 RMSE），
但数据不是真实测量。它不能替代真实数据，它替代的是**被嵌套结构毁掉的自由度**。
所有量程锚定到本项目盘上已核验的真实值（见 ANCHORS），而非凭空设定。

流程与正文管线一致（脚本 54）
----------------------------
观测层 y = β + ε，β = 1 − exp(−k t)；逐温度以 1-D 非线性最小二乘拟合 k̂_i；
再以 log k̂ 对 1/(R·T_K) 线性回归得 Ê_a（斜率取负）。这与脚本 54 的 fit_k → arrhenius 同构，
因此包含 k 估计这一非线性步骤——正是它使"准则能否预测真实误差"成为真问题而非恒等式。

输出
----
04outputs/123montecarlo_criterion_study.json
04outputs/123montecarlo_criterion_study-designs.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs" / "123montecarlo_criterion_study.json"
CSV = ROOT / "04outputs" / "123montecarlo_criterion_study-designs.csv"

R_GAS = 8.314e-3  # kJ/(mol·K)，与脚本 54/122 逐位一致

# --- 量程锚定：全部取自本项目盘上已核验的真实结果，不凭空设定 -------------------
ANCHORS = {
    "Ea_kJmol": {
        "range": (13.0, 70.0),
        "source": ("苹果 54c5_realkinetics.json: Red Delicious 51.0 / Granny Smith 61.4 / Royal Gala 68.5"
                   "（剔除 Pink Lady 的病态 858）；食品 99_realkinetics2.json per_series: 13.8 / 29.4 等；"
                   "牛油果 102_avocado_c5_pilot.json: 53.3"),
    },
    "sigma_lnk": {
        "range": (0.06, 0.60),
        "source": "苹果 σ_lnk=0.0621（54c5 full）；食品 per_series σ_lnk 0.34–0.58（99_realkinetics2）",
    },
    "T_C": {
        "range": (-18.0, 40.0),
        "source": "苹果 2–10℃；牛油果 10/20℃；食品 −18/5/25/40℃（三体系并集）",
    },
}

N_DESIGNS = 1000         # 独立设计数（非嵌套）——这正是脚本 122 缺的自由度
N_REPS = 400             # 每个设计的噪声重复次数 → 真值 RMSE
SEED = 20260725


# ---------------------------------------------------------------- 设计与准则
def fisher(T_C: np.ndarray, n_per_T: np.ndarray, sigma_lnk: float) -> np.ndarray:
    """(E_a, lnA) 基下的设计【总】Fisher，与 54c5_realkinetics.psi_design 修正后口径一致。

    【2026-08-19 订正，RYA-526】原实现在 JᵀJ 之外再按 n_per_T 加权一次，属重复计权。
    本脚本的 k̂ 由每温度 n_per_T 条重复曲线【联合拟合】得出（simulate_design），
    而 sigma_lnk_emp 是该聚合估计量自身的标准差——重复数已被吸收其中。
    再乘 n_per_T 即把重复计两次；且 n_per_T∈[3,20] 随设计变化、不是常数因子，
    会系统性扭曲跨设计比较。口径依据同 54c5：σ_lnk 已是每温度 ln k̂ 估计量的标准差，
    故 I_total = JᵀJ，不得再乘任何样本量因子（def:psi 的 Ψ_n=λmin(I_total)）。
    n_per_T 保留在签名中仅为调用方（124coverage_study）兼容，不参与计算。
    """
    T_K = np.asarray(T_C, float) + 273.15
    J = np.stack([-1.0 / (R_GAS * T_K), np.ones_like(T_K)], axis=1) / sigma_lnk
    return J.T @ J


def fisher_het(T_C: np.ndarray, n_per_T: np.ndarray, sigma_i: np.ndarray) -> np.ndarray:
    """逐温度噪声不同的 Fisher（正文的 Ψ_het）：每行按 1/σ_i² 加权。

    【2026-08-19 订正，RYA-526】原为 n_i/σ_i²，与 fisher() 同因去掉 n_i：σ_i 已是
    联合拟合后每温度 ln k̂ 估计量的标准差，重复数已吸收，再乘 n_i 即二次计权。
    """
    T_K = np.asarray(T_C, float) + 273.15
    J = np.stack([-1.0 / (R_GAS * T_K), np.ones_like(T_K)], axis=1)
    w = 1.0 / np.asarray(sigma_i, float) ** 2
    return (J * w[:, None]).T @ J


def _spectral_criteria(M: np.ndarray) -> dict:
    """给定一个 2x2 信息算子 M，产出【全部七个】同台判据 + 最小特征方向。

    【2026-08-23 新增，RYA-581 / 义务 de049d538e54】本函数是本轮订正的核心：
    此前 E/D 由 fisher_het 产出、而 A/c_deg/c_Ea/cond 由同方差 fisher 产出，
    七个判据被排在同一张对决表上比较，属【算子不匹配】——
    「条件数失败」这一结论因此没有把「判据选择」与「信息阵装配」拆开。
    现把判据计算与算子装配彻底解耦：装配一次算子，七个判据全部由它导出。
    调用方各自决定喂哪个算子，同一张表内只允许出现同一个算子的判据。
    """
    ev, evec = np.linalg.eigh(M)
    lmin, lmax = float(ev[0]), float(ev[1])
    Minv = np.linalg.inv(M)
    v_min = evec[:, 0]
    e_Ea = np.array([1.0, 0.0])
    return {
        "E": lmin,                                        # E-最优 = 本文 Ψ = λmin
        "D": float(np.sqrt(max(np.linalg.det(M), 0.0))),  # D-最优
        "A": 2.0 / float(np.trace(Minv)),                 # A-最优
        "c_deg": 1.0 / float(v_min @ Minv @ v_min),       # 沿退化方向的 c-最优
        "c_Ea": 1.0 / float(e_Ea @ Minv @ e_Ea),          # 只关心 E_a 的 c-最优
        "cond": lmax / max(lmin, 1e-300),
        "lambda_min": lmin,
        "lambda_max": lmax,
        "v_deg": v_min,
    }


def criteria_het(T_C, n_per_T, sigma_i) -> dict:
    """异方差定义量（正文 def:psi 的 Ψ）及【它自己的】最小特征方向。

    【2026-08-21 订正，RYA-569】原实现用 np.linalg.eigvalsh，只拿特征值、不拿特征向量，
    于是主循环只能借用 criteria()（同方差代理）的 v_deg 去投影误差——
    头条速率检验因此把【异方差算子的特征值】与【同方差算子的特征方向】配在一起，
    是两个不同算子的量。取证义务 1f9bcf45（reviewer 原级 critical，FP-RISK 封顶为 major）
    指出此事，回码复核属实。现改用 eigh，返回本算子自己的 v_deg，
    使头条的 Ψ 与「最难方向」出自同一个信息算子。

    【2026-08-23 扩充，RYA-581 / 义务 de049d538e54】原实现只返回 E/D 两个判据，
    于是 A/c_deg/c_Ea/cond 只能从同方差 fisher() 借，造成同台对决跨算子。
    现改为委托 _spectral_criteria(fisher_het(...))，七个判据一律由异方差算子导出。
    返回键为旧键的超集，旧调用方不受影响。
    """
    return _spectral_criteria(fisher_het(T_C, n_per_T, sigma_i))


def criteria(T_C, n_per_T, sigma_lnk) -> dict:
    """同方差特例代理（def:psi 的 σ_i≡σ̄ 特例）下的同一组七判据。

    【2026-08-23，RYA-581】改为委托 _spectral_criteria(fisher(...))，与 criteria_het
    共用同一段判据代数，从而保证「代理表」与「定义量表」的差别只在算子、不在算法。
    返回键与旧实现逐键相同，124coverage_study 等调用方不受影响。
    """
    return _spectral_criteria(fisher(T_C, n_per_T, sigma_lnk))


# ---------------------------------------------------------------- 仿真与反演
def fit_k_grid(t: np.ndarray, Y: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    """向量化 1-D 非线性最小二乘：对 Y[rep, obs] 逐 rep 拟合 β=1−exp(−k t) 的 k。

    做法：在对数 k 网格上算 SSE，取最小者，再用相邻三点抛物线细化（等价于一次牛顿步）。
    与逐点调用 curve_fit 的差别在 1e-3 量级以下，而快约两个数量级。
    """
    beta = 1.0 - np.exp(-k_grid[:, None] * t[None, :])            # (G, n_obs)
    sse = ((Y[:, None, :] - beta[None, :, :]) ** 2).sum(axis=2)   # (rep, G)
    j = np.argmin(sse, axis=1)
    j = np.clip(j, 1, len(k_grid) - 2)
    lg = np.log(k_grid)
    y0 = sse[np.arange(len(j)), j - 1]
    y1 = sse[np.arange(len(j)), j]
    y2 = sse[np.arange(len(j)), j + 1]
    denom = y0 - 2 * y1 + y2
    step = np.where(np.abs(denom) > 1e-300, 0.5 * (y0 - y2) / np.where(denom == 0, 1, denom), 0.0)
    step = np.clip(step, -1.0, 1.0)
    h = lg[1] - lg[0]
    return np.exp(lg[j] + step * h)


def simulate_design(rng, T_C, n_per_T, t_grid, Ea, lnA, sigma_y, n_reps) -> dict:
    """按正文管线（逐温度拟 k → log k 对 1/(R T) 线性回归）在噪声重复上得真值估计误差。"""
    T_K = np.asarray(T_C, float) + 273.15
    k_true = np.exp(lnA - Ea / (R_GAS * T_K))
    n_T = len(T_C)
    # k 网格：**不使用真值**。上下界只由该设计自己的时间窗 t_grid 决定，
    # 而 t_grid 是实验者开跑前就定好的量，故此网格在真实部署中可照搬。
    # 判据：k*t 太小则窗内几乎无衰减、太大则首点即已饱和，两端各留 4 个数量级余量。
    # （2026-08-06：原实现由 k_true 构造网格，属 oracle 信息，经取证 HP-FAKE-GT 指出后改此。
    #   改前/改后逐键比对见 04outputs/129kgrid_oracle_ablation.json：162 个数值键无一变动超过 2%。）
    _tmin = float(np.min(t_grid)); _tmax = float(np.max(t_grid))
    _k_lo = 1e-4 / max(_tmax, 1e-12)
    _k_hi = 1e+4 / max(_tmin, 1e-12)
    k_grid = np.exp(np.linspace(np.log(_k_lo), np.log(_k_hi), 2000))

    khat = np.empty((n_reps, n_T))
    _edge_hits = 0
    _edge_tot = 0
    for i in range(n_T):
        beta = 1.0 - np.exp(-k_true[i] * t_grid)                       # (n_t,)
        # 每温度 n_per_T 条独立曲线，各 n_t 个时间点
        Y = beta[None, None, :] + sigma_y * rng.standard_normal((n_reps, int(n_per_T[i]), len(t_grid)))
        Yf = Y.reshape(n_reps, -1)
        tf = np.tile(t_grid, int(n_per_T[i]))
        khat[:, i] = fit_k_grid(tf, Yf, k_grid)
        # 网格边界命中率：k_grid 现由 t_grid 生成、与真值无关。若估计量常被截在边界，
        # 「不可恢复档饱和」就可能是网格假象而非统计事实。故如实记录、随结果一并输出。
        # (2026-07-26 外审取证的确认复跑 HP-FAKE-GT 要求给出该率。)
        _edge_hits += int(np.sum((khat[:, i] <= k_grid[0] * (1 + 1e-12))
                                 | (khat[:, i] >= k_grid[-1] * (1 - 1e-12))))
        _edge_tot += int(khat.shape[0])

    x = 1.0 / (R_GAS * T_K)
    lk = np.log(np.clip(khat, 1e-300, None))
    # 逐 rep 的一元线性回归（闭式，向量化）：slope = -Ea
    xm = x.mean()
    Sxx = ((x - xm) ** 2).sum()
    slope = ((lk - lk.mean(axis=1, keepdims=True)) * (x - xm)).sum(axis=1) / Sxx
    intercept = lk.mean(axis=1) - slope * xm
    Ea_hat = -slope
    lnA_hat = intercept

    # 逐温度实测 log k̂ 的标准差——这正是实践者能观测到的 σ_lnk，用它喂 Fisher。
    # 【样本分割】σ 取自前半重复、评价用的 RMSE 取自后半重复：原实现两者同源于全部 400 次重复，
    # 而 E/A/c 用 σ、cond 不用，同源会系统性偏袒用 σ 的准则——正是准则排序结论所依赖之处。
    # 分割后两者独立，排序比较才成立。同时保留全样本口径 sigma_lnk_emp_full 以便对照。
    # (2026-07-26 外审取证的确认复跑 HP-FAKE-GT 指出同源问题，复算坐实后分割。)
    _half = n_reps // 2
    sigma_lnk_emp = np.std(lk[:_half], axis=0, ddof=1)
    sigma_lnk_emp_full = np.std(lk, axis=0, ddof=1)

    _eval = np.zeros(n_reps, bool); _eval[_half:] = True     # 评价集 = 后半重复（与 σ 所用前半不相交）
    ok = np.isfinite(Ea_hat) & np.isfinite(lnA_hat) & _eval
    if ok.sum() < (n_reps - _half) // 2:
        return {"rmse_Ea": np.nan, "frac_ok": float(ok.mean()), "sigma_lnk_emp": sigma_lnk_emp,
                "sigma_lnk_emp_full": sigma_lnk_emp_full,
                "grid_edge_hit_rate": float(_edge_hits / max(_edge_tot, 1))}
    err = np.stack([Ea_hat[ok] - Ea, lnA_hat[ok] - lnA], axis=1)
    return {
        "rmse_Ea": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "bias_Ea": float(np.mean(err[:, 0])),
        "err_stack": err,
        "frac_ok": float(ok.mean()),
        "sigma_lnk_emp": sigma_lnk_emp,
        "sigma_lnk_emp_full": sigma_lnk_emp_full,
        "grid_edge_hit_rate": float(_edge_hits / max(_edge_tot, 1)),
    }


def sample_design(rng) -> dict:
    """独立抽取一个设计——刻意与脚本 122 的嵌套子集相反，每个设计是自由的一次抽样。"""
    Tlo, Thi = ANCHORS["T_C"]["range"]
    n_T = int(rng.integers(3, 7))                                  # 3–6 个温度水平
    dT = float(np.exp(rng.uniform(np.log(2.0), np.log(55.0))))     # 对数均匀，覆盖窄窗到宽窗
    T_start = float(rng.uniform(Tlo, Thi - min(dT, Thi - Tlo)))
    span = min(dT, Thi - T_start)
    if rng.random() < 0.5:                                          # 一半等距、一半随机布点
        T_C = np.linspace(T_start, T_start + span, n_T)
    else:
        inner = np.sort(rng.uniform(0, 1, max(n_T - 2, 0)))
        T_C = T_start + span * np.concatenate([[0.0], inner, [1.0]])[:n_T]
        T_C = np.sort(T_C)
    n_per_T = rng.integers(3, 21, n_T).astype(float)               # 每温度 3–20 条曲线
    n_t = int(rng.integers(4, 8))
    Ea = float(rng.uniform(*ANCHORS["Ea_kJmol"]["range"]))
    # 直接抽【观测层】噪声（响应已归一化到 [0,1]），σ_lnk 不预设、由仿真实测得到，
    # 从而消除"σ_lnk→σ_y 随手换算"这一无根据的错配。σ_y 量程事后核对是否落回锚定的 σ_lnk 区间。
    sigma_y = float(np.exp(rng.uniform(np.log(0.005), np.log(0.08))))
    # lnA 定标：使窗中心温度的特征时间落在观测时窗内（否则设计本身就是废的，与准则无关）
    T_mid_K = float(np.mean(T_C)) + 273.15
    k_mid = float(np.exp(rng.uniform(np.log(0.05), np.log(0.5))))
    lnA = np.log(k_mid) + Ea / (R_GAS * T_mid_K)
    t_grid = np.linspace(0.0, 3.0 / k_mid, n_t + 1)[1:]            # 去掉 t=0（无信息）
    return dict(T_C=T_C, n_per_T=n_per_T, t_grid=t_grid, Ea=Ea, lnA=lnA,
                sigma_y=sigma_y, n_T=n_T, n_t=n_t, dT=float(T_C.max() - T_C.min()),
                N_obs=float(n_per_T.sum() * n_t), k_mid=k_mid)


# ---------------------------------------------------------------- 分析
def boot_ci(vals, rng, n=5000, q=(2.5, 97.5)):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return [float("nan")] * 2
    idx = rng.integers(0, len(vals), (n, len(vals)))
    m = np.median(vals[idx], axis=1)
    return [float(np.percentile(m, q[0])), float(np.percentile(m, q[1]))]


def partial_spearman(x, y, z) -> float:
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
        des = sample_design(rng)
        sim = simulate_design(rng, des["T_C"], des["n_per_T"], des["t_grid"],
                              des["Ea"], des["lnA"], des["sigma_y"], N_REPS)
        if not np.isfinite(sim["rmse_Ea"]):
            continue
        # 准则用【实测】σ_lnk 计算——即实践者手上真正有的量，不用任何预设换算
        sig = np.asarray(sim["sigma_lnk_emp"], float)
        if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
            continue
        sigma_lnk_mean = float(np.mean(sig))
        cr = criteria(des["T_C"], des["n_per_T"], sigma_lnk_mean)          # 同方差特例代理（def:psi 的 σ_i≡σ 特例）
        cr_het = criteria_het(des["T_C"], des["n_per_T"], sig)             # 异方差定义量（正文头条口径 Ψ）
        # 沿退化方向的真值误差（本文定理正是关于这个方向）
        # 【2026-08-21 订正，RYA-569 / 义务 1f9bcf45】两个口径各投影到【自己算子】的最小特征方向：
        #   rmse_deg_het = 头条（异方差定义量 Ψ=E_het 的同一算子方向）
        #   rmse_deg     = 同方差代理对照（E 的同一算子方向），保留以使代理块自成闭环
        # 二者夹角一并落盘，供审计复核方向错配的实际影响幅度。
        err = sim["err_stack"]
        v_hom, v_het = cr["v_deg"], cr_het["v_deg"]
        rmse_deg = float(np.sqrt(np.mean((err @ v_hom) ** 2)))
        rmse_deg_het = float(np.sqrt(np.mean((err @ v_het) ** 2)))
        _c = float(abs(v_hom @ v_het) / (np.linalg.norm(v_hom) * np.linalg.norm(v_het)))
        angle_hom_het_deg = float(np.degrees(np.arccos(min(max(_c, -1.0), 1.0))))
        rows.append({
            "n_T": des["n_T"], "n_t": des["n_t"], "dT": des["dT"], "N_obs": des["N_obs"],
            "grid_edge_hit_rate": sim["grid_edge_hit_rate"],
            "Ea_true": des["Ea"], "sigma_y": des["sigma_y"], "sigma_lnk": sigma_lnk_mean,
            # 【RYA-581 / de049d538e54】异方差算子（定义量）的全部七个判据——同台对决只用这一组
            "E_het": cr_het["E"], "D_het": cr_het["D"], "A_het": cr_het["A"],
            "c_deg_het": cr_het["c_deg"], "c_Ea_het": cr_het["c_Ea"], "cond_het": cr_het["cond"],
            "lambda_min_het": cr_het["lambda_min"], "lambda_max_het": cr_het["lambda_max"],
            # 同方差代理算子的同一组七判据——自成一表，不与上组混排
            "E": cr["E"], "D": cr["D"], "A": cr["A"], "c_deg": cr["c_deg"], "c_Ea": cr["c_Ea"],
            "cond": cr["cond"], "lambda_min": cr["lambda_min"], "lambda_max": cr["lambda_max"],
            "rmse_Ea": sim["rmse_Ea"], "rmse_deg": rmse_deg, "rmse_deg_het": rmse_deg_het,
            "angle_hom_het_deg": angle_hom_het_deg,
            "relerr_Ea": sim["rmse_Ea"] / des["Ea"], "frac_ok": sim["frac_ok"],
        })
        if (d + 1) % 50 == 0:
            print(f"  {d+1}/{N_DESIGNS} 设计完成 ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    print(f"有效设计 {len(df)}/{N_DESIGNS}，用时 {time.time()-t0:.0f}s")

    # 【2026-08-19 订正，RYA-537】原 CRIT 只含同方差代理 E/D，而正文 def:psi 把 Ψ 定义为
    # 异方差装配 λmin(Î₁[w])、w_i=g_i²/σ_i²，同方差式只是 σ_i≡σ 的特例。本脚本每个设计的
    # σ_lnk 实测逐温度不同（E/E_het 比值实测跨 0.0044–3.54），故代理与定义并非同一个量。
    # 现把 E_het/D_het 一并纳入评价列表，正文头条改报异方差口径（定义量），同方差代理保留
    # 为特例对照。（外审取证 feb912c2 指出，回码复核坐实后修正。）
    #
    # 【2026-08-23 订正，RYA-581 / 义务 de049d538e54】上一轮只把 E/D 换成了异方差装配，
    # A/c_deg/c_Ea/cond 仍留在同方差算子上，于是同一张对决表里有两个不同的信息算子。
    # 后果具体而非形式：「按条件数筛查是明确的失败」这一结论无法区分究竟是
    # 「条件数这个判据不行」还是「同方差装配不行」，即判据选择与信息阵装配没有被拆开。
    # 现改为【一张表一个算子】：头条表全部取自异方差定义量算子（CRIT_HET），
    # 同方差代理表全部取自同方差算子（CRIT_HOM），两表结构完全对称、逐项可比；
    # 旧的混排表作为 PRE-FIX 对照一并落盘（operator_mismatch_audit），供复核订正幅度。
    CRIT_HET = ["E_het", "D_het", "A_het", "c_deg_het", "c_Ea_het", "cond_het"]
    CRIT_HOM = ["E", "D", "A", "c_deg", "c_Ea", "cond"]
    CRIT_LEGACY_MIXED = ["E_het", "E", "D_het", "D", "A", "c_deg", "c_Ea", "cond"]   # 仅供 PRE-FIX 对照
    CRIT_ALL = CRIT_HET + CRIT_HOM
    err = df["relerr_Ea"].to_numpy()

    def _is_cond(k: str) -> bool:
        return k.split("_")[0] == "cond"

    # (1) 独立设计上的秩相关（这次 bootstrap 合法：设计彼此独立）
    #     两张表各自算子闭环；dT / N_obs 是纯设计量、不属任何算子，两表共用。
    marg = {k: float(spearmanr(df[k], err)[0]) for k in CRIT_HET}
    marg["dT"] = float(spearmanr(df["dT"], err)[0])
    marg["N_obs"] = float(spearmanr(df["N_obs"], err)[0])
    part = {k: partial_spearman(df[k].to_numpy(), err, df["dT"].to_numpy()) for k in CRIT_HET}
    marg_hom = {k: float(spearmanr(df[k], err)[0]) for k in CRIT_HOM}
    part_hom = {k: partial_spearman(df[k].to_numpy(), err, df["dT"].to_numpy()) for k in CRIT_HOM}

    # E 与 D 之差的 bootstrap（设计独立 → 重抽样合法）
    # 【2026-08-23，RYA-581】自助流按用途分开，不再共用一条 rb。
    # 起因是本轮的实测教训：判据表增删一列，会顺着 boot_ci 的调用次数把 rb 的流位置整体推移，
    # 于是速率律与饱和检验的置信区间【在没有任何相关改动的情况下】跟着变。
    # 分流之后，rb 只服务「判据表」（E−D 差 + 选型 regret），速率律与饱和各有独立流，
    # 此后改判据表不再扰动稿面所引的速率/饱和区间。点估计本来就是确定的，不受任何流影响。
    rb = np.random.default_rng(SEED + 1)          # 判据表：E−D 差、选型场景与其 regret 中位 CI
    rb_rate = np.random.default_rng(SEED + 11)    # 速率律：两轴与单轴受控的斜率自助
    rb_sat = np.random.default_rng(SEED + 12)     # 饱和检验：实测/外推比值的自助
    n = len(df)
    gaps = []
    Ev, Dv = df["E_het"].to_numpy(), df["D_het"].to_numpy()
    for _ in range(2000):
        i = rb.integers(0, n, n)
        gaps.append(abs(spearmanr(Ev[i], err[i])[0]) - abs(spearmanr(Dv[i], err[i])[0]))
    gaps = np.array(gaps)
    ED = {"point": float(np.mean(gaps)),
          "ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
          "P_D_at_least_as_good": float(np.mean(gaps <= 0))}

    # (2) 决策相关量：模拟真实的选型场景——从预算相近的候选设计中，各准则各选其最优，
    #     比较所选设计的【真实】误差。重复数千次以得到分布，而非 6 个分位箱的单点。
    #     这正是外审要求的"决策相关量"：相关系数之差不可读，误差上升百分之几可读。
    K_CAND = 8            # 每次决策面对的候选设计数（实践中通常在少数几个方案里挑）
    N_SCEN = 3000         # 决策场景数
    df = df.sort_values("N_obs").reset_index(drop=True)
    Nobs = df["N_obs"].to_numpy()
    relerr = df["relerr_Ea"].to_numpy()
    crit_vals = {k: df[k].to_numpy() for k in CRIT_ALL}
    picks = {k: [] for k in CRIT_ALL}
    agree_ED = 0
    agree_ED_hom = 0
    oracle_gap = []
    for _ in range(N_SCEN):
        # 预算配平：在 N_obs 排序上取一个窗口，窗口内预算相近
        lo = rb.integers(0, len(df) - K_CAND)
        idx = np.arange(lo, lo + K_CAND)
        if Nobs[idx].max() / max(Nobs[idx].min(), 1e-9) > 2.0:   # 预算差超过 2 倍则跳过
            continue
        best_true = relerr[idx].min()
        chosen = {}
        # 两张表在【同一批场景】上选型，故 het 与 hom 的 regret 逐项可比
        for k in CRIT_ALL:
            v = crit_vals[k][idx]
            j = idx[np.argmin(v)] if _is_cond(k) else idx[np.argmax(v)]
            chosen[k] = j
            picks[k].append(relerr[j] / best_true - 1.0)          # 相对【神谕最优】的惩罚
        agree_ED += int(chosen["E_het"] == chosen["D_het"])
        agree_ED_hom += int(chosen["E"] == chosen["D"])
        oracle_gap.append(relerr[idx].max() / best_true - 1.0)
    n_scen = len(picks["E_het"])
    picks = {k: {"median_regret_vs_oracle": float(np.median(v)),
                 "mean_regret_vs_oracle": float(np.mean(v)),
                 "p90_regret_vs_oracle": float(np.percentile(v, 90)),
                 "ci95_of_median": boot_ci(v, rb)} for k, v in picks.items()}
    decision = {
        "protocol": (f"每个场景从预算相近（N_obs 极差≤2×）的 {K_CAND} 个候选设计中，"
                     f"各准则各选其最优，记录所选设计真实相对误差相对该场景【神谕最优】的超额（regret）。"
                     f"共 {n_scen} 个场景。"
                     "【RYA-581】头条表 by_criterion 只含异方差定义量算子导出的六个判据，"
                     "同方差代理表另列于 by_criterion_homoscedastic_surrogate，两表同场景同协议。"),
        "n_scenarios": n_scen,
        "operator": "heteroscedastic (def:psi 定义量 Î₁[w]，w_i=g_i²/σ_i²)",
        "E_and_D_pick_same_design_rate": agree_ED / max(n_scen, 1),
        "worst_vs_best_in_candidate_set_median": float(np.median(oracle_gap)),
        "by_criterion": {k: picks[k] for k in CRIT_HET},
        "by_criterion_homoscedastic_surrogate": {k: picks[k] for k in CRIT_HOM},
        "E_and_D_pick_same_design_rate_homoscedastic_surrogate": agree_ED_hom / max(n_scen, 1),
    }

    # (3) 两轴验证：固定另一轴，误差对 Ψ 的 log-log 斜率应为 −1/2
    # 【2026-08-21 订正，RYA-569 / 义务 1f9bcf45】ycol 与 col 必须来自同一个信息算子：
    #   头条    col="E_het" ↔ ycol="rmse_deg_het"
    #   代理对照 col="E"     ↔ ycol="rmse_deg"
    # 原实现 ycol 恒为 "rmse_deg"，与 col="E_het" 配成跨算子对，即被取证指出的错配。
    def slope_vs_psi(sub, nboot=1000, col="E_het", ycol=None):
        if ycol is None:
            ycol = "rmse_deg_het" if col == "E_het" else "rmse_deg"
        if len(sub) < 8:
            return None
        lx, ly = np.log(sub[col].to_numpy()), np.log(sub[ycol].to_numpy())
        co = np.polyfit(lx, ly, 1)
        pred = np.polyval(co, lx)
        r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
        bs = []
        for _ in range(nboot):                       # 设计独立 → 重抽样合法
            i = rb_rate.integers(0, len(lx), len(lx))
            bs.append(np.polyfit(lx[i], ly[i], 1)[0])
        return {"slope": float(co[0]), "slope_ci95": [float(np.percentile(bs, 2.5)),
                                                      float(np.percentile(bs, 97.5))],
                "r2": float(r2), "n": int(len(sub)),
                "psi_col": col, "error_col": ycol}

    narrow = df[df.dT <= df.dT.median()]
    wide = df[df.dT > df.dT.median()]
    small = df[df.N_obs <= df.N_obs.median()]
    large = df[df.N_obs > df.N_obs.median()]
    two_axis = {
        "all": slope_vs_psi(df),
        "narrow_window_half": slope_vs_psi(narrow),
        "wide_window_half": slope_vs_psi(wide),
        "small_n_half": slope_vs_psi(small),
        "large_n_half": slope_vs_psi(large),
        "theory": -0.5,
        "note": ("误差沿退化方向对 Ψ 的 log-log 斜率；理论预测 −1/2。"
                 "【口径更正】按 ΔT 或按 N 中位数切半后，两个自变量在每个子组【仍同时变动】，"
                 "故这是跨粗分层的稳健性，不等于两轴分别受控验证；后者见 controlled_axis。"),
    }

    # 真正的单轴受控：把另一轴锁在窄带内（相对极差 ≤ TOL），只让目标轴变动。
    # 中位数切半做不到这一点——切完两轴仍在各自子组内一起变。
    # (2026-07-26 外审取证的确认复跑 HP-SCOPE-INFLATE 指出，复算坐实后补此分析。)
    def controlled(fix_col: str, tol: float = 0.25, col: str = "E_het", ycol=None) -> dict:
        out = []
        v = df[fix_col].to_numpy()
        lo_q = np.quantile(v, np.linspace(0, 1, 6))
        for i in range(5):
            band = df[(df[fix_col] >= lo_q[i]) & (df[fix_col] <= lo_q[i + 1])]
            if len(band) < 40:
                continue
            rng_ratio = float(band[fix_col].max() / max(band[fix_col].min(), 1e-12))
            if rng_ratio > 1 + tol * 4:      # 该带内固定轴仍变动过大则跳过，如实记录
                continue
            r = slope_vs_psi(band, col=col, ycol=ycol)
            r["band"] = [float(lo_q[i]), float(lo_q[i + 1])]
            r["fixed_axis_range_ratio"] = rng_ratio
            out.append(r)
        return {"bands": out,
                "n_bands_used": len(out),
                "slope_median": float(np.median([b["slope"] for b in out])) if out else None}

    two_axis["controlled_axis"] = {
        "vary_n_fix_dT": controlled("dT"),
        "vary_dT_fix_n": controlled("N_obs"),
        "note": ("在另一轴的窄带内（带内极差比 ≤2×）拟合 log-log 斜率，"
                 "使 Ψ 的变化主要来自目标轴。这才是两轴分别受控的检验。"),
    }

    # 同方差代理（σ_i≡σ̄）下的同一批拟合，作为特例对照一并落盘（RYA-537）
    two_axis_homo = {
        "all": slope_vs_psi(df, col="E"),
        "narrow_window_half": slope_vs_psi(narrow, col="E"),
        "wide_window_half": slope_vs_psi(wide, col="E"),
        "small_n_half": slope_vs_psi(small, col="E"),
        "large_n_half": slope_vs_psi(large, col="E"),
        "controlled_axis": {"vary_n_fix_dT": controlled("dT", col="E"),
                            "vary_dT_fix_n": controlled("N_obs", col="E")},
        "theory": -0.5,
        "note": ("同方差代理 E（σ_i 取均值）下的同一组拟合，误差投影用同一算子的方向 rmse_deg；"
                 "正文头条用异方差定义量 E_het 及其自己的方向 rmse_deg_het。"
                 "【RYA-569】本块与头条块各自算子闭环，不再跨算子配对。"),
    }

    # --- 不可恢复档：低 Ψ 端误差是否【饱和】？Cramér--Rao 会继续发散，饱和是三分律的独有预测 ---
    PSI_COL = "E_het"          # RYA-537：饱和检验同样按定义量（异方差 Ψ）分档
    ERR_COL = "rmse_deg_het"   # RYA-569：误差方向须与 PSI_COL 同算子（义务 1f9bcf45）
    lo_mask = df[PSI_COL] <= df[PSI_COL].quantile(0.15)
    hi_mask = df[PSI_COL] >= df[PSI_COL].quantile(0.50)
    def fit_half(mask):
        sub = df[mask]
        lx, ly = np.log(sub[PSI_COL].to_numpy()), np.log(sub[ERR_COL].to_numpy())
        return float(np.polyfit(lx, ly, 1)[0]), int(len(sub))
    slope_lo, n_lo = fit_half(lo_mask)
    slope_hi, n_hi = fit_half(hi_mask)
    # 用高-Ψ 段（可恢复档）外推到最低 Ψ，比较实测误差是否被压在外推值之下（=饱和）
    sub_hi = df[hi_mask]
    co_hi = np.polyfit(np.log(sub_hi[PSI_COL]), np.log(sub_hi[ERR_COL]), 1)
    sub_lo = df[lo_mask]
    pred_lo = np.exp(np.polyval(co_hi, np.log(sub_lo[PSI_COL])))
    ratio = (sub_lo[ERR_COL] / pred_lo).to_numpy()
    saturation = {
        "slope_low_Psi_15pct": slope_lo, "n_low": n_lo,
        "slope_high_Psi_50pct": slope_hi, "n_high": n_hi,
        "median_observed_over_extrapolated": float(np.median(ratio)),
        "ci95": boot_ci(ratio, rb_sat),
        "relerr_Ea_median_at_low_Psi": float(sub_lo["relerr_Ea"].median()),
        "note": ("可恢复档斜率外推到最低 15% 的 Ψ；比值 <1 即实测误差低于外推 = 误差饱和（第三档）。"
                 "Cramér--Rao 界不预测饱和，故这是三分律相对纯 CRB 的独有预测。"
                 "饱和的力学来源是 Θ 紧致：估计量被约束在有界参数空间内，误差不能无界增长。"),
    }

    # --- 义务 1f9bcf45 的取证对照块（RYA-569）-------------------------------------
    # 用【独立 RNG】重算三组配对，故绝不扰动上面任何量的随机流：
    #   (a) PRE-FIX 错配：Ψ=E_het 配同方差方向 rmse_deg —— 即被取证指出的旧头条
    #   (b) POST-FIX 正配：Ψ=E_het 配 rmse_deg_het —— 现头条
    #   (c) 两者之差，以及两算子最小特征方向夹角与 rmse 比值的分布
    # (a)(b) 在同一条 RNG 流上先后计算，故二者的自助区间可直接相减比较；
    # 头条块本身仍用主流程的 rb，数值以 res["two_axis_rate_validation"] 为准。
    rb_audit = np.random.default_rng(SEED + 2)

    def _fit_audit(sub, col, ycol, nboot=1000):
        if len(sub) < 8:
            return None
        lx, ly = np.log(sub[col].to_numpy()), np.log(sub[ycol].to_numpy())
        co = np.polyfit(lx, ly, 1)
        pred = np.polyval(co, lx)
        r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
        bs = [np.polyfit(lx[i], ly[i], 1)[0]
              for i in (rb_audit.integers(0, len(lx), len(lx)) for _ in range(nboot))]
        return {"slope": float(co[0]),
                "slope_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "r2": float(r2), "n": int(len(sub)), "psi_col": col, "error_col": ycol}

    _strata = {"all": df, "narrow_window_half": narrow, "wide_window_half": wide,
               "small_n_half": small, "large_n_half": large}
    _pre = {k: _fit_audit(v, "E_het", "rmse_deg") for k, v in _strata.items()}
    _post = {k: _fit_audit(v, "E_het", "rmse_deg_het") for k, v in _strata.items()}

    def _sat(err_col):
        lo = df[df[PSI_COL] <= df[PSI_COL].quantile(0.15)]
        hi = df[df[PSI_COL] >= df[PSI_COL].quantile(0.50)]
        c = np.polyfit(np.log(hi[PSI_COL]), np.log(hi[err_col]), 1)
        r = (lo[err_col] / np.exp(np.polyval(c, np.log(lo[PSI_COL])))).to_numpy()
        return {"slope_low_Psi_15pct": float(np.polyfit(np.log(lo[PSI_COL]), np.log(lo[err_col]), 1)[0]),
                "slope_high_Psi_50pct": float(c[0]),
                "median_observed_over_extrapolated": float(np.median(r))}

    _ang = df["angle_hom_het_deg"].to_numpy()
    _rat = (df["rmse_deg_het"] / df["rmse_deg"]).to_numpy()
    projection_operator_audit = {
        "obligation": "1f9bcf45",
        "defect": ("PRE-FIX 的头条把异方差算子的特征值 E_het 与同方差算子的最小特征方向 v_deg 配对；"
                   "POST-FIX 两者同出于异方差算子。"),
        "eigendirection_angle_deg": {
            "median": float(np.median(_ang)), "p95": float(np.percentile(_ang, 95)),
            "max": float(_ang.max()),
            "note": ("这是【实测】夹角。RYA-569 任务书预测中位 0.000°/最大 0.001°，"
                     "其论据（J 两列量级差 1e4）是把 R 当成 8.314 J/(mol·K) 的单位错误；"
                     "本脚本 R_GAS=8.314e-3 kJ/(mol·K)，两列量级比只有 2.1–2.6 倍。"
                     "预检见 04outputs/138eigendirection_angle_check.json。"),
        },
        "rmse_ratio_het_over_hom": {
            "median": float(np.median(_rat)), "p95": float(np.percentile(_rat, 95)),
            "max": float(_rat.max()), "min": float(_rat.min()),
            "note": "rmse 沿方向的二次型在主轴附近二阶平坦，故比值远比 sin(夹角) 小。",
        },
        "two_axis_pre_fix_mismatched": _pre,
        "two_axis_post_fix_matched": _post,
        "two_axis_slope_delta_post_minus_pre": {
            k: (None if (_pre[k] is None or _post[k] is None)
                else float(_post[k]["slope"] - _pre[k]["slope"])) for k in _strata},
        "saturation_pre_fix_mismatched": _sat("rmse_deg"),
        "saturation_post_fix_matched": _sat("rmse_deg_het"),
        "note": ("本块自带独立 RNG（SEED+2），与头条块的自助抽样流互不影响；"
                 "因此本块的 CI 与 res['two_axis_rate_validation'] 的 CI 数值不必逐位相同，"
                 "本块只用于 PRE/POST 的同流对照，稿面数字一律引头条块。"),
    }

    # --- 义务 de049d538e54 的取证对照块（RYA-581）--------------------------------
    # 用【独立 RNG】重算 PRE-FIX 的混排表，故绝不扰动上面任何量的随机流。
    #   PRE-FIX：E_het/D_het 取自异方差算子，而 A/c_deg/c_Ea/cond 取自同方差算子（跨算子同台）
    #   POST-FIX：六个判据全部取自异方差算子（本轮头条）
    # 同时把两处此前只活在脚本注释里、任何产物中都取不到的数落盘成字段
    # （外审核对报的两处真缺口：E/E_het 比值区间、网格边界命中率）。
    rb_op = np.random.default_rng(SEED + 3)
    # ⚠️ 本块位于 df.sort_values("N_obs") 之后，而上文的 err 取自【排序前】的 df。
    #    故本块一律用排序后的 relerr / df["dT"]，绝不用 err——否则两列错位、Spearman 全是垃圾。
    #    （首次实现即踩到这个坑：A 的 PRE 值算成 +0.0455，与旧产物 −0.9349 对不上，
    #      正是错位所致；此注释留作绊线。）
    _err_sorted = relerr
    _dT_sorted = df["dT"].to_numpy()

    _ratio_E = (df["E"] / df["E_het"]).to_numpy()
    _ratio_cond = (df["cond"] / df["cond_het"]).to_numpy()
    _edge = df["grid_edge_hit_rate"].to_numpy()

    def _stats(v):
        v = np.asarray(v, float)
        return {"min": float(v.min()), "p05": float(np.percentile(v, 5)),
                "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                "max": float(v.max())}

    operator_mismatch_audit = {
        "obligation": "de049d538e54",
        "defect": ("PRE-FIX 的准则对决把 E_het/D_het（异方差算子 fisher_het）与 "
                   "A/c_deg/c_Ea/cond（同方差算子 fisher）排在同一张表上比较；"
                   "两个算子并非相差一个公共标量，故该表无法把「判据选择」与「信息阵装配」拆开。"),
        # 【外审核对缺口 1】E/E_het 比值此前只写在本文件第 ~370 行的注释里，产物中取不到。
        "E_over_E_het_ratio": {
            **_stats(_ratio_E),
            "note": ("同方差代理 E 与异方差定义量 E_het 之比。若两算子只差一个公共标量，"
                     "该比值应为常数；实测跨越两个数量级，故不是公共标量，"
                     "混排表因此不是同一个信息阵上的判据对决。"
                     "正文 §sec:mc 所引区间即取自本字段的 min/max。"),
        },
        "cond_over_cond_het_ratio": {
            **_stats(_ratio_cond),
            "note": "条件数是尺度不变量，但对【装配口径】并不不变；本字段量化该敏感性。",
        },
        # 【外审核对缺口 2】grid-edge hit rate 此前只是 designs CSV 的一列，JSON 无该键。
        "grid_edge_hit_rate": {
            **_stats(_edge),
            "mean": float(_edge.mean()),
            "n_designs_with_any_hit": int(np.sum(_edge > 0)),
            "n_designs": int(len(_edge)),
            "note": ("k 搜索网格的边界命中率，逐设计统计。全为 0 即无一设计的 k̂ 被截在网格端点，"
                     "故不可恢复档的饱和不是网格假象。正文 §sec:mc 所引「命中率恒为 0」即本字段。"),
        },
        "spearman_marginal_pre_fix_mixed": {k: float(spearmanr(df[k], _err_sorted)[0])
                                            for k in CRIT_LEGACY_MIXED},
        "spearman_marginal_post_fix_heteroscedastic": {k: marg[k] for k in CRIT_HET},
        "spearman_partial_pre_fix_mixed": {
            k: partial_spearman(df[k].to_numpy(), _err_sorted, _dT_sorted)
            for k in CRIT_LEGACY_MIXED},
        "decision_regret_pre_fix_mixed": None,   # 下面按同一批场景填
        "note": ("PRE 表与 POST 表的差别只在 A/c_deg/c_Ea/cond 四列所用的算子；"
                 "E_het/D_het 两列两表相同（本轮之前即已是异方差）。"
                 "本块自带独立 RNG（SEED+3），与头条块的自助流互不影响。"),
    }

    # PRE-FIX 混排表在【同一批场景】上的 regret（复用上面已记录的 picks 原始序列不可行，
    # 因 picks 已被汇总；故用同一 K_CAND/N_SCEN 协议、独立 RNG 重跑一遍，
    # 只用于 PRE/POST 的量级对照，稿面数字一律引头条块。）
    _pre_cols = ["E_het", "D_het", "A", "c_deg", "c_Ea", "cond"]
    _pre_vals = {k: df[k].to_numpy() for k in _pre_cols}
    _pre_picks = {k: [] for k in _pre_cols}
    for _ in range(N_SCEN):
        lo = rb_op.integers(0, len(df) - K_CAND)
        idx = np.arange(lo, lo + K_CAND)
        if Nobs[idx].max() / max(Nobs[idx].min(), 1e-9) > 2.0:
            continue
        best_true = relerr[idx].min()
        for k in _pre_cols:
            v = _pre_vals[k][idx]
            j = idx[np.argmin(v)] if _is_cond(k) else idx[np.argmax(v)]
            _pre_picks[k].append(relerr[j] / best_true - 1.0)
    operator_mismatch_audit["decision_regret_pre_fix_mixed"] = {
        "n_scenarios": len(_pre_picks["E_het"]),
        "rng": "独立流 SEED+3，故与头条块场景不同批；只可用于量级对照",
        "by_criterion": {k: {"median_regret_vs_oracle": float(np.median(v)),
                             "mean_regret_vs_oracle": float(np.mean(v)),
                             "p90_regret_vs_oracle": float(np.percentile(v, 90))}
                         for k, v in _pre_picks.items()},
    }

    res = {
        "script": Path(__file__).name, "seed": SEED,
        "n_designs_requested": N_DESIGNS, "n_designs_valid": int(len(df)), "n_reps_per_design": N_REPS,
        "anchors": ANCHORS,
        "design_independence": "每个设计为独立抽样（温度窗/水平数/布点/重复数/θ/σ 各自随机），非嵌套子集",
        "bootstrap_streams": {
            "criterion_table": "default_rng(SEED+1)：E−D 差自助、3000 个选型场景、各判据 regret 中位 CI",
            "rate_law": "default_rng(SEED+11)：two_axis / controlled_axis / 同方差代理块的斜率自助",
            "saturation": "default_rng(SEED+12)：低-Ψ 端实测/外推比值的自助",
            "projection_operator_audit": "default_rng(SEED+2)",
            "operator_mismatch_audit": "default_rng(SEED+3)",
            "note": ("【RYA-581】此前全部共用 SEED+1 一条流，判据表增删一列即整体推移后续流位置，"
                     "使速率律与饱和的 CI 在无相关改动时也变。分流后互不影响。"
                     "点估计（斜率、中位数、Spearman）本来就与流无关，任何轮次都可逐位复算。"),
        },
        # 【2026-08-10 订正，RYA-105】原串写「%d 次重复的 RMSE」，但自 2026-07-26 起代码已做样本分割：
        # σ_logk 取自前 N_REPS/2 次重复、评价用的 RMSE 只取后 N_REPS/2 次（见上方 _half / _eval）。
        # 正文 §sec:mc 写的「前 200 / 后 200」是对的，错的是本产物的元数据串。
        "ground_truth": ("已知 θ；误差为【后 %d 次】重复（评价集，与估 σ_logk 所用的前 %d 次不相交）的 RMSE"
                         "（真统计估计误差，非 bootstrap 噪声传播）；每设计共 %d 次重复"
                         % (N_REPS - N_REPS // 2, N_REPS // 2, N_REPS)),
        "spearman_marginal": marg,
        "spearman_partial_controlling_dT": part,
        "spearman_marginal_homoscedastic_surrogate": marg_hom,
        "spearman_partial_controlling_dT_homoscedastic_surrogate": part_hom,
        "E_vs_D_gap_bootstrap": ED,
        "decision_relevant_regret": decision,
        "psi_convention": ("头条口径 = 异方差定义量 E_het = λmin(Î₁[w])，w_i=g_i²/σ_i²（正文 def:psi）；"
                           "【RYA-581】同台对决的六个判据（E/D/A/c_deg/c_Ea/cond）一律由【同一个】"
                           "异方差算子导出（列名后缀 _het）；同方差特例代理表（无后缀列）结构完全对称、"
                           "自成一表，两表不混排。此前 A/c_deg/c_Ea/cond 取自同方差算子而 E/D 取自异方差算子，"
                           "属算子不匹配，PRE/POST 对照见 operator_mismatch_audit。"
                           "【RYA-569】误差投影方向与 Ψ 必须同算子：头条用 rmse_deg_het"
                           "（异方差算子的最小特征方向），代理块用 rmse_deg（同方差算子方向）。"),
        "two_axis_rate_validation": two_axis,
        "two_axis_rate_validation_homoscedastic_surrogate": two_axis_homo,
        "projection_operator_audit": projection_operator_audit,
        "operator_mismatch_audit": operator_mismatch_audit,
        "unrecoverable_branch_saturation": saturation,
        "honesty": [
            "这是模拟研究：设计独立、真值已知、误差为真 RMSE，但数据非真实测量。",
            "它替代的是被嵌套结构毁掉的自由度，不替代真实数据。",
            "量程全部锚定到本项目已核验的真实值（见 anchors），非凭空设定。",
        ],
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print("\n=== 边际 Spearman · 头条表（全部取自异方差定义量算子）===")
    for k, v in sorted(marg.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:>12s} {v:+.3f}" + (f"   偏(控ΔT) {part[k]:+.3f}" if k in part else ""))
    print("=== 边际 Spearman · 同方差代理表（全部取自同方差算子）===")
    for k, v in sorted(marg_hom.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:>12s} {v:+.3f}   偏(控ΔT) {part_hom[k]:+.3f}")
    print(f"\nE−D 差 bootstrap: {ED['point']:+.3f} CI{[round(x,3) for x in ED['ci95']]} "
          f"P(D≥E)={ED['P_D_at_least_as_good']:.1%}")
    print(f"\n=== 决策相关：{n_scen} 个选型场景，各准则所选设计相对神谕最优的 regret ===")
    print(f"  （候选集内最差 vs 最优的真实误差差距中位数 {decision['worst_vs_best_in_candidate_set_median']:+.1%}，"
          f"即这个决策本身有多大空间）")
    for k in CRIT_HET + CRIT_HOM:
        v = picks[k]
        tag = "[het]" if k in CRIT_HET else "[hom]"
        print(f"  {tag} {k:>12s} 中位 {v['median_regret_vs_oracle']:+.1%}  均值 {v['mean_regret_vs_oracle']:+.1%}"
              f"  P90 {v['p90_regret_vs_oracle']:+.1%}  中位CI{[f'{x:+.1%}' for x in v['ci95_of_median']]}")
    print(f"  E 与 D 选中同一个设计的比例：het {decision['E_and_D_pick_same_design_rate']:.1%}"
          f" / hom {decision['E_and_D_pick_same_design_rate_homoscedastic_surrogate']:.1%}")
    _oma = operator_mismatch_audit
    print("\n=== 义务 de049d538e54：算子不匹配的实际影响 ===")
    _r = _oma["E_over_E_het_ratio"]
    print(f"  E/E_het 比值  min {_r['min']:.4f}  中位 {_r['median']:.4f}  max {_r['max']:.4f}")
    _rc = _oma["cond_over_cond_het_ratio"]
    print(f"  cond/cond_het 比值  min {_rc['min']:.4f}  中位 {_rc['median']:.4f}  max {_rc['max']:.4f}")
    _e = _oma["grid_edge_hit_rate"]
    print(f"  网格边界命中率  max {_e['max']:.6f}  有命中的设计数 {_e['n_designs_with_any_hit']}/{_e['n_designs']}")
    print("  PRE(混排) vs POST(同算子) 边际 Spearman：")
    for k in ("A", "c_deg", "c_Ea", "cond"):
        print(f"    {k:>7s} 同方差 {_oma['spearman_marginal_pre_fix_mixed'][k]:+.4f}"
              f"  →  {k+'_het':>11s} 异方差 {marg[k + '_het']:+.4f}")
    print("\n=== 两轴速率验证（理论斜率 −0.5）===")
    for k, v in two_axis.items():
        if isinstance(v, dict) and "slope" in v:
            print(f"  {k:>20s} slope {v['slope']:+.3f} CI{[round(x,3) for x in v['slope_ci95']]}"
                  f"  R²={v['r2']:.3f}  n={v['n']}")
    _ca = two_axis["controlled_axis"]
    print("  --- 单轴受控（另一轴锁在窄带内）---")
    for k in ("vary_n_fix_dT", "vary_dT_fix_n"):
        s = _ca[k]["slope_median"]
        print(f"  {k:>20s} 带数 {_ca[k]['n_bands_used']}  斜率中位数 "
              f"{format(s, '+.3f') if s is not None else 'n/a（无合格窄带）'}")
    print("\n=== 不可恢复档：低 Ψ 端饱和检验 ===")
    print(f"  高-Ψ 半（可恢复）斜率 {saturation['slope_high_Psi_50pct']:+.3f} (n={saturation['n_high']})；"
          f"最低 15% Ψ 斜率 {saturation['slope_low_Psi_15pct']:+.3f} (n={saturation['n_low']})")
    print(f"  实测/外推 中位比 {saturation['median_observed_over_extrapolated']:.3f} "
          f"CI{[round(x,3) for x in saturation['ci95']]}  （<1 = 饱和）")
    _poa = projection_operator_audit
    print("\n=== 义务 1f9bcf45：投影算子错配的实际影响（独立 RNG 同流对照）===")
    print(f"  两算子最小特征方向夹角  中位 {_poa['eigendirection_angle_deg']['median']:.4f}°"
          f"  p95 {_poa['eigendirection_angle_deg']['p95']:.4f}°"
          f"  最大 {_poa['eigendirection_angle_deg']['max']:.4f}°")
    print(f"  rmse_deg_het/rmse_deg   中位 {_poa['rmse_ratio_het_over_hom']['median']:.9f}"
          f"  p95 {_poa['rmse_ratio_het_over_hom']['p95']:.9f}"
          f"  最大 {_poa['rmse_ratio_het_over_hom']['max']:.9f}")
    for k, d in _poa["two_axis_slope_delta_post_minus_pre"].items():
        pre = _poa["two_axis_pre_fix_mismatched"][k]; post = _poa["two_axis_post_fix_matched"][k]
        if pre is None or post is None:
            continue
        print(f"  {k:>20s} 斜率 PRE {pre['slope']:+.6f} → POST {post['slope']:+.6f}  Δ {d:+.2e}"
              f"   R² {pre['r2']:.6f} → {post['r2']:.6f}")
    print(f"\n写出 {OUT}")


if __name__ == "__main__":
    main()
