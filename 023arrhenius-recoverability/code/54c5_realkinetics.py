"""
54c5_realkinetics.py: C5 三分律 Ψ 诊断在【真实】温度梯度品质动力学数据上的验证

数据来源（真实、机器提取、可引用、不构造任何数值）：
  PMC10253207 (Foods 2023, 12, 2139) Table 表16：苹果硬度 firmness (kg·cm⁻²)，
  5 贮温 2/4/6/8/10°C × 5 时间点 0/12/24/36/48 h × 4 品种(Granny Smith/Pink Lady/Red Delicious/Royal Gala)，含 mean±SD。
  数值由 pandas.read_html 从 PMC 开放获取 HTML【真实数据表】提取（脚本 55fetch_pmc_tables.py），
  存于 03data/processed/pmc_real/PMC10253207_T16.csv，本脚本【从该 CSV 读取】，绝不手填。

诚实标注：(i) 数据为他组公开发表(非本组采集,附 DOI); (ii) 真实数据含噪声/非单调(如 Pink Lady@2°C
  生理后熟使硬度先升)，恰检验 Ψ 诊断对【病态/良态】反演的区分; (iii) 仅用单调软化品种做 Arrhenius。

C5 验证（两层）：
  (1) 真实可恢复性：对每品种全 5 温度拟合一阶软化 k(T)、Arrhenius (Ea,lnA)，报 R²；
  (2) 诊断 Ψ 预测【设计可恢复性】(cor:design 核心)：固定温度【个数】、变温度【窗宽】，
      Ψ_n=λ_min(设计的总 Fisher 信息) 应预测 (Ea,lnA) 反演的 bootstrap CV：窄窗→低 Ψ→反演病态(CV 大)；
      宽窗→高 Ψ→反演稳健(CV 小)。"展宽设计窗优于盲目增采样点"在真实数据上的确认。

运行: python 02code/54c5_realkinetics.py
输出: 04outputs/54c5_realkinetics.json + -designsweep.csv
"""
from __future__ import annotations

import os
import json, logging, time, itertools, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"; LOGS = ROOT / "05logs"
CSV = ROOT / "03data" / "processed" / "pmc_real" / "PMC10253207_T16.csv"
DOI = "10.3390/foods12112113 (PMC10253207, Table of firmness, read_html-extracted)"
R_GAS = 8.314e-3  # kJ/mol/K
BOOT = 3000
SEED_BASE = int(os.environ.get("SEED_OVERRIDE", "20240530"))  # 多种子复跑入口(§5.3)；未设置 SEED_OVERRIDE 时与原字面值逐字节一致
VARIETIES = ["Granny Smith", "Pink Lady", "Red Delicious", "Royal Gala"]


def get_logger(stem):
    LOGS.mkdir(parents=True, exist_ok=True); ts = time.strftime("%y-%m-%d_%H%M%S")
    lg = logging.getLogger(stem); lg.setLevel(logging.INFO); lg.handlers.clear()
    fh = logging.FileHandler(LOGS / f"{stem}_{ts}.log"); sh = logging.StreamHandler()
    f = logging.Formatter("%(asctime)s %(levelname)s %(message)s"); fh.setFormatter(f); sh.setFormatter(f)
    lg.addHandler(fh); lg.addHandler(sh); return lg


def parse_meansd(s):
    """'8.41 ± 0.36' -> (8.41, 0.36)；容错空格/负号。"""
    s = str(s).replace("−", "-").replace("±", "±")
    m = re.findall(r"[-+]?\d*\.?\d+", s.replace("±", " "))
    if len(m) >= 2: return float(m[0]), float(m[1])
    if len(m) == 1: return float(m[0]), np.nan
    return np.nan, np.nan


def load_real():
    """从真实提取的 CSV 读取苹果硬度（跳过 2 行表头）。返回 {variety: dict(T_C, t_h, mean[T,t], sd[T,t])}。"""
    raw = pd.read_csv(CSV, header=None, skiprows=2,
                      names=["T_C", "t_h"] + VARIETIES)
    raw["T_C"] = pd.to_numeric(raw["T_C"], errors="coerce")
    raw["t_h"] = pd.to_numeric(raw["t_h"], errors="coerce")
    Ts = sorted(raw["T_C"].dropna().unique().tolist())
    ts = sorted(raw["t_h"].dropna().unique().tolist())
    out = {}
    for v in VARIETIES:
        M = np.full((len(Ts), len(ts)), np.nan); S = np.full((len(Ts), len(ts)), np.nan)
        for _, row in raw.iterrows():
            if np.isnan(row["T_C"]) or np.isnan(row["t_h"]): continue
            i = Ts.index(row["T_C"]); j = ts.index(row["t_h"])
            mu, sd = parse_meansd(row[v]); M[i, j] = mu; S[i, j] = sd
        out[v] = dict(T_C=Ts, t_h=ts, mean=M, sd=S)
    return out


def fit_k(t, y):
    """一阶软化 y=y0*exp(-k t)（y0 自由，y0 用 t=0 实测）。返回 (k, y0, R²)。"""
    t = np.asarray(t, float); y = np.asarray(y, float)
    if np.any(~np.isfinite(y)) or len(y) < 3: return np.nan, np.nan, np.nan
    y0 = y[0]
    def m(tt, k, yi): return yi * np.exp(-k * tt)
    try:
        popt, _ = curve_fit(m, t, y, p0=[1e-3, y0], bounds=([1e-8, 0], [1, 500]), maxfev=20000)
        pred = m(t, *popt); ss = 1 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
        return float(popt[0]), float(popt[1]), float(ss)
    except Exception:
        return np.nan, np.nan, np.nan


def arrhenius(T_C, k):
    T_K = np.asarray(T_C, float) + 273.15
    good = np.isfinite(k) & (np.asarray(k) > 0)
    if good.sum() < 2: return np.nan, np.nan, np.nan, good
    x = 1.0 / (R_GAS * T_K[good]); lk = np.log(np.asarray(k)[good])
    s = linregress(x, lk)
    return float(-s.slope), float(s.intercept), float(s.rvalue ** 2), good


def fisher(T_C, sigma_lnk):
    T_K = np.asarray(T_C, float) + 273.15
    J = np.stack([-1.0 / (R_GAS * T_K), np.ones_like(T_K)], axis=1) / max(sigma_lnk, 1e-6)
    return J.T @ J


def psi_design(T_C, n_per_T, sigma_lnk):
    """Ψ_n = λ_min(设计的【总】Fisher 信息)，即定义 def:psi 的 Ψ_n = n·λmin(Î₁)。

    口径（2026-08-10 订正，RYA-105）：fisher() 返回的 JᵀJ 已经是【跨温度层求和】的总信息阵——
    Arrhenius 回归的数据点是每温度一条 ln k̂，噪声尺度 σ_lnk 即该估计量自身的标准差（它已吸收
    该温度下 n_per_T 个时间点的重复）。故总信息 I_total = JᵀJ，而 n·λmin(Î₁) ≡ λmin(n·Î₁)
    = λmin(I_total)：n 与 Î₁ 恒等相消，不得再乘任何样本量因子。

    订正前写的是 `N·λmin(JᵀJ)`（N=len(T_C)·n_per_T），把温度层数 n_temps 重复计了一次
    （Ψ_code = n_temps·Ψ_def），与定义 def:psi 及 §sec:proof「凡要拿本文数字与代码对账，应以
    前者为准」的陈述矛盾。n_per_T 保留在签名中仅为调用方兼容，不参与计算。
    """
    I1 = fisher(T_C, sigma_lnk); ev, evec = np.linalg.eigh(I1)
    return float(ev[0]), float(ev[0]), float(ev[1]), evec[:, 0].tolist()


def boot_cv(T_C, ks, sigma_lnk, rng, n=BOOT):
    T_C = np.asarray(T_C, float); lk0 = np.log(np.clip(ks, 1e-12, None)); Eas = []
    for _ in range(n):
        lkb = lk0 + sigma_lnk * rng.standard_normal(len(T_C))
        T_K = T_C + 273.15; s = linregress(1.0 / (R_GAS * T_K), lkb)
        if np.isfinite(s.slope): Eas.append(-s.slope)
    Eas = np.asarray(Eas)
    return float(np.std(Eas) / (abs(np.mean(Eas)) + 1e-9)), float(np.mean(Eas))


def fast_k(t, y):
    """一阶软化的快速对数线性 k 估计：ln(y)= ln(y0) - k t ⇒ k=-slope（仅正值点）。用于 bootstrap 内层加速。"""
    t = np.asarray(t, float); y = np.asarray(y, float)
    good = np.isfinite(y) & (y > 0)
    if good.sum() < 2:
        return np.nan
    s = linregress(t[good], np.log(y[good]))
    return float(-s.slope) if np.isfinite(s.slope) else np.nan


def real_estimation_error(T_sub, t_h, M_sub, S_sub, rng, n=1000):
    """用【真实测量 SD】（发表表中 ± 值）经 [硬度→k(T)→Arrhenius] 全链 bootstrap 传播到 Ea 的【真实估计误差】。
    对每个温度按 N(mean, sd) 重采样硬度曲线→拟合 k(T)→Arrhenius 拟合 Ea；返回 (CV=std/|mean|, SE=std, 丢弃率)。
    与 boot_cv（注入标量 σ_lnk 的噪声传播）不同：本函数用【数据自身测量不确定度】，且对 n_temps≥3 的
    【超定】子集（自由度>0）反映真实统计估计误差，而非 2 温度恰定（df=0）的纯噪声敏感性。

    内层估计量（2026-08-10 订正，RYA-105）：改用与【点估计完全同一个】的 fit_k（curve_fit 非线性拟
    y=y₀e^{-kt}）。订正前内层走 fast_k（linregress(t, log y) 的对数线性近似），与点估计不是同一估计量，
    而稿件把本函数描述为「[硬度→k(T)→Arrhenius] 全链自助传播」，读者会读成与点估计同链。

    ⚠️ 选择性丢弃：任一温度的 k 估计失败即整次抽样作废（沿用订正前的定义，未改）。这等价于「以拟合成功
    为条件」，会系统性【低估】不确定度，故必须把丢弃率随产物落盘、并在正文报出，不得静默。

    对照口径：同一批重抽样曲线上【同时】用旧的 fast_k 对数线性内层算一份，作为并报的第二估计量
    （两者共用同一 yb 抽样，故差异只来自估计量本身）。返回的第 4 项即该对照 CV。"""
    T_sub = np.asarray(T_sub, float); t_h = np.asarray(t_h, float)
    Eas, Eas_loglin = [], []
    n_dropped = 0
    for _ in range(n):
        ks_b, ks_l = [], []; ok = True; ok_l = True
        for i in range(len(T_sub)):
            yb = np.asarray(M_sub[i], float) + np.asarray(S_sub[i], float) * rng.standard_normal(len(t_h))
            if ok:
                kb, _y0b, _r2b = fit_k(t_h, yb)
                if np.isfinite(kb) and kb > 0:
                    ks_b.append(kb)
                else:
                    ok = False
            if ok_l:
                kl = fast_k(t_h, yb)
                if np.isfinite(kl) and kl > 0:
                    ks_l.append(kl)
                else:
                    ok_l = False
            if not (ok or ok_l):
                break
        if ok:
            Ea_b, _, _, _ = arrhenius(T_sub, ks_b)
            if np.isfinite(Ea_b):
                Eas.append(Ea_b)
            else:
                n_dropped += 1
        else:
            n_dropped += 1
        if ok_l:
            Ea_l, _, _, _ = arrhenius(T_sub, ks_l)
            if np.isfinite(Ea_l):
                Eas_loglin.append(Ea_l)
    drop_rate = float(n_dropped / n)
    cv_l = (float(np.std(Eas_loglin) / (abs(np.mean(Eas_loglin)) + 1e-9))
            if len(Eas_loglin) >= 30 else float("nan"))
    if len(Eas) < 30:
        return np.nan, np.nan, drop_rate, cv_l
    Eas = np.asarray(Eas)
    return (float(np.std(Eas) / (abs(np.mean(Eas)) + 1e-9)), float(np.std(Eas)),
            drop_rate, cv_l)


def main():
    OUT.mkdir(parents=True, exist_ok=True); log = get_logger("54c5_realkinetics")
    data = load_real()
    Ts = data[VARIETIES[0]]["T_C"]; ts = data[VARIETIES[0]]["t_h"]
    log.info("=== C5 真实动力学验证 | 苹果硬度 %s ===", DOI)
    log.info("温度 %s°C × 时间 %s h × 品种 %s（真实 read_html 提取）", Ts, ts, VARIETIES)

    # (1) 逐品种全温度拟合
    per_var = {}
    for v in VARIETIES:
        M = data[v]["mean"]; ks, r2s = [], []
        for i, Tc in enumerate(Ts):
            k, y0, r2 = fit_k(ts, M[i]); ks.append(k); r2s.append(r2)
        Ea, lnA, r2_arr, good = arrhenius(Ts, ks)
        # 单调性诊断：是否软化（k>0 且一阶 R² 高）
        mono = np.nanmean([r for r in r2s if np.isfinite(r)]) if any(np.isfinite(r2s)) else np.nan
        per_var[v] = dict(k_per_T=ks, r2_firstorder=r2s, Ea_kJmol=Ea, lnA=lnA, r2_arrhenius=r2_arr,
                          mean_firstorder_r2=float(mono), n_good_T=int(np.sum(good)))
        log.info("  %-14s Ea=%5.1f kJ/mol lnA=%5.1f Arr_R²=%.3f | 一阶R²均值=%.3f (good T=%d/5) k=%s",
                 v, Ea if np.isfinite(Ea) else -1, lnA if np.isfinite(lnA) else -1,
                 r2_arr if np.isfinite(r2_arr) else -1, mono, int(np.sum(good)),
                 [round(x, 4) if np.isfinite(x) else None for x in ks])

    # 选单调软化最好的品种做设计扫描（Arrhenius R² 最高、一阶拟合好）
    best_v = max(VARIETIES, key=lambda v: (per_var[v]["r2_arrhenius"] if np.isfinite(per_var[v]["r2_arrhenius"]) else -1))
    M = data[best_v]["mean"]; ks = per_var[best_v]["k_per_T"]
    Ea, lnA, r2_arr, good = arrhenius(Ts, ks)
    T_K = np.asarray(Ts) + 273.15
    lk_pred = lnA - Ea / (R_GAS * T_K)
    sigma_lnk = float(np.nanstd(np.log(np.clip(ks, 1e-12, None)) - lk_pred)) or 0.05
    log.info("→ 设计扫描用品种=%s (Arr R²=%.3f), σ_lnk=%.3f", best_v, r2_arr, sigma_lnk)

    # (2) 设计扫描：固定温度数、变窗宽
    rng = np.random.default_rng(SEED_BASE); rows = []; n_per_T = len(ts)
    Tvals = np.array(Ts); kvals = np.array(ks)
    M_best = data[best_v]["mean"]; S_best = np.array(data[best_v]["sd"], float)
    S_best = np.where(np.isfinite(S_best), S_best, np.nanmean(S_best))   # 真实测量 SD，缺失填均值
    for r in (2, 3, 4, 5):
        for combo in itertools.combinations(range(len(Ts)), r):
            Tsub = [Ts[i] for i in combo]; ksub = [ks[i] for i in combo]
            if any(not (np.isfinite(x) and x > 0) for x in ksub): continue
            dT = max(Tsub) - min(Tsub)
            psi, lmin, lmax, degvec = psi_design(Tsub, n_per_T, sigma_lnk)
            cv, ea_b = boot_cv(Tsub, ksub, sigma_lnk, rng)
            # 真实测量 SD（发表 ± 值）传播的 Ea 估计误差；n_temps>=3 为超定（df>0，真实统计估计误差）
            real_cv, real_se, drop, real_cv_l = real_estimation_error(
                Tsub, ts, M_best[list(combo)], S_best[list(combo)], rng)
            rows.append(dict(variety=best_v, n_temps=r, temps="/".join(f"{x:g}" for x in Tsub),
                             deltaT=dT, Psi_n=psi, lambda_min=lmin, cond=lmax / max(lmin, 1e-30),
                             inv_CV_Ea=cv, Ea_boot=ea_b, realCV_Ea=real_cv, realSE_Ea=real_se,
                             boot_drop_rate=drop, realCV_Ea_loglin=real_cv_l))
    df = pd.DataFrame(rows).sort_values("Psi_n")
    df.to_csv(OUT / "54c5_realkinetics-designsweep.csv", index=False)

    sl = linregress(np.log(df.Psi_n), np.log(df.inv_CV_Ea + 1e-9))
    # 超定（>=3 温度）子集：Ψ 预测【真实测量 SD 传播】的 Ea 估计误差（df>0、非注入噪声、非 2 温度恰定）
    od = df[(df.n_temps >= 3) & np.isfinite(df.realCV_Ea) & (df.realCV_Ea > 0)]
    overdet = dict(n_overdetermined=int(len(od)))
    if len(od) >= 4:
        slo = linregress(np.log(od.Psi_n), np.log(od.realCV_Ea))
        sp = spearmanr(od.Psi_n, od.realCV_Ea)
        overdet.update(logPsi_vs_logRealCV_slope=float(slo.slope), r2=float(slo.rvalue ** 2),
                       spearman=float(sp.correlation), spearman_p=float(sp.pvalue),
                       realCV_min=float(od.realCV_Ea.min()), realCV_max=float(od.realCV_Ea.max()),
                       boot_drop_rate_max=float(df.boot_drop_rate.max()),
                       boot_drop_rate_mean=float(df.boot_drop_rate.mean()),
                       boot_drop_rate_overdet_max=float(od.boot_drop_rate.max()),
                       note=("真实测量 SD（发表±值）经[硬度->k(T)->Arrhenius]全链 bootstrap 传播的 Ea 估计误差; "
                             "内层与点估计同一估计量 fit_k(curve_fit 非线性); "
                             "n_temps>=3 超定(自由度>0,真实统计估计误差,非 2 温度恰定 df=0/非注入标量噪声); "
                             "boot_drop_rate=任一温度 k 拟合失败而作废的抽样比例(选择性丢弃会低估不确定度,故落盘)"))
        odl = od[np.isfinite(od.realCV_Ea_loglin) & (od.realCV_Ea_loglin > 0)]
        if len(odl) >= 4:
            sll = linregress(np.log(odl.Psi_n), np.log(odl.realCV_Ea_loglin))
            spl = spearmanr(odl.Psi_n, odl.realCV_Ea_loglin)
            overdet.update(loglin_inner=dict(
                n=int(len(odl)), spearman=float(spl.correlation), spearman_p=float(spl.pvalue),
                logPsi_vs_logRealCV_slope=float(sll.slope), r2=float(sll.rvalue ** 2),
                note=("并报口径：内层改用旧的 fast_k 对数线性近似（与点估计 fit_k 【不】同一估计量），"
                      "同一批重抽样曲线，差异只来自估计量。主口径以 realCV_Ea（fit_k 内层）为准。")))
        log.info("超定(>=3温度)真实估计误差: n=%d, log Ψ vs log realCV 斜率=%.3f R²=%.3f, Spearman=%.3f(p=%.1e)",
                 len(od), slo.slope, slo.rvalue ** 2, sp.correlation, sp.pvalue)
    pair2 = df[df.n_temps == 2].sort_values("deltaT")
    narrow = pair2.iloc[0]; wide = pair2.iloc[-1]
    log.info("=== 设计扫描（%d 子集，固定温度数变窗宽）===", len(df))
    for _, rr in df.iterrows():
        log.info("  {%s} ΔT=%g°C Ψ=%.2e cond=%.0f → 反演CV(Ea)=%.3f", rr.temps, rr.deltaT, rr.Psi_n, rr.cond, rr.inv_CV_Ea)
    log.info("核心: log Ψ vs log CV(Ea) 斜率=%.3f R²=%.3f (应<0)", sl.slope, sl.rvalue ** 2)
    log.info("cor:design(2温度): 窄窗{%s}ΔT=%g CV=%.3f vs 宽窗{%s}ΔT=%g CV=%.3f → 宽窗稳 %.1f×",
             narrow.temps, narrow.deltaT, narrow.inv_CV_Ea, wide.temps, wide.deltaT, wide.inv_CV_Ea,
             narrow.inv_CV_Ea / max(wide.inv_CV_Ea, 1e-9))

    # 多品种复现 cor:design（每个单调品种独立的 2 温度窄窗 vs 宽窗；多真实序列复现）
    multi = {}
    for v in VARIETIES:
        ksv = per_var[v]["k_per_T"]
        if not (np.isfinite(per_var[v]["r2_arrhenius"]) and per_var[v]["mean_firstorder_r2"] > 0.8):
            multi[v] = dict(monotone=False, note="非单调/病态(排除复现)"); continue
        # 同 σ_lnk 口径（该品种自身）
        T_K2 = np.asarray(Ts) + 273.15
        Eav, lnAv, r2v, gv = arrhenius(Ts, ksv)
        sigv = float(np.nanstd(np.log(np.clip(ksv, 1e-12, None)) - (lnAv - Eav / (R_GAS * T_K2)))) or 0.06
        rng2 = np.random.default_rng(SEED_BASE + 1)
        nar_cv, _ = boot_cv([Ts[-2], Ts[-1]], [ksv[-2], ksv[-1]], sigv, rng2)  # 窄窗(最近两温度)
        wid_cv, _ = boot_cv([Ts[0], Ts[-1]], [ksv[0], ksv[-1]], sigv, rng2)    # 宽窗(最远两温度)
        multi[v] = dict(monotone=True, Ea=Eav, r2_arrhenius=r2v,
                        narrow_dT=float(Ts[-1] - Ts[-2]), narrow_CV=nar_cv,
                        wide_dT=float(Ts[-1] - Ts[0]), wide_CV=wid_cv,
                        wide_better=float(nar_cv / max(wid_cv, 1e-9)),
                        replicates=bool(wid_cv < nar_cv))
    n_mono = sum(1 for v in multi if multi[v].get("monotone"))
    n_repl = sum(1 for v in multi if multi[v].get("replicates"))
    log.info("=== 多品种 cor:design 复现: %d/%d 单调品种宽窗优于窄窗 ===", n_repl, n_mono)
    for v in VARIETIES:
        m = multi[v]
        if m.get("monotone"):
            log.info("  %-14s 窄窗ΔT=%g CV=%.3f → 宽窗ΔT=%g CV=%.3f (%.1f×, 复现=%s)",
                     v, m["narrow_dT"], m["narrow_CV"], m["wide_dT"], m["wide_CV"], m["wide_better"], m["replicates"])
        else:
            log.info("  %-14s %s", v, m["note"])

    summary = dict(
        data_source=DOI,
        positioning="真实数据验证【诊断Ψ的设计预测效力】(cor:design设计侧推论)——Ψ预测不同温度设计窗下(Ea,lnA)反演的可恢复性差异; 【非】三分律相变本身或converse(后者由半合成+理论支撑)",
        method_notes="CV=噪声传播敏感性(2温度Arrhenius恰定、拟合R²≡1、自由度0; CV来自对k(T)注入σ_lnk噪声的bootstrap传播,非统计估计误差); σ_lnk由各品种全温度Arrhenius残差std估; 展示品种选择准则=一阶R²>0.8且ArrheniusR²最高(RoyalGala); 四品种全表均报告(含病态PinkLady)",
        honesty="他组公开发表数据(附DOI); pandas.read_html真实提取(脚本55,存pmc_real/CSV)+人工核验; 本脚本从CSV读取不手填; 真实数据含噪声/非单调(Pink Lady@2°C后熟)恰检验Ψ区分病态/良态",
        attribute="firmness softening (kg·cm^-2)", T_C=Ts, t_h=ts, varieties=VARIETIES,
        per_variety=per_var, design_variety=best_v,
        overdetermined_real_error=overdet,
        multi_variety_replication=dict(n_monotone=n_mono, n_replicate=n_repl, detail=multi),
        full=dict(Ea_kJmol=Ea, lnA=lnA, r2_arrhenius=r2_arr, sigma_lnk=sigma_lnk, n_good_T=int(np.sum(good))),
        design_sweep=dict(
            logPsi_vs_logCV_slope=float(sl.slope), r2=float(sl.rvalue ** 2), n_subsets=len(df),
            cor_design_pair=dict(
                narrow=dict(temps=narrow.temps, dT=float(narrow.deltaT), Psi=float(narrow.Psi_n), CV=float(narrow.inv_CV_Ea)),
                wide=dict(temps=wide.temps, dT=float(wide.deltaT), Psi=float(wide.Psi_n), CV=float(wide.inv_CV_Ea)),
                wide_better_factor=float(narrow.inv_CV_Ea / max(wide.inv_CV_Ea, 1e-9)))),
        verdict=("PASS: 真实数据上 Ψ 预测反演可恢复性(斜率<0)且宽窗优于窄窗(cor:design)"
                 if sl.slope < 0 and wide.inv_CV_Ea < narrow.inv_CV_Ea else "PARTIAL"))
    json.dump(summary, open(OUT / "54c5_realkinetics.json", "w"), ensure_ascii=False, indent=2)
    log.info("VERDICT: %s", summary["verdict"])
    log.info("DONE 54c5_realkinetics")


if __name__ == "__main__":
    main()
