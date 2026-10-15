"""
43trichotomy_formal.py: C5 可恢复性三分律的 FORMAL 多种子复跑（替换 39 pilot 数字）

vs 39 pilot：用项目约定 §5.4 FORMAL 固定 5 种子集合 + cluster bootstrap 95% CI（≥1000 重采样，种子为聚类单位）
+ 更细 n 网格 + 对关键 verdict（三分律分离、collapse log-log 斜率、Krug 匹配）给 mean/std/CI。
Krug 匹配自 2026-08-23 起以【加权】E_w[r]（权重 g^2/sigma^2）为 headline，与 lem:fisher 同口径；
旧的不加权口径以 slope_krug_unw / slope_relerr_unw / krug_match_unweighted 并存保留，供审计复算。
半合成 Arrhenius forward，scipy/numpy（CPU 主导，--device 仅占位）。

运行方式:
    cd <项目根目录>
    python 02code/43trichotomy_formal.py --device auto

输出文件:
    04outputs/43trichotomy_formal.json        — formal verdict + bootstrap CI
    04outputs/43trichotomy_formal-percell.csv  — 每 (regime,n,seed) 明细
    04outputs/43trichotomy_formal-agg.csv      — 每 (regime,n) mean/std/CI
    05logs/43trichotomy_formal_<ts>.log
"""
from __future__ import annotations
import argparse, json, logging, time
from pathlib import Path
import numpy as np
from scipy import optimize
from scipy.stats import spearmanr

R_GAS = 8.314
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"; LOGS = ROOT / "05logs"
RECOMMENDED_SEEDS = [20060515, 20041210, 19810915, 2023, 2024]  # 项目约定 §5.4 Formal 固定集合
T_REF_K = 283.15; PHI_TRUE = 34.0; LOGA_TRUE = 31.0
T_LOW_C = 5.0; DELTA_L = 10.0; SIGMA = 0.02; BETA_STAR = 0.5; T_REF_C = 10.0
N_LIST = [200, 500, 1000, 2000, 5000, 10000, 20000]; N0 = 200; DT0 = 16.0
REGIMES = {"fixed_recover": 0.0, "sub_recover": 0.25, "critical": 0.5, "subcritical": 1.0}
BOOT = 2000


def get_logger(stem):
    LOGS.mkdir(parents=True, exist_ok=True); ts = time.strftime("%y-%m-%d_%H%M%S")
    lg = logging.getLogger(stem); lg.setLevel(logging.INFO); lg.handlers.clear()
    fh = logging.FileHandler(LOGS / f"{stem}_{ts}.log"); sh = logging.StreamHandler()
    f = logging.Formatter("%(asctime)s %(levelname)s %(message)s"); fh.setFormatter(f); sh.setFormatter(f)
    lg.addHandler(fh); lg.addHandler(sh); return lg


def forward(t, T_C, phi, logA, beta=1.0):
    T_K = T_C + 273.15; lk = np.clip(logA - phi * (T_REF_K / T_K), -30, 5)
    return beta * (1 - np.exp(-np.clip(np.exp(lk) * t, 0, 30)))


def fit(t, T_C, y, rng, n_starts=10):
    def resid(th): return forward(t, T_C, th[0], th[1]) - y
    best = None
    for _ in range(n_starts):
        x0 = [rng.uniform(20, 55), rng.uniform(22, 42)]
        try:
            r = optimize.least_squares(resid, x0=x0, bounds=([8, 15], [80, 50]),
                                       method="trf", xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=8000)
            c = float(np.sum(r.fun ** 2))
            if best is None or c < best[2]: best = (float(r.x[0]), float(r.x[1]), c)
        except Exception: continue
    return (best[0], best[1]) if best else (PHI_TRUE, LOGA_TRUE)


def fim(t, T_C, phi, logA, sigma):
    T_K = T_C + 273.15; r = T_REF_K / T_K
    lk = np.clip(logA - phi * r, -30, 5); k = np.exp(lk); kt = np.clip(k * t, 0, 30)
    g = kt * np.exp(-kt); J = np.column_stack([-r * g, g])
    f1 = (J.T @ J) / (sigma ** 2) / max(len(t), 1)
    ev, evec = np.linalg.eigh(f1); lmin = float(max(ev[0], 1e-300)); vdeg = evec[:, 0]
    slope = float(vdeg[1] / vdeg[0]) if abs(vdeg[0]) > 1e-15 else np.nan
    return lmin, vdeg, slope


def krug(T_C):
    """旧口径：不加权的 E[r]（= T_ref / 调和均值温度）。

    保留原因：稿面 0.29% 一路的历史比对量由它产生，静默覆盖会让换数不可审。
    它不是 lem:fisher 断言的对象——见 krug_weighted。
    """
    T_K = T_C + 273.15; return float(T_REF_K / (len(T_K) / np.sum(1.0 / T_K)))


def krug_weighted(t, T_C, phi, logA, sigma):
    """lem:fisher 断言的对象：加权 E_w[r] = Σ g_i² r_i/σ_i² / Σ g_i²/σ_i²。

    fim() 装配的 Fisher 行是 g_i·[-r_i, 1]（g = kt·e^{-kt} 逐观测变化），其近零方向斜率
    是 g²/σ² 加权的 E_w[r]；旧 krug() 看不到 g，量的是另一条线。
    本设定 σ 为常数，故 σ² 在分子分母对消，退化为 Σg²r/Σg²；此处仍按公式显式写出。
    下面三行与 fim() 的对应三行逐字一致，改动其一必须同步另一处。
    (2026-08-23 RYA-588 修 9974b904a90b；本函数不消费随机数，故不扰动 RNG 流。)
    """
    T_K = T_C + 273.15; r = T_REF_K / T_K
    lk = np.clip(logA - phi * r, -30, 5); k = np.exp(lk); kt = np.clip(k * t, 0, 30)
    g = kt * np.exp(-kt); w = (g ** 2) / (sigma ** 2)
    return float(np.sum(w * r) / np.sum(w))


def boot_ci(vals, rng, stat=np.mean, n=BOOT):
    vals = np.asarray(vals); bs = [stat(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="auto"); ap.add_argument("--n_starts", type=int, default=10)
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); log = get_logger("43trichotomy_formal")
    log.info("FORMAL stage | seeds=%s (project conventions §5.4) | bootstrap=%d", RECOMMENDED_SEEDS, BOOT)
    rows = []
    for rname, p in REGIMES.items():
        for n in N_LIST:
            dT = float(np.clip(DT0 * (n / N0) ** (-p), 0.05, 60))
            for seed in RECOMMENDED_SEEDS:
                rng = np.random.default_rng(seed * 100003 + n)
                T_C = rng.uniform(T_LOW_C, T_LOW_C + dT, n); t = rng.uniform(0.5, DELTA_L, n)
                y = forward(t, T_C, PHI_TRUE, LOGA_TRUE) + rng.normal(0, SIGMA, n)
                lmin, vdeg, se = fim(t, T_C, PHI_TRUE, LOGA_TRUE, SIGMA); psi = float(n * lmin)
                ph, la = fit(t, T_C, y, rng, args.n_starts)
                ev = np.array([ph - PHI_TRUE, la - LOGA_TRUE]); ed = float(abs(ev @ vdeg))
                sk_w = krug_weighted(t, T_C, PHI_TRUE, LOGA_TRUE, SIGMA)  # headline：与 lem:fisher 同口径
                sk_unw = krug(T_C)                                        # 旧口径，保留供审计复算
                rows.append(dict(regime=rname, p=p, n=n, seed=seed, psi_n=psi, err_deg=ed,
                                 slope_emp=se, slope_krug_w=sk_w, slope_krug_unw=sk_unw,
                                 slope_relerr=float(abs(se - sk_w) / abs(sk_w)),
                                 slope_relerr_unw=float(abs(se - sk_unw) / abs(sk_unw))))
        log.info("regime=%-14s done", rname)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(OUT / "43trichotomy_formal-percell.csv", index=False)
    rng = np.random.default_rng(20240529)
    # 聚合 + cluster bootstrap CI（以 seed 为聚类单位）
    agg_rows = []
    for (rname, n), g in df.groupby(["regime", "n"]):
        per_seed = g.groupby("seed").err_deg.mean().values
        lo, hi = boot_ci(per_seed, rng)
        agg_rows.append(dict(regime=rname, n=n, psi_n=float(g.psi_n.mean()),
                             err_deg_mean=float(g.err_deg.mean()), err_deg_std=float(g.err_deg.std()),
                             err_ci_lo=lo, err_ci_hi=hi))
    agg = pd.DataFrame(agg_rows); agg.to_csv(OUT / "43trichotomy_formal-agg.csv", index=False)
    # verdicts + CI
    dfv = df[df.psi_n > 0].copy(); dfv["lp"] = np.log10(dfv.psi_n); dfv["le"] = np.log10(dfv.err_deg.clip(1e-6))
    rec = dfv[dfv.regime.isin(["sub_recover", "fixed_recover"]) & (dfv.psi_n > 1)]
    slope_fit = float(np.polyfit(rec["lp"], rec["le"], 1)[0])
    # collapse slope bootstrap CI (cluster by seed)
    sl_bs = []
    # 以 seed 为聚类单位的自助：抽到重复的种子必须【按重数复制其整块行】。
    # 原实现用 rec.seed.isin(ss) 做集合成员判断，重复种子被折叠成一次，
    # 于是抽的是「种子的随机子集」而非 cluster bootstrap，聚类间方差没被正确重抽。
    # (2026-07-26 由外审 L2 取证发现 HP-AGG-DRIFT，复算坐实后修正。)
    _blocks = {s: rec[rec.seed == s] for s in RECOMMENDED_SEEDS}
    for _ in range(BOOT):
        ss = rng.choice(RECOMMENDED_SEEDS, len(RECOMMENDED_SEEDS), replace=True)
        sub = pd.concat([_blocks[s] for s in ss], ignore_index=True)
        if len(sub) > 5: sl_bs.append(np.polyfit(sub["lp"], sub["le"], 1)[0])
    slope_ci = (float(np.percentile(sl_bs, 2.5)), float(np.percentile(sl_bs, 97.5)))
    # 点估计与区间必须是【同一个统计量】。原实现点估计取全行中位数，区间却把 5 个种子中位数
    # 喂给 boot_ci（其默认 stat=np.mean），于是报的是「种子中位数之均值」的自助区间——
    # 两者不同源，结果 CI 竟不含点估计（0.0028577 vs [0.0028697, 0.0029546]），
    # 四舍五入到 0.29% / [0.29,0.30]% 恰好把这个错配盖住。
    # 改为对【同一统计量】做以 seed 为聚类单位的自助：每次重抽整块种子行，再取全行中位数。
    # (2026-07-26 外审取证的确认复跑 HP-PHANTOM-RESULT 指出，复算坐实后修正。)
    # 加权列为 headline，不加权列同批次一并自助：同一次重抽同时喂两列，故两条 CI 配对，
    # 且 rng 的抽取次数与次序与改动前逐次相同（每轮仍只调用一次 rng.choice）。
    krug_med = float(df.slope_relerr.median())
    krug_med_unw = float(df.slope_relerr_unw.median())
    _kblocks = {s: df[df.seed == s] for s in RECOMMENDED_SEEDS}
    _kbs, _kbs_unw = [], []
    for _ in range(BOOT):
        _sub = pd.concat([_kblocks[s] for s in rng.choice(RECOMMENDED_SEEDS, len(RECOMMENDED_SEEDS),
                                                          replace=True)], ignore_index=True)
        _kbs.append(float(_sub.slope_relerr.median())); _kbs_unw.append(float(_sub.slope_relerr_unw.median()))
    krug_ci = (float(np.percentile(_kbs, 2.5)), float(np.percentile(_kbs, 97.5)))
    krug_ci_unw = (float(np.percentile(_kbs_unw, 2.5)), float(np.percentile(_kbs_unw, 97.5)))
    rho = float(spearmanr(dfv.psi_n, dfv.err_deg).correlation)
    summary = dict(stage="formal", seeds=RECOMMENDED_SEEDS, bootstrap=BOOT, n_list=N_LIST,
        trichotomy={rn: dict(psi_first=float(agg[(agg.regime == rn)].sort_values("n").psi_n.iloc[0]),
                             psi_last=float(agg[(agg.regime == rn)].sort_values("n").psi_n.iloc[-1]),
                             err_first=float(agg[(agg.regime == rn)].sort_values("n").err_deg_mean.iloc[0]),
                             err_last=float(agg[(agg.regime == rn)].sort_values("n").err_deg_mean.iloc[-1]))
                    for rn in REGIMES},
        collapse_slope=dict(fitted=slope_fit, ci95=slope_ci, predicted=-0.5,
                            verdict="matches -1/2" if abs(slope_fit + 0.5) < 0.2 else "deviates"),
        krug_match=dict(median_relerr=krug_med, ci95=krug_ci, verdict="PASS" if krug_med < 0.1 else "PARTIAL",
                        weighting="g^2/sigma^2", compared_to="E_w[r] = Sum g^2 r / Sum g^2 (lem:fisher 断言的对象)"),
        krug_match_unweighted=dict(median_relerr=krug_med_unw, ci95=krug_ci_unw,
                                   verdict="PASS" if krug_med_unw < 0.1 else "PARTIAL",
                                   weighting="none",
                                   compared_to="E[r] = T_ref / T_hm (不加权调和均值温度)",
                                   note=("旧口径，保留供审计复算 2026-08-23 之前稿面的 0.29% / CI[0.29,0.30]%。"
                                         "它不是 lem:fisher 断言的对象，稿面不再引用此列。")))
    with open(OUT / "43trichotomy_formal.json", "w") as f: json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("=== FORMAL VERDICT ===")
    for rn in REGIMES:
        v = summary["trichotomy"][rn]; log.info("  %-14s Ψ %.2e→%.2e | err %.3f→%.3f", rn, v["psi_first"], v["psi_last"], v["err_first"], v["err_last"])
    log.info("collapse slope %.3f CI[%.3f,%.3f] (pred -0.5) %s", slope_fit, slope_ci[0], slope_ci[1], summary["collapse_slope"]["verdict"])
    log.info("Krug(weighted E_w[r]) median relerr %.6f CI[%.6f,%.6f] %s", krug_med, krug_ci[0], krug_ci[1], summary["krug_match"]["verdict"])
    log.info("Krug(unweighted E[r], legacy) median relerr %.6f CI[%.6f,%.6f] %s", krug_med_unw, krug_ci_unw[0], krug_ci_unw[1], summary["krug_match_unweighted"]["verdict"])
    log.info("DONE 43trichotomy_formal")


if __name__ == "__main__":
    main()
