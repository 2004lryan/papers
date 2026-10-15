"""
126trichotomy_plugin_psi.py: 三分律在 **plug-in** 诊断量 Psi(theta_hat) 下的复现（去 oracle）

动机（取证发现 CONS-F001 / EXP 维度）：
    43trichotomy_formal.py 的半合成三分律实验以 **真值** (PHI_TRUE, LOGA_TRUE) 装配 Fisher，
    即报告的是 oracle 层 Psi(theta_0)；而稿件交付给读者的是 plug-in 层 Psi(theta_hat)，
    其初值来自响应观测。两者的区别稿中虽已定义（§relate-id (iv)），但半合成图未标注用的是哪一层。
    本脚本用**独立小预试验样本**估初值，重算 Psi，检验三分律分离是否在 plug-in 层同样成立。

与 43 的唯一差别：Psi 的 theta 来源。
    43:  Psi_oracle = n * lmin(I_1(theta_0; Phi))          <- 真值
    126: Psi_plugin = n * lmin(I_1(theta_hat_pilot; Phi))  <- 独立预试验样本上的廉价初拟合
设计 Phi（温度点与时间点）、噪声 sigma、样本量 n、种子集合、regime 网格与 43 逐位一致，
故两列可直接配对比较。预试验样本与主样本 **不相交**（独立 rng 流），无数据复用。

运行方式:
    python 02code/126trichotomy_plugin_psi.py

输出文件:
    04outputs/126trichotomy_plugin_psi.json       — verdict + 逐 regime 分离 + 相合性（本轮逐字节未变）
    04outputs/147oracle_plugin_error_pairing.json — oracle 对 plug-in 两列退化方向误差的配对差（见下）
    04outputs/126trichotomy_plugin_psi-percell.csv — 每 (regime,n,seed) 明细（两列 Psi 并排）
    05logs/126trichotomy_plugin_psi_<ts>.log

147oracle_plugin_error_pairing.json（RYA-583 新增；为何独立成件见该文件的 why_separate）:
    稿面报出「两列退化方向误差 140 个 cell 全不同，最大绝对差 / 最大相对差 / 两列 Spearman」，
    此前无任何产物字段存这三个数（外审核对判为 real_gap）。本块按 (regime,n,seed) 把
    43trichotomy_formal-percell.csv 的 err_deg 与本脚本的 err_deg 逐格配对后实算并落盘。
    两列之所以不逐位相同：两脚本主拟合的多起点初值区间不同（稿面已声明），故这是该差异的量化。
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.stats import spearmanr

R_GAS = 8.314
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"; LOGS = ROOT / "05logs"
RECOMMENDED_SEEDS = [20060515, 20041210, 19810915, 2023, 2024]  # 与 43 同一预注册集合
T_REF_K = 283.15; PHI_TRUE = 34.0; LOGA_TRUE = 31.0
T_LOW_C = 5.0; DELTA_L = 10.0; SIGMA = 0.02
N_LIST = [200, 500, 1000, 2000, 5000, 10000, 20000]; N0 = 200; DT0 = 16.0
REGIMES = {"fixed_recover": 0.0, "sub_recover": 0.25, "critical": 0.5, "subcritical": 1.0}
REGIME_IDX = {name: i for i, name in enumerate(REGIMES)}   # 确定性偏移，勿用 hash()（进程间随机）

N_PILOT = 40        # 预试验样本量：固定小常数，不随 n 增长（"少量曲线的廉价初拟合"）
PILOT_STARTS = 3    # 预试验拟合的多起点数：刻意低于主拟合的 10，体现"廉价"
PILOT_RETRY_STARTS = 12   # 收敛守卫触发后的重试起点数
PILOT_RMSE_TOL = 3.0      # 收敛守卫：拟合残差 RMSE > 该倍数 * sigma 即判该次预拟合未收敛
#   守卫只用可观测量（预试验残差 + 算法本就要求输入的噪声尺度 sigma），不使用真值，
#   故与 alg:psi 的输入集合一致、可在真实部署中照做。


def get_logger(stem):
    LOGS.mkdir(parents=True, exist_ok=True); ts = time.strftime("%y-%m-%d_%H%M%S")
    lg = logging.getLogger(stem); lg.setLevel(logging.INFO); lg.handlers.clear()
    fh = logging.FileHandler(LOGS / f"{stem}_{ts}.log"); sh = logging.StreamHandler()
    f = logging.Formatter("%(asctime)s %(levelname)s %(message)s"); fh.setFormatter(f); sh.setFormatter(f)
    lg.addHandler(fh); lg.addHandler(sh); return lg


def forward(t, T_C, phi, logA, beta=1.0):
    """与 43trichotomy_formal.forward 逐字一致。"""
    T_K = T_C + 273.15; lk = np.clip(logA - phi * (T_REF_K / T_K), -30, 5)
    return beta * (1 - np.exp(-np.clip(np.exp(lk) * t, 0, 30)))


def fit(t, T_C, y, rng, n_starts=10):
    """与 43trichotomy_formal.fit 逐字一致（多起点 trf 非线性最小二乘）。"""
    def resid(th): return forward(t, T_C, th[0], th[1]) - y
    best = None
    for _ in range(n_starts):
        x0 = np.array([rng.uniform(15, 60), rng.uniform(15, 45)])
        try:
            r = optimize.least_squares(resid, x0=x0, bounds=([8, 15], [80, 50]),
                                       method="trf", xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=8000)
            c = float(np.sum(r.fun ** 2))
            if best is None or c < best[2]: best = (float(r.x[0]), float(r.x[1]), c)
        except Exception: continue
    return (best[0], best[1]) if best else (PHI_TRUE, LOGA_TRUE)


def fim(t, T_C, phi, logA, sigma):
    """与 43trichotomy_formal.fim 逐字一致。theta 经 g=kt*exp(-kt) 权重真进入。"""
    T_K = T_C + 273.15; r = T_REF_K / T_K
    lk = np.clip(logA - phi * r, -30, 5); k = np.exp(lk); kt = np.clip(k * t, 0, 30)
    g = kt * np.exp(-kt); J = np.column_stack([-r * g, g])
    f1 = (J.T @ J) / (sigma ** 2) / max(len(t), 1)
    ev, evec = np.linalg.eigh(f1); lmin = float(max(ev[0], 1e-300)); vdeg = evec[:, 0]
    slope = float(vdeg[1] / vdeg[0]) if abs(vdeg[0]) > 1e-15 else np.nan
    return lmin, vdeg, slope


def pilot_rmse(tp, Tp, yp, phi, logA):
    """预试验拟合的残差 RMSE —— 只用预试验数据本身，不涉及真值。"""
    return float(np.sqrt(np.mean((forward(tp, Tp, phi, logA) - yp) ** 2)))


def pilot_fit_guarded(tp, Tp, yp, prng, sigma):
    """廉价预拟合 + 收敛守卫。

    返回 (phi_hat, logA_hat, rmse, n_starts_used, converged)。
    守卫判据：残差 RMSE <= PILOT_RMSE_TOL * sigma。不达标则加大起点数重试一次。
    退化分支（k 极大导致响应在所有点上已饱和）恰好表现为残差远大于噪声尺度，
    故该判据在实际部署中同样可用——实验者看到的就是"这条拟合根本没拟上"。
    """
    phi, logA = fit(tp, Tp, yp, prng, PILOT_STARTS)
    rmse = pilot_rmse(tp, Tp, yp, phi, logA)
    first_ok = bool(rmse <= PILOT_RMSE_TOL * sigma)   # 首次(廉价 3 起点)是否已收敛——即"不加守卫"的口径
    if first_ok:
        return phi, logA, rmse, PILOT_STARTS, True, first_ok
    phi2, logA2 = fit(tp, Tp, yp, prng, PILOT_RETRY_STARTS)
    rmse2 = pilot_rmse(tp, Tp, yp, phi2, logA2)
    if rmse2 <= rmse:
        phi, logA, rmse = phi2, logA2, rmse2
    return phi, logA, rmse, PILOT_STARTS + PILOT_RETRY_STARTS, bool(rmse <= PILOT_RMSE_TOL * sigma), first_ok


def loglog_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3: return float("nan"), float("nan")
    lx, ly = np.log(x[m]), np.log(y[m])
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx; ss = float(1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2))
    return float(b), ss


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); log = get_logger("126trichotomy_plugin_psi")
    log.info("plug-in Psi | seeds=%s | N_PILOT=%d starts=%d", RECOMMENDED_SEEDS, N_PILOT, PILOT_STARTS)
    rows = []
    for rname, p in REGIMES.items():
        for n in N_LIST:
            dT = float(np.clip(DT0 * (n / N0) ** (-p), 0.05, 60))
            for seed in RECOMMENDED_SEEDS:
                # 主样本 rng 流：与 43 逐位一致，故 Psi_oracle / err_deg 可与 43 直接对照
                rng = np.random.default_rng(seed * 100003 + n)
                T_C = rng.uniform(T_LOW_C, T_LOW_C + dT, n); t = rng.uniform(0.5, DELTA_L, n)
                y = forward(t, T_C, PHI_TRUE, LOGA_TRUE) + rng.normal(0, SIGMA, n)

                lmin_o, vdeg_o, se_o = fim(t, T_C, PHI_TRUE, LOGA_TRUE, SIGMA)
                psi_oracle = float(n * lmin_o)

                # 独立预试验样本：另起 rng 流，与主样本不相交（无数据复用，无需样本分割论证）
                # 种子含 regime 哈希，避免不同 regime 复用同一条预试验流
                prng = np.random.default_rng(seed * 100003 + n + 7919 + 65521 * REGIME_IDX[rname])
                Tp = prng.uniform(T_LOW_C, T_LOW_C + dT, N_PILOT)   # 同一设计窗，故设计 Phi 已知
                tp = prng.uniform(0.5, DELTA_L, N_PILOT)
                yp = forward(tp, Tp, PHI_TRUE, LOGA_TRUE) + prng.normal(0, SIGMA, N_PILOT)
                phi_hat, logA_hat, p_rmse, p_starts, p_conv, p_first = pilot_fit_guarded(tp, Tp, yp, prng, SIGMA)

                # plug-in：theta 用预试验估计，设计 (t, T_C) 仍是候选设计本身（实验前已知）
                lmin_p, vdeg_p, se_p = fim(t, T_C, phi_hat, logA_hat, SIGMA)
                psi_plugin = float(n * lmin_p)

                # 主拟合与退化方向误差：口径与 43 一致（用 oracle 退化方向，保证 err 列可比）
                ph, la = fit(t, T_C, y, rng, 10)
                ev = np.array([ph - PHI_TRUE, la - LOGA_TRUE]); err_deg = float(abs(ev @ vdeg_o))

                rows.append(dict(regime=rname, p=p, n=n, seed=seed,
                                 psi_oracle=psi_oracle, psi_plugin=psi_plugin,
                                 ratio=psi_plugin / psi_oracle if psi_oracle > 0 else float("nan"),
                                 phi_hat=phi_hat, logA_hat=logA_hat,
                                 phi_relerr=abs(phi_hat - PHI_TRUE) / PHI_TRUE,
                                 logA_relerr=abs(logA_hat - LOGA_TRUE) / LOGA_TRUE,
                                 pilot_rmse=p_rmse, pilot_starts=p_starts, pilot_converged=p_conv, pilot_first_ok=p_first,
                                 err_deg=err_deg, slope_oracle=se_o, slope_plugin=se_p))
        log.info("regime=%-14s done", rname)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "126trichotomy_plugin_psi-percell.csv", index=False, encoding="utf-8")

    res = {"script": "126trichotomy_plugin_psi.py", "seeds": RECOMMENDED_SEEDS,
           "n_pilot": N_PILOT, "pilot_starts": PILOT_STARTS, "n_cells": int(len(df)),
           "question": "半合成三分律的分离，在去 oracle 的 plug-in 诊断量 Psi(theta_hat) 下是否同样成立",
           "protocol": ("设计 Phi、噪声、n 网格、regime 与种子集合与 43trichotomy_formal.py 逐位一致；"
                        f"初值由独立的 {N_PILOT} 点预试验样本经 {PILOT_STARTS} 起点最小二乘拟合得到，"
                        "与主样本使用不相交的 rng 流，故无数据复用。")}

    conv = df[df.pilot_converged]      # 通过收敛守卫的格
    fail = df[~df.pilot_converged]

    # (0) 廉价预拟合本身的失败率 —— 这是本实验的第一项结果，不可隐藏
    res["pilot_convergence"] = {
        "guard": f"pilot residual RMSE <= {PILOT_RMSE_TOL} * sigma (sigma={SIGMA})",
        "n_cells": int(len(df)), "n_converged": int(len(conv)), "n_failed": int(len(fail)),
        "fail_rate": float(len(fail) / len(df)),
        "unguarded_first_attempt_fail_rate": float((~df.pilot_first_ok).mean()),
        "unguarded_first_attempt_n_failed": int((~df.pilot_first_ok).sum()),
        "unguarded_note": "首次 3 起点廉价预拟合的失败率，即稿中\"不加守卫时\"所引口径；守卫后的失败率见 fail_rate。",
        "failed_cells": [dict(regime=r.regime, n=int(r.n), seed=int(r.seed),
                              pilot_rmse=float(r.pilot_rmse), ratio=float(r.ratio))
                         for r in fail.itertuples()],
        "note": ("未通过守卫的格，其 Psi(theta_hat) 塌向 0（退化分支：k_hat 极大使响应在所有点已饱和，"
                 "g=kt*exp(-kt)->0）。方向上这是**保守**失败：诊断把该设计判为不可恢复而非误判为可恢复。"),
    }

    # (1) 两层 Psi 的相合性（守卫后）
    def cons(d):
        return {"n": int(len(d)),
                "spearman_psi_oracle_vs_plugin": float(spearmanr(d.psi_oracle, d.psi_plugin).statistic),
                "ratio_median": float(d.ratio.median()),
                "ratio_p05": float(d.ratio.quantile(0.05)), "ratio_p95": float(d.ratio.quantile(0.95)),
                "pilot_phi_relerr_median": float(d.phi_relerr.median()),
                "pilot_logA_relerr_median": float(d.logA_relerr.median())}
    res["consistency_converged"] = cons(conv)
    res["consistency_all_cells"] = cons(df)
    res["consistency_converged"]["ratio_median_by_n"] = {
        int(k): float(v) for k, v in conv.groupby("n").ratio.median().items()}

    # (2) 三分律分离：每个 regime 上 log Psi 对 log n 的斜率，理论值 1-2*rho
    def separation(d):
        sep = {}
        for rname, p in REGIMES.items():
            dd = d[d.regime == rname]
            b_o, r2_o = loglog_slope(dd.n, dd.psi_oracle)
            b_p, r2_p = loglog_slope(dd.n, dd.psi_plugin)
            sep[rname] = {"rho": p, "theory_slope": 1 - 2 * p,
                          "slope_oracle": b_o, "r2_oracle": r2_o,
                          "slope_plugin": b_p, "r2_plugin": r2_p,
                          "abs_gap_plugin_vs_theory": abs(b_p - (1 - 2 * p))}
        return sep
    sep_c = separation(conv); sep_a = separation(df)
    res["separation_logPsi_vs_logn_converged"] = sep_c
    res["separation_logPsi_vs_logn_all_cells"] = sep_a
    for tag, sep in (("converged", sep_c), ("all_cells", sep_a)):
        res[f"separation_verdict_{tag}"] = {
            "max_abs_gap_plugin_vs_theory": float(max(v["abs_gap_plugin_vs_theory"] for v in sep.values())),
            "all_regimes_within_0p10": bool(all(v["abs_gap_plugin_vs_theory"] < 0.10 for v in sep.values())),
            "ordering_preserved": bool(
                sep["fixed_recover"]["slope_plugin"] > sep["sub_recover"]["slope_plugin"]
                > sep["critical"]["slope_plugin"] > sep["subcritical"]["slope_plugin"]),
            "min_r2_plugin": float(min(v["r2_plugin"] for v in sep.values())),
            # 本实验要回答的是"换成 plug-in 后分离是否保持"，故主判据是 plugin vs oracle 的差；
            # 与理论值的差（max_abs_gap_plugin_vs_theory）是既有的有限 n 偏差，oracle 列同样存在，
            # 不由 plug-in 引入，见 slope_oracle 各行。
            "max_abs_gap_plugin_vs_oracle": float(max(
                abs(v["slope_plugin"] - v["slope_oracle"]) for v in sep.values())),
            "max_abs_gap_oracle_vs_theory": float(max(
                abs(v["slope_oracle"] - v["theory_slope"]) for v in sep.values())),
        }

    # (3) 可恢复侧 err ~ Psi^{-1/2}：两层各自的 log-log 斜率
    def rate(d):
        rec = d[d.regime.isin(["fixed_recover", "sub_recover"])]
        b_o, r2_o = loglog_slope(rec.psi_oracle, rec.err_deg)
        b_p, r2_p = loglog_slope(rec.psi_plugin, rec.err_deg)
        return {"theory": -0.5, "n": int(len(rec)),
                "slope_vs_psi_oracle": b_o, "r2_oracle": r2_o,
                "slope_vs_psi_plugin": b_p, "r2_plugin": r2_p}
    res["rate_recoverable_side_converged"] = rate(conv)
    res["rate_recoverable_side_all_cells"] = rate(df)

    # (4) oracle 层(43) 与 plug-in 层(本脚本) 两列退化方向误差的【配对】差
    #     稿面引用这三个数（全不同的 cell 数 / 最大绝对差 / 最大相对差 / 两列 Spearman），
    #     此前无产物字段承载（外审核对判 real_gap）。此处按 (regime,n,seed) 配对后实算。
    ref_csv = OUT / "43trichotomy_formal-percell.csv"
    pairing = None
    if ref_csv.exists():
        ref = pd.read_csv(ref_csv)
        key = ["regime", "n", "seed"]
        mg = ref[key + ["err_deg"]].merge(df[key + ["err_deg"]], on=key,
                                          suffixes=("_oracle_layer", "_plugin_layer"), how="inner")
        eo = mg["err_deg_oracle_layer"].to_numpy(float)
        ep = mg["err_deg_plugin_layer"].to_numpy(float)
        dabs = np.abs(eo - ep)
        drel = dabs / np.abs(eo)
        i_abs = int(np.argmax(dabs)); i_rel = int(np.argmax(drel))
        pairing = {
            "reference_file": ref_csv.name,
            "join_keys": key,
            "n_paired_cells": int(len(mg)),
            "n_cells_43": int(len(ref)), "n_cells_126": int(len(df)),
            "n_cells_differing": int((dabs > 0).sum()),
            "all_cells_differ": bool((dabs > 0).all()),
            "max_abs_diff": float(dabs.max()),
            "max_abs_diff_cell": {"regime": str(mg.regime.iloc[i_abs]),
                                  "n": int(mg.n.iloc[i_abs]), "seed": int(mg.seed.iloc[i_abs])},
            "max_rel_diff": float(drel.max()),
            "max_rel_diff_cell": {"regime": str(mg.regime.iloc[i_rel]),
                                  "n": int(mg.n.iloc[i_rel]), "seed": int(mg.seed.iloc[i_rel])},
            "median_abs_diff": float(np.median(dabs)),
            "median_rel_diff": float(np.median(drel)),
            "spearman_err_oracle_layer_vs_plugin_layer": float(spearmanr(eo, ep).statistic),
            "note": ("两列的唯一制度性差异是主拟合多起点初值区间不同（43: phi in [20,55], logA in [22,42]；"
                     "126: phi in [15,60], logA in [15,45]），故本块量化的正是稿面自陈的那处非严格单因子替换的幅度。"
                     "退化方向一律取 oracle 方向 vdeg(theta_0)，两列同口径。"),
        }
    else:
        pairing = {"status": "reference_missing", "reference_file": ref_csv.name}

    res["honesty"] = [
        "本脚本只替换 Psi 的 theta 来源（真值 -> 独立预试验估计），设计与噪声模型不变。",
        "err_deg 仍以 oracle 退化方向 vdeg(theta_0) 投影，使误差列与 43 可直接对照；"
        "若改用 plug-in 方向会同时变动被解释量与解释量，无法归因。",
        "这是半合成研究：真值已知、误差为对真值的偏差；它检验的是 plug-in 层是否保留分离，"
        "不是对真实数据三分律的验证。",
        "收敛守卫只用预试验残差与 sigma（alg:psi 本就要求的输入），不用真值；"
        "未加守卫的 all_cells 口径一并报出，不以守卫掩盖失败率。",
    ]
    (OUT / "126trichotomy_plugin_psi.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (OUT / "147oracle_plugin_error_pairing.json").write_text(
        json.dumps({"script": "126trichotomy_plugin_psi.py",
                    "produces": "147oracle_plugin_error_pairing.json",
                    "why_separate": (
                        "本块本应写进 126 的 json，但 126 的 json 是取证台账里义务 c13bd67a3388 "
                        "的 resolve 收据所指的证据文件，而收据是【每次门禁都重新验哈希】的。"
                        "把新字段写进那份产物会让一条既有收据当场自失效（实测：门禁因此判 BLOCK）。"
                        "故拆成独立产物：126 的 json 逐字节保持不变，新字段落在这里。"),
                    "honesty": [
                        "这是跨脚本配对（读 43 的 percell 与本脚本的 percell），不是 126 内部两列之差；",
                        "若 43 的 percell 不在盘上，本文件记 reference_missing 而不静默省略。"],
                    "oracle_vs_plugin_error_pairing": pairing},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log.info("pilot fail %d/%d (%.1f%%) | sep(conv) max gap=%.4f | rate(conv) plugin=%.4f oracle=%.4f",
             res["pilot_convergence"]["n_failed"], res["pilot_convergence"]["n_cells"],
             100 * res["pilot_convergence"]["fail_rate"],
             res["separation_verdict_converged"]["max_abs_gap_plugin_vs_theory"],
             res["rate_recoverable_side_converged"]["slope_vs_psi_plugin"],
             res["rate_recoverable_side_converged"]["slope_vs_psi_oracle"])
    print(json.dumps({k: res[k] for k in ("pilot_convergence", "consistency_converged",
                                          "separation_verdict_converged", "separation_verdict_all_cells",
                                          "rate_recoverable_side_converged")},
                     ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
