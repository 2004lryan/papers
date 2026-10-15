"""
107design_multivariety.py: 三品种设计扫描扩样本 + (品种×ΔT) 分层置换检验

为什么做（ryan 2026-07-22 选择路线 b）：
  106 的负控显示，论文称为「决定性」的第二层证据——「Ψ 在固定 ΔT 内仍有判别力，故非≈cond」
  ——在有效检验下只有 p≈0.03（最保守种子 0.065），组内精确置换 p=0.14~0.23，均不达 §7.6 的
  p<0.001。根因是样本量：单品种(Royal Gala)只有 16 个超定子集、9 个层自由度约 13。
  本脚本把样本扩到三个单调品种，检验第二层证据能否在更大样本下达标。

【关键论证：为什么可以合并三个品种】
  这不是「更多嵌套子集」，而是同一套设计实验的【3 次独立生物学重复】：
  Royal Gala / Granny Smith / Red Delicious 是三个不同品种的果实，各自独立测量、
  独立拟合各自的 k(T)、Ea、σ_lnk。同品种内 16 子集仍相互嵌套（这一点未变），
  但【跨品种是独立的】，故有效信息为 3 倍而非虚增。

【关键论证：跨品种异质性如何处理】
  三品种的 σ_lnk 不同（Ψ_n ∝ 1/σ²）、Ea 不同（68.5/61.4/51.0 kJ/mol）、
  realCV 绝对水平不同（Red Delicious 窄窗曾达 98.6）。
  处理办法 = 【分层置换 + 秩统计量】：标签只在「同品种 ∧ 同 ΔT」的层内打乱，
  永不跨品种流动，因此品种间的尺度差异在数学上【完全不进入零分布】。
  这比任何标准化/归一化都干净——无需假设 realCV 跨品种可比。
  层结构：3 品种 × {ΔT=4:3, ΔT=6:6, ΔT=8:7} = 9 层，置换空间 (3!·6!·7!)^3 ~ 1e22。

【品种纳入准则（沿用论文既有口径，非本轮新定）】
  论文 §res-c5 已载明：3/4 单调软化品种给出物理合理 Ea（Royal Gala 68.5 / Granny Smith 61.4 /
  Red Delicious 51.0 kJ/mol，Arrhenius R²=0.96/0.83/0.87），而 Pink Lady 因 2°C 生理后熟
  非单调、反演失稳（Ea=858、R²=0.55）被排除。本脚本形式化为：
      纳入 <=> 20 <= Ea <= 150 kJ/mol 且 Arrhenius R² >= 0.8
  并【同时报告】纳入 Pink Lady 的敏感性结果，避免该准则被读作事后挑选。

运行方式:
    cd <项目根目录绝对路径>
    python 02code/107design_multivariety.py --device auto

输出文件:
    04outputs/107design_multivariety.xlsx  — sheets: permutation / sweep / per_variety / strata
    04outputs/107design_multivariety.json  — 机读汇总
    04outputs/107design_multivariety-sweep.csv — 三品种合并设计扫描明细（流式产物）
    05logs/107design_multivariety_YY-MM-DD_HHMMSS.log — 运行日志

方法论证锚点: refine-logs/77NEGCTRL_PERMUTATION_DESIGN.md（§1.9 四段式），台账 M55–M58。
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from export_utils import RECOMMENDED_SEEDS, get_logger, table_sheet_name, write_script_workbook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"
STEM = Path(__file__).stem
N_PERM = 10000
EA_LO, EA_HI, R2_MIN = 20.0, 150.0, 0.8   # 品种纳入准则（论文既有口径的形式化）


def _load(stem: str):
    """加载既有脚本并复用其函数——不重写实现（M53 的教训：R_GAS 单位曾致 7e5 倍偏差）。"""
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "02code" / f"{stem}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {stem}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_{stem}"] = mod
    spec.loader.exec_module(mod)
    return mod


M54 = _load("54c5_realkinetics")
M106 = _load("106negctrl_permutation")


def composite_strata(variety: np.ndarray, dT: np.ndarray) -> np.ndarray:
    """(品种, ΔT) 复合层标签。置换只在层内进行，故品种间尺度差异不进入零分布。"""
    return np.array([f"{v}|{d:g}" for v, d in zip(variety, dT)])


def partial_spearman_multi(x: np.ndarray, y: np.ndarray, Z_cols: list[np.ndarray]) -> float:
    """控制【多个】协变量后 x 与 y 的偏 Spearman（品种用哑变量进入）。"""
    rx, ry = rankdata(x).astype(float), rankdata(y).astype(float)
    Z = np.column_stack([rankdata(z).astype(float) if z.dtype.kind == "f" else z
                         for z in Z_cols] + [np.ones(len(x))])
    res = []
    for r in (rx, ry):
        beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
        res.append(r - Z @ beta)
    a, b = res
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else float("nan")


def sweep_one_variety(v: str, data: dict, Ts: list, ts: list, rng: np.random.Generator) -> tuple:
    """对单个品种做完整 26 子集设计扫描，全部调用 54c5 的权威实现。"""
    M = data[v]["mean"]
    S = np.array(data[v]["sd"], float)
    S = np.where(np.isfinite(S), S, np.nanmean(S))
    ks = [M54.fit_k(ts, M[i])[0] for i in range(len(Ts))]
    Ea, lnA, r2_arr, good = M54.arrhenius(Ts, ks)
    T_K = np.asarray(Ts) + 273.15
    lk_pred = lnA - Ea / (M54.R_GAS * T_K)
    sigma_lnk = float(np.nanstd(np.log(np.clip(ks, 1e-12, None)) - lk_pred)) or 0.05
    n_per_T = len(ts)

    rows = []
    for r in (2, 3, 4, 5):
        for combo in itertools.combinations(range(len(Ts)), r):
            Tsub = [Ts[i] for i in combo]
            ksub = [ks[i] for i in combo]
            if any(not (np.isfinite(x) and x > 0) for x in ksub):
                continue
            psi, lmin, lmax, _ = M54.psi_design(Tsub, n_per_T, sigma_lnk)
            # 【2026-08-10 订正，RYA-105】54c5.real_estimation_error 现返回 4 元组
            # （新增 bootstrap 丢弃率与旧对数线性内层的并报 CV），此处同步解包并落盘。
            real_cv, real_se, drop, real_cv_l = M54.real_estimation_error(
                Tsub, ts, M[list(combo)], S[list(combo)], rng)
            rows.append(dict(variety=v, n_temps=r, temps="/".join(f"{x:g}" for x in Tsub),
                             deltaT=max(Tsub) - min(Tsub), Psi_n=psi, lambda_min=lmin,
                             cond=lmax / max(lmin, 1e-30), realCV_Ea=real_cv, realSE_Ea=real_se,
                             boot_drop_rate=drop, realCV_Ea_loglin=real_cv_l))
    meta = dict(variety=v, Ea_kJmol=float(Ea), lnA=float(lnA), r2_arrhenius=float(r2_arr),
                sigma_lnk=sigma_lnk, n_good_T=int(np.sum(good)),
                included=bool(EA_LO <= Ea <= EA_HI and r2_arr >= R2_MIN))
    return pd.DataFrame(rows), meta


def run_tests(sub: pd.DataFrame, tag: str, rng: np.random.Generator, n_perm: int,
              log=None) -> list[dict]:
    """在给定子集上跑主检验 + 对照 + 两项负控，全部用 (品种×ΔT) 复合层置换。

    非有限值必须【显式剔除并记录】，禁止静默截断（§本项目纪律）。Pink Lady 因 Arrhenius
    反演失稳（Ea=858、R²=0.55），16 个超定子集中 11 个 realCV 非有限。
    """
    n0 = len(sub)
    ok = np.isfinite(sub.realCV_Ea.to_numpy(float)) & np.isfinite(sub.Psi_n.to_numpy(float)) \
        & np.isfinite(sub.cond.to_numpy(float))
    sub = sub[ok].reset_index(drop=True)
    n_drop = n0 - len(sub)
    if n_drop and log is not None:
        log.log(f"  [{tag}] 剔除非有限行 {n_drop}/{n0}（realCV 或 Psi 或 cond 非有限），余 {len(sub)}")
    # 剔除后层可能变得过小，需重新体检
    st_chk = composite_strata(sub.variety.to_numpy(str), sub.deltaT.to_numpy(float))
    small = {s: int(c) for s, c in zip(*np.unique(st_chk, return_counts=True)) if c < 2}
    if small and log is not None:
        log.log(f"  [{tag}] ⚠ 剔除后出现无法置换的层（n<2）：{small}——其标签在置换中保持不动")
    if len(sub) < 8:
        raise ValueError(f"[{tag}] 剔除非有限后仅剩 {len(sub)} 行，不足以做置换检验")
    psi = sub.Psi_n.to_numpy(float)
    cv = sub.realCV_Ea.to_numpy(float)
    dT = sub.deltaT.to_numpy(float)
    var = sub.variety.to_numpy(str)
    st = composite_strata(var, dT)
    # 品种哑变量（去掉一列避免共线）
    dummies = [(var == u).astype(float) for u in np.unique(var)[1:]]

    out = []
    tests = {
        "主: Ψ 偏Spearman(控ΔT+品种) · (品种×ΔT)分层置换":
            (lambda a, b: partial_spearman_multi(a, b, [dT, *dummies]), psi, cv),
        "对照: cond 偏Spearman(控ΔT+品种) · 同分层":
            (lambda a, b: partial_spearman_multi(a, b, [dT, *dummies]), -sub.cond.to_numpy(float), cv),
        "参考: Ψ 边际Spearman · 同分层":
            (lambda a, b: float(spearmanr(a, b).statistic), psi, cv),
    }
    for name, (fn, xx, yy) in tests.items():
        r = M106.perm_test(fn, xx, yy, st, rng, n_perm)
        out.append(dict(样本=tag, 检验=name, n=len(sub), n_dropped=n_drop, n_strata=len(np.unique(st)), **r))

    rand_x = rng.normal(size=len(psi))
    out.append(dict(样本=tag, 检验="负控: 随机特征基线", n=len(sub), n_dropped=n_drop, n_strata=len(np.unique(st)),
                    **M106.perm_test(lambda a, b: partial_spearman_multi(a, b, [dT, *dummies]),
                                     rand_x, cv, st, rng, n_perm)))
    id_x = rankdata(sub.temps.astype(str).to_numpy()).astype(float)
    if abs(float(spearmanr(id_x, psi).statistic)) >= 0.999:
        raise ValueError("ID-only 探针与 Ψ 近乎同序，无法充当负控")
    out.append(dict(样本=tag, 检验="负控: ID-only 探针(字典序)", n=len(sub), n_dropped=n_drop, n_strata=len(np.unique(st)),
                    **M106.perm_test(lambda a, b: partial_spearman_multi(a, b, [dT, *dummies]),
                                     id_x, cv, st, rng, n_perm)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="纯 CPU 秩统计与 2x2 线性代数，device 仅记录")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()
    log = get_logger(STEM)
    log.log(f"=== {STEM} 启动 | device={args.device} | 置换 {args.n_perm} 次 | 种子 {list(RECOMMENDED_SEEDS)} ===")

    data = M54.load_real()
    Ts = data[M54.VARIETIES[0]]["T_C"]
    ts = data[M54.VARIETIES[0]]["t_h"]
    log.log(f"温度 {Ts}°C × 时间 {ts} h × 品种 {M54.VARIETIES}")

    all_rows, all_tests, metas = [], [], []
    for seed in RECOMMENDED_SEEDS:
        rng = np.random.default_rng(seed)
        sweeps = []
        for v in M54.VARIETIES:
            dfv, meta = sweep_one_variety(v, data, Ts, ts, rng)
            meta["seed"] = seed
            metas.append(meta)
            sweeps.append(dfv)
            if seed == RECOMMENDED_SEEDS[0]:
                log.log(f"  {v:<15s} Ea={meta['Ea_kJmol']:6.1f} kJ/mol  Arr_R²={meta['r2_arrhenius']:.3f}  "
                        f"σ_lnk={meta['sigma_lnk']:.4f}  纳入={meta['included']}")
        sweep = pd.concat(sweeps, ignore_index=True)
        sweep["seed"] = seed
        all_rows.append(sweep)

        inc = {m["variety"] for m in metas if m["seed"] == seed and m["included"]}
        od = sweep[sweep.n_temps >= 3]
        main_sub = od[od.variety.isin(inc)].reset_index(drop=True)
        full_sub = od.reset_index(drop=True)
        if seed == RECOMMENDED_SEEDS[0]:
            log.log(f"纳入品种 {sorted(inc)} → 主样本 {len(main_sub)} 个超定子集"
                    f"（对照：全四品种 {len(full_sub)} 个）")
            sizes = main_sub.groupby(["variety", "deltaT"]).size().to_dict()
            log.log(f"层结构（品种×ΔT）：{ {f'{k[0]}|ΔT={k[1]:g}': v for k, v in sizes.items()} }")

        all_tests += run_tests(main_sub, f"主样本(3单调品种, n={len(main_sub)})", rng, args.n_perm, log)
        all_tests += run_tests(full_sub, f"敏感性(含PinkLady, n={len(full_sub)})", rng, args.n_perm, log)

    per_seed = pd.DataFrame(all_tests)
    sweep_all = pd.concat(all_rows, ignore_index=True)
    agg = (per_seed.groupby(["样本", "检验"], sort=False)
           .agg(n=("n", "first"), n_strata=("n_strata", "first"),
                observed_mean=("observed", "mean"), observed_std=("observed", "std"),
                p_mean=("p_one_sided", "mean"), p_max=("p_one_sided", "max"),
                null_mean=("null_mean", "mean"), z_mean=("z_vs_null", "mean"),
                n_seeds=("observed", "count")).reset_index())

    log.log(f"=== 扩样本置换检验（{len(RECOMMENDED_SEEDS)} 种子 × {args.n_perm} 次置换，单尾）===")
    for r in agg.itertuples():
        log.log(f"  [{r.样本}] {r.检验:<44s} n={r.n:<3d} 层={r.n_strata} "
                f"obs={r.observed_mean:+.4f}±{r.observed_std:.4f} p_mean={r.p_mean:.2e} "
                f"p_max={r.p_max:.2e} 零中心={r.null_mean:+.4f}")

    m = agg[(agg.样本.str.startswith("主样本")) & (agg.检验.str.startswith("主:"))].iloc[0]
    prev = dict(n=16, p_max=0.0647, observed=-0.5472)   # 106 的单品种结果，用于直接对比
    verdict = {
        "主样本": str(m.样本), "主检验": str(m.检验),
        "n_子集": int(m.n), "n_层": int(m.n_strata), "有效自由度≈": int(m.n) - int(m.n_strata),
        "观测统计量": f"{m.observed_mean:+.4f} ± {m.observed_std:.4f}",
        "p_单尾_均值": float(m.p_mean), "p_单尾_最保守种子": float(m.p_max),
        "是否达 §7.6 的 p<0.001": bool(m.p_max < 0.001),
        "零分布中心": float(m.null_mean),
        "对比_单品种106": prev,
        "口径": ("标签只在(品种×ΔT)层内打乱，故三品种间 σ_lnk/Ea/realCV 尺度差异不进入零分布；"
                 "同品种内 16 子集仍嵌套，跨品种为独立生物学重复"),
    }
    log.log("VERDICT: " + json.dumps(verdict, ensure_ascii=False, indent=2))

    sweep_all.to_csv(OUT / f"{STEM}-sweep.csv", index=False)
    (OUT / f"{STEM}.json").write_text(json.dumps(
        {"verdict": verdict, "seeds": list(RECOMMENDED_SEEDS), "n_perm": args.n_perm,
         "criterion": dict(Ea_lo=EA_LO, Ea_hi=EA_HI, r2_min=R2_MIN,
                           note="沿用论文既有口径的形式化；同时报告含 Pink Lady 的敏感性"),
         "aggregate": agg.to_dict("records"), "per_variety": metas},
        ensure_ascii=False, indent=2), encoding="utf-8")
    write_script_workbook(__file__, {
        table_sheet_name(1, "permutation"): agg,
        table_sheet_name(2, "per_seed"): per_seed,
        table_sheet_name(3, "per_variety"): pd.DataFrame(metas),
        table_sheet_name(4, "sweep"): sweep_all,
    })
    log.log(f"DONE {STEM}")


if __name__ == "__main__":
    main()
