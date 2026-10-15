"""
127budget_matched_sensitivity.py: 「按条件数筛查代价高昂」在**收紧预算配平容差**下的敏感性

动机（取证发现 BASELINE-BC001）：
    123montecarlo_criterion_study.py 的候选集只要求 N_obs 极差 <= 2.0 倍即算"预算配平"。
    E/D/A/c 用重复加权 Fisher 信息，样本量大者直接占优；条件数对信息阵的公共缩放不变，
    对 N_obs 视而不见。故 cond 的部分劣势可能来自"它不看预算"这一定义性质，
    而非它作为设计筛查的失败。本脚本把容差从 2.0 收紧到 1.0（严格同预算），
    检验结论是否仍成立。

方法：完全复用 123 已产出的 1000 个设计（04outputs/123montecarlo_criterion_study-designs.csv），
     不重跑仿真，只重做选型循环，唯一变动的是容差 TOL。随机流与 123 同种子同用法。

运行方式:
    python 02code/127budget_matched_sensitivity.py

输出文件:
    04outputs/127budget_matched_sensitivity.json
    05logs/127budget_matched_sensitivity_<ts>.log
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"; LOGS = ROOT / "05logs"
DESIGNS = OUT / "123montecarlo_criterion_study-designs.csv"

SEED = 20060515          # 与 123 的选型循环同种子
N_SCEN = 3000
K_CAND = 8
# RYA-537：与 123 同步，头条口径改为定义量 E_het/D_het（异方差装配），
# 同方差代理 E/D/A/c 保留为特例对照。
# 【2026-08-23，RYA-581 / 义务 de049d538e54】与 123 同步改为【一张表一个算子】：
# 头条表六判据全部取自异方差定义量算子（_het 后缀列），同方差代理表结构对称、自成一表。
# 此前 E_het/D_het 与 A/c_deg/c_Ea/cond 混排，属跨算子同台，稿面所引的
# 「严格同预算下条件数中位超额」因此也混了装配口径，本轮一并订正。
CRIT_HET = ["E_het", "D_het", "A_het", "c_deg_het", "c_Ea_het", "cond_het"]
CRIT_HOM = ["E", "D", "A", "c_deg", "c_Ea", "cond"]
CRIT_ALL = CRIT_HET + CRIT_HOM


def _is_cond(k: str) -> bool:
    return k.split("_")[0] == "cond"
TOLS = [1.0, 1.05, 1.1, 1.25, 1.5, 2.0]   # 2.0 = 123 的原设置；1.0 = 严格同预算


def get_logger(stem):
    LOGS.mkdir(parents=True, exist_ok=True); ts = time.strftime("%y-%m-%d_%H%M%S")
    lg = logging.getLogger(stem); lg.setLevel(logging.INFO); lg.handlers.clear()
    fh = logging.FileHandler(LOGS / f"{stem}_{ts}.log"); sh = logging.StreamHandler()
    f = logging.Formatter("%(asctime)s %(levelname)s %(message)s"); fh.setFormatter(f); sh.setFormatter(f)
    lg.addHandler(fh); lg.addHandler(sh); return lg


def boot_ci(vals, rng, n=2000):
    vals = np.asarray(vals)
    bs = [float(np.median(rng.choice(vals, len(vals), replace=True))) for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def run_selection(df, tol, rb):
    """与 123montecarlo_criterion_study.py 的选型循环同构，唯一差别是预算容差 tol。

    关键：必须先按 N_obs 排序再取滑窗（123:319 `df.sort_values("N_obs")`），
    否则窗口内预算天然发散，容差过滤会误杀绝大多数场景。
    """
    df = df.sort_values("N_obs").reset_index(drop=True)
    Nobs = df["N_obs"].to_numpy(); relerr = df["relerr_Ea"].to_numpy()
    crit_vals = {k: df[k].to_numpy() for k in CRIT_ALL}
    picks = {k: [] for k in CRIT_ALL}
    agree_ED = 0; nobs_spread = []
    for _ in range(N_SCEN):
        lo = rb.integers(0, len(df) - K_CAND)
        idx = np.arange(lo, lo + K_CAND)
        spread = Nobs[idx].max() / max(Nobs[idx].min(), 1e-9)
        if spread > tol:
            continue
        nobs_spread.append(float(spread))
        best_true = relerr[idx].min()
        chosen = {}
        for k in CRIT_ALL:
            v = crit_vals[k][idx]
            j = idx[np.argmin(v)] if _is_cond(k) else idx[np.argmax(v)]
            chosen[k] = j
            picks[k].append(relerr[j] / best_true - 1.0)
        agree_ED += int(chosen["E_het"] == chosen["D_het"])
    n_scen = len(picks["E_het"])
    if n_scen == 0:
        return {"n_scenarios": 0, "note": "该容差下无可用候选集"}
    out = {"n_scenarios": int(n_scen),
           "operator": "heteroscedastic (def:psi 定义量)；同方差代理表另列",
           "nobs_spread_max": float(max(nobs_spread)), "nobs_spread_median": float(np.median(nobs_spread)),
           "E_and_D_pick_same_design_rate": float(agree_ED / n_scen),
           "by_criterion": {}, "by_criterion_homoscedastic_surrogate": {}}
    for k, v in picks.items():
        tgt = "by_criterion" if k in CRIT_HET else "by_criterion_homoscedastic_surrogate"
        out[tgt][k] = {"median_regret_vs_oracle": float(np.median(v)),
                       "mean_regret_vs_oracle": float(np.mean(v)),
                       "p90_regret_vs_oracle": float(np.percentile(v, 90)),
                       "ci95_of_median": boot_ci(v, rb)}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); log = get_logger("127budget_matched_sensitivity")
    df = pd.read_csv(DESIGNS)
    _missing = [c for c in ["N_obs", "relerr_Ea"] + CRIT_ALL if c not in df.columns]
    if _missing:
        raise SystemExit(f"designs CSV 缺列 {_missing}——请先重跑 123montecarlo_criterion_study.py")
    log.info("designs=%d  全部判据列在盘", len(df))

    res = {"script": "127budget_matched_sensitivity.py", "seed": SEED,
           "source_designs": str(DESIGNS.name), "n_designs": int(len(df)),
           "question": ("把 123 的预算配平容差从 2.0 倍收紧到严格同预算，"
                        "「按条件数筛查代价高昂」是否仍成立"),
           "protocol": ("完全复用 123 已产出的设计库，不重跑仿真；只重做选型循环，"
                        f"唯一变动量为容差 TOL in {TOLS}（2.0 即 123 原设置）。"),
           "by_tolerance": {}}
    for tol in TOLS:
        rb = np.random.default_rng(SEED)     # 每个容差用同一条流，使各档可配对比较
        res["by_tolerance"][str(tol)] = run_selection(df, tol, rb)
        r = res["by_tolerance"][str(tol)]
        if r.get("n_scenarios"):
            log.info("tol=%-5s n=%-5d cond_het median=%+.4f  E_het median=%+.4f  "
                     "(hom 对照 cond=%+.4f) spread<=%.3f",
                     tol, r["n_scenarios"], r["by_criterion"]["cond_het"]["median_regret_vs_oracle"],
                     r["by_criterion"]["E_het"]["median_regret_vs_oracle"],
                     r["by_criterion_homoscedastic_surrogate"]["cond"]["median_regret_vs_oracle"],
                     r["nobs_spread_max"])

    strict = res["by_tolerance"]["1.0"]
    loose = res["by_tolerance"]["2.0"]
    if strict.get("n_scenarios"):
        res["verdict"] = {
            "operator": "heteroscedastic (def:psi 定义量)；下列 *_homoscedastic_surrogate 为同方差对照",
            "cond_median_regret_strict": strict["by_criterion"]["cond_het"]["median_regret_vs_oracle"],
            "cond_median_regret_strict_ci95": strict["by_criterion"]["cond_het"]["ci95_of_median"],
            "cond_median_regret_orig_tol2": loose["by_criterion"]["cond_het"]["median_regret_vs_oracle"],
            "E_median_regret_strict": strict["by_criterion"]["E_het"]["median_regret_vs_oracle"],
            "cond_still_worse_than_E_strictly_matched": bool(
                strict["by_criterion"]["cond_het"]["median_regret_vs_oracle"]
                > strict["by_criterion"]["E_het"]["median_regret_vs_oracle"]),
            "n_scenarios_strict": strict["n_scenarios"],
            "cond_median_regret_strict_homoscedastic_surrogate":
                strict["by_criterion_homoscedastic_surrogate"]["cond"]["median_regret_vs_oracle"],
            "cond_median_regret_strict_ci95_homoscedastic_surrogate":
                strict["by_criterion_homoscedastic_surrogate"]["cond"]["ci95_of_median"],
            "E_median_regret_strict_homoscedastic_surrogate":
                strict["by_criterion_homoscedastic_surrogate"]["E"]["median_regret_vs_oracle"],
        }
    res["honesty"] = [
        "本脚本不产生新仿真，只在既有设计库上重做选型；故它检验的是选型口径的稳健性，不是新证据。",
        "严格同预算档的场景数远少于 2.0 档，其区间更宽，须按场景数解读。",
        "随机流与 123 不同：123 的 rb 在选型循环之前已被前序阶段消耗，本脚本每档重新播种。"
        "故本脚本的 tol=2.0 一档是同一规则下的**独立重抽**，不是 123 那一次场景集的逐位复现；"
        "可比较的是各容差档之间（它们共用同一条流），不可用来复算 123 自己的 cond 中位超额。",
        # 【2026-08-19 订正，RYA-526】原串硬编码「+89.3%」，两轮 123 产物都对不上
        # （旧 +89.1%、订正后 +107.2%）。硬编码的跨脚本数字必然随上游重跑而过期，
        # 故一律不再写死具体数值，只陈述「不可跨流复算」这一口径事实。
        # 【2026-08-23，RYA-581】此处原写死「1.467 倍」，与上面刚记过的教训自相矛盾：
        # 跨脚本硬编码的数字必然随上游重跑而过期。改为由本轮实测填入。
        ("实际落在窗口内的 N_obs 极差最大仅 %.3f 倍，故 123 名义上的 2.0 倍容差从未被触及。"
         % res["by_tolerance"]["2.0"]["nobs_spread_max"]),
    ]
    (OUT / "127budget_matched_sensitivity.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res.get("verdict", res["by_tolerance"]), ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
