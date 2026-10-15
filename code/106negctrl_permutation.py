"""
106negctrl_permutation.py: Ψ 诊断的负控套件——置换检验 + 随机特征基线 + ID-only 探针 (§7.6)

为什么必须做（§7.6 是投稿必要前提，本项目此前完全缺失）：
  C5 线五个脚本 grep `permut|置换|shuffle` 命中为 0。而论文 §「为何不报 p」用整段解释：
  16 个超定子集由同一批 5 个温度点组合派生、相互嵌套，Spearman 的名义 p 以独立为前提，
  在此会被严重放大而失去意义，故全文只报描述性秩相关。
  置换检验正是该问题的标准解：它不推导零分布而直接构造零分布，有效性不依赖独立性，
  只依赖零假设下标签的可交换性——而可交换性可由置换方案的选择来保证（见下）。

两种置换方案（并列报告，主/辅分明）：
  [主] ΔT 分层内置换：只在 ΔT 相同的子集之间打乱 realCV 标签。
       保留「窗宽决定误差量级」这一真实结构，检验的是【Ψ 是否携带窗宽之外的信息】,
       即论文称为「决定性」的偏 Spearman 的置换版。层大小 3/6/7 -> 置换空间 3!*6!*7! ~ 2.2e7。
  [辅] 全局置换：跨全部 16 子集打乱标签。偏松（反保守）——它把「温度数越多误差越低」
       的系统关系也一并打散，零分布会偏离 0，故仅作对照、不作主结论。
  诊断指标：两种方案的置换零分布中心都应报出。中心明显偏离 0 即说明该方案破坏了应保留的结构。

另两项负控（§7.6 要求）：
  [随机特征基线] 用与 Ψ 同形状的随机值替换 Ψ，其与 realCV 的相关应落在置换零分布内。
  [ID-only 探针] 仅用子集的序号（枚举顺序，与设计几何无关）作预测量，不应优于随机。

多种子（§5.3）：
  对 RECOMMENDED_SEEDS 的每个种子各有一份 designsweep 产物（由 105 归档），
  逐种子独立跑全套负控，报 mean±std，n_primary=5。

运行方式:
    cd <项目根目录绝对路径>
    python 02code/106negctrl_permutation.py --device auto

输出文件:
    04outputs/106negctrl_permutation.xlsx  — sheets: negative_control / per_seed / nulldist
    04outputs/106negctrl_permutation.json  — 机读汇总
    05logs/106negctrl_permutation_YY-MM-DD_HHMMSS.log — 运行日志

方法论证锚点: refine-logs/77NEGCTRL_PERMUTATION_DESIGN.md（§1.9 四段式），台账 M55–M57。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from export_utils import RECOMMENDED_SEEDS, get_logger, table_sheet_name, write_script_workbook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"
SEED_DIR = OUT / "105_seeds"
STEM = Path(__file__).stem
N_PERM = 10000          # §7.6 要求 >=1000；取 10000 使 p 的分辨率达 1e-4
MIN_STRATUM = 3         # 层内少于 3 个样本无法置换出非平凡排列，跳过


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """控制 z 后 x 与 y 的偏 Spearman（与 104/91 同口径）。"""
    rx, ry, rz = (rankdata(v).astype(float) for v in (x, y, z))
    Z = np.stack([rz, np.ones_like(rz)], axis=1)
    res = []
    for r in (rx, ry):
        beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
        res.append(r - Z @ beta)
    a, b = res
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else float("nan")


def stratified_permute(y: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """在每个 strata 层内独立打乱 y；层间不混合，从而保留层级结构。"""
    out = y.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if len(idx) >= 2:
            out[idx] = y[rng.permutation(idx)]
    return out


def perm_test(stat_fn, x: np.ndarray, y: np.ndarray, strata: np.ndarray | None,
              rng: np.random.Generator, n_perm: int = N_PERM) -> dict:
    """单尾置换检验（H1: 统计量比零分布更负，即判据越大误差越小）。

    p 用 (1 + #{null <= obs}) / (1 + n_perm)，分子分母各 +1 是标准的有偏修正，
    保证 p 恒 >0（否则 p=0 会被误读为"绝对显著"）。
    """
    obs = stat_fn(x, y)
    # 非有限守卫（CRITICAL）：若 obs 或输入含 NaN/Inf，np.sum(nan <= nan) 恒为 0，
    # p 会静默取到下限 1/(1+n_perm)，看起来极显著实为全 NaN——这是伪造显著性的路径。
    # 实战：107 首版把 Pink Lady（16 个超定子集中 11 个 realCV 非有限）并入敏感性臂，
    # 全部检验（含随机特征基线）都报 p=1e-4，正是被该基线的不可能取值暴露。
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"置换检验输入含非有限值：x 非有限 {int(np.sum(~np.isfinite(x)))} 个，"
                         f"y 非有限 {int(np.sum(~np.isfinite(y)))} 个——须在调用前显式剔除并记录")
    if not np.isfinite(obs):
        raise ValueError(f"观测统计量非有限（{obs}）——拒绝在其上做置换检验")
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = stratified_permute(y, strata, rng) if strata is not None else y[rng.permutation(len(y))]
        null[i] = stat_fn(x, yp)
    p = float((1 + int(np.sum(null <= obs))) / (1 + n_perm))
    return dict(observed=float(obs), p_one_sided=p,
                null_mean=float(null.mean()), null_std=float(null.std(ddof=1)),
                null_q025=float(np.percentile(null, 2.5)),
                null_q975=float(np.percentile(null, 97.5)),
                z_vs_null=float((obs - null.mean()) / null.std(ddof=1)) if null.std(ddof=1) > 0 else float("nan"),
                n_perm=n_perm, n_used=int(len(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="纯 CPU 秩统计，device 仅记录")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()
    log = get_logger(STEM)
    log.log(f"=== {STEM} 启动 | device={args.device} | 置换次数={args.n_perm} ===")

    seeds = list(RECOMMENDED_SEEDS)
    rows: list[dict] = []
    nulls: list[dict] = []
    for seed in seeds:
        f = SEED_DIR / f"54c5_realkinetics-designsweep-seed{seed}.csv"
        if not f.exists():
            raise FileNotFoundError(f"IRON RULE 违反：{f} 不在盘（先跑 105multiseed_realdata.py）")
        df = pd.read_csv(f)
        od = df[df.n_temps >= 3].reset_index(drop=True)
        if len(od) != 16:
            raise ValueError(f"IRON RULE 违反：seed={seed} 超定子集数 {len(od)} != 16")
        psi = od.Psi_n.to_numpy(float)
        cv = od.realCV_Ea.to_numpy(float)
        dT = od.deltaT.to_numpy(float)
        rng = np.random.default_rng(seed)

        # 层大小体检：太小的层无法提供置换自由度，须显式记录（禁止静默截断）
        sizes = {int(s): int((dT == s).sum()) for s in np.unique(dT)}
        usable = {k: v for k, v in sizes.items() if v >= MIN_STRATUM}
        log.log(f"seed={seed} 层大小 {sizes}；可置换层 {usable}"
                + ("" if usable == sizes else "  ⚠ 存在过小层，其内标签不参与打乱"))

        tests = {
            # [主] 分层置换 + 偏 Spearman：Ψ 是否携带 ΔT 之外的信息
            "主: Ψ 偏Spearman(控ΔT) · ΔT分层置换":
                (lambda a, b: partial_spearman(a, b, dT), psi, cv, dT),
            # [辅] 全局置换 + 边际 Spearman：偏松对照
            "辅: Ψ 边际Spearman · 全局置换":
                (lambda a, b: float(spearmanr(a, b).statistic), psi, cv, None),
            # [辅] 分层置换 + 边际 Spearman：层内边际关联
            "辅: Ψ 边际Spearman · ΔT分层置换":
                (lambda a, b: float(spearmanr(a, b).statistic), psi, cv, dT),
            # 对照判据：条件数（论文的主要竞品）
            "对照: cond 偏Spearman(控ΔT) · ΔT分层置换":
                (lambda a, b: partial_spearman(a, b, dT), -od.cond.to_numpy(float), cv, dT),
        }
        for name, (fn, xx, yy, st) in tests.items():
            r = perm_test(fn, xx, yy, st, rng, args.n_perm)
            rows.append(dict(seed=seed, 检验=name, **r))
            if seed == seeds[0]:
                nulls.append(dict(检验=name, seed=seed, null_mean=r["null_mean"],
                                  null_std=r["null_std"], null_q025=r["null_q025"],
                                  null_q975=r["null_q975"]))

        # 负控 1：随机特征基线——用同形状随机值替换 Ψ
        rand_x = rng.normal(size=len(psi))
        r = perm_test(lambda a, b: partial_spearman(a, b, dT), rand_x, cv, dT, rng, args.n_perm)
        rows.append(dict(seed=seed, 检验="负控: 随机特征基线(控ΔT)", **r))

        # 负控 2：ID-only 探针——必须使用【与 Ψ 无关】的 ID。
        # 【实战抓错记录】首版用 np.arange(16) 作 ID，结果与 Ψ_n 的偏 Spearman 完全相同
        # （−0.5472±0.0814，四位小数一致）。根因：54c5 输出 CSV 时 sort_values("Psi_n")，
        # 行号即 Ψ 的秩，而 Spearman 是秩统计量 —— 该探针实际上在测 Ψ 自己，毫无负控作用。
        # 改用温度组合字符串的字典序：它由枚举本身决定，与 Ψ 的数值无关。
        id_x = rankdata(od.temps.astype(str).to_numpy()).astype(float)
        assert abs(float(spearmanr(id_x, psi).statistic)) < 0.999, \
            "ID-only 探针与 Ψ 近乎同序，无法充当负控——请更换 ID 定义"
        r = perm_test(lambda a, b: partial_spearman(a, b, dT), id_x, cv, dT, rng, args.n_perm)
        rows.append(dict(seed=seed, 检验="负控: ID-only 探针(控ΔT, 字典序)", **r))
        rows[-1]["note"] = f"ID 与 Psi_n 的 Spearman={float(spearmanr(id_x, psi).statistic):+.3f}(须远离±1)"

        # 组内精确置换 p：逐 ΔT 层穷举全部排列，给出【精确】单尾 p。
        # 目的：论文 L460 报「ΔT=4/8 组内 Spearman −1.00/−0.96、p<5e-4」，
        # 但 ΔT=4 层 n=3，全部 3!=6 种排列中达到 ρ=−1 的占 1/6，故精确 p 的下限恒为 0.1667，
        # p<5e-4 在该层数学上不可能。此处给出可复算的精确值取代之。
        from itertools import permutations as _perms
        for dv in np.unique(dT):
            m = dT == dv
            xg, yg = psi[m], cv[m]
            ng = int(m.sum())
            if ng < 3 or ng > 8:      # >8 时排列数过大，改用随机置换（本数据最大 7，不会触发）
                continue
            obs_g = float(spearmanr(xg, yg).statistic)
            allp = list(_perms(range(ng)))
            null_g = np.array([spearmanr(xg, yg[list(q)]).statistic for q in allp])
            p_exact = float(np.sum(null_g <= obs_g + 1e-12) / len(allp))
            rows.append(dict(seed=seed, 检验=f"组内精确置换 ΔT={dv:.0f} (n={ng})",
                             observed=obs_g, p_one_sided=p_exact,
                             null_mean=float(null_g.mean()), null_std=float(null_g.std(ddof=1)),
                             null_q025=float(np.percentile(null_g, 2.5)),
                             null_q975=float(np.percentile(null_g, 97.5)),
                             z_vs_null=float("nan"), n_perm=len(allp),
                             note=f"穷举全部 {len(allp)} 种排列；该层精确 p 的理论下限={1/len(allp):.4f}"))

    per_seed = pd.DataFrame(rows)
    agg = (per_seed.groupby("检验", sort=False)
           .agg(observed_mean=("observed", "mean"), observed_std=("observed", "std"),
                p_mean=("p_one_sided", "mean"), p_max=("p_one_sided", "max"),
                null_mean=("null_mean", "mean"), null_std=("null_std", "mean"),
                z_mean=("z_vs_null", "mean"), n_seeds=("seed", "count"))
           .reset_index())

    log.log(f"=== 负控套件结果（{len(seeds)} 种子 × {args.n_perm} 次置换，单尾）===")
    for r in agg.itertuples():
        log.log(f"  {r.检验:<42s} obs={r.observed_mean:+.4f}±{r.observed_std:.4f} "
                f"p_mean={r.p_mean:.2e} p_max={r.p_max:.2e} 零分布中心={r.null_mean:+.4f} z={r.z_mean:+.2f}")

    main_row = agg[agg.检验.str.startswith("主:")].iloc[0]
    verdict = {
        "主检验": str(main_row.检验),
        "观测统计量": f"{main_row.observed_mean:+.4f} ± {main_row.observed_std:.4f}",
        "p_单尾_最大值(5种子中最保守)": float(main_row.p_max),
        "是否达 §7.6 的 p<0.001": bool(main_row.p_max < 0.001),
        "零分布中心": float(main_row.null_mean),
        "零分布中心是否≈0(方案有效性体检)": bool(abs(main_row.null_mean) < 0.05),
        "随机特征基线_p": float(agg[agg.检验.str.contains("随机特征")].p_mean.iloc[0]),
        "IDonly探针_p": float(agg[agg.检验.str.contains("ID-only")].p_mean.iloc[0]),
        "口径": ("p=(1+#{null<=obs})/(1+n_perm) 单尾；分层置换保留 ΔT 层结构，"
                 "故 p 的有效性不依赖子集独立性；全局置换偏松，仅作对照"),
    }
    log.log("VERDICT: " + json.dumps(verdict, ensure_ascii=False, indent=2))

    (OUT / f"{STEM}.json").write_text(json.dumps(
        {"verdict": verdict, "seeds": seeds, "n_perm": args.n_perm,
         "aggregate": agg.to_dict("records"), "per_seed": rows, "nulldist_seed0": nulls},
        ensure_ascii=False, indent=2), encoding="utf-8")
    write_script_workbook(__file__, {
        table_sheet_name(1, "negative_control"): agg,
        table_sheet_name(2, "per_seed"): per_seed,
        table_sheet_name(3, "nulldist"): pd.DataFrame(nulls),
    })
    log.log(f"DONE {STEM}")


if __name__ == "__main__":
    main()
