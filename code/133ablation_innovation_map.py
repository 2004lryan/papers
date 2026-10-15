"""
133ablation_innovation_map.py: 把四个创新点逐一对上一项消融（RYA-502）

本脚本做两件事，边界很清楚：

  (A) **组装**——不重算。把已在盘的消融产物按「创新点 -> 消融项」重排成一张可上稿的表：
      - 创新点 (1) Ψ 三分律：N 因子消融 + 谱泛函选择消融，来自 104ablation_psi_aggregation*.json
        （7 个文件 = 主文件 + 5 个预注册种子 + 1 个 regtest；**regtest 按 verdict 排除**，理由见
         下方 REGTEST_EXCLUSION，取证依据为 05logs/104ablation_psi_aggregation_26-07-22_200110.log）
      - 创新点 (1) 三档 ρ：来自 43trichotomy_formal.json / -agg.csv（REGIMES p=0/0.25/0.5/1.0）
      - 创新点 (3) 补偿线：来自 122criterion_bakeoff.json 的 lemma_fisher_numerical_check
  (B) **补跑**——创新点 (2)「均匀 LAQ 与三阶抵消」在盘上没有对应消融，本脚本现算一个：
      沿退化方向对（无噪）对数似然做三阶展开，量 |f'''(0)|/|f''(0)|；再把「log k 对 θ 仿射」
      这一条结构人为破坏（lk 加一项 c·(φ r)^2），看该比值是否失去有界性。
      这是对创新点 (2) 唯一可做的数据消融形式：被消融的不是一个超参，而是模型的仿射结构本身。

运行方式:
    cd <项目根目录>
    python 02code/133ablation_innovation_map.py

输出文件:
    04outputs/133ablation_innovation_map.json
    05logs/133ablation_innovation_map_<ts>.log

口径纪律（与全文一致）:
    - 真实数据侧一律描述性，**不报名义 p**（子集嵌套非独立）。
    - 104 的两个评价口径（主 = 16 超定 / 附 = 全 26）都出，不择优。
    - 种子展布报 5 个预注册种子的区间，不只报主文件。
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"
LOGS = ROOT / "05logs"

# ---- 与 43trichotomy_formal.py 逐位一致的模型常数（不得改动，改了就不是同一个体系）----
R_GAS = 8.314
T_REF_K = 283.15
PHI_TRUE = 34.0
LOGA_TRUE = 31.0
T_LOW_C = 5.0
DELTA_L = 10.0
SIGMA = 0.02
N_LIST = [200, 500, 1000, 2000, 5000, 10000, 20000]
N0 = 200
DT0 = 16.0
SEEDS = [20060515, 20041210, 19810915, 2023, 2024]  # 项目约定 §5.4 FORMAL 固定集合
RHO_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]
C_LIST = [0.0, 0.01, 0.1, 1.0]  # c=0 即本文 Arrhenius（log k 对 θ 仿射）

PRE_SEEDS = ["seed2023", "seed2024", "seed19810915", "seed20041210", "seed20060515"]

REGTEST_EXCLUSION = (
    "104ablation_psi_aggregation_regtest.json 排除于种子展布之外。判定依据是它的生成日志 "
    "05logs/104ablation_psi_aggregation_26-07-22_200110.log：该日志逐值给出本文件的全部 verdict "
    "（Psi=-0.8853 / 去N=-0.8853 / N因子增量=0.0 / cond=-0.7912 / sigma_recovered=0.06209759749048433），"
    "且自述 tag=_regtest；其口径行写的是 'Psi_n=n_temps·5·lambda_min'、N 取值 [15,20,25]，"
    "而 2026-08-10 订正后的 5 个种子运行写的是 'Psi_n=lambda_min(I_total)'、N 取值 [3,4,5]——"
    "两者恰差一个 n_per_T=5 的因子，即 122criterion_bakeoff.py 记载的「把 n_temps 计了两次」这一 bug。"
    "故 regtest 为 N 约定订正之前的陈旧产物，不并入展布。未删除，处置意见回报父任务。"
)


def get_logger(stem: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%y-%m-%d_%H%M%S")
    lg = logging.getLogger(stem)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fh = logging.FileHandler(LOGS / f"{stem}_{ts}.log")
    sh = logging.StreamHandler()
    f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(f)
    sh.setFormatter(f)
    lg.addHandler(fh)
    lg.addHandler(sh)
    return lg


# ============================ (A) 组装已有产物 ============================

def load_104(log) -> dict:
    """读 7 个 104 文件，列出键与行数后再取值（本项目铁律：先开文件列键，不假设结构）。"""
    files = {"main": OUT / "104ablation_psi_aggregation.json"}
    for s in PRE_SEEDS:
        files[s] = OUT / f"104ablation_psi_aggregation_{s}.json"
    files["regtest"] = OUT / "104ablation_psi_aggregation_regtest.json"

    raw = {}
    for tag, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"104 产物缺失: {p}")
        d = json.load(open(p, encoding="utf-8"))
        assert set(d.keys()) == {"verdict", "ablation", "within_group"}, f"{p} 顶层键与预期不符: {list(d.keys())}"
        assert len(d["ablation"]) == 18, f"{p} ablation 行数 {len(d['ablation'])} != 18"
        log.info("载入 %-12s | ablation=%d 行 within_group=%d 行 | Psi=%s",
                 tag, len(d["ablation"]), len(d["within_group"]), d["verdict"]["本文Psi_n有效相关"])
        raw[tag] = d
    return raw


def assemble_psi_ablation(raw: dict, log) -> dict:
    """创新点 (1)：N 因子消融 + 谱泛函选择消融，双口径 × (主文件 + 5 种子)。"""
    calibers = ["16超定(主)", "26全部(附)"]
    crits = ["Psi_n(E最优xN,本文)", "D最优xN2", "A最优xN inv", "deltaT(启发式)",
             "E最优 lambda_min(去N)", "D最优 det(I)", "A最优 tr(Iinv)", "条件数 cond", "n_temps(启发式)"]

    def row(tag, cal, crit, key):
        m = {r["判据"]: r for r in raw[tag]["ablation"] if r["口径"] == cal}
        return m[crit][key]

    table = {}
    for cal in calibers:
        table[cal] = {}
        for crit in crits:
            vals = {t: row(t, cal, crit, "有效相关") for t in ["main"] + PRE_SEEDS}
            pvals = {t: row(t, cal, crit, "偏有效相关") for t in ["main"] + PRE_SEEDS}
            seed_only = [vals[t] for t in PRE_SEEDS]
            table[cal][crit] = {
                "主文件_有效相关": vals["main"],
                "主文件_偏有效相关控ΔT": pvals["main"],
                "五种子_有效相关_min": float(np.min(seed_only)),
                "五种子_有效相关_max": float(np.max(seed_only)),
                "五种子_有效相关_mean": round(float(np.mean(seed_only)), 4),
                "逐种子_有效相关": {t: vals[t] for t in PRE_SEEDS},
                "n": row("main", cal, crit, "n"),
            }

    # N 因子增量：Ψ(带N) 与 λmin(去N) 之差，逐口径逐种子
    n_factor = {}
    for cal in calibers:
        inc = {}
        for t in ["main"] + PRE_SEEDS:
            psi = row(t, cal, "Psi_n(E最优xN,本文)", "有效相关")
            lmin = row(t, cal, "E最优 lambda_min(去N)", "有效相关")
            inc[t] = round(psi - lmin, 4)
        seed_only = [inc[t] for t in PRE_SEEDS]
        n_factor[cal] = {
            "主文件_增量": inc["main"],
            "五种子_增量_min": float(np.min(seed_only)),
            "五种子_增量_max": float(np.max(seed_only)),
            "逐种子_增量": {t: inc[t] for t in PRE_SEEDS},
            "方向恒为负": bool(all(v < 0 for v in seed_only + [inc["main"]])),
            "偏相关符号翻正_主文件": {
                "Psi_偏": row("main", cal, "Psi_n(E最优xN,本文)", "偏有效相关"),
                "去N_偏": row("main", cal, "E最优 lambda_min(去N)", "偏有效相关"),
            },
        }
        log.info("N 因子消融 | %s | 主文件 %+.4f | 五种子区间 [%+.4f, %+.4f] | 恒负=%s",
                 cal, inc["main"], np.min(seed_only), np.max(seed_only), n_factor[cal]["方向恒为负"])

    # 谱泛函之间的胜负（诚实记录：附口径 Ψ 略输 D×N²；ΔT 启发式在部分种子上胜出）
    # ⚠ 主口径上 Ψ、D×N²、A×N inv 的有效相关常**逐位相同**，"最强"是并列而非独占。
    #   故此处记录**全部**并列判据，不用 min() 取首序，避免把三方并列讲成 Ψ 独强。
    best, best_sole = {}, {}
    for cal in calibers:
        b, bs = {}, {}
        for t in ["main"] + PRE_SEEDS:
            m = {c: row(t, cal, c, "有效相关") for c in crits}
            top = min(m.values())                              # 有效相关越负越好
            winners = sorted([c for c in crits if m[c] == top])  # 并列者全列
            b[t] = winners
            bs[t] = winners[0] if len(winners) == 1 else None
        best[cal] = b
        best_sole[cal] = bs
        log.info("各抽样最强判据（并列全列）| %s | %s", cal,
                 {t: ("+".join(v) if len(v) > 1 else v[0]) for t, v in b.items()})

    PSI = "Psi_n(E最优xN,本文)"
    psi_top = {cal: sum(1 for t in ["main"] + PRE_SEEDS if PSI in best[cal][t]) for cal in calibers}
    psi_sole = {cal: sum(1 for t in ["main"] + PRE_SEEDS if best[cal][t] == [PSI]) for cal in calibers}
    log.info("Ψ 位列最强的抽样次数（含并列，共 6 次）: %s", psi_top)
    log.info("Ψ **独占**最强的抽样次数（共 6 次）: %s", psi_sole)

    return {"双口径判据表": table, "N因子消融": n_factor,
            "各抽样最强判据_并列全列": best,
            "各抽样唯一最强判据_并列则为null": best_sole,
            "Psi位列最强次数_含并列_共6次抽样": psi_top,
            "Psi独占最强次数_共6次抽样": psi_sole,
            "平票口径": "有效相关越负越好；主口径上 Psi_n、D最优xN2、A最优xN inv 三者常逐位相同，"
                       "此时按并列处理并列出全部并列判据，不按列表顺序择一，"
                       "故『Ψ 位列最强 4/6』指并列在内，『独占』另计。",
            "regtest排除理由": REGTEST_EXCLUSION,
            "口径说明": "有效相关<0 表示该判据判为更好的设计确实对应更低的真实反演误差；"
                        "描述性报告，子集嵌套非独立，不报名义 p。"}


def assemble_rho_trichotomy(log) -> dict:
    """创新点 (1) 的三档 ρ 消融：直接取 43trichotomy_formal 的既有产物，不补跑。"""
    j = json.load(open(OUT / "43trichotomy_formal.json", encoding="utf-8"))
    agg = pd.read_csv(OUT / "43trichotomy_formal-agg.csv")
    log.info("43trichotomy_formal-agg.csv 列: %s | 行: %d", list(agg.columns), len(agg))

    # 脚本 43 的 REGIMES: p 即正文的 ρ（ΔT_n = DT0·(n/N0)^(-ρ)）
    regime_rho = {"fixed_recover": 0.0, "sub_recover": 0.25, "critical": 0.5, "subcritical": 1.0}
    band = {0.0: "ρ<1/2（可恢复）", 0.25: "ρ<1/2（可恢复）", 0.5: "ρ=1/2（临界）", 1.0: "ρ>1/2（不可恢复）"}

    out = {}
    for rn, rho in regime_rho.items():
        g = agg[agg.regime == rn].sort_values("n")
        assert len(g) == len(N_LIST), f"{rn} 的 n 网格行数 {len(g)} != {len(N_LIST)}"
        tri = j["trichotomy"][rn]
        out[rn] = {
            "rho": rho,
            "档": band[rho],
            "Psi_n_首": tri["psi_first"], "Psi_n_末": tri["psi_last"],
            "err_首": tri["err_first"], "err_末": tri["err_last"],
            "Psi_倍变": round(tri["psi_last"] / tri["psi_first"], 4),
            "err_倍变": round(tri["err_last"] / tri["err_first"], 4),
        }
        log.info("ρ=%.2f %-24s | Ψ %8.3f -> %10.3f | err %.4f -> %.4f",
                 rho, band[rho], tri["psi_first"], tri["psi_last"], tri["err_first"], tri["err_last"])

    return {"三档": out,
            "n网格": j["n_list"], "种子": j["seeds"], "bootstrap": j["bootstrap"],
            "collapse_slope": j["collapse_slope"],
            "说明": "被消融的设计要素是收缩指数 ρ 本身（ΔT_n=c·n^(-ρ)）。四个 regime 覆盖 ρ<1/2 两档、"
                    "ρ=1/2 一档、ρ>1/2 一档，故三档齐备，无须补跑。"}


def assemble_compensation(log) -> dict:
    """创新点 (3)：补偿线 = Fisher 最小特征方向。取 122 的既有数值核验。"""
    d = json.load(open(OUT / "122criterion_bakeoff.json", encoding="utf-8"))
    lf = d["lemma_fisher_numerical_check"]
    log.info("引理 fisher 数值核验 | min|cos| = %.12f | 子集数 = %d",
             lf["min_abs_cos_krug_vs_min_eigvec"], d["n_subsets_total"])
    return {
        "min_abs_cos_krug_vs_min_eigvec": lf["min_abs_cos_krug_vs_min_eigvec"],
        "n_subsets": d["n_subsets_total"],
        "n_overdetermined": d["n_overdetermined"],
        "E_vs_D_verdict": d["E_vs_D_is_NOT_established"]["verdict"],
        "E_D_边际Spearman相同": (d["spearman_marginal_vs_realCV"]["crit_E"]
                               == d["spearman_marginal_vs_realCV"]["crit_D"]),
        "crit_E_marginal": d["spearman_marginal_vs_realCV"]["crit_E"],
        "crit_D_marginal": d["spearman_marginal_vs_realCV"]["crit_D"],
        "近奇异区中位条件数": d["criterion_collapse_in_near_singular_regime"]["median_condition_number"],
        "pairwise_E_vs_D": d["criterion_collapse_in_near_singular_regime"]["pairwise_spearman"]["crit_E|crit_D"],
        "pairwise_E_vs_cond": d["criterion_collapse_in_near_singular_regime"]["pairwise_spearman"]["crit_E|crit_cond"],
        "诚实订正": "122 自带 interpretation 末句称 D-最优「偏离真实误差排序」，被同一 JSON 的 "
                    "spearman_marginal_vs_realCV 推翻：crit_D 与 crit_E 的边际 Spearman 完全相同。"
                    "D 偏离的是 E 的排序（成对 0.9945），不是真实误差的排序。本表不沿用该末句。",
    }


# ============================ (B) 创新点 (2)：三阶抵消消融 ============================

def design(n: int, rho: float, seed: int):
    """与 43trichotomy_formal.py 逐位一致的设计生成。"""
    dT = float(np.clip(DT0 * (n / N0) ** (-rho), 0.05, 60))
    rng = np.random.default_rng(seed * 100003 + n)
    T_C = rng.uniform(T_LOW_C, T_LOW_C + dT, n)
    t = rng.uniform(0.5, DELTA_L, n)
    return T_C, t, dT


def laq_ratio(T_C, t, c: float):
    """沿退化方向的三阶/二阶比。

    被消融的模型族（外科式，只动仿射性这一条）:
        lk(θ) = logA - φ·r + c·r·(φ - φ*)²,    r = T_ref/T_K
    在评价点 θ=θ*=(φ*, logA*) 上：
        - 附加项取值 0        -> lk、k、z、g、响应 β 与本文 Arrhenius **逐点相同**
        - 附加项一阶导取值 0  -> ∂lk/∂θ 不变 -> Fisher Î₁、退化方向 u、Ψ_n **完全不变**
        - 附加项二阶导 = 2c·r -> ∂²_u lk 由恒 0 变为非零，**仅**仿射性被破坏
    c=0 即本文 Arrhenius。故 c 是一根只拧「log k 对 θ 是否仿射」的旋钮，不混入任何别的差异。

    响应: β = 1 - exp(-k·t)，z = k·t，∂β/∂lk = z·e^{-z} =: g，∂²β/∂lk² = g·(1-z)。
    重标度 r_n = sqrt(n·λmin(Î₁)) = sqrt(Ψ_n)，使无噪对数似然沿 u 的二阶系数恰为 -1（LAQ 归一化）。
    返回 |f'''(0)| / |f''(0)|；引理 laq 主张它在 ρ≤1/2 上有界。

    数值：λmin 由 J 的 SVD 取（奇异值平方），不走 eigh(JᵀJ)——后者把条件数平方，
    在窄窗设计上会把 λmin 打到下溢。
    """
    T_K = T_C + 273.15
    r = T_REF_K / T_K
    lk = LOGA_TRUE - PHI_TRUE * r          # 附加项在 θ* 处为 0
    k = np.exp(lk)
    z = k * t
    g = z * np.exp(-z)

    # lk 对 θ=(φ, logA) 的一阶偏导（在 θ* 处与 c 无关）
    dlk_dphi = -r
    dlk_dlogA = np.ones_like(r)
    J = np.column_stack([dlk_dphi * g, dlk_dlogA * g])  # ∂β/∂θ

    n = len(t)
    A = J / (SIGMA * np.sqrt(n))            # AᵀA = Î₁
    sv, V = np.linalg.svd(A, full_matrices=False)[1:]
    lmin = float(sv[-1] ** 2)
    u = V[-1, :]
    psi_n = float(n * lmin)
    r_n = float(np.sqrt(psi_n))

    # 沿 u 的 lk 方向导数
    d1 = u[0] * dlk_dphi + u[1] * dlk_dlogA               # ∂_u lk
    d2 = (u[0] ** 2) * (2.0 * c * r)                      # ∂²_u lk（c=0 时恒为 0）

    dbeta_u = g * d1                                       # ∂_u β
    d2beta_u = g * (1.0 - z) * d1 ** 2 + g * d2            # ∂²_u β

    f2 = -(np.sum(dbeta_u ** 2) / SIGMA ** 2) / r_n ** 2   # 应 == -1
    f3 = -(3.0 * np.sum(dbeta_u * d2beta_u) / SIGMA ** 2) / r_n ** 3
    ratio = float(abs(f3) / abs(f2))

    # 破坏仿射性所付出的代价有**闭式**：设 u=(u_φ,u_A) 为退化方向，则
    #   Σ r·g²·d1 = -σ²·Ψ_n·u_φ   （因 (JᵀJ)u = σ²Ψ_n·u 的第一分量）
    # 故 f3(c) - f3(0) = 6c·u_φ³ / sqrt(Ψ_n)  —— 精确恒等式，与 n、设计无关，只由 Ψ_n 控制。
    # 这把创新点 (2) 直接接回创新点 (1)：仿射性一旦破坏，三阶余项的存亡由同一个标量 Ψ_n 决定。
    f3_affine = -(3.0 * np.sum(dbeta_u * (g * (1.0 - z) * d1 ** 2)) / SIGMA ** 2) / r_n ** 3
    f3_c_closed = 6.0 * c * u[0] ** 3 / np.sqrt(psi_n)
    return dict(psi_n=psi_n, lmin=lmin, r_n=r_n, f2=float(f2), f3=float(f3),
                ratio=ratio, max_z=float(z.max()), affine_second_deriv_max=float(np.abs(d2).max()),
                f3_affine=float(f3_affine), f3_c_closed=float(f3_c_closed),
                closed_form_abserr=float(abs((f3 - f3_affine) - f3_c_closed)),
                closed_form_relerr=float(abs((f3 - f3_affine) - f3_c_closed)
                                         / max(abs(f3_c_closed), 1e-300)) if c > 0 else 0.0,
                cond_I1=float((sv[0] / sv[-1]) ** 2))


def loglog_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    lx, ly = np.log(x[ok]), np.log(y[ok])
    A = np.column_stack([lx, np.ones_like(lx)])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((ly - pred) ** 2)); ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), r2


def run_laq_ablation(log) -> dict:
    log.info("=== 创新点 (2) 三阶抵消消融：ρ × c 网格 ===")
    rows = []
    for rho in RHO_LIST:
        for c in C_LIST:
            for n in N_LIST:
                per_seed = []
                for seed in SEEDS:
                    T_C, t, dT = design(n, rho, seed)
                    per_seed.append(laq_ratio(T_C, t, c))
                rows.append(dict(
                    rho=rho, c=c, n=n, dT=dT,
                    ratio=float(np.mean([p["ratio"] for p in per_seed])),
                    psi_n=float(np.mean([p["psi_n"] for p in per_seed])),
                    f2=float(np.mean([p["f2"] for p in per_seed])),
                    d2lk_max=float(np.max([p["affine_second_deriv_max"] for p in per_seed])),
                    max_z=float(np.max([p["max_z"] for p in per_seed])),
                    closed_form_abserr=float(np.max([p["closed_form_abserr"] for p in per_seed])),
                    closed_form_relerr=float(np.max([p["closed_form_relerr"] for p in per_seed])),
                    cond_I1=float(np.max([p["cond_I1"] for p in per_seed])),
                    ratio_affine_only=float(np.mean([abs(p["f3_affine"]) for p in per_seed])),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "133ablation_innovation_map-laq.csv", index=False)

    # LAQ 归一化自检：二阶系数必须恒为 -1
    f2_dev = float(np.max(np.abs(df.f2 + 1.0)))
    log.info("LAQ 归一化自检 | max|f''(0)+1| = %.3e（应 ~0，否则重标度写错了）", f2_dev)
    assert f2_dev < 1e-9, f"LAQ 二阶归一化失败: {f2_dev}"

    # 仿射性自检：c=0 时 ∂²_u lk 必须恒为 0
    aff_dev = float(df[df.c == 0.0].d2lk_max.max())
    assert aff_dev == 0.0, f"c=0 应当仿射，实测 ∂²_u lk 最大 {aff_dev}"
    log.info("仿射性自检 | c=0 时 max|∂²_u lk| = %.1f（本文模型）；c=1.0 时 = %.4g（已破坏）",
             aff_dev, float(df[df.c == 1.0].d2lk_max.max()))

    # 外科性自检：Ψ_n 必须与 c 无关（消融只动二阶结构，不动 Fisher）
    piv = df.pivot_table(index=["rho", "n"], columns="c", values="psi_n")
    psi_dev = float(np.max(np.abs(piv.values - piv[0.0].values[:, None])))
    assert psi_dev == 0.0, f"Ψ_n 应与 c 无关（消融须外科），实测最大偏差 {psi_dev}"
    log.info("外科性自检 | Ψ_n 对 c 的最大偏差 = %.1f（0 即消融只动了仿射性，未动 Fisher）", psi_dev)

    # 闭式自检：f3(c) - f3(0) 应等于 6c·u_φ³/sqrt(Ψ_n)
    # 该恒等式依赖 u 为**精确**最小特征向量；近奇异设计上 cond(Î₁) 可达 1e10 量级，
    # 数值 u 的相对误差随之放大，故按**相对**误差判定，不宣称机器精度。
    cf_abs = float(df.closed_form_abserr.max())
    cf_rel = float(df.closed_form_relerr.max())
    cond_max = float(df.cond_I1.max())
    log.info("闭式自检 | max 绝对误差 %.3e | max 相对误差 %.3e | max cond(Î₁) %.3e",
             cf_abs, cf_rel, cond_max)
    assert cf_rel < 1e-4, f"闭式恒等式不成立（相对误差 {cf_rel}）"

    # 仿射项自身的标度（与 c 无关），单列出来
    aff_slopes = {}
    for rho in RHO_LIST:
        g0 = df[(df.rho == rho) & (df.c == 0.0)].sort_values("n")
        s, r2 = loglog_slope(g0.n.values, g0.ratio_affine_only.values)
        aff_slopes[f"rho={rho}"] = {"loglog_slope": round(s, 4), "R2": round(r2, 4)}

    slopes = {}
    for rho in RHO_LIST:
        slopes[f"rho={rho}"] = {}
        for c in C_LIST:
            g = df[(df.rho == rho) & (df.c == c)].sort_values("n")
            s, r2 = loglog_slope(g.n.values, g.ratio.values)
            slopes[f"rho={rho}"][f"c={c}"] = {
                "loglog_slope_ratio_vs_n": round(s, 4), "R2": round(r2, 4),
                "ratio_n200": round(float(g.ratio.iloc[0]), 6),
                "ratio_n20000": round(float(g.ratio.iloc[-1]), 6),
                "倍变": round(float(g.ratio.iloc[-1] / g.ratio.iloc[0]), 4),
            }
            log.info("ρ=%.2f c=%-6g | 斜率 %+.3f (R²=%.3f) | ratio %.4g -> %.4g (×%.3g)",
                     rho, c, s, r2, g.ratio.iloc[0], g.ratio.iloc[-1], g.ratio.iloc[-1] / g.ratio.iloc[0])

    # 判据：仿射（本文，c=0）在 ρ≤1/2 上比值不增；破坏仿射后在临界档是否转为增长
    affine_ok = all(slopes[f"rho={rho}"]["c=0.0"]["loglog_slope_ratio_vs_n"] < 0
                    for rho in [0.0, 0.25, 0.5])
    crit_broken = {f"c={c}": slopes["rho=0.5"][f"c={c}"]["loglog_slope_ratio_vs_n"] for c in C_LIST}

    verdict = {
        "仿射时三阶比在全部ρ上衰减": bool(all(
            slopes[f"rho={rho}"]["c=0.0"]["loglog_slope_ratio_vs_n"] < 0 for rho in RHO_LIST)),
        "仿射时逐ρ斜率": {f"rho={rho}": slopes[f"rho={rho}"]["c=0.0"]["loglog_slope_ratio_vs_n"]
                       for rho in RHO_LIST},
        "破坏仿射后逐ρ斜率_c=1": {f"rho={rho}": slopes[f"rho={rho}"]["c=1.0"]["loglog_slope_ratio_vs_n"]
                              for rho in RHO_LIST},
        "临界档ρ=1/2逐c斜率": crit_broken,
        "闭式": "f'''(c)-f'''(0) = 6c·u_φ³/sqrt(Ψ_n)；数值核验 max 相对误差 %.2e"
                "（受 cond(Î₁) 最大 %.1e 限制，非机器精度）" % (cf_rel, cond_max),
        "闭式含义": "破坏仿射性所付的代价恰由 Ψ_n 控制：ρ<1/2 时 Ψ_n→∞ 故代价消失，"
                   "ρ=1/2 时 Ψ_n→γ 故代价降为非零常数（不再消失），ρ>1/2 时 Ψ_n→0 故代价发散。"
                   "即三阶抵消与三分律受同一个标量支配。",
        "ρ>1/2不主张": "引理 laq 只主张 ρ≤1/2（正文注 laqscope）；ρ=0.75/1.0 的数字一并给出仅供参照，"
                      "不作为主张依据。",
    }
    log.info("VERDICT(2): %s", json.dumps(verdict, ensure_ascii=False))
    return {"网格": {"rho": RHO_LIST, "c": C_LIST, "n": N_LIST, "seeds": SEEDS},
            "模型": "lk(θ)=logA-φ·r+c·r·(φ-φ*)²；c=0 即本文 Arrhenius（log k 对 θ 仿射）。"
                    "在评价点 θ* 上响应、一阶导、Fisher、Ψ_n 与 c 无关，仅 ∂²_u lk 由 0 变非零。",
            "被消融的要素": "log k 对 θ 的仿射性本身（c 从 0 调开），而非某个超参",
            "量": "|f'''(0)|/|f''(0)|，f(s)=沿退化方向 u、以 r_n=sqrt(Ψ_n) 重标度的无噪对数似然",
            "自检": {"LAQ归一化_max_abs_f2_plus_1": f2_dev,
                    "外科性_Psi_n对c最大偏差": psi_dev,
                    "闭式恒等式_max_abserr": cf_abs,
                    "闭式恒等式_max_relerr": cf_rel,
                    "max_cond_I1": cond_max},
            "仿射项自身标度": aff_slopes,
            "slopes": slopes, "verdict": verdict}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = get_logger("133ablation_innovation_map")
    log.info("=== 133ablation_innovation_map 启动 | 四创新点 -> 消融项 ===")

    raw = load_104(log)
    psi_ab = assemble_psi_ablation(raw, log)
    rho_ab = assemble_rho_trichotomy(log)
    comp = assemble_compensation(log)
    laq = run_laq_ablation(log)

    summary = {
        "script": "133ablation_innovation_map.py",
        "创新点1_三分律": {"N因子与谱泛函消融": psi_ab, "三档rho消融": rho_ab},
        "创新点2_均匀LAQ三阶抵消": laq,
        "创新点3_补偿线": comp,
        "创新点4_设计法则": {
            "说明": "两轴消融（固定 ΔT 变 n / 固定 n 变 ΔT）已在 04outputs/123montecarlo_criterion_study.json，"
                    "并已写入正文 §sec:mc，本表不重复计算，只在消融表中登记其归属。",
            "产物": "04outputs/123montecarlo_criterion_study.json",
        },
        "纪律": "真实数据侧描述性、子集嵌套非独立、全文不报名义 p；两个评价口径都出，不择优。",
    }
    p = OUT / "133ablation_innovation_map.json"
    json.dump(summary, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log.info("写出 %s", p)
    log.info("DONE 133ablation_innovation_map")


if __name__ == "__main__":
    main()
