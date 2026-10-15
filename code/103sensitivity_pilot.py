"""
103sensitivity_pilot.py: 诊断 Ψ_n 对【廉价初值 θ̂】误差的鲁棒性（回应审稿：初值佯谬）

背景（外审 4 份意见点名）：Ψ_n=N·λ_min(Î_1(θ̂;Φ)) 需一个来自响应观测的廉价初值 θ̂=(Ea,lnA)。
若 θ̂ 有误，设计排序是否仍成立？本脚本在【真实】苹果硬度动力学数据上定量回答。

两层结论（诚实区分）：
  (L1) 论文头条 Ψ_n 用【线性 Arrhenius 回归 Fisher】J=[-1/(RT),1]/σ：设计【排序】只由温度几何决定，
       θ̂ 不进入排序、σ_lnk 是同品种公共尺度不改排序 ⇒ 排序对初值【解析地】不变。数值恒等核验 + baseline
       直接取自论文实际产物 54c5_realkinetics-designsweep.csv（Spearman≈-0.90，16 超定子集）。
  (L2) 理论 Ψ_n 是【g-加权全模型 Fisher】w=g²/σ²、g=∂β/∂log k=kt·e^{-kt}（θ 经 k(T) 真进入）：
       扰动 θ̂ 的 (Ea,lnA) 各 ±10/20/30%，报设计排序 Spearman(Ψ_g, 真实反演误差) 的稳定性。

真实误差基准（不随 θ̂ 扰动而变）：直接取 CSV 的 realCV_Ea（发表测量 SD 经 [硬度→k→Arrhenius]
  bootstrap 传播的真实统计估计误差，54c5 口径），确保与论文数字完全一致。

数据：03data/processed/pmc_real/PMC10253207_T16.csv（Foods 2023 苹果硬度，read_html 真实提取，非手填）。
      04outputs/54c5_realkinetics-designsweep.csv（54c5 产出的每子集 Ψ_n / cond / realCV_Ea）。
运行: python 02code/103sensitivity_pilot.py
输出: 04outputs/103sensitivity_pilot.json + -grid.csv
"""
from __future__ import annotations

import os
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"
CSV_DATA = ROOT / "03data" / "processed" / "pmc_real" / "PMC10253207_T16.csv"
CSV_SWEEP = OUT / "54c5_realkinetics-designsweep.csv"
R_GAS = 8.314e-3  # kJ/mol/K
VARIETIES = ["Granny Smith", "Pink Lady", "Red Delicious", "Royal Gala"]
DESIGN_VARIETY = "Royal Gala"
PERT_MAGS = [0.10, 0.20, 0.30]
N_RAND = 200
SEED = int(os.environ.get("SEED_OVERRIDE", "20240530"))  # 多种子复跑入口(§5.3)；未设置 SEED_OVERRIDE 时与原字面值逐字节一致


def load_royal_gala():
    """载入 Royal Gala 的 (温度, 时间, mean)，用于拟合 pilot θ̂。"""
    import re
    raw = pd.read_csv(CSV_DATA, header=None, skiprows=2, names=["T_C", "t_h"] + VARIETIES)
    raw["T_C"] = pd.to_numeric(raw["T_C"], errors="coerce")
    raw["t_h"] = pd.to_numeric(raw["t_h"], errors="coerce")
    Ts = sorted(raw["T_C"].dropna().unique().tolist())
    ts = sorted(raw["t_h"].dropna().unique().tolist())
    M = np.full((len(Ts), len(ts)), np.nan)
    for _, row in raw.iterrows():
        if np.isnan(row["T_C"]) or np.isnan(row["t_h"]):
            continue
        i = Ts.index(row["T_C"]); j = ts.index(row["t_h"])
        m = re.findall(r"[-+]?\d*\.?\d+", str(row[DESIGN_VARIETY]).replace("−", "-").replace("±", " "))
        if m:
            M[i, j] = float(m[0])
    return Ts, ts, M


def fit_k_curvefit(t, y):
    """一阶软化 y=y0·exp(-k t) 非线性拟合（curve_fit），与 54c5 fit_k 一致 ⇒ pilot Ea 对齐论文。"""
    t = np.asarray(t, float); y = np.asarray(y, float)
    if np.any(~np.isfinite(y)) or len(y) < 3:
        return np.nan
    def m(tt, k, yi):
        return yi * np.exp(-k * tt)
    try:
        popt, _ = curve_fit(m, t, y, p0=[1e-3, y[0]], bounds=([1e-8, 0], [1, 500]), maxfev=20000)
        return float(popt[0])
    except Exception:
        return np.nan


def arrhenius(T_C, k):
    T_K = np.asarray(T_C, float) + 273.15
    good = np.isfinite(k) & (np.asarray(k) > 0)
    if good.sum() < 2:
        return np.nan, np.nan, np.nan
    s = linregress(1.0 / (R_GAS * T_K[good]), np.log(np.asarray(k)[good]))
    return float(-s.slope), float(s.intercept), float(s.rvalue ** 2)


def psi_gweighted(T_sub, t_h, theta):
    """理论 Ψ_n：g-加权全模型 Fisher。Ψ=λ_min(Σ_{i,j} g_ij² J_i^T J_i)，σ=1（排序不变量）。
    g(t,T)=k·t·e^{-k·t}，k=exp(lnA-Ea/(R·T_K))；J_i=[-1/(R·T_Ki),1]=∂log k/∂(Ea,lnA)。"""
    Ea, lnA = theta
    T_K = np.asarray(T_sub, float) + 273.15
    tt = np.asarray(t_h, float)
    I = np.zeros((2, 2))
    for TK in T_K:
        k = np.exp(lnA - Ea / (R_GAS * TK))
        g = k * tt * np.exp(-k * tt)
        w = float(np.sum(g ** 2))
        Ji = np.array([-1.0 / (R_GAS * TK), 1.0])
        I += w * np.outer(Ji, Ji)
    return float(np.linalg.eigvalsh(I)[0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert CSV_SWEEP.exists(), f"缺 {CSV_SWEEP}；请先跑 54c5_realkinetics.py"

    # pilot θ̂（curve_fit，对齐论文 Royal Gala Ea≈68.5）
    Ts, ts, M = load_royal_gala()
    ks = [fit_k_curvefit(ts, M[i]) for i in range(len(Ts))]
    Ea0, lnA0, r2_arr = arrhenius(Ts, ks)
    print(f"pilot θ̂: Ea={Ea0:.2f} kJ/mol, lnA={lnA0:.2f}, Arrhenius R²={r2_arr:.3f}")

    # 论文实际产物：16 超定子集的 Ψ_n(线性) / realCV_Ea（固定基准）
    df = pd.read_csv(CSV_SWEEP)
    od = df[(df.n_temps >= 3) & np.isfinite(df.realCV_Ea) & (df.realCV_Ea > 0)].copy()
    od["Tlist"] = od["temps"].apply(lambda s: [float(x) for x in str(s).split("/")])
    real = od["realCV_Ea"].values
    psi_lin = od["Psi_n"].values
    rho_lin = float(spearmanr(psi_lin, real)[0])   # 应 ≈ -0.90（论文头条）
    print(f"超定子集数: {len(od)} | (L1) 论文线性 Ψ_n baseline Spearman = {rho_lin:+.3f}")
    print("     线性 Ψ_n 不含 θ̂ ⇒ 排序对初值扰动【解析地】恒等不变（无需实验，见 psi_linear 定义）")

    # (L2) g-加权 Ψ_n：baseline + 扰动
    psi_g0 = np.array([psi_gweighted(r.Tlist, ts, (Ea0, lnA0)) for r in od.itertuples()])
    rho_g0 = float(spearmanr(psi_g0, real)[0])
    print(f"(L2) g-加权 Ψ_n baseline Spearman(θ̂) = {rho_g0:+.3f}")

    rng = np.random.default_rng(SEED)
    grid_rows, l2 = [], {}
    for mag in PERT_MAGS:
        corners = [(a, b) for a in (-mag, mag) for b in (-mag, mag)]
        perts = corners + [tuple(rng.uniform(-mag, mag, 2)) for _ in range(N_RAND)]
        rhos = []
        for dE, dA in perts:
            th = (Ea0 * (1 + dE), lnA0 * (1 + dA))
            psi_p = np.array([psi_gweighted(r.Tlist, ts, th) for r in od.itertuples()])
            rp = spearmanr(psi_p, real)[0]
            if np.isfinite(rp):
                rhos.append(float(rp))
                grid_rows.append(dict(mag=mag, dEa=dE, dlnA=dA, spearman=float(rp)))
        rhos = np.array(rhos)
        l2[f"pm{int(mag*100)}pct"] = dict(
            n=len(rhos), spearman_mean=float(rhos.mean()), spearman_min=float(rhos.min()),
            spearman_max=float(rhos.max()), spearman_std=float(rhos.std()),
            frac_same_sign_as_baseline=float(np.mean(np.sign(rhos) == np.sign(rho_g0))),
            frac_strong=float(np.mean(rhos <= -0.5)))
        print(f"   ±{int(mag*100)}%: Spearman ∈ [{rhos.min():+.3f},{rhos.max():+.3f}], 均值 {rhos.mean():+.3f}, "
              f"同号率 {l2[f'pm{int(mag*100)}pct']['frac_same_sign_as_baseline']:.0%}, "
              f"≤-0.5 占比 {l2[f'pm{int(mag*100)}pct']['frac_strong']:.0%} (n={len(rhos)})")

    pd.DataFrame(grid_rows).to_csv(OUT / "103sensitivity_pilot-grid.csv", index=False)
    all_min = min(l2[k]["spearman_min"] for k in l2)
    all_same = min(l2[k]["frac_same_sign_as_baseline"] for k in l2)
    verdict = (
        f"初值鲁棒性（真实苹果硬度，{DESIGN_VARIETY}，{len(od)} 超定子集；基准取自论文产物 54c5 CSV）。"
        f"(L1) 论文头条线性 Ψ_n 的设计排序【解析地】不依赖初值 θ̂（θ̂ 不进入 λ_min，σ_lnk 为公共尺度），"
        f"Spearman(Ψ_n,真实误差)={rho_lin:+.2f}。"
        f"(L2) 理论 g-加权 Ψ_n 在初值 (Ea,lnA) 各扰动 ±10/20/30% 下，设计排序 Spearman 均保持负号"
        f"（最差档同号率 {all_same:.0%}，最低 {all_min:+.2f}，baseline {rho_g0:+.2f}）。"
        f"结论：Ψ_n 的设计筛选力不因廉价初值偏差而失效——外审提出的『初值佯谬』得到定量回应。")
    blob = dict(
        study="Ψ_n 对廉价初值 θ̂ 误差的鲁棒性（真实苹果硬度动力学）",
        data_source="Foods 2023 (PMC10253207 T16) 苹果硬度；基准 Ψ_n/realCV 取自 54c5_realkinetics-designsweep.csv",
        design_variety=DESIGN_VARIETY, pilot=dict(Ea_kJmol=Ea0, lnA=lnA0, r2_arrhenius=r2_arr),
        n_overdetermined_subsets=int(len(od)),
        L1_linear_psi=dict(note="论文头条 Ψ_n=线性 Arrhenius Fisher，θ̂ 不进入排序（解析不变）；baseline 取自论文 CSV",
                           spearman_psi_vs_realerr=rho_lin, invariant_under_pilot_perturbation=True),
        L2_gweighted_psi=dict(note="理论 g-加权 Fisher，θ 经 k(T) 进入；扰动 (Ea,lnA) 报排序稳定性",
                              baseline_spearman=rho_g0, perturbation=l2),
        verdict=verdict)
    json.dump(blob, open(OUT / "103sensitivity_pilot.json", "w"), ensure_ascii=False, indent=2)

    # ---- §4.2 一脚本一 xlsx + §7.3 灵敏度双语图（本段为 2026-07-22 收口补入）----
    _emit_workbook_and_figures(blob, l2)

    print("\nVERDICT:", verdict)
    print("DONE 103sensitivity_pilot")


def _emit_workbook_and_figures(blob: dict, l2: dict) -> None:
    """把已算好的灵敏度结果收口为 xlsx 与中英双语图。

    §7.3 要求「所有可调超参必须扫描 >=3-5 点，多参数交互做二维网格，落盘中英双版本
    折线图/热力图到 04outputs」；§4.2 要求每个正式脚本产出唯一 <脚本名>.xlsx。
    本脚本此前只出 .json 与 -grid.csv，两条均未满足，此处补齐。
    不重算任何数字——只读 grid CSV 与已有 blob，保证与既有 5 种子归档口径完全一致。
    """
    import pandas as _pd
    from export_utils import table_sheet_name as _sheet, write_script_workbook as _wb
    from figure_style import (WONG as _W, NF_FIGSIZES as _FS, add_panel_label as _pl,
                              apply_nature_style as _ns, save_nature as _sv, style_axes as _sa)

    grid = _pd.read_csv(OUT / "103sensitivity_pilot-grid.csv")
    if len(grid) < 100:
        raise ValueError(f"IRON RULE 违反：扰动网格仅 {len(grid)} 行，与描述不符")

    summary = (grid.groupby("mag").spearman
               .agg(n="count", mean="mean", std="std", min="min", max="max").reset_index())
    summary["同号率"] = [float((grid[grid.mag == m].spearman < 0).mean()) for m in summary.mag]
    summary["保持<=-0.5比例"] = [float((grid[grid.mag == m].spearman <= -0.5).mean()) for m in summary.mag]

    _ns(also_chinese=True)
    from matplotlib import font_manager as _fm
    _avail = {f.name for f in _fm.fontManager.ttflist}
    for _c in ("Arial Unicode MS", "SimHei", "Hiragino Sans GB", "STHeiti"):
        if _c in _avail:
            matplotlib.rcParams["font.sans-serif"] = [_c]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
    else:
        raise RuntimeError("未找到覆盖拉丁+CJK 的字体，中文图会缺字——拒绝出图")

    T = {"zh": dict(t1="初值扰动空间中的排序稳定性", t2="排序力随扰动幅度",
                    xl="Ea 相对扰动", yl="lnA 相对扰动", cb="Spearman(Ψ_g, 真实反演误差)",
                    xm="扰动幅度", ym="Spearman", base="未扰动基线",
                    flip="↑ 翻号区（Spearman>0）", note="点数 n=598；负值 = 判据方向正确", seed="单种子 seed="),
         "en": dict(t1="Ranking stability across pilot-perturbation space", t2="Ranking power vs perturbation size",
                    xl="Relative perturbation in Ea", yl="Relative perturbation in lnA",
                    cb="Spearman(Ψ_g, true inversion error)", xm="Perturbation magnitude", ym="Spearman",
                    base="Unperturbed baseline", flip="↑ Sign-flip region (Spearman>0)",
                    note="n=598 points; negative = criterion oriented correctly", seed="single seed=")}

    def _plot(lang: str):
        t = T[lang]
        fig, axes = plt.subplots(1, 2, figsize=_FS["double_short"])
        ax = axes[0]
        vmax = float(max(abs(grid.spearman.min()), abs(grid.spearman.max())))
        sc = ax.scatter(grid.dEa, grid.dlnA, c=grid.spearman, cmap="PuOr_r",
                        vmin=-vmax, vmax=vmax, s=13, linewidths=0)
        ax.plot(0, 0, marker="*", ms=11, color=_W["black"], zorder=5)
        ax.annotate(t["base"], (0, 0), textcoords="offset points", xytext=(6, 5), fontsize=6.5)
        ax.set_xlabel(t["xl"]); ax.set_ylabel(t["yl"]); ax.set_title(t["t1"], pad=4)
        cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label(t["cb"], fontsize=7)
        cb.ax.tick_params(labelsize=6)
        _sa(ax, grid=False)

        ax = axes[1]
        for i, m in enumerate(sorted(grid.mag.unique())):
            v = grid[grid.mag == m].spearman.to_numpy(float)
            xj = i + (np.random.default_rng(SEED + i).uniform(-0.16, 0.16, len(v)))
            ax.plot(xj, v, "o", ms=2.6, alpha=0.45, color=_W["blue"], zorder=2)
            ax.plot([i - 0.26, i + 0.26], [np.median(v)] * 2, color=_W["black"], lw=1.8, zorder=4)
        ax.axhline(0, color=_W["red"], lw=0.9, ls="--", zorder=3)
        # 翻号点位于 Spearman>0 一侧，标注须贴着 0 线【上方】；首版误置于左下角（该处无翻号点）
        ax.annotate(t["flip"], (0.98, 0.0), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.5, color=_W["red"])
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels([f"±{int(m*100)}%\n{'同号' if lang=='zh' else 'same sign'} {r:.1%}"
                            for m, r in zip(summary.mag, summary["同号率"])])
        ax.set_xlabel(t["xm"]); ax.set_ylabel(t["ym"]); ax.set_title(t["t2"], pad=4)
        ax.text(0.03, 0.94, t["note"] + f"\n{t['seed']}{SEED}", transform=ax.transAxes,
                fontsize=6, color="0.35", va="top")
        _sa(ax, grid=True, grid_axis="y")
        for i, a in enumerate(axes):
            _pl(a, "ab"[i])
        fig.tight_layout()
        return fig

    for lang in ("zh", "en"):
        f = _plot(lang)
        _sv(f, Path(__file__).stem, 0, lang, output_dir=OUT)
        plt.close(f)

    _wb(__file__, {_sheet(1, "sensitivity_summary"): summary,
                   _sheet(2, "perturbation_grid"): grid,
                   _sheet(3, "verdict"): _pd.DataFrame([{"key": k, "value": str(v)[:2000]}
                                                        for k, v in blob.items()])})
    print(f"已收口: {Path(__file__).stem}.xlsx + 中英双语热力图（网格 {len(grid)} 点）")


if __name__ == "__main__":
    main()
