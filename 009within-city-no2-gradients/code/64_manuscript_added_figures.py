"""
64_manuscript_added_figures.py: main-text Figures 1, 3 and 4 (figures a, b and c below) and the graphical abstract,
one set in Chinese and one in English (the graphical abstract in English only)

Figure a  Relationship between within-city NO₂ and built-up exposure
      (a) Exposure distribution of the analysis sample (station-years), marking the p10 and p90 that the main
          text uses to convert effects;
      (b) Binned scatter after demeaning within the same city and year, with one fitted line for each of the two
          periods 2015–2017 and 2022–2024; the slope is the pooled within-city gradient of that period. A single
          panel shows both "how large the gradient is" and "how much it flattened over the decade";
      (c)(d) What this axis measures: within the same city and year, the difference between p90 and p10 stations
          in the commuting-peak to overnight concentration ratio and in the weekday excess over weekends, one row
          each for NO₂, CO, SO₂ and PM2.5 (converted from the slopes of check ① in script 65; city-clustered 95%
          intervals). Pollutants emitted by traffic should show both time signatures; pollutants dominated by
          coal combustion and secondary formation should not.

Figure b  Robustness of the NO₂ gradient path
      Top: annual gradients — full sample (with 95% intervals), with meteorological-level interactions added,
          with meteorological-anomaly and climate-mean interactions added, and stations present in all ten years;
      Bottom: forest plots of three quantities — relative change 2015→2024, and mean annual change in 2017–2020
          and in 2020–2024.
      The data are taken from the outputs of script 62 (meteorology) and script 63 (station composition) and are
      not recomputed in this script.

Figure c  Timing of the narrowing and fleet growth
      Left: cumulative completion over 2017–2023 of the gradient narrowing and of the national NEV increase
          (the gradient with the bootstrap intervals of script 61); the fleet is drawn once more on the log scale
          used in the regressions: the log moves growth earlier and is the most lenient scale for the
          electrification hypothesis;
      Right: mean annual change in the three segments (bootstrap intervals of script 61) and, as bars, the mean
          annual fleet increase in 2017–2020 and 2020–2024 (the fleet series begins in 2017, so 2015–2017 has none).

Graphical abstract  A 13 cm × 5.2 cm canvas, legible at 13 cm × 5 cm and at least 1328 × 531 pixels; exported as
      PDF, PNG (300 dpi) and TIFF (600 dpi).

**Style**: the same as the three figures of script 52 — the same series colours (NO₂ red, fleet dark-grey dashed
line, Blue Sky Protection Campaign light-grey band), the same font rules (Chinese figures use a Chinese font with
ASCII minus signs, English figures use DejaVu Sans with true minus signs), and figure titles likewise.
The font and formatting functions are taken directly from script 52 rather than written a second time.

**Dependencies**: run scripts 51, 61, 62, 63 and 65 first (this script reads their workbooks).

Usage:
    python code/64_manuscript_added_figures.py

Output files (稿件新增图 = added manuscript figures, 图 = figure):
    outputs/64_稿件新增图-图a.pdf / -图b.pdf / -图c.pdf        (Chinese; a .png of the same name for each)
    outputs/64_稿件新增图-图a_en.pdf / -图b_en.pdf / -图c_en.pdf (English; likewise)
    outputs/64_稿件新增图-图GA_en.pdf / .tiff / .png             (graphical abstract)
    outputs/64_稿件新增图.xlsx                                   (plotting data of figures a–c)
    logs/64_manuscript_added_figures.log
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _load(name: str, file: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    if spec is None or spec.loader is None:
        raise SystemExit(f"{file} 无法加载")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export_utils = _load("export_utils", "01_export_utils.py")
m51 = _load("m51", "51_station_timing_and_power.py")
m52 = _load("m52", "52_station_figures.py")   # fonts, formatting and the national NEV series

STA, CITY, TIME, EXPO = m51.STA, m51.CITY, m51.TIME, m51.EXPO
STEM = "64_稿件新增图"
NO2 = m52.NO2
RED, GREY_NEV, BAND = "#C0392B", "0.25", "0.85"
EARLY, LATE = (2015, 2017), (2022, 2024)
N_BINS = 15
OUT = export_utils.OUTPUT_DIR
# Robustness rows: (workbook, column that selects the series, value, legend/row-label key, line style).
# Lines use only red and greys, so they stay distinguishable in black-and-white print
SPECS = [("63", "样本", "全样本", "spec_all", "-"),
         ("62", "模型", "M0", "spec_m0", None),
         ("62", "模型", "M1", "spec_m1", "--"),
         ("62", "模型", "M2", "spec_m2", ":"),
         ("63", "样本", "十年齐全站点", "spec_fixed", "-.")]
SPEC_COLOR = {"spec_all": RED, "spec_m0": "#E6A39C", "spec_m1": "0.35", "spec_m2": "0.55", "spec_fixed": "0.1"}
# Rows of figure a (c)(d): (species name in the workbook of script 65, axis label, colour). NO₂ red, the rest grey
SIG_ORDER = [("NO2", NO2, RED), ("CO", "CO", "0.25"), ("SO2", m52.SO2, "0.5"), ("PM2.5", m52.PM25, "0.65")]

TEXT = {
    "zh": {
        "a1_title": "暴露分布（分析样本）", "a1_x": "站点 500 m 内建成区占比", "a1_y": "站-年数",
        "a2_title": "城内 " + NO2 + " 梯度：两期对比",
        "a2_x": "建成区占比相对同城同年均值的偏离", "a2_y": "ln " + NO2 + " 相对同城同年均值的偏离",
        "a2_leg": "{a}–{b}：斜率 {slope}（{n} 站-年）",
        "c_title": "早晚高峰对夜间的浓度比", "c_x": "p90 与 p10 站点之差（%）",
        "d_title": "工作日超出周末的幅度", "d_x": "p90 与 p10 站点之差（百分点）",
        "b_title": "城内 " + NO2 + " 梯度路径对气象与站点构成的稳健性",
        "b_y": "城内 " + NO2 + " 梯度", "year": "年份",
        "spec_all": "全样本（主估计）", "spec_m0": "有气象数据的城市",
        "spec_m1": "加 气象水平 × 暴露", "spec_m2": "加 气象距平与气候均值 × 暴露",
        "spec_fixed": "十年齐全的 801 站",
        "f1": "2015→2024 相对变化（%）", "f2": "2017–2020 年均变化", "f3": "2020–2024 年均变化",
        "c1_title": "收窄与车队增长的累计完成度", "c1_y": "占 2017–2023 年变化的比例（%）",
        "c1_grad": "城内 " + NO2 + " 梯度收窄（95% 区间）", "c1_nev": "全国 NEV 保有量增长",
        "c1_nevlog": "同上，按回归所用的对数计", "c1_note": "2020 年：{g}% 对 {v}%（对数 {w}%）",
        "c2_title": "分段速率与各段车队增量", "c2_y": "梯度平均年变化（单位/年）",
        "c2_nev": "全国 NEV 年均增量（万辆/年）", "c2_share": "（车队 2017–2024 年\n增量的 {p}%）",
        "c2_na": "（车队序列\n始于 2017 年）",
        "band": "蓝天保卫战",
    },
    "en": {
        "a1_title": "Exposure distribution (analysis sample)", "a1_x": "Built-up fraction within 500 m",
        "a1_y": "Station-years",
        "a2_title": "Within-city " + NO2 + " gradient in two periods",
        "a2_x": "Built-up fraction relative to city-year mean",
        "a2_y": "ln " + NO2 + " relative to city-year mean",
        "a2_leg": "{a}–{b}: slope {slope} ({n} station-years)",
        "c_title": "Commuting-peak to night ratio", "c_x": "Difference, p90 vs p10 station (%)",
        "d_title": "Weekday excess over weekends", "d_x": "Difference, p90 vs p10 station (percentage points)",
        "b_title": "Robustness of the within-city " + NO2 + " gradient path",
        "b_y": "Within-city " + NO2 + " gradient", "year": "Year",
        "spec_all": "All stations (main estimate)", "spec_m0": "Cities with meteorology",
        "spec_m1": "+ meteorological levels × exposure",
        "spec_m2": "+ anomalies and climate × exposure", "spec_fixed": "801 stations present all ten years",
        "f1": "Change 2015→2024 (%)", "f2": "Mean change per year, 2017–2020",
        "f3": "Mean change per year, 2020–2024",
        "c1_title": "Cumulative completion, 2017–2023", "c1_y": "Share of the 2017–2023 change (%)",
        "c1_grad": "Narrowing of the " + NO2 + " gradient (95% interval)", "c1_nev": "Growth of the national NEV stock",
        "c1_nevlog": "Same, on the log scale used in the regressions", "c1_note": "2020: {g}% vs {v}% ({w}% in logs)",
        "c2_title": "Segment rates and fleet growth", "c2_y": "Mean change in gradient per year",
        "c2_nev": "National NEV growth ($10^4$ vehicles per year)",
        "c2_share": "({p}% of the fleet's\n2017–2024 growth)",
        "c2_na": "(fleet series\nbegins in 2017)",
        "band": "Blue Sky campaign",
    },
}


def panel(letter: str, title: str) -> str:
    """Prefix the panel title with its letter, matching (a)(b)… in the figure captions."""
    return f"({letter}) {title}"


def save(fig: Figure, tag: str) -> None:
    for suf in ("pdf", "png"):
        fig.savefig(export_utils.figure_output_path(STEM, tag, suf), dpi=300, bbox_inches="tight")


def binned(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Binned scatter and pooled slopes for the two periods. Demeaning is done within each city-year; the slope is
    the pooled within-city gradient of that period."""
    rows: list[dict[str, object]] = []
    slopes: list[dict[str, object]] = []
    g0 = d.dropna(subset=["ln_NO2", EXPO])
    for a, b in (EARLY, LATE):
        g = g0[(g0[TIME] >= a) & (g0[TIME] <= b)]
        x = g[EXPO] - g.groupby("城市年")[EXPO].transform("mean")
        y = g["ln_NO2"] - g.groupby("城市年")["ln_NO2"].transform("mean")
        slope = float((x * y).sum() / (x * x).sum())
        slopes.append({"时期": f"{a}–{b}", "合并城内梯度（去均值 OLS 斜率）": slope, "站-年": len(g),
                       "城市": g[CITY].nunique()})
        q = pd.qcut(x, N_BINS, labels=False, duplicates="drop")
        for k, idx in x.groupby(q).groups.items():
            rows.append({"时期": f"{a}–{b}", "分箱": int(k), "暴露偏离均值": float(x.loc[idx].mean()),
                         "lnNO2偏离均值": float(y.loc[idx].mean()), "站-年": len(idx)})
    return pd.DataFrame(rows), pd.DataFrame(slopes)


def _p90_p10(beta: float, span: float, log_scale: bool) -> float:
    """Convert a slope into the difference between p90 and p10 stations: log quantities become percentages, and
    proportions are multiplied by 100 to give percentage points."""
    return float(100 * np.expm1(beta * span)) if log_scale else float(100 * beta * span)


def signature_table(d: pd.DataFrame) -> pd.DataFrame:
    """Data for figure a (c)(d): the within-city-year slopes of check ① in script 65 (city-clustered SE), converted
    into the difference between p90 and p10 stations."""
    fp = pd.read_excel(OUT / "65_终审意见核验.xlsx", sheet_name="03_建成区轴指纹")
    e = d.dropna(subset=["ln_NO2", EXPO])[EXPO]
    span = float(e.quantile(0.90) - e.quantile(0.10))
    rows: list[dict[str, object]] = []
    for key, _, _ in SIG_ORDER:
        for qty, tag in (("ln 峰谷比", "c"), ("工作日−周末相对差", "d")):
            r = fp[(fp["物种"] == key) & (fp["量"] == qty)].iloc[0]
            b, se, lg = float(r["斜率"]), float(r["SE"]), tag == "c"
            rows.append({"物种": key, "量": qty, "分图": tag, "斜率": b, "SE": se, "p": float(r["p"]),
                         "p90−p10 暴露差": span, "p90−p10 差": _p90_p10(b, span, lg),
                         "下限": _p90_p10(b - 1.96 * se, span, lg), "上限": _p90_p10(b + 1.96 * se, span, lg)})
    return pd.DataFrame(rows)


def fig_a(d: pd.DataFrame, bins: pd.DataFrame, slopes: pd.DataFrame, sig: pd.DataFrame, lang: str) -> None:
    import matplotlib.pyplot as plt

    t = TEXT[lang]
    m52.use_font(lang)
    fig = plt.figure(figsize=(10.4, 6.9), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.6, 1])
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1.35])
    bot = outer[1].subgridspec(1, 2)
    ax1, ax2 = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    e = d.dropna(subset=["ln_NO2", EXPO])[EXPO]
    p10, p90 = float(e.quantile(0.10)), float(e.quantile(0.90))
    counts, _, _ = ax1.hist(e, bins=40, color="0.62", edgecolor="white", linewidth=0.4)
    ax1.set_ylim(0, float(np.max(counts)) * 1.15)   # headroom for the p10/p90 labels, clear of the tallest bar
    for v, lab, ha in ((p10, "p10", "left"), (p90, "p90", "right")):
        ax1.axvline(v, color=RED, lw=1.4, ls="--")
        ax1.text(v, ax1.get_ylim()[1] * 0.97, f" {lab} = {v:.4f} ", color=RED, fontsize=8.5, va="top", ha=ha)
    ax1.set_xlabel(t["a1_x"])
    ax1.set_ylabel(t["a1_y"])
    ax1.set_title(panel("a", t["a1_title"]))
    for (a, b), col, mk in ((EARLY, "0.45", "o"), (LATE, RED, "s")):
        lab = f"{a}–{b}"
        bb = bins[bins["时期"] == lab]
        s = slopes[slopes["时期"] == lab].iloc[0]
        ax2.plot(bb["暴露偏离均值"], bb["lnNO2偏离均值"], mk, color=col, ms=6)
        xs = np.linspace(bins["暴露偏离均值"].min(), bins["暴露偏离均值"].max(), 20)
        ax2.plot(xs, s["合并城内梯度（去均值 OLS 斜率）"] * xs, color=col, lw=2,
                 label=m52.fmt(t["a2_leg"], a=a, b=b, slope=f"{s['合并城内梯度（去均值 OLS 斜率）']:.3f}",
                               n=int(s["站-年"])))
    ax2.axhline(0, color="0.6", lw=0.8, ls=":")
    ax2.axvline(0, color="0.6", lw=0.8, ls=":")
    ax2.set_xlabel(t["a2_x"])
    ax2.set_ylabel(t["a2_y"])
    ax2.set_title(panel("b", t["a2_title"]))
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ys = np.arange(len(SIG_ORDER))[::-1]
    for j, tag in enumerate("cd"):
        axs = fig.add_subplot(bot[0, j])
        s = sig[sig["分图"] == tag].set_index("物种")
        for yv, (key, _, col) in zip(ys, SIG_ORDER, strict=True):
            r = s.loc[key]
            axs.errorbar(r["p90−p10 差"], yv, xerr=[[r["p90−p10 差"] - r["下限"]], [r["上限"] - r["p90−p10 差"]]],
                       fmt="o", color=col, ms=6, capsize=3, lw=1.6)
        axs.axvline(0, color="0.6", lw=0.8, ls=":")
        axs.set_yticks(ys)
        axs.set_yticklabels([lab for _, lab, _ in SIG_ORDER])
        axs.set_ylim(-0.6, len(SIG_ORDER) - 0.4)
        axs.set_xlabel(t[f"{tag}_x"])
        axs.set_title(panel(tag, t[f"{tag}_title"]))
    save(fig, "a" if lang == "zh" else "a_en")
    plt.close(fig)


def robustness_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Data for the path and forest plots: the NO₂ rows of scripts 62 and 63, ordered as in SPECS."""
    wb = {"62": OUT / "62_梯度气象核验.xlsx", "63": OUT / "63_梯度平衡面板核验.xlsx"}
    path_rows: list[dict[str, object]] = []
    forest_rows: list[dict[str, object]] = []
    for src, col, val, key, _ in SPECS:
        nar = pd.read_excel(wb[src], sheet_name="01_收窄幅度")
        seg = pd.read_excel(wb[src], sheet_name="02_分段速率")
        pth = pd.read_excel(wb[src], sheet_name="04_逐年梯度" if src == "62" else "03_逐年梯度")
        r = nar[(nar["序列"] == "NO2") & (nar[col] == val)].iloc[0]
        forest_rows.append({"行": key, "量": "f1", "点估计": r["2015→2024 相对变化%"],
                            "下限": r["bootstrap 95% 下限"], "上限": r["bootstrap 95% 上限"]})
        for seg_lab, qk in (("2017–2020", "f2"), ("2020–2024", "f3")):
            s = seg[(seg["序列"] == "NO2") & (seg[col] == val) & (seg["分段"] == seg_lab)].iloc[0]
            forest_rows.append({"行": key, "量": qk, "点估计": s["平均年变化（梯度单位/年）"],
                                "下限": s["bootstrap 95% 下限"], "上限": s["bootstrap 95% 上限"]})
        p = pth[(pth["序列"] == "NO2") & (pth[col] == val)]
        for _, q in p.iterrows():
            path_rows.append({"行": key, "年份": int(q["年份"]), "城内梯度": q["城内梯度"], "SE": q["城市聚类SE"]})
    return pd.DataFrame(path_rows), pd.DataFrame(forest_rows)


def fig_b(path: pd.DataFrame, forest: pd.DataFrame, lang: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    t = TEXT[lang]
    m52.use_font(lang)
    fig = plt.figure(figsize=(10.4, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.42, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    for _, _, _, key, ls in SPECS:
        if ls is None:
            continue
        p = path[path["行"] == key]
        ax.plot(p["年份"], p["城内梯度"], ls, color=SPEC_COLOR[key], lw=2 if key == "spec_all" else 1.6,
                marker="o" if key == "spec_all" else None, ms=4, label=t[key])
        if key == "spec_all":
            ax.fill_between(p["年份"], p["城内梯度"] - 1.96 * p["SE"], p["城内梯度"] + 1.96 * p["SE"],
                            color=RED, alpha=0.12, lw=0)
    ax.axvspan(2018, 2020, color=BAND, alpha=0.45, zorder=0)
    ax.text(2019, 0.985, t["band"], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9,
            color="0.35")
    ax.set_xticks(range(2015, 2025))
    ax.set_xlabel(t["year"])
    ax.set_ylabel(t["b_y"])
    ax.set_title(panel("a", t["b_title"]))
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=2)
    keys = [s[3] for s in SPECS]
    ys = np.arange(len(keys))[::-1]
    for j, qk in enumerate(("f1", "f2", "f3")):
        a = fig.add_subplot(gs[1, j])
        f = forest[forest["量"] == qk].set_index("行").loc[keys]
        for yv, key in zip(ys, keys, strict=True):
            r = f.loc[key]
            a.errorbar(r["点估计"], yv, xerr=[[r["点估计"] - r["下限"]], [r["上限"] - r["点估计"]]],
                       fmt="o", color=SPEC_COLOR[key], ms=5, capsize=3, lw=1.4)
        a.axvline(0, color="0.6", lw=0.8, ls=":")
        a.set_yticks(ys)
        a.set_yticklabels([t[k] for k in keys] if j == 0 else [], fontsize=8.5)
        a.set_ylim(-0.6, len(keys) - 0.4)
        a.xaxis.set_major_locator(MaxNLocator(4))
        a.set_title(panel("bcd"[j], t[qk]), fontsize=10)
    save(fig, "b" if lang == "zh" else "b_en")
    plt.close(fig)


def timing_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Cumulative completion (script 51) with its intervals (script 61), segment rates (script 61), and national NEV
    (the series of script 52)."""
    cum = pd.read_excel(OUT / "51_梯度路径与可检测性.xlsx", sheet_name="03_时序核对")
    ci = pd.read_excel(OUT / "61_梯度分段速率检验.xlsx", sheet_name="05_收窄完成度")
    cum = cum.merge(ci[["截至年份", "95% 区间下限", "95% 区间上限"]].rename(columns={"截至年份": "年份"}),
                    on="年份", how="left")
    seg = pd.read_excel(OUT / "61_梯度分段速率检验.xlsx", sheet_name="02_分段速率")
    seg = seg[(seg["序列"] == "NO2") & (seg["口径"] == "单位/年")
              & seg["分段"].isin(["2015–2017", "2017–2020", "2020–2024"])].copy()
    nev = m52.nev_series()
    # Fleet cumulative completion on the log scale used in the regressions (2017→2023 = 100), drawn alongside the
    # level scale
    ln_nev = np.log(nev)
    cum["NEV对数增量占比%"] = [100 * float((ln_nev.loc[y] - ln_nev.loc[2017]) / (ln_nev.loc[2023] - ln_nev.loc[2017]))
                          for y in cum["年份"]]
    total = float(nev.loc[2024] - nev.loc[2017])
    inc = {"2017–2020": (2017, 2020), "2020–2024": (2020, 2024)}
    seg["NEV年均增量_万辆"] = [float((nev.loc[inc[s][1]] - nev.loc[inc[s][0]]) / (inc[s][1] - inc[s][0]))
                           if s in inc else np.nan for s in seg["分段"]]
    seg["占2017_2024增量%"] = [100 * float(nev.loc[inc[s][1]] - nev.loc[inc[s][0]]) / total if s in inc else np.nan
                           for s in seg["分段"]]
    return cum, seg, nev


def fig_c(cum: pd.DataFrame, seg: pd.DataFrame, lang: str) -> None:
    import matplotlib.pyplot as plt

    t = TEXT[lang]
    m52.use_font(lang)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    yr = cum["年份"]
    g, v = cum["梯度收窄完成度%"], cum["NEV累计增量占比%"]
    lo = (g - cum["95% 区间下限"]).where(cum["95% 区间下限"].notna(), 0)
    hi = (cum["95% 区间上限"] - g).where(cum["95% 区间上限"].notna(), 0)
    ax1.fill_between(yr, v, g, where=g >= v, color=RED, alpha=0.10, lw=0)
    ax1.errorbar(yr, g, yerr=[lo, hi], fmt="o-", color=RED, lw=2, ms=6, capsize=3, label=t["c1_grad"])
    ax1.plot(yr, v, "s--", color=GREY_NEV, lw=1.6, ms=5, label=t["c1_nev"])
    ax1.plot(yr, cum["NEV对数增量占比%"], "^:", color="0.55", lw=1.5, ms=5, label=t["c1_nevlog"])
    ax1.axvspan(2018, 2020, color=BAND, alpha=0.45, zorder=0)
    r20 = cum[cum["年份"] == 2020].iloc[0]
    ax1.annotate(m52.fmt(t["c1_note"], g=round(float(r20["梯度收窄完成度%"])), v=round(float(r20["NEV累计增量占比%"])),
                         w=round(float(r20["NEV对数增量占比%"]))),
                 xy=(2020, float(r20["NEV累计增量占比%"])), xytext=(2020.35, 2),
                 fontsize=9, color="0.2", arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.8})
    ax1.set_ylim(-12, 135)
    ax1.set_xticks(list(yr))
    ax1.set_xlabel(t["year"])
    ax1.set_ylabel(t["c1_y"])
    ax1.set_title(panel("a", t["c1_title"]))
    h, lab = ax1.get_legend_handles_labels()
    order = [lab.index(t[k]) for k in ("c1_grad", "c1_nev", "c1_nevlog")]
    # Legend below the axes: in the upper-left corner its first row would overlap the top of the 2020 error bar
    ax1.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, -0.16),
               frameon=False, fontsize=9)
    xs = np.arange(len(seg))
    ax2b = ax2.twinx()
    ax2b.bar(xs, seg["NEV年均增量_万辆"].fillna(0), width=0.5, color="0.8", zorder=0)
    ax2b.set_ylabel(t["c2_nev"])
    ticks = [f"{r['分段']}\n" + (t["c2_na"] if np.isnan(r["NEV年均增量_万辆"])
                                  else m52.fmt(t["c2_share"], p=f"{r['占2017_2024增量%']:.0f}"))
             for _, r in seg.iterrows()]
    ax2b.set_ylim(0, 820)
    ax2.set_zorder(ax2b.get_zorder() + 1)
    ax2.patch.set_visible(False)
    ax2.errorbar(xs, seg["点估计"], yerr=[seg["点估计"] - seg["95% 区间下限"], seg["95% 区间上限"] - seg["点估计"]],
                 fmt="o", color=RED, ms=7, capsize=4, lw=1.8)
    ax2.axhline(0, color="0.6", lw=0.8, ls=":")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(ticks, fontsize=8.5)
    ax2.set_ylabel(t["c2_y"])
    ax2.set_title(panel("b", t["c2_title"]))
    fig.tight_layout()
    save(fig, "c" if lang == "zh" else "c_en")
    plt.close(fig)


def graphical_abstract(d: pd.DataFrame, path51: pd.DataFrame, cum: pd.DataFrame) -> None:
    """13 cm × 5.2 cm. Three blocks: ten-year gradient path → comparison of cumulative completion → conclusions.
    Every number on the figure is computed from the data."""
    import textwrap

    import matplotlib.pyplot as plt

    m52.use_font("en")
    plt.rcParams["font.size"] = 7
    fig = plt.figure(figsize=(13 / 2.54, 5.2 / 2.54))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], wspace=0.38, left=0.065, right=0.6, bottom=0.14,
                          top=0.76)
    fig.suptitle("Within-city " + NO2 + " exposure gradients narrowed ahead of electric vehicle growth",
                 fontsize=7.4, y=0.97, fontweight="bold")
    ax1 = fig.add_subplot(gs[0, 0])
    p = path51[path51["序列"] == "NO2"]
    base = float(p["城内梯度"].iloc[0])
    change = float(100 * p["城内梯度"].iloc[-1] / base) - 100
    ax1.plot(p["年份"], 100 * p["城内梯度"] / base, "o-", color=RED, lw=1.4, ms=2.5)
    ax1.axvspan(2018, 2020, color=BAND, alpha=0.5, zorder=0)
    # Band label inside the band, below the red line: at the top of the band it would overlap the 2017–2018 points,
    # and at the bottom it would crowd the size of the change in the lower-right corner
    ax1.text(2019, 55, "Blue Sky\ncampaign", ha="center", va="bottom", fontsize=5.2, color="0.35", linespacing=1.05)
    ax1.text(2023.9, 100 + change - 9, f"{change:.1f}%".replace("-", "\u2212"), ha="right", va="top", fontsize=7,
             color=RED)
    ax1.set_ylim(35, 108)
    ax1.set_xticks([2015, 2018, 2021, 2024])
    ax1.set_title("NO" + "$_2$" + " gradient, 2015 = 100", fontsize=6.8)
    ax1.tick_params(labelsize=6)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(cum["年份"], cum["梯度收窄完成度%"], "o-", color=RED, lw=1.4, ms=2.5)
    ax2.plot(cum["年份"], cum["NEV累计增量占比%"], "s--", color=GREY_NEV, lw=1.1, ms=2.2)
    r20 = cum[cum["年份"] == 2020].iloc[0]
    g20, v20 = float(r20["梯度收窄完成度%"]), float(r20["NEV累计增量占比%"])
    w20 = float(r20["NEV对数增量占比%"])
    ax2.annotate("", xy=(2020, g20 - 2), xytext=(2020, v20 + 2),
                 arrowprops={"arrowstyle": "<->", "color": "0.3", "lw": 0.7})
    # Annotation to the right of the arrow: on the left it would overlap the red line of 2018–2019
    ax2.text(2020.15, (g20 + v20) / 2 + 12, f"{g20:.0f}% vs\n{v20:.0f}%\nin 2020", fontsize=5.8, va="center",
             ha="left", color="0.2")
    ax2.text(2017.1, 88, "gradient\nnarrowing", fontsize=5.8, color=RED, va="top")
    ax2.text(2022.9, 20, "EV fleet\ngrowth", fontsize=5.8, color=GREY_NEV, ha="right", va="top")
    ax2.set_ylim(-5, 112)
    ax2.set_xticks([2017, 2019, 2021, 2023])
    ax2.set_title("Share of 2017–2023 change (%)", fontsize=6.8)
    ax2.tick_params(labelsize=6)
    g = d.dropna(subset=["ln_NO2", EXPO])
    # The last bullet states what the findings imply: the NO₂ excess at built-up stations narrowed mainly during the
    # Blue Sky Protection Campaign, and differences within cities must be tracked directly
    lines = [f"{g[STA].nunique():,} stations in {g[CITY].nunique()} cities: gradient {change:.1f}% in 2015–2024"
             .replace("-", "\u2212").replace("2015\u22122024", "2015–2024"),
             f"By 2020 the EV fleet had made {v20:.0f}% of its 2017–2023 growth ({w20:.0f}% in logs)",
             "SO\u2082 gradients narrowed faster, so the narrowing was not traffic-specific",
             "EV × exposure association not detected once an exposure trend is allowed",
             "The ambient NO\u2082 excess at built-up sites shrank mainly in the Blue Sky years; "
             "track contrasts inside cities directly"]
    yv = 0.80
    for line in lines:
        txt = textwrap.fill(line, width=40)
        fig.text(0.635, yv, "\u2022 " + txt.replace("\n", "\n   "), fontsize=5.8, va="top", ha="left",
                 linespacing=1.15)
        yv -= 0.03 + 0.052 * (txt.count("\n") + 1)
    for suf, dpi in (("pdf", 300), ("png", 300), ("tiff", 600)):
        fig.savefig(export_utils.figure_output_path(STEM, "GA_en", suf), dpi=dpi)
    plt.close(fig)
    plt.rcParams["font.size"] = 10


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    logger, log_path = export_utils.configure_file_logger("64_manuscript_added_figures")
    d = m51.load()
    bins, slopes = binned(d)
    path, forest = robustness_tables()
    cum, seg, nev = timing_tables()
    sig = signature_table(d)
    path51 = pd.read_excel(OUT / "51_梯度路径与可检测性.xlsx", sheet_name="02_逐年梯度路径")
    path51 = path51[path51["样本"] == "全部"]
    for lang in ("zh", "en"):
        fig_a(d, bins, slopes, sig, lang)
        fig_b(path, forest, lang)
        fig_c(cum, seg, lang)
    graphical_abstract(d, path51, cum)
    e = d.dropna(subset=["ln_NO2", EXPO])[EXPO]
    logger.info("图 a：暴露 p10 %.4f、p90 %.4f（%s 站-年）", e.quantile(0.1), e.quantile(0.9), len(e))
    for _, r in slopes.iterrows():
        logger.info("图 a：%s 合并城内梯度 %.4f（%s 站-年，%s 城）", r["时期"], r["合并城内梯度（去均值 OLS 斜率）"],
                    r["站-年"], r["城市"])
    for _, r in seg.iterrows():
        logger.info("图 c：%s 平均年变化 %.4f（95%% %.4f, %.4f），NEV 年均增量 %.1f 万辆，占 2017–2024 增量 %.1f%%",
                    r["分段"], r["点估计"], r["95% 区间下限"], r["95% 区间上限"], r["NEV年均增量_万辆"],
                    r["占2017_2024增量%"])
    logger.info("全国 NEV（万辆）：%s", nev.round(0).to_dict())
    out = export_utils.write_excel_workbook(STEM, [
        ("图a_分箱散点", bins), ("图a_分期斜率", slopes), ("图a_时间特征", sig),
        ("图b_逐年梯度", path), ("图b_森林图", forest),
        ("图c_累计完成度", cum), ("图c_分段速率", seg),
    ])
    print(f"✓ 稿件新增图完成（中英各三张 + 图形摘要）\n  数据: {out}\n  日志: {log_path}")


if __name__ == "__main__":
    main()
