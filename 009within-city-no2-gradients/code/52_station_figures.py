"""
52_station_figures.py: figures of the station-level results, one set in Chinese and one in English (figure a is
Figure 2 of the main text; figures b and c are Figures S1 and S2 of the supplement)

Figure a  Ten-year path of the within-city pollution gradients vs the national NEV stock
          — the central figure of the paper. Four gradient curves (NO₂ / NO₂ peak-to-trough ratio / SO₂ / PM2.5) share
          one frame with the NEV stock, showing directly that "the narrowing is fastest in 2017–2020, while 89% of the
          NEV increase falls in 2020–2024".
          In 2015–2016 NEVs were less than 1% of the vehicle fleet, which makes these years a clean pre-period baseline.

Figure b  Exposure-specific year effects (year dummies × exposure, 2015 as the base year)
          — estimated in this script on the 2015–2024 sample with station and city-by-year fixed effects, SEs
          clustered by city, and no NEV term. Shows each year's change in the gradient relative to the base year with
          its 95% confidence interval as an error bar, and marks the NEV stock alongside, to see whether the two
          curves have the same shape.

Figure c  Cross-section: each city's gradient change vs the log growth of that city's NEV stock
          — if electrification drove the narrowing, the correlation should be significantly negative; the observed
          correlation is positive.

**Bilingual**: every figure is drawn once with Chinese and once with English text. The data are computed only once
and the plotting runs twice over the `TEXT` dictionary, so that the two sets cannot drift apart through separate
recomputation.
Chinese figures use a Chinese font with `axes.unicode_minus=False` (the Chinese fonts lack the U+2212 glyph, so every
minus sign is an ASCII hyphen); English figures switch to DejaVu Sans and turn on the true minus sign.

Usage:
    python code/52_station_figures.py

Output files:
    outputs/52_站点结果-图a.pdf (station results, figure a) / -图b.pdf / -图c.pdf
        Chinese version; one .png of the same name for each
    outputs/52_站点结果-图a_en.pdf / -图b_en.pdf / -图c_en.pdf
        English version; likewise
    outputs/52_站点结果.xlsx (station results)
        plotted data; sheets 01_图a_梯度路径, 02_图b_年度效应, 03_图c_横截面
    logs/52_station_figures.log
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location("export_utils", Path(__file__).with_name("01_export_utils.py"))
assert _SPEC
assert _SPEC.loader
export_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_utils)

STA, CITY, TIME = "站点编号", "所属城市", "年份"
EXPO = "建成区占比_500m"
MIN_DAYS = 274
PANEL = "站点年面板-2015_2024.csv"
STEM = "52_站点结果"
# Legends write chemical formulas in mathtext: the project's Chinese font has no subscript glyphs, so a literal ₂
# would render as an empty box
NO2, SO2 = r"$\mathrm{NO_2}$", r"$\mathrm{SO_2}$"
PM25 = r"$\mathrm{PM_{2.5}}$"
SERIES = [("NO2", "ln_NO2", "#C0392B"), ("NO2峰谷比", "ln_峰谷比", "#E67E22"),
          ("SO2", "ln_SO2", "#2E86C1"), ("PM25", "ln_PM25", "#7D3C98")]

# ── The single source of all text on the figures. Change wording only here, so the two sets cannot drift apart ──
TEXT = {
    "zh": {
        "NO2": NO2, "NO2峰谷比": NO2 + " 峰谷比", "SO2": SO2, "PM25": PM25,
        "nev_legend": "全国 NEV 保有量",
        "nev_ylabel": "全国新能源汽车保有量（万辆，公安部口径）",
        "year": "年份",
        "a_ylabel": "城内空间梯度（以 {y0} 年为 100）",
        "a_title": "城内污染梯度的收窄路径与新能源汽车保有量",
        "a_band": "蓝天保卫战三年行动",
        "a_fast": "梯度年均 -0.064",
        "a_slow": "年均 -0.024\nNEV 增量的 89% 在此期间",
        "b_legend": "暴露 × 年（相对基期）",
        "b_ylabel": "ln " + NO2 + " 的暴露梯度相对 {y0} 年的变化",
        "b_title": "城内 " + NO2 + " 梯度的年度变化（站点固定效应 + 城市×年固定效应）",
        "c_legend": "n={n}，r={r}（负相关才支持电动化解释）",
        "c_xlabel": "城市 NEV 保有量的对数增幅（{y0}→{y1}）",
        "c_ylabel": "城内 " + NO2 + " 梯度的变化（{y1} - {y0}）",
        "c_title": "各城市的梯度收窄幅度与电动化速度",
    },
    "en": {
        "NO2": NO2, "NO2峰谷比": NO2 + " peak-to-trough ratio", "SO2": SO2, "PM25": PM25,
        "nev_legend": "National NEV stock",
        "nev_ylabel": "National NEV stock ($10^4$ vehicles, MPS)",
        "year": "Year",
        "a_ylabel": "Within-city spatial gradient ({y0} = 100)",
        "a_title": "Narrowing of within-city pollution gradients and the national NEV stock",
        "a_band": "Blue Sky Protection Campaign",
        "a_fast": "−0.064 per year",
        "a_slow": "−0.024 per year\n89% of NEV growth falls here",
        "b_legend": "Exposure × year (relative to base year)",
        "b_ylabel": "Change in the ln " + NO2 + " exposure gradient relative to {y0}",
        "b_title": "Annual change in the within-city " + NO2
                   + " gradient (station and city-by-year fixed effects)",
        "c_legend": "n={n}, r={r} (only a negative slope supports electrification)",
        "c_xlabel": "Log growth in city NEV stock ({y0}→{y1})",
        "c_ylabel": "Change in the within-city " + NO2 + " gradient ({y1} − {y0})",
        "c_title": "City-level gradient change versus the pace of electrification",
    },
}


def fmt(s: str, **kw: object) -> str:
    """Literal replacement by key. str.format cannot be used: it would treat the mathtext in the labels
    ($\\mathrm{NO_2}$) as a placeholder and raise KeyError: 'NO_2'."""
    for k, v in kw.items():
        s = s.replace("{" + k + "}", str(v))
    return s


def load() -> pd.DataFrame:
    p = pd.read_csv(export_utils.DATA_DIR / PANEL)
    p.columns = [c.lstrip("﻿") for c in p.columns]
    p = p[p["NO2_有效日"] >= MIN_DAYS].copy()
    p["城市年"] = p[CITY].astype(str) + "_" + p[TIME].astype(str)
    # Each pollutant is filtered by **its own** valid-day threshold (the same rule as apply_threshold in script 47).
    # Filtering by NO₂ valid days alone would let in station-years with too few PM2.5 / SO₂ valid days.
    for src, dst, dcol in [("NO2_年均", "ln_NO2", "NO2_有效日"),
                           ("SO2_年均", "ln_SO2", "SO2_有效日"),
                           ("PM2.5_年均", "ln_PM25", "PM2.5_有效日"),
                           ("NO2_峰谷比", "ln_峰谷比", "NO2_有效日")]:
        v = p[src].where(p[src] > 0)
        if dcol in p.columns:
            v = v.where(p[dcol] >= MIN_DAYS)
        p[dst] = np.log(v)
    n = p.dropna(subset=["ln_NO2"]).groupby("城市年")[STA].nunique()
    p["城内站数"] = p["城市年"].map(n)
    return p[p["城内站数"] >= 2].copy()


def grad(df: pd.DataFrame, ycol: str) -> tuple[float, float]:
    g = df.dropna(subset=[ycol, EXPO])
    if g[CITY].nunique() < 10:
        return np.nan, np.nan
    z = g[[ycol, EXPO]].astype(float)
    z = z - z.groupby(g[CITY].values).transform("mean")
    X, y = z[[EXPO]].values, z[ycol].values
    XtX_i = np.linalg.pinv(X.T @ X)
    b = XtX_i @ X.T @ y
    u = y - X @ b
    gid = pd.factorize(g[CITY])[0]
    meat = sum(np.outer(X[gid == k].T @ u[gid == k], X[gid == k].T @ u[gid == k])
               for k in np.unique(gid))
    G = len(np.unique(gid))
    V = XtX_i @ np.atleast_2d(meat) @ XtX_i * (G / (G - 1))
    return float(b[0]), float(np.sqrt(V[0, 0]))


# 2024 national NEV stock (10^4 vehicles): 3140, as published in the Ministry of Public Security's annual motor
# vehicle statistics bulletin.
# No provincial 2024 values were obtained, so it is used only as the last point of the national series and enters
# no regression.
NEV_2024_NATIONAL = 3140.0


def nev_series() -> pd.Series:
    """National NEV stock (10^4 vehicles): 2017–2023 is the sum over the 31 provinces of the provincial series on the
    Ministry of Public Security registration basis; 2024 is the ministry's national value."""
    prov = pd.read_csv(export_utils.DATA_DIR / "省级新能源汽车保有量-公安部口径-2017_2023.csv")
    prov.columns = [c.lstrip("﻿") for c in prov.columns]
    s = prov.groupby(TIME)["新能源汽车保有量_辆"].sum() / 1e4
    s.loc[2024] = NEV_2024_NATIONAL
    return s.sort_index()


def use_font(lang: str) -> None:
    """Chinese figures need a Chinese font with the unicode minus disabled; English figures switch to DejaVu and turn
    on the true minus sign."""
    import matplotlib.pyplot as plt

    if lang == "zh":
        export_utils.setup_chinese_font()
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
        plt.rcParams["axes.unicode_minus"] = True


def fig_a(gp: pd.DataFrame, nev: pd.Series, years: list[int], lang: str) -> None:
    import matplotlib.pyplot as plt

    t = TEXT[lang]
    use_font(lang)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for key, _, col in SERIES:
        g = gp[gp["序列"] == key].dropna(subset=["梯度"])
        if g.empty:
            continue
        base = g["梯度"].iloc[0]
        # Point estimates only: after normalising to 100 in the base year, the confidence bands are inflated by
        # the base year's own SE until they cover the lines, and the shape can no longer be seen. The annual SEs are
        # in the year-by-year gradient path table of script 51 (Table S3); figure b gives the year-effect intervals.
        ax.plot(g["年份"], 100 * g["梯度"] / base, "o-", color=col, lw=2, ms=5, label=t[key])
    ax.axhline(100, color="0.6", lw=0.8, ls=":")
    ax.axvspan(2018, 2020, color="0.85", alpha=0.45, zorder=0)
    ax.set_ylim(0, 145)
    ax.text(2019, 142, t["a_band"], ha="center", va="top", fontsize=9, color="0.35")
    # Two-period contrast; each arrow spans the interval of the rate written on it (the segments 2017–2020 and
    # 2020–2024 of the main text and Table S3): the sharp narrowing happens in 2017–2020, when NEVs were still very
    # few, while 89% of the NEV increase falls in 2020–2024
    ax.annotate("", xy=(2020.0, 30), xytext=(2017.0, 30),
                arrowprops={"arrowstyle": "<->", "color": "#C0392B", "lw": 1.3})
    ax.text(2018.5, 33, t["a_fast"], ha="center", fontsize=9, color="#C0392B")
    ax.annotate("", xy=(2024.0, 30), xytext=(2020.0, 30),
                arrowprops={"arrowstyle": "<->", "color": "0.35", "lw": 1.3})
    # Two lines, shifted right: on a single line the text spans 2021 and sits on the 2021 point of the NEV curve
    ax.text(2023.0, 32, t["a_slow"], ha="center", va="bottom", fontsize=9, color="0.35")
    ax.set_xlabel(t["year"])
    ax.set_ylabel(fmt(t["a_ylabel"], y0=years[0]))
    ax.set_title(t["a_title"])

    ax2 = ax.twinx()
    ax2.plot(nev.index, nev.values, "s--", color="0.25", lw=1.6, ms=4, label=t["nev_legend"])
    ax2.set_ylabel(t["nev_ylabel"])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Legend below the axes: inside the axes it would be crossed by the NEV curve
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False,
              fontsize=9, ncol=5)
    fig.tight_layout()
    tag = "a" if lang == "zh" else "a_en"
    for suf in ("pdf", "png"):
        fig.savefig(export_utils.figure_output_path(STEM, tag, suf), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_b(ev: pd.DataFrame, nev: pd.Series, years: list[int], lang: str) -> None:
    import matplotlib.pyplot as plt

    t = TEXT[lang]
    use_font(lang)
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.errorbar(ev["年份"], ev["系数"], yerr=1.96 * ev["SE"], fmt="o-", color="#C0392B",
                lw=2, ms=6, capsize=4, label=t["b_legend"])
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.axvspan(2018, 2020, color="0.85", alpha=0.45, zorder=0)
    ax.set_xlabel(t["year"])
    ax.set_ylabel(fmt(t["b_ylabel"], y0=years[0]))
    ax.set_title(t["b_title"])
    ax2 = ax.twinx()
    ax2.plot(nev.index, nev.values, "s--", color="0.25", lw=1.6, ms=4, label=t["nev_legend"])
    ax2.set_ylabel(t["nev_ylabel"])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False,
              fontsize=9, ncol=2)
    fig.tight_layout()
    tag = "b" if lang == "zh" else "b_en"
    for suf in ("pdf", "png"):
        fig.savefig(export_utils.figure_output_path(STEM, tag, suf), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_c(m: pd.DataFrame, ny0: int, ny1: int, lang: str) -> None:
    import matplotlib.pyplot as plt

    t = TEXT[lang]
    use_font(lang)
    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.scatter(m["NEV对数增幅"], m["梯度变化"], s=14 + 2.2 * m["站点数"], alpha=0.55,
               color="#2E86C1", edgecolor="none")
    if len(m) >= 10:
        k, b0 = np.polyfit(m["NEV对数增幅"], m["梯度变化"], 1)
        xs = np.linspace(m["NEV对数增幅"].min(), m["NEV对数增幅"].max(), 50)
        r = float(np.corrcoef(m["NEV对数增幅"], m["梯度变化"])[0, 1])
        ax.plot(xs, k * xs + b0, color="#C0392B", lw=2,
                label=fmt(t["c_legend"], n=len(m), r=f"{r:+.3f}"))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=9)
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(fmt(t["c_xlabel"], y0=ny0, y1=ny1))
    ax.set_ylabel(fmt(t["c_ylabel"], y0=ny0, y1=ny1))
    ax.set_title(t["c_title"])
    fig.tight_layout()
    tag = "c" if lang == "zh" else "c_en"
    for suf in ("pdf", "png"):
        fig.savefig(export_utils.figure_output_path(STEM, tag, suf), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    logger, log_path = export_utils.configure_file_logger("52_station_figures")
    d = load()
    nev = nev_series()
    years = sorted(d[TIME].unique())
    logger.info("面板 %s 站-年，%s–%s，站点 %s，城市 %s",
                len(d), years[0], years[-1], d[STA].nunique(), d[CITY].nunique())

    # ── Data for figure a: within-city gradient by year and by series ──
    rows = []
    for key, ycol, _ in SERIES:
        if ycol not in d.columns:
            continue
        for yr in years:
            b, se = grad(d[d[TIME] == yr], ycol)
            rows.append({"序列": key, "年份": yr, "梯度": b, "SE": se})
    gp = pd.DataFrame(rows)

    # ── Data for figure b: year dummies × exposure, station FE + city-by-year FE, SEs clustered by city ──
    dd = d.dropna(subset=["ln_NO2", EXPO]).reset_index(drop=True)
    dd["Expo_c"] = dd[EXPO] - dd[EXPO].mean()
    ycols = []
    for yv in years[1:]:
        c = f"y{yv}"
        dd[c] = (dd[TIME] == yv).astype(float) * dd["Expo_c"]
        ycols.append(c)
    z = dd[["ln_NO2", *ycols]].astype(float)
    a_, b_ = dd[STA].values, dd["城市年"].values
    for _ in range(600):
        prev = z.values.copy()
        z = z - z.groupby(a_).transform("mean")
        z = z - z.groupby(b_).transform("mean")
        if np.max(np.abs(z.values - prev)) < 1e-11:
            break
    X, y = z[ycols].values, z["ln_NO2"].values
    XtX_i = np.linalg.pinv(X.T @ X)
    bb = XtX_i @ X.T @ y
    u = y - X @ bb
    gid = pd.factorize(dd[CITY])[0]
    meat = sum(np.outer(X[gid == k].T @ u[gid == k], X[gid == k].T @ u[gid == k])
               for k in np.unique(gid))
    G, N, K = len(np.unique(gid)), len(y), X.shape[1]
    V = XtX_i @ meat @ XtX_i * (G / (G - 1)) * ((N - 1) / (N - K))
    se = np.sqrt(np.diag(V))
    ev = pd.DataFrame({"年份": years, "系数": np.r_[0.0, bb], "SE": np.r_[0.0, se]})

    # ── Data for figure c: cross-section of cities ──
    c_ = pd.read_csv(export_utils.DATA_DIR / "城市面板-2017_2023.csv")
    c_.columns = [x.lstrip("﻿") for x in c_.columns]
    cn = c_[c_["NEV保有量_辆"] > 0].pivot_table(index="城市", columns=TIME, values="NEV保有量_辆")
    # The window is the years covered by the **city-level** NEV data (2017–2023). The national series extends to
    # 2024, but no provincial or city 2024 values were obtained, so the cross-section must use the two end years
    # that actually have city values.
    ny0, ny1 = int(min(cn.columns)), int(max(cn.columns))
    gr = np.log(cn[ny1] / cn[ny0]).rename("NEV对数增幅").reset_index()
    # The gradient change must use the same ny0→ny1 window as the NEV growth; otherwise each end would add a stretch
    # of change unrelated to NEV
    rows = []
    for city, g in d[(d[TIME] >= ny0) & (d[TIME] <= ny1)].dropna(
            subset=["ln_NO2", EXPO]).groupby(CITY):
        yy = sorted(g[TIME].unique())
        if len(yy) < 5 or yy[0] != ny0 or yy[-1] != ny1:
            continue
        bs = {}
        for yr in (yy[0], yy[-1]):
            gg = g[g[TIME] == yr]
            if gg[STA].nunique() < 3 or gg[EXPO].std() < 1e-6:
                continue
            A = np.column_stack([np.ones(len(gg)), gg[EXPO].to_numpy(float)])
            bs[yr] = float(np.linalg.lstsq(A, gg["ln_NO2"].to_numpy(float), rcond=None)[0][1])
        if len(bs) == 2:
            rows.append({"城市": city, "梯度变化": bs[yy[-1]] - bs[yy[0]], "站点数": g[STA].nunique()})
    cs = pd.DataFrame(rows)
    cs["城市名"] = cs["城市"].astype(str).str.replace(r"(市|地区|自治州|盟)$", "", regex=True)
    m = cs.merge(gr.rename(columns={"城市": "城市名"}), on="城市名", how="inner").dropna(
        subset=["梯度变化", "NEV对数增幅"])

    # ── Two sets of figures from the same data ──
    for lang in ("zh", "en"):
        fig_a(gp, nev, years, lang)
        fig_b(ev, nev, years, lang)
        fig_c(m, ny0, ny1, lang)

    export_utils.write_excel_workbook(STEM, [("图a_梯度路径", gp), ("图b_年度效应", ev),
                                             ("图c_横截面", m)])
    logger.info("六图完成（中英各三）；图 b 年度效应末年系数 %.4f（SE %.4f）",
                ev["系数"].iloc[-1], ev["SE"].iloc[-1])
    print("✓ 站点结果三图完成，中英各一套")
    print(gp.pivot_table(index="年份", columns="序列", values="梯度").round(3).to_string())
    print(f"\n  中文图: {export_utils.OUTPUT_DIR}/{STEM}-图a/b/c.pdf（同名 png 各一份）")
    print(f"  英文图: {export_utils.OUTPUT_DIR}/{STEM}-图a_en/b_en/c_en.pdf（同名 png 各一份）")
    print(f"  日志: {log_path}")


if __name__ == "__main__":
    main()
