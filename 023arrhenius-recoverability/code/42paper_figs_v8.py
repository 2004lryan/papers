"""
42paper_figs_v8.py: v8 主稿出版级图（nature-figure house style，从真实 validated source data 生成；中英双语）

遵循项目约定 §8.1（主图禁单纯柱/折线/饼；用相图/雨云/散点+密度/多面板）+ §4.3（中英双版本，数据/配色/版式一致，仅文本语言不同）。
Wong colorblind-safe 调色盘，despine，panel 字母，矢量 PDF。源数据：04outputs/_from_server_v5/。

运行方式:
    cd <项目根>
    python 02code/42paper_figs_v8.py        # 同时输出 en 与 zh

输出文件:
    06doc/01manuscript/figures_v8/fig{1..4}_*.pdf       — 英文版
    06doc/01manuscript/figures_v8/fig{1..4}_*-zh.pdf    — 中文镜像版
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from figure_style import apply_nature_style, WONG, ENTITY_COLORS, CONTINUOUS_CMAP, REGION_FILL

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "04outputs" / "_from_server_v5"
OUT = ROOT / "06doc" / "01manuscript" / "figures_v8"
OUT.mkdir(parents=True, exist_ok=True)

# 三分律 4 regime 有序配色（需 4 个可辨色，无绿故无红绿混淆）：可恢复→蓝，p=0.25→天蓝，临界→橙，不可恢复→红
# 与全稿“可恢复=蓝”一致；临界/不可恢复同处退化侧（橙/红），保持 4 类可辨且色盲安全
REG_COLOR = {"fixed_recover": ENTITY_COLORS["recoverable"], "sub_recover": WONG["skyblue"],
             "critical": WONG["orange"], "subcritical": WONG["red"]}
LANG = "en"  # 由 main 切换


def L(en, zh):
    return zh if LANG == "zh" else en


def setfont():
    # 统一 Nature 风格（spine/legend/pdf-fonttype/unicode-minus），再按语言设置字体族与本图字号
    apply_nature_style(also_chinese=True)
    if LANG == "zh":
        fam = ["Arial Unicode MS", "Songti SC", "STSong", "Arial"]
    else:
        fam = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": fam,
        "font.size": 9, "axes.linewidth": 0.8, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    })


def panel(ax, t):
    ax.text(-0.16, 1.04, t, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")


def sfx():
    return "-zh" if LANG == "zh" else ""


def reglab(rn):
    return {"fixed_recover": L("recoverable (p=0)", "可恢复 (p=0)"),
            "sub_recover": L("recoverable (p=0.25)", "可恢复 (p=0.25)"),
            "critical": L("critical window (p=0.5)", "临界窗口 (p=0.5)"),
            "subcritical": L("unrecoverable (p=1)", "不可恢复 (p=1)")}[rn]


def fig1():
    # panel a = 3D 可恢复性曲面 Ψ(n,ΔT)（z 轴），临界流形 Ψ=γ 为橙脊，三分律投影于底面；
    # panel b = 清理后的 2D 相图（修复 R1-C 指认的"临界线标签压线 + p=0.25 裁切"低级瑕疵）。
    from matplotlib.colors import LightSource
    n0, dt0 = 200.0, 16.0
    xg = np.linspace(np.log10(1e2), np.log10(3e4), 90)
    yg = np.linspace(np.log10(0.12), np.log10(60), 90)
    X, Y = np.meshgrid(xg, yg)
    Psi = (10 ** X / n0) * (10 ** Y / dt0) ** 2     # Ψ=(n/n0)(ΔT/dt0)^2；Ψ=γ⇔ΔT∝n^{-1/2}（=旧橙线）
    Zc = np.clip(np.log10(Psi), -3.2, 3.2)
    fig = plt.figure(figsize=(7.4, 3.5))
    ax = fig.add_axes([0.005, 0.05, 0.55, 0.92], projection="3d")
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Zc, cmap=mpl.colormaps[CONTINUOUS_CMAP],
                   norm=mpl.colors.Normalize(-3.2, 3.2), vert_exag=0.1, blend_mode="soft")
    ax.plot_surface(X, Y, Zc, facecolors=rgb, rstride=2, cstride=2, linewidth=0,
                    antialiased=True, alpha=0.97, shade=False)
    nl = np.logspace(2, np.log10(3e4), 200); dtl = dt0 * (nl / n0) ** (-0.5)
    m = (dtl >= 0.12) & (dtl <= 60)
    ax.plot(np.log10(nl[m]), np.log10(dtl[m]), np.zeros(m.sum()), color=WONG["orange"], lw=3.0, zorder=10)
    zoff = -3.6
    ax.contourf(X, Y, Zc, levels=[-3.2, 0, 3.2], zdir="z", offset=zoff,
                colors=[REGION_FILL["nogo"], REGION_FILL["go"]], alpha=0.18)
    ax.plot(np.log10(nl[m]), np.log10(dtl[m]), np.full(m.sum(), zoff),
            color=WONG["orange"], lw=1.6, ls="--", zorder=5)
    ax.scatter([np.log10(n0)], [np.log10(dt0)], [0], color="k", s=28, depthshade=False, zorder=11)
    ax.plot([np.log10(n0)] * 2, [np.log10(dt0)] * 2, [zoff, 0], color="grey", lw=0.7, ls=":")
    ax.text(np.log10(2.6e4), np.log10(46), 2.7, L("recoverable", "可恢复") + "\n" + r"$\Psi>\gamma$",
            color=ENTITY_COLORS["recoverable"], fontsize=7.5, fontweight="bold", ha="center")
    ax.text(np.log10(7e2), np.log10(0.16), -2.9, L("unrecoverable", "不可恢复") + "\n" + r"$\Psi<\gamma$",
            color=WONG["orange"], fontsize=7.5, fontweight="bold", ha="center")
    ax.text(np.log10(1.1e3), np.log10(0.62), 0.4, L(r"critical $\Psi{=}\gamma$", r"临界 $\Psi{=}\gamma$"),
            color="#9c6500", fontsize=7, ha="center")
    ax.set_xticks([2, 3, 4]); ax.set_xticklabels([r"$10^2$", r"$10^3$", r"$10^4$"])
    ax.set_yticks([-1, 0, 1]); ax.set_yticklabels([r"$0.1$", r"$1$", r"$10$"])
    ax.set_zticks([-3, 0, 3]); ax.set_zticklabels([r"$10^{-3}$", r"$\gamma$", r"$10^{3}$"])
    ax.set_xlabel(L("sample size  n", "样本量  n"), labelpad=-3)
    ax.set_ylabel(L(r"design window  $\Delta T$ (K)", r"设计温窗  $\Delta T$ (K)"), labelpad=-3)
    ax.set_zlabel(L(r"recoverability  $\Psi_n$", r"可恢复性  $\Psi_n$"), labelpad=-6, rotation=90)
    ax.view_init(elev=22, azim=-58); ax.set_box_aspect((1, 1, 0.62), zoom=1.02)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.985, 0.985, 0.99, 1.0)); axis.pane.set_edgecolor((0.8, 0.8, 0.82, 1.0))
        axis.pane.set_linewidth(0.4); axis._axinfo["grid"].update(color=(0.82, 0.82, 0.85, 1.0), linewidth=0.3)
        axis.set_tick_params(pad=-1)
    ax.text2D(0.02, 0.98, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax2 = fig.add_axes([0.66, 0.16, 0.32, 0.78])
    nn = np.logspace(2, 4.5, 240); crit = dt0 * (nn / n0) ** (-0.5)
    ax2.fill_between(nn, crit, 60, color=REGION_FILL["go"], alpha=0.10)
    ax2.fill_between(nn, 0.05, crit, color=REGION_FILL["nogo"], alpha=0.10)
    ax2.plot(nn, crit, color=WONG["orange"], lw=2.4, zorder=5)
    ns = nn[nn >= n0]
    ax2.plot(ns, dt0 * (ns / n0) ** (-0.25), color=ENTITY_COLORS["recoverable"], lw=1.4, ls="--")
    ax2.plot(ns, dt0 * (ns / n0) ** (-1.0), color=WONG["red"], lw=1.4, ls="--")
    ax2.plot([n0], [dt0], "o", color="k", ms=6, zorder=6)
    ax2.annotate(L(r"design $(n_0,\Delta T_0)$", r"设计点 $(n_0,\Delta T_0)$"), (n0, dt0),
                 xytext=(n0 * 1.5, dt0 * 1.7), fontsize=6.5, color="k",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="grey"))
    ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlim(1e2, 3e4); ax2.set_ylim(0.1, 60)
    ax2.set_xlabel(L("sample size  n", "样本量  n")); ax2.set_ylabel(L(r"design window  $\Delta T$ (K)", r"设计温窗  $\Delta T$ (K)"))
    ax2.text(8e3, 30, L("recoverable", "可恢复"), color=ENTITY_COLORS["recoverable"], fontsize=8, ha="center", fontweight="bold")
    ax2.text(8e3, 0.14, L("unrecoverable", "不可恢复"), color=WONG["red"], fontsize=8, ha="center", fontweight="bold")
    ax2.text(2.2e3, dt0 * (2.2e3 / n0) ** (-0.5) * 1.18, r"$\Delta T\!\propto\!n^{-1/2}$",
             color="#9c6500", fontsize=7, rotation=-24, rotation_mode="anchor", ha="center")
    ax2.text(1.3e3, dt0 * (1.3e3 / n0) ** (-0.25) * 1.12, r"$p{=}0.25$", color=ENTITY_COLORS["recoverable"], fontsize=6.5, ha="center")
    ax2.text(1.3e3, dt0 * (1.3e3 / n0) ** (-1.0) * 0.78, r"$p{=}1$", color=WONG["red"], fontsize=6.5, ha="center")
    ax2.text(-0.22, 1.04, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="top")
    fig.savefig(OUT / f"fig1_framework{sfx()}.pdf"); plt.close(fig)


def fig2():
    agg = pd.read_csv(SRC / "43trichotomy_formal-agg.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for rn in REG_COLOR:
        s = agg[agg.regime == rn].sort_values("n")
        ax[0].plot(s.n, s.psi_n, "o-", ms=4, lw=1.4, color=REG_COLOR[rn], label=reglab(rn))
    ax[0].axhspan(0.3, 3, color="#E69F00", alpha=0.08); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(L("sample size  n", "样本量  n")); ax[0].set_ylabel(r"$\Psi_n=n\,\lambda_{\min}(\hat I_1)$")
    ax[0].legend(frameon=False, loc="lower left"); panel(ax[0], "a")
    psis = np.array([])
    for rn in REG_COLOR:
        s = agg[agg.regime == rn].sort_values("psi_n")
        ax[1].errorbar(s.psi_n, s.err_deg_mean, yerr=s.err_deg_std, fmt="o", ms=4.5, color=REG_COLOR[rn], capsize=2, lw=1, mec="k", mew=0.4)
        psis = np.concatenate([psis, s.psi_n.values])
    xg = np.logspace(np.log10(max(psis.min(), 1e-2)), np.log10(psis.max()), 50)
    ax[1].plot(xg, 0.9 * xg ** (-0.5), color="k", lw=1, ls=":", label=L(r"$\Psi_n^{-1/2}$ guide", r"$\Psi_n^{-1/2}$ 参考"))
    ax[1].axvspan(0.3, 3, color="#E69F00", alpha=0.10); ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\Psi_n$"); ax[1].set_ylabel(L("error along degenerate direction", "退化方向误差"))
    ax[1].text(0.9, 6, L("critical\nwindow", "临界\n窗口"), color="#B8860B", fontsize=7.5, ha="center")
    ax[1].legend(frameon=False, loc="upper right"); panel(ax[1], "b")
    fig.tight_layout(); fig.savefig(OUT / f"fig2_trichotomy{sfx()}.pdf"); plt.close(fig)


def fig3():
    # R1-C 升级：joint plot（主 hexbin + 上/右边际分布 + 相关系数注释），提升信息密度
    # 2026-08-23 RYA-588：本图【只此一处】改读 04outputs/ 正本，不读 SRC(_from_server_v5)。两条理由：
    #   (1) 数据卡 120DERIVED 登记的 fig:krug 数据源本来就写的是 04outputs/43trichotomy_formal-percell.csv，
    #       而 _from_server_v5 全仓未登记；此前读的是未登记快照，属「谁产出/谁引用」对不上的同族接线缺陷。
    #   (2) 加权列 slope_krug_w 是 43 本轮新出的；_from_server_v5 是已停机服务器的冻结快照、不可重跑，
    #       往里补列会毁掉它「服务器原样产出」的语义。
    # 数值影响实测：两份 percell 的 slope_krug 逐位相同（max|diff|=0），slope_emp max|diff|=1.44e-15。
    # fig2 仍读 SRC 的 agg（本轮不动），故此处不改全局 SRC。
    pc = pd.read_csv(ROOT / "04outputs" / "43trichotomy_formal-percell.csv")
    sk, se = pc.slope_krug_w.values, pc.slope_emp.values
    fig = plt.figure(figsize=(4.2, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.2, 1], height_ratios=[1, 4.2], wspace=0.05, hspace=0.05)
    axm = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[0, 0], sharex=axm); axr = fig.add_subplot(gs[1, 1], sharey=axm)
    hb = axm.hexbin(sk, se, gridsize=22, cmap=CONTINUOUS_CMAP, mincnt=1, linewidths=0.2)  # 统一连续色图 cividis
    lim = [min(sk.min(), se.min()), max(sk.max(), se.max())]
    axm.plot(lim, lim, color=WONG["orange"], lw=1.6, ls="--", label=L("identity", "恒等线"))  # 橙参考线
    r = float(np.corrcoef(sk, se)[0, 1])
    axm.text(0.05, 0.84, f"r = {r:.2f}", transform=axm.transAxes, fontsize=8.5, va="top", color="#333333", fontweight="bold")
    axm.set_xlabel(L(r"weighted Krug prediction  $\mathrm{E}_w[T_{\mathrm{ref}}/T]$",
                     r"加权 Krug 预测  $\mathrm{E}_w[T_{\mathrm{ref}}/T]$"))
    axm.set_ylabel(L(r"empirical slope  $\mathrm{d}\log A/\mathrm{d}\varphi$", r"经验斜率  $\mathrm{d}\log A/\mathrm{d}\varphi$"))
    axm.legend(frameon=False, loc="upper left")
    axt.hist(sk, bins=24, color=ENTITY_COLORS["recoverable"], alpha=0.6, edgecolor="white", linewidth=0.3)
    axr.hist(se, bins=24, orientation="horizontal", color=ENTITY_COLORS["recoverable"], alpha=0.6, edgecolor="white", linewidth=0.3)
    axt.axis("off"); axr.axis("off")
    cax = axm.inset_axes([0.60, 0.07, 0.32, 0.035])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal"); cb.set_label(L("count", "计数"), fontsize=6.5); cb.ax.tick_params(labelsize=6)
    fig.savefig(OUT / f"fig3_krug{sfx()}.pdf"); plt.close(fig)


def fig4():
    df = pd.read_csv(SRC / "44crossdevice_formal-byseed.csv")
    agg = df.groupby("K").agg(fm=("acc_func", "mean"), fs=("acc_func", "std"), dm=("acc_disc", "mean"),
                              ds=("acc_disc", "std"), psi=("psi_K", "mean")).reset_index()
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.4), constrained_layout=True)
    ax[0].axvspan(7, 12, color=REGION_FILL["go"], alpha=0.08)  # 可恢复带=蓝淡彩
    ax[0].plot(agg.K, agg.fm, "o-", color=ENTITY_COLORS["functional"], lw=1.6, ms=5, label=L("functional (common basis)", "函数型(共同基)"))
    ax[0].fill_between(agg.K, agg.fm - agg.fs, agg.fm + agg.fs, color=ENTITY_COLORS["functional"], alpha=0.15)
    ax[0].plot(agg.K, agg.dm, "s--", color=ENTITY_COLORS["discrete"], lw=1.4, ms=4.5, label=L("discrete (interp.)", "离散(插值)"))
    ax[0].fill_between(agg.K, agg.dm - agg.ds, agg.dm + agg.ds, color=ENTITY_COLORS["discrete"], alpha=0.12)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks([8, 16, 32, 64, 224]); ax[0].set_xticklabels(["8", "16", "32", "64", "224"]); ax[0].minorticks_off()
    ax[0].set_xlabel(L("device band count  K", "设备波段数  K"))
    ax[0].set_ylabel(L("cross-device transfer accuracy", "跨设备迁移精度")); ax[0].legend(frameon=False, loc="lower right")
    ax[0].text(0.10, 0.12, L("cheap\ndevice", "廉价\n设备"), transform=ax[0].transAxes, color=ENTITY_COLORS["recoverable"], fontsize=7.5, ha="center"); panel(ax[0], "a")
    g8 = df[df.K == 8].gain.values; brng = np.random.default_rng(20240529)
    bs = [g8[brng.integers(0, len(g8), len(g8))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); gm = float(g8.mean())
    rng = np.random.default_rng(0); xj = 1 + (rng.random(len(g8)) - 0.5) * 0.16
    ax[1].axhline(0, color="grey", lw=0.8, ls=":")
    ax[1].errorbar(1.0, gm, yerr=[[gm - lo], [hi - gm]], fmt="none", ecolor=WONG["orange"], elinewidth=1.6, capsize=4, zorder=2)
    ax[1].scatter(xj, g8, s=42, color=ENTITY_COLORS["functional"], edgecolor="k", lw=0.4, zorder=3)  # 增益=函数型量→蓝
    ax[1].hlines(gm, 0.84, 1.16, color=WONG["orange"], lw=2.2, zorder=4); ax[1].set_xlim(0.55, 1.55)
    ax[1].set_xticks([1]); ax[1].set_xticklabels(["K=8"]); ax[1].set_ylabel(L("functional − discrete gain", "函数型 − 离散 增益")); panel(ax[1], "b")
    ax[1].text(1.2, gm, f"+{gm:.3f}\n[{lo:.3f}, {hi:.3f}]", color=WONG["orange"], ha="left", va="center", fontsize=7)
    sc = ax[2].scatter(df.psi_K, df.gain, c=np.log2(df.K), cmap=CONTINUOUS_CMAP, s=26, edgecolor="k", lw=0.3)  # 统一连续色图 cividis
    ax[2].axhline(0, color="grey", lw=0.8, ls=":"); ax[2].set_xscale("symlog", linthresh=1e-4)
    ax[2].set_xlabel(L(r"recoverability index  $\Psi(K)$", r"可恢复性指标  $\Psi(K)$"))
    ax[2].set_ylabel(L("transfer gain", "迁移增益")); cb = fig.colorbar(sc, ax=ax[2], shrink=0.8)
    cb.set_label(r"$\log_2 K$", fontsize=8); panel(ax[2], "c")
    fig.savefig(OUT / f"fig4_crossdevice{sfx()}.pdf"); plt.close(fig)


if __name__ == "__main__":
    for lang in ["en", "zh"]:
        LANG = lang; setfont()
        fig1(); fig2(); fig3(); fig4()
        print(f"[{lang}] figures written")
    for f in sorted(OUT.glob("*.pdf")):
        if not f.name.startswith("._"):
            print("  ", f.name, f.stat().st_size, "bytes")
