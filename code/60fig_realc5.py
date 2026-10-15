r"""
60fig_realc5.py: C5 真实动力学验证出版级图（回应外审到 8.5：把 S1/S2 表升级为图）

全部从真实 54c5 输出读取（04outputs/54c5_realkinetics.json + -designsweep.csv），零手填。
三面板（Wong 配色、cividis 色标、无低级图型、中英双版）：
  a Arrhenius 线性图: ln k vs 1/(RT)，4 品种；3 单调软化品种落直线、Pink Lady 后熟非单调散乱(病态)。
  b 设计诊断 Ψ–反演 CV 散点: 26 温度子集，色标=温度窗宽 ΔT；Ψ 大→CV 小，log-log 斜率 -0.57。
  c 品种复现哑铃图: 3 单调品种窄窗 vs 宽窗 CV，宽窗一致更稳。

运行: python 02code/60fig_realc5.py
输出: 06doc/01manuscript/figures_v8/fig5_realc5{,-zh}.pdf
     （文件名必须与 Elsevier_{en,zh}.tex 的 \includegraphics 目标名逐字相同；改名请同时改稿面，否则 139 号机械闸报 BLOCK）
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from figure_style import apply_nature_style, WONG, CONTINUOUS_CMAP

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"
FIGDIR = ROOT / "06doc" / "01manuscript" / "figures_v8"; FIGDIR.mkdir(parents=True, exist_ok=True)
R_GAS = 8.314e-3
# 4 品种为领域内固定类别（非全稿规范实体），取 4 个互异 Wong 色（色盲安全，无红绿单独区分）
C_RG, C_GS, C_RD, C_PL = WONG["blue"], WONG["green"], WONG["orange"], WONG["pink"]
VARC = {"Royal Gala": C_RG, "Granny Smith": C_GS, "Red Delicious": C_RD, "Pink Lady": C_PL}
T_C = [2, 4, 6, 8, 10]
LANG = "en"


def Lz(en, zh): return zh if LANG == "zh" else en
def sfx(): return "-zh" if LANG == "zh" else ""


def setfont():
    apply_nature_style(also_chinese=True)  # 统一 Nature 风格
    fam = ["Arial Unicode MS", "Songti SC", "STSong", "Arial"] if LANG == "zh" else ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": fam,
        "font.size": 9, "axes.linewidth": 0.8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7})


def panel(ax, t):
    ax.text(-0.20, 1.05, t, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")


def vlabel(v):
    return {"Royal Gala": Lz("Royal Gala", "皇家嘎啦"), "Granny Smith": Lz("Granny Smith", "青苹果"),
            "Red Delicious": Lz("Red Delicious", "红元帅"), "Pink Lady": Lz("Pink Lady", "粉红佳人")}[v]


def make_fig():
    d = json.load(open(OUT / "54c5_realkinetics.json"))
    sweep = pd.read_csv(OUT / "54c5_realkinetics-designsweep.csv")
    pv = d["per_variety"]; mv = d["multi_variety_replication"]["detail"]

    fig = plt.figure(figsize=(9.4, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 0.95], wspace=0.42)

    axa = fig.add_subplot(gs[0, 0])
    TK = np.array(T_C) + 273.15; x = 1.0 / (R_GAS * TK)
    for v in ["Royal Gala", "Granny Smith", "Red Delicious", "Pink Lady"]:
        k = np.array(pv[v]["k_per_T"], float); good = np.isfinite(k) & (k > 0)
        lnk = np.log(k[good]); xa = x[good]
        axa.scatter(xa, lnk, s=22, color=VARC[v], zorder=3, edgecolor="white", linewidth=0.4)
        if pv[v]["r2_arrhenius"] > 0.7:
            sl = np.polyfit(xa, lnk, 1); xf = np.linspace(xa.min(), xa.max(), 20)
            axa.plot(xf, np.polyval(sl, xf), "-", color=VARC[v], lw=1.1, alpha=0.8,
                     label=f"{vlabel(v)} ($E_a${pv[v]['Ea_kJmol']:.0f}, $R^2${pv[v]['r2_arrhenius']:.2f})")
        else:
            axa.plot([], [], "o", color=VARC[v], label=f"{vlabel(v)} " + Lz("(non-mono.)", "(非单调)"))
    axa.set_xlabel(Lz("$1/(RT)$  (mol/kJ)", "$1/(RT)$  (mol/kJ)"))
    axa.set_ylabel(Lz("$\\ln k$  (softening rate)", "$\\ln k$（软化速率）"))
    # R1-C 必修：图例移出坐标区到面板下方（2 列），不再压在 Arrhenius 数据线上
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False,
               fontsize=5.8, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0)
    panel(axa, "a")

    axb = fig.add_subplot(gs[0, 1])
    sc = axb.scatter(sweep["Psi_n"], sweep["inv_CV_Ea"], c=sweep["deltaT"], cmap="cividis",
                     s=30, zorder=3, edgecolor="white", linewidth=0.4)
    # 叠加：超定(>=3温)子集的真实测量 SD 传播估计误差（强证据；非恰定 2 温噪声传播）
    if "realCV_Ea" in sweep.columns and "n_temps" in sweep.columns:
        od = sweep[(sweep["n_temps"] >= 3) & np.isfinite(sweep["realCV_Ea"]) & (sweep["realCV_Ea"] > 0)]
        rho_od = d.get("overdetermined_real_error", {}).get("spearman", float("nan"))
        if len(od) >= 4:
            axb.scatter(od["Psi_n"], od["realCV_Ea"], marker="D", s=24, facecolor="none",
                        edgecolor=C_RD, linewidth=1.0, zorder=4,
                        label=Lz(f"overdet. real est.err ($\\rho${rho_od:.2f})",
                                  f"超定真实估计误差 ($\\rho${rho_od:.2f})"))
    lp = np.log(sweep["Psi_n"]); lc = np.log(sweep["inv_CV_Ea"]); sl = np.polyfit(lp, lc, 1)
    xf = np.linspace(lp.min(), lp.max(), 30)
    axb.plot(np.exp(xf), np.exp(np.polyval(sl, xf)), "--", color="#555555", lw=1.0,
             label=Lz(f"slope {sl[0]:.2f}", f"斜率 {sl[0]:.2f}"))
    nar = sweep[sweep.temps == "8/10"].iloc[0]; wid = sweep[sweep.temps == "2/10"].iloc[0]
    axb.annotate(Lz("narrow {8,10}", "窄窗{8,10}"), (nar.Psi_n, nar.inv_CV_Ea), fontsize=6.2,
                 xytext=(8, 6), textcoords="offset points", color=C_RD)
    axb.annotate(Lz("wide {2,10}", "宽窗{2,10}"), (wid.Psi_n, wid.inv_CV_Ea), fontsize=6.2,
                 xytext=(-6, -12), textcoords="offset points", color=C_RG)
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel(Lz("diagnostic $\\Psi_n$", "诊断 $\\Psi_n$"))
    axb.set_ylabel(Lz("inversion CV($E_a$)", "反演 CV($E_a$)"))
    axb.legend(loc="upper right", frameon=False, fontsize=6.5)
    cb = fig.colorbar(sc, ax=axb, fraction=0.045, pad=0.03); cb.set_label(Lz("$\\Delta T$ (°C)", "$\\Delta T$ (°C)"), fontsize=7)
    panel(axb, "b")

    axc = fig.add_subplot(gs[0, 2])
    monos = [v for v in ["Royal Gala", "Granny Smith", "Red Delicious"] if mv[v].get("monotone")]
    y = np.arange(len(monos))
    for yi, v in zip(y, monos):
        m = mv[v]
        axc.plot([m["wide_CV"], m["narrow_CV"]], [yi, yi], "-", color="#cccccc", lw=1.2, zorder=1)
        axc.scatter(m["wide_CV"], yi, s=40, color=VARC[v], marker="o", zorder=3, label=Lz("wide window", "宽窗") if yi == 0 else "")
        axc.scatter(m["narrow_CV"], yi, s=40, color=VARC[v], marker="X", zorder=3, label=Lz("narrow window", "窄窗") if yi == 0 else "")
    axc.set_yticks(y); axc.set_yticklabels([vlabel(v) for v in monos], fontsize=7)
    axc.set_xscale("log")
    axc.set_xlabel(Lz("inversion CV($E_a$) (log)", "反演 CV($E_a$)（对数）"))
    axc.legend(loc="lower right", frameon=False, fontsize=6.5, handletextpad=0.3)
    panel(axc, "c")

    fig.savefig(FIGDIR / f"fig5_realc5{sfx()}.pdf"); plt.close(fig)
    print(f"saved fig5_realc5{sfx()}.pdf")


def main():
    global LANG
    for lang in ["en", "zh"]:
        LANG = lang; setfont(); make_fig()


if __name__ == "__main__":
    main()
