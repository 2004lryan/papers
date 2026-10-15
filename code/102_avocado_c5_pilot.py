"""
102_avocado_c5_pilot.py — C5 真实动力学第三体系：Hass 牛油果（004），补真统计估计误差与独立复现。

动机（回应审稿人对 C5 的四条点名，论文自己也已认下）：
    (1) "2 温度 Arrhenius 为恰定（R^2≡1、自由度 0），故 CV 为**噪声传播敏感性**、**并非统计估计误差**"
    (2) "16 个相互嵌套/温度重叠的**非独立**子集"
    (3) "**单一品种**（Royal Gala）"
    (4) 真实数据均来自他人论文的**汇总表**（PMC10253207 / PMC9319022），非原始数据

    004 能修 (1)(2)(3)(4)：478 个**完全独立**的 Hass 牛油果、每温度 127-155 个真实重复
    （达级5天数 SD = 2.7/0.6/0.7 天，真实个体间离散），且为**原始数据**。
    根因澄清：论文那条局限的根子**不是"温度只有 2 个"，而是"每温度只有一个已发表均值、无重复"**——
    无重复 => 只能对 k(T) 注入残差做 bootstrap => 那是敏感性。004 有真实个体重复
    => 对个体 bootstrap => 每温度速率有**真实抽样误差** => Ea 有**真实抽样分布**。

诚实边界（先声明，不许事后粉饰）：
    - 004 仅 **2 个可用温度**（T10=10C / T20=20C；Tam 为"常温"无标称值、且环境温度本身漂移，
      按 IRON RULE **不得**为其臆造温度值，故**排除出拟合**）。
    - 因此本脚本**不能**检验三分律的 **ΔT 相变**（那需要温度窗扫描）；只能检验**可恢复侧**
      的速率律 error ∝ Psi_n^{-1/2}（ΔT 固定时 Psi_n ∝ n）。
    - Tam 仅用作**外部一致性检查**：由 {T10,T20} 拟合的 Arrhenius 反推 Tam 应有的温度，
      看是否落在物理合理的实验室常温区间——这是**弱检验**，不作主张。

数据：${ARIS_RAW_ROOT}/004_hass_avocado_rgb_ripening（仓库外只读原始数据，不复制进项目；
      ARIS_RAW_ROOT 未设置时默认取 <项目根目录>/../../05data）
输出：04outputs/102_avocado_c5_pilot.json
"""
from __future__ import annotations

import os

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"
#: 原始数据只读引用、禁止复制进项目（项目约定），故此处指向项目外的只读路径。
#: 可移植性：优先读环境变量 AVOCADO_XLSX，其次读 AVOCADO_DATA_DIR 下的标准相对路径，
#: 最后回落到本机默认位置。数据集公开可获：Mendeley Data doi:10.17632/3xd9n945v8.1
_REL = Path("Hass Avocado Ripening Photographic Dataset") / "Avocado Ripening Dataset.xlsx"
# 【2026-08-10 订正，RYA-105】原为作者机绝对路径，既不可移植也会随发布件泄露私有目录结构。
# 改为「项目根目录的祖父目录下的 05data」并允许 ARIS_RAW_ROOT 覆盖。
_RAW_ROOT = Path(os.environ.get("ARIS_RAW_ROOT", str(ROOT.parents[1] / "05data")))
_DEFAULT_DIR = _RAW_ROOT / "004_hass_avocado_rgb_ripening"
if os.environ.get("AVOCADO_XLSX"):
    XLSX = Path(os.environ["AVOCADO_XLSX"])
elif os.environ.get("AVOCADO_DATA_DIR"):
    XLSX = Path(os.environ["AVOCADO_DATA_DIR"]) / _REL
else:
    XLSX = _DEFAULT_DIR / _REL

R_GAS = 8.314462618  # J/(mol*K)
TEMPS_K = {"T10": 283.15, "T20": 293.15}  # 仅标称已知者；Tam 无标称值，排除
RIPE_LEVEL = 5
N_BOOT = 2000
SEED = int(os.environ.get("SEED_OVERRIDE", "0"))  # 多种子复跑入口(§5.3)；未设置 SEED_OVERRIDE 时与原字面值逐字节一致


def load_verified() -> pd.DataFrame:
    """IRON RULE: 在盘 + 可载 + 结构与描述相符，三条全过才返回。"""
    if not XLSX.exists():
        raise FileNotFoundError(f"IRON RULE 违反：{XLSX} 不在盘")
    df = pd.read_excel(XLSX)
    need = {"Storage Group", "Sample", "Day of Experiment", "Ripening Index Classification"}
    if not need.issubset(df.columns):
        raise ValueError(f"IRON RULE 违反：列名不符，实得 {list(df.columns)}")
    if len(df) != 14722:
        raise ValueError(f"IRON RULE 违反：行数 {len(df)} != 描述 14722")
    if set(df["Storage Group"].unique()) != {"T10", "T20", "Tam"}:
        raise ValueError(f"IRON RULE 违反：储温组 {set(df['Storage Group'].unique())}")
    # 每果只在一个温度下（已核实）——若不成立则逐果 Ea 的排除理由失效，须重新审视
    if (df.groupby("Sample")["Storage Group"].nunique() > 1).any():
        raise ValueError("结构变更：出现跨温度追踪的果，需重新设计（可做逐果 Ea）")
    return df


def per_fruit_rate(df: pd.DataFrame, group: str) -> npt.NDArray[np.float64]:
    """逐果成熟速率 k_i = 1 / (首次达成熟级 5 的天数)。仅取真正走到级 5 的果。"""
    v = df[df["Storage Group"] == group]
    t5 = v[v["Ripening Index Classification"] == RIPE_LEVEL].groupby("Sample")[
        "Day of Experiment"
    ].min()
    t5 = t5[t5 > 0]
    arr: npt.NDArray[np.float64] = 1.0 / t5.to_numpy(dtype=np.float64)
    return arr


def fit_arrhenius(k_by_T: dict[str, float]) -> tuple[float, float]:
    """两点 Arrhenius：ln k = ln A - Ea/(R T)。返回 (Ea[J/mol], lnA)。"""
    Ts = np.array([TEMPS_K[g] for g in k_by_T])
    ks = np.array([k_by_T[g] for g in k_by_T])
    x = 1.0 / Ts
    y = np.log(ks)
    slope, intercept = np.polyfit(x, y, 1)
    return -slope * R_GAS, intercept


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = load_verified()

    rates = {g: per_fruit_rate(df, g) for g in ["T10", "T20", "Tam"]}
    res: dict[str, object] = {
        "study": "102_avocado_c5_pilot",
        "data": "004_hass_avocado_rgb_ripening (Mendeley 3xd9n945v8)，只读引用，未复制进项目",
        "honest_scope": (
            "004 仅 2 个标称已知温度（T10=10C, T20=20C）；Tam 为'常温'无标称值且环境温度漂移，"
            "按 IRON RULE 排除出拟合，不臆造温度。故本脚本**不检验三分律的 ΔT 相变**，"
            "只检验可恢复侧速率律 error ∝ Psi_n^{-1/2}（ΔT 固定 => Psi_n ∝ n）。"
        ),
        "n_fruits_reaching_level5": {g: int(len(v)) for g, v in rates.items()},
        "median_days_to_ripe": {
            g: float(np.median(1.0 / v)) for g, v in rates.items()
        },
        "sd_days_to_ripe_between_individuals": {
            g: float(np.std(1.0 / v, ddof=1)) for g, v in rates.items()
        },
    }

    # ── 全样本 Arrhenius（仅 T10/T20）──
    k_full = {g: float(np.mean(rates[g])) for g in TEMPS_K}
    Ea_full, lnA_full = fit_arrhenius(k_full)
    q10 = float(k_full["T20"] / k_full["T10"])
    res["full_sample"] = {
        "mean_rate_per_day": k_full,
        "Ea_kJ_per_mol": round(Ea_full / 1000, 2),
        "lnA": round(lnA_full, 3),
        "Q10": round(q10, 3),
        "Q10_in_biological_range_2_to_3": bool(2.0 <= q10 <= 3.0),
    }

    # ── 真统计估计误差：对**个体**做 bootstrap（这正是苹果汇总表给不出的）──
    boot_list: list[float] = []
    for _ in range(N_BOOT):
        kb = {
            g: float(np.mean(rng.choice(rates[g], size=len(rates[g]), replace=True)))
            for g in TEMPS_K
        }
        boot_list.append(fit_arrhenius(kb)[0])
    boots: npt.NDArray[np.float64] = np.asarray(boot_list, dtype=np.float64) / 1000.0
    res["real_statistical_error"] = {
        "method": "对独立个体重抽样（非对 k(T) 注入残差）=> 真抽样分布，非噪声传播敏感性",
        "n_boot": N_BOOT,
        "Ea_mean_kJ": round(float(boots.mean()), 3),
        "Ea_sd_kJ": round(float(boots.std(ddof=1)), 3),
        "Ea_CI95_kJ": [round(float(np.percentile(boots, 2.5)), 3),
                       round(float(np.percentile(boots, 97.5)), 3)],
        "Ea_CV": round(float(boots.std(ddof=1) / abs(boots.mean())), 4),
    }

    # ── 样本量扫描：ΔT 固定 => Psi_n ∝ n；检验 error ∝ Psi_n^{-1/2}（斜率应 ≈ -0.5）──
    # 必须**放回**抽样：模拟"若有 n 个独立果"的抽样方差 = sigma^2/n。
    # 若用 replace=False，从仅 127-155 的有限池抽 n 会触发**有限总体校正** (1-n/N)：
    # n=120 / N=127 时方差被压掉约 18 倍，把大 n 端人为拉低、斜率虚假变陡（实测 -0.69）。
    # 这是抽样设计 bug，不是理论被推翻。
    sweep: list[dict[str, float]] = []
    n_grid = [5, 10, 20, 40, 80, 120]
    for n in n_grid:
        err_list: list[float] = []
        for _ in range(N_BOOT // 4):
            kb = {
                g: float(np.mean(rng.choice(rates[g], size=n, replace=True)))
                for g in TEMPS_K
            }
            err_list.append(fit_arrhenius(kb)[0] / 1000.0)
        errs: npt.NDArray[np.float64] = np.asarray(err_list, dtype=np.float64)
        sweep.append({"n_per_temp": n, "Psi_n_proportional_to": n,
                      "Ea_sd_kJ": float(errs.std(ddof=1))})
    ln_n = np.log([s["n_per_temp"] for s in sweep])
    ln_e = np.log([s["Ea_sd_kJ"] for s in sweep])
    slope, _ = np.polyfit(ln_n, ln_e, 1)
    r2 = float(np.corrcoef(ln_n, ln_e)[0, 1] ** 2)
    res["n_sweep"] = {
        "grid": sweep,
        "log_Psi_vs_log_error_slope": round(float(slope), 4),
        "r2": round(r2, 4),
        "theory_predicts": -0.5,
        "note": "ΔT 固定 => Psi_n ∝ n；可恢复侧理论速率 error ≍ Psi_n^{-1/2} => 斜率 -0.5",
    }

    # ── Tam 一致性检查（弱检验，不作主张）──
    k_tam = float(np.mean(rates["Tam"]))
    T_tam_implied = 1.0 / (1.0 / TEMPS_K["T20"] - (np.log(k_tam / k_full["T20"]) * R_GAS / Ea_full))
    res["Tam_consistency_check"] = {
        "status": "弱检验，不作主张（Tam 无标称温度，此为反推）",
        "implied_T_celsius": round(float(T_tam_implied - 273.15), 2),
        "physically_plausible_lab_ambient": bool(15.0 <= T_tam_implied - 273.15 <= 30.0),
        "caveat": "反推值若≈T20，则 Tam 与 T20 近重合，无法提供第 3 个有效温度",
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "102_avocado_c5_pilot.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
