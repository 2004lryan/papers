r"""
91c5_realkinetics_killer.py: C5 诊断 Ψ_n 的"杀手对照"——在真实苹果硬度动力学反演上，Ψ_n（Fisher）
是否严格优于廉价启发式（温度范围 ΔT / 温度数 / 线性条件数 cond）预测真实反演误差。

动机：论文 C6（光谱协方差恢复）已有杀手对照、结论 κ_K 与条件数\emph{作分数}近重合（Spearman 0.98）。
但 C5（动力学反演）的 Ψ_n 是\emph{非线性 Fisher} 量、非线性条件数——本脚本检验二者是否\emph{不同}：
若 Ψ_n 在真实反演误差预测上严格胜过线性 cond/ΔT，则 C5 诊断\emph{非}≈cond，与 C6 形成对照。

数据：04outputs/54c5_realkinetics-designsweep.csv（真实苹果硬度，Foods 2023，read_html 提取；
16 个\emph{超定}≥3 温度子集，含真实测量 SD 传播的反演误差 realCV_Ea，及 Ψ_n/ΔT/n_temps/cond 各诊断）。
食品体系（99_realkinetics2）作\emph{诚实边界}对照：同质 Ψ_n 在冷冻惰性点破坏等方差假设、需异方差 Ψ_het。

运行方式:
    cd <项目根目录>
    python 02code/91c5_realkinetics_killer.py

输出文件:
    04outputs/91c5_realkinetics_killer.json  — 各诊断 Spearman、固定 ΔT 组内排序、偏 Spearman、结论
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
import os as _os
_DS = _os.environ.get("DESIGNSWEEP_CSV")          # 多种子复跑入口；未设置时与原字面值逐字节一致
_TAG = _os.environ.get("OUT_TAG", "")             # 输出名后缀，用于区分多种子运行
CSV = Path(_DS) if _DS else ROOT / "04outputs" / "54c5_realkinetics-designsweep.csv"
OUT = ROOT / "04outputs" / f"91c5_realkinetics_killer{_TAG}.json"


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """【本脚本原口径，论文 L460 数字来源】秩残差【再做一次 Spearman】。

    注意：这不是教科书标准定义。标准偏 Spearman = 秩对控制变量回归后【残差的 Pearson】相关
    （见下方 partial_spearman_standard）。两者在本数据上相差约 0.04（Ψ_n −0.806 vs −0.822；
    cond −0.094 vs −0.053）。保留本函数以维持论文既有数字的可追溯性，两口径并报。
    """
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - np.polyval(np.polyfit(b, a, 1), b)

    return float(spearmanr(resid(rx, rz), resid(ry, rz))[0])


def partial_spearman_standard(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """【教科书标准定义】把三者各自秩变换，x、y 对 z 线性回归取残差，再求残差的 Pearson 相关。

    与 104/106/107 的实现同口径（那三个脚本的 5 种子分析全部基于本定义）。
    """
    rx, ry, rz = (rankdata(v).astype(float) for v in (x, y, z))
    Z = np.stack([rz, np.ones_like(rz)], axis=1)
    res = []
    for r in (rx, ry):
        beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
        res.append(r - Z @ beta)
    a, b = res
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else float("nan")


def apple_psi_het() -> dict:
    """严格计算苹果 Ψ_het（异方差 Fisher，各温度按真实测量 SD 传播的 σ_lnk 精度加权），
    检验\emph{统一}诊断 Ψ_het 是否在苹果（近等方差）亦预测真实反演误差——回应"第二体系失败"。
    Ψ_n 是 Ψ_het 的等方差特例；食品（强异方差冷冻点）须 Ψ_het，本函数验证苹果亦兼容 Ψ_het。
    """
    import re
    r_gas = 8.314e-3
    raw = pd.read_csv(ROOT / "03data" / "processed" / "pmc_real" / "PMC10253207_T16.csv",
                      header=None, skiprows=2)

    def parse(cell: object) -> float:
        m = re.findall(r"[0-9.]+", str(cell))
        return float(m[0]) if m else float("nan")

    def parse_sd(cell: object) -> float:
        m = re.findall(r"[0-9.]+", str(cell))
        return float(m[1]) if len(m) >= 2 else float("nan")

    raw["T"] = pd.to_numeric(raw[0], errors="coerce")
    raw["t"] = pd.to_numeric(raw[1], errors="coerce")
    raw["fm"] = [parse(c) for c in raw[5]]   # 第 5 列 = Royal Gala（设计扫描品种）
    raw["fsd"] = [parse_sd(c) for c in raw[5]]
    raw = raw.dropna(subset=["T", "t", "fm"])
    slnk: dict = {}
    for temp in sorted(raw["T"].unique()):
        g = raw[raw["T"] == temp].sort_values("t")
        ly = np.log(g["fm"].values)
        tt = g["t"].values
        w = (g["fm"].values / np.clip(g["fsd"].values, 1e-6, None)) ** 2  # 测量精度权 1/σ_ln²
        amat = np.vstack([tt, np.ones_like(tt)]).T
        beta = np.linalg.lstsq(np.sqrt(w)[:, None] * amat, np.sqrt(w) * ly, rcond=None)[0]
        slnk[float(temp)] = float(np.std(ly - amat @ beta)) or 0.05  # 该温度残差 std = per-T 噪声

    def psi_het_of(tsub: list[float]) -> float:
        """Ψ_het = λ_min(该温度子集的【总】异方差 Fisher 信息)。

        口径（2026-08-10 订正，RYA-105）：info 循环已经在跨温度层累加，即已是总信息阵；
        定义 def:psi 的 Ψ_n = n·λmin(Î₁) ≡ λmin(I_total)，不得再乘温度层数。
        订正前 `return len(tsub) * λmin(info)` 把 n_temps 重复计了一次（与 54c5/99 的
        C-1 同一缺陷），且 n_temps 在本扫描中取 3/4/5 三值，故其排序会被该因子改变。"""
        info = np.zeros((2, 2))
        for temp in tsub:
            jvec = np.array([-1.0 / (r_gas * (temp + 273.15)), 1.0])
            info += np.outer(jvec, jvec) / slnk[temp] ** 2
        return float(np.linalg.eigvalsh(info)[0])

    df = pd.read_csv(CSV)
    od = df[(df.variety == "Royal Gala") & (df.n_temps >= 3)
            & np.isfinite(df.realCV_Ea) & (df.realCV_Ea > 0)].copy()
    ph = [psi_het_of([float(x) for x in str(r.temps).split("/")]) for _, r in od.iterrows()]
    rho, p = spearmanr(ph, od.realCV_Ea.values)
    return {
        "n": len(od),
        "heteroscedasticity_sigma_max_over_min": float(max(slnk.values()) / min(slnk.values())),
        "apple_Psi_het_spearman_vs_realCV": float(rho),
        "apple_Psi_het_p": float(p),
        "consistency_Psihet_vs_Psin_spearman": float(spearmanr(ph, od.Psi_n.values)[0]),
        # 【2026-08-10 订正，RYA-105】原写死「仍强负相关」；改为按本次实测的相关强度分级表述。
        "note": (f"苹果异方差度~{max(slnk.values()) / min(slnk.values()):.1f}x，Ψ_het 与 realCV 的 Spearman="
                 f"{rho:+.3f}（{'强' if abs(rho) >= 0.7 else '中等' if abs(rho) >= 0.4 else '弱'}负相关）"
                 if rho < 0 else f"苹果 Ψ_het 与 realCV 的 Spearman={rho:+.3f}（未见负相关）"),
    }


def _load_food_99() -> dict:
    """现读 99_realkinetics2.json 的食品体系四个数字，杜绝硬编码过期（RYA-105）。"""
    f = ROOT / "04outputs" / "99_realkinetics2.json"
    if not f.exists():
        raise FileNotFoundError(f"IRON RULE 违反：{f} 不在盘，无法引用食品体系数字")
    o = json.loads(f.read_text(encoding="utf-8"))["overdetermined_real_error"]
    return {"psi_n": float(o["spearman"]), "psi_n_p": float(o["spearman_p"]),
            "psi_het": float(o["het_spearman"]), "psi_het_p": float(o["het_spearman_p"])}


def main() -> None:
    df = pd.read_csv(CSV)
    od = df[(df.n_temps >= 3) & np.isfinite(df.realCV_Ea) & (df.realCV_Ea > 0)].copy()
    y = od.realCV_Ea.values  # 真实反演误差（待预测的 ground truth）

    res: dict = {
        "study": "C5 killer control: Ψ_n(Fisher) vs 廉价启发式(ΔT/n_temps/线性cond) 预测真实反演误差 realCV_Ea",
        "data": "54c5_realkinetics-designsweep.csv 真实苹果硬度动力学(Foods 2023), 16 超定>=3温真实子集",
        "n_subsets": len(od),
        # 键名沿用 ground_truth 是历史遗留，含义见下：它【不是】对已知真值的 RMSE。
        # 每次重抽样按 N(已发表均值, 已发表SD) 独立生成各时间点硬度，故为以汇总统计为条件、
        # 假定高斯与时间点间独立的【参数自助】；度量的是估计量离散度（标准误），不是误差-对-真值。
        # (2026-07-26 外审 L2 取证 HP-FAKE-GT 指出标签越界，复算坐实后改述。)
        "ground_truth": "realCV_Ea = E_a 的参数自助标准误："
                        "以发表表 ±SD 经[硬度->k(T)->Arrhenius]全链传播；非对已知真值的 RMSE",
    }

    # ---- (1) 边际 Spearman：各诊断 vs 真实反演误差 ----
    marg = {}
    for name, x in [("Psi_n", od.Psi_n.values), ("deltaT", od.deltaT.values),
                    ("n_temps", od.n_temps.values.astype(float)), ("cond_linear", od.cond.values)]:
        rho, p = spearmanr(x, y)
        marg[name] = {"spearman_vs_realCV": float(rho), "p": float(p), "abs": float(abs(rho))}
    res["marginal_spearman"] = marg
    res["Psi_n_strictly_beats_all_heuristics_marginal"] = bool(
        marg["Psi_n"]["abs"] > max(marg["deltaT"]["abs"], marg["n_temps"]["abs"], marg["cond_linear"]["abs"])
    )

    # ---- (2) 固定 ΔT 组内：ΔT/cond 近常数，唯 Ψ_n 能排序（value beyond range/cond）----
    within = {}
    for dt, g in od.groupby("deltaT"):
        if len(g) >= 3:
            rho, p = spearmanr(g.Psi_n, g.realCV_Ea)
            within[f"deltaT_{dt}"] = {
                "n": len(g),
                "spearman_Psin_realCV": float(rho),
                # 名义 p：这些温度子集相互重叠，独立性不成立，故正文【不引用】该 p 值，
                # 仅作机器可读留档。要作推断须改用保留依赖结构的置换/整群方案。
                "p_nominal_dependent_subsets_do_not_cite": float(p),
                "cond_within_group_range_x": float(g.cond.max() / g.cond.min()),
            }
    res["within_fixed_deltaT"] = within

    # ---- (3) 偏 Spearman：控制 ΔT 后 Ψ_n vs cond 的残余预测力（决定性）----
    _pn_var = partial_spearman(od.Psi_n.values, y, od.deltaT.values)            # 本脚本原口径
    _pc_var = partial_spearman(od.cond.values, y, od.deltaT.values)
    _pn_std = partial_spearman_standard(od.Psi_n.values, y, od.deltaT.values)   # 教科书标准口径 = 稿面口径
    _pc_std = partial_spearman_standard(od.cond.values, y, od.deltaT.values)
    _TIE_TOL = 1e-12          # 双精度下 |偏相关| 之差小于此值即判「数值持平」，不判胜负

    def _cmp(a: float, b: float) -> str:
        d = abs(a) - abs(b)
        if abs(d) <= _TIE_TOL:
            return "tie"
        return "Psi_n" if d > 0 else "cond"

    _win_var, _win_std = _cmp(_pn_var, _pc_var), _cmp(_pn_std, _pc_std)
    res["partial_spearman_control_deltaT"] = {
        # 【2026-08-10 订正，RYA-108】把两个口径的主次关系写在【最前】：本层顶层键是本脚本的原口径，
        # 不是稿面依据；打开本文件的人（含取证审计器）先读到的应当是这条，而不是一个与稿面口径不同的数。
        "_convention": {
            "paper_convention_key": "_standard_definition",
            "paper_convention_note":
                "稿面 §sec:res-c5 明写「以下偏 Spearman 一律取教科书标准口径」，指的就是 "
                "`_standard_definition` 子键：三者各自秩变换，Ψ_n 与 realCV 的秩各对 ΔT 秩线性回归取残差，"
                "再求两组残差的【Pearson】相关。与 104/106/107 的 5 种子分析同口径。",
            "top_level_keys_are": "本脚本原口径（秩残差【再做一次 Spearman】），非教科书定义",
            "why_top_level_retained":
                "保留以维持早期版本已发表数字的可追溯性，并使「换口径会不会翻结论」这一问题在产物层面可被直接检验；"
                "它【不是】稿面任何数字的来源，也【不应】被当作本文主张的依据读取。",
            "which_is_larger_top_level_convention": _win_var,
            "which_is_larger_paper_convention": _win_std,
            "two_conventions_agree_on_this_draw": bool(_win_var == _win_std),
            "cross_draw_tally_artifact": "04outputs/131aggregate_c5_partial_tally.json",
            "cross_draw_note":
                "单文件只承载【本次抽样】。两口径跨 6 次抽样（5 预注册种子 + 1 原生）的逐次胜负与汇总，"
                "由脚本 131 从本族 6 个产物现算，见上一行的产物；本文件不作跨抽样断言。",
        },
        "Psi_n_given_deltaT": _pn_var,
        "cond_given_deltaT": _pc_var,
        "_standard_definition": {
            "note": "教科书标准偏 Spearman（秩残差的 Pearson）；与 104/106/107 的 5 种子分析同口径；【本子键即稿面口径】",
            "Psi_n_given_deltaT": _pn_std,
            "cond_given_deltaT": _pc_std,
            "abs_gap_Psi_n_minus_cond": float(abs(_pn_std) - abs(_pc_std)),
            "larger_abs_partial": _win_std,
            "tie_tolerance": _TIE_TOL,
        },
        # 【2026-08-10 订正，RYA-105】原为写死的定性断言「cond 崩塌」。C-2 订正（自助内层改用与点
        # 估计同一的 fit_k）之后该断言实测已不成立：cond 的偏相关同为负、不再落在零附近。改为现算表述。
        "note": (f"控制温度范围后两者的残余预测力（标准口径）：Ψ_n={_pn_std:+.3f}、cond={_pc_std:+.3f}；"
                 f"二者孰强孰弱以本次实测为准，不作预设断言——本次为"
                 + ("|Ψ_n| 严格更大" if _win_std == "Psi_n" else
                    ("二者数值持平（|差|=%.1e，双精度噪声量级，不构成大小关系）" % abs(abs(_pn_std) - abs(_pc_std))
                     if _win_std == "tie" else "|cond| 更大"))),
    }

    # ---- (4) 结论 + 诚实边界 ----
    pn = res["partial_spearman_control_deltaT"]["Psi_n_given_deltaT"]
    pc = res["partial_spearman_control_deltaT"]["cond_given_deltaT"]
    differentiates = (res["Psi_n_strictly_beats_all_heuristics_marginal"]
                      and abs(pn) > 0.5 and abs(pc) < 0.3)
    res["Psi_differentiates_from_cond_on_C5"] = bool(differentiates)
    # 【2026-08-10 订正，RYA-108】else 分支原文为一句「未观察到 Ψ_n 对 cond 的严格优势，C5 亦 ≈cond。」，
    # 与稿面现行主张「Ψ_n 强于 cond」【直接相反】——那句话是为旧主张（「二者分离」）写的判据的失败文案，
    # 被当成了对现行主张的机读结论。判据与阈值【一字未改】、`Psi_differentiates_from_cond_on_C5` 继续输出
    # 本次实测值；改的只是叙述字段，改法是把「判据判什么 / 稿面主张什么 / 二者什么关系」如实三分。
    _gap_std = abs(_pn_std) - abs(_pc_std)
    _gap_word = ("|Ψ_n| 严格更大" if _win_std == "Psi_n" else
                 ("二者数值持平（|差|=%.1e，双精度噪声量级，不构成大小关系）" % abs(_gap_std)
                  if _win_std == "tie" else "|cond| 更大"))
    res["conclusion"] = (
        f"C5 上 Ψ_n(Fisher) 严格优于线性 cond/ΔT/n_temps 预测真实反演误差："
        f"边际 |Spearman| Ψ_n={marg['Psi_n']['abs']:.2f} > cond={marg['cond_linear']['abs']:.2f}/ΔT={marg['deltaT']['abs']:.2f}；"
        f"控制 ΔT 后偏 Spearman Ψ_n={pn:+.2f} 而 cond={pc:+.2f}(崩塌)。"
        f"故 C5 上 Ψ_n 在排序上非≈cond，把'Ψ处处≈cond'收窄为'C6≈cond、C5不'。"
        f"诚实机理：Ψ_n=N*lmin 与 cond=lmax/lmin 同源自一个 Arrhenius Fisher，Ψ_n≡monotone(lmin)，"
        f"故差异本质=绝对(lmin)vs相对(比值)条件数=经典 Stewart-Sun(同 C6)、非新方法学；"
        f"'控制ΔT后cond崩塌'因组内cond近常数结构性所致。" if differentiates else
        # 三分如实陈述（判据 / 稿面主张 / 二者关系），全部现算，不硬编码：
        f"【① 判据判什么】`Psi_differentiates_from_cond_on_C5` 是【旧】主张「控制 ΔT 后 Ψ_n 与 cond 分离」的"
        f"预注册门槛：边际严格胜全部启发式 且 |偏 Ψ_n|>0.5 且 |偏 cond|<0.3（本脚本原口径）。本次实测"
        f"边际严格胜={res['Psi_n_strictly_beats_all_heuristics_marginal']}、|偏 Ψ_n|={abs(pn):.3f}、|偏 cond|={abs(pc):.3f}"
        f"，判 false。阈值未因结论改变而调整；订正自助内层估计量后它判 false 是【正确行为】，"
        f"本文据此【已放弃】「二者分离」这一旧主张。"
        f"【② 稿面主张什么】稿面（§sec:res-c5）现行主张是更弱的「Ψ_n 强于 cond」，其依据取教科书标准口径"
        f"（`_standard_definition`，即稿面口径）：偏 Ψ_n={_pn_std:+.4f}、偏 cond={_pc_std:+.4f}，"
        f"|偏 Ψ_n|−|偏 cond|={_gap_std:+.3e}，本次抽样为{_gap_word}。"
        f"【③ 二者什么关系】「不分离」与「更强」并存且不矛盾：门槛问的是 cond 的残余预测力是否崩塌到 |偏|<0.3"
        f"（本次否，cond 同为负且量级相当），稿面问的是同一口径下 |偏相关| 谁更大。"
        f"故本文件既不支持「Ψ_n 与 cond 分离」，也不否证「Ψ_n 强于 cond」——后者是跨 6 次抽样的汇总主张，"
        f"逐次胜负与汇总由脚本 131 现算，见 04outputs/131aggregate_c5_partial_tally.json；本文件只承载本次抽样。"
    )
    # 【2026-08-10 订正，RYA-105】原实现把 ΔT 组内 p 值与食品体系的三个数字【硬编码】在本串里，
    # 上游 99 一改这些数字就静默过期（本轮 C-1/C-2 订正后实测已全部不符）。改为一律现算/现读。
    food = _load_food_99()
    _dtp = "、".join(f"ΔT={k.split('_')[1]} 组(n={v['n']}) p={v['p_nominal_dependent_subsets_do_not_cite']:.3f}"
                     for k, v in within.items())
    res["honest_boundaries"] = (
        f"(1) 单一真实数据集(苹果 Foods 2023)、{len(od)} 个\\emph{{非独立}}超定子集(温度重叠)、realCV 为测量 SD 传播"
        f"\\emph{{非}}独立重复，故为\\emph{{排序}}证据非强统计独立检验；"
        f"(2) 固定 ΔT 组内名义 p（子集重叠，正文不引用）：{_dtp}；"
        f"(3) 食品体系(99_realkinetics2)同质 Ψ_n 不直接复现(Spearman {food['psi_n']:+.2f}, p={food['psi_n_p']:.3f})、"
        f"需异方差 Ψ_het({food['psi_het']:+.2f}, p={food['psi_het_p']:.3f})——即 Ψ_n 优势\\emph{{条件于近等方差}}，强异方差须用 Ψ_het；"
        f"(4) 仅 C5(动力学)成立；C6(光谱线性恢复)κ_K 仍 ≈cond；"
        # 【2026-08-10 补，RYA-108】把判据的「所属主张」写进诚实边界，避免任何人把 false 读成对现行主张的否证。
        f"(5) `Psi_differentiates_from_cond_on_C5` 的阈值(|偏 Ψ_n|>0.5 且 |偏 cond|<0.3)属【旧】主张"
        f"「Ψ_n 与 cond 分离」的预注册门槛，非稿面现行主张「Ψ_n 强于 cond」的门槛；"
        f"C-2 订正(自助内层估计量改为与点估计同一的非线性最小二乘)后它判 false，本文因此放弃旧主张，"
        f"【未】为迎合结论调整该阈值；"
        f"(6) 本层顶层键与 `_standard_definition` 分属两个偏相关口径，稿面只用后者，两者的主次与本次是否同向"
        f"见 `partial_spearman_control_deltaT._convention`。"
    )

    # ---- (5) 统一异方差诊断 Ψ_het 跨 2 独立真实体系（苹果 computed + 食品 99 既有）----
    apple = apple_psi_het()
    res["unified_Psi_het_cross_system"] = {
        "apple_kinetics_computed": apple,
        "food_kinetics_99": {"Psi_het_spearman": food["psi_het"], "p": food["psi_het_p"],
                             "source": "04outputs/99_realkinetics2.json（现读，不再硬编码）",
                             "homoscedastic_Psi_n_fails": f"{food['psi_n']:+.2f} (p={food['psi_n_p']:.3f})"},
        "conclusion": (f"统一异方差 Fisher Ψ_het 跨 2 独立真实体系皆负相关："
                       f"苹果 Ψ_het={apple['apple_Psi_het_spearman_vs_realCV']:+.2f}"
                       f"(p={apple['apple_Psi_het_p']:.1e}, 异方差度 {apple['heteroscedasticity_sigma_max_over_min']:.1f}x)、"
                       f"食品 Ψ_het={food['psi_het']:+.2f}(p={food['psi_het_p']:.3f})；Ψ_n 是 Ψ_het 的等方差特例。"
                       f"'第二体系失败'实为用错诊断(同质 Ψ_n)，改用原则性 Ψ_het 则两体系一致。"
                       f"诚实：仅 2 体系、机理仍属绝对-vs-相对(robustness 非 novelty)。"),
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("=== C5 killer control: Ψ_n vs 廉价启发式 预测真实反演误差 ===")
    for k, v in marg.items():
        print(f"  {k:12s} |Spearman vs realCV| = {v['abs']:.3f}  (rho={v['spearman_vs_realCV']:+.3f}, p={v['p']:.1e})")
    print(f"  >> Ψ_n 边际严格胜全部启发式: {res['Psi_n_strictly_beats_all_heuristics_marginal']}")
    print("  固定 ΔT 组内 Spearman(Ψ_n, realCV):")
    for k, v in within.items():
        print(f"    {k}: {v['spearman_Psin_realCV']:+.3f} (n={v['n']}, cond组内仅变{v['cond_within_group_range_x']:.2f}x)")
    print(f"  偏 Spearman 控制 ΔT: Ψ_n={pn:+.3f}  vs  cond={pc:+.3f}")
    print(f"  >> Ψ_n 在 C5 上与 cond 区分(非≈cond): {res['Psi_differentiates_from_cond_on_C5']}")
    print(f"\n结论: {res['conclusion']}")
    print(f"诚实边界: {res['honest_boundaries']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
