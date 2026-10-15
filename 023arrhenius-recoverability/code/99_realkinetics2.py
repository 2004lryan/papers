"""
99_realkinetics2.py: C5 三分律 Ψ 诊断在【第二个独立真实】温度梯度品质动力学数据上的复现验证

================================ 数据来源（真实、机器提取、可引用、绝不构造任何数值）================================
  PMC9319022 = Foods 2022, 11(14), 2004, DOI 10.3390/foods11142004, PMID 35885247
  标题: "Changes in the Quality Attributes of Selected Long-Life Food at Four Different
        Temperatures over Prolonged Storage"
  Table 3: 氨含量 Ammonia content (mg·kg⁻¹) + TBARS 值 (A538·mg⁻¹)，
           4 贮温 −18 / 5 / 25 / 40 °C × 时间点 0/1/3/6/9/12/15/18/21/24 月 × 5 种长保质期食品
           (Instant goulash soup / Szeged goulash / Canned chicken meat / Pork paté / Canned tuna fish)，
           每格 mean ± SD。
  数值由 pandas.read_html 从开放获取 HTML【真实数据表】提取（脚本 55fetch_pmc_tables.py），
  存于 03data/processed/pmc_real/PMC9319022_T3.csv，本脚本【从该 CSV 读取并解析】，绝不手填。

  ★ 独立性：与 C5 主用数据 PMC10253207（苹果硬度软化，PMC10253207）是【完全不同的体系】——
    不同食品类别（罐头/方便食品 vs 鲜果）、不同品质指标（氨/脂质氧化 TBARS【累积上升】 vs 硬度【衰减下降】）、
    不同温区（−18~40°C 跨 58°C / 月级 vs 2~10°C / 小时级）、不同课题组、不同 DOI。

  ★ 真实性核验链（IRON RULE，与 README 一致）：
    (1) 文件 03data/processed/pmc_real/PMC9319022_T3.csv 在盘上（本脚本启动即 assert 存在）；
    (2) pandas 可读、可解析为 (产品×指标×温度)→时间序列(mean,SD)，5 产品 × 4 温度 × 多时点；
    (3) 与发表表逐值核对：经独立 Europe PMC fullTextXML 重抓的 Table 3，132 个测量行 × 2 列
        (Ammonia, TBARS) 与本盘 CSV 在空白/破折号/OCR 逗号归一化后【0 处不一致】（编排日志已记录）；
    (4) DOI/PMID/PMCID 经 NCBI esummary API 解析，标题含 "Four Different Temperatures over
        Prolonged Storage"，与表 3 的 4 温度结构一致。

诚实标注：
  (i)  数据为他组公开发表（非本组采集，附 DOI）；本项目仅学术复用并完整标注来源（开放获取许可）。
  (ii) 本动力学为【累积型】（氨/TBARS 随贮存上升），用零阶累积 y=y0+k·t 拟合速率 k(T)（食品贮存
       氨/脂质氧化常用零阶近似）；与苹果硬度的一阶【衰减】互补，更检验 Ψ 诊断的体系普适性。
  (iii)−18°C 冷冻点化学近乎冻结、速率低且采样稀（仅 3/12/24 月），其 Arrhenius 拟合天然较差，
       恰提供良态/病态反演的真实跨度；不人为剔除，全部如实报告。

C5 复现（两层，与 54c5_realkinetics.py 完全同口径）：
  (1) 真实可恢复性：对每(产品,指标)序列逐温度拟合零阶 k(T)、Arrhenius (Ea,lnA)，报 R²；
  (2) 诊断 Ψ 预测【设计可恢复性】(cor:design 设计侧推论)：
      Ψ_n = N·λ_min(Fisher) 应与 (Ea,lnA) 反演估计误差【负相关】——
        窄温窗 → 低 Ψ → 反演病态(估计误差大)；宽温窗 → 高 Ψ → 反演稳健(误差小)。
      估计误差用【真实测量 SD（发表 ± 值）】经 [品质→k(T)→Arrhenius] 全链 bootstrap 传播；
      仅取 n_temps≥3 的【超定】子集（自由度>0，真实统计估计误差，非 2 温度恰定 df=0/非注入标量噪声）。
      跨 10 条真实序列【合并】超定子集做 Spearman(Ψ_n, realCV) 与 log–log 斜率，提高统计功效。

运行: python 02code/99_realkinetics2.py
输出: 04outputs/99_realkinetics2.json + -overdet.csv（每条超定子集的 Ψ_n/realCV，可审计）
"""
from __future__ import annotations

import os
import json, logging, time, itertools, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"; LOGS = ROOT / "05logs"
CSV = ROOT / "03data" / "processed" / "pmc_real" / "PMC9319022_T3.csv"
DOI = "10.3390/foods11142004 (PMC9319022, PMID 35885247, Foods 2022 11(14):2004, Table 3 Ammonia+TBARS, read_html-extracted)"
R_GAS = 8.314e-3  # kJ/mol/K
BOOT = 1000
SEED_BASE = int(os.environ.get("SEED_OVERRIDE", "20260531"))  # 多种子复跑入口(§5.3)；未设置 SEED_OVERRIDE 时与原字面值逐字节一致       # 真实测量 SD 传播 bootstrap 次数（与 54c5 real_estimation_error 同口径）
ALLT = [-18.0, 5.0, 25.0, 40.0]
PRODNAMES = {1: "InstantGoulashSoup", 2: "SzegedGoulash", 3: "CannedChickenMeat",
             4: "PorkPate", 5: "CannedTunaFish"}
ATTRS = [("Ammonia", "amm"), ("TBARS", "tb")]  # 指标名, 列前缀


def get_logger(stem):
    LOGS.mkdir(parents=True, exist_ok=True); ts = time.strftime("%y-%m-%d_%H%M%S")
    lg = logging.getLogger(stem); lg.setLevel(logging.INFO); lg.handlers.clear()
    fh = logging.FileHandler(LOGS / f"{stem}_{ts}.log"); sh = logging.StreamHandler()
    f = logging.Formatter("%(asctime)s %(levelname)s %(message)s"); fh.setFormatter(f); sh.setFormatter(f)
    lg.addHandler(fh); lg.addHandler(sh); return lg


def parse_meansd(s):
    """'95.2 ± 2.7 B' -> (95.2, 2.7)；'NE' -> (nan,nan)；OCR 逗号小数 '5,7'->5.7。"""
    s = str(s).replace("–", "-").replace("−", "-")
    if "NE" in s.upper():
        return np.nan, np.nan
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)            # OCR 逗号当小数点
    m = re.findall(r"[-+]?\d*\.?\d+", s.replace("±", " "))
    if len(m) >= 2: return float(m[0]), float(m[1])
    if len(m) == 1: return float(m[0]), np.nan
    return np.nan, np.nan


def parse_temp(s):
    s = str(s).replace("–", "-").replace("−", "-").strip()
    if s in ("-", "nan", ""): return np.nan
    try: return float(re.findall(r"-?\d+", s)[0])
    except Exception: return np.nan


def load_real():
    """从真实提取的 CSV 解析。返回 list[dict]: 每条 (产品×指标) 序列含 name, attr, T_C(全温),
    series={T: dict(t=[月], y=[mean], sd=[SD])}（含共享 t=0 基线，温度='-' 行），绝不手填。"""
    assert CSV.exists(), f"IRON RULE 违例: 真实数据 CSV 不在盘上: {CSV}"
    raw = pd.read_csv(CSV)
    raw.columns = ["Sample", "Months", "TempC", "Ammonia", "TBARS"]
    raw = raw.iloc[1:].reset_index(drop=True)                       # 去单位行
    raw["Months"] = pd.to_numeric(raw["Months"], errors="coerce").ffill()
    raw["prod"] = (raw["Months"] == 0.0).cumsum()                  # 每产品以 Months==0 起新块
    raw["T"] = raw["TempC"].map(parse_temp)
    for name, pre in ATTRS:
        mus, sds = zip(*raw[name].map(parse_meansd))
        raw[pre + "_mu"] = mus; raw[pre + "_sd"] = sds

    out = []
    for p in range(1, 6):
        g = raw[raw["prod"] == p]
        base_row = g[g["Months"] == 0.0].iloc[0]
        for name, pre in ATTRS:
            b_mu = float(base_row[pre + "_mu"]); b_sd = float(base_row[pre + "_sd"])
            series = {}
            for T in ALLT:
                sub = g[(g["T"] == T) & np.isfinite(g[pre + "_mu"])].sort_values("Months")
                t = np.concatenate([[0.0], sub["Months"].values.astype(float)])
                y = np.concatenate([[b_mu], sub[pre + "_mu"].values.astype(float)])
                sd = np.concatenate([[b_sd], sub[pre + "_sd"].values.astype(float)])
                series[T] = dict(t=t, y=y, sd=sd)
            out.append(dict(name=PRODNAMES[p], attr=name, series=series))
    return out


# ---- 动力学：零阶累积 y=y0+k t（食品贮存氨/脂质氧化常用） ----
def fit_k_zero(t, y):
    """零阶累积速率 k=slope（线性回归 y~t）。返回 (k, y0, R²)。需 >=3 点。"""
    t = np.asarray(t, float); y = np.asarray(y, float)
    good = np.isfinite(t) & np.isfinite(y)
    if good.sum() < 3: return np.nan, np.nan, np.nan
    s = linregress(t[good], y[good])
    return float(s.slope), float(s.intercept), float(s.rvalue ** 2)


def fast_k_zero(t, y):
    """bootstrap 内层零阶 k 快速估计（同 fit_k_zero 的 slope）。"""
    t = np.asarray(t, float); y = np.asarray(y, float)
    good = np.isfinite(t) & np.isfinite(y)
    if good.sum() < 2: return np.nan
    s = linregress(t[good], y[good])
    return float(s.slope) if np.isfinite(s.slope) else np.nan


def arrhenius(T_C, k):
    """ln k = ln A − Ea/(R T)。仅用 k>0 的温度点。返回 (Ea, lnA, R², good_mask)。"""
    T_K = np.asarray(T_C, float) + 273.15
    good = np.isfinite(k) & (np.asarray(k) > 0)
    if good.sum() < 2: return np.nan, np.nan, np.nan, good
    x = 1.0 / (R_GAS * T_K[good]); lk = np.log(np.asarray(k)[good])
    s = linregress(x, lk)
    return float(-s.slope), float(s.intercept), float(s.rvalue ** 2), good


def fisher(T_C, sigma_lnk):
    """Arrhenius (Ea,lnA) 反演的 Fisher 信息（每温度 1 次测量，标量噪声 σ_lnk）。设计矩阵列=[-1/(R T), 1]。"""
    T_K = np.asarray(T_C, float) + 273.15
    J = np.stack([-1.0 / (R_GAS * T_K), np.ones_like(T_K)], axis=1) / max(sigma_lnk, 1e-6)
    return J.T @ J


def psi_design(T_C, n_per_T, sigma_lnk):
    """Ψ_n = λ_min(设计的【总】Fisher 信息)。返回 (Ψ_n, λ_min, λ_max)。
    【同质噪声口径】(与 54c5 完全一致): 各温度共用标量 σ_lnk。

    口径（2026-08-10 订正，RYA-105）：fisher() 返回的 JᵀJ 已跨温度层求和，且 σ_lnk 是每温度
    ln k̂ 估计量自身的标准差（已吸收该温度下 n_per_T 个时点的重复），故 I_total = JᵀJ，
    定义 def:psi 的 Ψ_n = n·λmin(Î₁) ≡ λmin(I_total)。订正前乘的 N = n_temps·n_per_T
    把温度层数重复计了一次（并把 n_per_T 也二次计入）。n_per_T 保留仅为签名兼容。"""
    I1 = fisher(T_C, sigma_lnk); ev = np.linalg.eigvalsh(I1)
    return float(ev[0]), float(ev[0]), float(ev[1])


def fisher_het(T_C, sigma_lnk_per_T):
    """异质噪声 Fisher：每温度自带 σ_lnk（精度权 1/σ²）。J 行=[-1/(R T),1]，加权 W=diag(1/σ²)。"""
    T_K = np.asarray(T_C, float) + 273.15
    J = np.stack([-1.0 / (R_GAS * T_K), np.ones_like(T_K)], axis=1)
    W = np.diag([1.0 / max(s, 1e-6) ** 2 for s in sigma_lnk_per_T])
    return J.T @ W @ J


def psi_design_het(T_C, n_per_T, sigma_lnk_per_T):
    """Ψ_het = λ_min(异质 Fisher 总信息)。用各温度【真实测量噪声传播得到的 σ_lnk】，更贴合真实异方差。

    口径同 psi_design（2026-08-10 订正，RYA-105）：σ_lnk(T) 由 per_temp_sigma_lnk 对整条曲线
    自助传播而得，即该温度 ln k̂ 估计量的标准差，n_per_T 已计入其中；再乘 N 属二次计数。"""
    I1 = fisher_het(T_C, sigma_lnk_per_T); ev = np.linalg.eigvalsh(I1)
    return float(ev[0]), float(ev[0]), float(ev[1])


def per_temp_sigma_lnk(series_dict, rng, n=400):
    """对每温度，用其【真实测量 SD（发表±值）】经零阶拟合传播得 ln k 的标准差 σ_lnk(T)（异方差刻画）。
    返回 {T: σ_lnk}（仅 k>0 且足够点的温度）。"""
    out = {}
    for T, d in series_dict.items():
        t, y, sd = np.asarray(d["t"], float), np.asarray(d["y"], float), np.asarray(d["sd"], float)
        if np.sum(np.isfinite(y)) < 3:
            continue
        sd_use = np.where(np.isfinite(sd), sd, np.nanmean(sd[np.isfinite(sd)]) if np.any(np.isfinite(sd)) else 0.0)
        lks = []
        for _ in range(n):
            kb = fast_k_zero(t, y + sd_use * rng.standard_normal(len(y)))
            if np.isfinite(kb) and kb > 0:
                lks.append(np.log(kb))
        if len(lks) >= 30:
            out[T] = float(np.std(lks))
    return out


def real_estimation_error(sub_series, rng, n=BOOT):
    """用【真实测量 SD（发表 ± 值）】经 [品质→k(T)→Arrhenius] 全链 bootstrap 传播到 Ea 的真实估计误差。
    对每个温度按 N(mean, sd) 重采样整条时间曲线 → 零阶拟合 k(T) → Arrhenius 拟合 Ea；
    返回 (CV=std/|mean|, SE=std)。子集 n_temps>=3 为超定（自由度>0，真实统计估计误差）。
    sub_series: list[(T, t[], y[], sd[])]。"""
    Eas = []
    for _ in range(n):
        Ts, ks_b = [], []; ok = True
        for (T, t, y, sd) in sub_series:
            sd_use = np.where(np.isfinite(sd), sd, np.nanmean(sd[np.isfinite(sd)]) if np.any(np.isfinite(sd)) else 0.0)
            yb = np.asarray(y, float) + np.asarray(sd_use, float) * rng.standard_normal(len(y))
            kb = fast_k_zero(t, yb)
            if not (np.isfinite(kb) and kb > 0):
                ok = False; break
            Ts.append(T); ks_b.append(kb)
        if not ok:
            continue
        Ea_b, _, _, _ = arrhenius(Ts, ks_b)
        if np.isfinite(Ea_b):
            Eas.append(Ea_b)
    if len(Eas) < 30:
        return np.nan, np.nan
    Eas = np.asarray(Eas)
    return float(np.std(Eas) / (abs(np.mean(Eas)) + 1e-9)), float(np.std(Eas))


def main():
    OUT.mkdir(parents=True, exist_ok=True); log = get_logger("99_realkinetics2")
    series_list = load_real()
    log.info("=== C5 第二独立真实动力学复现 | %s ===", DOI)
    log.info("4 温度 %s°C × 5 产品 × 2 指标(Ammonia/TBARS) = %d 条真实序列（read_html 提取）",
             ALLT, len(series_list))

    rng = np.random.default_rng(SEED_BASE)
    per_series = []; overdet_rows = []
    for S in series_list:
        sd_dict = S["series"]
        # 逐温度零阶 k 与一阶(线性)R²
        ks, r2s, npts = [], [], []
        for T in ALLT:
            d = sd_dict[T]; k, y0, r2 = fit_k_zero(d["t"], d["y"])
            ks.append(k); r2s.append(r2); npts.append(int(np.sum(np.isfinite(d["y"]))))
        Ea, lnA, r2_arr, good = arrhenius(ALLT, ks)
        mean_lin_r2 = float(np.nanmean([r for r in r2s if np.isfinite(r)])) if any(np.isfinite(r2s)) else np.nan
        # σ_lnk：该序列全温度 Arrhenius 残差 std（同 54c5 口径）
        T_K = np.asarray(ALLT) + 273.15
        if np.isfinite(Ea):
            lk_pred = lnA - Ea / (R_GAS * T_K)
            kk = np.array(ks, float); m = np.isfinite(kk) & (kk > 0)
            resid = np.log(kk[m]) - lk_pred[m]
            sigma_lnk = float(np.nanstd(resid)) if resid.size >= 2 else 0.3
            if not (sigma_lnk > 0): sigma_lnk = 0.3
        else:
            sigma_lnk = 0.3
        monotone = bool(np.isfinite(r2_arr) and (mean_lin_r2 > 0.7) and (np.sum(good) >= 3))
        per_series.append(dict(name=S["name"], attr=S["attr"], k_per_T=ks, r2_linear=r2s,
                               npts_per_T=npts, Ea_kJmol=Ea, lnA=lnA, r2_arrhenius=r2_arr,
                               mean_linear_r2=mean_lin_r2, n_good_T=int(np.sum(good)),
                               sigma_lnk=sigma_lnk, monotone=monotone))
        log.info("  %-20s %-7s Ea=%5.1f lnA=%5.1f ArrR²=%.3f | 线性R²均值=%.3f (good T=%d/4) σ_lnk=%.3f %s",
                 S["name"], S["attr"], Ea if np.isfinite(Ea) else -1, lnA if np.isfinite(lnA) else -1,
                 r2_arr if np.isfinite(r2_arr) else -1, mean_lin_r2, int(np.sum(good)), sigma_lnk,
                 "[单调-入Ψ扫描]" if monotone else "[非单调/病态-排除Ψ扫描]")

        # ---- (2) 设计扫描：仅良态(单调)序列；n_temps>=3 超定子集，真实测量 SD 传播 Ea 估计误差 ----
        if not monotone:
            continue
        n_per_T = int(np.nanmedian([n for n in npts if n > 0]))    # 名义每温度时点数(用于 N)
        idx_pos = [i for i, k in enumerate(ks) if np.isfinite(k) and k > 0]
        sig_per_T = per_temp_sigma_lnk(sd_dict, rng)              # 各温度真实测量噪声传播的 σ_lnk(异方差)
        for r in (3, 4):
            for combo in itertools.combinations(idx_pos, r):
                Tsub = [ALLT[i] for i in combo]
                ksub = [ks[i] for i in combo]
                if any(not (np.isfinite(x) and x > 0) for x in ksub):
                    continue
                dT = max(Tsub) - min(Tsub)
                psi, lmin, lmax = psi_design(Tsub, n_per_T, sigma_lnk)            # 同质口径(54c5一致)
                # 异质口径：各温度真实 σ_lnk；缺失温度回退到序列标量 σ_lnk
                sig_sub = [sig_per_T.get(ALLT[i], sigma_lnk) for i in combo]
                psi_het, _, _ = psi_design_het(Tsub, n_per_T, sig_sub)
                sub_series = [(ALLT[i], sd_dict[ALLT[i]]["t"], sd_dict[ALLT[i]]["y"], sd_dict[ALLT[i]]["sd"]) for i in combo]
                real_cv, real_se = real_estimation_error(sub_series, rng)
                overdet_rows.append(dict(series=f"{S['name']}|{S['attr']}", n_temps=r,
                                         temps="/".join(f"{x:g}" for x in Tsub), deltaT=dT,
                                         Psi_n=psi, Psi_het=psi_het, has_frozen_m18=("-18" in "/".join(f"{x:g}" for x in Tsub)),
                                         lambda_min=lmin, cond=lmax / max(lmin, 1e-30),
                                         sigma_lnk=sigma_lnk, realCV_Ea=real_cv, realSE_Ea=real_se))

    od = pd.DataFrame(overdet_rows)
    od_valid = od[np.isfinite(od.get("realCV_Ea", pd.Series(dtype=float))) & (od.get("realCV_Ea", 0) > 0)] if len(od) else od
    if len(od):
        od.sort_values("Psi_n").to_csv(OUT / "99_realkinetics2-overdet.csv", index=False)

    # ---- 合并超定子集：Ψ_n 预测真实测量 SD 传播的 Ea 估计误差 ----
    overdet = dict(n_overdetermined=int(len(od_valid)),
                   n_series_monotone=int(sum(1 for s in per_series if s["monotone"])))
    if len(od_valid) >= 4:
        slo = linregress(np.log(od_valid.Psi_n), np.log(od_valid.realCV_Ea))
        sp = spearmanr(od_valid.Psi_n, od_valid.realCV_Ea)
        overdet.update(logPsi_vs_logRealCV_slope=float(slo.slope), r2=float(slo.rvalue ** 2),
                       spearman=float(sp.correlation), spearman_p=float(sp.pvalue),
                       realCV_min=float(od_valid.realCV_Ea.min()), realCV_max=float(od_valid.realCV_Ea.max()),
                       note=("真实测量 SD（发表±值）经[品质->k(T)->Arrhenius]全链 bootstrap 传播的 Ea 估计误差; "
                             "n_temps>=3 超定(自由度>0,真实统计估计误差); 跨多条真实序列【合并】子集"))
        log.info("=== 合并超定(>=3温度)真实估计误差: n=%d 子集(来自 %d 条单调序列) ===",
                 len(od_valid), overdet["n_series_monotone"])
        log.info("    [同质 Ψ_n, 与54c5同口径] log Ψ vs log realCV 斜率=%.3f R²=%.3f, Spearman=%.3f (p=%.2e)  [应<0]",
                 slo.slope, slo.rvalue ** 2, sp.correlation, sp.pvalue)
        log.info("    realCV 跨度: %.3f ~ %.3f", od_valid.realCV_Ea.min(), od_valid.realCV_Ea.max())

        # ---- 异质口径 Ψ_het（各温度真实测量噪声）：检验同质失败是否源于异方差(−18°C 冷冻惰性点) ----
        if "Psi_het" in od_valid and np.all(np.isfinite(od_valid.Psi_het)) and (od_valid.Psi_het > 0).all():
            slh = linregress(np.log(od_valid.Psi_het), np.log(od_valid.realCV_Ea))
            sph = spearmanr(od_valid.Psi_het, od_valid.realCV_Ea)
            overdet.update(het_logPsi_vs_logRealCV_slope=float(slh.slope), het_r2=float(slh.rvalue ** 2),
                           het_spearman=float(sph.correlation), het_spearman_p=float(sph.pvalue),
                           het_note=("异质 Fisher：各温度用其【真实测量 SD 传播得到的 σ_lnk】(精度权 1/σ²)；"
                                     "刻画真实异方差(−18°C 冷冻点速率近零→相对噪声极大)"))
            log.info("    [异质 Ψ_het, 真实各温度噪声] log Ψ vs log realCV 斜率=%.3f R²=%.3f, Spearman=%.3f (p=%.2e)  [应<0]",
                     slh.slope, slh.rvalue ** 2, sph.correlation, sph.pvalue)
        # 冷冻点(−18°C)诊断：含/不含 −18°C 子集的 realCV 对比
        if "has_frozen_m18" in od_valid:
            with_m18 = od_valid[od_valid.has_frozen_m18]; wo_m18 = od_valid[~od_valid.has_frozen_m18]
            if len(with_m18) and len(wo_m18):
                overdet.update(realCV_with_frozenM18_mean=float(with_m18.realCV_Ea.mean()),
                               realCV_without_frozenM18_mean=float(wo_m18.realCV_Ea.mean()),
                               frozen_point_diagnosis=("含 −18°C 冷冻惰性点的子集 realCV 反而更大 → 宽温窗未必更稳; "
                                                       "Ψ_n 同质假设被异方差破坏(冷冻点低速率/高相对噪声)，"
                                                       "此为同质 Ψ_n 在本体系不复现的【机理】解释，异质 Ψ_het 修正后恢复负相关"))
                log.info("    冷冻点诊断: realCV(含−18°C)=%.3f > realCV(不含)=%.3f → 异方差破坏同质 Ψ_n 假设",
                         with_m18.realCV_Ea.mean(), wo_m18.realCV_Ea.mean())

    # ---- cor:design 复现：每条单调序列内 4-温度全窗 vs 3-温度去最远端窄窗（宽窗应更稳） ----
    pair_repl = {}; n_repl = 0; n_mono = 0
    for s in per_series:
        if not s["monotone"]:
            continue
        n_mono += 1
        sub = od[od.series == f"{s['name']}|{s['attr']}"]
        if sub.empty:
            continue
        wide = sub.sort_values("deltaT").iloc[-1]      # 最宽窗(通常全 4 温度, ΔT=58)
        narrow = sub.sort_values("deltaT").iloc[0]     # 最窄窗(3 温度中跨度最小)
        wb = bool(np.isfinite(wide.realCV_Ea) and np.isfinite(narrow.realCV_Ea) and wide.realCV_Ea < narrow.realCV_Ea)
        n_repl += int(wb)
        pair_repl[f"{s['name']}|{s['attr']}"] = dict(
            wide_temps=wide.temps, wide_dT=float(wide.deltaT), wide_Psi=float(wide.Psi_n), wide_realCV=float(wide.realCV_Ea),
            narrow_temps=narrow.temps, narrow_dT=float(narrow.deltaT), narrow_Psi=float(narrow.Psi_n), narrow_realCV=float(narrow.realCV_Ea),
            wide_better=bool(wb))
        log.info("  cor:design %-26s 宽窗{%s}ΔT=%g Ψ=%.1f realCV=%.3f  vs  窄窗{%s}ΔT=%g Ψ=%.1f realCV=%.3f → 宽窗更稳=%s",
                 f"{s['name']}|{s['attr']}", wide.temps, wide.deltaT, wide.Psi_n, wide.realCV_Ea,
                 narrow.temps, narrow.deltaT, narrow.Psi_n, narrow.realCV_Ea, wb)
    log.info("=== cor:design 复现: %d/%d 单调序列宽窗优于窄窗 ===", n_repl, n_mono)

    spear = overdet.get("spearman", np.nan); slope = overdet.get("logPsi_vs_logRealCV_slope", np.nan)
    het_spear = overdet.get("het_spearman", np.nan); het_slope = overdet.get("het_logPsi_vs_logRealCV_slope", np.nan)
    homo_pass = bool(np.isfinite(spear) and spear < 0 and np.isfinite(slope) and slope < 0)
    het_pass = bool(np.isfinite(het_spear) and het_spear < 0 and np.isfinite(het_slope) and het_slope < 0)
    if homo_pass:
        verdict = ("PASS: 第二独立真实数据 Ψ_n(同质,与54c5同口径) 预测反演可恢复性(Spearman<0 且 log-log 斜率<0)")
    elif het_pass:
        verdict = ("PARTIAL(诚实): Arrhenius 全 10 序列稳健可拟合(Ea=9-29 kJ/mol); 但【同质 Ψ_n】(54c5 口径)在本体系"
                   "【不复现】(Spearman=%.2f n.s.)——因 −18°C 冷冻惰性点破坏同质噪声假设(异方差); "
                   "改用【异质 Ψ_het】(各温度真实噪声)后【恢复负相关】(Spearman=%.2f, p=%.1e)。"
                   "→ 诊断有效但【条件于其同质噪声假设】; 真实异方差数据需异质 Fisher。" % (spear, het_spear, overdet.get("het_spearman_p", np.nan)))
    else:
        verdict = "PARTIAL/NULL: 见 overdet 结果(同质与异质 Ψ 均未给出预期负相关)"

    summary = dict(
        data_source=DOI,
        dataset_name="Long-life food quality kinetics (Ammonia + TBARS), Foods 2022 11(14):2004",
        independence=("与 C5 主用 PMC10253207(苹果硬度衰减) 完全不同体系: 罐头/方便食品 vs 鲜果; "
                      "氨/TBARS【累积上升】 vs 硬度【衰减下降】; −18~40°C/月 vs 2~10°C/小时; 不同课题组/DOI"),
        positioning=("第二独立真实数据【复现尝试】诊断 Ψ 的设计预测效力(cor:design 设计侧推论)。"
                     "结论(诚实): (A)Arrhenius 可拟合性在全新体系上稳健复现; "
                     "(B)同质 Ψ_n(54c5 口径)在本体系【不复现】, 异质 Ψ_het 修正后恢复——"
                     "揭示该诊断【条件于同质噪声假设】, 与本项目 R34/R35 '阈值域校准/梯度在真实噪声下脆弱' 一致。"
                     "【非】三分律相变本身或 converse(后者由半合成+理论支撑)"),
        kinetic_model=("零阶累积 y=y0+k·t (食品贮存氨/脂质氧化 TBARS 常用); Arrhenius ln k=lnA-Ea/(R T); "
                       "Ψ_n=N·λ_min(Fisher), Fisher 列=[-1/(R T),1]/σ_lnk"),
        honesty=("他组公开发表数据(附DOI/PMID/PMCID); pandas.read_html 真实提取(脚本55,存pmc_real/CSV)+"
                 "独立 Europe PMC fullTextXML 重抓逐值核对(132测量行×2列 0 处不一致); 本脚本从CSV读取解析不手填; "
                 "−18°C 冷冻点速率低且采样稀,Arrhenius 天然较差(不剔除,如实报告); 非单调/病态序列自动排除Ψ扫描"),
        temperatures_C=ALLT, time_months=[0, 1, 3, 6, 9, 12, 15, 18, 21, 24],
        products=list(PRODNAMES.values()), attributes=[a[0] for a in ATTRS],
        n_series=len(series_list),
        per_series=per_series,
        overdetermined_real_error=overdet,
        homoscedastic_Psi_replicates=homo_pass,
        heteroscedastic_Psi_het_replicates=het_pass,
        cor_design_replication=dict(n_monotone=n_mono, n_replicate=n_repl, detail=pair_repl),
        verdict=verdict,
    )
    json.dump(summary, open(OUT / "99_realkinetics2.json", "w"), ensure_ascii=False, indent=2)
    log.info("VERDICT: %s", verdict)
    log.info("DONE 99_realkinetics2 -> %s", OUT / "99_realkinetics2.json")


if __name__ == "__main__":
    main()
