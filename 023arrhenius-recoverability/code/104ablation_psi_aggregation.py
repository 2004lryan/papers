"""
104ablation_psi_aggregation.py: Ψ_n 的【聚合方式消融】与【N 因子 remove 消融】(§7.5)

背景（为什么必须做）：
  Ψ_n = N·λ_min(Î) 的全部立论是「取 min 而非 det/trace，因为可恢复性是瓶颈问题、判据必须非补偿」。
  但全项目 grep 确认：从未有任何脚本计算过 Arrhenius 设计 Fisher 的 det(Î) 或 tr(Î⁻¹)。
  而 §相关工作 已把 D-最优摆上台面（Rodriguez-Aragón–López-Fidalgo, CILS 2005，同刊同模型）。
  故「为什么不用 D-最优」目前只有理论答案、没有一个数——本脚本补上。

消融设计（§7.5 的 replace 与 remove 两类）：
  [replace 聚合方式] 同一个 2×2 Fisher Î，换四种经典最优设计判据聚合：
      E-最优  λ_min(Î)              越大越好   ← 本文 Ψ_n 去掉 N 后即是它
      D-最优  det(Î) = λ_min·λ_max  越大越好   ← 经典主流，审稿人必问
      A-最优  tr(Î⁻¹) = 1/λ_min+1/λ_max 越小越好
      条件数  λ_max/λ_min           越小越好   ← 已有对照(91)，此处并列复算
  [remove N 因子] Ψ_n = N·λ_min  vs  λ_min 单独：N 是否承担判别力？

两路交叉核验（IRON RULE，已实战抓错一次，见 _load_54c5 docstring）：
  σ_lnk 未存入 CSV。以 σ=1 重建 Fisher 得 λ_min^(1)，则 σ² = λ_min^(1)/λ_min^(CSV)。
  若重建正确，该 σ 在全部 26 个子集上必须是【同一个常数】——这既恢复了 σ，又验证了重建口径。
  另比对 cond（σ 无关量）与 Ψ_n = n_temps·5·λ_min，全部须 <1e-9 相对偏差。

评价口径（与论文一致，不新增主张）：
  主口径 = 16 个超定子集(n_temps>=3，自由度>0，realCV_Ea 为真实统计估计误差)；
  附口径 = 全 26 子集。相关性一律用 Spearman 秩相关 + 控制 ΔT 的偏 Spearman。
  【不报名义 p】：子集相互嵌套、非独立，与论文 §「为何不报 p」同口径。

运行方式:
    cd <项目根目录绝对路径>
    python 02code/104ablation_psi_aggregation.py --device auto

输出文件:
    04outputs/104ablation_psi_aggregation.xlsx  — sheets: ablation / raw_criteria / verification
    04outputs/104ablation_psi_aggregation.json  — 机读汇总
    05logs/104ablation_psi_aggregation_YY-MM-DD_HHMMSS.log — 运行日志

方法论证锚点: refine-logs/75ABLATION_PSI_AGGREGATION_DESIGN.md（§1.9 四段式），台账 M51–M53。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from export_utils import get_logger, write_script_workbook, table_sheet_name

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"
CSV = OUT / "54c5_realkinetics-designsweep.csv"
N_PER_T = 5          # 每温度的时间点数，与 54c5 的 n_per_T=len(ts) 一致（仅作记录，见下）
# 【2026-08-10 订正，RYA-105】Arrhenius 反演阶段的数据点是「每温度一条 ln k̂」，其噪声尺度
# σ_lnk 已吸收该温度下的 N_PER_T 个时点，故该阶段的样本量 n = n_temps，而设计的总 Fisher
# 信息 I_total = Σ_i JᵢᵀJᵢ（54c5.fisher 的返回值本身）。据此 Ψ_n = n·λmin(Î₁) ≡ λmin(I_total)。
# 订正前本脚本按 N = n_temps·N_PER_T 乘在【已经跨温度求和】的 λmin 上，重复计了 n_temps
# （并二次计入 N_PER_T），且把该缺陷写成了 psi_relerr<1e-9 的硬断言。
STEM = Path(__file__).stem


def _load_54c5():
    """直接加载 54c5_realkinetics 并复用其 fisher/psi_design。

    【为什么必须 import 而不是重写】首版本脚本按源码重写了 fisher，但把 R_GAS 写成
    8.314462618 J/(mol·K)，而 54c5 用的是 8.314e-3 kJ/(mol·K)（因 Ea 以 kJ/mol 报）。
    单位不一致导致重建 cond 偏离真值 7e5 倍，被本脚本的交叉核验闸门当场拦下。
    根治办法是不再维护第二份实现——消融判据必须与被消融对象出自同一个 Fisher。
    """
    spec = importlib.util.spec_from_file_location("_m54", ROOT / "02code" / "54c5_realkinetics.py")
    if spec is None or spec.loader is None:
        raise ImportError("无法加载 54c5_realkinetics.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m54"] = mod
    spec.loader.exec_module(mod)
    for fn in ("fisher", "psi_design"):
        if not hasattr(mod, fn):
            raise ImportError(f"54c5_realkinetics 缺少 {fn}，口径无法对齐")
    return mod


M54 = _load_54c5()


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """控制 z 后 x 与 y 的偏 Spearman：三者各自秩变换，x、y 对 z 线性回归取残差再求 Pearson。"""
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(v).astype(float) for v in (x, y, z))
    Z = np.stack([rz, np.ones_like(rz)], axis=1)
    res = []
    for r in (rx, ry):
        beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
        res.append(r - Z @ beta)
    a, b = res
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="本脚本为纯 numpy 2x2 线性代数，device 仅记录不参与计算")
    ap.add_argument("--designsweep", default=None,
                    help="设计扫描 CSV 路径；不给则用 54c5 默认产物（默认行为与改动前一致）。"
                         "多种子复跑时指向 04outputs/105_seeds/54c5_realkinetics-designsweep-seed<N>.csv")
    ap.add_argument("--tag", default="", help="输出文件名后缀，用于区分多种子运行；空则沿用原文件名")
    args = ap.parse_args()
    csv_path = Path(args.designsweep) if args.designsweep else CSV
    out_stem = f"{STEM}{args.tag}"
    log = get_logger(STEM)
    log.log(f"=== {STEM} 启动 | device={args.device}（纯 CPU 线性代数） ===")

    if not csv_path.exists():
        raise FileNotFoundError(f"IRON RULE 违反：{csv_path} 不在盘")
    log.log(f"设计扫描输入: {csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) != 26:
        raise ValueError(f"IRON RULE 违反：子集数 {len(df)} != 描述 26")
    need = {"n_temps", "temps", "deltaT", "Psi_n", "lambda_min", "cond", "realCV_Ea"}
    if not need.issubset(df.columns):
        raise ValueError(f"IRON RULE 违反：列缺失，实得 {list(df.columns)}")
    log.log(f"载入 {len(df)} 个温度子集，列 {list(df.columns)}")

    # ---------- (0) 两路交叉核验：重建 Fisher，恢复 σ_lnk 并验证口径一致 ----------
    ver_rows = []
    sigmas = []
    lam_min_rec, lam_max_rec = [], []
    for r in df.itertuples():
        T_sub = [float(x) for x in str(r.temps).split("/")]
        ev = np.linalg.eigvalsh(M54.fisher(T_sub, 1.0))      # σ=1 的重建，用 54c5 权威实现
        lmin1, lmax1 = float(ev[0]), float(ev[1])
        sigma2 = lmin1 / float(r.lambda_min)                  # σ² = λmin^(σ=1) / λmin^(CSV)
        sigmas.append(np.sqrt(sigma2))
        lam_min_rec.append(lmin1 / sigma2)
        lam_max_rec.append(lmax1 / sigma2)
        ver_rows.append(dict(temps=r.temps, n_temps=int(r.n_temps),
                             cond_csv=float(r.cond), cond_rebuilt=lmax1 / lmin1,
                             sigma_recovered=float(np.sqrt(sigma2))))
    sigmas = np.asarray(sigmas)
    lam_min_rec = np.asarray(lam_min_rec)
    lam_max_rec = np.asarray(lam_max_rec)

    ver = pd.DataFrame(ver_rows)
    sigma_spread = float(sigmas.max() / sigmas.min() - 1.0)
    cond_relerr = float(np.max(np.abs(ver.cond_rebuilt - ver.cond_csv) / ver.cond_csv))
    psi_relerr = float(np.max(np.abs(lam_min_rec - df.Psi_n.values) / df.Psi_n.values))
    log.log(f"交叉核验 | σ 恢复值展布 {sigma_spread:.2e}（应≈0，证明重建口径正确，σ={float(sigmas.mean()):.6f}）")
    log.log(f"交叉核验 | cond 重建相对偏差 {cond_relerr:.2e} | Ψ_n=λmin(I_total) 相对偏差 {psi_relerr:.2e}")
    for name, val in (("sigma_spread", sigma_spread), ("cond_relerr", cond_relerr), ("psi_relerr", psi_relerr)):
        if not val < 1e-9:
            raise ValueError(f"交叉核验失败：{name}={val:.3e} 未 <1e-9，重建口径与 54c5 不一致，拒绝继续")
    log.log("交叉核验 3/3 通过（全部 <1e-9），重建 Fisher 与 54c5 同口径")

    # ---------- (1) 四种聚合判据 + N 因子 ----------
    # 【订正】lam_*_rec 是【总】信息阵 I_total 的特征值，Ψ_n = λmin(I_total)（= CSV 的 Psi_n）。
    # 本消融比较的是「带样本量权重的总信息阵口径」与「去权重的每观测口径 Î₁ = I_total/N」，
    # N = n_temps（Arrhenius 阶段每温度一条 ln k̂）。两条线并列，才答得了「N 因子有没有贡献」。
    N = df.n_temps.values.astype(float)
    lmin, lmax = lam_min_rec, lam_max_rec          # 总信息阵特征值
    crit = pd.DataFrame({
        "variety": df.variety if "variety" in df.columns else "",
        "temps": df.temps, "n_temps": df.n_temps, "deltaT": df.deltaT,
        "realCV_Ea": df.realCV_Ea,
        "Psi_n(E最优xN,本文)": lmin,                       # = λmin(I_total)，本文 Ψ_n
        "E最优 lambda_min(去N)": lmin / N,                   # = λmin(Î₁)，去样本量权重
        "D最优 det(I)": (lmin / N) * (lmax / N),             # det(Î₁)
        "D最优xN2": lmin * lmax,                             # N²·det(Î₁) = det(I_total)
        "A最优 tr(Iinv)": N * (1.0 / lmin + 1.0 / lmax),     # tr(Î₁⁻¹)
        "A最优xN inv": 1.0 / lmin + 1.0 / lmax,              # tr(Î₁⁻¹)/N = tr(I_total⁻¹)
        "条件数 cond": lmax / lmin,
        "deltaT(启发式)": df.deltaT.values.astype(float),
        "n_temps(启发式)": df.n_temps.values.astype(float),
    })

    # 方向：越大越可恢复 = "+"，越小越可恢复 = "-"。用于把 Spearman 统一折算为「越好则误差越小」
    DIRECTION = {
        "Psi_n(E最优xN,本文)": "+", "E最优 lambda_min(去N)": "+",
        "D最优 det(I)": "+", "D最优xN2": "+",
        "A最优 tr(Iinv)": "-", "A最优xN inv": "-",
        "条件数 cond": "-", "deltaT(启发式)": "+", "n_temps(启发式)": "+",
    }

    def evaluate(sub: pd.DataFrame, tag: str) -> list[dict]:
        rows = []
        y = sub.realCV_Ea.values.astype(float)
        z = sub.deltaT.values.astype(float)
        for name, sign in DIRECTION.items():
            x = sub[name].values.astype(float)
            sp = float(spearmanr(x, y).statistic)
            # 期望符号：判据越好(越大/越小) → 误差越小。折算为「有效相关」，负值=符合预期方向
            eff = sp if sign == "+" else -sp
            # 【2026-08-16 订正，RYA-492-B】控 ΔT 的偏相关对【ΔT 自己】按构造无定义：
            # rank(ΔT) 对 rank(ΔT) 回归的残差恒为 0，此前落盘的 ±0.001~0.06 全是双精度舍入噪声
            # （残差范数 1e-14），仅把行序颠倒就能从 +0.0059 跳到 -0.0397，跨运行也不可复现。
            # 凡与控制变量秩确定等价者一律置 None，不再落一个看似是数的噪声。
            degenerate = abs(abs(float(spearmanr(x, z).statistic)) - 1.0) < 1e-12
            if np.unique(z).size <= 1 or degenerate:
                psp = peff = None
            else:
                psp = round(partial_spearman(x, y, z), 4)
                peff = psp if sign == "+" else round(-psp, 4)
            rows.append(dict(口径=tag, 判据=name, 方向=sign,
                             Spearman=round(sp, 4), 有效相关=round(eff, 4),
                             偏Spearman_控ΔT=psp, 偏有效相关=peff,
                             n=len(sub)))
        return rows

    od = crit[crit.n_temps >= 3].copy()
    if len(od) != 16:
        raise ValueError(f"IRON RULE 违反：超定子集数 {len(od)} != 描述 16")
    abl = pd.DataFrame(evaluate(od, "16超定(主)") + evaluate(crit, "26全部(附)"))
    abl = abl.sort_values(["口径", "有效相关"])

    log.log("=== 聚合方式消融（主口径 16 超定子集；有效相关越负=越好地预测低误差）===")
    for r in abl[abl.口径 == "16超定(主)"].itertuples():
        pe = "  未定义（该判据与 ΔT 秩确定等价）" if r.偏有效相关 is None else f"{r.偏有效相关:+.3f}"
        log.log(f"  {r.判据:<24s} 有效相关 {r.有效相关:+.3f} | 控ΔT后 {pe}")

    # ---------- (2) N 因子 remove 消融的直接对照 ----------
    a = float(abl[(abl.口径 == "16超定(主)") & (abl.判据 == "Psi_n(E最优xN,本文)")].有效相关.iloc[0])
    b = float(abl[(abl.口径 == "16超定(主)") & (abl.判据 == "E最优 lambda_min(去N)")].有效相关.iloc[0])
    n_gain = a - b
    log.log(f"N 因子 remove 消融：Ψ_n={a:+.4f} vs λmin={b:+.4f} → N 的增量 {n_gain:+.4f}（N 取值 {sorted(set(N[df.n_temps.values >= 3].tolist()))}）")

    # ---------- (3) 组内判别力（固定 ΔT）----------
    grp_rows = []
    for dT, g in od.groupby("deltaT"):
        if len(g) < 3:
            continue
        for name, sign in DIRECTION.items():
            x = g[name].values.astype(float)
            spread = float(np.max(x) / np.min(x)) if np.min(x) > 0 else float("nan")
            sp = float(spearmanr(x, g.realCV_Ea.values).statistic)
            grp_rows.append(dict(deltaT=float(dT), n=len(g), 判据=name,
                                 组内展布倍数=round(spread, 3),
                                 组内Spearman=round(sp, 4),
                                 组内有效相关=round(sp if sign == "+" else -sp, 4)))
    grp = pd.DataFrame(grp_rows)

    best_main = abl[abl.口径 == "16超定(主)"].iloc[0]
    verdict = {
        "主口径最佳判据": str(best_main.判据),
        "本文Psi_n有效相关": a,
        "去N的lambda_min有效相关": b,
        "N因子增量": round(n_gain, 6),
        "N因子是否有贡献": bool(abs(n_gain) > 1e-9),
        "D最优有效相关": float(abl[(abl.口径 == "16超定(主)") & (abl.判据 == "D最优 det(I)")].有效相关.iloc[0]),
        "A最优有效相关": float(abl[(abl.口径 == "16超定(主)") & (abl.判据 == "A最优 tr(Iinv)")].有效相关.iloc[0]),
        "条件数有效相关": float(abl[(abl.口径 == "16超定(主)") & (abl.判据 == "条件数 cond")].有效相关.iloc[0]),
        "交叉核验": dict(sigma_recovered=float(sigmas.mean()), sigma_spread=sigma_spread,
                        cond_relerr=cond_relerr, psi_relerr=psi_relerr),
        "口径说明": "有效相关<0 表示该判据判为更好的设计确实对应更低的真实反演误差；不报名义 p（子集嵌套非独立）",
    }
    log.log("VERDICT: " + json.dumps(verdict, ensure_ascii=False, indent=2))

    # 【2026-08-16 订正，RYA-492-B】pandas 会把 None 转成 float NaN，而 json.dumps 默认写出裸
    # 字面量 NaN——那不是合法 JSON，严格解析器（含多数非 Python 读者）会直接报错。统一写 null。
    def _jsonable(recs: list[dict]) -> list[dict]:
        return [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
                for r in recs]

    (OUT / f"{out_stem}.json").write_text(
        json.dumps({"verdict": verdict,
                    "ablation": _jsonable(abl.to_dict("records")),
                    "within_group": _jsonable(grp.to_dict("records"))},
                   ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if args.tag:
        log.log(f"tag={args.tag}：仅输出 {out_stem}.json，不重写主 workbook（§4.2 一脚本一 xlsx）")
        log.log(f"DONE {out_stem}")
        return
    write_script_workbook(__file__, {
        table_sheet_name(1, "ablation"): abl,
        table_sheet_name(2, "raw_criteria"): crit,
        table_sheet_name(3, "within_group"): grp,
        table_sheet_name(4, "verification"): ver,
    })
    log.log(f"DONE {STEM}")


if __name__ == "__main__":
    main()
