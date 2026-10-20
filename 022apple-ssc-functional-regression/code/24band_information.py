# -*- coding: utf-8 -*-
"""3.5 节长波区失效的成因证据：各波段区间的单波段相关性、协方差谱有效维数与原始光谱信噪比。

为什么要有这个脚本：3.5 节与结论第 (4) 条引用的「|r|>0.2 占比」「有效维数」原先只有
05logs/17evidence_gaps.log 一份日志，没有可重跑的脚本，不满足「结论可追溯到脚本」的要求。
本脚本按论文所写的定义重算，并断言与该日志逐位一致。

定义（与论文 3.5 节一致）：
  单波段相关系数  r(λ) = corr(SNV+SG-1D 光谱在 λ 处的取值, SSC)，报告 |r| 的最大值、中位数与 >0.2 的占比。
  有效维数        取该区间协方差谱的指数熵：设协方差矩阵特征值 λ_i≥0，p_i=λ_i/Σλ_i，
                  则有效维数 = exp(−Σ p_i ln p_i)。全部方差集中在 1 个方向时为 1，
                  均匀分散到 m 个方向时为 m，故其值越小说明波段间越冗余。
  信噪比          原始（未预处理）强度的均值 ÷ 相邻波长差分的稳健标准差
                  （1.4826×中位绝对偏差，逐样本求出后取中位），与 12figures.py 图 4(c) 同一算法。
只读 05data。
"""
import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np, warnings; warnings.filterwarnings('ignore')
with open('02code/08fda_smallsample.py') as _src: exec(_src.read().split('# ---------- 实验一')[0])   # X 原始、Xs 预处理后、y、wl

def eff_dim(M):
    """协方差谱的指数熵。M 为 (n_samples, n_bands)。"""
    ev = np.linalg.eigvalsh(np.cov(M, rowvar=False))
    ev = ev[ev > 0]; p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def snr(M):
    d = np.diff(M, axis=1)
    return float(M.mean() / np.median(1.4826 * np.median(np.abs(d - np.median(d, 1, keepdims=True)), 1)))

SEGS = [(590, 700), (700, 900), (900, 1000), (1000, 1100), (1100, 1250), (590, 900), (1000, 1250)]
# 与 05logs/17evidence_gaps.log 第 20~26 行逐位对账（波段数、|r|最大、|r|中位、占比、有效维数）
REF = {(590, 700): (307, 0.364, 0.216, 61.2, 2.5), (700, 900): (590, 0.513, 0.278, 60.8, 1.9),
       (900, 1000): (312, 0.327, 0.134, 37.2, 4.9), (1000, 1100): (325, 0.204, 0.051, 0.3, 37.0),
       (1100, 1250): (514, 0.298, 0.131, 23.0, 6.4), (590, 900): (897, 0.513, 0.236, 61.0, 2.5),
       (1000, 1250): (839, 0.298, 0.093, 14.2, 11.4)}

print(f"{'区间/nm':>14} {'波段':>5} {'|r|最大':>8} {'|r|中位':>8} {'|r|>0.2 占比':>12} {'有效维数':>9} {'信噪比':>8}")
bad = 0
for lo, hi in SEGS:
    m = (wl >= lo) & (wl < hi); Z = Xs[:, m]
    r = np.abs(np.array([np.corrcoef(Z[:, j], y)[0, 1] for j in range(Z.shape[1])]))
    n, rmax, rmed, pct, ed = int(m.sum()), float(r.max()), float(np.median(r)), float((r > 0.2).mean() * 100), eff_dim(Z)
    print(f"{lo:>6}-{hi:<7} {n:>5} {rmax:>8.3f} {rmed:>8.3f} {pct:>11.1f}% {ed:>9.1f} {snr(X[:, m]):>8.0f}")
    ref = REF[(lo, hi)]
    for k, (a, b, nd) in enumerate(zip((n, rmax, rmed, pct, ed), ref, (0, 3, 3, 1, 1), strict=False)):
        if abs(round(a, nd) - round(b, nd)) > 1e-9:
            bad += 1; print(f"      不一致：第 {k+1} 项 本脚本 {a:.3f} 日志 {b:.3f}")
print(f"\n与 05logs/17evidence_gaps.log 对账：{'全部一致' if bad == 0 else str(bad)+' 处不一致'}")
assert bad == 0
