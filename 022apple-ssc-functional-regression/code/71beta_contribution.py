# -*- coding: utf-8 -*-
"""
71beta_contribution.py: FPLR 部署模型的预测值按波段的分解（稿件 2.5 节，事后描述）

为什么要有这个脚本：2.5 节按回归系数函数绝对值最大的 400 个波长点划出权重窗口。系数的大小同时取决于该处光谱的尺度：
预处理后的光谱在 900 nm 以上样品间的变化很小（图 2b），系数大不等于对预测值的影响大。不把这一点拆开，
「模型的权重集中在哪里」会被读成「预测值由哪里决定」。本脚本把部署模型的预测值逐波长分解，给出各窗口的贡献。
这一分析在看到系数函数与平均光谱之后才提出，属事后描述（协议 C3 修订记录 2026-10-04 行）。

做法：
  1. 与 48rebuttal_R1_13_bspline_beta.py 相同：16unified_comparison.py 的共享段读数与预处理，三次 B 样条基 L=80，
     展开系数 C = to_coef(Xp, B, R)
  2. 部署模型的 β(λ) 取自 04outputs/48rebuttal_beta.csv。FPLR 的预测值为 ȳ + c_i′Jb，其中 J 由 gram_and_pen 按波长网格求积，
     故 c_i′Jb = Σ_j dt_j·x̃_i(t_j)·β(t_j)，x̃_i = B c_i 为样品 i 的光谱函数在网格上的取值。逐点贡献 g_ij = dt_j·x̃_i(t_j)·β(t_j)
     核对：以 48rebuttal_deploy.csv 选中的 λβ 在全部样品上重拟合，其预测值与 ȳ + Σ_j g_ij 的最大绝对差须小于 1e-6，否则停下
  3. 波段 w 的贡献 G_iw = Σ_{j∈w} g_ij。报告 G_w 的样品间标准差（°Brix），G_w 与预测值的协方差占预测值方差的比例
     （对波长的一个划分，各波段的比例之和为 1；波段之间的贡献相关时，单个比例可以为负或大于 1），
     以及 G_w 与其余全部波长的贡献之和（预测值减去 ȳ 再减去 G_w）的相关系数：负值表示两者此消彼长。
     标准差、方差与协方差一律取样本口径（除以 n−1），与表 1 及稿件其余各处的标准差相同
  4. 波段的取法：
       beta   48rebuttal_beta_windows.csv 中按 |β| 最大的 400 个点划出的窗口，及其余波长
       contrib 逐点贡献的样品间标准差最大的 400 个点，按与 48 相同的规则（相邻间隔大于 15 nm 断开、保留点数大于 15 的）
               划出的窗口，及其余波长

运行方式:
    cd <项目根目录>
    python3 02code/71beta_contribution.py > 05logs/71beta_contribution_<YY-MM-DD_HHMMSS>.log 2>&1

输入文件（只读）:
    05data 下两个年份的 Excel、04outputs/48rebuttal_beta.csv、48rebuttal_beta_windows.csv、48rebuttal_deploy.csv
输出文件:
    04outputs/71beta_contribution.csv            — 各波段：取法、起止波长、网格点数、贡献的标准差、协方差占比、
                                                   与其余波长贡献之和的相关系数、该波段光谱与系数的尺度
    04outputs/71beta_contribution_pointwise.csv  — 逐波长：β、光谱函数的样品间标准差、逐点贡献的标准差
    04outputs/71beta_contribution_samples.csv    — 逐样品：预测值减 ȳ，以及两种取法下各波段的贡献（上表各量由它算出，
                                                   42recompute_manuscript_numbers.py 据此复算稿件数字）
    05logs/71beta_contribution_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）
"""
import ast

import numpy as np
import pandas as pd

with open('02code/16unified_comparison.py') as _src:
    _SHARED = _src.read().split("METHODS=['PLSR'")[0]
exec(_SHARED)                                # 注入 wl、t、Xp、y 与 bspline_basis、gram_and_pen、to_coef、pflr

L_BASIS = 80
B = bspline_basis(t, L_BASIS)
J, R = gram_and_pen(B, t)
C = to_coef(Xp, B, R)
bt = pd.read_csv('04outputs/48rebuttal_beta.csv')
assert np.allclose(bt.wavelength_nm.to_numpy(), wl)
beta = bt.beta_deploy.to_numpy()
dt = np.gradient(t)
Xf = C @ B.T                                 # 光谱函数在波长网格上的取值，n × 2 048
g = Xf * beta * dt                           # 逐点贡献
total = g.sum(1)

dep = pd.read_csv('04outputs/48rebuttal_deploy.csv')
picked = ast.literal_eval(dep[dep.method == 'FPLR'].picked.iloc[0])
assert picked[0] == L_BASIS
pred = pflr(C, y, C, J, R, picked[1])
gap = float(np.max(np.abs(pred - (y.mean() + total))))
print(f'部署模型（L={picked[0]}，λβ={picked[1]:g}）在全部 {len(y)} 个样品上的预测值与 ȳ+Σg 的最大绝对差：{gap:.2e}')
assert gap < 1e-6, '逐点分解与部署模型的预测值不一致'
var_t = float(total.var(ddof=1))
print(f'预测值的样品间标准差 {total.std(ddof=1):.4f} °Brix')


per_sample = {'prediction_minus_ybar': total}


def band(kind, lo, hi, mask, n_top):
    G = g[:, mask].sum(1)
    per_sample[kind if kind.endswith('_rest') else f'{kind}_{lo:.2f}_{hi:.2f}'] = G
    return {'kind': kind, 'lo': lo, 'hi': hi, 'n_grid': int(mask.sum()), 'n_top': n_top, 'sd_contribution': float(G.std(ddof=1)),
            'cov_share': float(np.cov(G, total)[0, 1] / var_t),
            'corr_with_other_wavelengths': float(np.corrcoef(G, total - G)[0, 1]),
            'median_sd_x': float(np.median(Xf[:, mask].std(0, ddof=1))), 'median_abs_beta': float(np.median(np.abs(beta[mask])))}


rows = []
# ---- (a) 按 |β| 划出的窗口 ----
bw = pd.read_csv('04outputs/48rebuttal_beta_windows.csv')
bw = bw[bw.kind == '绝对值前400点']
rest = np.ones(len(wl), bool)
for r in bw.itertuples():
    m = (wl >= r.lo) & (wl <= r.hi)
    rows.append(band('beta', r.lo, r.hi, m, int(r.n_points)))
    rest &= ~m
rows.append(band('beta_rest', float('nan'), float('nan'), rest, 0))
# ---- (b) 按逐点贡献的标准差划出的窗口（规则同 48） ----
s = g.std(0, ddof=1)
top = np.sort(wl[np.argsort(-s)[:400]])
rest = np.ones(len(wl), bool)
for seg in np.split(top, np.where(np.diff(top) > 15)[0] + 1):
    if len(seg) > 15:
        m = (wl >= seg.min()) & (wl <= seg.max())
        rows.append(band('contrib', float(seg.min()), float(seg.max()), m, len(seg)))
        rest &= ~m
rows.append(band('contrib_rest', float('nan'), float('nan'), rest, 0))
out = pd.DataFrame(rows)
for kind in ('beta', 'contrib'):
    tot = out[out.kind.isin([kind, kind + '_rest'])].cov_share.sum()
    assert abs(tot - 1) < 1e-9, (kind, tot)
out.insert(0, 'sd_prediction', total.std(ddof=1))
out.to_csv('04outputs/71beta_contribution.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'wavelength_nm': wl, 'beta_deploy': beta, 'sd_x': Xf.std(0, ddof=1), 'sd_contribution': s}).to_csv(
    '04outputs/71beta_contribution_pointwise.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(per_sample).to_csv('04outputs/71beta_contribution_samples.csv', index=False, encoding='utf-8-sig')
print(out.round(4).to_string(index=False))
print('→ 04outputs/71beta_contribution.csv、71beta_contribution_pointwise.csv、71beta_contribution_samples.csv')
