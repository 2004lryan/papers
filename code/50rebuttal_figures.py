# -*- coding: utf-8 -*-
"""
50rebuttal_figures.py: 修订稿的图 1（配对差值森林图）、图 2（小样本两种选参策略）、图 3（跨批次校正热图）（导师意见 1、3、5，预注册 B10.2~B10.4）

为什么要有这个脚本：修订稿用三张图替代原稿中大段的统计数字叙述。每张图先写清它要支撑的那一句话：
  图 1  「统一协议下哪些方法对之间检出了 R² 差异、差值与不确定性有多大」——逐对给差值与修正 95% 置信区间，
        Holm 校正后显著者实心，未检出差异者空心；区间一律画出，不以「未达显著」暗示相等。
  图 2  「训练样本受限时，两种选参做法的典型表现与失效各是怎样」——中位数与四分位带表示典型表现，
        下方条形给出 20 次重抽中 R²<0 的次数，均值不入图（N≤30 时均值受少数失效支配，见 49 脚本说明）。
  图 3  「少量目标批次样品的校正效果随批次类型不同」——五种设置、四个方法、四个 k、三种策略全部入图，
        事后参照 O1、O2 单列并标明，不挑组合。
数字一律读产物 CSV；样式取自 figstyle.py；多面板图经 figstyle.save 的对齐审计后导出。
图内中文与数字、字母之间不留空格（与正文的空格规范一致）。FIG_LANG=en 时画英文稿用的英文版，
数据、版式与中文版相同，只换图内文字（T 字典逐项对应），文件名加 _en。

口径：
  图 1  47 的 paired_tests 产物的全部行（经 results47.resolve() 取得，只认有效标记：正式结果优先，其次部分结果），
        按族分组、族内保持脚本输出的顺序；Holm p<0.05 实心。
        RF、1D-CNN 未完成时主分析族少于 15 对，图题自动写出当前对数，不补画缺失的对。
  图 2  21smallsample_repeats.csv 的 20 次重抽：逐（N, 方法, 策略）取中位数与 25%、75% 分位；
        中位数与 49rebuttal_smallsample_summary.csv 逐项核对（差 <1e-12），失效次数取自该文件。
        PLSR 只有一种做法（潜变量数由训练集内 4 折交叉验证选取），各面板以灰色虚线作参照。
  图 3  46rebuttal_recal_summary.csv：S1~S3 取 200 次抽样的 R² 中位数；S0 取源模型对目标批次全部样品的 R²（tgt_R2），
        O1、O2 为使用全部目标样品真值的事后参照。色标以 0 为中心，低于 −1 的格按 −1 着色、格内数字仍为实际值。

运行方式:
    cd <项目根目录>
    python3 02code/50rebuttal_figures.py                   # 中文版（中文稿用）
    FIG_LANG=en python3 02code/50rebuttal_figures.py       # 英文版（英文稿用）

输出文件:
    04outputs/figs/Figure_forest_pairs.{pdf,svg,png}       — 图 1
    04outputs/figs/Figure_smallsample_median.{pdf,svg,png} — 图 2（对齐审计 JSON 存 04outputs/figs/qa/）
    04outputs/figs/Figure_recal_heatmap.{pdf,svg,png}      — 图 3
    04outputs/figs/Figure_{forest_pairs,smallsample_median,recal_heatmap}_en.{pdf,svg,png} — FIG_LANG=en 时的英文版
    05logs/50rebuttal_figures_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

不读 05data，只读 04outputs。
"""
from typing import Any
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory, offset_copy

sys.path.insert(0, '02code')
from figstyle import W2, C, METHOD, panel_label, save, zero_line  # noqa: E402
from results47 import resolve as resolve47  # noqa: E402

O = '04outputs'
MINUS = '−'
LANG = os.environ.get('FIG_LANG', 'zh')
if LANG not in ('zh', 'en'):
    raise SystemExit(f'FIG_LANG 只能取 zh 或 en，收到 {LANG}')
SUFFIX = '' if LANG == 'zh' else '_en'
# 图中全部文字，中英两版逐项对应；英文版按英文排印（数值区间用 en dash，运算符两侧留空格）
T: dict[str, Any] = {
    'zh': dict(
        fam_main='主分析（{n}对，Holm校正）', fam_basis='基函数敏感性（B样条−傅里叶，{n}对）',
        head_diff='差值[未校正95%置信区间]', xlab_forest='R²之差（前者−后者），25个配对外层折',
        sig='Holm校正后p<0.05', nosig='未检出差异',
        xlab_n='训练集样本量N', ylab_med='测试集R²（中位数）', ylab_neg='R²<0次数',
        leg=['固定超参数', '交叉验证选参', '25%~75%分位', 'PLSR中位数'],
        groups=['直接\n使用', '截距校正', '斜率-截距校正', '加样重建', '事后参照'],
        sets={'2018→2019': '2018→2019', '2019→2018': '2019→2018', '留新疆': '目标：新疆', '留山东': '目标：山东',
              '留陕西': '目标：陕西'}),
    'en': dict(
        fam_main='Main analysis ({n} pairs, Holm-adjusted)', fam_basis='Basis sensitivity (B-spline − Fourier, {n} pairs)',
        head_diff='Difference [unadjusted 95% CI]', xlab_forest='Difference in R² (first − second), 25 paired outer folds',
        sig='Holm-adjusted p < 0.05', nosig='No difference detected',
        xlab_n='Training set size N', ylab_med='Test R² (median)', ylab_neg='Draws with R² < 0',
        leg=['Fixed hyperparameters', 'Cross-validated selection', '25%–75% quantiles', 'PLSR median'],
        groups=['Direct\nuse', 'Intercept', 'Slope–intercept', 'Augmented refit', 'Post hoc'],
        # 竖排的设置名只有四行格子高（约 43 pt），「Target: Shandong」放不下；与「2018→2019」同一写法，箭头指向目标产地
        sets={'2018→2019': '2018→2019', '2019→2018': '2019→2018', '留新疆': '→Xinjiang', '留山东': '→Shandong',
              '留陕西': '→Shaanxi'}),
}[LANG]


def fmt(v, nd=2):
    return f'{v:.{nd}f}'.replace('-', MINUS)


# ============================ 图 1：配对差值森林图 ============================
def fig_forest():
    src = resolve47()
    if src.with_status('invalid'):
        raise SystemExit(f'47 的输入有未通过核对的项，不据此作图：{src.with_status("invalid")}')
    pt = pd.read_csv(src.path('paired_tests'))
    print(f'图 1 数据：{src.path("paired_tests")}（{"正式结果" if src.final else "部分结果，非最终"}）')
    fams = [f for f in ['主分析', '基函数敏感性'] if f in set(pt.family)]
    rows, ypos, heads, y = [], [], [], 0.0
    for f in fams:
        sub = pt[pt.family == f]
        n_fam = int(sub.family_size.iloc[0])
        heads.append((y, (T['fam_main'] if f == '主分析' else T['fam_basis']).format(n=n_fam)))
        y += 0.9
        for q in sub.itertuples():
            rows.append(q)
            ypos.append(y)
            y += 1.0
        y += 0.5
    fig = plt.figure(figsize=(W2, 0.30 + 0.175 * (y + 0.6)))
    ax = fig.add_axes((0.16, 0.13, 0.50, 0.84))
    tr_fig_y = blended_transform_factory(fig.transFigure, ax.transData)
    for q, yy in zip(rows, ypos, strict=True):
        col = C['black'] if q.family == '主分析' else C['gray']
        sig = q.holm_p < 0.05
        ax.plot([q.ci_lo, q.ci_hi], [yy, yy], color=col, lw=1.0, solid_capstyle='butt', zorder=2)
        ax.plot([q.diff], [yy], marker='o', ms=4.2, mfc=col if sig else 'white', mec=col, mew=0.9, zorder=3)
        ax.text(0.69, yy, f'{fmt(q.diff, 3)}  [{fmt(q.ci_lo, 3)}, {fmt(q.ci_hi, 3)}]', transform=tr_fig_y,
                va='center', ha='left', fontsize=6.5)
        # p 值小于 0.001 时写「<0.001」，三位小数写成 0.000 会被读成 p 为 0
        ax.text(0.995, yy, '<0.001' if q.holm_p < 0.001 else fmt(q.holm_p, 3), transform=tr_fig_y, va='center', ha='right', fontsize=6.5)
    for yy, lab in heads:
        ax.text(0.005, yy, lab, transform=tr_fig_y, va='center', ha='left', fontsize=7, fontweight='bold')
    # 区间为逐对（未经多重比较校正）的修正 t 区间；实心与否按 Holm 校正后的 p 判定，二者口径不同，表头写明
    ax.text(0.69, heads[0][0], T['head_diff'], transform=tr_fig_y, va='center', ha='left', fontsize=6.5, color=C['gray'])
    ax.text(0.995, heads[0][0], 'Holm p', transform=tr_fig_y, va='center', ha='right', fontsize=6.5, color=C['gray'])
    for f in fams:                                                     # 零线只画在各族数据行的范围内
        yf = [yy for q, yy in zip(rows, ypos, strict=True) if q.family == f]
        ax.plot([0, 0], [min(yf) - 0.55, max(yf) + 0.55], color=C['black'], lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([q.pair.split('：')[0] if q.family != '主分析' else q.pair for q in rows])
    ax.set_ylim(y - 0.2, -0.9)
    lo = min(pt.ci_lo.min(), 0.0)
    hi = max(pt.ci_hi.max(), 0.0)
    pad = 0.08 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(T['xlab_forest'])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    handles = [Line2D([], [], marker='o', ls='', ms=4.2, mfc=C['black'], mec=C['black'], label=T['sig']),
               Line2D([], [], marker='o', ls='', ms=4.2, mfc='white', mec=C['black'], label=T['nosig'])]
    ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=6.5, handletextpad=0.3)
    save(fig, 'Figure_forest_pairs' + SUFFIX, multi=False)
    print(f'  图 1：{len(rows)} 对（' + '、'.join(f'{f} {int((pt.family == f).sum())} 对' for f in fams) + '）')


# ============================ 图 2：小样本两种选参策略 ============================
def fig_smallsample():
    rep = pd.read_csv(f'{O}/21smallsample_repeats.csv')
    rep['method'] = rep['method'].replace({'PFLR': 'FPLR'})
    summ = pd.read_csv(f'{O}/49rebuttal_smallsample_summary.csv')
    g = rep.groupby(['N', 'method', 'strategy']).R2
    q = pd.DataFrame({'q25': g.quantile(0.25), 'q50': g.median(), 'q75': g.quantile(0.75)}).reset_index()
    chk = q.merge(summ, on=['N', 'method', 'strategy'], how='inner')
    if len(chk) != len(summ) or float(np.max(np.abs(chk.q50 - chk['median']))) > 1e-12:
        raise SystemExit('21smallsample_repeats.csv 的中位数与 49rebuttal_smallsample_summary.csv 不一致，先重跑 49 脚本')
    Ns = sorted(rep.N.unique())
    if min(Ns) <= 0:
        raise SystemExit('训练集样本量须为正数，才能画对数横轴')
    if float(q.q75.max()) >= 0.86:
        raise SystemExit('四分位带上界超出纵轴上限 0.86')
    fig = plt.figure(figsize=(W2, 2.95))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.55, 1.0], hspace=0.16, wspace=0.12,
                          left=0.085, right=0.995, top=0.855, bottom=0.14)
    color = {'FPCR': METHOD['FPCR'], 'FPLR': METHOD['PFLR'], 'SVR': METHOD['SVR']}
    plsr = q[q.method == 'PLSR'].sort_values('N')
    top: list[Any] = []
    bot: list[Any] = []
    for j, m in enumerate(['FPCR', 'FPLR', 'SVR']):
        a = fig.add_subplot(gs[0, j], sharey=top[0] if top else None)
        b = fig.add_subplot(gs[1, j], sharex=a, sharey=bot[0] if bot else None)
        top.append(a)
        bot.append(b)
        for st, ls in [('fixed', '-'), ('cv', (0, (3, 1.6)))]:
            d = q[(q.method == m) & (q.strategy == st)].sort_values('N')
            a.fill_between(d.N, d.q25, d.q75, color=color[m], alpha=0.16 if st == 'fixed' else 0.09, lw=0)
            a.plot(d.N, d.q50, color=color[m], lw=1.15, ls=ls, marker='o' if st == 'fixed' else 's', ms=2.6,
                   mfc=color[m] if st == 'fixed' else 'white', mec=color[m], mew=0.7, zorder=3)
        a.plot(plsr.N, plsr.q50, color=C['gray'], lw=0.9, ls=(0, (1, 1.2)), zorder=2)
        zero_line(a)
        a.set_xscale('log')
        a.set_ylim(float(q.q25.min()) - 0.05, 0.86)                   # 四分位带下界全部入图，不截断
        a.set_title(m, loc='center', pad=3)
        a.tick_params(labelbottom=False)
        width = 0.12
        for st, off, fc in [('fixed', -1, color[m]), ('cv', 1, 'white')]:     # 实心 = 固定超参数，空心 = 交叉验证，与上排标记一致
            d = summ[(summ.method == m) & (summ.strategy == st)].sort_values('N')
            xs = np.array(d.N, dtype=float) * np.exp(off * width / 2 * 1.05)
            b.bar(xs, d.n_neg, width=np.array(d.N, dtype=float) * width, color=fc, edgecolor=color[m], lw=0.6)
        b.set_xscale('log')
        b.set_xticks(Ns)
        b.set_xticklabels([str(n) for n in Ns])
        b.minorticks_off()
        a.minorticks_off()
        b.set_ylim(0, 15)
        b.set_yticks([0, 5, 10, 15])
        b.set_xlabel(T['xlab_n'])
        if j > 0:
            a.tick_params(labelleft=False)
            b.tick_params(labelleft=False)
    top[0].set_ylabel(T['ylab_med'])
    bot[0].set_ylabel(T['ylab_neg'])
    for a, lab in zip(top, ['(a)', '(b)', '(c)'], strict=True):
        panel_label(a, lab, dx=-2, dy=3)
    for b, lab in zip(bot, ['(d)', '(e)', '(f)'], strict=True):          # 放在右上角：N=200、400 两档无失效，条形不会压字
        b.text(0.98, 0.95, lab, transform=b.transAxes, ha='right', va='top', fontsize=9, fontweight='bold')
    handles: list[Any] = [(Line2D([], [], color=C['black'], lw=1.15, marker='o', ms=2.6),
                Patch(fc=C['black'], ec=C['black'], lw=0.6)),
               (Line2D([], [], color=C['black'], lw=1.15, ls=(0, (3, 1.6)), marker='s', ms=2.6, mfc='white'),
                Patch(fc='white', ec=C['black'], lw=0.6)),
               Patch(fc=C['lightgray'], ec='none'),
               Line2D([], [], color=C['gray'], lw=0.9, ls=(0, (1, 1.2)))]
    fig.legend(handles, T['leg'], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)}, loc='upper center', ncol=4,
               bbox_to_anchor=(0.54, 1.0), fontsize=6.5, handlelength=3.2, columnspacing=1.6)
    save(fig, 'Figure_smallsample_median' + SUFFIX)
    n80 = summ[summ.N == 80].set_index(['method', 'strategy'])
    print(f'  图 2：N=80 FPCR 固定/交叉验证中位数 {n80.loc[("FPCR", "fixed"), "median"]:.3f}/{n80.loc[("FPCR", "cv"), "median"]:.3f}，'
          f'R²<0 次数 {int(n80.loc[("FPCR", "fixed"), "n_neg"])}/{int(n80.loc[("FPCR", "cv"), "n_neg"])}')


# ============================ 图 3：跨批次校正热图 ============================
def fig_recal():
    s = pd.read_csv(f'{O}/46rebuttal_recal_summary.csv')
    SET = ['2018→2019', '2019→2018', '留新疆', '留山东', '留陕西']
    MS = ['PLSR', 'SVR', 'FPLR', 'FPCR']
    KS = [5, 10, 20, 40]
    cols = [('S0', None)] + [(st, k) for st in ['S1', 'S2', 'S3'] for k in KS] + [('O1', None), ('O2', None)]
    mat = np.full((len(SET) * len(MS), len(cols)), np.nan)
    for i, st_name in enumerate(SET):
        for j, m in enumerate(MS):
            sub = s[(s.setting == st_name) & (s.method == m)]
            if sub.empty:
                raise SystemExit(f'46rebuttal_recal_summary.csv 缺（{st_name}, {m}）')
            r = i * len(MS) + j
            for c, (st, k) in enumerate(cols):
                if st == 'S0':
                    mat[r, c] = float(sub.tgt_R2.iloc[0])
                elif st in ('O1', 'O2'):
                    mat[r, c] = float(sub[f'{st}_R2'].iloc[0])
                else:
                    cell = sub[(sub.strategy == st) & (sub.k == k)]
                    if len(cell) != 1:
                        raise SystemExit(f'（{st_name}, {m}, {st}, k={k}）应恰有一行')
                    mat[r, c] = float(cell.R2_med.iloc[0])
    if np.isnan(mat).any():
        raise SystemExit('热图矩阵含缺失值')
    # 列之间在策略分组处留缝：用 x 坐标偏移实现
    gap = 0.35
    xs, x = [], 0.0
    groups = [1, 4, 4, 4, 2]
    for gsz in groups:
        for _ in range(gsz):
            xs.append(x)
            x += 1.0
        x += gap
    xs_arr = np.array(xs)
    ys = []
    yv = 0.0
    for _i in range(len(SET)):
        for _j in range(len(MS)):
            ys.append(yv)
            yv += 1.0
        yv += gap
    ys_arr = np.array(ys)
    # 版式按物理尺寸定：左侧 34 pt 放旋转的设置名与方法名，底部 0.36 in 放色标；
    # 其余宽度全给 15 列，单格约 20 pt 宽，带负号的两位小数（约 16 pt）两侧各留约 2 pt
    fig_h = 4.0
    fig = plt.figure(figsize=(W2, fig_h))
    ax_left = 34 / 72 / W2
    ax_bottom = 0.36 / fig_h
    ax = fig.add_axes((ax_left, ax_bottom, 0.997 - ax_left, 0.995 - ax_bottom))
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=0.8)
    cmap = plt.get_cmap('RdBu')
    cell = 0.92                                                        # 格与格之间留白缝，不描边
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            rgba = cmap(norm(np.clip(v, -1.0, 0.8)))
            ax.add_patch(plt.Rectangle((xs_arr[c] - cell / 2, ys_arr[r] - cell / 2), cell, cell, fc=rgba, ec='none', lw=0))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(xs_arr[c], ys_arr[r], fmt(v, 2), ha='center', va='center', fontsize=6.5,
                    color='white' if lum < 0.5 else C['black'])
    ax.set_xlim(xs_arr[0] - 0.6, xs_arr[-1] + 0.6)
    ax.set_ylim(ys_arr[-1] + 0.6, ys_arr[0] - 2.9)
    ax.set_xticks([])
    ax.set_yticks(ys_arr)
    ax.set_yticklabels(MS * len(SET))
    ax.tick_params(axis='y', length=0, pad=1.5, labelsize=6.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # 列标签：k 值一行、策略名一行（直接写在数据坐标里，避免与刻度标签相撞）
    sub_lab = [''] + [str(k) for k in KS] * 3 + ['O1', 'O2']
    for xc, lab in zip(xs_arr, sub_lab, strict=True):
        ax.text(xc, ys_arr[0] - 0.95, lab, ha='center', va='center', fontsize=6.5)
    starts = np.cumsum([0] + groups[:-1])
    for st_i, gsz, nm in zip(starts, groups, T['groups'], strict=True):
        xc = (xs_arr[st_i] + xs_arr[st_i + gsz - 1]) / 2
        yb = ys_arr[0] - (0.62 if gsz == 1 else 1.45)
        # 两行的列名（英文版的 Direct / use）行距取 1.15：取 1.0 时两行的文字框相叠 0.6 pt，碰撞审计判 FAIL
        ax.text(xc, yb, nm, ha='center', va='bottom', fontsize=7, fontweight='bold', linespacing=1.15)
        if gsz == 4:
            ax.plot([xs_arr[st_i] - 0.45, xs_arr[st_i + gsz - 1] + 0.45], [ys_arr[0] - 1.38] * 2, color=C['black'], lw=0.5)
    # 设置名：旋转 90°，置于方法名左侧 27 pt 处，居中对齐该设置的四行
    tr_set = offset_copy(ax.get_yaxis_transform(), fig=fig, x=-27, y=0, units='points')
    for i, st_name in enumerate(SET):
        yc = (ys_arr[i * len(MS)] + ys_arr[i * len(MS) + len(MS) - 1]) / 2
        ax.text(0, yc, T['sets'][st_name], transform=tr_set, rotation=90, rotation_mode='anchor', ha='center', va='center',
                fontsize=7, fontweight='bold')
    # 色标：以 0 为中心的发散色，低于 −1 的格按 −1 着色（左端三角表示截断）
    ax_w = 0.997 - ax_left
    cax = fig.add_axes((ax_left + 0.58 * ax_w, 0.045, 0.40 * ax_w, 0.022))
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='min',
                      ticks=[-1.0, -0.5, 0.0, 0.4, 0.8])
    cb.outline.set_linewidth(0.4)  # type: ignore[operator]
    cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
    cb.ax.set_xticklabels([fmt(t, 1) if t != 0 else '0' for t in [-1.0, -0.5, 0.0, 0.4, 0.8]])
    fig.text(ax_left + 0.58 * ax_w - 0.012, 0.045 + 0.011, 'R²', ha='right', va='center', fontsize=6.5)
    save(fig, 'Figure_recal_heatmap' + SUFFIX, multi=False)
    print(f'  图 3：{mat.shape[0]} 行 × {mat.shape[1]} 列，最小 {mat.min():.3f}、最大 {mat.max():.3f}')


if __name__ == '__main__':
    print('画图：')
    fig_forest()
    fig_smallsample()
    fig_recal()
