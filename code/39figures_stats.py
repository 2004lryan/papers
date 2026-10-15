# -*- coding: utf-8 -*-
"""把三段最难读的统计数字叙述画成三张图。样式一律取自 figstyle.py，数字一律现读产物 CSV。

每张图先写清它要辩护的那一句话（figure contract 第 1 条），再决定面板分工：

图 2  方法之间哪些差异真的成立？
      (a) 效应量与不确定性：五个方法相对综合最优者的 GBID 差值 + 95% bootstrap CI
      (b) 同一批差值的贝叶斯后验分区：实质更优 / 落在实际等价区间 / 对方更优
      两个面板并排是为了让「配对检验未达显著」与「贝叶斯判为实质更优」在同一行里对上——
      1D-CNN 一行正是二者结论不同的那个反例。

图 5  这个排序在多少种设定改变下不变？
      (a) 九种设定下的名次轨迹（主分析 + 四项灵敏度 + 四种替代划分）
      (b) 五种划分下的 GBIDS 绝对水平：换协议改变的是水平而不是次序

图 6  模型离开训练分布后，失效的成分是什么？
      (a) 跨年 R² 与扣除整体偏差后的 R²（哑铃图），右侧给预测-实测相关系数
      (b) 共形预测区间的校准：同分布六条曲线贴住对角线，跨年 90% 水平明显掉下来
"""
import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from figstyle import W2, C, METHOD, panel_label, save

OUT = pathlib.Path('04outputs')
MS6 = ['SVR', 'PFLR', 'FPCR', 'PLSR', '1D-CNN', 'RF']          # 表 1 的精度次序
# figstyle 只固定了四个主对照的配色，另两个方法在此补齐，仍取自同一 Okabe–Ito 家族
MC = dict(METHOD, RF=C['green'], **{'1D-CNN': C['orange']})


def _read(name):
    return pd.read_csv(OUT / f'{name}.csv', encoding='utf-8-sig')


# ============================ 图 2 方法差异的统计判定 ============================
def fig_stats():
    pr = _read('23gbid_paired')
    pr['方法'] = pr['对比'].str.replace(' − SVR', '', regex=False)
    pr = pr.set_index('方法').reindex([m for m in MS6[::-1] if m != 'SVR'])   # 自下而上按精度递增

    fig = plt.figure(figsize=(W2, 2.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.30)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    y = np.arange(len(pr))

    # (a) 效应量 + 95% CI。实心＝Holm 校正后显著，空心＝未达显著，形状差异独立于颜色
    for i, (m, r) in enumerate(pr.iterrows()):
        sig = r['Holm_p'] < 0.05
        ax1.plot([r['CI下界'], r['CI上界']], [i, i], color=MC[m], lw=1.5, solid_capstyle='butt', zorder=2)
        ax1.plot(r['差值'], i, marker='o', ms=5.2, color=MC[m], zorder=3,
                 mfc=MC[m] if sig else 'white', mew=1.3)
        ax1.text(0.182, i, f"{r['差值']:.3f}", va='center', ha='right', fontsize=6.8, color=MC[m])
    ax1.axvline(0, color=C['black'], lw=0.6, ls=(0, (2.2, 2.2)), zorder=1)
    ax1.set_yticks(y, list(pr.index))
    ax1.set_xlim(-0.012, 0.185)
    ax1.set_xlabel('GBID 差值（相对 SVR，>0 为更差）')
    ax1.set_ylim(-0.62, len(pr) - 0.38)
    panel_label(ax1, '(a)', dx=-42)
    h_sig = ax1.plot([], [], 'o', ms=5.2, color=C['black'], label='Holm 校正后显著')[0]
    h_ns = ax1.plot([], [], 'o', ms=5.2, color=C['black'], mfc='white', mew=1.3, label='未达显著')[0]

    # (b) 同一批差值的贝叶斯后验三分区
    seg = [('P(SVR 实质更优)', 'P(SVR优)', C['black']),
           ('P(落在实际等价区间)', 'P(等价|Δ|<0.02)', C['sky']),
           ('P(该法实质更优)', 'P(该法优)', C['lightgray'])]
    left = np.zeros(len(pr))
    for lab, col, colour in seg:
        v = pr[col].to_numpy()
        ax2.barh(y, v, left=left, height=0.62, color=colour, edgecolor='white', lw=0.5, label=lab, zorder=2)
        for i, (a, b) in enumerate(zip(left, v, strict=False)):
            if b >= 0.13:                                       # 窄段不标，避免压字
                ax2.text(a + b / 2, i, f'{b:.2f}', ha='center', va='center', fontsize=6.5,
                         color='white' if colour == C['black'] else C['black'], zorder=3)
        left = left + v
    ax2.set_yticks(y, []); ax2.set_xlim(0, 1); ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xlabel('后验概率')
    ax2.set_ylim(-0.62, len(pr) - 0.38)
    panel_label(ax2, '(b)', dx=-14)

    # 两个面板的图例合并到图底一行：面板内任何位置都会被 0 参考线或误差线穿过
    hb, lb = ax2.get_legend_handles_labels()
    fig.legend([h_sig, h_ns] + hb, ['Holm 校正后显著', '未达显著'] + lb,
               loc='upper center', bbox_to_anchor=(0.5, 0.175), ncol=3, fontsize=6.6,
               handletextpad=0.4, columnspacing=1.4)
    fig.subplots_adjust(bottom=0.34)

    save(fig, 'Figure_6_forest',
         exemptions=[{'panels': ['a', 'b'], 'checks': ['panel-width'],
                      'reason': '(a) 为效应量主面板、(b) 为同一批对比的后验分区，'
                                '二者共享 y 轴类别但 x 轴量纲不同，宽度按信息量分配'}])


# ============================ 图 5 排序稳健性总览 ============================
def fig_robust():
    sen = _read('23gbid_sensitivity').set_index('方法')
    spl = _read('22split_scheme_summary')

    def _rank_lo_hi(s):                                          # 「4~4」→ (4, 4)
        a, b = str(s).split('~'); return int(a), int(b)

    # 九种设定：主分析 + 四项灵敏度（名次区间）+ 四种替代划分（按该划分内 GBIDS 重排名次）
    cols = ['主分析', 'q 参数', '两级权重', '基准值', '相关性', '随机 70%', '随机 75%', 'SPXY 75%', 'KS 75%']
    # 每列宽约 23 pt，6.2 pt 字号下最多容 2 个汉字，故显示名另取短标签
    labels = ['主分析', 'q', '权重', '基准', '相关', '随机\n70%', '随机\n75%', 'SPXY\n75%', 'KS\n75%']
    schemes = {'随机 70%': '随机70%', '随机 75%': '随机75%', 'SPXY 75%': 'SPXY75%', 'KS 75%': 'KS75%'}
    smap = {}
    for disp, key in schemes.items():
        sub = spl[spl.scheme.astype(str).str.replace(' ', '') == key]
        if sub.empty:                                            # 名称不一致时按出现顺序回退
            sub = spl[spl.scheme.astype(str).str.contains(key[:2], na=False)]
        order = sub.sort_values('R2_mean', ascending=False).method.tolist()
        smap[disp] = {m: i + 1 for i, m in enumerate(order)}

    rank = {}
    for m in MS6:
        r = [int(sen.loc[m, '主排名']), float(sen.loc[m, 'q敏感性平均排名']),
             _rank_lo_hi(sen.loc[m, '权重扰动排名区间'])[0], _rank_lo_hi(sen.loc[m, '基准敏感性排名区间'])[0],
             int(sen.loc[m, '主排名'])]                            # 相关性处理列为「一致」，即与主排名同
        r += [smap[c].get(m, np.nan) for c in cols[5:]]
        rank[m] = r

    fig = plt.figure(figsize=(W2, 2.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    x = np.arange(len(cols))
    for m in MS6:
        v = np.array(rank[m], dtype=float)
        ok = ~np.isnan(v)
        ax1.plot(x[ok], v[ok], '-', color=MC[m], lw=1.2, marker='o', ms=3.4, zorder=3,
                 mfc=MC[m], clip_on=False)
        ax1.text(x[ok][-1] + 0.22, v[ok][-1], m, va='center', ha='left', fontsize=6.8, color=MC[m],
                 zorder=5, bbox=dict(fc='white', ec='none', pad=0.6))
    ax1.axvspan(-0.5, 4.42, color=C['paleblue'], zorder=0, lw=0)
    ax1.text(2.0, 0.42, '同一划分，改评价设定', ha='center', fontsize=6.5, color=C['gray'])
    ax1.text(6.5, 0.42, '换样本划分', ha='center', fontsize=6.5, color=C['gray'])
    ax1.set_xticks(x, labels, rotation=0, ha='center', fontsize=6.2)
    ax1.set_yticks(range(1, 7)); ax1.invert_yaxis()
    ax1.set_ylim(6.6, 0.3); ax1.set_xlim(-0.5, len(cols) - 0.5 + 1.35)
    ax1.set_ylabel('按 GBIDS 的名次')
    panel_label(ax1, '(a)', dx=-32)

    # (b) 五种划分下四个方法的 GBIDS 绝对水平：水平在动，次序不动
    m4 = ['SVR', 'PFLR', 'FPCR', 'PLSR']
    sc = ['嵌套CV', '随机70%', '随机75%', 'SPXY75%', 'KS75%']
    sc_disp = ['嵌套 CV', '随机 70%', '随机 75%', 'SPXY 75%', 'KS 75%']
    gs_by = {}
    for s in sc:
        sub = spl[spl.scheme.astype(str).str.replace(' ', '') == s]
        if sub.empty:
            sub = spl[spl.scheme.astype(str).str.contains(s[:2], na=False)]
        gs_by[s] = sub.set_index('method')
    # GBIDS 由各划分自身的四项指标算出、不可跨划分比较，故纵轴取该划分内的 R²，避免混口径
    xs = np.arange(len(sc))
    for m in m4:
        ys = [float(gs_by[s].loc[m, 'R2_mean']) if m in gs_by[s].index else np.nan for s in sc]
        ax2.plot(xs, ys, '-o', color=MC[m], lw=1.1, ms=3.4, clip_on=False, zorder=3)
    ax2.set_xticks(xs, [s.replace(' ', '\n') for s in sc_disp], rotation=0, ha='center', fontsize=6.2)
    ax2.set_ylabel('R²（该划分内）')
    ax2.text(0.02, 0.03, '配色同 (a)', transform=ax2.transAxes, fontsize=6.2, color=C['gray'])
    ax2.set_xlim(-0.35, len(sc) - 0.65)
    panel_label(ax2, '(b)', dx=-30)

    save(fig, 'Figure_7_robust',
         exemptions=[{'panels': ['a', 'b'], 'checks': ['panel-width'],
                      'reason': '(a) 覆盖九种设定为主面板，(b) 只覆盖五种划分，横轴类别数不同'}])



# ============================ 图 6 跨批次失效分解 ============================
def fig_drift():
    ex = _read('31external_year_summary').set_index('method').reindex(MS6)
    cov = _read('33uncertainty_coverage')
    ood = _read('33uncertainty_ood').set_index('item')['value']

    fig = plt.figure(figsize=(W2, 2.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.34)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) 哑铃图：跨年 R²（空心）→ 扣除整体偏差后的 R²（实心）
    y = np.arange(len(ex))[::-1]
    for i, (m, r) in zip(y, ex.iterrows(), strict=False):
        ax1.plot([r['ext_R2'], r['ext_R2_bias']], [i, i], color=MC[m], lw=1.4, zorder=2,
                 solid_capstyle='round')
        ax1.plot(r['ext_R2'], i, 'o', ms=4.6, color=MC[m], mfc='white', mew=1.2, zorder=3)
        ax1.plot(r['ext_R2_bias'], i, 'o', ms=4.6, color=MC[m], zorder=3)
        ax1.annotate('', xy=(r['ext_R2_bias'], i), xytext=(r['ext_R2'] + 0.06, i), zorder=2,
                     arrowprops=dict(arrowstyle='-|>', color=MC[m], lw=0, mutation_scale=7, shrinkA=0, shrinkB=3))
        ax1.text(0.62, i, f"r={r['ext_r']:.2f}", va='center', ha='left', fontsize=6.6, color=MC[m])
    ax1.axvline(0, color=C['black'], lw=0.6, ls=(0, (2.2, 2.2)), zorder=1)
    ax1.set_yticks(y, list(ex.index))
    ax1.set_xlim(-1.62, 0.92); ax1.set_ylim(-0.6, len(ex) - 0.4)
    ax1.set_xlabel('跨年 R²')
    panel_label(ax1, '(a)', dx=-40)
    ax1.plot([], [], 'o', ms=4.6, color=C['black'], mfc='white', mew=1.2, label='原始')
    ax1.plot([], [], 'o', ms=4.6, color=C['black'], label='扣除整体偏差后')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.42, -0.24), ncol=2, fontsize=6.6,
               handletextpad=0.4, columnspacing=1.2)

    # (b) 共形区间校准：同分布贴对角线，跨年 90% 掉下来
    ax2.plot([0.45, 1.0], [0.45, 1.0], color=C['gray'], lw=0.7, ls=(0, (3, 2)), zorder=1)
    for m in MS6:
        s = cov[cov.method == m].sort_values('nominal')
        if s.empty: continue
        ax2.plot(s.nominal, s.empirical, '-', color=MC[m], lw=1.0, marker='o', ms=2.8, zorder=3)
    key = '跨年 {} 名义 0.90 的经验覆盖'
    for m in ['PLSR', 'SVR', 'FPCR', 'PFLR']:
        k = key.format(m)
        if k in ood.index:
            ax2.plot(0.90, float(ood[k]), marker='v', ms=5.0, color=MC[m], zorder=4, clip_on=False)
    ax2.set_xlim(0.45, 1.0); ax2.set_ylim(0.10, 1.0)
    ax2.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.set_xlabel('名义覆盖率'); ax2.set_ylabel('经验覆盖率')
    ax2.plot([], [], '-o', color=C['black'], ms=2.8, lw=1.0, label='同分布（折外）')
    ax2.plot([], [], 'v', color=C['black'], ms=5.0, label='跨年，名义 0.90')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.45, -0.24), ncol=1, fontsize=6.6,
               handletextpad=0.4)
    panel_label(ax2, '(b)', dx=-30)

    save(fig, 'Figure_8_drift',
         exemptions=[{'panels': ['a', 'b'], 'checks': ['panel-width'],
                      'reason': '(a) 为六方法的失效分解主面板，(b) 为区间校准的诊断面板，横轴量纲不同'}])


# ============================ 合并：方法差异的统计判定 + 排序稳健性（2×2） ============================
def fig_merged():
    """(a)(b) 取自 fig_stats，(c)(d) 取自 fig_robust；四个面板回答同一个问题：
    「SVR 领先这件事，有多确定、在多少种设定下仍成立？」"""
    pr = _read('23gbid_paired')
    pr['方法'] = pr['对比'].str.replace(' − SVR', '', regex=False)
    pr = pr.set_index('方法').reindex([m for m in MS6[::-1] if m != 'SVR'])
    sen = _read('23gbid_sensitivity').set_index('方法')
    spl = _read('22split_scheme_summary')

    fig = plt.figure(figsize=(W2, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.12],
                          wspace=0.30, hspace=0.78)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax3, ax4 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    y = np.arange(len(pr))

    for i, (m, r) in enumerate(pr.iterrows()):                    # (a) 效应量 + 95% CI
        sig = r['Holm_p'] < 0.05
        ax1.plot([r['CI下界'], r['CI上界']], [i, i], color=MC[m], lw=1.5, solid_capstyle='butt', zorder=2)
        ax1.plot(r['差值'], i, marker='o', ms=5.2, color=MC[m], zorder=3, mfc=MC[m] if sig else 'white', mew=1.3)
        ax1.text(0.182, i, f"{r['差值']:.3f}", va='center', ha='right', fontsize=6.8, color=MC[m])
    ax1.axvline(0, color=C['black'], lw=0.6, ls=(0, (2.2, 2.2)), zorder=1)
    ax1.set_yticks(y, list(pr.index)); ax1.set_xlim(-0.012, 0.185)
    ax1.set_xlabel('GBID 差值（相对 SVR，>0 为更差）'); ax1.set_ylim(-0.62, len(pr) - 0.38)
    panel_label(ax1, '(a)', dx=-42)
    h_sig = ax1.plot([], [], 'o', ms=5.2, color=C['black'], label='Holm 校正后显著')[0]
    h_ns = ax1.plot([], [], 'o', ms=5.2, color=C['black'], mfc='white', mew=1.3, label='未达显著')[0]

    seg = [('P(SVR 实质更优)', 'P(SVR优)', C['black']), ('P(落在实际等价区间)', 'P(等价|Δ|<0.02)', C['sky']),
           ('P(该法实质更优)', 'P(该法优)', C['lightgray'])]
    left = np.zeros(len(pr))                                      # (b) 贝叶斯后验三分区
    for lab, col, colour in seg:
        v = pr[col].to_numpy()
        ax2.barh(y, v, left=left, height=0.62, color=colour, edgecolor='white', lw=0.5, label=lab, zorder=2)
        for i, (a, b) in enumerate(zip(left, v, strict=False)):
            if b >= 0.13:
                ax2.text(a + b / 2, i, f'{b:.2f}', ha='center', va='center', fontsize=6.5,
                         color='white' if colour == C['black'] else C['black'], zorder=3)
        left = left + v
    ax2.set_yticks(y, []); ax2.set_xlim(0, 1); ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xlabel('后验概率'); ax2.set_ylim(-0.62, len(pr) - 0.38)
    panel_label(ax2, '(b)', dx=-14)
    hb, lb = ax2.get_legend_handles_labels()
    fig.legend([h_sig, h_ns] + hb, ['Holm 校正后显著', '未达显著'] + lb,
               loc='center', bbox_to_anchor=(0.5, 0.505), ncol=5, fontsize=6.3,
               handletextpad=0.4, columnspacing=1.0)

    def _lo(s): return int(str(s).split('~')[0])
    cols = ['主分析', 'q 参数', '两级权重', '基准值', '相关性', '随机 70%', '随机 75%', 'SPXY 75%', 'KS 75%']
    labels = ['主分析', 'q', '权重', '基准', '相关', '随机\n70%', '随机\n75%', 'SPXY\n75%', 'KS\n75%']
    schemes = {'随机 70%': '随机70%', '随机 75%': '随机75%', 'SPXY 75%': 'SPXY75%', 'KS 75%': 'KS75%'}
    smap = {}
    for disp, key in schemes.items():
        sub = spl[spl.scheme.astype(str).str.replace(' ', '') == key]
        order = sub.sort_values('R2_mean', ascending=False).method.tolist()
        smap[disp] = {m: i + 1 for i, m in enumerate(order)}
    x = np.arange(len(cols))                                      # (c) 九种设定下的名次
    for m in MS6:
        v = np.array([int(sen.loc[m, '主排名']), float(sen.loc[m, 'q敏感性平均排名']),
                      _lo(sen.loc[m, '权重扰动排名区间']), _lo(sen.loc[m, '基准敏感性排名区间']),
                      int(sen.loc[m, '主排名'])] + [smap[c].get(m, np.nan) for c in cols[5:]], dtype=float)
        ok = ~np.isnan(v)
        ax3.plot(x[ok], v[ok], '-', color=MC[m], lw=1.2, marker='o', ms=3.4, zorder=3, mfc=MC[m], clip_on=False)
        ax3.text(x[ok][-1] + 0.22, v[ok][-1], m, va='center', ha='left', fontsize=6.8, color=MC[m],
                 zorder=5, bbox=dict(fc='white', ec='none', pad=0.6))
    ax3.axvspan(-0.5, 4.42, color=C['paleblue'], zorder=0, lw=0)
    ax3.text(2.0, 0.42, '同一划分，改评价设定', ha='center', fontsize=6.5, color=C['gray'])
    ax3.text(6.5, 0.42, '换样本划分', ha='center', fontsize=6.5, color=C['gray'])
    ax3.set_xticks(x, labels, fontsize=6.2); ax3.set_yticks(range(1, 7))
    ax3.set_ylim(6.6, 0.3); ax3.set_xlim(-0.5, len(cols) - 0.5 + 1.35)
    ax3.set_ylabel('按 GBIDS 的名次')
    panel_label(ax3, '(c)', dx=-32)

    m4 = ['SVR', 'PFLR', 'FPCR', 'PLSR']                          # (d) 五种划分下的 R²
    sc = ['嵌套CV', '随机70%', '随机75%', 'SPXY75%', 'KS75%']
    sc_disp = ['嵌套\nCV', '随机\n70%', '随机\n75%', 'SPXY\n75%', 'KS\n75%']
    gs_by = {s: spl[spl.scheme.astype(str).str.replace(' ', '') == s].set_index('method') for s in sc}
    xs = np.arange(len(sc))
    for m in m4:
        ys = [float(gs_by[s].loc[m, 'R2_mean']) if m in gs_by[s].index else np.nan for s in sc]
        ax4.plot(xs, ys, '-o', color=MC[m], lw=1.1, ms=3.4, clip_on=False, zorder=3)
    ax4.set_xticks(xs, sc_disp, fontsize=6.2); ax4.set_ylabel('R²（该划分内）')
    ax4.text(0.02, 0.03, '配色同 (c)', transform=ax4.transAxes, fontsize=6.2, color=C['gray'])
    ax4.set_xlim(-0.35, len(sc) - 0.65)
    panel_label(ax4, '(d)', dx=-30)

    save(fig, 'Figure_6_stats',
         exemptions=[{'panels': ['a', 'b', 'c', 'd'], 'checks': ['panel-width'],
                      'reason': '左列为主面板（效应量、名次轨迹），右列为各自的配套面板，宽度按信息量分配'}])


if __name__ == '__main__':
    fig_stats(); fig_robust(); fig_drift(); fig_merged()
    print('三张统计图已导出到 04outputs/figs/')
