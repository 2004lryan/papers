# -*- coding: utf-8 -*-
"""
70batch_overview.py: 各批次的 SSC 统计特征与平均光谱（稿件 2.1 节的表 1 与图 2）

为什么要有这个脚本：目标刊同方向的论文与对标的中文论文，结果一节都先给样品参考值的统计特征与光谱（项目规范 18.1 的对标语料，
结论见 research-wiki/目标期刊论文/00VENUE_STUDY.md §4）。本稿的跨批次实验按年份与产地分批，各批次的 SSC 分布与平均光谱
是读 2.4 节的前提，此前只在附录 A4 有一句各年份、各产地的样本数与均值范围。

做法：
  1. 从 04outputs/59data_clean_list.csv 取清洗后保留的记录，按年份 × 产地统计样本数与 SSC 的均值、标准差、最小值、最大值
  2. 读两个年份的原始光谱（只读 05data），按 data_clean.keep_mask 取保留的记录并与清单逐行核对；求各批次的平均原始光谱，
     以及与 16unified_comparison.py 相同的预处理（SNV 后接 SG 一阶导，窗宽 15、二次多项式）之后的平均光谱
  3. 描述量：各批次平均原始光谱的峰位、峰值与全波段均值；两年在 650~700 nm 内的局部极小位置；2019 年与 2018 年
     同一产地平均强度之比；预处理前后，年份之间、新疆与另两个产地之间、山东与陕西之间平均光谱的差异（见 summary 各行的 note）
  4. 画图 2：(a) 各批次的平均原始光谱；(b) 预处理后的平均光谱。中英两版只有文字不同

运行方式:
    cd <项目根目录>
    PANEL_AUDIT_DIR=<对齐审计模块所在目录> python3 02code/70batch_overview.py                 # 中文版
    PANEL_AUDIT_DIR=<对齐审计模块所在目录> FIG_LANG=en python3 02code/70batch_overview.py     # 英文版（数值产物相同）

输入文件（只读）:
    05data 下两个年份的 Excel（路径取环境变量 APPLE_SSC_DATA）、04outputs/59data_clean_list.csv
输出文件:
    04outputs/70batch_stats.csv             — 表 1：年份、产地、样本数、SSC 均值、标准差、最小值、最大值（含合计行）
    04outputs/70batch_spectra_summary.csv   — 正文 2.1 节引用的描述量，一行一个量
    04outputs/70batch_mean_spectra.csv      — 各批次的平均原始光谱与平均预处理光谱（图 2 的源数据）
    04outputs/figs/Figure_spectra_batches[_en].pdf/.svg/.png
    05logs/70batch_overview_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import argrelextrema, savgol_filter

sys.path.insert(0, '02code')
import data_clean  # noqa: E402
from figstyle import LOCC, W2, panel_label, save, zero_line  # noqa: E402

D = os.environ.get('APPLE_SSC_DATA', '../../05data/001_apple_hyperspectral_multiyear')   # 数据不随代码分发，见 08github/data/DATA.md
FILES = {2018: '01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx', 2019: '1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx'}
O = '04outputs'
LANG = os.environ.get('FIG_LANG', 'zh')
if LANG not in ('zh', 'en'):
    raise SystemExit(f'FIG_LANG 只能取 zh 或 en，收到 {LANG}')
SUFFIX = '' if LANG == 'zh' else '_en'
AREA = {'zh': {1: '新疆', 2: '山东', 3: '陕西'}, 'en': {1: 'Xinjiang', 2: 'Shandong', 3: 'Shaanxi'}}
T = {'zh': dict(xlab='波长/nm', ya='透射强度/10³计数', yb='SNV+SG一阶导后的光谱值', yr='{y}年'),
     'en': dict(xlab='Wavelength (nm)', ya='Transmitted intensity (10³ counts)', yb='Spectral value after SNV + SG-1D', yr='{y}')}
LS = {2018: '-', 2019: (0, (4, 1.6))}


def raw(f):
    """与 59data_clean.py 相同的读法：首行为波长，第 3 行起为数据，第 2 049、2 050 列为 SSC 与产地编码。"""
    d = pd.read_excel(f'{D}/{f}', header=None)
    wl = d.iloc[0, :2048].astype(float).values
    X = d.iloc[2:, :2048].astype(float).values
    y = pd.to_numeric(d.iloc[2:, 2048], errors='coerce').values
    z = pd.to_numeric(d.iloc[2:, 2049], errors='coerce').values
    return X, y, z, wl


def snv(M):
    return (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def main():
    lst = pd.read_csv(f'{O}/59data_clean_list.csv')
    kept = lst[lst.keep]
    # ---- 1. 表 1 ----
    rows = []
    for yr in FILES:
        for o in (1, 2, 3):
            s = kept[(kept.year == yr) & (kept.origin == o)].SSC
            rows.append({'year': yr, 'origin': AREA['zh'][o], 'n': len(s), 'SSC_mean': s.mean(), 'SSC_sd': s.std(ddof=1),
                         'SSC_min': s.min(), 'SSC_max': s.max()})
    rows.append({'year': '合计', 'origin': '', 'n': len(kept), 'SSC_mean': kept.SSC.mean(), 'SSC_sd': kept.SSC.std(ddof=1),
                 'SSC_min': kept.SSC.min(), 'SSC_max': kept.SSC.max()})
    stats = pd.DataFrame(rows)
    assert int(stats.n.iloc[:-1].sum()) == int(stats.n.iloc[-1]) == 1111
    stats.to_csv(f'{O}/70batch_stats.csv', index=False, encoding='utf-8-sig')
    print(stats.round(3).to_string(index=False))

    # ---- 2. 平均光谱 ----
    mean_raw: dict[tuple[int, int], np.ndarray] = {}
    mean_pre: dict[tuple[int, int], np.ndarray] = {}
    wl = np.empty(0)
    for yr, f in FILES.items():
        X, y, z, wl = raw(f)
        keep = data_clean.keep_mask(X, y, z)
        assert (keep == lst[lst.year == yr].keep.values).all(), f'{yr} 年的保留记录与清洗清单不一致'
        Xk, zk = X[keep], z[keep].astype(int)
        Xp = savgol_filter(snv(Xk), 15, 2, deriv=1, axis=1)          # 与 16unified_comparison.py 的预处理相同
        for o in (1, 2, 3):
            assert int((zk == o).sum()) == int(stats[(stats.year == yr) & (stats.origin == AREA['zh'][o])].n.iloc[0])
            mean_raw[yr, o] = Xk[zk == o].mean(0)
            mean_pre[yr, o] = Xp[zk == o].mean(0)
    ms = pd.DataFrame({'wavelength_nm': wl})
    for (yr, o), v in mean_raw.items():
        ms[f'raw_{yr}_{AREA["zh"][o]}'] = v
    for (yr, o), v in mean_pre.items():
        ms[f'prep_{yr}_{AREA["zh"][o]}'] = v
    ms.to_csv(f'{O}/70batch_mean_spectra.csv', index=False, encoding='utf-8-sig')

    # ---- 3. 描述量 ----
    q = []
    for (yr, o), v in mean_raw.items():
        tag = f'{yr}_{AREA["zh"][o]}'
        q.append((f'peak_wl_{tag}', wl[v.argmax()], 'nm，平均原始光谱最大值所在波长'))
        q.append((f'peak_value_{tag}', v.max(), '计数，平均原始光谱的最大值'))
        q.append((f'mean_value_{tag}', v.mean(), '计数，平均原始光谱在全部 2 048 个波长点上的均值'))
    for yr in FILES:
        m = np.mean([mean_raw[yr, o] for o in (1, 2, 3)], axis=0)
        sm = savgol_filter(m, 51, 3)
        idx = [i for i in argrelextrema(sm, np.less, order=40)[0] if 650 <= wl[i] <= 700]
        assert len(idx) == 1, (yr, [wl[i] for i in idx])
        q.append((f'local_min_wl_{yr}', wl[idx[0]], 'nm，三产地平均原始光谱（51 点三次 SG 平滑后）在 650~700 nm 内的局部极小'))
    for o in (1, 2, 3):
        q.append((f'intensity_ratio_2019_over_2018_{AREA["zh"][o]}', mean_raw[2019, o].mean() / mean_raw[2018, o].mean(),
                  '同一产地 2019 年与 2018 年平均原始光谱全波段均值之比'))
    # 年份之间（同产地）与产地之间（同年份）平均光谱之差的均方根，除以六个批次平均光谱均方根的均值。
    # 产地之间分两组：新疆对另两个产地（4 对），山东对陕西（2 对）
    for name, M in (('raw', mean_raw), ('prep', mean_pre)):
        scale = np.mean([rms(v) for v in M.values()])
        groups = {'between_year': ([rms(M[2019, o] - M[2018, o]) / scale for o in (1, 2, 3)], '同产地两年平均光谱之差，3 个产地'),
                  'xinjiang_vs_other_area': ([rms(M[yr, 1] - M[yr, b]) / scale for yr in FILES for b in (2, 3)],
                                             '同年份新疆与另一产地平均光谱之差，4 对'),
                  'shandong_vs_shaanxi': ([rms(M[yr, 2] - M[yr, 3]) / scale for yr in FILES], '同年份山东与陕西平均光谱之差，2 对')}
        for gname, (vals, what) in groups.items():
            q.append((f'{name}_{gname}_rel_rms_min', min(vals), f'{what}；差的均方根 ÷ 六批次平均光谱均方根的均值，最小值'))
            q.append((f'{name}_{gname}_rel_rms_max', max(vals), '同上，最大值'))
    summ = pd.DataFrame(q, columns=['quantity', 'value', 'note'])
    summ.to_csv(f'{O}/70batch_spectra_summary.csv', index=False, encoding='utf-8-sig')
    print(summ.to_string(index=False))

    # ---- 4. 图 2 ----
    t = T[LANG]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.35), gridspec_kw={'wspace': 0.36})
    for ax, M, ylab, k in ((axes[0], mean_raw, t['ya'], 1e-3), (axes[1], mean_pre, t['yb'], 1.0)):
        for (yr, o), v in M.items():
            ax.plot(wl, v * k, color=LOCC[o], ls=LS[yr], lw=0.9)
        ax.set_xlabel(t['xlab'])
        ax.set_ylabel(ylab)
        ax.set_xlim(590, 1250)
        ax.set_xticks([600, 700, 800, 900, 1000, 1100, 1200])
    zero_line(axes[1])
    for ax, lab in zip(axes, ('(a)', '(b)'), strict=True):
        panel_label(ax, lab)
    handles = [Line2D([], [], color=LOCC[o], lw=1.2) for o in (1, 2, 3)] + [Line2D([], [], color='#1A1A1A', lw=0.9, ls=LS[yr]) for yr in FILES]
    labels = [AREA[LANG][o] for o in (1, 2, 3)] + [t['yr'].format(y=yr) for yr in FILES]
    axes[0].legend(handles, labels, loc='upper right', ncol=1)
    save(fig, 'Figure_spectra_batches' + SUFFIX)
    print(f'✅ 70 完成（FIG_LANG={LANG}）')


if __name__ == '__main__':
    main()
