# -*- coding: utf-8 -*-
"""图 1（1.4 节）：研究流程与嵌套评价协议。

这是全文唯一的示意图，它要辩护的一句话是：
「六个方法在同一批折、同样的候选配置数下比较，外层测试折不参与任何一次选参与拟合；
 在此协议之上回答三个问题（方法比较、小样本选参、跨批次校正），系数函数作解释。」

三行顺着正文的次序读下来即本文的做法全貌，分区编号用 (a)~(c)，与正文清洗规则的名称互不混淆：
  (a) 数据、预处理与两条建模路线（1.1~1.3 节）
  (b) 统一协议：5折×5次重复的嵌套交叉验证（1.4 节）
  (c) 三个问题与一项解释（2.2~2.5 节），卡片编号即小节号
图中不放任何结果数字与结论：示意图只交代「怎么做」，结论由正文与图 3~6 给出。

版式：卡片一律纯文字、不嵌微缩图；按插入正文后的实际尺寸（通栏 14.65 cm），正文字号 7 pt、
卡片标题 7.5 pt、区标题 8.5 pt，均不小于本刊插图文字的可读下限。橙色带上用墨色字，其余色带用白字。
中英两版只有文字不同，数据、配色、字号与版式一致（项目规范 4.x 中英双语图）。

数值一律与 1.1、1.4 节一致：样品数取协议 B10.10 数据清洗后的保留数（04outputs/59data_clean_summary.csv），
外层 5 折的训练、测试样品数由它算出（KFold 把余数分给前几折），内层 5 折、每个方法 12 组候选、5 个随机种子重复。
四位数不分节（与正文一致），五位及以上每三位空一格。中文版的中文与数字、字母之间不留空格，
数值与单位之间留一格（与正文的空格规范一致）；卡片标题与 2.2~2.5 节标题同名或为其简称。

运行方式:
    cd <项目根目录>
    python3 02code/40figure_flow.py                # 中文版 → 04outputs/figs/Figure_0_flow.*
    FIG_LANG=en python3 02code/40figure_flow.py    # 英文版 → 04outputs/figs/Figure_0_flow_en.*
"""
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, '02code')
from figstyle import W2, C, save  # noqa: E402

LANG = os.environ.get('FIG_LANG', 'zh')
if LANG not in ('zh', 'en'):
    raise SystemExit(f'FIG_LANG 只能取 zh 或 en，收到 {LANG}')
N_KEPT = int(pd.read_csv('04outputs/59data_clean_summary.csv', dtype={'year': str}).set_index('year').loc['合计', 'kept'])
TE_LO, TE_HI = N_KEPT // 5, N_KEPT // 5 + (1 if N_KEPT % 5 else 0)       # 外层测试折的样品数
TR_LO, TR_HI = N_KEPT - TE_HI, N_KEPT - TE_LO                              # 外层训练折的样品数


def num(n):
    """四位数不分节，五位及以上每三位空一格。"""
    return str(n) if n < 10000 else f'{n:,}'.replace(',', ' ')


def rng(a, b):
    sep = '~' if LANG == 'zh' else '–'
    return num(a) if a == b else f'{num(a)}{sep}{num(b)}'


# 图中全部文字，中英两版逐项对应
T: dict[str, Any] = {
    'zh': dict(
        z1='数据、预处理与两条建模路线',
        c1=[('样品与光谱', ['3个产地×2个年份', f'清洗后{num(N_KEPT)}个苹果', '590.7~1249.6 nm，2048点']),
            ('预处理', ['SNV+SG一阶导数', '5种方案比较后选定', '附录A3折内复核']),
            ('函数型建模', ['三次B样条基，L=80', 'FPLR：粗糙惩罚λβ', 'FPCR：主成分数K']),
            ('离散建模', ['PLSR、SVR', 'RF、1D-CNN', '输入2048个波长点'])],
        z2='统一协议：5折×5次重复的嵌套交叉验证', z2note='外层测试折不参与选参与拟合',
        outer='外层5折', inner='内层5折', train='训练', test='测试', unit='个', val='训练折内的验证折',
        steps=['每个方法12组候选，按内层RMSE均值选优',
               '选中配置在外层训练折重拟合，测试折评价',
               '5个随机种子重复划分，六种方法共用同一批折',
               '共25个配对外层折，作修正t检验'],
        z3='三个问题与一项解释（卡片编号即小节号）',
        c3=[('2.2  方法比较', ['六种方法同折、同候选数', '25个配对折的差值', '15对Holm校正']),
            ('2.3  小样本选参', ['训练集N=20~400', '固定超参数/交叉验证', '每个N重抽20次']),
            ('2.4  跨批次校正', ['5种年份或产地批次', '目标批次抽k=5~40个', '每个k重抽200次']),
            ('2.5  系数函数', ['全样本部署模型的β(λ)', '25个训练折的变动范围', '极值窗口与文献谱带对照'])]),
    'en': dict(
        z1='Data, preprocessing and two modelling routes',
        c1=[('Samples and spectra', ['3 areas × 2 years', f'{num(N_KEPT)} apples after cleaning', '590.7–1249.6 nm, 2048 pts']),
            ('Preprocessing', ['SNV + SG first derivative', 'chosen among 5 schemes', 'checked in Appendix A3']),
            ('Functional models', ['cubic B-splines, L = 80', 'FPLR: penalty λβ', 'FPCR: components K']),
            ('Discrete models', ['PLSR, SVR', 'RF, 1D-CNN', 'input: 2048 wavelengths'])],
        z2='Unified protocol: 5 × 5 repeated nested CV',
        z2note='test folds never used for tuning or fitting',
        outer='Outer 5 folds', inner='Inner 5 folds', train='training', test='test', unit='', val='validation fold',
        steps=['12 candidates per method, chosen by mean inner RMSE',
               'refit on the outer training fold, scored on its test fold',
               '5 random seeds; all six methods share the same folds',
               '25 paired outer folds, corrected t test'],
        z3='Three questions and one interpretation (card number = section)',
        c3=[('2.2  Method comparison', ['same folds, 12 candidates', 'differences on 25 folds', '15 pairs, Holm adjustment']),
            ('2.3  Small training sets', ['training N = 20–400', 'fixed vs cross-validated', '20 draws per N']),
            ('2.4  Batch recalibration', ['5 year or area settings', 'k = 5–40 target samples', '200 draws per k']),
            ('2.5  Coefficient function', ['β(λ) of the deployed model', 'variation over 25 refits', 'windows vs reported bands'])]),
}[LANG]

FZ, FC, FB, FS = 8.5, 7.5, 7.0, 7.0          # 区标题 / 卡片标题 / 卡片正文 / 协议图注
INK, SUB, HAIR, PAPER = C['black'], '#4A4A4A', '#C9C9C9', '#FCFCFC'
ZONE = [C['blue'], C['orange'], C['green'], C['purple']]
ZTX = ['white', INK, 'white', 'white']
H = 3.55                                       # 图高（英寸）
BH, RS, LH = 0.052, 0.010, 0.050               # 卡片色带高 / 圆角 / 行距（画布坐标）


def zone(ax, y0, y1, col, label, title, note=None):
    """一个编号区：圆角外框 + 区标题（可在同一行右端附一句声明）。"""
    ax.add_patch(FancyBboxPatch((0.010, y0), 0.980, y1 - y0, boxstyle=f'round,pad=0,rounding_size={RS}',
                                fc='white', ec=col, lw=0.8, zorder=1))
    cy = y1 - 0.034
    ax.text(0.024, cy, f'{label}  {title}', ha='left', va='center', fontsize=FZ, fontweight='bold',
            color=col, zorder=3)
    if note:
        ax.text(0.976, cy, note, ha='right', va='center', fontsize=FS, color=col, zorder=3)


def card(ax, x, y, w, h, title, col, tc, lines):
    """一张卡片：圆角底 + 顶部色带标题 + 若干行说明。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={RS}',
                                fc=PAPER, ec=HAIR, lw=0.6, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - BH), w, BH, boxstyle=f'round,pad=0,rounding_size={RS}',
                                fc=col, ec='none', zorder=3))
    head = ax.text(x + 0.010, y + h - BH / 2, title, ha='left', va='center', fontsize=FC, fontweight='bold',
                   color=tc, zorder=4)
    return [head] + [ax.text(x + 0.010, y + h - BH - 0.036 - i * LH, s, ha='left', va='center', fontsize=FB,
                             color=INK, zorder=4) for i, s in enumerate(lines)]


def check_text_layout(fig, ax, cards):
    """渲染后检查文字：任意两段文字的外接框不相交，卡片内文字不越出卡片右缘；不通过则停下，不导出。"""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = [(t.get_text(), t.get_window_extent(rend)) for t in ax.texts if t.get_text()]
    bad = [f'{a!r} 与 {b!r}' for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:] if ba.overlaps(bb)]
    for x, w, texts in cards:
        right = ax.transData.transform((x + w, 0))[0]
        bad += [f'{t.get_text()!r} 越出卡片' for t in texts if t.get_window_extent(rend).x1 > right - 1]
    if bad:
        raise SystemExit('流程图文字重叠或越界：' + '；'.join(bad))


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8, lw=1.0,
                                 color=INK, shrinkA=0, shrinkB=0, zorder=5))


def flow():
    fig = plt.figure(figsize=(W2, H))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    cw, cg = 0.2335, 0.0100
    xs = [0.022 + i * (cw + cg) for i in range(4)]
    cards = []

    # ==================== (a) 数据、预处理与两条建模路线 ====================
    zone(ax, 0.690, 0.995, ZONE[0], '(a)', T['z1'])
    ch = 0.215
    for x, col, tc, (t, ls) in zip(xs, [ZONE[0], ZONE[0], ZONE[1], ZONE[1]], [ZTX[0], ZTX[0], ZTX[1], ZTX[1]],
                                   T['c1'], strict=True):
        cards.append((x, cw, card(ax, x, 0.705, cw, ch, t, col, tc, ls)))

    # ==================== (b) 统一协议：5折×5次重复的嵌套交叉验证 ====================
    zone(ax, 0.320, 0.675, ZONE[2], '(b)', T['z2'], T['z2note'])
    ox, oy, ow, rh, gp = 0.035, 0.545, 0.190, 0.022, 0.009
    for r in range(5):
        for c in range(5):
            ax.add_patch(Rectangle((ox + c * ow / 5, oy - r * (rh + gp)), ow / 5 - 0.003, rh,
                                   fc=C['blue'] if c == r else C['lightgray'], ec='none', zorder=3))
    ax.text(ox, oy + rh + 0.018, T['outer'], fontsize=FS, fontweight='bold', color=INK, zorder=3)
    yb = oy - 4 * (rh + gp) - 0.040
    sep = '' if LANG == 'zh' else ' '                  # 中文版的中文与数字之间不留空格
    for dy, fc_, lb in [(0.0, C['lightgray'], f"{T['train']}{sep}{rng(TR_LO, TR_HI)}{T['unit']}"),
                        (-0.036, C['blue'], f"{T['test']}{sep}{rng(TE_LO, TE_HI)}{T['unit']}")]:
        ax.add_patch(Rectangle((ox, yb + dy), 0.020, rh * 0.8, fc=fc_, ec='none', zorder=3))
        ax.text(ox + 0.026, yb + dy + rh * 0.4, lb, va='center', fontsize=FS, color=SUB, zorder=3)
    arrow(ax, ox + ow + 0.012, oy - 2 * (rh + gp) + rh / 2, ox + ow + 0.052, oy - 2 * (rh + gp) + rh / 2)

    ix, iw = 0.295, 0.160
    for r in range(5):
        for c in range(5):
            ax.add_patch(Rectangle((ix + c * iw / 5, oy - r * (rh + gp)), iw / 5 - 0.003, rh,
                                   fc=C['sky'] if c == r else C['lightgray'], ec='none', zorder=3))
    ax.text(ix, oy + rh + 0.018, T['inner'], fontsize=FS, fontweight='bold', color=INK, zorder=3)
    ax.add_patch(Rectangle((ix, yb), 0.020, rh * 0.8, fc=C['sky'], ec='none', zorder=3))
    ax.text(ix + 0.026, yb + rh * 0.4, T['val'], va='center', fontsize=FS, color=SUB, zorder=3)
    arrow(ax, ix + iw + 0.012, oy - 2 * (rh + gp) + rh / 2, ix + iw + 0.052, oy - 2 * (rh + gp) + rh / 2)

    tx = ix + iw + 0.066
    for i, s in enumerate(T['steps']):
        ax.text(tx, oy + rh - 0.006 - i * 0.052, f'{i + 1}. {s}', fontsize=FS, color=INK, va='center', zorder=3)

    # ==================== (c) 三个问题与一项解释 ====================
    zone(ax, 0.005, 0.305, ZONE[3], '(c)', T['z3'])
    for x, (t, ls) in zip(xs, T['c3'], strict=True):
        cards.append((x, cw, card(ax, x, 0.020, cw, ch, t, ZONE[3], ZTX[3], ls)))

    check_text_layout(fig, ax, cards)
    save(fig, 'Figure_0_flow' + ('' if LANG == 'zh' else '_en'), multi=False)


if __name__ == '__main__':
    flow()
    print('流程图已导出到 04outputs/figs/Figure_0_flow' + ('' if LANG == 'zh' else '_en') + '.*')
