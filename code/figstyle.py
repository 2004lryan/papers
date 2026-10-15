# -*- coding: utf-8 -*-
"""论文插图的统一样式层：字体、字号、配色、面板标签、导出与对齐审计。所有画图脚本只从这里取样式。

约定（依据本刊模版「标目为 量/单位、图例使用中文」与出版级插图规范）：
  尺寸    通栏图宽 5.5 in（版心 14.65 cm，插入时缩至 14 cm），字号按最终尺寸设定，最小 6.5 pt
  字体    拉丁与数字 Arial，中文 思源黑体（Source Han Sans SC），逐字回退
  负号    刻度与文字一律用 U+2212（Arial 与思源黑体均含该字形），不用连字符
  坐标轴  只保留左、下两条轴线（L 形），刻度朝外，无网格
  配色    Okabe–Ito 色盲友好色系；方法固定配色：SVR 深灰黑、PLSR 中灰、PFLR 蓝、FPCR 朱红
  导出    PDF（Type 42 可编辑文字）+ SVG + 600 dpi PNG；多面板图先过对齐审计再导出，
          审计 JSON 存 04outputs/figs/qa/
"""
import os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
# 多面板对齐审计模块 audit_panel_alignment.py 不随代码分发：环境变量 PANEL_AUDIT_DIR 给出它所在的目录。
# 手头没有该模块时设 PANEL_AUDIT=off，图照常导出、不作对齐审计；两个变量都不设则报错退出。
_AUDIT_DIR = os.environ.get('PANEL_AUDIT_DIR')
if _AUDIT_DIR:
    sys.path.insert(0, os.path.expanduser(_AUDIT_DIR))
    from audit_panel_alignment import require_matplotlib_panel_alignment
elif os.environ.get('PANEL_AUDIT') == 'off':
    require_matplotlib_panel_alignment = None
else:
    raise SystemExit('figstyle: 未设环境变量 PANEL_AUDIT_DIR（多面板对齐审计模块 audit_panel_alignment.py 所在目录）；'
                     '没有该模块时设 PANEL_AUDIT=off，图照常导出、不作对齐审计。')

plt.rcParams.update({
    'font.family': ['Arial', 'Source Han Sans CN', 'DejaVu Sans'],   # 列表形式才会逐字回退：拉丁 Arial，中文思源黑体
    'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.borderaxespad': 0.2,
    'legend.labelspacing': 0.35, 'legend.handletextpad': 0.5,
    'axes.unicode_minus': True, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 600, 'lines.linewidth': 1.0,
    'axes.titlelocation': 'left', 'axes.titleweight': 'normal', 'axes.titlepad': 4,
})
W1, W2 = 3.35, 5.5                                   # 单栏 / 通栏宽（英寸）
C = dict(blue='#0072B2', vermilion='#D55E00', green='#009E73', orange='#E69F00', sky='#56B4E9',
         purple='#CC79A7', black='#1A1A1A', gray='#8C8C8C', lightgray='#CFCFCF', paleblue='#DCE9F5')
METHOD = {'SVR': C['black'], 'PLSR': C['gray'], 'PFLR': C['blue'], 'FPCR': C['vermilion']}
MARK = {'SVR': 'D', 'PLSR': 'o', 'PFLR': '^', 'FPCR': 's'}
LOC = {1: '新疆', 2: '山东', 3: '陕西'}
LOCC = {1: C['vermilion'], 2: C['blue'], 3: C['green']}
OUT = '04outputs/figs'; QA = f'{OUT}/qa'

def panel_label(ax, s, dx=-30, dy=2):
    """面板标签「(a)」：相对坐标区左上角固定点偏移，加粗 9 pt。"""
    off = ScaledTranslation(dx / 72, dy / 72, ax.figure.dpi_scale_trans)
    return ax.text(0, 1, s, transform=ax.transAxes + off, fontsize=9, fontweight='bold', ha='left', va='bottom')

def zero_line(ax, **kw):
    ax.axhline(0, color=C['black'], lw=0.5, ls=(0, (2, 2)), zorder=1, **kw)

def save(fig, name, multi=True, exclude=(), exemptions=None):
    """导出 PDF/SVG/PNG；多面板图先做渲染后对齐审计（1.5 pt 容差），不通过则抛错不导出。"""
    os.makedirs(QA, exist_ok=True)
    fig.canvas.draw()
    audited = multi and require_matplotlib_panel_alignment is not None
    if audited:
        require_matplotlib_panel_alignment(fig, json_out=f'{QA}/{name}.alignment.json',
                                           exclude_axes=list(exclude), exemptions=exemptions or [], strict=True)
    kw = dict(bbox_inches='tight', pad_inches=0.03)
    fig.savefig(f'{OUT}/{name}.pdf', **kw); fig.savefig(f'{OUT}/{name}.svg', **kw); fig.savefig(f'{OUT}/{name}.png', dpi=600, **kw)
    note = '，对齐审计通过' if audited else ('，未作对齐审计（PANEL_AUDIT=off）' if multi else '')
    plt.close(fig); print(f'  {name}: pdf/svg/png 已导出' + note)
