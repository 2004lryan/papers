# -*- coding: utf-8 -*-
"""图 3（单面板）：PFLR 的回归系数函数 β(λ) 与其极值窗口。

数据取自 04outputs/beta_function.csv（由 12figures.py 在 B 样条 L=80、λ 经 4 折交叉验证选出后落盘）。
按区间建模的精度与信噪比这两组数字改写进 3.4 节正文，本图只留系数函数本身——
它要辩护的是一句话：系数主峰 868~957 nm 跨越 900 nm 这一常规划分界线。
"""
import sys, pathlib
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from figstyle import W2, C, zero_line, save

b = pd.read_csv('04outputs/beta_function.csv')
wl, bs = b.wavelength_nm.to_numpy(), b.beta.to_numpy() / 1e5

fig, a = plt.subplots(figsize=(W2, 1.75), layout='constrained')
a.plot(wl, bs, color=C['blue'], lw=1.0, zorder=3); zero_line(a)
a.fill_between(wl, 0, bs, where=bs > 0, color=C['blue'], alpha=0.18, lw=0)
a.fill_between(wl, 0, bs, where=bs < 0, color=C['vermilion'], alpha=0.18, lw=0)
a.set_ylim(bs.min() * 1.15, bs.max() * 1.42)
for lo, hi, lab, al in [(868, 957, '868~957 nm', 0.12), (1125, 1153, '1 125~1 153 nm', 0.07)]:
    a.axvspan(lo, hi, color=C['blue'], alpha=al, lw=0, zorder=0)
    a.text((lo + hi) / 2, bs.max() * 1.22, lab, ha='center', va='bottom', fontsize=6.5, color=C['blue'])
a.axvline(900, ymax=0.84, color=C['gray'], lw=0.6, ls='--')
a.text(910, bs.min() * 1.05, '900 nm', fontsize=6.5, color=C['gray'], va='bottom', ha='left')
a.set_xlim(580, 1260); a.set_xticks([600, 700, 800, 900, 1000, 1100, 1200])
a.set_xlabel('波长/nm'); a.set_ylabel('回归系数 β(λ)/10⁵')
save(fig, 'Figure_4_beta', multi=False)
print('单面板系数函数图已导出')
