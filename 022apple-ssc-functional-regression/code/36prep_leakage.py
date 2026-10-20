# -*- coding: utf-8 -*-
"""预处理是先验固定还是选出来的：换一种预处理，表 3 的排序会不会变（投稿清单第 13 条的补充）。

为什么要有这个脚本：投稿前的诚信审计（2026-09-07）指出一个方法学缺口——
3.1 节的预处理是**看过表 1 的交叉验证结果之后**选定 SNV+SG-1D 的，而表 3 的主比较又用同一批
外层折。选择用到了后来被当作测试折的数据，属 Kapoor & Narayanan (2023) 分类里的
「L1 无干净分离 / 选择先于评价」。这是真实的方法论缺口，不能只在正文加一句免责了事。

严格的修法是把预处理选择嵌套进每个外层训练折内重跑全部六个方法，代价是重跑一遍表 3
（主分析六方法合计约 34 小时机时）。本脚本改问一个更直接、也更能说明问题的问题：
**这个选择究竟影响了什么？** 若把预处理换成其余四种中的任何一种，表 3 的方法排序是否改变？
若不变，则该选择虽不严格嵌套，但对论文的结论不构成实质影响，可如实登记其性质与影响范围；
若变了，则结论依赖于那次选择，必须改写。

范围与代价（如实登记）：只跑 PLSR、SVR、FPCR、PFLR 四个方法。RF 与 1D-CNN 在五种预处理上
重跑需约 25 小时机时，超出本轮预算；这四个方法已覆盖论文的核心论断——
「SVR 不低于两种函数型模型」与「函数型模型优于 PLSR」——即表 3 排序的主干。

协议与表 3 完全一致：外层 KFold(5, shuffle=True, seed=20041210)，内层 5 折选参，
每方法 12 组候选（02code/grids.py），只有预处理这一项在变。

预处理逐折拟合（2026-09-07 修正）：五种预处理中只有 MSC 用到跨样本统计量（参考谱）。
初版对全体样本一次算好参考谱再划折，外层测试折的光谱因此进入了参考谱，MSC 一行自身带有
K&N L1 类「预处理在划分前用全部数据拟合」的泄漏——本脚本原本要检验的正是这一类问题。
现改为在每次划分内部拟合：内层的参考谱只取内层训练样本，外层重拟合的参考谱只取外层训练折。
其余四种逐样本变换，不受此改动影响，数值与初版一致。

运行方式:
    cd <项目根目录>
    python3 02code/36prep_leakage.py

输出文件:
    04outputs/36prep_leakage.csv   — 5 种预处理 × 4 方法的嵌套 CV 结果与各自的排序
    05logs/36prep_leakage.log      — 运行日志（由调用方重定向）

只读 05data。
"""
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import kendalltau

with open('02code/16unified_comparison.py') as _src: exec(_src.read().split("METHODS=['PLSR'")[0])
sys.path.insert(0, '02code')
from grids import GRIDS                                          # noqa: E402

SEED, K_OUTER, K_INNER = 20041210, 5, 5
METHODS = ['PLSR', 'SVR', 'FPCR', 'PFLR']


def msc(A, ref):
    """MSC 是五种预处理里唯一用到跨样本统计量的一种，故参考谱由调用方给出。

    参考谱必须只来自训练样本：若取全体样本的均值谱，外层测试折的光谱就进入了参考谱，
    构成 Kapoor & Narayanan (2023) L1 类中的「预处理在划分前用全部数据拟合」。
    """
    out = np.empty_like(A)
    for i, r in enumerate(A):
        k, b = np.polyfit(ref, r, 1)
        out[i] = (r - b) / k
    return out


# (是否需要训练样本的参考谱, 变换函数)。需要的那一个在每次划分内单独拟合。
PREP = {
    '原始':      (False, lambda A, ref: A),
    'SNV':       (False, lambda A, ref: snv(A)),
    'MSC':       (True,  lambda A, ref: msc(A, ref)),
    'SG1D':      (False, lambda A, ref: savgol_filter(A, 15, 2, deriv=1, axis=1)),
    'SNV+SG1D':  (False, lambda A, ref: savgol_filter(snv(A), 15, 2, deriv=1, axis=1)),
}


def apply_prep(pname, train_idx):
    """按 train_idx 所指的训练样本拟合该预处理，再应用到全体样本，并覆盖全局 Xp / COEF。"""
    need_ref, f = PREP[pname]
    Xp_ = f(X, X[train_idx].mean(0) if need_ref else None)
    globals()['Xp'] = Xp_
    globals()['COEF'] = {L: to_coef(Xp_, BASIS[L][0], BASIS[L][2]) for L in BASIS}

folds = list(KFold(K_OUTER, shuffle=True, random_state=SEED).split(np.arange(len(y))))
print(f'预处理选择对表 3 排序的影响（协议与表 3 同：外层 {K_OUTER} 折、内层 {K_INNER} 折、'
      f'每方法 12 组候选，seed={SEED}）')
print(f'方法：{", ".join(METHODS)}；RF 与 1D-CNN 因机时超预算未纳入，理由见文件头\n')

rows = []
for pname in PREP:
    for m in METHODS:
        t0 = time.time()
        grid, fit = GRIDS[m], mk(m)[1]
        oof = np.zeros(len(y))
        for itr, ite in folds:
            inner = list(KFold(K_INNER, shuffle=True, random_state=SEED).split(itr))
            se = []
            for c in grid:
                iv_oof = np.zeros(len(itr))
                for ia, iv in inner:
                    apply_prep(pname, itr[ia])       # 内层：参考谱只来自内层训练样本
                    iv_oof[iv] = fit(c, itr[ia], itr[iv], SEED)
                se.append(mean_squared_error(y[itr], iv_oof))
            apply_prep(pname, itr)                   # 外层重拟合：参考谱只来自外层训练折
            oof[ite] = fit(grid[int(np.argmin(se))], itr, ite, SEED)
        r2 = float(r2_score(y, oof))
        rmse = float(np.sqrt(mean_squared_error(y, oof)))
        rows.append(dict(prep=pname, method=m, R2=r2, RMSE=rmse, RPD=float(np.std(y)) / rmse,
                         sec=round(time.time() - t0, 1)))
        print(f'  {pname:<10}{m:<8} R²={r2:.3f}  RMSE={rmse:.3f}  ({time.time()-t0:.0f}s)', flush=True)

d = pd.DataFrame(rows)
d['rank'] = d.groupby('prep').R2.rank(ascending=False).astype(int)
d.to_csv('04outputs/36prep_leakage.csv', index=False)

print('\n【各预处理下的方法排序】（1 = R² 最高）')
print(f'{"预处理":<11}' + ''.join(f'{m:>10}' for m in METHODS) + '   排序')
base = None
for p in PREP:
    sub = d[d.prep == p].set_index('method')
    order = tuple(sub.R2.sort_values(ascending=False).index)
    if base is None:
        base = order
    print(f'{p:<11}' + ''.join(f'{sub.loc[m, "R2"]:>10.3f}' for m in METHODS) +
          f'   {" > ".join(order)}' + ('' if order == base else '   ← 与首行不同'))

print('\n【与 SNV+SG1D（论文所选）的排序一致性】')
ref = d[d.prep == 'SNV+SG1D'].set_index('method').R2
same = 0
for p in PREP:
    if p == 'SNV+SG1D':
        continue
    cur = d[d.prep == p].set_index('method').R2
    tau = kendalltau(ref[METHODS].to_numpy(), cur[METHODS].to_numpy()).statistic
    ident = tuple(ref.sort_values(ascending=False).index) == tuple(cur.sort_values(ascending=False).index)
    same += ident
    print(f'  {p:<10} Kendall τ = {tau:+.3f}   排序{"完全相同" if ident else "不同"}')
print(f'\n结论：其余 4 种预处理中有 {same} 种给出与 SNV+SG1D 完全相同的方法排序。')
print('→ 04outputs/36prep_leakage.csv')
