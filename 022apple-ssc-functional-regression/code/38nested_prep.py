# -*- coding: utf-8 -*-
"""把预处理选择嵌套进每个外层训练折内，重做表 3 的方法比较。

为什么要有这个脚本：表 3 的六方法比较统一用 SNV+SG-1D 预处理，而这一选择是在看过表 1 的
交叉验证结果之后作出的（3.1 节），表 3 又复用同一批外层折。按 Kapoor & Narayanan (2023) 的
分类，这属于 L1 类「没有干净分离」中的「选择先于评价」：预处理不是先验固定的设定，而是一次
用到了后来被当作测试折的数据的选择。

`36prep_leakage.py` 换五种预处理各重跑一遍完整嵌套交叉验证，回答的是「换个预处理排序变不变」，
是灵敏度检验——它不消除这一条泄漏，只说明泄漏的后果有多大。本脚本做的是消除它本身：
把预处理并入超参数，在**每个外层训练折内部**用内层 5 折从 5 种预处理 × 12 组候选 = 60 个组合
中选优，再用选中的组合在整个外层训练折上重拟合、在外层测试折上评价。外层测试折的数据
不参与任何一次选择。

计算量与 36 脚本相同（同为 5 外层 × 5 内层 × 60 组合），因为 36 脚本本就把五种预处理各跑满了
一遍；差别只在于「按折内选」还是「按全局选」。故本检验不额外消耗机时预算。

RF 与 1D-CNN 未纳入，与 36 脚本同一理由：二者在五种预处理上重跑约需 25 h 机时，超出本研究的
算力预算。这两个方法在表 3 中排末两位，不承载本文的两条核心论断
（「SVR 不低于两种函数型模型」与「函数型模型不低于全波段 PLSR」）。

协议（与表 3 逐项对齐，只有「预处理何时选定」一项不同）：
  外层    KFold(5, shuffle=True, random_state=20041210)——与表 3 同一批折
  内层    在外层训练折内 KFold(5, shuffle=True, random_state=20041210)
  候选    5 种预处理 × 各方法 12 组超参 = 60 组，四个方法预算相同
  选参    内层折外预测的 MSE 最小
  评价    外层测试折上的折外预测拼合后一次算出 R²/RMSE/RPD

预处理逐样本计算（SNV 按行标准化、SG 一阶导沿波长卷积、MSC 以该折训练样本的均值谱为参考），
其中只有 MSC 用到跨样本统计量，故 MSC 的参考谱**只用外层训练折的样本**计算，再应用到测试折；
其余四种在划分前整体计算不构成泄漏。

运行方式:
    cd <项目根目录>
    python3 02code/38nested_prep.py

输出文件:
    04outputs/38nested_prep.csv   — 每个方法一行：嵌套选预处理的 R²/RMSE/RPD + 各折选中的预处理
    05logs/38nested_prep.log      — 运行日志（由调用方重定向）

只读 05data。
"""
from typing import Any
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

with open('02code/16unified_comparison.py') as _src: exec(_src.read().split("METHODS=['PLSR'")[0])
sys.path.insert(0, '02code')
from grids import GRIDS                                          # noqa: E402

SEED, K_OUTER, K_INNER = 20041210, 5, 5
METHODS = ['PLSR', 'SVR', 'FPCR', 'PFLR']
OUT = '04outputs/38nested_prep.csv'


def msc_fit(A, ref):
    """MSC 是唯一用到跨样本统计量的预处理，故参考谱由调用方给出（只能来自训练折）。"""
    out = np.empty_like(A)
    for i, r in enumerate(A):
        k, b = np.polyfit(ref, r, 1)
        out[i] = (r - b) / k
    return out


# 每个预处理写成 (需要训练折参考谱?, 变换函数)。需要参考谱的那一个在每次划分内单独拟合。
PREP = {
    '原始':      (False, lambda A, ref: A),
    'SNV':       (False, lambda A, ref: snv(A)),
    'MSC':       (True,  lambda A, ref: msc_fit(A, ref)),
    'SG1D':      (False, lambda A, ref: savgol_filter(A, 15, 2, deriv=1, axis=1)),
    'SNV+SG1D':  (False, lambda A, ref: savgol_filter(snv(A), 15, 2, deriv=1, axis=1)),
}

folds = list(KFold(K_OUTER, shuffle=True, random_state=SEED).split(np.arange(len(y))))

print(f'预处理选择嵌套进外层训练折（外层 {K_OUTER} 折、内层 {K_INNER} 折、'
      f'5 预处理 × 12 超参 = 60 组候选，seed={SEED}）')
print(f'方法：{", ".join(METHODS)}；RF 与 1D-CNN 因机时超预算未纳入，理由见文件头\n')


def apply_prep(pname, train_idx):
    """按 train_idx 所指的训练样本拟合该预处理，再应用到全体样本。

    需要参考谱的预处理（MSC）只用训练样本求参考谱；其余逐样本变换，训练集与否不影响结果。
    返回值覆盖全局 Xp 与 COEF，因为 mk(m)[1] 返回的 fit 函数读的就是这两个全局名。
    """
    need_ref, f = PREP[pname]
    ref = X[train_idx].mean(0) if need_ref else None
    Xp_ = f(X, ref)
    globals()['Xp'] = Xp_
    globals()['COEF'] = {L: to_coef(Xp_, BASIS[L][0], BASIS[L][2]) for L in BASIS}


rows = []
for m in METHODS:
    t0 = time.time()
    grid, fit = GRIDS[m], mk(m)[1]
    oof = np.zeros(len(y))
    picked = []
    for k, (itr, ite) in enumerate(folds):
        inner = list(KFold(K_INNER, shuffle=True, random_state=SEED).split(itr))
        best: Any = None
        best_se = np.inf
        for pname in PREP:
            # 内层的每一次拟合都只看外层训练折：参考谱取 itr[ia]，评价在 itr[iv]
            for c in grid:
                iv_oof = np.zeros(len(itr))
                for ia, iv in inner:
                    apply_prep(pname, itr[ia])
                    iv_oof[iv] = fit(c, itr[ia], itr[iv], SEED)
                se = mean_squared_error(y[itr], iv_oof)
                if se < best_se:
                    best_se, best = se, (pname, c)
        apply_prep(best[0], itr)                       # 选中的组合在整个外层训练折上重拟合
        oof[ite] = fit(best[1], itr, ite, SEED)
        picked.append(best[0])
        print(f'  {m:<6} fold={k}  选中预处理={best[0]:<9} 超参={best[1]}', flush=True)
    r2 = float(r2_score(y, oof))
    rmse = float(np.sqrt(mean_squared_error(y, oof)))
    rows.append(dict(method=m, R2=r2, RMSE=rmse, RPD=float(np.std(y)) / rmse,
                     picked='|'.join(picked),
                     n_snv_sg1d=sum(p == 'SNV+SG1D' for p in picked),
                     sec=round(time.time() - t0, 1)))
    print(f'  {m:<6} R²={r2:.3f}  RMSE={rmse:.3f}  各折选中={picked}  ({time.time()-t0:.0f}s)\n',
          flush=True)

d = pd.DataFrame(rows).sort_values('R2', ascending=False).reset_index(drop=True)
d.to_csv(OUT, index=False)

print('【嵌套选预处理后的方法排序】')
print(f'{"方法":<8}{"R²":>9}{"RMSE":>9}{"RPD":>8}   各折选中的预处理')
for _, r in d.iterrows():
    print(f'{r.method:<8}{r.R2:>9.3f}{r.RMSE:>9.3f}{r.RPD:>8.2f}   {r.picked}')
print(f'\n排序：{" > ".join(d.method)}')
print(f'→ {OUT}')
