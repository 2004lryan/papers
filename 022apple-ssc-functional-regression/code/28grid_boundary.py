# -*- coding: utf-8 -*-
"""网格边界灵敏度（项目规范 7.3：「嵌套 CV 选出的 best_params 不构成『参数已合理』，
需在 CV 外额外报告灵敏度」）。

第一轮网格下有四个方法把选中值压在自己网格的上边界（SVR 的 C=1 000 五折全中、FPCR 的 K=30
五折全中、RF 的树数 800 三折、1D-CNN 的层数 4 五折全中）。选中值落在边界上意味着网格之外
可能还有更好的配置，把网格内的最优当成该方法的能力就是低估。02code/grids.py 已按统一规则
重定这四个方法的网格（规则见该文件头）。本脚本回答重定之后还剩什么问题：

  ① 逐方法诊断第二轮的选中值是否仍落在网格边界上，并把边界分成两类：
     **结构性上限**（越过它模型就无法定义，如 FPCR 的 K ≤ L−1、1D-CNN 的层数受特征图长度限制）
     与**人为截断**（网格画到哪里纯属选择，如 SVR 的 C、RF 的树数）。前者不需要延伸试验，
     因为不存在可延伸的空间；只有后者需要。
  ② 对仍存在人为截断的方法，在原网格之外向上延伸，用同一套嵌套 CV 协议重跑，
     报告精度的相对变化，对照 7.3 的 2% 判定线。

**本脚本的结果不替换表 3。** 表 3 报告的是预注册的等预算网格（每方法 12 组候选）下的结果，
事后单独加大某一个方法的网格再回填，属于按结果反向调参；本脚本只作为局限性的量化说明。

运行方式:
    cd <项目根目录>
    python3 02code/28grid_boundary.py

输出文件:
    04outputs/28grid_boundary.csv — 逐方法的边界诊断与延伸试验结果
    05logs/28grid_boundary.log    — 运行日志（由调用方重定向）

只读 05data。
"""
import ast
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

with open('02code/16unified_comparison.py') as _src: exec(_src.read().split("METHODS=['PLSR'")[0])
sys.path.insert(0, '02code')
from grids import GRIDS

K_OUTER, K_INNER, SEED = 5, 5, 20041210
folds = list(KFold(K_OUTER, shuffle=True, random_state=SEED).split(np.arange(len(y))))
NCV = '04outputs/25nested_cv.csv'

# 每个方法：从 picked 字符串取出「被重定的那一维」的取值、该维的候选取值、上界的性质。
# 上界性质决定要不要做延伸试验：结构性上限之外没有可搜索的空间，延伸试验无从谈起。
DIMS = {
    'PLSR':   ('潜变量数', lambda s: int(s),
               sorted(GRIDS['PLSR']), None),
    'SVR':    ('惩罚系数 C', lambda s: ast.literal_eval(s)[0],
               sorted({c for c, _ in GRIDS['SVR']}), '人为'),
    'RF':     ('树数', lambda s: ast.literal_eval(s)[0],
               sorted({n for n, _ in GRIDS['RF']}), '人为'),
    'FPCR':   ('主成分数 K', lambda s: ast.literal_eval(s)[1],
               sorted({k for _, k in GRIDS['FPCR']}), '结构：K ≤ L−1 = 79，主成分数不能超过基函数个数'),
    'PFLR':   ('粗糙惩罚 λ', lambda s: ast.literal_eval(s)[1],
               sorted({lam for _, lam in GRIDS['PFLR']}), '人为'),
    '1D-CNN': ('卷积层数', lambda s: ast.literal_eval(s)['nl'],
               sorted({c['nl'] for c in GRIDS['1D-CNN']}),
               '结构：每层长度降至 1/4，第 5 层后剩 2 个点，第 6 层为 0'),
}


def diagnose():
    """读第二轮的折内选中值，判断每个方法还有没有卡在网格边界上。"""
    if not os.path.exists(NCV):
        print(f"找不到 {NCV}，先跑 02code/25nested_cv.py"); return {}
    df = pd.read_csv(NCV)
    df = df[df.seed == SEED]
    out = {}
    print("【第二轮网格的边界诊断】看每个方法五折选中的取值落在网格的什么位置\n")
    print(f"{'方法':<9}{'被重定的维度':<14}{'五折选中值':<26}{'网格上界':>10}  判定")
    for m, (dim, get, cand, kind) in DIMS.items():
        a = df[df.method == m]
        if not len(a):
            print(f"{m:<9}{dim:<14}{'（未跑完）':<26}{str(cand[-1]):>10}  待补"); continue
        v = [get(s) for s in a.picked]
        hi = sum(x == cand[-1] for x in v)
        lo = sum(x == cand[0] for x in v)
        if hi and kind == '人为':
            note = f"⚠ {hi}/{len(v)} 折卡上界，需延伸试验"
        elif hi:
            note = f"{hi}/{len(v)} 折触上界，但那是{kind}"
        elif lo:
            note = f"{lo}/{len(v)} 折落在下界"
        else:
            note = "全部在网格内部"
        print(f"{m:<9}{dim:<14}{str(v):<26}{str(cand[-1]):>10}  {note}")
        out[m] = dict(values=v, hit_high=hi, hit_low=lo, kind=kind)
    print()
    return out


def nested(grid, fitf, tag):
    """与 25nested_cv.py 完全相同的嵌套 CV：内层选参、外层训练折重训练、外层测试折评价。"""
    r2s, picks = [], []
    for itr, ite in folds:
        inner = list(KFold(K_INNER, shuffle=True, random_state=SEED).split(itr))
        best, bc = np.inf, grid[0]
        for c in grid:
            e = float(np.mean([np.sqrt(mean_squared_error(y[itr[iv]], fitf(c, itr[ia], itr[iv], SEED)))
                               for ia, iv in inner]))
            if e < best: best, bc = e, c
        r2s.append(r2_score(y[ite], fitf(bc, itr, ite, SEED))); picks.append(str(bc))
    print(f"  {tag:<30} R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}  选中 {picks}", flush=True)
    return float(np.mean(r2s)), picks


hits = diagnose()
rows = []
print(f"【延伸试验】嵌套 CV K_outer={K_OUTER}/K_inner={K_INNER}，seed={SEED}\n")

# ---- SVR：第二轮 C 的上界 10^4 仍被一折选中，向上延伸两个数量级 ----
f_svr = mk('SVR')[1]
S0 = GRIDS['SVR']
S1 = [(C_, g_) for C_ in [10, 100, 1000, 10000, 100000, 1000000] for g_ in ['scale', 1e-3, 1e-4]]
print(f"SVR（第二轮 12 组 vs 向上延伸至 {len(S1)} 组）：")
t0 = time.time()
a0, p0 = nested(S0, f_svr, '第二轮 C≤10^4（12 组）')
a1, p1 = nested(S1, f_svr, f'延伸 C≤10^6（{len(S1)} 组）')
print(f"  用时 {time.time()-t0:.0f}s\n")
rows += [dict(method='SVR', grid='第二轮 12 组', R2=a0, picked=';'.join(p0)),
         dict(method='SVR', grid=f'延伸 {len(S1)} 组', R2=a1, picked=';'.join(p1))]

# ---- RF：第二轮树数的上界 1 600 被两折选中，向上延伸到 3 200 ----
# 为什么固定 max_features=0.1：第二轮五个外层折全部选中 0.1，这一维已经收敛；而 RF 的耗时
# 与「树数 × 每棵树用到的特征数」成正比，把 max_features 一并放开会让延伸试验的算力涨到
# 十小时量级。固定这一维只延伸树数，代价是无法排除「更多树需要配另一个 max_features」这种
# 交互效应，这一局限如实记在下面的输出里。
f_rf = mk('RF')[1]
R0 = [(n, 0.1) for n in [400, 800, 1600]]
R1 = [(n, 0.1) for n in [400, 800, 1600, 3200]]
print("RF（固定 max_features=0.1，只延伸树数）：")
t0 = time.time()
b0, q0 = nested(R0, f_rf, '第二轮 树数≤1600（3 组）')
b1, q1 = nested(R1, f_rf, '延伸 树数≤3200（4 组）')
print(f"  用时 {time.time()-t0:.0f}s\n")
rows += [dict(method='RF', grid='第二轮 树数≤1600（固定 mf=0.1）', R2=b0, picked=';'.join(q0)),
         dict(method='RF', grid='延伸 树数≤3200（固定 mf=0.1）', R2=b1, picked=';'.join(q1))]

df = pd.DataFrame(rows)
df.to_csv('04outputs/28grid_boundary.csv', index=False, encoding='utf-8-sig')

print("【相对变化】7.3 判据：相对变化 >2% 须在正文讨论敏感性")
for m, (x0, x1) in [('SVR', (a0, a1)), ('RF', (b0, b1))]:
    rel = 100 * (x1 - x0) / abs(x0)
    print(f"  {m:<6} 延伸后 R² 由 {x0:.3f} 变为 {x1:.3f}，相对变化 {rel:+.2f}%"
          f"{'  ← 超过判定线' if abs(rel) > 2 else ''}")

print("\n【不做延伸试验的三个方法及理由】")
print("  FPCR    K=79 已是结构性上限（K ≤ L−1，主成分数不能超过基函数个数），无可延伸空间")
print("  1D-CNN  层数 5 已是结构性上限（每层长度降至 1/4，第 5 层后剩 2 个点，第 6 层为 0）；")
print("          宽度方向 32 虽是上界，但卷积开销与宽度平方成正比，延伸到 64 需数小时量级，")
print("          按 7.3「算力受限须显式记录降级范围」登记为未做项")
print("  PLSR / PFLR  五折选中值全部落在网格内部，不存在边界截断")
print("\n已写出 04outputs/28grid_boundary.csv")
