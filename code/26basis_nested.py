# -*- coding: utf-8 -*-
"""表 2（基函数族 × 基个数）改按 7.1 的嵌套交叉验证重跑。

为什么要有这个脚本：08fda_smallsample.py 的「实验一」用的是 3 次 70%/30% 随机划分、
内层 4 折选参，与表 3 改用嵌套 CV 之后的评价口径不一致——同一篇论文里两张表若用不同的
评价协议，读者无法把「傅里叶基 L=80 最好」与表 3 的方法排序放在一起看。本脚本把该实验
搬到与 25nested_cv.py 完全相同的协议上：外层 KFold(5, shuffle=True)、内层 KFold(5)，
每个外层折独立选参、重训练、评价。

候选网格：PFLR 的 λ_β 取 logspace(-10, 2, 13)；FPCR 的主成分个数 K 按**基个数的比例**取
round(L × {0.1, 0.2, …, 0.9})，即每个 L 都得到 9 个候选、且恒满足 K ≤ L−1。
之所以用比例而非固定的 K 列表：K 的可取上限本就是 L−1，用固定列表（如 K ≤ 30）会让 L=80 这一行
被上限截断、L=20 那一行却搜遍全程，各行的搜索预算不对等，比较基函数个数的结论就会偏向小 L。
两法各扫各的网格，不与表 3 的「12 组等预算」挂钩——本表的用途是在函数型方法内部选基函数族与
基个数，不参与方法之间的优劣判定。
只读 05data。
"""
import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

with open('02code/08fda_smallsample.py') as _src: exec(_src.read().split('# ---------- 实验一')[0])   # 数据、基系统、pflr/fpcr

K_OUTER, K_INNER, SEED = 5, 5, 20041210
LAM_GRID = np.logspace(-10, 2, 13)

def pick_inner(C, yy, J, R, kind, grid):
    """在给定的（外层训练折）样本上用 K_INNER 折交叉验证选参。"""
    best, bp = np.inf, grid[0]
    for gpar in grid:
        e = [mean_squared_error(yy[va], pflr(C[tr], yy[tr], C[va], J, R, gpar) if kind == 'pflr'
                                else fpcr(C[tr], yy[tr], C[va], J, gpar))
             for tr, va in KFold(K_INNER, shuffle=True, random_state=SEED).split(C)]
        if np.mean(e) < best: best, bp = np.mean(e), gpar
    return bp

folds = list(KFold(K_OUTER, shuffle=True, random_state=SEED).split(np.arange(len(y))))
print(f"表 2 重跑：嵌套 CV K_outer={K_OUTER} / K_inner={K_INNER}，seed={SEED}，n={len(y)}\n")
print(f"{'基族':<8}{'L':>5}{'PFLR R²':>12}{'FPCR R²':>12}{'PFLR RMSE':>12}{'FPCR RMSE':>12}")
rows = []
for fam, fn in [('B样条', bspline_basis), ('傅里叶', fourier_basis)]:
    for L in [20, 40, 60, 80]:
        B = fn(t, L); J, R = gram_and_pen(B, t); C = to_coef(Xs, B, R)
        Kg = sorted({max(1, int(round(L * f))) for f in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]})
        r1, r2_, m1, m2 = [], [], [], []
        for itr, ite in folds:
            lam = pick_inner(C[itr], y[itr], J, R, 'pflr', LAM_GRID)
            p1 = pflr(C[itr], y[itr], C[ite], J, R, lam)
            K = pick_inner(C[itr], y[itr], J, R, 'fpcr', Kg)
            p2 = fpcr(C[itr], y[itr], C[ite], J, K)
            r1.append(r2_score(y[ite], p1)); r2_.append(r2_score(y[ite], p2))
            m1.append(np.sqrt(mean_squared_error(y[ite], p1)))
            m2.append(np.sqrt(mean_squared_error(y[ite], p2)))
        print(f"{fam:<8}{L:>5}{np.mean(r1):>12.3f}{np.mean(r2_):>12.3f}{np.mean(m1):>12.3f}{np.mean(m2):>12.3f}")
        rows.append(dict(basis=fam, L=L, PFLR_R2=np.mean(r1), PFLR_R2_sd=np.std(r1),
                         FPCR_R2=np.mean(r2_), FPCR_R2_sd=np.std(r2_),
                         PFLR_RMSE=np.mean(m1), PFLR_RMSE_sd=np.std(m1),
                         FPCR_RMSE=np.mean(m2), FPCR_RMSE_sd=np.std(m2)))
df = pd.DataFrame(rows); df.to_csv('04outputs/26basis_nested.csv', index=False, encoding='utf-8-sig')
b = df.loc[df.PFLR_R2.idxmax()]
print(f"\nPFLR 最优：{b.basis} L={int(b.L)}  R²={b.PFLR_R2:.3f}")
b2 = df.loc[df.FPCR_R2.idxmax()]
print(f"FPCR 最优：{b2.basis} L={int(b2.L)}  R²={b2.FPCR_R2:.3f}")
print("\n已写出 04outputs/26basis_nested.csv")
