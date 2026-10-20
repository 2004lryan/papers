# -*- coding: utf-8 -*-
"""表 1（预处理 × 建模方法）改按 7.1 的交叉验证重跑。

为什么要有这个脚本：04baseline_models.py 用的是 2 次 70%/30% 随机划分，与表 3 改用嵌套交叉
验证之后的评价口径不一致。本脚本把外层评价换成与 25nested_cv.py 相同的 KFold(5, shuffle=True)，
使全文报告的每一个精度都出自 5 折交叉验证。

超参数的处理与 04 脚本一致，不作改动：PLSR 的潜变量数在每个外层训练折内以 5 折交叉验证选取，
SVR、RF 与 1D-CNN 用各自的默认设置。这是有意为之——本表的用途是在四个模型上同时观察
「换预处理会怎样」，模型侧固定不动才能把差异归因到预处理；方法之间的对等超参数搜索在表 3
的统一协议下进行（每方法 12 组候选、同一批折）。
只读 05data。
"""
import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np, pandas as pd, time, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 复用 04 脚本的数据装载、5 种预处理、CNN 定义与 run_cnn（切到实验循环之前）
with open('02code/04baseline_models.py') as _src: exec(_src.read().split('SEEDS=[0,1]')[0])

K_OUTER, K_INNER, SEED = 5, 5, 20041210
folds = list(KFold(K_OUTER, shuffle=True, random_state=SEED).split(np.arange(len(y))))
print(f"表 1 重跑：外层 KFold({K_OUTER}, shuffle=True) seed={SEED}；PLSR 潜变量数在折内以 "
      f"{K_INNER} 折交叉验证选取，其余三法用默认设置\n")
print(f"{'预处理':<10}{'方法':<10}{'R²':>8}{'RMSE':>9}{'RPD':>7}")
print("-" * 46)

rows = {}
for pname, pf in PREP.items():
    Xp = pf(X)
    for mname in ['PLSR', 'SVR', 'RF', '1D-CNN']:
        t0 = time.time(); Rm = []
        for itr, ite in folds:
            Xtr, Xte, ytr, yte = Xp[itr], Xp[ite], y[itr], y[ite]
            if mname == 'PLSR':
                bn, bs_ = 8, np.inf
                for nc in range(2, 26, 2):
                    e = [mean_squared_error(ytr[va], PLSRegression(n_components=nc)
                         .fit(Xtr[tr], ytr[tr]).predict(Xtr[va]).ravel())
                         for tr, va in KFold(K_INNER, shuffle=True, random_state=SEED).split(Xtr)]
                    if np.mean(e) < bs_: bs_, bn = np.mean(e), nc
                yp = PLSRegression(n_components=bn).fit(Xtr, ytr).predict(Xte).ravel()
            elif mname == 'SVR':
                mu, sdv = Xtr.mean(0), Xtr.std(0) + 1e-8
                yp = SVR(C=100, gamma='scale', epsilon=0.1).fit((Xtr - mu) / sdv, ytr).predict((Xte - mu) / sdv)
            elif mname == 'RF':
                yp = RandomForestRegressor(300, n_jobs=-1, random_state=SEED).fit(Xtr, ytr).predict(Xte)
            else:
                yp = run_cnn(Xtr, ytr, Xte, seed=SEED)
            Rm.append(metrics(yte, yp))
        a = np.array(Rm); mu_ = a.mean(0); sd_ = a.std(0)
        rows[(pname, mname)] = (mu_, sd_)
        print(f"{pname:<10}{mname:<10}{mu_[0]:>8.3f}{mu_[1]:>9.3f}{mu_[2]:>7.2f}   "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("-" * 46)

k = max(rows, key=lambda kk: rows[kk][0][0])
print(f"\n最优组合：{k[0]} + {k[1]}   R²={rows[k][0][0]:.3f}  RMSE={rows[k][0][1]:.3f}  RPD={rows[k][0][2]:.2f}")
# 逐预处理取四法均值，作为「哪种预处理整体最稳」的依据
print("\n各预处理在四个模型上的平均 R²：")
for pname in PREP:
    v = np.mean([rows[(pname, m)][0][0] for m in ['PLSR', 'SVR', 'RF', '1D-CNN']])
    print(f"  {pname:<10}{v:.3f}")
pd.DataFrame([{'预处理': a, '方法': b, 'R2': v[0][0], 'R2_sd': v[1][0], 'RMSE': v[0][1],
               'RMSE_sd': v[1][1], 'RPD': v[0][2], 'RPD_sd': v[1][2]}
              for (a, b), v in rows.items()]).to_csv('04outputs/27prep_nested.csv',
                                                     index=False, encoding='utf-8-sig')
print("\n已写出 04outputs/27prep_nested.csv")
