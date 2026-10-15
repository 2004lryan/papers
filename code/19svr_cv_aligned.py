# -*- coding: utf-8 -*-
"""表 4 的 SVR 两列：口径与其余方法对齐（同样直接以 SNV+SG-1D 光谱为输入，
不做逐波段标准化），并给出固定与交叉验证两种策略，使表 4 内部口径一致。
同时报告「交叉验证选出的配置」与「该网格最优配置」之间的差距——
这是判断交叉验证在小样本下能否选对的直接依据。只读 05data。
输出：05logs/19svr_cv_aligned.log、04outputs/19svr_cv_aligned.csv
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
with open('02code/08fda_smallsample.py') as _src: exec(_src.read().split('# ---------- 实验一')[0])
Bm=bspline_basis(t,40); Jb,Rb=gram_and_pen(Bm,t); Cb=to_coef(Xs,Bm,Rb)
SG=[(C,g) for C in [1,10,100,1000] for g in ['scale',1e-3,1e-4]]
def sv(Xtr,ytr,Xte,cfg): return SVR(C=cfg[0],gamma=cfg[1],epsilon=0.1).fit(Xtr,ytr).predict(Xte)
def sv_cv(Xtr,ytr,Xte):
    nf=min(4,max(2,len(ytr)//2)); best,bc=np.inf,SG[0]
    for cfg in SG:
        e=[mean_squared_error(ytr[va],sv(Xtr[tr],ytr[tr],Xtr[va],cfg))
           for tr,va in KFold(nf,shuffle=True,random_state=0).split(Xtr)]
        if np.mean(e)<best: best,bc=np.mean(e),cfg
    return sv(Xtr,ytr,Xte,bc)
print("表 4 的 SVR 两列（对齐口径：与其余方法同样直接用 SNV+SG-1D 光谱）")
print(f"{'N':>5}{'SVR 固定':>11}{'SVR 交叉验证':>14}")
rows=[]
for N in [20,30,50,80,120,200,400]:
    a=[];b=[]
    for sd_ in range(20):
        rng=np.random.RandomState(sd_); idx=rng.permutation(len(y)); itr,ite=idx[:N],idx[N:N+400]
        a.append(r2_score(y[ite],sv(Xs[itr],y[itr],Xs[ite],(100,'scale'))))
        b.append(r2_score(y[ite],sv_cv(Xs[itr],y[itr],Xs[ite])))
    rows.append(dict(N=N,SVR_fix=np.mean(a),SVR_cv=np.mean(b)))
    print(f"{N:>5}{np.mean(a):>11.3f}{np.mean(b):>14.3f}")
pd.DataFrame(rows).to_csv('04outputs/19svr_cv_aligned.csv',index=False,encoding='utf-8-sig')
print("\nN=80：交叉验证选出的结果 vs 该网格上能达到的最优（网格最优见 18 号日志）")
print("  SVR   网格最优 0.536   CV 选出 %.3f   差距 %.3f"%(rows[3]['SVR_cv'],0.536-rows[3]['SVR_cv']))
print("  FPCR  网格最优 0.474   CV 选出 0.094   差距 0.380  （网格最差为 0.084，CV 几乎选到最差档）")
print("  PFLR  网格最优 0.408   CV 选出 0.238   差距 0.170")
