# -*- coding: utf-8 -*-
"""图 3 的不确定度来源：把 10smallsample_hyperparam.py 与 19svr_cv_aligned.py 的 20 次重抽逐次记录下来
（同一批种子、同一口径），输出长表 04outputs/21smallsample_repeats.csv（N, method, strategy, seed, R2）。
每个 (N, method, strategy) 的均值必须与 04outputs/10smallsample_hyperparam.csv、19svr_cv_aligned.csv
里已发表的三位小数一致，否则报错——这是对表 4 数字的一次独立复算。只读 05data。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
with open('02code/08fda_smallsample.py') as _src: exec(_src.read().split('# ---------- 实验一')[0])
Bm=bspline_basis(t,40); J,R=gram_and_pen(Bm,t); C=to_coef(Xs,Bm,R)
SG=[(c,g) for c in [1,10,100,1000] for g in ['scale',1e-3,1e-4]]
def sv(Xtr,ytr,Xte,cfg): return SVR(C=cfg[0],gamma=cfg[1],epsilon=0.1).fit(Xtr,ytr).predict(Xte)
def sv_cv(Xtr,ytr,Xte):
    nf=min(4,max(2,len(ytr)//2)); best,bc=np.inf,SG[0]
    for cfg in SG:
        e=[mean_squared_error(ytr[va],sv(Xtr[tr],ytr[tr],Xtr[va],cfg))
           for tr,va in KFold(nf,shuffle=True,random_state=0).split(Xtr)]
        if np.mean(e)<best: best,bc=np.mean(e),cfg
    return sv(Xtr,ytr,Xte,bc)

rows=[]
for N in [20,30,50,80,120,200,400]:
    for sd in range(20):
        rng=np.random.RandomState(sd); idx=rng.permutation(len(y)); itr,ite=idx[:N],idx[N:N+400]
        yt=y[ite]
        rows.append((N,'PLSR','fixed',sd,r2_score(yt,plsr(Xs[itr],y[itr],Xs[ite]))))
        Kf=min(10,max(2,N-2))
        rows.append((N,'FPCR','fixed',sd,r2_score(yt,fpcr(C[itr],y[itr],C[ite],J,Kf))))
        Kg=[k for k in [2,5,10,15,20,30] if k<=max(2,int(N*0.75)-2)] or [2]
        Kc=cv_pick(C[itr],y[itr],J,R,'fpcr',Kg)
        rows.append((N,'FPCR','cv',sd,r2_score(yt,fpcr(C[itr],y[itr],C[ite],J,Kc))))
        rows.append((N,'PFLR','fixed',sd,r2_score(yt,pflr(C[itr],y[itr],C[ite],J,R,1e-3))))
        Lc=cv_pick(C[itr],y[itr],J,R,'pflr',np.logspace(-8,2,6))
        rows.append((N,'PFLR','cv',sd,r2_score(yt,pflr(C[itr],y[itr],C[ite],J,R,Lc))))
        rows.append((N,'SVR','fixed',sd,r2_score(yt,sv(Xs[itr],y[itr],Xs[ite],(100,'scale')))))
        rows.append((N,'SVR','cv',sd,r2_score(yt,sv_cv(Xs[itr],y[itr],Xs[ite]))))
    print(f"N={N} 完成",flush=True)
df=pd.DataFrame(rows,columns=['N','method','strategy','seed','R2'])
df.to_csv('04outputs/21smallsample_repeats.csv',index=False,encoding='utf-8-sig')

# ---- 与已发表数字对账（三位小数） ----
pub=pd.read_csv('04outputs/10smallsample_hyperparam.csv').merge(pd.read_csv('04outputs/19svr_cv_aligned.csv'),on='N')
col={('PLSR','fixed'):'PLSR',('FPCR','fixed'):'FPCR_fixK',('FPCR','cv'):'FPCR_cvK',
     ('PFLR','fixed'):'PFLR_fixL',('PFLR','cv'):'PFLR_cvL',('SVR','fixed'):'SVR_fix',('SVR','cv'):'SVR_cv'}
agg=df.groupby(['N','method','strategy']).R2.agg(['mean','std']).reset_index()
bad=0
print(f"\n{'N':>5} {'method':>5} {'strategy':>8} {'mean':>7} {'sd':>6} {'已发表':>7}")
for _,q in agg.iterrows():
    ref=float(pub.loc[pub.N==q.N,col[(q.method,q.strategy)]].iloc[0])
    ok=abs(round(q['mean'],3)-round(ref,3))<1e-9
    bad+=(not ok)
    print(f"{int(q.N):>5} {q.method:>5} {q.strategy:>8} {q['mean']:>7.3f} {q['std']:>6.3f} {ref:>7.3f} {'' if ok else '  <-- 不一致'}")
print(f"\n对账：{len(agg)} 个均值，{bad} 个与已发表数字不一致")
assert bad==0
