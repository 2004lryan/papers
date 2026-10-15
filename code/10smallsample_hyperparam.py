# -*- coding: utf-8 -*-
"""小样本下的超参选择困境：CV 选 K vs 固定 K，对 FPCR/PFLR 的影响。
对应申报书 B 关键技术 (a)「FCPCA 在样本量不足时本征基不稳」。只读 05data。"""
from typing import Any
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
with open('02code/08fda_smallsample.py') as _src: exec(_src.read().split('# ---------- 实验一')[0])   # 复用数据与模型定义
B,J,R=None,None,None
Bm=bspline_basis(t,40); J,R=gram_and_pen(Bm,t); C=to_coef(Xs,Bm,R)
print("\n【小样本 × 超参选择策略】B样条 L=40，20 次重抽平均，R²")
print(f"{'N':>5} | {'PLSR':>7} | {'FPCR固定K=10':>12}{'FPCR-CV选K':>12} | {'PFLR固定λ':>11}{'PFLR-CV选λ':>12}")
rows=[]
for N in [20,30,50,80,120,200,400]:
    r: dict[str, Any] = {'N':N,'PLSR':[],'FPCR_fixK':[],'FPCR_cvK':[],'PFLR_fixL':[],'PFLR_cvL':[]}
    for sd in range(20):
        rng=np.random.RandomState(sd); idx=rng.permutation(len(y)); itr,ite=idx[:N],idx[N:N+400]
        r['PLSR'].append(r2_score(y[ite],plsr(Xs[itr],y[itr],Xs[ite])))
        Kf=min(10,max(2,N-2))
        r['FPCR_fixK'].append(r2_score(y[ite],fpcr(C[itr],y[itr],C[ite],J,Kf)))
        Kg=[k for k in [2,5,10,15,20,30] if k<=max(2,int(N*0.75)-2)] or [2]
        Kc=cv_pick(C[itr],y[itr],J,R,'fpcr',Kg)
        r['FPCR_cvK'].append(r2_score(y[ite],fpcr(C[itr],y[itr],C[ite],J,Kc)))
        r['PFLR_fixL'].append(r2_score(y[ite],pflr(C[itr],y[itr],C[ite],J,R,1e-3)))
        Lc=cv_pick(C[itr],y[itr],J,R,'pflr',np.logspace(-8,2,6))
        r['PFLR_cvL'].append(r2_score(y[ite],pflr(C[itr],y[itr],C[ite],J,R,Lc)))
    m={k:(np.mean(v) if k!='N' else v) for k,v in r.items()}
    rows.append(m)
    print(f"{N:>5} | {m['PLSR']:>7.3f} | {m['FPCR_fixK']:>12.3f}{m['FPCR_cvK']:>12.3f} | {m['PFLR_fixL']:>11.3f}{m['PFLR_cvL']:>12.3f}")
df=pd.DataFrame(rows); df.to_csv('04outputs/10smallsample_hyperparam.csv',index=False,encoding='utf-8-sig')
print("\n差值（固定 − CV选），正值说明「固定超参优于交叉验证选」：")
print(f"{'N':>5}{'FPCR':>9}{'PFLR':>9}")
for _,q in df.iterrows():
    print(f"{int(q.N):>5}{q.FPCR_fixK-q.FPCR_cvK:>9.3f}{q.PFLR_fixL-q.PFLR_cvL:>9.3f}")
