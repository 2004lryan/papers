# -*- coding: utf-8 -*-
"""两件事：
① 口径对齐——统一协议里 SVR 直接用 SNV+SG-1D 光谱（与 PLSR/FPCR/PFLR 相同），
   而 17smallsample_with_svr.py 里的 SVR 额外做了逐波段标准化。两处不一致。
   本脚本按统一协议的口径（不额外标准化）重跑小样本 SVR，并同时给出标准化版本，
   以便判断该差异是否影响结论。
② 超参数敏感性实测——3.4 节称「函数型模型的超参数选偏一档即显著改变模型复杂度，
   而 SVR 的 C 与 gamma 影响相对平缓」。这一说法此前无证据。
   本脚本在 N=80 下把每个方法 12 组候选逐个跑出来，报告其 R² 的极差与标准差。
只读 05data。
"""
from typing import Any
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.svm import SVR
from sklearn.metrics import r2_score

with open('02code/08fda_smallsample.py') as _src: exec(_src.read().split('# ---------- 实验一')[0])
Bm=bspline_basis(t,40); Jb,Rb=gram_and_pen(Bm,t); Cb=to_coef(Xs,Bm,Rb)

def svr_raw(Xtr,ytr,Xte,cfg):          # 与 16unified_comparison.py 同一口径：不额外标准化
    return SVR(C=cfg[0],gamma=cfg[1],epsilon=0.1).fit(Xtr,ytr).predict(Xte)
def svr_std(Xtr,ytr,Xte,cfg):          # 逐波段标准化版本
    mu,sd=Xtr.mean(0),Xtr.std(0); sd[sd==0]=1
    return SVR(C=cfg[0],gamma=cfg[1],epsilon=0.1).fit((Xtr-mu)/sd,ytr).predict((Xte-mu)/sd)

print("\n【① 口径对齐】SVR 固定超参（C=100, gamma='scale'），每个 N 重抽 20 次取平均")
print(f"{'N':>5}{'SVR 不标准化':>14}{'SVR 标准化':>12}{'FPCR 固定K':>12}{'PFLR 固定λ':>12}")
rows=[]
for N in [20,30,50,80,120,200,400]:
    a=[];b=[];c=[];d=[]
    for sd_ in range(20):
        rng=np.random.RandomState(sd_); idx=rng.permutation(len(y)); itr,ite=idx[:N],idx[N:N+400]
        a.append(r2_score(y[ite],svr_raw(Xs[itr],y[itr],Xs[ite],(100,'scale'))))
        b.append(r2_score(y[ite],svr_std(Xs[itr],y[itr],Xs[ite],(100,'scale'))))
        c.append(r2_score(y[ite],fpcr(Cb[itr],y[itr],Cb[ite],Jb,min(10,max(2,N-2)))))
        d.append(r2_score(y[ite],pflr(Cb[itr],y[itr],Cb[ite],Jb,Rb,1e-3)))
    rows.append(dict(N=N,SVR_raw=np.mean(a),SVR_std=np.mean(b),FPCR=np.mean(c),PFLR=np.mean(d)))
    print(f"{N:>5}{np.mean(a):>14.3f}{np.mean(b):>12.3f}{np.mean(c):>12.3f}{np.mean(d):>12.3f}")
pd.DataFrame(rows).to_csv('04outputs/18svr_aligned.csv',index=False,encoding='utf-8-sig')

print("\n【② 超参数敏感性】N=80，每个方法把 12 组候选逐个跑出（20 次重抽平均的 R²）")
GR: dict[str, list[Any]] = {'SVR':[(C,g) for C in [1,10,100,1000] for g in ['scale',1e-3,1e-4]],
    'FPCR':[(L,K) for L in [40] for K in [2,3,5,8,10,12,15,20,25,30,35,40]],
    'PFLR':[(40,lam) for lam in [1e-8,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]]}
BAS={40:(Bm,Jb,Rb)}; CO={40:Cb}
def run(name,cfg,itr,ite):
    if name=='SVR': return svr_raw(Xs[itr],y[itr],Xs[ite],cfg)
    L,h=cfg
    if name=='FPCR': return fpcr(CO[L][itr],y[itr],CO[L][ite],BAS[L][1],min(int(h),max(2,80-2)))
    return pflr(CO[L][itr],y[itr],CO[L][ite],BAS[L][1],BAS[L][2],h)
N=80; summ: list[Any] = []
for name in ['SVR','FPCR','PFLR']:
    per=[]
    for cfg in GR[name]:
        e=[]
        for sd_ in range(20):
            rng=np.random.RandomState(sd_); idx=rng.permutation(len(y)); itr,ite=idx[:N],idx[N:N+400]
            e.append(r2_score(y[ite],run(name,cfg,itr,ite)))
        per.append(np.mean(e))
    per_a = np.array(per)
    print(f"\n  {name}（12 组候选的 R²）：")
    for cfg,v in zip(GR[name],per_a, strict=False): print(f"      {str(cfg):<22}{v:>8.3f}")
    print(f"      → 极差 {per_a.max()-per_a.min():.3f}   标准差 {per_a.std():.3f}   最优 {per_a.max():.3f}   最差 {per_a.min():.3f}")
    summ.append(dict(method=name,rng=per_a.max()-per_a.min(),sd=per_a.std(),best=per_a.max(),worst=per_a.min()))
print("\n  汇总：12 组候选之间的 R² 极差")
for q in summ: print(f"      {q['method']:<6} 极差 {q['rng']:.3f}  标准差 {q['sd']:.3f}  最优 {q['best']:.3f}  最差 {q['worst']:.3f}")
pd.DataFrame(summ).to_csv('04outputs/18hyperparam_sensitivity.csv',index=False,encoding='utf-8-sig')
