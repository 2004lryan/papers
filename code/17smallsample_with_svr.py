# -*- coding: utf-8 -*-
"""把 SVR 补进小样本对比。

为什么要补：统一协议下 SVR 在全样本上达到 R²=0.817，是六个方法里最高的；
若小样本一节只拿 PLSR 作离散对照，「函数型方法在小样本占优」这一结论就没有和
最强的离散基线较量过。本脚本沿用 10smallsample_hyperparam.py 的全部设定
（B样条 L=40、N∈{20,…,400}、每个 N 重抽 20 次、测试集取后 400 条），
只增加 SVR 的两种超参选择策略：
  SVR 固定    C=100，gamma='scale'（与 04baseline_models.py 一致）
  SVR 交叉验证 建模集内 4 折，在 12 组 (C, gamma) 候选里选，与 FPCR/PFLR 的 CV 同一口径
只读 05data。

注：本脚本的 SVR 做了逐波段标准化，与统一协议（直接用 SNV+SG-1D 光谱）不同。
表 4 采用的是口径对齐后的 19svr_cv_aligned.py 的结果；两种口径的对照见 18svr_align_and_sensitivity.py。
"""
from typing import Any
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

with open('02code/08fda_smallsample.py') as _src: exec(_src.read().split('# ---------- 实验一')[0])   # 复用数据与模型定义
Bm=bspline_basis(t,40); J,R=gram_and_pen(Bm,t); C=to_coef(Xs,Bm,R)

SVR_GRID=[(c,g) for c in [1,10,100,1000] for g in ['scale',1e-3,1e-4]]
assert len(SVR_GRID)==12

def svr_fit(Xtr,ytr,Xte,cfg):
    mu,sd=Xtr.mean(0),Xtr.std(0); sd[sd==0]=1
    return SVR(C=cfg[0],gamma=cfg[1],epsilon=0.1).fit((Xtr-mu)/sd,ytr).predict((Xte-mu)/sd)

def svr_cv(Xtr,ytr,Xte):
    """建模集内 4 折选 (C, gamma)，与 cv_pick 对 FPCR/PFLR 的做法同一口径。"""
    nf=min(4,max(2,len(ytr)//2))
    best,bc=np.inf,SVR_GRID[0]
    for cfg in SVR_GRID:
        e=[]
        for tr,va in KFold(nf,shuffle=True,random_state=0).split(Xtr):
            e.append(mean_squared_error(ytr[va],svr_fit(Xtr[tr],ytr[tr],Xtr[va],cfg)))
        if np.mean(e)<best: best,bc=np.mean(e),cfg
    return svr_fit(Xtr,ytr,Xte,bc)

print("\n【小样本：补入 SVR】B样条 L=40，每个 N 重抽 20 次取平均，R²")
print(f"{'N':>5} | {'PLSR':>7}{'SVR固定':>9}{'SVR-CV':>9} | {'FPCR固定K':>10}{'PFLR固定λ':>10}")
rows=[]
for N in [20,30,50,80,120,200,400]:
    r: dict[str, Any] = {'N':N,'PLSR':[],'SVR_fix':[],'SVR_cv':[],'FPCR_fixK':[],'PFLR_fixL':[]}
    for sd in range(20):
        rng=np.random.RandomState(sd); idx=rng.permutation(len(y)); itr,ite=idx[:N],idx[N:N+400]
        r['PLSR'].append(r2_score(y[ite],plsr(Xs[itr],y[itr],Xs[ite])))
        r['SVR_fix'].append(r2_score(y[ite],svr_fit(Xs[itr],y[itr],Xs[ite],(100,'scale'))))
        r['SVR_cv'].append(r2_score(y[ite],svr_cv(Xs[itr],y[itr],Xs[ite])))
        r['FPCR_fixK'].append(r2_score(y[ite],fpcr(C[itr],y[itr],C[ite],J,min(10,max(2,N-2)))))
        r['PFLR_fixL'].append(r2_score(y[ite],pflr(C[itr],y[itr],C[ite],J,R,1e-3)))
    m={k:(np.mean(v) if k!='N' else v) for k,v in r.items()}
    rows.append(m)
    print(f"{N:>5} | {m['PLSR']:>7.3f}{m['SVR_fix']:>9.3f}{m['SVR_cv']:>9.3f} | "
          f"{m['FPCR_fixK']:>10.3f}{m['PFLR_fixL']:>10.3f}")
df=pd.DataFrame(rows); df.to_csv('04outputs/17smallsample_with_svr.csv',index=False,encoding='utf-8-sig')
print("\n函数型方法（固定超参）相对最强离散基线的差值，正值说明函数型占优：")
print(f"{'N':>5}{'FPCR−max(PLSR,SVR)':>22}{'PFLR−max(PLSR,SVR)':>22}")
for _,q in df.iterrows():
    best=max(q.PLSR,q.SVR_fix,q.SVR_cv)
    print(f"{int(q.N):>5}{q.FPCR_fixK-best:>22.3f}{q.PFLR_fixL-best:>22.3f}")
