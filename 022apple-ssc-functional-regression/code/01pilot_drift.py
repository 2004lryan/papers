# -*- coding: utf-8 -*-
"""idea 阶段 pilot：量化苹果 NIR 糖度模型的【产地漂移】与【年份漂移】。只读 05data，不写回。"""
import os
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

D = os.environ.get('APPLE_SSC_DATA', '../../05data/001_apple_hyperspectral_multiyear')   # 数据不随代码分发，见 08github/data/DATA.md
LOC={1:'新疆',2:'山东',3:'陕西'}

def load(f):
    d=pd.read_excel(f,header=None)
    wl=d.iloc[0,:2048].astype(float).values
    X=d.iloc[2:,:2048].astype(float).values
    y=pd.to_numeric(d.iloc[2:,2048],errors='coerce').values
    z=pd.to_numeric(d.iloc[2:,2049],errors='coerce').values
    ok=np.isfinite(y)&np.isfinite(z)&(y>5)&(y<25)&np.isfinite(X).all(1)   # 剔除糖度离群(如56.1)
    return X[ok],y[ok],z[ok].astype(int),wl

def snv(X):
    m=X.mean(1,keepdims=True); s=X.std(1,keepdims=True); s[s==0]=1
    return (X-m)/s

X18,y18,z18,wl=load(f"{D}/01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx")
X19,y19,z19,_ =load(f"{D}/1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx")
X18,X19=snv(X18),snv(X19)
print(f"清洗后 2018 n={len(y18)} 2019 n={len(y19)}  波段 {len(wl)} ({wl[0]:.1f}-{wl[-1]:.1f}nm)")
for nm,y,z in [('2018',y18,z18),('2019',y19,z19)]:
    print(f"  {nm} 产地分布 " + "  ".join(f"{LOC[k]}:n={int((z==k).sum())},SSC={y[z==k].mean():.2f}±{y[z==k].std():.2f}" for k in sorted(set(z))))

NC=12
def fit_eval(Xtr,ytr,Xte,yte):
    m=PLSRegression(n_components=NC).fit(Xtr,ytr)
    p=m.predict(Xte).ravel()
    return r2_score(yte,p), np.sqrt(mean_squared_error(yte,p)), p.mean()-yte.mean()

def cv_within(X,y,k=5):
    r2s,rmses=[],[]
    for tr,te in KFold(k,shuffle=True,random_state=0).split(X):
        r,rm,_=fit_eval(X[tr],y[tr],X[te],y[te]); r2s.append(r); rmses.append(rm)
    return np.mean(r2s),np.mean(rmses)

print("\n================ 基线：同年同产地 5 折 CV（无漂移）================")
base={}
for nm,X,y,z in [('2018',X18,y18,z18),('2019',X19,y19,z19)]:
    for k in sorted(set(z)):
        m=z==k; r2,rm=cv_within(X[m],y[m]); base[(nm,k)]=(r2,rm)
        print(f"  {nm} {LOC[k]}: R2={r2:6.3f}  RMSE={rm:.3f}")

print("\n================ A. 产地漂移（同年、同仪器，跨产地）================")
A=[]
for nm,X,y,z in [('2018',X18,y18,z18),('2019',X19,y19,z19)]:
    for a in sorted(set(z)):
        for b in sorted(set(z)):
            if a==b: continue
            r2,rm,bias=fit_eval(X[z==a],y[z==a],X[z==b],y[z==b]); A.append((r2,rm))
            print(f"  {nm} {LOC[a]}→{LOC[b]}: R2={r2:7.3f}  RMSE={rm:6.3f}  bias={bias:+.2f}")

print("\n================ B. 年份漂移（同产地、同仪器，2018→2019）================")
B=[]
for k in sorted(set(z18)&set(z19)):
    r2,rm,bias=fit_eval(X18[z18==k],y18[z18==k],X19[z19==k],y19[z19==k]); B.append((r2,rm))
    print(f"  {LOC[k]} 2018→2019: R2={r2:7.3f}  RMSE={rm:6.3f}  bias={bias:+.2f}")

print("\n================ C. 双漂移（跨产地 + 跨年份）================")
C=[]
for a in sorted(set(z18)):
    for b in sorted(set(z19)):
        if a==b: continue
        r2,rm,bias=fit_eval(X18[z18==a],y18[z18==a],X19[z19==b],y19[z19==b]); C.append((r2,rm))
        print(f"  {LOC[a]}2018→{LOC[b]}2019: R2={r2:7.3f}  RMSE={rm:6.3f}  bias={bias:+.2f}")

print("\n================ 汇总（RMSE 越低越好）================")
b_=np.mean([v[1] for v in base.values()])
print(f"  基线(同年同产地CV)   RMSE={b_:.3f}   R2={np.mean([v[0] for v in base.values()]):.3f}")
for nm,S in [('A 仅产地漂移',A),('B 仅年份漂移',B),('C 双漂移',C)]:
    rm=np.mean([v[1] for v in S]); r2=np.mean([v[0] for v in S])
    print(f"  {nm:<14} RMSE={rm:.3f}  ({rm/b_:.2f}× 基线)   R2={r2:7.3f}")
