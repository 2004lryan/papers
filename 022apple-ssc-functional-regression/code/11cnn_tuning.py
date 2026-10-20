# -*- coding: utf-8 -*-
"""给 1D-CNN 一次公平的超参搜索（此前 CNN 未调参而 PLSR 有内层 CV，对比不公平）。
在最优预处理 SNV+SG1D 上搜索 8 组配置，内层验证集选优，外层测试集报告。只读 05data。"""
from typing import Any
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import torch, torch.nn as nn
D = os.environ.get('APPLE_SSC_DATA', '../../05data/001_apple_hyperspectral_multiyear')   # 数据不随代码分发，见 08github/data/DATA.md
def load(f):
    d=pd.read_excel(f,header=None)
    X=d.iloc[2:,:2048].astype(float).values
    y=pd.to_numeric(d.iloc[2:,2048],errors='coerce').values
    z=pd.to_numeric(d.iloc[2:,2049],errors='coerce').values
    ok=np.isfinite(y)&np.isfinite(z)&(y>5)&(y<25)&np.isfinite(X).all(1)
    return X[ok],y[ok]
X1,y1=load(f"{D}/01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx")
X2,y2=load(f"{D}/1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx")
X=np.vstack([X1,X2]); y=np.r_[y1,y2]
def snv(M):
    m=M.mean(1,keepdims=True); s=M.std(1,keepdims=True); s[s==0]=1; return (M-m)/s
Xp=savgol_filter(snv(X),15,2,deriv=1,axis=1)

class CNN(nn.Module):
    def __init__(s,w=16,dp=0.3,nl=3):
        super().__init__()
        L=[]; c=1
        for i in range(nl):
            o=w*(2**i)
            L+=[nn.Conv1d(c,o,9 if i==0 else 5,stride=2,padding=4 if i==0 else 2),
                nn.BatchNorm1d(o),nn.ReLU(),nn.MaxPool1d(2)]
            c=o
        L+=[nn.AdaptiveAvgPool1d(8)]
        s.f=nn.Sequential(*L)
        s.h=nn.Sequential(nn.Flatten(),nn.Dropout(dp),nn.Linear(c*8,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,x): return s.h(s.f(x))

def fit(Xtr,ytr,Xte,cfg,seed,ep):
    torch.manual_seed(seed)
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-8; ym,ys=ytr.mean(),ytr.std()
    A=torch.tensor(((Xtr-mu)/sd)[:,None,:],dtype=torch.float32)
    B=torch.tensor(((Xte-mu)/sd)[:,None,:],dtype=torch.float32)
    Y=torch.tensor(((ytr-ym)/ys)[:,None],dtype=torch.float32)
    m=CNN(cfg['w'],cfg['dp'],cfg['nl'])
    opt=torch.optim.Adam(m.parameters(),cfg['lr'],weight_decay=cfg['wd'])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep); lf=nn.MSELoss(); n=len(A)
    for _e in range(ep):
        m.train(); idx=torch.randperm(n)
        for i in range(0,n,cfg['bs']):
            b=idx[i:i+cfg['bs']]; opt.zero_grad(); lf(m(A[b]),Y[b]).backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad(): return m(B).numpy().ravel()*ys+ym

GRID: list[Any] = []
for w,nl in [(16,3),(32,3),(16,4)]:
    for lr in [1e-3,3e-4]:
        for dp in [0.2,0.4]:
            GRID.append(dict(w=w,nl=nl,lr=lr,dp=dp,wd=1e-4,bs=32,ep=200))
print(f"候选配置 {len(GRID)} 组；训练/验证/测试 = 0.56/0.14/0.30")
itr,ite=train_test_split(np.arange(len(y)),test_size=0.3,random_state=0)
i2,iva=train_test_split(itr,test_size=0.2,random_state=0)
best = -9.0
bcfg: Any = None
for k,cfg in enumerate(GRID):
    p=fit(Xp[i2],y[i2],Xp[iva],cfg,0,cfg['ep'])
    r=r2_score(y[iva],p)
    print(f"  [{k+1:>2}/{len(GRID)}] w={cfg['w']} nl={cfg['nl']} lr={cfg['lr']:g} dp={cfg['dp']}  验证 R²={r:.3f}")
    if r>best: best,bcfg=r,cfg
print(f"\n选中：{bcfg}  验证 R²={best:.3f}")
R=[]
for sd in [0,1,2]:
    a,b=train_test_split(np.arange(len(y)),test_size=0.3,random_state=sd)
    p=fit(Xp[a],y[a],Xp[b],bcfg,sd,bcfg['ep'])
    R.append([r2_score(y[b],p),np.sqrt(mean_squared_error(y[b],p))])
m_=np.mean(R,0)
print(f"调参后 1D-CNN 测试集：R²={m_[0]:.3f}  RMSEP={m_[1]:.3f}  RPD={np.std(y)/m_[1]:.2f}")
print("（对照：未调参 CNN R²=0.594；PLSR 0.740/0.847；PFLR 0.781/0.780）")
pd.DataFrame([{'model':'1D-CNN(tuned)','R2':m_[0],'RMSEP':m_[1],**bcfg}]).to_csv(
    '04outputs/11cnn_tuned.csv',index=False,encoding='utf-8-sig')
