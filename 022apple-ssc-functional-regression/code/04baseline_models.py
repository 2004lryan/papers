# -*- coding: utf-8 -*-
"""贴合版方案验证：标准建模场景下，各类方法预测苹果糖度的精度对比。
场景 = 2018+2019 合并、随机划分（这是「预测建模」论文的常规设定）。
比较：PLSR / SVR / RF / 1D-CNN，各配 4 种预处理。只读 05data。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import torch, torch.nn as nn
import data_clean   # 协议 B10.10 数据清洗四条规则的唯一实现（02code/data_clean.py）

D = os.environ.get('APPLE_SSC_DATA', '../../05data/001_apple_hyperspectral_multiyear')   # 数据不随代码分发，见 08github/data/DATA.md
def load(f,yr):
    d=pd.read_excel(f,header=None)
    X=d.iloc[2:,:2048].astype(float).values
    y=pd.to_numeric(d.iloc[2:,2048],errors='coerce').values
    z=pd.to_numeric(d.iloc[2:,2049],errors='coerce').values
    ok=np.isfinite(y)&np.isfinite(z)&(y>5)&(y<25)&np.isfinite(X).all(1)
    ok&=data_clean.keep_mask(X,y,z)   # 协议 B10.10 数据清洗：四条规则在该年份原始记录上判定，剔除其并集
    return X[ok],y[ok],z[ok].astype(int),np.full(ok.sum(),yr)
X1,y1,z1,t1=load(f"{D}/01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx",2018)
X2,y2,z2,t2=load(f"{D}/1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx",2019)
X=np.vstack([X1,X2]); y=np.r_[y1,y2]; z=np.r_[z1,z2]; t=np.r_[t1,t2]
print(f"合并样本 {len(y)}（2018:{len(y1)} + 2019:{len(y2)}），波段 2048，SSC {y.mean():.2f}±{y.std():.2f} °Brix\n")

def snv(A):
    m=A.mean(1,keepdims=True); s=A.std(1,keepdims=True); s[s==0]=1; return (A-m)/s
def msc(A,ref=None):
    ref=A.mean(0) if ref is None else ref; out=np.empty_like(A)
    for i,r in enumerate(A):
        k,b=np.polyfit(ref,r,1); out[i]=(r-b)/k
    return out
PREP={'原始':lambda A:A,'SNV':snv,'MSC':msc,
      'SG1D':lambda A:savgol_filter(A,15,2,deriv=1,axis=1),
      'SNV+SG1D':lambda A:savgol_filter(snv(A),15,2,deriv=1,axis=1)}

def metrics(yt,yp):
    r2=r2_score(yt,yp); rm=np.sqrt(mean_squared_error(yt,yp))
    return r2,rm,np.std(yt)/rm   # R2, RMSEP, RPD

class CNN1D(nn.Module):
    def __init__(s,L):
        super().__init__()
        s.f=nn.Sequential(nn.Conv1d(1,16,9,stride=2,padding=4),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(2),
                          nn.Conv1d(16,32,7,stride=2,padding=3),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),
                          nn.Conv1d(32,64,5,stride=2,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.AdaptiveAvgPool1d(8))
        s.h=nn.Sequential(nn.Flatten(),nn.Dropout(0.3),nn.Linear(64*8,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,x): return s.h(s.f(x))

def run_cnn(Xtr,ytr,Xte,seed=0,ep=80):
    torch.manual_seed(seed); dev='cpu'
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-8; ym,ys=ytr.mean(),ytr.std()
    A=torch.tensor(((Xtr-mu)/sd)[:,None,:],dtype=torch.float32)
    B=torch.tensor(((Xte-mu)/sd)[:,None,:],dtype=torch.float32)
    Y=torch.tensor(((ytr-ym)/ys)[:,None],dtype=torch.float32)
    m=CNN1D(Xtr.shape[1]).to(dev); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep); lf=nn.MSELoss()
    n=len(A); bs=64
    for _e in range(ep):
        m.train(); idx=torch.randperm(n)
        for i in range(0,n,bs):
            b=idx[i:i+bs]; opt.zero_grad(); l=lf(m(A[b]),Y[b]); l.backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad(): return (m(B).numpy().ravel()*ys+ym)

SEEDS=[0,1]
print(f"{'预处理':<10}{'方法':<10}{'R²':>8}{'RMSEP':>9}{'RPD':>7}")
print("-"*46)
best={}
for pname,pf in PREP.items():
    Xp=pf(X)
    for mname in ['PLSR','SVR','RF','1D-CNN']:
        R=[]
        for sd_ in SEEDS:
            Xtr,Xte,ytr,yte=train_test_split(Xp,y,test_size=0.3,random_state=sd_,stratify=z*10000+t)
            if mname=='PLSR':
                bn,bs_=8,1e9
                for nc in range(2,26,2):
                    e=[]
                    for tr,va in KFold(5,shuffle=True,random_state=0).split(Xtr):
                        p=PLSRegression(n_components=nc).fit(Xtr[tr],ytr[tr]).predict(Xtr[va]).ravel()
                        e.append(mean_squared_error(ytr[va],p))
                    if np.mean(e)<bs_: bs_,bn=np.mean(e),nc
                yp=PLSRegression(n_components=bn).fit(Xtr,ytr).predict(Xte).ravel()
            elif mname=='SVR':
                mu,sdv=Xtr.mean(0),Xtr.std(0)+1e-8
                yp=SVR(C=100,gamma='scale',epsilon=0.1).fit((Xtr-mu)/sdv,ytr).predict((Xte-mu)/sdv)
            elif mname=='RF':
                yp=RandomForestRegressor(300,n_jobs=-1,random_state=sd_).fit(Xtr,ytr).predict(Xte)
            else:
                yp=run_cnn(Xtr,ytr,Xte,seed=sd_)
            R.append(metrics(yte,yp))
        R_a = np.array(R); mu_=R_a.mean(0)
        best[(pname,mname)]=mu_
        print(f"{pname:<10}{mname:<10}{mu_[0]:>8.3f}{mu_[1]:>9.3f}{mu_[2]:>7.2f}")
    print("-"*46)
k=max(best,key=lambda k:best[k][0])
print(f"\n最优组合：{k[0]} + {k[1]}   R²={best[k][0]:.3f}  RMSEP={best[k][1]:.3f}  RPD={best[k][2]:.2f}")
pd.DataFrame([{'预处理':a,'方法':b,'R2':v[0],'RMSEP':v[1],'RPD':v[2]} for (a,b),v in best.items()]).to_csv(
    '04outputs/04baseline_models.csv',index=False,encoding='utf-8-sig')
