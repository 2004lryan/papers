# -*- coding: utf-8 -*-
"""关键验证：1D-CNN 在 2018+2019(1300 样本) 上输给 PLSR，是样本量问题还是方法问题？
用 2025 数据（3228 条光谱，约 2.5 倍样本）单独建模复测。只读 05data。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import torch, torch.nn as nn

D = os.environ.get('APPLE_SSC_DATA', '../../05data/001_apple_hyperspectral_multiyear')   # 数据不随代码分发，见 08github/data/DATA.md
spec=pd.read_csv(f"{D}/新疆-山东-甘肃苹果光谱数据2025.csv")
ssc =pd.read_csv(f"{D}/新疆-山东-甘肃苹果糖度数据2025.csv",encoding='gbk')
wcol=[c for c in spec.columns if c.endswith('nm')]
key=spec['实际苹果编号'].astype(str).str.split('-',n=1,expand=True)
spec['code'],spec['face']=key[0],key[1]
long=ssc.melt(id_vars='苹果编码',var_name='face',value_name='SSC').rename(columns={'苹果编码':'code'})
df=spec.merge(long,on=['code','face'],how='inner')
df=df[(df.SSC>5)&(df.SSC<25)]
X=df[wcol].values.astype(float); y=df['SSC'].values.astype(float)
prov=df['code'].str.extract(r'^([A-Za-z]+)')[0].values
ok=np.isfinite(X).all(1)&np.isfinite(y); X,y,prov=X[ok],y[ok],prov[ok]
print(f"2025 配对成功 {len(y)} 条（光谱 {len(spec)} × 糖度 {len(ssc)}×5 面）")
print(f"波段 {X.shape[1]}（{wcol[0]}–{wcol[-1]}）  SSC {y.mean():.2f}±{y.std():.2f} °Brix")
print("产地:", dict(pd.Series(prov).value_counts()))

def snv(A):
    m=A.mean(1,keepdims=True); s=A.std(1,keepdims=True); s[s==0]=1; return (A-m)/s
prep=lambda A: savgol_filter(snv(A),15,2,deriv=1,axis=1)   # 上一轮的最优预处理 SNV+SG1D

class CNN1D(nn.Module):
    def __init__(s):
        super().__init__()
        s.f=nn.Sequential(nn.Conv1d(1,16,9,stride=2,padding=4),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(2),
                          nn.Conv1d(16,32,7,stride=2,padding=3),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),
                          nn.Conv1d(32,64,5,stride=2,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.AdaptiveAvgPool1d(8))
        s.h=nn.Sequential(nn.Flatten(),nn.Dropout(0.3),nn.Linear(64*8,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,x): return s.h(s.f(x))

def run_cnn(Xtr,ytr,Xte,seed=0,ep=120):
    torch.manual_seed(seed)
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-8; ym,ys=ytr.mean(),ytr.std()
    A=torch.tensor(((Xtr-mu)/sd)[:,None,:],dtype=torch.float32)
    B=torch.tensor(((Xte-mu)/sd)[:,None,:],dtype=torch.float32)
    Y=torch.tensor(((ytr-ym)/ys)[:,None],dtype=torch.float32)
    m=CNN1D(); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep); lf=nn.MSELoss(); n=len(A)
    for _e in range(ep):
        m.train(); idx=torch.randperm(n)
        for i in range(0,n,64):
            b=idx[i:i+64]; opt.zero_grad(); lf(m(A[b]),Y[b]).backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad(): return m(B).numpy().ravel()*ys+ym

Xp=prep(X)
print(f"\n{'样本量':<8}{'方法':<10}{'R²':>8}{'RMSEP':>9}{'RPD':>7}")
print("-"*44)
for frac,tag in [(0.25,'~800'),(0.5,'~1600'),(1.0,'全部')]:
    rng=np.random.RandomState(0); sub=rng.choice(len(y),int(len(y)*frac),replace=False)
    Xs,ys_=Xp[sub],y[sub]
    for name in ['PLSR','SVR','1D-CNN']:
        R=[]
        for sd_ in [0,1]:
            Xtr,Xte,ytr,yte=train_test_split(Xs,ys_,test_size=0.3,random_state=sd_)
            if name=='PLSR':
                bn,bs=8,1e9
                for nc in range(2,26,2):
                    e=[mean_squared_error(ytr[va],PLSRegression(n_components=nc).fit(Xtr[tr],ytr[tr]).predict(Xtr[va]).ravel())
                       for tr,va in KFold(5,shuffle=True,random_state=0).split(Xtr)]
                    if np.mean(e)<bs: bs,bn=np.mean(e),nc
                yp=PLSRegression(n_components=bn).fit(Xtr,ytr).predict(Xte).ravel()
            elif name=='SVR':
                mu,sv=Xtr.mean(0),Xtr.std(0)+1e-8
                yp=SVR(C=100,gamma='scale').fit((Xtr-mu)/sv,ytr).predict((Xte-mu)/sv)
            else:
                yp=run_cnn(Xtr,ytr,Xte,seed=sd_)
            R.append([r2_score(yte,yp),np.sqrt(mean_squared_error(yte,yp)),np.std(yte)/np.sqrt(mean_squared_error(yte,yp))])
        m_=np.mean(R,0); print(f"{tag:<8}{name:<10}{m_[0]:>8.3f}{m_[1]:>9.3f}{m_[2]:>7.2f}")
    print("-"*44)
