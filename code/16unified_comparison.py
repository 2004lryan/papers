# -*- coding: utf-8 -*-
"""统一协议下的六方法对比：同一预处理、同一批划分、同一选参方式、同样 12 组候选。

为什么要有这个脚本：表 3 原先把 04baseline（2 次划分）、08fda（3 次划分、内层 4 折 CV）、
11cnn（1 次验证集选参、3 次测试划分）三处的数字并排，重复次数与选参强度都不同，
横向说「谁优于谁」不成立。本脚本把六个方法放进同一套协议里重跑。

协议（六个方法完全一致）：
  预处理  SNV + Savitzky-Golay 一阶导（窗宽 15，二次多项式）
  外层    5 次重复，seed 0..4，7:3 随机划分——同一 seed 下六个方法拿到同一划分
  选参    从建模集再切 20% 作内层验证集，在 12 组候选里按验证 RMSE 选优，
          再用整个建模集重新拟合选中的配置，最后在预测集上评价
  报告    5 次重复的均值 ± 标准差

只读 05data。
"""
from typing import Any
import os, time; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import torch, torch.nn as nn
import data_clean   # 协议 B10.10 数据清洗四条规则的唯一实现（02code/data_clean.py）

D = os.environ.get('APPLE_SSC_DATA', '../../05data/001_apple_hyperspectral_multiyear')   # 数据不随代码分发，见 08github/data/DATA.md
def load(f):
    d=pd.read_excel(f,header=None)
    wl=d.iloc[0,:2048].astype(float).values
    X=d.iloc[2:,:2048].astype(float).values
    y=pd.to_numeric(d.iloc[2:,2048],errors='coerce').values
    z=pd.to_numeric(d.iloc[2:,2049],errors='coerce').values
    ok=np.isfinite(y)&np.isfinite(z)&(y>5)&(y<25)&np.isfinite(X).all(1)
    ok&=data_clean.keep_mask(X,y,z)   # 协议 B10.10 数据清洗：四条规则在该年份原始记录上判定，剔除其并集
    return X[ok],y[ok],z[ok].astype(int),wl
X1,y1,z1,wl=load(f"{D}/01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx")
X2,y2,z2,_ =load(f"{D}/1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx")
X=np.vstack([X1,X2]); y=np.r_[y1,y2]

def snv(M):
    m=M.mean(1,keepdims=True); s=M.std(1,keepdims=True); s[s==0]=1; return (M-m)/s
Xp=savgol_filter(snv(X),15,2,deriv=1,axis=1)
t=(wl-wl.min())/(wl.max()-wl.min())
SDY=float(np.std(y))
print(f"n={len(y)}  p={X.shape[1]}  波长 {wl[0]:.1f}-{wl[-1]:.1f}nm  SSC={y.mean():.2f}±{SDY:.2f}")

# ================= 函数型基系统（与 08fda_smallsample.py 同一实现）=================
def bspline_basis(t,L,k=3):
    ni=L-k-1
    inner=np.linspace(0,1,ni+2)[1:-1] if ni>0 else np.array([])
    kn=np.r_[np.zeros(k+1), inner, np.ones(k+1)]
    return np.asarray(BSpline.design_matrix(np.clip(t,0,1-1e-12),kn,k).todense())
def fourier_basis(t,L):
    B=[np.ones_like(t)]
    for k in range(1,L//2+1): B += [np.sin(2*np.pi*k*t), np.cos(2*np.pi*k*t)]
    return np.array(B[:L]).T
def gram_and_pen(B,t):
    dt=np.gradient(t)
    J=(B*dt[:,None]).T@B
    d2=np.gradient(np.gradient(B,t,axis=0),t,axis=0)
    R=(d2*dt[:,None]).T@d2
    return J, R+1e-8*np.eye(B.shape[1])
def to_coef(Xd,B,R,lam_s=1e-6):
    return np.linalg.solve(B.T@B+lam_s*R, B.T@Xd.T).T
def pflr(Ctr,ytr,Cte,J,R,lam):
    Z=Ctr@J; Zt=Cte@J; ym=ytr.mean(); ZtZ=Z.T@Z
    sc=np.trace(ZtZ)/max(np.trace(R),1e-30)
    b=np.linalg.solve(ZtZ+lam*sc*R+1e-12*np.eye(len(R)), Z.T@(ytr-ym))
    return Zt@b+ym
def fpcr(Ctr,ytr,Cte,J,K):
    M=np.linalg.cholesky(J+1e-10*np.eye(len(J)))
    A=Ctr@M; At=Cte@M; mu=A.mean(0); A0=A-mu
    P=np.linalg.svd(A0,full_matrices=False)[2][:K].T
    T1=np.c_[np.ones(len(A0)),A0@P]; Tt1=np.c_[np.ones(len(At)),(At-mu)@P]
    return Tt1@np.linalg.lstsq(T1,ytr,rcond=None)[0]

# 基系统缓存 + J/R 量级诊断（回应「J~1e-4 而 R~1e6」这一说法需有据）
BASIS={}
for L in [40,80]:
    B=fourier_basis(t,L); J,Rm=gram_and_pen(B,t)
    BASIS[L]=(B,J,Rm)
    print(f"  傅里叶基 L={L:>3}：tr(J)={np.trace(J):.3e}  tr(R)={np.trace(Rm):.3e}"
          f"  量级比 {np.trace(Rm)/np.trace(J):.2e}")
COEF={L:to_coef(Xp,*[BASIS[L][0],BASIS[L][2]]) for L in BASIS}
print()

# ================= 1D-CNN（与 11cnn_tuning.py 同一网络与训练流程）=================
class CNN(nn.Module):
    def __init__(s,w=16,dp=0.3,nl=3):
        super().__init__(); L=[]; c=1
        for i in range(nl):
            o=w*(2**i)
            L+=[nn.Conv1d(c,o,9 if i==0 else 5,stride=2,padding=4 if i==0 else 2),
                nn.BatchNorm1d(o),nn.ReLU(),nn.MaxPool1d(2)]; c=o
        L+=[nn.AdaptiveAvgPool1d(8)]; s.f=nn.Sequential(*L)
        s.h=nn.Sequential(nn.Flatten(),nn.Dropout(dp),nn.Linear(c*8,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,x): return s.h(s.f(x))
def cnn_fit(Xtr,ytr,Xte,cfg,seed):
    torch.manual_seed(seed)
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-8; ym,ys=ytr.mean(),ytr.std()
    A=torch.tensor(((Xtr-mu)/sd)[:,None,:],dtype=torch.float32)
    Bt=torch.tensor(((Xte-mu)/sd)[:,None,:],dtype=torch.float32)
    Y=torch.tensor(((ytr-ym)/ys)[:,None],dtype=torch.float32)
    m=CNN(cfg['w'],cfg['dp'],cfg['nl'])
    opt=torch.optim.Adam(m.parameters(),cfg['lr'],weight_decay=cfg['wd'])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['ep']); lf=nn.MSELoss(); n=len(A)
    for _e in range(cfg['ep']):
        m.train(); idx=torch.randperm(n)
        for i in range(0,n,cfg['bs']):
            b=idx[i:i+cfg['bs']]; opt.zero_grad(); lf(m(A[b]),Y[b]).backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad(): return m(Bt).numpy().ravel()*ys+ym

# ================= 六个方法：各 12 组候选，统一 predict(候选, 训练下标, 预测下标) =================
CNN_GRID=[dict(w=w,nl=nl,lr=lr,dp=dp,wd=1e-4,bs=32,ep=200)
          for w,nl in [(16,3),(32,3),(16,4)] for lr in [1e-3,3e-4] for dp in [0.2,0.4]]

def mk(name):
    g: list[Any]
    if name=='PLSR':
        g=list(range(2,26,2))
        def f(c,a,b,sd): return PLSRegression(n_components=min(c,len(a)-1)).fit(Xp[a],y[a]).predict(Xp[b]).ravel()
    elif name=='SVR':
        g=[(C,gm) for C in [1,10,100,1000] for gm in ['scale',1e-3,1e-4]]
        def f(c,a,b,sd): return SVR(C=c[0],gamma=c[1]).fit(Xp[a],y[a]).predict(Xp[b])
    elif name=='RF':
        g=[(n,mf) for n in [200,400,800] for mf in ['sqrt',0.1,0.3,1.0]]
        def f(c,a,b,sd): return RandomForestRegressor(n_estimators=c[0],max_features=c[1],
                                    random_state=sd,n_jobs=-1).fit(Xp[a],y[a]).predict(Xp[b])
    elif name=='1D-CNN':
        g=CNN_GRID
        def f(c,a,b,sd): return cnn_fit(Xp[a],y[a],Xp[b],c,sd)
    elif name=='FPCR':
        g=[(L,K) for L in [40,80] for K in [5,10,15,20,25,30]]
        def f(c,a,b,sd):
            L,K=c; return fpcr(COEF[L][a],y[a],COEF[L][b],BASIS[L][1],K)
    elif name=='PFLR':
        g=[(L,lam) for L in [40,80] for lam in [1e-8,1e-6,1e-4,1e-2,1e0,1e2]]
        def f(c,a,b,sd):
            L,lam=c; return pflr(COEF[L][a],y[a],COEF[L][b],BASIS[L][1],BASIS[L][2],lam)
    return g,f

METHODS=['PLSR','SVR','RF','1D-CNN','FPCR','PFLR']
# 协议说明放在这里而不是文件头：本行以上的部分被 9 个下游脚本 exec 复用，
# 而它们跑的是嵌套交叉验证，写在文件头会让它们的日志头声明一个并未执行的协议。
print("协议：SNV+SG1D 预处理；5 次 7:3 划分（seed 0-4，六法共用）；")
print("      建模集再切 20% 作内层验证集，12 组候选选优后用整个建模集重拟合\n")
for m in METHODS: assert len(mk(m)[0])==12, (m,len(mk(m)[0]))
print(f"六个方法各 {12} 组候选，候选数相同\n")

SEEDS=[0,1,2,3,4]
res: dict[str, list[Any]] = {m: [] for m in METHODS}
picked: dict[str, list[Any]] = {m: [] for m in METHODS}
for m in METHODS:
    grid,fit=mk(m); t0=time.time()
    for sd in SEEDS:
        itr,ite=train_test_split(np.arange(len(y)),test_size=0.3,random_state=sd)
        isub,iva=train_test_split(itr,test_size=0.2,random_state=sd)     # 内层验证集
        best,bc=np.inf,grid[0]
        for c in grid:
            e=np.sqrt(mean_squared_error(y[iva],fit(c,isub,iva,sd)))
            if e<best: best,bc=e,c
        p=fit(bc,itr,ite,sd)                                             # 用整个建模集重拟合
        r2=r2_score(y[ite],p); rm=float(np.sqrt(mean_squared_error(y[ite],p)))
        res[m].append((r2,rm,np.std(y[ite])/rm)); picked[m].append(str(bc))
        print(f"  {m:<8} seed={sd}  选中={str(bc):<34} R²={r2:.3f}  RMSEP={rm:.3f}")
    print(f"  {m:<8} 用时 {time.time()-t0:.0f}s\n")

print("="*76)
print("【统一协议下的六方法对比】5 次 7:3 划分，均值 ± 标准差")
print(f"{'方法':<10}{'R²':>18}{'RMSEP/°Brix':>20}{'RPD':>16}")
rows=[]
for m in METHODS:
    a=np.array(res[m])
    print(f"{m:<10}{a[:,0].mean():>10.3f}±{a[:,0].std():<7.3f}"
          f"{a[:,1].mean():>12.3f}±{a[:,1].std():<7.3f}{a[:,2].mean():>9.2f}±{a[:,2].std():<6.2f}")
    rows.append(dict(method=m,R2=a[:,0].mean(),R2_sd=a[:,0].std(),RMSEP=a[:,1].mean(),
                     RMSEP_sd=a[:,1].std(),RPD=a[:,2].mean(),RPD_sd=a[:,2].std(),
                     picked=';'.join(picked[m])))
print("\n各次划分选中的配置：")
for m in METHODS: print(f"  {m:<8} {picked[m]}")

# 配对比较：同一 seed 下 PFLR 与其他方法的 R² 差
print("\n同一划分下 PFLR 相对各方法的 R² 差（5 次配对，均值 ± 标准差）：")
pf=np.array(res['PFLR'])[:,0]
for m in METHODS:
    if m=='PFLR': continue
    d=pf-np.array(res[m])[:,0]
    print(f"  PFLR − {m:<8} {d.mean():+.3f} ± {d.std():.3f}   5 次中占优 {int((d>0).sum())}/5")
pd.DataFrame(rows).to_csv('04outputs/16unified_comparison.csv',index=False,encoding='utf-8-sig')
print("\n已写出 04outputs/16unified_comparison.csv")
