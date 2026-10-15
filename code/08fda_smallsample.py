# -*- coding: utf-8 -*-
"""按 008/009 申请书的 FDA 路线实现并验证：
  光谱 -> B样条/傅里叶基展开(粗糙惩罚) -> FPCA / 惩罚函数型线性回归(PFLR)
对照离散基线 PLSR。核心场景 = 小样本（009 关键技术 (a)：样本量不足时本征基不稳）。
只读 05data。"""
import os
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.interpolate import BSpline
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
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
X=np.vstack([X1,X2]); y=np.r_[y1,y2]; z=np.r_[z1,z2]
def snv(M):
    m=M.mean(1,keepdims=True); s=M.std(1,keepdims=True); s[s==0]=1; return (M-m)/s
Xs=savgol_filter(snv(X),15,2,deriv=1,axis=1)          # 与前几轮同一预处理，保证公平
t=(wl-wl.min())/(wl.max()-wl.min())                    # 波长归一到 [0,1]
print(f"n={len(y)}  p={X.shape[1]}  波长 {wl[0]:.1f}-{wl[-1]:.1f}nm  SSC={y.mean():.2f}±{y.std():.2f}")

# ---------- 基函数系统 ----------
def bspline_basis(t,L,k=3):
    """L 个 k 次 B 样条基。clamped 节点向量长度须为 L+k+1。"""
    ni=L-k-1                                            # 内部节点数
    if ni<0: raise ValueError(f"L={L} 对 k={k} 太小")
    inner=np.linspace(0,1,ni+2)[1:-1] if ni>0 else np.array([])
    kn=np.r_[np.zeros(k+1), inner, np.ones(k+1)]
    assert len(kn)==L+k+1, (len(kn),L+k+1)
    tt=np.clip(t,0,1-1e-12)
    return np.asarray(BSpline.design_matrix(tt,kn,k).todense())

def fourier_basis(t,L):
    B=[np.ones_like(t)]
    for k in range(1,L//2+1):
        B += [np.sin(2*np.pi*k*t), np.cos(2*np.pi*k*t)]
    return np.array(B[:L]).T

def gram_and_pen(B,t):
    dt=np.gradient(t)
    J=(B*dt[:,None]).T@B                                  # ∫φ_j φ_k
    d2=np.gradient(np.gradient(B,t,axis=0),t,axis=0)      # 二阶导
    R=(d2*dt[:,None]).T@d2                                # ∫φ_j'' φ_k''
    return J, R+1e-8*np.eye(B.shape[1])

def to_coef(Xd,B,R,lam_s=1e-6):
    G=B.T@B+lam_s*R
    return np.linalg.solve(G, B.T@Xd.T).T                 # n×L 基系数

# ---------- 三种函数型模型 ----------
def pflr(Ctr,ytr,Cte,J,R,lam):
    """惩罚函数型线性回归。lam 为【相对】平滑参数：实际惩罚 = lam*(tr(Z'Z)/tr(R))*R，
    使其与数据项同量级——否则 J~1e-4 而 R~1e6，任何绝对 lam 都会把 beta 压成 0。"""
    Z=Ctr@J; Zt=Cte@J; ym=ytr.mean()
    ZtZ=Z.T@Z
    sc=np.trace(ZtZ)/max(np.trace(R),1e-30)
    A=ZtZ+lam*sc*R+1e-12*np.eye(len(R))
    b=np.linalg.solve(A, Z.T@(ytr-ym))
    return Zt@b+ym
def fpcr(Ctr,ytr,Cte,J,K):
    M=np.linalg.cholesky(J+1e-10*np.eye(len(J)))          # J 的平方根，函数内积
    A=Ctr@M; At=Cte@M
    mu=A.mean(0); A0=A-mu
    U,S,Vt=np.linalg.svd(A0,full_matrices=False)
    P=Vt[:K].T
    T=A0@P; Tt=(At-mu)@P
    T1=np.c_[np.ones(len(T)),T]; Tt1=np.c_[np.ones(len(Tt)),Tt]
    coef=np.linalg.lstsq(T1,ytr,rcond=None)[0]
    return Tt1@coef

def cv_pick(Ctr,ytr,J,R,kind,grid):
    best,bp=1e18,grid[0]
    for g in grid:
        e=[]
        for tr,va in KFold(4,shuffle=True,random_state=0).split(Ctr):
            p = pflr(Ctr[tr],ytr[tr],Ctr[va],J,R,g) if kind=='pflr' else fpcr(Ctr[tr],ytr[tr],Ctr[va],J,g)
            e.append(mean_squared_error(ytr[va],p))
        if np.mean(e)<best: best,bp=np.mean(e),g
    return bp

def plsr(Xtr,ytr,Xte):
    bn,bs=2,1e18
    hi=max(2,min(25,int(len(ytr)*0.75)-1,Xtr.shape[1]))   # 4 折内层，训练份约 0.75n
    for nc in range(2,hi+1,2):
        e=[mean_squared_error(ytr[va],PLSRegression(n_components=nc).fit(Xtr[tr],ytr[tr]).predict(Xtr[va]).ravel())
           for tr,va in KFold(4,shuffle=True,random_state=0).split(Xtr)]
        if np.mean(e)<bs: bs,bn=np.mean(e),nc
    bn=min(bn,len(ytr)-1,Xtr.shape[1])
    return PLSRegression(n_components=bn).fit(Xtr,ytr).predict(Xte).ravel()

# ---------- 实验一：基函数族与基个数（008 的「基函数优选」）----------
print("\n【实验一】基函数族 × 基个数（全样本 7:3，3 次划分平均）")
print(f"{'基族':<8}{'L':>5}{'PFLR R²':>10}{'FPCR R²':>10}{'PFLR RMSEP':>12}")
cache={}
for fam,fn in [('B样条',bspline_basis),('傅里叶',fourier_basis)]:
    for L in [20,40,60,80]:
        B=fn(t,L); J,R=gram_and_pen(B,t); C=to_coef(Xs,B,R)
        cache[(fam,L)]=(B,J,R,C)
        r1,r2,rm=[],[],[]
        for sd in [0,1,2]:
            itr,ite=train_test_split(np.arange(len(y)),test_size=0.3,random_state=sd)
            lam=cv_pick(C[itr],y[itr],J,R,'pflr',np.logspace(-10,2,13))
            p1=pflr(C[itr],y[itr],C[ite],J,R,lam)
            K=cv_pick(C[itr],y[itr],J,R,'fpcr',[5,10,15,20,min(30,L-1)])
            p2=fpcr(C[itr],y[itr],C[ite],J,K)
            r1.append(r2_score(y[ite],p1)); r2.append(r2_score(y[ite],p2))
            rm.append(np.sqrt(mean_squared_error(y[ite],p1)))
        print(f"{fam:<8}{L:>5}{np.mean(r1):>10.3f}{np.mean(r2):>10.3f}{np.mean(rm):>12.3f}")

# ---------- 实验二：小样本（009 关键技术 (a)）----------
print("\n【实验二】小样本：训练集只有 N 条时，函数型 vs 离散")
B,J,R,C=cache[('B样条',40)]
print(f"{'N_train':>8}{'PLSR':>9}{'FPCR':>9}{'PFLR':>9}   (R²，20 次重抽平均)")
rows=[]
for N in [20,30,50,80,120,200,400]:
    a,b,c=[],[],[]
    for sd in range(20):
        rng=np.random.RandomState(sd)
        idx=rng.permutation(len(y)); itr,ite=idx[:N],idx[N:N+400]
        a.append(r2_score(y[ite],plsr(Xs[itr],y[itr],Xs[ite])))
        Kg=[k for k in [2,5,10,15,20,30] if k<=max(2,int(N*0.75)-2)] or [2]
        K=cv_pick(C[itr],y[itr],J,R,'fpcr',Kg)      # 与实验一同样用 CV 选 K，避免人为设限
        b.append(r2_score(y[ite],fpcr(C[itr],y[itr],C[ite],J,K)))
        lam=cv_pick(C[itr],y[itr],J,R,'pflr',np.logspace(-8,2,6))
        c.append(r2_score(y[ite],pflr(C[itr],y[itr],C[ite],J,R,lam)))
    rows.append([N,np.mean(a),np.mean(b),np.mean(c)])
    print(f"{N:>8}{np.mean(a):>9.3f}{np.mean(b):>9.3f}{np.mean(c):>9.3f}")
pd.DataFrame(rows,columns=['N_train','PLSR','FPCR','PFLR']).to_csv('04outputs/08fda_smallsample.csv',index=False,encoding='utf-8-sig')

# ---------- 实验三：跨产地小样本迁移（009 关键技术 (b) 负迁移）----------
print("\n【实验三】跨产地：源域全量建模，目标域仅 N 条时的三种做法")
LOC={1:'新疆',2:'山东',3:'陕西'}
print(f"{'迁移':<14}{'N':>4}{'源模型直用':>11}{'目标域重建PLSR':>15}{'目标域重建PFLR':>15}")
for s_,t_ in [(2,1),(3,1),(1,2)]:
    ms,mt=z==s_,z==t_
    lam_src=cv_pick(C[ms],y[ms],J,R,'pflr',np.logspace(-10,2,13))
    for N in [10,20,40]:
        d0,d1,d2=[],[],[]
        for sd in range(15):
            rng=np.random.RandomState(sd)
            ti=np.where(mt)[0]; rng.shuffle(ti); tr,te=ti[:N],ti[N:]
            d0.append(r2_score(y[te], pflr(C[ms],y[ms],C[te],J,R,lam_src)))
            d1.append(r2_score(y[te], plsr(Xs[tr],y[tr],Xs[te])) if N>6 else np.nan)
            lam_t=cv_pick(C[tr],y[tr],J,R,'pflr',np.logspace(-8,2,6)) if N>8 else 1e-4
            d2.append(r2_score(y[te], pflr(C[tr],y[tr],C[te],J,R,lam_t)))
        print(f"{LOC[s_]}→{LOC[t_]:<10}{N:>4}{np.nanmean(d0):>11.3f}{np.nanmean(d1):>15.3f}{np.nanmean(d2):>15.3f}")
