# -*- coding: utf-8 -*-
"""关键验证：2018/2019 的成功是否依赖 1000-1250nm？把波段砍到 2025 的范围看掉多少。
若砍完也崩 → 2025 的失败是波段范围所致，是仪器选型问题，不是数据坏。
评价协议与 25nested_cv.py 一致：外层 KFold(5, shuffle=True)，潜变量数在每个外层训练折内以 5 折交叉验证选取。
只读 05data。"""
import os
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
D = os.environ.get('APPLE_SSC_DATA', '../../05data/001_apple_hyperspectral_multiyear')   # 数据不随代码分发，见 08github/data/DATA.md
def load(f):
    d=pd.read_excel(f,header=None)
    wl=d.iloc[0,:2048].astype(float).values
    X=d.iloc[2:,:2048].astype(float).values
    y=pd.to_numeric(d.iloc[2:,2048],errors='coerce').values
    z=pd.to_numeric(d.iloc[2:,2049],errors='coerce').values
    ok=np.isfinite(y)&np.isfinite(z)&(y>5)&(y<25)&np.isfinite(X).all(1)
    return X[ok],y[ok],z[ok].astype(int),wl
X1,y1,z1,wl=load(f"{D}/01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx")
X2,y2,z2,_ =load(f"{D}/1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx")
X=np.vstack([X1,X2]); y=np.r_[y1,y2]
print(f"2018+2019 合并 n={len(y)}，波段 {wl[0]:.1f}-{wl[-1]:.1f} nm\n")
def snv(M):
    m=M.mean(1,keepdims=True); s=M.std(1,keepdims=True); s[s==0]=1; return (M-m)/s
K_OUTER=K_INNER=5; SEED=20041210          # 与 25nested_cv.py 同一评价协议（项目规范 7.1）
def ev(Xs,tag):
    Xp=savgol_filter(snv(Xs),15,2,deriv=1,axis=1); R=[]
    for itr,ite in KFold(K_OUTER,shuffle=True,random_state=SEED).split(Xp):
        Xtr,Xte,ytr,yte=Xp[itr],Xp[ite],y[itr],y[ite]
        bn,bs=8,1e9
        for nc in range(2,26,2):
            e=[mean_squared_error(ytr[va],PLSRegression(n_components=nc).fit(Xtr[tr],ytr[tr]).predict(Xtr[va]).ravel())
               for tr,va in KFold(K_INNER,shuffle=True,random_state=SEED).split(Xtr)]
            if np.mean(e)<bs: bs,bn=np.mean(e),nc
        yp=PLSRegression(n_components=bn).fit(Xtr,ytr).predict(Xte).ravel()
        R.append([r2_score(yte,yp),np.sqrt(mean_squared_error(yte,yp))])
    m_=np.mean(R,0); print(f"  {tag:<32} {Xs.shape[1]:>5} 波段  R²={m_[0]:>7.3f}  RMSE={m_[1]:.3f}  RPD={np.std(y)/m_[1]:.2f}")
    return m_[0]
print("波段区间对 2018+2019 建模精度的影响：")
full=ev(X,"全波段 590-1250nm（原始）")
cut =ev(X[:,wl<=1001],"590-1001nm（= 2025 的上限）")
for lo,hi in [(590,900),(900,1250),(1000,1250),(1100,1250),(950,1100)]:
    m=(wl>=lo)&(wl<=hi)
    if m.sum()>30: ev(X[:,m],f"{lo}-{hi}nm")
print(f"\n判读：砍到 1001nm 后 R² 由 {full:.3f} → {cut:.3f}（保留 {cut/full*100:.0f}%）")
print("      若 cut 仍明显 >0.3，则 2025 的失败不能只归因于波段范围，需另找原因。")
