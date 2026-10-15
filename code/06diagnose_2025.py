# -*- coding: utf-8 -*-
"""诊断：2025 数据为何 R²≈0。查配对、光谱质量、波段范围、产地分层。只读 05data。"""
import os
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
D = os.environ.get('APPLE_SSC_DATA', '../../05data/001_apple_hyperspectral_multiyear')   # 数据不随代码分发，见 08github/data/DATA.md
spec=pd.read_csv(f"{D}/新疆-山东-甘肃苹果光谱数据2025.csv")
ssc =pd.read_csv(f"{D}/新疆-山东-甘肃苹果糖度数据2025.csv",encoding='gbk')
wcol=[c for c in spec.columns if c.endswith('nm')]
wl=np.array([float(c[:-2]) for c in wcol])
k=spec['实际苹果编号'].astype(str).str.split('-',n=1,expand=True)
spec['code'],spec['face']=k[0],k[1]

print("【1】配对完整性")
print("  光谱 face 取值:",dict(spec['face'].value_counts()))
print("  糖度列:",list(ssc.columns))
print("  光谱 code 数:",spec['code'].nunique()," 糖度 code 数:",ssc['苹果编码'].nunique())
print("  交集 code 数:",len(set(spec['code'])&set(ssc['苹果编码'].astype(str))))

print("\n【2】光谱质量")
A=spec[wcol].values.astype(float)
print(f"  反射率范围 [{np.nanmin(A):.4f}, {np.nanmax(A):.4f}]；含 0 的波段数 {(A==0).any(0).sum()}；含 NaN 行 {np.isnan(A).any(1).sum()}")
zero_frac=(A==0).mean(0)
bad=wl[zero_frac>0.01]
print(f"  >1% 样本为 0 的波段: {len(bad)} 个" + (f"（{bad.min():.0f}–{bad.max():.0f} nm）" if len(bad) else ""))
# 同一苹果 5 个面之间的光谱相关性（配对正确的话应该很高）
g=spec.groupby('code')
cors=[]
for _c, d in list(g)[:200]:
    if len(d)>=3:
        M=d[wcol].values.astype(float)
        C=np.corrcoef(M); cors.append(C[np.triu_indices_from(C,1)].mean())
print(f"  同果不同面光谱平均相关系数: {np.nanmean(cors):.3f}（低则说明行与果的对应有问题）")

print("\n【3】糖度标签")
long=ssc.melt(id_vars='苹果编码',var_name='face',value_name='SSC').rename(columns={'苹果编码':'code'})
long['code']=long['code'].astype(str)
df=spec.merge(long,on=['code','face'],how='inner')
print(f"  配对 {len(df)} 条；SSC {df.SSC.mean():.2f}±{df.SSC.std():.2f}，范围 [{df.SSC.min():.1f},{df.SSC.max():.1f}]")
d=df[(df.SSC>5)&(df.SSC<25)]
print(f"  清洗后 {len(d)} 条；SSC {d.SSC.mean():.2f}±{d.SSC.std():.2f}")
print("  同果 5 面糖度的组内标准差(均值): %.3f  vs 果间标准差 %.3f"%(
    d.groupby('code').SSC.std().mean(), d.groupby('code').SSC.mean().std()))

print("\n【4】换波段区间 / 分产地 / 按果聚合 重测（PLSR + SNV+SG1D）")
def snv(M):
    m=M.mean(1,keepdims=True); s=M.std(1,keepdims=True); s[s==0]=1; return (M-m)/s
def ev(X,y,tag):
    if len(y)<60: print(f"  {tag:<34} 样本不足 {len(y)}"); return
    Xp=savgol_filter(snv(X),15,2,deriv=1,axis=1)
    R=[]
    for sd in [0,1]:
        Xtr,Xte,ytr,yte=train_test_split(Xp,y,test_size=0.3,random_state=sd)
        bn,bs=8,1e9
        for nc in range(2,26,2):
            e=[mean_squared_error(ytr[va],PLSRegression(n_components=nc).fit(Xtr[tr],ytr[tr]).predict(Xtr[va]).ravel())
               for tr,va in KFold(5,shuffle=True,random_state=0).split(Xtr)]
            if np.mean(e)<bs: bs,bn=np.mean(e),nc
        yp=PLSRegression(n_components=bn).fit(Xtr,ytr).predict(Xte).ravel()
        R.append([r2_score(yte,yp),np.sqrt(mean_squared_error(yte,yp))])
    m_=np.mean(R,0); print(f"  {tag:<34} n={len(y):<5} R²={m_[0]:>7.3f}  RMSEP={m_[1]:.3f}")

X=d[wcol].values.astype(float); y=d.SSC.values.astype(float); pr=d['code'].str[:2].values
ev(X,y,"全波段 391-1001nm 全产地")
for lo,hi,t in [(700,1001,"700-1001nm"),(800,1001,"800-1001nm"),(900,1001,"900-1001nm"),(391,700,"391-700nm(可见光)")]:
    m=(wl>=lo)&(wl<=hi); ev(X[:,m],y,f"{t} 全产地")
for p in ['XJ','SD','GS']:
    m=pr==p; ev(X[m],y[m],f"全波段 仅{p}")
# 按果聚合：5 面光谱平均 → 果均糖度
agg=d.groupby('code').agg({**{c:'mean' for c in wcol},'SSC':'mean'}).reset_index()
ev(agg[wcol].values.astype(float),agg.SSC.values.astype(float),"按果聚合(5面均值)")
