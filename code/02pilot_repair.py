# -*- coding: utf-8 -*-
"""idea 阶段 pilot-2：漂移能不能便宜地修好？
比较 3 种策略在目标域仅有 N 个标注样本时的挽回能力：
  (a) 不校正  (b) slope-bias 校正(只调一次线性映射)  (c) 用这 N 个样本从头建模
只读 05data。"""
import os
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

D = os.environ.get('APPLE_SSC_DATA', '../../05data/001_apple_hyperspectral_multiyear')   # 数据不随代码分发，见 08github/data/DATA.md
LOC = {1: '新疆', 2: '山东', 3: '陕西'}
def load(f):
    d=pd.read_excel(f,header=None)
    X=d.iloc[2:,:2048].astype(float).values
    y=pd.to_numeric(d.iloc[2:,2048],errors='coerce').values
    z=pd.to_numeric(d.iloc[2:,2049],errors='coerce').values
    ok=np.isfinite(y)&np.isfinite(z)&(y>5)&(y<25)&np.isfinite(X).all(1)
    return X[ok],y[ok],z[ok].astype(int)
def snv(X):
    m=X.mean(1,keepdims=True); s=X.std(1,keepdims=True); s[s==0]=1; return (X-m)/s
X18,y18,z18=load(f"{D}/01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx")
X19,y19,z19=load(f"{D}/1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx")
X18,X19=snv(X18),snv(X19)
rmse=lambda a,b: float(np.sqrt(mean_squared_error(a,b)))

CASES=[("山东2018→新疆2019",X18[z18==2],y18[z18==2],X19[z19==1],y19[z19==1]),
       ("陕西2018→新疆2019",X18[z18==3],y18[z18==3],X19[z19==1],y19[z19==1]),
       ("山东2018→新疆2018",X18[z18==2],y18[z18==2],X18[z18==1],y18[z18==1]),
       ("新疆2018→新疆2019",X18[z18==1],y18[z18==1],X19[z19==1],y19[z19==1])]
NS=[5,10,20,40]; R=30
print(f"{'迁移对':<20}{'不校正':>8}" + "".join(f"{f'bias修N={n}':>12}" for n in NS) + "".join(f"{f'重建模N={n}':>12}" for n in NS))
for name,Xs,ys,Xt,yt in CASES:
    src=PLSRegression(n_components=12).fit(Xs,ys)
    p_all=src.predict(Xt).ravel(); raw=rmse(yt,p_all)
    rowb,rowf=[],[]
    for n in NS:
        bs,fs=[],[]
        for r in range(R):
            rng=np.random.RandomState(r); idx=rng.choice(len(yt),n,replace=False)
            hold=np.setdiff1d(np.arange(len(yt)),idx)
            # (b) slope-bias：在 n 个样本上把源模型预测线性重映射
            a,b=np.polyfit(p_all[idx],yt[idx],1)
            bs.append(rmse(yt[hold], a*p_all[hold]+b))
            # (c) 只用 n 个目标样本从头建 PLS
            nc=max(1,min(6,n-1))
            f=PLSRegression(n_components=nc).fit(Xt[idx],yt[idx])
            fs.append(rmse(yt[hold], f.predict(Xt[hold]).ravel()))
        rowb.append(np.mean(bs)); rowf.append(np.mean(fs))
    print(f"{name:<20}{raw:>8.3f}" + "".join(f"{v:>12.3f}" for v in rowb) + "".join(f"{v:>12.3f}" for v in rowf))
print("\n参考：同年同产地 5 折 CV 基线 RMSE≈0.66（见 01pilot_drift.log）")
