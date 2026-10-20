# -*- coding: utf-8 -*-
"""idea 阶段 pilot-3：误差机理是【纯平移】还是【平移+增益】？
在全部 15 个迁移对上比较 4 种策略：
  raw      不校正
  bias1    只修截距（1 参数：把预测的均值对齐目标域少量样本）
  slope2   斜率+截距（2 参数，polyfit 一阶）
  refit    只用这 N 个目标域样本重新建 PLS
若 bias1 ≈ slope2 → 误差主体是纯平移；若 slope2 明显更好 → 增益也变了。
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

# 15 个迁移对：同年跨产地 6 + 跨年同产地 3 + 双跨 6
PAIRS=[]
for a in (1,2,3):
    for b in (1,2,3):
        if a!=b:
            PAIRS.append((f"{LOC[a]}18→{LOC[b]}18","产地",X18,y18,z18,a,X18,y18,z18,b))
            PAIRS.append((f"{LOC[a]}18→{LOC[b]}19","双跨",  X18,y18,z18,a,X19,y19,z19,b))
    PAIRS.append((f"{LOC[a]}18→{LOC[a]}19","年份",X18,y18,z18,a,X19,y19,z19,a))

NS=[3,5,10,20,40,80]; R=30
rows=[]
for name,kind,Xs,ys,zs,a,Xt,yt,zt,b in PAIRS:
    Xs_,ys_=Xs[zs==a],ys[zs==a]; Xt_,yt_=Xt[zt==b],yt[zt==b]
    src=PLSRegression(n_components=12).fit(Xs_,ys_)
    p=src.predict(Xt_).ravel(); raw=rmse(yt_,p)
    rec={'pair':name,'kind':kind,'raw':raw}
    for n in NS:
        b1,s2,rf=[],[],[]
        for r in range(R):
            rng=np.random.RandomState(r); idx=rng.choice(len(yt_),n,replace=False)
            hold=np.setdiff1d(np.arange(len(yt_)),idx)
            b1.append(rmse(yt_[hold], p[hold] + (yt_[idx]-p[idx]).mean()))          # 1 参数
            sl,ic=np.polyfit(p[idx],yt_[idx],1)
            s2.append(rmse(yt_[hold], sl*p[hold]+ic))                                # 2 参数
            m=PLSRegression(n_components=max(1,min(6,n-1))).fit(Xt_[idx],yt_[idx])
            rf.append(rmse(yt_[hold], m.predict(Xt_[hold]).ravel()))                 # 重建模
        rec[f'bias1_N{n}']=np.mean(b1); rec[f'slope2_N{n}']=np.mean(s2); rec[f'refit_N{n}']=np.mean(rf)
    rows.append(rec)
df=pd.DataFrame(rows)

print("========== 按漂移类型汇总（RMSE 均值，°Brix）==========")
g=df.groupby('kind')
cols=['raw']+[f'{s}_N{n}' for n in NS for s in ('bias1','slope2','refit')]
summ=g[cols].mean().reindex(['产地','年份','双跨'])
print(f"{'类型':<6}{'n对':>4}{'不校正':>9}" + "".join(f"{f'{s}N{n}':>10}" for n in NS for s in ('bias1','slope2','refit')))
for k,r in summ.iterrows():
    print(f"{k:<6}{int((df.kind==k).sum()):>4}{r['raw']:>9.3f}" + "".join(f"{r[f'{s}_N{n}']:>10.3f}" for n in NS for s in ('bias1','slope2','refit')))

print("\n========== 机理判读：1 参数 vs 2 参数 ==========")
for n in NS:
    d=(summ[f'slope2_N{n}']-summ[f'bias1_N{n}'])
    print(f"  N={n:>2}: slope2−bias1 = " + "  ".join(f"{k}:{v:+.3f}" for k,v in d.items()) + "   (负=修增益有用；正=样本太少估斜率反而更差)")

print("\n========== 校正 vs 重建模 的交叉点 ==========")
for k in ['产地','年份','双跨']:
    best=[('N'+str(n), summ.loc[k,f'slope2_N{n}'], summ.loc[k,f'bias1_N{n}'], summ.loc[k,f'refit_N{n}']) for n in NS]
    print(f"  {k}: " + "  ".join(f"{a}(校正最优{min(b,c):.2f} vs 重建模{d:.2f}){'✓校正' if min(b,c)<d else '✗重建模'}" for a,b,c,d in best))

df.to_csv('04outputs/03pilot_mechanism.csv',index=False,encoding='utf-8-sig')
print("\n逐对明细 → 04outputs/03pilot_mechanism.csv")
print("\n========== 最坏 5 对（不校正 RMSE 最高）==========")
w=df.nlargest(5,'raw')[['pair','kind','raw','bias1_N5','slope2_N5','refit_N5','slope2_N20','refit_N20']]
print(w.to_string(index=False,float_format=lambda v:f"{v:.3f}"))
