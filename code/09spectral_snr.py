# -*- coding: utf-8 -*-
"""查 900-1250nm 表现差是否源于数据质量：分区间看信噪比、动态范围、与糖度的相关性。
用于判定「实测 590-900nm 主导」与申报书 B「970nm 糖带锚定」的不一致是不是处理问题。只读 05data。"""
import os
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D = os.environ.get('APPLE_SSC_DATA', '../../05data/001_apple_hyperspectral_multiyear')   # 数据不随代码分发，见 08github/data/DATA.md
def load(f):
    d=pd.read_excel(f,header=None)
    wl=d.iloc[0,:2048].astype(float).values
    X=d.iloc[2:,:2048].astype(float).values
    y=pd.to_numeric(d.iloc[2:,2048],errors='coerce').values
    z=pd.to_numeric(d.iloc[2:,2049],errors='coerce').values
    ok=np.isfinite(y)&np.isfinite(z)&(y>5)&(y<25)&np.isfinite(X).all(1)
    return X[ok],y[ok],wl
X1,y1,wl=load(f"{D}/01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx")
X2,y2,_ =load(f"{D}/1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx")
X=np.vstack([X1,X2]); y=np.r_[y1,y2]
print(f"n={len(y)}  原始强度（未预处理）")
# 逐波段：信号=样本间均值，噪声=相邻波段差分的稳健尺度（光谱应光滑，差分即噪声）
d1=np.diff(X,axis=1)
noise=1.4826*np.median(np.abs(d1-np.median(d1,axis=1,keepdims=True)),axis=1)  # 每条谱一个噪声尺度
print(f"每条光谱的噪声尺度(MAD of 一阶差分)：中位 {np.median(noise):.2f}")
print(f"\n{'区间(nm)':<14}{'波段数':>6}{'平均强度':>10}{'谱内噪声':>10}{'SNR':>8}{'|与SSC相关|最大':>14}{'饱和(≥65000)占比':>16}")
for lo,hi in [(590,700),(700,800),(800,900),(900,1000),(1000,1100),(1100,1250)]:
    m=(wl>=lo)&(wl<hi); Xs=X[:,m]
    lvl=Xs.mean()
    nz=np.median(1.4826*np.median(np.abs(np.diff(Xs,axis=1)-np.median(np.diff(Xs,axis=1),axis=1,keepdims=True)),axis=1))
    r=np.array([abs(np.corrcoef(Xs[:,j],y)[0,1]) for j in range(0,Xs.shape[1],5)])
    sat=(Xs>=65000).mean()
    print(f"{lo}-{hi:<9}{m.sum():>6}{lvl:>10.0f}{nz:>10.1f}{lvl/max(nz,1e-9):>8.1f}{np.nanmax(r):>14.3f}{sat:>16.4f}")
print("\n判读：若 900nm 以上 SNR 显著低或饱和占比高 → 数据质量问题；")
print("      若 SNR 正常但与 SSC 相关性低 → 该区间在本数据上确实信息少，可作为实证反馈。")
