# -*- coding: utf-8 -*-
"""直接测「小样本下交叉验证为何为 FPCR 选错超参数」：
① 20 次重抽里交叉验证各选中了哪个 K，与逐次事后最优 K 相比偏到哪里；
② 交叉验证误差估计本身的折间波动有多大（4 折之间 MSE 的相对标准差）。
只读 05data。输出：05logs/20cv_selection_mechanism.log
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
with open('02code/08fda_smallsample.py') as _src: exec(_src.read().split('# ---------- 实验一')[0])
Bm=bspline_basis(t,40); Jb,Rb=gram_and_pen(Bm,t); Cb=to_coef(Xs,Bm,Rb)
KG=[2,5,10,15,20,30]                      # 与 10smallsample_hyperparam.py 的 CV 网格一致
N=80
sel=[];post=[];cvsd=[];r2sel=[];r2post=[]
for sd_ in range(20):
    rng=np.random.RandomState(sd_); idx=rng.permutation(len(y)); itr,ite=idx[:N],idx[N:N+400]
    cvm={}; foldsd={}
    for K in KG:
        e=[mean_squared_error(y[itr][va],fpcr(Cb[itr][tr],y[itr][tr],Cb[itr][va],Jb,K))
           for tr,va in KFold(4,shuffle=True,random_state=0).split(Cb[itr])]
        cvm[K]=np.mean(e); foldsd[K]=np.std(e)/max(np.mean(e),1e-12)
    Kcv=min(cvm, key=lambda k: cvm[k])
    r2s={K:r2_score(y[ite],fpcr(Cb[itr],y[itr],Cb[ite],Jb,K)) for K in KG}   # 事后参照，不用于建模
    Kpost=max(r2s, key=lambda k: r2s[k])
    sel.append(Kcv); post.append(Kpost); cvsd.append(foldsd[Kcv])
    r2sel.append(r2s[Kcv]); r2post.append(r2s[Kpost])
print(f"N={N}，FPCR，交叉验证网格 K∈{KG}，20 次重抽\n")
print("① 交叉验证选中的 K 的分布：", dict(sorted(Counter(sel).items())))
print("   事后最优 K 的分布      ：", dict(sorted(Counter(post).items())))
print(f"   两者一致的次数：{sum(a==b for a,b in zip(sel,post, strict=False))}/20")
print(f"\n② 交叉验证所选 K 的实际 R² 均值 {np.mean(r2sel):.3f}，事后最优 K 的 R² 均值 {np.mean(r2post):.3f}"
      f"，平均损失 {np.mean(r2post)-np.mean(r2sel):.3f}")
print(f"\n③ 交叉验证误差估计的折间相对标准差（所选 K 处）：中位 {np.median(cvsd)*100:.0f}%，"
      f"范围 {min(cvsd)*100:.0f}%~{max(cvsd)*100:.0f}%")
print("   —— 折间波动达到这个量级时，各候选 K 的 CV 均值之差常被折间噪声淹没。")
