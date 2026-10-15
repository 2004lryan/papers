# -*- coding: utf-8 -*-
"""3.7 节：样本划分方式对结论的影响，并产出 LLH-GBID 综合评价所需的原始指标。

问题：主协议用随机划分（训练集 910、测试集 390，seed 0–4）。本脚本检验主结论
（各方法排序；函数型方法仅相对 PLSR 占优）是否依赖于划分方式。
划分方式：
  ⓪ 嵌套 CV：与 25nested_cv.py 完全相同的外层 5 折与内层 5 折（seed=20041210）。本脚本独立重算这一行，
     并断言与 04outputs/25nested_cv.csv 逐位一致——两个脚本的取数与预处理路径不同（本脚本走 08fda，
     25 走 16unified），逐位相同才说明表 3 的数字不依赖于某一条代码路径；
  ① 随机 70%：重复留出法，5 次随机划分（seed 0–4），训练集内再切 20% 作固定验证集选参；
  ② 随机 75%：seed 0–4；
  ③ SPXY 75%：Galvão 等 2005，x 距离与 y 距离各自按最大值归一化后相加，按 Kennard–Stone 流程选训练集（确定性，1 次）；
  ④ KS 75%：仅用 x 距离的 Kennard–Stone（确定性，1 次）。
  距离在建模所用的 SNV+SG-1D 光谱上计算；SPXY 按其定义使用全部样本的参考值。
模型与选参：PLSR、SVR、FPCR、PFLR，候选网格取 02code/grids.py 的第二轮网格（与表 3 同一份，否则表 8 各行不可比）。
  内层选参按各行自身的协议：嵌套 CV 行用训练折内的 5 折交叉验证，四种留出法行用训练集内 20% 的固定验证集；
  选中配置后一律用整个训练集（或外层训练折）重拟合，再在测试集（或外层测试折）上评价。
指标：R²、RMSE、RPD=s_y(test)/RMSE 按本领域惯例报告；
  另按 refine-logs/01PROJECT_PROTOCOL.md B9 锁定的口径记录 LLH-GBID 所需的四项——
  R²、MAE、绝对误差的 90% 分位数 P90、平均偏差 bias=mean(ŷ−y)，
  以及效用化所需的基准量：均值预测基准模型（以训练集均值预测测试集）的 MAE 与 P90、训练集与测试集的 SSC 标准差。
  综合评价本身不在本脚本内完成，由 23gbid_evaluation.py 统一计算（B9 参数已先行锁定）。
输出：04outputs/22split_scheme.csv（逐次）、04outputs/22split_scheme_summary.csv（汇总）。只读 05data。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
K_OUTER=K_INNER=5; NCV_SEED=20041210          # 与 25nested_cv.py 一致
with open('02code/08fda_smallsample.py') as _src: exec(_src.read().split('# ---------- 实验一')[0])
BASIS={}
for L in [40,80]:
    B=fourier_basis(t,L); J_,R_=gram_and_pen(B,t); BASIS[L]=(B,J_,R_)
COEF={L:to_coef(Xs,BASIS[L][0],BASIS[L][2]) for L in BASIS}
import sys as _sys; _sys.path.insert(0,'02code')
from grids import GRIDS          # 与表 3 同一份第二轮网格，否则表 8 的各行不可比
GRID={k:GRIDS[k] for k in ['PLSR','SVR','FPCR','PFLR']}
for g in GRID.values(): assert len(g)==12
def fit(name,c,a,b):
    if name=='PLSR': return PLSRegression(n_components=min(c,len(a)-1)).fit(Xs[a],y[a]).predict(Xs[b]).ravel()
    if name=='SVR':  return SVR(C=c[0],gamma=c[1]).fit(Xs[a],y[a]).predict(Xs[b])
    if name=='FPCR': L,K=c; return fpcr(COEF[L][a],y[a],COEF[L][b],BASIS[L][1],K)
    L,lam=c; return pflr(COEF[L][a],y[a],COEF[L][b],BASIS[L][1],BASIS[L][2],lam)

def kennard_stone(D,n_train):
    """经典 KS：先取相距最远的两点，再逐个加入「到已选集最小距离」最大的点。"""
    n=len(D); i,j=np.unravel_index(np.argmax(D),D.shape); sel=[i,j]
    rest=np.ones(n,bool); rest[[i,j]]=False; dmin=np.minimum(D[i],D[j])
    while len(sel)<n_train:
        cand=np.where(rest)[0]; k=cand[np.argmax(dmin[cand])]
        sel.append(k); rest[k]=False; dmin=np.minimum(dmin,D[k])
    return np.array(sel)
n=len(y); ntr75=int(round(0.75*n))
Dx=cdist(Xs,Xs); Dy=np.abs(y[:,None]-y[None,:])
def splits(scheme):
    if scheme=='嵌套CV':  return list(KFold(K_OUTER,shuffle=True,random_state=NCV_SEED).split(np.arange(n)))
    if scheme=='随机70%': return [train_test_split(np.arange(n),test_size=0.3,random_state=sd) for sd in range(5)]
    if scheme=='随机75%': return [train_test_split(np.arange(n),test_size=0.25,random_state=sd) for sd in range(5)]
    if scheme=='KS75%':  tr=kennard_stone(Dx,ntr75); return [(tr,np.setdiff1d(np.arange(n),tr))]
    if scheme=='SPXY75%': tr=kennard_stone(Dx/Dx.max()+Dy/Dy.max(),ntr75); return [(tr,np.setdiff1d(np.arange(n),tr))]
def metrics(ytr,yte,p):
    """原始指标 + 效用化所需的基准量（口径见 B9 §9.1）。基准模型 = 以训练集均值预测测试集。"""
    e=np.asarray(p)-yte; ae=np.abs(e); rm=float(np.sqrt(mean_squared_error(yte,p))); sdv=float(np.std(yte))
    be=np.abs(float(np.mean(ytr))-yte)
    return dict(R2=float(r2_score(yte,p)),RMSE=rm,MAE=float(mean_absolute_error(yte,p)),
                P90=float(np.quantile(ae,0.9)),BIAS=float(np.mean(e)),RPD=sdv/rm,
                base_MAE=float(be.mean()),base_P90=float(np.quantile(be,0.9)),
                s_y_train=float(np.std(ytr)),s_y_test=sdv)

SCHEMES=['嵌套CV','随机70%','随机75%','SPXY75%','KS75%']; METHODS=['PLSR','SVR','FPCR','PFLR']
print(f"n={n}；随机 70%：训练 {int(round(0.7*n))} / 测试 {n-int(round(0.7*n))}；75%：训练 {ntr75} / 测试 {n-ntr75}")
rows=[]; t0=time.time()
for scheme in SCHEMES:
    for k,(itr,ite) in enumerate(splits(scheme)):
        print(f"\n[{scheme}] 第 {k+1} 次：训练 {len(itr)} 个（SSC {y[itr].mean():.2f}±{y[itr].std():.2f}，{y[itr].min():.1f}~{y[itr].max():.1f}）"
              f"  测试 {len(ite)} 个（{y[ite].mean():.2f}±{y[ite].std():.2f}，{y[ite].min():.1f}~{y[ite].max():.1f}）")
        if scheme=='嵌套CV':
            inner=list(KFold(K_INNER,shuffle=True,random_state=NCV_SEED).split(itr))
        else:
            isub,iva=train_test_split(itr,test_size=0.2,random_state=k)
        for name in METHODS:
            best,bc=np.inf,GRID[name][0]
            for c in GRID[name]:
                if scheme=='嵌套CV':
                    e=float(np.mean([np.sqrt(mean_squared_error(y[itr[iv]],fit(name,c,itr[ia],itr[iv])))
                                     for ia,iv in inner]))
                else:
                    e=np.sqrt(mean_squared_error(y[iva],fit(name,c,isub,iva)))
                if e<best: best,bc=e,c
            m=metrics(y[itr],y[ite],fit(name,bc,itr,ite))
            rows.append(dict(scheme=scheme,rep=k,n_train=len(itr),n_test=len(ite),method=name,picked=str(bc),
                             y_train_min=y[itr].min(),y_train_max=y[itr].max(),y_test_min=y[ite].min(),y_test_max=y[ite].max(),**m))
            print(f"   {name:<5} 选中={str(bc):<16} R²={m['R2']:.3f} RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f} P90={m['P90']:.3f} bias={m['BIAS']:+.3f} RPD={m['RPD']:.2f}")
print(f"\n用时 {time.time()-t0:.0f}s")
df=pd.DataFrame(rows); df.to_csv('04outputs/22split_scheme.csv',index=False,encoding='utf-8-sig')

# ---- 对账：本脚本独立重算的嵌套 CV 行必须与 25nested_cv.py 逐折逐位一致 ----
ref=pd.read_csv('04outputs/25nested_cv.csv')
ref=ref[ref.seed==NCV_SEED]
bad=0
for name in METHODS:
    a=df[(df.scheme=='嵌套CV')&(df.method==name)].sort_values('rep')
    b=ref[ref.method==name].sort_values('fold')
    assert len(a)==len(b)==K_OUTER,(name,len(a),len(b))
    for col in ['R2','RMSE','MAE','P90','BIAS','RPD']:
        for k_,(v1,v2) in enumerate(zip(a[col].values,b[col].values, strict=False)):
            if abs(v1-v2)>1e-9:
                bad+=1; print(f"  对账不一致：{name} fold={k_} {col} 本脚本 {v1:.6f} 25 脚本 {v2:.6f}")
    if list(a.picked)!=list(b.picked.astype(str)):
        bad+=1; print(f"  对账不一致：{name} 选中配置 {list(a.picked)} vs {list(b.picked)}")
print(f"嵌套 CV 行与 25nested_cv.py 对账：{'逐位一致' if bad==0 else str(bad)+' 处不一致'}")
assert bad==0

# ---- 汇总：均值 ± 标准差；各划分下按 R² 与按 DISO 的排序；PFLR−PLSR、SVR−PFLR 的配对差 ----
_grp=df.groupby(['scheme','method'])[['R2','RMSE','MAE','P90','BIAS','RPD']]
g=_grp.mean().add_suffix('_mean').join(_grp.std(ddof=0).add_suffix('_std')).join(_grp.size().rename('n')).reset_index()   # 标准差与表 3 同口径（总体标准差，ddof=0）
g['scheme']=pd.Categorical(g.scheme,SCHEMES); g['method']=pd.Categorical(g.method,METHODS); g=g.sort_values(['scheme','method'])
g.to_csv('04outputs/22split_scheme_summary.csv',index=False,encoding='utf-8-sig')
print("\n【汇总】")
for scheme in SCHEMES:
    s=g[g.scheme==scheme]
    print(f"\n{scheme}")
    for _,r in s.iterrows():
        sd=lambda c: f"±{r[c+'_std']:.3f}" if r['n']>1 else ''
        print(f"  {r.method:<5} R²={r.R2_mean:.3f}{sd('R2')}  RMSE={r.RMSE_mean:.3f}{sd('RMSE')}  MAE={r.MAE_mean:.3f}{sd('MAE')}  P90={r.P90_mean:.3f}{sd('P90')}  bias={r.BIAS_mean:+.3f}{sd('BIAS')}  RPD={r.RPD_mean:.2f}")
    print("  按 R² 排序：", ' > '.join(s.sort_values('R2_mean',ascending=False).method.astype(str)),
          "｜ 按 MAE 排序：", ' > '.join(s.sort_values('MAE_mean').method.astype(str)))
    piv=df[df.scheme==scheme].pivot(index='rep',columns='method',values='R2')
    d1=piv.PFLR-piv.PLSR; d2=piv.SVR-piv.PFLR
    print(f"  同一划分配对差：PFLR−PLSR {d1.mean():+.3f}（{int((d1>0).sum())}/{len(d1)} 次占优）；SVR−PFLR {d2.mean():+.3f}（{int((d2>0).sum())}/{len(d2)} 次占优）")
print("\n已写出 04outputs/22split_scheme.csv 与 04outputs/22split_scheme_summary.csv")
