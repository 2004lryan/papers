# -*- coding: utf-8 -*-
"""图 2：折外预测值-实测值散点图（点按局部密度着色）。样式见 figstyle.py。

数据来源：02code/25nested_cv.py 落盘的逐样本折外预测 04outputs/25nested_cv_pred.csv。
嵌套交叉验证中每个样本恰好在一个外层折里充当测试样本，把 5 折的预测拼起来即得到全部
1 300 个样本的折外预测——这正是嵌套 CV 相对重复留出法的一个直接好处：散点图不再只画
某一次划分抽到的那 390 个样本，而是覆盖全体样本，且每个点的预测都来自未见过它的模型。

图中标注的 R² 与 RMSE 按拼合后的全部折外预测计算，因此与表 3 报告的「5 折均值」不是同一个
统计量（前者是把 5 折的预测合起来算一次，后者是逐折算完再平均），脚本内断言二者之差在
0.02 以内，超出即报错，以免读者把两个数字当成同一个。
只读 05data。
"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import sys; sys.path.insert(0,'02code')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
from figstyle import *      # 多面板图在 figstyle.save 内经 require_matplotlib_panel_alignment(strict=True) 审计后才导出

PRIMARY_SEED = 20041210
SHOW = ['SVR','PFLR','PLSR']        # 最优离散方法、最优函数型方法、行业基准
pred = pd.read_csv('04outputs/25nested_cv_pred.csv')
ncv  = pd.read_csv('04outputs/25nested_cv.csv')
pred = pred[pred.seed == PRIMARY_SEED]; ncv = ncv[ncv.seed == PRIMARY_SEED]
for m in SHOW:
    assert pred[pred.method==m].idx.nunique() == len(pred[pred.method==m]), f'{m} 折外预测有重复样本'

lo,hi=8,19
fig,ax=plt.subplots(1,3,figsize=(W2,2.3),layout='constrained',sharex=True,sharey=True)
cmap=matplotlib.colors.LinearSegmentedColormap.from_list('dens',[C['paleblue'],C['blue'],'#052F5F'])
sc=None
for a,name in zip(ax,SHOW, strict=False):
    q=pred[pred.method==name].sort_values('idx')
    yt=q.y.values; p=q.pred.values
    r2=r2_score(yt,p); rm=float(np.sqrt(mean_squared_error(yt,p)))
    fold_mean_r2 = float(ncv[ncv.method==name].R2.mean())
    assert abs(r2-fold_mean_r2)<0.02, f'{name} 拼合折外 R²={r2:.3f} 与逐折均值 {fold_mean_r2:.3f} 相差过大'
    dens=gaussian_kde(np.vstack([yt,p]))(np.vstack([yt,p])); dens=dens/dens.max()
    o=np.argsort(dens)                                     # 高密度点画在上层
    sc=a.scatter(yt[o],p[o],c=dens[o],s=4,cmap=cmap,vmin=0,vmax=1,edgecolors='none',zorder=2)
    a.plot([lo,hi],[lo,hi],ls='--',c=C['vermilion'],lw=0.9,zorder=3)
    a.set_xlim(lo,hi); a.set_ylim(lo,hi); a.set_aspect('equal'); a.set_xticks([8,10,12,14,16,18])
    a.set_title(f'{name}'); a.set_xlabel('实测值/°Brix')
    a.text(0.05,0.95,f'R²={r2:.3f}\nRMSE={rm:.3f} °Brix',transform=a.transAxes,va='top',ha='left',fontsize=7,linespacing=1.4)
    print(f"  {name:<5} 折外 n={len(yt)}  R²={r2:.3f}  RMSE={rm:.3f}   （逐折均值 R²={fold_mean_r2:.3f}）")
ax[0].set_ylabel('预测值/°Brix')
# matplotlib 的 stub 把 scatter 返回值记为 Any | None、把 colorbar.outline 记为 Spine，
# 与实际返回的 PathCollection / 可设线宽的 outline 不符；此处按实际行为忽略。
cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, aspect=25)   # type: ignore[arg-type]
cb.set_label('样本点相对密度')
cb.outline.set_linewidth(0.5)   # type: ignore[union-attr,operator]
save(fig,'Figure_5_scatter',exclude=[cb.ax])
print("Figure_5_scatter 完成（嵌套 CV 折外预测，n=%d）"%len(pred[pred.method==SHOW[0]]))
