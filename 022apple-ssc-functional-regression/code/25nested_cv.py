# -*- coding: utf-8 -*-
"""统一协议下六方法的嵌套交叉验证（项目规范 7.1 默认强制的评价协议）。

为什么要有这个脚本：16unified_comparison.py 用的是「5 次 70%/30% 重复随机划分 + 建模集内
20% 固定验证集选参」，属重复留出法——测试集之间互相重叠、并不覆盖全部样本，内层也只有一次
验证集，选参本身的方差进不了误差棒。项目规范 7.1 对本项目定的是**嵌套交叉验证（默认强制）**：
外层 K_outer=5、内层 K_inner=5、回归用 KFold(shuffle=True)，每折独立（内层选参 → 外层重训练 →
外层测试，折间不共享超参）。7.2 进一步规定统计单位 n_primary=K_outer、CI 走 fold 层级 cluster
bootstrap。本脚本按该协议重跑，取代 16 脚本作为表 3 及其下游（表 5~表 8、图 2）的数据来源。

协议（六个方法完全一致，与 16 脚本共用同一份预处理、同一批 12 组候选网格）：
  候选网格 六个方法各 12 组，由 02code/grids.py 统一给出（第二轮网格，重定理由见该文件头）
  预处理  SNV + Savitzky-Golay 一阶导（窗宽 15，二次多项式）
          两者均为逐样本运算（SNV 按行去均值除标准差、SG 沿波长卷积），不使用其他样本的信息，
          故在划分之前整体计算不构成信息泄漏；基函数展开同样逐样本求解，同理。
  外层    KFold(n_splits=5, shuffle=True, random_state=seed)，六个方法共用同一批折
  内层    在外层训练折上再作 KFold(n_splits=5, shuffle=True, random_state=seed)，
          12 组候选各算 5 折的 RMSE 均值，取最小者
  重训练  用整个外层训练折重新拟合选中的配置，在外层测试折上评价
  种子    项目规范 422 行的 Formal 集合去重后 5 个；按种子分段落盘，跑完一个存一次
  报告    每折的 R² / RMSE / RPD / MAE / P90 / 平均偏差，以及均值预测基准的 MAE / P90

只读 05data。
"""
import os, sys, time; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# 复用 16 脚本的数据、预处理、基系统、六个方法的候选网格与拟合函数（到 METHODS 定义之前为止）
with open('02code/16unified_comparison.py') as _src: exec(_src.read().split("METHODS=['PLSR'")[0])
sys.path.insert(0, '02code')
from grids import GRIDS          # 第二轮候选网格，唯一定义处；重定理由见 grids.py 文件头

METHODS = ['PLSR', 'SVR', 'RF', '1D-CNN', 'FPCR', 'PFLR']
if os.environ.get('NCV_METHODS'): METHODS = os.environ['NCV_METHODS'].split(',')   # 分段跑用
for m in METHODS:
    assert len(GRIDS[m]) == 12, (m, len(GRIDS[m]))
K_OUTER, K_INNER = 5, 5
SEEDS = [20041210, 20240404, 20060515, 20251203, 19810915]      # Formal 集合去重后 5 个
if os.environ.get('NCV_SEEDS'): SEEDS = [int(x) for x in os.environ['NCV_SEEDS'].split(',')]
OUT = '04outputs/25nested_cv.csv'
PRED = '04outputs/25nested_cv_pred.csv'

print(f"n={len(y)}  嵌套 CV：K_outer={K_OUTER}  K_inner={K_INNER}  种子 {SEEDS}")
print("六个方法各 12 组候选，共用同一批外层折与内层折\n")

def fold_metrics(ytr, yte, p):
    """一个外层折上的全部指标；base_* 为「以训练折均值预测」这一无技能基准的同名指标。"""
    e = np.asarray(p) - yte; ae = np.abs(e)
    rm = float(np.sqrt(mean_squared_error(yte, p))); sdte = float(np.std(yte))
    be = np.abs(float(np.mean(ytr)) - yte)
    return dict(R2=float(r2_score(yte, p)), RMSE=rm, RPD=sdte / rm,
                MAE=float(ae.mean()), P90=float(np.quantile(ae, 0.9)), BIAS=float(e.mean()),
                base_MAE=float(be.mean()), base_P90=float(np.quantile(be, 0.9)),
                s_y_train=float(np.std(ytr)), s_y_test=sdte)

rows, preds = [], []
if os.path.exists(OUT):                                         # 断点续跑：已完成的种子不重算
    rows = pd.read_csv(OUT).to_dict('records')
    preds = pd.read_csv(PRED).to_dict('records')
done = {(r['seed'], r['method']) for r in rows}

for seed in SEEDS:
    t_seed = time.time()
    folds = list(KFold(n_splits=K_OUTER, shuffle=True, random_state=seed).split(np.arange(len(y))))
    for m in METHODS:
        if (seed, m) in done:
            print(f"  seed={seed} {m:<8} 已完成，跳过"); continue
        grid, fit = GRIDS[m], mk(m)[1]; t0 = time.time()
        for k, (itr, ite) in enumerate(folds):
            inner = list(KFold(n_splits=K_INNER, shuffle=True, random_state=seed).split(itr))
            best, bc = np.inf, grid[0]
            for c in grid:
                se = [mean_squared_error(y[itr[iv]], fit(c, itr[ia], itr[iv], seed)) for ia, iv in inner]
                e = float(np.mean([np.sqrt(s) for s in se]))
                if e < best: best, bc = e, c
            p = fit(bc, itr, ite, seed)                          # 用整个外层训练折重训练
            mt = fold_metrics(y[itr], y[ite], p)
            rows.append(dict(seed=seed, method=m, fold=k, picked=str(bc), **mt))
            preds += [dict(seed=seed, method=m, fold=k, idx=int(i), y=float(y[i]), pred=float(v))
                      for i, v in zip(ite, p, strict=False)]
            print(f"  seed={seed} {m:<8} fold={k}  选中={str(bc):<34} "
                  f"R²={mt['R2']:.3f}  RMSE={mt['RMSE']:.3f}")
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding='utf-8-sig')
        pd.DataFrame(preds).to_csv(PRED, index=False, encoding='utf-8-sig')
        print(f"  seed={seed} {m:<8} 用时 {time.time()-t0:.0f}s，已落盘\n", flush=True)
    df = pd.DataFrame(rows); d = df[df.seed == seed]
    print(f"=== seed={seed} 完成，用时 {time.time()-t_seed:.0f}s ===")
    for m in METHODS:
        a = d[d.method == m]
        if len(a): print(f"  {m:<8} R²={a.R2.mean():.3f}±{a.R2.std(ddof=0):.3f}  "
                         f"RMSE={a.RMSE.mean():.3f}  RPD={a.RPD.mean():.2f}")
    print(flush=True)

df = pd.DataFrame(rows)
print("=" * 76)
print(f"【嵌套交叉验证下的六方法对比】K_outer={K_OUTER}, K_inner={K_INNER}, "
      f"n_seeds={df.seed.nunique()}，统计单位 n_primary={K_OUTER}")
print(f"{'方法':<10}{'R²':>18}{'RMSE/°Brix':>20}{'RPD':>16}")
for m in METHODS:
    a = df[df.method == m]
    print(f"{m:<10}{a.R2.mean():>10.3f}±{a.R2.std(ddof=0):<7.3f}"
          f"{a.RMSE.mean():>12.3f}±{a.RMSE.std(ddof=0):<7.3f}{a.RPD.mean():>9.2f}±{a.RPD.std(ddof=0):<6.2f}")
print(f"\n已写出 {OUT} 与 {PRED}")
