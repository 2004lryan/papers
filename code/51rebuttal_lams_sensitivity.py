# -*- coding: utf-8 -*-
"""
51rebuttal_lams_sensitivity.py: 光谱展开平滑参数 λs 的敏感性（交叉审稿：1.1 节 λs=10⁻⁶ 缺取值依据）

为什么要有这个脚本：1.1 节把 B 样条展开的平滑参数 λs 固定为 10⁻⁶，正文没有给出取值依据。
λs 并非可以忽略的小量：SNV+SG-1D 光谱在三次 B 样条 L=80 上展开时，λs·R 的对角元约为 Φ′Φ 对角元的 0.1 倍，
拟合光谱与不加惩罚时相差数个百分点。若 FPLR、FPCR 的精度随 λs 明显变化，表 1 的函数型结果就依赖一个未经选择的设定。
本脚本在表 1 的同一协议下（5 折×5 次重复嵌套交叉验证、同一批外层与内层折、grids.py 第二轮 12 组候选、
内层取 5 折 RMSE 均值最小者）把 λs 取 0、10⁻⁸、10⁻⁶、10⁻⁴，重算两个函数型模型的外层 R²，
并记录各 λs 下拟合光谱与 λs=0 时的相对差、拟合残差与光谱之比。

复现校验：λs=10⁻⁶ 的逐折 R² 必须与 04outputs/45rebuttal_ncv_bspline_{FPLR,FPCR}.csv 逐位一致（差 <1e-9），
不一致说明本脚本的协议与表 1 不同，直接报错、不写产物。

运行方式:
    cd <项目根目录>
    python3 02code/51rebuttal_lams_sensitivity.py > 05logs/51rebuttal_lams_sensitivity_<时间戳>.log 2>&1

输出:
    04outputs/51rebuttal_lams_folds.csv    — 逐（λs, 方法, 种子, 折）的选中配置、R²、RMSE
    04outputs/51rebuttal_lams_summary.csv  — 逐（λs, 方法）的 25 折 R² 均值与标准差、与 λs=10⁻⁶ 逐折配对差值的均值与范围，
                                              以及该 λs 下拟合光谱与 λs=0 的相对差、残差与光谱之比
只读 05data。
"""
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# 数据、预处理、基函数、pflr/fpcr 与 mk()：与表 1（45 脚本）共用 16 脚本的同一份实现
with open('02code/16unified_comparison.py') as _src:
    _SHARED = _src.read().split("METHODS=['PLSR'")[0]
exec(_SHARED)
sys.path.insert(0, '02code')
from grids import GRIDS  # noqa: E402

SEEDS = [20041210, 20240404, 20060515, 20251203, 19810915]          # 与 45 脚本相同的 5 个种子
K_OUTER, K_INNER, L_BASIS = 5, 5, 80
LAMS = [0.0, 1e-8, 1e-6, 1e-4]
REF_LAM = 1e-6                                                       # 表 1 所用取值
LABEL = {'PFLR': 'FPLR', 'FPCR': 'FPCR'}

_B = bspline_basis(t, L_BASIS)
_J, _Rm = gram_and_pen(_B, t)
BASIS = {L_BASIS: (_B, _J, _Rm)}          # mk() 的闭包在调用时按名取 BASIS、COEF，模块级重新赋值即生效
print(f"三次 B 样条 L={L_BASIS}：diag(Φ′Φ) 中位数 {np.median(np.diag(_B.T @ _B)):.3e}，"
      f"diag(R) 中位数 {np.median(np.diag(_Rm)):.3e}", flush=True)

fold_rows, spec_rows = [], []
F0 = to_coef(Xp, _B, _Rm, 0.0) @ _B.T
for lam_s in LAMS:
    COEF = {L_BASIS: to_coef(Xp, _B, _Rm, lam_s)}
    fitted = COEF[L_BASIS] @ _B.T
    spec_rows.append(dict(lam_s=lam_s,
                          fit_rel_diff_vs0=float(np.linalg.norm(fitted - F0) / np.linalg.norm(F0)),
                          resid_over_signal=float(np.linalg.norm(Xp - fitted) / np.linalg.norm(Xp))))
    print(f"λs={lam_s:g}：拟合光谱与 λs=0 的相对差 {spec_rows[-1]['fit_rel_diff_vs0']:.2%}，"
          f"残差/光谱 {spec_rows[-1]['resid_over_signal']:.2%}", flush=True)
    for m in ('PFLR', 'FPCR'):
        fit = mk(m)[1]
        t0 = time.time()
        for sd in SEEDS:
            outer = list(KFold(n_splits=K_OUTER, shuffle=True, random_state=sd).split(np.arange(len(y))))
            for fold, (itr, ite) in enumerate(outer):
                inner = list(KFold(n_splits=K_INNER, shuffle=True, random_state=sd).split(itr))
                best, bc = np.inf, GRIDS[m][0]
                for cfg in GRIDS[m]:
                    se = [float(mean_squared_error(y[itr[iv]], fit(cfg, itr[ia], itr[iv], sd))) for ia, iv in inner]
                    e = float(np.mean(np.sqrt(se)))
                    if e < best:
                        best, bc = e, cfg
                p = fit(bc, itr, ite, sd)
                fold_rows.append(dict(lam_s=lam_s, method=LABEL[m], seed=sd, fold=fold, picked=str(bc),
                                      R2=float(r2_score(y[ite], p)), RMSE=float(np.sqrt(mean_squared_error(y[ite], p)))))
        g = pd.DataFrame([r for r in fold_rows if r['lam_s'] == lam_s and r['method'] == LABEL[m]])
        print(f"  λs={lam_s:g} {LABEL[m]}：25 折 R²={g.R2.mean():.4f}±{g.R2.std(ddof=1):.4f}  用时 {time.time() - t0:.0f}s",
              flush=True)

folds = pd.DataFrame(fold_rows)

# 复现校验：λs=10⁻⁶ 必须逐折复现表 1 的函数型结果
for lab in ('FPLR', 'FPCR'):
    ref = pd.read_csv(f'04outputs/45rebuttal_ncv_bspline_{lab}.csv')
    mine = folds[(folds.lam_s == REF_LAM) & (folds.method == lab)]
    mg = mine.merge(ref[['seed', 'fold', 'R2']], on=['seed', 'fold'], suffixes=('', '_t1'))
    assert len(mg) == 25, (lab, len(mg))
    gap = float((mg.R2 - mg.R2_t1).abs().max())
    assert gap < 1e-9, f'{lab}：λs=10⁻⁶ 未复现表 1（逐折最大差 {gap:.2e}），协议与 45 脚本不一致'
    print(f'复现校验通过：{lab} 在 λs=10⁻⁶ 下 25 折 R² 与 45 产物逐折最大差 {gap:.1e}', flush=True)

base = folds[folds.lam_s == REF_LAM].set_index(['method', 'seed', 'fold']).R2
rows = []
for (lam_s, lab), g in folds.groupby(['lam_s', 'method']):
    d = g.set_index(['method', 'seed', 'fold']).R2 - base.loc[g.set_index(['method', 'seed', 'fold']).index]
    sp = next(s for s in spec_rows if s['lam_s'] == lam_s)
    rows.append(dict(lam_s=lam_s, method=lab, n_folds=len(g), R2_mean=float(g.R2.mean()), R2_sd=float(g.R2.std(ddof=1)),
                     diff_vs_ref_mean=float(d.mean()), diff_vs_ref_min=float(d.min()), diff_vs_ref_max=float(d.max()),
                     fit_rel_diff_vs0=sp['fit_rel_diff_vs0'], resid_over_signal=sp['resid_over_signal']))
summary = pd.DataFrame(rows)

for df, path in ((folds, '04outputs/51rebuttal_lams_folds.csv'), (summary, '04outputs/51rebuttal_lams_summary.csv')):
    tmp = f'{path}.tmp'
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, path)
print('\n' + summary.to_string(index=False), flush=True)
print('已写 04outputs/51rebuttal_lams_folds.csv、04outputs/51rebuttal_lams_summary.csv', flush=True)
