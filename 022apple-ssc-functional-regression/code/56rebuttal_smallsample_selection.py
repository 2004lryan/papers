# -*- coding: utf-8 -*-
"""
56rebuttal_smallsample_selection.py: 小样本下函数型模型选参的稳健化对策试验（预注册 B10.8）

为什么要有这个脚本：小样本试验（21 脚本、稿件 3.2 节）只诊断出「按 4 折交叉验证误差最小选参，
函数型模型在少数抽样中严重失效」，对策留作待研究的问题。本脚本按 B10.8 登记的规则检验两种标准对策：
一倍标准误准则（1-SE）与重复交叉验证。对策是在失效已被观察到之后设定的，所以检验用新的 100 次抽样
（seed 20~119，未在任何已有计算中用过）；原 20 次抽样（seed 0~19）只并列对照。

选参臂（FPCR、FPLR 各算一遍；PLSR、SVR 未出现失效，只按原做法并列作参照）：
  fixed   FPCR 取 K=min(10, max(2, N−2))，FPLR 取 λβ=1e-3
  cv      KFold(4, shuffle=True, random_state=0) 验证均方误差的均值最小（原做法，同 08 脚本的 cv_pick）
  onese   同一 4 折；均值最小的候选为 g*，SE=sd(ddof=1)/√4；在均值 ≤ m(g*)+SE(g*) 的候选中取最简约者
          （FPCR 取 K 最小者，FPLR 取 λβ 最大者）
  repcv   4 折重复 10 次（random_state=0~9），40 个折的均方误差均值最小

复现门：先在原 20 次抽样上重算 fixed、cv 两臂，与 04outputs/21smallsample_repeats.csv 逐次比对
（|ΔR²| ≤ 1e-9），任一不符即中止，不写任何产物。函数实现取自 02code/08fda_smallsample.py 原文。

运行方式:
    cd <项目根目录>
    python3 02code/56rebuttal_smallsample_selection.py > 05logs/56rebuttal_smallsample_selection_<时间戳>.log 2>&1

输出文件:
    04outputs/56rebuttal_selection_draws.csv    — 逐次抽样：seed_set, N, method, arm, seed, R2, selected
    04outputs/56rebuttal_selection_summary.csv  — 每个（seed_set, N, method, arm）的次数、中位数、均值、最低值、R²<0 次数
    05logs/56rebuttal_smallsample_selection_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

只读 05data。
"""
import os
import time
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

T0 = time.time()
NS: dict = {}
with open('02code/08fda_smallsample.py', encoding='utf-8') as _src:
    exec(_src.read().split('# ---------- 实验一')[0], NS)          # 数据读取、预处理、基函数与三个估计器，取原文
Xs, y, t = NS['Xs'], NS['y'], NS['t']
fpcr, pflr, plsr = NS['fpcr'], NS['pflr'], NS['plsr']
Bm = NS['bspline_basis'](t, 40)
J, R = NS['gram_and_pen'](Bm, t)
C = NS['to_coef'](Xs, Bm, R)

N_GRID = [20, 30, 50, 80, 120, 200, 400]
K_ALL = [2, 5, 10, 15, 20, 30]
LAM_ALL = [float(v) for v in np.logspace(-8, 2, 6)]
SVR_GRID = [(c, g) for c in [1, 10, 100, 1000] for g in ['scale', 1e-3, 1e-4]]
N_REP = 10
SEED_SETS = {'原20次': range(0, 20), '新100次': range(20, 120)}


def fold_mse(kind, Ctr, ytr, g, state):
    """一个候选在一次 4 折划分上的各折验证均方误差。"""
    out = []
    for tr, va in KFold(4, shuffle=True, random_state=state).split(Ctr):
        p = pflr(Ctr[tr], ytr[tr], Ctr[va], J, R, g) if kind == 'pflr' else fpcr(Ctr[tr], ytr[tr], Ctr[va], J, g)
        out.append(mean_squared_error(ytr[va], p))
    return np.asarray(out)


def pick(kind, Ctr, ytr, grid):
    """返回 cv、onese、repcv 三个臂各自选中的候选。grid 按「简约到复杂」之外的原顺序给出，简约方向在函数内处理。"""
    e0 = np.array([fold_mse(kind, Ctr, ytr, g, 0) for g in grid])          # 候选 × 4 折
    m = e0.mean(1)
    i_cv = int(np.argmin(m))                                               # 并列时取先出现者，同 cv_pick 的严格小于
    se = e0[i_cv].std(ddof=1) / np.sqrt(e0.shape[1])
    ok = [i for i in range(len(grid)) if m[i] <= m[i_cv] + se]
    i_1se = min(ok, key=lambda i: grid[i]) if kind == 'fpcr' else max(ok, key=lambda i: grid[i])
    er = np.array([np.concatenate([fold_mse(kind, Ctr, ytr, g, s) for s in range(N_REP)]) for g in grid])
    i_rep = int(np.argmin(er.mean(1)))
    return grid[i_cv], grid[i_1se], grid[i_rep]


def svr_fit(Xtr, ytr, Xte, cfg):
    return SVR(C=cfg[0], gamma=cfg[1], epsilon=0.1).fit(Xtr, ytr).predict(Xte)


def svr_cv(Xtr, ytr, Xte):
    nf = min(4, max(2, len(ytr) // 2))
    best, bc = np.inf, SVR_GRID[0]
    for cfg in SVR_GRID:
        e = [mean_squared_error(ytr[va], svr_fit(Xtr[tr], ytr[tr], Xtr[va], cfg))
             for tr, va in KFold(nf, shuffle=True, random_state=0).split(Xtr)]
        if np.mean(e) < best:
            best, bc = float(np.mean(e)), cfg
    return svr_fit(Xtr, ytr, Xte, bc)


rows = []
for set_name, seeds in SEED_SETS.items():
    for N in N_GRID:
        for sd in seeds:
            idx = np.random.RandomState(sd).permutation(len(y))
            itr, ite = idx[:N], idx[N:N + 400]
            yt = y[ite]
            rows.append((set_name, N, 'PLSR', 'cv', sd, r2_score(yt, plsr(Xs[itr], y[itr], Xs[ite])), ''))
            rows.append((set_name, N, 'SVR', 'fixed', sd, r2_score(yt, svr_fit(Xs[itr], y[itr], Xs[ite], (100, 'scale'))), ''))
            rows.append((set_name, N, 'SVR', 'cv', sd, r2_score(yt, svr_cv(Xs[itr], y[itr], Xs[ite])), ''))
            kf = min(10, max(2, N - 2))
            kg = [k for k in K_ALL if k <= max(2, int(N * 0.75) - 2)] or [2]
            picks = {'FPCR': dict(zip(('fixed', 'cv', 'onese', 'repcv'), (kf, *pick('fpcr', C[itr], y[itr], kg)), strict=True)),
                     'FPLR': dict(zip(('fixed', 'cv', 'onese', 'repcv'), (1e-3, *pick('pflr', C[itr], y[itr], LAM_ALL)), strict=True))}
            for meth, d in picks.items():
                for arm, g in d.items():
                    p = fpcr(C[itr], y[itr], C[ite], J, g) if meth == 'FPCR' else pflr(C[itr], y[itr], C[ite], J, R, g)
                    rows.append((set_name, N, meth, arm, sd, r2_score(yt, p), g))
        print(f"{set_name} N={N} 完成，累计 {time.time() - T0:.0f} s", flush=True)

df = pd.DataFrame(rows, columns=['seed_set', 'N', 'method', 'arm', 'seed', 'R2', 'selected'])

# ---- 复现门：原 20 次抽样的 fixed、cv 两臂须与 21 脚本的逐次记录一致 ----
ref = pd.read_csv('04outputs/21smallsample_repeats.csv').rename(columns={'strategy': 'arm'})
ref['method'] = ref['method'].replace({'PFLR': 'FPLR'})
ref.loc[ref.method == 'PLSR', 'arm'] = 'cv'                                  # 21 脚本把 PLSR 记在 fixed 名下，其潜变量数实由 4 折选取
mine = df[(df.seed_set == '原20次') & df.arm.isin(['fixed', 'cv'])]
chk = mine.merge(ref, on=['N', 'method', 'arm', 'seed'], suffixes=('', '_ref'), how='outer', indicator=True)
assert (chk['_merge'] == 'both').all(), chk['_merge'].value_counts().to_dict()
gap = float((chk.R2 - chk.R2_ref).abs().max())
print(f"\n复现门：与 21smallsample_repeats.csv 比对 {len(chk)} 条，最大 |ΔR²| = {gap:.2e}")
assert gap <= 1e-9, '原 20 次抽样复现不了 21 脚本的结果，中止'

summ = (df.groupby(['seed_set', 'N', 'method', 'arm'], sort=False).R2
        .agg(n='size', median='median', mean='mean', min='min', n_neg=lambda s: int((s < 0).sum())).reset_index())
df.to_csv('04outputs/56rebuttal_selection_draws.csv', index=False, encoding='utf-8-sig')
summ.to_csv('04outputs/56rebuttal_selection_summary.csv', index=False, encoding='utf-8-sig')

pd.set_option('display.width', 200)
for set_name in SEED_SETS:
    for meth in ('FPCR', 'FPLR'):
        q = summ[(summ.seed_set == set_name) & (summ.method == meth)]
        print(f"\n{set_name} · {meth}：R²<0 次数 / 中位数")
        print(q.pivot(index='N', columns='arm', values='n_neg')[['fixed', 'cv', 'onese', 'repcv']].to_string())
        print(q.pivot(index='N', columns='arm', values='median')[['fixed', 'cv', 'onese', 'repcv']].round(3).to_string())

# ---- B10.8 的判读（只对新 100 次抽样；FPCR 与 FPLR 分别判） ----
new = summ[summ.seed_set == '新100次'].set_index(['method', 'arm', 'N'])
print("\nB10.8 判读（新 100 次抽样，N=50、80、120）：")
for meth in ('FPCR', 'FPLR'):
    neg_cv = [int(new.loc[(meth, 'cv', n), 'n_neg']) for n in (50, 80, 120)]
    print(f"  (i) {meth} 的 CV 臂 R²<0 次数：{neg_cv}")
    for arm in ('onese', 'repcv'):
        neg = [int(new.loc[(meth, arm, n), 'n_neg']) for n in (50, 80, 120)]
        dmed = [float(new.loc[(meth, arm, n), 'median'] - new.loc[(meth, 'cv', n), 'median']) for n in (50, 80, 120)]
        ok = all(v == 0 for v in neg) and all(d >= -0.02 for d in dmed)
        print(f"  (ii) {meth} {arm}: R²<0 次数 {neg}，中位数差 {[round(d, 3) for d in dmed]} → {'有效' if ok else '无效'}")
print(f"\n总耗时 {time.time() - T0:.0f} s")
