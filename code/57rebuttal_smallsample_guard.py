# -*- coding: utf-8 -*-
"""
57rebuttal_smallsample_guard.py: 小样本失效的逐样品剖析，与按杠杆值回退、按参考值范围截断两种预测端对策的确认试验（预注册 B10.9）

为什么要有这个脚本：B10.8 的两种选参端对策（一倍标准误准则、重复交叉验证）在 100 次抽样上都没有消除函数型模型的
R²<0。对那 100 次抽样作逐样品剖析后看到（清洗前的数据），失效抽样的平方误差集中在个别测试光谱上，且集中在同样 6 条光谱；
交叉验证按训练集里的典型样品选中弱正则化的配置，训练集未覆盖的异常光谱在低方差方向上得分极大，预测外推失控。
据此设两种只用待测光谱本身、不用其参考值的预测端对策，并在再取的 100 次抽样（seed 120~219，未在任何已有计算中用过）
上作确认；B10.8 的 100 次抽样（seed 20~119）上的结果只作探查并列。
协议 B10.10 起在清洗后的数据上运行：光谱名单不写死，按 B10.9 的规则（B10.8 的 100 次抽样、N=50/80/120 的 cv 臂
R²<0 抽样中，误差最大 3 个里出现不少于 5 次）先在当前数据上导出，序号为 08 脚本读入顺序（自 0 起），名单可以为空；
清洗前按同一规则导出的是 6 条（HOT_PRE_CLEAN，清洗前的序号），只作记录。

臂（每个 N、每次抽样）：
  fixed   FPCR 取 K=min(10, max(2, N−2))，FPLR 取 λβ=1e-3（同 21、56 脚本，只作参照）
  cv      4 折交叉验证误差均值最小（原做法；FPCR、FPLR、PLSR、SVR）
  lev     在 cv 选中的配置上，对每条待测光谱算杠杆值；超过训练集平均杠杆值的 3 倍时，沿候选网格向更强的正则化
          逐级回退（FPCR 取更小的 K，FPLR 取更大的 λβ，PLSR 取更少的潜变量），直到不超过该级的 3 倍平均杠杆值，
          或到达最强正则化的候选（FPCR、FPLR、PLSR）
  clip    cv 选中配置的预测值截断到训练集参考值的最小值与最大值之间（FPCR、FPLR、PLSR、SVR）

杠杆值：FPCR 为 1/n + Σ_k s_k²/Σ_i s_ik²（训练集平均为 (K+1)/n）；PLSR 为 1/n + t'(T'T)⁻¹t（平均 (A+1)/n）；
FPLR 为 z'(Z'Z+λβ·c·R)⁻¹z，与训练集各样品同式计算的均值相比。

复现门：先在 seed 20~119 上重算 fixed、cv 两臂，与 04outputs/56rebuttal_selection_draws.csv 逐次比对
（|ΔR²| ≤ 1e-9），任一不符即中止，不写任何产物。函数实现取自 02code/08fda_smallsample.py 原文。

运行方式:
    cd <项目根目录>
    python3 02code/57rebuttal_smallsample_guard.py > 05logs/57rebuttal_smallsample_guard_<时间戳>.log 2>&1

输出文件:
    04outputs/57rebuttal_guard_draws.csv     — 逐次抽样：seed_set, N, method, arm, seed, R2, selected, frac_backed, hot_in_test, hot_backed
    04outputs/57rebuttal_guard_summary.csv   — 每个（seed_set, N, method, arm）的次数、中位数、均值、最低值、R²<0 次数、回退占比中位数
    04outputs/57rebuttal_guard_anatomy.csv   — cv 臂每个 R²<0 的抽样：误差最大的 1、3、5 个测试样品占平方误差总和的份额、去掉最大 5 个后的 R²
    04outputs/57rebuttal_guard_spectra.csv   — 名单上各条光谱的原始光谱特征、在失效抽样「误差最大 3 个」中出现的次数、落入测试集与被回退的次数
    04outputs/57rebuttal_guard_dataset.csv   — 数据集层面的事实：逐点相同的原始光谱组、含检测器上限读数的光谱
    04outputs/57rebuttal_guard_mainfold.csv  — 主分析（47 正式结果）中名单上光谱的误差份额，及评价时剔除它们后的 25 折 R² 均值
    05logs/57rebuttal_smallsample_guard_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

只读 05data。
"""
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

N_GRID = [20, 30, 50, 80, 120, 200, 400]
N_READ = (50, 80, 120)
K_ALL = [2, 5, 10, 15, 20, 30]
LAM_ALL = [float(v) for v in np.logspace(-8, 2, 6)]
SVR_GRID = [(c, g) for c in [1, 10, 100, 1000] for g in ['scale', 1e-3, 1e-4]]
LEV_MULT = 3.0
ADC_MAX = 65535.0
HOT_MIN = 5                                  # B10.9 的名单规则：误差最大 3 个中出现不少于 5 次
HOT_PRE_CLEAN = (71, 115, 125, 126, 131, 223)   # 清洗前（1 300 个样品的读入顺序）按同一规则导出的 6 条，只作记录
SEED_SETS = {'B10.8的100次': range(20, 120), '再取100次': range(120, 220)}
# 名单、剖析与主分析三个产物的列：名单或失效抽样为空时照样写出表头，下游（58、ms_numbers）按列读取
SPECTRA_COLS = ['idx', 'year', 'origin', 'SSC', 'raw_max', 'raw_max_quantile', 'raw_mean', 'raw_mean_quantile',
                'n_points_at_adc_max', 'identical_to']
ANATOMY_COLS = ['seed_set', 'N', 'method', 'seed', 'selected', 'R2', 'share_top1', 'share_top3', 'share_top5', 'R2_drop5',
                'n_abs_gt5', 'top3_idx', 'top5_idx']
MAINFOLD_COLS = ['method', 'n_folds', 'R2_all', 'R2_excl_hot', 'delta', 'sse_share_hot', 'max_abs_err_hot', 'max_abs_err_all']
USED, FRESH = 'B10.8的100次', '再取100次'


def fpcr_lev(Ctr, ytr, Cte, M, cand, k_sel, mult=LEV_MULT):
    """FPCR 的杠杆回退。cand 为候选 K，k_sel 为 cv 选中的 K。返回（预测，各待测光谱实际所用的 K）。"""
    n = len(ytr)
    A, At = Ctr @ M, Cte @ M
    mu = A.mean(0)
    _, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
    T, Tt = (A - mu) @ Vt.T, (At - mu) @ Vt.T
    chain = sorted((k for k in cand if k <= k_sel), reverse=True)            # 由弱正则化到强正则化
    out, used, done = np.empty(len(Cte)), np.empty(len(Cte)), np.zeros(len(Cte), bool)
    for k in chain:
        coef = np.linalg.lstsq(np.c_[np.ones(n), T[:, :k]], ytr, rcond=None)[0]
        h = 1.0 / n + ((Tt[:, :k] ** 2) / (S[:k] ** 2)).sum(1)
        take = ~done & ((h <= mult * (k + 1) / n) | (k == chain[-1]))
        out[take] = (np.c_[np.ones(len(Tt)), Tt[:, :k]] @ coef)[take]
        used[take] = k
        done |= take
    return out, used


def fplr_lev(Ctr, ytr, Cte, J, R, cand, lam_sel, mult=LEV_MULT):
    """FPLR 的杠杆回退。返回（预测，各待测光谱实际所用的 λβ）。估计式同 08 脚本的 pflr。"""
    Z, Zt, ym = Ctr @ J, Cte @ J, ytr.mean()
    ZtZ = Z.T @ Z
    sc = np.trace(ZtZ) / max(np.trace(R), 1e-30)
    chain = sorted(v for v in cand if v >= lam_sel * (1 - 1e-9))              # 由弱正则化到强正则化
    out, used, done = np.empty(len(Cte)), np.empty(len(Cte)), np.zeros(len(Cte), bool)
    for lam in chain:
        A = ZtZ + lam * sc * R + 1e-12 * np.eye(len(R))
        b = np.linalg.solve(A, Z.T @ (ytr - ym))
        h_tr = np.einsum('ij,ij->i', Z, np.linalg.solve(A, Z.T).T)
        h_te = np.einsum('ij,ij->i', Zt, np.linalg.solve(A, Zt.T).T)
        take = ~done & ((h_te <= mult * h_tr.mean()) | (lam == chain[-1]))
        out[take] = (Zt @ b + ym)[take]
        used[take] = lam
        done |= take
    return out, used


def plsr_pick(Xtr, ytr):
    """PLSR 的潜变量数，选法同 08 脚本的 plsr（4 折、步长 2）。"""
    bn, bs = 2, 1e18
    hi = max(2, min(25, int(len(ytr) * 0.75) - 1, Xtr.shape[1]))
    for nc in range(2, hi + 1, 2):
        e = [mean_squared_error(ytr[va], PLSRegression(n_components=nc).fit(Xtr[tr], ytr[tr]).predict(Xtr[va]).ravel())
             for tr, va in KFold(4, shuffle=True, random_state=0).split(Xtr)]
        if np.mean(e) < bs:
            bs, bn = float(np.mean(e)), nc
    return min(bn, len(ytr) - 1, Xtr.shape[1])


def plsr_lev(Xtr, ytr, Xte, nc_sel, mult=LEV_MULT):
    """PLSR 的杠杆回退。返回（cv 选中配置的预测，回退后的预测，各待测光谱实际所用的潜变量数）。"""
    n = len(ytr)
    chain = sorted({nc_sel, *range(2, nc_sel + 1, 2)}, reverse=True)
    out, used, done = np.empty(len(Xte)), np.empty(len(Xte)), np.zeros(len(Xte), bool)
    p_sel = None
    for nc in chain:
        m = PLSRegression(n_components=nc).fit(Xtr, ytr)
        p = m.predict(Xte).ravel()
        if p_sel is None:
            p_sel = p
        T, Tt = m.x_scores_, m.transform(Xte)
        h = 1.0 / n + np.einsum('ij,ij->i', Tt, np.linalg.solve(T.T @ T, Tt.T).T)
        take = ~done & ((h <= mult * (nc + 1) / n) | (nc == chain[-1]))
        out[take] = p[take]
        used[take] = nc
        done |= take
        if done.all():
            break
    return p_sel, out, used


def svr_cv(Xtr, ytr, Xte):
    """SVR 的 4 折选参与预测，同 56 脚本。"""
    nf = min(4, max(2, len(ytr) // 2))
    best, bc = np.inf, SVR_GRID[0]
    for cfg in SVR_GRID:
        e = [mean_squared_error(ytr[va], SVR(C=cfg[0], gamma=cfg[1], epsilon=0.1).fit(Xtr[tr], ytr[tr]).predict(Xtr[va]))
             for tr, va in KFold(nf, shuffle=True, random_state=0).split(Xtr)]
        if np.mean(e) < best:
            best, bc = float(np.mean(e)), cfg
    return SVR(C=bc[0], gamma=bc[1], epsilon=0.1).fit(Xtr, ytr).predict(Xte)


def anatomy(y_true, pred, ids):
    """一次抽样的逐样品剖析：误差最大的 1、3、5 个测试样品占平方误差总和的份额，去掉最大 5 个后的 R²。"""
    e2 = (y_true - pred) ** 2
    o = np.argsort(e2)[::-1]
    sst = ((y_true - y_true.mean()) ** 2).sum()
    return {'share_top1': e2[o[0]] / e2.sum(), 'share_top3': e2[o[:3]].sum() / e2.sum(), 'share_top5': e2[o[:5]].sum() / e2.sum(),
            'R2_drop5': 1 - e2[o[5:]].sum() / sst, 'n_abs_gt5': int((np.abs(y_true - pred) > 5).sum()),
            'top3_idx': ' '.join(str(int(i)) for i in ids[o[:3]]), 'top5_idx': ' '.join(str(int(i)) for i in ids[o[:5]])}


def derive_hot(y, C, J, R, cv_pick, fpcr, pflr):
    """B10.9 的名单规则：在 B10.8 的 100 次抽样（seed 20~119）、N=50、80、120 上，FPCR、FPLR 的 cv 臂 R²<0 的抽样里，
    误差最大 3 个测试样品中出现不少于 HOT_MIN 次的光谱（两种方法分别计数，取并集）。抽样、选参与预测与 main() 的
    cv 臂逐项相同，main() 算完全部抽样后再按同一规则导出一次，两者须相同。"""
    c: dict[str, Counter] = {'FPCR': Counter(), 'FPLR': Counter()}
    for N in N_READ:
        for sd in SEED_SETS[USED]:
            idx = np.random.RandomState(sd).permutation(len(y))
            itr, ite = idx[:N], idx[N:N + 400]
            ytr, yt = y[itr], y[ite]
            kg = [k for k in K_ALL if k <= max(2, int(N * 0.75) - 2)] or [2]
            k_cv = cv_pick(C[itr], ytr, J, R, 'fpcr', kg)
            lam_cv = cv_pick(C[itr], ytr, J, R, 'pflr', LAM_ALL)
            for meth, p0 in (('FPCR', fpcr(C[itr], ytr, C[ite], J, k_cv)), ('FPLR', pflr(C[itr], ytr, C[ite], J, R, lam_cv))):
                if r2_score(yt, p0) < 0:
                    c[meth].update(int(i) for i in anatomy(yt, p0, ite)['top3_idx'].split())
    return tuple(sorted({i for m in c for i, k in c[m].items() if k >= HOT_MIN}))


def main():
    t0 = time.time()
    ns: dict = {}
    with open('02code/08fda_smallsample.py', encoding='utf-8') as src:
        exec(src.read().split('# ---------- 实验一')[0], ns)       # 数据读取、预处理、基函数与估计器，取原文
    X, Xs, y, z, t = ns['X'], ns['Xs'], ns['y'], ns['z'], ns['t']
    year = np.r_[np.full(len(ns['y1']), 2018), np.full(len(ns['y2']), 2019)]
    fpcr, pflr, cv_pick = ns['fpcr'], ns['pflr'], ns['cv_pick']
    Bm = ns['bspline_basis'](t, 40)
    J, R = ns['gram_and_pen'](Bm, t)
    C = ns['to_coef'](Xs, Bm, R)
    M = np.linalg.cholesky(J + 1e-10 * np.eye(len(J)))
    HOT = derive_hot(y, C, J, R, cv_pick, fpcr, pflr)
    print(f"按 B10.9 的规则在当前数据（{len(y)} 个样品）上导出的光谱名单：{list(HOT)}（清洗前为 {list(HOT_PRE_CLEAN)}，序号不可直接对照）", flush=True)
    sys.path.insert(0, '02code')
    from results47 import resolve
    if HOT and not resolve().final:
        # 名单非空时要对主分析作误差份额剖析（只读 47 正式结果）；先查，免得算完全部抽样才发现 47 未就绪
        raise SystemExit('名单非空，主分析剖析须读 47 正式结果，而 47 尚无有效的正式结果：先运行 47 正式版')

    only = os.environ.get('GUARD_SEED_SETS')                     # 只用于登记前在已用过的抽样上试跑（不写产物），正式运行不设
    sets = {k: v for k, v in SEED_SETS.items() if not only or k in only.split(',')}
    rows, arows = [], []
    for set_name, seeds in sets.items():
        for N in N_GRID:
            for sd in seeds:
                idx = np.random.RandomState(sd).permutation(len(y))
                itr, ite = idx[:N], idx[N:N + 400]
                ytr, yt = y[itr], y[ite]
                lo, hi = ytr.min(), ytr.max()
                hot = np.isin(ite, HOT)

                def add(meth, arm, pred, sel='', backed=None, _a=(set_name, N, sd, yt, hot)):
                    fb = float(backed.mean()) if backed is not None else np.nan
                    hb = int((backed & _a[4]).sum()) if backed is not None else -1
                    rows.append((_a[0], _a[1], meth, arm, _a[2], r2_score(_a[3], pred), sel, fb, int(_a[4].sum()), hb))

                nc = plsr_pick(Xs[itr], ytr)
                p_cv, p_lev, used = plsr_lev(Xs[itr], ytr, Xs[ite], nc)
                add('PLSR', 'cv', p_cv, nc)
                add('PLSR', 'lev', p_lev, nc, used < nc)
                add('PLSR', 'clip', np.clip(p_cv, lo, hi), nc)
                p_svr = svr_cv(Xs[itr], ytr, Xs[ite])
                add('SVR', 'cv', p_svr)
                add('SVR', 'clip', np.clip(p_svr, lo, hi))

                kg = [k for k in K_ALL if k <= max(2, int(N * 0.75) - 2)] or [2]
                k_cv = cv_pick(C[itr], ytr, J, R, 'fpcr', kg)
                lam_cv = cv_pick(C[itr], ytr, J, R, 'pflr', LAM_ALL)
                kf = min(10, max(2, N - 2))
                add('FPCR', 'fixed', fpcr(C[itr], ytr, C[ite], J, kf), kf)
                add('FPLR', 'fixed', pflr(C[itr], ytr, C[ite], J, R, 1e-3), 1e-3)
                for meth, sel in (('FPCR', k_cv), ('FPLR', lam_cv)):
                    if meth == 'FPCR':
                        p0 = fpcr(C[itr], ytr, C[ite], J, sel)
                        p1, u = fpcr_lev(C[itr], ytr, C[ite], M, kg, sel)
                        backed = u < sel
                    else:
                        p0 = pflr(C[itr], ytr, C[ite], J, R, sel)
                        p1, u = fplr_lev(C[itr], ytr, C[ite], J, R, LAM_ALL, sel)
                        backed = u > sel * (1 + 1e-9)
                    assert np.allclose(p0[~backed], p1[~backed], rtol=1e-6, atol=1e-6), (meth, N, sd)   # 未回退的光谱两臂预测须一致
                    add(meth, 'cv', p0, sel)
                    add(meth, 'lev', p1, sel, backed)
                    add(meth, 'clip', np.clip(p0, lo, hi), sel)
                    if r2_score(yt, p0) < 0:
                        arows.append({'seed_set': set_name, 'N': N, 'method': meth, 'seed': sd, 'selected': sel,
                                      'R2': r2_score(yt, p0), **anatomy(yt, p0, ite)})
            print(f"{set_name} N={N} 完成，累计 {time.time() - t0:.0f} s", flush=True)

    df = pd.DataFrame(rows, columns=['seed_set', 'N', 'method', 'arm', 'seed', 'R2', 'selected', 'frac_backed', 'hot_in_test', 'hot_backed'])
    an = pd.DataFrame(arows, columns=ANATOMY_COLS)

    # ---- 复现门：seed 20~119 的 fixed、cv 两臂须与 56 脚本的逐次记录一致 ----
    if USED in sets:
        ref = pd.read_csv('04outputs/56rebuttal_selection_draws.csv')
        ref = ref[(ref.seed_set == '新100次') & ref.arm.isin(['fixed', 'cv']) & ~((ref.method == 'SVR') & (ref.arm == 'fixed'))]
        mine = df[(df.seed_set == USED) & df.arm.isin(['fixed', 'cv'])]
        chk = mine.merge(ref[['N', 'method', 'arm', 'seed', 'R2']], on=['N', 'method', 'arm', 'seed'], suffixes=('', '_ref'), how='outer', indicator=True)
        assert (chk['_merge'] == 'both').all(), chk['_merge'].value_counts().to_dict()
        gap = float((chk.R2 - chk.R2_ref).abs().max())
        print(f"\n复现门：与 56rebuttal_selection_draws.csv 比对 {len(chk)} 条，最大 |ΔR²| = {gap:.2e}")
        assert gap <= 1e-9, 'seed 20~119 复现不了 56 脚本的结果，中止'

    summ = (df.groupby(['seed_set', 'N', 'method', 'arm'], sort=False)
            .agg(n=('R2', 'size'), median=('R2', 'median'), mean=('R2', 'mean'), min=('R2', 'min'),
                 n_neg=('R2', lambda s: int((s < 0).sum())), frac_backed_median=('frac_backed', 'median')).reset_index())

    # ---- 光谱名单：全部抽样算完后按同一规则（B10.8 的 100 次抽样、N=50/80/120 的失效抽样里，误差最大 3 个中出现 ≥5 次）再导出一次，须与先行导出的相同 ----
    def top3_counts(set_name):
        c: dict[str, Counter] = {'FPCR': Counter(), 'FPLR': Counter()}
        if len(an):
            q = an[(an.seed_set == set_name) & an.N.isin(N_READ)]
            for meth, s in zip(q.method, q.top3_idx, strict=True):
                c[meth].update(int(i) for i in s.split())
        return c
    if USED in sets:
        cu = top3_counts(USED)
        derived = sorted({i for m in cu for i, k in cu[m].items() if k >= 5})
        print(f"\nB10.8 的 100 次抽样上按规则导出的光谱：{derived}")
        assert tuple(derived) == HOT, (derived, HOT)

    mx, me = X.max(1), X.mean(1)
    groups = defaultdict(list)
    for i in range(len(X)):
        groups[X[i].tobytes()].append(i)
    dup = [v for v in groups.values() if len(v) > 1]
    partner = {i: [j for j in v if j != i] for v in dup for i in v}
    srows = []
    for i in HOT:
        r = {'idx': i, 'year': int(year[i]), 'origin': int(z[i]), 'SSC': float(y[i]), 'raw_max': float(mx[i]),
             'raw_max_quantile': float(np.mean(mx <= mx[i])), 'raw_mean': float(me[i]), 'raw_mean_quantile': float(np.mean(me <= me[i])),
             'n_points_at_adc_max': int((X[i] >= ADC_MAX).sum()), 'identical_to': ' '.join(map(str, partner.get(i, [])))}
        for set_name in sets:
            c = top3_counts(set_name)
            for meth in ('FPCR', 'FPLR'):
                r[f'top3_{meth}_{set_name}'] = c[meth][i]
        srows.append(r)
    spectra = pd.DataFrame(srows, columns=SPECTRA_COLS + [f'top3_{m}_{s}' for s in sets for m in ('FPCR', 'FPLR')])
    dsum = pd.DataFrame([
        ('n_samples', len(y)),
        ('identical_spectrum_groups', len(dup)),
        ('samples_in_identical_groups', sum(len(v) for v in dup)),
        ('identical_groups_same_origin_year', int(sum(len({(int(z[i]), int(year[i])) for i in v}) == 1 for v in dup))),
        ('identical_groups_adjacent_rows', int(sum(max(v) - min(v) == len(v) - 1 for v in dup))),
        ('identical_groups_ssc_absdiff_median', float(np.median([max(y[v]) - min(y[v]) for v in dup])) if dup else np.nan),
        ('identical_groups_ssc_absdiff_max', float(max(max(y[v]) - min(y[v]) for v in dup)) if dup else np.nan),
        ('spectra_with_adc_max_reading', int((mx >= ADC_MAX).sum())),
        ('spectra_with_10plus_adc_max_readings', int(((X >= ADC_MAX).sum(1) >= 10).sum())),
    ], columns=['item', 'value'])
    dsum['note'] = ''
    dsum.loc[dsum.item == 'spectra_with_adc_max_reading', 'note'] = ' '.join(
        f'{int(i)}:{int((X[i] >= ADC_MAX).sum())}' for i in np.where(mx >= ADC_MAX)[0])       # 序号:达到上限读数的波长点数
    dsum.loc[dsum.item == 'identical_spectrum_groups', 'note'] = ' '.join('+'.join(map(str, v)) for v in dup)

    # ---- 主分析（47 正式结果）中名单上光谱的误差份额；只描述，不重做检验。名单为空时无从剖析，产物只写表头 ----
    mrows = []
    source = resolve() if HOT else None
    if source is not None:
        assert source.final, '主分析稳健性只读 47 正式结果'
    paths = [str(p) for p in source.marker.get('inputs45', [])] if source is not None else []
    for meth in (('SVR', 'FPLR', 'FPCR', 'PLSR', '1D-CNN', 'RF') if HOT else ()):
        stem = f"45rebuttal_ncv_{'bspline' if meth.startswith('FP') else 'none'}_{meth.replace('-', '')}_pred.csv"
        hit = [p for p in paths if Path(p).name == stem]
        assert len(hit) == 1, (meth, hit)
        d = pd.read_csv(hit[0])
        assert np.abs(d.y.to_numpy() - y[d.idx.to_numpy()]).max() < 1e-9, f'{meth}: 45 的样品序号与 08 脚本的读入顺序不一致'
        d['e2'] = (d.pred - d.y) ** 2

        def fold_r2(g):
            return 1 - g.e2.sum() / ((g.y - g.y.mean()) ** 2).sum()
        keep = d[~d.idx.isin(HOT)]
        r_all = float(np.mean([fold_r2(g) for _, g in d.groupby(['seed', 'fold'])]))
        r_exc = float(np.mean([fold_r2(g) for _, g in keep.groupby(['seed', 'fold'])]))
        mrows.append({'method': meth, 'n_folds': d.groupby(['seed', 'fold']).ngroups, 'R2_all': r_all, 'R2_excl_hot': r_exc,
                      'delta': r_exc - r_all, 'sse_share_hot': float(d[d.idx.isin(HOT)].e2.sum() / d.e2.sum()),
                      'max_abs_err_hot': float(np.sqrt(d[d.idx.isin(HOT)].e2.max())), 'max_abs_err_all': float(np.sqrt(d.e2.max()))})
    mainfold = pd.DataFrame(mrows, columns=MAINFOLD_COLS)
    if not HOT:
        print('名单为空：主分析的误差份额剖析无从进行，57rebuttal_guard_mainfold.csv 只写表头', flush=True)

    if only:
        print('试跑（设了 GUARD_SEED_SETS）：不写产物')
    else:
        df.to_csv('04outputs/57rebuttal_guard_draws.csv', index=False, encoding='utf-8-sig')
        summ.to_csv('04outputs/57rebuttal_guard_summary.csv', index=False, encoding='utf-8-sig')
        an.to_csv('04outputs/57rebuttal_guard_anatomy.csv', index=False, encoding='utf-8-sig')
        spectra.to_csv('04outputs/57rebuttal_guard_spectra.csv', index=False, encoding='utf-8-sig')
        dsum.to_csv('04outputs/57rebuttal_guard_dataset.csv', index=False, encoding='utf-8-sig')
        mainfold.to_csv('04outputs/57rebuttal_guard_mainfold.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 220)
    for set_name in sets:
        for meth, arms in (('FPCR', ['fixed', 'cv', 'lev', 'clip']), ('FPLR', ['fixed', 'cv', 'lev', 'clip']),
                           ('PLSR', ['cv', 'lev', 'clip']), ('SVR', ['cv', 'clip'])):
            q = summ[(summ.seed_set == set_name) & (summ.method == meth)]
            print(f"\n{set_name} · {meth}：R²<0 次数 / 中位数 / lev 臂回退占比中位数")
            print(q.pivot(index='N', columns='arm', values='n_neg')[arms].to_string())
            print(q.pivot(index='N', columns='arm', values='median')[arms].round(3).to_string())
            if 'lev' in arms:
                print(q[q.arm == 'lev'].set_index('N').frac_backed_median.round(3).to_string())
    print(f'\n名单上的 {len(HOT)} 条光谱：'); print(spectra.to_string(index=False))
    print('\n数据集层面：'); print(dsum.to_string(index=False))
    print('\n主分析：'); print(mainfold.round(4).to_string(index=False))

    # ---- B10.9 的判读（只对再取的 100 次抽样） ----
    if FRESH in sets:
        new = summ[summ.seed_set == FRESH].set_index(['method', 'arm', 'N'])
        print(f"\nB10.9 判读（{FRESH}，N=50、80、120）：")
        for meth in ('FPCR', 'FPLR'):
            neg_cv = [int(new.loc[(meth, 'cv', n), 'n_neg']) for n in N_READ]
            print(f"  (i) {meth} 的 cv 臂 R²<0 次数：{neg_cv}")
            for arm in ('lev', 'clip'):
                neg = [int(new.loc[(meth, arm, n), 'n_neg']) for n in N_READ]
                dmed = [float(new.loc[(meth, arm, n), 'median'] - new.loc[(meth, 'cv', n), 'median']) for n in N_READ]
                ok = all(v == 0 for v in neg) and all(d >= -0.02 for d in dmed)
                print(f"  (ii) {meth} {arm}: R²<0 次数 {neg}，中位数差 {[round(d, 3) for d in dmed]} → {'有效' if ok else '无效'}")
            a = an[(an.seed_set == FRESH) & (an.method == meth) & an.N.isin(N_READ)] if len(an) else an
            if len(a):
                print(f"  (iii) {meth} 的 cv 臂失效抽样 {len(a)} 个：误差最大 3 个测试样品占平方误差总和的份额中位数 {a.share_top3.median():.3f}；"
                      f"去掉最大 5 个后的 R² 中位数 {a.R2_drop5.median():.3f}、最低 {a.R2_drop5.min():.3f}、<0 的 {int((a.R2_drop5 < 0).sum())} 个")
            h = df[(df.seed_set == FRESH) & (df.method == meth) & (df.arm == 'lev') & df.N.isin(N_READ)]
            print(f"  (iii) {meth}: 名单上的 {len(HOT)} 条光谱落入测试集共 {int(h.hot_in_test.sum())} 条次，其中被回退 {int(h.hot_backed.sum())} 条次")
    print(f"\n总耗时 {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
