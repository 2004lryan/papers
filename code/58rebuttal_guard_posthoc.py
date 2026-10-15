# -*- coding: utf-8 -*-
"""
58rebuttal_guard_posthoc.py: 杠杆回退的事后描述（非确认性）：对其余测试光谱的精度有无影响，结论对倍数取值是否敏感

为什么要有这个脚本：B10.9 的确认试验只回答「按登记的规则（3 倍平均杠杆值）回退后 R²<0 是否消失」。查新评审
（2026-09-21）指出还须交代两件事：回退是否以其余测试光谱的精度为代价，
以及结论是否依赖 3 这个取值。二者都在看过确认结果之后才提出，只作描述，不改 B10.9 的规则、判据与结论。

内容（FPCR、FPLR；两组各 100 次抽样，seed 20~119 与 120~219；N 同 57 脚本）：
  ① 把 57 脚本按 B10.9 的规则导出的光谱名单（04outputs/57rebuttal_guard_spectra.csv 的 idx 列；协议 B10.10 起为清洗后
     数据上的名单，可以为空）从测试集里除去后重算 R²：cv 臂与各回退规则并列（其余光谱上的精度）
  ② 回退规则取平均杠杆值的 2、3、4 倍，以及该级训练集的最大杠杆值，各自的 R²<0 次数、中位数与回退占比
倍数为 3 的结果须与 57 脚本的 lev 臂逐次一致（|ΔR²| ≤ 1e-9），否则中止、不写产物。

运行方式:
    cd <项目根目录>
    python3 02code/58rebuttal_guard_posthoc.py > 05logs/58rebuttal_guard_posthoc_<时间戳>.log 2>&1

输出文件:
    04outputs/58rebuttal_guard_posthoc_draws.csv    — 逐次抽样：seed_set, N, method, seed, rule, R2_all, R2_rest, frac_backed
    04outputs/58rebuttal_guard_posthoc_summary.csv  — 每个（seed_set, N, method, rule）的中位数与 R²<0 次数（全部测试光谱、除去名单上的光谱之后）
    05logs/58rebuttal_guard_posthoc_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

只读 05data。
"""
import importlib.util
import os
import time
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

_spec = importlib.util.spec_from_file_location('guard57', '02code/57rebuttal_smallsample_guard.py')
assert _spec is not None
assert _spec.loader is not None
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

RULES = ('x2', 'x3', 'x4', 'max')


def _ok(rule, h_te, h_mean, h_max):
    return h_te <= h_max if rule == 'max' else h_te <= float(rule[1:]) * h_mean


def fpcr_levels(Ctr, ytr, Cte, M, chain):
    """chain 由弱正则化到强正则化。返回各级的（预测，待测光谱杠杆值，训练集平均杠杆值，训练集最大杠杆值）。"""
    n = len(ytr)
    A, At = Ctr @ M, Cte @ M
    mu = A.mean(0)
    _, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
    T, Tt = (A - mu) @ Vt.T, (At - mu) @ Vt.T
    out = {}
    for k in chain:
        coef = np.linalg.lstsq(np.c_[np.ones(n), T[:, :k]], ytr, rcond=None)[0]
        h_tr = 1.0 / n + ((T[:, :k] ** 2) / (S[:k] ** 2)).sum(1)
        h_te = 1.0 / n + ((Tt[:, :k] ** 2) / (S[:k] ** 2)).sum(1)
        out[k] = (np.c_[np.ones(len(Tt)), Tt[:, :k]] @ coef, h_te, (k + 1) / n, float(h_tr.max()))
    return out


def fplr_levels(Ctr, ytr, Cte, J, R, chain):
    Z, Zt, ym = Ctr @ J, Cte @ J, ytr.mean()
    ZtZ = Z.T @ Z
    sc = np.trace(ZtZ) / max(np.trace(R), 1e-30)
    out = {}
    for lam in chain:
        A = ZtZ + lam * sc * R + 1e-12 * np.eye(len(R))
        b = np.linalg.solve(A, Z.T @ (ytr - ym))
        h_tr = np.einsum('ij,ij->i', Z, np.linalg.solve(A, Z.T).T)
        h_te = np.einsum('ij,ij->i', Zt, np.linalg.solve(A, Zt.T).T)
        out[lam] = (Zt @ b + ym, h_te, float(h_tr.mean()), float(h_tr.max()))
    return out


def backoff(levels, chain, rule):
    """沿 chain 逐级回退；返回（预测，是否被回退）。"""
    m = len(levels[chain[0]][0])
    out, done, backed = np.empty(m), np.zeros(m, bool), np.zeros(m, bool)
    for j, g in enumerate(chain):
        pred, h_te, h_mean, h_max = levels[g]
        take = ~done & (_ok(rule, h_te, h_mean, h_max) | (j == len(chain) - 1))
        out[take] = pred[take]
        backed[take] = j > 0
        done |= take
    return out, backed


def main():
    t0 = time.time()
    ns: dict = {}
    with open('02code/08fda_smallsample.py', encoding='utf-8') as src:
        exec(src.read().split('# ---------- 实验一')[0], ns)
    Xs, y, t = ns['Xs'], ns['y'], ns['t']
    Bm = ns['bspline_basis'](t, 40)
    J, R = ns['gram_and_pen'](Bm, t)
    C = ns['to_coef'](Xs, Bm, R)
    M = np.linalg.cholesky(J + 1e-10 * np.eye(len(J)))
    HOT = tuple(int(i) for i in pd.read_csv('04outputs/57rebuttal_guard_spectra.csv').idx)   # 57 按 B10.9 的规则导出的名单
    print(f'57 导出的光谱名单：{list(HOT)}', flush=True)
    rows = []
    for set_name, seeds in G.SEED_SETS.items():
        for N in G.N_GRID:
            for sd in seeds:
                idx = np.random.RandomState(sd).permutation(len(y))
                itr, ite = idx[:N], idx[N:N + 400]
                ytr, yt = y[itr], y[ite]
                rest = ~np.isin(ite, HOT)
                kg = [k for k in G.K_ALL if k <= max(2, int(N * 0.75) - 2)] or [2]
                k_cv = ns['cv_pick'](C[itr], ytr, J, R, 'fpcr', kg)
                lam_cv = ns['cv_pick'](C[itr], ytr, J, R, 'pflr', G.LAM_ALL)
                for meth, chain in (
                        ('FPCR', sorted((k for k in kg if k <= k_cv), reverse=True)),
                        ('FPLR', sorted(v for v in G.LAM_ALL if v >= lam_cv * (1 - 1e-9)))):
                    lev = fpcr_levels(C[itr], ytr, C[ite], M, chain) if meth == 'FPCR' else fplr_levels(C[itr], ytr, C[ite], J, R, chain)
                    p0 = lev[chain[0]][0]
                    rows.append((set_name, N, meth, sd, 'cv', r2_score(yt, p0), r2_score(yt[rest], p0[rest]), 0.0))
                    for rule in RULES:
                        p, backed = backoff(lev, chain, rule)
                        rows.append((set_name, N, meth, sd, rule, r2_score(yt, p), r2_score(yt[rest], p[rest]), float(backed.mean())))
            print(f"{set_name} N={N} 完成，累计 {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows, columns=['seed_set', 'N', 'method', 'seed', 'rule', 'R2_all', 'R2_rest', 'frac_backed'])

    # ---- 一致性门：cv 与倍数 3 须与 57 脚本的 cv、lev 两臂逐次一致 ----
    ref = pd.read_csv('04outputs/57rebuttal_guard_draws.csv')
    ref = ref[ref.method.isin(['FPCR', 'FPLR']) & ref.arm.isin(['cv', 'lev'])].assign(rule=lambda q: q.arm.map({'cv': 'cv', 'lev': 'x3'}))
    chk = df[df.rule.isin(['cv', 'x3'])].merge(ref[['seed_set', 'N', 'method', 'seed', 'rule', 'R2', 'frac_backed']],
                                               on=['seed_set', 'N', 'method', 'seed', 'rule'], suffixes=('', '_57'), how='outer', indicator=True)
    assert (chk['_merge'] == 'both').all(), chk['_merge'].value_counts().to_dict()
    gap = float((chk.R2_all - chk.R2).abs().max())
    print(f"\n一致性门：与 57rebuttal_guard_draws.csv 比对 {len(chk)} 条，最大 |ΔR²| = {gap:.2e}")
    assert gap <= 1e-9, '与 57 脚本的结果不一致，中止'
    lv = chk[chk.rule == 'x3']
    assert float((lv.frac_backed - lv.frac_backed_57).abs().max()) <= 1e-12

    summ = (df.groupby(['seed_set', 'N', 'method', 'rule'], sort=False)
            .agg(n=('R2_all', 'size'), median_all=('R2_all', 'median'), n_neg_all=('R2_all', lambda s: int((s < 0).sum())),
                 median_rest=('R2_rest', 'median'), n_neg_rest=('R2_rest', lambda s: int((s < 0).sum())),
                 min_rest=('R2_rest', 'min'), frac_backed_median=('frac_backed', 'median')).reset_index())
    df.to_csv('04outputs/58rebuttal_guard_posthoc_draws.csv', index=False, encoding='utf-8-sig')
    summ.to_csv('04outputs/58rebuttal_guard_posthoc_summary.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 220)
    order = ['cv', *RULES]
    for set_name in G.SEED_SETS:
        for meth in ('FPCR', 'FPLR'):
            q = summ[(summ.seed_set == set_name) & (summ.method == meth)]
            print(f"\n{set_name} · {meth}")
            for col, title in (('n_neg_all', '全部测试光谱：R²<0 次数'), ('median_all', '全部测试光谱：R² 中位数'),
                               ('n_neg_rest', '除去名单上的光谱后：R²<0 次数'), ('median_rest', '除去名单上的光谱后：R² 中位数'),
                               ('frac_backed_median', '被回退的测试光谱占比（中位数）')):
                print(title)
                print(q.pivot(index='N', columns='rule', values=col)[order].round(3).to_string())
    print(f"\n总耗时 {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
