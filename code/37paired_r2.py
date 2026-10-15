# -*- coding: utf-8 -*-
"""基于折级 R² 的全对配对检验（补投稿清单第 13 条与 7.2 的一处缺口）。

为什么要有这个脚本：23gbid_evaluation.py 的配对检验建立在 **LLH-GBID 综合指数**上，
且只做「各方法 vs 综合最优（SVR）」这 5 个对比。但论文结论 (1) 与摘要的核心论断是
**「PFLR 优于使用全部 2 048 个波段的 PLSR」**，其依据是 R²（0.797 对 0.763）与「5 折中 4 折占优」——
这一对既不在那 5 个对比里，用的也不是同一个指标。投稿前的诚信审计（2026-09-07）指出：
两者的 ±1 标准差区间重叠（0.758~0.836 对 0.745~0.781），
而论文没有给出针对这一对、基于 R² 的配对检验，读者无从判断该优势是否超出折间波动。

本脚本按 7.2 的规定补齐：对**全部 15 个方法对**在折级 R² 上做 Nadeau–Bengio 修正 t 检验
（σ²_corr = (1/J + n_te/n_tr)·σ̂²，5 折下 n_te/n_tr = 1/(K−1) = 0.25，df = J−1 = 4），
Holm 控制族错误率，并给出贝叶斯相关 t 检验在 ROPE=±0.02 下的三个概率。
做全对而不只做缺的那一对，是为了避免「按结果挑对己方有利的对比来补检验」。

运行方式:
    cd <项目根目录>
    python3 02code/37paired_r2.py

输出文件:
    04outputs/37paired_r2.csv   — 15 个方法对的差值、修正 t、p、Holm p 与三个贝叶斯概率
    05logs/37paired_r2.log      — 运行日志（由调用方重定向）

不读 05data。
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from baycomp import CorrelatedTTest

SEED, K = 20041210, 5
METHODS = ['PLSR', 'SVR', 'RF', '1D-CNN', 'FPCR', 'PFLR']
EPS = 0.02                                    # 与 23gbid_evaluation.py 同一个实际等价区间

d = pd.read_csv('04outputs/25nested_cv.csv')
d = d[d.seed == SEED]
R = {m: d[d.method == m].sort_values('fold').R2.to_numpy() for m in METHODS}
for m in METHODS:
    assert len(R[m]) == K, (m, len(R[m]))

rho = 1.0 / (K - 1)                           # n_te/n_tr：5 折下测试折与训练折的样本比
print(f'折级 R² 的全对配对检验（seed={SEED}，J={K} 折，Nadeau–Bengio 修正 ρ={rho}，df={K-1}，ROPE=±{EPS}）')
print('各方法 5 折 R² 均值：' + '  '.join(f'{m}={R[m].mean():.3f}' for m in METHODS) + '\n')

rows = []
for a, b in itertools.combinations(METHODS, 2):
    diff = R[a] - R[b]
    mu = float(diff.mean())
    var = float(diff.var(ddof=1))
    se = float(np.sqrt(var * (1.0 / K + rho)))          # Nadeau–Bengio 修正标准误
    t = mu / se if se > 0 else np.inf
    p = float(2 * stats.t.sf(abs(t), df=K - 1))
    # probs(x, y, rope) 第一项是「x 比 y 大超过 rope」的概率；R² 越大越好，直接传
    p_a, p_eq, p_b = CorrelatedTTest.probs(R[a], R[b], rope=EPS)
    rows.append(dict(对比=f'{a} − {b}', 差值=mu, NB标准误=se, t=t, p=p,
                     n_a_wins=int((diff > 0).sum()),
                     **{f'P({a}优)': p_a, 'P(等价)': p_eq, f'P({b}优)': p_b}))

r = pd.DataFrame(rows).sort_values('p').reset_index(drop=True)
r['Holm_p'] = np.maximum.accumulate(np.minimum((len(r) - np.arange(len(r))) * r.p.values, 1.0))
r['结论'] = np.where(r.Holm_p < 0.05, '差异显著', '未达显著')
cols = ['对比', '差值', 'NB标准误', 't', 'p', 'Holm_p', 'n_a_wins', '结论']
r[cols].to_csv('04outputs/37paired_r2.csv', index=False, encoding='utf-8-sig')

print(f'{"对比":<18}{"差值":>9}{"NB 标准误":>11}{"t":>8}{"p":>9}{"Holm p":>9}{"前者占优折数":>10}  结论')
for _, x in r.iterrows():
    print(f'{x.对比:<18}{x.差值:>9.4f}{x.NB标准误:>11.4f}{x.t:>8.2f}{x.p:>9.4f}'
          f'{x.Holm_p:>9.4f}{x.n_a_wins:>10d}/5  {x.结论}')

print('\n【论文结论 (1) 所依据的那一对】')
pf = r[r.对比 == 'FPCR − PFLR']
pp = [x for x in rows if x['对比'] == 'PLSR − PFLR'][0]
print(f'  PFLR 相对 PLSR：差值 {-pp["差值"]:+.4f}，5 折中 PFLR 占优 {5 - pp["n_a_wins"]} 折，'
      f'Nadeau–Bengio 修正后 p={float(r[r.对比 == "PLSR − PFLR"].p.iloc[0]):.4f}，'
      f'Holm 校正后 p={float(r[r.对比 == "PLSR − PFLR"].Holm_p.iloc[0]):.4f}')
print(f'  贝叶斯（ROPE=±{EPS}）：P(PLSR 优)={pp["P(PLSR优)"]:.3f}，P(等价)={pp["P(等价)"]:.3f}，'
      f'P(PFLR 优)={pp["P(PFLR优)"]:.3f}')
print('\n→ 04outputs/37paired_r2.csv')
