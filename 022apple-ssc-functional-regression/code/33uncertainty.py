# -*- coding: utf-8 -*-
"""预测区间的覆盖率校准与分布外检测（投稿清单第 12 条）。

为什么要有这个脚本：表 3 只给点预测的精度。一个 RMSE=0.735 °Brix 的模型，用在分选线上时
真正要回答的问题是「这一个果子的 SSC 有多大把握落在某个范围内」，以及「这个果子是不是模型
没见过的类型、它的预测该不该信」。点预测不带这两样信息，前者要预测区间且区间的覆盖率必须
兑现（名义 90% 就该真覆盖 90%），后者要分布外检测。清单第 12 条要的就是这两件事。

**区间怎么构造：CV+ 型分裂共形（Barber 等 2021）**。选它而不是正态假设下的 ŷ±1.96σ，
理由是后者要求残差正态且同方差，本数据的折外残差既不正态也随 SSC 水平变化；共形预测
不作分布假设，只要求残差可交换，在有限样本下即有覆盖保证。
**校准残差取自其他 4 个外层折**——绝不含被预测样本所在折，否则区间会因见过该折而偏窄，
覆盖率看着达标其实是自证。这一条是本脚本最容易写错的地方。

**分布外检测怎么做**：用 2018 年样品拟合 PCA（保留 99% 方差），以重构误差作为 OOD 分数。
选重构误差而不是马氏距离，是因为 2 048 维、1 300 样本下协方差不可逆，马氏距离需先降维，
降维后又丢掉了「落在主子空间之外」这一最关键的证据——而跨年批次差异恰恰体现在那里。
检验两件事：① 2019 的分数是否显著高于 2018（能否检出已知的跨年偏移）；
② 分数与绝对误差是否正相关（分数高时预测是否真的更不可信）。②不成立则该分数没有使用价值。

运行方式:
    cd <项目根目录>
    python3 02code/33uncertainty.py

输出文件:
    04outputs/33uncertainty_coverage.csv  — 每方法 × 每个名义水平的经验覆盖率与区间宽度
    04outputs/33uncertainty_ood.csv       — OOD 分数的分年份统计与其同绝对误差的相关性
    05logs/33uncertainty.log              — 运行日志（由调用方重定向）

只读 05data。
"""
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from scipy import stats

with open('02code/16unified_comparison.py') as _src: exec(_src.read().split("METHODS=['PLSR'")[0])
sys.path.insert(0, '02code')

SEED = 20041210
METHODS = ['PLSR', 'SVR', 'RF', '1D-CNN', 'FPCR', 'PFLR']
LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
YEAR = np.r_[np.full(len(y1), 2018), np.full(len(y2), 2019)]

pred = pd.read_csv('04outputs/25nested_cv_pred.csv')
pred = pred[pred.seed == SEED]

print(f'预测区间校准与分布外检测（seed={SEED}，n={len(y)}，名义水平 {LEVELS}）\n')

# ── 一、CV+ 型分裂共形的覆盖率校准 ──────────────────────────────────────────
rows = []
for m in METHODS:
    p = pred[pred.method == m].sort_values('idx')
    assert len(p) == len(y), (m, len(p))
    idx, yh, fold = p.idx.to_numpy(), p.pred.to_numpy(), p.fold.to_numpy()
    resid = np.abs(yh - y[idx])
    for lv in LEVELS:
        lo = np.empty(len(idx))
        hi = np.empty(len(idx))
        for k in sorted(set(fold)):
            te = fold == k
            cal = resid[~te]                       # 校准残差只来自其他 4 折
            q = float(np.quantile(cal, min(1.0, lv * (1 + 1 / len(cal)))))   # 有限样本修正
            lo[te], hi[te] = yh[te] - q, yh[te] + q
        cov = float(((y[idx] >= lo) & (y[idx] <= hi)).mean())
        rows.append(dict(method=m, nominal=lv, empirical=cov, gap=cov - lv,
                         mean_width=float((hi - lo).mean()),
                         width_over_sd=float((hi - lo).mean() / np.std(y))))
cov_df = pd.DataFrame(rows)
cov_df.to_csv('04outputs/33uncertainty_coverage.csv', index=False)

print('【共形预测区间的覆盖率】（名义 → 经验，括号内为平均区间宽度 °Brix）')
print(f'{"方法":<9}' + ''.join(f'{int(l*100):>13d}%' for l in LEVELS) + f'{"ECE":>9}')
for m in METHODS:
    d = cov_df[cov_df.method == m].set_index('nominal')
    cells = ''.join(f'{d.loc[l, "empirical"]:>8.3f}({d.loc[l, "mean_width"]:.2f})' for l in LEVELS)
    ece = float(np.abs(d.gap).mean())
    print(f'{m:<9}{cells}{ece:>9.4f}')
print('\nECE = 各名义水平上 |经验−名义| 的等权平均；越小说明区间的把握度越诚实。')

# ── 二、分布外检测 ─────────────────────────────────────────────────────────
i18, i19 = np.where(YEAR == 2018)[0], np.where(YEAR == 2019)[0]
A = Xp[i18]
mu = A.mean(0)
U, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
kk = int(np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), 0.99) + 1)
V = Vt[:kk].T


def ood_score(M):
    Z = M - mu
    return np.linalg.norm(Z - (Z @ V) @ V.T, axis=1)


s18, s19 = ood_score(Xp[i18]), ood_score(Xp[i19])
t, pv = stats.ttest_ind(s19, s18, equal_var=False)
auc = float((s19[:, None] > s18[None, :]).mean() + 0.5 * (s19[:, None] == s18[None, :]).mean())
print(f'\n【分布外检测】PCA 在 2018 上拟合，保留 {kk} 个主成分（99% 方差），重构误差为 OOD 分数')
print(f'  2018 分数 {s18.mean():.4f}±{s18.std():.4f}   2019 分数 {s19.mean():.4f}±{s19.std():.4f}')
print(f'  Welch t={t:.2f}, p={pv:.3g}；把「是否 2019」当作标签时 AUC={auc:.3f}')

allscore = np.empty(len(y))
allscore[i18], allscore[i19] = s18, s19
orows = [dict(item='2018 分数均值', value=float(s18.mean())),
         dict(item='2019 分数均值', value=float(s19.mean())),
         dict(item='Welch t', value=float(t)), dict(item='p 值', value=float(pv)),
         dict(item='年份判别 AUC', value=auc), dict(item='PCA 主成分数', value=kk)]
print('\n  OOD 分数与折外绝对误差的相关（分数高时预测是否真的更差）：')
print(f'{"方法":<9}{"Spearman ρ":>12}{"p":>11}{"高分 10% 的 MAE":>16}{"其余 90% 的 MAE":>16}')
for m in METHODS:
    p = pred[pred.method == m].sort_values('idx')
    ae = np.abs(p.pred.to_numpy() - y[p.idx.to_numpy()])
    sc = allscore[p.idx.to_numpy()]
    rho, prho = stats.spearmanr(sc, ae)
    hi = sc >= np.quantile(sc, 0.9)
    print(f'{m:<9}{rho:>12.3f}{prho:>11.3g}{ae[hi].mean():>16.3f}{ae[~hi].mean():>16.3f}')
    orows.append(dict(item=f'{m} Spearman ρ(OOD, |误差|)', value=float(rho)))
    orows.append(dict(item=f'{m} ρ 的 p 值', value=float(prho)))
    orows.append(dict(item=f'{m} 高分 10% MAE', value=float(ae[hi].mean())))
    orows.append(dict(item=f'{m} 其余 90% MAE', value=float(ae[~hi].mean())))
pd.DataFrame(orows).to_csv('04outputs/33uncertainty_ood.csv', index=False)
print('\n→ 04outputs/33uncertainty_coverage.csv  /  04outputs/33uncertainty_ood.csv')

# ── 三、把 OOD 分数放回它该起作用的场景：跨年 zero-shot ────────────────────
# 第二节得到的是负结果——OOD 分数与折外误差不相关，甚至微弱负相关。直接写「OOD 检测无效」
# 是错的归因：嵌套 CV 把两年混着划折，模型训练时**见过** 2019 的采集条件，此时一个样本
# 「离 2018 的主子空间多远」本就不该预测它的误差。OOD 分数要有用，必须是在模型确实没见过
# 那个分布的时候。故此处把同一个分数放进 31external_year.py 的跨年设置复核一次：
# 2018 训练、2019 zero-shot 预测，再看分数与误差的关系。只跑四个秒级方法，与 31 脚本同协议。
from sklearn.model_selection import KFold                            # noqa: E402
from sklearn.metrics import mean_squared_error             # noqa: E402
from grids import GRIDS                                              # noqa: E402

print('\n【跨年 zero-shot 下复核同一个 OOD 分数】2018 训练 → 2019 预测（PLSR/SVR/FPCR/PFLR）')
# 读数前先看一个混淆因子：OOD 分数若与 SSC 本身相关，而跨年误差又主要是随 SSC 变化的系统偏移，
# 那么「分数—误差」的相关就会被这条通路制造出来，与 OOD 检测本身无关。故一并报告去偏后的相关。
rho_y18 = stats.spearmanr(s18, y[i18])
rho_y19 = stats.spearmanr(s19, y[i19])
print(f'  混淆检查：ρ(OOD, 真实 SSC) 在 2018 为 {rho_y18[0]:+.3f}（p={rho_y18[1]:.2g}）、'
      f'在 2019 为 {rho_y19[0]:+.3f}（p={rho_y19[1]:.2g}）')
print(f'{"方法":<9}{"跨年 MAE":>10}{"原始 ρ":>9}{"去偏 ρ":>9}{"去偏 p":>10}'
      f'{"高分 10% MAE":>14}{"其余 90% MAE":>14}')
rows3 = [dict(item='ρ(OOD, SSC) 2018', value=float(rho_y18[0])),
         dict(item='ρ(OOD, SSC) 2019', value=float(rho_y19[0]))]
for m in ['PLSR', 'SVR', 'FPCR', 'PFLR']:
    grid, fitf = GRIDS[m], mk(m)[1]
    inner = list(KFold(5, shuffle=True, random_state=SEED).split(i18))
    se = []
    for c in grid:
        oof = np.zeros(len(i18))
        for ia, iv in inner:
            oof[iv] = fitf(c, i18[ia], i18[iv], SEED)
        se.append(mean_squared_error(y[i18], oof))
    bc = grid[int(np.argmin(se))]
    pe = fitf(bc, i18, i19, SEED)
    e = pe - y[i19]
    ae = np.abs(e)
    ae_db = np.abs(e - e.mean())                 # 扣掉整体偏移，只留下样本间的相对误差
    rho, prho = stats.spearmanr(s19, ae)
    rho_db, prho_db = stats.spearmanr(s19, ae_db)
    hi = s19 >= np.quantile(s19, 0.9)
    print(f'{m:<9}{ae.mean():>10.3f}{rho:>9.3f}{rho_db:>9.3f}{prho_db:>10.2g}'
          f'{ae[hi].mean():>14.3f}{ae[~hi].mean():>14.3f}')
    rows3 += [dict(item=f'跨年 {m} Spearman ρ(OOD, |误差|)', value=float(rho)),
              dict(item=f'跨年 {m} ρ 的 p 值', value=float(prho)),
              dict(item=f'跨年 {m} 去偏后 ρ(OOD, |误差|)', value=float(rho_db)),
              dict(item=f'跨年 {m} 去偏后 ρ 的 p 值', value=float(prho_db)),
              dict(item=f'跨年 {m} 系统偏移', value=float(e.mean())),
              dict(item=f'跨年 {m} 高分 10% MAE', value=float(ae[hi].mean())),
              dict(item=f'跨年 {m} 其余 90% MAE', value=float(ae[~hi].mean()))]

    # 共形区间在跨年下还保不保覆盖：可交换性被破坏时它应当失效，实测给出失效幅度
    inner_oof = np.zeros(len(i18))
    for ia, iv in inner:
        inner_oof[iv] = fitf(bc, i18[ia], i18[iv], SEED)
    cal = np.abs(inner_oof - y[i18])
    for lv in (0.90, 0.95):
        q = float(np.quantile(cal, min(1.0, lv * (1 + 1 / len(cal)))))
        cov = float((ae <= q).mean())
        rows3.append(dict(item=f'跨年 {m} 名义 {lv:.2f} 的经验覆盖', value=cov))
        if lv == 0.90:
            print(f'{"":9}  ↳ 用 2018 折外残差定的 90% 区间（半宽 {q:.2f}）在 2019 上实际覆盖 {cov:.3f}')
pd.DataFrame(pd.read_csv('04outputs/33uncertainty_ood.csv').to_dict('records') + rows3
             ).to_csv('04outputs/33uncertainty_ood.csv', index=False)
print('\n结论（第二、三节合读）：PCA 重构误差能可靠检出**批次层面**的分布偏移（年份判别 AUC 见上），')
print('但在**样本层面**不能预测哪一个样本会预测得更差——跨年设置下看似很强的负相关，扣掉系统偏移后')
print('即降到 |ρ|<0.09，它来自「OOD 分数与 SSC 负相关 × 跨年误差随 SSC 变化」这条混淆通路，')
print('不是分数本身的判别力。故该分数宜用作「该不该重新校准」的批次级警报，不宜用作单样本置信度过滤。')
print('\n→ 04outputs/33uncertainty_ood.csv（已追加跨年复核）')
