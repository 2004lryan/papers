# -*- coding: utf-8 -*-
"""跨年份外部验证：在一个采集年份上建模，在另一个年份上 zero-shot（项目规范 投稿清单第 11 条）。

为什么要有这个脚本：表 3 的嵌套交叉验证把 2018、2019 两年的样品混在一起划折，每个测试折里都
既有 2018 也有 2019 的果子，模型见过测试年份的采集条件。这样得到的 R² 回答的是「同一批数据内
能预测多准」，而不是「换一年、换一批果子还准不准」。论文 3.6 节的跨产地泛化同理——它按产地划分，
但两年仍然混着。清单第 11 条要的是**完全不参与训练的独立集合上的 zero-shot 落差**。

为什么用年份而不是别的划分做外部集：本数据的两个年份是**两次独立采集**，实测差异很大——
原始光谱均值 2018 为 3 615、2019 为 7 227（近两倍，反映采集增益/仪器状态不同），
SSC 分布 12.92±1.47 对 13.89±1.68（Welch t=−11.11，p=1.9e−27）。协变量与标签同时发生偏移，
正是外部验证要考察的情形。相比之下按产地划分时两年混合，年份这一最大的批次因素被平均掉了。
本设计不需要新采集数据，但它给出的是真正的 zero-shot：测试年份的样品一个也没进过训练。

**边界（必须在论文里写明，不得夸大）**：两个年份同属一个课题组、同一台仪器、同一套采样规程，
因此这是**时间外部验证**（temporal external validation），不等于第三方独立复现。

协议（与表 3 等预算，保证落差可归因于跨年而非调参强度差异）：
  选参    在训练年份**内部**做 KFold(5, shuffle=True, seed) 折外 CV，在同一 12 组候选里按 RMSE 选优
  重拟合  用整个训练年份重新拟合选中的配置
  评价    在另一年份上一次性预测，测试年份从未参与选参与拟合
  参照    同时报告选参时那 5 折的折外指标（in-domain），落差 = zero-shot − in-domain

两个方向都跑（2018→2019 与 2019→2018），因为单向结果无法区分「跨年掉点」与「某一年更难预测」。

运行方式:
    cd <项目根目录>
    python3 02code/31external_year.py

输出文件:
    04outputs/31external_year.csv   — 每个方法 × 每个方向一行，含 in-domain 与 zero-shot 两套指标
    05logs/31external_year.log      — 运行日志（由调用方重定向）

只读 05data。
"""
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

with open('02code/16unified_comparison.py') as _src: exec(_src.read().split("METHODS=['PLSR'")[0])
sys.path.insert(0, '02code')
from grids import GRIDS                                          # noqa: E402

SEED = 20041210
K_INNER = 5
METHODS = ['PLSR', 'SVR', 'FPCR', 'PFLR', 'RF', '1D-CNN']        # 由快到慢，便于中途查看
if os.environ.get('EXT_METHODS'):
    METHODS = os.environ['EXT_METHODS'].split(',')
OUT = '04outputs/31external_year.csv'

# 年份标签：16 脚本里 X=np.vstack([X1,X2])，故前 len(y1) 行是 2018、其后是 2019
YEAR = np.r_[np.full(len(y1), 2018), np.full(len(y2), 2019)]
assert len(YEAR) == len(y), '年份标签与样本数不一致'
IDX = {yr: np.where(YEAR == yr)[0] for yr in (2018, 2019)}
DIRECTIONS = [(2018, 2019), (2019, 2018)]


def metrics(ytr, yte, p):
    """指标 + 把跨年失效拆成「整体平移」与「判别能力」两部分。

    为什么要拆：跨年 zero-shot 的 R² 可能因一个常数偏移就变成负数，而 R² 为负只说明
    「不如直接用均值预测」，不说明模型没学到光谱与 SSC 的关系。化学计量学里跨批次迁移的
    常规处置正是先做 slope/bias 校正再看剩余误差，故这里同时报告：
      R2_bias — 只减掉平均偏差（校正截距）后的 R²，回答「若知道新年份的整体水平，还准不准」；
      r       — 预测与实测的皮尔逊相关，不受任何平移与缩放影响，是判别能力的下界刻画。
    三者并列才能把「模型失效」与「校准漂移」区分开，只报 R² 会把后者误判成前者。
    """
    p = np.asarray(p, dtype=float)
    e = p - yte
    rm = float(np.sqrt(mean_squared_error(yte, p)))
    r = float(np.corrcoef(p, yte)[0, 1]) if np.std(p) > 1e-12 else float('nan')
    return dict(R2=float(r2_score(yte, p)), RMSE=rm, RPD=float(np.std(yte)) / rm,
                MAE=float(np.abs(e).mean()), BIAS=float(e.mean()),
                R2_bias=float(r2_score(yte, p - e.mean())), r=r)


print(f'跨年份外部验证（seed={SEED}，内层 {K_INNER} 折选参，12 组候选／方法）')
print(f'  2018: n={len(IDX[2018])}  SSC={y[IDX[2018]].mean():.2f}±{y[IDX[2018]].std():.2f}')
print(f'  2019: n={len(IDX[2019])}  SSC={y[IDX[2019]].mean():.2f}±{y[IDX[2019]].std():.2f}\n')

rows = pd.read_csv(OUT).to_dict('records') if os.path.exists(OUT) else []
done = {(r['method'], r['train_year']) for r in rows}

for m in METHODS:
    grid, fit = GRIDS[m], mk(m)[1]
    for tr_yr, te_yr in DIRECTIONS:
        if (m, tr_yr) in done:
            print(f'  {m:<8} {tr_yr}→{te_yr} 已完成，跳过')
            continue
        t0 = time.time()
        itr_all, ite = IDX[tr_yr], IDX[te_yr]

        # ① 训练年份内部 5 折 CV 选参；顺带留下 in-domain 折外预测作参照
        inner = list(KFold(K_INNER, shuffle=True, random_state=SEED).split(itr_all))
        se, oof = [], np.zeros((len(grid), len(itr_all)))
        for ci, c in enumerate(grid):
            for ia, iv in inner:
                p = fit(c, itr_all[ia], itr_all[iv], SEED)
                oof[ci, iv] = p
            se.append(mean_squared_error(y[itr_all], oof[ci]))
        best = int(np.argmin(se))
        bc = grid[best]

        # ② 用整个训练年份重拟合 → ③ 在另一年份 zero-shot
        p_ext = fit(bc, itr_all, ite, SEED)

        ind = metrics(y[itr_all], y[itr_all], oof[best])          # in-domain（同年折外）
        ext = metrics(y[itr_all], y[ite], p_ext)                  # zero-shot（跨年）
        rows.append(dict(method=m, train_year=tr_yr, test_year=te_yr, picked=str(bc),
                         n_train=len(itr_all), n_test=len(ite),
                         **{f'ind_{k}': v for k, v in ind.items()},
                         **{f'ext_{k}': v for k, v in ext.items()},
                         drop_R2=ind['R2'] - ext['R2'], sec=round(time.time() - t0, 1)))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f'  {m:<8} {tr_yr}→{te_yr}  同年 R²={ind["R2"]:.3f}  跨年 R²={ext["R2"]:>7.3f}  '
              f'去偏后 {ext["R2_bias"]:.3f}  r={ext["r"]:.3f}  偏差 {ext["BIAS"]:+.3f}  '
              f'RMSE {ind["RMSE"]:.3f}→{ext["RMSE"]:.3f}  选中 {bc}  用时 {time.time() - t0:.0f}s', flush=True)

d = pd.DataFrame(rows)

# 论文表 9 报的是「按方法汇总」的派生量，与本文件的逐方向行不同：偏差先取绝对值再平均
# （两个方向一负一正，直接平均会互相抵消）。派生量单独落盘，使正文里的每个数字都能在产物中查到。
if len(d) == len(METHODS) * 2 or set(d.method) >= set(METHODS):
    dd = d.assign(ext_absBIAS=d.ext_BIAS.abs())
    summ = (dd.groupby('method')[['ind_R2', 'ext_R2', 'ext_R2_bias', 'ext_r', 'ext_absBIAS', 'drop_R2']]
            .mean().round(3).reset_index())   # 与正文的 .3f 同位数，便于逐字核对
    summ.to_csv('04outputs/31external_year_summary.csv', index=False)
    print('\n按方法汇总（论文表 9 的取数口径）→ 04outputs/31external_year_summary.csv')
    print(summ.to_string(index=False))

print(f'\n【跨年 zero-shot 落差汇总】（{len(d)} 行）')
print(f'{"方法":<10}{"方向":<14}{"同年 R²":>9}{"跨年 R²":>9}{"去偏 R²":>9}{"相关 r":>8}{"偏差":>8}')
for _, r in d.iterrows():
    print(f'{r.method:<10}{r.train_year}→{r.test_year:<9}{r.ind_R2:>9.3f}{r.ext_R2:>9.3f}'
          f'{r.ext_R2_bias:>9.3f}{r.ext_r:>8.3f}{r.ext_BIAS:>8.3f}')
g = (d.groupby('method')[['ind_R2', 'ext_R2', 'ext_R2_bias', 'ext_r', 'drop_R2']]
     .mean().sort_values('ext_R2_bias', ascending=False))
print(f'\n{"方法":<10}{"同年均值":>10}{"跨年 R²":>10}{"去偏 R²":>10}{"相关 r":>9}{"落差":>9}')
for m_, r in g.iterrows():
    print(f'{m_:<10}{r.ind_R2:>10.3f}{r.ext_R2:>10.3f}{r.ext_R2_bias:>10.3f}{r.ext_r:>9.3f}{r.drop_R2:>9.3f}')
print(f'\n→ {OUT}')
