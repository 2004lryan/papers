# -*- coding: utf-8 -*-
"""
49rebuttal_derived_stats.py: 修订稿引用的派生统计量，由已有产物现算落盘（导师意见 3、5，预注册 B10.3、B10.4；不新增任何模型拟合）

为什么要有这个脚本：修订稿要引用几类不在任何产物里直接出现的汇总量——小样本 20 次重抽的中位数、最低值与 R²<0 的次数，
逐次配对的占优次数，跨批次设置中目标批次参考值超出源批次范围的比例，系数函数分位带在极值窗口内外的相对宽度。
这些量若只在构建脚本或临时分析里算，稿件数字就没有可复核的落点。本脚本把它们一次算出、写成 CSV，
构建脚本与 42 复算脚本读取同一份产物。

口径：
  小样本    输入 04outputs/21smallsample_repeats.csv（每个 N、方法、策略 20 次重抽的测试集 R²）。
            均值与 21 脚本、表 A2 同源；中位数、最低值、R²<0 次数为描述量，是看到均值受少数严重失效支配后补报的（C3 已登记）。
            标准误 = 20 次的样本标准差 / √20。逐次配对按同一种子比较两者的 R²，报前者更高的次数。
            方法标签 PFLR 输出为 FPLR（B10.6）。
  跨批次    输入 46rebuttal_recal_pred.csv（目标批次逐样品真值）与 46rebuttal_recal_batches.csv（源批次最小、最大值），
            统计目标批次 SSC 低于源批次最小值、高于最大值的样品数。
  系数函数  输入 48rebuttal_beta.csv 与 48rebuttal_beta_windows.csv。区域取「绝对值前 400 点」规则下点数 ≥90 的两个窗口
            （与图 5 标注的窗口相同）及其余波长；逐点分位带宽度 = 97.5% 分位 − 2.5% 分位，
            报其与该点部署 β 绝对值之比的中位数、25 条曲线同号的点数占比。

运行方式:
    cd <项目根目录>
    python3 02code/49rebuttal_derived_stats.py

输出文件:
    04outputs/49rebuttal_smallsample_summary.csv — 每个（N, 方法, 策略）的均值、中位数、最低、最高、R²<0 次数、标准误
    04outputs/49rebuttal_smallsample_paired.csv  — 每个 N 下指定配对中前者 R² 更高的次数（共 20 次）
    04outputs/49rebuttal_recal_range.csv         — 每种批次设置下目标批次超出源批次 SSC 范围的样品数与比例
    04outputs/49rebuttal_beta_band.csv           — 系数函数极值窗口内外的同号占比与分位带相对宽度
    05logs/49rebuttal_derived_stats_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

不读 05data，只读 04outputs。
"""
import numpy as np
import pandas as pd

O = '04outputs'
LABEL = {'PFLR': 'FPLR'}
N_DRAW = 20

# ---------------- 小样本 ----------------
rep = pd.read_csv(f'{O}/21smallsample_repeats.csv')
rep['method'] = rep['method'].replace(LABEL)
cnt = rep.groupby(['N', 'method', 'strategy']).size()
if not (cnt == N_DRAW).all():
    raise SystemExit(f'21smallsample_repeats.csv 中存在非 {N_DRAW} 次重抽的组合：\n{cnt[cnt != N_DRAW]}')
g = rep.groupby(['N', 'method', 'strategy']).R2
small = pd.DataFrame({'mean': g.mean(), 'median': g.median(), 'min': g.min(), 'max': g.max(),
                      'n_neg': g.apply(lambda v: int((v < 0).sum())),
                      'se': g.std(ddof=1) / np.sqrt(N_DRAW)}).reset_index()
small.to_csv(f'{O}/49rebuttal_smallsample_summary.csv', index=False, encoding='utf-8-sig')

wide = rep.pivot_table(index=['N', 'seed'], columns=['method', 'strategy'], values='R2')
PAIRS = [(('SVR', 'fixed'), ('FPCR', 'fixed')), (('SVR', 'fixed'), ('FPLR', 'fixed')),
         (('FPCR', 'fixed'), ('PLSR', 'fixed')), (('FPLR', 'fixed'), ('PLSR', 'fixed')),
         (('FPCR', 'cv'), ('FPCR', 'fixed')), (('FPLR', 'cv'), ('FPLR', 'fixed')), (('SVR', 'cv'), ('SVR', 'fixed')),
         (('FPCR', 'cv'), ('PLSR', 'fixed')), (('FPLR', 'cv'), ('PLSR', 'fixed')),
         (('SVR', 'cv'), ('FPCR', 'cv')), (('SVR', 'cv'), ('FPLR', 'cv'))]
paired = []
for n_tr, sub in wide.groupby(level='N'):
    for a, b in PAIRS:
        paired.append(dict(N=int(n_tr), a=f'{a[0]}_{a[1]}', b=f'{b[0]}_{b[1]}',
                           n_a_higher=int((sub[a] > sub[b]).sum()), n_draws=len(sub)))
paired_df = pd.DataFrame(paired)
paired_df.to_csv(f'{O}/49rebuttal_smallsample_paired.csv', index=False, encoding='utf-8-sig')

print('【小样本】每个（N, 方法, 策略）20 次重抽的测试集 R²')
print(f'{"N":>5} {"method":>6} {"strategy":>8} {"mean":>8} {"median":>8} {"min":>8} {"max":>8} {"n_neg":>6} {"se":>7}')
for q in small.itertuples():
    print(f'{q.N:>5} {q.method:>6} {q.strategy:>8} {q.mean:>8.3f} {q.median:>8.3f} {q.min:>8.3f} {q.max:>8.3f} {q.n_neg:>6} {q.se:>7.3f}')
print('\n【小样本逐次配对】前者 R² 更高的次数（共 20 次）')
print(paired_df.pivot_table(index='N', columns=['a', 'b'], values='n_a_higher').to_string())

# ---------------- 跨批次：目标批次超出源批次范围 ----------------
pred = pd.read_csv(f'{O}/46rebuttal_recal_pred.csv')
bat = pd.read_csv(f'{O}/46rebuttal_recal_batches.csv')
rng_rows = []
for st in ['2018→2019', '2019→2018', '留新疆', '留山东', '留陕西']:
    src = bat[(bat.level == '源批次') & (bat.name == st)]
    if len(src) != 1:
        raise SystemExit(f'46rebuttal_recal_batches.csv 中「源批次 {st}」应恰有一行')
    y_t = pred[(pred.setting == st) & (pred.method == 'PLSR')].sort_values('idx').y.to_numpy(dtype=float)
    for m in ['SVR', 'FPLR', 'FPCR']:                       # 四个方法的目标批次真值须一致
        y_m = pred[(pred.setting == st) & (pred.method == m)].sort_values('idx').y.to_numpy(dtype=float)
        if len(y_m) != len(y_t) or float(np.max(np.abs(y_m - y_t))) > 1e-9:
            raise SystemExit(f'{st}：{m} 与 PLSR 的目标批次真值不一致')
    lo, hi = float(src['min'].iloc[0]), float(src['max'].iloc[0])
    n_lo, n_hi = int((y_t < lo).sum()), int((y_t > hi).sum())
    rng_rows.append(dict(setting=st, src_min=lo, src_max=hi, n_tgt=len(y_t), n_below=n_lo, n_above=n_hi,
                         frac_outside=(n_lo + n_hi) / len(y_t)))
rng_df = pd.DataFrame(rng_rows)
rng_df.to_csv(f'{O}/49rebuttal_recal_range.csv', index=False, encoding='utf-8-sig')
print('\n【跨批次】目标批次 SSC 超出源批次取值范围的样品')
for q in rng_df.itertuples():
    print(f'  {q.setting:<10} 源范围 {q.src_min:.2f} ~ {q.src_max:.2f}  目标 n= {q.n_tgt}  低于 {q.n_below}  高于 {q.n_above}  '
          f'占比 {100 * q.frac_outside:.1f} %')

# ---------------- 系数函数：分位带相对宽度 ----------------
beta = pd.read_csv(f'{O}/48rebuttal_beta.csv')
win = pd.read_csv(f'{O}/48rebuttal_beta_windows.csv')
big = win[(win.kind == '绝对值前400点') & (win.n_points >= 90)].sort_values('lo')
if len(big) != 2:
    raise SystemExit(f'点数 ≥90 的极值窗口应为 2 个，实为 {len(big)} 个')
wl = beta.wavelength_nm.to_numpy(dtype=float)
band = (beta.beta_q975 - beta.beta_q025).to_numpy(dtype=float)
absb = np.abs(beta.beta_deploy.to_numpy(dtype=float))
stable = beta.sign_stable.astype(bool).to_numpy()
masks = [(f'{q.lo:.2f}~{q.hi:.2f}', q.lo, q.hi, (wl >= q.lo) & (wl <= q.hi)) for q in big.itertuples()]
inside = np.zeros(len(wl), dtype=bool)
for *_rest, mk in masks:
    inside |= mk
masks.append(('其余波长', np.nan, np.nan, ~inside))
band_rows = []
for name, lo_w, hi_w, mk in masks:
    band_rows.append(dict(region=name, lo=lo_w, hi=hi_w, n_points=int(mk.sum()), frac_sign_stable=float(stable[mk].mean()),
                          median_band_over_absbeta=float(np.median(band[mk] / np.maximum(absb[mk], 1e-12))),
                          median_band_over_peak=float(np.median(band[mk]) / absb.max())))
band_df = pd.DataFrame(band_rows)
band_df.to_csv(f'{O}/49rebuttal_beta_band.csv', index=False, encoding='utf-8-sig')
print('\n【系数函数】极值窗口内外：25 条训练折 β 同号的点数占比，分位带宽度与部署 |β| 之比的中位数')
for q in band_df.itertuples():
    print(f'  {q.region:<16} 点数 {q.n_points}  同号占比 {q.frac_sign_stable:.3f}  带宽/|β| 中位 {q.median_band_over_absbeta:.2f}  '
          f'带宽/全局峰值 中位 {q.median_band_over_peak:.3f}')

print('\n→ 04outputs/49rebuttal_smallsample_summary.csv / 49rebuttal_smallsample_paired.csv / '
      '49rebuttal_recal_range.csv / 49rebuttal_beta_band.csv')
