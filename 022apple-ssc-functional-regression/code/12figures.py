# -*- coding: utf-8 -*-
"""生成图 1（光谱）、图 3（小样本学习曲线）、图 4（回归系数函数与波段贡献）。样式见 figstyle.py。
图 3 的不确定度来自 04outputs/21smallsample_repeats.csv（20 次重抽逐次结果，由 21smallsample_repeats.py 生成并已与表 4 对账）。
输出 04outputs/figs/*.pdf/.svg/.png。只读 05data。"""
import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import sys; sys.path.insert(0,'02code')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
with open('02code/08fda_smallsample.py') as _src: exec(_src.read().split('# ---------- 实验一')[0])   # 数据、预处理、基函数、模型
from figstyle import *      # 多面板图在 figstyle.save 内经 require_matplotlib_panel_alignment(strict=True) 审计后才导出
from sklearn.cross_decomposition import PLSRegression
print("画图：")

# ======================= 图 1：三产地平均光谱 =======================
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4), layout='constrained')
for k in [1, 2, 3]:
    m = z == k; mu = X[m].mean(0); sd = X[m].std(0)
    ax[0].plot(wl, mu / 1e4, color=LOCC[k], lw=1.0, label=f'{LOC[k]}（n={m.sum()}）')
    ax[0].fill_between(wl, (mu - sd) / 1e4, (mu + sd) / 1e4, color=LOCC[k], alpha=0.13, lw=0)
    ax[1].plot(wl, Xs[m].mean(0), color=LOCC[k], lw=0.9, label=LOC[k])
ax[0].set_xlabel('波长/nm'); ax[0].set_ylabel('透射强度/10⁴ a.u.'); ax[0].set_title('(a) 原始光谱（阴影为 ±1 标准差）')
ax[1].set_xlabel('波长/nm'); ax[1].set_ylabel('一阶导数/a.u.'); ax[1].set_title('(b) SNV+SG 一阶导数光谱')
ax[0].legend(loc='upper right'); ax[1].legend(loc='lower right')
for a in ax: a.set_xlim(580, 1260); a.set_xticks([600, 800, 1000, 1200])
# (b) 局部放大：850~1 000 nm 的形态差异（放在右上空白区，不遮主曲线）
ins = ax[1].inset_axes([0.50, 0.52, 0.46, 0.40])
for k in [1, 2, 3]:
    m = (wl >= 850) & (wl <= 1000); ins.plot(wl[m], Xs[z == k][:, m].mean(0), color=LOCC[k], lw=0.8)
ins.set_xlim(850, 1000); ins.set_xticks([850, 900, 950, 1000]); ins.tick_params(labelsize=6, length=1.8, width=0.5)
ins.set_title('850~1 000 nm', fontsize=6.5, pad=2, loc='center')
for s in ins.spines.values(): s.set_linewidth(0.5)
_rect, _lines = ax[1].indicate_inset_zoom(ins, edgecolor=C['gray'], lw=0.6, alpha=0.9)
for _l in _lines: _l.set_visible(False)                          # 只留放大区域的方框，连接线会压住刻度数字
save(fig, 'Figure_1_spectra', exclude=[ins])

# ======================= 图 3：小样本学习曲线 + N=80 训练集/测试集散点 =======================
rep = pd.read_csv('04outputs/21smallsample_repeats.csv')
g = rep.groupby(['N', 'method', 'strategy']).R2.agg(['mean', 'std', 'count']).reset_index()
g['se'] = g['std'] / np.sqrt(g['count'])
NAME = {'PLSR': 'PLSR（离散）', 'SVR': 'SVR', 'FPCR': 'FPCR', 'PFLR': 'FPLR'}   # 缩写按 B10.6 改为 FPLR，数据键仍为 PFLR
# 图 3 与图 4 原为一张 2×3 的六面板图。拆成两张各 1×3，是排版决定而非内容改动：
# 六面板图连同中英文图注高约 11.5 cm，在正文新增 3.7~3.10 四节之后无论放在哪一段之后，
# 都会在前一页留下 8 cm 以上的空白（实测 45 种落点组合，最好的一种仍有一页填充率 71%）。
# 拆开后每张约 5.5 cm，两张可分别就近排在各自被引用的段落之后。面板内容与数据一字未改。
fig = plt.figure(figsize=(W2, 2.15))
fig.subplots_adjust(left=0.1, right=0.995, bottom=0.19, top=0.90, wspace=0.38)
ax = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
for a, strat, title in [(ax[0], 'fixed', '(a) 固定超参数'), (ax[1], 'cv', '(b) 交叉验证选超参数')]:
    for mth in ['PLSR', 'SVR', 'FPCR', 'PFLR']:
        d = g[(g.method == mth) & (g.strategy == ('fixed' if mth == 'PLSR' else strat))].sort_values('N')
        a.fill_between(d.N, d['mean'] - d.se, d['mean'] + d.se, color=METHOD[mth], alpha=0.15, lw=0)
        a.plot(d.N, d['mean'], marker=MARK[mth], ms=3, color=METHOD[mth], lw=1.1, label=NAME[mth],
               ls='-' if mth != 'PLSR' or strat == 'fixed' else '-')
    a.set_title(title); a.set_ylim(-0.75, 0.9)
# (c) 成对差值：同一次重抽下 交叉验证 − 固定
piv = rep.pivot_table(index=['N', 'method', 'seed'], columns='strategy', values='R2').reset_index()
piv = piv[piv.method != 'PLSR']; piv['d'] = piv.cv - piv.fixed
dg = piv.groupby(['N', 'method']).d.agg(['mean', 'std', 'count']).reset_index(); dg['se'] = dg['std'] / np.sqrt(dg['count'])
for mth in ['SVR', 'FPCR', 'PFLR']:
    d = dg[dg.method == mth].sort_values('N')
    ax[2].fill_between(d.N, d['mean'] - d.se, d['mean'] + d.se, color=METHOD[mth], alpha=0.15, lw=0)
    ax[2].plot(d.N, d['mean'], marker=MARK[mth], ms=3, color=METHOD[mth], lw=1.1, label=NAME[mth])
ax[2].set_title('(c) 两种策略之差'); ax[2].set_ylabel('ΔR²（交叉验证 − 固定）')
for a in ax:
    a.set_xscale('log'); a.set_xticks([20, 50, 100, 200, 400]); a.set_xticklabels(['20', '50', '100', '200', '400'])
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); a.set_xlabel('训练集样本量 N'); zero_line(a)
ax[0].set_ylabel('决定系数 R²')
LEG = dict(frameon=True, facecolor='white', edgecolor='none', framealpha=0.85, borderpad=0.55, labelspacing=0.45)   # 半透明白底：图例压在阴影带/散点上时仍可读
ax[0].legend(loc='lower right', **LEG); ax[2].legend(loc='lower right', **LEG)
# ---- 第二行 (d)(e)(f)：N=80、固定超参数下一次代表性重抽的训练集 / 测试集散点 ----
# 代表性重抽的选法：20 次重抽中，六个（模型×策略）组合的测试集 R² 与各自 20 次均值的偏差平方和最小的一次（seed 3）。
# 抽样、基系统（B 样条 L=40）与固定超参数均与 21smallsample_repeats.py（表 4）相同，并与其逐次结果对账。
from sklearn.svm import SVR as _SVR
SD_REP, N80 = 3, 80
B40 = bspline_basis(t, 40); J40, R40 = gram_and_pen(B40, t); C40 = to_coef(Xs, B40, R40)
rng = np.random.RandomState(SD_REP); idx = rng.permutation(len(y)); itr, ite = idx[:N80], idx[N80:N80 + 400]
PRED = {'SVR': lambda I: _SVR(C=100, gamma='scale', epsilon=0.1).fit(Xs[itr], y[itr]).predict(Xs[I]),
        'FPCR': lambda I: fpcr(C40[itr], y[itr], C40[I], J40, 10),
        'PFLR': lambda I: pflr(C40[itr], y[itr], C40[I], J40, R40, 1e-3)}
lo, hi = 8, 19
save(fig, 'Figure_2_smallsample')                       # 图 3：(a)(b)(c) 三面板趋势图
fig2 = plt.figure(figsize=(W2, 2.15))
fig2.subplots_adjust(left=0.1, right=0.995, bottom=0.19, top=0.90, wspace=0.38)
axs2 = [fig2.add_subplot(1, 3, i + 1) for i in range(3)]
for a, mth, lab in zip(axs2, ['SVR', 'FPCR', 'PFLR'], ['(a)', '(b)', '(c)'], strict=False):
    ptr, pte = PRED[mth](itr), PRED[mth](ite)
    r2tr, r2te = r2_score(y[itr], ptr), r2_score(y[ite], pte); rm = np.sqrt(mean_squared_error(y[ite], pte))
    ref = rep[(rep.N == N80) & (rep.method == mth) & (rep.strategy == 'fixed') & (rep.seed == SD_REP)].R2.iloc[0]
    assert abs(round(r2te, 3) - round(ref, 3)) < 1e-9, (mth, r2te, ref)          # 与表 4 所用的逐次结果一致
    a.scatter(y[ite], pte, s=5, color=C['blue'], alpha=0.5, edgecolors='none', label='测试集（n=400）', zorder=2)
    a.scatter(y[itr], ptr, s=10, color=C['orange'], edgecolors='white', linewidths=0.3, label='训练集（N=80）', zorder=3)
    a.plot([lo, hi], [lo, hi], ls='--', c=C['black'], lw=0.7, zorder=1)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_xticks([8, 10, 12, 14, 16, 18]); a.set_yticks([8, 10, 12, 14, 16, 18])
    a.set_title(f'{lab} {mth}（N=80）'); a.set_xlabel('实测值/°Brix')
    a.text(0.04, 0.96, f'训练集 R²={r2tr:.3f}\n测试集 R²={r2te:.3f}\nRMSE={rm:.3f} °Brix', transform=a.transAxes,
           va='top', ha='left', fontsize=6.5, linespacing=1.35)
    print(f"  图 4{lab} {mth}: 训练集 R²={r2tr:.3f}  测试集 R²={r2te:.3f}  RMSE={rm:.3f} °Brix")
axs2[0].set_ylabel('预测值/°Brix'); axs2[0].legend(loc='lower right', handletextpad=0.3, **LEG)
save(fig2, 'Figure_2b_scatter')

# ======================= 图 4：回归系数函数 + 波段贡献 =======================
# (a) β(λ)：B 样条 L=80，λ 由 4 折交叉验证从 13 点对数网格选出（与 3.5 节一致）
Bm = bspline_basis(t, 80); J, R = gram_and_pen(Bm, t); Cc = to_coef(Xs, Bm, R)
lam = cv_pick(Cc, y, J, R, 'pflr', np.logspace(-10, 2, 13))
Z = Cc @ J; ZtZ = Z.T @ Z; sc = np.trace(ZtZ) / max(np.trace(R), 1e-30)
bcoef = np.linalg.solve(ZtZ + lam * sc * R + 1e-12 * np.eye(len(R)), Z.T @ (y - y.mean()))
beta = Bm @ bcoef
# (b)(c) 六个子区间各自建模的精度与原始光谱信噪比（与 3.5 节正文一致）
# 评价协议与 25nested_cv.py、07band_range_test.py 一致：外层 KFold(5, shuffle=True)，
# 潜变量数在每个外层训练折内以 5 折交叉验证选取——每个区间各自选到最合适的潜变量数，
# 才是「该区间单独建模能做到多好」的公平度量。
from sklearn.model_selection import KFold
K_OUTER = K_INNER = 5; SEED_FIG = 20041210
segs = [(590, 700), (700, 800), (800, 900), (900, 1000), (1000, 1100), (1100, 1250)]
r2s = []; snr = []
for lo, hi in segs:
    m = (wl >= lo) & (wl < hi); Xseg = Xs[:, m]; R2 = []
    for a_, b_ in KFold(K_OUTER, shuffle=True, random_state=SEED_FIG).split(Xseg):
        bn, bs_ = 8, np.inf
        for nc in range(2, 26, 2):
            e = [mean_squared_error(y[a_][va], PLSRegression(n_components=nc)
                 .fit(Xseg[a_][tr], y[a_][tr]).predict(Xseg[a_][va]).ravel())
                 for tr, va in KFold(K_INNER, shuffle=True, random_state=SEED_FIG).split(a_)]
            if np.mean(e) < bs_: bs_, bn = np.mean(e), nc
        R2.append(r2_score(y[b_], PLSRegression(n_components=bn).fit(Xseg[a_], y[a_]).predict(Xseg[b_]).ravel()))
    r2s.append(np.mean(R2))
    print(f"  图 4(b) {lo}-{hi}nm  R²={np.mean(R2):.3f}")
    Xr_ = X[:, m]; d = np.diff(Xr_, axis=1)
    snr.append(Xr_.mean() / np.median(1.4826 * np.median(np.abs(d - np.median(d, 1, keepdims=True)), 1)))

fig = plt.figure(figsize=(W2, 3.9), layout='constrained')
gs = fig.add_gridspec(3, 1, height_ratios=[2.3, 1, 1])
a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1], sharex=a); c = fig.add_subplot(gs[2], sharex=a)
bs = beta / 1e5
a.plot(wl, bs, color=C['blue'], lw=1.0, zorder=3); zero_line(a)
a.fill_between(wl, 0, bs, where=bs > 0, color=C['blue'], alpha=0.18, lw=0)
a.fill_between(wl, 0, bs, where=bs < 0, color=C['vermilion'], alpha=0.18, lw=0)
a.set_ylim(bs.min() * 1.15, bs.max() * 1.42)                      # 顶部留出标注带，标注不压曲线
for lo, hi, lab, al in [(868, 957, '868~957 nm', 0.12), (1125, 1153, '1 125~1 153 nm', 0.07)]:
    a.axvspan(lo, hi, color=C['blue'], alpha=al, lw=0, zorder=0)
    a.text((lo + hi) / 2, bs.max() * 1.22, lab, ha='center', va='bottom', fontsize=6.5, color=C['blue'])
a.axvline(900, ymax=0.84, color=C['gray'], lw=0.6, ls='--')          # 分界线止于标注带之下
a.text(910, bs.min() * 1.05, '900 nm', fontsize=6.5, color=C['gray'], va='bottom', ha='left')
a.set_ylabel('回归系数 β(λ)/10⁵'); a.set_title('(a) 惩罚函数型线性回归的回归系数函数')
for ax_, vals, lab, ttl, fmt in [(b, r2s, '决定系数 R²', '(b) 各波段区间单独建模的精度', '{:.3f}'),
                                 (c, snr, '信噪比', '(c) 各波段区间原始光谱的信噪比', '{:.0f}')]:
    for (lo, hi), v in zip(segs, vals, strict=False):
        col = (C['blue'] if v > 0.3 else C['lightgray']) if ax_ is b else C['lightgray']
        ax_.bar(lo, v, width=hi - lo, align='edge', color=col, edgecolor='white', lw=0.6)
        ax_.text((lo + hi) / 2, max(v, 0) + 0.03 * max(vals), fmt.format(v), ha='center', va='bottom', fontsize=6.5)   # 负值标在零线上方
    ax_.set_ylabel(lab); ax_.set_title(ttl)
b.set_ylim(-0.12, 0.85); zero_line(b); c.set_ylim(0, max(snr) * 1.25)
a.tick_params(labelbottom=False); b.tick_params(labelbottom=False)
c.set_xlabel('波长/nm'); a.set_xlim(580, 1260); c.set_xticks([600, 700, 800, 900, 1000, 1100, 1200])
save(fig, 'Figure_4_combined')

np.savetxt('04outputs/beta_function.csv', np.c_[wl, beta], delimiter=',', header='wavelength_nm,beta', comments='', fmt='%.6f')
print(f"\nβ(λ) 已存 04outputs/beta_function.csv；选中 λ={lam:.3g}")
top = np.argsort(-np.abs(beta))[:400]
print("β 绝对值最大的波长区间（前 400 点聚合）：")
tw = np.sort(wl[top]); brk = np.where(np.diff(tw) > 15)[0]
for s_ in np.split(tw, brk + 1):
    if len(s_) > 15: print(f"  {s_.min():.0f}–{s_.max():.0f} nm  ({len(s_)} 点)")
print("各区间 R²：", ' '.join(f'{lo}-{hi}:{v:.3f}' for (lo, hi), v in zip(segs, r2s, strict=False)))
print("各区间信噪比：", ' '.join(f'{lo}-{hi}:{v:.0f}' for (lo, hi), v in zip(segs, snr, strict=False)))
