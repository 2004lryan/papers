# -*- coding: utf-8 -*-
"""LLH-GBID 综合评价（项目规范 第十五章）。参数在 refine-logs/01PROJECT_PROTOCOL.md 第二部分 B9 已先行锁定。

范围：
  ① 主分析 —— 表 3 的统一协议，即 项目规范 7.1 默认强制的**嵌套交叉验证**
     （K_outer=5、K_inner=5，由 02code/25nested_cv.py 产出）。本脚本读该脚本落盘的**逐样本预测**
     04outputs/25nested_cv_pred.csv，从预测值独立重算每个外层折的全部指标，并断言与
     04outputs/25nested_cv.csv 中记录的值逐位一致（不一致即报错），因此是对表 3 的一次独立复算。
     统计单位 n_primary = K_outer = 5（7.2）；多种子只用于排序稳定性检查，不并入统计单位。
  ② 稳健性 —— 划分方式：读 04outputs/22split_scheme.csv（22split_scheme.py 产出的重复留出法
     随机 70%/75%、SPXY 75%、Kennard-Stone 75%），与嵌套 CV 并列比较排序是否改变。

指标层（B9 §9.1，四个维度各一项，两级等权）：
  D1 整体拟合  R²    效益型  I=1  L=0（理论无技能）
  D2 平均误差  MAE   成本型  I=0  L=均值预测基准模型在同一测试集上的 MAE
  D3 大误差风险 P90  成本型  I=0  L=同一基准模型的 P90            （P90 = 绝对误差的 90% 分位数）
  D4 系统偏差  bias  目标型  T=0  τ±=0.25·s_y(train)  δ±=0
  RMSE 与 RPD 在固定测试集上与 R² 互为严格单调变换，按 B9 不纳入综合评价，仅在表 3 照常报告。

产出（04outputs/）：
  23gbid_raw.csv         原始指标表（稿件表 5）：四种划分 × 各方法 × R²/MAE/P90/bias + GBIDS
  23gbid_main.csv        综合结果表（稿件表 6）：Pass/ū/C_B/C_W/GBID/GBIDS/平均排名/第一名频率
  23gbid_sensitivity.csv 敏感性分析总表（稿件表 7）
  23gbid_dimensions.csv  各方法在 4 个维度上的效用 U_d
  23gbid_diag.csv        截断前诊断与指标相关性诊断
只读 05data。
"""
import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, kendalltau, rankdata
import scipy.stats as stt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys; sys.path.insert(0, '02code')
from export_utils import (compute_utility, compute_target_utility, compute_gbid,
                          gbid_sensitivity_q, gbid_sensitivity_weights, gbid_bootstrap)

# 复用 16unified_comparison.py 的数据、基系统、候选网格与拟合函数（切到 METHODS 定义之前）
with open('02code/16unified_comparison.py') as _src: exec(_src.read().split("METHODS=['PLSR'")[0])

METHODS = ['PLSR', 'SVR', 'RF', '1D-CNN', 'FPCR', 'PFLR']
if os.environ.get('NCV_METHODS'): METHODS = os.environ['NCV_METHODS'].split(',')   # 分段冒烟用
SEEDS = [0, 1, 2, 3, 4]
ALPHA_TAU = 0.25                       # B9 §9.1：τ± = 0.25 · s_y(train)
MNAMES = ['R2', 'MAE', 'P90', 'BIAS']
DIMS = ['D1 整体拟合', 'D2 平均误差', 'D3 大误差风险', 'D4 系统偏差']
DIM_GROUPS = [np.array([0]), np.array([1]), np.array([2]), np.array([3])]
W_MAIN = np.full(4, 0.25)
V_MAIN = [np.array([1.0]) for _ in range(4)]


def raw_metrics(ytr, yte, p):
    """原始指标 + 效用化基准量。与 22split_scheme.py 的 metrics() 同一口径。"""
    e = np.asarray(p) - yte; ae = np.abs(e)
    rm = float(np.sqrt(mean_squared_error(yte, p))); sdv = float(np.std(yte))
    be = np.abs(float(np.mean(ytr)) - yte)
    return dict(R2=float(r2_score(yte, p)), RMSE=rm, MAE=float(mean_absolute_error(yte, p)),
                P90=float(np.quantile(ae, 0.9)), BIAS=float(np.mean(e)), RPD=sdv / rm,
                base_MAE=float(be.mean()), base_P90=float(np.quantile(be, 0.9)),
                s_y_train=float(np.std(ytr)), s_y_test=sdv)


# ===================== ① 主分析：嵌套 CV，从逐样本预测独立重算全部指标 =====================
PRIMARY_SEED = 20041210                       # B1 Formal 集合的第 1 个；其余种子仅作排序稳定性检查
ncv = pd.read_csv('04outputs/25nested_cv.csv')
pred = pd.read_csv('04outputs/25nested_cv_pred.csv')
seeds_done = sorted(ncv.seed.unique())
print(f"【主分析】嵌套交叉验证 K_outer=5 / K_inner=5；已完成种子 {seeds_done}，"
      f"主分析取 seed={PRIMARY_SEED}\n")
assert PRIMARY_SEED in seeds_done, f"主种子 {PRIMARY_SEED} 尚未跑完"

rows, bad = [], 0
for m in METHODS:
    for k in sorted(ncv.fold.unique()):
        q = pred[(pred.seed == PRIMARY_SEED) & (pred.method == m) & (pred.fold == k)]
        rec = ncv[(ncv.seed == PRIMARY_SEED) & (ncv.method == m) & (ncv.fold == k)].iloc[0]
        ite = q.idx.values; itr = np.setdiff1d(np.arange(len(y)), ite)
        mm = raw_metrics(y[itr], q.y.values, q.pred.values)          # 从预测值独立重算
        for kk in ['R2', 'RMSE', 'MAE', 'P90', 'BIAS', 'RPD', 'base_MAE', 'base_P90', 's_y_train', 's_y_test']:
            if abs(mm[kk] - float(rec[kk])) > 1e-9:
                bad += 1; print(f"  不一致：{m} fold={k} {kk} 本脚本 {mm[kk]:.6f} 25 脚本 {float(rec[kk]):.6f}")
        rows.append(dict(scheme='嵌套CV', rep=k, method=m, picked=str(rec.picked), **mm))
        print(f"  {m:<7} fold={k}  R²={mm['R2']:.3f}  RMSE={mm['RMSE']:.3f}  MAE={mm['MAE']:.3f}  "
              f"P90={mm['P90']:.3f}  bias={mm['BIAS']:+.3f}")
print(f"\n与 25nested_cv.csv 逐位对账：{'全部一致' if bad == 0 else str(bad)+' 处不一致'}")
assert bad == 0
main = pd.DataFrame(rows)
N_TR = int(np.median([len(y) - len(pred[(pred.seed == PRIMARY_SEED) & (pred.method == METHODS[0])
                                        & (pred.fold == k)]) for k in sorted(ncv.fold.unique())]))
N_TE = len(y) - N_TR
print(f"  外层折规模：训练 {N_TR}、测试 {N_TE}（Nadeau–Bengio 修正项 n_te/n_tr = {N_TE/N_TR:.3f}）")

# 并入 22split_scheme.py 的另外三种划分方式
COLS = ['scheme', 'rep', 'method', 'picked', 'R2', 'RMSE', 'MAE', 'P90', 'BIAS', 'RPD',
        'base_MAE', 'base_P90', 's_y_train', 's_y_test']
oth = pd.read_csv('04outputs/22split_scheme.csv')
# 22split_scheme.py 也重算了一份嵌套 CV 的行，用途是用另一条代码路径与 25nested_cv.py 逐位对账
# （该断言在 22split 内部执行且已通过）。主分析的嵌套 CV 数据一律取 main，即从 25nested_cv_pred.csv
# 的折外预测重算的那一份；此处剔除 22split 的同名 scheme，以免同一个外层折被计入两次。
oth = oth[oth.scheme != '嵌套CV'][COLS]
main = main[COLS]
allraw = pd.concat([main, oth], ignore_index=True)
allraw.to_csv('04outputs/23gbid_rawmetrics.csv', index=False, encoding='utf-8-sig')


# ===================== ② 效用化（B9 §9.1）=====================
def utilities(df, methods, worst_case=False):
    """返回 u (n_models, 4, n_reps) 与截断前 u_tilde。worst_case=True 时基准取候选集最差值（仅敏感性分析用）。"""
    reps = sorted(df.rep.unique())
    u = np.zeros((len(methods), 4, len(reps))); ut = np.zeros_like(u)
    for k, rp in enumerate(reps):
        d = df[df.rep == rp].set_index('method').loc[methods]
        if worst_case:                                        # 候选集最差值（15.11③，仅敏感性分析）
            L_r2 = float(d.R2.min()); L_mae = float(d.MAE.max()); L_p90 = float(d.P90.max())
            tau = max(float(np.abs(d.BIAS).max()), 1e-10)
        else:                                                 # B9 主分析基准
            L_r2 = 0.0; L_mae = float(d.base_MAE.iloc[0]); L_p90 = float(d.base_P90.iloc[0])
            tau = ALPHA_TAU * float(d.s_y_train.iloc[0])
        a = compute_utility(d.R2.values, I=1.0, L=L_r2, direction='benefit')
        b = compute_utility(d.MAE.values, I=0.0, L=L_mae, direction='cost')
        c = compute_utility(d.P90.values, I=0.0, L=L_p90, direction='cost')
        e = compute_target_utility(d.BIAS.values, T=0.0, tau_minus=tau, tau_plus=tau)
        for j, r in enumerate([a, b, c, e]):
            u[:, j, k] = r['u']; ut[:, j, k] = r['u_tilde']
    return u, ut


u_all, ut_all = utilities(main, METHODS)
u_mean = u_all.mean(axis=2)                                   # (n_models, 4)

# 截断前诊断（15.11⑥）
diag = []
for j, nm in enumerate(MNAMES):
    flat = ut_all[:, j, :].ravel()
    diag.append(dict(指标=nm, 下截断比例=float((flat <= 0).mean()), 上截断比例=float((flat >= 1).mean()),
                     截断前最小=float(flat.min()), 截断前最大=float(flat.max())))
diag = pd.DataFrame(diag)
print("\n【截断前诊断】（任一截断比例 > 50% 须在敏感性分析中补充调整结果）")
print(diag.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
assert (diag.下截断比例 <= 0.5).all(), "存在下截断比例超 50% 的指标"
assert (diag.上截断比例 <= 0.5).all(), "存在上截断比例超 50% 的指标"

# 指标相关性诊断（15.11①）：候选方法 × 重复 矩阵上的 Spearman
mat = np.column_stack([main.pivot_table(index=['rep'], columns='method', values=c)
                       .loc[:, METHODS].values.ravel() for c in MNAMES])
rho = spearmanr(mat).statistic
print("\n【指标相关性诊断】Spearman ρ（候选方法 × 5 次划分）")
print(pd.DataFrame(rho, index=MNAMES, columns=MNAMES).to_string(float_format=lambda v: f'{v:+.3f}'))
hi = [(MNAMES[i], MNAMES[j], rho[i, j]) for i in range(4) for j in range(i + 1, 4) if abs(rho[i, j]) > 0.9]
print("  |ρ|>0.9 的指标对：", hi if hi else "无")

# ===================== ③ 主分析 GBID + 统计推断（B9 §9.3/§9.5）=====================
g = compute_gbid(u_mean, W_MAIN, V_MAIN, DIM_GROUPS, q_B=2.0, q_W=2.0)
bs = gbid_bootstrap(u_all, W_MAIN, V_MAIN, DIM_GROUPS, n_bootstrap=2000, ci_level=0.95, seed=42)
u_R2 = u_mean[:, 0]; U_D1 = g['U_d'][:, 0]
Pass = ((u_R2 >= 0.5) & (U_D1 >= 0.5)).astype(int)            # B9 §9.3 软否决
main_rank = rankdata(np.where(Pass == 1, 0, 1) * 10 + g['gbid'], method='min')

tab6 = pd.DataFrame(dict(方法=METHODS, Pass=Pass, u_bar=g['u_bar'], C_B=g['C_B'], C_W=g['C_W'],
                         GBID=g['gbid'], GBIDS=g['gbids'],
                         GBIDS_sd=bs['gbids_sd'], GBIDS_ci_lo=bs['gbids_ci_lo'], GBIDS_ci_hi=bs['gbids_ci_hi'],
                         平均排名=bs['mean_rank'], 第一名频率=bs['first_place_pct'], 主排名=main_rank))
tab6.to_csv('04outputs/23gbid_main.csv', index=False, encoding='utf-8-sig')
# 各方法各维度的效用 U_d（正文引用其取值范围，须可追溯）
ud = pd.DataFrame(g['U_d'], columns=DIMS); ud.insert(0, '方法', METHODS)
ud.to_csv('04outputs/23gbid_dimensions.csv', index=False, encoding='utf-8-sig')
print("\n【各维度效用 U_d】")
print(ud.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print("\n【综合结果表（稿件表 6）】主分析 q_B=q_W=2，5 次划分，percentile bootstrap 2000 次")
print(tab6.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

# 配对差（7.2）：Nadeau–Bengio 修正 t 检验给 p，fold 层级 cluster bootstrap 给 CI（B9 §9.5）
gb_rep = np.array([compute_gbid(u_all[:, :, k], W_MAIN, V_MAIN, DIM_GROUPS)['gbid'] for k in range(u_all.shape[2])])
rho_nb = N_TE / N_TR                                          # Nadeau–Bengio：σ²_corr = (1/J + n_te/n_tr)·σ̂²
best = int(np.argmin(g['gbid'])); pairs = []
rng = np.random.default_rng(42)
for i, m in enumerate(METHODS):
    if i == best: continue
    d = gb_rep[:, i] - gb_rep[:, best]; J = len(d)
    se = float(np.sqrt(d.var(ddof=1) * (1 / J + rho_nb)))
    tstat = float(d.mean() / se) if se > 0 else np.inf
    pval = float(2 * stt.t.sf(abs(tstat), df=J - 1))
    bsd = np.array([rng.choice(d, J, replace=True).mean() for _ in range(2000)])   # cluster bootstrap（折为单位）
    lo, hi_ = np.percentile(bsd, [2.5, 97.5])
    pairs.append(dict(对比=f"{m} − {METHODS[best]}", 差值=d.mean(), NB标准误=se, t=tstat,
                      CI下界=lo, CI上界=hi_, p=pval,
                      Cohen_d=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.inf))
pairs = pd.DataFrame(pairs).sort_values('p').reset_index(drop=True)
# Holm：按 p 升序，第 i 个乘以 (m−i)，再取累积最大以保证单调
pairs['Holm_p'] = np.maximum.accumulate(np.minimum((len(pairs) - np.arange(len(pairs))) * pairs.p.values, 1.0))
EPS = 0.02                                                    # B9 §9.5 实际等价区间
_sig = pairs.Holm_p < 0.05
_within = (pairs.CI下界.abs() <= EPS) & (pairs.CI上界.abs() <= EPS)
pairs['结论'] = np.where(~_sig, '综合表现相近（未达显著）',
                np.where(_within, f'差异显著但整个 CI 落在等价区间内（|Δ|≤ε={EPS}）',
                np.where(pairs.差值.abs() <= EPS, f'差异显著，点估计小于 ε={EPS} 而 CI 上界超出', '显著劣于最优')))
print(f"\n【配对差（相对综合最优 {METHODS[best]}）】GBID 越小越好；"
      f"p 由 Nadeau–Bengio 修正 t 检验（df={u_all.shape[2]-1}）给出，CI 由 2 000 次 fold 层级 cluster bootstrap 给出，Holm 校正")
print(pairs.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# 贝叶斯相关 t 检验（7.2 指定的多模型比较主方法；ROPE 取与上面同一个 ε）
# CorrelatedTTest 正是 Nadeau–Bengio 的贝叶斯版本：它把 k 折交叉验证的折间相关直接建进后验，
# 相关系数取 ρ = n_te/(n_te+n_tr) = 1/K_outer。相对 p 值的好处是能直接回答
# 「两法之差落在实际等价区间内的概率有多大」，而不是只能说「未达显著」。
from baycomp import CorrelatedTTest
bay = []
for i, m in enumerate(METHODS):
    if i == best: continue
    # probs(x, y, rope) 的第一项是「x 大于 y 超过 rope」的概率，故传入 −GBID 使「大」即「好」
    p_best, p_eq, p_m = CorrelatedTTest.probs(-gb_rep[:, best], -gb_rep[:, i], rope=EPS)
    bay.append({'对比': f"{m} − {METHODS[best]}", f'P({METHODS[best]}优)': p_best,
                f'P(等价|Δ|<{EPS})': p_eq, 'P(该法优)': p_m})
bay = pd.DataFrame(bay)
print(f"\n【贝叶斯相关 t 检验】ROPE=±{EPS}（B9 §9.5 的实际等价区间），折间相关按 K 折交叉验证取 1/K_outer")
print(bay.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
pairs = pairs.merge(bay, on='对比', how='left')
pairs.to_csv('04outputs/23gbid_paired.csv', index=False, encoding='utf-8-sig')

# ===================== ④ 四项稳健性分析（B9 §9.4）=====================
sq = gbid_sensitivity_q(u_mean, W_MAIN, V_MAIN, DIM_GROUPS)
sw = gbid_sensitivity_weights(u_mean, W_MAIN, V_MAIN, DIM_GROUPS, n_samples=2000, kappa_B=10.0, kappa_W=10.0, seed=42)

# 权重 ±20% 扰动
pert = []
for d0 in range(4):
    for f in (0.8, 1.2):
        W2 = W_MAIN.copy(); W2[d0] *= f; W2 /= W2.sum()
        pert.append(rankdata(compute_gbid(u_mean, W2, V_MAIN, DIM_GROUPS)['gbid'], method='min'))
pert_a = np.array(pert)

# 基准敏感性：固定基准 vs 候选集最差值
u_worst, _ = utilities(main, METHODS, worst_case=True)
r_worst = rankdata(compute_gbid(u_worst.mean(2), W_MAIN, V_MAIN, DIM_GROUPS)['gbid'], method='min')

# 相关性处理：等权 / 相关性折扣 / Mahalanobis
rho_bar = np.array([np.mean([abs(rho[j, k]) for k in range(4) if k != j]) for j in range(4)])
W_disc = (1 / (1 + rho_bar)); W_disc /= W_disc.sum()
r_disc = rankdata(compute_gbid(u_mean, W_disc, V_MAIN, DIM_GROUPS)['gbid'], method='min')
Sig = np.corrcoef(mat, rowvar=False) + 1e-8 * np.eye(4); Wh = np.diag(np.sqrt(W_MAIN))
gm = np.array([float(np.sqrt(max((1 - u) @ Wh @ np.linalg.inv(Sig) @ Wh @ (1 - u), 0))) for u in u_mean])
r_maha = rankdata(gm, method='min')
r_corr = np.vstack([main_rank, r_disc, r_maha])

tab7 = pd.DataFrame(dict(
    方法=METHODS, 主排名=main_rank, q敏感性平均排名=sq['mean_rank'],
    权重扰动第一名概率=sw['first_place_pct'],
    权重扰动排名区间=[f"{int(min(sw['rankings'][i].min(), pert_a[:, i].min()))}~"
                     f"{int(max(sw['rankings'][i].max(), pert_a[:, i].max()))}" for i in range(len(METHODS))],
    基准敏感性排名区间=[f"{int(min(main_rank[i], r_worst[i]))}~{int(max(main_rank[i], r_worst[i]))}" for i in range(len(METHODS))],
    相关性处理一致性=['一致' if len(set(r_corr[:, i])) == 1 else f"{int(r_corr[:, i].min())}~{int(r_corr[:, i].max())}" for i in range(len(METHODS))],
    重采样第一名频率=bs['first_place_pct']))
tab7.to_csv('04outputs/23gbid_sensitivity.csv', index=False, encoding='utf-8-sig')
print("\n【敏感性分析总表（稿件表 7）】")
print(tab7.to_string(index=False, float_format=lambda v: f'{v:.1f}'))
print(f"\n  q 配置间 Kendall τ 最小值：{sq['kendall_tau'][np.triu_indices(4,1)].min():.3f}")
print(f"  权重扰动与主排序的平均 Kendall τ：{sw['mean_kendall']:.3f}")
print(f"  基准敏感性（候选集最差值）与主排序 Kendall τ：{kendalltau(main_rank, r_worst).statistic:.3f}")
print(f"  相关性折扣权重 W_d = {np.round(W_disc,3)}；与主排序 Kendall τ：{kendalltau(main_rank, r_disc).statistic:.3f}"
      f"；Mahalanobis 与主排序 Kendall τ：{kendalltau(main_rank, r_maha).statistic:.3f}")

# ===================== ⑤ 四种划分方式下的 GBIDS（稿件表 5）=====================
M4 = ['PLSR', 'SVR', 'FPCR', 'PFLR']
SCHEMES = ['嵌套CV', '随机70%', '随机75%', 'SPXY75%', 'KS75%']
tab5 = []
for scheme in SCHEMES:
    df = allraw[allraw.scheme == scheme]
    meth = METHODS if scheme == '嵌套CV' else M4
    uu, _ = utilities(df, meth)
    gg = compute_gbid(uu.mean(2), W_MAIN, V_MAIN, DIM_GROUPS)
    rk = rankdata(gg['gbid'], method='min')
    for i, m in enumerate(meth):
        a = df[df.method == m]
        tab5.append(dict(scheme=scheme, method=m, n=len(a),
                         R2=a.R2.mean(), R2_sd=a.R2.std(ddof=0), MAE=a.MAE.mean(), MAE_sd=a.MAE.std(ddof=0),
                         P90=a.P90.mean(), P90_sd=a.P90.std(ddof=0), BIAS=a.BIAS.mean(), BIAS_sd=a.BIAS.std(ddof=0),
                         GBID=gg['gbid'][i], GBIDS=gg['gbids'][i], rank=int(rk[i])))
tab5 = pd.DataFrame(tab5); tab5.to_csv('04outputs/23gbid_raw.csv', index=False, encoding='utf-8-sig')
diag.to_csv('04outputs/23gbid_diag.csv', index=False, encoding='utf-8-sig')
print("\n【原始指标表（稿件表 5）】嵌套 CV 及四种重复留出/确定性划分下的原始指标与 0—100 综合指数得分 GBIDS")
print(tab5.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
for scheme in SCHEMES:
    s_ = tab5[tab5.scheme == scheme].sort_values('rank')
    print(f"  {scheme:<8} 按 GBIDS 排序：{' > '.join(s_.method)}")
# ===================== ⑥ 种子稳定性（多种子只降低方差估计，不并入统计单位）=====================
if len(seeds_done) > 1:
    print("\n【种子稳定性】各已完成种子单独按 GBIDS 排序（n_primary 仍为 K_outer=5）")
    stab = []
    for sd_ in seeds_done:
        rr = []
        for m in METHODS:
            for k in sorted(ncv.fold.unique()):
                q = pred[(pred.seed == sd_) & (pred.method == m) & (pred.fold == k)]
                if not len(q): break
                itr = np.setdiff1d(np.arange(len(y)), q.idx.values)
                rr.append(dict(rep=k, method=m, **raw_metrics(y[itr], q.y.values, q.pred.values)))
        dfr = pd.DataFrame(rr)
        if dfr.method.nunique() < len(METHODS): 
            print(f"  seed={sd_} 尚未跑全六个方法，跳过"); continue
        uu, _ = utilities(dfr, METHODS)
        gg = compute_gbid(uu.mean(2), W_MAIN, V_MAIN, DIM_GROUPS)
        order = [METHODS[i] for i in np.argsort(gg['gbid'])]
        print(f"  seed={sd_}: {' > '.join(order)}   GBIDS {np.round(gg['gbids'][np.argsort(gg['gbid'])],1)}")
        stab.append(dict(seed=sd_, 排序=' > '.join(order),
                         **{METHODS[i]: gg['gbids'][i] for i in range(len(METHODS))}))
    if stab: pd.DataFrame(stab).to_csv('04outputs/23gbid_seedstability.csv', index=False, encoding='utf-8-sig')

print("\n已写出 23gbid_rawmetrics.csv / 23gbid_raw.csv / 23gbid_main.csv / 23gbid_sensitivity.csv / "
      "23gbid_paired.csv / 23gbid_diag.csv")
