# -*- coding: utf-8 -*-
"""修订稿引用的数字：一律从产物 CSV 与实验日志现读，稿件构建脚本（13、14）只从本模块取数。

为什么单列一个模块：返修稿的数字来自 21、25、27、36、38、44~49、51、54~61 等脚本的产物，
摘要、正文、图注、结论与附录会在多处引用同一个量。旧稿把数字手抄进两个构建脚本，
出过「产物重跑而正文未同步」的事故。
这里每个量只算一次；正文里凡是描述数据模式的句子（如「N=50~200 时交叉验证选参的 FPLR 中位数最高」），
都在本模块用断言守住，数据一变而句子不再成立时直接报错，不让构建脚本生成与数据矛盾的稿件。
全部数字取协议 B10.10 清洗后的数据；清洗前的对照量只在附录 A9 出现，取自 59 从提交 2331884 导出的对照表。

口径：
  表 1、配对检验  47 的 table1、paired_tests 产物（主分析为 B 样条 L=80），经 results47.resolve() 取得：
                  只认有效标记，正式结果优先，其次部分结果；部分结果里若有未通过核对的方法即停下
                  （45 的产物也在该标记的输入哈希之内，下文直接读 45 预测文件的几处因此同受约束）
  主分析弱折      61rebuttal_mainfold_summary.csv、61rebuttal_mainfold_spectra.csv、61rebuttal_mainfold_folds.csv（事后描述）
  小样本          49rebuttal_smallsample_summary.csv（中位数、最低值、R²<0 次数）、21smallsample_repeats.csv（逐次配对）、
                  18svr_aligned.csv（SVR 标准化口径）、05logs/18svr_align_and_sensitivity.log（N=80 逐个候选）、
                  05logs/20cv_selection_mechanism.log（交叉验证选中的 K 与误差波动）
  选参对策        56rebuttal_selection_*.csv（B10.8）、57rebuttal_guard_*.csv（B10.9）、58rebuttal_guard_posthoc_summary.csv
  跨批次          46rebuttal_recal_summary.csv、46rebuttal_recal_source.csv、46rebuttal_recal_batches.csv、49rebuttal_recal_range.csv
  系数函数        48rebuttal_beta_windows.csv、49rebuttal_beta_band.csv
  预处理与复核    27prep_nested.csv、36prep_leakage.csv、38nested_prep.csv、44fully_nested*.csv、25nested_cv.csv、
                  45rebuttal_ncv_*_pred.csv（种子 20041210 的合并预测）、51rebuttal_lams_summary.csv（展开平滑参数 λs 的敏感性）
  综合评价        55rebuttal_A5_summary.csv、55rebuttal_A5_pairs.csv、55rebuttal_A5_diag.csv（47 正式结果的 25 折 GBIDS）
  网格            grids.py、47 的 boundary 产物、54 的第三轮产物（稿中称初始网格、主网格、外移网格，即协议的第一、二、三轮）
  数据清洗        59data_clean_summary.csv、59data_clean_list.csv、59raw_reference.csv（清洗前的对照量）

运行方式（单独运行时打印全部数字，供核对）:
    cd <项目根目录>
    python3 02code/ms_numbers.py

只读 04outputs 与 05logs，不写任何文件。
"""
import math
import os
import json
from typing import Any
import re
import ast
from pathlib import Path

import pandas as pd
import numpy as np

from results47 import resolve as resolve47
from results54 import resolve as resolve54, out_path as out54

OUT = '04outputs'
LOGS = '05logs'
MINUS = '−'
ALL6 = ['SVR', 'FPLR', 'FPCR', 'PLSR', 'RF', '1D-CNN']
_READS: list[str] = []                              # 本模块读过的产物与日志，末尾统一核对生成时间


def rd(path, **kw):
    """读产物 CSV 并记下路径。"""
    _READS.append(str(path))
    return pd.read_csv(path, **kw)


def thou(n):
    """四位及以上整数每三位空一格，如 1 500。"""
    return f'{int(n):,}'.replace(',', ' ')


def fm(v, nd=3):
    """数字按位数格式化，负号用 U+2212。"""
    return f'{v:.{nd}f}'.replace('-', MINUS)


def rg(a, b, nd=3):
    """区间「a~b」，两端各自带符号。"""
    return f'{fm(a, nd)}~{fm(b, nd)}'


def up(v, nd=2):
    """「不超过 x」一类上界：向上取整到 nd 位，写出的界不小于真实值。"""
    q = 10 ** nd
    return fm(math.ceil(v * q - 1e-9) / q, nd)


def dn(v, nd=2):
    """「x 以上」一类下界：向下取整到 nd 位，写出的界不大于真实值。"""
    q = 10 ** nd
    return fm(math.floor(v * q + 1e-9) / q, nd)


def pct2(v):
    return f'{v:.2f}'.replace('-', MINUS)


def pct1(x: float) -> str:
    """比例写成百分数、保留一位小数，按四舍五入（格式化默认逢五取偶，会把 16.25 写成 16.2）。"""
    return f"{math.floor(x * 1000 + 0.5 + 1e-9) / 10:.1f}"


def pct0(x: float) -> str:
    """比例写成整数百分数，按四舍五入。"""
    return f"{math.floor(x * 100 + 0.5 + 1e-9):d}"


def join_cn(items):
    """「a、b 与 c」。"""
    items = [str(x) for x in items]
    return items[0] if len(items) == 1 else '、'.join(items[:-1]) + ' 与 ' + items[-1]


# 清洗后的样品数（协议 B10.10），全文的样品数核对以它为准
KEPT_N = int(rd(f'{OUT}/59data_clean_summary.csv', dtype={'year': str}).set_index('year').loc['合计', 'kept'])

# ============================ 表 1 与配对检验（47）============================
R47 = resolve47()
T1 = rd(R47.path('table1'))
MAIN = T1[T1.analysis.str.startswith('主分析')].set_index('method')
FOUR = T1[T1.analysis.str.startswith('敏感性')].set_index('method')
DONE = [m for m in ALL6 if m in MAIN.index]
PENDING = [m for m in ALL6 if m not in MAIN.index]          # RF、1D-CNN 未完成时非空，稿件据此标【待补】
assert not R47.with_status('invalid'), f'47 的输入有未通过核对的项，不得据此生成稿件：{R47.with_status("invalid")}'
assert R47.final == (not PENDING), f'47 产物为{"正式" if R47.final else "部分"}结果，而表 1 未完成的方法为 {PENDING}'
assert set(DONE) >= {'SVR', 'FPLR', 'FPCR', 'PLSR'}, DONE
assert (MAIN.n_folds == 25).all()
assert (MAIN.n_seeds == 5).all()
ORDER = list(MAIN.R2.sort_values(ascending=False).index)          # 表 1 的行序：R² 均值由高到低


def acc(m, col='R2', nd=3, src=MAIN):
    return f'{src.loc[m, col]:.{nd}f}±{src.loc[m, col + "_sd"]:.{nd}f}'


PT = rd(R47.path('paired_tests'))
FAM_MAIN = int(PT.loc[PT.family == '主分析', 'family_size'].iloc[0])     # 六法齐备后应为 15
FAM_FINAL = FAM_MAIN == 15
_PTM = PT[PT.family == '主分析']
MAIN_NSIG = int((_PTM.holm_p < 0.05).sum())     # 主分析各对中 Holm 显著的对数


def pair(name, family='主分析'):
    r = PT[(PT.family == family) & (PT.pair == name)]
    assert len(r) == 1, (family, name)
    return r.iloc[0]


def dci(name, family='主分析'):
    """「差值（95% 置信区间 lo~hi）」中的两部分。"""
    r = pair(name, family)
    return fm(r['diff']), rg(r.ci_lo, r.ci_hi)


P_FL = pair('FPLR − PLSR')
P_FC = pair('FPCR − PLSR')
P_SL = pair('SVR − FPLR')
P_SC = pair('SVR − FPCR')
P_SP = pair('SVR − PLSR')
P_LC = pair('FPLR − FPCR')
B_L = pair('FPLR：B 样条 − 傅里叶', '基函数敏感性')
B_C = pair('FPCR：B 样条 − 傅里叶', '基函数敏感性')
# 3.1 节：SVR 与 PLSR 之差 Holm 校正后显著；函数型模型与 PLSR、SVR 以及两种函数型模型之间、两种基之间均未检出差异
assert P_SP.holm_p < 0.05, P_SP.holm_p
assert P_SP['diff'] > 0, P_SP['diff']
for _r in (P_FL, P_FC, P_SL, P_SC, P_LC, B_L, B_C):
    assert _r.holm_p >= 0.05, (_r.pair, _r.holm_p)
assert P_FL.ci_lo > 0, '正文写「未校正区间不含 0」'
assert P_FC.ci_lo > 0, '正文写「未校正区间不含 0」'
assert P_SL.ci_lo < 0 < P_SL.ci_hi
assert P_SC.ci_lo < 0 < P_SC.ci_hi
assert B_L.ci_lo < 0 < B_L.ci_hi, '3.1 节写基函数一族中只有 FPCR 的未校正区间不含 0'
assert B_C.ci_hi < 0, '3.1 节写基函数一族中只有 FPCR 的未校正区间不含 0'
assert ORDER[0] == 'SVR', '正文写 SVR 的 R² 最高'
assert ORDER[1:4] == ['FPLR', 'FPCR', 'PLSR'], ORDER
# 基函数敏感性：两对差值均不拒绝正态（2.5 节的规则下不另报 Wilcoxon）
assert (PT.loc[PT.family == '基函数敏感性', 'shapiro_p'] >= 0.05).all(), PT.loc[PT.family == '基函数敏感性', ['pair', 'shapiro_p']]
# 2.5 节的规则：差值 Shapiro-Wilk p<0.05 时另报 Wilcoxon；产物须为这些行给出补充检验。
_main_nonnormal = _PTM[_PTM.shapiro_p < 0.05]
assert _main_nonnormal.wilcoxon_p.notna().all(), _main_nonnormal[['pair', 'shapiro_p', 'wilcoxon_p']]
MAIN_NONNORMAL = list(_main_nonnormal.pair)
MAIN_NONNORMAL_WMAX = float(_main_nonnormal.wilcoxon_p.max())
# 3.1 节写「差值非正态的各对补充 Wilcoxon 检验 p 均小于 0.001」
assert MAIN_NONNORMAL_WMAX < 0.001, _main_nonnormal[['pair', 'shapiro_p', 'wilcoxon_p']]
# 非正态的各对全部涉及函数型模型（3.1 节以此衔接弱折一段）
assert all(('FPLR' in p) or ('FPCR' in p) for p in MAIN_NONNORMAL), MAIN_NONNORMAL
SIG_MAIN = list(_PTM.loc[_PTM['结论'] == '差异显著', 'pair'])
assert (_PTM.loc[_PTM.pair.isin(SIG_MAIN), 'diff'] > 0).all(), '显著对的差值方向须与「高于」一致（配对名前者减后者）'


def sig_phrase(pairs):
    """把显著对写成「A 高于 B、C 与 D，E、F 高于 G」：按较高一方分组，较低一方相同的高方合并；次序按表 1。"""
    win: dict[str, list[str]] = {}
    for p in pairs:
        a, b = [s.strip() for s in p.split('−')]
        win.setdefault(a, []).append(b)
    groups: dict[tuple, list[str]] = {}
    for a in sorted(win, key=ORDER.index):
        groups.setdefault(tuple(sorted(win[a], key=ORDER.index)), []).append(a)
    return '，'.join(f'{join_cn(ws)} 高于 {join_cn(ls)}' for ls, ws in groups.items())


SIG_TEXT = sig_phrase(SIG_MAIN)

# ============================ 附录 A5 LLH-GBID（55）===========================
A5_SUMMARY = rd(f'{OUT}/55rebuttal_A5_summary.csv').set_index('method')
A5_PAIRS = rd(f'{OUT}/55rebuttal_A5_pairs.csv')
A5_DIAG = rd(f'{OUT}/55rebuttal_A5_diag.csv').set_index('metric')
A5_ORDER = list(A5_SUMMARY.GBIDS_mean.sort_values(ascending=False).index)
A5_SIG = A5_PAIRS[A5_PAIRS['结论'] == '差异显著'].copy()
A5_NONSIG = A5_PAIRS[A5_PAIRS['结论'] == '未检出差异'].copy()
A5_HIGH = str(A5_SUMMARY.GBIDS_mean.idxmax())
A5_LOW = str(A5_SUMMARY.GBIDS_mean.idxmin())
A5_DIAG_OVER50 = A5_DIAG[(A5_DIAG.lower_clip_ratio > 0.5) | (A5_DIAG.upper_clip_ratio > 0.5)].copy()
A5_DIAG_ALLZERO = bool(((A5_DIAG.lower_clip_ratio == 0) & (A5_DIAG.upper_clip_ratio == 0)).all())
assert set(A5_SUMMARY.index) == set(ALL6), A5_SUMMARY.index
assert (A5_SUMMARY.n_folds == 25).all()
assert (A5_PAIRS.family == 'A5 GBIDS').all()
assert (A5_PAIRS.family_size == 15).all()
assert len(A5_PAIRS) == 15
assert len(A5_SIG) + len(A5_NONSIG) == 15
assert (A5_SIG['diff'] > 0).all(), 'A5 显著对的差值方向须与配对名一致'
# A5 与 R² 主分析逐对判定的异同：两表的 15 对同名，逐对比较 Holm 判定
_a5v = A5_PAIRS.set_index('pair')['结论']
_r2v = _PTM.set_index('pair')['结论']
assert set(_a5v.index) == set(_r2v.index), set(_a5v.index) ^ set(_r2v.index)
A5_ONLY = [p for p in A5_PAIRS.pair if _a5v[p] == '差异显著' and _r2v[p] == '未检出差异']   # GBIDS 显著而 R² 未检出
R2_ONLY = [p for p in A5_PAIRS.pair if _a5v[p] == '未检出差异' and _r2v[p] == '差异显著']   # R² 显著而 GBIDS 未检出
A5_SAME = 15 - len(A5_ONLY) - len(R2_ONLY)


def a5_score(method):
    """附录 A5 的「GBIDS 均值±标准差（Pass 次数/25）」格式。"""
    row = A5_SUMMARY.loc[method]
    return f'{fm(row.GBIDS_mean, 2)}±{fm(row.GBIDS_sd, 2)}（{int(row.Pass_n)}/25）'


def _a5_wilcoxon(row):
    """2.5 节的报告规则：差值 Shapiro-Wilk p<0.05 时另报 Wilcoxon 符号秩检验。"""
    if not row.shapiro_p < 0.05:
        return ''
    p = 'p<0.001' if row.wilcoxon_p < 0.001 else f'p={row.wilcoxon_p:.3f}'
    return f'；差值非正态，Wilcoxon 符号秩检验 {p}'


def a5_sig_text(row):
    """附录 A5 显著对的「名称：差值（95% CI）」格式。"""
    return f'{row.pair}：{fm(row["diff"], 2)}（95% 置信区间 {rg(row.ci_lo, row.ci_hi, 2)}{_a5_wilcoxon(row)}）'


def a5_nonsig_text(row):
    """附录 A5 未检出差异对的「名称 差值（95% CI）」格式（B10.2：未检出差异也给差值与区间）。"""
    return f'{row.pair} {fm(row["diff"], 2)}（{rg(row.ci_lo, row.ci_hi, 2)}{_a5_wilcoxon(row)}）'


def a5_vs_r2(short=False):
    """GBIDS 与 R² 两种口径逐对判定的异同。"""
    parts = []
    if A5_ONLY:
        parts.append(f"{'、'.join(A5_ONLY)} 这 {len(A5_ONLY)} 对在 GBIDS 上差异显著而在 R² 上未检出差异")
    if R2_ONLY:
        parts.append(f"{'、'.join(R2_ONLY)} 这 {len(R2_ONLY)} 对在 R² 上差异显著而在 GBIDS 上未检出差异")
    if not parts:
        return '15 对的判定与 R² 相同' if short else '15 对两种口径的判定相同'
    return '，'.join(parts) + ('' if short else f'，其余 {A5_SAME} 对两种口径的判定相同')


# 第二轮网格边界诊断；端点次数由表 A1 直接报告，不在本模块把触发结果改写成“未触发”。
BND = rd(R47.path('boundary'))
BND_MAX = int(BND[['at_min', 'at_max']].to_numpy().max())

# ============================ 第三轮网格（54）============================
R54 = resolve54()
T154 = rd(out54('table1'))
PT54 = rd(out54('paired_tests'))
BND54 = rd(out54('boundary'))
FOLD54 = {
    m: rd(next(p for p in R54['work_inputs'] if p.endswith(suffix)))
    for m, suffix in {
        'SVR': '45rebuttal_ncv_none_SVR.csv',
        '1D-CNN': '45rebuttal_ncv_none_1DCNN.csv',
    }.items()
}
PRED54 = {
    m: rd(next(p for p in R54['work_inputs'] if p.endswith(suffix)))
    for m, suffix in {
        'SVR': '45rebuttal_ncv_none_SVR_pred.csv',
        '1D-CNN': '45rebuttal_ncv_none_1DCNN_pred.csv',
    }.items()
}
for _suffix in ('45rebuttal_ncv_none_SVR.csv', '45rebuttal_ncv_none_1DCNN.csv',
                '45rebuttal_ncv_none_SVR_pred.csv', '45rebuttal_ncv_none_1DCNN_pred.csv'):
    assert sum(p.endswith(_suffix) for p in R54['work_inputs']) == 1, _suffix
R54T = T154.set_index(['round', 'method'])
R2_OLD = {m: R54T.loc[('第二轮', m)] for m in ('SVR', '1D-CNN')}
R2_NEW = {m: R54T.loc[('第三轮', m)] for m in ('SVR', '1D-CNN')}
R2_NEW_TABLE = T154[T154['round'] == '第三轮'].set_index('method')
F1_54 = PT54[PT54.family.str.startswith('F1')].copy()
F2_54 = PT54[PT54.family.str.startswith('F2')].copy()
F2_54['same_decision_bool'] = F2_54.same_decision.astype(str).str.lower().eq('true')
F2_SIG = F2_54[F2_54['结论'] == '差异显著'].copy()
F2_CHANGED = F2_54[~F2_54.same_decision_bool].copy()
F2_NONSIG = F2_54[F2_54['结论'] == '未检出差异'].copy()
assert len(T154) == 4
assert len(F1_54) == 2
assert len(F2_54) == 15
assert all(abs(float(R2_OLD[m].R2) - float(MAIN.loc[m, 'R2'])) < 1e-9 for m in ('SVR', '1D-CNN')), '54 的第二轮须为表 1 的主分析'
_const54 = []
for row in PRED54['1D-CNN'].groupby(['seed', 'fold']):
    g = row[1]
    if float(g.pred.std(ddof=0)) < 1e-12:
        _const54.append((int(row[0][0]), int(row[0][1]), float(FOLD54['1D-CNN'].set_index(['seed', 'fold']).loc[row[0], 'R2'])))
CONST54: tuple[tuple[int, int, float], ...] = tuple(_const54)
_const_keys54 = {(s, f) for s, f, _ in CONST54}
NONCONST54 = FOLD54['1D-CNN'][~FOLD54['1D-CNN'].apply(lambda r: (int(r.seed), int(r.fold)) in _const_keys54, axis=1)]
assert len(CONST54) + len(NONCONST54) == 25


def inner_degradation_stats(records: pd.DataFrame, threshold: float = 2.0) -> dict:
    """从缓存记录统计内层退化；输入列为 kind、cfg、v。"""
    inner = records[records.kind == 'inner'].copy()
    inner['degraded'] = inner.v.map(lambda values: [float(x) > threshold for x in values])
    exploded = inner[['cfg', 'degraded']].explode('degraded')
    bad = exploded[exploded.degraded.astype(bool)]
    by_lr = bad.cfg.map(lambda c: c['lr']).value_counts().to_dict()
    by_w = bad.cfg.map(lambda c: c['w']).value_counts().to_dict()
    flat = np.array([float(x) for v in inner.v for x in v])
    ok, over = flat[flat <= threshold], flat[flat > threshold]
    return {'total': int(len(bad)), 'by_lr': by_lr, 'by_w': by_w,
            'near': int(sum(1.8 < float(x) < 2.2 for v in inner.v for x in v)),
            'n_cfg_any': int(inner.degraded.map(any).sum()), 'n_cfg': int(len(inner)), 'n_fit': int(len(flat)),
            'max_ok': float(ok.max()) if len(ok) else float('nan'),
            'min_bad': float(over.min()) if len(over) else float('nan'),
            'med_bad': float(np.median(over)) if len(over) else float('nan')}


def rf_tree_platform_stats(records: pd.DataFrame) -> pd.DataFrame:
    """按特征比例汇总树数 800→1600 的内层 RMSE 相对变化（纯函数）。"""
    inner = records[records.kind == 'inner'].copy()
    inner['ratio'] = inner.cfg.map(lambda c: c[1])
    inner['trees'] = inner.cfg.map(lambda c: c[0])
    inner['rmse'] = inner.v.map(lambda values: float(np.mean(np.sqrt(values))))
    rows = []
    for ratio, group in inner.groupby('ratio'):
        by_fold = group.pivot_table(index=['seed', 'fold'], columns='trees', values='rmse', aggfunc='first')
        relative = 100 * (by_fold[1600] / by_fold[800] - 1)
        rows.append({'feature_ratio': ratio, 'relative_change': float(relative.median())})
    return pd.DataFrame(rows)


def _cache_records(paths, method):
    rows = []
    for path in paths:
        _READS.append(str(path))
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                rec = json.loads(line)
                kind, seed, fold, cfg = json.loads(rec['k'])
                rows.append({'kind': kind, 'seed': seed, 'fold': fold, 'cfg': cfg, 'v': rec['v'], 'method': method})
    return pd.DataFrame(rows)


_cache54 = _cache_records([p for p in R54['cache_inputs'] if '45shift/' in p and '1DCNN' in p], '1D-CNN')
_cache47 = _cache_records([p for p in R47.marker['caches'] if '1DCNN' in p], '1D-CNN')
_rf47 = _cache_records([p for p in R47.marker['caches'] if 'none_RF' in p], 'RF')
DEG54 = inner_degradation_stats(_cache54)
DEG47 = inner_degradation_stats(_cache47)
RF_PLATFORM = rf_tree_platform_stats(_rf47)
RF_PLATFORM_RANGE = (float(RF_PLATFORM.relative_change.min()), float(RF_PLATFORM.relative_change.max()))
_bnd_main = BND[BND.analysis.str.startswith('主分析')]
# 表 A1 注：非数值取值的实际大小（47 的端点诊断按实际取值判端点）
_note_scale = str(_bnd_main.loc[(_bnd_main.method == 'SVR') & (_bnd_main.dim == 'gamma'), 'value_note'].iloc[0])
_note_feat = str(_bnd_main.loc[(_bnd_main.method == 'RF') & (_bnd_main.dim == '特征比例'), 'value_note'].iloc[0])
_m_scale = re.search(r'([0-9.]+)~([0-9.]+)', _note_scale)
_m_feat = re.search(r'sqrt=(\d+); 0\.1=(\d+); 0\.3=(\d+); 1\.0=(\d+)', _note_feat)
assert _m_scale is not None, _note_scale
assert _m_feat is not None, _note_feat
SCALE_RANGE = tuple(float(x) for x in _m_scale.groups())
RF_NFEAT = tuple(int(x) for x in _m_feat.groups())
assert SCALE_RANGE[0] > 1e-3, SCALE_RANGE                       # scale 大于网格中的 10⁻³，是该维的最大端
assert RF_NFEAT[0] < RF_NFEAT[1] < RF_NFEAT[2] < RF_NFEAT[3], RF_NFEAT   # √p 为特征比例维的最小端
SECOND_OVER50 = {
    f'{row.method} {row.dim}': int(max(row.at_min, row.at_max))
    for _, row in _bnd_main.iterrows()
    if max(row.at_min, row.at_max) > row.n_folds / 2
}
# 附录 A1 写：第二轮中过半落在同一端点的维度，按 2.5 节的规则只对 SVR 与 1D-CNN 设第三轮；RF 树数为事后例外
assert {k.split()[0] for k in SECOND_OVER50} <= {'SVR', '1D-CNN', 'RF'}, SECOND_OVER50
assert 'SVR gamma' in SECOND_OVER50, SECOND_OVER50
assert '1D-CNN 层数' in SECOND_OVER50, SECOND_OVER50
_selected = []
for _, row in FOLD54['1D-CNN'].iterrows():
    _selected_cfg = ast.literal_eval(row.picked)
    _matches = _cache54[
        (_cache54.seed == row.seed) & (_cache54.fold == row.fold) &
        (_cache54.kind == 'inner') & (_cache54.cfg == _selected_cfg)
    ]
    assert len(_matches) == 1, (row.seed, row.fold, _selected_cfg, len(_matches))
    _selected.append(_matches.iloc[0])
_selected_records = pd.DataFrame(_selected)
SEL_DEG54 = int(sum(float(x) > 2.0 for values in _selected_records.v for x in values))
DEG54_BY_LR = [(lr, int(DEG54['by_lr'].get(lr, 0))) for lr in (3e-4, 1e-3, 3e-3)]
DEG54_BY_W = [(w, int(DEG54['by_w'].get(w, 0))) for w in (16, 32, 64, 128)]
THIRD_FREE = BND54[BND54.dim != '层数']
THIRD_FREE_MAX = int(THIRD_FREE[['at_min', 'at_max']].to_numpy().max())
THIRD_FREE_ARGMAX = THIRD_FREE.iloc[int(np.argmax(THIRD_FREE[['at_min', 'at_max']].max(axis=1).to_numpy()))]
assert THIRD_FREE_MAX <= 12, '附录 A6 写第三轮自由维未过半落在端点、网格至此固定'
assert (F1_54.holm_p >= 0.05).all(), '3.1 节写 SVR、1D-CNN 的第三轮与第二轮之差均未检出差异'
assert (F2_SIG['diff'] > 0).all()


def changed_text():
    """第三轮重做 15 对比较时判定改变的各对：「X、Y 变为未检出差异，Z 变为差异显著」。"""
    to_ns = [str(p) for p in F2_CHANGED[F2_CHANGED['结论'] == '未检出差异'].pair]
    to_s = [str(p) for p in F2_CHANGED[F2_CHANGED['结论'] == '差异显著'].pair]
    parts = []
    if to_ns:
        parts.append(f"{'、'.join(to_ns)} 这 {len(to_ns)} 对变为未检出差异")
    if to_s:
        parts.append(f"{'、'.join(to_s)} 这 {len(to_s)} 对变为差异显著")
    return '，'.join(parts) + '，其余各对的判定不变' if parts else '15 对的判定均不变'


# ============================ 预处理选择与复核（27、36、38、44、25）============================
PREP = rd(f'{OUT}/27prep_nested.csv')
PREP_MEAN = PREP.groupby('预处理').R2.mean()
assert PREP_MEAN.idxmax() == 'SNV+SG1D', PREP_MEAN
PREP_BEST = float(PREP_MEAN.max())
PREP_OTHERS = PREP_MEAN.drop('SNV+SG1D')
DEF_SVR = float(PREP[(PREP.预处理 == 'SNV+SG1D') & (PREP.方法 == 'SVR')].R2.iloc[0])    # SVR 默认超参数
DEF_PLSR = float(PREP[(PREP.预处理 == 'SNV+SG1D') & (PREP.方法 == 'PLSR')].R2.iloc[0])  # PLSR 潜变量数在训练折内选取
assert DEF_SVR < DEF_PLSR, '正文写 SVR 与 PLSR 的先后随调参条件互换'
assert MAIN.loc['SVR', 'R2'] > MAIN.loc['PLSR', 'R2'], '正文写 SVR 与 PLSR 的先后随调参条件互换'

LEAK = rd(f'{OUT}/36prep_leakage.csv')                 # 单个种子，5 种预处理 × 4 法，各自 12 组候选
LEAK_RANK = LEAK.pivot(index='prep', columns='method', values='rank')
assert (LEAK_RANK.SVR == 1).all()
assert (LEAK_RANK.PLSR == 4).all()
SVR_BY_PREP = LEAK[LEAK.method == 'SVR'].set_index('prep').R2
SVR_PREP_BEST = str(SVR_BY_PREP.idxmax())
assert SVR_PREP_BEST != 'SNV+SG1D', '附录写 SNV+SG-1D 不是 SVR 的最优预处理'
PREP_CN = {'原始': '不处理', 'SNV': 'SNV', 'MSC': 'MSC', 'SG1D': 'SG-1D', 'SNV+SG1D': 'SNV+SG-1D'}

NP = rd(f'{OUT}/38nested_prep.csv').set_index('method')
BASIS = rd(f'{OUT}/26basis_nested.csv')                  # 单个种子，两族基 × L∈{20,40,60,80}
for _fam in ('B样条', '傅里叶'):
    for _col in ('PFLR_R2', 'FPCR_R2'):
        assert int(BASIS[BASIS.basis == _fam].set_index('L')[_col].idxmax()) == 80, (_fam, _col)
BASIS80 = BASIS[BASIS.L == 80].set_index('basis')
FN = pd.concat([rd(f'{OUT}/44fully_nested.csv'), rd(f'{OUT}/44fully_nested_rf.csv'),
                rd(f'{OUT}/44fully_nested_cnn.csv')]).set_index('method')
FN['method_std'] = FN.index.map(lambda s: 'FPLR' if s == 'PFLR' else s)
FN = FN.set_index('method_std')
assert set(FN.index) == set(ALL6), FN.index
# 全嵌套复核（44）的 R² 按 5 个外层测试折的合并预测计算，与之比较的同一种子主分析也须按合并预测计算：
# 六法均取 45 正式结果种子 20041210 的逐样品预测（函数型模型取 B 样条基）。
SEED1 = 20041210
_pred45 = pd.concat([rd(p) for p in sorted(Path(OUT).glob('45rebuttal_ncv_*_pred.csv'))
                     if not p.name.startswith('._')])
_pred45 = _pred45[(_pred45.seed == SEED1) & _pred45.basis.isin(['bspline', 'none'])]


def _pooled_r2(g):
    assert len(g) == KEPT_N, len(g)
    assert g.idx.nunique() == KEPT_N
    return 1 - float(((g.y - g.pred) ** 2).sum()) / float(((g.y - g.y.mean()) ** 2).sum())


S1POOL = pd.Series({m: _pooled_r2(g) for m, g in _pred45.groupby('method')})
assert set(S1POOL.index) == set(ALL6), S1POOL.index
_f45 = rd(f'{OUT}/45rebuttal_ncv_none_PLSR.csv')
_f25 = rd(f'{OUT}/25nested_cv.csv')
_fx = _f45[_f45.seed == SEED1][['fold', 'R2']].merge(_f25[(_f25.seed == SEED1) & (_f25.method == 'PLSR')][['fold', 'R2']],
                                                      on='fold', suffixes=('_45', '_25'))
assert len(_fx) == 5
assert (_fx.R2_45 - _fx.R2_25).abs().max() < 1e-9, '45 与 25 脚本在种子 20041210 下须为同一批外层折'
FN_ORDER = list(FN.R2.sort_values(ascending=False).index)
S1_ORDER = list(S1POOL.sort_values(ascending=False).index)
FN_DELTA = (FN.R2 - S1POOL).reindex(ALL6)                          # 全嵌套 − 同一种子固定预处理与基设定
FN_SVR_PREPS = FN.loc['SVR', 'prep'].split('|')
FN_SVR_MODE = pd.Series(FN_SVR_PREPS).value_counts()
assert all(b.endswith('80') for _fm in ('FPLR', 'FPCR') for b in FN.loc[_fm, 'basis'].split('|')), '附录写函数型模型的 L 各折均为 80'
FN_FOURIER = {m: sum(b.startswith('傅里叶') for b in FN.loc[m, 'basis'].split('|')) for m in ('FPLR', 'FPCR')}
FN_PREP_MODE = {m: pd.Series(FN.loc[m, 'prep'].split('|')).value_counts() for m in ('FPLR', 'FPCR', 'SVR', 'PLSR')}
FN_PREP_MODE = {m: (str(v.index[0]), int(v.iloc[0])) for m, v in FN_PREP_MODE.items()}
assert all(n >= 3 for _, n in FN_PREP_MODE.values()), '附录写各法的预处理「多选」某一种'
assert SVR_BY_PREP['SNV'] > SVR_BY_PREP['SNV+SG1D'], '附录写 SVR 在 SNV 下高于 SNV+SG-1D'
SEEDS5 = [int(s) for s in rd(f'{OUT}/45rebuttal_ncv_none_PLSR.csv').seed.unique()]
assert len(SEEDS5) == 5, SEEDS5

# λs 敏感性（51）：λs 取 0、10⁻⁸、10⁻⁴ 时两个函数型模型 25 折 R² 均值与表 1（λs=10⁻⁶）之差
LAMS = rd(f'{OUT}/51rebuttal_lams_summary.csv')
assert set(LAMS.lam_s) == {0.0, 1e-8, 1e-6, 1e-4}, LAMS
assert (LAMS.n_folds == 25).all()
for _lab in ('FPLR', 'FPCR'):
    _ref = LAMS[(LAMS.lam_s == 1e-6) & (LAMS.method == _lab)].R2_mean.iloc[0]
    assert abs(_ref - MAIN.loc[_lab, 'R2']) < 1e-9, (_lab, _ref, MAIN.loc[_lab, 'R2'])
LAMS_ALT = LAMS[LAMS.lam_s != 1e-6]
LAMS_MAXDIFF = float(LAMS_ALT.diff_vs_ref_mean.abs().max())
LAMS_DIFF = LAMS_ALT.set_index(['method', 'lam_s']).diff_vs_ref_mean          # (方法, λs) → 与 λs=10⁻⁶ 的 R² 均值之差
LAMS_FIT = LAMS.groupby('lam_s').fit_rel_diff_vs0.first()                     # λs → 拟合光谱与 λs=0 的相对差

# ============================ 主分析的弱折（61，事后描述）============================
MF = rd(f'{OUT}/61rebuttal_mainfold_summary.csv').set_index('method')
MFS = rd(f'{OUT}/61rebuttal_mainfold_spectra.csv')
MFF = rd(f'{OUT}/61rebuttal_mainfold_folds.csv')
assert set(MF.index) == set(ALL6), MF.index
assert all(abs(MF.loc[m, 'R2_mean'] - MAIN.loc[m, 'R2']) < 1e-9 for m in ALL6), '61 的 25 折均值须与表 1 相同'
MF_HOT = sorted(int(i) for i in MFS.idx.unique())                   # 清洗后序号
MF_HOT_OLD = sorted(int(i) for i in MFS.old_idx.unique())            # 清洗前序号
assert len(MF_HOT) == 2, MF_HOT
WEAK = (int(MF.loc['FPLR', 'worst_seed']), int(MF.loc['FPLR', 'worst_fold']))
assert WEAK == (int(MF.loc['FPCR', 'worst_seed']), int(MF.loc['FPCR', 'worst_fold'])), '两种函数型模型的最弱一折相同'
for _fm in ('FPLR', 'FPCR'):
    assert sorted(int(i) for i in MF.loc[_fm, 'worst_top5_idx'].split()[:2]) == MF_HOT, (_fm, MF.loc[_fm, 'worst_top5_idx'])
_sd = MF.R2_sd
assert _sd['FPLR'] > 1.8 * _sd['SVR'], '3.1 节写函数型模型的标准差约为 SVR 的两倍'
assert _sd['FPCR'] > 1.8 * _sd['SVR'], '3.1 节写函数型模型的标准差约为 SVR 的两倍'
_fold_r2 = {m: rd(f"{OUT}/45rebuttal_ncv_{'bspline' if m.startswith('FP') else 'none'}_{m.replace('-', '')}.csv")
            .set_index(['seed', 'fold']).R2 for m in ALL6}
WEAK_R2 = {m: float(_fold_r2[m].loc[WEAK]) for m in ALL6}
# 名单光谱在弱折中的预测误差（FPLR、PLSR；偏高为正）
_wk = MFS[(MFS.seed == WEAK[0]) & (MFS.fold == WEAK[1])].set_index(['method', 'idx'])
WEAK_ERR = {m: [float(_wk.loc[(m, i), 'err']) for i in MF_HOT] for m in ('FPLR', 'FPCR', 'PLSR', 'SVR')}
assert all(v > 0 for v in WEAK_ERR['FPLR']), '3.1 节写 FPLR 对二者的预测偏高'
HOT_ABS = {m: (float(MFS[MFS.method == m].err.abs().min()), float(MFS[MFS.method == m].err.abs().max())) for m in ALL6}
assert all(len(MFS[MFS.method == m]) == 10 for m in ALL6), '2 条光谱各在 5 个种子下落入测试折 1 次'
# 两条光谱的原始光谱特征（61 的产物）：一条最大读数为全体最低
MF_RAWQ = MFS.groupby('idx').raw_max_quantile.first()
MF_WEAKEST = int(MF_RAWQ.idxmin())
assert abs(float(MF_RAWQ.min()) - 1 / KEPT_N) < 1e-9, '3.1 节写其一的最大读数为全体最低'
# 评价时除去这 2 条光谱后的 25 折 R²
EXCL = {m: (float(MF.loc[m, 'R2_mean_excl_hot']), float(MF.loc[m, 'R2_sd_excl_hot'])) for m in ALL6}
assert all(EXCL[m][1] < _sd[m] * 0.7 for m in ('FPLR', 'FPCR')), '附录写除去后函数型模型的标准差大幅降低'
# 杠杆回退的事后应用
LEVM = {m: (float(MF.loc[m, 'R2_mean_lev']), float(MF.loc[m, 'R2_sd_lev'])) for m in ('FPLR', 'FPCR', 'PLSR')}
LEV_WEAK = {m: float(MF.loc[m, 'worst_R2_lev']) for m in ('FPLR', 'FPCR', 'PLSR')}
LEV_HOT = {m: (int(MF.loc[m, 'hot_backed']), int(MF.loc[m, 'hot_in_test'])) for m in ('FPLR', 'FPCR', 'PLSR')}
LEV_FRAC_MAIN = {m: float(MF.loc[m, 'frac_backed_median']) for m in ('FPLR', 'FPCR', 'PLSR')}
_mff = MFF.assign(d=lambda q: q.R2_lev - q.R2)
_other = _mff[~((_mff.seed == WEAK[0]) & (_mff.fold == WEAK[1]))]
LEV_OTHER = {m: (float(_other[_other.method == m].d.min()), float(_other[_other.method == m].d.max())) for m in ('FPLR', 'FPCR', 'PLSR')}
LEV_DMEAN = {m: LEVM[m][0] - float(MF.loc[m, 'R2_mean']) for m in ('FPLR', 'FPCR', 'PLSR')}
_hl = MFS[MFS.method.isin(['FPLR', 'FPCR'])]
LEV_HOT_ERR = (float(_hl.err.abs().max()), float(_hl.err_lev.abs().max()))       # 回退前后名单光谱的最大绝对误差
assert all(LEV_HOT[m][0] == LEV_HOT[m][1] for m in ('FPLR', 'FPCR')), '附录写 2 条光谱每次落入测试折均被回退'
assert all(LEV_WEAK[m] > WEAK_R2[m] + 0.05 for m in ('FPLR', 'FPCR')), '附录写弱折的 R² 回退后明显升高'
assert LEV_HOT_ERR[1] < LEV_HOT_ERR[0], LEV_HOT_ERR

# ============================ 小样本（21、49、18、20）============================
SS = rd(f'{OUT}/49rebuttal_smallsample_summary.csv').set_index(['N', 'method', 'strategy'])
NS = sorted({i[0] for i in SS.index})


def ss(N, m, st, col):
    return float(SS.loc[(N, m, st), col])


REP = rd(f'{OUT}/21smallsample_repeats.csv')
REP['method'] = REP.method.replace({'PFLR': 'FPLR'})
_w = REP[(REP.N == 80) & (REP.method == 'FPCR')].pivot(index='seed', columns='strategy', values='R2')
CV80_HI = int((_w.cv > _w.fixed + 1e-12).sum())
CV80_LO = int((_w.cv < _w.fixed - 1e-12).sum())
CV80_EQ = len(_w) - CV80_HI - CV80_LO
_pp = rd(f'{OUT}/49rebuttal_smallsample_paired.csv')
assert int(_pp[(_pp.N == 80) & (_pp.a == 'FPCR_cv') & (_pp.b == 'FPCR_fixed')].n_a_higher.iloc[0]) == CV80_HI
PP = _pp.set_index(['N', 'a', 'b']).n_a_higher

_ARMS = [(m, st) for m in ('FPLR', 'FPCR', 'SVR') for st in ('fixed', 'cv')] + [('PLSR', 'fixed')]
MID = [50, 80, 120, 200]                                   # 3.2 节与摘要的「训练样本为 50~200 个」
# 3.2 节：N=50~200 时交叉验证选参的 FPLR 中位数在各方法与做法中最高，FPCR（交叉验证）次之；N=400 时 SVR 最高
for _n in MID:
    _med = {a: ss(_n, a[0], a[1], 'median') for a in _ARMS}
    _rank = sorted(_med, key=lambda a: -_med[a])
    assert _rank[0] == ('FPLR', 'cv'), (_n, _rank[:3])
    assert _rank[1] == ('FPCR', 'cv'), (_n, _rank[:3])
assert max(_ARMS, key=lambda a: ss(400, a[0], a[1], 'median')) == ('SVR', 'cv')
FPLR_MID_MED = [ss(n, 'FPLR', 'cv', 'median') for n in MID]
# 交叉验证选参与固定超参数：N≥50 时两种函数型模型交叉验证的中位数均高于固定超参数
CV_GAIN = {m: [ss(n, m, 'cv', 'median') - ss(n, m, 'fixed', 'median') for n in NS if n >= 50] for m in ('FPLR', 'FPCR')}
assert all(v > 0 for m in CV_GAIN for v in CV_GAIN[m]), CV_GAIN
assert min(CV_GAIN['FPLR']) > 0.2, '摘要写 FPLR 交叉验证选参比固定超参数高 0.2 以上'
CV_GAIN_ALL = (min(min(v) for v in CV_GAIN.values()), max(max(v) for v in CV_GAIN.values()))
# 同一做法内的比较：固定超参数时 SVR 在每个 N 的均值高于两种函数型模型；交叉验证选参时 N=30~200 的 FPLR 均值高于 SVR
for _N in NS:
    for _fm in ('FPCR', 'FPLR'):
        assert ss(_N, 'SVR', 'fixed', 'mean') > ss(_N, _fm, 'fixed', 'mean'), (_N, _fm)
FPLR_OVER_SVR_N = [n for n in NS if ss(n, 'FPLR', 'cv', 'mean') > ss(n, 'SVR', 'cv', 'mean')]
assert FPLR_OVER_SVR_N == [30, 50, 80, 120, 200], FPLR_OVER_SVR_N
# 逐次配对：N=50~200 时交叉验证选参的 FPLR 高于 SVR 的次数（20 次抽样）
FPLR_VS_SVR = [int(20 - PP.loc[(n, 'SVR_cv', 'FPLR_cv')]) for n in MID]
assert min(FPLR_VS_SVR) >= 14, FPLR_VS_SVR
# R²<0：N≤30 时各方法与做法均多次出现；N≥80 时均不再出现（20 次抽样）
LOW_NEG = {n: [int(ss(n, a[0], a[1], 'n_neg')) for a in _ARMS] for n in (20, 30)}
assert min(LOW_NEG[20]) >= 5, LOW_NEG
assert min(LOW_NEG[30]) >= 1, LOW_NEG
assert all(ss(n, a[0], a[1], 'n_neg') == 0 for n in NS if n >= 80 for a in _ARMS)
LOW_NEG_RANGE = (min(min(v) for v in LOW_NEG.values()), max(max(v) for v in LOW_NEG.values()))   # 摘要：N≤30 时各方法均有 R²<0 的次数范围
N50_NEG = {a: int(ss(50, a[0], a[1], 'n_neg')) for a in _ARMS if ss(50, a[0], a[1], 'n_neg') > 0}
NEG_MAX_N = max(n for n in NS for a in _ARMS if ss(n, a[0], a[1], 'n_neg') > 0)
assert NEG_MAX_N == 50, NEG_MAX_N
# PLSR 的中位数在 N=30~200 均最低
for _n in (30, 50, 80, 120, 200):
    _lo = min(_ARMS, key=lambda a: ss(_n, a[0], a[1], 'median'))
    assert _lo == ('PLSR', 'fixed'), (_n, _lo)

SVR_STD = rd(f'{OUT}/18svr_aligned.csv').set_index('N')
assert all(SVR_STD.loc[n, 'SVR_std'] > SVR_STD.loc[n, 'SVR_raw'] for n in (20,)), '附录写标准化在 N=20 时升高'
assert all(SVR_STD.loc[n, 'SVR_std'] < SVR_STD.loc[n, 'SVR_raw'] for n in NS if n >= 30), '附录写标准化在 N≥30 时降低'

_READS.append(f'{LOGS}/18svr_align_and_sensitivity.log')
_log18 = Path(f'{LOGS}/18svr_align_and_sensitivity.log').read_text(encoding='utf-8')
_fpcr_block = _log18.split('FPCR（12 组候选的 R²）：')[1].split('→')[0]
K80 = {int(k): float(v) for k, v in re.findall(r'\(40, (\d+)\)\s+(-?\d+\.\d+)', _fpcr_block)}
assert len(K80) == 12, K80
K80_BEST = max(K80, key=lambda k: K80[k])
K80_HI = (min(K80[k] for k in (20, 25, 30)), max(K80[k] for k in (20, 25, 30)))
assert K80_BEST in (20, 25, 30), K80
K80_10 = K80[10]

_READS.append(f'{LOGS}/20cv_selection_mechanism.log')
_log20 = Path(f'{LOGS}/20cv_selection_mechanism.log').read_text(encoding='utf-8')
_m = re.search(r'交叉验证选中的 K 的分布：\s*(\{[^}]*\})', _log20)
assert _m is not None
CV_K = {int(a): int(b) for a, b in re.findall(r'(\d+):\s*(\d+)', _m.group(1))}
_m = re.search(r'中位\s*(\d+)%', _log20)
assert _m is not None
CV_RSD = int(_m.group(1))
_m = re.search(r'交叉验证所选 K 的实际 R² 均值 ([0-9.]+)，事后最优 K 的 R² 均值 ([0-9.]+)，平均损失 ([0-9.]+)', _log20)
assert _m is not None
CV_K_R2, BEST_K_R2, CV_K_LOSS = (float(x) for x in _m.groups())
assert CV_K.get(10, 0) == CV80_EQ, (CV_K, CV80_EQ)          # 与固定 K=10 相同的次数即 R² 相等的次数
assert min(CV_K) >= 15, '附录写交叉验证选中的 K 均不小于 15'
assert len(CV_K) >= 2, CV_K
CV_K_TXT = '、'.join(str(k) for k in sorted(CV_K)[:-1]) + f' 或 {max(CV_K)}'   # 3.2 节「交叉验证选中的 K 为 15、20 或 30」

# ============================ 选参对策与逐样品剖析（56、57、58；协议 B10.8、B10.9）============================
G_N = (50, 80, 120)                                   # 判读所用的三档样本量
G_SEL, G_USED, G_FRESH = '新100次', 'B10.8的100次', '再取100次'   # 56 的 seed 20~119；57 的 seed 20~119 与 120~219
_s56 = rd(f'{OUT}/56rebuttal_selection_summary.csv')
S56 = _s56[_s56.seed_set == G_SEL].set_index(['method', 'arm', 'N'])
D56 = rd(f'{OUT}/56rebuttal_selection_draws.csv').query("seed_set == @G_SEL")
S57 = rd(f'{OUT}/57rebuttal_guard_summary.csv').set_index(['seed_set', 'method', 'arm', 'N'])
D57 = rd(f'{OUT}/57rebuttal_guard_draws.csv')
AN57 = rd(f'{OUT}/57rebuttal_guard_anatomy.csv') if Path(f'{OUT}/57rebuttal_guard_anatomy.csv').stat().st_size > 5 else pd.DataFrame()
SP57 = rd(f'{OUT}/57rebuttal_guard_spectra.csv') if Path(f'{OUT}/57rebuttal_guard_spectra.csv').stat().st_size > 5 else pd.DataFrame()
DS57 = rd(f'{OUT}/57rebuttal_guard_dataset.csv').set_index('item').value
S58 = rd(f'{OUT}/58rebuttal_guard_posthoc_summary.csv').set_index(['seed_set', 'method', 'rule', 'N'])
assert (S56.n == 100).all()
assert (S57.n == 100).all()
assert (S58.n == 100).all()
assert int(float(DS57['n_samples'])) == KEPT_N, '57 须在清洗后的数据上运行'


def g56(m, arm, col, n):
    return float(S56.loc[(m, arm, n), col])


def g57(m, arm, col, n, s=G_FRESH):
    return float(S57.loc[(s, m, arm, n), col])


def g58(m, rule, col, n, s=G_FRESH):
    return float(S58.loc[(s, m, rule, n), col])


def neg56(m, arm):
    return [int(g56(m, arm, 'n_neg', n)) for n in G_N]


def neg57(m, arm, s=G_FRESH):
    return [int(g57(m, arm, 'n_neg', n, s)) for n in G_N]


def cnt(v):
    """三档次数的写法：26、15、8。"""
    return '、'.join(str(int(x)) for x in v)


_FUN = ('FPCR', 'FPLR')
# 57 在 seed 20~119 上的 cv 臂与 56 逐格一致（同一批抽样的两次计算）
assert all(neg57(m, 'cv', G_USED) == neg56(m, 'cv') for m in _FUN)
# B10.8：清洗后交叉验证选参在三档均无 R²<0，两种选参端对策亦然；判定只由中位数之差决定
assert all(v == 0 for m in _FUN for a in ('cv', 'onese', 'repcv') for v in neg56(m, a)), '3.2 节写三档均无 R²<0'
D1SE = {m: [g56(m, 'onese', 'median', n) - g56(m, 'cv', 'median', n) for n in G_N] for m in _FUN}
DREP = {m: [g56(m, 'repcv', 'median', n) - g56(m, 'cv', 'median', n) for n in G_N] for m in _FUN}
ONESE_OK = {m: min(D1SE[m]) >= -0.02 for m in _FUN}
REPCV_OK = {m: min(DREP[m]) >= -0.02 for m in _FUN}
assert not any(ONESE_OK.values()), '附录写一倍标准误准则按判据为无效（中位数低于交叉验证选参超过 0.02）'
assert all(REPCV_OK.values()), '附录写重复交叉验证按判据为有效'
D1SE_RANGE = (min(min(v) for v in D1SE.values()), max(max(v) for v in D1SE.values()))
DREP_RANGE = (min(min(v) for v in DREP.values()), max(max(v) for v in DREP.values()))
# B10.9：再取的 100 次抽样上各臂的 R²<0 次数与中位数；判据同 B10.9
GUARD_OK = {(m, a): all(v == 0 for v in neg57(m, a)) and all(g57(m, a, 'median', n) - g57(m, 'cv', 'median', n) >= -0.02 for n in G_N)
            for m in _FUN for a in ('lev', 'clip')}
CV57_NEG = {m: neg57(m, 'cv') for m in _FUN}
LEV_D = {m: [g57(m, 'lev', 'median', n) - g57(m, 'cv', 'median', n) for n in G_N] for m in _FUN}
CLIP_D = {m: [g57(m, 'clip', 'median', n) - g57(m, 'cv', 'median', n) for n in G_N] for m in _FUN}
LEV_FRAC = [g57(m, 'lev', 'frac_backed_median', n) for m in _FUN for n in G_N]      # 被回退的测试光谱占比（中位数）
HOT57 = [] if SP57.empty else [int(i) for i in SP57.idx]                               # 清洗后按 B10.9 规则导出的名单
N_AN57 = {s: (0 if AN57.empty else int(((AN57.seed_set == s) & AN57.N.isin(G_N)).sum())) for s in (G_USED, G_FRESH)}
# 58：倍数 2、3、4 下的 R²<0 次数与中位数（事后描述）
X_NEG = {r: sum(int(g58(m, r, 'n_neg_all', n)) for m in _FUN for n in G_N) for r in ('x2', 'x3', 'x4')}
X_MED = {r: [g58(m, r, 'median_all', n) - g58(m, 'cv', 'median_all', n) for m in _FUN for n in G_N] for r in ('x2', 'x3', 'x4')}


def _rng(v, nd=3):
    return rg(min(v), max(v), nd)


def _verdict(ok):
    return '有效' if ok else '无效'


def _chg(v, nd=3):
    """一组差值的写法：全为正写「升高 a~b」，全为负写「降低 a~b」，否则「变化 a~b」。"""
    if min(v) > 0:
        return '升高 ' + rg(min(v), max(v), nd)
    if max(v) < 0:
        return '降低 ' + rg(-max(v), -min(v), nd)
    return '变化 ' + rg(min(v), max(v), nd)


# B10.9 判读 (iv)：两臂都有效时正文写 R²<0 总次数较少者，相同则写三档中位数之和较高者，另一臂进附录；都无效时正文如实写明，
# 并把成因剖析作为本节的主要结果。FPCR 与 FPLR 分别判读
MED_SUM = {(m, a): sum(g57(m, a, 'median', n) for n in G_N) for m in _FUN for a in ('lev', 'clip')}


def _main_arm(m):
    ok = {a: GUARD_OK[(m, a)] for a in ('lev', 'clip')}
    if not any(ok.values()):
        return None
    if not all(ok.values()):
        return 'lev' if ok['lev'] else 'clip'
    tot = {a: sum(neg57(m, a)) for a in ('lev', 'clip')}
    if tot['lev'] != tot['clip']:
        return min(('lev', 'clip'), key=lambda a: tot[a])
    return max(('lev', 'clip'), key=lambda a: MED_SUM[(m, a)])


MAIN_ARM = {m: _main_arm(m) for m in _FUN}
_cv_clean = all(v == 0 for m in _FUN for v in CV57_NEG[m])
_lev_neg = {m: neg57(m, 'lev') for m in _FUN}
_clip_neg = {m: neg57(m, 'clip') for m in _FUN}
_lev_d_all = [v for m in _FUN for v in LEV_D[m]]
_clip_d_all = [v for m in _FUN for v in CLIP_D[m]]
# 再取的 100 次抽样上交叉验证选参的失效抽样（B10.9 判读 (iii) 的逐样品剖析），与各臂在同一抽样上的 R²
_d57f = D57[(D57.seed_set == G_FRESH) & D57.N.isin(G_N) & D57.method.isin(_FUN)]
_fail = _d57f[(_d57f.arm == 'cv') & (_d57f.R2 < 0)]
_anf = AN57[(AN57.seed_set == G_FRESH) & AN57.N.isin(G_N)] if not AN57.empty else AN57
assert len(_anf) == len(_fail), (len(_anf), len(_fail))
FAIL57: dict[str, dict[str, Any]] = {}
for _r in _fail.itertuples():
    _arms = _d57f[(_d57f.method == _r.method) & (_d57f.N == _r.N) & (_d57f.seed == _r.seed)].set_index('arm').R2
    _a = _anf[(_anf.method == _r.method) & (_anf.N == _r.N) & (_anf.seed == _r.seed)]
    assert len(_a) == 1, (_r.method, _r.seed)
    assert abs(float(_a.R2.iloc[0]) - float(_r.R2)) < 1e-12, (_r.method, _r.seed)
    FAIL57[_r.method] = {'N': int(_r.N), 'seed': int(_r.seed), 'cv': float(_arms['cv']), 'lev': float(_arms['lev']),
                         'clip': float(_arms['clip']), 'share3': float(_a.share_top3.iloc[0]),
                         'top3': [int(i) for i in str(_a.top3_idx.iloc[0]).split()]}
if not _cv_clean:
    # 3.2 节与附录 A8 的写法对应的数据事实：FPCR、FPLR 各 1 次失效，均在 N=50；FPLR 的一次误差最大的 2 个测试样品即主分析弱折的 2 条光谱，
    # 回退后 R² 转正；FPCR 的一次误差不集中（误差最大 3 个样品的份额不到 FPLR 那次的一半）
    assert sorted(FAIL57) == ['FPCR', 'FPLR'], FAIL57
    assert all(sum(CV57_NEG[m]) == 1 for m in _FUN), CV57_NEG
    assert sorted(FAIL57['FPLR']['top3'][:2]) == MF_HOT, (FAIL57['FPLR']['top3'], MF_HOT)
    assert FAIL57['FPLR']['lev'] > 0, FAIL57['FPLR']
    assert FAIL57['FPLR']['clip'] > 0, FAIL57['FPLR']
    assert FAIL57['FPCR']['share3'] < 0.5 * FAIL57['FPLR']['share3'], FAIL57
    assert not set(FAIL57['FPCR']['top3']) & set(MF_HOT), '3.2 节写 FPCR 失效的一次误差最大的 3 个样品不含这 2 条光谱'
    assert FAIL57['FPCR']['lev'] < 0, '3.2 节写两种做法均未消除 FPCR 的这次失效'
    assert FAIL57['FPCR']['clip'] < 0, '3.2 节写两种做法均未消除 FPCR 的这次失效'
    assert MAIN_ARM == {'FPCR': None, 'FPLR': 'clip'}, ('3.2 节按判读 (iv) 写 FPLR 的范围截断、FPCR 两种做法均无效', MAIN_ARM)
    assert _lev_neg['FPLR'] == _clip_neg['FPLR'], '3.2 节把两种做法的次数合写'
    assert _lev_neg['FPCR'] == _clip_neg['FPCR'], '3.2 节把两种做法的次数合写'
if _cv_clean:
    GUARD_SENTENCE = ("再取的 100 次抽样上，交叉验证选参在三档已无 R²<0，回退后亦无"
                      if all(v == 0 for m in _FUN for v in _lev_neg[m]) else
                      f"再取的 100 次抽样上，交叉验证选参在三档已无 R²<0，回退后 FPCR 与 FPLR 为 {cnt(_lev_neg['FPCR'])} 次与 {cnt(_lev_neg['FPLR'])} 次")
    GUARD_SENTENCE += f"，中位数{_chg(_lev_d_all)}"
else:
    _fl57, _fc57 = FAIL57['FPLR'], FAIL57['FPCR']
    _cv57_txt = (f"各有 {cnt(CV57_NEG['FPCR'])} 次" if CV57_NEG['FPCR'] == CV57_NEG['FPLR']
                 else f"有 {cnt(CV57_NEG['FPCR'])} 次与 {cnt(CV57_NEG['FPLR'])} 次")
    GUARD_SENTENCE = (f"再取的 100 次抽样上，交叉验证选参的 FPCR 与 FPLR 在三档{_cv57_txt} R²<0。"
                      f"FPLR 在两种做法下均为 {cnt(_clip_neg['FPLR'])} 次，按判据均有效，按事先规则报告三档中位数之和较高的范围截断："
                      f"中位数{_chg(CLIP_D['FPLR'])}，失效的一次误差最大的 2 个测试样品即这 2 条光谱，"
                      f"截断使其 R² 由 {fm(_fl57['cv'])} 升至 {_fl57['clip']:.3f}（杠杆回退见附录 A8）。"
                      f"FPCR 在两种做法下仍为 {cnt(_clip_neg['FPCR'])} 次，按判据均无效：失效的一次 R² 为 {fm(_fc57['cv'])}，"
                      f"误差最大的 3 个样品只占平方误差的 {pct0(_fc57['share3'])}%，不含这 2 条光谱")
# 预测端方案还从随机种子 20~119 的失效抽样导出异常光谱名单，用来核对回退能否识别它们
_hot_txt = ('方案还从随机种子 20~119 的失效抽样导出一份异常光谱名单，用来核对回退能否识别它们；清洗后这批抽样没有失效，名单为空'
            if not HOT57 else f'方案从随机种子 20~119 的失效抽样导出的异常光谱名单共 {len(HOT57)} 条')
# 附录 A8 写判读 (iv) 的执行：FPLR 两臂 R²<0 次数相同，按三档中位数之和取范围截断；FPCR 两臂均无效
_iv_txt = ('按事先写明的判读规则，两种做法都有效时正文报告 R²<0 次数较少者，次数相同时报告三档中位数之和较高者：'
           f"FPLR 两种做法三档的 R²<0 次数均为 {sum(_clip_neg['FPLR'])}，中位数之和范围截断为 {MED_SUM[('FPLR', 'clip')]:.3f}、"
           f"杠杆回退为 {MED_SUM[('FPLR', 'lev')]:.3f}，3.2 节报告范围截断；FPCR 两种做法均无效，3.2 节如实写明并给出失效抽样的剖析。")
assert MAIN_ARM == {'FPCR': None, 'FPLR': 'clip'}, MAIN_ARM
assert sum(_clip_neg['FPLR']) == sum(_lev_neg['FPLR']), '附录 A8 写 FPLR 两种做法的 R²<0 次数相同'
_fail_txt = '' if _cv_clean else (
    f"再取的 100 次抽样中交叉验证选参的失效抽样为 FPCR、FPLR 各 1 次（均为 N={FAIL57['FPLR']['N']}）："
    f"FPLR 的一次误差最大的 3 个测试样品占平方误差总和的 {pct0(FAIL57['FPLR']['share3'])}%，其中 2 个即 3.1 节弱折的 2 条光谱，"
    f"杠杆回退与范围截断后 R² 由 {fm(FAIL57['FPLR']['cv'])} 升至 {FAIL57['FPLR']['lev']:.3f} 与 {FAIL57['FPLR']['clip']:.3f}；"
    f"FPCR 的一次 R² 为 {fm(FAIL57['FPCR']['cv'])}，误差最大的 3 个测试样品占 {pct0(FAIL57['FPCR']['share3'])}%，"
    f"回退与截断后为 {fm(FAIL57['FPCR']['lev'])} 与 {fm(FAIL57['FPCR']['clip'])}。")
GUARD_APPENDIX = (
    "表 A6 表明，" + ("交叉验证选参在三档已无 R²<0；" if _cv_clean else
                      f"交叉验证选参的 FPCR 与 FPLR 在三档有 {cnt(CV57_NEG['FPCR'])} 次与 {cnt(CV57_NEG['FPLR'])} 次 R²<0；")
    + f"杠杆回退后 FPCR 与 FPLR 为 {cnt(_lev_neg['FPCR'])} 次与 {cnt(_lev_neg['FPLR'])} 次，中位数较交叉验证选参{_chg(_lev_d_all)}；"
    + f"范围截断后为 {cnt(_clip_neg['FPCR'])} 次与 {cnt(_clip_neg['FPLR'])} 次，中位数{_chg(_clip_d_all)}。"
    + ("按判据，杠杆回退与范围截断" + "，".join(f"对 {m} 均为{_verdict(GUARD_OK[(m, 'lev')])}" for m in ('FPCR', 'FPLR'))
       if all(GUARD_OK[(m, 'lev')] == GUARD_OK[(m, 'clip')] for m in ('FPCR', 'FPLR')) else
       "按判据，" + "，".join(f"{lab}对 FPCR 为{_verdict(GUARD_OK[('FPCR', a)])}、对 FPLR 为{_verdict(GUARD_OK[('FPLR', a)])}"
                            for a, lab in (('lev', '杠杆回退'), ('clip', '范围截断'))))
    + ("；由于交叉验证选参本身已无失效，这些判定只反映中位数的差别。" if _cv_clean else "。")
    + _iv_txt
    + _fail_txt
    + _hot_txt + "。"
    + f"被回退的测试光谱占比的中位数为 {pct1(min(LEV_FRAC))}%~{pct1(max(LEV_FRAC))}%；"
    + f"倍数改取 2、3、4 时三档的 R²<0 次数合计为 {X_NEG['x2']}、{X_NEG['x3']}、{X_NEG['x4']} 次，"
    + f"中位数较交叉验证选参依次{'、'.join(_chg(X_MED[x]) for x in ('x2', 'x3'))} 与{_chg(X_MED['x4'])}（倍数 2、4 为事后描述）。")

# 数据层面（清洗后）
DUP_GROUPS = int(float(DS57['identical_spectrum_groups']))
ADC_SPECTRA = int(float(DS57['spectra_with_adc_max_reading']))
assert DUP_GROUPS == 0, '清洗后无逐点相同的光谱组'
assert ADC_SPECTRA == 0, '清洗后无含检测器上限读数的光谱'

# ============================ 跨批次（46、49）============================
RS = rd(f'{OUT}/46rebuttal_recal_summary.csv')
SRC = rd(f'{OUT}/46rebuttal_recal_source.csv').set_index(['setting', 'method'])
SETS = ['2018→2019', '2019→2018', '留新疆', '留山东', '留陕西']
M4 = ['PLSR', 'SVR', 'FPLR', 'FPCR']


def rmed(st, m, strat, k):
    r = RS[(RS.setting == st) & (RS.method == m) & (RS.strategy == strat) & (RS.k == k)]
    assert len(r) == 1, (st, m, strat, k)
    return float(r.R2_med.iloc[0])


def rq(st, m, strat, k):
    r = RS[(RS.setting == st) & (RS.method == m) & (RS.strategy == strat) & (RS.k == k)].iloc[0]
    return float(r.R2_lo), float(r.R2_hi)


S0 = {(st, m): float(SRC.loc[(st, m), 'tgt_R2']) for st in SETS for m in M4}
O1 = {(st, m): float(SRC.loc[(st, m), 'O1_R2']) for st in SETS for m in M4}
O2 = {(st, m): float(SRC.loc[(st, m), 'O2_R2']) for st in SETS for m in M4}
_hard = [S0[(st, m)] for st in SETS[:3] for m in M4]
S0_HARD_NEG = sum(v < 0 for v in _hard)
S0_EASY = [S0[(st, m)] for st in SETS[3:] for m in M4]
assert min(S0_EASY) > 0
SRC_CV = (float(SRC.src_cv_R2.min()), float(SRC.src_cv_R2.max()))

# 2018→2019：k=10 截距校正与 O1 之差；斜率-截距与加样重建都低于截距校正
Y1_S1 = [rmed('2018→2019', m, 'S1', 10) for m in M4]
Y1_GAP = [O1[('2018→2019', m)] - rmed('2018→2019', m, 'S1', 10) for m in M4]
assert max(Y1_GAP) < 0.03, '正文写「与事后参照相差不足 0.03」'
assert all(rmed('2018→2019', m, s, 10) < rmed('2018→2019', m, 'S1', 10) for m in M4 for s in ('S2', 'S3'))
# 2019→2018：k≥10 时斜率-截距优于截距（四法）
assert all(rmed('2019→2018', m, 'S2', k) > rmed('2019→2018', m, 'S1', k) for m in M4 for k in (10, 20, 40)), '正文写 k≥10 时四法均如此'
# 误差结构：事后斜率-截距校正（O2）比事后截距校正（O1）多出的部分
O21_Y1 = [O2[('2018→2019', m)] - O1[('2018→2019', m)] for m in M4]
O21_Y2 = [O2[('2019→2018', m)] - O1[('2019→2018', m)] for m in M4]
assert max(abs(v) for v in O21_Y1) < 0.01, '正文写 2018→2019 时 O2 与 O1 相差不足 0.01'
assert min(O21_Y2) > 0.05, O21_Y2
# 留新疆：k≤40 的截距与斜率-截距校正中位数上限；k=40 加样重建
XJ_LIN_MAX = max(rmed('留新疆', m, s, k) for m in M4 for s in ('S1', 'S2') for k in (5, 10, 20, 40))
XJ_O2 = [O2[('留新疆', m)] for m in M4]
XJ_S3_40 = {m: rmed('留新疆', m, 'S3', 40) for m in M4}
assert all(XJ_S3_40[m] > XJ_LIN_MAX for m in M4), '正文写 k=40 的加样重建使四法均高于线性校正的上限'
assert max(XJ_S3_40, key=lambda m: XJ_S3_40[m]) == 'SVR', XJ_S3_40
# 2019→2018：k=40 时各法三种做法的最好值
Y2_BEST40 = {m: max(rmed('2019→2018', m, s, 40) for s in ('S1', 'S2', 'S3')) for m in M4}
# 留山东、留陕西：k=5 斜率-截距低于不校正的组合数（同一评价集上比较）
EASY_S2_LOSS = sum(rmed(st, m, 'S2', 5) < rmed(st, m, 'S0', 5) for st in ('留山东', '留陕西') for m in M4)
assert EASY_S2_LOSS == 2 * len(M4), '3.3 节写 k=5 的斜率-截距校正在山东、陕西两种设置下均低于不校正'
EASY_GAIN = max(rmed(st, m, s, k) - rmed(st, m, 'S0', k) for st in ('留山东', '留陕西') for m in M4
                for s in ('S1', 'S2', 'S3') for k in (5, 10, 20, 40))

# 附录表 A3 只列 k=10 的分位；正文写「分位区间随 k 增大而收窄」，按 S1~S3 的平均区间宽度逐 k 守住
_RS13 = RS[RS.strategy.isin(['S1', 'S2', 'S3'])].assign(width=lambda x: x.R2_hi - x.R2_lo)
Q_WIDTH = _RS13.groupby('k').width.mean()
assert Q_WIDTH.is_monotonic_decreasing, Q_WIDTH

RANGE = rd(f'{OUT}/49rebuttal_recal_range.csv').set_index('setting')
BAT = rd(f'{OUT}/46rebuttal_recal_batches.csv').set_index('name')

# ============================ 数据清洗（59，协议 B10.10）============================
CLEAN = rd(f'{OUT}/59data_clean_summary.csv', dtype={'year': str}).set_index('year')
CLEAN_LIST = rd(f'{OUT}/59data_clean_list.csv')
_CA = CLEAN.loc['合计']
RAW_N = {k: int(CLEAN.loc[k, 'raw']) for k in ('2018', '2019', '合计')}
RULE_N = {r: int(_CA[r]) for r in ('R1', 'R2', 'R3', 'R4')}
R3_GROUPS = int(_CA.R3_groups)
REMOVED, KEPT = int(_CA.removed), int(_CA.kept)
assert KEPT == KEPT_N
R2_SSC = CLEAN_LIST.loc[CLEAN_LIST.R2, 'SSC'].tolist()
assert len(R2_SSC) == RULE_N['R2'] == 1, '2.1 节写 R2 剔除 1 个并给出其记录值'
# 2.1 节规则②：其余记录的 SSC 范围（写作「均在 a~b°Brix 之间」，两端各向外取整）
SSC_OTHER = (float(CLEAN_LIST.loc[~CLEAN_LIST.R2, 'SSC'].min()), float(CLEAN_LIST.loc[~CLEAN_LIST.R2, 'SSC'].max()))
assert SSC_OTHER[1] < 20 < R2_SSC[0], (SSC_OTHER, R2_SSC)
assert int(_CA.R1_and_R4) == RULE_N['R4'], '2.1 节写 R4 命中的光谱均已被 R1 剔除'
assert KEPT == int(BAT.loc['全体', 'n']), '清洗汇总与 46 的批次描述样品数须相同'
assert all(int(CLEAN.loc[k, 'kept']) == int(BAT.loc[k, 'n']) for k in ('2018', '2019'))

# 主分析弱折的 2 条光谱在 PLSR 同作杠杆回退时的变化（61，附录 A8）：每次回退到的潜变量数、预测是否移向全体样品均值、
# 各条回退前后的绝对误差范围。最大读数最低的一条 PLSR 原本预测较准，回退后误差全部增大；另一条全部减小
_hp = MFS[MFS.method == 'PLSR']
assert _hp.backed.astype(str).eq('True').all(), '附录写这 2 条光谱在 PLSR 下每次都被回退'
PLSR_LEV_K = sorted({int(v) for v in _hp.used_lev})
assert len(PLSR_LEV_K) == 1, PLSR_LEV_K
_ybar = float(BAT.loc['全体', 'mean'])
assert bool(((_hp.pred_lev - _ybar).abs() < (_hp.pred - _ybar).abs()).all()), '附录写回退后预测移向样品均值'
PLSR_HOT_RANGE = {int(i): ((float(g.err.abs().min()), float(g.err.abs().max())),
                           (float(g.err_lev.abs().min()), float(g.err_lev.abs().max()))) for i, g in _hp.groupby('idx')}
_oth = [i for i in PLSR_HOT_RANGE if i != MF_WEAKEST]
assert len(_oth) == 1, PLSR_HOT_RANGE
PLSR_WEAKEST, PLSR_OTHER = PLSR_HOT_RANGE[MF_WEAKEST], PLSR_HOT_RANGE[_oth[0]]
assert PLSR_WEAKEST[1][0] > PLSR_WEAKEST[0][1], '附录写最大读数最低的一条回退后误差增大'
assert PLSR_OTHER[1][1] < PLSR_OTHER[0][0], '附录写另一条回退后误差减小'
assert PLSR_WEAKEST[0][1] < PLSR_OTHER[0][0], '附录写 PLSR 原本对最大读数最低的一条预测较准'
PRE_N = int(_CA.n_before)                                  # 清洗前（2026-09-21 定稿时）的样品数
# 2.1 节写：清洗前只按 5<SSC<25°Brix 筛去 1 条，即规则②命中的记录
assert RAW_N['合计'] - PRE_N == 1, (RAW_N, PRE_N)
assert not 5 < R2_SSC[0] < 25, R2_SSC
# KFold 把余数分给前几折：外层 5 折与假想的 10 折下各折测试、训练样品数的范围（2.3 节）
TE5 = (KEPT // 5, KEPT // 5 + (1 if KEPT % 5 else 0))
TR5 = (KEPT - TE5[1], KEPT - TE5[0])
TE10 = (KEPT // 10, KEPT // 10 + (1 if KEPT % 10 else 0))
TR10 = (KEPT - TE10[1], KEPT - TE10[0])
# 清洗前的 6 条异常光谱（B10.9 名单）各被哪条规则剔除
_HOT_PRE = (71, 115, 125, 126, 131, 223)
_hp = CLEAN_LIST[CLEAN_LIST.old_ok & CLEAN_LIST.old_idx.isin(_HOT_PRE)].set_index('old_idx')
assert len(_hp) == 6
HOT_PRE_REMOVED = [int(i) for i in _hp.index[~_hp.keep]]
HOT_PRE_KEPT = [int(i) for i in _hp.index[_hp.keep]]
assert sorted(int(x) for x in _hp.loc[HOT_PRE_KEPT, 'new_idx']) == MF_HOT, '保留的 2 条即主分析弱折中误差最大的 2 条'
assert all(bool(_hp.loc[i, 'R1']) for i in HOT_PRE_REMOVED), '被剔除的 4 条均为规则①命中'
HOT_PRE_R3 = [i for i in HOT_PRE_REMOVED if bool(_hp.loc[i, 'R3'])]
HOT_PRE_R4 = [i for i in HOT_PRE_REMOVED if bool(_hp.loc[i, 'R4'])]
assert len(HOT_PRE_R3) == 2, HOT_PRE_R3
assert len(HOT_PRE_R4) == 1, HOT_PRE_R4
# 保留的 2 条（主分析弱折）按规则①的距离与所在年份的阈值、中位数（附录 A9）：二者低于阈值，故未被规则①剔除
_hk = CLEAN_LIST[CLEAN_LIST.keep & CLEAN_LIST.new_idx.isin(MF_HOT)].set_index('new_idx')
assert len(_hk) == 2, _hk
assert (_hk.year == 2018).all(), _hk
_d18 = CLEAN_LIST.loc[CLEAN_LIST.year == 2018, 'R1_dist']
R1_THR18, R1_MED18 = float(np.percentile(_d18, 90)), float(_d18.median())      # 与 data_clean.rule_flags 的 np.percentile(d, 90) 同式
HOT_KEPT_R1 = [float(_hk.loc[i, 'R1_dist']) for i in [MF_WEAKEST] + [i for i in MF_HOT if i != MF_WEAKEST]]   # 先写最大读数最低的一条
assert max(HOT_KEPT_R1) < R1_THR18, (HOT_KEPT_R1, R1_THR18)
assert CLEAN_LIST.loc[(CLEAN_LIST.year == 2018) & CLEAN_LIST.R1, 'R1_dist'].min() > R1_THR18
# 2.4 节：最大的 k=40 不到每个目标批次的 1/8
assert 40 * 8 < min(int(BAT.loc[b, 'n']) for b in ('2018', '2019', '新疆', '山东', '陕西')), BAT.n

# ============================ 系数函数（48、49）============================
BAND = rd(f'{OUT}/49rebuttal_beta_band.csv').set_index('region')
WIN = rd(f'{OUT}/48rebuttal_beta_windows.csv')
WIN_TOP = WIN[(WIN.kind == '绝对值前400点') & (WIN.n_points >= 90)].sort_values('n_points', ascending=False)
assert len(WIN_TOP) == 2, WIN_TOP
MAINW, SUBW = WIN_TOP.iloc[0], WIN_TOP.iloc[1]
WIN_ABS = WIN[WIN.kind == '绝对值前400点'].sort_values('n_points', ascending=False)
WIN_N = len(WIN_ABS)
WIN_REST_MAX = int(WIN_ABS.n_points.iloc[2:].max())
BAND_MAIN = BAND.loc[f'{MAINW.lo:.2f}~{MAINW.hi:.2f}']          # 49 的区段名按两位小数写
BAND_SUB = BAND.loc[f'{SUBW.lo:.2f}~{SUBW.hi:.2f}']
BAND_REST = BAND.loc['其余波长']
# 3.4 节写前 400 点「主要落在」两个窗口；符号一致比例的分母是窗口范围内的全部波长点（49 产物的 n_points）
assert (int(MAINW.n_points) + int(SUBW.n_points)) / 400 > 0.8
assert int(BAND_SUB.n_points) == int(SUBW.n_points)
assert BAND_SUB.median_band_over_absbeta > BAND_MAIN.median_band_over_absbeta, '3.4 节写后一窗口的幅值更不稳定'


def nm(v, nd=1):
    """波长写法：四位整数部分每三位空一格，如 1 124.9。"""
    s = f'{v:.{nd}f}'
    ip, _, dp = s.partition('.')
    if len(ip) >= 4:
        ip = f'{ip[:-3]} {ip[-3:]}'
    return ip + ('.' + dp if dp else '')


# ============================ 清洗前后的对照（附录 A9，协议 B10.10「报告」一段所列四项）============================
# 清洗前的量取自 59 从提交 2331884（2026-09-21 定稿）导出的 59raw_reference.csv，清洗后的量取本模块上文的同一口径
RAW_REF = rd(f'{OUT}/59raw_reference.csv')


def _raw_ref(item):
    q = RAW_REF[RAW_REF.item == item]
    assert len(q), item
    return q


CMP_T1 = pd.DataFrame({'pre': _raw_ref('表1_R2均值').set_index('key').value.astype(float),
                       'post': MAIN.R2.astype(float)}).reindex(ALL6)
# 附录 A9 的升幅按表 A7 所列的三位小数计算，使正文与表逐项对得上
_CMP_T1_UP_DISP = [float(f'{b:.3f}') - float(f'{a:.3f}') for a, b in zip(CMP_T1.pre, CMP_T1.post, strict=True)]
CMP_T1_UP = (min(_CMP_T1_UP_DISP), max(_CMP_T1_UP_DISP))
assert CMP_T1_UP[0] > 0, '附录 A9 写清洗后各方法的 R² 均值均升高'
_pp = _raw_ref('六法15对').set_index('key')
CMP_PAIRS = pd.DataFrame({'pre_diff': _pp.value.astype(float), 'pre_verdict': _pp.key2,
                          'post_diff': _PTM.set_index('pair')['diff'].reindex(_pp.index),
                          'post_verdict': _PTM.set_index('pair')['结论'].reindex(_pp.index)})
CMP_FLIP = CMP_PAIRS[CMP_PAIRS.pre_verdict != CMP_PAIRS.post_verdict]
# 附录 A5 的 GBIDS 15 对在清洗前后的判定（附录 A9）
_pa5 = _raw_ref('A5_GBIDS15对').set_index('key')
CMP_A5 = pd.DataFrame({'pre_verdict': _pa5.key2, 'post_verdict': A5_PAIRS.set_index('pair')['结论']})
assert len(CMP_A5) == 15, CMP_A5
assert CMP_A5.notna().all().all(), CMP_A5
CMP_A5_FLIP = CMP_A5[CMP_A5.pre_verdict != CMP_A5.post_verdict]
_sp = _raw_ref('小样本_cv_R2负值次数')
CMP_SS = pd.DataFrame([{'method': r.key, 'N': int(r.key2), 'pre': int(float(r.value)),
                        'post': int(SS.loc[(int(r.key2), r.key, 'cv'), 'n_neg'])} for r in _sp.itertuples()])
_rp = _raw_ref('跨批次_2018→2019_k10_S1').set_index('key')
CMP_RC = pd.DataFrame({'pre': _rp.value.astype(float),
                       'post': pd.Series({m: rmed('2018→2019', m, 'S1', 10) for m in _rp.index})})
CMP_SS_FUN = CMP_SS[CMP_SS.method.isin(_FUN)]


def _pre_arm_ok(item, m, arm):
    q = _raw_ref(item).assign(N=lambda t: t.key2.astype(float).astype(int)).set_index(['key', 'N'])
    neg = [int(float(q.loc[(f'{m}|{arm}', n), 'value'])) for n in G_N]
    d = [float(q.loc[(f'{m}|{arm}', n), 'extra']) - float(q.loc[(f'{m}|cv', n), 'extra']) for n in G_N]
    return all(v == 0 for v in neg) and min(d) >= -0.02


_ARMS_ALL = [('选参端对策', 'onese', '一倍标准误准则', ONESE_OK), ('选参端对策', 'repcv', '重复交叉验证', REPCV_OK),
             ('预测端对策', 'lev', '杠杆回退', {m: GUARD_OK[(m, 'lev')] for m in _FUN}),
             ('预测端对策', 'clip', '范围截断', {m: GUARD_OK[(m, 'clip')] for m in _FUN})]
CMP_GUARD = pd.DataFrame([{'arm': lab, 'method': m, 'pre': _pre_arm_ok(item, m, a), 'post': bool(post[m])}
                          for item, a, lab, post in _ARMS_ALL for m in _FUN])
# 清洗前的判定与协议 B10.8、B10.9 的结果段一致
assert CMP_GUARD.set_index(['arm', 'method']).pre.to_dict() == {
    ('一倍标准误准则', 'FPCR'): False, ('一倍标准误准则', 'FPLR'): False, ('重复交叉验证', 'FPCR'): False, ('重复交叉验证', 'FPLR'): False,
    ('杠杆回退', 'FPCR'): True, ('杠杆回退', 'FPLR'): True, ('范围截断', 'FPCR'): True, ('范围截断', 'FPLR'): False}, CMP_GUARD
# 清洗前按同一判读规则 (iv) 在正文报告的预测端做法（协议 B10.9 结果段：两种模型均为杠杆回退）
def _pre_main_arm(m):
    q = _raw_ref('预测端对策').assign(N=lambda t: t.key2.astype(float).astype(int)).set_index(['key', 'N'])
    ok = {a: _pre_arm_ok('预测端对策', m, a) for a in ('lev', 'clip')}
    if not any(ok.values()):
        return None
    if not all(ok.values()):
        return 'lev' if ok['lev'] else 'clip'
    tot = {a: sum(int(float(q.loc[(f'{m}|{a}', n), 'value'])) for n in G_N) for a in ('lev', 'clip')}
    if tot['lev'] != tot['clip']:
        return min(('lev', 'clip'), key=lambda a: tot[a])
    return max(('lev', 'clip'), key=lambda a: sum(float(q.loc[(f'{m}|{a}', n), 'extra']) for n in G_N))


PRE_MAIN_ARM = {m: _pre_main_arm(m) for m in _FUN}
assert PRE_MAIN_ARM == {'FPCR': 'lev', 'FPLR': 'lev'}, PRE_MAIN_ARM
# 附录 A8 另取 100 次抽样的理由：方案在清洗前设定时，N=50~120 的失效频率最低为每 20 次 1 次；
# 某档频率若仍为这一最低值，100 次中一次都不出现与 20 次中一次都不出现的概率
PRE_FAIL_MIN = min(CMP_SS_FUN.pre) / 20
P0_100, P0_20 = (1 - PRE_FAIL_MIN) ** 100, (1 - PRE_FAIL_MIN) ** 20
assert min(CMP_SS_FUN.pre) >= 1


# ============================ 产物生成时间 ============================
# 全部数字须取自协议 B10.10 清洗之后生成的产物。服务器结果取回之前，本机仍留着清洗前的同名文件
# （如 44fully_nested_rf.csv），照读会把清洗前的数排进稿件；故以清洗规则的唯一实现 02code/data_clean.py 的修改时间为基准，
# 本模块读过的每个文件都不得早于它（exFAT 的时间戳精度为 2 s，留 2 s 余量）。不取 59 产物的时间：59 只写清单，
# 读数入口经 data_clean.keep_mask() 现算剔除，59 为补充输出而重跑时，早先已在清洗后数据上算出的产物并未过时。
_CLEAN_T = Path('02code/data_clean.py').stat().st_mtime
STALE = sorted({p for p in _READS if Path(p).stat().st_mtime < _CLEAN_T - 2})
# 试排草稿时可用 MS_STALE_OK 点名放行少数尚待服务器重算的文件（逗号分隔的文件名），放行的文件照样读、并逐个打印警告；
# 正式构建不设此变量
_STALE_OK = {x.strip() for x in os.environ.get('MS_STALE_OK', '').split(',') if x.strip()}
for _p in [p for p in STALE if Path(p).name in _STALE_OK]:
    print(f'⚠️ 草稿：{_p} 生成于数据清洗之前，按 MS_STALE_OK 放行，稿中由它得出的数字不可用')
STALE = [p for p in STALE if Path(p).name not in _STALE_OK]
assert not STALE, f'以下文件生成于数据清洗之前，须先重算：{STALE}'


if __name__ == '__main__':
    print('已完成方法', DONE, '未完成', PENDING, '主分析族', FAM_MAIN)
    for m in DONE:
        print(f'  {m:<6} R²={acc(m)}  RMSE={acc(m, "RMSE")}  RPD={acc(m, "RPD", 2)}')
    for r in PT.itertuples():
        print(f'  {r.family} {r.pair}: {fm(r.diff)} [{rg(r.ci_lo, r.ci_hi)}] Holm {r.holm_p:.3f} 占优 {r.a_wins}/25')
    print('显著对', SIG_TEXT, '；非正态', MAIN_NONNORMAL)
    print('GBIDS 与 R² 的异同', a5_vs_r2())
    print('第三轮', changed_text())
    print('网格端点最多折数', BND_MAX, SECOND_OVER50, '；预处理四法平均', f'{PREP_BEST:.3f}', rg(PREP_OTHERS.min(), PREP_OTHERS.max()))
    print('默认超参数 SVR/PLSR', f'{DEF_SVR:.3f}/{DEF_PLSR:.3f}')
    print('三层全嵌套', FN.R2.round(3).to_dict(), '同种子', S1POOL.round(3).to_dict())
    print('弱折', WEAK, {m: round(v, 3) for m, v in WEAK_R2.items()}, '名单', MF_HOT, MF_HOT_OLD, '误差', WEAK_ERR)
    print('除去名单', EXCL, '杠杆回退', LEVM, LEV_WEAK, LEV_OTHER)
    print('小样本 FPLR 中位数', [round(v, 3) for v in FPLR_MID_MED], 'cv−fixed', CV_GAIN_ALL, 'FPLR>SVR 次数', FPLR_VS_SVR)
    print('B10.8', D1SE_RANGE, DREP_RANGE, 'B10.9', GUARD_OK, CV57_NEG, '名单', HOT57)
    print('S0 难设置负值数', S0_HARD_NEG, '易设置', rg(min(S0_EASY), max(S0_EASY)), '源内 CV R²', rg(*SRC_CV))
    print('2018→2019 k=10 S1', [round(x, 3) for x in Y1_S1], '距 O1', [round(x, 3) for x in Y1_GAP])
    print('留新疆线性校正上限', round(XJ_LIN_MAX, 3), 'O2', [round(x, 3) for x in XJ_O2], 'S3 k=40', XJ_S3_40)
    print('A9', CMP_T1_UP, len(CMP_FLIP), CMP_SS_FUN.to_dict('records'))
    print('全体检查通过')
