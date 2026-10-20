# -*- coding: utf-8 -*-
"""42recompute_manuscript_numbers.py: 按定义从低一层产物复算修订稿主结论链的数字，与稿件字面值逐一对照。

为什么单列一个脚本：29manuscript_check.py 第七项只能证明「稿件里那串三位小数在某个产物里出现过」，
证不了「它是 FPLR−PLSR 在 25 个配对折上的修正 t 检验区间下界」；而稿件的数字全部取自 02code/ms_numbers.py，
拿同一份代码核对自己等于自审。本脚本不导入 ms_numbers，也不读它所用的汇总表（47 的检验表、49 的小样本汇总、
46 的抽样汇总、49 的分位带汇总），而是从更低一层的逐折、逐次抽样、逐波长产物按各量的定义重算，
再到稿件全文里找对应的字面值。找的尽量是带上下文的片段（如「0.037（95%置信区间0.007~0.068）」），
不是孤立的三位小数；稿件用文字描述的数据模式（「均高于」「均不超过」）另按复算值判定是否成立。

口径（与稿件 1.4~1.6 节一致；复算项名称与注释里的节、图、表编号均为现稿编号）：
  表 2        25 个外层折的均值 ± 样本标准差（ddof=1）
  配对检验    同一种子、同一外层折上的 R² 之差，J=25；修正重复 k 折 t 检验，方差因子 1/J + 1/(k−1)=1/J+1/4，自由度 24；
              95% 置信区间不作多重比较校正；Holm 校正按主分析与基函数敏感性两族分别计算
  小样本      每个 N 抽样 20 次；表 A2 为均值（样本标准差/√20）
  跨批次      200 次抽样的中位数与 2.5%~97.5% 分位（numpy.nanquantile 线性插值）
  「不超过 x」类上界按向上取整到所写位数判定

判据：「未找到 0 项」，否则以退出码 1 结束。RF 与 1D-CNN 的 5 折×5 次结果落地之前，表 2 这两行及其配对检验
自动跳过并在末尾列出；六个方法齐备后主分析族自动扩为 15 对。

运行方式:
    cd <项目根目录>
    MS_DOCX=<稿件 docx 路径> python3 02code/42recompute_manuscript_numbers.py > 05logs/42recompute.log 2>&1
    R42_NO_DOCX=1 python3 02code/42recompute_manuscript_numbers.py

输出文件:
    仅标准输出（逐项对照表与未找到清单）；只读，不修改任何稿件或产物，不读 05data。
"""
import itertools
from decimal import ROUND_HALF_UP, Decimal
import ast
import json
import glob
import hashlib
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from docx import Document
from scipy import stats
from sklearn.metrics import r2_score

OUT, LOGS = '04outputs', '05logs'
DOCX = os.environ.get('MS_DOCX', '06doc/01manuscript/苹果SSC函数型小样本预测-投稿稿.docx')
NO_DOCX = os.environ.get('R42_NO_DOCX') == '1'
MINUS = '−'
SEED0 = 20041210                                   # 附录 A3 单种子复核所用的种子
_CLEAN_IMPL = Path('02code/data_clean.py')
_CLEAN_MTIME = _CLEAN_IMPL.stat().st_mtime
PENDING: list[tuple[str, str]] = []
_REQUIRED_54 = [
    '04outputs/54rebuttal_table1.csv', '04outputs/54rebuttal_paired_tests.csv',
    '04outputs/54rebuttal_crossround.csv', '04outputs/54rebuttal_boundary.csv',
    '04outputs/54rebuttal_provenance.csv', '04outputs/54rebuttal_COMPLETE.json',
]
_REQUIRED_61 = [
    '04outputs/61rebuttal_mainfold_folds.csv', '04outputs/61rebuttal_mainfold_spectra.csv',
    '04outputs/61rebuttal_mainfold_summary.csv',
]


def _fresh(path: str | Path) -> bool:
    target = Path(path)
    if not target.exists() or target.stat().st_mtime < _CLEAN_MTIME - 2:
        item = (str(target), '不存在或早于 data_clean.py 2 s')
        if item not in PENDING:
            PENDING.append(item)
        return False
    return True


def _read_csv(path: str | Path, name: str) -> pd.DataFrame | None:
    if not _fresh(path):
        PENDING.append((name, str(path)))
        return None
    return pd.read_csv(path)

_txt: list[str] = []
if not NO_DOCX:
    _doc = Document(DOCX)
    _txt = [p.text for p in _doc.paragraphs]
    for _tb in _doc.tables:                        # 各表的数字在单元格里，按行逐格拼接
        for _row in _tb.rows:
            _txt += [c.text for c in _row.cells]
# 汉字与数字之间的空格、千分位空格一律压掉；负号统一成 ASCII 连字符后比较
FLAT = re.sub(r'\s+', '', '\n'.join(_txt)).replace(MINUS, '-')

rows: list[tuple[str, str, bool, str]] = []


def chk(name: str, lit: str) -> None:
    """lit 为按复算值写出的稿件应有片段。"""
    rows.append((name, lit, True if NO_DOCX else re.sub(r'\s+', '', lit).replace(MINUS, '-') in FLAT, 'chk'))


def cond(name: str, ok: bool, what: str) -> None:
    """稿件用文字描述的数据模式：按复算值判定是否成立。"""
    rows.append((name, what, bool(ok), 'cond'))


def f(v: float, nd: int = 3) -> str:
    s = f'{v:.{nd}f}'
    if float(s) == 0:
        s = f'-{0:.{nd}f}' if v < 0 else f'{0:.{nd}f}'
    return s.replace('-', MINUS)


def rg(a: float, b: float, nd: int = 3) -> str:
    return f'{f(a, nd)}~{f(b, nd)}'


def up(v: float, nd: int = 2) -> str:
    q = 10 ** nd
    return f(math.ceil(v * q - 1e-9) / q, nd)


def pct(v: float) -> str:
    if v == 0:
        return '0'
    return str(Decimal(repr(v * 100)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)) + '%'


def exp10(v: float) -> str:
    """10 的整数次幂按稿件文字层写法：10−6（上标在文字层里被拍平）。"""
    e = round(math.log10(v))
    return '1' if e == 0 else ('10' if e == 1 else f'10{e}')


def p3(v: float) -> str:
    return '<0.001' if v < 0.001 else f'{v:.3f}'


def pw(v: float) -> str:
    """「p<0.001」或「p=0.414」中 p 之后的部分。"""
    return '<0.001' if v < 0.001 else f'={v:.3f}'


# ============================ 表 2 与配对检验（45 逐折）============================
ALL6 = ['SVR', 'FPLR', 'FPCR', 'PLSR', 'RF', '1D-CNN']
STEM = {'SVR': 'none_SVR', 'FPLR': 'bspline_FPLR', 'FPCR': 'bspline_FPCR', 'PLSR': 'none_PLSR',
        'RF': 'none_RF', '1D-CNN': 'none_1DCNN'}
FOLDS: dict[str, pd.DataFrame] = {}
for _m in ALL6:
    _p = Path(f'{OUT}/45rebuttal_ncv_{STEM[_m]}.csv')
    if _p.exists():
        if not _fresh(_p):
            continue
        _d = pd.read_csv(_p)
        assert len(_d) == 25, _m
        assert len(_d[['seed', 'fold']].drop_duplicates()) == 25, _m
        FOLDS[_m] = _d
    else:
        PENDING.append((f'表1 {_m}', str(_p)))
DONE = [m for m in ALL6 if m in FOLDS]
PEND = [m for m in ALL6 if m not in FOLDS]

for _m in DONE:
    _d = FOLDS[_m]
    rep = '函数型' if _m in ('FPLR', 'FPCR') else '离散'
    chk(f'表2 {_m} 一行', f'{_m}{rep}{_d.R2.mean():.3f}±{_d.R2.std(ddof=1):.3f}{_d.RMSE.mean():.3f}±{_d.RMSE.std(ddof=1):.3f}'
                         f'{_d.RPD.mean():.2f}±{_d.RPD.std(ddof=1):.2f}')
    _pred_path = Path(f'{OUT}/45rebuttal_ncv_{STEM[_m]}_pred.csv')
    _pred = _read_csv(_pred_path, f'表1 {_m} 逐样品预测')
    if _pred is not None:
        _metric_rows = []
        for (_seed, _fold), _group in _pred.groupby(['seed', 'fold']):
            _y = _group.y.to_numpy(dtype=float)
            _p = _group.pred.to_numpy(dtype=float)
            _rmse = float(np.sqrt(np.mean((_y - _p) ** 2)))
            _metric_rows.append((int(_seed), int(_fold), float(r2_score(_y, _p)), _rmse,
                                 float(np.std(_y, ddof=0) / _rmse)))
        _metric = pd.DataFrame(_metric_rows, columns=['seed', 'fold', 'R2', 'RMSE', 'RPD'])
        _joined = _d[['seed', 'fold', 'R2', 'RMSE', 'RPD']].merge(_metric, on=['seed', 'fold'], suffixes=('_listed', '_recomputed'))
        _max_diff = {column: float(np.max(np.abs(_joined[f'{column}_listed'] - _joined[f'{column}_recomputed'])))
                     for column in ('R2', 'RMSE', 'RPD')}
        cond(f'表2 {_m} 逐折定义复算', all(value <= 1e-9 for value in _max_diff.values()),
             '最大差 ' + '、'.join(f'{key}={value:.3e}' for key, value in _max_diff.items()))
if DONE:
    cond('摘要 SVR 的 R² 在已完成方法中最高', max(DONE, key=lambda m: FOLDS[m].R2.mean()) == 'SVR', 'SVR 均值最大')


def paired(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    mg = a[['seed', 'fold', 'R2']].merge(b[['seed', 'fold', 'R2']], on=['seed', 'fold'], suffixes=('_a', '_b'))
    assert len(mg) == 25
    dd = (mg.R2_a - mg.R2_b).to_numpy(dtype=float)
    j = len(dd)
    se = math.sqrt((1 / j + 1 / 4) * dd.var(ddof=1))
    t = float(dd.mean()) / se
    half = float(stats.t.ppf(0.975, j - 1)) * se
    return dict(d=float(dd.mean()), lo=float(dd.mean()) - half, hi=float(dd.mean()) + half,
                p=float(2 * stats.t.sf(abs(t), j - 1)), sw=float(stats.shapiro(dd).pvalue),
                wx=float(stats.wilcoxon(dd).pvalue))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj = [0.0] * len(ps)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


PAIRS = {(a, b): paired(FOLDS[a], FOLDS[b]) for a, b in itertools.combinations(DONE, 2)}
for _v, _h in zip(PAIRS.values(), holm([v['p'] for v in PAIRS.values()]), strict=True):
    _v['holm'] = _h


def pr(a: str, b: str) -> dict[str, float]:
    if (a, b) in PAIRS:
        return PAIRS[(a, b)]
    s = PAIRS[(b, a)]
    return dict(s, d=-s['d'], lo=-s['hi'], hi=-s['lo'])


_x, _y = pr('FPLR', 'PLSR'), pr('FPCR', 'PLSR')
chk('2.2 FPLR、FPCR 与 PLSR 之差及区间', f"FPLR、FPCR与PLSR之差分别为{f(_x['d'])}（{rg(_x['lo'], _x['hi'])}）与{f(_y['d'])}（{rg(_y['lo'], _y['hi'])}）")
chk('2.2 两对 Holm 校正后 p', f"Holm校正后p为{f(_x['holm'])}与{f(_y['holm'])}")
cond('2.2 两对未校正区间不含 0、Holm 后未检出', _x['lo'] > 0 and _y['lo'] > 0 and min(_x['holm'], _y['holm']) >= 0.05,
     '区间下界 >0 且 Holm p≥0.05')
_s1, _s2 = pr('SVR', 'FPLR'), pr('SVR', 'FPCR')
chk('2.2 SVR 与 FPLR、FPCR 之差及区间', f"之差为{f(_s1['d'])}（{rg(_s1['lo'], _s1['hi'])}）与{f(_s2['d'])}（{rg(_s2['lo'], _s2['hi'])}）")
chk('2.2 不能排除的幅度', f"FPLR比SVR低{f(_s1['hi'])}，也不能排除高{f(-_s1['lo'])}")
_sp1 = pr('SVR', 'PLSR')
chk('2.2 SVR 与 PLSR 之差及区间', f"SVR与PLSR之差为{f(_sp1['d'])}（95%置信区间{rg(_sp1['lo'], _sp1['hi'])}），Holm校正后差异显著")
if len(DONE) == len(ALL6):
    _sig_pairs = [pair for pair, value in PAIRS.items() if value['holm'] < 0.05]
    _nonnormal = [pair for pair, value in PAIRS.items() if value['sw'] < 0.05]
    # 显著对写成「较高一方 高于 较低一方」：较低一方集合相同的高方合并，组与组内的次序按表 2 的 R² 均值
    _order6 = sorted(ALL6, key=lambda m: -FOLDS[m].R2.mean())
    _win: dict[str, list[str]] = {}
    for _a, _b in _sig_pairs:
        _hi_m, _lo_m = (_a, _b) if PAIRS[(_a, _b)]['d'] > 0 else (_b, _a)
        _win.setdefault(_hi_m, []).append(_lo_m)
    _grp: dict[tuple[str, ...], list[str]] = {}
    for _hi_m in sorted(_win, key=_order6.index):
        _grp.setdefault(tuple(sorted(_win[_hi_m], key=_order6.index)), []).append(_hi_m)

    def _cn(xs: list[str]) -> str:
        return xs[0] if len(xs) == 1 else '、'.join(xs[:-1]) + '与' + xs[-1]

    _sig_phrase = '，'.join(f"{_cn(ws)}高于{_cn(list(ls))}" for ls, ws in _grp.items())
    chk('2.2 主分析显著集合', f"15对比较中，{_sig_phrase}，差异显著，其余{15 - len(_sig_pairs)}对未检出差异")
    chk('结论（1）主分析显著集合', f"统一协议下，{_sig_phrase}，差异显著")
    chk('2.2 六法的 R² 次序', f"SVR的R²最高，随后依次为{'、'.join(_order6[1:-1])}与{_order6[-1]}")
    cond('2.2 SVR 的 R² 最高', _order6[0] == 'SVR', str(_order6))
    cond('1.6、2.2 主分析非正态分支', all('FPLR' in pair or 'FPCR' in pair for pair in _nonnormal)
         and all(PAIRS[pair]['wx'] < 0.001 for pair in _nonnormal),
         f'{len(_nonnormal)} 对均涉及函数型模型且 Wilcoxon p<0.001')
    chk('2.2 差值非正态的对数', f"差值非正态的{len(_nonnormal)}对均涉及函数型模型，补充的Wilcoxon检验p均小于0.001")
else:
    PENDING.append(('3.1、结论（1）主分析显著集合', '04outputs/45rebuttal_ncv_none_RF.csv'))
    PENDING.append(('2.5、3.1 主分析非正态分支', '04outputs/45rebuttal_ncv_none_RF.csv'))
chk('摘要 SVR、FPLR 与 PLSR 的比较', f"SVR的R²为{FOLDS['SVR'].R2.mean():.3f}，与PLSR之差经Holm校正差异显著；"
                              f"FPLR为{FOLDS['FPLR'].R2.mean():.3f}，与二者之差均未检出差异")
chk('Abstract SVR、FPLR 与 PLSR 的比较', f"SVR gave R² of {FOLDS['SVR'].R2.mean():.3f} and differed from PLSR after Holm adjustment. "
                                    f"FPLR gave {FOLDS['FPLR'].R2.mean():.3f} with no difference detected from either")
cond('摘要 SVR 与 PLSR 之差经 Holm 校正差异显著', pr('SVR', 'PLSR')['holm'] < 0.05 and pr('SVR', 'PLSR')['d'] > 0,
     f"Holm p {pr('SVR', 'PLSR')['holm']:.4f}")
cond('摘要 FPLR 与 PLSR、SVR 经 Holm 后均未检出差异', _x['holm'] >= 0.05 and pr('SVR', 'FPLR')['holm'] >= 0.05,
     f"Holm p {_x['holm']:.3f}、{pr('SVR', 'FPLR')['holm']:.3f}")

# ============================ 2.2 第 2 段：函数型模型折间波动来自的外层折（45 逐折、逐样品预测）============================
_sd = {m: FOLDS[m].R2.std(ddof=1) for m in DONE}
cond('2.2 函数型模型 25 折标准差约为 SVR 的两倍', 'SVR' in _sd and all(1.8 * _sd['SVR'] < _sd[m] < 2.5 * _sd['SVR'] for m in ('FPLR', 'FPCR')),
     '、'.join(f'{m} {_sd[m]:.3f}' for m in ('FPLR', 'FPCR', 'SVR')))
chk('2.2 25 折标准差', f"函数型模型25折的标准差（FPLR{_sd['FPLR']:.3f}、FPCR{_sd['FPCR']:.3f}）约为SVR（{_sd['SVR']:.3f}）的两倍")
_wk_row = FOLDS['FPLR'].loc[FOLDS['FPLR'].R2.idxmin()]
WEAK = (int(_wk_row.seed), int(_wk_row.fold))
cond('2.2 两种函数型模型的最弱一折相同', tuple(int(v) for v in FOLDS['FPCR'].loc[FOLDS['FPCR'].R2.idxmin(), ['seed', 'fold']]) == WEAK, str(WEAK))
_wr2 = {m: float(FOLDS[m].set_index(['seed', 'fold']).R2.loc[WEAK]) for m in DONE}
chk('2.2 弱折的四法 R²', f"随机种子{WEAK[0]}的第{WEAK[1] + 1}折中两者的R²为{_wr2['FPLR']:.3f}与{_wr2['FPCR']:.3f}，"
                      f"同折SVR与PLSR为{_wr2['SVR']:.3f}与{_wr2['PLSR']:.3f}")
_PRED: dict[str, pd.DataFrame] = {}
for _m in DONE:
    _pp45 = _read_csv(f'{OUT}/45rebuttal_ncv_{STEM[_m]}_pred.csv', f'3.1 {_m} 逐样品预测')
    if _pp45 is not None:
        _PRED[_m] = _pp45
# 弱折放大的是函数型模型与离散方法之间各对差值的方差；两种函数型模型在该折一同偏低，二者之差的方差不受影响
if len(DONE) == 6:
    _r2f = {m: FOLDS[m].set_index(['seed', 'fold']).R2 for m in ALL6}

    def _var_ratio(a: str, b: str) -> float:
        """差值的方差：25 折的与去掉弱折后 24 折的之比。"""
        dd = _r2f[a] - _r2f[b]
        return float(dd.var(ddof=1) / dd.drop(index=WEAK).var(ddof=1))

    _fd = [(a, b) for a in ('FPLR', 'FPCR') for b in ('SVR', 'PLSR', 'RF', '1D-CNN')]
    _vr = {f'{a}−{b}': _var_ratio(a, b) for a, b in _fd}
    chk('2.2 弱折放大方差的对数', f"这一折放大了函数型模型与离散方法之间{len(_fd)}对差值的方差")
    cond('2.2 弱折使函数型模型与离散方法之间各对差值的方差增大', min(_vr.values()) > 1.5, '；'.join(f'{k} {v:.2f}' for k, v in _vr.items()))
    cond('2.2 弱折未放大 FPLR−FPCR 差值的方差', _var_ratio('FPLR', 'FPCR') < 1.05, f"{_var_ratio('FPLR', 'FPCR'):.2f}")
# 1.6 节：外层训练折的重叠比例，由逐样品预测文件里各折的测试样品算
if 'SVR' in _PRED:
    _te16 = {k: set(g.idx) for k, g in _PRED['SVR'].groupby(['seed', 'fold'])}
    _all16 = set(_PRED['SVR'].idx)
    _tr16 = {k: _all16 - v for k, v in _te16.items()}
    _same16 = [len(_tr16[a] & _tr16[b]) / len(_tr16[a]) for a in _tr16 for b in _tr16 if a < b and a[0] == b[0]]
    _cross16 = [len(_tr16[a] & _tr16[b]) / len(_tr16[a]) for a in _tr16 for b in _tr16 if a < b and a[0] != b[0]]
    chk('1.6 外层训练折的重叠', "外层训练折在同一次重复内两两重叠3/4，不同重复之间约4/5")
    cond('1.6 同一次重复内训练折重叠 3/4', all(abs(v - 0.75) < 0.001 for v in _same16),
         f'{min(_same16):.4f}~{max(_same16):.4f}（{len(_same16)} 对）')
    cond('1.6 不同重复之间训练折重叠约 4/5', abs(float(np.mean(_cross16)) - 0.8) < 0.005 and min(_cross16) > 0.78 and max(_cross16) < 0.83,
         f'均值 {np.mean(_cross16):.3f}，{min(_cross16):.3f}~{max(_cross16):.3f}（{len(_cross16)} 对）')
if all(m in _PRED for m in ('FPLR', 'FPCR', 'PLSR')):
    _wf = {m: _PRED[m][(_PRED[m].seed == WEAK[0]) & (_PRED[m].fold == WEAK[1])].assign(e=lambda q: q.pred - q.y) for m in ('FPLR', 'FPCR', 'PLSR')}
    _top2 = {m: _wf[m].reindex(_wf[m].e.abs().sort_values(ascending=False).index).head(2) for m in ('FPLR', 'FPCR')}
    HOT2 = sorted(int(i) for i in _top2['FPLR'].idx)
    cond('2.2 两种函数型模型误差最大的 2 个测试样品相同', sorted(int(i) for i in _top2['FPCR'].idx) == HOT2, str(HOT2))
    _share = {m: float((_top2[m].e ** 2).sum() / (_wf[m].e ** 2).sum()) for m in ('FPLR', 'FPCR')}
    chk('A8 弱折 2 个样品的平方误差份额', f"误差最大的2个测试样品占该折平方误差总和的{_share['FPLR']:.0%}与{_share['FPCR']:.0%}")
    _e_fplr = [float(_wf['FPLR'].set_index('idx').e.loc[i]) for i in HOT2]
    _e_plsr = [float(_wf['PLSR'].set_index('idx').e.loc[i]) for i in HOT2]
    chk('2.2 弱折 2 条光谱的预测误差', f"FPLR对二者的预测分别偏高{_e_fplr[0]:.1f}与{_e_fplr[1]:.1f}°Brix，PLSR为{f(_e_plsr[0], 1)}与{f(_e_plsr[1], 1)}°Brix")
    cond('2.2 FPLR 对二者的预测均偏高', min(_e_fplr) > 0, str([round(v, 2) for v in _e_fplr]))

    def _excl_stats(m: str) -> tuple[float, float]:
        """评价时除去这 2 条光谱：含它们的测试折按其余样品重算 R²，25 折的均值与标准差。"""
        kept = _PRED[m][~_PRED[m].idx.isin(HOT2)]
        r = np.array([r2_score(g.y, g.pred) for _, g in kept.groupby(['seed', 'fold'])])
        assert len(r) == 25, (m, len(r))
        return float(r.mean()), float(r.std(ddof=1))

    _EXCL = {m: _excl_stats(m) for m in _PRED}
    _xs = [f"{_EXCL[m][1]:.3f}" for m in ('FPLR', 'FPCR')]
    chk('2.2 除去 2 条光谱后的 25 折标准差', f"评价时除去它们，FPLR与FPCR的25折标准差降至{_xs[0] if _xs[0] == _xs[1] else '与'.join(_xs)}（事后描述，附录A8）")
    cond('2.2 除去 2 条光谱后函数型模型的 25 折标准差降幅过四成（折间波动主要来自这 2 条光谱）',
         all(_EXCL[m][1] < 0.6 * _sd[m] for m in ('FPLR', 'FPCR')), '、'.join(f"{m} {_sd[m]:.3f}→{_EXCL[m][1]:.3f}" for m in ('FPLR', 'FPCR')))
    _hot_rows = {m: _PRED[m][_PRED[m].idx.isin(HOT2)] for m in _PRED}
    cond('A8 这 2 条光谱在 5 次重复中各落入测试折 1 次', all(sorted(int(v) for v in q.groupby('idx').size()) == [5, 5] and q.seed.nunique() == 5
                                                  for q in _hot_rows.values()), '各方法逐条 5 次')
    _hot_err = {m: float((q.pred - q.y).abs().max()) for m, q in _hot_rows.items()}
    if len(_PRED) == len(ALL6):
        chk('A8 除去 2 条光谱后六法的 R²', '评价时除去它们，六种方法25折的R²均值与标准差依次为' + '、'.join(
            f"{m}{_EXCL[m][0]:.3f}±{_EXCL[m][1]:.3f}" for m in sorted(ALL6, key=lambda q: -FOLDS[q].R2.mean())))
        chk('A8 2 条光谱在各方法下的最大绝对误差', f"FPLR、FPCR的绝对误差最大为{_hot_err['FPLR']:.1f}与{_hot_err['FPCR']:.1f}°Brix，"
                                          f"SVR、1D-CNN、RF与PLSR分别为{_hot_err['SVR']:.1f}、{_hot_err['1D-CNN']:.1f}、{_hot_err['RF']:.1f}、{_hot_err['PLSR']:.1f}°Brix")
        cond('A8 2 条光谱的最大绝对误差以函数型模型为大', min(_hot_err['FPLR'], _hot_err['FPCR']) > max(_hot_err[m] for m in ('SVR', '1D-CNN', 'RF', 'PLSR')),
             '、'.join(f'{m} {v:.2f}' for m, v in _hot_err.items()))
else:
    HOT2 = []

# ==================== 第三轮（稿中称外移网格）：从逐折、预测与缓存低层产物复算 ====================
_r3_svr = _read_csv('04outputs/45shift/04outputs/45rebuttal_ncv_none_SVR.csv', '第三轮 SVR')
_r3_cnn = _read_csv('04outputs/45shift/04outputs/45rebuttal_ncv_none_1DCNN.csv', '第三轮 1D-CNN')
_r3p = _read_csv('04outputs/45shift/04outputs/45rebuttal_ncv_none_1DCNN_pred.csv', '第三轮 1D-CNN 预测')
if _r3_svr is None or _r3_cnn is None or _r3p is None:
    _r3 = {}
else:
    _r3 = {'SVR': _r3_svr, '1D-CNN': _r3_cnn}
assert _r3p is not None
for _m in ('SVR', '1D-CNN'):
    _d = _r3[_m]
    chk(f'表2 {_m} 外移网格一行', f'{_m}（外移网格）离散{_d.R2.mean():.3f}±{_d.R2.std(ddof=1):.3f}{_d.RMSE.mean():.3f}±{_d.RMSE.std(ddof=1):.3f}'
                         f'{_d.RPD.mean():.2f}±{_d.RPD.std(ddof=1):.2f}')
_const = []
for (_seed, _fold), _g in _r3p.groupby(['seed', 'fold']):
    if float(_g.pred.std(ddof=0)) < 1e-6:
        _const.append((int(_seed), int(_fold)))
_r3n = _r3['1D-CNN'][~_r3['1D-CNN'].apply(lambda x: (int(x.seed), int(x.fold)) in set(_const), axis=1)]
cond('外移网格 1D-CNN 常数预测折数', len(_const) >= 0, f'{len(_const)} 折（预测标准差<1e-6）')
_r3_pick: dict[str, list[Any]] = {m: list(_r3[m].picked.map(ast.literal_eval)) for m in ('SVR', '1D-CNN')}
_svr3_specs = {'C': (0, [100, 1000, 10000]), 'gamma': (1, [0.1, 1, 'scale', 10])}
_svr3_end = []
for _pos, _values in _svr3_specs.values():
    _vals = [cast(tuple[Any, ...], _cfg)[_pos] for _cfg in _r3_pick['SVR']]
    _grid_values = cast(list[Any], _values)
    _svr3_end.extend([sum(v == _grid_values[0] for v in _vals), sum(v == _grid_values[-1] for v in _vals)])
_svr3_end_max = max(_svr3_end)
_cnn_specs = {'w': [16, 32, 64, 128], 'lr': [3e-4, 1e-3, 3e-3]}
_cnn_end = []
for _dim, _values in _cnn_specs.items():
    _vals = [cast(dict[str, Any], _cfg)[_dim] for _cfg in _r3_pick['1D-CNN']]
    _grid_values = cast(list[Any], _values)
    _cnn_end.extend([sum(v == _grid_values[0] for v in _vals), sum(v == _grid_values[-1] for v in _vals)])
_third_free_max = max(_cnn_end + _svr3_end)
_r3cnn_index = _r3['1D-CNN'].set_index(['seed', 'fold'])
def _f_signed(v):
    return f'{v:.4f}'.replace('-', MINUS)
chk('附录A6 外移网格 1D-CNN 常数折', f"外移网格下有{len(_const)}个外层折退化为常数预测（" + '、'.join(f'随机种子{s}第{k + 1}折：R²={_f_signed(float(_r3cnn_index.loc[(s, k), "R2"]))}' for s, k in _const) + '）')
chk('附录A6 外移网格 1D-CNN 其余折', f'其余{len(_r3n)}折为{_r3n.R2.mean():.3f}±{_r3n.R2.std(ddof=1):.3f}')
chk('表2 注 主网格与外移网格', f'未注明者为主网格（主分析）；外移网格仅用于检查网格端点（附录A6），其中1D-CNN有{len(_const)}折为常数预测。')
chk('附录A6 外移网格两法的 R² 均值', f"外移网格下SVR与1D-CNN的R²均值为{_r3['SVR'].R2.mean():.3f}与{_r3['1D-CNN'].R2.mean():.3f}")
_r3all = {**FOLDS, **_r3}
_f1 = [paired(_r3['SVR'], FOLDS['SVR']), paired(_r3['1D-CNN'], FOLDS['1D-CNN'])]
_f1h = holm([x['p'] for x in _f1])
for _name, _v, _h in zip(('SVR', '1D-CNN'), _f1, _f1h, strict=True):
    _f1_sentence = (
        f"SVR与主网格之差{f(_v['d'])}（{rg(_v['lo'], _v['hi'])}）"
        if _name == 'SVR' else
        f"1D-CNN与主网格之差{f(_v['d'])}（{rg(_v['lo'], _v['hi'])}），均未检出差异"
    )
    chk(f'F1 {_name} 数值', _f1_sentence)
    cond(f'F1 {_name} 未检出且非正态补充检验', _h >= 0.05 and _v['sw'] < 0.05 and _v['wx'] >= 0.05, 'Holm p≥0.05、Shapiro p<0.05、Wilcoxon p≥0.05')

# ============================ 1.1 与附录 A9：清洗清单及清洗前低层复算 ============================
_clean_path = Path(f'{OUT}/59data_clean_list.csv')
if _clean_path.exists():
    _clean = pd.read_csv(_clean_path)
    _N_KEPT = int(_clean.keep.sum())                      # 清洗后样品数
    chk('1.1 清洗前样品数', f"原始数据共{len(_clean):,}".replace(',', ' ') + '条记录')
    chk('1.1 清洗后样品数', f"共剔除{int((~_clean.keep).sum())}个，保留{int(_clean.keep.sum()):,}".replace(',', ' ') + '个')
    chk('A9 剔除样品数', f"剔除{int((~_clean.keep).sum())}个")
    _r2_ssc = _clean.loc[_clean.R2, 'SSC']
    _ssc_other = _clean.loc[~_clean.R2, 'SSC']
    chk('1.1 规则①命中数', f"欧氏距离高于90%分位数者（{int(_clean.R1.sum())}个）")
    _BRIX_RANGE_MAX = 45.0                                # 折光仪量程上限（ATAGO PR-101 系列为 0.0~45.0%，厂商规格，不出自本项目数据）
    chk('1.1 规则②命中数与其余记录的范围', f"20°Brix者（{int(_clean.R2.sum())}个，记录值{float(_r2_ssc.iloc[0]):.2f}°Brix超出折光仪0~{_BRIX_RANGE_MAX:.0f}°Brix的量程，"
        f"其余为{math.floor(float(_ssc_other.min()) * 10 + 1e-9) / 10:.1f}~{math.ceil(float(_ssc_other.max()) * 10 - 1e-9) / 10:.1f}°Brix）")
    cond('1.1 规则②只命中 1 条且其余记录均低于 20°Brix', len(_r2_ssc) == 1 and float(_r2_ssc.iloc[0]) > 20 and float(_ssc_other.max()) < 20,
         f'{float(_r2_ssc.iloc[0]):.2f}；其余最大 {float(_ssc_other.max()):.2f}')
    cond('1.1 规则②命中的记录值超出折光仪量程', float(_r2_ssc.iloc[0]) > _BRIX_RANGE_MAX, f'{float(_r2_ssc.iloc[0]):.2f} > {_BRIX_RANGE_MAX:.0f}')
    chk('1.1 规则③命中数', f"整组剔除（{int(_clean.loc[_clean.R3, 'dup_group'].nunique())}组{int(_clean.R3.sum())}个）")
    # 稿中写「逐点相同而 SSC 不同」；data_clean.py 按光谱逐点相同成组、不看 SSC，故另核每组内 SSC 互不相同
    _dup_nun = _clean[_clean.R3].groupby(['year', 'dup_group']).SSC.agg(['nunique', 'size'])
    cond('1.1 规则③各组内 SSC 互不相同', bool((_dup_nun['nunique'] == _dup_nun['size']).all()),
         f"{int((_dup_nun['nunique'] == _dup_nun['size']).sum())}/{len(_dup_nun)} 组")
    chk('1.1 规则④命中数', f"65535者（{int(_clean.R4.sum())}个，均已被光谱离群规则剔除）")
    cond('1.1 规则④命中者均被规则①剔除', bool((_clean.R1 | ~_clean.R4).all()), f'{int((_clean.R4 & _clean.R1).sum())}/{int(_clean.R4.sum())}')
    chk('A9 年份构成', f"2018、2019年各{int((_clean.year == 2018).sum())}、{int((_clean.year == 2019).sum())}条")
    _kept = _clean[_clean.keep]
    _year_n = _kept.groupby('year').size()
    _origin_n = _kept.groupby('origin').size()
    _all = _kept.SSC
    chk('1.1 保留样品的 SSC', f"保留{int(_kept.shape[0]):,}".replace(',', ' ') + f"个，其SSC为({_all.mean():.2f}±{_all.std(ddof=1):.2f})°Brix（各批次的统计结果见表1）")
    _bmeans = [float(_kept[_kept.year == y].SSC.mean()) for y in (2018, 2019)] + [float(_kept[_kept.origin == o].SSC.mean()) for o in (1, 2, 3)]
    chk('A4 各批次构成', f"清洗后2018、2019年各{int(_year_n.loc[2018])}、{int(_year_n.loc[2019])}个样品，"
        f"新疆、山东、陕西各{int(_origin_n.loc[1])}、{int(_origin_n.loc[2])}、{int(_origin_n.loc[3])}个，"
        f"各年份与各产地的SSC均值为{min(_bmeans):.2f}~{max(_bmeans):.2f}°Brix")
    _bt46 = pd.read_csv(f'{OUT}/46rebuttal_recal_batches.csv').set_index('name')
    cond('A4 产地编码 1、2、3 依次为新疆、山东、陕西（与 46 的批次表样品数一致）',
         [int(_origin_n.loc[o]) for o in (1, 2, 3)] == [int(_bt46.loc[k, 'n']) for k in ('新疆', '山东', '陕西')], '样品数逐项相同')
    cond('1.5 最大的 k=40 不到每个目标批次的 1/8', 40 * 8 < min(int(_year_n.min()), int(_origin_n.min())),
         f'最小目标批次 {min(int(_year_n.min()), int(_origin_n.min()))} 个')
    # 清洗前只按 5<SSC<25°Brix 筛去 1 条，即规则②的记录（附录 A9 第一句）
    _pre_drop = _clean[~_clean.old_ok]
    chk('A9 清洗前的筛除', f"清洗前仅筛去SSC不在5~25°Brix之内的{len(_pre_drop)}条（即SSC异常规则剔除的记录），全部{int(_clean.old_ok.sum()):,}".replace(',', ' ')
        + "个样品上的结果仅作对照")
    cond('A9 清洗前筛去的即规则②的记录', len(_pre_drop) == 1 and bool(_pre_drop.R2.all()) and not 5 < float(_pre_drop.SSC.iloc[0]) < 25,
         f'{len(_pre_drop)} 条，SSC {float(_pre_drop.SSC.iloc[0]):.2f}')
    # 1.4、1.6：5 折与 10 折时的测试、训练样品数，由保留样品数算出并与 45 的逐样品预测核对
    _k5 = sorted({_N_KEPT // 5, -(-_N_KEPT // 5)})
    _k10 = sorted({_N_KEPT // 10, -(-_N_KEPT // 10)})
    _nk = f"{_N_KEPT:,}".replace(',', ' ')
    if 'SVR' in _PRED:
        cond('1.4 外层折大小与 45 的逐样品预测一致', sorted({int(v) for v in _PRED['SVR'].groupby(['seed', 'fold']).size()}) == _k5, str(_k5))
    chk('1.4 外层折大小', f"外层将{_nk}个样品随机分为5折（每折{_k5[0]}或{_k5[-1]}个）")
    chk('1.4 5 折与 10 折的训练、测试样品数', f"10折仅将训练集由{_N_KEPT - _k5[-1]}~{_N_KEPT - _k5[0]}个增至{_N_KEPT - _k10[-1]}~{_N_KEPT - _k10[0]}个，"
                                      f"却使测试折减半为{_k10[0]}~{_k10[-1]}个")
    chk('1.6 各折的测试与训练样品数', f"各折为{_k5[-1]}/{_N_KEPT - _k5[-1]}与{_k5[0]}/{_N_KEPT - _k5[0]}")
    chk('摘要 保留样品数', f"剔除离群光谱与错误记录后保留{_nk}个样品")
    chk('Abstract 保留样品数', f"and {_nk} samples remained after outlying spectra and faulty records were removed")
    chk('2.3 第 1 段 全部样品数', f"上一小节用全部{_nk}个样品")
    chk('表A7 注 样品数', f"清洗前为全部{int(_clean.old_ok.sum()):,}".replace(',', ' ') + f"个样品，清洗后为{_nk}个")
    if HOT2:
        _h2 = _clean[_clean.keep & _clean.new_idx.isin(HOT2)]
        cond('2.2 弱折 2 条光谱均为 2018 年新疆批次', len(_h2) == 2 and bool((_h2.year == 2018).all() and (_h2.origin == 1).all()),
             str(_h2[['new_idx', 'year', 'origin']].values.tolist()))
        # 规则①：2018 年的阈值为该年全部记录欧氏距离的 90% 分位数；先核对清单的命中与距离一致
        _d18 = _clean.loc[_clean.year == 2018, 'R1_dist']
        _thr18 = float(np.percentile(_d18, 90))
        cond('A9 2018 年规则①的命中与距离阈值一致', bool((_clean[_clean.year == 2018].R1 == (_d18 > _thr18)).all()), f'阈值 {_thr18:.2f}')


def _git_csv(commit: str, path: str) -> pd.DataFrame | None:
    try:
        raw = subprocess.check_output(['git', 'show', f'{commit}:{path}'], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        PENDING.append((f'A9 清洗前 {path}', f'git show {commit}:{path}'))
        return None
    from io import BytesIO
    return pd.read_csv(BytesIO(raw))


_pre_stems = {m: ('none_1DCNN' if m == '1D-CNN' else (f'none_{m}' if m not in ('FPLR', 'FPCR') else f'bspline_{m}')) for m in ALL6}
_pre_folds: dict[str, pd.DataFrame] = {}
for _m, _stem in _pre_stems.items():
    _pre = _git_csv('2331884', f'04outputs/45rebuttal_ncv_{_stem}.csv')
    if _pre is not None:
        _pre_folds[_m] = _pre
_A7_ORDER = ['SVR', 'FPLR', 'FPCR', 'PLSR', '1D-CNN', 'RF']        # 表 A7 与表 1 的方法次序、配对的前后
if len(_pre_folds) == len(ALL6) and len(DONE) == len(ALL6):
    _t7 = [f"{float(_pre_folds[m].R2.mean()):.3f}" for m in ALL6], [f"{float(FOLDS[m].R2.mean()):.3f}" for m in ALL6]
    for _m, _a, _b in zip(ALL6, *_t7, strict=True):
        chk(f'表A7 R² 均值 {_m}', f"{_m}{_a}{_b}")
    _up7 = [float(FOLDS[m].R2.mean() - _pre_folds[m].R2.mean()) for m in ALL6]
    _up7_tab = [float(b) - float(a) for a, b in zip(*_t7, strict=True)]
    chk('A9 清洗后 R² 均值的升幅', f"清洗后各方法的R²均值升高{rg(min(_up7), max(_up7), 2)}")
    cond('A9 升幅按原始值与按表 A7 所列三位小数相减，写到两位小数相同',
         rg(min(_up7), max(_up7), 2) == rg(min(_up7_tab), max(_up7_tab), 2), f'{min(_up7):.4f}~{max(_up7):.4f}；{min(_up7_tab):.3f}~{max(_up7_tab):.3f}')
    cond('A9 各方法的 R² 均值均升高', min(_up7) > 0, f'{min(_up7):.3f}')
    _pre_pairs = {(a, b): paired(_pre_folds[a], _pre_folds[b]) for a, b in itertools.combinations(_A7_ORDER, 2)}
    for _v, _h in zip(_pre_pairs.values(), holm([v['p'] for v in _pre_pairs.values()]), strict=True):
        _v['holm'] = _h
    _flips = []
    for (_a, _b), _v in _pre_pairs.items():
        _post = pr(_a, _b)
        _vp = '差异显著' if _v['holm'] < 0.05 else '未检出差异'
        _vq = '差异显著' if _post['holm'] < 0.05 else '未检出差异'
        chk(f'表A7 {_a} − {_b}', f"{_a}−{_b}{f(_v['d'])}（{_vp}）{f(_post['d'])}（{_vq}）")
        if _vp != _vq:
            _flips.append(f"{_a}−{_b}（{_vp}→{_vq}）")
    chk('A9 判定改变的各对', f"15对中{len(_flips)}对的判定改变：" + '、'.join(_flips))
else:
    PENDING.append(('表 A7 与 A9 清洗前后的表 1 与 15 对', '45 的六法逐折（清洗前取提交 2331884）'))

_pre_small = _git_csv('2331884', '04outputs/21smallsample_repeats.csv')
if _pre_small is not None:
    _pre_small['method'] = _pre_small.method.replace({'PFLR': 'FPLR'})
    _pre_neg = _pre_small[_pre_small.N.isin([50, 80, 120])].groupby(['N', 'method', 'strategy']).R2.apply(lambda x: int((x < 0).sum()))
    _post_small = pd.read_csv(f'{OUT}/21smallsample_repeats.csv')
    _post_small['method'] = _post_small.method.replace({'PFLR': 'FPLR'})
    _post_neg = _post_small[_post_small.N.isin([50, 80, 120])].groupby(['N', 'method', 'strategy']).R2.apply(lambda x: int((x < 0).sum()))
    for _n in (50, 80, 120):
        for _m in ('FPCR', 'FPLR', 'SVR'):
            chk(f'表A7 小样本 {_m} N={_n}', f"{_m}，N={_n}{int(_pre_neg.loc[(_n, _m, 'cv')])}{int(_post_neg.loc[(_n, _m, 'cv')])}")
    _pre_fun = [int(_pre_neg.loc[(n, m, 'cv')]) for n in (50, 80, 120) for m in ('FPCR', 'FPLR')]
    _post_fun = [int(_post_neg.loc[(n, m, 'cv')]) for n in (50, 80, 120) for m in ('FPCR', 'FPLR')]
    chk('A9 小样本失效次数的变化', f"交叉验证选参的函数型模型在N=50~120的R²<0次数由{min(_pre_fun)}~{max(_pre_fun)}次降为{min(_post_fun)}~{max(_post_fun)}次")
    chk('2.3 清洗前的小样本失效次数', f"清洗前（附录A9），N=50~120时交叉验证选参的FPCR与FPLR在20次抽样中有{min(_pre_fun)}~{max(_pre_fun)}次R²<0")
    chk('A9 清洗后只剩的失效次数', f"在20次抽样中仅剩{sum(_post_fun)}次R²<0")
    # 清洗前的逐样品剖析做在另取的两批抽样上（随机种子 20~219），20 次抽样（随机种子 0~19）的失效没有逐样品剖析
    chk('2.3 清洗前逐样品剖析所在的抽样', "按附录A8的方案另取的抽样中，失效集中在6条异常光谱上")
    _pre_an = _git_csv('2331884', '04outputs/57rebuttal_guard_anatomy.csv')
    if _pre_an is not None:
        cond('2.3 清洗前的逐样品剖析只含另取的两批抽样', int(_pre_an.seed.min()) == 20 and int(_pre_an.seed.max()) == 219,
             f'随机种子 {int(_pre_an.seed.min())}~{int(_pre_an.seed.max())}，{len(_pre_an)} 行')
    # 摘要的选参建议限于训练样本不少于 50 个：N≥50 时两种函数型模型交叉验证选参的中位数都高于固定超参数，N=30 时 FPCR 相反
    _med = _post_small.groupby(['N', 'method', 'strategy']).R2.median()
    chk('摘要 选参建议的样本量范围', "训练样本不少于50个时宜以交叉验证为函数型模型选参")
    cond('摘要 N≥50 时函数型模型交叉验证选参的中位数均高于固定超参数',
         all(_med.loc[(n, m, 'cv')] > _med.loc[(n, m, 'fixed')] for n in sorted(set(_post_small.N)) if n >= 50 for m in ('FPCR', 'FPLR')),
         f"N≥50 的样本量 {[int(n) for n in sorted(set(_post_small.N)) if n >= 50]}")
    cond('摘要 N=30 时 FPCR 交叉验证选参的中位数低于固定超参数', bool(_med.loc[(30, 'FPCR', 'cv')] < _med.loc[(30, 'FPCR', 'fixed')]),
         f"{_med.loc[(30, 'FPCR', 'cv')]:.3f} 对 {_med.loc[(30, 'FPCR', 'fixed')]:.3f}")

_pre_recal = _git_csv('2331884', '04outputs/46rebuttal_recal_draws.csv')
if _pre_recal is not None:
    _pre_1819 = _pre_recal[(_pre_recal.setting == '2018→2019') & (_pre_recal.strategy == 'S1') & (_pre_recal.k == 10)]
    if not _pre_1819.empty:
        _post_rc = pd.read_csv(f'{OUT}/46rebuttal_recal_draws.csv')
        _post_1819 = _post_rc[(_post_rc.setting == '2018→2019') & (_post_rc.strategy == 'S1') & (_post_rc.k == 10)]
        for _m in ('FPCR', 'FPLR', 'PLSR', 'SVR'):
            chk(f'表A7 截距校正 {_m}', f"{_m}{float(_pre_1819[_pre_1819.method == _m].R2.median()):.3f}{float(_post_1819[_post_1819.method == _m].R2.median()):.3f}")

# 后续模块仍有少量逐折/逐点输入直接读取；统一在这里做一次门禁，避免某个服务器结果
# 未到位时在模块中途抛 FileNotFoundError，也避免把过期旧文件当作清洗后结果。
_later_inputs = [
    '04outputs/47rebuttal_boundary.csv', '04outputs/55rebuttal_A5_pairs.csv',
    '04outputs/55rebuttal_A5_folds.csv', '04outputs/55rebuttal_A5_diag.csv',
    '04outputs/44fully_nested.csv', '04outputs/44fully_nested_rf.csv',
    '04outputs/44fully_nested_cnn.csv', '04outputs/36prep_leakage.csv',
    '04outputs/26basis_nested.csv', '04outputs/51rebuttal_lams_folds.csv', '04outputs/51rebuttal_lams_summary.csv',
    '04outputs/27prep_nested.csv', '04outputs/46rebuttal_recal_batches.csv',
    '04outputs/18svr_aligned.csv', '04outputs/56rebuttal_selection_draws.csv',
    '04outputs/57rebuttal_guard_draws.csv', '04outputs/58rebuttal_guard_posthoc_draws.csv',
    '04outputs/57rebuttal_guard_anatomy.csv', '04outputs/57rebuttal_guard_spectra.csv',
    '04outputs/57rebuttal_guard_dataset.csv', '04outputs/46rebuttal_recal_draws.csv',
    '04outputs/46rebuttal_recal_source.csv', '04outputs/49rebuttal_recal_range.csv',
    '04outputs/48rebuttal_beta.csv', *_REQUIRED_54, *_REQUIRED_61,
]
for _input in _later_inputs:
    _fresh(_input)

# 后续第三轮替换六法与附录 A1/A3/A5 的比较要求六个清洗后主分析方法及服务器产物齐备。
# 条件不满足时跳过该块，后面的 1.1、2.3~2.5 与 A2/A4/A7/A8 继续独立复算。
_full_server = len(DONE) == len(ALL6) and all(Path(path).exists() and Path(path).stat().st_mtime >= _CLEAN_MTIME - 2 for path in (
    '04outputs/47rebuttal_boundary.csv', '04outputs/55rebuttal_A5_pairs.csv',
    '04outputs/55rebuttal_A5_folds.csv', '04outputs/55rebuttal_A5_diag.csv'))
# 附录 A3 的三层全嵌套复核另需服务器上 44 的三份产物；未到位时只跳过这一小块
_fn_ready = all(Path(path).exists() and Path(path).stat().st_mtime >= _CLEAN_MTIME - 2 for path in (
    '04outputs/44fully_nested.csv', '04outputs/44fully_nested_rf.csv', '04outputs/44fully_nested_cnn.csv'))
if _full_server:
    _sub = dict(FOLDS)
    _sub.update(_r3)
    _f2pairs = list(itertools.combinations(ALL6, 2))
    _f2 = [paired(_sub[a], _sub[b]) for a, b in _f2pairs]
    _f2h = holm([x['p'] for x in _f2])
    _main_h = {pair: PAIRS[pair]['holm'] if pair in PAIRS else pr(*pair)['holm'] for pair in _f2pairs}
    for (_name, _v, _h) in zip(('SVR：外移网格 − 主网格', '1D-CNN：外移网格 − 主网格'), _f1, _f1h, strict=True):
        _holm_text = p3(_h)
        _decision_text = '未检出差异' if _h >= 0.05 else '差异显著'
        _wilcoxon_text = f'（Wilcoxon p{pw(_v["wx"])}）' if _v['sw'] < 0.05 else ''
        chk(f'表A4 {_name} 数值', ''.join((_name, f(_v['d']), rg(_v['lo'], _v['hi']), _holm_text, _decision_text + _wilcoxon_text, '－')))
    _changed = []
    for (_pair, _v, _h) in zip(_f2pairs, _f2, _f2h, strict=True):
        _decision = _h < 0.05
        _main_decision = _main_h[_pair] < 0.05
        if _decision != _main_decision:
            _changed.append(' − '.join(_pair))
        _holm_text = p3(_h)
        _decision_text = '差异显著' if _decision else '未检出差异'
        _main_text = '差异显著' if _main_decision else '未检出差异'
        _wilcoxon = f'（Wilcoxon p{pw(_v["wx"])}）' if _v['sw'] < 0.05 else ''
        _pair_text = '1D-CNN − RF' if _pair == ('RF', '1D-CNN') else ' − '.join(_pair)
        if _pair == ('RF', '1D-CNN'):
            _table_v = dict(_v, d=-_v['d'], lo=-_v['hi'], hi=-_v['lo'])
        else:
            _table_v = _v
        chk(f'表A4 {_pair_text} 数值', ''.join((_pair_text, f(_table_v['d']), rg(_table_v['lo'], _table_v['hi']), _holm_text, _decision_text + _wilcoxon, _main_text)))
        cond(f'F2 {" − ".join(_pair)} 数值', np.isfinite(_v['d']) and np.isfinite(_v['lo']) and np.isfinite(_v['hi']),
             f'{" − ".join(_pair)} {f(_v["d"])} [{rg(_v["lo"], _v["hi"])}]')
    _changed_norm = [p.replace('RF − 1D-CNN', '1D-CNN − RF') for p in _changed]
    cond('F2 判定改变的各对均由差异显著变为未检出差异', all(_main_h[pp] < 0.05 <= h for pp, h in zip(_f2pairs, _f2h, strict=True)
                                               if ' − '.join(pp) in _changed), str(_changed_norm))
    chk('2.2、A6 外移网格判定改变的各对', f"重新进行15对比较，{'、'.join(_changed_norm)}这{ {2: '两'}.get(len(_changed_norm), len(_changed_norm)) }对变为未检出差异，其余各对的判定不变")
    chk('结论（1）外移网格判定改变的各对', f"差异显著（{'、'.join(_changed_norm)}在外移网格下未检出差异）")
    chk('2.2 其余各对未检出差异', f"差异显著，其余{len(PAIRS) - sum(v['holm'] < 0.05 for v in PAIRS.values())}对未检出差异")
    chk('A6 外移网格的规则（RF 树数一维的例外）', 'RF树数一维按事后例外未平移，附录A1')
    chk('A6 外移网格的规则（1D-CNN 层数）', '将该维向外平移（1D-CNN层数已达结构上限，固定为5；')
    # 外移网格登记于清洗之前：当时主网格的端点次数取自提交 2331884 的 47rebuttal_boundary.csv
    _pre_bd = _git_csv('2331884', '04outputs/47rebuttal_boundary.csv')
    if _pre_bd is not None:
        _pb = _pre_bd[_pre_bd.analysis == '主分析'].copy()
        _pb['end'] = _pb[['at_min', 'at_max']].max(axis=1)
        _pb_over = {(r.method, r.dim): int(r.end) for r in _pb.itertuples() if r.end > 12.5}
        _pb_order = [('SVR', 'gamma'), ('1D-CNN', '宽度'), ('1D-CNN', '层数'), ('1D-CNN', '学习率'), ('RF', '树数')]
        cond('A6 清洗前主网格过半落在同一端点的维度同为这 5 个', sorted(_pb_over) == sorted(_pb_order), str(_pb_over))
        chk('A6 外移网格确定于清洗之前', '外移网格在清洗之前已按当时主网格的选中次数确定（同样的5个维度，落在同一端点的次数依次为'
            + '、'.join(str(_pb_over.get(k, 0)) for k in _pb_order) + '次），清洗后按表A1的次数重新判定，所用网格未变')
    cond('A6 外移网格文件自登记后未变（协议 C3 2026-09-15 登记的 SHA-256）',
         hashlib.sha256(Path('02code/grids_boundary_shift.py').read_bytes()).hexdigest()
         == 'a4579b1efa4a60e446bcd66e7606b70b136f5bcf1718bac03366c658775630f9', '02code/grids_boundary_shift.py')
    cond('F2 判定改变的各对均涉及 1D-CNN', all('1D-CNN' in p for p in _changed_norm), str(_changed_norm))

    def _memo_fresh(pattern: str) -> list[str]:
        """同一方法的格点缓存按数据指纹分文件；只取修改时间不早于 data_clean.py 的一套（清洗后数据上算出）。"""
        paths = [q for q in glob.glob(pattern) if Path(q).stat().st_mtime >= _CLEAN_MTIME - 2]
        fps = {re.sub(r'_shard\d+$', '', Path(q).stem).rsplit('_', 1)[-1] for q in paths}
        assert len(fps) == 1, (pattern, sorted(fps))
        return paths

    def _inner_mse(paths: list[str]) -> list[tuple[dict[str, Any], list[float], tuple[int, int]]]:
        out = []
        for _path in paths:
            with open(_path, encoding='utf-8') as _fh:
                for _line in _fh:
                    _rec = json.loads(_line); _key = json.loads(_rec['k'])
                    if _key[0] == 'inner':
                        out.append((_key[3], [float(x) for x in _rec['v']], (int(_key[1]), int(_key[2]))))
        return out

    _in3 = _inner_mse(_memo_fresh('04outputs/45shift/04outputs/45memo_none_1DCNN_*.jsonl'))
    _in2 = _inner_mse(_memo_fresh('04outputs/45memo_none_1DCNN_*.jsonl'))
    _n_fit3 = sum(len(v) for _, v, _ in _in3)
    _deg3 = sum(x > 2 for _, v, _ in _in3 for x in v)
    _deg2 = sum(x > 2 for _, v, _ in _in2 for x in v)
    _by_lr = {lr: sum(x > 2 for c, v, _ in _in3 if c['lr'] == lr for x in v) for lr in (3e-4, 1e-3, 3e-3)}
    _selected_cfgs = {(int(row.seed), int(row.fold), tuple(sorted(ast.literal_eval(row.picked).items()))) for _, row in _r3['1D-CNN'].iterrows()}
    _sel_bad = sum(x > 2 for c, v, sf in _in3 if (sf[0], sf[1], tuple(sorted(c.items()))) in _selected_cfgs for x in v)
    chk('附录A6 内层退化次数', f"{_n_fit3:,}".replace(',', ' ') + f"次内层拟合中{_deg3}次退化")
    chk('附录A6 按学习率的退化次数', f"按学习率3×10{MINUS}4、10{MINUS}3、3×10{MINUS}3依次为{_by_lr[3e-4]}、{_by_lr[1e-3]}、{_by_lr[3e-3]}次")
    chk('附录A6 选中候选内层退化', f'选中候选的内层退化次数为{_sel_bad}')
    chk('附录A6 主网格退化次数', f'主网格按同一判据退化{_deg2}次')
    cond('附录A6 拟合总数为 25 折×12 组候选×5 次内层、按学习率的分解之和等于总数', _n_fit3 == 25 * 12 * 5 and sum(_by_lr.values()) == _deg3,
         f'{_n_fit3} 次；{sum(_by_lr.values())}/{_deg3}')

    # 表 A1 注：非数值取值的实际大小。特征数按 sklearn 的 max_features 取整口径（⌊比例×p⌋，√p 取 ⌊√p⌋）由 p=2048 重算
    _p_wave = 2048
    _nfeat = [int(math.sqrt(_p_wave)), int(0.1 * _p_wave), int(0.3 * _p_wave), _p_wave]
    chk('表A1注 RF 特征比例对应的波长点数', f"依次对应{_nfeat[0]}、{_nfeat[1]}、{_nfeat[2]}、{_nfeat[3]}个波长点（向下取整），√p为该维的最小端")
    _bnd47 = pd.read_csv(f'{OUT}/47rebuttal_boundary.csv')
    _scale_note = str(_bnd47.loc[(_bnd47.analysis == '主分析') & (_bnd47.method == 'SVR') & (_bnd47.dim == 'gamma'), 'value_note'].iloc[0])
    _scale_m = re.search(r'([0-9.]+)~([0-9.]+)', _scale_note)
    assert _scale_m is not None, _scale_note
    chk('表A1注 gamma=scale 的实际取值', f"在各外层训练折上为{float(_scale_m.group(1)):g}~{float(_scale_m.group(2)):g}，是该维的最大端")
    cond('表A1注 scale 大于网格其余取值', float(_scale_m.group(1)) > 1e-3, f'{_scale_m.group(1)} > 0.001')

    # 附录 A2：小样本试验 SVR 的 12 组候选，取自 21 脚本的网格定义
    with open('02code/21smallsample_repeats.py', encoding='utf-8') as _sg_fh:
        _sg_src = _sg_fh.read()
    cond('A2 SVR 小样本候选的网格定义', "SG=[(c,g) for c in [1,10,100,1000] for g in ['scale',1e-3,1e-4]]" in _sg_src, '21 脚本的 SG 定义')
    chk('A2 SVR 小样本候选', f"SVR的C∈{{1,10,100,1000}}×gamma∈{{scale,10{MINUS}3,10{MINUS}4}}共12组")

    # 附录 A5：六法的 GBIDS 与 Pass 次数、15 对配对比较，由 55 的逐折结果按 1.6 节同一检验重算
    _a5f = pd.read_csv(f'{OUT}/55rebuttal_A5_folds.csv')
    _a5f = _a5f[_a5f.basis.isin(['none', 'bspline'])] if 'basis' in _a5f else _a5f
    _g5 = {m: _a5f[_a5f.method == m].rename(columns={'GBIDS': 'R2_'})[['seed', 'fold', 'R2_']].rename(columns={'R2_': 'R2'}) for m in ALL6}
    assert all(len(v) == 25 for v in _g5.values()), {m: len(v) for m, v in _g5.items()}
    _a5_order = sorted(ALL6, key=lambda m: -float(_g5[m].R2.mean()))
    chk('A5 六法 GBIDS 与 Pass', '六种方法的GBIDS均值±标准差与Pass次数依次为' + '、'.join(
        f"{m}{float(_g5[m].R2.mean()):.2f}±{float(_g5[m].R2.std(ddof=1)):.2f}（{int(_a5f[_a5f.method == m].Pass.astype(bool).sum())}/25）" for m in _a5_order))
    _a5_pairs = {(a, b): paired(_g5[a], _g5[b]) for a, b in itertools.combinations(_A7_ORDER, 2)}
    for _v, _h in zip(_a5_pairs.values(), holm([v['p'] for v in _a5_pairs.values()]), strict=True):
        _v['holm'] = _h
    _a5p = pd.read_csv(f'{OUT}/55rebuttal_A5_pairs.csv').set_index('pair')
    cond('A5 重算的 15 对与 55 配对表一致', all(abs(_a5_pairs[(a, b)]['d'] - float(_a5p.loc[f'{a} − {b}', 'diff'])) < 1e-9
                                             and abs(_a5_pairs[(a, b)]['holm'] - float(_a5p.loc[f'{a} − {b}', 'holm_p'])) < 1e-9 for a, b in _a5_pairs), '差值与 Holm p 逐对一致')
    for (_a, _b), _v in _a5_pairs.items():
        _ci = rg(_v['lo'], _v['hi'], 2)
        _wtxt = f"；差值非正态，Wilcoxon符号秩检验p{pw(_v['wx'])}" if _v['sw'] < 0.05 else ''
        chk(f'A5 {_a} − {_b}', f"{_a}−{_b}：{f(_v['d'], 2)}（95%置信区间{_ci}{_wtxt}）" if _v['holm'] < 0.05
            else f"{_a}−{_b}{f(_v['d'], 2)}（{_ci}{_wtxt}）")
    _a5_diff_pairs = [(a, b) for (a, b), v in _a5_pairs.items() if (v['holm'] < 0.05) != (pr(a, b)['holm'] < 0.05)]
    _a5_diff = [f'{a} − {b}' for a, b in _a5_diff_pairs]
    cond('A5 与 R² 判定不同的各对均为 GBIDS 显著而 R² 未检出', all(_a5_pairs[q]['holm'] < 0.05 for q in _a5_diff_pairs), str(_a5_diff))
    _a5_ns = sum(v['holm'] >= 0.05 for v in _a5_pairs.values())
    chk('A5 显著与未检出的对数', f"Holm校正后差异显著；其余{_a5_ns}对未检出差异")
    chk('A5 与 R² 判定不同的各对', f"与R²的判定（2.2节）相比，{'、'.join(_a5_diff)}这{len(_a5_diff)}对在GBIDS上差异显著而在R²上未检出差异，其余{15 - len(_a5_diff)}对两种口径的判定相同")
    chk('附录 A6 引表 A4', '外移网格的比较另成两族（与主网格之差2对、替换后的六种方法比较15对）各作Holm校正。2.2节外移网格比较的逐对结果见表A4')
    # 附录 A5：未检出差异的 4 对中 PLSR−1D-CNN 的未校正区间不含 0，稿件另给它的 Holm 校正后 p
    _a5_pc = _a5_pairs[('PLSR', '1D-CNN')]
    chk('A5 PLSR − 1D-CNN 的 Holm p', f"其中PLSR−1D-CNN的未校正区间不含0，但Holm校正后p为{_a5_pc['holm']:.3f}")
    cond('A5 PLSR − 1D-CNN 未校正区间不含 0 而 Holm 校正后未检出差异', _a5_pc['lo'] > 0 and _a5_pc['holm'] >= 0.05,
         f"区间 {_a5_pc['lo']:.2f}~{_a5_pc['hi']:.2f}，Holm p {_a5_pc['holm']:.3f}")
    # 附录 A9：GBIDS 15 对在清洗前后的判定，清洗前取提交 2331884 的 55 逐折结果按同一检验重算
    _pre_a5f = _git_csv('2331884', '04outputs/55rebuttal_A5_folds.csv')
    if _pre_a5f is not None:
        _pre_a5f = _pre_a5f[_pre_a5f.basis.isin(['none', 'bspline'])] if 'basis' in _pre_a5f else _pre_a5f
        _pg5 = {m: _pre_a5f[_pre_a5f.method == m][['seed', 'fold', 'GBIDS']].rename(columns={'GBIDS': 'R2'}) for m in ALL6}
        assert all(len(v) == 25 for v in _pg5.values()), {m: len(v) for m, v in _pg5.items()}
        _pre_a5_pairs = {(a, b): paired(_pg5[a], _pg5[b]) for a, b in itertools.combinations(_A7_ORDER, 2)}
        for _v, _h in zip(_pre_a5_pairs.values(), holm([v['p'] for v in _pre_a5_pairs.values()]), strict=True):
            _v['holm'] = _h

        def _verdict_a5(h: float) -> str:
            return '差异显著' if h < 0.05 else '未检出差异'

        _a5_flip = [f"{a} − {b}（{_verdict_a5(_pre_a5_pairs[(a, b)]['holm'])}→{_verdict_a5(_a5_pairs[(a, b)]['holm'])}）"
                    for a, b in _a5_pairs if (_pre_a5_pairs[(a, b)]['holm'] < 0.05) != (_a5_pairs[(a, b)]['holm'] < 0.05)]
        chk('A9 GBIDS 判定的变化', ("附录A5的GBIDS15对中，" + '、'.join(_a5_flip) + f"的判定改变，其余{15 - len(_a5_flip)}对不变")
            if _a5_flip else '附录A5的GBIDS15对的判定均不变')
    _a5_rms = np.sqrt(((1 - _a5f[['u_R2', 'u_MAE', 'u_P90', 'u_BIAS']]) ** 2).mean(axis=1))
    cond('A5 GBID 为 4 项效用缺口的均方根、GBIDS=100(1−GBID)', float(np.abs(_a5f.GBID - _a5_rms).max()) < 1e-9 and float(np.abs(_a5f.GBIDS - 100 * (1 - _a5f.GBID)).max()) < 1e-9,
         f'最大绝对差 {float(np.abs(_a5f.GBID - _a5_rms).max()):.1e}')
    _a5d = pd.read_csv(f'{OUT}/55rebuttal_A5_diag.csv')
    cond('A5 四项指标上下截断比例均为 0', bool(((_a5d.lower_clip_ratio == 0) & (_a5d.upper_clip_ratio == 0)).all()), '全为 0')
    chk('A5 截断比例', '四项指标的上、下截断比例均为0')

    _fr = {m: pd.read_csv(f'{OUT}/45rebuttal_ncv_fourier_{m}.csv') for m in ('FPLR', 'FPCR')}
    _bs = {m: paired(FOLDS[m], _fr[m]) for m in ('FPLR', 'FPCR')}
    _bh = holm([_bs['FPLR']['p'], _bs['FPCR']['p']])
    chk('2.2 B 样条基与傅里叶基之差', f"之差为{f(_bs['FPLR']['d'])}（{rg(_bs['FPLR']['lo'], _bs['FPLR']['hi'])}）与"
                                 f"{f(_bs['FPCR']['d'])}（{rg(_bs['FPCR']['lo'], _bs['FPCR']['hi'])}）")
    cond('2.2 基函数两对 Holm 后未检出、差值不拒绝正态（不另报 Wilcoxon）', min(_bh) >= 0.05 and min(v['sw'] for v in _bs.values()) >= 0.05,
         f"Holm p {_bh[0]:.3f}、{_bh[1]:.3f}；Shapiro p {_bs['FPLR']['sw']:.3f}、{_bs['FPCR']['sw']:.3f}")
    chk('2.2 FPCR 基函数之差的 Holm p', f"后者的未校正区间不含0，但Holm校正后p为{f(_bh[1])}")
    cond('2.2 只有 FPCR 一对的未校正区间不含 0', _bs['FPCR']['hi'] < 0 and _bs['FPLR']['lo'] < 0 < _bs['FPLR']['hi'],
         f"FPLR {rg(_bs['FPLR']['lo'], _bs['FPLR']['hi'])}；FPCR {rg(_bs['FPCR']['lo'], _bs['FPCR']['hi'])}")

    # 表 A1：25 个外层折中的选中值与次数（按次数降序、同次数按取值升序）
    def counts(m: str, pos: int, fmt: Callable[[float], str]) -> str:
        """picked 形如 14、(80, 50)、(80, 1e-06)、(1000, 'scale')；取第 pos 个数值分量计数。"""
        vals = [float(re.findall(r'-?\d+(?:\.\d+)?(?:e-?\d+)?', s)[pos]) for s in FOLDS[m].picked.astype(str)]
        tally: dict[float, int] = {}
        for v in vals:
            tally[v] = tally.get(v, 0) + 1
        return '、'.join(f'{fmt(v)}（{n}）' for v, n in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])))


    chk('表A1 PLSR 选中', 'PLSR潜变量数2,4,6,8,10,12,14,16,18,20,22,24' + counts('PLSR', 0, lambda v: f'{int(v)}'))
    chk('表A1 FPCR 选中', counts('FPCR', 1, lambda v: f'{int(v)}'))
    chk('表A1 FPLR 选中', counts('FPLR', 1, exp10))
    chk('表A1 SVR 选中（C）', 'C：' + counts('SVR', 0, exp10))

    def _num(x: Any) -> float:
        return float(x)

    _pk = {m: [ast.literal_eval(str(v)) for v in FOLDS[m].picked] for m in ALL6}
    _p_wave = 2048
    # 各维（取值序列, 网格最小端, 网格最大端）；gamma='scale' 为最大端，RF 特征比例 √p 为最小端（表 A1 注）
    _dims = {
        'PLSR 潜变量数': ([int(v) for v in _pk['PLSR']], 2, 24),
        'SVR C': ([c[0] for c in _pk['SVR']], 10, 10000),
        'SVR gamma': (['scale' if c[1] == 'scale' else _num(c[1]) for c in _pk['SVR']], 1e-4, 'scale'),
        'RF 树数': ([c[0] for c in _pk['RF']], 400, 1600),
        'RF 特征比例': ([c[1] for c in _pk['RF']], 'sqrt', 1.0),
        '1D-CNN 宽度': ([c['w'] for c in _pk['1D-CNN']], 16, 32),
        '1D-CNN 层数': ([c['nl'] for c in _pk['1D-CNN']], 3, 5),
        '1D-CNN 学习率': ([c['lr'] for c in _pk['1D-CNN']], 3e-4, 1e-3),
        'FPCR K': ([c[1] for c in _pk['FPCR']], 5, 79),
        'FPLR λβ': ([_num(c[1]) for c in _pk['FPLR']], 1e-10, 10),
    }

    def _same(v: Any, e: Any) -> bool:
        return (v == e) if isinstance(v, str) or isinstance(e, str) else abs(float(v) - float(e)) <= 1e-12 * max(1.0, abs(float(e)))

    _ends = {k: (sum(_same(v, lo) for v in vals), sum(_same(v, hi) for v in vals)) for k, (vals, lo, hi) in _dims.items()}
    _over50 = {k: max(e) for k, e in _ends.items() if max(e) > 12.5}
    chk('A1 端点次数最大值', f"各维选中网格端点的次数见表A1，最多为{max(max(e) for e in _ends.values())}次")
    _over_order = ['SVR gamma', '1D-CNN 宽度', '1D-CNN 层数', '1D-CNN 学习率', 'RF 树数']
    cond('A1 过半落在同一端点的维度恰为 5 个', sorted(_over50) == sorted(_over_order), str(_over50))
    chk('A1 主网格过半端点', '主网格中过半落在同一端点的维度为' + '、'.join(f'{k.replace("SVR gamma", "SVR的gamma")}（{_over50[k]}次）'
                                                                    for k in _over_order if k in _over50))

    _rf_rows = []
    for _path in _memo_fresh('04outputs/45memo_none_RF_*.jsonl'):
        with open(_path, encoding='utf-8') as _fh:
            for _line in _fh:
                _rec = json.loads(_line); _key = json.loads(_rec['k'])
                if _key[0] == 'inner':
                    _rf_rows.append((_key[1], _key[2], _key[3][0], _key[3][1], float(np.mean(np.sqrt(_rec['v'])))))
    _rf_frame = pd.DataFrame(_rf_rows, columns=['seed', 'fold', 'trees', 'ratio', 'rmse'])
    _rf_platform = []
    for _ratio, _group in _rf_frame.groupby('ratio'):
        _wide = _group.pivot_table(index=['seed', 'fold'], columns='trees', values='rmse', aggfunc='first')
        _rf_platform.append(float((100 * (_wide[1600] / _wide[800] - 1)).median()))
    _pct2 = lambda value: f'{value:.2f}'.replace('-', MINUS)
    chk('A1 RF 树数平台', f'内层均方根误差的相对变化按4档特征比例取中位数为{_pct2(min(_rf_platform))}%~{_pct2(max(_rf_platform))}%')
    # 第三轮自由维：SVR 的 C 与 gamma、1D-CNN 的宽度与学习率在各自第三轮网格上的端点次数
    _free = {'SVR 的C': ([c[0] for c in _r3_pick['SVR']], 100, 10000),
             'SVR 的gamma': ([c[1] for c in _r3_pick['SVR']], 0.1, 10),
             '1D-CNN 的宽度': ([c['w'] for c in _r3_pick['1D-CNN']], 16, 128),
             '1D-CNN 的学习率': ([c['lr'] for c in _r3_pick['1D-CNN']], 3e-4, 3e-3)}
    _free_ends = {k: (sum(_same(v, lo) for v in vals), sum(_same(v, hi) for v in vals)) for k, (vals, lo, hi) in _free.items()}
    chk('A6 外移网格各自由维的端点次数', '；'.join(f"{k}为最小端{a}次、最大端{b}次" for k, (a, b) in _free_ends.items()))
    _free_max = max(max(v) for v in _free_ends.values())
    _free_arg = [k for k, v in _free_ends.items() if max(v) == _free_max]
    cond('A1 外移网格自由维端点次数的最大者唯一', len(_free_arg) == 1, str(_free_arg))
    chk('A1 外移网格自由维端点', f"外移网格中自由维的端点次数最多为{_free_max}次（{_free_arg[0].replace('的', '')}）")

    # 附录 A1：初始网格那次运行（提交 a9641e3：清洗前 1 300 个样品、单个种子 5 个外层折、函数型模型为傅里叶基）
    # 的候选、各折选中值与重定主网格时删去的取值；主网格的范围取自 02code/grids.py
    _ini = _git_csv('a9641e3', '04outputs/25nested_cv.csv')
    try:
        _ini_src = subprocess.check_output(['git', 'show', 'a9641e3:02code/16unified_comparison.py'], stderr=subprocess.DEVNULL).decode('utf-8')
    except subprocess.CalledProcessError:
        _ini_src = ''
        PENDING.append(('A1 初始网格的候选', 'git show a9641e3:02code/16unified_comparison.py'))
    if _ini is not None and _ini_src:
        _ini_defs = {
            'PLSR 2~24 步长 2': 'g=list(range(2,26,2))',
            'SVR': "g=[(C,gm) for C in [1,10,100,1000] for gm in ['scale',1e-3,1e-4]]",
            'RF': "g=[(n,mf) for n in [200,400,800] for mf in ['sqrt',0.1,0.3,1.0]]",
            '1D-CNN': 'for w,nl in [(16,3),(32,3),(16,4)] for lr in [1e-3,3e-4] for dp in [0.2,0.4]]',
            'FPCR': 'g=[(L,K) for L in [40,80] for K in [5,10,15,20,25,30]]',
            'FPLR': 'g=[(L,lam) for L in [40,80] for lam in [1e-8,1e-6,1e-4,1e-2,1e0,1e2]]',
            '傅里叶基': 'B=fourier_basis(t,L); J,Rm=gram_and_pen(B,t)',
        }
        _ini_miss = [k for k, v in _ini_defs.items() if _ini_src.count(v) != 1]
        cond('A1 初始网格的候选与基（当时脚本的定义逐条在）', not _ini_miss, str(_ini_miss))
        chk('A1 初始网格 SVR 与 RF', 'SVR为C∈{1, 10, 102, 103}×gamma∈{scale, 10−3, 10−4}，RF为树数∈{200, 400, 800}×4档特征比例')
        chk('A1 初始网格 1D-CNN', '1D-CNN为(宽度, 层数)∈{(16, 3), (32, 3), (16, 4)}×学习率∈{10−3, 3×10−4}×丢弃率∈{0.2, 0.4}')
        chk('A1 初始网格 函数型模型', 'FPCR为L∈{40, 80}×K∈{5, 10, 15, 20, 25, 30}，FPLR为L∈{40, 80}×λβ∈{10−8, 10−6, 10−4, 10−2, 1, 102}，两种函数型模型均用傅里叶基')
        _ipk = {m: [ast.literal_eval(str(v)) for v in g.sort_values('fold').picked] for m, g in _ini.groupby('method')}
        cond('A1 初始网格那次运行为单个种子、5 个外层折、六种方法', [int(v) for v in _ini.seed.unique()] == [SEED0]
             and sorted(_ipk) == sorted(['PLSR', 'SVR', 'RF', '1D-CNN', 'FPCR', 'PFLR']) and all(len(v) == 5 for v in _ipk.values()),
             str(_ini.groupby('method').size().to_dict()))
        _n_c = sum(c[0] == 1000 for c in _ipk['SVR'])
        _n_k = sum(c[1] == 30 for c in _ipk['FPCR'])
        _n_nl = sum(c['nl'] == 4 for c in _ipk['1D-CNN'])
        _n_tr = sum(c[0] == 800 for c in _ipk['RF'])
        cond('A1 C、K、层数三维的上边界选中折数相同', _n_c == _n_k == _n_nl, f'{_n_c}/{_n_k}/{_n_nl}')
        chk('A1 初始网格上边界的选中折数', f'SVR的C（1000）、FPCR的K（30）与1D-CNN的层数（4）在{_n_c}折中均选中初始网格的上边界，RF的树数（800）在{_n_tr}折中选中上边界')
        _n_c1 = sum(c[0] == 1 for c in _ipk['SVR'])
        _n_l100 = sum(float(c[1]) == 100.0 for c in _ipk['PFLR'])
        cond('A1 C=1 与 λβ=10² 从未被选中', _n_c1 == 0 and _n_l100 == 0, f'{_n_c1}/{_n_l100}')
        _n_200 = sum(c[0] == 200 for c in _ipk['RF'])
        _n_dp = sum(c['dp'] == 0.4 for c in _ipk['1D-CNN'])
        _n_l40 = sum(c[0] == 40 for c in _ipk['FPCR'])
        chk('A1 删去而被选中过的取值', f'其余被选中过：200棵树{_n_200}折，丢弃率0.4为{_n_dp}折，L=40为{_n_l40}折（FPCR）')
        cond('A1 L=40 未被 FPLR 选中', sum(c[0] == 40 for c in _ipk['PFLR']) == 0, str(_ipk['PFLR']))
        _n_sc = sum(c[1] == 'scale' for c in _ipk['SVR'])
        chk('A1 gamma 一维未调整', f'SVR的gamma一维未作调整，而scale在{_n_sc}折中均被选中')
        cond('A1 初始网格上 gamma 各折均为 scale', _n_sc == 5, str(_ipk['SVR']))
        import runpy
        _G = runpy.run_path('02code/grids.py')['GRIDS']
        _g_c = sorted({c for c, _ in _G['SVR']}); _g_k = sorted(k for _, k in _G['FPCR'])
        _g_nl = sorted({c['nl'] for c in _G['1D-CNN']}); _g_tr = sorted({n for n, _ in _G['RF']})
        cond('A1 主网格 C、K、层数、树数的取值', _g_c == [10, 100, 1000, 10000] and _g_k == [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 79]
             and _g_nl == [3, 4, 5] and _g_tr == [400, 800, 1600], f'{_g_c} {_g_k} {_g_nl} {_g_tr}')
        chk('A1 主网格四维的范围', f'C取{_g_c[0]}~104，K取{_g_k[0]}~{_g_k[-1]}，层数取{_g_nl[0]}~{_g_nl[-1]}，树数取{_g_tr[0]}~{_g_tr[-1]}')
        cond('A1 主网格的丢弃率固定为 0.2、L 固定为 80、gamma 三个取值未变',
             {c['dp'] for c in _G['1D-CNN']} == {0.2} and {L for L, _ in _G['FPCR']} | {L for L, _ in _G['PFLR']} == {80}
             and [g for _, g in _G['SVR'][:3]] == ['scale', 1e-3, 1e-4], '')
        cond('A1 主网格不含 C=1、200 棵树与 λβ=10²', 1 not in _g_c and 200 not in _g_tr and max(lam for _, lam in _G['PFLR']) < 100, '')
        chk('A1 删去取值的归并', '主网格的丢弃率固定为0.2，L固定为80（附录A3）')

    # ============================ 2.2 末段与附录 A3：单种子复核 ============================
    def pooled(stem: str) -> tuple[float, pd.DataFrame]:
        p = pd.read_csv(f'{OUT}/45rebuttal_ncv_{stem}_pred.csv')
        p = p[p.seed == SEED0]
        assert len(p) == _N_KEPT, stem
        assert p.idx.nunique() == _N_KEPT, stem
        return float(r2_score(p.y, p.pred)), p


    FIX = {}                                        # 附录 A3「固定设定」六法：45 正式预测中种子 20041210 的合并外层预测 R²（与 ms_numbers 的 S1POOL 同口径）
    for _m in ALL6:
        FIX[_m], _pp = pooled(STEM[_m])
    if _fn_ready:
        _fn = pd.concat([pd.read_csv(f'{OUT}/44fully_nested{s}.csv') for s in ('', '_rf', '_cnn')])
        _fn['method'] = _fn.method.replace({'PFLR': 'FPLR'})
        _fn = _fn.set_index('method').sort_values('R2', ascending=False)
        _prep_cn = {'原始': '不处理', 'SG1D': 'SG-1D', 'SNV+SG1D': 'SNV+SG-1D'}
        chk('A3 三层全嵌套六法', '六个方法的R²依次为' + '、'.join(f'{m}{_fn.loc[m, "R2"]:.3f}' for m in _fn.index))
        _fix_order = sorted(FIX, key=lambda m: -FIX[m])
        chk('A3 固定设定六法', '（同为合并预测）依次为' + '、'.join(f'{m}{FIX[m]:.3f}' for m in _fix_order))
        _dl = [float(_fn.loc[m, 'R2'] - FIX[m]) for m in ALL6]
        chk('A3 全嵌套带来的变化与排序', f"全嵌套使各方法变化{rg(min(_dl), max(_dl))}，" + ('排序不变' if list(_fn.index) == _fix_order else '排序有所改变'))
        chk('2.2 全嵌套的变化与排序', f"六种方法的R²较同一种子的主分析变化为{rg(min(_dl), max(_dl))}，排序为{'、'.join(list(_fn.index)[:-1])}与{list(_fn.index)[-1]}")
        _svr_prep = str(_fn.loc['SVR', 'prep']).split('|')
        chk('A3 SVR 折内选中 SNV 的折数', f"SVR在5折中有{_svr_prep.count('SNV')}折选中SNV")
        _bF = str(_fn.loc['FPLR', 'basis']).split('|')
        _bC = str(_fn.loc['FPCR', 'basis']).split('|')
        _mF = pd.Series(str(_fn.loc['FPLR', 'prep']).split('|')).value_counts()
        _mC = pd.Series(str(_fn.loc['FPCR', 'prep']).split('|')).value_counts()
        cond('A3 FPLR 与 FPCR 选中最多的预处理相同且唯一', _mF.index[0] == _mC.index[0] and (len(_mF) == 1 or _mF.iloc[0] > _mF.iloc[1])
             and (len(_mC) == 1 or _mC.iloc[0] > _mC.iloc[1]), f'{dict(_mF)}；{dict(_mC)}')
        chk('A3 函数型基与预处理的折内选择', f"5折中分别有{sum('傅里叶' in b for b in _bF)}折与{sum('傅里叶' in b for b in _bC)}折选中傅里叶基、"
                                  f"{int(_mF.iloc[0])}折与{int(_mC.iloc[0])}折选中{_prep_cn.get(_mF.index[0], _mF.index[0])}")
        cond('A3 函数型 L 各折均为 80', all(b.endswith('80') for b in _bF + _bC), '全部 80')
    else:
        PENDING.append(('A3 三层全嵌套与 3.1 末段', '04outputs/44fully_nested{,_rf,_cnn}.csv'))

    _pl = pd.read_csv(f'{OUT}/36prep_leakage.csv')
    _svr_pl = _pl[_pl.method == 'SVR'].set_index('prep').R2
    chk('A3 SVR 预处理对比', f"SVR在{_svr_pl.idxmax()}下的R²最高（{_svr_pl.max():.3f}），在SNV+SG-1D下为{_svr_pl['SNV+SG1D']:.3f}")
    cond('A3 五种预处理下 SVR 均第一、PLSR 均末位',
         all(_pl[(_pl.prep == p) & (_pl.method == 'SVR')].R2.iloc[0] == _pl[_pl.prep == p].R2.max() and
             _pl[(_pl.prep == p) & (_pl.method == 'PLSR')].R2.iloc[0] == _pl[_pl.prep == p].R2.min() for p in _pl.prep.unique()),
         '逐预处理成立')
    _bn = pd.read_csv(f'{OUT}/26basis_nested.csv')
    _b80 = _bn[_bn.L == 80].set_index('basis')
    chk('A3 L=80 的基函数对照', f"B样条基的R²为{_b80.loc['B样条', 'PFLR_R2']:.3f}与{_b80.loc['B样条', 'FPCR_R2']:.3f}，"
                           f"傅里叶基为{_b80.loc['傅里叶', 'PFLR_R2']:.3f}与{_b80.loc['傅里叶', 'FPCR_R2']:.3f}")
    cond('A3 两种基、两个模型均以 L=80 最高', all(_bn[_bn.basis == b][c].idxmax() == _bn[(_bn.basis == b) & (_bn.L == 80)].index[0]
                                             for b in ('B样条', '傅里叶') for c in ('PFLR_R2', 'FPCR_R2')), '逐项成立')
    _lf = pd.read_csv(f'{OUT}/51rebuttal_lams_folds.csv')
    _lm = _lf.groupby(['method', 'lam_s']).R2.mean()
    _ld = {m: [float(_lm.loc[(m, lam)] - _lm.loc[(m, 1e-6)]) for lam in (0.0, 1e-8, 1e-4)] for m in ('FPLR', 'FPCR')}
    chk('A3 λs 敏感性', f"FPLR的R²均值较λs=10-6依次变化{'、'.join(f(v) for v in _ld['FPLR'])}，FPCR依次变化{'、'.join(f(v) for v in _ld['FPCR'])}")
    _lsum = pd.read_csv(f'{OUT}/51rebuttal_lams_summary.csv')
    _lfit = _lsum.groupby('lam_s').fit_rel_diff_vs0.first()
    chk('A3 拟合光谱的相对变化', f"与不加惩罚相比，拟合光谱在λs=10-6时变化{_lfit.loc[1e-6]:.1%}，在10-4时变化{_lfit.loc[1e-4]:.1%}")
    chk('1.3.1 λs 变化范围', f"R²均值的变化为{rg(min(v for vs in _ld.values() for v in vs), max(v for vs in _ld.values() for v in vs))}")
    _pn = pd.read_csv(f'{OUT}/27prep_nested.csv')
    _pm = _pn.groupby('预处理').R2.mean()
    chk('1.2 预处理择优', f"SNV+SG-1D的平均R²为{_pm['SNV+SG1D']:.3f}，高于其余方案的{rg(_pm.drop('SNV+SG1D').min(), _pm.drop('SNV+SG1D').max())}")
    _pd = _pn[_pn.预处理 == 'SNV+SG1D'].set_index('方法').R2
    chk('A3 固定超参数', f"沿用1.2节的固定超参数时（5折交叉验证，各折均值；SVR为C=100、gamma为scale，输入按训练折逐波长标准化），SVR的R²为{_pd['SVR']:.3f}，低于在训练折内选取潜变量数的PLSR（{_pd['PLSR']:.3f}）")
    _src27 = Path('02code/27prep_nested.py').read_text(encoding='utf-8')
    cond('A3 预处理择优时 SVR 的固定设置与输入标准化',
         _src27.count("yp = SVR(C=100, gamma='scale', epsilon=0.1).fit((Xtr - mu) / sdv, ytr).predict((Xte - mu) / sdv)") == 1
         and _src27.count('mu, sdv = Xtr.mean(0), Xtr.std(0) + 1e-8') == 1, '02code/27prep_nested.py')
    chk('1.2 三种离散方法的固定超参数', '其余三种方法用固定超参数、不作调参')
    chk('A3 全嵌套复核的第一阶段候选', 'PLSR、SVR、RF与1D-CNN为附录A1的初始网格，FPCR的K取5~30的12个值，FPLR的λβ取10−8~103')
    _src44 = Path('02code/44fully_nested.py').read_text(encoding='utf-8')
    cond('A3 第一阶段候选的定义', _src44.count("'FPCR': [5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30],") == 1
         and _src44.count("'PFLR': [float(10.0 ** e) for e in range(-8, 4)],") == 1
         and _src44.count("'SVR':  [(C, g) for C in [1, 10, 100, 1000] for g in ['scale', 1e-3, 1e-4]],") == 1
         and _src44.count("'RF':   [(n, mf) for n in [200, 400, 800] for mf in ['sqrt', 0.1, 0.3, 1.0]],") == 1, '02code/44fully_nested.py')
    chk('A3 基设定复核的候选', "此项复核中λβ取10−10~102的13个值，K取L的0.1~0.9倍")

# 附录 A1：两个平台上数据的比对（45 上传脚本写下的日志：原始光谱、参考值与折划分的哈希，预处理后光谱的逐值比较）
_up_logs = [sorted(glob.glob(f'05logs/45rebuttal_R1_4_repeated_ncv_{t}-upload_*.log'))[-1:] for t in ('RF', '1DCNN')]
if all(_up_logs):
    _up_max = []
    for (_lp,) in _up_logs:
        _lt = Path(_lp).read_text(encoding='utf-8')
        _ex = re.findall(r'(本机|服务器)\s+EXACT n=(\d+) p=(\d+) dtype=float64\s+X=(\w+)\s+y=(\w+)\s+folds=(\w+)', _lt)
        cond(f'A1 两平台的原始光谱、参考值与折划分相同（{Path(_lp).name}）', len(_ex) == 2 and _ex[0][1:] == _ex[1][1:] and _ex[0][1:3] == ('1111', '2048'),
             str([e[3:] for e in _ex]))
        _mx = re.search(r'最大绝对差 ([0-9.e+-]+)（容差', _lt)
        _up_max.append(float(_mx.group(1)) if _mx else float('nan'))
    _e15 = max(_up_max) / 1e-15
    chk('A1 两平台预处理后光谱的最大绝对差', f"预处理后光谱的最大绝对差为{_e15:.1f}×10{MINUS}15")
else:
    PENDING.append(('A1 两平台数据的比对', '05logs/45rebuttal_R1_4_repeated_ncv_{RF,1DCNN}-upload_*.log'))

# ============================ 1.1：样品构成（46 批次表）============================
_bt = pd.read_csv(f'{OUT}/46rebuttal_recal_batches.csv').set_index('name')

# ============================ 2.3 与附录 A2：小样本（21 逐次抽样）============================
_rep = pd.read_csv(f'{OUT}/21smallsample_repeats.csv')
_rep['method'] = _rep.method.replace({'PFLR': 'FPLR'})
_gr = _rep.groupby(['N', 'method', 'strategy']).R2
SS = pd.DataFrame({'mean': _gr.mean(), 'median': _gr.median(), 'min': _gr.min(), 'n': _gr.count(),
                   'n_neg': _gr.apply(lambda v: int((v < 0).sum())), 'se': _gr.std(ddof=1) / np.sqrt(_gr.count())})
assert (SS.n == 20).all()
NS = sorted({i[0] for i in SS.index})


def ss(n: int, m: str, st: str, c: str) -> float:
    return float(SS.loc[(n, m, st), c])


_w = _rep[(_rep.N == 80) & (_rep.method == 'FPCR')].pivot(index='seed', columns='strategy', values='R2')
_hi, _lo = int((_w.cv > _w.fixed + 1e-12).sum()), int((_w.cv < _w.fixed - 1e-12).sum())
_mid = (50, 80, 120, 200)
_fplr_mid = [ss(n, 'FPLR', 'cv', 'median') for n in _mid]
chk('2.3、结论（2）FPLR 中位数范围', f"N=50~200时，交叉验证选参的FPLR中位数为{rg(min(_fplr_mid), max(_fplr_mid))}")
cond('2.3 FPLR、FPCR 的中位数次序', all(
    ss(n, 'FPLR', 'cv', 'median') > ss(n, 'FPCR', 'cv', 'median')
    and ss(n, 'FPCR', 'cv', 'median') > max(ss(n, m, st, 'median')
                                           for m in ('PLSR', 'SVR') for st in ('fixed', 'cv')
                                           if not (m == 'PLSR' and st == 'cv'))
    for n in _mid), 'N=50~200 时 FPLR 第一、FPCR 第二')
_fplr_vs_svr = []
for _n in _mid:
    _cmp = _rep[(_rep.N == _n) & (_rep.method.isin(['FPLR', 'SVR'])) & (_rep.strategy == 'cv')].pivot(
        index='seed', columns='method', values='R2')
    _fplr_vs_svr.append(int((_cmp.FPLR > _cmp.SVR).sum()))
chk('A2 FPLR 逐次高于 SVR 的次数', f"N=50~200时逐次比较，FPLR在20次抽样中有{min(_fplr_vs_svr)}~{max(_fplr_vs_svr)}次高于交叉验证选参的SVR")
_gain_cv = [ss(n, m, 'cv', 'median') - ss(n, m, 'fixed', 'median') for n in NS if n >= 50 for m in ('FPCR', 'FPLR')]
chk('2.3 交叉验证选参的中位数增益', f"N≥50时，两种函数型模型交叉验证选参的中位数比固定超参数高{rg(min(_gain_cv), max(_gain_cv))}（逐次比较与选中的K见附录A2）")
chk('结论（2）交叉验证选参的中位数增益', f"N≥50时两种函数型模型交叉验证选参的中位数比固定超参数高{rg(min(_gain_cv), max(_gain_cv))}，但固定超参数时SVR（输入未另作标准化）"
                             "在各样本量下的均值均高于两种函数型模型；训练样本不超过30个时各方法均不可靠")
chk('摘要 小样本', f"训练样本为50~200个时，交叉验证选参的FPLR的R²中位数为{rg(min(_fplr_mid), max(_fplr_mid))}，在各方法中最高，比其固定超参数时高0.2以上；"
                 "样本不超过30个时，各方法均出现R²<0")
chk('Abstract 小样本', f"With 50-200 training samples, cross-validated FPLR gave the highest median R² ({min(_fplr_mid):.3f}-{max(_fplr_mid):.3f}), "
                      "and its median was more than 0.2 higher than with fixed hyperparameters; with 30 or fewer, every method gave R²<0 in some draws")
_fplr_gain_mid = [ss(n, 'FPLR', 'cv', 'median') - ss(n, 'FPLR', 'fixed', 'median') for n in _mid]
cond('摘要 N=50~200 时 FPLR 交叉验证选参的中位数比固定超参数高 0.2 以上', min(_fplr_gain_mid) > 0.2, f'最小 {min(_fplr_gain_mid):.3f}')
cond('2.3 N=400 与 PLSR 排序', ss(400, 'SVR', 'cv', 'median') == max(ss(400, m, st, 'median')
     for m in ('PLSR', 'SVR', 'FPCR', 'FPLR') for st in ('fixed', 'cv') if not (m == 'PLSR' and st == 'cv'))
     and all(ss(n, 'PLSR', 'fixed', 'median') == min(ss(n, m, st, 'median') for m in ('PLSR', 'SVR', 'FPCR', 'FPLR')
             for st in ('fixed', 'cv') if not (m == 'PLSR' and st == 'cv')) for n in (30, 50, 80, 120, 200)),
     'N=400 时 SVR 最高；N=30~200 时 PLSR 最低')
_low_neg = [int(ss(n, m, st, 'n_neg')) for n in (20, 30) for m, st in
            [(m, s) for m in ('SVR', 'FPCR', 'FPLR') for s in ('fixed', 'cv')] + [('PLSR', 'fixed')]]
chk('2.3 N≤30 的负值次数', f"训练样本不超过30个时，各方法与做法出现{min(_low_neg)}~{max(_low_neg)}次R²<0，均不可靠")
cond('摘要 N≤30 时各方法与做法均出现 R²<0', min(_low_neg) >= 1, f'最少 {min(_low_neg)} 次')
_n50_neg = {(m, st): int(ss(50, m, st, 'n_neg')) for m in ('SVR', 'FPCR', 'FPLR') for st in ('fixed', 'cv')
             if ss(50, m, st, 'n_neg') > 0}
chk('2.3 N=50 与 N≥80 的负值次数', f"N=50时仍有{len(_n50_neg)}种组合各出现1次，N≥80时均无")
cond('2.3 N=50 负值分支', all(v == 1 for v in _n50_neg.values()) and all(ss(n, m, st, 'n_neg') == 0
     for n in NS if n >= 80 for m in ('SVR', 'FPCR', 'FPLR') for st in ('fixed', 'cv')), str(_n50_neg))
cond('2.3 固定超参数时 SVR 均值最高', all(ss(n, 'SVR', 'fixed', 'mean') > ss(n, m, 'fixed', 'mean')
                                              for n in NS for m in ('FPCR', 'FPLR')), '每个 N 均高于 FPCR、FPLR')
cond('2.3 交叉验证时 FPLR 均值高于 SVR', all(ss(n, 'FPLR', 'cv', 'mean') > ss(n, 'SVR', 'cv', 'mean')
                                                for n in (30, 50, 80, 120, 200)), 'N=30~200 逐项成立')
_sv = pd.read_csv(f'{OUT}/18svr_aligned.csv').set_index('N')
cond('2.3 SVR 标准化分支', _sv.loc[20, 'SVR_std'] > _sv.loc[20, 'SVR_raw']
     and all(_sv.loc[n, 'SVR_std'] < _sv.loc[n, 'SVR_raw'] for n in NS if n >= 30),
     'N=20 升高，N≥30 均降低')
cond('A2 18 脚本的 SVR 与 FPCR 列与 21 的均值一致', all(abs(_sv.loc[n, 'SVR_raw'] - ss(n, 'SVR', 'fixed', 'mean')) < 5e-4 and
                                                    abs(_sv.loc[n, 'FPCR'] - ss(n, 'FPCR', 'fixed', 'mean')) < 5e-4 for n in NS), '差 <5e-4')
_nf = max(n for n in NS for m in ('FPCR', 'FPLR') if ss(n, m, 'cv', 'n_neg') > 0)
_ns = max(n for n in NS if ss(n, 'SVR', 'cv', 'n_neg') > 0)
chk('A2 标准化敏感性 N=20', f"若另作标准化，固定超参数的SVR均值在N=20时升高（{_sv.loc[20, 'SVR_std']:.3f}对{_sv.loc[20, 'SVR_raw']:.3f}）")
chk('A2 标准化敏感性 N=50', f"在N≥30时降低（N=50时{_sv.loc[50, 'SVR_std']:.3f}对{_sv.loc[50, 'SVR_raw']:.3f}）")
for _n in NS:
    cells = [('PLSR', 'fixed'), ('SVR', 'fixed'), ('SVR', 'cv'), ('FPCR', 'fixed'), ('FPCR', 'cv'), ('FPLR', 'fixed'), ('FPLR', 'cv')]
    chk(f'表A2 N={_n}', f'{_n}' + ''.join(f"{f(ss(_n, m, s, 'mean'))}({ss(_n, m, s, 'se'):.3f})" for m, s in cells))

_log18 = Path(f'{LOGS}/18svr_align_and_sensitivity.log').read_text(encoding='utf-8')
_blk = _log18.split('FPCR（12 组候选的 R²）：')[1].split('→')[0]
_k80 = {int(k): float(v) for k, v in re.findall(r'\(40, (\d+)\)\s+(-?\d+\.\d+)', _blk)}
assert len(_k80) == 12
_stab = [_k80[k] for k in (8, 10, 12, 15)]
chk('A2 逐个 K 的评估', f"K=10时为{_k80[10]:.3f}，K取20、25、30时为{rg(min(_k80[k] for k in (20, 25, 30)), max(_k80[k] for k in (20, 25, 30)))}")
chk('A2 N=80 两种做法逐次比较', f"N=80时逐次比较两种做法，交叉验证选参的FPCR有{_hi}次高于固定K=10、{_lo}次较低")
_log20 = Path(f'{LOGS}/20cv_selection_mechanism.log').read_text(encoding='utf-8')
_mk = re.search(r'交叉验证选中的 K 的分布：\s*(\{[^}]*\})', _log20)
_mr = re.search(r'中位\s*(\d+)%', _log20)
assert _mk is not None
assert _mr is not None
_ck = {int(a): int(b) for a, b in re.findall(r'(\d+):\s*(\d+)', _mk.group(1))}
chk('A2 选中 K 的次数与误差波动', f"交叉验证在20次抽样中选中K=15、20、30的次数依次为{_ck.get(15, 0)}、{_ck.get(20, 0)}、{_ck.get(30, 0)}，"
                             f"所选K处4折误差的相对标准差中位数为{_mr.group(1)}%")
cond('A2 选中 K=10 的次数等于逐次比较中「选中同一 K」的次数', _ck.get(10, 0) == 20 - _hi - _lo, '相等')
chk('1.5 选参端两种做法的理由', f"N=80时FPCR所选K处4折验证误差的相对标准差中位数为{_mr.group(1)}%（附录A2），选参易受划分影响")
if _pre_small is not None:
    # 两端另取 100 次抽样的理由：清洗前 N=50~120 交叉验证选参的函数型模型各档失效频率的最低值，在 100 次与 20 次中一次都不出现的概率
    _p_min = min(_pre_fun) / 20
    chk('1.5 另取 100 次抽样的理由', f"若某档失效频率为清洗前各档最低的{Decimal(repr(_p_min * 100)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)}%，则"
                                f"100次全不出现的概率约{(1 - _p_min) ** 100:.1%}，20次约{(1 - _p_min) ** 20:.0%}")
    cond('1.5 清洗前各档均有失效（最低频率大于 0）', min(_pre_fun) >= 1, str(_pre_fun))
_m20 = re.search(r'交叉验证所选 K 的实际 R² 均值 ([0-9.]+)，事后最优 K 的 R² 均值 ([0-9.]+)，平均损失 ([0-9.]+)', _log20)
assert _m20 is not None
chk('A2 所选 K 的平均 R² 与事后最优之差', f"所选K的平均R²为{_m20.group(1)}，比逐次事后最优的K低{_m20.group(3)}")
cond('A2 平均损失等于事后最优与所选 K 的平均 R² 之差', abs(float(_m20.group(2)) - float(_m20.group(1)) - float(_m20.group(3))) <= 0.0011,
     f'{_m20.group(2)} − {_m20.group(1)}')

# ============================ 2.3 与附录 A8：小样本失效的对策与成因（56、57、58 逐次抽样）============================
# 不读 56、57、58 的汇总表，只读逐次抽样与逐个失效抽样的产物，按定义重算次数、中位数与份额
_d56 = pd.read_csv(f'{OUT}/56rebuttal_selection_draws.csv')
_d56 = _d56[_d56.seed_set == '新100次']
_d57 = pd.read_csv(f'{OUT}/57rebuttal_guard_draws.csv')
_d58 = pd.read_csv(f'{OUT}/58rebuttal_guard_posthoc_draws.csv')
_an = pd.read_csv(f'{OUT}/57rebuttal_guard_anatomy.csv')
_sp = pd.read_csv(f'{OUT}/57rebuttal_guard_spectra.csv').set_index('idx')
_U, _F, _GN = 'B10.8的100次', '再取100次', (50, 80, 120)
assert set(_d56.seed) == set(range(20, 120))
assert set(_d57[_d57.seed_set == _U].seed) == set(range(20, 120))
assert set(_d57[_d57.seed_set == _F].seed) == set(range(120, 220))
chk('A8 另取抽样的限定', "各另取此前未在任何计算中用过的100次抽样")
_s21 = set(pd.read_csv(f'{OUT}/21smallsample_repeats.csv').seed)
cond('A8 两批另取抽样与 20 次抽样的随机种子互不重合',
     _s21 == set(range(20)) and not (_s21 & set(_d56.seed)) and not (set(_d56.seed) & set(_d57[_d57.seed_set == _F].seed)),
     '随机种子 0~19、20~119、120~219')


def _cell(df: pd.DataFrame, m: str, arm: str, n: int, col: str = 'arm') -> pd.Series:
    q = df[(df.method == m) & (df[col] == arm) & (df.N == n)].R2
    assert len(q) == 100, (m, arm, n, len(q))
    return q


def _neg(df: pd.DataFrame, m: str, arm: str, col: str = 'arm') -> list[int]:
    return [int((_cell(df, m, arm, n, col) < 0).sum()) for n in _GN]


def _cnt(v: list[int]) -> str:
    return '、'.join(str(x) for x in v)


_f57 = _d57[_d57.seed_set == _F]
_u57 = _d57[_d57.seed_set == _U]
_d1_fpcr = [float(_cell(_d56, 'FPCR', 'onese', n).median() - _cell(_d56, 'FPCR', 'cv', n).median()) for n in _GN]
_drep_fpcr = [float(_cell(_d56, 'FPCR', 'repcv', n).median() - _cell(_d56, 'FPCR', 'cv', n).median()) for n in _GN]
_d1_fplr = [float(_cell(_d56, 'FPLR', 'onese', n).median() - _cell(_d56, 'FPLR', 'cv', n).median()) for n in _GN]
_drep_fplr = [float(_cell(_d56, 'FPLR', 'repcv', n).median() - _cell(_d56, 'FPLR', 'cv', n).median()) for n in _GN]
_sel_neg = {m: _neg(_d56, m, 'cv') for m in ('FPCR', 'FPLR')}
_one_se_drop = min(-np.array(_d1_fpcr + _d1_fplr)), max(-np.array(_d1_fpcr + _d1_fplr))
_repeat_delta = min(_drep_fpcr + _drep_fplr), max(_drep_fpcr + _drep_fplr)
chk('2.3、A8 另取 100 次抽样的选参端次数',
    f"另取的100次抽样上（附录A8），N=50、80、120三档交叉验证选参的FPCR与FPLR均无R²<0，"
    f"一倍标准误准则使中位数降低{rg(*_one_se_drop)}，重复交叉验证与单次交叉验证的中位数之差为{rg(*_repeat_delta)}")
cond('2.3 FPCR 选参端判据', min(_d1_fpcr) < -0.02 and min(_drep_fpcr) >= -0.02,
     'FPCR：一倍标准误无效、重复交叉验证有效')
cond('2.3 FPLR 选参端判据', min(_d1_fplr) < -0.02 and min(_drep_fplr) >= -0.02,
     'FPLR：一倍标准误无效、重复交叉验证有效')
_fail80 = {a: set(_d56[(_d56.N == 80) & (_d56.method == 'FPCR') & (_d56.arm == a) & (_d56.R2 < 0)].seed) for a in ('cv', 'onese', 'repcv')}
_anu = _an[(_an.seed_set == _U) & _an.N.isin(_GN)]
_anf = _an[(_an.seed_set == _F) & _an.N.isin(_GN)]
cond('A8 剖析产物的失效抽样数与逐次抽样一致', len(_anu) == sum(_neg(_u57, m, 'cv')[i] for m in ('FPCR', 'FPLR') for i in range(3))
     and len(_anf) == sum(_neg(_f57, m, 'cv')[i] for m in ('FPCR', 'FPLR') for i in range(3)), f'{len(_anu)}、{len(_anf)} 个')
_dr_u = [float(_anu[_anu.method == m].R2_drop5.median()) for m in ('FPCR', 'FPLR')]
_dr_f = [float(_anf[_anf.method == m].R2_drop5.median()) for m in ('FPCR', 'FPLR')]
if not _anu.empty:
    chk('2.3 失效抽样的平方误差份额与剔除后的 R²', f"误差最大的3个占平方误差总和的{float(_anu.share_top3.median()):.0%}（中位数），剔除误差最大的5个后R²中位数为{_dr_u[0]:.3f}与{_dr_u[1]:.3f}")
    chk('A8 第一组失效抽样数', f"FPCR{int((_anu.method == 'FPCR').sum())}个、FPLR{int((_anu.method == 'FPLR').sum())}个")
if not _anf.empty:
    _anf_f = _anf[_anf.method == 'FPLR'].iloc[0]
    _anf_c = _anf[_anf.method == 'FPCR'].iloc[0]
    _f57_194 = _f57[(_f57.method == 'FPLR') & (_f57.N == 50) & (_f57.seed == _anf_f.seed)]
    _f57_201 = _f57[(_f57.method == 'FPCR') & (_f57.N == 50) & (_f57.seed == _anf_c.seed)]
    _f57_194_lev = float(_f57_194[_f57_194.arm == 'lev'].R2.iloc[0])
    _f57_194_clip = float(_f57_194[_f57_194.arm == 'clip'].R2.iloc[0])
    _f57_201_lev = float(_f57_201[_f57_201.arm == 'lev'].R2.iloc[0])
    _f57_201_clip = float(_f57_201[_f57_201.arm == 'clip'].R2.iloc[0])
    _top3_f = [int(x) for x in str(_anf_f.top3_idx).split()]
    _top3_c = [int(x) for x in str(_anf_c.top3_idx).split()]
    cond('2.3 FPLR 失效一次误差最大的 2 个测试样品即弱折的 2 条光谱', sorted(_top3_f[:2]) == HOT2, f'{_top3_f}；{HOT2}')
    cond('2.3 FPCR 失效一次误差最大的 3 个样品不含这 2 条光谱、回退与截断后仍为负',
         not set(_top3_c) & set(HOT2) and _f57_201_lev < 0 and _f57_201_clip < 0,
         f'{_top3_c}；回退后 {_f57_201_lev:.4f}，截断后 {_f57_201_clip:.4f}')
    chk('A8 第二组的份额与剔除后的 R²', f"再取的100次抽样中N={_GN[0]}~{_GN[-1]}时交叉验证选参的失效抽样为FPCR、FPLR各1次（均为N=50）：FPLR的一次误差最大的3个测试样品占平方误差总和的{_anf_f.share_top3:.0%}，其中2个即2.2节弱折的2条光谱，杠杆回退与范围截断后R²由{f(_anf_f.R2)}升至{f(_f57_194_lev)}与{f(_f57_194_clip)}；FPCR的一次 R² 为 {f(_anf_c.R2)}，误差最大的3个测试样品占{_anf_c.share_top3:.0%}，回退与截断后为 {f(_f57_201_lev)}与 {f(_f57_201_clip)}")
_pre_sp = _git_csv('2331884', '04outputs/57rebuttal_guard_spectra.csv')
if _pre_sp is not None:
    _pre_sp = _pre_sp.set_index('idx')
    if _clean_path.exists() and HOT2:
        _h6 = _clean[_clean.old_ok & _clean.old_idx.isin(list(_pre_sp.index))]
        _h6_rm, _h6_kp = _h6[~_h6.keep], _h6[_h6.keep]
        cond('2.3、A9 六条异常光谱中清洗剔除 4 条、保留的 2 条即弱折的 2 条',
             len(_h6) == 6 and len(_h6_rm) == 4 and sorted(int(i) for i in _h6_kp.new_idx) == HOT2,
             f'剔除 {sorted(int(i) for i in _h6_rm.old_idx)}；保留 {sorted(int(i) for i in _h6_kp.old_idx)}→{sorted(int(i) for i in _h6_kp.new_idx)}')
        cond('A9 剔除的 4 条均为规则①命中，其中 1 条另为规则④、2 条另为规则③命中',
             bool(_h6_rm.R1.all()) and int(_h6_rm.R4.sum()) == 1 and int(_h6_rm.R3.sum()) == 2, str(_h6_rm[['old_idx', 'R1', 'R3', 'R4']].values.tolist()))
        _low_old = int(_pre_sp.raw_max_quantile.idxmin())
        _kp_old = sorted(int(i) for i in _h6_kp.old_idx)
        _oth_old = [i for i in _kp_old if i != _low_old]
        cond('A9 保留的 2 条为最大读数最低的 1 条与未见单一特征的 1 条', _low_old in _kp_old and len(_oth_old) == 1
             and int(_pre_sp.loc[_oth_old[0], 'n_points_at_adc_max']) == 0 and pd.isna(_pre_sp.loc[_oth_old[0], 'identical_to']),
             f'最大读数最低 {_low_old}；另一条 {_oth_old}')
        _dist2 = [float(_h6_kp.set_index('old_idx').loc[i, 'R1_dist']) for i in [_low_old] + _oth_old]
        chk('A9 保留的 2 条的规则①距离', f"二者在光谱离群规则下的距离为{_dist2[0]:.1f}与{_dist2[1]:.1f}，低于2018年的阈值{_thr18:.1f}（中位数{float(_d18.median()):.1f}），故未被剔除")
        cond('A9 二者的距离均低于 2018 年阈值', max(_dist2) < _thr18, f'{_dist2}；阈值 {_thr18:.2f}')
    cond('2.3 清洗前六条异常光谱的模式', list(_pre_sp.index) == [71, 115, 125, 126, 131, 223]
         and list(_pre_sp.index[_pre_sp.n_points_at_adc_max > 0]) == [71]
         and abs(float(_pre_sp.loc[115, 'raw_max_quantile']) - 1 / 1300) < 1e-9
         and int(float(_pre_sp.loc[125, 'identical_to'])) == 126
         and bool((_pre_sp.year == 2018).all() and (_pre_sp.origin == 1).all()), '清洗前为1条饱和、1条最大读数最低、1对逐点相同，均为2018年新疆批次')
else:
    PENDING.append(('3.2 清洗前六条异常光谱的模式', 'git show 2331884:04outputs/57rebuttal_guard_spectra.csv'))
cond('A8 随机种子 20~119 的失效抽样为 0、导出的光谱名单为空', _sp.empty and _anu.empty, f'失效抽样 {len(_anu)} 个；名单 {len(_sp)} 条')
chk('A8 名单为空的范围', f"清洗后这批抽样在N={_GN[0]}~{_GN[-1]}未出现失效，名单为空")
_cv57 = {m: _neg(_f57, m, 'cv') for m in ('FPCR', 'FPLR')}
_lev57 = {m: _neg(_f57, m, 'lev') for m in ('FPCR', 'FPLR')}
_clip57 = {m: _neg(_f57, m, 'clip') for m in ('FPCR', 'FPLR')}
cond('2.3 再取抽样上 FPCR 与 FPLR 的交叉验证失效次数相同（合写）', _cv57['FPCR'] == _cv57['FPLR'], str(_cv57))
chk('2.3 再取 100 次抽样的交叉验证失效次数', f"再取的100次抽样上，交叉验证选参的FPCR与FPLR在三档各有{_cnt(_cv57['FPCR'])}次R²<0")
chk('A8 再取 100 次抽样的次数', f"交叉验证选参的FPCR与FPLR在三档有{_cnt(_cv57['FPCR'])}次与{_cnt(_cv57['FPLR'])}次R²<0；"
                             f"杠杆回退后FPCR与FPLR为{_cnt(_lev57['FPCR'])}次与{_cnt(_lev57['FPLR'])}次")
cond('2.3、A8 FPCR 预测端对策判据', _cv57['FPCR'] == [1, 0, 0] and _lev57['FPCR'] == [1, 0, 0]
     and _clip57['FPCR'] == [1, 0, 0]
     and all(_cell(_f57, 'FPCR', a, n).median() - _cell(_f57, 'FPCR', 'cv', n).median() >= -0.02
             for a in ('lev', 'clip') for n in _GN), 'FPCR：交叉验证、杠杆回退与范围截断次数及中位数变化')
cond('2.3、A8 FPLR 预测端对策判据', _cv57['FPLR'] == [1, 0, 0] and _lev57['FPLR'] == [0, 0, 0]
     and _clip57['FPLR'] == [0, 0, 0]
     and all(_cell(_f57, 'FPLR', a, n).median() - _cell(_f57, 'FPLR', 'cv', n).median() >= -0.02
             for a in ('lev', 'clip') for n in _GN), 'FPLR：交叉验证、杠杆回退与范围截断次数及中位数变化')
_lev_delta = [float(_cell(_f57, m, 'lev', n).median() - _cell(_f57, m, 'cv', n).median()) for m in ('FPCR', 'FPLR') for n in _GN]
cond('A8 范围截断次数分支', _clip57 == {'FPCR': [1, 0, 0], 'FPLR': [0, 0, 0]}, str(_clip57))
_clip_delta = [float(_cell(_f57, m, 'clip', n).median() - _cell(_f57, m, 'cv', n).median()) for m in ('FPCR', 'FPLR') for n in _GN]
chk('A8 杠杆回退与范围截断的次数与中位数变化', f"杠杆回退后FPCR与FPLR为{_cnt(_lev57['FPCR'])}次与{_cnt(_lev57['FPLR'])}次，中位数较交叉验证选参升高{rg(min(_lev_delta), max(_lev_delta))}；"
                                    f"范围截断后为{_cnt(_clip57['FPCR'])}次与{_cnt(_clip57['FPLR'])}次，中位数升高{rg(min(_clip_delta), max(_clip_delta))}")


# 协议 B10.9 判读 (iv)：两种做法都有效时正文写 R²<0 总次数较少者，相同则写三档中位数之和较高者；FPCR、FPLR 分别判读。
# 清洗前后用同一函数，清洗前取提交 2331884 的逐次抽样
def _guard_ok(df: pd.DataFrame, m: str, a: str) -> bool:
    return all(v == 0 for v in _neg(df, m, a)) and all(
        float(_cell(df, m, a, n).median() - _cell(df, m, 'cv', n).median()) >= -0.02 for n in _GN)


def _med_sum(df: pd.DataFrame, m: str, a: str) -> float:
    return sum(float(_cell(df, m, a, n).median()) for n in _GN)


def _arm_iv(df: pd.DataFrame, m: str) -> str | None:
    ok = {a: _guard_ok(df, m, a) for a in ('lev', 'clip')}
    if not any(ok.values()):
        return None
    if not all(ok.values()):
        return 'lev' if ok['lev'] else 'clip'
    tot = {a: sum(_neg(df, m, a)) for a in ('lev', 'clip')}
    if tot['lev'] != tot['clip']:
        return min(('lev', 'clip'), key=lambda a: tot[a])
    return max(('lev', 'clip'), key=lambda a: _med_sum(df, m, a))


_iv = {m: _arm_iv(_f57, m) for m in ('FPCR', 'FPLR')}
cond('2.3 判读 (iv)：FPLR 报告范围截断、FPCR 两种做法均无效', _iv == {'FPCR': None, 'FPLR': 'clip'}, str(_iv))
cond('2.3 两种做法的 R²<0 次数相同（合写）', _lev57 == _clip57, f'{_lev57}；{_clip57}')
_clip_d_fplr = [float(_cell(_f57, 'FPLR', 'clip', n).median() - _cell(_f57, 'FPLR', 'cv', n).median()) for n in _GN]
if not _anf.empty:
    chk('2.3 预测端对策的判定与两次失效', f"FPLR在两种做法下均为{_cnt(_clip57['FPLR'])}次，按判据均有效，按事先规则报告三档中位数之和较高的范围截断："
        f"中位数升高{rg(min(_clip_d_fplr), max(_clip_d_fplr))}，失效的那一次误差最大的2个测试样品即上述2条光谱，"
        f"截断使其R²由{f(_anf_f.R2)}升至{f(_f57_194_clip)}（杠杆回退见附录A8）。"
        f"FPCR在两种做法下仍为{_cnt(_clip57['FPCR'])}次，按判据均无效：失效的一次R²为{f(_anf_c.R2)}，"
        f"误差最大的3个样品仅占平方误差的{_anf_c.share_top3:.0%}，不包含上述2条光谱")
chk('A8 判读 (iv) 的执行', f"FPLR两种做法三档的R²<0次数均为{sum(_clip57['FPLR'])}，中位数之和范围截断为{_med_sum(_f57, 'FPLR', 'clip'):.3f}、"
                         f"杠杆回退为{_med_sum(_f57, 'FPLR', 'lev'):.3f}，2.3节报告范围截断；FPCR两种做法均无效")
chk('结论（2）预测端对策', f"范围截断与杠杆回退对FPLR均有效，FPCR的{sum(_cv57['FPCR'])}次失效误差分散，两种做法均未消除")
chk('局限 预测端做法消除的失效', f"N={_GN[0]}~{_GN[-1]}时预测端两种做法消除的函数型模型失效仅有FPLR的{sum(_cv57['FPLR']) - sum(_clip57['FPLR'])}次")
cond('局限 两种做法均未消除 FPCR 的失效', sum(_cv57['FPCR']) == sum(_clip57['FPCR']) == sum(_lev57['FPCR']), f"{_cv57['FPCR']}；{_clip57['FPCR']}；{_lev57['FPCR']}")
chk('结论（2）失效来源的范围', f"N={_GN[0]}~{_GN[-1]}时清洗前函数型模型的失效与FPLR在再取抽样中的失效，主要来自训练集未覆盖的异常光谱")
cond('局限 两种做法对 FPLR 消除的失效次数相同', sum(_lev57['FPLR']) == sum(_clip57['FPLR']), f'{_lev57}；{_clip57}')
# 附录 A9：四种对策清洗前后的判定（同一判据），与 2.3 节在清洗前后按判读 (iv) 报告的做法
_pre56 = _git_csv('2331884', '04outputs/56rebuttal_selection_draws.csv')
_pre57 = _git_csv('2331884', '04outputs/57rebuttal_guard_draws.csv')
if _pre56 is not None and _pre57 is not None:
    _pre56 = _pre56[_pre56.seed_set == '新100次']
    _pre57 = _pre57[_pre57.seed_set == _F]
    assert set(_pre56.seed) == set(range(20, 120))
    assert set(_pre57.seed) == set(range(120, 220))
    _vd = {True: '有效', False: '无效'}
    _arms9 = [('一倍标准误准则', 'onese', _pre56, _d56), ('重复交叉验证', 'repcv', _pre56, _d56),
              ('杠杆回退', 'lev', _pre57, _f57), ('范围截断', 'clip', _pre57, _f57)]
    _chg9: list[str] = []
    _same9: list[str] = []
    for _lab9, _a9, _dpre, _dpost in _arms9:
        _grp9: dict[tuple[bool, bool], list[str]] = {}
        for _m9 in ('FPCR', 'FPLR'):
            _grp9.setdefault((_guard_ok(_dpre, _m9, _a9), _guard_ok(_dpost, _m9, _a9)), []).append(_m9)
        _c9 = [f"对{'、'.join(ms)}由{_vd[pb[0]]}变为{_vd[pb[1]]}" for pb, ms in _grp9.items() if pb[0] != pb[1]]
        _s9 = [f"对{'、'.join(ms)}前后均为{_vd[pb[0]]}" for pb, ms in _grp9.items() if pb[0] == pb[1]]
        if _c9:
            _chg9.append(_lab9 + '、'.join(_c9))
        if _s9:
            _same9.append(_lab9 + '、'.join(_s9))
    chk('A9 各对策清洗前后的判定', "按附录A8的判据，清洗前后各对策的判定：" + "，".join(_chg9) + "；" + "，".join(_same9) + "。")
    _iv_pre = {m: _arm_iv(_pre57, m) for m in ('FPCR', 'FPLR')}
    cond('A9 清洗前按判读 (iv) 报告的做法为两种模型均杠杆回退', _iv_pre == {'FPCR': 'lev', 'FPLR': 'lev'}, str(_iv_pre))
    chk('A9 2.3 节清洗前后报告的做法', "按同一判读规则，2.3节报告的预测端做法清洗前FPCR与FPLR均为杠杆回退，清洗后FPLR为范围截断、FPCR两种做法均无效")
    chk('A9 再取抽样上的失效次数', f"选参端另取的100次抽样中没有出现，预测端再取的100次抽样中FPCR与FPLR各{sum(_cv57['FPCR'])}次")
cond('2.3 同作范围截断时 N=50~200 函数型模型的中位数高于 SVR 与 PLSR',
     all(_cell(_f57, m, 'clip', n).median() > max(_cell(_f57, 'SVR', 'clip', n).median(), _cell(_f57, 'PLSR', 'clip', n).median())
         for m in ('FPCR', 'FPLR') for n in (50, 80, 120, 200)), '逐项成立')
cond('A8 同作范围截断时 N=50、80、120 的次序为 FPLR、FPCR、SVR、PLSR，N=400 时 SVR 最高',
     all(_cell(_f57, 'FPLR', 'clip', n).median() > _cell(_f57, 'FPCR', 'clip', n).median() > _cell(_f57, 'SVR', 'clip', n).median()
         > _cell(_f57, 'PLSR', 'clip', n).median() for n in _GN)
     and all(_cell(_f57, 'SVR', 'clip', 400).median() > _cell(_f57, m, 'clip', 400).median() for m in ('FPCR', 'FPLR', 'PLSR')), '逐项成立')


def _pct1(x: float) -> str:
    """百分数一位小数，四舍五入（不用格式化默认的逢五取偶）。"""
    return str(Decimal(repr(round(x * 100, 6))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


_frac = [float(_f57[(_f57.method == m) & (_f57.arm == 'lev') & (_f57.N == n)].frac_backed.median()) for m in ('FPCR', 'FPLR') for n in _GN]
chk('A8 被回退的测试光谱占比', f"被回退的测试光谱占比的中位数为{_pct1(min(_frac))}%~{_pct1(max(_frac))}%")
_f58 = _d58[_d58.seed_set == _F]


def _m58(m: str, rule: str, n: int, col: str) -> pd.Series:
    q = _f58[(_f58.method == m) & (_f58.rule == rule) & (_f58.N == n)][col]
    assert len(q) == 100
    return q


_xneg = {rule: sum(int((_m58(m, rule, n, 'R2_all') < 0).sum()) for m in ('FPCR', 'FPLR') for n in _GN)
          for rule in ('x2', 'x3', 'x4')}
chk('A8 倍数 2、3、4 的失效次数', f"倍数改取2、3、4时三档的R²<0次数合计为{_xneg['x2']}、{_xneg['x3']}、{_xneg['x4']}次")
_xmed = {rule: [float(_m58(m, rule, n, 'R2_all').median() - _m58(m, 'cv', n, 'R2_all').median())
               for m in ('FPCR', 'FPLR') for n in _GN] for rule in ('x2', 'x3', 'x4')}
def _change(v: float) -> str:
    return ('降低' if v < 0 else '升高') + f(abs(v))
def _change_range(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    return ('降低' if hi < 0 else '升高') + rg(min(abs(lo), abs(hi)), max(abs(lo), abs(hi)))
chk('A8 倍数 2、3、4 的中位数变化', f"倍数改取2、3、4时三档的R²<0次数合计为{_xneg['x2']}、{_xneg['x3']}、{_xneg['x4']}次，中位数较交叉验证选参依次{_change_range(_xmed['x2'])}、{_change_range(_xmed['x3'])}与{_change_range(_xmed['x4'])}")
cond('A8 PLSR 作杠杆回退后各档中位数变化不超过 0.02',
     max(abs(float(_f57[(_f57.method == 'PLSR') & (_f57.arm == 'lev') & (_f57.N == n)].R2.median()
                   - _f57[(_f57.method == 'PLSR') & (_f57.arm == 'cv') & (_f57.N == n)].R2.median())) for n in NS) < 0.02, '<0.02')
chk('2.3 一倍标准误与重复交叉验证的中位数变化', f"一倍标准误准则使中位数降低{rg(*_one_se_drop)}，重复交叉验证与单次交叉验证的中位数之差为{rg(*_repeat_delta)}")
chk('A8 选参端对策的判定', f"一倍标准误准则的中位数比交叉验证选参低{rg(*_one_se_drop)}，按判据为无效，重复交叉验证与之相差{rg(*_repeat_delta)}，按判据为有效")
cond('A8 清洗后交叉验证选参与两种做法在三档均无 R²<0', all(v == 0 for m in ('FPCR', 'FPLR') for a in ('cv', 'onese', 'repcv') for v in _neg(_d56, m, a)),
     '、'.join(f'{m}-{a} {_neg(_d56, m, a)}' for m in ('FPCR', 'FPLR') for a in ('cv', 'onese', 'repcv')))
for _m in ('FPCR', 'FPLR'):
    for _n in NS:
        chk(f'表A5 {_m} N={_n}', f"{_m}{_n}" + ''.join(f"{int((_cell(_d56, _m, a, _n) < 0).sum())}（{f(float(_cell(_d56, _m, a, _n).median()))}）"
                                                       for a in ('fixed', 'cv', 'onese', 'repcv')))
for _m in ('FPCR', 'FPLR', 'PLSR', 'SVR'):
    for _n in NS:
        _cells = ''.join('/' if (_m == 'SVR' and a == 'lev') else
                         f"{int((_cell(_f57, _m, a, _n) < 0).sum())}（{f(float(_cell(_f57, _m, a, _n).median()))}）" for a in ('cv', 'lev', 'clip'))
        _fb = '/' if _m == 'SVR' else _pct1(float(_f57[(_f57.method == _m) & (_f57.arm == 'lev') & (_f57.N == _n)].frac_backed.median()))
        chk(f'表A6 {_m} N={_n}', f"{_m}{_n}{_cells}{_fb}")
# 数据层面与主分析：从 05data 之外可得的最低层产物（57 的 dataset、mainfold 由脚本直接自原始光谱与 45 的逐样品预测算出）
_ds = pd.read_csv(f'{OUT}/57rebuttal_guard_dataset.csv').set_index('item').value
if not _sp.empty:
    chk('A8 逐点相同的光谱组', f"有{int(_ds['identical_spectrum_groups'])}组共{int(_ds['samples_in_identical_groups'])}个样品的原始光谱逐点相同而SSC不同")
    chk('A8 组内 SSC 极差', f"组内SSC极差的中位数为{float(_ds['identical_groups_ssc_absdiff_median']):.2f}°Brix、最大{float(_ds['identical_groups_ssc_absdiff_max']):.2f}°Brix")
_mainfold_ready = all(_fresh(path) for path in _REQUIRED_61)
if _mainfold_ready and HOT2:
    _sp61 = pd.read_csv(f'{OUT}/61rebuttal_mainfold_spectra.csv')
    _fo61 = pd.read_csv(f'{OUT}/61rebuttal_mainfold_folds.csv')
    cond('A8 61 的光谱即 2.2 弱折的 2 条', sorted({int(i) for i in _sp61.idx}) == HOT2, str(sorted({int(i) for i in _sp61.idx})))
    cond('A8 61 的逐折 R²（回退前）与 45 逐折一致', all(abs(float(r.R2) - float(FOLDS[r.method].set_index(['seed', 'fold']).R2.loc[(int(r.seed), int(r.fold))])) < 1e-9
                                              for r in _fo61.itertuples()), f'{len(_fo61)} 行')
    _fun61 = _sp61[_sp61.method.isin(['FPLR', 'FPCR'])]
    cond('A8 这 2 条光谱在 FPLR、FPCR 下每次都被回退', len(_fun61) == 20 and bool(_fun61.backed.astype(bool).all()), f'{len(_fun61)} 行')
    _wk61 = _fo61[(_fo61.seed == WEAK[0]) & (_fo61.fold == WEAK[1])].set_index('method')
    _is_wk = (_fo61.seed == WEAK[0]) & (_fo61.fold == WEAK[1])
    _dlev = {m: (_fo61[(_fo61.method == m) & ~_is_wk].R2_lev - _fo61[(_fo61.method == m) & ~_is_wk].R2) for m in ('FPLR', 'FPCR')}
    _ml = {m: (float(_fo61[_fo61.method == m].R2_lev.mean()), float(_fo61[_fo61.method == m].R2_lev.std(ddof=1))) for m in ('FPLR', 'FPCR', 'PLSR')}
    _fb61 = {m: float(_fo61[_fo61.method == m].frac_backed.median()) for m in ('FPLR', 'FPCR')}
    chk('A8 主分析杠杆回退', f"上述2条光谱在FPLR、FPCR下每次均被回退，绝对误差最大降至{float(_fun61.err_lev.abs().max()):.1f}°Brix，"
                         f"弱折的R²升至{_wk61.loc['FPLR', 'R2_lev']:.3f}与{_wk61.loc['FPCR', 'R2_lev']:.3f}")
    chk('A8 主分析回退的占比、其余各折与 25 折均值', f"被回退的测试光谱占比中位数为{_pct1(_fb61['FPLR'])}%与{_pct1(_fb61['FPCR'])}%，"
                                         f"其余各折R²的变化为{rg(_dlev['FPLR'].min(), _dlev['FPLR'].max())}与{rg(_dlev['FPCR'].min(), _dlev['FPCR'].max())}，"
                                         f"25折均值为{_ml['FPLR'][0]:.3f}±{_ml['FPLR'][1]:.3f}与{_ml['FPCR'][0]:.3f}±{_ml['FPCR'][1]:.3f}")
    _pl61 = _sp61[_sp61.method == 'PLSR']
    _ybar = float(_PRED['PLSR'].groupby('idx').y.first().mean())
    cond('A8 PLSR 下这 2 条光谱每次都被回退到 2 个潜变量、预测移向样品均值',
         len(_pl61) == 10 and bool(_pl61.backed.astype(bool).all()) and bool((_pl61.used_lev.astype(float) == 2).all())
         and all(abs(float(r.pred_lev) - _ybar) < abs(float(r.pred) - _ybar) for r in _pl61.itertuples()), f'样品均值 {_ybar:.2f}')
    chk('A8 PLSR 杠杆回退', f"PLSR同作回退后25折均值为{_ml['PLSR'][0]:.3f}：上述2条光谱每次均被回退到2个潜变量，预测移向样品均值")
    _raw = _sp61.drop_duplicates('idx').set_index('idx')
    _low_new = int(_raw.raw_max.idxmin())
    _oth_new = [i for i in HOT2 if i != _low_new]
    cond('2.2、A8 其一的最大读数为全体最低', abs(float(_raw.loc[_low_new, 'raw_max_quantile']) - 1 / _N_KEPT) < 1e-9, f'{_raw.loc[_low_new, "raw_max_quantile"]:.6f}')
    _pa, _pb = _pl61[_pl61.idx == _low_new], _pl61[_pl61.idx == _oth_new[0]]
    cond('A8 PLSR 原本对最大读数最低的一条预测较准', float(_pa.err.abs().max()) < float(_pb.err.abs().min()), f'{_pa.err.abs().max():.2f} < {_pb.err.abs().min():.2f}')
    chk('A8 PLSR 两条光谱误差变化', f"PLSR原本预测较准的最大读数最低的一条，绝对误差由{_pa.err.abs().min():.1f}~{_pa.err.abs().max():.1f}°Brix增至"
                               f"{_pa.err_lev.abs().min():.1f}~{_pa.err_lev.abs().max():.1f}°Brix，另一条由{_pb.err.abs().min():.1f}~{_pb.err.abs().max():.1f}°Brix"
                               f"降至{_pb.err_lev.abs().min():.1f}~{_pb.err_lev.abs().max():.1f}°Brix")
else:
    PENDING.append(('A8 主分析弱折的杠杆回退', '04outputs/61rebuttal_mainfold_{folds,spectra,summary}.csv'))

# ============================ 2.4 与附录 A4：跨批次（46 逐次抽样）============================
_dr = pd.read_csv(f'{OUT}/46rebuttal_recal_draws.csv')
_gq = _dr.groupby(['setting', 'method', 'k', 'strategy']).R2
Q = pd.DataFrame({'med': _gq.median(), 'lo': _gq.agg(lambda v: np.nanquantile(v, 0.025)),
                  'hi': _gq.agg(lambda v: np.nanquantile(v, 0.975)), 'n': _gq.count()})
assert (Q.n == 200).all()
_tails = _gq.agg(lambda v: (int((v < np.nanquantile(v, 0.025)).sum()), int((v > np.nanquantile(v, 0.975)).sum())))
cond('1.5 200 次抽样在 2.5%、97.5% 分位之外约各 5 次', all(min(t) >= 4 and max(t) <= 5 for t in _tails)
     and sum(t == (5, 5) for t in _tails) >= 0.99 * len(_tails),
     f'{sum(t == (5, 5) for t in _tails)}/{len(_tails)} 组两侧各 5 次，其余为并列值所致')
chk('1.5 200 次抽样', "各抽样200次，2.5%、97.5%分位之外约各5次")
M4 = ['PLSR', 'SVR', 'FPLR', 'FPCR']


def med(st: str, m: str, s: str, k: int) -> float:
    return float(Q.loc[(st, m, k, s), 'med'])


_src = pd.read_csv(f'{OUT}/46rebuttal_recal_source.csv').set_index(['setting', 'method'])
_hard = [float(_src.loc[(st, m), 'tgt_R2']) for st in ('2018→2019', '2019→2018', '留新疆') for m in M4]
_easy = [float(_src.loc[(st, m), 'tgt_R2']) for st in ('留山东', '留陕西') for m in M4]
chk('2.4 源批次内交叉验证 R²', f"源批次内交叉验证的R²为{rg(_src.src_cv_R2.min(), _src.src_cv_R2.max())}")
chk('2.4 直接使用', f"共12个组合中有{sum(v < 0 for v in _hard)}个R²为负，以山东、陕西为目标产地时为{rg(min(_easy), max(_easy))}")
_rng = pd.read_csv(f'{OUT}/49rebuttal_recal_range.csv')
chk('2.4 目标超出源范围的比例', f"样品仅占{pct(float(_rng.frac_outside.min()))}~{pct(float(_rng.frac_outside.max()))}")
_o = {(st, m): (float(_src.loc[(st, m), 'O1_R2']), float(_src.loc[(st, m), 'O2_R2'])) for st, m in _src.index}
cond('2.4 2018→2019 时 O2 与 O1 相差不足 0.01', max(abs(_o[('2018→2019', m)][1] - _o[('2018→2019', m)][0]) for m in M4) < 0.01, '<0.01')
_y1 = [med('2018→2019', m, 'S1', 10) for m in M4]
_g1 = [_o[('2018→2019', m)][0] - med('2018→2019', m, 'S1', 10) for m in M4]
chk('2.4 2018→2019 截距校正', f"R²中位数升至{rg(min(_y1), max(_y1))}，与O1相差{rg(min(_g1), max(_g1))}")
chk('摘要 2018→2019 截距校正', f"作截距校正后R²中位数为{rg(min(_y1), max(_y1))}")
chk('结论（3）2018→2019 截距校正与 O1 之差', f"k=10的截距校正与事后参照O1相差{rg(min(_g1), max(_g1))}")
chk('2.4 2018→2019 时 O2 与 O1', "2018→2019时，事后参照O2与O1相差不足0.01，即加估斜率几乎没有增益")
cond('2.4 2018→2019 k=10 另两种做法均更低', all(med('2018→2019', m, s, 10) < med('2018→2019', m, 'S1', 10) for m in M4 for s in ('S2', 'S3')), '逐项成立')
_o21 = [_o[('2019→2018', m)][1] - _o[('2019→2018', m)][0] for m in M4]
chk('2.4 2019→2018 O2 比 O1 高', f"O2比O1高{rg(min(_o21), max(_o21), 2)}")
cond('2.4 2019→2018 k≥10 斜率-截距校正均优于截距校正',
     all(med('2019→2018', m, 'S2', k) > med('2019→2018', m, 'S1', k) for m in M4 for k in (10, 20, 40)), '逐项成立')
_xo2 = [_o[('留新疆', m)][1] for m in M4]
_xlin = max(med('留新疆', m, s, k) for m in M4 for s in ('S1', 'S2') for k in (5, 10, 20, 40))
chk('2.4 新疆 O2', f"O2也仅为{rg(min(_xo2), max(_xo2))}")
chk('2.4 新疆线性校正上界', f"线性校正在k≤40内的R²中位数不超过{up(_xlin)}")
chk('摘要 新疆线性校正上界', f"以新疆为目标产地时，线性校正后不超过{up(_xlin)}")
_s3_40 = [med('留新疆', m, 'S3', 40) for m in M4]
chk('结论（3）新疆', f"以新疆为目标产地时线性校正后的R²中位数不超过{up(_xlin)}，k=40的加样重建使四种方法升至{min(_s3_40):.3f}~{max(_s3_40):.3f}。")
chk('Abstract 跨批次', f"intercept correction with 10 target samples gave median R² of {min(_y1):.3f}-{max(_y1):.3f}; "
                      f"with Xinjiang as the target area, linear corrections left it at or below {up(_xlin)}")
chk('2.4 新疆 k=40 加样重建', f"k=40的加样重建使四种方法升至{min(med('留新疆', m, 'S3', 40) for m in M4):.3f}~{max(med('留新疆', m, 'S3', 40) for m in M4):.3f}，以SVR最高")
_gain = max(med(st, m, s, k) - med(st, m, 'S0', k) for st in ('留山东', '留陕西') for m in M4 for s in ('S1', 'S2', 'S3') for k in (5, 10, 20, 40))
_loss = sum(med(st, m, 'S2', 5) < med(st, m, 'S0', 5) for st in ('留山东', '留陕西') for m in M4)
chk('2.4 山东、陕西', f"校正增益不超过{up(_gain)}，k=5的斜率-截距校正均低于不校正")
cond('2.4 k=5 的斜率-截距校正在山东、陕西的 8 个组合中均低于不校正', _loss == 8, f'{_loss} 个')
_y2_best = {m: max(med('2019→2018', m, s, 40) for s in ('S1', 'S2', 'S3')) for m in M4}
chk('2.4 2019→2018 k=40 函数型与离散模型比较',
    f"2019→2018时，k=40下FPLR、FPCR三种做法中最高的R²中位数为{_y2_best['FPLR']:.3f}与{_y2_best['FPCR']:.3f}，低于PLSR与SVR的{_y2_best['PLSR']:.3f}与{_y2_best['SVR']:.3f}")
_wd = Q[Q.index.get_level_values('strategy').isin(['S1', 'S2', 'S3'])].assign(w=lambda x: x.hi - x.lo).groupby(level='k').w.mean()
chk('A4 平均分位区间宽度', f"依次为{'、'.join(f'{_wd.loc[k]:.2f}' for k in (5, 10, 20, 40))}")
cond('A4 S1~S3 的平均分位区间宽度随 k 增大而收窄', all(_wd.loc[a] > _wd.loc[b] for a, b in ((5, 10), (10, 20), (20, 40))), '逐档递减')
for _st in ('2018→2019', '2019→2018', '留新疆', '留山东', '留陕西'):
    for _m in M4:
        chk(f'表A3 {_st} {_m}', _m + ''.join(f"{f(Q.loc[(_st, _m, 10, s), 'med'], 2)}[{f(Q.loc[(_st, _m, 10, s), 'lo'], 2)},"
                                            f"{f(Q.loc[(_st, _m, 10, s), 'hi'], 2)}]" for s in ('S1', 'S2', 'S3')))

# ============================ 2.5 与附录 A6：回归系数函数（48 逐波长）============================
_bt2 = pd.read_csv(f'{OUT}/48rebuttal_beta.csv')
_wl = _bt2.wavelength_nm.to_numpy(dtype=float)
_ab = np.abs(_bt2.beta_deploy.to_numpy(dtype=float))
_top = np.sort(_wl[np.argsort(-_ab)[:400]])
_segs = [s for s in np.split(_top, np.where(np.diff(_top) > 15)[0] + 1) if len(s) > 15]
_big = sorted(sorted(_segs, key=len, reverse=True)[:2], key=lambda s: float(s[0]))
cond('A7 窗口数', len(_segs) == 4, f'{len(_segs)} 个')
_rest = sorted(len(s) for s in _segs if all(s is not b for b in _big))
chk('A7 其余窗口的点数上界', f"2.5节所列两个以外的{len(_rest)}个均不超过{max(_rest)}个点")
chk('2.5 两个窗口', f"主要落在{_big[0][0]:.0f}~{_big[0][-1]:.0f}nm与{_big[1][0]:.0f}~{_big[1][-1]:.0f}nm两个窗口（各{len(_big[0])}与{len(_big[1])}个点）")
_st2 = _bt2.sign_stable.astype(bool).to_numpy()
_band = (_bt2.beta_q975 - _bt2.beta_q025).to_numpy(dtype=float)
_k1 = (_wl >= _big[0][0]) & (_wl <= _big[0][-1])
_k2 = (_wl >= _big[1][0]) & (_wl <= _big[1][-1])
_ko = ~(_k1 | _k2)


def ratio(mask: np.ndarray) -> float:
    return float(np.median(_band[mask] / np.maximum(_ab[mask], 1e-12)))


chk('2.5 分位带不含 0 的比例与带宽', f"2.5%~97.5%分位带在前一窗口范围内{pct(float(_st2[_k1].mean()))}的波长点不含零（两个窗口之外的波长为{pct(float(_st2[_ko].mean()))}），"
                            f"带宽中位数为|β|的{ratio(_k1):.2f}倍；后一窗口这一比例为{pct(float(_st2[_k2].mean()))}，"
                            f"但带宽约为|β|的{ratio(_k2):.2f}倍")
cond('2.5 sign_stable 一列即分位带不含 0', bool((((_bt2.beta_q025 > 0) | (_bt2.beta_q975 < 0)).to_numpy() == _st2).all()),
     f'{int(_st2.sum())}/{len(_st2)} 个波长点')
chk('A7 比例的分母', "2.5节分位带不含零的比例以窗口范围内的全部波长点为分母")
chk('A7 分母', f"{_big[0][0]:.0f}~{_big[0][-1]:.0f}nm窗口范围内共{int(_k1.sum())}个波长点")

# ============================ 2.1 节：各批次的 SSC 统计（表 1）与平均光谱（图 2）============================
# 表 1 由清洗清单逐条记录重算；平均光谱的描述量由图 2 的源数据（各批次的平均光谱）重算，不读 70 的汇总表
def hu_n(v: float, nd: int) -> str:
    """显式四舍五入（Python 的格式化逢五取偶）。"""
    return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)).replace('-', MINUS)


def sci_n(v: float) -> str:
    """a×10^b 在稿件文字里的样子：上标与正文相连，如 2.7×10−4。"""
    b = math.floor(math.log10(abs(v)))
    return f'{hu_n(v / 10 ** b, 1)}×10{str(b).replace("-", MINUS)}'


def lst_n(vals: list[str]) -> str:
    return '、'.join(vals[:-1]) + '与' + vals[-1]


_AREA_n = {1: '新疆', 2: '山东', 3: '陕西'}
_kp70 = pd.read_csv(_clean_path)
_kp70 = _kp70[_kp70.keep]
_bs_n = {(y, o): _kp70[(_kp70.year == y) & (_kp70.origin == o)].SSC for y in (2018, 2019) for o in (1, 2, 3)}
for (_y_n, _o_n), _v_n in _bs_n.items():
    chk(f'表1 {_y_n} {_AREA_n[_o_n]}', f"{_y_n}{_AREA_n[_o_n]}{len(_v_n)}{hu_n(_v_n.mean(), 2)}±{hu_n(_v_n.std(ddof=1), 2)}{hu_n(_v_n.min(), 1)}~{hu_n(_v_n.max(), 1)}")
chk('表1 合计', f"合计{len(_kp70)}{hu_n(_kp70.SSC.mean(), 2)}±{hu_n(_kp70.SSC.std(ddof=1), 2)}{hu_n(_kp70.SSC.min(), 1)}~{hu_n(_kp70.SSC.max(), 1)}")
_bm_n = {k: float(v.mean()) for k, v in _bs_n.items()}
_bsd_n = {k: float(v.std(ddof=1)) for k, v in _bs_n.items()}
_hi_n, _lo_n = max(_bm_n, key=lambda k: _bm_n[k]), min(_bm_n, key=lambda k: _bm_n[k])
chk('2.1 各批次 SSC 均值与标准差的范围', f"各批次的SSC均值为{hu_n(min(_bm_n.values()), 2)}~{hu_n(max(_bm_n.values()), 2)}°Brix，标准差为{hu_n(min(_bsd_n.values()), 2)}~"
    f"{hu_n(max(_bsd_n.values()), 2)}°Brix；{_hi_n[0]}年{_AREA_n[_hi_n[1]]}批次的均值最高，{_lo_n[0]}年{_AREA_n[_lo_n[1]]}批次最低")
_ms70 = pd.read_csv(f'{OUT}/70batch_mean_spectra.csv')
_wl70 = _ms70.wavelength_nm.to_numpy(dtype=float)
_raw70 = {(y, o): _ms70[f'raw_{y}_{_AREA_n[o]}'].to_numpy(dtype=float) for y in (2018, 2019) for o in (1, 2, 3)}
_pre70 = {(y, o): _ms70[f'prep_{y}_{_AREA_n[o]}'].to_numpy(dtype=float) for y in (2018, 2019) for o in (1, 2, 3)}
cond('2.1 平均光谱的波长点数', len(_wl70) == 2048, f'{len(_wl70)} 个')
_pk70 = [float(_wl70[v.argmax()]) for v in _raw70.values()]
_w70 = (_wl70 >= 650) & (_wl70 <= 700)
_vl70 = [float(_wl70[_w70][np.mean([_raw70[y, o] for o in (1, 2, 3)], axis=0)[_w70].argmin()]) for y in (2018, 2019)]
chk('2.1 峰位与局部极小', f"最大值位于{hu_n(min(_pk70), 0)}~{hu_n(max(_pk70), 0)}nm，{hu_n(float(np.mean(_vl70)), 0)}nm附近有一局部极小")
cond('2.1 两年的局部极小都在 650~700 nm 的内部且相差不到 1 nm', all(650 < v < 700 for v in _vl70) and abs(_vl70[0] - _vl70[1]) < 1, f'{_vl70}')
cond('2.1 900 nm 以上强度较低（各批次平均光谱在 900 nm 以上的最大值不到其峰值的 1/5）',
     all(v[_wl70 > 900].max() < v.max() / 5 for v in _raw70.values()), f"{max(float(v[_wl70 > 900].max() / v.max()) for v in _raw70.values()):.3f}")
_rat70 = [float(_raw70[2019, o].mean() / _raw70[2018, o].mean()) for o in (1, 2, 3)]
chk('2.1 两年平均强度之比', f"同一产地2019年的全波段平均强度为2018年的{hu_n(min(_rat70), 1)}~{hu_n(max(_rat70), 1)}倍")


def _rms_n(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def _rel_n(M: dict[tuple[int, int], np.ndarray]) -> tuple[list[float], list[float], list[float]]:
    """相对差异：两个批次平均光谱之差的均方根 ÷ 六条批次平均光谱均方根的平均值。返回（年份之间、新疆对另两个产地、山东对陕西）。"""
    sc = float(np.mean([_rms_n(v) for v in M.values()]))
    return ([_rms_n(M[2019, o] - M[2018, o]) / sc for o in (1, 2, 3)], [_rms_n(M[y, 1] - M[y, b]) / sc for y in (2018, 2019) for b in (2, 3)],
            [_rms_n(M[y, 2] - M[y, 3]) / sc for y in (2018, 2019)])


def _r2_n(v: list[float]) -> str:
    return f'{hu_n(min(v), 2)}~{hu_n(max(v), 2)}'


_r0_n, _r1_n = _rel_n(_raw70), _rel_n(_pre70)
chk('2.1 预处理前的相对差异', f"同一产地两个年份之间的相对差异为{_r2_n(_r0_n[0])}，同一年份新疆与另两个产地之间为{_r2_n(_r0_n[1])}，山东与陕西之间为{_r2_n(_r0_n[2])}")
chk('2.1 预处理后的相对差异', f"年份之间降至{_r2_n(_r1_n[0])}，山东与陕西之间降至{_r2_n(_r1_n[2])}，而新疆与另两个产地之间为{_r2_n(_r1_n[1])}")
cond('2.1 预处理后山东与陕西之间最小、新疆与另两个产地之间最大（三组的取值范围互不重叠）',
     max(_r1_n[2]) < min(_r1_n[0]) and max(_r1_n[0]) < min(_r1_n[1]), f'{_r2_n(_r1_n[2])} < {_r2_n(_r1_n[0])} < {_r2_n(_r1_n[1])}')
cond('2.1 预处理后年份之间、山东与陕西之间的相对差异都比预处理前小', max(_r1_n[0]) < min(_r0_n[0]) and max(_r1_n[2]) < min(_r0_n[2]), '两组的最大值均小于预处理前的最小值')

# ============================ 2.5 节与附录 A7：预测值按波段的分解（71 的逐样品、逐波长产物）============================
# 标准差、方差与协方差均取样本口径（ddof=1），与表 1、表 2 的标准差相同
_ps_n = pd.read_csv(f'{OUT}/71beta_contribution_samples.csv')
_pw_n = pd.read_csv(f'{OUT}/71beta_contribution_pointwise.csv')
_tot_n = _ps_n.prediction_minus_ybar.to_numpy(dtype=float)
_bc_n = [c for c in _ps_n.columns if c.startswith('beta_') and c != 'beta_rest']
_cc_n = [c for c in _ps_n.columns if c.startswith('contrib_') and c != 'contrib_rest']
cond('A7 样品数与两种取法的波段数', len(_ps_n) == 1111 and len(_bc_n) == 4 == len(_cc_n), f'{len(_ps_n)} 个样品，{len(_bc_n)}、{len(_cc_n)} 个波段')
cond('A7 两种取法下各波段贡献之和都等于预测值减训练集均值',
     max(float(np.abs(_ps_n[[*_bc_n, 'beta_rest']].sum(1) - _tot_n).max()), float(np.abs(_ps_n[[*_cc_n, 'contrib_rest']].sum(1) - _tot_n).max())) < 1e-9, '最大绝对差 < 1e-9')


def _bname_n(c: str) -> str:
    lo, hi = c.split('_')[1:]
    return f'{hu_n(float(lo), 0)}~{hu_n(float(hi), 0)}'


def _sd_n(c: str) -> float:
    return float(_ps_n[c].std(ddof=1))


def _share_n(c: str) -> float:
    return float(np.cov(_ps_n[c], _tot_n)[0, 1] / _tot_n.var(ddof=1))


def _sdx_n(c: str) -> float:
    lo, hi = (float(x) for x in c.split('_')[1:])
    return float(_pw_n.sd_x[(_pw_n.wavelength_nm >= lo) & (_pw_n.wavelength_nm <= hi)].median())


# 按 |β| 划出的窗口须与 48 逐波长产物重算的一致；按逐点贡献标准差划出的波段由 71 的逐波长产物按同一规则重算
cond('A7 按 |β| 划出的 4 个窗口与 48 逐波长产物重算的一致',
     [(_bname_n(c)) for c in _bc_n] == [f'{hu_n(float(sg[0]), 0)}~{hu_n(float(sg[-1]), 0)}' for sg in sorted(_segs, key=lambda sg: float(sg[0]))], str([_bname_n(c) for c in _bc_n]))
_ctop_n = np.sort(_pw_n.wavelength_nm.to_numpy(dtype=float)[np.argsort(-_pw_n.sd_contribution.to_numpy(dtype=float))[:400]])
_csegs_n = [sg for sg in np.split(_ctop_n, np.where(np.diff(_ctop_n) > 15)[0] + 1) if len(sg) > 15]
cond('A7 按逐点贡献标准差划出的 4 个波段与逐波长产物重算的一致，且含全部 400 个点',
     [_bname_n(c) for c in _cc_n] == [f'{hu_n(float(sg[0]), 0)}~{hu_n(float(sg[-1]), 0)}' for sg in _csegs_n] and sum(len(sg) for sg in _csegs_n) == 400, str([_bname_n(c) for c in _cc_n]))
_w1_n, _w2_n = _bc_n[0], _bc_n[2]
cond('2.5 正文所列两个窗口即 868~955 与 1079~1115 nm', (_bname_n(_w1_n), _bname_n(_w2_n)) == (f'{_big[0][0]:.0f}~{_big[0][-1]:.0f}', f'{_big[1][0]:.0f}~{_big[1][-1]:.0f}'),
     f'{_bname_n(_w1_n)}、{_bname_n(_w2_n)}')
_corr1_n = float(np.corrcoef(_ps_n[_w1_n], _tot_n - _ps_n[_w1_n])[0, 1])
chk('2.5 光谱函数的样品间标准差', f"在前一窗口为{sci_n(_sdx_n(_w1_n))}、在后一窗口为{sci_n(_sdx_n(_w2_n))}，而在{_bname_n(_cc_n[0])}nm为{sci_n(_sdx_n(_cc_n[0]))}")
chk('2.5 前一窗口的贡献', f"前一窗口贡献的标准差为{hu_n(_sd_n(_w1_n), 2)}°Brix，而预测值本身的标准差为{hu_n(float(_tot_n.std(ddof=1)), 2)}°Brix；该贡献与预测值的协方差为预测值方差的"
    f"{hu_n(_share_n(_w1_n), 3)}倍，与其余全部波长贡献之和的相关系数为{hu_n(_corr1_n, 2)}")
chk('2.5 后一窗口的贡献', f"后一窗口贡献的标准差仅为{hu_n(_sd_n(_w2_n), 2)}°Brix")
chk('2.5 按贡献排序的波段', f"最大的400个波长点落在{lst_n([_bname_n(c) for c in _cc_n])}nm四个波段，其贡献与预测值的协方差依次为预测值方差的{lst_n([hu_n(_share_n(c), 2) for c in _cc_n])}倍")
chk('A7 按 |β| 的 4 个窗口', f"（{lst_n([_bname_n(c) for c in _bc_n])}nm）的标准差依次为{lst_n([hu_n(_sd_n(c), 2) for c in _bc_n])}°Brix，前三个的比例为{lst_n([hu_n(_share_n(c), 3) for c in _bc_n[:3]])}，"
    f"第四个的绝对值小于0.001；窗口以外的波长为{hu_n(_sd_n('beta_rest'), 2)}°Brix与{hu_n(_share_n('beta_rest'), 3)}")
cond('A7 第四个窗口的比例绝对值小于 0.001', abs(_share_n(_bc_n[3])) < 0.001, f'{_share_n(_bc_n[3]):.5f}')
chk('A7 前一窗口与其余波长贡献的相关', f"868~955nm窗口的贡献与其余全部波长贡献之和的相关系数为{hu_n(_corr1_n, 2)}")
chk('A7 按贡献的 4 个波段', f"得4个波段：{lst_n([_bname_n(c) for c in _cc_n])}nm，标准差依次为{lst_n([hu_n(_sd_n(c), 2) for c in _cc_n])}°Brix，比例为{lst_n([hu_n(_share_n(c), 2) for c in _cc_n])}；"
    f"其余波长为{hu_n(_sd_n('contrib_rest'), 2)}°Brix与{hu_n(_share_n('contrib_rest'), 2)}")
chk('A7 光谱函数的样品间标准差', f"在868~955nm窗口为{sci_n(_sdx_n(_w1_n))}，在1079~1115nm窗口为{sci_n(_sdx_n(_w2_n))}，在{_bname_n(_cc_n[0])}nm为{sci_n(_sdx_n(_cc_n[0]))}")
cond('A7 比例之和为 1（两种取法）', abs(sum(_share_n(c) for c in [*_bc_n, 'beta_rest']) - 1) < 1e-9 and abs(sum(_share_n(c) for c in [*_cc_n, 'contrib_rest']) - 1) < 1e-9, '两种取法均为 1')
_log71 = sorted(glob.glob(f'{LOGS}/71beta_contribution_*.log'))
_gap71 = re.search(r'最大绝对差：([0-9.e+-]+)', Path(_log71[-1]).read_text(encoding='utf-8')) if _log71 else None
cond('A7 逐点分解与部署模型预测值之差不超过 1e-13（71 的运行日志）', _gap71 is not None and float(_gap71.group(1)) <= 1e-13,
     f'{_gap71.group(1) if _gap71 else "日志缺失"}（{Path(_log71[-1]).name if _log71 else ""}）')
chk('结论（4）两个窗口', f"FPLR回归系数函数的绝对值在{_bname_n(_w1_n)}nm与{_bname_n(_w2_n)}nm两个窗口最大")
cond('结论（4）仅前一窗口对预测值有明显影响：后一窗口贡献的标准差不到预测值标准差的 1/10，前一窗口的大于预测值的标准差',
     _sd_n(_w2_n) < float(_tot_n.std(ddof=1)) / 10 < float(_tot_n.std(ddof=1)) < _sd_n(_w1_n), f'{_sd_n(_w2_n):.3f}、{float(_tot_n.std(ddof=1)):.3f}、{_sd_n(_w1_n):.3f}')
# 1.3.2 节：截距与两个数值稳定项，核对建模代码的原文
with open('02code/16unified_comparison.py', encoding='utf-8') as _fh16:
    _src16 = _fh16.read()
cond('1.3.2 R 上加 1e-8 倍单位矩阵、Z′Z+λβsR 上加 1e-12 倍单位矩阵、截距取训练集均值（16unified_comparison.py）',
     'return J, R+1e-8*np.eye(B.shape[1])' in _src16 and 'np.linalg.solve(ZtZ+lam*sc*R+1e-12*np.eye(len(R)), Z.T@(ytr-ym))' in _src16, '代码原文两处')
chk('1.3.2 数值稳定项', f"求解前在R上加10{MINUS}8倍单位矩阵，在Z′Z+λβsR上加10{MINUS}12倍单位矩阵")

# ============================ 事后分析的标注（协议 C3：主分析弱折的剖析）============================
chk('摘要 事后剖析', "事后剖析表明其折间波动主要来自2条光谱上的外推")
chk('Abstract post hoc', "a post hoc analysis traced its spread across folds mainly to extrapolation on two spectra")
chk('结论（1）事后描述', "函数型模型的折间波动主要来自2条光谱上的外推（事后描述）")
chk('结论（2）事后描述', "清洗后主分析的弱折也来自其中2条（事后描述）")

# ============================ 汇总 ============================
print(f'稿件：{"未读取（R42_NO_DOCX=1）" if NO_DOCX else DOCX}')
print(f'已完成方法 {DONE}；未落地、本轮跳过 {PEND}；主分析族 {len(PAIRS)} 对')
print(f'{"复算项":44s}  在稿中  稿件应有的片段或应成立的模式')
print('-' * 110)
bad = [(n, lit) for n, lit, hit, _kind in rows if not hit]
for name, lit, hit, kind in rows:
    status = '待核' if NO_DOCX and kind == 'chk' else ('✓' if hit else '★')
    print(f'{name:44s}  {status:^4s}  {lit}')
print(f'\n复算 {len(rows)} 项，其中 {len(bad)} 项在稿件中找不到或模式不成立')
for name, lit in bad:
    print(f'  ★ {name}: {lit}')
if PENDING:
    print('\n待产物')
    for name, path in PENDING:
        print(f'  - {name}：{path}')
exit_code = 1 if bad or PENDING else 0
print(f'退出码 {exit_code}')
if exit_code:
    sys.exit(1)
