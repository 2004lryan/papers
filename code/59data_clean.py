# -*- coding: utf-8 -*-
"""
59data_clean.py: 按协议 B10.10 的四条规则清洗两个年份的原始记录，输出逐行剔除清单、汇总与清洗前的对照量

为什么要有这个脚本：清洗规则在 data_clean.py 里只有一处实现，16 号共享段与 08 号文件头读数时直接调用；
本脚本把每一行命中了哪条规则、R1 距离、清洗前后的样品编号落成清单，供稿件 2.1 节与附录引用，
并按登记值逐项断言（剔除数、重合数、保留数与产地构成），任何一项对不上即中止、不写产物。
清洗前的对照量（协议 B10.10「报告」一段所列四项）从提交 2331884 的产物读取，那是 2026-09-21 定稿时的结果。

运行方式:
    cd <项目根目录>
    python3 02code/59data_clean.py > 05logs/59data_clean_<YY-MM-DD_HHMMSS>.log 2>&1

输出文件:
    04outputs/59data_clean_list.csv       — 逐行：年份、Excel 数据行号、清洗前编号、清洗后编号、SSC、产地、R1~R4、R1 距离、是否保留、
                                            达到检测器上限的波长点数、按 015 的波长范围（590.67~1 001.12 nm）判定的 R1（只作对照）
    04outputs/59data_clean_summary.csv    — 各年份各规则的命中数、重合数、并集、保留数与产地构成、SSC 统计
    04outputs/59raw_reference.csv         — 清洗前的对照量（表 1 R² 均值、15 对差值与 Holm 判定、小样本失效次数、跨批次截距校正、
                                            选参端与预测端各对策臂在 N=50、80、120 的 R²<0 次数与中位数）
    05logs/59data_clean_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

只读 05data。
"""
import io
import os
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, '02code')
import data_clean  # noqa: E402

D = os.environ.get('APPLE_SSC_DATA', '../../05data/001_apple_hyperspectral_multiyear')   # 数据不随代码分发，见 08github/data/DATA.md
FILES = {2018: '01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx', 2019: '1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx'}
ORIGIN = {1: '新疆', 2: '山东', 3: '陕西'}
RAW_COMMIT = '2331884'          # 2026-09-21 定稿时的产物
OLD_HOT = (71, 115, 125, 126, 131, 223)
# 协议 B10.10 登记的剔除结果；任何一项对不上即中止
EXPECT: dict[str, Any] = {'R1': {2018: 63, 2019: 67}, 'R2': {2018: 1, 2019: 0}, 'R3': {2018: 37, 2019: 32}, 'R3_groups': {2018: 18, 2019: 16},
          'R4': {2018: 2, 2019: 0}, 'union': 190, 'keep': {2018: 537, 2019: 574},
          'origin': {2018: {1: 161, 2: 181, 3: 195}, 2019: {1: 203, 2: 176, 3: 195}}}


def raw(f):
    """与 16、08 两处 load() 相同的读法，但不做任何筛选；另返回波长（首行）。"""
    d = pd.read_excel(f'{D}/{f}', header=None)
    wl = d.iloc[0, :2048].astype(float).values
    X = d.iloc[2:, :2048].astype(float).values
    y = pd.to_numeric(d.iloc[2:, 2048], errors='coerce').values
    z = pd.to_numeric(d.iloc[2:, 2049], errors='coerce').values
    return X, y, z, wl


def git_csv(path):
    r = subprocess.run(['git', 'show', f'{RAW_COMMIT}:{path}'], capture_output=True, check=True)
    return pd.read_csv(io.StringIO(r.stdout.decode('utf-8-sig')))


def main():
    rows, spectra = [], []
    for yr, f in FILES.items():
        X, y, z, wl = raw(f)
        fl, dist = data_clean.rule_flags(X, y, z)
        # 对照：R1 只用 015 的波长范围（590.67~1 001.12 nm）时的判定，只进清单与汇总，不参与清洗（协议 B10.10 差异①）
        band = (wl >= 590.67) & (wl <= 1001.12)
        fl015, _ = data_clean.rule_flags(X[:, band], y, z)
        old_ok = np.isfinite(y) & np.isfinite(z) & (y > 5) & (y < 25) & np.isfinite(X).all(1)   # 清洗前两处 load() 的筛选
        keep = data_clean.keep_mask(X, y, z)
        assert (keep == (fl['valid'] & ~(fl['R1'] | fl['R2'] | fl['R3'] | fl['R4']))).all()
        groups: dict[bytes, list[int]] = {}
        for i in np.flatnonzero(fl['valid']):
            groups.setdefault(X[i].tobytes(), []).append(int(i))
        gid = {i: k for k, v in enumerate(g for g in groups.values() if len(g) > 1) for i in v}
        # 对照：015 的 R3 在其波长范围（590.67~1 001.12 nm）上按保留 10 位小数比较；本稿按全部 2 048 点逐位比较。
        # 两种比较在本数据上命中同一批记录（只进断言与日志，不参与清洗）
        for tag, M in (('015 波段逐位', X[:, band]), ('015 波段保留 10 位小数', np.round(X[:, band], 10))):
            g015: dict[bytes, list[int]] = {}
            for i in np.flatnonzero(fl['valid']):
                g015.setdefault(M[i].tobytes(), []).append(int(i))
            hit015 = sorted(i for v in g015.values() if len(v) > 1 for i in v)
            assert hit015 == sorted(np.flatnonzero(fl['R3']).tolist()), (yr, tag)
            print(f'{yr} 年 R3 对照（{tag}）：{sum(len(v) > 1 for v in g015.values())} 组 {len(hit015)} 个，与全波段逐位比较的命中相同')
        for i in range(len(y)):
            rows.append({'year': yr, 'row': i, 'old_ok': bool(old_ok[i]), 'keep': bool(keep[i]), 'SSC': y[i], 'origin': int(z[i]),
                         **{r: bool(fl[r][i]) for r in data_clean.RULES}, 'R1_dist': dist[i],
                         'n_adc_max': int((X[i] >= data_clean.ADC_MAX).sum()), 'R1_015band': bool(fl015['R1'][i]),
                         'dup_group': f'{yr}-{gid[i]}' if i in gid else ''})
            spectra.append(X[i])
    df = pd.DataFrame(rows)
    df.insert(2, 'old_idx', np.where(df.old_ok, df.old_ok.cumsum() - 1, -1))
    df.insert(3, 'new_idx', np.where(df.keep, df.keep.cumsum() - 1, -1))
    assert int(df.old_ok.sum()) == 1300
    assert (df.old_idx[df.old_ok] == np.arange(1300)).all()

    # 跨年份不应有逐位相同的光谱（R3 按年份判定的前提）
    keys = pd.Series([s.tobytes() for s in spectra])
    cross = df.assign(k=keys).groupby('k').year.nunique()
    assert int((cross > 1).sum()) == 0, '存在跨年份逐位相同的光谱'

    # ---- 按登记值逐项核对 ----
    for yr in FILES:
        q = df[df.year == yr]
        for r in data_clean.RULES:
            assert int(q[r].sum()) == EXPECT[r][yr], (yr, r, int(q[r].sum()))
        assert q.dup_group[q.dup_group != ''].nunique() == EXPECT['R3_groups'][yr]
        assert int(q.keep.sum()) == EXPECT['keep'][yr]
        assert q[q.keep].origin.value_counts().sort_index().to_dict() == EXPECT['origin'][yr]
    anyrule = df[list(data_clean.RULES)].any(axis=1)
    assert int(anyrule.sum()) == EXPECT['union']
    assert int((~df.keep).sum()) == EXPECT['union']
    assert int((df.R1 & df.R3).sum()) == 10
    assert int((df.R4 & ~df.R1).sum()) == 0
    assert int((df.R2 & (df.R1 | df.R3 | df.R4)).sum()) == 0

    df.to_csv('04outputs/59data_clean_list.csv', index=False, encoding='utf-8-sig')

    # ---- 汇总 ----
    summ = []
    for label in (2018, 2019, '合计'):
        q = df if label == '合计' else df[df.year == label]
        rec = {'year': label, 'raw': len(q)}
        rec.update({r: int(q[r].sum()) for r in data_clean.RULES})
        rec.update({'R3_groups': int(q.dup_group[q.dup_group != ''].nunique()), 'R1_and_R3': int((q.R1 & q.R3).sum()),
                    'R1_and_R4': int((q.R1 & q.R4).sum()), 'removed': int((~q.keep).sum()), 'kept': int(q.keep.sum())})
        rec.update({f'kept_{ORIGIN[o]}': int((q.keep & (q.origin == o)).sum()) for o in ORIGIN})
        rec.update({'R1_015band': int(q.R1_015band.sum()), 'R1_and_R1_015band': int((q.R1 & q.R1_015band).sum())})
        for tag, m in (('before', q.old_ok), ('after', q.keep)):
            s = q.SSC[m]
            rec.update({f'SSC_mean_{tag}': s.mean(), f'SSC_sd_{tag}': s.std(ddof=1), f'SSC_min_{tag}': s.min(), f'SSC_max_{tag}': s.max(),
                        f'n_{tag}': int(m.sum())})
        summ.append(rec)
    sm = pd.DataFrame(summ)
    sm.to_csv('04outputs/59data_clean_summary.csv', index=False, encoding='utf-8-sig')

    # ---- 清洗前的对照量（提交 2331884 的产物）----
    ref = []
    t1 = git_csv('04outputs/47rebuttal_table1.csv')
    for _, r in t1[t1.analysis == '主分析（B 样条）'].iterrows():
        ref.append({'item': '表1_R2均值', 'key': r.method, 'key2': '', 'value': r.R2, 'extra': r.R2_sd})
    pt = git_csv('04outputs/47rebuttal_paired_tests.csv')
    for _, r in pt[pt.family == '主分析'].iterrows():
        ref.append({'item': '六法15对', 'key': r.pair, 'key2': r['结论'], 'value': r['diff'], 'extra': r.holm_p})
    ss = git_csv('04outputs/49rebuttal_smallsample_summary.csv')
    for _, r in ss[ss.N.isin([50, 80, 120]) & (ss.strategy == 'cv')].iterrows():
        ref.append({'item': '小样本_cv_R2负值次数', 'key': r.method, 'key2': int(r.N), 'value': r.n_neg, 'extra': r['median']})
    rc = git_csv('04outputs/46rebuttal_recal_summary.csv')
    for _, r in rc[(rc.setting == '2018→2019') & (rc.k == 10) & (rc.strategy == 'S1')].iterrows():
        ref.append({'item': '跨批次_2018→2019_k10_S1', 'key': r.method, 'key2': '', 'value': r.R2_med, 'extra': r.O1_R2})
    # 附录 A9 列出各对策臂判定的变化：B10.8（56，随机种子 20~119）与 B10.9（57，随机种子 120~219）的逐档 R²<0 次数与中位数
    for item, path, seed_set, arms in (('选参端对策', '04outputs/56rebuttal_selection_summary.csv', '新100次', ('cv', 'onese', 'repcv')),
                                       ('预测端对策', '04outputs/57rebuttal_guard_summary.csv', '再取100次', ('cv', 'lev', 'clip'))):
        g = git_csv(path)
        q = g[(g.seed_set == seed_set) & g.method.isin(['FPCR', 'FPLR']) & g.N.isin([50, 80, 120]) & g.arm.isin(arms)]
        assert len(q) == 2 * len(arms) * 3, (item, len(q))
        assert (q.n == 100).all(), item
        for _, r in q.iterrows():
            ref.append({'item': item, 'key': f'{r.method}|{r.arm}', 'key2': int(r.N), 'value': r.n_neg, 'extra': r['median']})
    # 附录 A9 列出 GBIDS 判定的变化：附录 A5 的 15 对（55 的配对表）
    a5 = git_csv('04outputs/55rebuttal_A5_pairs.csv')
    assert len(a5) == 15, len(a5)
    for _, r in a5.iterrows():
        ref.append({'item': 'A5_GBIDS15对', 'key': r.pair, 'key2': r['结论'], 'value': r['diff'], 'extra': r.holm_p})
    rf = pd.DataFrame(ref)
    assert len(rf[rf.item == '六法15对']) == 15
    assert len(rf[rf.item == '表1_R2均值']) == 6
    rf.to_csv('04outputs/59raw_reference.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 220)
    print('协议 B10.10 数据清洗：四条规则在原始记录上各自判定，剔除其并集')
    print(sm.T.to_string(header=False))
    print('\n清洗前编号的 6 条光谱（10.9 名单）：')
    print(df[df.old_idx.isin(OLD_HOT)][['year', 'row', 'old_idx', 'new_idx', 'SSC', 'origin', 'R1', 'R2', 'R3', 'R4', 'R1_dist']].to_string(index=False))
    print('\nR2 命中：')
    print(df[df.R2][['year', 'row', 'old_idx', 'SSC', 'origin', 'R1_dist']].to_string(index=False))
    print('\nR4 命中：')
    print(df[df.R4][['year', 'row', 'old_idx', 'SSC', 'origin', 'R1', 'R1_dist']].to_string(index=False))
    print(f"\n清洗前对照量（提交 {RAW_COMMIT}）：{len(rf)} 项，写入 04outputs/59raw_reference.csv")
    print(rf.to_string(index=False))


if __name__ == '__main__':
    main()
