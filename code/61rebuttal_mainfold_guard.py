# -*- coding: utf-8 -*-
"""
61rebuttal_mainfold_guard.py: 主分析中函数型模型弱折的逐样品剖析与杠杆回退的事后应用（协议 C3 2026-09-25 登记，事后描述）

为什么要有这个脚本：清洗后（协议 B10.10）FPLR、FPCR 的 25 折 R² 标准差约为 SVR 的两倍，来自 seed 20041210 fold 0
这一折（两者 R² 为 0.648 与 0.640，同折 SVR 为 0.810）；该折误差最大的两个测试样品，正是 B10.9 清洗前名单 6 条光谱中
未被清洗规则剔除的 2 条。清洗前的小样本剖析认为函数型模型在训练集未覆盖的异常光谱上外推失控，这里在清洗后的主分析
（每个外层训练折约 888 个样品）上作同样的剖析，并把 B10.9 的杠杆回退原样事后应用到外层测试折，看它能否识别并纠正
这两条光谱、对其余光谱有无代价。只作描述，不改变主分析的比较、检验与 Holm 族。

做法：
  剖析     六法在 25 个外层折上对名单光谱的预测误差；评价时除去名单光谱后各法 25 折 R² 的均值与标准差；
           各法最弱一折中误差最大的 5 个测试样品及其清洗前序号
  杠杆回退 在 45 的 FPLR、FPCR（三次 B 样条基，L=80）与 PLSR 各外层折选中的配置上，对外层测试折的每条光谱计算
           杠杆值，超过训练集平均杠杆值的 3 倍时沿第二轮 12 组候选逐级回退到更强的正则化（B10.9 的规则与实现，
           函数取自 02code/57rebuttal_smallsample_guard.py 原文）
  名单     B10.9 清洗前名单（57 脚本的 HOT_PRE_CLEAN）中被清洗保留者，经 04outputs/59data_clean_list.csv 换成清洗后序号

复现门：选中配置的预测与 45 的逐样品预测之差不超过 1e-9，任一不符即中止，不写产物。
RF、1D-CNN 的逐样品预测在服务器上算出后取回；未齐时设 MF_ALLOW_PARTIAL=1 只打印已有方法的结果、不写产物。

运行方式:
    cd <项目根目录>
    python3 02code/61rebuttal_mainfold_guard.py > 05logs/61rebuttal_mainfold_guard_<时间戳>.log 2>&1

输出文件:
    04outputs/61rebuttal_mainfold_folds.csv    — 逐方法、逐外层折：选中配置、R²（原预测与杠杆回退后）、回退光谱数与占比、名单光谱在测试折中的条数与被回退的条数
    04outputs/61rebuttal_mainfold_spectra.csv  — 名单光谱每次落入测试折：六法的预测值与误差，三法回退后的预测值、误差与所用配置
    04outputs/61rebuttal_mainfold_summary.csv  — 各方法 25 折 R² 的均值与标准差（原预测、评价时除去名单光谱、杠杆回退后），最弱一折及其误差最大的 5 个测试样品
    05logs/61rebuttal_mainfold_guard_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

只读 05data。
"""
import ast
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
sys.path.insert(0, '02code')
import ncv_manifest as nm  # noqa: E402
from grids import GRIDS  # noqa: E402

OUT = '04outputs'
K_OUTER, L_BASIS = 5, 80
SEEDS = [20041210, 20240404, 20060515, 20251203, 19810915]
METHODS6 = ('SVR', 'FPLR', 'FPCR', 'PLSR', '1D-CNN', 'RF')
GUARDED = ('FPCR', 'FPLR', 'PLSR')
PARTIAL = os.environ.get('MF_ALLOW_PARTIAL', '0') == '1'
GATE_TOL = 1e-9


def load57():
    """B10.9 的杠杆回退函数取 57 脚本原文（模块导入，不运行其 main）。"""
    spec = importlib.util.spec_from_file_location('guard57', '02code/57rebuttal_smallsample_guard.py')
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pred_path(meth):
    tag = 'bspline' if meth.startswith('FP') else 'none'
    return Path(OUT) / f"45rebuttal_ncv_{tag}_{meth.replace('-', '')}_pred.csv"


def fold_r2(g):
    return 1 - ((g.pred - g.y) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum()


def main():
    t0 = time.time()
    g57 = load57()
    ns: dict = {}
    exec(nm.shared16_text(), ns)                       # 数据（含 B10.10 清洗）、预处理、基函数与估计器，同 45
    y, Xp = ns['y'], ns['Xp']
    _B, J, R, C = nm.basis_system(ns, 'bspline', L_BASIS)
    M = np.linalg.cholesky(J + 1e-10 * np.eye(len(J)))
    n = len(y)

    # ---- 名单：B10.9 清洗前 6 条中被清洗保留者，换成清洗后序号 ----
    lst = pd.read_csv(f'{OUT}/59data_clean_list.csv')
    lst = lst[lst.old_ok]
    hit = lst[lst.old_idx.isin(g57.HOT_PRE_CLEAN)]
    assert len(hit) == len(g57.HOT_PRE_CLEAN), hit
    kept = hit[hit.keep]
    HOT = tuple(sorted(int(i) for i in kept.new_idx))
    old_of = dict(zip(lst[lst.keep].new_idx.astype(int), lst[lst.keep].old_idx.astype(int), strict=True))
    assert len(old_of) == n, (len(old_of), n)
    print(f'样品数 {n}；B10.9 清洗前名单 {list(g57.HOT_PRE_CLEAN)} 中被清洗保留 {len(HOT)} 条：'
          f'清洗后序号 {list(HOT)}（清洗前 {[old_of[i] for i in HOT]}）', flush=True)

    folds = {sd: list(KFold(n_splits=K_OUTER, shuffle=True, random_state=sd).split(np.arange(n))) for sd in SEEDS}

    # ---- 六法的逐样品预测（45 产物）：序号与真值须与本次读入的清洗后数据一致 ----
    preds: dict[str, pd.DataFrame] = {}
    for meth in METHODS6:
        p = pred_path(meth)
        d = pd.read_csv(p) if p.exists() else None
        ok = d is not None and len(d) == len(SEEDS) * n and d.idx.max() < n and \
            float(np.abs(d.y.to_numpy() - y[d.idx.to_numpy()]).max()) < 1e-9
        if not ok:
            msg = f'{meth}：{p} 缺失或与清洗后数据不一致（行数、序号或真值不符）'
            if PARTIAL:
                print(f'  跳过 {msg}', flush=True)
                continue
            raise SystemExit(msg)
        assert d is not None
        for sd in SEEDS:
            for fold, (_itr, ite) in enumerate(folds[sd]):
                got = np.sort(d[(d.seed == sd) & (d.fold == fold)].idx.to_numpy())
                assert np.array_equal(got, np.sort(ite)), (meth, sd, fold)
        d['e'] = d.pred - d.y
        preds[meth] = d
    missing = [m for m in METHODS6 if m not in preds]
    if missing and not PARTIAL:
        raise SystemExit(f'缺少 {missing} 的逐样品预测')

    # ---- 杠杆回退（B10.9 的规则），先过复现门 ----
    frows, gpred = [], {}
    for meth in GUARDED:
        if meth not in preds:
            continue
        tag = 'bspline' if meth.startswith('FP') else 'none'
        res = pd.read_csv(f"{OUT}/45rebuttal_ncv_{tag}_{meth}.csv")
        d = preds[meth]
        gap, out = 0.0, []
        for sd in SEEDS:
            for fold, (itr, ite) in enumerate(folds[sd]):
                row = res[(res.seed == sd) & (res.fold == fold)]
                assert len(row) == 1, (meth, sd, fold)
                picked = ast.literal_eval(str(row.picked.iloc[0]))
                ref = d[(d.seed == sd) & (d.fold == fold)].set_index('idx').pred.loc[ite].to_numpy()
                sel: float
                if meth == 'FPCR':
                    sel = int(picked[1])
                    p0 = ns['fpcr'](C[itr], y[itr], C[ite], J, sel)
                    p1, used = g57.fpcr_lev(C[itr], y[itr], C[ite], M, [k for _, k in GRIDS['FPCR']], sel)
                    backed = used < sel
                elif meth == 'FPLR':
                    sel = float(picked[1])
                    p0 = ns['pflr'](C[itr], y[itr], C[ite], J, R, sel)
                    p1, used = g57.fplr_lev(C[itr], y[itr], C[ite], J, R, [float(v) for _, v in GRIDS['PFLR']], sel)
                    backed = used > sel * (1 + 1e-9)
                else:
                    sel = int(picked)
                    p0 = PLSRegression(n_components=min(sel, len(itr) - 1)).fit(Xp[itr], y[itr]).predict(Xp[ite]).ravel()
                    _p_sel, p1, used = g57.plsr_lev(Xp[itr], y[itr], Xp[ite], sel)
                    backed = used < sel
                gap = max(gap, float(np.abs(p0 - ref).max()))
                assert np.allclose(p0[~backed], p1[~backed], rtol=1e-6, atol=1e-6), (meth, sd, fold)
                hot = np.isin(ite, HOT)
                frows.append({'method': meth, 'seed': sd, 'fold': fold, 'picked': str(picked), 'n_test': len(ite),
                              'R2': r2_score(y[ite], p0), 'R2_lev': r2_score(y[ite], p1),
                              'n_backed': int(backed.sum()), 'frac_backed': float(backed.mean()),
                              'hot_in_test': int(hot.sum()), 'hot_backed': int((hot & backed).sum())})
                for i, v1, u, b in zip(ite, p1, used, backed, strict=True):
                    if i in HOT:
                        out.append({'method': meth, 'seed': sd, 'fold': fold, 'idx': int(i), 'pred_lev': float(v1),
                                    'used': float(u), 'backed': bool(b)})
        print(f'复现门 {meth}：选中配置的预测与 45 逐样品预测之差最大 {gap:.2e}', flush=True)
        if gap > GATE_TOL:
            raise SystemExit(f'{meth}：复现不了 45 的预测（最大差 {gap:.2e} > {GATE_TOL:g}），中止，不写产物')
        gpred[meth] = pd.DataFrame(out)
    fdf = pd.DataFrame(frows)

    # ---- 剖析：名单光谱的误差、除去名单光谱后的 R²、最弱一折 ----
    srows, sprows = [], []
    for meth, d in preds.items():
        per = d.groupby(['seed', 'fold']).apply(fold_r2)
        per_ex = d[~d.idx.isin(HOT)].groupby(['seed', 'fold']).apply(fold_r2)
        (wsd, wf), wr = per.idxmin(), float(per.min())
        w = d[(d.seed == wsd) & (d.fold == wf)].assign(ae=lambda q: q.e.abs()).sort_values('ae', ascending=False)
        e2 = w.e ** 2
        row = {'method': meth, 'R2_mean': float(per.mean()), 'R2_sd': float(per.std(ddof=1)),
               'R2_mean_excl_hot': float(per_ex.mean()), 'R2_sd_excl_hot': float(per_ex.std(ddof=1)),
               'worst_seed': int(wsd), 'worst_fold': int(wf), 'worst_R2': wr,
               'worst_top2_sse_share': float(e2.iloc[:2].sum() / e2.sum()),
               'worst_top5_idx': ' '.join(str(int(i)) for i in w.idx.iloc[:5]),
               'worst_top5_old_idx': ' '.join(str(old_of[int(i)]) for i in w.idx.iloc[:5]),
               'worst_top5_abs_err': ' '.join(f'{v:.3f}' for v in w.ae.iloc[:5]),
               'hot_max_abs_err': float(d[d.idx.isin(HOT)].e.abs().max()),
               'all_max_abs_err': float(d.e.abs().max())}
        if meth in GUARDED and meth in gpred:
            q = fdf[fdf.method == meth]
            row.update({'R2_mean_lev': float(q.R2_lev.mean()), 'R2_sd_lev': float(q.R2_lev.std(ddof=1)),
                        'worst_R2_lev': float(q[(q.seed == wsd) & (q.fold == wf)].R2_lev.iloc[0]),
                        'max_abs_dR2_other_folds': float((q.R2_lev - q.R2)[~((q.seed == wsd) & (q.fold == wf))].abs().max()),
                        'frac_backed_median': float(q.frac_backed.median()),
                        'hot_in_test': int(q.hot_in_test.sum()), 'hot_backed': int(q.hot_backed.sum())})
        srows.append(row)
        for r in d[d.idx.isin(HOT)].itertuples():
            sp = {'method': meth, 'seed': int(r.seed), 'fold': int(r.fold), 'idx': int(r.idx), 'old_idx': old_of[int(r.idx)],
                  'y': float(r.y), 'pred': float(r.pred), 'err': float(r.e)}
            if meth in gpred:
                g = gpred[meth]
                gg = g[(g.seed == r.seed) & (g.fold == r.fold) & (g.idx == r.idx)]
                assert len(gg) == 1, (meth, r.seed, r.fold, r.idx)
                sp.update({'pred_lev': float(gg.pred_lev.iloc[0]), 'err_lev': float(gg.pred_lev.iloc[0] - r.y),
                           'used_lev': float(gg.used.iloc[0]), 'backed': bool(gg.backed.iloc[0])})
            sprows.append(sp)
    summ = pd.DataFrame(srows)
    spec = pd.DataFrame(sprows)

    # 名单光谱的原始特征：清洗记录中的年份、产地、R1 距离与饱和点数，以及原始光谱的最大读数、平均读数在清洗后样品中的分位
    lk = lst.set_index('old_idx')
    X = ns['X']
    mx, me = X.max(1), X.mean(1)
    feats = pd.DataFrame([{'idx': i, 'old_idx': old_of[i], 'year': int(lk.loc[old_of[i], 'year']),
                           'origin': int(lk.loc[old_of[i], 'origin']), 'SSC': float(lk.loc[old_of[i], 'SSC']),
                           'R1_dist': float(lk.loc[old_of[i], 'R1_dist']), 'n_adc_max': int(lk.loc[old_of[i], 'n_adc_max']),
                           'raw_max': float(mx[i]), 'raw_max_quantile': float(np.mean(mx <= mx[i])),
                           'raw_mean_quantile': float(np.mean(me <= me[i]))}
                          for i in HOT])
    assert np.abs(feats.SSC.to_numpy() - y[list(HOT)]).max() < 1e-9, '清洗记录与读入数据的序号不一致'

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print('\n名单光谱的原始特征：'); print(feats.to_string(index=False))
    print('\n各方法：'); print(summ.round(4).to_string(index=False))
    print('\n名单光谱每次落入测试折的预测：'); print(spec.round(3).to_string(index=False))
    if len(fdf):
        print('\n杠杆回退逐折（R² 变化超过 0.005 的折）：')
        big = fdf[(fdf.R2_lev - fdf.R2).abs() > 0.005]
        print(big.round(4).to_string(index=False) if len(big) else '  无')

    if missing:
        print(f'\n部分结果（缺 {missing}，设了 MF_ALLOW_PARTIAL）：不写产物')
    else:
        fdf.to_csv(f'{OUT}/61rebuttal_mainfold_folds.csv', index=False, encoding='utf-8-sig')
        spec.merge(feats[['idx', 'R1_dist', 'n_adc_max', 'raw_max', 'raw_max_quantile', 'raw_mean_quantile']], on='idx').to_csv(
            f'{OUT}/61rebuttal_mainfold_spectra.csv', index=False, encoding='utf-8-sig')
        summ.to_csv(f'{OUT}/61rebuttal_mainfold_summary.csv', index=False, encoding='utf-8-sig')
        print(f'\n已写出 {OUT}/61rebuttal_mainfold_{{folds,spectra,summary}}.csv')
    print(f'总耗时 {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
