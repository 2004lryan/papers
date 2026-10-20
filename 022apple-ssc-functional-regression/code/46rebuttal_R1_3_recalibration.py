# -*- coding: utf-8 -*-
"""
46rebuttal_R1_3_recalibration.py: 跨批次的前瞻性小样本校正验证（导师意见 3，预注册 B10.3）

为什么要有这个脚本：原 3.4 节「扣除整体偏差后 R² 回升」所扣的偏差，是用外部集自身全部样品的
真实平均残差算出来的，属事后分解，推不出「用少量新批次样品校正截距即可」这一处置建议。
本脚本把它改成前瞻设计：先在源批次上建模，再从目标批次随机抽 k 个样品做校正，
只在其余从未参与校正的目标样品上评价；k 预先给定，每个 k 独立重抽 200 次。

设计（与 B10.3 逐项一致）：
  批次设置  留一年份 2 种（2018→2019、2019→2018）；留一产地 3 种（以另两个产地两年的样品建模，
            预测该产地两年的样品）。前者为「新采收季、产地已覆盖」，后者为「新产地、采收季已覆盖」
  方法      PLSR、SVR、FPLR、FPCR；函数型用三次 B 样条基 L=80（与表 1 主分析同）
  源模型    源批次内 KFold(5, shuffle=True, random_state=20041210) 从 grids.py 的 12 组候选中选参
            （准则同表 1：5 折 RMSE 的均值最小），用整个源批次重拟合，对目标批次全部样品预测
  k         {5, 10, 20, 40}；抽样器 numpy.random.default_rng(20041210)，按（设置, k）预先生成 200 组
            抽样下标，四个方法共用同一批抽样
  策略      S0 不校正；S1 截距校正 ŷ + mean_k(y − ŷ)；S2 斜率-截距校正（校正集上最小二乘 y = a + b·ŷ）；
            S3 加样重建（校正集并入源批次，沿用源批次选中的超参数重拟合，不重新选参）
  参照      O1 以目标批次全部样品的平均残差作截距校正、O2 以全部样品作斜率-截距校正，均在全部目标
            样品上评价；二者用到评价样品自身的真值，只作上限参照，不是可部署的策略
  评价      评价集（目标批次去掉校正集）上的 R²、RMSE 与平均偏差

断点续跑：按（设置, 方法）粒度落盘。每条 source、pred、draw 记录都带来源指纹 fp，指纹覆盖真值 y、预处理后
光谱 Xp、B 样条基系数、年份与产地标签、16 脚本的共享实现、grids.py、全部设置与种子、k、N_DRAW、
逐（设置, k）的抽样下标、计算函数的源码以及 Python/NumPy/SciPy/sklearn/torch 版本。已有产物中的某个
（设置, 方法）只有同时满足以下各条才沿用：三类记录的指纹均等于本次指纹；抽样结果的（k, 抽样号, 策略）
恰为期望键集合且无重复；S0、S1、S3 的三项指标全部有限，S2 仅在标记 failed 时三项同为缺失、否则全部有限；
源模型记录恰一行且数值有限（O2 仅在 O2_failed 时允许缺失）；预测的样品下标恰为目标批次、真值与当前数据一致、
预测有限。任一条不满足即整组重算。

运行方式:
    cd <项目根目录>
    python3 02code/46rebuttal_R1_3_recalibration.py

输出文件:
    04outputs/46rebuttal_recal_source.csv   — 每个（设置, 方法）的源批次选参结果与目标批次未校正指标
    04outputs/46rebuttal_recal_pred.csv     — 源模型对目标批次逐样品的预测
    04outputs/46rebuttal_recal_draws.csv    — 逐次抽样、逐策略的评价指标（断点续跑按（设置, 方法）粒度）
    04outputs/46rebuttal_recal_summary.csv  — 200 次抽样的中位数与 2.5%~97.5% 分位，含两种事后参照
    04outputs/46rebuttal_recal_batches.csv  — 全体、年份、产地、产地×年份及各设置源/目标批次的样本量与 SSC 统计，
                                              年份行含原始行数与剔除行的 SSC
    05logs/46rebuttal_R1_3_recalibration_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

只读 05data。
"""
import hashlib
import inspect
import json
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import data_clean   # 协议 B10.10 数据清洗四条规则的唯一实现（02code/data_clean.py）
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

with open('02code/16unified_comparison.py') as _src:
    _SHARED = _src.read().split("METHODS=['PLSR'")[0]
exec(_SHARED)
sys.path.insert(0, '02code')
from grids import GRIDS  # noqa: E402

warnings.simplefilter('default', RuntimeWarning)       # 16 脚本全局忽略了告警；数值告警照常进日志

SEED, K_INNER, L_BASIS, N_DRAW = 20041210, 5, 80, 200
K_LIST = [5, 10, 20, 40]
METHODS = ['PLSR', 'SVR', 'PFLR', 'FPCR']
LABEL = {'PFLR': 'FPLR'}

# 函数型基改为三次 B 样条 L=80（覆盖 16 脚本的傅里叶 BASIS/COEF，mk() 调用时按名取全局量）
_B = bspline_basis(t, L_BASIS)
_J, _Rm = gram_and_pen(_B, t)
BASIS = {L_BASIS: (_B, _J, _Rm)}
COEF = {L_BASIS: to_coef(Xp, _B, _Rm)}

# 批次标签：16 脚本里 X = vstack([X1, X2])，前 len(y1) 行为 2018；产地码 1 新疆、2 山东、3 陕西
YEAR = np.r_[np.full(len(y1), 2018), np.full(len(y2), 2019)]
ORIGIN = np.r_[z1, z2].astype(int)
assert len(YEAR) == len(ORIGIN) == len(y)
ONAME = {1: '新疆', 2: '山东', 3: '陕西'}
SETTINGS = [('留一年份', '2018→2019', YEAR == 2018, YEAR == 2019),
            ('留一年份', '2019→2018', YEAR == 2019, YEAR == 2018)]
SETTINGS += [('留一产地', f'留{ONAME[o]}', ORIGIN != o, ORIGIN == o) for o in (1, 2, 3)]

OUT_SRC = '04outputs/46rebuttal_recal_source.csv'
OUT_PRED = '04outputs/46rebuttal_recal_pred.csv'
OUT_DRAW = '04outputs/46rebuttal_recal_draws.csv'
OUT_SUM = '04outputs/46rebuttal_recal_summary.csv'
OUT_BATCH = '04outputs/46rebuttal_recal_batches.csv'


def metr(yt, p):
    p = np.asarray(p, dtype=float)
    return dict(R2=float(r2_score(yt, p)), RMSE=float(np.sqrt(mean_squared_error(yt, p))),
                BIAS=float(np.mean(p - yt)))


def atomic_csv(df, path):
    tmp = f'{path}.tmp'
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, path)


def slope_bias(p_cal, y_cal):
    """校正集上最小二乘 y = a + b·ŷ；设计矩阵秩不足（校正集预测值全相同）时返回 None，由调用方记为失败。"""
    A = np.c_[np.ones(len(p_cal)), p_cal]
    coef, _res, rank, _sv = np.linalg.lstsq(A, y_cal, rcond=None)
    return None if rank < 2 else (float(coef[0]), float(coef[1]))


def select_on_source(fit, grid, src):
    """源批次内 5 折交叉验证选参，准则与表 1 相同；并列取网格中靠前者。"""
    folds = list(KFold(n_splits=K_INNER, shuffle=True, random_state=SEED).split(src))
    best, bc, oof_best = np.inf, grid[0], None
    for c in grid:
        oof = np.zeros(len(src))
        rm = []
        for ia, iv in folds:
            p = fit(c, src[ia], src[iv], SEED)
            oof[iv] = p
            rm.append(np.sqrt(mean_squared_error(y[src[iv]], p)))
        e = float(np.mean(rm))
        if e < best:
            best, bc, oof_best = e, c, oof
    return bc, best, oof_best


# 抽样下标预先生成：按（设置, k）各 200 组，四个方法共用，保证策略与方法之间是配对比较
rng = np.random.default_rng(SEED)
DRAWS = {}
for _kind, name, _s, tmask in SETTINGS:
    n_t = int(tmask.sum())
    for k in K_LIST:
        DRAWS[(name, k)] = [rng.choice(n_t, size=k, replace=False) for _ in range(N_DRAW)]


def run_pair(kind, name, smask, tmask, m):
    """一个（设置, 方法）：源批次选参与重拟合、目标批次预测、200 次抽样 × 4 种策略的评价。
    返回源模型记录、逐样品预测记录、抽样评价记录（均未带指纹）。"""
    src, tgt = np.where(smask)[0], np.where(tmask)[0]
    yt = y[tgt]
    lab = LABEL.get(m, m)
    fit = mk(m)[1]
    bc, cv_rmse, oof = select_on_source(fit, GRIDS[m], src)
    p = np.asarray(fit(bc, src, tgt, SEED), dtype=float)             # 源模型对目标批次全部样品
    if not np.all(np.isfinite(p)):
        raise SystemExit(f'{name} {lab}：源模型对目标批次的预测含非有限值')
    s0 = metr(yt, p)
    res = yt - p
    o1 = metr(yt, p + res.mean())
    sb_all = slope_bias(p, yt)
    o2 = (dict(R2=np.nan, RMSE=np.nan, BIAS=np.nan) if sb_all is None
          else metr(yt, sb_all[0] + sb_all[1] * p))
    r = float(np.corrcoef(p, yt)[0, 1])
    src_row = dict(kind=kind, setting=name, method=lab, picked=str(bc), n_src=len(src), O2_failed=sb_all is None,
                   n_tgt=len(tgt), src_cv_rmse=cv_rmse, src_cv_R2=float(r2_score(y[src], oof)),
                   tgt_R2=s0['R2'], tgt_RMSE=s0['RMSE'], tgt_BIAS=s0['BIAS'], tgt_r=r,
                   O1_R2=o1['R2'], O1_RMSE=o1['RMSE'], O2_R2=o2['R2'], O2_RMSE=o2['RMSE'])
    preds = [dict(setting=name, method=lab, idx=int(i), y=float(y[i]), pred=float(v))
             for i, v in zip(tgt, p, strict=True)]
    rec = []
    for k in K_LIST:
        for b, dr in enumerate(DRAWS[(name, k)]):
            ev = np.setdiff1d(np.arange(len(tgt)), dr)
            y_ev, p_ev = yt[ev], p[ev]
            rec.append(dict(setting=name, method=lab, k=k, draw=b, strategy='S0', **metr(y_ev, p_ev)))
            off = float(np.mean(yt[dr] - p[dr]))
            rec.append(dict(setting=name, method=lab, k=k, draw=b, strategy='S1', **metr(y_ev, p_ev + off)))
            sb = slope_bias(p[dr], yt[dr])
            if sb is None:
                rec.append(dict(setting=name, method=lab, k=k, draw=b, strategy='S2',
                                R2=np.nan, RMSE=np.nan, BIAS=np.nan, failed=True))
            else:
                rec.append(dict(setting=name, method=lab, k=k, draw=b, strategy='S2',
                                **metr(y_ev, sb[0] + sb[1] * p_ev)))
            p3 = np.asarray(fit(bc, np.r_[src, tgt[dr]], tgt[ev], SEED), dtype=float)
            if not np.all(np.isfinite(p3)):
                raise SystemExit(f'{name} {lab} k={k} 抽样 {b}：加样重建的预测含非有限值')
            rec.append(dict(setting=name, method=lab, k=k, draw=b, strategy='S3', **metr(y_ev, p3)))
    return src_row, preds, rec, s0, o1, o2, r


def provenance_fp():
    """来源指纹：决定 source/pred/draw 三类记录取值的全部输入与代码，任一项改变即整组重算。"""
    import scipy
    import sklearn
    h = hashlib.sha256()
    for arr in (y, Xp, COEF[L_BASIS], YEAR, ORIGIN):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(json.dumps([SEED, K_INNER, L_BASIS, N_DRAW, K_LIST, METHODS, [GRIDS[m] for m in METHODS],
                         [(kd, nm) for kd, nm, _s, _t in SETTINGS]], sort_keys=True, default=str).encode())
    for (nm, k), lst in sorted(DRAWS.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        h.update(f'{nm}|{k}'.encode())
        for dr in lst:
            h.update(np.asarray(dr, dtype=np.int64).tobytes())
    h.update(_SHARED.encode())
    with open('02code/grids.py', encoding='utf-8') as fh:
        h.update(fh.read().encode())
    for fn in (metr, slope_bias, select_on_source, run_pair):
        h.update(inspect.getsource(fn).encode())
    h.update(json.dumps([sys.version, np.__version__, scipy.__version__, sklearn.__version__,
                         torch.__version__]).encode())
    return h.hexdigest()[:16]


FP = provenance_fp()
print(f'跨批次前瞻校正：{len(SETTINGS)} 种设置 × {len(METHODS)} 个方法 × k∈{K_LIST} × {N_DRAW} 次抽样  指纹 {FP}')
for kind, name, smask, tmask in SETTINGS:
    print(f'  {kind} {name}: 源 n={int(smask.sum())} SSC={y[smask].mean():.2f}±{y[smask].std():.2f}；'
          f'目标 n={int(tmask.sum())} SSC={y[tmask].mean():.2f}±{y[tmask].std():.2f}')
print(flush=True)

_STR = {'fp': str, 'setting': str, 'method': str, 'strategy': str, 'picked': str, 'kind': str}
src_rows = pd.read_csv(OUT_SRC, dtype=_STR).to_dict('records') if os.path.exists(OUT_SRC) else []
pred_rows = pd.read_csv(OUT_PRED, dtype=_STR).to_dict('records') if os.path.exists(OUT_PRED) else []
draw_df = pd.read_csv(OUT_DRAW, dtype=_STR) if os.path.exists(OUT_DRAW) else pd.DataFrame()
STRATS = ('S0', 'S1', 'S2', 'S3')
EXPECTED = {(k, b, st) for k in K_LIST for b in range(N_DRAW) for st in STRATS}
TGT_IDX = {name: set(np.where(tmask)[0].tolist()) for _k, name, _s, tmask in SETTINGS}
SRC_NUM = ['src_cv_rmse', 'src_cv_R2', 'tgt_R2', 'tgt_RMSE', 'tgt_BIAS', 'tgt_r', 'O1_R2', 'O1_RMSE']


def _is_true(v):
    return v is True or str(v) == 'True'


def complete_pairs():
    """（设置, 方法）算完成须同时满足（任一不满足整组重算）：
    ① 三类记录的指纹都等于本次指纹 FP；
    ② 抽样结果的（k, 抽样号, 策略）恰为期望键集合且无重复；S0/S1/S3 三项指标全部有限，
       S2 仅在 failed=True 时三项同为缺失，否则全部有限；非 S2 行不得带 failed=True；
    ③ 源模型记录恰一行，SRC_NUM 各列有限，O2_R2/O2_RMSE 仅在 O2_failed=True 时允许缺失；
    ④ 预测的样品下标恰为该设置的目标批次、各一条，真值与当前数据一致，预测有限。"""
    ok: set[tuple[str, str]] = set()
    if draw_df.empty or not {'fp', 'setting', 'method', 'k', 'draw', 'strategy', 'R2', 'RMSE', 'BIAS'} <= set(draw_df.columns):
        return ok
    src_df, pred_df = pd.DataFrame(src_rows), pd.DataFrame(pred_rows)
    if src_df.empty or pred_df.empty or 'fp' not in src_df.columns or 'fp' not in pred_df.columns:
        return ok
    for (name, lab), a in draw_df.groupby(['setting', 'method']):
        if name not in TGT_IDX or not (a.fp.astype(str) == FP).all():
            continue
        keys = list(zip(a.k.astype(int), a.draw.astype(int), a.strategy, strict=True))
        if len(keys) != len(EXPECTED) or set(keys) != EXPECTED:
            continue
        m3 = a[['R2', 'RMSE', 'BIAS']].to_numpy(dtype=float)
        failed = (a['failed'].map(_is_true).to_numpy(dtype=bool) if 'failed' in a.columns
                  else np.zeros(len(a), dtype=bool))
        is_s2 = (a.strategy == 'S2').to_numpy()
        valid = np.where(is_s2 & failed, np.isnan(m3).all(axis=1), np.isfinite(m3).all(axis=1))
        if not valid.all() or (failed & ~is_s2).any():
            continue
        s = src_df[(src_df.setting == name) & (src_df.method == lab)]
        if len(s) != 1 or str(s.fp.iloc[0]) != FP:
            continue
        if not np.all(np.isfinite(s[SRC_NUM].to_numpy(dtype=float))):
            continue
        o2 = s[['O2_R2', 'O2_RMSE']].to_numpy(dtype=float)
        if not (np.isnan(o2).all() if _is_true(s.O2_failed.iloc[0]) else np.isfinite(o2).all()):
            continue
        pp = pred_df[(pred_df.setting == name) & (pred_df.method == lab)]
        if not (len(pp) == len(TGT_IDX[name]) and set(pp.idx.astype(int)) == TGT_IDX[name]
                and (pp.fp.astype(str) == FP).all()
                and np.all(np.isfinite(pp[['y', 'pred']].to_numpy(dtype=float)))):
            continue
        if float(np.max(np.abs(pp.y.to_numpy(dtype=float) - y[pp.idx.astype(int).to_numpy()]))) > 1e-9:
            continue
        ok.add((name, lab))
    return ok


done = complete_pairs()
n_old = 0 if draw_df.empty else len(draw_df.groupby(['setting', 'method']))
src_rows = [r for r in src_rows if (r['setting'], r['method']) in done]
pred_rows = [r for r in pred_rows if (r['setting'], r['method']) in done]
if not draw_df.empty:
    draw_df = draw_df[[(a, b) in done for a, b in zip(draw_df.setting, draw_df.method, strict=True)]]
print(f'已完成且来源相符的（设置, 方法）：{sorted(done)}；已有产物中判废整组重算的 {n_old - len(done)} 组', flush=True)

for kind, name, smask, tmask in SETTINGS:
    for m in METHODS:
        lab = LABEL.get(m, m)
        if (name, lab) in done:
            print(f'  {name} {lab} 已完成，跳过')
            continue
        t0 = time.time()
        src_row, preds, rec, s0, o1, o2, r = run_pair(kind, name, smask, tmask, m)
        src_rows.append(dict(src_row, fp=FP))
        pred_rows += [dict(x, fp=FP) for x in preds]
        rd = pd.DataFrame(rec).assign(fp=FP)
        draw_df = pd.concat([draw_df, rd], ignore_index=True)
        atomic_csv(pd.DataFrame(src_rows), OUT_SRC)
        atomic_csv(pd.DataFrame(pred_rows), OUT_PRED)
        atomic_csv(draw_df.round(6), OUT_DRAW)
        med = rd.groupby(['k', 'strategy']).R2.median().unstack()
        print(f'  {name} {lab:<5} 选中 {src_row["picked"]:<22} 源内 CV R²={src_row["src_cv_R2"]:.3f}  '
              f'目标 R²={s0["R2"]:.3f} 偏差 {s0["BIAS"]:+.3f} r={r:.3f}  O1={o1["R2"]:.3f} O2={o2["R2"]:.3f}  '
              f'({time.time() - t0:.0f}s)', flush=True)
        print('      R² 中位数（行 k，列策略）：\n' + med.round(3).to_string().replace('\n', '\n      '), flush=True)

# 汇总：200 次抽样的中位数与 2.5%~97.5% 分位；S1 优于 S0 的次数占比；并入两种事后参照
g = draw_df.groupby(['setting', 'method', 'k', 'strategy'])
summ = g.agg(R2_med=('R2', 'median'), R2_lo=('R2', lambda v: np.nanquantile(v, 0.025)),
             R2_hi=('R2', lambda v: np.nanquantile(v, 0.975)), RMSE_med=('RMSE', 'median'),
             RMSE_lo=('RMSE', lambda v: np.nanquantile(v, 0.025)), RMSE_hi=('RMSE', lambda v: np.nanquantile(v, 0.975)),
             BIAS_med=('BIAS', 'median')).reset_index()
w = draw_df.pivot_table(index=['setting', 'method', 'k', 'draw'], columns='strategy', values='R2').reset_index()
frac = w.assign(S1_gt_S0=(w.S1 > w.S0)).groupby(['setting', 'method', 'k']).S1_gt_S0.mean().reset_index()
summ = summ.merge(frac, on=['setting', 'method', 'k'], how='left')
summ = summ.merge(pd.DataFrame(src_rows)[['setting', 'method', 'kind', 'picked', 'src_cv_R2', 'tgt_R2',
                                          'tgt_BIAS', 'tgt_r', 'O1_R2', 'O2_R2', 'O1_RMSE', 'O2_RMSE']],
                  on=['setting', 'method'], how='left')
FAILED = draw_df['failed'].map(_is_true).astype(bool) if 'failed' in draw_df else pd.Series(False, index=draw_df.index)
n_fail = int(FAILED.sum())
summ['S2_failed_draws'] = 0
if n_fail:
    nf = draw_df[FAILED].groupby(['setting', 'method', 'k']).size()
    summ = summ.merge(nf.rename('n_f').reset_index(), on=['setting', 'method', 'k'], how='left')
    summ['S2_failed_draws'] = summ.pop('n_f').fillna(0).astype(int)
print(f'斜率-截距校正因校正集设计矩阵秩不足而失败的抽样：{n_fail} 次')
atomic_csv(summ, OUT_SUM)


# 批次描述（2.1 节样品构成与 3.3 节批次漂移取数）：各层级及各设置源、目标批次的样本量与 SSC 统计（标准差 ddof=1）；
# 年份行另记原始行数与按 16 脚本 load() 规则剔除的行数（SSC 不在 (5, 25) °Brix、SSC/产地码缺失或光谱含非有限值，
# 以及协议 B10.10 数据清洗四条规则的并集；逐行命中情况见 59data_clean.py 的清单）
def _desc(level, name, mask):
    v = y[mask]
    return dict(level=level, name=name, n=int(mask.sum()), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                min=float(v.min()), max=float(v.max()))


batch_rows = [_desc('全体', '全体', np.ones(len(y), dtype=bool))]
for yr, fn in ((2018, f'{D}/01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx'),
               (2019, f'{D}/1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx')):
    raw = pd.read_excel(fn, header=None)
    y_raw = pd.to_numeric(raw.iloc[2:, 2048], errors='coerce').to_numpy(dtype=float)
    z_raw = pd.to_numeric(raw.iloc[2:, 2049], errors='coerce').to_numpy(dtype=float)
    x_raw = raw.iloc[2:, :2048].astype(float).to_numpy()
    x_ok = np.isfinite(x_raw).all(1)
    keep = np.isfinite(y_raw) & np.isfinite(z_raw) & (y_raw > 5) & (y_raw < 25) & x_ok
    keep &= data_clean.keep_mask(x_raw, y_raw, z_raw)
    if int(keep.sum()) != int((YEAR == yr).sum()):
        raise SystemExit(f'{yr} 年原始表按 load() 规则保留 {int(keep.sum())} 行，与已载入的 {int((YEAR == yr).sum())} 行不符')
    batch_rows.append(dict(_desc('年份', str(yr), YEAR == yr), n_raw=len(y_raw), n_excluded=int((~keep).sum())))
batch_rows += [_desc('产地', ONAME[o], ORIGIN == o) for o in (1, 2, 3)]
batch_rows += [_desc('产地×年份', f'{ONAME[o]}{yr}', (ORIGIN == o) & (YEAR == yr)) for o in (1, 2, 3) for yr in (2018, 2019)]
for _kind, name, smask, tmask in SETTINGS:
    batch_rows += [_desc('源批次', name, smask), _desc('目标批次', name, tmask)]
atomic_csv(pd.DataFrame(batch_rows), OUT_BATCH)

print('\n' + '=' * 80)
print('【R² 中位数：k=10 时各策略，与事后参照 O1/O2】')
k10 = summ[summ.k == 10].pivot_table(index=['setting', 'method'], columns='strategy', values='R2_med')
k10 = k10.join(summ[summ.k == 10].groupby(['setting', 'method'])[['tgt_R2', 'O1_R2', 'O2_R2']].first())
print(k10.round(3).to_string())
print(f'\n→ {OUT_SRC}\n→ {OUT_PRED}\n→ {OUT_DRAW}\n→ {OUT_SUM}\n→ {OUT_BATCH}')
