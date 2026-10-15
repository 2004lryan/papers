# -*- coding: utf-8 -*-
"""
54rebuttal_R1_4_boundary_shift.py: 第三轮网格（B10.1 网格边界诊断触发）的核对、跨轮核对、检验与两轮并列报告
（协议 C3 2026-09-15 第三轮网格行）

为什么要有这个脚本：第二轮 25 个外层折中，SVR 的 gamma 全选网格最大值 scale，1D-CNN 的层数、学习率与宽度过半选在
最大端，网格可能截断了这两个方法。按 B10.1，两法按 grids.py 的规则①~④ 平移网格（02code/grids_boundary_shift.py）
再跑一轮、两轮并列报告；47 的主族（第二轮 15 对）不因此改动。本脚本核对第三轮结果、与第二轮逐项比对，并按协议登记的
两族作检验，把两族与主族的判定并列出来。

输入核对（全程持有 47 运行锁与第三轮工作目录锁；先把 47 标记、第三轮 manifest 与其派生的两轮缓存集合一次读入，
写入系统临时目录中的私有镜像；缓存相关核对与计算只读镜像。47 核对函数仍从原文取出执行，见 results54.load47）：
  前提      47 的正式结果有效（results47.resolve() 为正式）；本脚本读取的第二轮 45 汇总、47 的表 1 与配对检验，其 SHA-256
            与 47 完成标记记录的相同。
  工作目录  04outputs/45shift 的 02code/ 中，45、16、ncv_manifest.py 与项目文件逐字节相同，grids.py 与
            02code/grids_boundary_shift.py 相同，后者的 SHA-256 为协议 C3 登记值（results54.GRIDS_SHA256）。
  ①~④、⑥   在私有镜像的第三轮工作目录中执行 47 的核对：覆盖、汇总行（选中配置属于第三轮网格）、预测行（真值与十项逐折指标重算）、来源清单
            （代码为工作目录中的代码，其中 grids 为第三轮网格；SVR 的缓存指纹按 45 的定义在工作目录中以本机数据与环境重算；
            1D-CNN 为服务器分片写出的 SHIFT_CUDA_MEMO_FINGERPRINT，服务器环境与光谱哈希取 47 的常量）、格点缓存复核。
            同指纹缓存的集合与 SHA-256 写入完成标记，新增、删除或改动均使发布或 resolve() 拒绝；其他指纹只列为残留。
            工作目录中不得有两法之外的 45 产物。
  跨轮核对  用于第三轮网格（47 的⑤参照 25nested_cv.csv，只适用于第二轮网格）：两轮共有候选（SVR 3 组、1D-CNN 4 组）在 25 个外层折上
            的内层 MSE，以及两轮选中同一候选的外层折的重训练预测。SVR 须与第二轮格点缓存 JSON 解析后逐元素数值严格相等（第二轮缓存先与 47 完成标记
            记录的哈希核对）；1D-CNN 只列不判（GPU 上 AdaptiveAvgPool 的反向传播没有确定性实现）。
  表 1 复算 第二轮 SVR、1D-CNN 的表 1 行由本脚本按 47 的汇总口径重算，须与 47 的表 1 一致（差 ≤1e-12），两轮并列才可比。

检验（协议 C3；修正重复 k 折 t 检验、置信区间、非正态回退与 α 同 B10.2，用 47 的 corrected_test 与 holm）：
  F1  SVR、1D-CNN 各自的第三轮减第二轮（同一种子、同一外层折的 R² 差），2 对，Holm
  F2  六法 15 对，SVR 与 1D-CNN 取第三轮结果、其余四法取第二轮结果（网格未变，同折配对），Holm 按 15 对；
      每行并列 47 主族同一对的 Holm p 与结论，same_decision 标出两族判定（Holm p<0.05 与否）是否相同。
      判定不同的对，正文与结论须同时给出两轮结果并写明差异来自网格平移。
另附：第三轮网格的端点诊断（口径同 47：非数值取值按 scikit-learn 拟合时实际使用的数值排序）。

运行方式:
    cd <项目根目录>
    python3 02code/54rebuttal_R1_4_boundary_shift.py
    SHIFT_PRINT_ONLY=1 python3 02code/54rebuttal_R1_4_boundary_shift.py   （尚有方法未完成时查看进度：只打印，不写产物）

输出文件（全部核对通过才写；发布与读取见 results54.py）:
    04outputs/54rebuttal_table1.csv       — SVR、1D-CNN 两轮并列的汇总指标
    04outputs/54rebuttal_paired_tests.csv — F1 与 F2 的检验结果，F2 行并列 47 主族的判定
    04outputs/54rebuttal_crossround.csv   — 两轮共有候选的逐项比对
    04outputs/54rebuttal_boundary.csv     — 第三轮网格的端点诊断
    04outputs/54rebuttal_provenance.csv   — 第三轮逐（方法, 种子, 折）的真值溯源核对
    04outputs/54rebuttal_COMPLETE.json    — 完成标记
    05logs/54rebuttal_R1_4_boundary_shift_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

只读 05data：真值与预处理后光谱由原始数据集重算（执行 16 脚本的共享段，与 45、47 同一实现）。
"""
import glob
import io
import itertools
import json
import os
import re
import atexit
import shutil
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')      # 须先于 numpy 与 torch 的导入

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

sys.path.insert(0, '02code')
import grids as grids2  # noqa: E402
import grids_boundary_shift as gbs  # noqa: E402
import ncv_manifest as nm  # noqa: E402
import results47 as r47  # noqa: E402
import results54 as r54  # noqa: E402

PRINT_ONLY = os.environ.get('SHIFT_PRINT_ONLY', '0') == '1'
TAG = '第三轮'
SHIFT_METHODS = ['SVR', '1D-CNN']
# 服务器上第三轮 1D-CNN 的缓存指纹由服务器上的 45 分片进程算出（其中含服务器的 Xp 字节与完整 Python 版本串，本机无法重算）。
# 开跑后取自分片写出的格点缓存文件名 shift/04outputs/45memo_none_1DCNN_<指纹>_shard<i>.jsonl（服务器的项目目录下）填入；
# cnn_fetch.sh 取回时核对清单与之相同。未填入时 1D-CNN 一律判为未通过。
# 2026-09-25 服务器在清洗后数据上（协议 B10.10）第三轮分片写出的缓存文件名 45memo_none_1DCNN_<指纹>_shard*.jsonl 中的指纹（e2e54 按整行改写此常量，行尾不加注释）
SHIFT_CUDA_MEMO_FINGERPRINT = '0a5587e34fd4c4fd'
FUNCS47 = ('read_csv_bytes', 'brief', 'manifest_problems', 'check_method', 'cache_key', 'cache_problems', 'fold_vec',
           'corrected_test', 'holm', 'dims_of', 'value_note')
CONSTS47 = ('K', 'R', 'N_TE', 'N_TR', 'N', 'J', 'RHO', 'L_BASIS', 'ORDER', 'KEY', 'SEED_LIST', 'SEEDS', 'EXPECTED_KEYS',
            'LABEL_OF_STEM', 'FILE_RE', 'SERVER_CPU_ENV', 'CUDA_ENV', 'SERVER_XP_SHA256', 'RF_N_JOBS', 'STR_COLS',
            'NUM_COLS', 'METRIC_COLS', 'PRED_COLS', 'TEST_COLS')
T1_NUM = ['R2', 'R2_sd', 'RMSE', 'RMSE_sd', 'RPD', 'RPD_sd', 'MAE', 'BIAS', 'seed_range']
assert list(gbs.GRIDS) == SHIFT_METHODS

# ---------- 0. 锁、占用与清理 ----------
if not os.path.isdir(r54.WORK):
    sys.exit(f'尚无第三轮工作目录 {r54.WORK}（本机 SVR 由 02code/deploy45/shift_local.sh 建立并运行）')
_RUN_LOCK = r47.acquire_run_lock()
if _RUN_LOCK is None:
    sys.exit(f'47 的运行锁 {r47.RUN_LOCK} 被占用（47 在运行，或 cnn_fetch.sh 正在发布取回的结果），本次不运行')
_WORK_LOCK = r54.acquire_work_lock()
if _WORK_LOCK is None:
    sys.exit(f'工作目录锁 {r54.WORK_LOCK} 被占用（shift_local.sh 正在运行第三轮），本次不运行')
BUSY = r47.activity(r54.WORK)[0] + r47.activity('.')[0]
if BUSY and not PRINT_ONLY:
    sys.exit('第三轮或第二轮的 45 输入正在被改写，须等其结束：\n  ' + '\n  '.join(BUSY))
if BUSY:
    print('【注意】45 的输入正在被改写，以下结果可能随即失效：\n  ' + '\n  '.join(BUSY))
if not PRINT_ONLY:
    _removed = r54.invalidate_stale()
    if _removed:
        print('正式路径上的旧产物没有有效完成标记，已删除：\n  ' + '\n  '.join(_removed))

# ---------- 1. 前提：47 的正式结果；工作目录的代码 ----------
try:
    SRC47 = r47.resolve()
except SystemExit as e:
    sys.exit(f'第三轮以 47 的正式结果为参照，47 的产物无效：{e}')
if not SRC47.final:
    sys.exit('47 只有部分结果；第三轮以 47 的正式结果为参照')
_wc = r54.work_code_problems()
if _wc:
    sys.exit('第三轮工作目录的代码不符：\n  ' + '\n  '.join(_wc))

ns = r54.load47(FUNCS47, CONSTS47, base=dict(np=np, pd=pd, json=json, glob=glob, os=os, io=io, re=re, stats=stats, nm=nm))
K, N, N_TE = ns['K'], ns['N'], ns['N_TE']
SEED_LIST, ORDER = ns['SEED_LIST'], ns['ORDER']
assert set(SHIFT_METHODS) <= set(ORDER)
read_csv_bytes, fold_vec, corrected_test, holm = ns['read_csv_bytes'], ns['fold_vec'], ns['corrected_test'], ns['holm']
ROUND2_BASIS = {m: ('bspline' if m in ('FPLR', 'FPCR') else 'none') for m in ORDER}
ROUND2_SUMMARY = {m: f"04outputs/45rebuttal_ncv_{ROUND2_BASIS[m]}_{m.replace('-', '')}.csv" for m in ORDER}
READ47 = [r47.COMPLETE, SRC47.path('table1'), SRC47.path('paired_tests')]

# ---------- 2. 输入快照：先冻结 marker/manifest，再派生精确缓存集合；此后只读系统临时镜像 ----------
_meta_paths = r54.work_inputs() + sorted(ROUND2_SUMMARY.values()) + READ47
_META = r54.snapshot(_meta_paths)
M47 = json.loads(_META[r47.COMPLETE].decode('utf-8'))
_cache_paths = r54.cache_inputs(_META)
_round2_caches = sorted(p for p in _cache_paths if not p.startswith(r54.WORK + '/'))
_missing2 = [p for p in _round2_caches if not os.path.exists(p)]
if _missing2:
    sys.exit(f'第二轮格点缓存不存在或与 47 完成标记记录的哈希不符：{_missing2}')
try:
    _CACHE = r54.snapshot(_cache_paths)
except FileNotFoundError as e:
    sys.exit(f'第三轮格点缓存缺失：{e.filename}')
SNAP = {**_META, **_CACHE}
_work_code_bytes = r54.snapshot([f'{r54.WORK}/{p}' for p in r54.WORK_COPIES])
_MIRROR = r54.materialize_snapshot({**SNAP, **_work_code_bytes})
atexit.register(shutil.rmtree, _MIRROR, ignore_errors=True)
# 端到端测试钩子：快照完成后修改活动缓存中的一条合法 inner 记录，验证计算只读镜像且发布拒绝活动输入漂移。
_test_after_snapshot = os.environ.get('SHIFT_TEST_AFTER_SNAPSHOT_MUTATE')
if _test_after_snapshot:
    with open(_test_after_snapshot, encoding='utf-8') as _fh:
        _test_lines = _fh.readlines()
    _test_changed = False
    for _test_i, _test_line in enumerate(_test_lines):
        if not _test_line.strip():
            continue
        _test_rec = json.loads(_test_line)
        _test_kind, _test_seed, _test_fold, _test_cfg = json.loads(_test_rec['k'])
        if _test_kind == 'inner' and _test_cfg[0] in (100, 1000) and _test_cfg[1] == 'scale':
            _test_rec['v'][0] = float(_test_rec['v'][0]) + 0.125
            _test_lines[_test_i] = json.dumps(_test_rec, ensure_ascii=False, sort_keys=True) + '\n'
            _test_changed = True
            break
    if not _test_changed:
        raise RuntimeError('测试钩子找不到第三轮 SVR 的共有候选 inner 记录')
    with open(_test_after_snapshot, 'w', encoding='utf-8') as _fh:
        _fh.writelines(_test_lines)


def frozen_path(path):
    return os.path.join(_MIRROR, path)


CODE54 = r54.code_sha()
_bad_cache47 = [p for p in _round2_caches
                if (M47.get('caches', {}).get(p) or {}).get('sha256') != nm.sha256_bytes(SNAP[p])]
if _bad_cache47:
    sys.exit(f'第二轮格点缓存不存在或与 47 完成标记记录的哈希不符：{_bad_cache47}')
_bad47 = [p for p in ROUND2_SUMMARY.values() if (M47.get('inputs') or {}).get(p) != nm.sha256_bytes(SNAP[p])]
_bad47 += [p for p in READ47[1:] if ((M47.get('outputs') or {}).get(p) or {}).get('sha256') != nm.sha256_bytes(SNAP[p])]
if M47 != SRC47.marker or _bad47:
    sys.exit(f'读入的 47 完成标记、第二轮 45 汇总或 47 产物与 47 完成标记记录的不符：{_bad47 or [r47.COMPLETE]}')

# ---------- 3. 原始数据：真值与预处理后光谱（执行 16 脚本的共享段，与 45 同一实现）----------
_ns: dict = {}
exec(nm.shared16_text(), _ns)
Y_DATA = np.asarray(_ns['y'], dtype=float)
if len(Y_DATA) != N:
    sys.exit(f'原始数据集读入 {len(Y_DATA)} 个样品，应为 {N}')
DATA_NOW = {'none': nm.data_sha256(_ns['y'], _ns['Xp'])}
ENV_NOW = nm.env_info('cpu')
P_FEATURES = int(np.asarray(_ns['Xp']).shape[1])
SCALE_GAMMAS = r47.svr_scale_gammas(_ns['Xp'], SEED_LIST, K)
TEST_SETS = {(sd, f): set(te.tolist()) for sd in ns['SEEDS']
             for f, (_tr, te) in enumerate(KFold(K, shuffle=True, random_state=sd).split(np.arange(N)))}

# ---------- 4. 读入工作目录中 45 的汇总：只收两法，文件里的标签须与文件名相符 ----------
SNAP_WORK = {p[len(r54.WORK) + 1:]: b for p, b in SNAP.items()
             if p.startswith(r54.WORK + '/04outputs/45rebuttal_ncv_')}
EXPECTED = {(TAG, m): 'none' for m in SHIFT_METHODS}
status: dict[str, str] = {}
problems: list[str] = []
label_bad: dict[tuple[str, str], str] = {}
frames = []
for _p, _bytes in SNAP_WORK.items():
    _mt = ns['FILE_RE'].match(_p)
    _m = ns['LABEL_OF_STEM'].get(_mt.group(2), '') if _mt else ''
    if not _mt or EXPECTED.get((TAG, _m)) != _mt.group(1):
        status[f'文件/{r54.WORK}/{_p}'] = 'invalid'
        problems.append(f'{r54.WORK} 下有预期之外的 45 产物：{_p}')
        continue
    if _mt.group(3) != '.csv':
        continue
    _df = read_csv_bytes(_bytes, dtype=ns['STR_COLS'])
    if (not {'basis', 'method', 'seed', 'fold'} <= set(_df.columns) or set(_df.basis.astype(str)) != {_mt.group(1)}
            or set(_df.method.astype(str)) != {_m}):
        label_bad[(TAG, _m)] = f'{TAG} {_m}：{_p} 里的基或方法标签与文件名不符'
        continue
    frames.append(_df)
d3 = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['basis', 'method', 'seed', 'fold'])
d3['seed'] = d3.seed.astype(int)
d3['fold'] = d3.fold.astype(int)

# ---------- 5. 在工作目录中执行 47 的核对 ①~④、⑥ ----------
SERVER_METHODS = {'1D-CNN': ('cuda', ns['CUDA_ENV'], SHIFT_CUDA_MEMO_FINGERPRINT)}
CACHES: dict[str, dict] = {}
prov: list[dict] = []
_cwd = os.getcwd()
os.chdir(frozen_path(r54.WORK))
try:
    CODE_NOW = nm.code_sha256()
    try:
        EXPECTED_MEMO = {('none', 'SVR'): r47.memo_fingerprint45(dict(
            np=np, torch=_ns['torch'], y=_ns['y'], Xp=_ns['Xp'], COEF={}, L_BASIS=ns['L_BASIS'], K_OUTER=K, K_INNER=K,
            GRIDS=gbs.GRIDS, _SHARED=nm.shared16_text(), BASIS_NAME='bspline', FUNCTIONAL=('PFLR', 'FPCR'),
            N_JOBS=ns['RF_N_JOBS'], DEVICE='cpu'), 'SVR')}
        FOLD_METRICS = r47.fold_metrics45()
    except Exception as e:  # noqa: BLE001 — 任何失败都意味着核对无从进行，停下
        sys.exit(f'无法在工作目录中按 45 的定义重算缓存指纹或取得逐折指标定义：{type(e).__name__}: {e}')
    ns.update(GRIDS=gbs.GRIDS, EXPECTED=EXPECTED, SERVER_METHODS=SERVER_METHODS, EXPECTED_MEMO=EXPECTED_MEMO,
              CODE_NOW=CODE_NOW, DATA_NOW=DATA_NOW, ENV_NOW=ENV_NOW, SNAP=SNAP_WORK, d=d3, label_bad=label_bad,
              TEST_SETS=TEST_SETS, Y_DATA=Y_DATA, FOLD_METRICS=FOLD_METRICS, CACHES=CACHES, SCALE_GAMMAS=SCALE_GAMMAS)
    print(f'第三轮应有的缓存指纹  none/SVR={EXPECTED_MEMO[("none", "SVR")]}  '
          f'none/1D-CNN={SHIFT_CUDA_MEMO_FINGERPRINT or "（未登记）"}（服务器）')
    print(f'工作目录代码 grids={CODE_NOW["grids"][:16]}  45={CODE_NOW["45"][:16]}')
    for _m in SHIFT_METHODS:
        if _m in SERVER_METHODS and not SERVER_METHODS[_m][2]:
            _st, _probs, _pv = 'invalid', [f'{TAG} {_m}：尚未登记服务器上第三轮的缓存指纹 SHIFT_CUDA_MEMO_FINGERPRINT'], []
        else:
            _st, _probs, _pv = ns['check_method'](TAG, _m)
        status[f'{TAG}/{_m}'] = _st
        problems += _probs
        if _st == 'ok':
            prov += _pv
    for _m in SHIFT_METHODS:
        _label = _m.replace('-', '')
        _selected = EXPECTED_MEMO.get(('none', _m), '') if _m == 'SVR' else SHIFT_CUDA_MEMO_FINGERPRINT
        _residue = [p for p in r54.cache_residue(_META, _cache_paths)
                    if os.path.basename(p).startswith(f'45memo_none_{_label}_')]
        if _residue:
            print(f'【缓存残留】{_m} 存在其他指纹缓存（忽略，不纳入 54 输入）：' + '；'.join(_residue))
finally:
    os.chdir(_cwd)

FINAL = all(v == 'ok' for v in status.values())
pv = pd.DataFrame(prov, columns=['analysis', 'method', 'seed', 'fold', 'n', 'max_abs_y_diff'])
print(f'【真值溯源】已核对 {len(pv)} 个（方法, 种子, 折），预测文件 y 与原始数据集真值的最大绝对差 '
      f'{(pv.max_abs_y_diff.max() if len(pv) else float("nan")):.1e}')
print('【各方法状态】' + '  '.join(f'{k}={v}' for k, v in status.items()))
if problems:
    print('【输入缺失、未跑完或未通过核对】\n  ' + '\n  '.join(problems))
if not FINAL and not PRINT_ONLY:
    sys.exit('输入不完整或未通过核对，不输出结果（查看进度可设 SHIFT_PRINT_ONLY=1，只打印不写产物）')
OK3 = [m for m in SHIFT_METHODS if status.get(f'{TAG}/{m}') == 'ok']
if not OK3:
    sys.exit('第三轮没有通过核对的方法')

# ---------- 6. 第二轮：各方法的逐折结果（字节已与 47 完成标记核对）----------
r2 = pd.concat([read_csv_bytes(SNAP[p], dtype=ns['STR_COLS']) for p in ROUND2_SUMMARY.values()], ignore_index=True)
r2['seed'] = r2.seed.astype(int)
r2['fold'] = r2.fold.astype(int)
for _m in ORDER:
    _a = r2[r2.method == _m]
    if set(zip(_a.seed, _a.fold, strict=True)) != ns['EXPECTED_KEYS'] or len(_a) != ns['J'] or \
            set(_a.basis.astype(str)) != {ROUND2_BASIS[_m]}:
        sys.exit(f'第二轮 {_m} 的汇总不是 {ns["J"]} 个（种子, 折）各一行')


# ---------- 7. 表 1：两轮并列，第二轮的行须与 47 的表 1 一致 ----------
def t1_row(a, m, rnd):
    """与 47 表 1 同一汇总口径。"""
    per_seed = a.groupby('seed').R2.mean()
    return dict(round=rnd, method=m, basis=a.basis.iloc[0], n_folds=len(a), n_seeds=a.seed.nunique(),
                R2=a.R2.mean(), R2_sd=a.R2.std(ddof=1), RMSE=a.RMSE.mean(), RMSE_sd=a.RMSE.std(ddof=1),
                RPD=a.RPD.mean(), RPD_sd=a.RPD.std(ddof=1), MAE=a.MAE.mean(), BIAS=a.BIAS.mean(),
                seed_means=';'.join(f'{v:.4f}' for v in per_seed.values),
                seed_range=float(per_seed.max() - per_seed.min()), device=a.device.iloc[0])


rows = []
t1_47 = read_csv_bytes(SNAP[SRC47.path('table1')])
t1_47 = t1_47[t1_47.analysis == '主分析（B 样条）'].set_index('method')
t1_fail = []
for _m in OK3:
    _r2row = t1_row(r2[r2.method == _m], _m, '第二轮')
    _ref = t1_47.loc[_m]
    _off = [c for c in T1_NUM if not abs(float(_r2row[c]) - float(_ref[c])) <= 1e-12]
    if _off or int(_ref.n_folds) != _r2row['n_folds'] or str(_ref.seed_means) != _r2row['seed_means']:
        t1_fail.append(f'{_m}：按 47 口径重算的第二轮表 1 行与 47 的表 1 不同 {_off}')
    rows += [_r2row, t1_row(d3[d3.method == _m], _m, TAG)]
t1 = pd.DataFrame(rows)
print('【表 1】两轮并列（均值 ± 样本标准差，25 个外层折）')
for _, _r in t1.iterrows():
    print(f'  {_r.method:<7} {_r["round"]:<4} R²={_r.R2:.4f}±{_r.R2_sd:.4f}  RMSE={_r.RMSE:.4f}±{_r.RMSE_sd:.4f}  '
          f'RPD={_r.RPD:.2f}±{_r.RPD_sd:.2f}  各次重复 {_r.seed_means}（极差 {_r.seed_range:.4f}）')
if t1_fail:
    print('【表 1 复算未通过】\n  ' + '\n  '.join(t1_fail))

# ---------- 8. 跨轮核对 ----------
xrows: list[dict] = []
xprob: list[str] = []
CACHES_USED: dict[str, dict] = {}
for _m in OK3:
    _label = _m.replace('-', '')
    _c2_identified = [(p, identity) for p in (M47.get('caches') or {})
                      if (identity := r54.cache_identity(p)) is not None
                      and identity[:3] == ('04outputs', 'none', _label)]
    _c2 = sorted(p for p, _identity in _c2_identified)
    _fp2 = {identity[3] for _p, identity in _c2_identified}
    if not _c2 or len(_fp2) != 1:
        xprob.append(f'{_m}：47 完成标记中没有唯一一套第二轮格点缓存（{_c2}）')
        continue
    _stale = [p for p in _c2 if p not in SNAP or nm.sha256_bytes(SNAP[p]) != M47['caches'][p]['sha256']]
    if _stale:
        xprob.append(f'{_m}：第二轮格点缓存不存在或与 47 完成标记记录的哈希不符 {_stale}')
        continue
    _v2, _p2, _rec2 = r54.load_cache([frozen_path(p) for p in _c2], method_key=_m, memo=_fp2.pop(), grid=grids2.GRIDS[_m], seeds=ns['SEEDS'],
                                     k=K, n_test=N_TE)
    _memo3 = EXPECTED_MEMO[('none', 'SVR')] if _m == 'SVR' else SHIFT_CUDA_MEMO_FINGERPRINT
    _c3 = sorted(f'{r54.WORK}/{p}' for p in CACHES if os.path.basename(p).startswith(f'45memo_none_{_label}_'))
    _v3, _p3, _rec3 = r54.load_cache([frozen_path(p) for p in _c3], method_key=_m, memo=_memo3, grid=gbs.GRIDS[_m], seeds=ns['SEEDS'], k=K,
                                     n_test=N_TE)
    _pk2 = {(int(r.seed), int(r.fold)): str(r.picked) for r in r2[r2.method == _m].itertuples()}
    _pk3 = {(int(r.seed), int(r.fold)): str(r.picked) for r in d3[d3.method == _m].itertuples()}
    _exact = _m == 'SVR'
    _rows, _pr = r54.crossround(_v2, _v3, grid2=grids2.GRIDS[_m], grid3=gbs.GRIDS[_m], picked2=_pk2, picked3=_pk3,
                                seeds=SEED_LIST, k=K, exact=_exact)
    for _row in _rows:
        _same = _row['n_missing'] == 0 and _row['n_equal'] == _row['n_points']
        xrows.append(dict(method=_m, gated=_exact, **_row,
                          verdict=('JSON 解析后逐元素数值严格相等' if _same else '不一致') if _exact else '服务器 GPU 上运行，只列不判'))
    xprob += [f'{_m}：{q}' for q in _p2 + _p3 + _pr]
    CACHES_USED.update({p: _rec2[frozen_path(p)] for p in _c2})
    CACHES_USED.update({p: _rec3[frozen_path(p)] for p in _c3})
xr = pd.DataFrame(xrows, columns=['method', 'gated', 'kind', 'candidate', 'n_points', 'n_missing', 'n_equal',
                                  'max_abs_diff', 'verdict'])
print('【跨轮核对】两轮共有候选的内层 MSE 与两轮选中同一候选的外层折的重训练预测')
for _, _r in xr.iterrows():
    print(f'  {_r.method:<7} {_r.kind:<5} {_r.candidate:<70} 共 {_r.n_points:>2} 项  缺 {_r.n_missing}  '
          f'JSON 解析后逐元素数值严格相等 {_r.n_equal:>2}  最大差 {_r.max_abs_diff:.2e}  → {_r.verdict}')
if xprob:
    print('【跨轮核对未通过】\n  ' + '\n  '.join(xprob))
if (xprob or t1_fail) and not PRINT_ONLY:
    sys.exit('跨轮核对或表 1 复算未通过，须先查因，不写任何产物')

# ---------- 9. 检验：F1（两轮之差）与 F2（第三轮网格下的 15 对），F2 并列 47 主族 ----------
tests = []
for _m in OK3:
    tests.append(dict(family='F1 第三轮−第二轮', pair=f'{_m}：第三轮 − 第二轮',
                      **corrected_test(fold_vec(d3, _m), fold_vec(r2, _m))))


def round_frame(m):
    """F2 中方法 m 取哪一轮的结果：触发平移的两法取第三轮，其余四法取第二轮。"""
    return d3 if m in SHIFT_METHODS else r2


if OK3 == SHIFT_METHODS:
    for _a, _b in itertools.combinations(ORDER, 2):
        tests.append(dict(family='F2 第三轮网格', pair=f'{_a} − {_b}',
                          **corrected_test(fold_vec(round_frame(_a), _a), fold_vec(round_frame(_b), _b))))
else:
    print(f'\n注意：{sorted(set(SHIFT_METHODS) - set(OK3))} 未通过核对或尚未跑完，F2 须等两法齐备')
tt = pd.DataFrame(tests)
tt['holm_p'] = np.nan
for _fam in tt.family.unique():
    _sel = tt.family == _fam
    tt.loc[_sel, 'holm_p'] = holm(tt.loc[_sel, 'p'].to_numpy())
tt['family_size'] = tt.groupby('family').pair.transform('count')
tt['结论'] = np.where(tt.holm_p < 0.05, '差异显著', '未检出差异')
_main = read_csv_bytes(SNAP[SRC47.path('paired_tests')])
_main = _main[_main.family == '主分析'].set_index('pair')
_is_f2 = tt.family.str.startswith('F2')
tt['main_holm_p'] = [float(_main.holm_p[p]) if f2 else np.nan for p, f2 in zip(tt.pair, _is_f2, strict=True)]
tt['main_结论'] = [str(_main['结论'][p]) if f2 else '' for p, f2 in zip(tt.pair, _is_f2, strict=True)]
tt['same_decision'] = [((h < 0.05) == (mh < 0.05)) if f2 else np.nan
                       for h, mh, f2 in zip(tt.holm_p, tt.main_holm_p, _is_f2, strict=True)]
tt = tt[list(ns['TEST_COLS']) + ['main_holm_p', 'main_结论', 'same_decision']]
if _is_f2.any() and len(tt[_is_f2]) != 15:
    sys.exit(f'F2 应为 15 对，实为 {int(_is_f2.sum())} 对')
print(f'\n【检验】修正重复 k 折 t 检验（J={ns["J"]}，n_te/n_tr={ns["RHO"]}，df={ns["J"] - 1}），Holm 按族校正')
for _, _x in tt.iterrows():
    _extra = (f'  [Shapiro p={_x.shapiro_p:.3f}，Wilcoxon p={_x.wilcoxon_p:.4f}，仅描述]'
              if _x.shapiro_p < 0.05 else '')
    _cmp = (f'  | 主族 Holm p={_x.main_holm_p:.4f} {_x["main_结论"]}{"" if _x.same_decision else "  ← 判定不同"}'
            if str(_x.family).startswith('F2') else '')
    print(f'  {_x.family:<12}{_x.pair:<22}{_x["diff"]:>+9.4f}   [{_x.ci_lo:+.4f}, {_x.ci_hi:+.4f}]'
          f'{_x.p:>9.4f}{_x.holm_p:>9.4f}{_x.a_wins:>4d}/{_x.n_pairs}  {_x.结论}{_cmp}{_extra}')

# ---------- 10. 第三轮网格的端点诊断（口径同 47）----------
brows = []
for _m in OK3:
    _a = d3[d3.method == _m]
    _grid = gbs.GRIDS[_m]
    _gstr = {str(c): c for c in _grid}
    _picked = [_gstr[str(pk)] for pk in _a.picked.astype(str)]       # 选中配置属于网格已在核对②中确认
    for _dim in ns['dims_of'](_m, _grid[0]):
        _vals: list = []
        for _c in _grid:
            _v = ns['dims_of'](_m, _c)[_dim]
            if _v not in _vals:
                _vals.append(_v)
        _ranges = {str(v): r47.grid_value_range(_m, _dim, v, n_features=P_FEATURES, scale_gammas=SCALE_GAMMAS)
                   for v in _vals}
        _lo, _hi = r47.grid_extremes(_ranges)
        _pv = [str(ns['dims_of'](_m, c)[_dim]) for c in _picked]
        brows.append(dict(analysis=TAG, method=_m, dim=_dim, n_folds=len(_a), grid_values=str(_vals),
                          value_note=ns['value_note'](_m, _dim, _ranges),
                          at_min=int(sum(v == _lo for v in _pv)) if _lo is not None else np.nan,
                          at_max=int(sum(v == _hi for v in _pv)) if _hi is not None else np.nan,
                          picked_counts='; '.join(f'{k}×{v}' for k, v in
                                                  pd.Series([str(x) for x in _pv]).value_counts().items())))
bd = pd.DataFrame(brows)
print('\n【网格端点诊断】第三轮网格，逐维统计选中值落在数值最小 / 最大端点的折数')
for _, _r in bd.iterrows():
    _lo_s = '不判' if pd.isna(_r.at_min) else f'{int(_r.at_min)}/{_r.n_folds}'
    _hi_s = '不判' if pd.isna(_r.at_max) else f'{int(_r.at_max)}/{_r.n_folds}'
    print(f'  {_r.method:<7}{_r.dim:<6} 最小端 {_lo_s}  最大端 {_hi_s}  选中分布 {_r.picked_counts}'
          + (f'  （{_r.value_note}）' if _r.value_note else ''))
_half = bd[(bd.at_min > bd.n_folds / 2) | (bd.at_max > bd.n_folds / 2)]
print('  过半落在同一端点的维度：' + ('无' if _half.empty else '; '.join(f'{r.method} {r.dim}' for _, r in _half.iterrows()))
      + '（按协议如实报告，网格平移以第三轮为限）' * (not _half.empty))

# ---------- 11. 发布 ----------
if PRINT_ONLY:
    sys.exit('SHIFT_PRINT_ONLY=1：只打印，不写产物')
_busy = r47.activity(r54.WORK)[0] + r47.activity('.')[0]
if _busy:
    sys.exit('全部核对通过，但 45 的输入正在被改写，不发布；待其结束后重跑：\n  ' + '\n  '.join(_busy))
_info = dict(status=status, problems=problems,
             fingerprints={m: str(d3[d3.method == m].fingerprint.iloc[0]) for m in SHIFT_METHODS},
             memo_fingerprints={'SVR': EXPECTED_MEMO[('none', 'SVR')], '1D-CNN': SHIFT_CUDA_MEMO_FINGERPRINT},
             caches=CACHES_USED, family_sizes={str(k): int(v) for k, v in tt.groupby('family').size().items()})
try:
    _mp = r54.publish({'table1': t1, 'paired_tests': tt, 'crossround': xr, 'boundary': bd, 'provenance': pv},
                      snapshot_bytes=SNAP, code=CODE54, info=_info)
except RuntimeError as e:
    sys.exit(str(e))
print('\n→ 正式结果：' + ' / '.join(r54.out_path(n) for n in r54.NAMES) + f'\n→ 标记 {_mp}')
