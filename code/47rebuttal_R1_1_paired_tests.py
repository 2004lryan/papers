# -*- coding: utf-8 -*-
"""
47rebuttal_R1_1_paired_tests.py: 表 1 的修正重复 k 折 t 检验、Holm 校正与置信区间（导师意见 1、4，预注册 B10.2）

为什么要有这个脚本：原稿把「配对检验未达显著」写成了「不低于」「等精度降维」「同一水平」，
把「不能拒绝零假设」当成了「接受零假设」。本脚本按 B10.2 在 5 折 × 5 次重复的 25 个外层折上
重做全部 15 对比较，并把报告口径做成机械规则：Holm 校正后 p<0.05 才写「差异显著」，
否则一律写「未检出差异」并给出差值的 95% 置信区间，R² 上不作任何等价或非劣判断。

统计口径（B10.2）：
  单位      同一次重复、同一外层折上两个方法的 R² 之差，J = k·r = 25
  检验      修正重复 k 折 t 检验（Nadeau & Bengio 2003；Bouckaert & Frank 2004）
            SE = √((1/J + n_te/n_tr)·σ̂²_d)，n_te/n_tr 取 k 折的训练/测试比例 1/(k−1) = 0.25，自由度 J−1 = 24，双侧
            （协议 B10.10 清洗后 1 111 个样品，5 折的测试折为 223、222、222、222、222 个，各折比例 0.251 与 0.250）
  置信区间  d̄ ± t_{0.975,24}·SE（外层训练折两两重叠 3/4，折级差值不独立，不用折级 bootstrap）
  多重比较  主分析 15 对一族，Holm；基函数敏感性（FPLR、FPCR 各自 B 样条减傅里叶）另成一族 2 对，Holm
  非正态    25 个差值 Shapiro–Wilk p<0.05 时另列 Wilcoxon 符号秩 p，仅描述，不用于论断
另附：各方法 25 折的均值 ± 样本标准差、每次重复的均值、网格端点选中次数（7.3 边界诊断；非数值取值按 scikit-learn
拟合时实际使用的数值排序，SVR 的 gamma='scale' 为 1/(特征数×训练集方差)，RF 的特征比例按实际特征数），
以及种子 20041210 与 25nested_cv.csv 的逐折对照。

输入核对（每个（分析, 方法）逐项核对；未通过的方法不进入任何产物，有任一方法未通过即不出正式结果）：
  ① 覆盖    恰好 5 个种子 × 5 折、无重复；04outputs 下不得有预期之外的 45 产物，文件里的基与方法标签须与文件名相符
  ② 汇总行  各折 n_train、n_test 与 KFold(5) 在 1 111 个样品上的划分相同（测试折 223 或 222 个）；数值列有限；
            选中配置属于 grids.py 该方法的网格；
            设备为 cpu，1D-CNN 为 cuda（B10.1）；结果指纹唯一
  ③ 预测行  方法、基、设备、结果指纹与汇总行相同；每折样品集合恰为 KFold(5, shuffle=True, seed) 的该折测试集；
            预测有限；y 与原始数据集真值逐样品一致；由数据集真值与逐样品预测、按 45 自己的 fold_metrics()
            （results47.fold_metrics45()）重算该折的 R²、RMSE、RPD、MAE、P90、BIAS、两项基准指标与训练/测试集标准差，
            逐项与汇总行一致（相对差 ≤1e-9；表 1 与稿件的数字取自这些列）
  ④ 来源清单（ncv_manifest.py，45 在 5 个种子全部完成后写出）
            清单记录的汇总与预测文件的路径、SHA-256、行数与读入的相同；缓存指纹等于本轮应有的值：本机运行的方法（PLSR、
            SVR、两族基的 FPLR 与 FPCR）为按 45 的 fingerprint() 定义以本机数据、网格与环境重算的值
            （results47.memo_fingerprint45）；服务器上运行的 1D-CNN 与 RF 为服务器分片写出的值 CUDA_MEMO_FINGERPRINT、
            RF_MEMO_FINGERPRINT；结果指纹等于由该缓存指纹与当前 45 脚本、清单模块的 SHA-256 算出的值，并与汇总行、预测行
            的指纹相同；RF 的并行线程数为 RF_N_JOBS；代码（45 全文、16 共享段、grids.py、清单模块）与当前文件相同；
            真值、预处理后光谱、基系数的 SHA-256 与本脚本由原始数据重算的相同（服务器上运行的方法的预处理后光谱为服务器上的
            那一份 SERVER_XP_SHA256，见该常量的注释）；网格、种子、折数、L 与当前相同；运行环境：本机运行的方法与本机当前
            环境相同，1D-CNN 与声明的服务器环境 CUDA_ENV 相同，RF 与 SERVER_CPU_ENV 相同。
            缓存指纹相同说明结果是按本轮的数据、代码、网格与环境配置写出的，不说明结果数值来自哪份缓存，后者见 ⑥
  ⑤ 可复现性门禁（B10.1）  种子 20041210 与 25nested_cv.csv 逐折对照：PLSR、SVR 与傅里叶基 FPLR、FPCR
            须逐折 R² 差 <1e-9 且选中配置相同，否则退出查因、不写任何产物；1D-CNN（参照在本机 CPU、新结果在服务器 GPU）
            与 RF（参照在本机 arm64、新结果在服务器 x86_64，浮点差异使树的分裂不同，见 SERVER_CPU_ENV 的注释）只列不判
  ⑥ 格点缓存复核  按本轮应有的缓存指纹找到 45 写出的格点缓存 04outputs/45memo_<基>_<方法>_<缓存指纹>*.jsonl
            （服务器上运行的方法的由 cnn_fetch.sh 从服务器取回），只采用方法与指纹相符、键与取值合格的记录（与 45 读缓存的规则相同），
            同一键取值不同即不通过；按 45 的选参规则（候选按网格顺序，内层 5 折 RMSE 的均值严格更小者胜）复算 25 个外层折
            的选中配置与内层 RMSE，须与汇总行相同（差 ≤1e-12）；逐样品预测须等于缓存里该折选中配置的重训练预测
            （差 ≤1e-12）。核对过的缓存文件的 SHA-256 与采用的记录数写入标记备查（下游读取时不重读缓存）。
            ①~⑥ 查出的是陈旧、混杂、残缺的结果与汇总出错；缓存与结果被有意一并伪造的情形，文件核对无从排除，须重算

产物的发布（results47.py）：全部核对与计算都在写任何文件之前完成。
  正式  各方法均通过①~⑥时，先写入 04outputs 下的临时目录，删除旧完成标记，逐个原子替换到正式路径，
        最后原子写 04outputs/47rebuttal_COMPLETE.json（输入、代码与产物的 SHA-256，各方法状态）。
  部分  PT_ALLOW_PARTIAL=1 且有方法缺失、未跑完或未通过核对时，只用通过核对的方法写到 04outputs/47partial/，
        标记 47partial_PARTIAL.json 记下各方法状态；正式路径一概不写。
  每次运行开始时，正式路径上没有有效完成标记的产物一律删除；下游只经 results47.resolve() 读取。
  运行锁    全程持有 04outputs/.47run.lock；锁被占用（另一个 47 在运行，或 cnn_fetch.sh 正在发布取回的结果）时直接退出。
            本机有工作目录为本项目（或读不到工作目录）的 45 进程、或列不出本机进程时（results47.activity()），
            正式运行不开始、部分模式照常但不发布正式结果；本机 RF 的防重复启动锁与取回暂存目录只作提示；
            发布前重读输入与代码，与开始时的快照不同即不发布。

运行方式:
    cd <项目根目录>
    python3 02code/47rebuttal_R1_1_paired_tests.py
    PT_ALLOW_PARTIAL=1 python3 02code/47rebuttal_R1_1_paired_tests.py     （尚有方法未完成时查看进度、生成草稿用）

输出文件（正式结果；部分结果为 04outputs/47partial/47partial_<名>.csv 与 47partial_PARTIAL.json）:
    04outputs/47rebuttal_table1.csv        — 各方法（主分析与傅里叶敏感性）的汇总指标
    04outputs/47rebuttal_paired_tests.csv  — 15 对主分析与 2 对基函数敏感性的检验结果
    04outputs/47rebuttal_boundary.csv      — 各方法选中值落在网格端点的折数
    04outputs/47rebuttal_repro.csv         — 种子 20041210 与表 1 旧结果的逐折对照
    04outputs/47rebuttal_provenance.csv    — 逐（分析, 方法, 种子, 折）的真值溯源核对
    04outputs/47rebuttal_COMPLETE.json     — 完成标记
    05logs/47rebuttal_R1_1_paired_tests_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

只读 05data：真值、预处理后光谱与基系数由原始数据集重算（执行 16 脚本的共享段，与 45 同一实现），逐样品核对
45 预测文件里的 y 并以数据集真值重算逐折指标，不以预测文件携带的 y 作为真值来源；核对结果写入 provenance 产物。
"""
import glob
import io
import itertools
import json
import os
import re
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')      # 须先于 numpy 与 torch 的导入

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

sys.path.insert(0, '02code')
from grids import GRIDS  # noqa: E402
import ncv_manifest as nm  # noqa: E402
import results47 as r47  # noqa: E402

K, R, N = 5, 5, 1111           # N：协议 B10.10 数据清洗后的样品数（2018 年 537、2019 年 574）
# KFold(5) 把 N 个样品分成 N//5 或 N//5+1 个的测试折（余数分给前几折）：1 111 个时为 223、222、222、222、222
N_TE = {f: N // K + (1 if f < N % K else 0) for f in range(K)}
N_TR = {f: N - N_TE[f] for f in range(K)}
J = K * R
# 修正项取 k 折的训练/测试比例 1/(k−1)（Bouckaert & Frank 2004）；清洗前 260/1 040 与此相同，清洗后各折为 0.251 与 0.250
RHO = 1 / (K - 1)
L_BASIS = 80
ORDER = ['SVR', 'FPLR', 'FPCR', 'PLSR', '1D-CNN', 'RF']
KEY = {'FPLR': 'PFLR'}                    # 输出标签 → grids.py 的键
SEED_LIST = [20041210, 20240404, 20060515, 20251203, 19810915]
SEEDS = set(SEED_LIST)
EXPECTED_KEYS = {(sd, f) for sd in SEEDS for f in range(K)}
ALLOW_PARTIAL = os.environ.get('PT_ALLOW_PARTIAL', '0') == '1'
# 每个（分析, 方法）应来自哪个基的 45 产物
EXPECTED = {('主分析', m): ('bspline' if m in ('FPLR', 'FPCR') else 'none') for m in ORDER}
EXPECTED.update({('傅里叶敏感性', m): 'fourier' for m in ('FPLR', 'FPCR')})
TAG_OF_BASIS = {'none': '主分析', 'bspline': '主分析', 'fourier': '傅里叶敏感性'}
LABEL_OF_STEM = {m.replace('-', ''): m for m in ORDER}
FILE_RE = re.compile(r'^04outputs/45rebuttal_ncv_([a-z]+)_([A-Za-z0-9]+)(\.csv|_pred\.csv|_manifest\.json)$')
# 服务器上运行的方法只认这一套服务器环境（B10.7 与 C3 的设备分工；35export_code.py 的 requirements 说明同此）：
# 1D-CNN 在 GPU 上（CUDA_ENV），RF 在 CPU 上（SERVER_CPU_ENV，即 CUDA_ENV 去掉 GPU 三项）。
# RF 于 2026-09-14 由本机改到服务器。服务器与本机对同一内层格点算出的 MSE 相差 6.5e−3~1.4e−2，
# 服务器上另装与本机相同的 scikit-learn 1.6.1 时差值不变，差异来自两种指令集的浮点运算使树的分裂不同，换环境无从消除；
# 本机 45 已算完的种子 20041210 前 4 折与 25nested_cv.csv 逐折 R² 差为 0、选中配置相同（05logs/rf_platform_check_*.log）。
SERVER_CPU_ENV = {'platform': 'Linux x86_64', 'python': '3.10.21', 'numpy': '2.2.6', 'scipy': '1.15.3', 'sklearn': '1.7.2',
                  'pandas': '2.3.3', 'torch': '2.11.0+cu128'}
CUDA_ENV = {'platform': 'Linux x86_64', 'python': '3.10.21', 'numpy': '2.2.6', 'scipy': '1.15.3', 'sklearn': '1.7.2',
            'pandas': '2.3.3', 'torch': '2.11.0+cu128', 'cuda': '12.8', 'cudnn': 91900,
            'gpu': 'NVIDIA GeForce RTX 4090'}
assert {k: v for k, v in CUDA_ENV.items() if k not in ('cuda', 'cudnn', 'gpu')} == SERVER_CPU_ENV
# 服务器上读到的原始光谱、真值与折下标与本机逐位相同，而预处理后光谱 Xp 经 Savitzky–Golay 求导的浮点舍入随指令集不同
# （协议 B10.10 清洗后的数据，本机 arm64 与服务器 x86_64：2 275 328 个元素中 2 273 132 个末位不同，最大绝对差 1.19e−15，
# |Xp| 最大 0.105；2026-09-25 服务器新实例上由 cnn_upload.sh 核对，记入 05logs/45rebuttal_R1_4_repeated_ncv_RF-upload_*.log）。
# 部署前由 02code/deploy45/cnn_upload.sh 逐值核对（容差 1e−12，记入 05logs/45rebuttal_R1_4_repeated_ncv_<方法>-upload_*.log）；
# 服务器上运行的方法（1D-CNN、RF）的清单里的 Xp 哈希只认服务器上的这一份，真值 y 仍须与本机逐位相同。
SERVER_XP_SHA256 = 'a468cf211cc2eb27b394aa24722608261d48dc8863c022be93cea9890598d925'
# 1D-CNN 的缓存指纹由服务器上的 45 分片进程算出（其中含服务器的 Xp 字节与完整 Python 版本串，本机无法重算）。
# 取自 2026-09-25 00:58（服务器时间）起 8 个分片在清洗后数据上写出的格点缓存文件名
# 04outputs/45memo_none_1DCNN_12eff7a003a54b08_shard{0..7}.jsonl（服务器的项目目录下）；cnn_fetch.sh 取回时核对清单与之相同。
CUDA_MEMO_FINGERPRINT = '12eff7a003a54b08'
# RF 的缓存指纹同样由服务器上的 45 分片进程算出。取自 2026-09-25 00:54（服务器时间）起 7 个分片在清洗后数据上写出的格点缓存文件名
# 04outputs/45memo_none_RF_dd8d00634bac7dda_shard{0..6}.jsonl（服务器的项目目录下）；cnn_fetch.sh 取回时核对清单与之相同。
RF_MEMO_FINGERPRINT = 'dd8d00634bac7dda'
RF_N_JOBS = 2                    # RF 每个分片的并行线程数：服务器上 NCV_METHOD=RF 以 RNCV_JOBS=2 运行（deploy45/ncv_method.sh）
# 服务器上运行的方法 → （设备, 环境, 缓存指纹）
SERVER_METHODS = {'1D-CNN': ('cuda', CUDA_ENV, CUDA_MEMO_FINGERPRINT), 'RF': ('cpu', SERVER_CPU_ENV, RF_MEMO_FINGERPRINT)}

STR_COLS = {c: str for c in ('basis', 'method', 'picked', 'fingerprint', 'device')}
NUM_COLS = ['inner_rmse', 'R2', 'RMSE', 'RPD', 'MAE', 'P90', 'BIAS', 'base_MAE', 'base_P90', 's_y_train', 's_y_test']
METRIC_COLS = NUM_COLS[1:]            # 由逐样品预测重算的列（inner_rmse 由格点缓存复核，见 ⑥）
PRED_COLS = {'basis', 'method', 'seed', 'fold', 'idx', 'y', 'pred', 'fingerprint', 'device'}


def read_csv_bytes(b, **kw):
    return pd.read_csv(io.BytesIO(b), encoding='utf-8-sig', **kw)


def brief(v, n=160):
    s = json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
    return s if len(s) <= n else s[:n] + '…'


# ---------- 0. 运行锁、清理与输入快照：此后一切核对与计算只用这份字节，完成标记记下的就是它们的哈希 ----------
_RUN_LOCK = r47.acquire_run_lock()
if _RUN_LOCK is None:
    sys.exit(f'47 的运行锁 {r47.RUN_LOCK} 被占用（另一个 47 在运行，或 cnn_fetch.sh 正在发布取回的结果），本次不运行')
BUSY, NOTES = r47.activity()
if BUSY and not ALLOW_PARTIAL:
    sys.exit('47 的输入正在被改写，正式运行须等其结束：\n  ' + '\n  '.join(BUSY))
if BUSY:
    print('【注意】47 的输入正在被改写，本次只出部分结果，且可能随即失效：\n  ' + '\n  '.join(BUSY))
if NOTES:
    print('【提示】以下目录只作提示，不影响本次运行：\n  ' + '\n  '.join(NOTES))
_removed = r47.invalidate_stale_final()
if _removed:
    print('正式路径上的旧产物没有有效完成标记，已删除：\n  ' + '\n  '.join(_removed))
SNAP = r47.snapshot_inputs()
CODE47 = r47.code_sha()
CODE_NOW = nm.code_sha256()

# ---------- 1. 原始数据：真值、预处理后光谱与两族基系数的哈希（执行 16 脚本的共享段，与 45 同一实现）----------
_ns: dict = {}
exec(nm.shared16_text(), _ns)
Y_DATA = np.asarray(_ns['y'], dtype=float)
if len(Y_DATA) != N:
    sys.exit(f'原始数据集读入 {len(Y_DATA)} 个样品，应为 {N}')
COEF_NOW = {_b: nm.basis_system(_ns, _b, L_BASIS)[3] for _b in ('bspline', 'fourier')}
DATA_NOW = {'none': nm.data_sha256(_ns['y'], _ns['Xp'])}
for _b in ('bspline', 'fourier'):
    DATA_NOW[_b] = nm.data_sha256(_ns['y'], _ns['Xp'], COEF_NOW[_b])
ENV_NOW = nm.env_info('cpu')
# 本机运行的方法应有的缓存指纹：按 45 自己的 fingerprint() 定义，以本机数据、网格与环境重算
EXPECTED_MEMO: dict[tuple[str, str], str] = {}
try:
    for _b, _k in (('none', 'PLSR'), ('none', 'SVR'), ('bspline', 'PFLR'), ('bspline', 'FPCR'),
                   ('fourier', 'PFLR'), ('fourier', 'FPCR')):
        EXPECTED_MEMO[(_b, _k)] = r47.memo_fingerprint45(dict(
            np=np, torch=_ns['torch'], y=_ns['y'], Xp=_ns['Xp'], COEF={L_BASIS: COEF_NOW[_b]} if _b in COEF_NOW else {},
            L_BASIS=L_BASIS, K_OUTER=K, K_INNER=K, GRIDS=GRIDS, _SHARED=nm.shared16_text(),
            BASIS_NAME=_b if _b in COEF_NOW else 'bspline', FUNCTIONAL=('PFLR', 'FPCR'), N_JOBS=RF_N_JOBS,
            DEVICE='cpu'), _k)
except Exception as e:  # noqa: BLE001 — 任何失败都意味着核对无从进行，停下
    sys.exit(f'无法按 45 的 fingerprint() 定义重算缓存指纹：{type(e).__name__}: {e}')
print('本轮应有的缓存指纹  ' + '  '.join(f'{b}/{k}={v}' for (b, k), v in EXPECTED_MEMO.items())
      + ''.join(f'  none/{m}={v[2]}（服务器）' for m, v in SERVER_METHODS.items()))
print(f'本机环境 {brief(ENV_NOW, 400)}')
print(f'数据哈希 y={DATA_NOW["none"]["y"][:16]}  Xp={DATA_NOW["none"]["Xp"][:16]}  '
      f'COEF(bspline)={DATA_NOW["bspline"]["COEF"][:16]}  COEF(fourier)={DATA_NOW["fourier"]["COEF"][:16]}')

# ---------- 2. 读入 45 的汇总：只收预期的（基, 方法），文件里的标签须与文件名相符 ----------
status: dict[str, str] = {}               # '主分析/RF' → ok / missing / incomplete / invalid；多余文件记为 '文件/<路径>'
problems: list[str] = []
label_bad: dict[tuple[str, str], str] = {}
frames = []
for _p, _bytes in SNAP.items():
    if _p == r47.REFERENCE:
        continue
    _mt = FILE_RE.match(_p)
    _m = LABEL_OF_STEM.get(_mt.group(2), '') if _mt else ''
    _tag = TAG_OF_BASIS.get(_mt.group(1), '') if _mt else ''
    if not _mt or EXPECTED.get((_tag, _m)) != _mt.group(1):
        status[f'文件/{_p}'] = 'invalid'
        problems.append(f'04outputs 下有预期之外的 45 产物：{_p}')
        continue
    if _mt.group(3) != '.csv':
        continue
    _df = read_csv_bytes(_bytes, dtype=STR_COLS)
    if (not {'basis', 'method', 'seed', 'fold'} <= set(_df.columns) or set(_df.basis.astype(str)) != {_mt.group(1)}
            or set(_df.method.astype(str)) != {_m}):
        label_bad[(_tag, _m)] = f'{_tag} {_m}：{_p} 里的基或方法标签与文件名不符'
        continue
    frames.append(_df)
if not frames:
    sys.exit('尚无 45 脚本的产物')
d = pd.concat(frames, ignore_index=True)
d['seed'] = d.seed.astype(int)
d['fold'] = d.fold.astype(int)

TEST_SETS = {(sd, f): set(te.tolist()) for sd in SEEDS
             for f, (_tr, te) in enumerate(KFold(K, shuffle=True, random_state=sd).split(np.arange(N)))}
try:
    FOLD_METRICS = r47.fold_metrics45()
    _probe = FOLD_METRICS(Y_DATA[:10], Y_DATA[10:20], Y_DATA[10:20] + 0.1)
except Exception as e:  # noqa: BLE001 — 取不到 45 的指标定义即无从核对，停下
    sys.exit(f'无法取得 45 的 fold_metrics() 定义：{type(e).__name__}: {e}')
if list(_probe) != METRIC_COLS:
    sys.exit(f'45 的 fold_metrics() 返回的指标 {list(_probe)} 与 47 核对的列 {METRIC_COLS} 不同')


def manifest_problems(tag, m, btag, a, pr, csv_p, pred_p):
    """④ 来源清单的核对。"""
    mp = nm.manifest_path(csv_p)
    head = f'{tag} {m}：来源清单'
    if mp not in SNAP:
        return [f'{head}缺失（{mp}）']
    try:
        man = json.loads(SNAP[mp].decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return [f'{head}无法解析：{e}']
    if not isinstance(man, dict):
        return [f'{head}格式不符']
    key = KEY.get(m, m)
    dev, env, memo = SERVER_METHODS[m] if m in SERVER_METHODS else ('cpu', ENV_NOW, EXPECTED_MEMO[(btag, key)])
    want = {'schema': nm.SCHEMA, 'analysis_basis': btag, 'method': m, 'method_key': key,
            'memo_fingerprint': memo, 'fingerprint': nm.row_fingerprint(memo, CODE_NOW),
            'n_jobs': RF_N_JOBS if m == 'RF' else None,
            'device': dev, 'seeds': SEED_LIST, 'k_outer': K, 'k_inner': K,
            'L': L_BASIS if m in ('FPLR', 'FPCR') else None, 'grid': nm.normalized(GRIDS[key]),
            'code': CODE_NOW, 'env': env,
            'data': {**DATA_NOW['none'], 'Xp': SERVER_XP_SHA256} if m in SERVER_METHODS else DATA_NOW[btag]}
    probs = [f'{head}的 {k} 不符：清单记为 {brief(man.get(k))}，应为 {brief(v)}' for k, v in want.items()
             if json.dumps(man.get(k), sort_keys=True) != json.dumps(v, sort_keys=True)]
    files = man['files'] if isinstance(man.get('files'), dict) else {}
    for kind, path, n_rows in (('csv', csv_p, len(a)), ('pred', pred_p, len(pr))):
        f = files[kind] if isinstance(files.get(kind), dict) else {}
        if f.get('path') != path or f.get('sha256') != nm.sha256_bytes(SNAP[path]) or f.get('rows') != n_rows:
            probs.append(f'{head}记录的 {kind} 文件（路径、SHA-256 或行数）与读入的 {path} 不符')
    fps = set(a.fingerprint.astype(str)) | set(pr.fingerprint.astype(str))
    if fps != {want['fingerprint']}:
        probs.append(f'{head}：汇总行、预测行的结果指纹 {sorted(fps)} 不是本轮应有的 {want["fingerprint"]}')
    return probs


def check_method(tag, m):
    """返回（状态, 问题列表, 真值溯源行）。"""
    btag = EXPECTED[(tag, m)]
    if (tag, m) in label_bad:
        return 'invalid', [label_bad[(tag, m)]], []
    a = d[(d.basis == btag) & (d.method == m)]
    if a.empty:
        return 'missing', [f'{tag} {m}：尚无结果'], []
    keys = list(zip(a.seed, a.fold, strict=True))
    if len(keys) != len(set(keys)) or not set(keys) <= EXPECTED_KEYS:
        return 'invalid', [f'{tag} {m}：{len(keys)} 行，（种子, 折）有重复或超出预期'], []
    if set(keys) != EXPECTED_KEYS:
        return 'incomplete', [f'{tag} {m}：{len(keys)}/{J} 折，尚未跑完'], []
    probs: list[str] = []
    if a.fingerprint.isna().any() or a.fingerprint.nunique() != 1:
        probs.append(f'{tag} {m}：指纹不唯一 {sorted(a.fingerprint.astype(str).unique())}')
    if not {'n_train', 'n_test', 'picked', 'device'} <= set(a.columns):
        return 'invalid', probs + [f'{tag} {m}：汇总行缺列'], []
    if not ((pd.to_numeric(a.n_train, errors='coerce') == a.fold.map(N_TR)).all()
            and (pd.to_numeric(a.n_test, errors='coerce') == a.fold.map(N_TE)).all()):
        probs.append(f'{tag} {m}：训练/测试样本数与 KFold({K}) 的划分不符（应为 {N_TR}/{N_TE}）')
    if not set(NUM_COLS) <= set(a.columns) or not np.all(np.isfinite(a[NUM_COLS].to_numpy(dtype=float))):
        probs.append(f'{tag} {m}：数值列缺失或含非有限值')
    allowed = {str(c) for c in GRIDS[KEY.get(m, m)]}
    off = sorted(set(a.picked.astype(str)) - allowed)
    if off:
        probs.append(f'{tag} {m}：选中配置不在 grids.py 的网格内 {off}')
    want_dev = 'cuda' if m == '1D-CNN' else 'cpu'
    if set(a.device.astype(str)) != {want_dev}:
        probs.append(f'{tag} {m}：设备 {sorted(set(a.device.astype(str)))} ≠ {{{want_dev}}}')
    csv_p = f"04outputs/45rebuttal_ncv_{btag}_{m.replace('-', '')}.csv"
    pred_p = csv_p[:-4] + '_pred.csv'
    if pred_p not in SNAP:
        return 'invalid', probs + [f'{tag} {m}：缺逐样品预测文件 {pred_p}'], []
    pr = read_csv_bytes(SNAP[pred_p], dtype=STR_COLS)
    if not PRED_COLS <= set(pr.columns):
        return 'invalid', probs + [f'{tag} {m}：预测文件缺列 {sorted(PRED_COLS - set(pr.columns))}'], []
    if set(pr.method.astype(str)) != {m} or set(pr.basis.astype(str)) != {btag}:
        probs.append(f'{tag} {m}：预测文件的方法或基与汇总行不符')
    if set(pr.device.astype(str)) != set(a.device.astype(str)) or set(pr.device.astype(str)) != {want_dev}:
        probs.append(f'{tag} {m}：预测文件的设备 {sorted(set(pr.device.astype(str)))} 与汇总行不符或 ≠ {{{want_dev}}}')
    if set(pr.fingerprint.astype(str)) != set(a.fingerprint.astype(str)):
        probs.append(f'{tag} {m}：逐样品预测文件的指纹与汇总行不同')
    groups = pr.groupby(['seed', 'fold'])
    if {(int(s), int(f)) for s, f in groups.groups} != EXPECTED_KEYS:
        probs.append(f'{tag} {m}：预测文件覆盖 {len(groups)} 个（种子, 折），与预期的 {J} 个不符')
    prov = []
    for (sd, f), g in groups:
        idx = g.idx.astype(int)
        if idx.duplicated().any() or set(idx) != TEST_SETS.get((int(sd), int(f)), set()):
            probs.append(f'{tag} {m} seed={sd} fold={f}：预测的样品集合不是该折测试集')
            continue
        if not np.all(np.isfinite(g[['y', 'pred']].to_numpy(dtype=float))):
            probs.append(f'{tag} {m} seed={sd} fold={f}：预测含非有限值')
            continue
        y_true = Y_DATA[idx.to_numpy()]
        dy = float(np.max(np.abs(g.y.to_numpy(dtype=float) - y_true)))
        prov.append(dict(analysis=tag, method=m, seed=int(sd), fold=int(f), n=len(g), max_abs_y_diff=dy))
        if dy > 1e-9:
            probs.append(f'{tag} {m} seed={sd} fold={f}：预测文件的 y 与原始数据集真值不符（最大差 {dy:.2e}）')
            continue
        row = a[(a.seed == sd) & (a.fold == f)]
        if len(row) != 1:
            probs.append(f'{tag} {m} seed={sd} fold={f}：汇总行不是恰好一行')
            continue
        order = np.argsort(idx.to_numpy())                 # 与 45 相同：测试样品按下标升序
        te = idx.to_numpy()[order]
        mt = FOLD_METRICS(Y_DATA[np.setdiff1d(np.arange(N), te)], Y_DATA[te], g.pred.to_numpy(dtype=float)[order])
        off = [c for c in METRIC_COLS if not abs(mt[c] - float(row[c].iloc[0])) <= 1e-9 * max(1.0, abs(mt[c]))]
        if off:
            probs.append(f'{tag} {m} seed={sd} fold={f}：由数据集真值与逐样品预测按 45 的 fold_metrics 重算的 '
                         f'{off} 与汇总行不一致（重算 {[round(mt[c], 6) for c in off]}，'
                         f'汇总行 {[round(float(row[c].iloc[0]), 6) for c in off]}）')
    probs += manifest_problems(tag, m, btag, a, pr, csv_p, pred_p)
    if not probs:
        probs += cache_problems(tag, m, btag, a, pr)
    return ('ok' if not probs else 'invalid'), probs, prov


CACHES: dict[str, dict] = {}           # 核对过的格点缓存文件 → SHA-256 与采用的记录数（写入标记）


def cache_key(kind, sd, f, cfg):
    """与 45 的 key_inner、key_refit 相同的键。"""
    return json.dumps([kind, sd, f, cfg], sort_keys=True, ensure_ascii=False, default=str)


def cache_problems(tag, m, btag, a, pr):
    """⑥ 格点缓存复核。"""
    key = KEY.get(m, m)
    memo = SERVER_METHODS[m][2] if m in SERVER_METHODS else EXPECTED_MEMO[(btag, key)]
    pattern = f"04outputs/45memo_{btag}_{m.replace('-', '')}_{memo}*.jsonl"
    head = f'{tag} {m}：格点缓存'
    files = sorted(p for p in glob.glob(pattern) if not os.path.basename(p).startswith('._'))
    if not files:
        return [f'{head}缺失（{pattern}）']
    grid = GRIDS[key]
    cfg_ok = {json.dumps(c, sort_keys=True, default=str) for c in grid}
    values: dict[str, list] = {}
    probs = []
    for fp in files:
        with open(fp, 'rb') as fh:
            raw = fh.read()
        n_ok = n_bad = 0
        for line in raw.decode('utf-8').splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                kind, sd, f, cfg = json.loads(rec['k'])
                v = rec['v']
                valid = (rec.get('m') == key and rec.get('fp') == memo and kind in ('inner', 'refit')
                         and sd in SEEDS and f in range(K) and json.dumps(cfg, sort_keys=True, default=str) in cfg_ok
                         and isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)
                         and bool(np.all(np.isfinite(v))) and len(v) == (K if kind == 'inner' else N_TE[f]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                n_bad += 1
                continue
            if rec['k'] in values and values[rec['k']] != v:
                probs.append(f'{head}：同一键取值不同 {rec["k"]}（{fp}）')
            values[rec['k']] = v
            n_ok += 1
        CACHES[fp] = {'sha256': nm.sha256_bytes(raw), 'records': n_ok, 'skipped': n_bad}
    for sd, f in sorted(EXPECTED_KEYS):
        inner = [v for v in (values.get(cache_key('inner', sd, f, cfg)) for cfg in grid) if v is not None]
        if len(inner) != len(grid):
            probs.append(f'{head}：seed={sd} fold={f} 缺内层格点')
            continue
        best, bc = np.inf, grid[0]
        for cfg, v in zip(grid, inner, strict=True):
            e = float(np.mean(np.sqrt(v)))
            if e < best:
                best, bc = e, cfg
        row = a[(a.seed == sd) & (a.fold == f)]
        if str(bc) != str(row.picked.iloc[0]) or abs(best - float(row.inner_rmse.iloc[0])) > 1e-12:
            probs.append(f'{head}：seed={sd} fold={f} 按缓存复算的选中配置 {bc}（内层 RMSE {best:.10g}）与汇总行的 '
                         f'{row.picked.iloc[0]}（{float(row.inner_rmse.iloc[0]):.10g}）不同')
            continue
        refit = values.get(cache_key('refit', sd, f, bc))
        g = pr[(pr.seed.astype(int) == sd) & (pr.fold.astype(int) == f)]
        got = g.pred.to_numpy(dtype=float)[np.argsort(g.idx.astype(int).to_numpy())]
        if refit is None or len(refit) != len(got) or float(np.max(np.abs(got - np.asarray(refit)))) > 1e-12:
            probs.append(f'{head}：seed={sd} fold={f} 逐样品预测与缓存里选中配置 {bc} 的重训练预测不符')
    return probs


prov: list[dict] = []
for (_tag, _m) in EXPECTED:
    _st, _probs, _pv_rows = check_method(_tag, _m)
    status[f'{_tag}/{_m}'] = _st
    problems += _probs
    if _st == 'ok':
        prov += _pv_rows
FINAL = all(v == 'ok' for v in status.values())
_pv = pd.DataFrame(prov, columns=['analysis', 'method', 'seed', 'fold', 'n', 'max_abs_y_diff'])
print(f'【真值溯源】已核对 {len(_pv)} 个（分析, 方法, 种子, 折），预测文件 y 与原始数据集真值的最大绝对差 '
      f'{(_pv.max_abs_y_diff.max() if len(_pv) else float("nan")):.1e}')
print('【各方法状态】' + '  '.join(f'{k}={v}' for k, v in status.items()))
if problems:
    print('【输入缺失、未跑完或未通过核对】\n  ' + '\n  '.join(problems))
if not FINAL and not ALLOW_PARTIAL:
    sys.exit('输入不完整或未通过核对，不输出检验结果（查看进度可设 PT_ALLOW_PARTIAL=1，结果写到 04outputs/47partial/）')
OK_MAIN = [m for m in ORDER if status[f'主分析/{m}'] == 'ok']
OK_FOUR = [m for m in ('FPLR', 'FPCR') if status[f'傅里叶敏感性/{m}'] == 'ok']
if not OK_MAIN:
    sys.exit('主分析没有通过核对的方法，不输出任何结果')

# ---------- 可复现性门禁（B10.1）：种子 20041210 与表 1 旧结果逐折对照，先于任何产物写出 ----------
# 与参照同在本机 CPU 上运行的方法（PLSR、SVR 与傅里叶基的 FPLR、FPCR）必须逐折 R² 差 <1e-9 且选中配置完全相同，
# 任一不一致即停下查因、不写任何产物；服务器上运行的 1D-CNN（GPU）与 RF（x86_64 CPU）运行平台与参照不同，只列不判。
# 正式运行时上述四个方法缺任何一个都视为未通过；部分结果模式只核对已经跑出的方法（含未通过其他核对的）。
old = read_csv_bytes(SNAP[r47.REFERENCE], dtype={'picked': str})
old = old[old.seed == 20041210].replace({'method': {'PFLR': 'FPLR'}})
GATED = [('none', 'PLSR'), ('none', 'SVR'), ('fourier', 'FPLR'), ('fourier', 'FPCR')]
rrows, repro_fail = [], []
for btag, m in GATED + [('none', m) for m in SERVER_METHODS]:
    a = d[(d.basis == btag) & (d.method == m) & (d.seed == 20041210)].sort_values('fold')
    b = old[old.method == m].sort_values('fold')
    if len(a) != K or len(b) != K:
        if (btag, m) in GATED and (FINAL or len(a) > 0):
            repro_fail.append(f'{m}（基={btag}）：新 {len(a)} 折 / 旧 {len(b)} 折，无法逐折对照')
        continue
    diff = np.abs(a.R2.to_numpy(dtype=float) - b.R2.to_numpy(dtype=float))
    same_pick = bool((a.picked.astype(str).to_numpy() == b.picked.astype(str).to_numpy()).all())
    gated = (btag, m) in GATED
    ok = bool(diff.max() < 1e-9 and same_pick)
    if gated and not ok:
        repro_fail.append(f'{m}（基={btag}）：逐折 R² 最大差 {diff.max():.2e}，选中配置相同={same_pick}')
    rrows.append(dict(method=m, basis=btag, device=a.device.iloc[0], gated=gated, max_abs_diff_R2=float(diff.max()),
                      same_picked=same_pick, new_R2=a.R2.mean(), old_R2=b.R2.mean(),
                      verdict=('一致' if ok else '不一致') if gated else '运行平台不同，只列不判'))
rp = pd.DataFrame(rrows, columns=['method', 'basis', 'device', 'gated', 'max_abs_diff_R2', 'same_picked', 'new_R2',
                                  'old_R2', 'verdict'])
print('【可复现性门禁】种子 20041210 与 25nested_cv.csv 逐折对照')
for _, r in rp.iterrows():
    print(f'  {r.method:<7} 基={r.basis:<8} 设备={r.device:<5} 逐折 R² 最大差 {r.max_abs_diff_R2:.2e}  '
          f'选中配置相同={r.same_picked}  新 {r.new_R2:.4f} / 旧 {r.old_R2:.4f}  → {r.verdict}')
if repro_fail:
    sys.exit('可复现性核对未通过，须先查因，不写任何产物：\n  ' + '\n  '.join(repro_fail))

main = d[d.basis.isin(['none', 'bspline']) & d.method.isin(OK_MAIN)]
four = d[(d.basis == 'fourier') & d.method.isin(OK_FOUR)]


def fold_vec(df, m):
    a = df[df.method == m].sort_values(['seed', 'fold'])
    return a.set_index(['seed', 'fold']).R2


def corrected_test(ra, rb):
    """ra、rb 为按（种子, 折）对齐的两串 R²；返回差值、修正标准误、t、p、置信区间等。"""
    if set(ra.index) != set(rb.index) or ra.index.has_duplicates or rb.index.has_duplicates:
        raise SystemExit('配对两侧的（种子, 折）集合不一致或有重复')
    if len(ra) != J:
        raise SystemExit(f'配对数 {len(ra)} ≠ {J}')
    idx = ra.index
    dd = (ra.loc[idx] - rb.loc[idx]).to_numpy()
    n = len(dd)
    mu, var = float(dd.mean()), float(dd.var(ddof=1))
    se = float(np.sqrt((1.0 / n + RHO) * var))
    df_ = n - 1
    if var == 0:                          # 退化：差值全相同
        tval = 0.0 if mu == 0 else float(np.sign(mu) * np.inf)
        p = 1.0 if mu == 0 else 0.0
    else:
        tval = mu / se
        p = float(2 * stats.t.sf(abs(tval), df=df_))
    q = float(stats.t.ppf(0.975, df=df_))
    sw = float(stats.shapiro(dd).pvalue) if n >= 3 and var > 0 else np.nan
    wx = float(stats.wilcoxon(dd).pvalue) if sw < 0.05 and np.any(dd != 0) else np.nan
    return dict(n_pairs=n, diff=mu, se_corr=se, t=tval, df=df_, p=p, ci_lo=mu - q * se, ci_hi=mu + q * se,
                a_wins=int((dd > 0).sum()), shapiro_p=sw, wilcoxon_p=wx)


def holm(p):
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


# ---------- 表 1 汇总 ----------
rows = []
for tag, df in [('主分析（B 样条）', main), ('敏感性（傅里叶）', four)]:
    for m in ORDER:
        a = df[df.method == m]
        if a.empty:
            continue
        per_seed = a.groupby('seed').R2.mean()
        rows.append(dict(analysis=tag, method=m, basis=a.basis.iloc[0], n_folds=len(a), n_seeds=a.seed.nunique(),
                         R2=a.R2.mean(), R2_sd=a.R2.std(ddof=1), RMSE=a.RMSE.mean(), RMSE_sd=a.RMSE.std(ddof=1),
                         RPD=a.RPD.mean(), RPD_sd=a.RPD.std(ddof=1), MAE=a.MAE.mean(), BIAS=a.BIAS.mean(),
                         seed_means=';'.join(f'{v:.4f}' for v in per_seed.values),
                         seed_range=float(per_seed.max() - per_seed.min()), device=a.device.iloc[0]))
t1 = pd.DataFrame(rows)
print(f'【表 1】5 折 × {R} 次重复嵌套交叉验证，均值 ± 样本标准差（{J} 个外层折）')
for _, r in t1.iterrows():
    print(f'  {r.analysis:<10} {r.method:<7} R²={r.R2:.4f}±{r.R2_sd:.4f}  RMSE={r.RMSE:.4f}±{r.RMSE_sd:.4f}  '
          f'RPD={r.RPD:.2f}±{r.RPD_sd:.2f}  各次重复 {r.seed_means}（极差 {r.seed_range:.4f}）')

# ---------- 配对检验 ----------
tests = []
missing = [m for m in ORDER if m not in OK_MAIN]
if missing:
    print(f'\n注意：{missing} 未通过核对或尚未跑完，主分析的 15 对须等其齐备；以下只检验已通过核对的方法之间的对')
for a_, b_ in itertools.combinations(OK_MAIN, 2):
    tests.append(dict(family='主分析', pair=f'{a_} − {b_}', **corrected_test(fold_vec(main, a_), fold_vec(main, b_))))
fam_main = [i for i, x in enumerate(tests) if x['family'] == '主分析']
for m in ['FPLR', 'FPCR']:
    if m in OK_FOUR and m in OK_MAIN:
        tests.append(dict(family='基函数敏感性', pair=f'{m}：B 样条 − 傅里叶',
                          **corrected_test(fold_vec(main, m), fold_vec(four, m))))
TEST_COLS = ['family', 'pair', 'n_pairs', 'diff', 'se_corr', 't', 'df', 'p', 'ci_lo', 'ci_hi', 'a_wins', 'shapiro_p',
             'wilcoxon_p', 'holm_p', 'family_size', '结论']
tt = pd.DataFrame(tests)
if tt.empty:
    tt = pd.DataFrame(columns=TEST_COLS)
else:
    tt['holm_p'] = np.nan
    for fam in tt.family.unique():
        sel = tt.family == fam
        tt.loc[sel, 'holm_p'] = holm(tt.loc[sel, 'p'].to_numpy())
    tt['family_size'] = tt.groupby('family').pair.transform('count')
    tt['结论'] = np.where(tt.holm_p < 0.05, '差异显著', '未检出差异')
    if not FINAL or len(fam_main) != 15:
        tt['结论'] += '（输入不完整，非最终）'
    print(f'\n【配对检验】修正重复 k 折 t 检验（J={J}，n_te/n_tr={RHO}，df={J - 1}），Holm 按族校正')
    print(f'  {"族":<8}{"对比":<22}{"差值":>9}{"95% CI":>22}{"修正SE":>9}{"t":>8}{"p":>9}{"Holm p":>9}'
          f'{"前者胜":>7}  结论')
    for _, x in tt.iterrows():
        extra = f'  [Shapiro p={x.shapiro_p:.3f}，Wilcoxon p={x.wilcoxon_p:.4f}，仅描述]' if x.shapiro_p < 0.05 else ''
        print(f'  {x.family:<8}{x.pair:<22}{x["diff"]:>+9.4f}   [{x.ci_lo:+.4f}, {x.ci_hi:+.4f}]'
              f'{x.se_corr:>9.4f}{x.t:>8.2f}{x.p:>9.4f}{x.holm_p:>9.4f}{x.a_wins:>4d}/{x.n_pairs}  {x.结论}{extra}')
if FINAL and len(fam_main) != 15:
    sys.exit(f'六法均通过核对，但主分析只有 {len(fam_main)} 对，应为 15 对')


# ---------- 网格端点诊断（7.3，逐维）----------
def dims_of(m, cfg):
    """把一个候选拆成各超参数维度的取值。"""
    if m == 'PLSR':
        return {'潜变量数': cfg}
    if m == 'SVR':
        return {'C': cfg[0], 'gamma': cfg[1]}
    if m == 'RF':
        return {'树数': cfg[0], '特征比例': cfg[1]}
    if m == '1D-CNN':
        return {'宽度': cfg['w'], '层数': cfg['nl'], '学习率': cfg['lr']}
    if m == 'FPCR':
        return {'K': cfg[1]}
    return {'λβ': cfg[1]}


P_FEATURES = int(np.asarray(_ns['Xp']).shape[1])
SCALE_GAMMAS = r47.svr_scale_gammas(_ns['Xp'], SEED_LIST, K)     # 45 各次 SVR 拟合所用训练集上 gamma='scale' 的范围


def value_note(method_key, dim, ranges):
    """换算过的非数值取值的说明，写入产物与日志。"""
    if method_key == 'RF' and dim == '特征比例':
        return '实际特征数 ' + '; '.join(f'{v}={int(lo)}' for v, (lo, _hi) in ranges.items())
    if method_key == 'SVR' and dim == 'gamma':
        return f'scale 在各训练集上为 {SCALE_GAMMAS[0]:.4g}~{SCALE_GAMMAS[1]:.4g}'
    return ''


brows = []
for tag, df in [('主分析', main), ('傅里叶', four)]:
    for m in ORDER:
        a = df[df.method == m]
        if a.empty:
            continue
        grid = GRIDS[KEY.get(m, m)]
        gstr = {str(c): c for c in grid}
        picked = [gstr[str(pk)] for pk in a.picked.astype(str)]         # 选中配置属于网格已在输入核对②中确认
        for dim in dims_of(KEY.get(m, m), grid[0]):
            vals = []
            for c in grid:
                v = dims_of(KEY.get(m, m), c)[dim]
                if v not in vals:
                    vals.append(v)
            ranges = {str(v): r47.grid_value_range(KEY.get(m, m), dim, v, n_features=P_FEATURES,
                                                   scale_gammas=SCALE_GAMMAS) for v in vals}
            lo, hi = r47.grid_extremes(ranges)
            pv = [str(dims_of(KEY.get(m, m), c)[dim]) for c in picked]
            brows.append(dict(analysis=tag, method=m, dim=dim, n_folds=len(a), grid_values=str(vals),
                              value_note=value_note(KEY.get(m, m), dim, ranges),
                              at_min=int(sum(v == lo for v in pv)) if lo is not None else np.nan,
                              at_max=int(sum(v == hi for v in pv)) if hi is not None else np.nan,
                              picked_counts='; '.join(f'{k}×{v}' for k, v in
                                                      pd.Series([str(x) for x in pv]).value_counts().items())))
bd = pd.DataFrame(brows)
print('\n【网格端点诊断】逐维统计选中值落在该维数值最小 / 最大端点的折数（非数值取值按 scikit-learn 拟合时实际使用的数值'
      '排序；区间相交无从判定的一端记为「不判」）')


def _count(v, n):
    return '不判' if pd.isna(v) else f'{int(v)}/{n}'


for _, r in bd.iterrows():
    print(f'  {r.analysis:<6}{r.method:<7}{r.dim:<6} 最小端 {_count(r.at_min, r.n_folds)}  最大端 {_count(r.at_max, r.n_folds)}  '
          f'选中分布 {r.picked_counts}' + (f'  （{r.value_note}）' if r.value_note else ''))
half = bd[(bd.at_min > bd.n_folds / 2) | (bd.at_max > bd.n_folds / 2)]
print('  过半落在同一端点的维度：' + ('无' if half.empty else '; '.join(
    f'{r.analysis} {r.method} {r.dim}' for _, r in half.iterrows())))

# ---------- 发布：正式结果与部分结果分开存放，标记最后写 ----------
_fps = {f'{t}/{m}': str(d[(d.basis == EXPECTED[(t, m)]) & (d.method == m)].fingerprint.iloc[0])
        for (t, m) in EXPECTED if status[f'{t}/{m}'] == 'ok'}
_info = dict(status=status, problems=problems, fingerprints=_fps, caches=CACHES,
             family_sizes={str(k): int(v) for k, v in tt.groupby('family').size().items()} if len(tt) else {})
if FINAL and (BUSY or r47.busy_reasons()):
    sys.exit('全部方法通过核对，但 47 的输入正在被改写，不发布正式结果；待其结束后重跑：\n  '
             + '\n  '.join(BUSY or r47.busy_reasons()))
try:
    _mp = r47.publish({'table1': t1, 'paired_tests': tt, 'boundary': bd, 'repro': rp, 'provenance': _pv},
                      final=FINAL, snapshot=SNAP, code=CODE47, info=_info)
except RuntimeError as e:
    sys.exit(str(e))
print(f'\n→ {"正式结果" if FINAL else "部分结果（非最终，只含通过核对的方法）"}：'
      + ' / '.join(r47.out_path(n, FINAL) for n in r47.NAMES) + f'\n→ 标记 {_mp}')
