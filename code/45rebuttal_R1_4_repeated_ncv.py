# -*- coding: utf-8 -*-
"""
45rebuttal_R1_4_repeated_ncv.py: 表 1 的 5 折 × 5 次重复嵌套交叉验证（导师意见 4 与批注 6，预注册 B10.1）

为什么要有这个脚本：表 1 原先只用种子 20041210 做了一次 5 折嵌套交叉验证。修正 t 检验在
5 个外层折上的方差因子为 1/5 + 0.25 = 0.45、自由度仅 4，统计功效太低（37paired_r2.csv 中
SVR 相对 RF 高 0.121、5 折全胜，Holm 校正后 p 仍为 0.82）。本脚本按 B10.1 把同一协议重复
5 次（B1 去重后的 5 个种子），得到 25 个外层折，供 47rebuttal_R1_1_paired_tests.py 作修正重复
k 折 t 检验。函数型模型的主分析基为三次 B 样条 L=80，傅里叶基 L=80 作为敏感性结果并列。

协议（与 25nested_cv.py 逐项一致，只多了种子与基族两个维度）：
  预处理  SNV + SG 一阶导（窗宽 15，二次多项式），逐样本运算，划分前整体计算不构成泄漏
  外层    KFold(5, shuffle=True, random_state=seed)，六个方法在同一种子下共用同一批折
  内层    在外层训练折上 KFold(5, shuffle=True, random_state=seed)；12 组候选各算 5 折 RMSE 的
          均值，取最小者（并列取网格中靠前者）
  重训练  用整个外层训练折重拟合选中配置，在外层测试折上评价
  网格    grids.py 第二轮，六个方法各 12 组，不作改动
  函数型  RNCV_BASIS=bspline（主分析）或 fourier（敏感性），L=80，基系数由 16 脚本的 to_coef 求解
  1D-CNN  RNCV_DEVICE=cuda 时在 GPU 上训练：网络、优化器、调度器、epoch 与 16 脚本的 cnn_fit 逐项一致，
          mini-batch 排列在 CPU 上生成；开启 cudnn 确定性卷积与 warn_only 的确定性算法
          （AdaptiveAvgPool 的 CUDA 反向传播没有确定性实现，只能告警不能强制，告警照常打印进日志）

断点与缓存：每个内层格点（种子, 折, 候选）的 5 个内层 MSE、每次重训练的测试折预测，算完即追加写入
缓存文件并 fsync。缓存文件名带缓存指纹（数据、预处理后光谱、基系数、网格、基族、L、方法、16 共享段、库版本、
拟合代码的 SHA-256 前 16 位），其中任一项改变都换新文件，旧缓存不会被误用。所有拟合都是确定性的（RF 与 CNN
每次调用都以同一种子重置），命中缓存与重算得到同一个值。同一键在缓存里出现两次而取值不同即报错停下。
并行：RNCV_SHARD='i/n' 时本进程只算全局序号 ≡ i (mod n) 的内层格点，写入自己独占的分片缓存文件后退出，
多进程之间不共享任何可写文件；选择、重训练与写 CSV 由不带 RNCV_SHARD 的进程完成。该进程启动时
统计缺失的内层格点：RNCV_REQUIRE_COMPLETE=1 时有缺失即报错退出（远程分片收工后用），否则自行补算。
CSV 一律先写临时文件再原子替换。汇总行与预测行都写结果指纹（缓存指纹再加本脚本全文与 ncv_manifest.py 的
SHA-256，见该模块说明）与设备。已有 CSV 里的某个种子只有同时满足以下各条才算完成、才会被沿用：
5 折齐全；汇总行与预测行的方法、基族、结果指纹、设备都与本次运行相符；选中配置属于网格；
全部数值列有限；逐样品预测恰好覆盖每个样品一次、每折样品集合就是该折测试集；由预测重算的 R² 与
汇总行一致。任一条不满足，该种子整组丢弃重算，旧代码、旧网格或旧环境算出的结果不会被当作本次结果。
来源清单：某个（基, 方法）的 5 个种子全部完成后，按种子、折排序重写两份 CSV，读回再核一遍全部种子，
通过后写 <汇总 CSV 名>_manifest.json（结果文件、代码、数据的 SHA-256 与库版本、设备，字段见 ncv_manifest.py），
47 脚本据此认证来源。本次运行要改动某组 CSV 时先删去它的旧清单，清单因此只会描述它旁边的那两份文件。
1D-CNN 的正式结果只在 GPU 上产生（B10.1）：选中 1D-CNN 而 RNCV_DEVICE 不是 cuda 时直接退出；本机冒烟检查
须显式设 RNCV_SANITY_CPU_CNN=1，此时产物改写到 04outputs/45sanity_ncv_*，47 脚本不读取。

运行方式:
    cd <项目根目录>
    RNCV_BASIS=bspline RNCV_METHODS=PFLR,FPCR python3 02code/45rebuttal_R1_4_repeated_ncv.py
    RNCV_BASIS=fourier RNCV_METHODS=PFLR,FPCR python3 02code/45rebuttal_R1_4_repeated_ncv.py
    RNCV_METHODS=PLSR python3 02code/45rebuttal_R1_4_repeated_ncv.py（SVR 同理）
    RF 在远程服务器的 CPU 上分片运行（每个分片 RNCV_JOBS=2），由 02code/deploy45/ 下的脚本启动、取回并核对
    1D-CNN 在远程 GPU 上运行：bash 02code/deploy45/cnn_upload.sh → 服务器上在项目目录运行 bash cnn_boot.sh 8
        （8 个分片以 RNCV_SHARD=i/8 填格点，守候在分片收工后以 RNCV_REQUIRE_COMPLETE=1 汇总）
        → 本机 bash 02code/deploy45/cnn_fetch.sh 取回并核对

输出文件（每个（基, 方法）一组，由唯一一个进程写；47 脚本合并读取）:
    04outputs/45rebuttal_ncv_<基>_<方法>.csv       — 逐（种子, 折）的指标与选中配置
    04outputs/45rebuttal_ncv_<基>_<方法>_pred.csv  — 逐样品的外层测试折预测
    04outputs/45rebuttal_ncv_<基>_<方法>_manifest.json — 来源清单（5 个种子全部完成后写出）
    04outputs/45memo_<基>_<方法>_<缓存指纹>[_shard<i>].jsonl — 格点级缓存（不进 git）
    基的取值：bspline / fourier（仅 FPLR、FPCR）/ none（其余四法）
    05logs/45rebuttal_R1_4_repeated_ncv_<标签>_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

方法标签：输出一律写 FPLR（B10.6），代码内部沿用 grids.py 的键 'PFLR'。
只读 05data。
"""
from typing import Any
import glob
import hashlib
import inspect
import io
import json
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
DEVICE = os.environ.get('RNCV_DEVICE', 'cpu')
if DEVICE == 'cuda':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # 须在 CUDA 初始化之前设置
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, '02code')
import ncv_manifest as nm  # noqa: E402

# 数据、预处理、两族基函数、pflr/fpcr、CNN 网络与 cnn_fit、mk()：与表 1 共用同一份实现
_SHARED = nm.shared16_text()
exec(_SHARED)
from grids import GRIDS  # noqa: E402
CODE_SHA = nm.code_sha256()          # 本次运行的代码：45 全文、16 共享段、grids.py、ncv_manifest.py

if DEVICE == 'cuda':
    # 16 脚本开头全局忽略了告警；GPU 路径要让 torch 的确定性告警照常进日志
    warnings.filterwarnings('default', category=UserWarning, module=r'torch.*')

BASIS_NAME = os.environ.get('RNCV_BASIS', 'bspline')
assert BASIS_NAME in ('bspline', 'fourier'), BASIS_NAME
ALL_METHODS = ['PLSR', 'SVR', 'RF', '1D-CNN', 'FPCR', 'PFLR']
METHODS = os.environ.get('RNCV_METHODS', ','.join(ALL_METHODS)).split(',')
assert set(METHODS) <= set(ALL_METHODS), METHODS
SANITY_CPU_CNN = os.environ.get('RNCV_SANITY_CPU_CNN', '0') == '1'
if '1D-CNN' in METHODS and DEVICE != 'cuda' and not SANITY_CPU_CNN:
    raise SystemExit('1D-CNN 的正式结果须在 GPU 上跑（B10.1）：请设 RNCV_DEVICE=cuda；仅作本机冒烟检查时设 '
                     'RNCV_SANITY_CPU_CNN=1（产物写到 04outputs/45sanity_ncv_*，47 脚本不读取）')
SEEDS = [20041210, 20240404, 20060515, 20251203, 19810915]          # B1 去重后 5 个
if os.environ.get('RNCV_SEEDS'):
    SEEDS = [int(s) for s in os.environ['RNCV_SEEDS'].split(',')]
    assert set(SEEDS) <= {20041210, 20240404, 20060515, 20251203, 19810915}, SEEDS
K_OUTER, K_INNER, L_BASIS = 5, 5, 80
N_JOBS = int(os.environ.get('RNCV_JOBS', '1'))          # RF 的并行核数须由调用方显式给出，默认单核
REQUIRE_COMPLETE = os.environ.get('RNCV_REQUIRE_COMPLETE', '0') == '1'
_SH = os.environ.get('RNCV_SHARD', '')
try:
    SHARD_I, SHARD_N = (int(_SH.split('/')[0]), int(_SH.split('/')[1])) if _SH else (0, 1)
except (ValueError, IndexError):
    raise SystemExit(f"RNCV_SHARD 须为 'i/n' 形式，收到 {_SH!r}") from None
if not (SHARD_N >= 1 and 0 <= SHARD_I < SHARD_N):
    raise SystemExit(f'RNCV_SHARD 须满足 n≥1 且 0≤i<n，收到 {_SH!r}')
ALL_SEEDS = [20041210, 20240404, 20060515, 20251203, 19810915]
LABEL = {'PFLR': 'FPLR'}
FUNCTIONAL = ('PFLR', 'FPCR')

# 函数型基：覆盖 16 脚本里的 BASIS/COEF。mk() 里的闭包在调用时才按名取全局量，故覆盖即生效；
# fourier 分支与 16 脚本的 BASIS[80]、COEF[80] 为同一函数同一参数，逐位相同。47 用同一个 basis_system 重算基系数的哈希。
_B, _J, _Rm, _C = nm.basis_system(globals(), BASIS_NAME, L_BASIS)
BASIS = {L_BASIS: (_B, _J, _Rm)}
COEF = {L_BASIS: _C}
print(f'函数型基：{BASIS_NAME} L={L_BASIS}  tr(J)={np.trace(_J):.3e}  tr(R)={np.trace(_Rm):.3e}  '
      f'量级比 {np.trace(_Rm) / np.trace(_J):.2e}')

if DEVICE == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f'GPU 环境：torch {torch.__version__}  CUDA {torch.version.cuda}  cuDNN {torch.backends.cudnn.version()}  '
          f'设备 {torch.cuda.get_device_name(0)}  cudnn.deterministic=True  benchmark=False  '
          f'use_deterministic_algorithms(warn_only=True)', flush=True)


def cnn_fit_dev(Xtr, ytr, Xte, cfg, seed):
    """与 16 脚本的 cnn_fit 逐项一致，只把张量放到 DEVICE 上；mini-batch 排列仍在 CPU 上生成。"""
    dev = torch.device(DEVICE)
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = ytr.mean(), ytr.std()
    A = torch.tensor(((Xtr - mu) / sd)[:, None, :], dtype=torch.float32, device=dev)
    Bt = torch.tensor(((Xte - mu) / sd)[:, None, :], dtype=torch.float32, device=dev)
    Y = torch.tensor(((ytr - ym) / ys)[:, None], dtype=torch.float32, device=dev)
    net = CNN(cfg['w'], cfg['dp'], cfg['nl']).to(dev)
    opt = torch.optim.Adam(net.parameters(), cfg['lr'], weight_decay=cfg['wd'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['ep'])
    lf = nn.MSELoss()
    n = len(A)
    for _e in range(cfg['ep']):
        net.train()
        idx = torch.randperm(n)
        for i in range(0, n, cfg['bs']):
            b = idx[i:i + cfg['bs']].to(dev)
            opt.zero_grad()
            lf(net(A[b]), Y[b]).backward()
            opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        return net(Bt).cpu().numpy().ravel() * ys + ym


def fitter(m):
    """返回 fit(候选, 训练下标, 预测下标, 种子)。PLSR/SVR/FPCR/PFLR 直接用 16 脚本的 mk()。"""
    if m == 'RF':
        def f(c, a, b, sd):
            return RandomForestRegressor(n_estimators=c[0], max_features=c[1], random_state=sd,
                                         n_jobs=N_JOBS).fit(Xp[a], y[a]).predict(Xp[b])
        return f
    if m == '1D-CNN':
        if DEVICE == 'cpu':
            return mk('1D-CNN')[1]

        def g(c, a, b, sd):
            return cnn_fit_dev(Xp[a], y[a], Xp[b], c, sd)
        return g
    return mk(m)[1]


def basis_tag(m):
    return BASIS_NAME if m in FUNCTIONAL else 'none'


_FP: dict[str, str] = {}


def fingerprint(m):
    """缓存指纹：缓存里的值（内层 MSE 与重训练预测）只取决于这些量，任一项改变即换新缓存文件。"""
    if m in _FP:
        return _FP[m]
    h = hashlib.sha256()
    for arr in (y, Xp, COEF[L_BASIS] if m in FUNCTIONAL else np.zeros(1)):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(json.dumps([basis_tag(m), L_BASIS if m in FUNCTIONAL else None, m, K_OUTER, K_INNER,
                         GRIDS[m]], sort_keys=True, default=str).encode())
    h.update(_SHARED.encode())
    with open('02code/grids.py', encoding='utf-8') as fh:
        h.update(fh.read().encode())
    import scipy
    import sklearn
    h.update(json.dumps([sys.version, np.__version__, scipy.__version__, sklearn.__version__,
                         torch.__version__]).encode())
    if m == '1D-CNN':
        h.update((DEVICE + inspect.getsource(cnn_fit_dev)).encode())
        if DEVICE == 'cuda':
            h.update(json.dumps([torch.version.cuda, torch.backends.cudnn.version(),
                                 torch.cuda.get_device_name(0)]).encode())
    if m == 'RF':
        h.update((inspect.getsource(fitter) + f'n_jobs={N_JOBS}').encode())
    _FP[m] = h.hexdigest()[:16]
    return _FP[m]


def row_fingerprint(m):
    """结果指纹：写进汇总行与预测行。缓存指纹再加本脚本全文与 ncv_manifest.py 的 SHA-256，
    本脚本任何改动都会让已有 CSV 整组按缓存重算（缓存本身不失效）。"""
    return nm.row_fingerprint(fingerprint(m), CODE_SHA)


def out_paths(m):
    prefix = '45sanity_ncv' if (m == '1D-CNN' and DEVICE != 'cuda') else '45rebuttal_ncv'
    stem = f"04outputs/{prefix}_{basis_tag(m)}_{LABEL.get(m, m).replace('-', '')}"
    return f'{stem}.csv', f'{stem}_pred.csv'


def run_device(m):
    return DEVICE if m == '1D-CNN' else 'cpu'


def memo_stem(m):
    return f"04outputs/45memo_{basis_tag(m)}_{LABEL.get(m, m).replace('-', '')}_{fingerprint(m)}"


def memo_own_path(m):
    return f'{memo_stem(m)}_shard{SHARD_I}.jsonl' if SHARD_N > 1 else f'{memo_stem(m)}.jsonl'


_MEMO: dict[str, Any] = {}


def memo_load(m):
    """读回本指纹下的主缓存与全部分片缓存，逐条校验后才采用：
    方法与指纹相符；键为 ['inner'|'refit', 种子, 折, 候选] 且种子、折、候选都在期望集合内；
    inner 的值是 K_INNER 个有限数，refit 的值是该折测试样品数个有限数。
    不合格的记录不采用（对应格点会重算）并计数报告；同一键取值不一致即报错停下。"""
    _MEMO.clear()
    files = sorted(f for f in glob.glob(f'{memo_stem(m)}*.jsonl') if not os.path.basename(f).startswith('._'))
    cfg_ok = {json.dumps(c, sort_keys=True, default=str) for c in GRIDS[m]}
    n_bad = 0
    for f in files:
        with open(f, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    kind, sd, fold, cfg = json.loads(rec['k'])
                    v = rec['v']
                    valid = (rec.get('m') == m and rec.get('fp') == fingerprint(m)
                             and kind in ('inner', 'refit') and sd in ALL_SEEDS and fold in range(K_OUTER)
                             and json.dumps(cfg, sort_keys=True, default=str) in cfg_ok
                             and isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)
                             and bool(np.all(np.isfinite(v)))
                             and len(v) == (K_INNER if kind == 'inner' else len(fold_test(sd, fold))))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    valid = False
                if not valid:
                    n_bad += 1
                    continue
                k = rec['k']
                if k in _MEMO and _MEMO[k] != v:
                    raise SystemExit(f'缓存冲突：同一键取值不同 {k}（{f}）')
                _MEMO[k] = v
    if files:
        print(f'  缓存 {len(files)} 个文件，采用 {len(_MEMO)} 个键，不合格记录 {n_bad} 条（对应格点会重算）',
              flush=True)


def memo_put(m, key, val):
    """追加一行并 fsync；每个进程只写自己独占的文件。记录带方法与指纹，供读回时核对。"""
    _MEMO[key] = val
    with open(memo_own_path(m), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps({'k': key, 'v': val, 'm': m, 'fp': fingerprint(m)}, ensure_ascii=False) + '\n')
        fh.flush()
        os.fsync(fh.fileno())


def fold_test(seed, fold):
    return list(KFold(n_splits=K_OUTER, shuffle=True, random_state=seed).split(np.arange(len(y))))[fold][1]


def key_inner(seed, fold, cfg):
    return json.dumps(['inner', seed, fold, cfg], sort_keys=True, ensure_ascii=False, default=str)


def key_refit(seed, fold, cfg):
    return json.dumps(['refit', seed, fold, cfg], sort_keys=True, ensure_ascii=False, default=str)


def inner_mses(m, fit, seed, fold, itr, cfg):
    """一个内层格点：5 个内层验证折的 MSE（未命中缓存才计算）。"""
    k = key_inner(seed, fold, cfg)
    if k not in _MEMO:
        inner = list(KFold(n_splits=K_INNER, shuffle=True, random_state=seed).split(itr))
        se = [float(mean_squared_error(y[itr[iv]], fit(cfg, itr[ia], itr[iv], seed))) for ia, iv in inner]
        memo_put(m, k, se)
    return _MEMO[k]


def fold_metrics(ytr, yte, p):
    """与 25nested_cv.py 同名函数一致；base_* 为「以训练折均值预测」这一无技能基准。"""
    e = np.asarray(p) - yte
    ae = np.abs(e)
    rm = float(np.sqrt(mean_squared_error(yte, p)))
    sdte = float(np.std(yte))
    be = np.abs(float(np.mean(ytr)) - yte)
    return dict(R2=float(r2_score(yte, p)), RMSE=rm, RPD=sdte / rm,
                MAE=float(ae.mean()), P90=float(np.quantile(ae, 0.9)), BIAS=float(e.mean()),
                base_MAE=float(be.mean()), base_P90=float(np.quantile(be, 0.9)),
                s_y_train=float(np.std(ytr)), s_y_test=sdte)


def atomic_csv(rows, path):
    tmp = f'{path}.tmp'
    pd.DataFrame(rows).to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, path)


NUM_COLS = ['inner_rmse', 'R2', 'RMSE', 'RPD', 'MAE', 'P90', 'BIAS', 'base_MAE', 'base_P90', 's_y_train', 's_y_test']
STR_COLS = {c: str for c in ('basis', 'method', 'picked', 'fingerprint', 'device')}


def complete_seeds(m, rows, preds):
    """某种子算完成（任一条不满足，该种子整组丢弃重算）：
    ① 汇总行 5 折各一行；方法、基族、结果指纹、设备都与本次运行相符；选中配置属于网格；
       训练/测试样本数与该折相符；全部数值列有限；
    ② 预测行的方法、基族、结果指纹、设备与本次运行相符；覆盖 0..n−1 每个样品恰一次，每折的样品集合就是该折测试集；
       真值与当前数据一致、预测有限；由预测重算的 R² 与汇总行一致（差 <1e-9）。"""
    r, p = pd.DataFrame(rows), pd.DataFrame(preds)
    ok: set[int] = set()
    need_r = {'basis', 'method', 'seed', 'fold', 'picked', 'fingerprint', 'device', 'n_train', 'n_test', *NUM_COLS}
    need_p = {'basis', 'method', 'seed', 'fold', 'idx', 'y', 'pred', 'fingerprint', 'device'}
    if r.empty or p.empty or not need_r <= set(r.columns) or not need_p <= set(p.columns):
        return ok
    lab, btag, fp, dev = LABEL.get(m, m), basis_tag(m), row_fingerprint(m), run_device(m)
    grid_str = {str(c) for c in GRIDS[m]}
    for sd, a in r.groupby('seed'):
        sd = int(sd)
        if sd not in ALL_SEEDS or sorted(a.fold.astype(int).tolist()) != list(range(K_OUTER)):
            continue
        if not ((a.method.astype(str) == lab).all() and (a.basis.astype(str) == btag).all()
                and (a.fingerprint.astype(str) == fp).all() and (a.device.astype(str) == dev).all()
                and a.picked.astype(str).isin(grid_str).all()):
            continue
        if not np.all(np.isfinite(a[NUM_COLS].to_numpy(dtype=float))):
            continue
        b = p[p.seed.astype(int) == sd]
        if not ((b.method.astype(str) == lab).all() and (b.basis.astype(str) == btag).all()
                and (b.fingerprint.astype(str) == fp).all() and (b.device.astype(str) == dev).all()):
            continue
        if len(b) != len(y) or set(b.idx.astype(int)) != set(range(len(y))):
            continue
        if not np.all(np.isfinite(b[['y', 'pred']].to_numpy(dtype=float))):
            continue
        if float(np.max(np.abs(b.y.to_numpy(dtype=float) - y[b.idx.astype(int).to_numpy()]))) > 1e-9:
            continue
        good = True
        for f in range(K_OUTER):
            te = fold_test(sd, f)
            bf, af = b[b.fold.astype(int) == f], a[a.fold.astype(int) == f]
            if (set(bf.idx.astype(int)) != set(te.tolist()) or int(af.n_test.iloc[0]) != len(te)
                    or int(af.n_train.iloc[0]) != len(y) - len(te)
                    or abs(float(r2_score(bf.y, bf.pred)) - float(af.R2.iloc[0])) > 1e-9):
                good = False
                break
        if good:
            ok.add(sd)
    return ok


def write_manifest(m, rows, preds, out, pred):
    """5 个种子全部完成后：按种子、折排序重写两份 CSV，从磁盘读回再核一遍全部种子，通过后写来源清单。"""
    lab = LABEL.get(m, m)
    order = {sd: i for i, sd in enumerate(ALL_SEEDS)}
    rows = sorted(rows, key=lambda r: (order[int(r['seed'])], int(r['fold'])))
    preds = sorted(preds, key=lambda r: (order[int(r['seed'])], int(r['fold'])))
    atomic_csv(preds, pred)
    atomic_csv(rows, out)
    with open(out, 'rb') as fh:
        cb = fh.read()
    with open(pred, 'rb') as fh:
        pb = fh.read()
    r_disk = pd.read_csv(io.BytesIO(cb), dtype=STR_COLS, encoding='utf-8-sig')
    p_disk = pd.read_csv(io.BytesIO(pb), dtype=STR_COLS, encoding='utf-8-sig')
    if (len(r_disk) != len(ALL_SEEDS) * K_OUTER or len(p_disk) != len(ALL_SEEDS) * len(y)
            or complete_seeds(m, r_disk.to_dict('records'), p_disk.to_dict('records')) != set(ALL_SEEDS)):
        raise SystemExit(f'{lab}：从磁盘读回的 CSV 未通过完整性核对，不写来源清单')
    man = nm.build_manifest(
        analysis_basis=basis_tag(m), method=lab, method_key=m, fingerprint=row_fingerprint(m),
        memo_fingerprint=fingerprint(m), device=run_device(m), n_jobs=N_JOBS if m == 'RF' else None,
        seeds=ALL_SEEDS, k_outer=K_OUTER, k_inner=K_INNER, L=L_BASIS if m in FUNCTIONAL else None, grid=GRIDS[m],
        csv_path=out, csv_bytes=cb, csv_rows=len(r_disk), pred_path=pred, pred_bytes=pb, pred_rows=len(p_disk),
        code=CODE_SHA, data=nm.data_sha256(y, Xp, COEF[L_BASIS] if m in FUNCTIONAL else None),
        env=nm.env_info(run_device(m)))
    nm.write_json_atomic(man, nm.manifest_path(out))
    print(f'  {lab} 基={basis_tag(m)}：5 个种子读回核对通过，来源清单 → {nm.manifest_path(out)}', flush=True)


FOLDS ={sd: list(KFold(n_splits=K_OUTER, shuffle=True, random_state=sd).split(np.arange(len(y))))
         for sd in SEEDS}
for m in METHODS:
    assert len(GRIDS[m]) == 12, (m, len(GRIDS[m]))

print(f'n={len(y)}  嵌套交叉验证 K_outer={K_OUTER} K_inner={K_INNER}  种子 {SEEDS}')
print(f'方法 {METHODS}  函数型基 {BASIS_NAME}  设备 {DEVICE}  RF n_jobs={N_JOBS}  '
      f'分片 {SHARD_I}/{SHARD_N}  缺格点即停={REQUIRE_COMPLETE}\n', flush=True)

if SHARD_N > 1:
    # 分片只填内层格点，不做选择、不重训练、不写 CSV：它看到的格点是残缺的。
    for m in METHODS:
        memo_load(m)
        fit = fitter(m)
        t0 = time.time()
        gi, n_new, n_hit = -1, 0, 0
        for sd in SEEDS:
            for fold, (itr, _ite) in enumerate(FOLDS[sd]):
                for cfg in GRIDS[m]:
                    gi += 1
                    if gi % SHARD_N != SHARD_I:
                        continue
                    if key_inner(sd, fold, cfg) in _MEMO:
                        n_hit += 1
                        continue
                    inner_mses(m, fit, sd, fold, itr, cfg)
                    n_new += 1
                    print(f'  [{SHARD_I}/{SHARD_N}] {m} seed={sd} fold={fold} 第 {gi:>3} 格点 {cfg} 已存 '
                          f'(新算 {n_new}，复用 {n_hit}，{time.time() - t0:.0f}s)', flush=True)
        print(f'  [{SHARD_I}/{SHARD_N}] {m} 本分片填完：新算 {n_new}，复用 {n_hit}', flush=True)
    sys.exit(0)

for m in METHODS:
    memo_load(m)
    fit = fitter(m)
    lab, btag = LABEL.get(m, m), basis_tag(m)
    out, pred = out_paths(m)
    rows = pd.read_csv(out, dtype=STR_COLS).to_dict('records') if os.path.exists(out) else []
    preds = pd.read_csv(pred, dtype=STR_COLS).to_dict('records') if os.path.exists(pred) else []
    done = complete_seeds(m, rows, preds)
    n_old = len({int(r['seed']) for r in rows})
    rows = [r for r in rows if int(r['seed']) in done]          # 不完整或来源不符的种子整组重算
    preds = [r for r in preds if int(r['seed']) in done]
    if n_old > len(done):
        print(f'  {lab}：已有产物中 {n_old - len(done)} 个种子不完整或来源与本次运行不符，整组重算', flush=True)
    todo = [sd for sd in SEEDS if sd not in done]
    missing = sum(key_inner(sd, fold, cfg) not in _MEMO
                  for sd in todo for fold in range(K_OUTER) for cfg in GRIDS[m])
    print(f'  {lab} 基={btag} 缓存指纹 {fingerprint(m)} 结果指纹 {row_fingerprint(m)}：已完成种子 {sorted(done)}，'
          f'待算种子 {todo}，其中缺失的内层格点 {missing} 个', flush=True)
    if REQUIRE_COMPLETE and missing:
        raise SystemExit(f'{lab}：要求格点齐全但缺 {missing} 个，分片可能未收工')
    man_path = nm.manifest_path(out)
    if done != set(ALL_SEEDS) and os.path.exists(man_path):
        os.remove(man_path)                  # 这组 CSV 将被改动或本就不完整：旧清单不再描述它们
        print(f'  {lab}：已删除不再对应的旧来源清单 {man_path}', flush=True)
    for sd in todo:
        t0 = time.time()
        new_rows, new_preds = [], []
        for fold, (itr, ite) in enumerate(FOLDS[sd]):
            best, bc = np.inf, GRIDS[m][0]
            for cfg in GRIDS[m]:
                e = float(np.mean(np.sqrt(inner_mses(m, fit, sd, fold, itr, cfg))))
                if e < best:
                    best, bc = e, cfg
            kr = key_refit(sd, fold, bc)
            if kr not in _MEMO:
                memo_put(m, kr, [float(v) for v in fit(bc, itr, ite, sd)])
            p = np.asarray(_MEMO[kr])
            assert len(p) == len(ite), (lab, sd, fold, len(p), len(ite))
            mt = fold_metrics(y[itr], y[ite], p)
            new_rows.append(dict(basis=btag, method=lab, seed=sd, fold=fold, picked=str(bc),
                                 inner_rmse=best, n_train=len(itr), n_test=len(ite), fingerprint=row_fingerprint(m),
                                 device=run_device(m), **mt))
            new_preds += [dict(basis=btag, method=lab, seed=sd, fold=fold, idx=int(i), y=float(y[i]),
                               pred=float(v), fingerprint=row_fingerprint(m), device=run_device(m))
                          for i, v in zip(ite, p, strict=True)]
            print(f'  {lab:<7} 基={btag:<8} seed={sd} fold={fold} 选中={str(bc):<36} '
                  f'R²={mt["R2"]:.4f} RMSE={mt["RMSE"]:.4f}', flush=True)
        rows += new_rows
        preds += new_preds
        atomic_csv(preds, pred)
        atomic_csv(rows, out)
        a = pd.DataFrame(new_rows)
        print(f'  {lab:<7} 基={btag:<8} seed={sd}  R²={a.R2.mean():.4f}±{a.R2.std(ddof=1):.4f}  '
              f'用时 {time.time() - t0:.0f}s，已落盘 {out}\n', flush=True)

    if {int(r['seed']) for r in rows} == set(ALL_SEEDS):
        write_manifest(m, rows, preds, out, pred)
    else:
        print(f'  {lab} 基={btag}：尚有种子未完成，不写来源清单', flush=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        per_seed = df.groupby('seed').R2.mean()
        print(f'【{lab} 基={btag}】折数 {len(df)}  R²={df.R2.mean():.4f}±{df.R2.std(ddof=1):.4f}  '
              f'RMSE={df.RMSE.mean():.4f}  各次重复均值 {np.round(per_seed.values, 4).tolist()}\n', flush=True)
