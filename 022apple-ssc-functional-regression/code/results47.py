# -*- coding: utf-8 -*-
"""47 脚本产物的发布与读取：正式结果与部分结果分开存放，下游（ms_numbers.py、50rebuttal_figures.py）只经本模块取数。

为什么要有这个模块：47 原先在 PT_ALLOW_PARTIAL=1 时把部分结果写到正式文件名，正式运行失败后，旧的部分统计
仍留在正式路径上，下游脚本照读不误（第 4 轮部署前代码审查）。现在的规则：
  正式结果  04outputs/47rebuttal_<名>.csv，外加完成标记 04outputs/47rebuttal_COMPLETE.json。
            只在全部核对与门禁通过后写出：先把全部文件写进 04outputs 下的临时目录，删除旧完成标记，
            逐个原子替换到正式路径，最后原子写出完成标记。发布中途出错时正式路径上没有完成标记，下游一律拒读。
  部分结果  04outputs/47partial/47partial_<名>.csv，外加标记 47partial_PARTIAL.json。只含通过全部核对的方法，
            标记记下每个（分析, 方法）的状态（ok / missing / incomplete / invalid）。部分结果永不写到正式路径。
  标记内容  47 读入的全部输入文件（04outputs 下 45 的汇总、预测与清单，以及 25nested_cv.csv）的 SHA-256，
            45 与 47 相关代码的 SHA-256，每个产物的 SHA-256 与行数，以及各方法状态。
  读取      resolve() 先核正式标记：产物、输入、代码的哈希与当前文件逐一相符，且 04outputs 下 45 产物的文件集合
            与标记记录的相同，才返回正式结果；否则再核部分标记；两者都不成立即报错停下。
            47 运行之后任何输入或代码被改动、增删，标记即失效，须重跑 47。
  清理      invalidate_stale_final()：正式路径上的产物若没有有效完成标记，连同失效的标记一并删除（47 每次运行开始时调用），
            保证正式路径上只可能存在经完成标记认证的结果。正式发布成功后，部分结果目录随之清除。
  并发      47 全程持有 04outputs/.47run.lock 的排他锁（acquire_run_lock()，进程退出即释放），两个 47 不会交错发布。
            cnn_fetch.sh 向 04outputs 发布取回的文件期间持有同一把锁：47 在运行时取回脚本取不到锁、不发布；取回脚本
            发布期间 47 取不到锁、即停。45 不取这把锁，activity() 列出本机工作目录为本项目的 45 进程（工作目录读不到的
            也计入，无从排除；列不出进程时同样按占用），47 的正式运行遇到即停。rf_local.sh 只经它启动的 45 进程改写
            47 的输入，取回脚本只在持锁时向 04outputs 发布文件，故本机 RF 的防重复启动锁目录与取回暂存目录只作提示，
            不据此判定占用。
            publish() 在改动正式路径之前重读全部输入与代码，与运行开始时的快照不同即不发布；写出标记后再核一次，
            不符即删去标记。输入在 47 发布之后才被改写时，标记随即失效，下游拒读。
  逐折指标  fold_metrics45()：取 45 脚本自己的 fold_metrics() 定义，47 据此由数据集真值与逐样品预测重算每折的
            R²、RMSE、RPD、MAE、P90、BIAS、两项基准指标与训练/测试集标准差，逐项核对汇总行。
  缓存指纹  memo_fingerprint45()：取 45 脚本自己的 fingerprint() 定义，以调用方给出的数据、网格、并行核数与环境
            重算缓存指纹。47 据此要求本机运行的方法的来源清单记录的缓存指纹等于本轮应有的值。指纹相同说明结果是按本轮的
            数据、代码、网格与环境配置写出的，不说明结果数值来自哪份格点缓存；后者由 47 的格点缓存复核检验
            （读 45 写出的缓存，复算每折的选中配置并比对逐样品预测），核对过的缓存文件的 SHA-256 与记录数写入标记备查；
            resolve() 核对的是输入、代码与产物的哈希，不重读缓存。
  端点诊断  svr_scale_gammas()、grid_value_range()、grid_extremes()：47 统计选中值落在网格端点的折数时，非数值取值
            按 scikit-learn 拟合时实际使用的数值排序。SVR 的 gamma='scale' 为 1/(特征数×训练集方差)，随训练集而变，取 45
            各次拟合所用训练集上的最小与最大值；RF 的特征比例按实际参与分裂的特征数（'sqrt' 为 ⌊√p⌋，比例 f 为 ⌊f·p⌋）。
            各取值的区间两两不交才判定端点，某一端无从判定时该端不计。

不读 05data（数据由调用方传入）。所有路径相对项目根目录。
"""
import ast
from collections.abc import Callable
from dataclasses import dataclass
import fcntl
import glob
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from ncv_manifest import sha256_bytes, sha256_file, write_json_atomic

OUT = '04outputs'
NAMES = ('table1', 'paired_tests', 'boundary', 'repro', 'provenance')
COMPLETE = f'{OUT}/47rebuttal_COMPLETE.json'
PARTIAL_DIR = f'{OUT}/47partial'
PARTIAL = f'{PARTIAL_DIR}/47partial_PARTIAL.json'
REFERENCE = f'{OUT}/25nested_cv.csv'
SCRIPT45 = '02code/45rebuttal_R1_4_repeated_ncv.py'
CODE_FILES = (SCRIPT45, '02code/16unified_comparison.py', '02code/grids.py',
              '02code/ncv_manifest.py', '02code/47rebuttal_R1_1_paired_tests.py', '02code/results47.py')
SCHEMA = '47rebuttal-marker-1'
RUN_LOCK = f'{OUT}/.47run.lock'
RF_LOCK = f'{OUT}/.45rf_local.lock'           # 02code/deploy45/rf_local.sh 的防重复启动锁（目录），只作提示
FETCH_GLOB = f'{OUT}/.45cnn_fetch.*'          # 02code/deploy45/cnn_fetch.sh 的取回暂存目录（1D-CNN 与 RF 共用），只作提示
FETCH_OWNER = 'owner'                         # cnn_fetch.sh 在暂存目录里写下自己的 PID
PROC_TIMEOUT_S = 30                           # ps 与 lsof 的时限
FP45_FUNCS = ('basis_tag', 'fitter', 'fingerprint')


def out_path(name: str, final: bool) -> str:
    if name not in NAMES:
        raise KeyError(name)
    return f'{OUT}/47rebuttal_{name}.csv' if final else f'{PARTIAL_DIR}/47partial_{name}.csv'


def marker_path(final: bool) -> str:
    return COMPLETE if final else PARTIAL


def inputs45() -> list[str]:
    """04outputs 下 45 的全部产物（汇总、预测、清单）；exFAT 的 ._ 伴随文件与写到一半的 .tmp 文件不算。"""
    return sorted(p for p in glob.glob(f'{OUT}/45rebuttal_ncv_*')
                  if not os.path.basename(p).startswith('._') and not p.endswith('.tmp'))


def snapshot_inputs() -> dict[str, bytes]:
    """一次读入 47 的全部输入的字节；47 的核对与计算只用这份字节，标记里记的就是它们的哈希。"""
    snap = {}
    for p in inputs45() + [REFERENCE]:
        with open(p, 'rb') as fh:
            snap[p] = fh.read()
    return snap


def code_sha() -> dict[str, str]:
    return {p: sha256_file(p) for p in CODE_FILES}


def _load(path: str):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh), None
    except (OSError, json.JSONDecodeError) as e:
        return None, f'{path} 无法读取：{e}'


def marker_problems(final: bool) -> list[str]:
    """核对标记与当前文件是否相符；返回不符之处，空表即有效。"""
    mp = marker_path(final)
    if not os.path.exists(mp):
        return [f'没有标记 {mp}']
    mk, err = _load(mp)
    if err:
        return [err]
    probs = []
    if not isinstance(mk, dict) or mk.get('schema') != SCHEMA or mk.get('kind') != ('final' if final else 'partial'):
        return [f'{mp} 的格式或类别不符']
    inputs = mk.get('inputs') or {}
    for p, h in inputs.items():
        if not os.path.exists(p):
            probs.append(f'输入已不存在：{p}')
        elif sha256_file(p) != h:
            probs.append(f'输入在 47 运行后被改动：{p}')
    if REFERENCE not in inputs:
        probs.append(f'标记未记录参照 {REFERENCE}')
    if sorted(mk.get('inputs45') or []) != inputs45() or set(mk.get('inputs45') or []) | {REFERENCE} != set(inputs):
        probs.append('04outputs 下 45 产物的文件集合与 47 运行时不同')
    code = mk.get('code') or {}
    if set(code) != set(CODE_FILES):
        probs.append('标记记录的代码文件集合不符')
    for p, h in code.items():
        if not os.path.exists(p) or sha256_file(p) != h:
            probs.append(f'代码在 47 运行后被改动：{p}')
    outputs = mk.get('outputs') or {}
    if set(outputs) != {out_path(n, final) for n in NAMES}:
        probs.append('标记记录的产物集合不符')
    for p, o in outputs.items():
        if not os.path.exists(p):
            probs.append(f'产物不存在：{p}')
        elif sha256_file(p) != o.get('sha256'):
            probs.append(f'产物在 47 运行后被改动：{p}')
    status = mk.get('status') or {}
    if not status:
        probs.append('标记未记录各方法状态')
    if final and any(v != 'ok' for v in status.values()):
        probs.append('正式标记中有未通过核对的方法')
    return probs


@dataclass(frozen=True)
class Source47:
    final: bool
    marker: dict

    def path(self, name: str) -> str:
        return out_path(name, self.final)

    def with_status(self, *values: str) -> list[str]:
        return sorted(k for k, v in self.marker['status'].items() if v in values)


def resolve() -> Source47:
    """下游取 47 产物的唯一入口：有效的正式结果优先，其次有效的部分结果，否则报错停下。"""
    pf = marker_problems(True)
    if not pf:
        return Source47(True, _load(COMPLETE)[0])
    pp = marker_problems(False)
    if not pp:
        return Source47(False, _load(PARTIAL)[0])
    raise SystemExit('47 的产物没有有效标记，须先运行 02code/47rebuttal_R1_1_paired_tests.py：\n  正式结果：'
                     + '；'.join(pf) + '\n  部分结果：' + '；'.join(pp))


def invalidate_stale_final() -> list[str]:
    """正式路径上的产物没有有效完成标记时，连同失效的标记一并删除；返回删除的文件。"""
    if not marker_problems(True):
        return []
    removed = []
    for p in [out_path(n, True) for n in NAMES] + [COMPLETE]:
        if os.path.exists(p):
            os.remove(p)
            removed.append(p)
    return removed


def clear_partial() -> list[str]:
    removed = []
    for p in [out_path(n, False) for n in NAMES] + [PARTIAL]:
        if os.path.exists(p):
            os.remove(p)
            removed.append(p)
    return removed


def acquire_run_lock():
    """取得 47 的运行锁（RUN_LOCK 上的 fcntl.flock 排他锁，持有到返回的文件对象关闭或进程退出）。
    另一进程持有时返回 None。"""
    fh = open(RUN_LOCK, 'a', encoding='utf-8')     # noqa: SIM115 — 锁须在函数返回后继续持有，由调用方保留文件对象
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def _lsof_path(raw: bytes) -> str:
    """lsof -F 输出的路径把非 ASCII 字节写成 \\xNN、反斜杠写成 \\\\，还原为原路径。"""
    out = bytearray()
    i = 0
    while i < len(raw):
        if raw[i:i + 2] == b'\\x' and re.fullmatch(rb'[0-9a-fA-F]{2}', raw[i + 2:i + 4]):
            out.append(int(raw[i + 2:i + 4], 16))
            i += 4
        elif raw[i:i + 2] == b'\\\\':
            out.append(0x5C)
            i += 2
        else:
            out.append(raw[i])
            i += 1
    return out.decode('utf-8', errors='replace')


def _process_cwd(pid: int) -> str | None:
    """进程的工作目录；读不到时（进程不在、其他用户的进程、权限受限、lsof 缺失、出错或超时）为 None。"""
    if sys.platform.startswith('linux'):
        try:
            return os.readlink(f'/proc/{pid}/cwd')
        except OSError:
            return None
    try:
        r = subprocess.run(['lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn'], capture_output=True, check=False,
                           timeout=PROC_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    for ln in r.stdout.splitlines():
        if ln.startswith(b'n'):
            return _lsof_path(ln[1:])
    return None


def _ps(field: str) -> dict[int, str]:
    r = subprocess.run(['ps', '-axww', '-o', f'pid=,{field}='], capture_output=True, text=True, check=True,
                       timeout=PROC_TIMEOUT_S)
    rows = {}
    for ln in r.stdout.splitlines():
        pid, _, rest = ln.strip().partition(' ')
        if pid.isdigit():
            rows[int(pid)] = rest.strip()
    return rows


def _cwd_state(pid: int, root: str) -> str:
    """进程工作目录与 root 的关系：'here' 相同；'elsewhere' 读得到且不同；'unknown' 读不到（见 _process_cwd()）。"""
    cwd = _process_cwd(pid)
    if cwd is None:
        return 'unknown'
    return 'here' if os.path.realpath(cwd) == os.path.realpath(root) else 'elsewhere'


def _running_45_states(root: str) -> dict[int, str]:
    """可执行文件是 python、命令行含 45 脚本路径、且工作目录是 root 或读不到的进程 → {PID: 'here' | 'unknown'}。
    列不出进程（ps 缺失、出错或超时）时抛出 OSError 或 subprocess.SubprocessError。"""
    comm = _ps('comm')
    states = {}
    for pid, args in _ps('args').items():
        if pid == os.getpid() or SCRIPT45 not in args:
            continue
        if not os.path.basename(comm.get(pid, '')).lower().startswith('python'):
            continue
        state = _cwd_state(pid, root)
        if state != 'elsewhere':
            states[pid] = state
    return states


def running_45(root: str = '.') -> list[int]:
    """可能在改写 root 下 47 输入的 45 脚本进程：可执行文件是 python，命令行含 45 脚本路径，且工作目录是 root
    或读不到（无从排除）；工作目录读得到且在别处（项目副本或其他目录）的同名脚本不算。"""
    return sorted(_running_45_states(root))


def _read_text(path: str) -> str:
    try:
        with open(path, encoding='utf-8') as fh:
            return fh.read()
    except OSError:
        return ''


def activity(root: str = '.') -> tuple[list[str], list[str]]:
    """返回（可能正在改写 47 输入的活动, 只作提示的目录）。前者空表表示没有。
    占用只按 45 进程判定；取回脚本发布期间持有 47 的运行锁，由 acquire_run_lock() 互斥，不在这里判定。"""
    busy: list[str] = []
    notes: list[str] = []
    try:
        states = _running_45_states(root)
    except (OSError, subprocess.SubprocessError) as e:
        busy.append(f'列不出本机进程（{type(e).__name__}: {e}），无从排除本项目有 45 脚本进程在运行')
        states = {}
    if states:
        unknown = [p for p in sorted(states) if states[p] == 'unknown']
        busy.append(f'本项目有 45 脚本进程在运行（PID {" ".join(map(str, sorted(states)))}），运行结束时会写出汇总、预测与清单'
                    + (f'；其中 PID {" ".join(map(str, unknown))} 的工作目录读不到，无从排除属于本项目，按占用处理'
                       if unknown else ''))
    lock = os.path.join(root, RF_LOCK)
    if os.path.isdir(lock):
        holders = [_read_text(os.path.join(lock, n)).strip() for n in ('pid', 'run_pid')]
        notes.append(f'本机 RF 的防重复启动锁 {RF_LOCK} 存在（记录的 PID {holders}）。rf_local.sh 只经它启动的 45 进程'
                     '改写 47 的输入，占用已按 45 进程判定；是中断后的遗留时，rf_local.sh 下次启动时清除')
    for d in sorted(p for p in glob.glob(os.path.join(root, FETCH_GLOB)) if not os.path.basename(p).startswith('._')):
        owner = _read_text(os.path.join(d, FETCH_OWNER)).strip()
        notes.append(f'服务器结果的取回暂存目录 {d} 存在（属主 PID {owner or "未记录"}）。取回脚本只在持有 47 的运行锁时'
                     '向 04outputs 发布，暂存目录本身不改动 47 的输入；是中断后的遗留时可删除')
    return busy, notes


def busy_reasons(root: str = '.') -> list[str]:
    """正在改写 47 输入的活动；空表表示没有。"""
    return activity(root)[0]


def fold_metrics45(script: str = SCRIPT45) -> Callable[..., dict]:
    """返回 45 脚本自己的 fold_metrics(ytr, yte, p) 定义：从 45 原文取出该顶层函数，以原文件名编译，
    命名空间只含它用到的 np、mean_squared_error、r2_score。45 里没有唯一的定义时抛 ValueError。"""
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score
    with open(script, encoding='utf-8') as fh:
        tree = ast.parse(fh.read(), filename=script)
    found: list[ast.stmt] = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fold_metrics']
    if len(found) != 1:
        raise ValueError(f'{script} 里没有唯一的 fold_metrics 顶层定义')
    ns: dict = {'__name__': '_fold_metrics45', 'np': np, 'mean_squared_error': mean_squared_error, 'r2_score': r2_score}
    exec(compile(ast.Module(body=found, type_ignores=[]), script, 'exec'), ns)
    func: Callable[..., dict] = ns['fold_metrics']
    return func


def memo_fingerprint45(values: dict, method_key: str, script: str = SCRIPT45) -> str:
    """按 45 脚本自己的 fingerprint() 定义重算缓存指纹。

    从 45 原文取出 basis_tag、fitter、fingerprint 三个顶层函数定义，以原文件名与行号编译（fingerprint 里的
    inspect.getsource(fitter) 因而读到的仍是 45 原文），在只含 values 的命名空间里调用 fingerprint(method_key)。
    values 须给出这些函数用到的全局量：np、torch、y、Xp、COEF、L_BASIS、K_OUTER、K_INNER、GRIDS、_SHARED、
    BASIS_NAME、FUNCTIONAL、N_JOBS、DEVICE。缺少任一项时抛 NameError 或 KeyError。"""
    with open(script, encoding='utf-8') as fh:
        tree = ast.parse(fh.read(), filename=script)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FP45_FUNCS]
    if sorted(n.name for n in found) != sorted(FP45_FUNCS):
        raise ValueError(f'{script} 里没有 {FP45_FUNCS} 的顶层定义')
    body: list[ast.stmt] = list(found)
    ns = {'__name__': '_memo_fingerprint45', 'hashlib': hashlib, 'inspect': inspect, 'json': json, 'sys': sys,
          '_FP': {}, **values}
    exec(compile(ast.Module(body=body, type_ignores=[]), script, 'exec'), ns)
    return str(ns['fingerprint'](method_key))


def svr_scale_gammas(xp, seeds, k: int) -> tuple[float, float]:
    """SVR 的 gamma='scale' 在 45 各次拟合所用训练集上的最小与最大取值（scikit-learn 取 1/(特征数 × 训练集方差)）。
    训练集的构造与 45 相同：每个种子的 k 个外层训练集（KFold(k, shuffle=True, random_state=种子) 作用于全部样品），
    以及各外层训练集上以同一种子再作 KFold 得到的 k 个内层训练集。"""
    import numpy as np
    from sklearn.model_selection import KFold
    xp = np.asarray(xp, dtype=float)
    p = xp.shape[1]
    gammas = []
    for sd in seeds:
        for itr, _ite in KFold(n_splits=k, shuffle=True, random_state=sd).split(np.arange(len(xp))):
            gammas.append(1.0 / (p * xp[itr].var()))
            for ia, _iv in KFold(n_splits=k, shuffle=True, random_state=sd).split(itr):
                gammas.append(1.0 / (p * xp[itr[ia]].var()))
    return float(min(gammas)), float(max(gammas))


def grid_value_range(method_key: str, dim: str, value, *, n_features: int,
                     scale_gammas: tuple[float, float]) -> tuple[float, float]:
    """网格端点诊断用：候选在某一维上的取值换成 scikit-learn 拟合时实际使用的数值，返回（下界, 上界）。
    RF 的特征比例：实际参与分裂的特征数，'sqrt' 为 max(1, ⌊√p⌋)，浮点比例 f 为 max(1, ⌊f·p⌋)；
    SVR 的 gamma='scale'：scale_gammas（见 svr_scale_gammas()）；其余数值取值原样返回。
    其他非数值取值抛 ValueError。"""
    if method_key == 'RF' and dim == '特征比例':
        if value == 'sqrt':
            n = max(1, int(math.sqrt(n_features)))
        elif isinstance(value, float):
            n = max(1, int(value * n_features))
        else:
            raise ValueError(f'RF 的特征比例取值 {value!r} 无法换算为特征数')
        return float(n), float(n)
    if method_key == 'SVR' and dim == 'gamma' and value == 'scale':
        return float(scale_gammas[0]), float(scale_gammas[1])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), float(value)
    raise ValueError(f'{method_key} 的 {dim} 取值 {value!r} 无法换算为数值')


def grid_extremes(ranges: dict[str, tuple[float, float]]) -> tuple[str | None, str | None]:
    """ranges 为 取值名 → (下界, 上界)。返回数值最小与最大的取值名：最小者的上界严格小于其余每个取值的下界，
    最大者的下界严格大于其余每个取值的上界。区间与其他取值相交或相等、无从判定时该端为 None；只有一个取值时两端都是它。"""
    keys = list(ranges)
    lows = [a for a in keys if all(ranges[a][1] < ranges[b][0] for b in keys if b != a)]
    highs = [a for a in keys if all(ranges[a][0] > ranges[b][1] for b in keys if b != a)]
    return (lows[0] if lows else None), (highs[0] if highs else None)


def publish(frames: dict, *, final: bool, snapshot: dict[str, bytes], code: dict[str, str], info: dict) -> str:
    """把 NAMES 中的每个 DataFrame 发布为正式或部分结果，最后写标记；返回标记路径。
    snapshot 为 47 读入的输入字节（snapshot_inputs()），code 为 47 开始运行时的 code_sha()，
    info 至少含 status（各（分析, 方法）的状态）。
    改动正式路径之前重读输入与代码，与 snapshot、code 不同即抛 RuntimeError，不改动任何产物；
    写出标记后再核一次，不符即删去标记并抛 RuntimeError。"""
    if set(frames) != set(NAMES):
        raise ValueError(sorted(frames))
    if not info.get('status') or (final and any(v != 'ok' for v in info['status'].values())):
        raise ValueError('正式发布要求全部方法状态为 ok')
    if not final:
        os.makedirs(PARTIAL_DIR, exist_ok=True)
    stage = tempfile.mkdtemp(prefix='.47stage_', dir=OUT)      # 与正式路径同一文件系统，os.replace 才是原子的
    try:
        staged = {}
        for n in NAMES:
            sp = os.path.join(stage, os.path.basename(out_path(n, final)))
            frames[n].to_csv(sp, index=False, encoding='utf-8-sig')
            staged[n] = sp
        marker = {
            'schema': SCHEMA, 'kind': 'final' if final else 'partial',
            'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'inputs': {p: sha256_bytes(b) for p, b in snapshot.items()},
            'inputs45': sorted(p for p in snapshot if p != REFERENCE),
            'code': code,
            'outputs': {out_path(n, final): {'sha256': sha256_file(staged[n]), 'rows': int(len(frames[n]))}
                        for n in NAMES},
            **info,
        }
        changed = sorted(set(snapshot) ^ set(inputs45() + [REFERENCE]))
        if not changed:
            now = snapshot_inputs()
            changed = sorted(p for p in snapshot if now.get(p) != snapshot[p])
        changed += sorted(p for p, h in code_sha().items() if code.get(p) != h)
        if changed:
            raise RuntimeError(f'47 运行期间输入或代码被改动或增删，未发布：{changed}')
        mp = marker_path(final)
        if os.path.exists(mp):
            os.remove(mp)
        for n in NAMES:
            os.replace(staged[n], out_path(n, final))
        write_json_atomic(marker, mp)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    probs = marker_problems(final)
    if probs:
        os.remove(mp)
        raise RuntimeError('发布后标记与当前文件不符（输入或代码在发布过程中被改动），已删去标记：' + '；'.join(probs))
    if final:
        clear_partial()
    return mp
