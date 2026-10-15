# -*- coding: utf-8 -*-
"""第三轮网格（协议 B10.1 网格边界诊断触发，C3 2026-09-15 第三轮网格行）的工作目录、47 核对函数的取出、格点缓存的
跨轮比对，以及 54 脚本产物的发布与读取。

为什么要有这个模块：第三轮用未改动的 45 在另一套网格（grids_boundary_shift.py）上运行，核对与报告须与 47 同一份实现，
而 47 与 results47.py 的哈希记在第二轮的完成标记里，改动其中任一个都会让第二轮的正式结果失效。所以第三轮的核对由
54 脚本从 47 原文取出函数执行，发布与读取另写在本模块，规则与 results47.py 相同。
  工作目录  WORK=04outputs/45shift，布局与项目根目录相同：02code/ 放 45、16、ncv_manifest.py 的逐字节副本，以及改名为
            grids.py 的 grids_boundary_shift.py；45 的产物与格点缓存在其中的 04outputs/。本机的 SVR 由 deploy45/shift_local.sh
            在该目录中运行，服务器上的 1D-CNN 由 deploy45/cnn_fetch.sh（NCV_GRID=boundary_shift）取回到同一位置。
            45、ncv_manifest.py、results47.py 与 47 的一切路径都相对当前目录；54 把一次读入的输入字节按相同布局物化到
            系统临时目录的私有镜像，并在镜像中执行 47 的核对函数；缓存相关核对与计算只读取该镜像。
            work_code_problems()：副本须与项目文件逐字节相同，第三轮网格的 SHA-256 须为协议登记值 GRIDS_SHA256。
  取出 47   load47()：从 47 原文取出指定的顶层函数与赋值语句，以原文件名编译，在调用方给出的命名空间里按原文顺序执行。
  跨轮核对  load_cache() 按 45 读格点缓存的规则读取并校验记录；crossround() 比较两轮共有候选的内层 MSE，以及两轮选中
            同一候选的外层折的重训练预测。
  发布      publish()：先把全部产物写进 04outputs 下的临时目录；改动正式路径前复核输入字节、代码、47 完成标记，以及两轮
            同指纹缓存的文件集合与 SHA-256，任一变化即不发布。通过后删除旧标记、逐个原子替换产物，最后原子写完成标记；
            写出后再核一次，不符即删去标记。
  缓存集合  第三轮按 45 的规则选择 45memo_none_<LABEL>_<16位指纹>*.jsonl，指纹后的任意后缀均属于同一集合；第二轮使用
            47 完成标记登记的精确文件集合。参与核对和计算的选中缓存来自私有镜像，其他身份的缓存只作为活动目录残留提示。
  读取      resolve()：47 的正式结果有效，且其完成标记与 54 运行时读到的逐字节相同；54 标记记录的输入、代码与产物的哈希
            与当前文件逐一相符；工作目录中 45 产物的文件集合与运行时相同；才返回标记，否则报错停下。
            54 在锁内一次读入第二轮登记缓存和第三轮 manifest 指纹缓存；标记记录每个文件的 SHA-256，resolve() 每次核对
            同指纹集合无增删且哈希不变，其他指纹缓存忽略并由 54 运行日志列为残留。
  清理      invalidate_stale()：正式路径上的产物没有有效完成标记时，连同失效的标记一并删除（54 每次运行开始时调用）。
  并发      54 全程持有 47 的运行锁 04outputs/.47run.lock（cnn_fetch.sh 向工作目录发布取回的结果期间也持有它），
            以及工作目录锁 WORK_LOCK（shift_local.sh 运行 45 期间持有）；工作目录中有 45 进程时不运行（results47.activity(WORK)）。

不读 05data。所有路径相对项目根目录。
"""
import ast
import fcntl
import glob
import json
import os
import re
import shutil
import tempfile
import time

import numpy as np

import results47 as r47
from ncv_manifest import sha256_bytes, sha256_file, write_json_atomic

OUT = '04outputs'
WORK = f'{OUT}/45shift'
WORK_LOCK = f'{WORK}/.45run.lock'
GRIDS_SRC = '02code/grids_boundary_shift.py'
GRIDS_SHA256 = 'a4579b1efa4a60e446bcd66e7606b70b136f5bcf1718bac03366c658775630f9'   # 协议 C3 第三轮网格行登记值
SCRIPT54 = '02code/54rebuttal_R1_4_boundary_shift.py'
SCRIPT47 = '02code/47rebuttal_R1_1_paired_tests.py'
# 工作目录中的代码（相对工作目录的路径）→ 须与之逐字节相同的项目文件
WORK_COPIES = {'02code/45rebuttal_R1_4_repeated_ncv.py': '02code/45rebuttal_R1_4_repeated_ncv.py',
               '02code/16unified_comparison.py': '02code/16unified_comparison.py',
               '02code/ncv_manifest.py': '02code/ncv_manifest.py',
               '02code/grids.py': GRIDS_SRC}
CODE_FILES = (SCRIPT54, '02code/results54.py', SCRIPT47, '02code/results47.py', '02code/45rebuttal_R1_4_repeated_ncv.py',
              '02code/16unified_comparison.py', '02code/ncv_manifest.py', '02code/grids.py', GRIDS_SRC)
NAMES = ('table1', 'paired_tests', 'crossround', 'boundary', 'provenance')
COMPLETE = f'{OUT}/54rebuttal_COMPLETE.json'
SCHEMA = '54rebuttal-marker-1'
CACHE_NAME = re.compile(r'^45memo_([^_]+)_([^_]+)_([0-9a-f]{16})(.*)\.jsonl$')


def out_path(name: str) -> str:
    if name not in NAMES:
        raise KeyError(name)
    return f'{OUT}/54rebuttal_{name}.csv'


def work_inputs() -> list[str]:
    """工作目录中 45 的汇总、预测与清单；缓存由已冻结的 marker/manifest 字节另行派生。"""
    return sorted(p for p in glob.glob(f'{WORK}/{OUT}/45rebuttal_ncv_*')
                  if not os.path.basename(p).startswith('._') and not p.endswith('.tmp'))


def cache_inputs(snapshot_bytes: dict[str, bytes]) -> list[str]:
    """按已冻结的 47 marker 与第三轮 manifest 派生缓存；第三轮集合与 45 的 <memo_stem>*.jsonl 相同。"""
    marker = json.loads(snapshot_bytes[r47.COMPLETE].decode('utf-8'))
    labels: set[str] = set()
    paths: set[str] = set()
    for path, raw in snapshot_bytes.items():
        if not path.startswith(f'{WORK}/{OUT}/45rebuttal_ncv_') or not path.endswith('_manifest.json'):
            continue
        try:
            manifest = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        memo = manifest.get('memo_fingerprint')
        method = manifest.get('method')
        if not isinstance(memo, str) or re.fullmatch(r'[0-9a-f]{16}', memo) is None or not isinstance(method, str):
            continue
        label = method.replace('-', '')
        labels.add(label)
        paths.update(glob.glob(f'{WORK}/{OUT}/45memo_none_{label}_{memo}*.jsonl'))
    paths.update(p for p in (marker.get('caches') or {})
                 if any(os.path.basename(p).startswith(f'45memo_none_{label}_') for label in labels))
    return sorted(paths)


def cache_identity(path: str) -> tuple[str, str, str, str] | None:
    """缓存身份为（目录、basis、label、16 位指纹）；指纹后的任意后缀属于同一集合。"""
    match = CACHE_NAME.fullmatch(os.path.basename(path))
    if not match:
        return None
    basis, label, memo, _suffix = match.groups()
    return os.path.dirname(path), basis, label, memo


def cache_residue(snapshot_bytes: dict[str, bytes], selected: list[str]) -> list[str]:
    """列出第三轮目录中其他指纹缓存；它们不参与接受或计算。"""
    all_cache = {p for p in glob.glob(f'{WORK}/{OUT}/45memo_*.jsonl')
                 if not os.path.basename(p).startswith('._')}
    selected_identities = {identity for p in selected if (identity := cache_identity(p)) is not None}
    return sorted(p for p in all_cache if cache_identity(p) not in selected_identities)


def materialize_snapshot(snapshot_bytes: dict[str, bytes]) -> str:
    """把一次读入的输入字节写入系统临时目录，保留相对布局；调用方负责删除。"""
    root = tempfile.mkdtemp(prefix='54rebuttal_')
    try:
        for path, raw in snapshot_bytes.items():
            target = os.path.join(root, path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, 'wb') as fh:
                fh.write(raw)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return root


def cache_marker_problems(marker: dict) -> list[str]:
    """核对标记冻结的缓存集合和字节；其他指纹缓存按 45 语义忽略。"""
    expected = marker.get('cache_inputs') or {}
    probs = []
    for p, h in expected.items():
        if not os.path.exists(p):
            probs.append(f'缓存已不存在：{p}')
        elif sha256_file(p) != h:
            probs.append(f'缓存已改动：{p}')
    identities = {identity for p in expected if (identity := cache_identity(p)) is not None}
    for directory, basis, label, memo in sorted(identities):
        pattern = os.path.join(directory, f'45memo_{basis}_{label}_{memo}*.jsonl')
        actual = sorted(q for q in glob.glob(pattern) if cache_identity(q) == (directory, basis, label, memo))
        same = sorted(q for q in expected if cache_identity(q) == (directory, basis, label, memo))
        if actual != same:
            probs.append(f'缓存集合已改变：{pattern}')
    return probs


def snapshot(paths) -> dict[str, bytes]:
    """一次读入给定文件的字节；54 的核对与计算只用这份字节，标记里记的就是它们的哈希。"""
    snap = {}
    for p in paths:
        with open(p, 'rb') as fh:
            snap[p] = fh.read()
    return snap


def code_sha() -> dict[str, str]:
    """54 结果所依赖的代码：CODE_FILES 与工作目录中的代码副本。"""
    out = {p: sha256_file(p) for p in CODE_FILES}
    for p in WORK_COPIES:
        wp = f'{WORK}/{p}'
        out[wp] = sha256_file(wp) if os.path.exists(wp) else ''
    return out


def work_code_problems() -> list[str]:
    """工作目录中的代码副本与项目文件逐字节相同，第三轮网格为协议登记的那一份；返回不符之处。"""
    probs = []
    if sha256_file(GRIDS_SRC) != GRIDS_SHA256:
        probs.append(f'{GRIDS_SRC} 的 SHA-256 不是协议 C3 登记的 {GRIDS_SHA256}')
    for wp, src in WORK_COPIES.items():
        path = f'{WORK}/{wp}'
        if not os.path.exists(path):
            probs.append(f'工作目录缺少 {path}')
        elif sha256_file(path) != sha256_file(src):
            probs.append(f'工作目录的 {path} 与项目的 {src} 不同')
    return probs


def acquire_work_lock():
    """取得工作目录锁（WORK_LOCK 上的 fcntl.flock 排他锁，持有到返回的文件对象关闭或进程退出）。
    另一进程持有时返回 None；工作目录不存在时抛 FileNotFoundError。"""
    fh = open(WORK_LOCK, 'a', encoding='utf-8')     # noqa: SIM115 — 锁须在函数返回后继续持有，由调用方保留文件对象
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def load47(funcs, consts, base: dict, script: str = SCRIPT47) -> dict:
    """从 47 原文取出名为 funcs 的顶层函数与目标名属于 consts 的顶层赋值语句，以原文件名与行号编译，在 base 的副本里
    按原文顺序执行，返回该命名空间。每个名字须恰有一处定义，否则抛 ValueError。"""
    with open(script, encoding='utf-8') as fh:
        tree = ast.parse(fh.read(), filename=script)
    picked: list[ast.stmt] = []
    seen: dict[str, int] = {}
    for node in tree.body:
        names: set[str] = set()
        if isinstance(node, ast.FunctionDef) and node.name in funcs:
            names = {node.name}
        elif isinstance(node, ast.Assign):
            targets = [t for tg in node.targets for t in (tg.elts if isinstance(tg, ast.Tuple) else [tg])]
            names = {t.id for t in targets if isinstance(t, ast.Name) and t.id in consts}
        if names:
            picked.append(node)
            for n in names:
                seen[n] = seen.get(n, 0) + 1
    wanted = set(funcs) | set(consts)
    bad = sorted(n for n in wanted if seen.get(n) != 1)
    if bad:
        raise ValueError(f'{script} 里下列名字不是恰有一处顶层定义：{bad}')
    ns = {'__name__': '_from47', **base}
    exec(compile(ast.Module(body=picked, type_ignores=[]), script, 'exec'), ns)
    return ns


def cache_key(kind, sd, f, cfg) -> str:
    """与 45 的 key_inner、key_refit 相同的键。"""
    return json.dumps([kind, sd, f, cfg], sort_keys=True, ensure_ascii=False, default=str)


def cfg_json(cfg) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def load_cache(paths, *, method_key: str, memo: str, grid, seeds, k: int, n_test: int | dict[int, int]):
    """按 45 读格点缓存的规则读取 paths：只采用方法与缓存指纹相符、键为 [inner|refit, 种子, 折, 候选] 且各项在期望集合内、
    取值为有限数（inner 为 k 个，refit 为 n_test 个；n_test 为 {折号: 测试折样品数} 时按折取）的记录；同一键取值不同记为问题。
    返回（键 → 取值, 问题列表, 文件 → {sha256, records, skipped}）。"""
    ok_cfg = {cfg_json(c) for c in grid}
    values: dict[str, list] = {}
    probs: list[str] = []
    records: dict[str, dict] = {}
    for path in paths:
        with open(path, 'rb') as fh:
            raw = fh.read()
        n_ok = n_bad = 0
        for line in raw.decode('utf-8').splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                kind, sd, f, cfg = json.loads(rec['k'])
                v = rec['v']
                valid = (rec.get('m') == method_key and rec.get('fp') == memo and kind in ('inner', 'refit')
                         and sd in seeds and f in range(k) and cfg_json(cfg) in ok_cfg
                         and isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)
                         and bool(np.all(np.isfinite(v)))
                         and len(v) == (k if kind == 'inner' else n_test[f] if isinstance(n_test, dict) else n_test))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                n_bad += 1
                continue
            if rec['k'] in values and values[rec['k']] != v:
                probs.append(f'{path}：同一键取值不同 {rec["k"]}')
            values[rec['k']] = v
            n_ok += 1
        records[path] = {'sha256': sha256_bytes(raw), 'records': n_ok, 'skipped': n_bad}
    return values, probs, records


def crossround(values2: dict, values3: dict, *, grid2, grid3, picked2: dict, picked3: dict, seeds, k: int, exact: bool):
    """两轮格点缓存的比对。picked2、picked3 为（种子, 折）→ 选中配置的字符串（45 汇总行的 picked 列）。
    逐项比较：两轮共有候选（按 45 缓存键的写法判定相同）在每个（种子, 折）上的内层 MSE；两轮选中同一候选的外层折的
    重训练预测。exact 为真时，没有共有候选、任一项缺失或 JSON 解析后逐元素数值不严格相等都记为问题；否则只列出差值。
    返回（逐候选的比对行, 问题列表）。"""
    keys2 = {cfg_json(c) for c in grid2}
    shared = [c for c in grid3 if cfg_json(c) in keys2]
    rows: list[dict] = []
    probs: list[str] = []
    if not shared:
        probs.append('两轮网格没有共有候选，无从跨轮核对')

    def compare(kind, cfg, cells):
        n_eq, diff, missing = 0, 0.0, 0
        for sd, f in cells:
            key = cache_key(kind, sd, f, cfg)
            a, b = values2.get(key), values3.get(key)
            if a is None or b is None or len(a) != len(b):
                missing += 1
                continue
            # JSON 解析后逐元素数值严格相等；这不是原始 JSON 文本或浮点格式的字节比较。
            n_eq += int(np.array_equal(np.asarray(a), np.asarray(b)))
            diff = max(diff, float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))))
        return n_eq, diff, missing

    cells = [(sd, f) for sd in seeds for f in range(k)]
    for cfg in shared:
        n_eq, diff, missing = compare('inner', cfg, cells)
        rows.append(dict(kind='inner', candidate=str(cfg), n_points=len(cells), n_missing=missing, n_equal=n_eq,
                         max_abs_diff=diff))
        if exact and (missing or n_eq != len(cells)):
            probs.append(f'共有候选 {cfg} 的内层 MSE：{len(cells)} 个（种子, 折）中缺 {missing} 个、JSON 解析后逐元素数值严格相等 {n_eq} 个，'
                         f'最大差 {diff:.3e}')
    by_str = {str(c): c for c in shared}
    same = [(sd, f) for sd, f in cells if picked2.get((sd, f)) is not None
            and picked2.get((sd, f)) == picked3.get((sd, f)) and picked3.get((sd, f)) in by_str]
    for name in sorted({picked3[c] for c in same}):
        sub = [c for c in same if picked3[c] == name]
        n_eq, diff, missing = compare('refit', by_str[name], sub)
        rows.append(dict(kind='refit', candidate=name, n_points=len(sub), n_missing=missing, n_equal=n_eq,
                         max_abs_diff=diff))
        if exact and (missing or n_eq != len(sub)):
            probs.append(f'两轮都选中 {name} 的 {len(sub)} 个外层折的重训练预测：缺 {missing} 个、JSON 解析后逐元素数值严格相等 {n_eq} 个，'
                         f'最大差 {diff:.3e}')
    return rows, probs


def _load(path: str):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh), None
    except (OSError, json.JSONDecodeError) as e:
        return None, f'{path} 无法读取：{e}'


def marker_problems() -> list[str]:
    """核对完成标记与当前文件是否相符；返回不符之处，空表即有效。"""
    if not os.path.exists(COMPLETE):
        return [f'没有标记 {COMPLETE}']
    mk, err = _load(COMPLETE)
    if err:
        return [err]
    if not isinstance(mk, dict) or mk.get('schema') != SCHEMA:
        return [f'{COMPLETE} 的格式不符']
    probs = [f'47 的正式结果无效：{p}' for p in r47.marker_problems(True)]
    inputs = mk.get('inputs') or {}
    if r47.COMPLETE not in inputs:
        probs.append(f'标记未记录 47 的完成标记 {r47.COMPLETE}')
    for p, h in inputs.items():
        if not os.path.exists(p):
            probs.append(f'输入已不存在：{p}')
        elif sha256_file(p) != h:
            probs.append(f'输入在 54 运行后被改动：{p}')
    if sorted(mk.get('work_inputs') or []) != work_inputs() or not set(work_inputs()) <= set(inputs):
        probs.append(f'{WORK} 下 45 产物的文件集合与 54 运行时不同')
    code = mk.get('code') or {}
    if set(code) != set(code_sha()):
        probs.append('标记记录的代码文件集合不符')
    for p, h in code.items():
        if not os.path.exists(p) or sha256_file(p) != h:
            probs.append(f'代码在 54 运行后被改动：{p}')
    outputs = mk.get('outputs') or {}
    if set(outputs) != {out_path(n) for n in NAMES}:
        probs.append('标记记录的产物集合不符')
    for p, o in outputs.items():
        if not os.path.exists(p):
            probs.append(f'产物不存在：{p}')
        elif sha256_file(p) != o.get('sha256'):
            probs.append(f'产物在 54 运行后被改动：{p}')
    status = mk.get('status') or {}
    if not status or any(v != 'ok' for v in status.values()):
        probs.append('标记中各方法状态缺失或有未通过核对的方法')
    probs += cache_marker_problems(mk)
    return probs


def resolve() -> dict:
    """下游取第三轮结果的唯一入口：完成标记有效时返回标记内容，否则报错停下。"""
    probs = marker_problems()
    if probs:
        raise SystemExit('第三轮（54）的产物没有有效完成标记，须先运行 02code/54rebuttal_R1_4_boundary_shift.py：\n  '
                         + '\n  '.join(probs))
    return dict(_load(COMPLETE)[0])


def invalidate_stale() -> list[str]:
    """正式路径上的产物没有有效完成标记时，连同失效的标记一并删除；返回删除的文件。"""
    if not marker_problems():
        return []
    removed = []
    for p in [out_path(n) for n in NAMES] + [COMPLETE]:
        if os.path.exists(p):
            os.remove(p)
            removed.append(p)
    return removed


def publish(frames: dict, *, snapshot_bytes: dict[str, bytes], code: dict[str, str], info: dict) -> str:
    """把 NAMES 中的每个 DataFrame 发布为正式结果，最后写完成标记；返回标记路径。
    snapshot_bytes 为 54 读入的输入字节（须含 47 的完成标记与工作目录中 45 的全部产物），code 为运行开始时的 code_sha()，
    info 至少含 status（各方法状态，须全部为 ok）。改动正式路径之前重读输入与代码、复核 47 的完成标记，与快照不同即抛
    RuntimeError，不改动任何产物；写出标记后再核一次，不符即删去标记并抛 RuntimeError。"""
    if set(frames) != set(NAMES):
        raise ValueError(sorted(frames))
    if not info.get('status') or any(v != 'ok' for v in info['status'].values()):
        raise ValueError('发布要求全部方法状态为 ok')
    snap_work = sorted(p for p in snapshot_bytes if p.startswith(f'{WORK}/{OUT}/45rebuttal_ncv_')
                       and not os.path.basename(p).startswith('._') and not p.endswith('.tmp'))
    if r47.COMPLETE not in snapshot_bytes or not snap_work:
        raise ValueError('快照须含 47 的完成标记与工作目录中 45 的产物')
    stage = tempfile.mkdtemp(prefix='.54stage_', dir=OUT)      # 与正式路径同一文件系统，os.replace 才是原子的
    mp = COMPLETE
    try:
        staged = {}
        for n in NAMES:
            sp = os.path.join(stage, os.path.basename(out_path(n)))
            frames[n].to_csv(sp, index=False, encoding='utf-8-sig')
            staged[n] = sp
        marker = {
            'schema': SCHEMA, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'inputs': {p: sha256_bytes(b) for p, b in snapshot_bytes.items()},
            'cache_inputs': {p: sha256_bytes(b) for p, b in snapshot_bytes.items()
                             if os.path.basename(p).startswith('45memo_') and p.endswith('.jsonl')},
            'work_inputs': snap_work,
            'code': code,
            'outputs': {out_path(n): {'sha256': sha256_file(staged[n]), 'rows': int(len(frames[n]))} for n in NAMES},
            **info,
        }
        changed = [f'{WORK} 下 45 产物的文件集合'] if work_inputs() != snap_work else []
        changed += cache_marker_problems(marker)
        now = snapshot([p for p in snapshot_bytes if os.path.exists(p)])
        changed += sorted(p for p in snapshot_bytes if now.get(p) != snapshot_bytes[p])
        changed += sorted(p for p, h in code_sha().items() if code.get(p) != h)
        changed += [f'47 的完成标记（{q}）' for q in r47.marker_problems(True)]
        if changed:
            raise RuntimeError(f'54 运行期间输入、代码或 47 的正式结果被改动，未发布：{changed}')
        if os.path.exists(mp):
            os.remove(mp)
        for n in NAMES:
            os.replace(staged[n], out_path(n))
        write_json_atomic(marker, mp)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    probs = marker_problems()
    if probs:
        os.remove(mp)
        raise RuntimeError('发布后标记与当前文件不符（输入或代码在发布过程中被改动），已删去标记：' + '；'.join(probs))
    return mp
