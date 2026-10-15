# -*- coding: utf-8 -*-
"""45 脚本结果的来源清单（manifest）：由 45 写出、由 47 核对，两边共用本模块的哈希与字段定义。

为什么要有这个模块：47 原先只要求每个方法「指纹唯一、预测文件与汇总行指纹相同」，没有要求这个指纹就是
本轮代码、数据与环境应当算出的那一个。由别的代码、网格或环境算出、但自身一致的整套 CSV 也能进入最终统计
（第 4 轮部署前代码审查）。现在 45 在某个（基, 方法）的 5 个种子全部完成后写一份清单，记下结果文件、代码、
数据的 SHA-256 与库版本、设备；47 逐项与当前代码、由原始数据重算的哈希、声明的运行环境比对，任一不符即不出
最终结果。写与核共用同一套定义，避免两边各写一份而悄悄不一致。

指纹分两层：
  缓存指纹  45 的 fingerprint()：只含决定格点缓存取值的量（数据、预处理后光谱、基系数、网格、16 共享段、库版本、
            拟合函数源码），45 其余代码的改动不会让格点缓存失效。
  结果指纹  row_fingerprint()：缓存指纹再加 45 脚本全文与本模块全文的 SHA-256，写进每一行汇总与预测。
            45 只沿用结果指纹与本次运行相同的已有种子，所以选择、重训练、汇总逻辑有任何改动，
            已有 CSV 都会整组按格点缓存重算，旧代码写出的行不会被当作本次结果。

清单字段（schema ncv-manifest-1）：
  analysis_basis / method / method_key / device / n_jobs        本组结果是什么、在哪算的
  seeds / k_outer / k_inner / L / grid                           协议参数
  fingerprint / memo_fingerprint                                 结果指纹与缓存指纹
  files.csv / files.pred                                         路径、SHA-256、行数
  code                                                           45 全文、16 共享段、grids.py、本模块的 SHA-256
  data                                                           真值 y、预处理后光谱 Xp、基系数 COEF（非函数型为 null）的 SHA-256
  env                                                            平台、Python 与各库版本；cuda 另记 CUDA、cuDNN 与显卡型号

不读 05data；除 write_json_atomic 写出的清单外不写任何文件。所有路径相对项目根目录。
"""
import hashlib
import json
import os
import platform
import time

import numpy as np

SCRIPT45 = '02code/45rebuttal_R1_4_repeated_ncv.py'
SCRIPT16 = '02code/16unified_comparison.py'
GRIDS_PY = '02code/grids.py'
THIS = '02code/ncv_manifest.py'
SHARED_END = "METHODS=['PLSR'"
SCHEMA = 'ncv-manifest-1'


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, 'rb') as fh:
        return sha256_bytes(fh.read())


def sha256_array(a) -> str:
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


def shared16_text() -> str:
    """16 脚本 METHODS 定义之前的共享段：数据读取、预处理、两族基函数、pflr/fpcr、CNN 网络与 mk()。"""
    with open(SCRIPT16, encoding='utf-8') as fh:
        return fh.read().split(SHARED_END)[0]


def code_sha256() -> dict:
    """决定 45 结果的代码：45 全文、16 共享段、grids.py 与本模块。"""
    return {'45': sha256_file(SCRIPT45), '16_shared': sha256_bytes(shared16_text().encode('utf-8')),
            'grids': sha256_file(GRIDS_PY), 'manifest': sha256_file(THIS)}


def row_fingerprint(memo_fp: str, code: dict) -> str:
    """结果指纹：缓存指纹 + 45 全文与本模块的 SHA-256。"""
    return sha256_bytes(json.dumps([memo_fp, code['45'], code['manifest']]).encode())[:16]


def basis_system(ns: dict, basis: str, L: int):
    """函数型基系统：基矩阵、Gram 矩阵 J、粗糙惩罚 R，以及预处理后光谱在该基下的系数（45 与 47 共用）。
    ns 为执行过 16 共享段的命名空间；basis 取 'bspline' 或 'fourier'。"""
    B = ns[f'{basis}_basis'](ns['t'], L)
    J, R = ns['gram_and_pen'](B, ns['t'])
    return B, J, R, ns['to_coef'](ns['Xp'], B, R)


def data_sha256(y, Xp, coef=None) -> dict:
    return {'y': sha256_array(y), 'Xp': sha256_array(Xp), 'COEF': None if coef is None else sha256_array(coef)}


def env_info(device: str) -> dict:
    """运行环境：平台、Python 与实际导入的库版本；device 为 cuda 时另记 CUDA、cuDNN 版本与显卡型号。"""
    import pandas
    import scipy
    import sklearn
    import torch
    env = {'platform': f'{platform.system()} {platform.machine()}', 'python': platform.python_version(),
           'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
           'pandas': pandas.__version__, 'torch': torch.__version__}
    if device == 'cuda':
        env.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0))
    return env


def normalized(obj):
    """网格等协议参数按 JSON 规范化（元组变列表、非 JSON 类型转字符串），便于两边逐字比较。"""
    return json.loads(json.dumps(obj, default=str))


def manifest_path(csv_path: str) -> str:
    if not csv_path.endswith('.csv'):
        raise ValueError(csv_path)
    return csv_path[:-4] + '_manifest.json'


def build_manifest(*, analysis_basis: str, method: str, method_key: str, fingerprint: str, memo_fingerprint: str,
                   device: str, n_jobs, seeds: list, k_outer: int, k_inner: int, L, grid,
                   csv_path: str, csv_bytes: bytes, csv_rows: int, pred_path: str, pred_bytes: bytes, pred_rows: int,
                   code: dict, data: dict, env: dict) -> dict:
    return {
        'schema': SCHEMA, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'analysis_basis': analysis_basis, 'method': method, 'method_key': method_key,
        'device': device, 'n_jobs': n_jobs,
        'seeds': [int(s) for s in seeds], 'k_outer': int(k_outer), 'k_inner': int(k_inner), 'L': L,
        'grid': normalized(grid),
        'fingerprint': fingerprint, 'memo_fingerprint': memo_fingerprint,
        'files': {'csv': {'path': csv_path, 'sha256': sha256_bytes(csv_bytes), 'rows': int(csv_rows)},
                  'pred': {'path': pred_path, 'sha256': sha256_bytes(pred_bytes), 'rows': int(pred_rows)}},
        'code': code, 'data': data, 'env': env,
    }


def write_json_atomic(obj: dict, path: str) -> None:
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, sort_keys=True)
        fh.write('\n')
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
