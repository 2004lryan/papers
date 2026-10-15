# -*- coding: utf-8 -*-
"""
data_clean.py: 数据清洗四条规则的唯一实现（协议 B10.10，照课题组 015 项目的数据预处理）

作为模块导入使用，不单独运行。16unified_comparison.py 共享段与 08fda_smallsample.py 文件头的 load()
读入一个年份的原始记录后调用 keep_mask()，经这两处读数的脚本都使用清洗后的数据；逐行清单由
59data_clean.py 输出。

四条规则都在一个年份的原始记录（2 048 个光谱点、SSC、产地编码均为有限值的行）上各自判定，剔除其并集：
    R1 光谱离群   年份内对各波长点标准化（sklearn StandardScaler，总体标准差），计算到均值的欧氏距离，
                  剔除严格大于 90% 分位数者，即 015 的 10% 固定分位剔除
                  （015 02code/02_data_processing.py::detect_outliers_mahalanobis）
    R2 SSC 越界   SSC > 20 °Brix（015 02code/01_export_utils.py::SSC_PHYSICAL_MAX）
    R3 逐位相同   2 048 点逐点相同的光谱整组剔除：同一条光谱不可能属于 SSC 不同的果实（015 同文件硬规则 2）
    R4 检测器饱和 任一点读数达到 65 535（本稿增补）
与 015 的两处差异（理由见协议 B10.10）：R1 用本稿建模的全部 2 048 个波长点；四条规则都在原始记录上判定，
重复组整组剔除不受 R1 影响。
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

SSC_MAX = 20.0         # R2：阈值沿用课题组此前对这批数据的预处理
ADC_MAX = 65535.0      # R4：光谱仪 16 位检测器的读数上限
R1_PCT = 90.0          # R1：剔除距离高于 90% 分位数者
RULES = ('R1', 'R2', 'R3', 'R4')


def rule_flags(X, y, z):
    """逐行判定四条规则。

    X 为一个年份 Excel 的全部数据行（行 × 2 048 个光谱点），y 为 SSC，z 为产地编码。
    返回 (flags, dist)：flags 为 {'valid', 'R1', 'R2', 'R3', 'R4'} → 与行数等长的 bool 数组；
    dist 为 R1 的距离（无效行为 nan）。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(y)
    valid = np.isfinite(X).all(1) & np.isfinite(y) & np.isfinite(z)
    iv = np.flatnonzero(valid)

    sc = StandardScaler().fit_transform(X[iv])
    d = np.linalg.norm(sc - sc.mean(0), axis=1)
    dist = np.full(n, np.nan)
    dist[iv] = d
    r1 = np.zeros(n, bool)
    r1[iv[d > np.percentile(d, R1_PCT)]] = True

    r2 = valid & (y > SSC_MAX)

    groups: dict[bytes, list[int]] = {}
    for i in iv:
        groups.setdefault(X[i].tobytes(), []).append(int(i))
    r3 = np.zeros(n, bool)
    for members in groups.values():
        if len(members) > 1:
            r3[members] = True

    r4 = np.zeros(n, bool)
    r4[iv] = X[iv].max(1) >= ADC_MAX
    return {'valid': valid, 'R1': r1, 'R2': r2, 'R3': r3, 'R4': r4}, dist


def keep_mask(X, y, z):
    """保留的行：有效且四条规则都未命中。"""
    f, _ = rule_flags(X, y, z)
    return f['valid'] & ~(f['R1'] | f['R2'] | f['R3'] | f['R4'])
