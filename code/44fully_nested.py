# -*- coding: utf-8 -*-
"""把预处理、基设定、候选网格这三层选择全部嵌套进外层训练折，重做方法比较。

为什么要有这个脚本：投稿前的诚信审计（2026-09-07）指出，有 6 处选择步骤位于外层评价之外，
分属候选网格、预处理、MSC 参考谱与基设定四个环节：
  候选网格（2 处）    候选网格经两轮，第二轮取值依据第一轮在同一批外层折上的边界诊断；
  预处理（2 处）      折内嵌套选预处理（38nested_prep.py）只覆盖四个方法；
  MSC 参考谱（1 处）  36prep_leakage.py 的 MSC 参考谱取自全体样本，已在该脚本改正；
  基设定（1 处）      基设定（基族与 L）在 3.1 节选定，用的也是同一批外层折。

38nested_prep.py 只消除了其中一层（预处理）。本脚本把三层一起放进折内：在每个外层训练折内部，
用内层 5 折从「5 预处理 × 基设定 × 12 组超参」中选优，其中网格本身还要先作一次边界诊断与重定；
外层测试折的数据不参与任何一次选择，也不参与网格的形状。

**折内重定规则**（grids.py 那四条的可自动执行版本——折内没有人工判断的余地，规则必须确定性）：
  只有当第一轮选中值落在该维的边界上时才重定；重定为「以选中值为中心重铺 12 个候选」，
  并受该超参的结构上限约束（K ≤ L−1、潜变量数 ≤ 训练样本数−1、λ 取整数次幂）。
  这满足 grids.py 的四条：① 使选中值落入内部；② 候选数仍为 12；③ 扩到结构上限为止；
  ④ 只删从未被选中的那一端。与正文第二轮网格的差别在于取值的铺法由规则算出而非人工给定，
  这是嵌套化的必要代价：人工给定的那一版本身就是被指控的那一步。

基设定作为独立一层而不占 12 个候选预算，与正文的两层结构一致（正文 3.1 节先定基设定，
3.2 节的 12 个候选全给正则化参数）；差别只在于本脚本的基设定也由折内的内层 CV 选出。
正文第一轮的 12 个候选对函数型方法是 L∈{40,80} × 6 个取值；L 拆到基设定层后，
K 与 λ_β 在第一轮的同一取值范围内铺满 12 个，等预算约束因而不变。

范围：PLSR / SVR / FPCR / PFLR。RF 与 1D-CNN 未纳入，理由与 38nested_prep.py 相同且已实测：
本机单次拟合 RF(n=1600, mf=1.0) 为 295 s、1D-CNN(200 epoch) 为 125 s，按本脚本的组合数
分别约需 21 h 与 83 h，超出算力预算。这两个方法在表 1 中排末两位，不承载本文的两条核心论断
（「SVR 不低于两种函数型模型」与「函数型模型不低于全波段 PLSR」）。

协议（与表 1 逐项对齐，只有「三层选择何时作出」一项不同）：
  外层    KFold(5, shuffle=True, random_state=20041210)——与表 1 同一批折
  内层    在外层训练折内 KFold(5, shuffle=True, random_state=20041210)
  候选    5 预处理 ×（函数型 4 种基设定）× 12 组超参，网格含一次折内边界诊断与重定
  选参    内层折外预测的 MSE 最小
  评价    外层测试折上的折外预测拼合后一次算出 R²/RMSE/RPD

输出：04outputs/44fully_nested.csv。只读 05data。
"""
from typing import Any
import hashlib
import json
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR as _SVR

with open('02code/16unified_comparison.py') as _src: exec(_src.read().split("METHODS=['PLSR'")[0])
sys.path.insert(0, '02code')

SEED, K_OUTER, K_INNER = 20041210, 5, 5
# 格点缓存与逐折中间结果的文件名带数据指纹（原始光谱与真值字节的 SHA-256 前 16 位）：数据一变（如协议 B10.10 的清洗）
# 即换新文件，续跑不会沿用别的数据算出的得分
DATA_FP = hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()[:16]
# 默认四个方法、CPU、输出 44fully_nested.csv —— 与本地已跑的那一轮完全一致。
# 远程扩到 RF / 1D-CNN 时用环境变量覆盖，不改默认行为：
#   FN_METHODS=RF,1D-CNN  FN_DEVICE=cuda  FN_JOBS=64  FN_OUT=04outputs/44fully_nested_rfcnn.csv
METHODS = os.environ.get('FN_METHODS', 'PLSR,SVR,FPCR,PFLR').split(',')
OUT = os.environ.get('FN_OUT', '04outputs/44fully_nested.csv')
DEVICE = os.environ.get('FN_DEVICE', 'cpu')
N_JOBS = int(os.environ.get('FN_JOBS', '-1'))   # 远程留余量给同机其他任务

# 并行填缓存：FN_SHARD='i/n' 让本进程只算序号 ≡ i (mod n) 的格点，其余留给同时运行的分片。
# 1D-CNN 单折第一轮 300 个格点、每格点 5 次内层拟合，实测 27 s/次 → 单折 11.2 h、五折 56 h；
# 而 bs=32 下 GPU 利用率只有 3~21%（时间花在 kernel launch 与同步上，不是算力），
# 多进程同卡并行恰好填这段空闲。分片进程只写格点得分，不做选择、不写 partial/CSV；
# 填完后由一个不带 FN_SHARD 的进程读缓存出结果，故分片数不影响任何数字。
_SH = os.environ.get('FN_SHARD', '')
SHARD_I, SHARD_N = (int(_SH.split('/')[0]), int(_SH.split('/')[1])) if _SH else (0, 1)


def msc_fit(A, ref):
    """MSC 是唯一用到跨样本统计量的预处理，参考谱只能来自训练折。"""
    out = np.empty_like(A)
    for i, r in enumerate(A):
        k, b = np.polyfit(ref, r, 1)
        out[i] = (r - b) / k
    return out


PREP = {
    '原始':     (False, lambda A, ref: A),
    'SNV':      (False, lambda A, ref: snv(A)),
    'MSC':      (True,  lambda A, ref: msc_fit(A, ref)),
    'SG1D':     (False, lambda A, ref: savgol_filter(A, 15, 2, deriv=1, axis=1)),
    'SNV+SG1D': (False, lambda A, ref: savgol_filter(snv(A), 15, 2, deriv=1, axis=1)),
}

# 基设定：2 个基族 × L ∈ {40, 80}。函数型方法专用；PLSR/SVR 的这一层为空。
BASIS_FAM = {'傅里叶': fourier_basis, 'B样条': bspline_basis}
BASIS_SETTINGS = [(fam, L) for fam in BASIS_FAM for L in (40, 80)]

# 第一轮网格，与 16unified_comparison.py 的 mk() 逐字一致（不是 grids.py 的第二轮）
FIRST: dict[str, list[Any]] = {   # 六个方法的候选形状不同（整数 / 二元组 / 配置字典）
    'PLSR': list(range(2, 26, 2)),
    'SVR':  [(C, g) for C in [1, 10, 100, 1000] for g in ['scale', 1e-3, 1e-4]],
    # L 拆到基设定层之后，K 与 λ_β 在第一轮的同一取值范围内铺满 12 个，
    # 以保持 7.4 的等预算约束（正文第一轮的 12 个候选是 L∈{40,80} × 6 个取值）。
    'FPCR': [5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30],          # K ∈ [5, 30]，L 由基设定层给出
    'PFLR': [float(10.0 ** e) for e in range(-8, 4)],   # λ_β ∈ [1e-8, 1e3]
    'RF':   [(n, mf) for n in [200, 400, 800] for mf in ['sqrt', 0.1, 0.3, 1.0]],
    '1D-CNN': [dict(w=w, nl=nl, lr=lr, dp=dp, wd=1e-4, bs=32, ep=200)
               for w, nl in [(16, 3), (32, 3), (16, 4)] for lr in [1e-3, 3e-4] for dp in [0.2, 0.4]],
}


def recentre(vals, picked, lo, hi, log10_step=False):
    """以 picked 为中心重铺 12 个候选，clip 到 [lo, hi]。用于折内重定。"""
    n = 12
    if log10_step:
        e = int(round(np.log10(picked)))
        es = list(range(e - n // 2, e - n // 2 + n))
        es = [x for x in es if lo <= x <= hi]
        while len(es) < n:                      # 触到上下限时从另一端补齐
            es = ([es[0] - 1] if es[0] - 1 >= lo else []) + es + \
                 ([es[-1] + 1] if es[-1] + 1 <= hi else [])
            if es[0] - 1 < lo and es[-1] + 1 > hi:
                break
        return [float(10.0 ** x) for x in es[:n]]
    step = max(1, int(round(np.median(np.diff(sorted(vals))))))
    out = [picked + step * (i - n // 2) for i in range(n)]
    out = [int(np.clip(v, lo, hi)) for v in out]
    out = sorted(set(out))
    while len(out) < n:                          # 去重后不足 12 个，向未触限的一端补
        if out[-1] + step <= hi:
            out.append(out[-1] + step)
        elif out[0] - step >= lo:
            out.insert(0, out[0] - step)
        else:
            break
    return out[:n]


def redefine(method, picked, grid, L=None, n_train=None):
    """一次折内边界诊断与重定。返回 (新网格, 是否重定)。选中值不在边界则原样返回。"""
    if method == 'PLSR':
        if picked in (min(grid), max(grid)):
            return recentre(grid, picked, 2, min(200, n_train - 1)), True
    elif method == 'SVR':
        Cs = sorted({c for c, _ in grid}); gms = [g for _, g in grid[:3]]
        pc = picked[0]
        if pc in (min(Cs), max(Cs)):
            e = int(round(np.log10(pc)))
            new_Cs = [float(10.0 ** x) for x in range(e - 2, e + 2)]
            return [(C, g) for C in new_Cs for g in gms], True
    elif method == 'FPCR':
        if picked in (min(grid), max(grid)):
            return recentre(grid, picked, 2, L - 1), True          # 结构上限 K ≤ L−1
    elif method == 'PFLR':
        if picked in (min(grid), max(grid)):
            return recentre(grid, picked, -12, 4, log10_step=True), True
    elif method == 'RF':
        ns = sorted({n for n, _ in grid}); mfs = [mf for _, mf in grid[:4]]
        if picked[0] in (min(ns), max(ns)):          # 树数是被截断的那一维
            e = int(round(np.log2(picked[0])))
            new_ns = [int(2 ** x) for x in range(e - 1, e + 2)]
            return [(n, mf) for n in new_ns for mf in mfs], True
    elif method == '1D-CNN':
        nls = sorted({c['nl'] for c in grid})
        if picked['nl'] in (min(nls), max(nls)):     # 层数是被截断的那一维；5 为结构上限
            base = [(w, nl) for w in (16, 32)
                    for nl in range(max(2, picked['nl'] - 1), min(5, picked['nl'] + 1) + 1)]
            out = [dict(w=w, nl=nl, lr=lr, dp=0.2, wd=1e-4, bs=32, ep=200)
                   for w, nl in base for lr in (1e-3, 3e-4)]
            return out[:12] if len(out) >= 12 else grid, len(out) >= 12
    return grid, False


def prep_of(pname, train_idx):
    """按 train_idx 所指的训练样本拟合该预处理，再应用到全体样本。"""
    need_ref, f = PREP[pname]
    return f(X, X[train_idx].mean(0) if need_ref else None)


_basis_cache: dict[Any, Any] = {}


def basis_of(fam, L):
    if (fam, L) not in _basis_cache:
        B = BASIS_FAM[fam](t, L)
        J, Rm = gram_and_pen(B, t)
        _basis_cache[(fam, L)] = (B, J, Rm)
    return _basis_cache[(fam, L)]


def fit_one(method, cfg, Xd, coef, J, Rm, a, b):
    """在样本索引 a 上拟合、在 b 上预测。Xd 为该预处理后的谱；coef/J/Rm 为该基设定下的量。"""
    if method == 'PLSR':
        return PLSRegression(n_components=min(cfg, len(a) - 1)).fit(Xd[a], y[a]).predict(Xd[b]).ravel()
    if method == 'SVR':
        return _SVR(C=cfg[0], gamma=cfg[1]).fit(Xd[a], y[a]).predict(Xd[b])
    if method == 'RF':
        return RandomForestRegressor(n_estimators=cfg[0], max_features=cfg[1],
                                     random_state=SEED, n_jobs=N_JOBS).fit(Xd[a], y[a]).predict(Xd[b])
    if method == '1D-CNN':
        return cnn_fit_dev(Xd[a], y[a], Xd[b], cfg, SEED)
    if method == 'FPCR':
        return fpcr(coef[a], y[a], coef[b], J, min(cfg, len(a) - 1))
    return pflr(coef[a], y[a], coef[b], J, Rm, cfg)


def cnn_fit_dev(Xtr, ytr, Xte, cfg, seed):
    """与共享块 cnn_fit 逐项一致（同网络、同优化器、同调度器、同种子、同 epoch），只改计算设备。

    共享块被 9 个脚本 exec 复用，改它会动到论文已有的数字，故在此另写一份而不是改原函数。
    mini-batch 的随机排列固定在 CPU 上生成：`torch.randperm(n, device='cuda')` 走的是 CUDA 随机数流，
    与 CPU 不是同一串数，会让每个 epoch 的 batch 顺序完全不同（实测逐样本差达 1.349 °Brix，
    远超浮点误差量级）。改回 CPU 生成后，剩余差异只来自 dropout 的设备相关随机流与浮点归约顺序。
    跑 GPU 前必须做一次同配置的 CPU/GPU 对照并把差异如实记入 05logs/44_devcheck.log。
    """
    dev = torch.device(DEVICE)
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = ytr.mean(), ytr.std()
    A = torch.tensor(((Xtr - mu) / sd)[:, None, :], dtype=torch.float32, device=dev)
    Bt = torch.tensor(((Xte - mu) / sd)[:, None, :], dtype=torch.float32, device=dev)
    Y = torch.tensor(((ytr - ym) / ys)[:, None], dtype=torch.float32, device=dev)
    m = CNN(cfg['w'], cfg['dp'], cfg['nl']).to(dev)
    opt = torch.optim.Adam(m.parameters(), cfg['lr'], weight_decay=cfg['wd'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['ep'])
    lf = nn.MSELoss()
    n = len(A)
    for _e in range(cfg['ep']):
        m.train()
        idx = torch.randperm(n)          # 在 CPU 上生成，与共享块 cnn_fit 取到同一串随机数
        for i in range(0, n, cfg['bs']):
            b = idx[i:i + cfg['bs']].to(dev)
            opt.zero_grad()
            lf(m(A[b]), Y[b]).backward()
            opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        return m(Bt).cpu().numpy().ravel() * ys + ym


def build_cache(settings, itr, functional):
    """每个 (预处理, 基设定) 组合下的谱与系数只算一次，供两轮网格共用。"""
    cache = {}
    for pname in PREP:
        Xd = prep_of(pname, itr)
        for fam, L in settings:
            if functional:
                B, J, Rm = basis_of(fam, L)
                cache[(pname, fam, L)] = (Xd, to_coef(Xd, B, Rm), J, Rm)
            else:
                cache[(pname, fam, L)] = (Xd, None, None, None)
    return cache


def search(method, grid, cache, itr, inner, fold):
    """在给定网格下扫全部 (预处理 × 基设定 × 网格)，返回最优 (se, pname, fam, L, cfg)。

    grid 对函数型方法是超参列表（K 或 λ_β），基设定由 cache 的键给出；
    评价只用外层训练折 itr 内部的内层折外预测。

    每个 (折, 预处理, 基设定, 超参) 的内层得分算完即落盘（见 memo_put），
    进程被杀后重启只重算未落盘的那一个格点。记忆化不改变任何结果：
    fit_one 对每个方法都是确定性的（CNN 每次调用都 torch.manual_seed(SEED)，
    不依赖调用顺序），命中缓存与重算得到同一个 se；遍历顺序不变，
    故 `se < best[0]` 的平局取舍也与不带缓存时逐位一致。
    """
    best: tuple[Any, ...] = (np.inf, None, None, None, None)
    for (pname, fam, L), (Xd, coef, J, Rm) in cache.items():
        for cfg in grid:
            key = memo_key(fold, pname, fam, L, cfg)
            if key in _MEMO:
                se = _MEMO[key]
            else:
                iv_oof = np.zeros(len(itr))
                for ia, iv in inner:
                    iv_oof[iv] = fit_one(method, cfg, Xd, coef, J, Rm, itr[ia], itr[iv])
                se = float(mean_squared_error(y[itr], iv_oof))
                memo_put(method, key, se)
            if se < best[0]:
                best = (se, pname, fam, L, cfg)
    return best


folds = list(KFold(K_OUTER, shuffle=True, random_state=SEED).split(np.arange(len(y))))

print(f'三层选择全部嵌套进外层训练折（外层 {K_OUTER} 折、内层 {K_INNER} 折，seed={SEED}）')
print(f'本次方法：{", ".join(METHODS)}；设备 {DEVICE}；预处理 {len(PREP)} 种；'
      f'函数型基设定 {len(BASIS_SETTINGS)} 种（仅 FPCR/PFLR 用到）；网格 12 组，含一次折内边界诊断与重定')
if SHARD_N > 1:
    print(f'分片 {SHARD_I}/{SHARD_N}：只算本分片的格点得分并落盘，'
          f'三层选择与汇总由不带 FN_SHARD 的进程完成\n')
else:
    print()

def part_path(m):
    """逐折中间结果的落盘路径。长任务被宿主重启打断过一次（2026-09-12 01:21，
    1D-CNN 跑完 4/5 折后容器重启，26 h 的计算因为没有中间落盘而全部作废），故每折算完即存。"""
    return f"{os.path.dirname(OUT) or '.'}/44partial_{m.replace('-', '')}_{DATA_FP}.npz"


def load_part(m, n):
    """读回已完成的折：返回 (oof 数组, 已完成折号集合, trace 列表)。"""
    f = part_path(m)
    if not os.path.exists(f):
        return np.zeros(n), set(), []
    d = np.load(f, allow_pickle=True)
    return d['oof'], set(d['done'].tolist()), list(d['trace'])


def save_part(m, oof, done, trace):
    np.savez(part_path(m), oof=oof, done=np.array(sorted(done)), trace=np.array(trace, dtype=object))


# 折级断点不够细：本项目的容器实测会在十小时内重启两次（2026-09-12 01:21 与 11:07~11:27），
# 而 1D-CNN 单折要 1.7 h 以上，重启一次就整折作废、连断点都写不出来。
# 故把粒度降到内层搜索的单个格点——每个格点约 5 次拟合，重启最多丢这一格。
_MEMO: dict[str, float] = {}


def memo_path(m):
    return f"{os.path.dirname(OUT) or '.'}/44memo_{m.replace('-', '')}_{DATA_FP}.jsonl"


def memo_key(fold, pname, fam, L, cfg):
    """折号必须进键：不同外层折的训练集与内层划分都不同，得分不可跨折复用。"""
    return json.dumps([fold, pname, fam, L, cfg], sort_keys=True, ensure_ascii=False, default=str)


def memo_load(m):
    """读回已落盘的格点得分。末行可能被杀进程截断，按行容错。"""
    _MEMO.clear()
    f = memo_path(m)
    if not os.path.exists(f):
        return
    with open(f, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            _MEMO[rec['k']] = rec['se']


def memo_put(m, key, se):
    """追加一行并 fsync：容器可能在任何时刻被杀，缓冲区里的行等于没写。"""
    _MEMO[key] = se
    with open(memo_path(m), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps({'k': key, 'se': se}, ensure_ascii=False) + '\n')
        fh.flush()
        os.fsync(fh.fileno())


def fill_shard(m, k, grid, cache, itr, inner, rnd, t0):
    """把本分片负责的格点算出来落盘，返回 (新算数, 命中数)。不做选择、不写 partial/CSV。"""
    n_new = n_hit = 0
    gi = -1
    for (pname, fam, L), (Xd, coef, J, Rm) in cache.items():
        for cfg in grid:
            gi += 1
            if gi % SHARD_N != SHARD_I:
                continue                            # 归别的分片
            key = memo_key(k, pname, fam, L, cfg)
            if key in _MEMO:
                n_hit += 1
                continue                            # 已落盘
            iv_oof = np.zeros(len(itr))
            for ia, iv in inner:
                iv_oof[iv] = fit_one(m, cfg, Xd, coef, J, Rm, itr[ia], itr[iv])
            memo_put(m, key, float(mean_squared_error(y[itr], iv_oof)))
            n_new += 1
            print(f'  [{SHARD_I}/{SHARD_N}] {m} fold={k} 第{rnd}轮 第 {gi:>3} 格点 '
                  f'{pname}/{fam or "—"}{L or ""}/{cfg} 已存  '
                  f'(本轮新算 {n_new}，复用 {n_hit}，{time.time() - t0:.0f}s)', flush=True)
    return n_new, n_hit


def first_round_full(m, k, cache):
    """本折第一轮网格是否已被所有分片填满 —— 填满才能确定重定网格。"""
    return all(memo_key(k, pname, fam, L, cfg) in _MEMO
               for (pname, fam, L) in cache for cfg in FIRST[m])


if SHARD_N > 1:
    # 分片填缓存：只算格点得分，选择与汇总交给不带 FN_SHARD 的进程。
    # 刻意不写 partial/CSV —— 分片自己看到的 best 是残缺的（别的格点在别的进程里）。
    # 第二轮网格由第一轮的选中值决定，所以要等所有分片把本折第一轮填满才能推出来；
    # 推的过程只查缓存（search 全命中），与汇总进程遍历同一份缓存、同一顺序，得到同一个
    # 选中值与同一个重定网格，故第二轮的分片划分也与汇总进程要的格点集合一致。
    for m in METHODS:
        memo_load(m)
        functional = m in ('FPCR', 'PFLR')
        settings = BASIS_SETTINGS if functional else [(None, None)]
        t0 = time.time()
        tot_new = tot_hit = 0
        for rnd in (1, 2):
            for k, (itr, _ite) in enumerate(folds):
                inner = list(KFold(K_INNER, shuffle=True, random_state=SEED).split(itr))
                cache = build_cache(settings, itr, functional)
                if rnd == 1:
                    grid = FIRST[m]
                else:
                    while not first_round_full(m, k, cache):
                        print(f'  [{SHARD_I}/{SHARD_N}] {m} fold={k} 等其他分片填完第一轮…',
                              flush=True)
                        time.sleep(60)
                        memo_load(m)
                    _se1, _p1, _f1, L1, c1 = search(m, FIRST[m], cache, itr, inner, k)
                    grid, moved = redefine(m, c1, FIRST[m], L=L1, n_train=len(itr))
                    if not moved:
                        print(f'  [{SHARD_I}/{SHARD_N}] {m} fold={k} 未触发重定，无第二轮',
                              flush=True)
                        continue
                a, b = fill_shard(m, k, grid, cache, itr, inner, rnd, t0)
                tot_new += a
                tot_hit += b
        print(f'  [{SHARD_I}/{SHARD_N}] {m} 本分片填完：新算 {tot_new}，复用 {tot_hit}，'
              f'{time.time() - t0:.0f}s', flush=True)
    sys.exit(0)

rows = []
for m in METHODS:
    t0 = time.time()
    functional = m in ('FPCR', 'PFLR')
    settings = BASIS_SETTINGS if functional else [(None, None)]
    oof, done, trace = load_part(m, len(y))
    memo_load(m)
    if done or _MEMO:
        print(f'  {m}: 续跑，已完成折 {sorted(done)}，'
              f'另有 {len(_MEMO)} 个格点得分可复用', flush=True)
    for k, (itr, ite) in enumerate(folds):
        if k in done:
            continue
        inner = list(KFold(K_INNER, shuffle=True, random_state=SEED).split(itr))
        cache = build_cache(settings, itr, functional)

        g1 = FIRST[m]
        se1, p1, f1, L1, c1 = search(m, g1, cache, itr, inner, k)
        g2, moved = redefine(m, c1, g1, L=L1, n_train=len(itr))
        if moved:
            se2, p2, f2, L2, c2 = search(m, g2, cache, itr, inner, k)
            if se2 < se1:
                se, pn, fam, L, cfg = se2, p2, f2, L2, c2
            else:                       # 重定后的网格没有更好，保留第一轮的选择
                se, pn, fam, L, cfg = se1, p1, f1, L1, c1
        else:
            se, pn, fam, L, cfg = se1, p1, f1, L1, c1

        Xd, coef, J, Rm = cache[(pn, fam, L)]
        oof[ite] = fit_one(m, cfg, Xd, coef, J, Rm, itr, ite)
        trace.append(dict(fold=k, prep=pn, fam=fam, L=L, cfg=cfg, redefined=moved))
        done.add(k)
        save_part(m, oof, done, trace)          # 每折即时落盘，宿主重启后可续
        print(f'  {m:<5} fold={k}  预处理={pn:<9} 基设定={fam or "—"}/{L or "—"}  '
              f'超参={cfg}  网格重定={"是" if moved else "否"}  (已存盘)', flush=True)

    trace = sorted(trace, key=lambda x: x['fold'])
    r2 = float(r2_score(y, oof))
    rmse = float(np.sqrt(mean_squared_error(y, oof)))
    rows.append(dict(method=m, R2=r2, RMSE=rmse, RPD=float(np.std(y)) / rmse,
                     prep='|'.join(str(x['prep']) for x in trace),
                     basis='|'.join(f'{x["fam"]}{x["L"]}' if x['fam'] else '—' for x in trace),
                     n_redefined=sum(x['redefined'] for x in trace),
                     sec=round(time.time() - t0, 1)))
    print(f'  {m:<5} R²={r2:.3f}  RMSE={rmse:.3f}  '
          f'网格重定 {sum(x["redefined"] for x in trace)}/{K_OUTER} 折  ({time.time()-t0:.0f}s)\n', flush=True)

df = pd.DataFrame(rows).sort_values('R2', ascending=False)
df.to_csv(OUT, index=False, encoding='utf-8-sig')
print('【三层全嵌套后的方法排序】')
print(f'{"方法":<8}{"R²":>8}{"RMSE":>9}{"RPD":>8}   各折选中的预处理 / 基设定')
for _, r in df.iterrows():
    print(f'{r.method:<8}{r.R2:>8.3f}{r.RMSE:>9.3f}{r.RPD:>8.2f}   {r.prep}   {r.basis}')
print('\n排序：' + ' > '.join(df.method))
print(f'→ {OUT}')
