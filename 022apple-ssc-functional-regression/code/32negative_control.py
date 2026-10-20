# -*- coding: utf-8 -*-
"""标签置换负控：把 SSC 标签打乱后重跑，看模型还能不能"预测"出东西（投稿清单第 13 条）。

为什么要有这个脚本：表 3 报出 R²=0.799 时，读者有权问一个问题——**这个数字有没有可能是流程本身
制造出来的？** 光谱预处理若跨折使用了全体样本的统计量、折划分若被复用于选参与评价、指标若算错方向，
都会在**标签与光谱毫无关系时**依然给出正的 R²。这类错误的特征是不报错、不中断、看起来一切正常，
只能靠负控暴露：把 y 随机打乱，切断光谱与标签的一切真实关联，再把**完全相同的流程**跑一遍。
若流程干净，零分布应当集中在 0 附近甚至为负（R² 的期望在无信息时略小于 0）；
若零分布中心明显为正，说明流程里有泄漏，表 3 的数字不能用。

为什么固定超参数而不在置换数据上重新选参：置换检验的严格形式要求把整条流水线（含选参）
放进每一次置换，代价是 12 倍。本脚本改为固定该方法在主分析 5 折中**选中次数最多的配置**，
理由是零假设下光谱与标签独立，内层选参无法从噪声里选出系统性有利的配置，因而不会制造正的 R²；
这一简化只影响功效不影响一类错误率。代价写在这里：本检验能排除"流程制造信号"，
不能排除"选参过程本身制造信号"——后者由嵌套 CV 的结构（内层只用训练折）在设计上排除。

为什么每个方法的观测值要用同一固定配置重算、而不是直接取表 3 的嵌套 CV R²：
零分布是在"固定配置 + 5 折 CV"这个协议下生成的，观测值必须同协议才可比。
两者的差值一并报出，用于确认固定配置没有高估。

算力与次数：置换次数由各方法单次 5 折 CV 的实测耗时决定，如实登记，不按一个数字一刀切。
1D-CNN 单次 5 折需约 2 600 s，即使 100 次也要 73 小时，**本轮不做**并在结论里写明。
这不削弱负控的效力：预处理是逐样本的（SNV 按行、SG 按行，无跨样本统计量），
六个方法共用同一批折划分与同一套预处理产物，泄漏若存在必然同时出现在所有方法上，
故已完成的方法足以判定流程是否干净。

运行方式:
    cd <项目根目录>
    python3 02code/32negative_control.py            # 默认跑 PLSR,SVR,FPCR,PFLR
    NC_METHODS=RF NC_PERM=200 python3 02code/32negative_control.py

输出文件:
    04outputs/32negative_control.csv       — 每方法一行：观测 R²、零分布统计量、p 值
    04outputs/32negative_control_perm.csv  — 每次置换的 R²（零分布本体，供复算与画图）
    05logs/32negative_control.log          — 运行日志（由调用方重定向）

只读 05data。
"""
import ast
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

with open('02code/16unified_comparison.py') as _src: exec(_src.read().split("METHODS=['PLSR'")[0])
sys.path.insert(0, '02code')

SEED = 20041210
K = 5
N_PERM = int(os.environ.get('NC_PERM', 1000))
METHODS = os.environ.get('NC_METHODS', 'PLSR,SVR,FPCR,PFLR').split(',')
OUT, PERM_OUT = '04outputs/32negative_control.csv', '04outputs/32negative_control_perm.csv'

# 主分析 5 折中选中次数最多的配置——负控固定用它
ncv = pd.read_csv('04outputs/25nested_cv.csv')
ncv = ncv[ncv.seed == SEED]


def mode_cfg(m):
    s = ncv[ncv.method == m].picked.value_counts().idxmax()
    return (int(s) if m == 'PLSR' else ast.literal_eval(s)), s


folds = list(KFold(K, shuffle=True, random_state=SEED).split(np.arange(len(y))))


def cv_r2(m, cfg, yy, fit):
    """固定配置下的 5 折折外 R²——观测值与每一次置换都走这一个函数，保证协议同一。"""
    oof = np.zeros(len(yy))
    for itr, ite in folds:
        oof[ite] = fit(cfg, itr, ite, SEED, yy)
    return float(r2_score(yy, oof))


print(f'标签置换负控（seed={SEED}，{K} 折，置换 {N_PERM} 次／方法）')
print(f'方法：{", ".join(METHODS)}\n')

rows = pd.read_csv(OUT).to_dict('records') if os.path.exists(OUT) else []
perm_rows = pd.read_csv(PERM_OUT).to_dict('records') if os.path.exists(PERM_OUT) else []
done = {r['method'] for r in rows}

for m in METHODS:
    if m in done:
        print(f'  {m:<8} 已完成，跳过')
        continue
    t0 = time.time()
    cfg, cfg_s = mode_cfg(m)
    base_fit = mk(m)[1]

    # 把 fit 包一层，使它能在置换后的标签上训练（原实现读全局 y）
    def fit(c, itr, ite, sd, yy, _f=base_fit):
        global y
        keep = y
        try:
            y = yy
            return _f(c, itr, ite, sd)
        finally:
            y = keep

    obs = cv_r2(m, cfg, y, fit)
    nested = float(ncv[ncv.method == m].R2.mean())

    rng = np.random.default_rng(SEED)
    null = []
    for i in range(N_PERM):
        r2i = cv_r2(m, cfg, y[rng.permutation(len(y))], fit)
        null.append(r2i)
        perm_rows.append(dict(method=m, i=i, R2=r2i))
        if (i + 1) % max(1, N_PERM // 10) == 0:
            print(f'    {m} 置换 {i + 1}/{N_PERM}  零分布均值 {np.mean(null):+.4f}  '
                  f'最大 {np.max(null):+.4f}  用时 {time.time() - t0:.0f}s', flush=True)
    null_a = np.asarray(null)
    n_ge = int((null_a >= obs).sum())
    p = (1 + n_ge) / (1 + N_PERM)
    rows.append(dict(method=m, config=cfg_s, n_perm=N_PERM, obs_R2=obs, nested_R2=nested,
                     null_mean=float(null_a.mean()), null_sd=float(null_a.std()),
                     null_max=float(null_a.max()), null_q95=float(np.quantile(null_a, 0.95)),
                     n_ge_obs=n_ge, p_value=p, sec=round(time.time() - t0, 1)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(perm_rows).to_csv(PERM_OUT, index=False)
    print(f'  {m:<8} 观测 R²={obs:.3f}（嵌套 CV {nested:.3f}）  '
          f'零分布 {null_a.mean():+.4f}±{null_a.std():.4f}，最大 {null_a.max():+.4f}  '
          f'p={p:.4f}（{N_PERM} 次中 {n_ge} 次 ≥ 观测）  用时 {time.time() - t0:.0f}s\n', flush=True)

d = pd.DataFrame(rows)
print('【标签置换负控汇总】')
print(f'{"方法":<9}{"置换次数":>7}{"观测 R²":>9}{"零分布均值":>11}{"零分布最大":>11}{"p 值":>9}')
for _, r in d.iterrows():
    print(f'{r.method:<9}{r.n_perm:>7d}{r.obs_R2:>9.3f}{r.null_mean:>11.4f}{r.null_max:>11.4f}{r.p_value:>9.4f}')
print('\n判据：零分布均值应接近 0（无信息时 R² 期望略小于 0），观测值应远在零分布之外。')
print(f'→ {OUT}  /  {PERM_OUT}')
