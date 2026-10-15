# -*- coding: utf-8 -*-
"""把产出论文数字的模型落盘并出模型卡（项目规范 第五章 Item 6 与第零章 F1）。

为什么要有这个脚本：F1 是四条静默失效红线之一，其特征是**违反时不报错、不中断、看起来一切正常**。
表 3 及其下游的每一个数字都来自六个方法在 5 个外层折上的模型，但这些模型此前跑完即弃，
`07checkpoints/` 只有一个 .gitkeep。第五章 Item 6 对此写得很直白：「用的是 sklearn 所以没权重可存」
不成立，joblib 存得下，而且必须存。项目 020 正是靠这个理由一路跑到投稿才发现 checkpoint 目录全空。

落盘口径：**一个方法一个 checkpoint**，内容是该方法 5 个外层折的模型与各折选中的配置。
不按折拆成 30 个目录，是因为表 3 报告的是 5 折的均值，5 折模型合起来才对应论文里的一行。

**落盘的东西必须能复现论文数字**，否则存了也没有证据价值。故本脚本对每个方法都做一次回放：
用落盘对象里的参数重算 5 折的折外预测，与 `04outputs/25nested_cv_pred.csv` 逐位比对，
差异超过 1e-9 即 raise。该断言对六个方法一律生效，1D-CNN 也不例外：它在 CPU 上按同一种子、
同一批次顺序重训练，2026-09-06 实测最大绝对差为 1.78e-15，与其余五个方法同在浮点精度极限上。两个函数型模型没有 sklearn 那样的估计器对象，本脚本为它们写了
拆出拟合参数的版本（fit_pflr / fit_fpcr），回放断言同时验证了这两个实现与
`16unified_comparison.py` 中原函数的一致性。

stage 标注：一律 `pilot`，理由见 `refine-logs/01PROJECT_PROTOCOL.md` 的「模型卡 stage 的具名声明」。
它表示「不经过 CUDA」，不表示这些模型是预实验产物。

运行方式:
    cd <项目根目录>
    python3 02code/30save_final_models.py

输出文件:
    07checkpoints/25nested_cv_<方法>/best.joblib   — 该方法 5 个外层折的模型与配置
    07checkpoints/25nested_cv_<方法>/model_card.md — 模型卡（第十一章 Item 3）
    05logs/30save_final_models.log                 — 运行日志（由调用方重定向）

只读 05data。
"""
import ast
import os
import subprocess
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
from export_utils import save_checkpoint                       # noqa: E402

SEED = 20041210
K_OUTER = 5
METHODS = ['PLSR', 'SVR', 'RF', '1D-CNN', 'FPCR', 'PFLR']
folds = list(KFold(K_OUTER, shuffle=True, random_state=SEED).split(np.arange(len(y))))
ncv = pd.read_csv('04outputs/25nested_cv.csv')
ncv = ncv[ncv.seed == SEED]
pred = pd.read_csv('04outputs/25nested_cv_pred.csv')
pred = pred[pred.seed == SEED]
GIT = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()[:12]


# ── 两个函数型模型：把拟合参数拆出来，使其可落盘、可回放 ───────────────────────
def fit_pflr(Ctr, ytr, J, R, lam):
    """与 16unified_comparison.pflr 同一套解，只是返回参数而非预测值。"""
    Z = Ctr @ J
    ym = float(ytr.mean())
    ZtZ = Z.T @ Z
    sc = np.trace(ZtZ) / max(np.trace(R), 1e-30)
    b = np.linalg.solve(ZtZ + lam * sc * R + 1e-12 * np.eye(len(R)), Z.T @ (ytr - ym))
    return dict(b=b, ym=ym)


def apply_pflr(par, Cte, J):
    return Cte @ J @ par['b'] + par['ym']


def fit_fpcr(Ctr, ytr, J, K):
    """与 16unified_comparison.fpcr 同一套解，只是返回参数而非预测值。"""
    M = np.linalg.cholesky(J + 1e-10 * np.eye(len(J)))
    A = Ctr @ M
    mu = A.mean(0)
    A0 = A - mu
    P = np.linalg.svd(A0, full_matrices=False)[2][:K].T
    T1 = np.c_[np.ones(len(A0)), A0 @ P]
    coef = np.linalg.lstsq(T1, ytr, rcond=None)[0]
    return dict(M=M, mu=mu, P=P, coef=coef)


def apply_fpcr(par, Cte):
    At = Cte @ par['M']
    return np.c_[np.ones(len(At)), (At - par['mu']) @ par['P']] @ par['coef']


def picked_of(m, k):
    """该方法在第 k 个外层折上内层交叉验证选中的配置。"""
    s = ncv[(ncv.method == m) & (ncv.fold == k)].picked.iloc[0]
    return ast.literal_eval(s) if m != 'PLSR' else int(s)


ARCH = {'PLSR': '偏最小二乘回归（潜变量数为超参）',
        'SVR': 'RBF 核支持向量回归（C 与 gamma 为超参）',
        'RF': '随机森林回归（树数与每次分裂的特征比例为超参）',
        '1D-CNN': '一维卷积网络：nl 个 Conv1d(stride 2)+BN+ReLU+MaxPool(2) 后接 AdaptiveAvgPool(8) 与两层全连接',
        'FPCR': '函数型主成分回归（傅里叶基 L=80，主成分数 K 为超参）',
        'PFLR': '惩罚函数型线性回归（傅里叶基 L=80，粗糙惩罚 λ_β 为超参）'}
INTENDED = ('用途：由近红外透射光谱（590~1 250 nm，2 048 点，SNV+SG 一阶导预处理）预测红富士苹果的'
            '可溶性固形物含量。边界：训练数据来自新疆阿克苏、山东烟台、陕西洛川三产地两年份共 1 300 个样品，'
            '不覆盖其他品种、产地、仪器与采集条件；模型不含域自适应，跨产地迁移的落差见论文 3.6 节。'
            '不得用于分级定价等对个体样品有实质后果的判定。')
LICENSE = ('数据为课题组自采，未公开发布，按需申请；模型权重随论文一并留存，不含任何个人信息，'
           '无伦理审查适用项（试材为水果，不涉人不涉动物）。')

print(f'把产出论文数字的模型落盘（seed={SEED}，K_outer={K_OUTER}，git {GIT}）\n')
summary = []
for m in METHODS:
    t0 = time.time()
    fitted, cfgs, replay = [], [], []
    for k, (itr, ite) in enumerate(folds):
        c = picked_of(m, k)
        cfgs.append(c)
        if m == 'PLSR':
            mo = PLSRegression(n_components=min(c, len(itr) - 1)).fit(Xp[itr], y[itr])
            p = mo.predict(Xp[ite]).ravel()
        elif m == 'SVR':
            mo = SVR(C=c[0], gamma=c[1]).fit(Xp[itr], y[itr])
            p = mo.predict(Xp[ite])
        elif m == 'RF':
            mo = RandomForestRegressor(n_estimators=c[0], max_features=c[1],
                                       random_state=SEED, n_jobs=-1).fit(Xp[itr], y[itr])
            p = mo.predict(Xp[ite])
        elif m == '1D-CNN':
            torch.manual_seed(SEED)                            # 与 cnn_fit 的种子设定一致
            mu, sd = Xp[itr].mean(0), Xp[itr].std(0) + 1e-8
            ym, ys = y[itr].mean(), y[itr].std()
            A = torch.tensor(((Xp[itr] - mu) / sd)[:, None, :], dtype=torch.float32)
            Bt = torch.tensor(((Xp[ite] - mu) / sd)[:, None, :], dtype=torch.float32)
            Y = torch.tensor(((y[itr] - ym) / ys)[:, None], dtype=torch.float32)
            net = CNN(c['w'], c['dp'], c['nl'])
            opt = torch.optim.Adam(net.parameters(), c['lr'], weight_decay=c['wd'])
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c['ep'])
            lf = nn.MSELoss()
            n = len(A)
            for _ in range(c['ep']):
                net.train()
                idx = torch.randperm(n)
                for i in range(0, n, c['bs']):
                    b = idx[i:i + c['bs']]
                    opt.zero_grad(); lf(net(A[b]), Y[b]).backward(); opt.step()
                sch.step()
            net.eval()
            with torch.no_grad():
                p = net(Bt).numpy().ravel() * ys + ym
            mo = dict(state_dict={kk: v.clone() for kk, v in net.state_dict().items()},
                      cfg=c, x_mu=mu, x_sd=sd, y_mu=float(ym), y_sd=float(ys))
        elif m == 'FPCR':
            L, K = c
            mo = fit_fpcr(COEF[L][itr], y[itr], BASIS[L][1], K)
            p = apply_fpcr(mo, COEF[L][ite])
        else:                                                  # PFLR
            L, lam = c
            mo = fit_pflr(COEF[L][itr], y[itr], BASIS[L][1], BASIS[L][2], lam)
            p = apply_pflr(mo, COEF[L][ite], BASIS[L][1])
        fitted.append(mo)
        replay.append((ite, p))

    # ── 回放核对：落盘对象重算的折外预测须与 25nested_cv_pred.csv 逐位一致 ──
    worst = 0.0
    for k, (ite, p) in enumerate(replay):
        ref = pred[(pred.method == m) & (pred.fold == k)].set_index('idx').loc[ite, 'pred'].to_numpy()
        worst = max(worst, float(np.max(np.abs(np.asarray(p) - ref))))
    if worst > 1e-9:
        raise AssertionError(f'{m} 的回放预测与 25nested_cv_pred.csv 不一致，最大绝对差 {worst:.3e}')

    allp = np.concatenate([p for _, p in replay])
    alli = np.concatenate([ite for ite, _ in replay])
    r2_pool = r2_score(y[alli], allp)
    a = ncv[ncv.method == m]
    run = f'25nested_cv_{m}'
    res = save_checkpoint(
        dict(folds=[dict(fold=k, config=cfgs[k], model=fitted[k],
                         train_idx=folds[k][0], test_idx=folds[k][1]) for k in range(K_OUTER)],
             seed=SEED, method=m),
        run,
        dict(arch=ARCH[m],
             params=('每折一个模型，共 5 个；逐折选中的配置见 training_config' if m != '1D-CNN'
                     else '每折一个模型，共 5 个；3 折宽度 32（1 135 937 参数）、2 折宽度 16（350 433 参数）'),
             flops='未测（本项目模型规模下非瓶颈，算力记录见 05logs/25nested_cv.log 的逐方法用时）',
             ckpt_sha='<由 save_checkpoint 回填>',
             training_config=f'嵌套 CV K_outer=5 / K_inner=5，seed={SEED}；'
                             f'预处理 SNV+SG 一阶导（窗宽 15，二次）；候选网格见 02code/grids.py（第二轮）；'
                             f'逐折选中：{cfgs}',
             training_hardware='cpu（8 核，无 CUDA）',
             deterministic_guarantee='numpy/sklearn 固定 random_state；torch 每次训练前 manual_seed；'
                                     '未开启 torch 确定性算子（CPU 通道无 cudnn）',
             stage='pilot',
             eval_summary=f'5 折均值 R²={a.R2.mean():.3f}±{a.R2.std(ddof=0):.3f}，'
                          f'RMSE={a.RMSE.mean():.3f}±{a.RMSE.std(ddof=0):.3f} °Brix，'
                          f'RPD={a.RPD.mean():.2f}；拼合 5 折折外预测的整体 R²={r2_pool:.3f}',
             intended_use_and_limits=INTENDED,
             license_ethics=LICENSE,
             traceability=f'git {GIT}；产出脚本 02code/25nested_cv.py 与本脚本 02code/30save_final_models.py；'
                          f'数字见 04outputs/25nested_cv.csv 与 25nested_cv_pred.csv；'
                          f'日志 05logs/25nested_cv.log；stage 标注的理由见 '
                          f'refine-logs/01PROJECT_PROTOCOL.md「模型卡 stage 的具名声明」'),
        tag='best')
    print(f'  {m:<8} 回放最大绝对差 {worst:.2e}  用时 {time.time()-t0:>6.0f}s  → {res.get("ckpt", run)}')
    summary.append(dict(method=m, replay_max_abs_diff=worst, r2_pooled=r2_pool,
                        ckpt=res.get('ckpt', ''), sha256=res.get('sha256', '')))

pd.DataFrame(summary).to_csv('04outputs/30save_final_models.csv', index=False, encoding='utf-8-sig')
print('\n六个方法的模型与模型卡已落 07checkpoints/，清单见 04outputs/30save_final_models.csv')
