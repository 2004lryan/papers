# -*- coding: utf-8 -*-
"""
48rebuttal_R1_13_bspline_beta.py: 主分析基（三次 B 样条 L=80）下 FPLR 的回归系数函数、其在 25 个外层训练折上的
变动范围，以及 FPLR 与 FPCR 部署模型的落盘（导师批注 6，预注册 B10.1；项目规范 第零章 F1、7.1）

为什么要有这个脚本：主分析基由傅里叶改为 B 样条之后，系数函数必须由同一基、同一网格、同一选参准则重新得到。
原稿的 β(λ) 取自 12figures.py，是以 4 折交叉验证在 13 点网格上选 λ 的结果，与表 1 的协议不一致。
另外，单条 β 曲线说明不了它的形态有多稳定：本脚本取 45 脚本在 5 折 × 5 次重复嵌套交叉验证中每个外层训练折
选中的 λβ，在该训练折上重拟合并求 β，给出 25 条曲线的逐点 2.5%~97.5% 分位带。25 个外层训练折两两共享
一半以上的样品，该带表示「换一批训练样品时 β 的变动」，不是置信区间，正文按此表述。

部署模型（7.1）：在全部样品（协议 B10.10 数据清洗后）上以 KFold(5, shuffle=True, random_state=20041210) 从 grids.py 的 12 组候选中
选参（准则与表 1 相同：5 折 RMSE 的均值最小，并列取网格中靠前者），用全部样品重拟合。部署模型的分数不替代表 1。
回放核对：25 个外层训练折重拟合得到的测试折预测，须与 45 脚本 _pred.csv 逐位一致（最大绝对差 <1e-9），否则停下。

极值窗口的判读规则沿用 12figures.py（原图所用规则，未作调整）：取部署 β 绝对值最大的 400 个波长点，
按相邻间隔 >15 nm 断开，保留点数 >15 的窗口。另报告 25 条曲线同号（分位带不跨 0）的连续波长区间。

运行方式:
    cd <项目根目录>
    python3 02code/48rebuttal_R1_13_bspline_beta.py               # 全部重算、落盘并画图
    python3 02code/48rebuttal_R1_13_bspline_beta.py --plot-only   # 只由已落盘的 48rebuttal_beta*.csv 重画图 A2，
                                                                  # 不读 05data、不重算、不改 CSV 与模型文件
    FIG_LANG=en python3 02code/48rebuttal_R1_13_bspline_beta.py --plot-only   # 英文稿用的英文版，文件名加 _en

输出文件:
    04outputs/48rebuttal_beta.csv            — 逐波长的部署 β 与 25 条外层训练折 β 的 2.5%/50%/97.5% 分位
    04outputs/48rebuttal_beta_windows.csv    — 极值窗口与同号区间
    04outputs/48rebuttal_deploy.csv          — 部署模型选参时 12 组候选的 5 折 RMSE 均值与选中配置
    04outputs/figs/Figure_5_beta_bspline.{pdf,svg,png} — 附录图 A2（FIG_LANG=en 时为 Figure_5_beta_bspline_en.*）
    07checkpoints/48rebuttal_{FPLR,FPCR}_bspline_deploy/best.joblib 与 model_card.md
    05logs/48rebuttal_R1_13_bspline_beta_YY-MM-DD_HHMMSS.log — 运行日志（由调用方重定向）

只读 05data。
"""
import ast
import os
import subprocess
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

LANG = os.environ.get('FIG_LANG', 'zh')
if LANG not in ('zh', 'en'):
    raise SystemExit(f'FIG_LANG 只能取 zh 或 en，收到 {LANG}')

def draw_beta_figure(wl, beta_dep, q025, q975, wdf):
    """附录图 A2：部署 β 与 25 条外层训练折 β 的 2.5%~97.5% 分位带，标出点数最多的两个窗口。
    数字写法与正文一致：四位数不分节；中文与数字、字母之间不留空格，数值与单位之间留一格。"""
    sys.path.insert(0, '02code')
    import matplotlib.pyplot as plt
    from figstyle import W2, C, zero_line, save
    # 英文图例须短于约 40 个字符，否则在左上角会压到 868~955 nm 的窗口标注
    lab = {'zh': dict(band='25个外层训练折的2.5%~97.5%分位', dep='全样本部署模型', sep='~', x='波长/nm', y='回归系数β(λ)/10'),
           'en': dict(band='2.5%–97.5% quantiles, 25 fold refits', dep='Deployed model (all samples)', sep='–',
                      x='Wavelength (nm)', y='Coefficient β(λ)/10')}[LANG]
    scale = 10 ** int(np.floor(np.log10(np.max(np.abs(np.r_[q025, q975])))))
    fig, ax = plt.subplots(figsize=(W2, 1.9), layout='constrained')
    ax.fill_between(wl, q025 / scale, q975 / scale, color=C['sky'], alpha=0.35, lw=0, label=lab['band'])
    ax.plot(wl, beta_dep / scale, color=C['blue'], lw=1.0, zorder=3, label=lab['dep'])
    zero_line(ax)
    ytop = np.max(q975 / scale)
    ax.set_ylim(np.min(q025 / scale) * 1.15, ytop * 1.45)
    # 图中只标跨度最大的两个窗口（各含 90 个以上波长点）；其余三个窗口宽不足 10 nm，数值见 48rebuttal_beta_windows.csv
    ylo, yhi = ax.get_ylim()
    band_top = (ytop * 1.18 - ylo) / (yhi - ylo) - 0.02                  # 底纹止于窗口标注下方，标注不压底纹边缘
    for _, w in wdf[(wdf.kind == '绝对值前400点') & (wdf.n_points >= 90)].iterrows():
        ax.axvspan(w.lo, w.hi, ymin=0, ymax=band_top, color=C['blue'], alpha=0.10, lw=0, zorder=0)
        ax.text((w.lo + w.hi) / 2, ytop * 1.18, f'{w.lo:.0f}{lab["sep"]}{w.hi:.0f} nm', ha='center', va='bottom',
                fontsize=6.5, color=C['blue'])
    ax.set_xlim(580, 1260)
    ax.set_xticks([600, 700, 800, 900, 1000, 1100, 1200])
    ax.set_xticklabels(['600', '700', '800', '900', '1000', '1100', '1200'])
    ax.set_xlabel(lab['x'])
    sup = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
    ax.set_ylabel(lab['y'] + str(int(np.log10(scale))).translate(sup))   # 用上标字符而非 mathtext，中文才不丢字
    ax.legend(loc='upper left', fontsize=6.5, ncol=1)
    save(fig, 'Figure_5_beta_bspline' + ('' if LANG == 'zh' else '_en'), multi=False)


if '--plot-only' in sys.argv:
    _b = pd.read_csv('04outputs/48rebuttal_beta.csv')
    draw_beta_figure(_b.wavelength_nm.to_numpy(), _b.beta_deploy.to_numpy(), _b.beta_q025.to_numpy(),
                     _b.beta_q975.to_numpy(), pd.read_csv('04outputs/48rebuttal_beta_windows.csv'))
    print('→ 04outputs/figs/Figure_5_beta_bspline.*（只画图：读 48rebuttal_beta.csv 与 48rebuttal_beta_windows.csv）')
    sys.exit(0)

with open('02code/16unified_comparison.py') as _src:
    _SHARED = _src.read().split("METHODS=['PLSR'")[0]
exec(_SHARED)
warnings.simplefilter('default', RuntimeWarning)
sys.path.insert(0, '02code')
from grids import GRIDS  # noqa: E402
from export_utils import save_checkpoint  # noqa: E402

SEED, L_BASIS = 20041210, 80
SEEDS = [20041210, 20240404, 20060515, 20251203, 19810915]
B80 = bspline_basis(t, L_BASIS)
J80, R80 = gram_and_pen(B80, t)
C80 = to_coef(Xp, B80, R80)
GIT = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()[:12]


def fit_fplr(Ctr, ytr, lam):
    """与 16 脚本的 pflr 同一套解，返回基系数 b 与训练均值，供落盘与回放。"""
    Z = Ctr @ J80
    ym = float(ytr.mean())
    ZtZ = Z.T @ Z
    sc = np.trace(ZtZ) / max(np.trace(R80), 1e-30)
    b = np.linalg.solve(ZtZ + lam * sc * R80 + 1e-12 * np.eye(len(R80)), Z.T @ (ytr - ym))
    return dict(b=b, ym=ym)


def apply_fplr(par, Cte):
    return Cte @ J80 @ par['b'] + par['ym']


def fit_fpcr(Ctr, ytr, K):
    """与 16 脚本的 fpcr 同一套解，返回参数。"""
    M = np.linalg.cholesky(J80 + 1e-10 * np.eye(len(J80)))
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


# ---------- 1. 回放核对 + 25 个外层训练折的 β ----------
ncv = pd.read_csv('04outputs/45rebuttal_ncv_bspline_FPLR.csv', dtype={'picked': str, 'fingerprint': str})
prd = pd.read_csv('04outputs/45rebuttal_ncv_bspline_FPLR_pred.csv', dtype={'fingerprint': str})
if len(ncv) != 25 or ncv.fingerprint.nunique() != 1:
    raise SystemExit('45 脚本的 B 样条 FPLR 结果不完整或来源不唯一')
betas, worst = [], 0.0
for sd in SEEDS:
    for f, (itr, ite) in enumerate(KFold(5, shuffle=True, random_state=sd).split(np.arange(len(y)))):
        row = ncv[(ncv.seed == sd) & (ncv.fold == f)]
        L, lam = ast.literal_eval(row.picked.iloc[0])      # 形如 "(80, 1e-06)"，来自 grids.py 的候选
        if L != L_BASIS or (L, lam) not in GRIDS['PFLR']:
            raise SystemExit(f'选中配置不在网格内：{row.picked.iloc[0]}')
        par = fit_fplr(C80[itr], y[itr], lam)
        p = apply_fplr(par, C80[ite])
        ref = prd[(prd.seed == sd) & (prd.fold == f)].set_index('idx').loc[ite, 'pred'].to_numpy()
        worst = max(worst, float(np.max(np.abs(p - ref))))
        betas.append(B80 @ par['b'])
if worst > 1e-9:
    raise SystemExit(f'回放核对未通过：25 个外层折的重拟合预测与 45 脚本产物最大绝对差 {worst:.3e}')
betas_arr = np.array(betas)
print(f'回放核对通过：25 个外层训练折重拟合的测试折预测与 45 脚本逐位一致（最大绝对差 {worst:.1e}）')

# ---------- 2. 部署模型：全样本 5 折选参 + 全样本重拟合 ----------
folds = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(y))))
dep_rows, deploy = [], {}
for m, fit, app in [('PFLR', fit_fplr, apply_fplr), ('FPCR', fit_fpcr, apply_fpcr)]:
    best, bc = np.inf, GRIDS[m][0]
    for c in GRIDS[m]:
        e = float(np.mean([np.sqrt(mean_squared_error(y[iv], app(fit(C80[ia], y[ia], c[1]), C80[iv])))
                           for ia, iv in folds]))
        dep_rows.append(dict(method='FPLR' if m == 'PFLR' else m, candidate=str(c), cv_rmse=e))
        if e < best:
            best, bc = e, c
    par = fit(C80, y, bc[1])
    deploy[m] = (bc, best, par, float(r2_score(y, app(par, C80))))
    for r in dep_rows:
        if r['method'] == ('FPLR' if m == 'PFLR' else m):
            r['picked'] = str(bc)
    print(f"部署模型 {'FPLR' if m == 'PFLR' else m}：选中 {bc}，5 折 RMSE 均值 {best:.4f} °Brix，"
          f'全样本拟合 R²={deploy[m][3]:.4f}（训练集内，不作精度报告）')
pd.DataFrame(dep_rows).to_csv('04outputs/48rebuttal_deploy.csv', index=False, encoding='utf-8-sig')

beta_dep = B80 @ deploy['PFLR'][2]['b']
q025, q50, q975 = np.quantile(betas_arr, [0.025, 0.5, 0.975], axis=0)
stable = (q025 > 0) | (q975 < 0)
pd.DataFrame(dict(wavelength_nm=wl, beta_deploy=beta_dep, beta_q025=q025, beta_q50=q50, beta_q975=q975,
                  sign_stable=stable)).to_csv('04outputs/48rebuttal_beta.csv', index=False, encoding='utf-8-sig')

# ---------- 3. 窗口判读 ----------
win = []
top = np.sort(wl[np.argsort(-np.abs(beta_dep))[:400]])
for s_ in np.split(top, np.where(np.diff(top) > 15)[0] + 1):
    if len(s_) > 15:
        msk = (wl >= s_.min()) & (wl <= s_.max())
        pk = wl[msk][np.argmax(np.abs(beta_dep[msk]))]
        win.append(dict(kind='绝对值前400点', lo=float(s_.min()), hi=float(s_.max()), n_points=len(s_),
                        peak_wl=float(pk), sign='+' if beta_dep[wl == pk][0] > 0 else '−',
                        frac_sign_stable=float(stable[msk].mean())))
edges = np.flatnonzero(np.diff(np.r_[0, stable.astype(int), 0]))
for a0, a1 in zip(edges[::2], edges[1::2], strict=True):
    if a1 - a0 > 15:
        win.append(dict(kind='25条同号区间', lo=float(wl[a0]), hi=float(wl[a1 - 1]), n_points=int(a1 - a0),
                        peak_wl=float(wl[a0:a1][np.argmax(np.abs(beta_dep[a0:a1]))]),
                        sign='+' if q50[a0:a1].mean() > 0 else '−', frac_sign_stable=1.0))
wdf = pd.DataFrame(win)
wdf.to_csv('04outputs/48rebuttal_beta_windows.csv', index=False, encoding='utf-8-sig')
print('\n【极值窗口与同号区间】')
print(wdf.round(3).to_string(index=False))
print(f'\n25 条曲线同号的波长点占比 {stable.mean():.1%}；部署 β 与 25 条曲线中位数的相关系数 '
      f'{np.corrcoef(beta_dep, q50)[0, 1]:.4f}')

# ---------- 4. 附录图 A2 ----------
draw_beta_figure(wl, beta_dep, q025, q975, wdf)

# ---------- 5. 部署模型落盘（F1）----------
INTENDED = ('用途：由近红外透射光谱（590~1 250 nm，2 048 点，SNV+SG 一阶导预处理）预测红富士苹果的可溶性固形物含量。'
            f'边界：训练数据来自新疆阿克苏、山东烟台、陕西洛川三产地两年份共 {len(y)} 个样品（协议 B10.10 数据清洗后），'
            '不覆盖其他品种、产地、仪器与'
            '采集条件；换采收年份或产地时须按论文 3.3 节用新批次样品重新校正，未经校正直接使用的跨批次 R² 可为负。'
            '不得用于分级定价等对个体样品有实质后果的判定。')
LICENSE = '数据为课题组自采，未公开发布，按需申请；模型不含个人信息；试材为水果，无伦理审查适用项。'
for m in ('PFLR', 'FPCR'):
    bc, best, par, r2fit = deploy[m]
    lab = 'FPLR' if m == 'PFLR' else m
    res = save_checkpoint(
        dict(method=lab, basis='bspline', L=L_BASIS, config=bc, params=par, wavelength_nm=wl,
             basis_matrix=B80, gram_J=J80, penalty_R=R80, coef_lam_s=1e-6,
             preprocessing='SNV + Savitzky-Golay 一阶导（窗宽 15，二次多项式）', seed=SEED),
        f'48rebuttal_{lab}_bspline_deploy',
        dict(arch=('惩罚函数型线性回归：三次 B 样条基 L=80，二阶导粗糙惩罚 λβ·s·R，s=tr(Z′Z)/tr(R)' if m == 'PFLR'
                   else '函数型主成分回归：三次 B 样条基 L=80，Gram 矩阵 Cholesky 变换后 SVD 取前 K 个主成分'),
             params=f'基系数 {L_BASIS} 个' + ('' if m == 'PFLR' else f'，主成分 {bc[1]} 个及其回归系数'),
             flops='未测（线性模型，单样品预测为一次 2 048×80 与 80×80 矩阵乘）',
             ckpt_sha='<由 save_checkpoint 回填>',
             training_config=(f'全部 {len(y)} 个样品（协议 B10.10 数据清洗后）；KFold(5, shuffle=True, random_state={SEED}) 从 grids.py 的 12 组候选中'
                              f'选参（5 折 RMSE 均值最小），选中 {bc}，5 折 RMSE 均值 {best:.4f} °Brix；全样本重拟合'),
             training_hardware='cpu（Apple M1，8 核，无 CUDA）',
             deterministic_guarantee=(f'闭式解（线性方程组与奇异值分解），无随机性；多线程 BLAS 下末位浮点可能不同，'
                                      f'文件哈希因此不保证逐次相同（回放核对的最大绝对差 {worst:.1e}）'),
             stage='pilot',
             eval_summary=(f'部署模型不另报精度；其方法的精度以 5 折 × 5 次重复嵌套交叉验证为准（04outputs/47rebuttal_table1.csv）。'
                           f'全样本拟合 R²={r2fit:.4f} 为训练集内数值，仅作记录'),
             intended_use_and_limits=INTENDED,
             license_ethics=LICENSE,
             traceability=(f'git {GIT}；产出脚本 02code/48rebuttal_R1_13_bspline_beta.py；选参记录 04outputs/48rebuttal_deploy.csv；'
                           f'系数函数 04outputs/48rebuttal_beta.csv；stage 标注理由见 refine-logs/01PROJECT_PROTOCOL.md'
                           f'「模型卡 stage 的具名声明」')),
        tag='best')
    print(f'  {lab} 部署模型已落盘 → {res.get("ckpt", "")}  sha256 {res.get("sha256", "")[:16]}')
print('\n→ 04outputs/48rebuttal_beta.csv / 48rebuttal_beta_windows.csv / 48rebuttal_deploy.csv / figs/Figure_5_beta_bspline.*')
