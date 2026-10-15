"""91_mango_half_unit.py: 芒果干物质任务上半数暴露对照 Δ_half 的划分单位——按行、按取样点、按采收批（补充材料 S9.9）。

为什么:
    74 号的 S_half 臂把留出季的样本按行对半分成 A、B：A 半带标签进训练池，只在 B 半上评估，用
    Δ_half = S_half − R^h 量「见过留出域、但没见过被评估样本」时的抬高。公开的芒果表里没有果实编号，
    元数据与干物质值完全相同的相邻两行有 5641 对（见下面「取样点与采收批」），按行对半时同一取样点的
    两条谱可能一条进 A、一条进 B；同一采收批的样本也会分到两半（各格的实际占比记在结果表里）。芒果
    4 个留出域上的 Δ_half 因此可能含有「见过被评估样本的另一条谱」或「见过被评估样本所在采收批」的成分。
    本脚本在主口径的芒果任务上把对半分的单位换成取样点与采收批各做一次，与按行的现有划分并排，
    看 Δ_half 是否随划分单位变化。

任务（与 74 号主口径的芒果任务相同，数据、类别、折内数组都经现有函数得到）:
    数据    74 号 DATASETS["mango_season"] 登记的加载函数 cdl.load_mango(data_root, 种子)：每季按种子抽至多
            500 条（四季都够 500 条），光谱取表头里列名为纯数字的全部列，目标为干物质
    类别    干物质按源季三分位分 3 类（cdl.prepare_fold 里的 tertile_labels，阈值只用源季）
    域      采收季 S1–S4；每次留出一季、其余三季为源域，共 4 个留出域；折内数组取 74 号 fold_arrays（逐谱 SNV）
    编码器  cnn / mlp / resnet / transformer 按原型网络元训练，pls / pca 在训练池上直接拟合；训练一律调
            74 号 obtain(编码器, x, y, in_dim, args, device, 种子)
    预算    K = 5；元训练 1000 情节、每格 50 个评估情节（pilot 时 200 情节、10 个评估情节）；权重不落盘

臂（每个（留出季, 编码器, 种子）里 R 只训一次，S_half 在三种划分单位下各训一次）:
    R       折限定  : 训练池 = 源域
    S_half  半数暴露 : 训练池 = 源域 ∪ 留出季的 A 半（带标签），只在 B 半上评估
    R_B     即 R^h  : R 的编码器在同一个 B 半、同一批评估情节上的评估
    对照    Δ_half = S_half − R_B。三种单位的 B 半是不同的样本集，各自配各自的 R_B

划分单位（都只把留出季分成 A、B，源域不动）:
    row   按行：直接调 74 号 half_split(fa, dom_of_row, 留出季, 种子)，与 74 号在同一种子下实际用的划分逐行
          相同（各类的行按 default_rng([种子, 74, 11]) 对半，奇数时 B 多一个）
    pair  按取样点：各类的取样点按 default_rng([种子, 91, 1, 采收季编号]) 对半（奇数时 B 多一个点），
          同一取样点的各条谱同进 A 或同进 B
    pop   按采收批：同一采收批（Pop 列）的全部谱同进 A 或同进 B，三种里最严。一个采收批里三类样本都可能
          有，各类的划分互相牵连，所以整体选一组采收批进 A 半，使 A 半各类的样本数尽量接近该类的一半（向下取整）：
          按 default_rng([种子, 91, 2, 采收季编号]) 给出的随机次序逐个考虑采收批，放进 A 半能让「A 半各类
          样本数与目标之差的平方和」变小就放，否则留在 B 半；这样做 200 遍，先取两半里不足 K+2 条的类
          最少的，再取各类偏差绝对值之和最小的（并列取最先出现的）。各类未必正好对半，实际的 A、B 样本数
          与各类样本数逐格写进结果表
    pair、pop 的训练池照 half_split 的后半段构造：xall 去掉留出季的 B 半，行序不变（pool_without_b）；
    按行划分时用它复算一遍，须与 half_split 返回的训练池逐元素相同，否则中止。

取样点与采收批（由表内数值归纳，数据说明里没有这一项）:
    取样点  Season、Region、Date、Type、Cultivar、Pop、Temp、DM 这 8 列完全相同的行算同一个取样点（定义同
            90 号的 POINT_KEYS）。全表 11691 行按此归成 6050 组：两行一组的 5641 组，都是表里相邻的两行；
            一行一组的 409 组；没有多于两行的组。只按干物质值看相邻行，得到的也是这 5641 对。抽看过的
            几对里，两条谱只在第三位有效数字上有出入，相邻的不同组之间差别明显更大。数据说明把一行记作
            一个果，表里没有果实编号；「同一取样点的两次扫描」是由这些数值归纳出来的，不是数据说明给出的字段
    采收批  Pop 列，112 个取值；四季各 25、19、50、18 个，没有一个采收批跨两季或跨两个产区
    行归属  cdl.load_mango 只返回光谱与干物质。本脚本另读一遍同一个文件，先在全表上给取样点编号，再照
            load_mango 的步骤抽样，取样点编号、采收批与原表行号跟着抽出来的行走；抽出的光谱与干物质须与
            load_mango 返回的数组逐元素相同，折内再按 prepare_fold 的类别筛选对到评估数组的行上并核对，
            任何一步对不上即中止（load_task、held_meta）
    每季只抽 500 条，同一取样点的两条谱都被抽中的只是一部分；各留出季里这样的样本数记在结果表的
    「留出季成对样本数」一列。

跳过的格（不降级评估；逐条记在结果表的「跳过的格」表里，不训练、不改用别的类别集合或别的划分补上）:
    留出季合法类不足 3 个（74 号 fold_arrays 返回空）时，该留出季三种单位都跳过。
    B 半里有类不足 K+2 条时该单位跳过：K 条作支持、至少再有 2 条作查询，是 cdl.eval_kshot 的门槛，也是
    74 号评 B 半三臂的条件（任务只有 3 类，74 号的「达标的类不少于 3 个」与「每个类都达标」是同一件事）。
    pair、pop 两种单位另要求 A 半里每类也不少于 K+2 条，保证每个类都有带标签的留出季样本进训练池；
    row 单位照 74 号只看 B 半。

逐格另记的划分情况:
    A、B 两半的样本数、各类样本数、采收批数；B 半样本里「同一取样点的另一条谱在 A 半」的占比与
    「同一采收批有样本在 A 半」的占比（pair 单位前者须为 0，pop 单位两者都须为 0，否则中止）；
    A 半各行原表行号的指纹（同一（留出季, 种子, 单位）在各编码器族上应相同，汇总时核对）。
    B 半的查询样本按「同点的另一条谱是否在 A 半」分两层，各臂在两层上的准确率分开记（全部评估情节的
    查询人次合计）。按行划分时，S_half − R_B 在两层之间的差是同一次划分内「见过另一条谱」多出来的部分，
    不受划分之间波动的影响；另两种单位下前一层是空的。

限定:
    三种单位的 B 半是不同的样本集，单位之间的差（pair − row、pop − row）里含划分本身的波动；在同一留出季、
    编码器、种子内配对只消掉训练与数据抽样的部分。
    按取样点、按采收批划分后，同一取样点的两条谱同在 B 半，评估情节里可能一条作支持、一条作查询；
    按行划分时也有这种情形，只是少一些。这一点 S_half 与 R_B 相同（同一批情节），绝对水平在单位之间不可直接比。
    分层准确率的「其余」一层里也有这样的样本，两臂同受影响，在臂间的差里大体抵消。
    pop 单位的 A、B 两半只是尽量接近对半，训练池大小与 row、pair 略有出入（见各格样本数）。
    row 单位与 74 号的正式结果走同一代码路径；换了机器或显卡，深度族的数值未必逐位相同（对账见汇总表）。

随机数:
    新增的抽样只有两处：按取样点对半 default_rng([种子, 91, 1, 采收季编号])、按采收批划分
    default_rng([种子, 91, 2, 采收季编号])。调用现有函数的地方保持它们原有的播种：每季抽样
    sub.sample(random_state=种子)（cdl.load_mango）、按行对半 [种子, 74, 11]（74 号）、元训练
    eu.set_random_seed(种子)（每次训练进入时重新播种）、评估情节 np.random.seed(种子)（crossdataset_lib）。

复用:
    export_utils      路径常量、get_device、set_random_seed、enforce_publication_grade_device、get_logger、
                      log_experiment_header、init_wandb、track_compute、compute_sha256、table_sheet_name
    74 号脚本         DATASETS（芒果的加载函数）、fold_arrays、half_split、obtain、kshot_episodes、_t_interval、
                      load_long（对账时读 74 号的正式逐种子表）；导入 74 号会连带导入 71、20 号
    90 号脚本         数据文件的相对路径与 SHA-256 登记值、META_STR_COLS、POINT_KEYS、STAGE_BUDGET
    crossdataset_lib  load_mango（经 74 号 DATASETS）、tertile_labels、_apply_preprocess、eval_kshot；
                      prepare_fold、meta_train 分别经 74 号 fold_arrays、obtain 调用
    encoders          ENCODERS、ARCH_DESC；build_encoder、fit_linear_encoder 经 74 号 obtain 调用

输入:
    <data_root>/041_mango_anderson_crossseason/NAnderson2020MendeleyMangoNIRData.csv（读入时核对 SHA-256）
    04outputs/formal_seeds/74_multi_encoder__<编码器>__seed<S>.xlsx（只读；--aggregate 对账用，没有就不对账）

运行方式:
    cd <项目根>
    # 预跑（种子 2004；元训练 200 情节、每格 10 个评估情节，结果文件名带 __pilot）
    python 02code/91_mango_half_unit.py --device cuda --stage pilot --seed 2004 --data_root <公开数据根目录>
    # Formal（CUDA；每次一个种子；--encoders 缺省为全部六族，也可以只给一族以便并行）
    python 02code/91_mango_half_unit.py --device cuda --stage formal --seed 20060515 \
        --encoders transformer --data_root <公开数据根目录>
    # 汇总（缺省汇总正式结果；加 --stage pilot 则汇总预跑结果，另存一份）
    python 02code/91_mango_half_unit.py --aggregate

输出文件:
    04outputs/formal_seeds/91_mango_half_unit__<编码器>__seed<S>[__pilot].xlsx
        表a 逐格结果（一行 = 留出季 × 半分单位 × 臂）、表b 逐情节、跳过的格、compute。给了 --tag 时 <编码器>
        之后多一段 __<tag>；正式阶段改了 --k、--episodes、--reps、--lin_dim 必须给 --tag，
        不带标签的文件只留给缺省设定
    04outputs/91_mango_half_unit.xlsx（--aggregate；有标签时加 __<tag>，--stage pilot 时再加 __pilot）
        表a 逐留出季对照、表b 各季合并、表c 单位间配对差、表d 与74号对账、表e 对账逐种子、
        表f 按行划分的分层准确率、表g 长格式、跳过的格、表注、compute；表a、b、c、f 的均值与 t 区间
        上下限按全精度写出，取用方显示时只舍入一次
    05logs/91_mango_half_unit_YY-MM-DD_HHMMSS.log
    不写入、不覆盖 74、90 号等现有脚本的任何结果文件。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import re
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import crossdataset_lib as cdl
import encoders as enc_lib
import export_utils as eu

NDArray = npt.NDArray[Any]
Key = tuple[str, str, str, int]  # 汇总时一格的键：(留出季, 编码器, 半分单位, 种子)
m74: Any = importlib.import_module("74_multi_encoder_leakage")
m90: Any = importlib.import_module("90_mango_region_task")

STEM = "91_mango_half_unit"
DATASET = "mango_season"  # 74 号 DATASETS 里芒果任务的登记名
MANGO_CSV: str = m90.MANGO_CSV
MANGO_SHA256: str = m90.MANGO_SHA256  # 数据卡登记值
META_STR_COLS: tuple[str, ...] = tuple(m90.META_STR_COLS)  # 按字符串读入的元数据列
POINT_KEYS: list[str] = list(m90.POINT_KEYS)  # 这 8 列全同的行算同一取样点
POINT_SIZES = (5641, 409, 0)  # 全表里两行一组、一行一组、多于两行的取样点个数（文件头里的归纳值）
LOAD_CAP, LOAD_MIN_ROWS = 500, 60  # 同 cdl.load_mango：每季抽至多 500 条；不足 60 行的季不作为域
UNITS: dict[str, str] = {"row": "按行（74 号现有划分）", "pair": "按取样点", "pop": "按采收批"}
STREAMS: dict[str, int] = {"pair": 1, "pop": 2}  # 随机流 default_rng([种子, 91, 流编号, 采收季编号])
ARM_HALF, ARM_REF = "S_half", "R_B"  # 74 号的臂名；R_B 即 R^h
MIN_EXTRA = 2  # 每类除 K 条支持外至少还要 2 条：cdl.eval_kshot 与 74 号 B 半三臂的门槛都是 K+2
POP_TRIES = 200  # 按采收批划分时试的随机次序数
TOL_74 = 0.02  # 与 74 号正式结果对账的容差
STAGE_BUDGET: dict[str, dict[str, int]] = {name: dict(val) for name, val in m90.STAGE_BUDGET.items()}
DEFAULT_K, DEFAULT_LIN_DIM = 5, 16
SEED_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
SHEET_CELL = eu.table_sheet_name(0, "逐格结果")
SHEET_EPIS = eu.table_sheet_name(1, "逐情节")
SHEET_SKIP = "跳过的格"
SKIP_COLS = ["留出季", "编码器", "种子", "半分单位", "原因", "A各类样本数", "B各类样本数"]
# 逐种子结果表里记录本次设定的列；汇总时这几列必须在全部文件里一致
SETTING_COLS = ["K", "元训练情节", "评估情节设定", "成分数上限"]
STRATA = ("同点谱在A半", "其余")  # B 半查询样本的两层
SPLIT_MEAN_COLS = ("A样本数", "B样本数", "B半同点谱在A半占比", "B半同批有样本在A半占比")
NOTES: tuple[tuple[str, str], ...] = (
    (
        "全部",
        "主口径的芒果干物质任务（按采收季留出，4 个留出域）。S_half 的留出季对半分按三种单位各做一次："
        "row = 按行（74 号现有划分）、pair = 按取样点、pop = 按采收批。Δ_half = S_half − R_B（R_B 即 R^h："
        "R 的编码器在同一个 B 半、同一批评估情节上的评估）。三种单位的 B 半是不同的样本集。",
    ),
    ("表a", "各臂一栏是宏 F1 的种子均值；种子 t 区间是逐种子配对差的 95% t 区间；样本数与占比是种子均值。"),
    (
        "表b",
        "各季合并：每个种子先对各留出季取均值，再对种子求均值与 t 区间（只用各留出季都有结果的种子）；"
        "「各族合并」一行再对编码器取均值。为正留出季 = 种子均值为正的留出季数 / 有结果的留出季数。",
    ),
    (
        "表c",
        "Δ_half之差 = 该单位的 Δ_half − row 单位的 Δ_half，同一留出季、编码器、种子内配对。"
        "两种单位的 B 半不是同一批样本，差里含划分本身的波动。",
    ),
    (
        "表d、表e",
        "row 单位与 74 号正式逐种子表里同一（留出季, 编码器, 种子）的 S_half、R_B 并排；两者同一代码路径，"
        f"同机同环境下应逐位相同。超过容差 = Δ_half 相差超过 {TOL_74}。R全部样本 = R 臂在留出季全部样本上的宏 F1。",
    ),
    (
        "表f",
        "只用 row 单位。B 半的查询样本按「同点的另一条谱是否在 A 半」分两层，准确率按全部评估情节的查询人次合计。"
        "差的差 =（S_half − R_B，同点谱在 A 半的一层）−（S_half − R_B，其余一层），"
        "是同一次划分内「见过另一条谱」多出来的准确率。",
    ),
)


# ═══════════════════════════════════════════════════════════════════════════════
# 一、数据：主口径的芒果任务，加上各行的取样点与采收批
# ═══════════════════════════════════════════════════════════════════════════════
class RowMeta(NamedTuple):
    """与光谱数组逐行对应的归属。"""

    rows: NDArray  # 原表行号（0 起，不含表头）
    points: NDArray  # 取样点编号（在全表上编号）
    pops: NDArray  # 采收批（Pop 列，字符串）


class MangoTask(NamedTuple):
    """主口径的芒果任务数据与各域各行的归属。"""

    doms: dict[str, tuple[NDArray, NDArray]]  # 域名 → (未做预处理的光谱, 干物质)，即 cdl.load_mango 的返回值
    meta: dict[str, RowMeta]  # 域名 → 各行的归属
    season: dict[str, int]  # 域名 → 采收季编号


def mate_rows(points: NDArray) -> NDArray:
    """各行所在的取样点在这一组样本里是否还有别的行。"""
    _, inv, cnt = np.unique(points, return_inverse=True, return_counts=True)
    out: NDArray = cnt[inv] > 1
    return out


def load_task(root: str, seed: int, logger: eu.Logger) -> MangoTask:
    """读主口径的芒果任务，并给抽出来的每一行配上原表行号、取样点编号与采收批。

    光谱与干物质取 74 号 DATASETS 登记的加载函数（cdl.load_mango）的返回值，任务数据与 74 号相同。
    该函数不返回元数据，这里另读一遍同一个文件：先在全表上按 POINT_KEYS 给取样点编号，再照它的步骤
    （按季筛选、不足 60 行的季略去、sub.sample(min(500, 行数), random_state=种子)）抽样，取样点编号、
    采收批与原表行号随抽出来的行带出；抽出的光谱与干物质须与加载函数返回的数组逐元素相同，否则中止。
    元数据列按字符串读入（同 90 号；品种一列既有数字编号又有字母名），不影响光谱列与干物质的取值。
    """
    path = os.path.join(root, MANGO_CSV)
    sha = eu.compute_sha256(path)
    if sha != MANGO_SHA256:
        raise ValueError(f"芒果数据文件的 SHA-256 与数据卡登记值不符：{path}\n  实际值={sha}")
    logger.log(f"数据文件：{path}（SHA-256 与数据卡登记值一致）")
    loaded: tuple[dict[str, tuple[NDArray, NDArray]], bool] = m74.DATASETS[DATASET]["loader"](root, seed)
    doms, is_reg = loaded
    if not is_reg:
        raise RuntimeError("芒果任务的目标应为连续值（干物质），加载函数却报告为类别")

    df = pd.read_csv(path, dtype=dict.fromkeys(META_STR_COLS, str))
    wl = [c for c in df.columns if c.isdigit()]
    df["_point"] = df.groupby(POINT_KEYS, sort=False, dropna=False).ngroup()
    size = np.bincount(df["_point"].to_numpy())
    found = (int((size == 2).sum()), int((size == 1).sum()), int((size > 2).sum()))
    logger.log(
        f"读入 {len(df)} 行、{len(wl)} 个波长；取样点 {len(size)} 个：两行的 {found[0]} 个、一行的 {found[1]} 个、"
        f"多于两行的 {found[2]} 个；采收批 {df['Pop'].nunique()} 个"
    )
    if found != POINT_SIZES:
        logger.log(f"  [WARN] 取样点的行数分布与文件头里的归纳值 {POINT_SIZES} 不同，先核对分组键再用结果")
    meta: dict[str, RowMeta] = {}
    season_of: dict[str, int] = {}
    for season in sorted(df["Season"].unique().tolist()):
        sub = df[df["Season"] == season]
        if len(sub) < LOAD_MIN_ROWS:
            continue
        sub = sub.sample(min(LOAD_CAP, len(sub)), random_state=seed)
        name = f"S{season}"
        if name not in doms:
            raise RuntimeError(f"{name} 不在加载函数返回的域里：{list(doms)}")
        same_x = np.array_equal(sub[wl].to_numpy(np.float64), doms[name][0])
        same_y = np.array_equal(sub["DM"].to_numpy(np.float64), doms[name][1])
        if not (same_x and same_y):
            raise RuntimeError(f"{name}：照 cdl.load_mango 的步骤重做抽样后与它返回的数组不一致，行归属对不上")
        meta[name] = RowMeta(sub.index.to_numpy(), sub["_point"].to_numpy(), sub["Pop"].to_numpy().astype(str))
        season_of[name] = int(season)
    if list(meta) != list(doms):
        raise RuntimeError(f"域不一致：加载函数给出 {list(doms)}，本脚本重做抽样得到 {list(meta)}")
    for name, m in meta.items():
        logger.log(
            f"  {name}：{len(m.rows)} 条，取样点 {len(np.unique(m.points))} 个"
            f"（两条谱都被抽中的样本 {int(mate_rows(m.points).sum())} 条），采收批 {len(np.unique(m.pops))} 个"
        )
    return MangoTask(doms, meta, season_of)


# ═══════════════════════════════════════════════════════════════════════════════
# 二、留出季各行的归属与三种划分
# ═══════════════════════════════════════════════════════════════════════════════
def held_meta(task: MangoTask, held: str, fa: cdl.FoldArrays) -> RowMeta:
    """留出季各行的归属，对到 fa.xh 的行上。

    cdl.prepare_fold 对留出季做的是：干物质按源季三分位记类（源季 = 其余各域按域字典的次序拼接），只留
    合法类的行，行序不变。这里重做同一筛选，并核对筛出的类别与光谱同 fa.yh、fa.xh 逐元素相同。光谱照
    prepare_fold 的写法先 np.vstack 再逐谱 SNV：数组的内存排布相同，逐行求和的次序才相同，结果才逐位一致。
    """
    y_src = np.concatenate([task.doms[d][1] for d in task.doms if d != held])
    labels = cdl.tertile_labels(y_src, task.doms[held][1])
    mask = np.isin(labels, fa.classes)
    feats = cdl._apply_preprocess(np.vstack([task.doms[held][0]]), "snv")
    if not np.array_equal(labels[mask], fa.yh) or not np.array_equal(feats[mask], fa.xh):
        raise RuntimeError(f"留出 {held}：行归属与 prepare_fold 的类别筛选对不上")
    m = task.meta[held]
    return RowMeta(m.rows[mask], m.points[mask], m.pops[mask])


def split_points(yh: NDArray, classes: list[int], points: NDArray, rng: np.random.Generator) -> NDArray:
    """按取样点对半：各类的取样点随机取一半（向下取整）进 A，返回 xh 各行是否在 A 半。

    同一取样点各行的干物质相同，所以都在同一类里，不会被拆开。
    """
    in_a = np.zeros(len(yh), dtype=bool)
    for c in classes:
        ids = np.unique(points[yh == c])
        perm = rng.permutation(len(ids))
        in_a[(yh == c) & np.isin(points, ids[perm[: len(ids) // 2]])] = True
    return in_a


def split_pops(yh: NDArray, classes: list[int], pops: NDArray, rng: np.random.Generator, need: int) -> NDArray:
    """按采收批划分：整批进 A 或进 B，使 A 半各类的样本数尽量接近该类的一半，返回 xh 各行是否在 A 半。

    目标是各类样本数的一半（向下取整，与按行对半时 A 半的大小相同）。每一遍按随机次序逐个考虑采收批，
    放进 A 半能让「A 半各类样本数与目标之差的平方和」变小就放；共做 POP_TRIES 遍。各遍之间先比
    「A、B 两半里样本数不足 need 的类有几个」（越少越好，need 即跳过规则里的 K+2），再比各类偏差绝对值
    之和，并列取最先出现的。
    """
    ids = np.unique(pops)
    counts = np.array([[int(((pops == g) & (yh == c)).sum()) for c in classes] for g in ids], dtype=np.int64)
    total = counts.sum(axis=0)
    target = total // 2
    best: NDArray = np.zeros(len(ids), dtype=bool)
    best_score: tuple[int, int] | None = None
    for _ in range(POP_TRIES):
        pick: NDArray = np.zeros(len(ids), dtype=bool)
        acc: NDArray = np.zeros(len(classes), dtype=np.int64)
        for g in rng.permutation(len(ids)):
            if int(((acc + counts[g] - target) ** 2).sum()) < int(((acc - target) ** 2).sum()):
                acc = acc + counts[g]
                pick[g] = True
        n_short = int(((acc < need) | (total - acc < need)).sum())
        score = (n_short, int(np.abs(acc - target).sum()))
        if best_score is None or score < best_score:
            best, best_score = pick, score
    out: NDArray = np.isin(pops, ids[best])
    return out


def pool_without_b(
    fa: cdl.FoldArrays, dom_of_row: NDArray, held: str, in_a: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """S_half 的训练池，返回 (训练池 x, 训练池 y, B 半在 xh 中的下标)。

    照 74 号 half_split 的后半段：训练池 = xall 去掉留出季的 B 半，行序不变。
    """
    held_rows = np.where(dom_of_row == held)[0]
    keep = np.ones(len(fa.yall), dtype=bool)
    keep[held_rows[~in_a]] = False
    return fa.xall[keep], fa.yall[keep], np.where(~in_a)[0]


def make_split(
    fa: cdl.FoldArrays, dom_of_row: NDArray, held: str, seed: int, season: int, unit: str, hm: RowMeta, need: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """按给定单位把留出季分成 A、B，返回 (xh 各行是否在 A 半, 训练池 x, 训练池 y, B 半在 xh 中的下标)。

    row   调 m74.half_split，随机流 [种子, 74, 11]；再用 pool_without_b 复算训练池并逐元素比对
    pair  split_points，随机流 [种子, 91, 1, 采收季编号]
    pop   split_pops，随机流 [种子, 91, 2, 采收季编号]；need 是两半里每类至少要有的条数（K+2）
    """
    if unit == "row":
        xa, ya, b_idx = m74.half_split(fa, dom_of_row, held, seed)
        in_a: NDArray = np.ones(len(fa.yh), dtype=bool)
        in_a[b_idx] = False
        xa2, ya2, b2 = pool_without_b(fa, dom_of_row, held, in_a)
        if not (np.array_equal(xa2, xa) and np.array_equal(ya2, ya) and np.array_equal(b2, b_idx)):
            raise RuntimeError(f"留出 {held}：pool_without_b 复算的训练池与 74 号 half_split 的结果不一致")
        return in_a, xa, ya, b_idx
    rng = np.random.default_rng([seed, 91, STREAMS[unit], season])
    in_a = (
        split_points(fa.yh, fa.classes, hm.points, rng)
        if unit == "pair"
        else split_pops(fa.yh, fa.classes, hm.pops, rng, need)
    )
    xa, ya, b_idx = pool_without_b(fa, dom_of_row, held, in_a)
    return in_a, xa, ya, b_idx


def fingerprint(rows: NDArray) -> str:
    """一组原表行号的指纹：排序后取 SHA-256 的前 16 位十六进制。"""
    return hashlib.sha256(np.sort(rows.astype(np.int64)).tobytes()).hexdigest()[:16]


def class_counts(y: NDArray, classes: list[int]) -> str:
    """各类样本数，写成「类:条数」、逗号分隔。"""
    return ",".join(f"{int(c)}:{int((y == c).sum())}" for c in classes)


def split_info(in_a: NDArray, yh: NDArray, classes: list[int], hm: RowMeta) -> dict[str, Any]:
    """一次划分的情况：两半的大小、各类样本数、采收批数，B 半与 A 半在取样点、采收批上的重叠，A 半指纹。"""
    in_b = ~in_a
    n_b = int(in_b.sum())
    mate = float(np.isin(hm.points[in_b], hm.points[in_a]).mean()) if n_b else float("nan")
    batch = float(np.isin(hm.pops[in_b], hm.pops[in_a]).mean()) if n_b else float("nan")
    return {
        "A样本数": int(in_a.sum()),
        "B样本数": n_b,
        "A各类样本数": class_counts(yh[in_a], classes),
        "B各类样本数": class_counts(yh[in_b], classes),
        "A采收批数": len(np.unique(hm.pops[in_a])),
        "B采收批数": len(np.unique(hm.pops[in_b])),
        "B半同点谱在A半占比": mate,
        "B半同批有样本在A半占比": batch,
        "A半指纹": fingerprint(hm.rows[in_a]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 三、评估与单个留出季
# ═══════════════════════════════════════════════════════════════════════════════
class CellEval(NamedTuple):
    """一格的评估结果。"""

    f1: float  # 宏 F1 均值（cdl.eval_kshot 的返回值）
    acc: float  # 准确率均值
    f1s: list[float]  # 逐情节宏 F1
    accs: list[float]  # 逐情节准确率
    hits: tuple[int, int]  # 两层查询（同点谱在 A 半、其余）里判对的人次
    totals: tuple[int, int]  # 两层查询的人次


def eval_cell(
    enc: nn.Module,
    feats: NDArray,
    labels: NDArray,
    classes: list[int],
    mate: NDArray,
    device: torch.device,
    k: int,
    reps: int,
    seed: int,
) -> CellEval:
    """B 半上的 K-shot 评估；mate 标出各评估样本的同点另一条谱是否在 A 半。

    宏 F1 均值就是 cdl.eval_kshot 的返回值（与 74 号 run_fold 同一调用）。它只返回均值，准确率、逐情节数值
    与分层的判对人次在 m74.kshot_episodes 给出的同一批情节上另算一遍（抽样与 eval_kshot 逐次相同，原型与
    预测的算法同 m74.proto_f1）。两遍的宏 F1 均值须一致（差 ≤ 1e-9，同 74 号路线面板的自检），否则中止。
    """
    f1_ref = cdl.eval_kshot(enc, feats, labels, classes, device, k, reps, seed)
    f1s: list[float] = []
    accs: list[float] = []
    hits = [0, 0]
    totals = [0, 0]
    enc.eval()
    with torch.no_grad():
        for sup, q in m74.kshot_episodes(labels, classes, k, reps, seed):
            protos: list[NDArray] = []
            for s in sup:
                feat = enc(torch.from_numpy(feats[s]).float().to(device)).cpu().numpy().mean(0)
                protos.append(feat / (np.linalg.norm(feat) + 1e-8))
            fq = enc(torch.from_numpy(feats[q]).float().to(device)).cpu().numpy()
            pred = np.asarray(classes)[(fq @ np.stack(protos).T).argmax(1)]
            ok = pred == labels[q]
            f1s.append(float(f1_score(labels[q], pred, average="macro", labels=classes)))
            accs.append(float(ok.mean()))
            for j, layer in enumerate((mate[q], ~mate[q])):
                hits[j] += int(ok[layer].sum())
                totals[j] += int(layer.sum())
    mine = float(np.mean(f1s)) if f1s else float("nan")
    same = (np.isnan(mine) and np.isnan(f1_ref)) or abs(mine - f1_ref) <= 1e-9
    if not same:
        raise RuntimeError(f"逐情节复算的宏 F1 均值 {mine} 与 cdl.eval_kshot 的 {f1_ref} 不一致")
    acc = float(np.mean(accs)) if accs else float("nan")
    return CellEval(f1_ref, acc, f1s, accs, (hits[0], hits[1]), (totals[0], totals[1]))


def run_fold(
    task: MangoTask,
    held: str,
    encoder: str,
    args: argparse.Namespace,
    device: torch.device,
    logger: eu.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """单个留出季：R 训一次，三种划分单位各训一个 S_half，各在自己的 B 半上评估 S_half 与 R_B。

    返回 (逐格记录, 逐情节记录, 跳过的格)。与 74 号 run_fold 相同的部分都是直接调用现有函数：
        折内数组  m74.fold_arrays(域字典, True, 留出季, K)
        训练      m74.obtain(编码器, x, y, in_dim, args, device, 种子)；R 的训练池是 (xs, ys)，S_half 的训练池
                  是 xall 去掉留出季的 B 半。深度族每次训练进入时按种子重新播种，各臂初始化相同
        评估      cdl.eval_kshot(编码器, B 半 x, B 半 y, 类别, device, K, 情节数, 种子)；同一个 B 半上两臂的
                  情节相同
    本脚本另加的：R 在留出季全部样本上的评估（「R_全部样本」，与 74 号 R 臂同一调用，对账用）；划分情况与
    分层准确率；跳过规则（见文件头）。S_src、S_shuf 两臂与本对照无关，不跑。
    """
    seed = int(args.seed)
    base: dict[str, Any] = {"留出季": held, "编码器": encoder, "种子": seed}
    got = m74.fold_arrays(task.doms, True, held, args.k)
    if got is None:
        why = "留出季合法类不足 3 个（74 号 fold_arrays 返回空）"
        logger.log(f"    留出 {held}：{why}，三种划分单位都跳过")
        return [], [], [{**base, "半分单位": u, "原因": why, "A各类样本数": "", "B各类样本数": ""} for u in UNITS]
    fa, dom_of_row = got
    hm = held_meta(task, held, fa)
    need = args.k + MIN_EXTRA
    enc_r = m74.obtain(encoder, fa.xs, fa.ys, fa.in_dim, args, device, seed)
    f1_r_all = float(cdl.eval_kshot(enc_r, fa.xh, fa.yh, fa.classes, device, args.k, args.reps, seed))
    tail: dict[str, Any] = {
        "R_全部样本": f1_r_all,
        "留出季样本数": len(fa.yh),
        "留出季成对样本数": int(mate_rows(hm.points).sum()),
        "留出季取样点数": len(np.unique(hm.points)),
        "留出季采收批数": len(np.unique(hm.pops)),
        "源域样本数": len(fa.ys),
        "mmd2": fa.mmd2,
        "元训练情节": args.episodes,
        "评估情节设定": args.reps,
        "成分数上限": args.lin_dim,
        "标签": args.tag,
    }
    rows: list[dict[str, Any]] = []
    ep_rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    n_pairs = tail["留出季成对样本数"]
    logger.log(f"    留出 {held}（{len(fa.yh)} 条，其中成对 {n_pairs} 条）：R 在全部样本上 = {f1_r_all:.4f}")
    for unit in UNITS:
        in_a, xa, ya, b_idx = make_split(fa, dom_of_row, held, seed, task.season[held], unit, hm, need)
        info = split_info(in_a, fa.yh, fa.classes, hm)
        mate_share, batch_share = info["B半同点谱在A半占比"], info["B半同批有样本在A半占比"]
        if (unit != "row" and mate_share > 0) or (unit == "pop" and batch_share > 0):
            raise RuntimeError(f"留出 {held} / {unit}：划分后仍有取样点或采收批同时落在 A、B 两半")
        yb = fa.yh[b_idx]
        short_b = [int(c) for c in fa.classes if int((yb == c).sum()) < need]
        short_a = [int(c) for c in fa.classes if unit != "row" and int((fa.yh[in_a] == c).sum()) < need]
        if short_b or short_a:
            parts = [f"{half} 半里类 {short}" for half, short in (("B", short_b), ("A", short_a)) if short]
            why = f"{'、'.join(parts)} 不足 K+{MIN_EXTRA} = {need} 条"
            skips.append(
                {**base, "半分单位": unit, "原因": why, **{c: info[c] for c in ("A各类样本数", "B各类样本数")}}
            )
            logger.log(f"      {unit}：{why}（A {info['A各类样本数']}；B {info['B各类样本数']}），跳过")
            continue
        enc_h = m74.obtain(encoder, xa, ya, fa.in_dim, args, device, seed)
        xb = fa.xh[b_idx]
        mate_b = (~in_a & np.isin(hm.points, hm.points[in_a]))[b_idx]
        f1_of: dict[str, float] = {}
        for arm, enc, n_train in ((ARM_HALF, enc_h, len(ya)), (ARM_REF, enc_r, len(fa.ys))):
            res = eval_cell(enc, xb, yb, fa.classes, mate_b, device, args.k, args.reps, seed)
            head = {**base, "半分单位": unit, "臂": arm}
            row: dict[str, Any] = {
                **head,
                "K": args.k,
                "宏F1": res.f1,
                "准确率": res.acc,
                "情节数": len(res.f1s),
                "宏F1情节SD": float(np.std(res.f1s)) if res.f1s else float("nan"),
                "训练池样本数": n_train,
                **info,
            }
            for name, hit, total in zip(STRATA, res.hits, res.totals, strict=True):
                row[f"查询准确率_{name}"] = hit / total if total else float("nan")
                row[f"查询人次_{name}"] = total
            rows.append({**row, **tail})
            for i, (f1_i, acc_i) in enumerate(zip(res.f1s, res.accs, strict=True)):
                ep_rows.append({**head, "情节": i, "宏F1": f1_i, "准确率": acc_i})
            f1_of[arm] = res.f1
        logger.log(
            f"      {unit}：S_half={f1_of[ARM_HALF]:.4f}  R_B={f1_of[ARM_REF]:.4f}  "
            f"Δ_half={f1_of[ARM_HALF] - f1_of[ARM_REF]:+.4f}（A {info['A样本数']} / B {info['B样本数']}；"
            f"B 半同点谱在 A 半 {mate_share:.1%}，同批有样本在 A 半 {batch_share:.1%}）"
        )
    return rows, ep_rows, skips


# ═══════════════════════════════════════════════════════════════════════════════
# 四、结果文件与汇总
# ═══════════════════════════════════════════════════════════════════════════════
def seed_file(encoder: str, tag: str, seed: int, stage: str) -> str:
    """逐种子结果表的路径：91_mango_half_unit__<编码器>[__<tag>]__seed<S>[__pilot].xlsx。"""
    seg = f"__{tag}" if tag else ""
    suffix = "" if stage == "formal" else "__pilot"
    return os.path.join(SEED_DIR, f"{STEM}__{encoder}{seg}__seed{seed}{suffix}.xlsx")


def seed_files(stage: str, tag: str) -> list[str]:
    """给定阶段与标签的全部逐种子结果表；文件名逐段匹配，带别的标签或别的阶段的文件不会混进来。"""
    seg = re.escape(f"__{tag}" if tag else "")
    suffix = "" if stage == "formal" else "__pilot"
    pat = re.compile(rf"{STEM}__({'|'.join(enc_lib.ENCODERS)}){seg}__seed\d+{suffix}\.xlsx")
    names = sorted(os.listdir(SEED_DIR)) if os.path.isdir(SEED_DIR) else []
    return [os.path.join(SEED_DIR, n) for n in names if pat.fullmatch(n)]


def arm_table(cells: pd.DataFrame, col: str) -> dict[str, dict[Key, float]]:
    """臂 → {(留出季, 编码器, 半分单位, 种子): 数值}；同一格出现两次即报错。"""
    out: dict[str, dict[Key, float]] = {ARM_HALF: {}, ARM_REF: {}}
    cols = (cells["臂"], cells["留出季"], cells["编码器"], cells["半分单位"], cells["种子"], cells[col])
    for arm, held, enc, unit, seed, val in zip(*cols, strict=True):
        key = (str(held), str(enc), str(unit), int(seed))
        if key in out[str(arm)]:
            raise ValueError(f"重复格：{arm} {key}")
        out[str(arm)][key] = float(val)
    return out


def paired_delta(f1: dict[str, dict[Key, float]]) -> dict[Key, float]:
    """逐格的 Δ_half = S_half − R_B（同一留出季、编码器、半分单位、种子）；缺一臂或非有限值的格不计。"""
    out: dict[Key, float] = {}
    for key, val in f1[ARM_HALF].items():
        gap = val - f1[ARM_REF].get(key, float("nan"))
        if np.isfinite(gap):
            out[key] = gap
    return out


def by_seed(values: dict[Key, float], held: str, enc: str, unit: str) -> dict[int, float]:
    """一个（留出季, 编码器, 半分单位）里 种子 → 数值。"""
    return {key[3]: val for key, val in values.items() if key[:3] == (held, enc, unit)}


def merge_seeds(parts: list[dict[int, float]]) -> dict[int, float]:
    """若干组 {种子: 数值} 合并成每个种子的均值；只留各组都有的种子。"""
    if not parts:
        return {}
    common = set(parts[0])
    for part in parts[1:]:
        common &= set(part)
    return {s: float(np.mean([part[s] for part in parts])) for s in sorted(common)}


def stat_cols(name: str, values: list[float]) -> dict[str, Any]:
    """一组逐种子数值的均值、95% t 区间、为正个数与个数；列名以 name 打头。

    均值与区间上下限按全精度写出（「_种子t区间」一列是 4 位小数的文字，供阅读）。
    """
    lo, hi = m74._t_interval(values)
    return {
        name: float(np.mean(values)),
        f"{name}_种子t区间": f"{lo:.4f}–{hi:.4f}",
        f"{name}_种子t区间下限": float(lo),
        f"{name}_种子t区间上限": float(hi),
        f"{name}_为正种子数": sum(v > 0 for v in values),
        f"{name}_配对种子数": len(values),
    }


def merged_rows(
    values: dict[Key, float], unit: str, encs: list[str], helds: list[str], name: str
) -> list[dict[str, Any]]:
    """一种半分单位下各季合并的逐种子统计：每个编码器一行，另加「各族合并」一行。

    每个种子先对各留出季取均值（只用各留出季都有数值的种子），再对种子求均值与 t 区间；
    「各族合并」是各编码器的逐种子合并值再对编码器取均值。
    """
    out: list[dict[str, Any]] = []
    families: list[dict[int, float]] = []
    for enc in encs:
        per_held = [d for d in (by_seed(values, held, enc, unit) for held in helds) if d]
        merged = merge_seeds(per_held)
        if not merged:
            continue
        families.append(merged)
        held_means = [float(np.mean(list(d.values()))) for d in per_held]
        out.append(
            {
                "编码器": enc,
                "留出季数": len(per_held),
                **stat_cols(name, list(merged.values())),
                f"{name}_为正留出季": f"{sum(v > 0 for v in held_means)}/{len(held_means)}",
            }
        )
    overall = merge_seeds(families)
    if overall:
        out.append({"编码器": "各族合并", "留出季数": len(helds), **stat_cols(name, list(overall.values()))})
    return out


def fold_table(
    cells: pd.DataFrame,
    f1: dict[str, dict[Key, float]],
    delta: dict[Key, float],
    helds: list[str],
    encs: list[str],
    skip_n: dict[tuple[str, str, str], int],
) -> list[dict[str, Any]]:
    """逐（留出季, 编码器, 半分单位）：两臂与 Δ_half 的种子统计，及划分情况的种子均值。"""
    out: list[dict[str, Any]] = []
    for held in helds:
        for enc in encs:
            for unit in UNITS:
                d = by_seed(delta, held, enc, unit)
                n_skip = skip_n.get((held, enc, unit), 0)
                if not d and not n_skip:
                    continue
                row: dict[str, Any] = {"留出季": held, "编码器": enc, "半分单位": unit, "跳过的种子数": n_skip}
                if d:
                    seeds = sorted(d)
                    for arm in (ARM_HALF, ARM_REF):
                        row[arm] = float(np.mean([f1[arm][(held, enc, unit, s)] for s in seeds]))
                    row.update(stat_cols("Δ_half", [d[s] for s in seeds]))
                    sel = (cells["留出季"] == held) & (cells["编码器"] == enc) & (cells["半分单位"] == unit)
                    sub = cells[sel & (cells["臂"] == ARM_HALF)]
                    for col in SPLIT_MEAN_COLS:
                        row[f"{col}_种子均值"] = float(sub[col].mean())
                out.append(row)
    return out


def diff_table(delta: dict[Key, float], helds: list[str], encs: list[str]) -> list[dict[str, Any]]:
    """pair − row、pop − row：Δ_half 之差，同一留出季、编码器、种子内配对；另有各季合并与各族合并的行。"""
    out: list[dict[str, Any]] = []
    for other in UNITS:
        if other == "row":
            continue
        diff: dict[Key, float] = {}
        for (held, enc, unit, seed), val in delta.items():
            ref = delta.get((held, enc, "row", seed))
            if unit == other and ref is not None:
                diff[(held, enc, other, seed)] = val - ref
        label = f"{other} − row"
        for enc in encs:
            for held in helds:
                d = by_seed(diff, held, enc, other)
                if d:
                    stats = stat_cols("Δ_half之差", list(d.values()))
                    out.append({"对比": label, "留出季": held, "编码器": enc, **stats})
        for row in merged_rows(diff, other, encs, helds, "Δ_half之差"):
            out.append({"对比": label, "留出季": "各季合并", **row})
    return out


def strata_table(cells: pd.DataFrame, helds: list[str], encs: list[str]) -> list[dict[str, Any]]:
    """row 单位下 B 半查询的分层准确率：两臂在两层上的种子均值，及两层之间 S_half − R_B 的差（差的差）。"""
    sub = cells[cells["半分单位"] == "row"]
    if sub.empty:
        return []
    acc_mate = arm_table(sub, f"查询准确率_{STRATA[0]}")
    acc_rest = arm_table(sub, f"查询准确率_{STRATA[1]}")
    n_mate = arm_table(sub, f"查询人次_{STRATA[0]}")[ARM_HALF]
    did: dict[Key, float] = {}
    for key, half_mate in acc_mate[ARM_HALF].items():
        ref_mate = acc_mate[ARM_REF].get(key, float("nan"))
        gap_rest = acc_rest[ARM_HALF].get(key, float("nan")) - acc_rest[ARM_REF].get(key, float("nan"))
        val = (half_mate - ref_mate) - gap_rest
        if np.isfinite(val):
            did[key] = val
    out: list[dict[str, Any]] = []
    for enc in encs:
        for held in helds:
            d = by_seed(did, held, enc, "row")
            if not d:
                continue
            keys = [(held, enc, "row", s) for s in sorted(d)]
            row: dict[str, Any] = {"留出季": held, "编码器": enc}
            for name, tab in zip(STRATA, (acc_mate, acc_rest), strict=True):
                for arm in (ARM_HALF, ARM_REF):
                    row[f"{arm}_{name}"] = float(np.mean([tab[arm][key] for key in keys]))
            row[f"查询人次_{STRATA[0]}_种子均值"] = round(float(np.mean([n_mate[key] for key in keys])), 1)
            row.update(stat_cols("差的差", [did[key] for key in keys]))
            out.append(row)
    for row in merged_rows(did, "row", encs, helds, "差的差"):
        out.append({"留出季": "各季合并", **row})
    return out


def reconcile(
    cells: pd.DataFrame, f1: dict[str, dict[Key, float]], logger: eu.Logger
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """row 单位与 74 号正式逐种子表的对账，返回 (逐留出季 × 编码器, 逐种子)；没有 74 号的表时返回两张空表。

    74 号的表经 m74.load_long("formal") 读入（各文件的第一张表：dataset、held、编码器、种子、臂、F1、n_eval），
    只取 dataset = mango_season 的行；S_half、R_B 两臂与 R 臂（留出季全部样本）逐（留出季, 编码器, 种子）并排。
    只对本脚本已有的格对账；74 号表里没有的格在备注里写明。
    """
    try:
        ref_df = m74.load_long("formal")
    except FileNotFoundError:
        logger.log("  [WARN] 没有 74 号的正式逐种子结果表，不对账")
        return [], []
    ref_df = ref_df[ref_df["dataset"] == DATASET]
    missing = (float("nan"), -1)
    ref: dict[tuple[str, str, str, int], tuple[float, int]] = {}
    ref_cols = (ref_df["held"], ref_df["编码器"], ref_df["臂"], ref_df["种子"], ref_df["F1"], ref_df["n_eval"])
    for held, enc, arm, seed, val, n_eval in zip(*ref_cols, strict=True):
        ref[(str(held), str(enc), str(arm), int(seed))] = (float(val), -1 if pd.isna(n_eval) else int(n_eval))
    sub = cells[(cells["半分单位"] == "row") & (cells["臂"] == ARM_HALF)]
    own: dict[tuple[str, str, int], tuple[float, int]] = {}
    own_cols = (sub["留出季"], sub["编码器"], sub["种子"], sub["R_全部样本"], sub["B样本数"])
    for held, enc, seed, r_all, n_b in zip(*own_cols, strict=True):
        own[(str(held), str(enc), int(seed))] = (float(r_all), int(n_b))

    detail: list[dict[str, Any]] = []
    for held, enc, seed in sorted(own, key=lambda t: (t[0], enc_lib.ENCODERS.index(t[1]), t[2])):
        own_h = f1[ARM_HALF].get((held, enc, "row", seed), float("nan"))
        own_r = f1[ARM_REF].get((held, enc, "row", seed), float("nan"))
        r_all, n_b = own[(held, enc, seed)]
        ref_h = ref.get((held, enc, ARM_HALF, seed), missing)
        ref_r = ref.get((held, enc, ARM_REF, seed), missing)
        ref_full = ref.get((held, enc, "R", seed), missing)
        gap = (own_h - own_r) - (ref_h[0] - ref_r[0])
        detail.append(
            {
                "留出季": held,
                "编码器": enc,
                "种子": seed,
                "74号_S_half": ref_h[0],
                "本脚本_S_half": own_h,
                "S_half之差": own_h - ref_h[0],
                "74号_R_B": ref_r[0],
                "本脚本_R_B": own_r,
                "R_B之差": own_r - ref_r[0],
                "74号_Δ_half": ref_h[0] - ref_r[0],
                "本脚本_Δ_half": own_h - own_r,
                "Δ_half之差": gap,
                "74号_R全部样本": ref_full[0],
                "本脚本_R全部样本": r_all,
                "R全部样本之差": r_all - ref_full[0],
                "74号_B样本数": ref_h[1],
                "本脚本_B样本数": n_b,
                "超过容差": bool(abs(gap) > TOL_74),
                "备注": "" if np.isfinite(gap) else "74 号正式逐种子表或本脚本缺这一格的数值，未比较",
            }
        )

    summary: list[dict[str, Any]] = []
    for held, enc in dict.fromkeys((r["留出季"], r["编码器"]) for r in detail):
        part = [r for r in detail if (r["留出季"], r["编码器"]) == (held, enc) and np.isfinite(r["Δ_half之差"])]
        row: dict[str, Any] = {"留出季": held, "编码器": enc, "配对种子数": len(part)}
        if part:
            mean_own = float(np.mean([r["本脚本_Δ_half"] for r in part]))
            mean_ref = float(np.mean([r["74号_Δ_half"] for r in part]))
            row.update(
                {
                    "74号_Δ_half": round(mean_ref, 4),
                    "本脚本row_Δ_half": round(mean_own, 4),
                    "均值之差": round(mean_own - mean_ref, 4),
                    "逐种子最大绝对差_Δ_half": round(max(abs(r["Δ_half之差"]) for r in part), 4),
                    "逐种子最大绝对差_S_half": round(max(abs(r["S_half之差"]) for r in part), 4),
                    "逐种子最大绝对差_R_B": round(max(abs(r["R_B之差"]) for r in part), 4),
                    "B样本数全部相同": all(r["74号_B样本数"] == r["本脚本_B样本数"] for r in part),
                    "均值相差超过容差": bool(abs(mean_own - mean_ref) > TOL_74),
                    "超过容差的种子数": sum(bool(r["超过容差"]) for r in part),
                }
            )
        summary.append(row)

    gaps = [abs(r["Δ_half之差"]) for r in detail if np.isfinite(r["Δ_half之差"])]
    flagged = [r for r in summary if r.get("均值相差超过容差")]
    logger.log(
        f"\n与 74 号正式逐种子表对账：并排 {len(gaps)} 格（留出季 × 编码器 × 种子），缺 {len(detail) - len(gaps)} 格；"
        f"Δ_half 的最大绝对差 {max(gaps, default=float('nan')):.2e}；"
        f"种子均值相差超过 {TOL_74} 的（留出季, 编码器）有 {len(flagged)} 个"
    )
    for r in flagged:
        old, new = r["74号_Δ_half"], r["本脚本row_Δ_half"]
        logger.log(f"  [超过容差] {r['留出季']} / {r['编码器']}：74 号 {old:.4f}，本脚本 row {new:.4f}")
    return summary, detail


def aggregate(stage: str, tag: str, logger: eu.Logger) -> None:
    """跨种子汇总一个（阶段, 标签）下的全部逐种子结果表。

    表a 逐留出季对照；表b 各季合并；表c 单位间配对差；表d、表e 与 74 号正式结果对账（只在正式阶段、
    缺省设定下做）；表f 按行划分的分层准确率；另有长格式、跳过的格、表注、compute。
    """
    files = seed_files(stage, tag)
    if not files:
        raise FileNotFoundError(f"没有 91 号的逐种子结果（阶段 {stage}，标签 {tag or '无'}）")
    cell_parts: list[Any] = []
    skip_parts: list[Any] = []
    cost_parts: list[Any] = []
    for f in files:
        part = pd.read_excel(f, sheet_name=SHEET_CELL)
        if not part.empty:
            cell_parts.append(part)
        part = pd.read_excel(f, sheet_name=SHEET_SKIP)
        if not part.empty:
            skip_parts.append(part)
        part = pd.read_excel(f, sheet_name="compute")
        part.insert(0, "文件", os.path.basename(f))
        cost_parts.append(part)
    if not cell_parts:
        raise ValueError("逐种子结果表里没有评过的格（全部被跳过，见各表的「跳过的格」）")
    cells = pd.concat(cell_parts, ignore_index=True)
    skip_df = pd.concat(skip_parts, ignore_index=True) if skip_parts else pd.DataFrame(columns=SKIP_COLS)
    cost_df = pd.concat(cost_parts, ignore_index=True)
    settings = cells[SETTING_COLS].drop_duplicates()
    if len(settings) != 1:
        raise ValueError(
            "待汇总的逐种子结果表设定不一致（换了设定的运行用 --tag 分开）：\n" + settings.to_string(index=False)
        )
    seeds_all = sorted({int(s) for s in cells["种子"].tolist()})
    if stage == "formal" and set(seeds_all) != set(eu.RECOMMENDED_SEEDS):
        logger.log(f"  [WARN] 种子集合 {seeds_all} 不是 Formal 5 种子，汇总仅作进度查看")
    logger.log(f"读入 {len(files)} 个逐种子结果表（阶段 {stage}，标签 {tag or '无'}），种子 {seeds_all}")
    n_mixed = int((cells.groupby(["留出季", "种子", "半分单位"])["A半指纹"].nunique() > 1).sum())
    if n_mixed:
        raise ValueError(f"有 {n_mixed} 个（留出季, 种子, 半分单位）在各编码器族上的 A 半指纹不同：划分应与编码器无关")

    f1 = arm_table(cells, "宏F1")
    delta = paired_delta(f1)
    helds = sorted({str(h) for h in cells["留出季"].tolist()})
    present = {str(e) for e in cells["编码器"].tolist()}
    encs = [e for e in enc_lib.ENCODERS if e in present]
    skip_n: dict[tuple[str, str, str], int] = {}
    for held, enc, unit in zip(skip_df["留出季"], skip_df["编码器"], skip_df["半分单位"], strict=True):
        cell_key = (str(held), str(enc), str(unit))
        skip_n[cell_key] = skip_n.get(cell_key, 0) + 1
    if skip_n:
        logger.log(f"  [WARN] 有 {sum(skip_n.values())} 个（留出季, 编码器, 种子, 半分单位）被跳过，见「跳过的格」表")

    fold_df = pd.DataFrame(fold_table(cells, f1, delta, helds, encs, skip_n))
    cross_rows = [{"半分单位": u, **row} for u in UNITS for row in merged_rows(delta, u, encs, helds, "Δ_half")]
    cross_df = pd.DataFrame(cross_rows)
    diff_df = pd.DataFrame(diff_table(delta, helds, encs))
    strata_df = pd.DataFrame(strata_table(cells, helds, encs))
    logger.log("\n各季合并的 Δ_half（逐种子配对）\n" + cross_df.to_string(index=False))
    if not diff_df.empty:
        merged_diff = diff_df[diff_df["留出季"] == "各季合并"]
        logger.log("\n单位之间 Δ_half 的配对差（各季合并）\n" + merged_diff.to_string(index=False))
    if not strata_df.empty:
        merged_strata = strata_df[strata_df["留出季"] == "各季合并"]
        logger.log("\n按行划分时两层查询之间的差的差（各季合并）\n" + merged_strata.to_string(index=False))

    notes = [{"表": name, "说明": text} for name, text in NOTES]
    notes.append({"表": "设定", "说明": "；".join(f"{c} = {settings.iloc[0][c]}" for c in SETTING_COLS)})
    rec_rows: list[dict[str, Any]] = []
    rec_detail: list[dict[str, Any]] = []
    if stage == "formal" and not tag:
        rec_rows, rec_detail = reconcile(cells, f1, logger)
    else:
        notes.append({"表": "表d、表e", "说明": "本次未对账：只有正式阶段、缺省设定的结果与 74 号的正式结果同预算。"})
    if stage != "formal":
        notes.append({"表": "阶段", "说明": "预跑结果（减小的预算），只用于检查流程，不进稿件。"})

    seg = f"__{tag}" if tag else ""
    suffix = "" if stage == "formal" else "__pilot"
    out = os.path.join(eu.OUTPUT_DIR, f"{STEM}{seg}{suffix}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        fold_df.to_excel(writer, sheet_name=eu.table_sheet_name(0, "逐留出季对照"), index=False)
        cross_df.to_excel(writer, sheet_name=eu.table_sheet_name(1, "各季合并"), index=False)
        diff_df.to_excel(writer, sheet_name=eu.table_sheet_name(2, "单位间配对差"), index=False)
        pd.DataFrame(rec_rows).to_excel(writer, sheet_name=eu.table_sheet_name(3, "与74号对账"), index=False)
        pd.DataFrame(rec_detail).to_excel(writer, sheet_name=eu.table_sheet_name(4, "对账逐种子"), index=False)
        strata_df.to_excel(writer, sheet_name=eu.table_sheet_name(5, "按行划分的分层准确率"), index=False)
        cells.to_excel(writer, sheet_name=eu.table_sheet_name(6, "长格式"), index=False)
        skip_df.to_excel(writer, sheet_name=SHEET_SKIP, index=False)
        pd.DataFrame(notes).to_excel(writer, sheet_name="表注", index=False)
        cost_df.to_excel(writer, sheet_name="compute", index=False)
    logger.log(f"汇总表 → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 五、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="91_mango_half_unit：芒果干物质任务上 S_half 的划分单位对照")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--stage", default=None, choices=["pilot", "formal"], help="缺省：运行时 pilot，--aggregate 时 formal"
    )
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED, help="一次一个种子")
    parser.add_argument("--encoders", default=",".join(enc_lib.ENCODERS), help="逗号分隔的编码器族，缺省全部六族")
    parser.add_argument("--data_root", default="05data")
    parser.add_argument("--episodes", type=int, default=None, help="元训练情节数；缺省 formal 1000、pilot 200")
    parser.add_argument("--reps", type=int, default=None, help="每格评估情节数；缺省 formal 50、pilot 10")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="每类支持样本数")
    parser.add_argument("--lin_dim", type=int, default=DEFAULT_LIN_DIM, help="线性族（pls / pca）的成分数上限")
    parser.add_argument("--tag", default="", help="写进输出文件名的标签；formal 阶段改了设定时必须给")
    parser.add_argument("--aggregate", action="store_true", help="汇总 --stage、--tag 对应的逐种子结果表")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.stage = args.stage or ("formal" if args.aggregate else "pilot")
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("--tag 只能含字母、数字、下划线与连字符")
    if args.aggregate:
        logger = eu.get_logger(STEM)
        aggregate(args.stage, args.tag, logger)
        logger.close()
        return

    stage_default = STAGE_BUDGET[args.stage]
    args.episodes = stage_default["episodes"] if args.episodes is None else args.episodes
    args.reps = stage_default["reps"] if args.reps is None else args.reps
    enc_list = list(dict.fromkeys(e.strip() for e in args.encoders.split(",") if e.strip()))
    unknown = sorted(set(enc_list) - set(enc_lib.ENCODERS))
    if unknown or not enc_list:
        parser.error(f"--encoders 须从 {list(enc_lib.ENCODERS)} 里取；未知：{unknown}")
    if min(args.k, args.episodes, args.reps, args.lin_dim) < 1:
        parser.error("--k、--episodes、--reps、--lin_dim 须为正整数")
    defaults: dict[str, Any] = {"k": DEFAULT_K, "lin_dim": DEFAULT_LIN_DIM, **stage_default}
    changed = [name for name, val in defaults.items() if getattr(args, name) != val]
    if args.stage == "formal" and changed and not args.tag:
        parser.error(f"formal 阶段改了 {changed}：请同时给 --tag，不带标签的结果文件只留给缺省设定")

    logger = eu.get_logger(STEM)
    device = eu.get_device(args.device)
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "芒果干物质任务（按采收季留出）上 S_half 的划分单位对照",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "编码器": [f"{e}（{enc_lib.ARCH_DESC[e]}）" for e in enc_list],
            "臂": f"R 只训一次；{ARM_HALF} 在各划分单位下各训一次；{ARM_REF}（即 R^h）在同一个 B 半上评估",
            "划分单位": [f"{u}（{desc}）" for u, desc in UNITS.items()],
            "K / 元训练情节 / 每格评估情节": f"{args.k} / {args.episodes} / {args.reps}",
            "线性族成分数上限": args.lin_dim,
            "非缺省设定 / 标签": f"{changed or '无'} / {args.tag or '无'}",
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))
    task = load_task(args.data_root, args.seed, logger)
    os.makedirs(SEED_DIR, exist_ok=True)
    for encoder in enc_list:
        logger.log(f"\n【{encoder}】{enc_lib.ARCH_DESC[encoder]}")
        rows: list[dict[str, Any]] = []
        ep_rows: list[dict[str, Any]] = []
        skips: list[dict[str, Any]] = []
        with eu.track_compute(f"芒果半分单位 / {encoder}", logger, run) as budget:
            for held in task.doms:
                fold_rows, fold_eps, fold_skips = run_fold(task, held, encoder, args, device, logger)
                rows += fold_rows
                ep_rows += fold_eps
                skips += fold_skips
        out = seed_file(encoder, args.tag, args.seed, args.stage)
        cost = {"seed": args.seed, "stage": args.stage, "encoder": encoder, **budget}
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name=SHEET_CELL, index=False)
            pd.DataFrame(ep_rows).to_excel(writer, sheet_name=SHEET_EPIS, index=False)
            pd.DataFrame(skips, columns=SKIP_COLS).to_excel(writer, sheet_name=SHEET_SKIP, index=False)
            pd.DataFrame([cost]).to_excel(writer, sheet_name="compute", index=False)
        logger.log(f"结果表 → {out}（评了 {len(rows) // 2} 格，跳过 {len(skips)} 格）")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
