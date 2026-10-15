"""90_mango_region_task.py: 第二个产地任务——芒果产区二分类（NT 对 QLD）按采收季留出的各暴露臂（补充材料 S9.10）。

为什么:
    主口径的 18 个留出域里只有苹果的 2 个是产地任务，苹果的产地标签又是申报值。Anderson 等 2020 公开的
    芒果数据（主口径里按干物质三分位用过）带有采集方记录的产区（NT、QLD），四个采收季里两个产区都有样本。
    本脚本把产区当类别、采收季当域，在这个任务上重跑 74 号的各暴露臂，看暴露带来的抬高是否复现。
    这是主口径之外单独报告的补充任务：任务是看过主结果之后才定的，不并入 18 个留出域的任何汇总，
    也不写入 74、75 号等现有脚本的结果文件。

任务:
    类别    产区，NT 记 0、QLD 记 1（二分类）
    域      采收季 S1–S4；每次留出一季、其余三季为源域，共 4 个留出域
    光谱    表头里列名为纯数字的全部 306 个波长（285–1200 nm，步长 3 nm），逐谱 SNV；取列同
            crossdataset_lib.load_mango，预处理调用 prepare_fold 所用的 _apply_preprocess
    抽样    每季每个产区按种子抽至多 --cap 条（缺省 500，0 为不设上限），随机流
            default_rng([种子, 90, 季, 类])；先按版本筛选品种，再抽样
    版本    all     全部 10 个品种
            shared  只取两个产区都有的四个品种（Caly、HG、KP、R2E2）

臂（训练池、训练配方与评估口径同 74 号 run_fold；逐项对照见本文件 run_fold 的文档字符串）:
    R       折限定  : 训练池 = 源域
    S_src   目标暴露 : 训练池 = 源域 ∪ 留出季（带标签）
    S_shuf  标签置乱 : 训练池同 S_src，留出季样本的产区标签在该季内随机置换；pca 不看标签，不跑这一臂
    S_half  半数暴露 : 留出季各产区的样本按种子对半分成 A、B；训练池 = 源域 ∪ A（带标签），只在 B 上评估；
                      R 与 S_src 的编码器也在同一个 B 上评估，记作 R_B（即 R^h）与 S_src_B
    对照    Δ_rep = S_src − R；Δ_lab = S_src − S_shuf；Δ_half = S_half − R_B
    编码器  cnn / mlp / resnet / transformer 按原型网络元训练，pls / pca 在训练池上直接拟合（encoders.py）
    预算    K = 5；元训练 1000 情节、每格 50 个评估情节（pilot 时 200 情节、10 个评估情节）；权重不落盘

限定（写稿时照此限定，不作为产地化学的证据）:
    产区与品种相关。all 版里 1201、1243、4069、LadyG、LadyJ 只在 NT，Keitt 只在 QLD，编码器可以靠品种
    把两个产区分开；shared 版只留两个产区都有的四个品种，但各季各产区的品种构成仍不相同（例如 S4 的 NT
    没有 HG、S2 的 QLD 没有 KP）。产区与采收批嵌套：112 个采收批没有一个同时出现在两个产区。
    各留出域两个产区的品种构成写在逐种子结果表的「品种构成」表里。
    表里没有果实编号。元数据与干物质值完全相同的相邻两行有 5641 对，另有 409 行单独成组；本脚本把这样的
    一组行叫作一个取样点（由表内数值归纳，数据说明里没有这一项）。74 号的 half_split 按行对半，同一取样点
    的两条谱可能一条进 A、一条进 B。缺省沿用按行对半（--half_unit row），并把各格 B 半样本里「同点的另一条
    谱在 A 半」的占比记进结果表；--half_unit pair 改为按取样点对半（正式阶段须同时给 --tag）。
    评估情节里支持样本的同点另一条谱也可能落在查询集里，这一点各臂相同。

随机数:
    新增的抽样只有两处：每季每个产区的抽样 default_rng([种子, 90, 季, 类])，按取样点对半
    default_rng([种子, 90, 11])（季取 1–4，与 11 不相撞）。调用现有函数的地方保持它们原有的播种：
    标签置乱 [种子, 74, 7]、按行对半 [种子, 74, 11]（74 号），元训练 eu.set_random_seed(种子)、
    评估情节 np.random.seed(种子)（crossdataset_lib）。

复用:
    export_utils      路径常量、get_device、set_random_seed、enforce_publication_grade_device、get_logger、
                      log_experiment_header、init_wandb、track_compute、compute_sha256、table_sheet_name
    74 号脚本         shuffled_pool、half_split、obtain、kshot_episodes、_t_interval，以及 HALF_ARMS 与
                      CONTRASTS 的定义（导入 74 号会连带导入 71、20 号，这几个脚本要一起在 02code 里）
    crossdataset_lib  FoldArrays、_apply_preprocess、mmd2_linear、eval_kshot；meta_train 经 74 号 obtain 调用
    encoders          ENCODERS、ARCH_DESC；build_encoder、fit_linear_encoder 经上面的函数调用

输入:
    <data_root>/041_mango_anderson_crossseason/NAnderson2020MendeleyMangoNIRData.csv（读入时核对 SHA-256）

运行方式:
    cd <项目根>
    # 预跑（种子 2004；元训练 200 情节、每格 10 个评估情节，结果文件名带 __pilot）
    python 02code/90_mango_region_task.py --device cuda --stage pilot --seed 2004 --variant all \
        --data_root <公开数据根目录>
    # Formal（CUDA；每次一个种子 × 一个版本；--encoders 缺省为全部六族，也可以只给一族以便并行）
    python 02code/90_mango_region_task.py --device cuda --stage formal --seed 20060515 --variant shared \
        --encoders transformer --data_root <公开数据根目录>
    # 汇总（缺省汇总正式结果；加 --stage pilot 则汇总预跑结果，另存一份）
    python 02code/90_mango_region_task.py --aggregate

输出文件:
    04outputs/formal_seeds/90_mango_region_task__<版本>__<编码器>__seed<S>[__pilot].xlsx
        表a 逐臂结果（一行一格）、表b 品种构成、表c 逐情节、compute。给了 --tag 时 <编码器> 之后多一段
        __<tag>；正式阶段改了 --cap、--half_unit、--k、--episodes、--reps、--lin_dim 必须给 --tag，
        不带标签的文件只留给缺省设定
    04outputs/90_mango_region_task.xlsx（--aggregate；有标签时加 __<tag>，--stage pilot 时再加 __pilot）
        表a 逐留出季对照、表b 跨季汇总、表c R臂绝对水平、表d 品种构成、表e 长格式、表注、compute；
        表a–c 的均值按全精度写出，取用方显示时只舍入一次
    05logs/90_mango_region_task_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import argparse
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
m74: Any = importlib.import_module("74_multi_encoder_leakage")

STEM = "90_mango_region_task"
MANGO_CSV = os.path.join("041_mango_anderson_crossseason", "NAnderson2020MendeleyMangoNIRData.csv")
MANGO_SHA256 = "86159e2df9244fa52f821856ea003f562ef7f6a031dd08794e3cd4458f0f153f"  # 数据卡登记值
REGIONS = ("NT", "QLD")  # 类别编号 0、1
SHARED_CULTIVARS = ("Caly", "HG", "KP", "R2E2")  # 两个产区都有的品种
VARIANTS = {"all": "全部品种", "shared": "只取两个产区都有的四个品种"}
META_STR_COLS = ("Set", "Region", "Date", "Type", "Cultivar", "Pop", "Temp")  # 按字符串读入的元数据列
POINT_KEYS = ["Season", "Region", "Date", "Type", "Cultivar", "Pop", "Temp", "DM"]  # 全同的行算同一取样点
MIN_SEASON_ROWS = 60  # 同 cdl.load_mango：不足 60 行的季不作为域
MIN_SRC_PER_CLASS = 15  # 同 cdl.prepare_fold：源域每类至少 15 条
MIN_CLASSES = 2  # 两个产区缺一即不评（cdl.prepare_fold 与 74 号 run_fold 里的门槛是 3 类）
ARMS = ("R", "S_src", "S_shuf", "S_half")  # 同 74 号 ARMS
HALF_ARMS: dict[str, str] = dict(m74.HALF_ARMS)  # R、S_src 的编码器在 B 半上的评估：R_B（即 R^h）、S_src_B
EVAL_ARMS: tuple[str, ...] = (*ARMS, *HALF_ARMS)
CONTRASTS: dict[str, tuple[str, str]] = {name: m74.CONTRASTS[name] for name in ("Δ_rep", "Δ_lab", "Δ_half")}
STAGE_BUDGET: dict[str, dict[str, int]] = {
    "formal": {"episodes": 1000, "reps": 50},
    "pilot": {"episodes": 200, "reps": 10},
}
DEFAULT_CAP, DEFAULT_K, DEFAULT_LIN_DIM = 500, 5, 16
SEED_DIR = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
SHEET_ARMS = eu.table_sheet_name(0, "逐臂结果")
SHEET_COMP = eu.table_sheet_name(1, "品种构成")
SHEET_EPIS = eu.table_sheet_name(2, "逐情节")
# 逐种子结果表里记录本次设定的列；汇总时这几列必须在全部文件里一致
SETTING_COLS = ["K", "抽样上限", "S_half划分单位", "元训练情节", "评估情节设定", "成分数上限"]
NOTES: tuple[tuple[str, str], ...] = (
    (
        "全部",
        "补充任务：芒果产区二分类（NT 对 QLD），按采收季留出，不并入主口径 18 个留出域的汇总。"
        "产区与品种、采收批相关（见「品种构成」表），不作为产地化学的证据。",
    ),
    (
        "表a",
        "Δ_rep = S_src − R；Δ_lab = S_src − S_shuf；Δ_half = S_half − R_B（R_B 即 R^h：R 的编码器在同一个 B 半上"
        "的评估）。各臂一栏是宏 F1 的种子均值；种子 t 区间是逐种子配对差的 95% t 区间。"
        "pca 不看标签，没有 S_shuf 与 Δ_lab。",
    ),
    ("表b", "跨季均值 = 各留出季种子均值的平均；为正留出季 = 种子均值为正的留出季数 / 有这一对照的留出季数。"),
    (
        "表c",
        "二分类：准确率的随机水平是 0.5；两类评估样本等量时宏 F1 的随机水平也是 0.5，不等量时略低"
        "（各格两类的评估样本数见长格式表的「评估_NT」「评估_QLD」两列）。",
    ),
)


# ═══════════════════════════════════════════════════════════════════════════════
# 一、数据：采收季为域、产区为类别
# ═══════════════════════════════════════════════════════════════════════════════
class RegionData(NamedTuple):
    """按采收季整理好的产区任务数据。"""

    doms: dict[str, tuple[NDArray, NDArray]]  # 域名 → (未做预处理的光谱, 产区编号)
    points: dict[str, NDArray]  # 域名 → 各行的取样点编号
    counts: dict[tuple[str, str, str], tuple[int, int]]  # (域名, 产区, 品种) → (筛选后行数, 抽到的条数)


def load_region_task(root: str, seed: int, variant: str, cap: int, logger: eu.Logger) -> RegionData:
    """读芒果表，整理成「域 = 采收季、类别 = 产区」的数组。

    读表与取列照 cdl.load_mango：同一个文件，光谱取表头里列名为纯数字的全部列，不足 60 行的季不作为域；
    逐谱预处理（SNV）留到 prepare_fold 里做，次序与 cdl.prepare_fold 相同。与 load_mango 不同的三处：
    类别取 Region 而不是干物质；按版本筛选品种；抽样改为每季每个产区各抽至多 cap 条
    （default_rng([种子, 90, 季, 类])；抽中的行在各产区内保持表内次序，每季先 NT 后 QLD 拼接），
    筛选在前、抽样在后。元数据列按字符串读入（品种一列既有数字编号又有字母名），不影响光谱列的取值。
    """
    path = os.path.join(root, MANGO_CSV)
    sha = eu.compute_sha256(path)
    if sha != MANGO_SHA256:
        raise ValueError(f"芒果数据文件的 SHA-256 与数据卡登记值不符：{path}\n  实际值={sha}")
    logger.log(f"数据文件：{path}（SHA-256 与数据卡登记值一致）")
    df = pd.read_csv(path, dtype=dict.fromkeys(META_STR_COLS, str))
    wl = [c for c in df.columns if c.isdigit()]
    found = sorted(df["Region"].unique().tolist())
    if found != list(REGIONS):
        raise ValueError(f"产区取值 {found} 与预期 {list(REGIONS)} 不符")
    by_region = [set(df.loc[df["Region"] == r, "Cultivar"].tolist()) for r in REGIONS]
    in_both = by_region[0] & by_region[1]
    if in_both != set(SHARED_CULTIVARS):
        raise ValueError(f"两个产区都有的品种是 {sorted(in_both)}，与 SHARED_CULTIVARS 不符")
    df["_point"] = df.groupby(POINT_KEYS, sort=False, dropna=False).ngroup()
    size = np.bincount(df["_point"].to_numpy())
    logger.log(
        f"读入 {len(df)} 行、{len(wl)} 个波长（{wl[0]}–{wl[-1]} nm）；取样点 {len(size)} 个："
        f"两行的 {int((size == 2).sum())} 个、一行的 {int((size == 1).sum())} 个、多于两行的 {int((size > 2).sum())} 个"
    )
    if variant == "shared":
        df = df[df["Cultivar"].isin(SHARED_CULTIVARS)]

    doms: dict[str, tuple[NDArray, NDArray]] = {}
    points: dict[str, NDArray] = {}
    counts: dict[tuple[str, str, str], tuple[int, int]] = {}
    for season in sorted(df["Season"].unique().tolist()):
        sub = df[df["Season"] == season]
        name = f"S{season}"
        if len(sub) < MIN_SEASON_ROWS:
            logger.log(f"  [WARN] {name} 筛选后只有 {len(sub)} 行（不足 {MIN_SEASON_ROWS}），不作为域")
            continue
        xs: list[NDArray] = []
        ys: list[NDArray] = []
        ps: list[NDArray] = []
        for cls, region in enumerate(REGIONS):
            part = sub[sub["Region"] == region]
            n = len(part)
            pos: NDArray = np.arange(n)
            if 0 < cap < n:
                pos = np.sort(np.random.default_rng([seed, 90, int(season), cls]).permutation(n)[:cap])
            cult = part["Cultivar"].to_numpy().astype(str)
            for cv in np.unique(cult):
                counts[(name, region, str(cv))] = (int((cult == cv).sum()), int((cult[pos] == cv).sum()))
            xs.append(part[wl].to_numpy(np.float64)[pos])
            ys.append(np.full(len(pos), cls, dtype=np.int64))
            ps.append(part["_point"].to_numpy()[pos])
        spec = np.vstack(xs)
        if not np.isfinite(spec).all():
            raise ValueError(f"{name}：光谱含缺失值或非有限值")
        doms[name] = (spec, np.concatenate(ys))
        points[name] = np.concatenate(ps)
    if len(doms) < 2:
        raise ValueError(f"可用的采收季只有 {list(doms)}，无法按季留出")
    logger.log(f"版本 {variant}（{VARIANTS[variant]}），每季每个产区至多 {cap or '不限'} 条：")
    for name, (_, y) in doms.items():
        rows_all = [sum(v[0] for (d, r, _), v in counts.items() if d == name and r == reg) for reg in REGIONS]
        used = " / ".join(f"{reg} {int((y == c).sum())}" for c, reg in enumerate(REGIONS))
        logger.log(f"  {name}：{used} 条（筛选后 {rows_all[0]} / {rows_all[1]} 行）")
    return RegionData(doms, points, counts)


def composition_rows(data: RegionData, held: str, base: dict[str, Any]) -> list[dict[str, Any]]:
    """一个留出域的品种构成：留出季与源域里两个产区各品种的筛选后行数与抽到的条数。"""
    out: list[dict[str, Any]] = []
    for scope in ("留出季", "源域"):
        agg: dict[tuple[str, str], tuple[int, int]] = {}
        for (dom, region, cultivar), (n_all, n_used) in data.counts.items():
            if (dom == held) != (scope == "留出季"):
                continue
            prev = agg.get((region, cultivar), (0, 0))
            agg[(region, cultivar)] = (prev[0] + n_all, prev[1] + n_used)
        for region in REGIONS:
            total = sum(v[1] for (r, _), v in agg.items() if r == region)
            for (r, cultivar), (n_all, n_used) in sorted(agg.items()):
                if r != region:
                    continue
                out.append(
                    {
                        **base,
                        "留出季": held,
                        "范围": scope,
                        "产区": region,
                        "品种": cultivar,
                        "筛选后行数": n_all,
                        "样本数": n_used,
                        "占该产区比例": n_used / total if total else float("nan"),
                    }
                )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 二、折内数组、S_half 的训练池与评估
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_fold(data: RegionData, held: str, k: int) -> tuple[cdl.FoldArrays, NDArray, NDArray] | None:
    """一个留出季的折内数组，返回 (折内数组, xall 各行所属的域名, xh 各行的取样点编号)；两个产区缺一时返回 None。

    照 cdl.prepare_fold 与 74 号 fold_arrays 写：原函数在合法类少于 3 个时返回 None，二分类无法直接调用。
    照抄的部分：源域 = 其余各季按域字典的次序拼接；先拼接再逐谱 SNV（cdl._apply_preprocess）；
    源域每类至少 15 条、留出季每类至少 K+2 条才算合法类；xall 按域字典的次序拼接全部域；
    mmd2 取 cdl.mmd2_linear(xs, xh)；行归属的核对同 fold_arrays。
    因二分类而不同：类别本来就是整数编号，不走连续目标的三分位；合法类少于 2 个（两个产区缺一）即不评
    这一折，原函数是少于 3 个；原函数「至多取 6 类」的上限用不上，略去。
    """
    doms = data.doms
    names = list(doms.keys())
    src = [d for d in names if d != held]
    in_dim = doms[names[0]][0].shape[1]
    ys_all = np.concatenate([doms[d][1] for d in src])
    yh_all = doms[held][1]
    cs_src = {int(c) for c in np.unique(ys_all) if int((ys_all == c).sum()) >= MIN_SRC_PER_CLASS}
    cs_held = {int(c) for c in np.unique(yh_all) if int((yh_all == c).sum()) >= k + 2}
    classes = sorted(cs_src & cs_held)
    if len(classes) < MIN_CLASSES:
        return None

    def build(dom_list: list[str]) -> tuple[NDArray, NDArray]:
        feats = cdl._apply_preprocess(np.vstack([doms[d][0] for d in dom_list]), "snv")
        labels = np.concatenate([doms[d][1] for d in dom_list])
        mask = np.isin(labels, classes)
        return feats[mask], labels[mask]

    xs, ys = build(src)
    xall, yall = build(names)
    xh, yh = build([held])
    fa = cdl.FoldArrays(classes, xs, ys, xall, yall, xh, yh, in_dim, cdl.mmd2_linear(xs, xh))
    rows: list[str] = []
    for d in names:
        rows += [d] * int(np.isin(doms[d][1], classes).sum())
    dom_of_row = np.asarray(rows)
    if len(dom_of_row) != len(fa.yall) or not np.array_equal(fa.xall[dom_of_row == held], fa.xh):
        raise RuntimeError(f"留出 {held}：行归属与折内数组不一致")
    return fa, dom_of_row, data.points[held][np.isin(yh_all, classes)]


def half_pool(
    fa: cdl.FoldArrays, dom_of_row: NDArray, held: str, seed: int, points: NDArray, unit: str
) -> tuple[NDArray, NDArray, NDArray]:
    """S_half 的训练池，返回 (训练池 x, 训练池 y, B 半在 xh 中的下标)。

    unit = "row"   直接调用 m74.half_split：留出季各产区的样本按行对半（奇数时 B 多一个），随机流 [种子, 74, 11]。
    unit = "pair"  同一取样点的光谱同进 A 或同进 B：各产区的取样点按种子对半（奇数时 B 多一个），随机流
                   default_rng([种子, 90, 11])；其余照 half_split：训练池 = xall 去掉留出季的 B 半，行序不变。
    """
    if unit == "row":
        xa, ya, b_idx = m74.half_split(fa, dom_of_row, held, seed)
        return xa, ya, b_idx
    rng = np.random.default_rng([seed, 90, 11])
    in_a = np.zeros(len(fa.yh), dtype=bool)
    for c in fa.classes:
        ids = np.unique(points[fa.yh == c])
        perm = rng.permutation(len(ids))
        in_a[(fa.yh == c) & np.isin(points, ids[perm[: len(ids) // 2]])] = True
    held_rows = np.where(dom_of_row == held)[0]
    keep = np.ones(len(fa.yall), dtype=bool)
    keep[held_rows[~in_a]] = False
    return fa.xall[keep], fa.yall[keep], np.where(~in_a)[0]


def eval_cell(
    enc: nn.Module,
    feats: NDArray,
    labels: NDArray,
    classes: list[int],
    device: torch.device,
    k: int,
    reps: int,
    seed: int,
) -> tuple[float, float, list[float], list[float]]:
    """留出域 K-shot 评估，返回 (宏 F1 均值, 准确率均值, 逐情节宏 F1, 逐情节准确率)。

    宏 F1 均值就是 cdl.eval_kshot 的返回值（与 74 号 run_fold 同一调用）。它只返回均值，准确率与逐情节数值
    在 m74.kshot_episodes 给出的同一批情节上另算一遍（抽样与 eval_kshot 逐次相同，原型与预测的算法同
    m74.proto_f1）。两遍的宏 F1 均值须一致（差 ≤ 1e-9，同 74 号路线面板的自检），否则中止。
    """
    f1_ref = cdl.eval_kshot(enc, feats, labels, classes, device, k, reps, seed)
    f1s: list[float] = []
    accs: list[float] = []
    enc.eval()
    with torch.no_grad():
        for sup, q in m74.kshot_episodes(labels, classes, k, reps, seed):
            protos: list[NDArray] = []
            for s in sup:
                feat = enc(torch.from_numpy(feats[s]).float().to(device)).cpu().numpy().mean(0)
                protos.append(feat / (np.linalg.norm(feat) + 1e-8))
            fq = enc(torch.from_numpy(feats[q]).float().to(device)).cpu().numpy()
            pred = np.asarray(classes)[(fq @ np.stack(protos).T).argmax(1)]
            f1s.append(float(f1_score(labels[q], pred, average="macro", labels=classes)))
            accs.append(float((pred == labels[q]).mean()))
    mine = float(np.mean(f1s)) if f1s else float("nan")
    same = (np.isnan(mine) and np.isnan(f1_ref)) or abs(mine - f1_ref) <= 1e-9
    if not same:
        raise RuntimeError(f"逐情节复算的宏 F1 均值 {mine} 与 cdl.eval_kshot 的 {f1_ref} 不一致")
    return f1_ref, float(np.mean(accs)) if accs else float("nan"), f1s, accs


# ═══════════════════════════════════════════════════════════════════════════════
# 三、单个留出季
# ═══════════════════════════════════════════════════════════════════════════════
def run_fold(
    data: RegionData,
    held: str,
    encoder: str,
    args: argparse.Namespace,
    device: torch.device,
    logger: eu.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """单个留出季：各臂训练 + 全部样本与 B 半上的评估，返回 (逐臂记录, 逐情节记录)；两个产区缺一时返回空表。

    照 74 号 run_fold 重写：原函数经 cdl.prepare_fold 准备数组，合法类少于 3 个即跳过，二分类无法直接调用。
    照抄的部分（都是直接调用现有函数，随机流不变）：
        训练池  R = (xs, ys)；S_src = (xall, yall)；S_shuf = m74.shuffled_pool(...)；
                S_half = m74.half_split(...)（--half_unit row 时）；pca 不跑 S_shuf
        训练    各臂按 R、S_src、S_shuf、S_half 的次序调用 m74.obtain(编码器, x, y, in_dim, args, device, 种子)：
                深度族走 cdl.meta_train（进入时按种子重新播种，各臂初始化相同），
                线性族走 enc_lib.fit_linear_encoder（成分数上限 args.lin_dim）
        评估    R、S_src、S_shuf 在留出季全部样本上，S_half、R_B、S_src_B 在 B 半上；宏 F1 取
                cdl.eval_kshot(编码器, x, y, 类别, device, K, 情节数, 种子) 的返回值，同一批样本上各臂情节相同
    因二分类而不同：
        B 半三臂的条件  原为「B 半里样本数 ≥ K+2 的类不少于 3 个」，这里是两个产区在 B 半里都不少于 K+2 条
        元训练的情节    cdl.meta_train 用 ArrSampler(feats, labels, 5, 10)，类别数不超过 6 时每个情节用全部
                        类别：这里每个情节 2 类、每类 5 条支持 + 10 条查询，共 30 条（主口径各数据集是
                        3–6 类，每类条数相同）
        评估的情节      每类 K 条支持，留出季（或 B 半）其余样本全部作查询；某类不足 K+2 条时该情节作废
                        （规则未改，缺省设定下不会触发）；宏 F1 对两类取平均
        pls            类别指示矩阵只有互补的两列，成分数上限仍取 args.lin_dim
    另加的部分（不改变上面的数值）：
        准确率、有效情节数与逐情节数值（eval_cell）；B 半样本里同一取样点的另一条谱落在 A 半的占比；
        --half_unit pair 时 S_half 按取样点对半；没有路线面板
    """
    seed = int(args.seed)
    got = prepare_fold(data, held, args.k)
    if got is None:
        logger.log(f"    留出 {held}：有产区样本不足（源域每类 ≥ {MIN_SRC_PER_CLASS}、留出季每类 ≥ K+2），此折不评")
        return [], []
    fa, dom_of_row, points_h = got
    arms = [a for a in ARMS if not (encoder == "pca" and a == "S_shuf")]
    pools: dict[str, tuple[NDArray, NDArray]] = {"R": (fa.xs, fa.ys), "S_src": (fa.xall, fa.yall)}
    if "S_shuf" in arms:
        pools["S_shuf"] = m74.shuffled_pool(fa, dom_of_row, held, seed)
    xa, ya, b_idx = half_pool(fa, dom_of_row, held, seed, points_h, args.half_unit)
    pools["S_half"] = (xa, ya)
    xb, yb = fa.xh[b_idx], fa.yh[b_idx]
    classes_b = [c for c in fa.classes if int((yb == c).sum()) >= args.k + 2]
    in_b = np.zeros(len(fa.yh), dtype=bool)
    in_b[b_idx] = True
    mate_share = float(np.isin(points_h[in_b], points_h[~in_b]).mean())
    if args.half_unit == "pair" and mate_share > 0:
        raise RuntimeError(f"留出 {held}：按取样点对半后仍有取样点同时落在 A、B 两半")

    encs: dict[str, nn.Module] = {}
    for arm in arms:
        x, y = pools[arm]
        encs[arm] = m74.obtain(encoder, x, y, fa.in_dim, args, device, seed)
    base: dict[str, Any] = {
        "版本": args.variant,
        "留出季": held,
        "编码器": encoder,
        "种子": seed,
        "K": args.k,
    }
    tail: dict[str, Any] = {
        **{f"留出季_{reg}": int((fa.yh == c).sum()) for c, reg in enumerate(REGIONS)},
        **{f"源域_{reg}": int((fa.ys == c).sum()) for c, reg in enumerate(REGIONS)},
        "mmd2": fa.mmd2,
        "B半同点谱在A半占比": mate_share,
        "抽样上限": args.cap,
        "S_half划分单位": args.half_unit,
        "元训练情节": args.episodes,
        "评估情节设定": args.reps,
        "成分数上限": args.lin_dim,
        "标签": args.tag,
    }
    rows: list[dict[str, Any]] = []
    ep_rows: list[dict[str, Any]] = []

    def record(arm: str, src_arm: str, scope: str, x_ev: NDArray, y_ev: NDArray, classes: list[int]) -> None:
        f1, acc, f1s, accs = eval_cell(encs[src_arm], x_ev, y_ev, classes, device, args.k, args.reps, seed)
        head = {**base, "臂": arm, "评估样本": scope}
        rows.append(
            {
                **head,
                "宏F1": f1,
                "准确率": acc,
                "情节数": len(f1s),
                "宏F1情节SD": float(np.std(f1s)) if f1s else float("nan"),
                "n_eval": len(y_ev),
                **{f"评估_{reg}": int((y_ev == c).sum()) for c, reg in enumerate(REGIONS)},
                "n_train": len(pools[src_arm][1]),
                **tail,
            }
        )
        pairs = enumerate(zip(f1s, accs, strict=True))
        ep_rows.extend({**head, "情节": i, "宏F1": f1_i, "准确率": acc_i} for i, (f1_i, acc_i) in pairs)

    for arm in [a for a in arms if a != "S_half"]:
        record(arm, arm, "全部", fa.xh, fa.yh, fa.classes)
    if len(classes_b) >= MIN_CLASSES:
        keep = np.isin(yb, classes_b)
        for arm, src_arm in (("S_half", "S_half"), *HALF_ARMS.items()):
            record(arm, src_arm, "B半", xb[keep], yb[keep], classes_b)
    else:
        logger.log(f"    留出 {held}：B 半里有产区不足 K+2 条，不评 B 半三臂")
    size = " / ".join(f"{reg} {tail[f'留出季_{reg}']}" for reg in REGIONS)
    msg = "  ".join(f"{r['臂']}={r['宏F1']:.4f}" for r in rows)
    logger.log(f"    留出 {held}（{size}；B 半同点谱在 A 半 {mate_share:.0%}）：{msg}")
    return rows, ep_rows


# ═══════════════════════════════════════════════════════════════════════════════
# 四、结果文件与汇总
# ═══════════════════════════════════════════════════════════════════════════════
def seed_file(variant: str, encoder: str, tag: str, seed: int, stage: str) -> str:
    """逐种子结果表的路径：90_mango_region_task__<版本>__<编码器>[__<tag>]__seed<S>[__pilot].xlsx。"""
    seg = f"__{tag}" if tag else ""
    suffix = "" if stage == "formal" else "__pilot"
    return os.path.join(SEED_DIR, f"{STEM}__{variant}__{encoder}{seg}__seed{seed}{suffix}.xlsx")


def seed_files(stage: str, tag: str) -> list[str]:
    """给定阶段与标签的全部逐种子结果表；文件名逐段匹配，带别的标签或别的阶段的文件不会混进来。"""
    seg = re.escape(f"__{tag}" if tag else "")
    suffix = "" if stage == "formal" else "__pilot"
    pat = re.compile(rf"{STEM}__({'|'.join(VARIANTS)})__({'|'.join(enc_lib.ENCODERS)}){seg}__seed\d+{suffix}\.xlsx")
    names = sorted(os.listdir(SEED_DIR)) if os.path.isdir(SEED_DIR) else []
    return [os.path.join(SEED_DIR, n) for n in names if pat.fullmatch(n)]


def arm_seed_values(sub: pd.DataFrame, col: str) -> dict[tuple[str, int], float]:
    """一个（版本, 编码器, 留出季）格里 (臂, 种子) → 数值；同一格出现两次即报错。"""
    out: dict[tuple[str, int], float] = {}
    for arm, seed, val in zip(sub["臂"], sub["种子"], sub[col], strict=True):
        key = (str(arm), int(seed))
        if key in out:
            raise ValueError(f"重复格：{key}")
        out[key] = float(val)
    return out


def aggregate(stage: str, tag: str, logger: eu.Logger) -> None:
    """跨种子汇总一个（阶段, 标签）下的全部逐种子结果表：逐留出季对照、跨季汇总、R 臂的绝对水平。"""
    files = seed_files(stage, tag)
    if not files:
        raise FileNotFoundError(f"没有 90 号的逐种子结果（阶段 {stage}，标签 {tag or '无'}）")
    long_df = pd.concat([pd.read_excel(f, sheet_name=SHEET_ARMS) for f in files], ignore_index=True)
    comp_df = pd.concat([pd.read_excel(f, sheet_name=SHEET_COMP) for f in files], ignore_index=True)
    cost_parts: list[Any] = []
    for f in files:
        part = pd.read_excel(f, sheet_name="compute")
        part.insert(0, "文件", os.path.basename(f))
        cost_parts.append(part)
    cost_df = pd.concat(cost_parts, ignore_index=True)
    settings = long_df[SETTING_COLS].drop_duplicates()
    if len(settings) != 1:
        raise ValueError(
            "待汇总的逐种子结果表设定不一致（换了设定的运行用 --tag 分开）：\n" + settings.to_string(index=False)
        )
    seeds_all = sorted({int(s) for s in long_df["种子"].tolist()})
    if stage == "formal" and set(seeds_all) != set(eu.RECOMMENDED_SEEDS):
        logger.log(f"  [WARN] 种子集合 {seeds_all} 不是 Formal 5 种子，汇总仅作进度查看")
    logger.log(f"读入 {len(files)} 个逐种子结果表（阶段 {stage}，标签 {tag or '无'}），种子 {seeds_all}")

    fold_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for enc in enc_lib.ENCODERS:
            cell = long_df[(long_df["版本"] == variant) & (long_df["编码器"] == enc)]
            if cell.empty:
                continue
            deltas: dict[str, list[float]] = {name: [] for name in CONTRASTS}
            level: list[tuple[float, float]] = []
            for held in sorted({str(h) for h in cell["留出季"].tolist()}):
                sub = cell[cell["留出季"] == held]
                f1 = arm_seed_values(sub, "宏F1")
                acc = arm_seed_values(sub, "准确率")
                seeds = sorted({s for _, s in f1})
                row: dict[str, Any] = {"版本": variant, "留出季": held, "编码器": enc, "种子数": len(seeds)}
                for arm in EVAL_ARMS:
                    vals = [f1[(arm, s)] for s in seeds if np.isfinite(f1.get((arm, s), np.nan))]
                    if vals:
                        row[arm] = float(np.mean(vals))
                for name, (a, b) in CONTRASTS.items():
                    dl = [f1[(a, s)] - f1[(b, s)] for s in seeds if (a, s) in f1 and (b, s) in f1]
                    dl = [d for d in dl if np.isfinite(d)]
                    if not dl:
                        continue
                    lo, hi = m74._t_interval(dl)
                    row[name] = float(np.mean(dl))
                    row[f"{name}_种子t区间"] = f"{lo:.4f}–{hi:.4f}"
                    row[f"{name}_为正种子数"] = sum(d > 0 for d in dl)
                    row[f"{name}_配对种子数"] = len(dl)
                    deltas[name].append(float(np.mean(dl)))
                fold_rows.append(row)
                r_f1 = [f1[("R", s)] for s in seeds if np.isfinite(f1.get(("R", s), np.nan))]
                r_acc = [acc[("R", s)] for s in seeds if np.isfinite(acc.get(("R", s), np.nan))]
                if r_f1 and r_acc:
                    level.append((float(np.mean(r_f1)), float(np.mean(r_acc))))
                    level_rows.append(
                        {
                            "版本": variant,
                            "编码器": enc,
                            "留出季": held,
                            "种子数": len(r_f1),
                            "R_宏F1": level[-1][0],
                            "R_宏F1_种子SD": float(np.std(r_f1, ddof=1)) if len(r_f1) > 1 else float("nan"),
                            "R_准确率": level[-1][1],
                        }
                    )
            n_seeds = int(cell["种子"].nunique())
            crow: dict[str, Any] = {
                "版本": variant,
                "编码器": enc,
                "留出季数": int(cell["留出季"].nunique()),
                "种子数": n_seeds,
            }
            for name, means in deltas.items():
                if means:
                    crow[f"{name}_跨季均值"] = float(np.mean(means))
                    crow[f"{name}_为正留出季"] = f"{sum(v > 0 for v in means)}/{len(means)}"
            cross_rows.append(crow)
            if level:
                level_rows.append(
                    {
                        "版本": variant,
                        "编码器": enc,
                        "留出季": "各季均值",
                        "种子数": n_seeds,
                        "R_宏F1": float(np.mean([v[0] for v in level])),
                        "R_宏F1_种子SD": float("nan"),
                        "R_准确率": float(np.mean([v[1] for v in level])),
                    }
                )
    fold_df, cross_df, level_df = pd.DataFrame(fold_rows), pd.DataFrame(cross_rows), pd.DataFrame(level_rows)
    logger.log("\n跨季汇总\n" + cross_df.to_string(index=False))
    logger.log("\nR 臂的绝对水平（二分类，随机水平 0.5）\n" + level_df.to_string(index=False))

    # 品种构成与编码器无关，同一（版本, 种子）在各编码器的文件里重复出现，去重后对种子取均值
    comp_keys = ["版本", "留出季", "范围", "产区", "品种"]
    comp_df["品种"] = comp_df["品种"].astype(str)
    comp_df = comp_df.drop_duplicates(subset=[*comp_keys, "种子"])
    comp_rows: list[dict[str, Any]] = []
    for vals_key, sub in comp_df.groupby(comp_keys):
        comp_rows.append(
            {
                **dict(zip(comp_keys, vals_key, strict=True)),
                "筛选后行数": int(sub["筛选后行数"].iloc[0]),
                "样本数_种子均值": round(float(sub["样本数"].mean()), 1),
                "占该产区比例_种子均值": round(float(sub["占该产区比例"].mean()), 4),
                "种子数": int(sub["种子"].nunique()),
            }
        )
    notes = [{"表": name, "说明": text} for name, text in NOTES]
    notes.append({"表": "设定", "说明": "；".join(f"{c} = {settings.iloc[0][c]}" for c in SETTING_COLS)})
    if stage != "formal":
        notes.append({"表": "阶段", "说明": "预跑结果（减小的预算），只用于检查流程，不进稿件。"})

    seg = f"__{tag}" if tag else ""
    suffix = "" if stage == "formal" else "__pilot"
    out = os.path.join(eu.OUTPUT_DIR, f"{STEM}{seg}{suffix}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        fold_df.to_excel(writer, sheet_name=eu.table_sheet_name(0, "逐留出季对照"), index=False)
        cross_df.to_excel(writer, sheet_name=eu.table_sheet_name(1, "跨季汇总"), index=False)
        level_df.to_excel(writer, sheet_name=eu.table_sheet_name(2, "R臂绝对水平"), index=False)
        pd.DataFrame(comp_rows).to_excel(writer, sheet_name=eu.table_sheet_name(3, "品种构成"), index=False)
        long_df.to_excel(writer, sheet_name=eu.table_sheet_name(4, "长格式"), index=False)
        pd.DataFrame(notes).to_excel(writer, sheet_name="表注", index=False)
        cost_df.to_excel(writer, sheet_name="compute", index=False)
    logger.log(f"汇总表 → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 五、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="90_mango_region_task：芒果产区二分类上的各暴露臂（补充任务）")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--stage", default=None, choices=["pilot", "formal"], help="缺省：运行时 pilot，--aggregate 时 formal"
    )
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED, help="一次一个种子")
    parser.add_argument(
        "--variant", default="all", choices=list(VARIANTS), help="all = 全部品种；shared = 两个产区共有的四个品种"
    )
    parser.add_argument("--encoders", default=",".join(enc_lib.ENCODERS), help="逗号分隔的编码器族，缺省全部六族")
    parser.add_argument("--data_root", default="05data")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="每季每个产区至多抽取的条数；0 = 不设上限")
    parser.add_argument(
        "--half_unit",
        default="row",
        choices=["row", "pair"],
        help="S_half 的划分单位：row = 按行（同 74 号），pair = 按取样点",
    )
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
    if args.cap < 0 or min(args.k, args.episodes, args.reps, args.lin_dim) < 1:
        parser.error("--cap 不能为负；--k、--episodes、--reps、--lin_dim 须为正整数")
    defaults: dict[str, Any] = {
        "cap": DEFAULT_CAP,
        "half_unit": "row",
        "k": DEFAULT_K,
        "lin_dim": DEFAULT_LIN_DIM,
        **stage_default,
    }
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
            "任务": "芒果产区二分类（NT 对 QLD）按采收季留出的各暴露臂（补充任务，不并入主口径）",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "版本": f"{args.variant}（{VARIANTS[args.variant]}）",
            "编码器": [f"{e}（{enc_lib.ARCH_DESC[e]}）" for e in enc_list],
            "臂": [*ARMS, *HALF_ARMS],
            "每季每个产区的抽样上限": args.cap or "不设",
            "K / 元训练情节 / 每格评估情节": f"{args.k} / {args.episodes} / {args.reps}",
            "线性族成分数上限": args.lin_dim,
            "S_half 划分单位": args.half_unit,
            "非缺省设定 / 标签": f"{changed or '无'} / {args.tag or '无'}",
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))
    data = load_region_task(args.data_root, args.seed, args.variant, args.cap, logger)
    comp_base = {"版本": args.variant, "种子": args.seed}
    comp_rows = [r for held in data.doms for r in composition_rows(data, held, comp_base)]
    os.makedirs(SEED_DIR, exist_ok=True)
    for encoder in enc_list:
        logger.log(f"\n【{encoder}】{enc_lib.ARCH_DESC[encoder]}")
        rows: list[dict[str, Any]] = []
        ep_rows: list[dict[str, Any]] = []
        with eu.track_compute(f"芒果产区任务 {args.variant} / {encoder}", logger, run) as budget:
            for held in data.doms:
                fold_rows, fold_eps = run_fold(data, held, encoder, args, device, logger)
                rows += fold_rows
                ep_rows += fold_eps
        if not rows:
            raise SystemExit("没有可评的留出季：检查 --cap 与 --k")
        out = seed_file(args.variant, encoder, args.tag, args.seed, args.stage)
        cost = {"seed": args.seed, "stage": args.stage, "variant": args.variant, "encoder": encoder, **budget}
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name=SHEET_ARMS, index=False)
            pd.DataFrame(comp_rows).to_excel(writer, sheet_name=SHEET_COMP, index=False)
            pd.DataFrame(ep_rows).to_excel(writer, sheet_name=SHEET_EPIS, index=False)
            pd.DataFrame([cost]).to_excel(writer, sheet_name="compute", index=False)
        logger.log(f"结果表 → {out}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
