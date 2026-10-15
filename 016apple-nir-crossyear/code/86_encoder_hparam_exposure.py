"""86_encoder_hparam_exposure.py: 编码器超参数的暴露（按留出域验证得分选编码器配置的抬高量，补充材料 S9.4）。

为什么:
    稿件里超参数级的暴露只测了基线：SVM 与 PLS-DA 的超参数在留出域情节上选（苹果 76 号，
    公开数据集 74 号路线面板）。编码器自身的超参数没有暴露臂：深度族的学习率、线性族的成分数
    若按留出域带标签样本上的得分来定，结果抬高多少，没有测过。
    本脚本每个配置都只在源域上训练或拟合，编码器权重不接触留出域；留出域只经由
    「选哪个配置」进入。

设计（一格 = 一个留出域 × 一个编码器族 × 一个种子）:
    配置网格:
        深度族（cnn / mlp / resnet / transformer）
            学习率取缺省值的 1/3、1、3 倍。缺省值在苹果与公开数据集的正式配方里都是 1e-3
            （64 号 --lr、crossdataset_lib.meta_train），四族相同；其余设定同正式配方。
            每个配置训一条仅源域的轨迹到满预算（苹果 2000、公开数据集 1000 情节），取末次权重
            （85 号 train_trajectory，不带快照）。
        线性族（pls / pca）
            成分数取 4、16、32（缺省 16，即 64、74 号的 --lin_dim），在源域训练池上由
            encoders.fit_linear_encoder 直接拟合，不训练。训练池的取法同 64 号 fit_linear
            （苹果：全部光谱，每果每面位一条）与 74 号 obtain（公开数据集：源域全部样本）。
    验证得分
        每个配置在三个验证池上各 30 个验证情节的查询准确率均值：src 源域、tgt 留出域全体、
        tgt_A 留出域 A 半。验证池、情节构成与独立随机数流都是 85 号的；同一格内各配置用
        同一批验证情节（即 85 号末次快照所用的那一批）。
    评估
        K=5、每格 50 次支持集重采样的宏 F1，每个配置都在留出域全体与 B 半上各评一次
        （85 号的评估函数与 A/B 划分）；同一格内各配置用同一批评估情节。
    选法
        源域选    源域验证得分最高的配置
        留出域选  留出域验证得分最高的配置
        A半选     留出域 A 半验证得分最高的配置
        并列时缺省配置优先，其次取网格里靠前的。
    对照（种子内配对作差）:
        Δ_hp      = 留出域选的配置 − 源域选的配置，在留出域全体上评估
        Δ_hp_half = A半选的配置 − 源域选的配置，在 B 半上评估

与 85 号的关系:
    训练函数、验证得分、评估、A/B 划分与随机数流都直接调用 85 号，所以验证不消耗训练所用的
    全局随机数这一点同样成立。深度族的缺省配置（学习率 × 1）与 85 号的 last 是同一条轨迹的
    同一份权重，验证情节与评估情节也相同：它的源域、留出域验证得分与留出域全体上的宏 F1
    应与 85 号逐位相同。--aggregate 时逐格核对，结果写进汇总表（表e），不相同的格在日志里列出。

范围:
    编码器族  cnn / mlp / resnet / transformer / pls / pca
    留出域    同 85 号：苹果两年设计 2 个、三年设计 3 个留出年份，公开数据集 16 个留出域
    主口径    苹果两年设计 + 公开数据集 = 18 个留出域；苹果三年设计单列

输入:
    同 85 号（苹果冻结数据与数据卡；--data_root 下的五个公开数据集）
    --aggregate 的核对另读 04outputs/formal_seeds/85_checkpoint_paired__*.xlsx（可以没有）

复用:
    85_checkpoint_paired  build_cells 与各格的 evaluate、linear_xy；train_trajectory、val_score、
                          stage_budget、last_index、check_recipe；命令行与范围解析；
                          结果文件的读写；domain_table、family_table
    encoders              fit_linear_encoder、ENCODERS、DEEP_ENCODERS、LINEAR_ENCODERS
    export_utils          设备、播种、Formal 的 CUDA 闸、日志、算力记录

运行方式（从项目根运行；一次一个种子；按范围与编码器分文件，可并行）:
    cd <项目根>
    # pilot：情节预算与评估重采样次数各取正式值的 1/5
    python 02code/86_encoder_hparam_exposure.py --device cuda --stage pilot --seed 2004 \
        --data_root <公开数据集根目录>
    # formal（要求 CUDA）
    python 02code/86_encoder_hparam_exposure.py --device cuda --stage formal --seed 20060515 \
        --scope public --encoders resnet --data_root <公开数据集根目录>
    python 02code/86_encoder_hparam_exposure.py --device cuda --stage formal --seed 20060515 \
        --scope apple --encoders pls,pca
    # 汇总（缺省读正式结果；加 --stage pilot 读 pilot 结果）
    python 02code/86_encoder_hparam_exposure.py --aggregate

输出文件:
    04outputs/formal_seeds/86_encoder_hparam_exposure__<范围>__<编码器>__seed<S>[__pilot].xlsx
        表a 逐配置结果：一行 = 一格 × 一个配置，含超参数取值、三个验证得分、留出域全体与 B 半上的
            宏 F1 与评估情节数，以及三种选法各自是否选中该配置
        表b 逐格选择与Δ：一行一格，三种选法选中的配置、对照两侧的宏 F1、Δ_hp 与 Δ_hp_half
        compute
        <范围> 的写法同 85 号
    04outputs/86_encoder_hparam_exposure.xlsx（--aggregate；pilot 时文件名加 __pilot）
        表a 逐留出域（Δ 的种子均值、种子 t 区间、为正种子数）、表b 按族汇总（主口径与苹果三年设计）、
        表c 逐格、表d 逐配置、表e 与 85 号末次检查点的核对、compute
    05logs/86_encoder_hparam_exposure_YY-MM-DD_HHMMSS.log
    不写权重文件；不读写 64、72、74 号的结果表与 07checkpoints 下的权重。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import encoders as enc_lib
import export_utils as eu

m85 = importlib.import_module("85_checkpoint_paired")

STEM = "86_encoder_hparam_exposure"
# 深度族：学习率相对缺省值的倍数；线性族：成分数（缺省 16，即 64、74 号 --lin_dim 的缺省值）
LR_GRID: tuple[tuple[str, float], ...] = (("lr×1/3", 1.0 / 3.0), ("lr×1", 1.0), ("lr×3", 3.0))
N_COMP_GRID = (4, 16, 32)
N_COMP_DEFAULT = 16
# 选法 → 按哪个验证池的得分选配置
SELECT: dict[str, str] = {"源域选": m85.POOL_SRC, "留出域选": m85.POOL_TGT, "A半选": m85.POOL_TGT_A}
# 对照 → (被减数的选法, 减数的选法, 在哪部分留出域样本上评估)
CONTRASTS: dict[str, tuple[str, str, str]] = {
    "Δ_hp": ("留出域选", "源域选", m85.EVAL_ALL),
    "Δ_hp_half": ("A半选", "源域选", m85.EVAL_B),
}
# 汇总表里取种子均值的列：各对照两侧的宏 F1
MEAN_COLS = [f"F1_{subset}_{sel}" for hi, lo, subset in CONTRASTS.values() for sel in (lo, hi)]
KEYS: list[str] = list(m85.KEYS)  # 一格的键
CHECK_TOL = 1e-9  # 与 85 号末次检查点核对的容差（同一台机器上同一份权重，应为 0）


@dataclass(frozen=True)
class Config:
    """一个编码器配置：只有一个超参数可以与缺省配置不同。"""

    label: str
    param: str
    value: float
    default: bool


def config_grid(family: str, lr_default: float) -> list[Config]:
    """该族的配置网格：深度族三个学习率，线性族三个成分数。"""
    if family in enc_lib.LINEAR_ENCODERS:
        return [Config(f"n_comp={n}", "n_comp", float(n), n == N_COMP_DEFAULT) for n in N_COMP_GRID]
    return [Config(label, "lr", lr_default * mult, mult == 1.0) for label, mult in LR_GRID]


# ═══════════════════════════════════════════════════════════════════════════════
# 一、逐配置的编码器、验证得分与评估
# ═══════════════════════════════════════════════════════════════════════════════
def fit_config(
    cell: Any, family: str, cfg: Config, episodes: int, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """按一个配置得到仅源域的编码器（eval 模式）与随配置记下的几项。

    深度族：85 号 train_trajectory 训满预算、取末次权重，不带快照与验证。
    线性族：在源域训练池上直接拟合（pca 不看类别），调用同 64 号 fit_linear、74 号 obtain。
    """
    if family in enc_lib.LINEAR_ENCODERS:
        x, y = cell.linear_xy()
        fitted, n_eff = enc_lib.fit_linear_encoder(family, x, y if family == "pls" else None, int(cfg.value))
        return fitted.to(device).eval(), {"训练情节数": 0, "实际成分数": n_eff}
    encoder: nn.Module = m85.train_trajectory(cell, family, cfg.value, episodes, device)
    finite = all(bool(torch.isfinite(p).all()) for p in encoder.parameters())
    return encoder, {"训练情节数": episodes, "权重有限": finite}


def pick_config(results: list[dict[str, Any]], col: str) -> int | None:
    """col 得分最高的配置在 results 里的位置；没有有限得分时返回 None。

    并列时缺省配置优先，其次取网格里靠前的。
    """
    order = sorted(range(len(results)), key=lambda i: (not results[i]["缺省配置"], i))
    best: int | None = None
    for i in order:
        score = float(results[i][col])
        if np.isfinite(score) and (best is None or score > float(results[best][col])):
            best = i
    return best


def run_cell(
    cell: Any, family: str, stage: str, device: torch.device, logger: eu.Logger
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """一格：逐配置得到仅源域的编码器，算验证得分与评估成绩，再按各选法取配置。

    返回 (逐配置记录, 本格的选择与 Δ)。
    """
    episodes, reps = m85.stage_budget(cell.recipe, stage)
    idx = m85.last_index(episodes)  # 验证情节与 85 号末次快照所用的是同一批
    base: dict[str, Any] = {
        "范围": cell.scope,
        "dataset": cell.dataset,
        "held": cell.held,
        "编码器": family,
        "种子": cell.seed,
    }
    results: list[dict[str, Any]] = []
    for cfg in config_grid(family, float(cell.recipe.lr)):
        encoder, extra = fit_config(cell, family, cfg, episodes, device)
        row: dict[str, Any] = {
            **base,
            "配置": cfg.label,
            "超参数": cfg.param,
            "取值": cfg.value,
            "缺省配置": cfg.default,
            **extra,
        }
        for pool in m85.POOLS:
            row[m85.VAL_COL[pool]] = float(m85.val_score(encoder, cell, pool, idx, device))
        for subset in (m85.EVAL_ALL, m85.EVAL_B):
            got = cell.evaluate(encoder, subset, reps, device)
            row[f"F1_{subset}"] = float("nan") if got is None else float(got[0])
            row[f"情节数_{subset}"] = 0 if got is None else int(got[1])
        results.append(row)

    picks = {name: pick_config(results, m85.VAL_COL[pool]) for name, pool in SELECT.items()}
    for name, chosen in picks.items():
        for i, row in enumerate(results):
            row[f"{name}中"] = i == chosen
    summary: dict[str, Any] = {**base, "配置数": len(results)}
    for name, chosen in picks.items():
        summary[f"{name}_配置"] = None if chosen is None else results[chosen]["配置"]
    for delta, (hi, lo, subset) in CONTRASTS.items():
        i_hi, i_lo = picks[hi], picks[lo]
        if i_hi is None or i_lo is None:
            continue
        f_hi = float(results[i_hi][f"F1_{subset}"])
        f_lo = float(results[i_lo][f"F1_{subset}"])
        if not (np.isfinite(f_hi) and np.isfinite(f_lo)):
            continue
        summary[f"F1_{subset}_{hi}"] = f_hi
        summary[f"F1_{subset}_{lo}"] = f_lo
        summary[delta] = f_hi - f_lo
        summary[f"{delta}_同选"] = i_hi == i_lo

    src_col, tgt_col = m85.VAL_COL[m85.POOL_SRC], m85.VAL_COL[m85.POOL_TGT]
    f1_col = f"F1_{m85.EVAL_ALL}"
    shown = "  ".join(f"{r['配置']}[{r[src_col]:.3f}/{r[tgt_col]:.3f}]={r[f1_col]:.4f}" for r in results)
    logger.log(f"    [{family}] {cell.dataset} 留出 {cell.held}（配置[源域/留出域验证]=宏F1）：{shown}")
    picked = "，".join(f"{name}={summary[name + '_配置']}" for name in SELECT)
    deltas = "，".join(f"{d}={summary[d]:+.4f}" for d in CONTRASTS if d in summary)
    logger.log(f"      选中：{picked}；{deltas}")
    return results, summary


# ═══════════════════════════════════════════════════════════════════════════════
# 二、汇总与核对
# ═══════════════════════════════════════════════════════════════════════════════
def check_against_85(configs: pd.DataFrame, stage: str, logger: eu.Logger) -> pd.DataFrame:
    """深度族缺省配置与 85 号 last 的逐格核对：同一条轨迹的同一份权重，三个量应逐位相同。

    返回一行一格的绝对差与是否一致；没有 85 号的结果或没有重合的格时返回空表。
    """
    try:
        ref = m85.load_seed_sheet(m85.STEM, stage, 0)
    except FileNotFoundError:
        logger.log("  [核对] 没有 85 号的逐种子结果，未与 85 号核对")
        return pd.DataFrame()
    ref = ref[ref["检查点选法"] == "last"]
    default_deep = configs["缺省配置"].astype(bool) & configs["编码器"].isin(enc_lib.DEEP_ENCODERS)
    both = configs[default_deep].merge(ref, on=KEYS, suffixes=("", "_85"))
    if both.empty:
        logger.log("  [核对] 本脚本与 85 号没有重合的格，未核对")
        return pd.DataFrame()
    out = both[KEYS].copy()
    out["绝对差_F1"] = (both[f"F1_{m85.EVAL_ALL}"] - both["F1"]).abs()
    for pool in (m85.POOL_SRC, m85.POOL_TGT):
        col = m85.VAL_COL[pool]
        out[f"绝对差_{col}"] = (both[col] - both[f"{col}_85"]).abs()
    worst = out[[c for c in out.columns if c.startswith("绝对差_")]].max(axis=1)
    out["一致"] = worst <= CHECK_TOL
    bad = out[~out["一致"]]
    logger.log(f"  [核对] 与 85 号 last 比对 {len(out)} 格，最大绝对差 {float(worst.max()):.2e}")
    if len(bad):
        logger.log(f"  [WARN] {len(bad)} 格与 85 号不一致（容差 {CHECK_TOL:g}），前 30 格：")
        logger.log(bad.head(30).to_string(index=False))
    return out


def aggregate(stage: str, logger: eu.Logger) -> None:
    """汇总逐种子结果：逐留出域的 Δ_hp 与 Δ_hp_half、按族的跨留出域汇总，并与 85 号核对。"""
    configs = m85.load_seed_sheet(STEM, stage, 0)
    cells = m85.load_seed_sheet(STEM, stage, 1)
    m85.check_unique(configs, [*KEYS, "配置"], "86 号逐配置结果")
    m85.check_unique(cells, KEYS, "86 号逐格结果")
    m85.warn_seeds(cells, stage, logger)
    dom = m85.domain_table(cells, list(CONTRASTS), MEAN_COLS)
    fam = m85.family_table(dom, list(CONTRASTS), logger)
    logger.log("按口径 × 编码器族\n" + fam.to_string(index=False))
    check = check_against_85(configs, stage, logger)
    out = m85.summary_path(STEM, stage)
    tables = [
        ("逐留出域", dom),
        ("按族汇总", fam),
        ("逐格", cells),
        ("逐配置", configs),
        ("与85号末次检查点核对", check),
    ]
    m85.write_workbook(out, tables, m85.load_seed_sheet(STEM, stage, "compute"))
    logger.log(f"汇总表 → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 三、主流程
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = m85.make_parser("86_encoder_hparam_exposure：编码器超参数的暴露", enc_lib.ENCODERS)
    args = parser.parse_args()
    args.stage = args.stage or ("formal" if args.aggregate else "pilot")
    logger = eu.get_logger(STEM)
    if args.aggregate:
        aggregate(args.stage, logger)
        logger.close()
        return

    families = m85.parse_subset(args.encoders, enc_lib.ENCODERS, "编码器")
    plan = m85.scope_plan(args.scope, args.designs, args.datasets)
    m85.check_recipe()
    device = eu.get_device(args.device)
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    ep_apple, reps = m85.stage_budget(m85.APPLE_RECIPE, args.stage)
    ep_public, _ = m85.stage_budget(m85.PUBLIC_RECIPE, args.stage)
    eu.log_experiment_header(
        logger,
        {
            "任务": "编码器超参数的暴露",
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "范围": [f"{tag}：{chosen}" for _, tag, chosen in plan],
            "编码器": families,
            "深度族学习率网格（相对缺省值）": [label for label, _ in LR_GRID],
            "线性族成分数网格": list(N_COMP_GRID),
            "训练情节（苹果 / 公开数据集）": f"{ep_apple} / {ep_public}",
            "每池验证情节": m85.VAL_EPISODES,
            "K / 每格评估重采样": f"{m85.K_SHOT} / {reps}",
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))
    for scope, tag, chosen in plan:
        cells = m85.build_cells(scope, chosen, args.data_root, args.seed, logger)
        for family in families:
            logger.log(f"\n{'=' * 60}\n【{tag} × {family}】{enc_lib.ARCH_DESC[family]}")
            config_rows: list[dict[str, Any]] = []
            cell_rows: list[dict[str, Any]] = []
            with eu.track_compute(f"编码器超参数暴露 {tag} × {family}", logger, run) as budget:
                for cell in cells:
                    results, summary = run_cell(cell, family, args.stage, device, logger)
                    config_rows += results
                    cell_rows.append(summary)
            out = m85.seed_path(STEM, tag, family, args.seed, args.stage)
            compute = pd.DataFrame(
                [{"seed": args.seed, "stage": args.stage, "scope": tag, "encoder": family, **budget}]
            )
            tables = [("逐配置结果", pd.DataFrame(config_rows)), ("逐格选择与Δ", pd.DataFrame(cell_rows))]
            m85.write_workbook(out, tables, compute)
            logger.log(f"结果表 → {out}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
