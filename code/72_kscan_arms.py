"""
72_kscan_arms.py: 协议对齐的编码器臂在各 K 下的单面评估，与只在源年份上选参的 SVM、PLS-DA 逐情节配对。

为什么：
    方法比较（20 号面板、71 号源年份选参）用单面查询、K ∈ {1, 5, 10, 20, 50}；64 号的编码器臂只在 K=5、
    全面位均值特征下评估。要回答「目标年暴露是否改变方法的先后次序」，须让同一协议训练的折限定编码器 R、
    合并池编码器 S_pool 与目标暴露编码器 S_src 在与 SVM、PLS-DA 完全相同的情节上评估：同一批支持果、
    同一面位、同一批查询。

设计：
    编码器   64 号在 three_year（默认）或 two_year（--design two_year）/ canonical 下训练的 R、S_src（每种子每折一个）
             与 S_pool（每种子一个）。
             只载入、不训练：模型卡逐项与 64 号设定一致、训练已走完、权重 SHA-256 与卡相符；
             任一项不符即中止（本脚本不会覆盖 64 号的权重）
    评估     20 号 _proto_eval（融合权重 0）：支持集与查询集各果随机取一面；重采样种子 S+i，i = 0..49。
             抽样顺序与 20 号 _eval_sklearn、71 号 episode 相同，三者逐情节同一批支持果与面位
    K        1, 5, 10, 20, 50（与 71 号相同）
    聚合     先对每个种子求三折均值，再对种子求均值与 SD；与 71 号所选超参数的 SVM、PLS-DA 按种子配对作差，
             并列出每个 K 下三条无泄漏路线（R、SVM、PLS-DA）与把 R 换成 S_pool 后的先后次序

运行方式:
    cd <项目根>
    # 每个正式种子一次（CUDA，须先有 64 号 Formal 的 three_year canonical 编码器）
    python 02code/72_kscan_arms.py --device cuda --stage formal --seed 20060515
    # 同仪器两年设计（须先有 64 号 Formal 的 two_year canonical 编码器；输出文件名带 two_year 段）
    python 02code/72_kscan_arms.py --device cuda --stage formal --seed 20060515 --design two_year
    # 五个种子都有之后聚合（需 71 号结果表）
    python 02code/72_kscan_arms.py --aggregate

输出文件:
    04outputs/formal_seeds/72_kscan__seed<S>[__pilot].xlsx — 逐折逐 K 逐臂的宏 F1 均值与情节 SD
    04outputs/72_kscan_arms.xlsx（--aggregate）— 三折均值、与 SVM/PLS-DA 的配对差、先后次序
    05logs/72_kscan_arms_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

import export_utils as eu
from protonet import ProtoEncoder, build_apple_data, build_year_region_data, encode_batch

NDArray = npt.NDArray[Any]
m20 = importlib.import_module("20_unified_singleface_kshot")
m64: Any = importlib.import_module("64_leakage_controls")
ARMS = ("R", "S_src", "S_pool")
K_LIST = (1, 5, 10, 20, 50)
VARIANT = "canonical"
YEARS = (2018, 2019, 2025)


def _refuse_training(run_name: str, *_: Any, **__: Any) -> Any:
    raise SystemExit(f"{run_name}：没有可复用的 64 号编码器（见上一行「不复用」原因），本脚本不训练")


def load_encoders(
    apple_data: Any, n_bands: int, args: argparse.Namespace, device: torch.device, logger: eu.Logger
) -> dict[str, dict[int, ProtoEncoder]]:
    """按 64 号 main 的同一组训练/验证集与模型卡字段载入编码器；借 64 号 obtain_encoder 的逐项核对。"""
    m64.train_encoder = _refuse_training  # obtain_encoder 核对不过时会调它；在这里改成直接中止
    folds = m64.DESIGNS[args.design]
    tag = f"64_{args.design}_{VARIANT}"
    out: dict[str, dict[int, ProtoEncoder]] = {a: {} for a in ARMS}
    pooled: ProtoEncoder | None = None
    for tgt, cfg in sorted(folds.items()):
        src = {e: apple_data[e] for e in cfg["src"]}
        union = {e: apple_data[e] for e in cfg["src"] + cfg["tgt"]}
        card = {
            "design": args.design,
            "data_variant": VARIANT,
            "fold_year": tgt,
            **{k: getattr(args, k) for k in m64.LEGACY_CARD_PROTOCOL},
        }
        for arm, (tr, va) in {"R": (src, src), "S_src": (union, src)}.items():
            enc, _ = m64.obtain_encoder(
                f"{tag}_{arm}_seed{args.seed}_fold{tgt}",
                tr,
                va,
                "",
                n_bands,
                args,
                device,
                logger,
                {**card, "arm": arm},
            )
            out[arm][tgt] = enc
        if pooled is None:
            pooled, _ = m64.obtain_encoder(
                f"{tag}_S_pool_seed{args.seed}",
                union,
                union,
                "",
                n_bands,
                args,
                device,
                logger,
                {**card, "arm": "S_pool", "fold_year": "all"},
            )
        out["S_pool"][tgt] = pooled
    return out


def face_features(
    encoder: ProtoEncoder, year_data: Any, device: torch.device
) -> dict[int, dict[str, dict[Any, NDArray]]]:
    """与 20 号 load_fold_features 相同：逐面编码，按 年份→产地→苹果编号 存成 (面数, 特征维) 数组。"""
    specs: list[NDArray] = []
    index: list[tuple[int, str, Any, int]] = []
    for year in YEARS:
        for reg in year_data[year]:
            for aid, faces in year_data[year][reg].items():
                for fi in range(len(faces)):
                    specs.append(faces[fi])
                    index.append((year, reg, aid, fi))
    f_all = encode_batch(encoder, np.stack(specs).astype(np.float32), device, batch_size=2048)
    nested: dict[int, dict[str, dict[Any, list[Any]]]] = {y: {} for y in YEARS}
    for (year, reg, aid, fi), f in zip(index, f_all, strict=True):
        nested[year].setdefault(reg, {}).setdefault(aid, [None] * len(year_data[year][reg][aid]))[fi] = f
    return {y: {r: {a: np.stack(v) for a, v in regs.items()} for r, regs in nested[y].items()} for y in YEARS}


def run_seed(args: argparse.Namespace) -> None:
    device = eu.get_device(args.device)
    logger = eu.get_logger("72_kscan_arms")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "编码器臂的单面 K 扫描（与 SVM、PLS-DA 同情节）",
            "设计 / 数据": f"{args.design} / {VARIANT}",
            "臂": ARMS,
            "K": K_LIST,
            "阶段": args.stage,
            "设备": str(device),
            "种子": args.seed,
            "重采样": args.reps,
        },
    )
    df = m64.load_data(VARIANT, logger)
    wl_cols = eu.get_spectrum_columns(df)
    apple_data = build_apple_data(df, wl_cols)
    year_data = build_year_region_data(df, wl_cols)
    rows: list[dict[str, Any]] = []
    with eu.track_compute("编码器臂 K 扫描", logger, None) as budget:
        encoders = load_encoders(apple_data, len(wl_cols), args, device, logger)
        cache: dict[int, Any] = {}
        # 三年设计沿用 20 号的折配置（与此前的正式结果逐位相同）；两年设计取 64 号的折配置
        folds = m20.FOLD_CFG if args.design == "three_year" else m64.DESIGNS[args.design]
        for tgt, cfg in sorted(folds.items()):
            for arm in ARMS:
                enc = encoders[arm][tgt]
                feats = cache.setdefault(id(enc), face_features(enc, year_data, device))
                for k in K_LIST:
                    f1s = [
                        float(m20._proto_eval(feats, year_data, tgt, k, args.seed + i, cfg["src_years"], 0.0)[0])
                        for i in range(args.reps)
                    ]
                    rows.append(
                        {
                            "种子": args.seed,
                            "设计": args.design,
                            "目标年份": tgt,
                            "臂": arm,
                            "K": k,
                            "F1均值": float(np.mean(f1s)),
                            "F1情节SD": float(np.std(f1s)),
                        }
                    )
                logger.log(
                    f"  留出 {tgt} {arm}：" + " ".join(f"K={r['K']}:{r['F1均值']:.4f}" for r in rows[-len(K_LIST) :])
                )
    suffix = "" if args.stage == "formal" else "__pilot"
    out_dir = os.path.join(eu.OUTPUT_DIR, "formal_seeds")
    os.makedirs(out_dir, exist_ok=True)
    seg = "" if args.design == "three_year" else f"{args.design}__"
    out_path = os.path.join(out_dir, f"72_kscan__{seg}seed{args.seed}{suffix}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name=eu.table_sheet_name(0, "逐折逐K逐臂"), index=False)
        pd.DataFrame([{"seed": args.seed, "stage": args.stage, **budget}]).to_excel(
            writer, sheet_name="compute", index=False
        )
    logger.log(f"结果表 → {out_path}")
    logger.close()


def aggregate() -> None:
    logger = eu.get_logger("72_kscan_arms")
    seeds = [int(s) for s in eu.RECOMMENDED_SEEDS]
    files = [os.path.join(eu.OUTPUT_DIR, "formal_seeds", f"72_kscan__seed{s}.xlsx") for s in seeds]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"缺正式种子结果：{[os.path.basename(f) for f in missing]}")
    arms = pd.concat([pd.read_excel(f, sheet_name=0) for f in files], ignore_index=True)
    t71 = pd.read_excel(os.path.join(eu.OUTPUT_DIR, "71_source_only_tuning.xlsx"), sheet_name=0)
    base = t71.rename(columns={"方法": "臂", "目标宏F1(所选)": "F1均值"})[["种子", "目标年份", "臂", "K", "F1均值"]]
    base["臂"] = base["臂"].map({"SVM-Kshot": "SVM(源年份选参)", "PLS-DA-Kshot": "PLS-DA(源年份选参)"})
    long = pd.concat([arms[["种子", "目标年份", "臂", "K", "F1均值"]], base], ignore_index=True)
    long = long[long["种子"].isin(seeds)]
    # 每个种子的三折均值
    per_seed = long.groupby(["臂", "K", "种子"])["F1均值"].mean().unstack("臂")
    summ = per_seed.groupby("K").agg(["mean", "std"]).round(4)
    summ.columns = [f"{a}_{'均值' if s == 'mean' else '种子SD'}" for a, s in summ.columns]
    summ = summ.reset_index()
    pair_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    rivals = ("SVM(源年份选参)", "PLS-DA(源年份选参)")
    for k, sub in per_seed.groupby("K"):
        for arm in ARMS:
            for rival in rivals:
                d = sub[arm] - sub[rival]
                pair_rows.append(
                    {
                        "K": k,
                        "对比": f"{arm} − {rival}",
                        "均值": round(float(d.mean()), 4),
                        "种子最小": round(float(d.min()), 4),
                        "种子最大": round(float(d.max()), 4),
                        "为正的种子数": int((d > 0).sum()),
                    }
                )
        means = sub.mean()
        for lead_arm in ("R", "S_pool"):
            cands = [lead_arm, *rivals]
            order = sorted(cands, key=lambda c: -float(means[c]))
            rank_rows.append(
                {
                    "K": k,
                    "网络的编码器": lead_arm,
                    "第一": order[0],
                    "第二": order[1],
                    "第三": order[2],
                    **{c: round(float(means[c]), 4) for c in cands},
                }
            )
    fold_mean = long.groupby(["臂", "K", "目标年份"])["F1均值"].mean()
    folds = fold_mean.unstack("目标年份").round(4).reset_index()
    # 折为单位的配对检验，口径与 42 号相同：Nadeau-Bengio 修正（n_train=100, n_test=50），Holm 跨 K
    test_rows: list[dict[str, Any]] = []
    years = sorted(long["目标年份"].unique())
    for arm in ARMS:
        for rival in rivals:
            ks = sorted(long["K"].unique())
            res = [
                eu.compare_methods(
                    [float(fold_mean[(arm, k, y)]) for y in years],
                    [float(fold_mean[(rival, k, y)]) for y in years],
                    n_train=100,
                    n_test=50,
                    n_bootstrap=2000,
                    seed=seeds[0],
                )
                for k in ks
            ]
            holm = eu.apply_multi_comp_correction([r["p_value"] for r in res], method="holm")
            for k, r, p_adj in zip(ks, res, holm["p_adjusted"], strict=True):
                test_rows.append(
                    {
                        "对比": f"{arm} − {rival}",
                        "K": k,
                        "折级均值差": round(r["mean_diff"], 4),
                        "p_raw": round(r["p_value"], 4),
                        "p_holm": round(p_adj, 4),
                        "Cohen_d": round(r["cohens_d"], 3),
                    }
                )
    pairs, ranks, tests = pd.DataFrame(pair_rows), pd.DataFrame(rank_rows), pd.DataFrame(test_rows)
    logger.log("三折均值（种子均值、种子 SD）\n" + summ.to_string(index=False))
    logger.log("配对差\n" + pairs.to_string(index=False))
    logger.log("先后次序\n" + ranks.to_string(index=False))
    logger.log("折级配对检验（n=3，Nadeau-Bengio 修正，Holm 跨 K）\n" + tests.to_string(index=False))
    out = eu.write_script_workbook(
        __file__,
        {
            0: ("三折均值", summ),
            1: ("配对差", pairs),
            2: ("先后次序", ranks),
            3: ("逐折均值", folds),
            4: ("折级检验", tests),
        },
    )
    logger.log(f"结果表 → {out}")
    logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="72_kscan_arms：编码器臂的单面 K 扫描")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--design", default="three_year", choices=sorted(m64.DESIGNS))
    parser.add_argument("--aggregate", action="store_true", help="只汇总三年设计（两年设计由 75 号汇总）")
    # 以下与 64 号默认值一致，只用于核对模型卡（本脚本不训练）
    parser.add_argument("--episodes", type=int, default=2000)
    for key, val in m64.LEGACY_CARD_PROTOCOL.items():
        parser.add_argument(f"--{key}", type=type(val), default=val)
    args = parser.parse_args()
    args.reuse_ckpt = True
    if args.aggregate:
        aggregate()
    else:
        run_seed(args)


if __name__ == "__main__":
    main()
