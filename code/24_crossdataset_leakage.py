"""24_crossdataset_leakage.py: 跨数据集表示层评估泄漏对照（案例→现象）。

在多个真实光谱/高光谱数据集上复现表示层评估泄漏：每数据集含天然分布漂移轴（仪器/季/空间），
对每个"留一域"折，比较共享编码器（全域元训练）vs 折限定编码器（仅源域），K-shot ProtoNet
(alpha=0) 评估留出域，Delta=共享-折限定 macro-F1，并算源vs留出的线性 MMD^2。回归目标按源域
三分位分 3 类（防泄漏）；HSI 用原生地物类。汇总长格式供混合效应统计（25_mixed_effects_leakage）。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE python 02code/24_crossdataset_leakage.py --device auto --stage pilot
    # Formal（CUDA 5 种子）:
    python 02code/24_crossdataset_leakage.py --device cuda --stage formal \
        --seeds 20060515,20041210,19810915,2023,2024
    # 敏感性（非默认设置写到带后缀的工作簿，不覆盖主结果）:
    #   --class_cap 0            不设 6 类上限，评估全部合法类
    #   --ip_buffer 10           Indian Pines 各象限去掉距分界线 10 像素以内的像素（空间缓冲带）
    #   --datasets indianpines_spatial   只跑指定数据集

类集合：回归目标按源域三分位分 3 类（阈值只用源域）；每折评估的类 = 源域样本 ≥15 且留出域样本 ≥ K+2 的类
（K-shot 部署时支持集本身就揭示了目标域有哪些类；留出域标签只用于核对「该类在留出域存在且够组成支持集+查询集」，
两臂同折共用同一类集合），默认按类编号取前 6 类。

输出文件:
    04outputs/24_crossdataset_leakage.xlsx  — 各数据集折级 Delta 汇总 + 长格式 + 算力
    05logs/24_crossdataset_leakage_YY-MM-DD_HHMMSS.log — 运行日志
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.io as sio
import torch
from sklearn.metrics import f1_score

import export_utils as eu
from protonet import ProtoEncoder, proto_loss

warnings.filterwarnings("ignore")

NDArray = npt.NDArray[Any]


# ─────────────────────────── 预处理工具 ───────────────────────────
def snv(spec: NDArray) -> NDArray:
    """逐光谱标准正态变量变换（L1 安全：仅用该光谱自身统计）。"""
    out: NDArray = (spec - spec.mean(1, keepdims=True)) / (spec.std(1, keepdims=True) + 1e-8)
    return out


def tertile_labels(y_src: NDArray, y_all: NDArray) -> NDArray:
    """按源域三分位把连续目标分 3 类（阈值仅用源域，防泄漏）。"""
    q1, q2 = np.quantile(y_src, [1 / 3, 2 / 3])
    return np.digitize(y_all, [q1, q2]).astype(int)


def mmd2_linear(a: NDArray, b: NDArray) -> float:
    """线性核 MMD^2（原始光谱均值差平方范数）。"""
    return float(np.sum((a.mean(0) - b.mean(0)) ** 2))


# ─────────────────────────── 数据集加载 ───────────────────────────
def load_corn(root: str, seed: int) -> tuple[dict[str, tuple[NDArray, NDArray]], bool]:
    mat = sio.loadmat(f"{root}/036_corn_nir_3instruments/corn.mat")
    prot = np.asarray(mat["propvals"][0, 0]["data"], dtype=np.float64)[:, 2]
    doms = {
        key: (np.asarray(mat[key][0, 0]["data"], dtype=np.float64), prot) for key in ["m5spec", "mp5spec", "mp6spec"]
    }
    return doms, True


def load_tablet(root: str, seed: int) -> tuple[dict[str, tuple[NDArray, NDArray]], bool]:
    mat = sio.loadmat(f"{root}/037_tablet_nir_shootout2002/nir_shootout_2002.mat")
    parts = ["calibrate", "test", "validate"]
    doms = {}
    for inst in ["1", "2"]:
        spec = np.vstack([np.asarray(mat[f"{p}_{inst}"][0, 0]["data"], dtype=np.float64) for p in parts])
        yval = np.vstack([np.asarray(mat[f"{p}_Y"][0, 0]["data"], dtype=np.float64) for p in parts])
        doms[f"instr{inst}"] = (spec, yval[:, 0])
    return doms, True


def load_mango(root: str, seed: int) -> tuple[dict[str, tuple[NDArray, NDArray]], bool]:
    df = pd.read_csv(f"{root}/041_mango_anderson_crossseason/NAnderson2020MendeleyMangoNIRData.csv")
    wl = [c for c in df.columns if c.isdigit()]
    doms: dict[str, tuple[NDArray, NDArray]] = {}
    for season in sorted(df["Season"].unique()):
        sub = df[df["Season"] == season]
        if len(sub) < 60:
            continue
        sub = sub.sample(min(500, len(sub)), random_state=seed)
        doms[f"S{season}"] = (sub[wl].to_numpy(np.float64), sub["DM"].to_numpy(np.float64))
    return doms, True


def load_indianpines(root: str, seed: int, buffer: int = 0) -> tuple[dict[str, tuple[NDArray, NDArray]], bool]:
    base = f"{root}/034_hsi_classification_benchmarks"
    cube = sio.loadmat(f"{base}/Indian_pines_corrected.mat")["indian_pines_corrected"].astype(np.float64)
    gt = sio.loadmat(f"{base}/Indian_pines_gt.mat")["indian_pines_gt"].astype(int)
    height, width, n_band = cube.shape
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dom = (rows >= height // 2).astype(int) * 2 + (cols >= width // 2).astype(int)
    # 空间缓冲带：去掉离象限分界线不足 buffer 像素的像素（buffer=0 时不去）
    edge = (np.abs(rows - height // 2 + 0.5) < buffer) | (np.abs(cols - width // 2 + 0.5) < buffer)
    spec, lab, dvec = cube.reshape(-1, n_band), gt.reshape(-1), dom.reshape(-1)
    keep = (lab > 0) & ~edge.reshape(-1)
    spec, lab, dvec = spec[keep], lab[keep], dvec[keep]
    doms = {f"Q{q}": (spec[dvec == q], lab[dvec == q]) for q in sorted(np.unique(dvec))}
    return doms, False  # native 分类


DATASETS: dict[str, dict[str, Any]] = {
    "corn_instr": {"loader": load_corn, "drift": "仪器", "modality": "NIR点谱"},
    "tablet_instr": {"loader": load_tablet, "drift": "仪器", "modality": "NIR点谱"},
    "mango_season": {"loader": load_mango, "drift": "采收季", "modality": "NIR点谱"},
    "indianpines_spatial": {"loader": load_indianpines, "drift": "空间域", "modality": "HSI成像"},
}


# ─────────────────────────── 元训练与评估 ───────────────────────────
class ArrSampler:
    """通用 (X,y) episode 采样器（最多 max_way 类）。"""

    def __init__(self, feats: NDArray, labels: NDArray, k: int, q: int, max_way: int = 6) -> None:
        self.by_class = {c: np.where(labels == c)[0] for c in np.unique(labels)}
        self.feats, self.k, self.q, self.max_way = feats, k, q, max_way
        self.classes = list(np.unique(labels))

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if len(self.classes) <= self.max_way:
            chosen = self.classes
        else:
            chosen = list(np.random.choice(self.classes, self.max_way, replace=False))
        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        for label, cls in enumerate(chosen):
            idx = self.by_class[cls].copy()
            np.random.shuffle(idx)
            need = self.k + self.q
            if len(idx) < need:
                idx = np.tile(idx, need // len(idx) + 1)
            ch = idx[:need]
            for i in ch[: self.k]:
                sup_x.append(self.feats[i])
                sup_y.append(label)
            for i in ch[self.k :]:
                qry_x.append(self.feats[i])
                qry_y.append(label)
        return (
            torch.from_numpy(np.stack(sup_x)).float(),
            torch.tensor(sup_y, dtype=torch.long),
            torch.from_numpy(np.stack(qry_x)).float(),
            torch.tensor(qry_y, dtype=torch.long),
            len(chosen),
        )


def meta_train(
    feats: NDArray, labels: NDArray, in_dim: int, device: torch.device, episodes: int, seed: int
) -> ProtoEncoder:
    """元训练 ProtoEncoder（共享或折限定由调用方传入的样本集决定）。"""
    eu.set_random_seed(seed)
    enc = ProtoEncoder(in_dim, feat_dim=64).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    smp = ArrSampler(feats, labels, 5, 10)
    enc.train()
    for _ in range(episodes):
        sup_x, sup_y, qry_x, qry_y, n_way = smp.sample()
        opt.zero_grad()
        loss, _ = proto_loss(enc, sup_x, sup_y, qry_x, qry_y, n_way, device)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    return enc


def eval_kshot(
    enc: ProtoEncoder,
    feats: NDArray,
    labels: NDArray,
    classes: list[int],
    device: torch.device,
    k: int,
    reps: int,
    seed: int,
) -> float:
    """留出域 K-shot ProtoNet(alpha=0) 评估，返回 reps 次重复的宏 F1 均值。"""
    np.random.seed(seed)
    enc.eval()
    f1s: list[float] = []
    with torch.no_grad():
        for _ in range(reps):
            proto: dict[int, NDArray] = {}
            used: set[int] = set()
            ok = True
            for cls in classes:
                idx = np.where(labels == cls)[0].copy()
                np.random.shuffle(idx)
                if len(idx) < k + 2:
                    ok = False
                    break
                sup = idx[:k]
                used.update(sup.tolist())
                feat = enc(torch.from_numpy(feats[sup]).float().to(device)).cpu().numpy().mean(0)
                proto[cls] = feat / (np.linalg.norm(feat) + 1e-8)
            if not ok:
                continue
            qidx = [i for i in range(len(labels)) if i not in used and labels[i] in classes]
            if not qidx:
                continue
            proto_mat = np.stack([proto[c] for c in classes])
            feat_q = enc(torch.from_numpy(feats[qidx]).float().to(device)).cpu().numpy()
            pred = np.array(classes)[(feat_q @ proto_mat.T).argmax(1)]
            f1s.append(f1_score(np.array(labels)[qidx], pred, average="macro", labels=classes))
    return float(np.mean(f1s)) if f1s else float("nan")


def run_fold(
    doms: dict[str, tuple[NDArray, NDArray]],
    is_reg: bool,
    held: str,
    device: torch.device,
    episodes: int,
    reps: int,
    k: int,
    seed: int,
    class_cap: int = 6,
) -> dict[str, Any] | None:
    """单个留一域折：共享 vs 折限定编码器，返回 Delta/MMD 记录（类不足则 None）。"""
    names = list(doms.keys())
    src = [d for d in names if d != held]
    in_dim = doms[names[0]][0].shape[1]

    y_src_cont = np.concatenate([doms[d][1] for d in src])

    def lab(v: NDArray) -> NDArray:
        return tertile_labels(y_src_cont, v) if is_reg else v.astype(int)

    ys_all = lab(np.concatenate([doms[d][1] for d in src]))
    yh_all = lab(doms[held][1])
    cs_src = {c for c in np.unique(ys_all) if (ys_all == c).sum() >= 15}
    cs_held = {c for c in np.unique(yh_all) if (yh_all == c).sum() >= k + 2}
    classes = sorted(cs_src & cs_held)
    if class_cap > 0:
        classes = classes[:class_cap]
    if len(classes) < 3:
        return None

    def build(dom_list: list[str]) -> tuple[NDArray, NDArray]:
        feats = snv(np.vstack([doms[d][0] for d in dom_list]))
        labels = np.concatenate([lab(doms[d][1]) for d in dom_list])
        mask = np.isin(labels, classes)
        return feats[mask], labels[mask]

    xs, ys = build(src)
    xall, yall = build(names)
    xh, yh = build([held])
    enc_shared = meta_train(xall, yall, in_dim, device, episodes, seed)
    enc_fold = meta_train(xs, ys, in_dim, device, episodes, seed)
    f_sh = eval_kshot(enc_shared, xh, yh, classes, device, k, reps, seed)
    f_fr = eval_kshot(enc_fold, xh, yh, classes, device, k, reps, seed)
    return {
        "held": held,
        "shared": f_sh,
        "fold_restricted": f_fr,
        "delta": f_sh - f_fr,
        "mmd2": mmd2_linear(xs, xh),
        "n_classes": len(classes),
        "classes": ",".join(str(c) for c in classes),
        "n_held": len(yh),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="跨数据集表示层评估泄漏对照")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    parser.add_argument("--seeds", default="2004", help="逗号分隔种子；formal 用 5 固定种子")
    parser.add_argument("--data_root", default="05data")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--class_cap", type=int, default=6, help="每折最多评估的类数；0 = 不设上限")
    parser.add_argument("--ip_buffer", type=int, default=0, help="Indian Pines 象限分界线两侧剔除的像素宽度")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="逗号分隔的数据集名")
    args = parser.parse_args()
    ds_list = [d for d in args.datasets.split(",") if d]
    unknown = sorted(set(ds_list) - set(DATASETS))
    if unknown:
        raise SystemExit(f"未知数据集：{unknown}")
    tag_parts = []
    if args.class_cap != 6:
        tag_parts.append(f"cap{args.class_cap}")
    if args.ip_buffer:
        tag_parts.append(f"buf{args.ip_buffer}")
    if ds_list != list(DATASETS):
        tag_parts.append("-".join(ds_list))
    tag = "_".join(tag_parts)

    device = eu.get_device(args.device)
    logger = eu.get_logger("24_crossdataset_leakage")
    seeds = [int(s) for s in args.seeds.split(",")]
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "跨数据集评估泄漏",
            "阶段": args.stage,
            "设备": str(device),
            "种子": seeds,
            "数据集": ds_list,
            "每折类数上限": args.class_cap or "不设",
            "Indian Pines 缓冲带像素": args.ip_buffer,
            "K / 元训练情节 / 每折评估重复": f"{args.k} / {args.episodes} / {args.reps}",
        },
    )
    run = eu.init_wandb(__file__, seeds[0], eu.PROJECT_CODE, args.stage, vars(args))

    long_rows: list[dict[str, Any]] = []
    with eu.track_compute("跨数据集泄漏", logger, run) as budget:
        for dsname in ds_list:
            meta = DATASETS[dsname]
            loader = meta["loader"]
            for seed in seeds:
                if dsname == "indianpines_spatial":
                    doms, is_reg = loader(args.data_root, seed, buffer=args.ip_buffer)
                else:
                    doms, is_reg = loader(args.data_root, seed)
                for held in list(doms.keys()):
                    rec = run_fold(doms, is_reg, held, device, args.episodes, args.reps, args.k, seed, args.class_cap)
                    if rec is None:
                        logger.log(f"  [{dsname} seed{seed}] 留出{held}: 合法类不足 3 个，此折不评")
                        continue
                    rec.update(
                        {"dataset": dsname, "drift_axis": meta["drift"], "modality": meta["modality"], "seed": seed}
                    )
                    long_rows.append(rec)
                    logger.log(
                        f"  [{dsname} seed{seed}] 留出{rec['held']}: 共享={rec['shared']:.4f} "
                        f"折限定={rec['fold_restricted']:.4f} Delta={rec['delta']:+.4f} MMD^2={rec['mmd2']:.3f}"
                    )

    long_df = pd.DataFrame(long_rows)
    summ = (
        long_df.groupby(["dataset", "drift_axis", "modality", "held"])
        .agg(
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            shared_mean=("shared", "mean"),
            fold_restricted_mean=("fold_restricted", "mean"),
            mmd2=("mmd2", "mean"),
        )
        .reset_index()
    )
    ds_summ = (
        long_df.groupby(["dataset", "drift_axis", "modality"])
        .agg(
            n_folds=("held", "nunique"),
            delta_mean=("delta", "mean"),
            pos_frac=("delta", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    n_pos = int((summ["delta_mean"] > 0).sum())
    logger.log(
        f"\n★ 汇总: {len(summ)} 折级, Delta>0 折占比={n_pos}/{len(summ)}="
        f"{n_pos / max(len(summ), 1):.0%}, 总Delta均值={summ['delta_mean'].mean():+.4f}"
    )

    sheets: dict[object, object] = {
        eu.table_sheet_name(1, "数据集汇总"): ds_summ,
        eu.table_sheet_name(2, "折级汇总"): summ,
        eu.table_sheet_name(3, "长格式"): long_df,
        eu.table_sheet_name(4, "compute"): pd.DataFrame([dict(budget)]),
    }
    stem = os.path.splitext(os.path.basename(__file__))[0]
    out = eu.write_script_workbook(f"{stem}__{tag}.py" if tag else __file__, sheets)
    logger.log(f"已写 {out}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
