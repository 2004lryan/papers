"""
39_face_instrument_confound.py: 把仪器/面位混淆从跨年标签互证里拆出来（LOEO-2025，零 GPU）。

问题背景:
    源质心完全锚定（src_alpha=1.0）时 LOEO-2025 折宏 F1 = 0.199，低于 3 类随机基线 0.333。
    该塌陷同时可由「2025 山东标签不可信」与「换仪器 + 单面对多面协议失配」解释，两者当前
    完全混淆。本脚本的唯一目标是把后者拆出来，判定 0.199 里多少归仪器/面位、多少归标签。

四个 Block:
    A  编码器空间：复现 α∈{0,0.4,1} 的 LOEO-2025 宏 F1 → 逐类拆解 → 面位口径消融
       （ALL/EQ/EQ1/TOP/BOT）→ 域校正消融（none/CTR/ZSC）。回答「面位对齐能救回多少」。
    B  光谱空间（无编码器）最近质心网格：预处理 {SNV,+D1,+D2} × 域校正 {none,CTR,ZSC}
       × 面位 4 档。**口径预选依据是源域内部 2018↔2019 迁移，完全不含 2025**，故与 2025
       标签无关，不构成「调口径调到好看为止」。
    C  抗域漂移的标签互证：score_R(x) = -||x - c_R||，只比较目标域内部的排序，对任何
       全局平移/缩放不变 → AUC + 甘肃阴性对照 + 源域内部标定 + 苹果级 bootstrap CI。
    D  反方向：2025 建原型 → 判 2018/2019，检验不对称性。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/39_face_instrument_confound.py

输出文件:
    04outputs/39_face_instrument_confound-图a.pdf / -fig-a.pdf  — 面位×α 消融（面位对齐救不回来）
    04outputs/39_face_instrument_confound-图b.pdf / -fig-b.pdf  — 预处理/域校正恢复网格
    04outputs/39_face_instrument_confound-图c.pdf / -fig-c.pdf  — 抗漂移 AUC 森林图（含 GS 阴性对照）
    04outputs/39_face_instrument_confound.xlsx                   — A/B/C/D 全部数值 + 算力
    05logs/39_face_instrument_confound_YY-MM-DD_HHMMSS.log      — 运行日志
    07checkpoints/39_face_instrument_confound_prespec/best.joblib + model_card.md

说明:
    编码器一律取 10 的 fold-2025 折限定编码器（07checkpoints/10_loeo_year_foldwise_seed2004_fold2025），
    与被解释的那条 0.199 同源。全流程 CPU、无训练、不动 03data，故 stage='pilot'。
    CTR/ZSC 为**直推式**域校正：只用目标域的无标签一阶/二阶矩，不接触任何目标标签；
    它无法制造类别特异的一致性，故不会伪造标签互证结论（见 §C 的 GS 阴性对照）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_utils as eu
from protonet import ProtoEncoder, encode_batch

NDArray = npt.NDArray[Any]

TGT_YEAR = 2025
SRC_YEARS = (2018, 2019)
SRC_REGIONS = ("XJ", "SD", "SX")
TGT_REGIONS = ("GS", "SD", "XJ")  # 与 sorted(year_data[2025].keys()) 一致
FACE_POLICIES = ("ALL", "EQ", "EQ1", "TOP", "BOT")
FACE_LABEL_ZH = {
    "ALL": "全 5 面均值（原口径）",
    "EQ": "仅赤道 3 面均值（协议对齐）",
    "EQ1": "赤道随机单面（协议全对齐）",
    "TOP": "仅顶部",
    "BOT": "仅底部",
}
FACE_LABEL_EN = {
    "ALL": "All 5 facets (original)",
    "EQ": "Equatorial 3 facets (aligned)",
    "EQ1": "Single equatorial facet",
    "TOP": "Top only",
    "BOT": "Bottom only",
}
CORRECTIONS = ("none", "CTR", "ZSC")
CORR_LABEL_ZH = {"none": "无校正", "CTR": "域均值居中", "ZSC": "域 z-score"}
CORR_LABEL_EN = {"none": "No correction", "CTR": "Domain mean-centring", "ZSC": "Domain z-score"}
SPEC_PREPS = ("SNV", "SNV+D1", "SNV+D2")
ALPHAS = (0.0, 0.4, 1.0)
K_SHOT = 5
N_REPS = 50
N_BOOT = 2000
WONG = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999999"]

# 面位对齐的先验口径：2018/2019 每果只在赤道面采 1 条（稿件 §1.1），故协议对齐档 = EQ。
# 该选择由采集协议决定，不看任何 2025 结果，是设计层面的预设而非事后挑选。
PRESPEC_FACE = "EQ"


# ═══════════════════════════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════════════════════════
def unit(v: NDArray) -> NDArray:
    """按最后一维 L2 归一化。"""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.asarray(v / np.maximum(n, 1e-8))


def domain_correct(mat: NDArray, domains: list[NDArray], mode: str) -> NDArray:
    """按域做直推式一阶/二阶矩对齐（不使用任何标签）。"""
    if mode == "none":
        return mat
    out = mat.copy()
    for m in domains:
        block = out[m]
        mu = block.mean(0, keepdims=True)
        if mode == "CTR":
            out[m] = block - mu
        elif mode == "ZSC":
            sd = block.std(0, keepdims=True)
            sd[sd < 1e-12] = 1.0
            out[m] = (block - mu) / sd
    return out


def auc_mwu(a: NDArray, b: NDArray) -> tuple[float, float]:
    """P(score_a > score_b) 及双侧 Mann-Whitney P 值。"""
    res = mannwhitneyu(a, b, alternative="two-sided")
    return float(res.statistic) / (len(a) * len(b)), float(res.pvalue)


def auc_only(a: NDArray, b: NDArray) -> float:
    """只算 AUC（bootstrap 内层用，省掉 P 值开销）。"""
    order = np.argsort(np.concatenate([a, b]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # 处理并列：同值取平均秩
    allv = np.concatenate([a, b])
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    r_a = ranks[: len(a)].sum()
    return float((r_a - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def boot_auc_ci(a: NDArray, b: NDArray, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    """苹果级 bootstrap 的 AUC 95% 百分位区间。"""
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = auc_only(rng.choice(a, size=len(a), replace=True), rng.choice(b, size=len(b), replace=True))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ═══════════════════════════════════════════════════════════════════════════════
# 数据与面位索引
# ═══════════════════════════════════════════════════════════════════════════════
class FaceIndex:
    """面位级索引：把 (年份, 产地, 苹果编号) 映射到 all_9elements 的行号。"""

    def __init__(self, df: pd.DataFrame) -> None:
        self.year = df["年份"].to_numpy()
        self.region = df["产地"].to_numpy().astype(str)
        self.apple = df["苹果编号"].to_numpy().astype(str)
        self.face = df["面位"].to_numpy().astype(str)
        self.domains = [self.year == y for y in (*SRC_YEARS, TGT_YEAR)]
        self._rows: dict[tuple[int, str], dict[str, NDArray]] = {}
        idx_all = np.arange(len(df))
        for y in (*SRC_YEARS, TGT_YEAR):
            for r in np.unique(self.region[self.year == y]):
                m = (self.year == y) & (self.region == r)
                bucket: dict[str, NDArray] = {}
                for aid in np.unique(self.apple[m]):
                    bucket[aid] = idx_all[m & (self.apple == aid)]
                self._rows[(y, str(r))] = bucket

    def apples(self, year: int, region: str) -> list[str]:
        return sorted(self._rows[(year, region)].keys())

    def rows(self, year: int, region: str, aid: str) -> NDArray:
        return self._rows[(year, region)][aid]

    def select_faces(self, year: int, rows: NDArray, policy: str, rng: np.random.Generator | None) -> NDArray:
        """按面位口径挑行号；2018/2019 天然单面，任何口径都取那一条。"""
        if year != TGT_YEAR or policy == "ALL":
            return rows
        faces = self.face[rows]
        if policy == "EQ":
            sel = rows[np.char.startswith(faces, "侧面")]
        elif policy == "EQ1":
            cand = rows[np.char.startswith(faces, "侧面")]
            sel = np.array([rng.choice(cand)]) if (len(cand) and rng is not None) else cand
        elif policy == "TOP":
            sel = rows[faces == "顶部"]
        elif policy == "BOT":
            sel = rows[faces == "底部"]
        else:
            raise ValueError(f"未知面位口径 {policy}")
        return sel if len(sel) else rows


def apple_matrix(
    mat: NDArray,
    fidx: FaceIndex,
    year: int,
    region: str,
    policy: str,
    rng: np.random.Generator | None = None,
    renorm: bool = True,
) -> tuple[NDArray, list[str]]:
    """苹果级特征矩阵（选定面位后取均值）。renorm=True 用于余弦空间（编码器特征）。"""
    aids = fidx.apples(year, region)
    feats = [mat[fidx.select_faces(year, fidx.rows(year, region, a), policy, rng)].mean(0) for a in aids]
    arr = np.stack(feats)
    return (unit(arr) if renorm else arr), aids


# ═══════════════════════════════════════════════════════════════════════════════
# Block A —— 编码器空间：复现 0.199 并逐层拆解
# ═══════════════════════════════════════════════════════════════════════════════
def source_prototypes(feat: NDArray, fidx: FaceIndex, policy: str) -> dict[str, NDArray]:
    """源域类原型：先各源年份取全体苹果均值，再跨年份平均（与 17_alpha_sensitivity 同构）。"""
    protos: dict[str, NDArray] = {}
    for reg in SRC_REGIONS:
        per_year = []
        for y in SRC_YEARS:
            arr, _ = apple_matrix(feat, fidx, y, reg, policy)
            per_year.append(unit(arr.mean(0)))
        protos[reg] = unit(np.mean(per_year, axis=0))
    return protos


def eval_alpha_once(
    feat: NDArray,
    fidx: FaceIndex,
    src_protos: dict[str, NDArray],
    tgt_apples: dict[str, tuple[NDArray, list[str]]],
    alpha: float,
    seed: int,
    drop_gs: bool = False,
) -> dict[str, Any]:
    """单次 K-shot 评估：返回宏 F1 与逐类召回。

    与 17_alpha_sensitivity.eval_protonet_alpha 同一判决规则：支持集按 sorted(产地) 依次抽样，
    目标原型=支持集均值，α 融合源原型后重归一化，查询=该折支持集之外的全部苹果，余弦最近原型。
    """
    regions = [r for r in TGT_REGIONS if not (drop_gs and r == "GS")]
    rng = np.random.default_rng(seed)
    support: dict[str, list[str]] = {}
    for reg in regions:
        aids = tgt_apples[reg][1]
        k = min(K_SHOT, max(1, len(aids) - 1))
        support[reg] = list(rng.choice(np.array(aids), size=k, replace=False))

    protos = []
    for reg in regions:
        arr, aids = tgt_apples[reg]
        pos = [aids.index(a) for a in support[reg]]
        p = unit(arr[pos].mean(0))
        if alpha > 0 and reg in src_protos:
            fused = (1 - alpha) * p + alpha * src_protos[reg]
            p = unit(fused)
        protos.append(p)
    proto_mat = np.stack(protos)

    y_true: list[str] = []
    y_pred: list[str] = []
    for reg in regions:
        arr, aids = tgt_apples[reg]
        keep = [i for i, a in enumerate(aids) if a not in support[reg]]
        sim = arr[keep] @ proto_mat.T
        y_pred += [regions[i] for i in sim.argmax(axis=1)]
        y_true += [reg] * len(keep)
    yt = np.array(y_true)
    yp = np.array(y_pred)
    out: dict[str, Any] = {"macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0))}
    for reg in regions:
        out[f"recall_{reg}"] = float((yp[yt == reg] == reg).mean()) if (yt == reg).any() else float("nan")
    return out


def run_block_a(feat_raw: NDArray, fidx: FaceIndex, seeds: list[int], logger: eu.Logger) -> pd.DataFrame:
    """面位口径 × 域校正 × α 的三向消融（编码器空间）。"""
    rows: list[dict[str, Any]] = []
    for corr in CORRECTIONS:
        feat = unit(domain_correct(feat_raw, fidx.domains, corr)) if corr != "none" else feat_raw
        for policy in FACE_POLICIES:
            face_rng = np.random.default_rng(20260817) if policy == "EQ1" else None
            src_p = source_prototypes(feat, fidx, policy)
            tgt = {r: apple_matrix(feat, fidx, TGT_YEAR, r, policy, face_rng) for r in TGT_REGIONS}
            for alpha in ALPHAS:
                for drop_gs in (False, True):
                    recs = [
                        eval_alpha_once(feat, fidx, src_p, tgt, alpha, s + i, drop_gs)
                        for s in seeds
                        for i in range(N_REPS)
                    ]
                    agg: dict[str, Any] = {
                        "域校正": corr,
                        "面位口径": policy,
                        "alpha": alpha,
                        "类别空间": "XJ/SD（剔除源域没有的 GS）" if drop_gs else "XJ/SD/GS（原口径）",
                        "随机基线": round(1 / (2 if drop_gs else 3), 3),
                        "重复数": len(recs),
                    }
                    for key in recs[0]:
                        vals = np.array([r[key] for r in recs], dtype=float)
                        agg[key] = round(float(np.nanmean(vals)), 4)
                        agg[f"{key}_sd"] = round(float(np.nanstd(vals)), 4)
                    rows.append(agg)
            logger.log(f"  [A] 域校正={corr:5s} 面位={policy:4s} 完成")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Block B —— 光谱空间最近质心 + 源域内部预选
# ═══════════════════════════════════════════════════════════════════════════════
def spectral_variant(x: NDArray, fidx: FaceIndex, prep: str, corr: str) -> NDArray:
    z = x.copy()
    if prep == "SNV+D1":
        z = savgol_filter(z, 15, 2, deriv=1, axis=1)
    elif prep == "SNV+D2":
        z = savgol_filter(z, 15, 2, deriv=2, axis=1)
    return domain_correct(z, fidx.domains, corr)


def nc_predict(mat: NDArray, protos: dict[str, NDArray]) -> NDArray:
    keys = np.array(list(protos))
    p = np.stack([protos[k] for k in keys])
    d = ((mat[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    return np.asarray(keys[np.argmin(d, axis=1)])


def run_block_b_source_internal(x: NDArray, fidx: FaceIndex, logger: eu.Logger) -> pd.DataFrame:
    """源域内部 2018↔2019 迁移：同仪器、同赤道单面、同「各产地当地市场」采购。

    **不含任何 2025 数据**，故可作为与 2025 标签完全无关的口径预选依据。
    """
    rows: list[dict[str, Any]] = []
    for prep in SPEC_PREPS:
        for corr in CORRECTIONS:
            z = spectral_variant(x, fidx, prep, corr)
            for a, b in ((2018, 2019), (2019, 2018)):
                protos = {r: apple_matrix(z, fidx, a, r, "ALL", renorm=False)[0].mean(0) for r in SRC_REGIONS}
                yt: list[str] = []
                yp: list[str] = []
                for r in SRC_REGIONS:
                    arr, _ = apple_matrix(z, fidx, b, r, "ALL", renorm=False)
                    yp += list(nc_predict(arr, protos))
                    yt += [r] * len(arr)
                yta, ypa = np.array(yt), np.array(yp)
                rows.append(
                    {
                        "预处理": prep,
                        "域校正": corr,
                        "方向": f"{a}→{b}",
                        "宏F1": round(float(f1_score(yta, ypa, average="macro", zero_division=0)), 4),
                        **{
                            f"{r}召回": round(float((ypa[yta == r] == r).mean()), 4) for r in SRC_REGIONS
                        },
                        "SD误判到SX": round(float((ypa[yta == "SD"] == "SX").mean()), 4),
                    }
                )
    df = pd.DataFrame(rows)
    logger.log(f"  [B-源域内部] {len(df)} 行")
    return df


def run_block_b_target(x: NDArray, fidx: FaceIndex, logger: eu.Logger) -> pd.DataFrame:
    """光谱空间：2025 苹果 → 2018/2019 类质心 {XJ,SD,SX} 的指派分布。"""
    rows: list[dict[str, Any]] = []
    for prep in SPEC_PREPS:
        for corr in CORRECTIONS:
            z = spectral_variant(x, fidx, prep, corr)
            for policy in ("ALL", "EQ", "TOP", "BOT"):
                protos = {
                    r: np.mean(
                        [apple_matrix(z, fidx, y, r, policy, renorm=False)[0].mean(0) for y in SRC_YEARS], axis=0
                    )
                    for r in SRC_REGIONS
                }
                rec: dict[str, dict[str, float]] = {}
                for t in TGT_REGIONS:
                    arr, _ = apple_matrix(z, fidx, TGT_YEAR, t, policy, renorm=False)
                    pred = nc_predict(arr, protos)
                    rec[t] = {r: float((pred == r).mean()) for r in SRC_REGIONS}
                p2 = {r: protos[r] for r in ("XJ", "SD")}
                yt: list[str] = []
                yp: list[str] = []
                for t in ("XJ", "SD"):
                    arr, _ = apple_matrix(z, fidx, TGT_YEAR, t, policy, renorm=False)
                    yp += list(nc_predict(arr, p2))
                    yt += [t] * len(arr)
                rows.append(
                    {
                        "预处理": prep,
                        "域校正": corr,
                        "面位口径": policy,
                        "XJ→XJ": round(rec["XJ"]["XJ"], 4),
                        "SD→SD": round(rec["SD"]["SD"], 4),
                        "SD→XJ": round(rec["SD"]["XJ"], 4),
                        "SD→SX": round(rec["SD"]["SX"], 4),
                        "GS→XJ": round(rec["GS"]["XJ"], 4),
                        "GS→SD": round(rec["GS"]["SD"], 4),
                        "GS→SX": round(rec["GS"]["SX"], 4),
                        "二类XJvsSD宏F1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
                    }
                )
    logger.log(f"  [B-目标域] {len(rows)} 行")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Block C —— 抗域漂移的标签互证（AUC + GS 阴性对照）
# ═══════════════════════════════════════════════════════════════════════════════
def run_block_c(x: NDArray, fidx: FaceIndex, rng: np.random.Generator, logger: eu.Logger) -> pd.DataFrame:
    """目标域内部排序统计量：对任何作用于整个目标域的平移/缩放不变。

    score_R(x) = -||x - c_R||，c_R 由 2018/2019 算出。比较两组 2025 苹果的 score 排序，
    全局域漂移对两组同等作用 → 被 AUC 消掉；剩下的差异只能来自类别特异的结构。
    """
    rows: list[dict[str, Any]] = []
    contrasts = [
        ("SD", "SD", "GS", "山东锚：宣称SD vs 宣称GS（GS 为阴性对照）"),
        ("SD", "SD", "XJ", "山东锚：宣称SD vs 宣称XJ"),
        ("XJ", "XJ", "GS", "新疆锚：宣称XJ vs 宣称GS（设计上的阳性对照，实测不成立）"),
        ("XJ", "XJ", "SD", "新疆锚：宣称XJ vs 宣称SD"),
    ]
    for prep in SPEC_PREPS:
        for corr in CORRECTIONS:
            z = spectral_variant(x, fidx, prep, corr)
            for policy in ("ALL", "EQ"):
                protos = {
                    r: np.mean(
                        [apple_matrix(z, fidx, y, r, policy, renorm=False)[0].mean(0) for y in SRC_YEARS], axis=0
                    )
                    for r in SRC_REGIONS
                }
                mats = {t: apple_matrix(z, fidx, TGT_YEAR, t, policy, renorm=False)[0] for t in TGT_REGIONS}

                def score(
                    t: str,
                    r: str,
                    _mats: dict[str, NDArray] = mats,
                    _protos: dict[str, NDArray] = protos,
                ) -> NDArray:
                    return np.asarray(-np.linalg.norm(_mats[t] - _protos[r], axis=1))

                for anchor, ga, gb, desc in contrasts:
                    a, b = score(ga, anchor), score(gb, anchor)
                    auc, pval = auc_mwu(a, b)
                    lo, hi = boot_auc_ci(a, b, rng)
                    rows.append(
                        {
                            "预处理": prep,
                            "域校正": corr,
                            "面位口径": policy,
                            "对比": desc,
                            "锚点": anchor,
                            "AUC": round(auc, 4),
                            "CI95下": round(lo, 4),
                            "CI95上": round(hi, 4),
                            "P": f"{pval:.2e}",
                            "n组A": len(a),
                            "n组B": len(b),
                        }
                    )
            logger.log(f"  [C] {prep} / {corr} 完成")
    return pd.DataFrame(rows)


def run_block_c_calibration(x: NDArray, fidx: FaceIndex, rng: np.random.Generator) -> pd.DataFrame:
    """同一统计量在源域内部的标定：2018 锚 → 2019 果，同仪器同面位、标签口径一致。

    这一格给出「该统计量在标签确实可用时应当长什么样」的参照，用来判断 2025 上的数值
    是「统计量没有分辨力」还是「标签确实对不上」。
    """
    rows: list[dict[str, Any]] = []
    for prep in SPEC_PREPS:
        for corr in CORRECTIONS:
            z = spectral_variant(x, fidx, prep, corr)
            protos = {r: apple_matrix(z, fidx, 2018, r, "ALL", renorm=False)[0].mean(0) for r in SRC_REGIONS}
            mats = {r: apple_matrix(z, fidx, 2019, r, "ALL", renorm=False)[0] for r in SRC_REGIONS}
            for anchor, ga, gb in (("SD", "SD", "XJ"), ("SD", "SD", "SX"), ("XJ", "XJ", "SD"), ("XJ", "XJ", "SX")):
                a = -np.linalg.norm(mats[ga] - protos[anchor], axis=1)
                b = -np.linalg.norm(mats[gb] - protos[anchor], axis=1)
                auc, pval = auc_mwu(a, b)
                lo, hi = boot_auc_ci(a, b, rng)
                rows.append(
                    {
                        "预处理": prep,
                        "域校正": corr,
                        "对比": f"{anchor}锚：2019{ga} vs 2019{gb}",
                        "AUC": round(auc, 4),
                        "CI95下": round(lo, 4),
                        "CI95上": round(hi, 4),
                        "P": f"{pval:.2e}",
                    }
                )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Block D —— 反方向 + 漂移几何
# ═══════════════════════════════════════════════════════════════════════════════
def run_block_d_reverse(x: NDArray, fidx: FaceIndex) -> pd.DataFrame:
    """反方向：2025 建质心 → 判 2018/2019 的 XJ/SD（SX 在 2025 无对应，不参评）。"""
    rows: list[dict[str, Any]] = []
    for prep in SPEC_PREPS:
        for corr in CORRECTIONS:
            z = spectral_variant(x, fidx, prep, corr)
            for policy in ("ALL", "EQ"):
                protos = {r: apple_matrix(z, fidx, TGT_YEAR, r, policy, renorm=False)[0].mean(0) for r in ("XJ", "SD")}
                for y in SRC_YEARS:
                    yt: list[str] = []
                    yp: list[str] = []
                    for r in ("XJ", "SD"):
                        arr, _ = apple_matrix(z, fidx, y, r, "ALL", renorm=False)
                        yp += list(nc_predict(arr, protos))
                        yt += [r] * len(arr)
                    yta, ypa = np.array(yt), np.array(yp)
                    rows.append(
                        {
                            "预处理": prep,
                            "域校正": corr,
                            "2025面位口径": policy,
                            "被判年份": y,
                            "二类宏F1": round(float(f1_score(yta, ypa, average="macro", zero_division=0)), 4),
                            "XJ召回": round(float((ypa[yta == "XJ"] == "XJ").mean()), 4),
                            "SD召回": round(float((ypa[yta == "SD"] == "SD").mean()), 4),
                            "随机基线": 0.5,
                        }
                    )
    return pd.DataFrame(rows)


def run_block_e_two_anchor(x: NDArray, fidx: FaceIndex) -> pd.DataFrame:
    """E1 两锚判决：源域各类苹果在「2025 宣称SD 锚」与「2025 宣称GS 锚」之间二选一。

    两个锚都是「非新疆」，故本题剔除了「新疆 vs 其余」这条最容易的边界，逼问的正是
    2025 那批宣称山东的果**是否特异地像山东**。判据是源域 SD 与源域 SX 的选择比之差：
    若 2025 的山东标签只是「不是新疆」的同义词，两者的选择比应当一致。
    """
    rows: list[dict[str, Any]] = []
    for prep in SPEC_PREPS:
        for corr in CORRECTIONS:
            z = spectral_variant(x, fidx, prep, corr)
            for policy in ("ALL", "EQ"):
                anchors = {
                    "2025SD": apple_matrix(z, fidx, TGT_YEAR, "SD", policy, renorm=False)[0].mean(0),
                    "2025GS": apple_matrix(z, fidx, TGT_YEAR, "GS", policy, renorm=False)[0].mean(0),
                }
                rec: dict[str, float] = {}
                for r in SRC_REGIONS:
                    arr = np.concatenate(
                        [apple_matrix(z, fidx, y, r, "ALL", renorm=False)[0] for y in SRC_YEARS], axis=0
                    )
                    pred = nc_predict(arr, anchors)
                    rec[r] = float((pred == "2025SD").mean())
                rows.append(
                    {
                        "预处理": prep,
                        "域校正": corr,
                        "2025面位口径": policy,
                        "源SD选2025SD锚": round(rec["SD"], 4),
                        "源SX选2025SD锚": round(rec["SX"], 4),
                        "源XJ选2025SD锚": round(rec["XJ"], 4),
                        "SD−SX差（>0 才是山东特异）": round(rec["SD"] - rec["SX"], 4),
                    }
                )
    return pd.DataFrame(rows)


def run_block_e_impostor(x: NDArray, fidx: FaceIndex) -> pd.DataFrame:
    """E2 冒名对照：把「2025 宣称SD」换成「2025 宣称GS」当锚，其余不变，判源域 XJ vs SD。

    若换成甘肃当锚也一样好，说明反方向那条高分只是「非新疆」检测器，不构成山东标签的证据。
    """
    rows: list[dict[str, Any]] = []
    for prep in SPEC_PREPS:
        for corr in CORRECTIONS:
            z = spectral_variant(x, fidx, prep, corr)
            for policy in ("ALL", "EQ"):
                for impostor, tag in (("SD", "真锚：2025宣称SD"), ("GS", "冒名锚：2025宣称GS")):
                    protos = {
                        "XJ": apple_matrix(z, fidx, TGT_YEAR, "XJ", policy, renorm=False)[0].mean(0),
                        "SD": apple_matrix(z, fidx, TGT_YEAR, impostor, policy, renorm=False)[0].mean(0),
                    }
                    yt: list[str] = []
                    yp: list[str] = []
                    for r in ("XJ", "SD"):
                        arr = np.concatenate(
                            [apple_matrix(z, fidx, y, r, "ALL", renorm=False)[0] for y in SRC_YEARS], axis=0
                        )
                        yp += list(nc_predict(arr, protos))
                        yt += [r] * len(arr)
                    yta, ypa = np.array(yt), np.array(yp)
                    rows.append(
                        {
                            "预处理": prep,
                            "域校正": corr,
                            "2025面位口径": policy,
                            "锚来源": tag,
                            "源域二类宏F1": round(float(f1_score(yta, ypa, average="macro", zero_division=0)), 4),
                            "源XJ召回": round(float((ypa[yta == "XJ"] == "XJ").mean()), 4),
                            "源SD召回": round(float((ypa[yta == "SD"] == "SD").mean()), 4),
                        }
                    )
    return pd.DataFrame(rows)


def run_block_e_permutation(
    x: NDArray, fidx: FaceIndex, rng: np.random.Generator, n_perm: int = 2000
) -> pd.DataFrame:
    """E3 标签置换负控：在苹果层面随机置换 2025 的 XJ/SD 声明标签，重做反方向二类判决。

    真值若显著高于置换零分布，说明反方向的高分确实来自 2025 声明标签所携带的信息，
    而非任意二分都能得到的几何巧合。
    """
    rows: list[dict[str, Any]] = []
    configs = (
        ("SNV+D1", "ZSC", "EQ"),  # 预选口径
        ("SNV+D1", "ZSC", "ALL"),
        ("SNV+D2", "ZSC", "EQ"),
        ("SNV+D1", "CTR", "EQ"),
        ("SNV", "none", "EQ"),  # 冻结正典口径
    )
    for prep, corr, policy in configs:
        z = spectral_variant(x, fidx, prep, corr)
        pool = np.concatenate([apple_matrix(z, fidx, TGT_YEAR, r, policy, renorm=False)[0] for r in ("XJ", "SD")])
        n_xj = len(fidx.apples(TGT_YEAR, "XJ"))
        src_mats = {
            r: np.concatenate([apple_matrix(z, fidx, y, r, "ALL", renorm=False)[0] for y in SRC_YEARS], axis=0)
            for r in ("XJ", "SD")
        }

        def score(
            idx_xj: NDArray,
            _pool: NDArray = pool,
            _src: dict[str, NDArray] = src_mats,
        ) -> float:
            mask = np.zeros(len(_pool), bool)
            mask[idx_xj] = True
            protos = {"XJ": _pool[mask].mean(0), "SD": _pool[~mask].mean(0)}
            yt: list[str] = []
            yp: list[str] = []
            for r in ("XJ", "SD"):
                yp += list(nc_predict(_src[r], protos))
                yt += [r] * len(_src[r])
            return float(f1_score(np.array(yt), np.array(yp), average="macro", zero_division=0))

        obs = score(np.arange(n_xj))
        null = np.array([score(rng.permutation(len(pool))[:n_xj]) for _ in range(n_perm)])
        rows.append(
            {
                "预处理": prep,
                "域校正": corr,
                "2025面位口径": policy,
                "真标签宏F1": round(obs, 4),
                "置换零分布均值": round(float(null.mean()), 4),
                "置换零分布97.5分位": round(float(np.percentile(null, 97.5)), 4),
                "置换P": round(float((null >= obs).sum() + 1) / (n_perm + 1), 4),
                "置换次数": n_perm,
            }
        )
    return pd.DataFrame(rows)


def run_block_d_geometry(x: NDArray, feat: NDArray, fidx: FaceIndex) -> pd.DataFrame:
    """漂移几何：源→目标的位移里，类别无关的公共分量占多大。

    δ_R = c_R(2025) - c_R(2018/2019)，R ∈ {XJ, SD}。公共分量 = (δ_XJ+δ_SD)/2。
    cos(δ_XJ, δ_SD) 越接近 1、公共占比越高 → 位移越像整体测量系统偏置（换仪器/换协议），
    越不像逐产地各自不同的地理化学差异。**本表用到了 2025 的声明标签，属描述性刻画而非检验。**
    """
    rows: list[dict[str, Any]] = []

    def geom(mat: NDArray, tag: str, policy: str, renorm: bool) -> None:
        deltas = {}
        for r in ("XJ", "SD"):
            cs = np.mean([apple_matrix(mat, fidx, y, r, policy, renorm=renorm)[0].mean(0) for y in SRC_YEARS], axis=0)
            ct = apple_matrix(mat, fidx, TGT_YEAR, r, policy, renorm=renorm)[0].mean(0)
            deltas[r] = ct - cs
        common = (deltas["XJ"] + deltas["SD"]) / 2
        spec = {r: deltas[r] - common for r in deltas}
        cos = float(
            deltas["XJ"] @ deltas["SD"] / (np.linalg.norm(deltas["XJ"]) * np.linalg.norm(deltas["SD"]) + 1e-12)
        )
        nc = float(np.linalg.norm(common))
        ns = float(np.mean([np.linalg.norm(v) for v in spec.values()]))
        rows.append(
            {
                "空间": tag,
                "面位口径": policy,
                "cos(δ_XJ, δ_SD)": round(cos, 4),
                "公共位移模长": round(nc, 4),
                "类别特异位移模长(均)": round(ns, 4),
                "公共分量占比": round(nc / (nc + ns + 1e-12), 4),
            }
        )

    for policy in ("ALL", "EQ"):
        geom(x, "光谱空间(SNV)", policy, renorm=False)
        geom(feat, "编码器嵌入空间", policy, renorm=True)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 图
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig_a(df_a: pd.DataFrame) -> Any:
    def _plot(lang: str) -> Any:
        zh = lang == "zh"
        sub = df_a[(df_a["域校正"] == "none") & (df_a["类别空间"].str.startswith("XJ/SD/GS"))]
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, pol in enumerate(FACE_POLICIES):
            s = sub[sub["面位口径"] == pol].sort_values("alpha")
            ax.errorbar(
                s["alpha"],
                s["macro_f1"],
                yerr=s["macro_f1_sd"],
                marker="osd^v"[i],
                markersize=6,
                capsize=3,
                linewidth=1.8,
                color=WONG[i % len(WONG)],
                label=(FACE_LABEL_ZH if zh else FACE_LABEL_EN)[pol],
            )
        ax.axhline(1 / 3, color="#999999", linestyle=":", linewidth=1.2)
        ax.text(
            0.02,
            1 / 3 + 0.012,
            "3 类随机基线 0.333" if zh else "3-class chance 0.333",
            fontsize=9,
            color="#555555",
        )
        ax.set_xlabel("源质心锚定权重 α" if zh else "Source-anchoring weight α")
        ax.set_ylabel("LOEO-2025 宏 F1" if zh else "LOEO-2025 macro-F1")
        ax.set_title(
            "面位口径对源质心锚定塌陷无实质影响" if zh else "Facet alignment does not rescue source-anchored transfer"
        )
        ax.set_ylim(0, 0.85)
        ax.set_xticks(list(ALPHAS))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="upper right")
        fig.tight_layout()
        return fig

    return _plot


def make_fig_b(df_bt: pd.DataFrame, prespec_prep: str, prespec_corr: str) -> Any:
    def _plot(lang: str) -> Any:
        zh = lang == "zh"
        sub = df_bt[df_bt["面位口径"] == PRESPEC_FACE].copy()
        sub["cfg"] = sub["预处理"] + " / " + sub["域校正"].map(CORR_LABEL_ZH if zh else CORR_LABEL_EN)
        sub = sub.sort_values("二类XJvsSD宏F1")
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        colors = [
            WONG[5] if (r["预处理"] == prespec_prep and r["域校正"] == prespec_corr) else WONG[0]
            for _, r in sub.iterrows()
        ]
        ax.barh(sub["cfg"], sub["二类XJvsSD宏F1"], color=colors, height=0.68)
        ax.axvline(0.5, color="#999999", linestyle=":", linewidth=1.3)
        ax.text(0.505, -0.6, "二类随机基线 0.5" if zh else "2-class chance 0.5", fontsize=9, color="#555555")
        ax.set_xlabel(
            "2025 果指派到 2018/2019 质心的二类（XJ vs SD）宏 F1"
            if zh
            else "2-class (XJ vs SD) macro-F1, 2025 fruit → 2018/2019 centroids"
        )
        ax.set_title(
            f"域级校正口径决定恢复程度（面位固定为{FACE_LABEL_ZH[PRESPEC_FACE]}；橙=源域内部预选口径）"
            if zh
            else (
                f"Recovery is driven by domain-level correction "
                f"({FACE_LABEL_EN[PRESPEC_FACE]}; orange = pre-specified)"
            ),
            fontsize=11,
        )
        ax.set_xlim(0, 1.0)
        ax.grid(alpha=0.25, axis="x")
        fig.tight_layout()
        return fig

    return _plot


def make_fig_c(df_c: pd.DataFrame, df_cal: pd.DataFrame, prespec_prep: str, prespec_corr: str) -> Any:
    def _plot(lang: str) -> Any:
        zh = lang == "zh"
        sub = df_c[
            (df_c["预处理"] == prespec_prep)
            & (df_c["域校正"] == prespec_corr)
            & (df_c["面位口径"] == PRESPEC_FACE)
        ]
        cal = df_cal[(df_cal["预处理"] == prespec_prep) & (df_cal["域校正"] == prespec_corr)]
        labels_zh = list(sub["对比"]) + ["【源域标定】" + s for s in cal["对比"]]
        labels_en = [
            "SD anchor: SD vs GS (neg. ctrl)",
            "SD anchor: SD vs XJ",
            "XJ anchor: XJ vs GS (pos. ctrl)",
            "XJ anchor: XJ vs SD",
            "[calib] SD anchor: 2019 SD vs XJ",
            "[calib] SD anchor: 2019 SD vs SX",
            "[calib] XJ anchor: 2019 XJ vs SD",
            "[calib] XJ anchor: 2019 XJ vs SX",
        ]
        vals = list(sub["AUC"]) + list(cal["AUC"])
        lo = list(sub["CI95下"]) + list(cal["CI95下"])
        hi = list(sub["CI95上"]) + list(cal["CI95上"])
        labels = labels_zh if zh else labels_en
        y = np.arange(len(vals))[::-1]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        cols = [WONG[0]] * len(sub) + [WONG[2]] * len(cal)
        for yi, v, a, b, c in zip(y, vals, lo, hi, cols, strict=False):
            ax.plot([a, b], [yi, yi], color=c, linewidth=2.2)
            ax.plot(v, yi, "o", color=c, markersize=7)
        ax.axvline(0.5, color="#999999", linestyle=":", linewidth=1.3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel(
            "AUC（0.5 = 声明标签不携带任何源域特异信息）"
            if zh
            else "AUC (0.5 = declared label carries no source-specific signal)"
        )
        ax.set_title(
            f"抗域漂移的跨年标签互证（{prespec_prep} / {CORR_LABEL_ZH[prespec_corr]} / {FACE_LABEL_ZH[PRESPEC_FACE]}）"
            if zh
            else f"Drift-robust cross-year label cross-validation ({prespec_prep} / {CORR_LABEL_EN[prespec_corr]})",
            fontsize=11,
        )
        ax.grid(alpha=0.25, axis="x")
        fig.tight_layout()
        return fig

    return _plot


# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="39：拆分仪器/面位混淆与标签问题（LOEO-2025）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--encoder_seed", type=int, default=eu.PILOT_SEED)
    parser.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = parser.parse_args()

    device = eu.get_device(args.device)
    logger = eu.get_logger("39_face_instrument_confound")
    eu.set_random_seed(args.seed, logger)
    eu.enforce_publication_grade_device(device, stage=args.stage, logger=logger)
    seeds = list(eu.RECOMMENDED_SEEDS)
    eu.log_experiment_header(
        logger,
        {
            "任务": "拆分仪器/面位混淆与标签问题（L3-1）",
            "阶段": args.stage,
            "设备": str(device),
            "折": "LOEO-2025（源=2018+2019）",
            "K": K_SHOT,
            "种子": f"{eu.n_distinct_seeds(seeds)} 个正式种子 × {N_REPS} 次重采样",
            "面位口径": list(FACE_POLICIES),
            "域校正": list(CORRECTIONS),
            "预处理": list(SPEC_PREPS),
            "bootstrap": N_BOOT,
        },
    )
    run = eu.init_wandb(__file__, args.seed, eu.PROJECT_CODE, args.stage, vars(args))

    df = eu.load_all_elements_df(verify_sha256=True).reset_index(drop=True)
    wl_cols = eu.get_spectrum_columns(df)
    x_spec = df[wl_cols].to_numpy().astype(np.float64)
    fidx = FaceIndex(df)
    logger.log(f"数据：{x_spec.shape[0]} 条面位光谱 / {len(wl_cols)} 波段")
    logger.log(f"2025 面位分布：{pd.Series(fidx.face[fidx.year == TGT_YEAR]).value_counts().to_dict()}")

    ckpt_path = os.path.join(
        eu.CHECKPOINT_DIR, f"10_loeo_year_foldwise_seed{args.encoder_seed}_fold{TGT_YEAR}", "best.pt"
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"缺 fold-2025 编码器：{ckpt_path}（先跑 10_loeo_year_foldwise）")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder = ProtoEncoder(len(wl_cols), feat_dim=int(ckpt.get("feat_dim", 128)))
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.to(device).eval()
    feat = encode_batch(encoder, x_spec.astype(np.float32), device)
    logger.log(f"折限定编码器：{os.path.relpath(ckpt_path)}  嵌入 {feat.shape}")

    rng = np.random.default_rng(args.seed)
    with eu.track_compute("仪器/面位混淆拆分", logger, run) as budget:
        logger.log("\n【Block A】编码器空间：面位口径 × 域校正 × α 三向消融")
        df_a = run_block_a(feat, fidx, seeds, logger)
        base = df_a[
            (df_a["域校正"] == "none")
            & (df_a["面位口径"] == "ALL")
            & (df_a["类别空间"].str.startswith("XJ/SD/GS"))
        ].set_index("alpha")
        logger.log(
            "  复现校验（无校正 / 全 5 面 / 3 类）：α=0 {:.4f}（稿件 0.671）｜α=0.4 {:.4f}（稿件 0.414）"
            "｜α=1 {:.4f}（稿件 0.199）".format(
                base.loc[0.0, "macro_f1"], base.loc[0.4, "macro_f1"], base.loc[1.0, "macro_f1"]
            )
        )
        logger.log(
            "  α=1 逐类召回：XJ {:.3f}｜SD {:.3f}｜GS {:.3f}".format(
                base.loc[1.0, "recall_XJ"], base.loc[1.0, "recall_SD"], base.loc[1.0, "recall_GS"]
            )
        )

        logger.log("\n【Block B-1】源域内部 2018↔2019 迁移（不含 2025，作口径预选依据）")
        df_bs = run_block_b_source_internal(x_spec, fidx, logger)
        sel = df_bs.groupby(["预处理", "域校正"])["宏F1"].mean().sort_values(ascending=False)
        prespec_prep, prespec_corr = sel.index[0]
        logger.log(f"  预选口径（源域内部宏F1 最高）：{prespec_prep} / {prespec_corr}  = {sel.iloc[0]:.4f}")
        logger.log(f"  预选面位口径（由采集协议决定，非事后挑选）：{PRESPEC_FACE}")
        logger.log("  源域内部排序：\n" + sel.round(4).to_string())

        logger.log("\n【Block B-2】光谱空间：2025 果 → 2018/2019 质心指派")
        df_bt = run_block_b_target(x_spec, fidx, logger)

        logger.log("\n【Block C】抗域漂移标签互证 AUC（+ 甘肃阴性对照 + 源域标定）")
        df_c = run_block_c(x_spec, fidx, rng, logger)
        df_cal = run_block_c_calibration(x_spec, fidx, rng)

        logger.log("\n【Block D】反方向与漂移几何")
        df_dr = run_block_d_reverse(x_spec, fidx)
        df_dg = run_block_d_geometry(x_spec, feat, fidx)

        logger.log("\n【Block E】反方向的三道负控（两锚判决 / 甘肃冒名 / 标签置换）")
        df_e1 = run_block_e_two_anchor(x_spec, fidx)
        df_e2 = run_block_e_impostor(x_spec, fidx)
        df_e3 = run_block_e_permutation(x_spec, fidx, rng)
        logger.log("  [E3] 置换负控：\n" + df_e3.to_string(index=False))

    key_c = df_c[
        (df_c["预处理"] == prespec_prep) & (df_c["域校正"] == prespec_corr) & (df_c["面位口径"] == PRESPEC_FACE)
    ]
    logger.log("\n── 预选口径下的标签互证判定 ──")
    logger.log(key_c[["对比", "AUC", "CI95下", "CI95上", "P"]].to_string(index=False))
    logger.log("\n── 漂移几何 ──")
    logger.log(df_dg.to_string(index=False))

    figs = (
        make_fig_a(df_a),
        make_fig_b(df_bt, prespec_prep, prespec_corr),
        make_fig_c(df_c, df_cal, prespec_prep, prespec_corr),
    )
    for idx, fn in enumerate(figs):
        zh, en = eu.save_figure_bilingual(fn, "39_face_instrument_confound", idx)
        logger.log(f"图{chr(97 + idx)} → {os.path.basename(zh)} / {os.path.basename(en)}")

    # F1：产出论文数字的最终判决模型（预选口径下的源质心 + 域校正统计量）落盘
    z_pre = spectral_variant(x_spec, fidx, prespec_prep, prespec_corr)
    final_model = {
        "kind": "source-anchored nearest-centroid (declared-origin cross-year probe)",
        "prep": prespec_prep,
        "domain_correction": prespec_corr,
        "face_policy": PRESPEC_FACE,
        "source_centroids": {
            r: np.mean(
                [apple_matrix(z_pre, fidx, y, r, PRESPEC_FACE, renorm=False)[0].mean(0) for y in SRC_YEARS],
                axis=0,
            )
            for r in SRC_REGIONS
        },
        "wavelengths_nm": eu.get_spectrum_wavelengths(df, wl_cols),
    }
    ck = eu.save_checkpoint(
        final_model,
        "39_face_instrument_confound_prespec",
        {
            "arch": f"NearestCentroid(3 源类) on {prespec_prep} + 域{prespec_corr} + 面位{PRESPEC_FACE}",
            "params": f"centroids shape=(3, {z_pre.shape[1]})",
            "flops": "N/A（最近质心，非逐层前向）",
            "stage": args.stage,
            "training_hardware": str(device),  # 实际用到的设备，而非本机可用的最高档设备
            "deterministic_guarantee": "full（确定性质心，无随机训练）",
            "fold": "LOEO-2025",
            "selection_rule": "预处理由源域内部 2018↔2019 迁移宏F1 预选（不含 2025）；面位由采集协议预设为赤道面",
            "note": "本模型仅用于标签互证探针，不是论文的产地判别主模型",
        },
    )
    logger.log(f"checkpoint → {os.path.relpath(ck['ckpt_path'])}  sha256={ck['sha256'][:16]}…")

    budget_df = pd.DataFrame([{"seeds": eu.n_distinct_seeds(seeds), "reps": N_REPS, "n_boot": N_BOOT, **budget}])
    xlsx = eu.write_script_workbook(
        __file__,
        {
            0: ("A_面位域校正alpha三向消融", df_a),
            1: ("B1_源域内部2018_2019预选", df_bs),
            2: ("B2_光谱空间指派分布", df_bt),
            3: ("C_抗漂移标签互证AUC", df_c),
            4: ("C_源域内部标定", df_cal),
            5: ("D_反方向2025到源域", df_dr),
            6: ("D_漂移几何", df_dg),
            7: ("E1_两锚判决SD对GS", df_e1),
            8: ("E2_甘肃冒名对照", df_e2),
            9: ("E3_标签置换负控", df_e3),
            "compute": budget_df,
        },
    )
    logger.log(f"结果 → {xlsx}")
    logger.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
