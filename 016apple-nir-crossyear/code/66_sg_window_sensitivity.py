"""66_sg_window_sensitivity.py: 光谱质控（饱和谱）与 SG 平滑物理窗口宽度的敏感性。

背景一（饱和谱）：原始数据里有一小部分光谱在探测器上限处被截顶。2025 年的 CSV 为逐谱最小–最大归一化值
（每条谱最小值 0、最大值 1），正常谱只有一个波段取到 1；被截顶的谱则有一段连续波段并列取最大值。
2018/2019 为三次扫描平均的透射计数，16 位满量程 65 535。质控分两档（在原始数据上判，与建模无关）：
    严格档（截顶）:
        2025       : 全波段中与该谱最大值并列（差 < 1e-6）的波段数 ≥ --qc_tied_bands
        2018/2019  : 590–1001 nm 窗口内任一波段 ≥ --qc_counts 计数
    宽松档（截顶 + 动态范围压缩，仅 2025）: 另剔除 590–1001 nm 窗口内标准差低于 Tukey 下篱笆
        （Q1 − --qc_tukey_k × IQR，按全部 2025 光谱计）的光谱。这类谱窗口内最小值约 0.6–0.9（正常约 0.26），
        可能是感兴趣区内部分像素截顶后与未截顶像素平均的结果，也可能是果实本身的低对比度；只在敏感性分析里剔除。
被判出的光谱仍在冻结数据里（冻结数据未做此项质控）；本脚本给出剔除它们之后的对照。
背景二（SG 窗口）：现行预处理在各仪器原生网格上做 15 点、二阶 SG 平滑。2018/2019 光纤光谱仪原生间隔约
0.34 nm，15 点约 4.7 nm；2025 成像仪的数据本就落在 229 波段公共网格上（间隔约 1.80 nm），15 点约 25.2 nm。

变体（其余步骤 裁剪 → 三次样条对齐到公共网格 → SNV 与 01preprocess.py 逐行相同）:
    V0  冻结数据（全部 4529 条，稿件现用口径；K-Shot 结果须复现 Formal 归档）
    Q0  严格档质控，现行 SG（2018/2019 原生 15 点 ≈4.7 nm；2025 公共网格 15 点 ≈25.2 nm）
    QB  宽松档质控，现行 SG
    Q1  严格档质控，等宽约 25 nm（2018/2019 原生 75 点；2025 公共网格 15 点）
    Q2  严格档质控，等宽约 7 nm（2018/2019 原生 23 点；2025 公共网格 5 点）
    Q3  严格档质控，不平滑
自校验：①a 按现行 SG 重建的全部行，须与在本机直接调用 01preprocess.py 各函数得到的结果逐位相同（max|Δ| = 0）；
       ①b 与冻结文件的差异（严格档保留行）只作记录，上限 --v0_tol：冻结文件由同一流程在另一环境生成，
           float32 舍入经 SNV 在低方差谱上放大，量级 1e-5；
       ② V0 的 K-Shot 路线复现 Formal 归档：最近质心逐 (折, K, 种子) 格 ≤ --nc_tol，SVM 与 PLS-DA 按 (折, K)
          5 种子均值 ≤ --selfcheck_tol。任一不过即拒绝写出。
每种变体报告：
    (a) 域差距：年份均值光谱两两之间的欧氏距离；同一年内各产地均值光谱两两之间的欧氏距离；二者之比；
    (b) 只看支持集的光谱空间 K-Shot 路线（SVM、PLS-DA、最近质心）的 LOEO 宏 F1（情节由 20 号脚本生成）；
        Q0−V0、QB−V0 为两档质控的效应，Q1/Q2/Q3−Q0 为 SG 窗口的效应。
另把 Q0、QB、Q1、Q3 写到 03data/processed/variants/（float32 npz：保留行掩码、光谱、冻结数据行序指纹），
供 GPU 上的编码器泄漏对照（64 号脚本 --data_variant）复用。

运行方式:
    cd <项目根>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/66_sg_window_sensitivity.py --workers 6

输出文件:
    04outputs/66_sg_window_sensitivity.xlsx          — 质控清单、变体定义、自校验、域差距、K-Shot 结果
    03data/processed/variants/all_9elements__<V>.npz — Q0、QB、Q1、Q3 光谱（float32）+ 保留行掩码 + 行序指纹
    03data/processed/variants/variants-datasheet.md  — 变体数据卡（生成方式、质控判据、SHA-256）
    05logs/66_sg_window_sensitivity_YY-MM-DD_HHMMSS.log

设备说明:
    纯 CPU（numpy/scipy/sklearn），与表 S6 的集成列同一口径（--stage pilot），5 个正式种子、每折每 K 50 次重采样。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import export_utils as eu  # noqa: E402
from protonet import build_year_region_data  # noqa: E402

NDArray = npt.NDArray[Any]


def _load_module(alias: str, filename: str) -> Any:
    path = os.path.join(_THIS_DIR, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRE = _load_module("pre01", "01preprocess.py")
P20 = _load_module("panel20", "20_unified_singleface_kshot.py")

FOLDS = (2018, 2019, 2025)
METHODS = ["SVM-Kshot", "PLS-DA-Kshot", "NearestCentroid"]
# 变体：(2018/2019 原生网格 SG 点数, 2025 公共网格 SG 点数)；None = 不平滑
REBUILT: dict[str, dict[str, Any]] = {
    "Q0": {
        "old": 15,
        "new": 15,
        "qc": "严格",
        "说明": "严格档质控；现行 SG：2018/2019 原生 15 点，2025 公共网格 15 点",
    },
    "QB": {"old": 15, "new": 15, "qc": "宽松", "说明": "宽松档质控（截顶 + 2025 动态范围压缩）；现行 SG"},
    "Q1": {
        "old": 75,
        "new": 15,
        "qc": "严格",
        "说明": "严格档质控；等宽约 25 nm：2018/2019 原生 75 点，2025 公共网格 15 点",
    },
    "Q2": {
        "old": 23,
        "new": 5,
        "qc": "严格",
        "说明": "严格档质控；等宽约 7 nm：2018/2019 原生 23 点，2025 公共网格 5 点",
    },
    "Q3": {"old": None, "new": None, "qc": "严格", "说明": "严格档质控；不平滑：只裁剪、三次样条对齐、SNV"},
}
ALL_VARIANTS = ["V0", *REBUILT]
PERSIST = ("Q0", "QB", "Q1", "Q3")
REF = {"Q0": "V0", "QB": "V0", "Q1": "Q0", "Q2": "Q0", "Q3": "Q0"}
VARIANT_DIR = os.path.join(eu.PROCESSED_DIR, "variants")


# ═══════════════════════════════════════════════════════════════════════════════
# 一、质控与变体重建（与 01preprocess.rebuild_processed 逐行同构，只换 SG 窗口）
# ═══════════════════════════════════════════════════════════════════════════════
def build_variant(
    sg_old: int | None, sg_new: int | None, qc_tied: int, qc_counts: float, tukey_k: float, logger: eu.Logger
) -> tuple[pd.DataFrame, dict[str, NDArray], dict[str, float]]:
    """返回 (与冻结数据同行序的全部行, {严格, 宽松} 质控标记, SG 物理跨度)。"""
    df25_raw = pd.read_csv(PRE.RAW_2025)
    spec_cols_25 = [c for c in df25_raw.columns if "nm" in str(c)]
    wl_25 = np.array([float(c.replace("nm", "")) for c in spec_cols_25])
    mask25 = (wl_25 >= PRE.OVERLAP_LOW) & (wl_25 <= PRE.OVERLAP_HIGH)
    common_wl = wl_25[mask25]
    common_cols_25 = np.array(spec_cols_25)[mask25].tolist()
    wl_col_names = [f"{w:.2f}nm" for w in common_wl]
    full25 = df25_raw[spec_cols_25].to_numpy(dtype=float)
    tied = (full25 >= full25.max(axis=1, keepdims=True) - 1e-6).sum(axis=1)
    sat25 = np.asarray(tied >= qc_tied)
    x25_raw = df25_raw[common_cols_25].to_numpy().astype(np.float32)
    sd_win = df25_raw[common_cols_25].to_numpy(dtype=float).std(axis=1)
    q1, q3 = np.percentile(sd_win, [25, 75])
    compressed25 = np.asarray(sd_win < q1 - tukey_k * (q3 - q1))
    origin_25 = df25_raw["图像名称"].str[:2].to_numpy()
    apple_id_25 = df25_raw["实际苹果编号"].str.split("-").str[0].to_numpy()
    face_25 = df25_raw["实际苹果编号"].str.split("-").str[1].to_numpy()

    spans: dict[str, float] = {"2025窗口SD_Tukey下篱笆": float(q1 - tukey_k * (q3 - q1))}
    parts: list[pd.DataFrame] = []
    flags: list[NDArray] = []
    broad: list[NDArray] = []
    for year_str, raw_path in (("2018", PRE.RAW_2018), ("2019", PRE.RAW_2019)):
        x_raw, wl_src, orig, ssc = PRE.load_old_year(raw_path, year_str, logger)
        crop = (wl_src >= PRE.OVERLAP_LOW - 1) & (wl_src <= PRE.OVERLAP_HIGH + 1)
        x_crop, wl_crop = x_raw[:, crop], wl_src[crop]
        flags.append(np.asarray((x_crop >= qc_counts).any(axis=1)))
        broad.append(flags[-1])
        x_s = PRE.sg_smooth(x_crop, window=sg_old) if sg_old is not None else x_crop
        spans[f"SG跨度_{year_str}_nm"] = float((sg_old - 1) * np.diff(wl_crop).mean()) if sg_old else 0.0
        x = PRE.snv(PRE.align_to_target_wavelengths(x_s, wl_crop, common_wl))
        ids = PRE.make_synthetic_apple_ids(orig, year_str)
        parts.append(PRE.build_element_df(x, orig, year_str, wl_col_names, ids, ssc=ssc))
    x25 = PRE.snv(PRE.sg_smooth(x25_raw, window=sg_new) if sg_new is not None else x25_raw)
    spans["SG跨度_2025_nm"] = float((sg_new - 1) * np.diff(common_wl).mean()) if sg_new else 0.0
    parts.append(PRE.build_element_df(x25, origin_25, "2025", wl_col_names, apple_ids=apple_id_25, faces=face_25))
    flags.append(sat25)
    broad.append(sat25 | compressed25)
    df_all = pd.concat(parts, axis=0, ignore_index=True)
    df_all["年份"] = df_all["年份"].astype(int)
    return df_all, {"严格": np.concatenate(flags), "宽松": np.concatenate(broad)}, spans


def reference_rebuild(logger: eu.Logger) -> NDArray:
    """在本机直接调用 01preprocess.py 的函数（与 rebuild_processed 同序）得到全部行的光谱，供自校验①a。"""
    df25_raw = pd.read_csv(PRE.RAW_2025)
    spec_cols_25 = [c for c in df25_raw.columns if "nm" in str(c)]
    wl_25 = np.array([float(c.replace("nm", "")) for c in spec_cols_25])
    mask25 = (wl_25 >= PRE.OVERLAP_LOW) & (wl_25 <= PRE.OVERLAP_HIGH)
    common_wl = wl_25[mask25]
    x25_raw = df25_raw[np.array(spec_cols_25)[mask25].tolist()].to_numpy().astype(np.float32)
    blocks = []
    for year_str, raw_path in (("2018", PRE.RAW_2018), ("2019", PRE.RAW_2019)):
        x_raw, wl_src, _, _ = PRE.load_old_year(raw_path, year_str, logger)
        blocks.append(PRE.preprocess_old(x_raw, wl_src, common_wl, year_str, logger))
    blocks.append(PRE.snv(PRE.sg_smooth(x25_raw)))
    return np.vstack(blocks).astype(np.float64)


def row_fingerprint(df: pd.DataFrame) -> str:
    """行序指纹：年份/产地/苹果编号/面位 逐行拼接后的 SHA-256。"""
    key = "\n".join(
        f"{a}|{b}|{c}|{d}" for a, b, c, d in df[["年份", "产地", "苹果编号", "面位"]].itertuples(index=False)
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# 二、域差距
# ═══════════════════════════════════════════════════════════════════════════════
def domain_gap(df: pd.DataFrame, wl_cols: list[str]) -> dict[str, float]:
    x = df[wl_cols].to_numpy(dtype=float)
    out: dict[str, float] = {}
    year_mean = {y: x[(df["年份"] == y).to_numpy()].mean(axis=0) for y in FOLDS}
    for a, b in combinations(FOLDS, 2):
        out[f"年份均值距离_{a}-{b}"] = float(np.linalg.norm(year_mean[a] - year_mean[b]))
    within: list[float] = []
    for y in FOLDS:
        sub = df["年份"] == y
        regs = sorted(df.loc[sub, "产地"].unique())
        means = {r: x[(sub & (df["产地"] == r)).to_numpy()].mean(axis=0) for r in regs}
        for r1, r2 in combinations(regs, 2):
            d = float(np.linalg.norm(means[r1] - means[r2]))
            out[f"同年产地均值距离_{y}_{r1}-{r2}"] = d
            within.append(d)
    out["同年产地均值距离_平均"] = float(np.mean(within))
    out["历史年-2025年份距离_平均"] = float(np.mean([out["年份均值距离_2018-2025"], out["年份均值距离_2019-2025"]]))
    out["2018-2019年份距离"] = out["年份均值距离_2018-2019"]
    out["跨年(历史→2025)/同年产地 距离比"] = out["历史年-2025年份距离_平均"] / out["同年产地均值距离_平均"]
    out["跨年(2018↔2019)/同年产地 距离比"] = out["2018-2019年份距离"] / out["同年产地均值距离_平均"]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 三、K-Shot 光谱空间路线（并行单元：变体 × 折 × K × 种子）
# ═══════════════════════════════════════════════════════════════════════════════
_W: dict[str, Any] = {}


def _worker_init(payload: dict[str, bytes]) -> None:
    _W["year_data"] = {}
    for v, blob in payload.items():
        df = pd.read_pickle(io.BytesIO(blob))
        _W["year_data"][v] = build_year_region_data(df, eu.get_spectrum_columns(df))


def _task(task: tuple[str, int, int, int, int]) -> tuple[tuple[str, int, int, int], dict[str, float]]:
    v, y, k, fseed, n_reps = task
    yd = _W["year_data"][v]
    acc: dict[str, list[float]] = {m: [] for m in METHODS}
    for i in range(n_reps):
        s = fseed + i
        acc["SVM-Kshot"].append(P20._eval_sklearn(yd, y, k, s, "svm")[0])
        acc["PLS-DA-Kshot"].append(P20._eval_sklearn(yd, y, k, s, "pls")[0])
        acc["NearestCentroid"].append(P20.eval_nearest_centroid(yd, y, k, s)[0])
    return (v, y, k, fseed), {m: float(np.mean(vals)) for m, vals in acc.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="66_sg_window_sensitivity：饱和谱质控与 SG 物理窗口宽度敏感性")
    parser.add_argument("--k_list", default="1,3,5,10,20,50")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed_list", default=",".join(str(s) for s in eu.RECOMMENDED_SEEDS))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--qc_tied_bands", type=int, default=10, help="2025：与最大值并列的波段数达到此值判为截顶")
    parser.add_argument("--qc_counts", type=float, default=64000.0, help="2018/2019：窗口内任一波段达此计数判为饱和")
    parser.add_argument("--qc_tukey_k", type=float, default=1.5, help="宽松档：2025 窗口标准差的 Tukey 下篱笆系数")
    parser.add_argument("--v0_tol", type=float, default=1e-4)
    parser.add_argument("--nc_tol", type=float, default=1e-4)
    parser.add_argument("--selfcheck_tol", type=float, default=5e-4)
    args = parser.parse_args()
    k_values = [int(k) for k in args.k_list.split(",")]
    formal_seeds = [int(s) for s in args.seed_list.split(",")]

    logger = eu.get_logger("66_sg_window_sensitivity")
    eu.set_random_seed(formal_seeds[0], logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "饱和谱质控与 SG 平滑物理窗口宽度敏感性",
            "阶段": "pilot（纯 CPU）",
            "正式种子": formal_seeds,
            "每折每K重采样": args.seeds,
            "K": k_values,
            "质控判据": f"严格档：2025 并列最大值波段 ≥ {args.qc_tied_bands}，2018/2019 计数 ≥ {args.qc_counts:.0f}；"
            f"宽松档另加 2025 窗口标准差 < Q1 − {args.qc_tukey_k}×IQR",
        },
    )
    frozen = eu.load_all_elements_df(verify_sha256=True)
    wl_cols = eu.get_spectrum_columns(frozen)

    # ── 1. 重建变体、质控标记、自校验① ─────────────────────────────────────────
    variants: dict[str, pd.DataFrame] = {"V0": frozen}
    def_rows: list[dict[str, Any]] = [{"变体": "V0", "说明": "冻结数据（全部光谱，稿件现用口径）"}]
    tiers: dict[str, NDArray] | None = None
    for v, spec in REBUILT.items():
        df_v, flags, spans = build_variant(
            spec["old"], spec["new"], args.qc_tied_bands, args.qc_counts, args.qc_tukey_k, logger
        )
        if list(eu.get_spectrum_columns(df_v)) != list(wl_cols) or len(df_v) != len(frozen):
            raise RuntimeError(f"{v} 的波段列或行数与冻结数据不一致")
        for col in ("年份", "产地", "苹果编号", "面位"):
            if not (df_v[col].astype(str).to_numpy() == frozen[col].astype(str).to_numpy()).all():
                raise RuntimeError(f"{v} 的行序（{col}）与冻结数据不一致")
        if tiers is None:
            tiers = flags
        elif not all((tiers[t] == flags[t]).all() for t in tiers):
            raise RuntimeError("各变体的质控标记不一致")
        if v == "Q0":
            ref_now = reference_rebuild(logger)
            mine = df_v[wl_cols].to_numpy(dtype=float)
            exact = float(np.abs(mine - ref_now).max())
            dev = np.abs(mine - frozen[wl_cols].to_numpy(dtype=float))
            kept = ~flags["严格"]
            max_kept = float(dev[kept].max())
            worst = int(np.flatnonzero(kept)[dev[kept].max(axis=1).argmax()])
            logger.log(
                f"── 自校验①a：本脚本重建 vs 本机直接调用 01preprocess.py，全部行 max|Δ|={exact:.2e}（须为 0）\n"
                f"── 自校验①b：vs 冻结文件，严格档保留行 max|Δ|={max_kept:.2e}（上限 {args.v0_tol:.0e}；"
                f"最大处 {frozen.loc[worst, '年份']} {frozen.loc[worst, '苹果编号']}-{frozen.loc[worst, '面位']}）；"
                f"截顶行 max|Δ|={float(dev[~kept].max()):.2e}（恒定谱，照录）"
            )
            if exact > 0.0 or max_kept > args.v0_tol:
                logger.close()
                raise RuntimeError(f"自校验① FAIL：①a={exact:.2e}，①b={max_kept:.2e}，拒绝写出。")
        keep = ~flags["宽松" if spec["qc"] == "宽松" else "严格"]
        variants[v] = df_v[keep].reset_index(drop=True)
        def_rows.append(
            {
                "变体": v,
                "质控档": spec["qc"],
                "保留光谱数": int(keep.sum()),
                "说明": spec["说明"],
                **{k: round(val, 4) for k, val in spans.items()},
            }
        )
    if tiers is None:
        raise RuntimeError("未生成任何变体")
    ident = frozen[["年份", "产地", "苹果编号", "面位"]]
    qc_df = (
        ident.assign(质控类别=np.where(tiers["严格"], "截顶（严格档）", "动态范围压缩（仅宽松档）"))
        .loc[tiers["宽松"]]
        .reset_index(drop=True)
    )
    sums = []
    for tier, mask in tiers.items():
        g = (
            frozen.loc[mask]
            .groupby(["年份", "产地"])
            .agg(**{f"{tier}_剔除光谱数": ("苹果编号", "size"), f"{tier}_涉及苹果数": ("苹果编号", "nunique")})
            .reset_index()
        )
        sums.append(g)
    tot = (
        frozen.groupby(["年份", "产地"])
        .agg(光谱总数=("苹果编号", "size"), 苹果总数=("苹果编号", "nunique"))
        .reset_index()
    )
    qc_sum = tot.merge(sums[0], on=["年份", "产地"], how="left").merge(sums[1], on=["年份", "产地"], how="left")
    qc_sum = qc_sum.fillna(0)
    for tier, mask in tiers.items():
        whole = frozen.assign(_f=mask).groupby(["年份", "产地", "苹果编号"])["_f"].all()
        logger.log(f"── 质控（{tier}档）：{int(mask.sum())} 条光谱；因此整果剔除 {int(whole.sum())} 个")
    logger.log(qc_sum.to_string(index=False))

    # ── 2. 域差距 ──────────────────────────────────────────────────────────────
    gap_df = pd.DataFrame(
        [{"变体": v, **{k: round(val, 4) for k, val in domain_gap(df, wl_cols).items()}} for v, df in variants.items()]
    )
    logger.log(
        "\n── 域差距 ──\n"
        + gap_df[
            [
                "变体",
                "历史年-2025年份距离_平均",
                "2018-2019年份距离",
                "同年产地均值距离_平均",
                "跨年(历史→2025)/同年产地 距离比",
                "跨年(2018↔2019)/同年产地 距离比",
            ]
        ].to_string(index=False)
    )

    # ── 3. K-Shot 光谱空间路线 ─────────────────────────────────────────────────
    payload: dict[str, bytes] = {}
    for v, df in variants.items():
        buf = io.BytesIO()
        df.to_pickle(buf)
        payload[v] = buf.getvalue()
    tasks = [(v, y, k, s, args.seeds) for v in ALL_VARIANTS for y in FOLDS for k in k_values for s in formal_seeds]
    cell: dict[tuple[str, int, int, int], dict[str, float]] = {}
    with eu.track_compute("质控与 SG 窗口敏感性 K-Shot", logger) as budget:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=ctx, initializer=_worker_init, initargs=(payload,)
        ) as ex:
            for n_done, (key, res) in enumerate(ex.map(_task, tasks), start=1):
                cell[key] = res
                if n_done % 90 == 0 or n_done == len(tasks):
                    logger.log(f"  完成 {n_done}/{len(tasks)} 个单元")

    # 自校验②：V0 复现 Formal 归档
    formal = _load_module("panel42", "42_ensemble_kshot.py").load_formal_panel()
    if formal is None:
        logger.close()
        raise RuntimeError("自校验②无法执行：未找到 20 号脚本的 Formal 归档，拒绝写出。")

    def archived(m: str, y: int, k: int, s: int) -> float | None:
        hit = formal[(formal["方法"] == m) & (formal["目标年份"] == y) & (formal["K"] == k) & (formal["种子"] == s)]
        return float(hit["F1均值"].iloc[0]) if len(hit) == 1 else None

    check_rows: list[dict[str, Any]] = []
    for y in FOLDS:
        for k in k_values:
            for m in METHODS:
                mine = [cell[("V0", y, k, s)][m] for s in formal_seeds]
                theirs = [archived(m, y, k, s) for s in formal_seeds]
                th = [float(t) for t in theirs if t is not None]
                if len(th) != len(mine):
                    continue
                per_cell = [abs(a - b) for a, b in zip(mine, th, strict=True)]
                check_rows.append(
                    {
                        "方法": m,
                        "目标年份": y,
                        "K": k,
                        "逐格max|Δ|": round(max(per_cell), 5),
                        "5种子均值|Δ|": round(abs(float(np.mean(mine)) - float(np.mean(th))), 5),
                    }
                )
    check_df = pd.DataFrame(check_rows)
    nc_worst = float(check_df[check_df["方法"] == "NearestCentroid"]["逐格max|Δ|"].max())
    mean_worst = float(check_df[check_df["方法"] != "NearestCentroid"]["5种子均值|Δ|"].max())
    logger.log(f"── 自校验②：最近质心逐格 max|Δ|={nc_worst:.5f}；SVM/PLS-DA 按折×K 均值 max|Δ|={mean_worst:.5f}")
    complete = len(check_df) == len(FOLDS) * len(k_values) * len(METHODS)
    if not complete or nc_worst > args.nc_tol or mean_worst > args.selfcheck_tol:
        logger.log(check_df.to_string(index=False))
        logger.close()
        raise RuntimeError("自校验② FAIL：V0 未复现 Formal 归档，拒绝写出。")

    def fold_mean(v: str, y: int, k: int, m: str) -> float:
        return float(np.mean([cell[(v, y, k, s)][m] for s in formal_seeds]))

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for v in ALL_VARIANTS:
        ref = REF.get(v, "")
        for k in k_values:
            srow: dict[str, Any] = {"变体": v, "K": k}
            frow: dict[str, Any] = {"变体": v, "K": k}
            for m in METHODS:
                vals = [fold_mean(v, y, k, m) for y in FOLDS]
                srow[m] = round(float(np.mean(vals)), 4)
                if v != "V0":
                    base = float(np.mean([fold_mean(ref, y, k, m) for y in FOLDS]))
                    srow[f"{m}_与{ref}之差"] = round(float(np.mean(vals)) - base, 4)
                for y, val in zip(FOLDS, vals, strict=True):
                    frow[f"{m}@{y}"] = round(val, 4)
            summary_rows.append(srow)
            fold_rows.append(frow)
    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    logger.log(
        "\n── K-Shot 三折均值（变体 × K；Q0、QB 对 V0 比，Q1–Q3 对 Q0 比）──\n" + summary_df.to_string(index=False)
    )

    # ── 4. 落盘变体光谱（供 GPU 泄漏对照复用）─────────────────────────────────
    os.makedirs(VARIANT_DIR, exist_ok=True)
    fp = row_fingerprint(frozen)
    card_lines = [
        "# 质控与 SG 窗口变体光谱（由 02code/66_sg_window_sensitivity.py 生成，下游只读）",
        "",
        f"- 严格档质控：2025 与该谱最大值并列的波段数 ≥ {args.qc_tied_bands}；"
        f"2018/2019 在 590–1001 nm 窗口内任一波段 ≥ {args.qc_counts:.0f} 计数；共 {int(tiers['严格'].sum())} 条",
        f"- 宽松档质控：严格档 + 2025 窗口标准差 < Q1 − {args.qc_tukey_k}×IQR；共 {int(tiers['宽松'].sum())} 条",
        "- 冻结数据 `03data/processed/all_9elements.csv` 的行序指纹（年份|产地|苹果编号|面位 的 SHA-256）：",
        f"  `{fp}`",
        "- 每个 npz 含 `keep_mask`（冻结数据各行是否保留）、`spectra`（保留行 × 229 波段，float32）、"
        "`wavelength_cols`、`row_fingerprint`",
        "- 用法：读入冻结数据，核对行序指纹，按 `keep_mask` 取行，再以 `spectra` 替换波段列"
        "（64 号脚本 `--data_variant`）",
        "",
        "| 变体 | 说明 | 文件 | SHA-256 |",
        "|---|---|---|---|",
    ]
    for v in PERSIST:
        path = os.path.join(VARIANT_DIR, f"all_9elements__{v}.npz")
        np.savez_compressed(
            path,
            keep_mask=~tiers["宽松" if REBUILT[v]["qc"] == "宽松" else "严格"],
            spectra=variants[v][wl_cols].to_numpy(dtype=np.float32),
            wavelength_cols=np.array(wl_cols),
            row_fingerprint=np.array(fp),
        )
        card_lines.append(f"| {v} | {REBUILT[v]['说明']} | `{os.path.basename(path)}` | `{eu.compute_sha256(path)}` |")
        logger.log(f"  变体光谱 → {path}")
    with open(os.path.join(VARIANT_DIR, "variants-datasheet.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(card_lines) + "\n")

    budget_df = pd.DataFrame(
        [{"正式种子数": eu.n_distinct_seeds(tuple(formal_seeds)), "每折每K重采样": args.seeds, **budget}]
    )
    out_path = os.path.join(eu.OUTPUT_DIR, "66_sg_window_sensitivity.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        qc_sum.to_excel(writer, sheet_name=eu.table_sheet_name(0, "质控汇总"), index=False)
        qc_df.to_excel(writer, sheet_name=eu.table_sheet_name(1, "饱和光谱清单"), index=False)
        pd.DataFrame(def_rows).to_excel(writer, sheet_name=eu.table_sheet_name(2, "变体定义"), index=False)
        check_df.to_excel(writer, sheet_name=eu.table_sheet_name(3, "自校验"), index=False)
        gap_df.to_excel(writer, sheet_name=eu.table_sheet_name(4, "域差距"), index=False)
        summary_df.to_excel(writer, sheet_name=eu.table_sheet_name(5, "KShot三折均值"), index=False)
        fold_df.to_excel(writer, sheet_name=eu.table_sheet_name(6, "KShot折级均值"), index=False)
        budget_df.to_excel(writer, sheet_name="compute", index=False)
    logger.log(f"\n结果表 → {out_path}")
    logger.close()


if __name__ == "__main__":
    main()
