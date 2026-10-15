"""
01preprocess.py: 苹果 NIR 光谱三年数据预处理 + 数据集卡（016 唯一 01 段位数据入口）。

本脚本为项目唯一的预处理入口，合并原 02_preprocess.py 的
原始 → 处理流水线（全部为逐样本 L1 变换，无跨样本 fit 型 L2），并保留数据集卡（§4.6）
与 SHA-256 完整性校验。

预处理流水线（逐样本 L1，§4.5 白名单）:
    - 2018/2019（Excel，原始计数值）+ 2025（CSV，反射率）三年数据加载；
    - 波长裁剪到公共重叠区(590–1001nm) → SG 平滑(window=15,poly=2,axis=1，逐样本) →
      三次样条插值对齐到 2025 的 229 波段网格(逐样本) → SNV(逐样本，行内均值方差)；
    - 构建 9 元素结构化数据((年份,产地)×9)，每行附 element_id/苹果编号/面位/SSC；
    - 2025 额外输出苹果级(5 面位均值)聚合；产物统一写入 03data/processed/。
    上述变换的输出均不随"看了多少样本"改变 → 满足 §4.5 L1 边界。降维/标准化等 L2
    （仅用于可视化或建模）留在 03_eda.py / 折内 Pipeline，不在本脚本。

运行方式:
    cd <project root>
    python 02code/01preprocess.py              # 默认：校验冻结产物 + 生成数据集卡
    python 02code/01preprocess.py --rebuild     # 从原始数据再生 processed/（覆盖，附差异报告）
    python 02code/01preprocess.py --plot        # 生成预处理前后对比图 01preprocess-图a

输出文件:
    04outputs/01preprocess-图a.pdf / -fig-a.pdf    — 预处理前后对比图（--plot 时生成）
    03data/processed/all_9elements.csv             — 9 元素面位级光谱（--rebuild 时再生）
    03data/processed/apple_index.csv               — 苹果粒度索引（--rebuild 时再生）
    03data/processed/2025_apple_level.csv          — 2025 苹果级聚合（--rebuild 时再生）
    03data/processed/all_9elements-datasheet.md    — 9 元素面位级数据集卡
    03data/processed/apple_index-datasheet.md       — 苹果粒度索引数据集卡
    03data/processed/2025_apple_level-datasheet.md  — 2025 苹果级聚合数据集卡
    05logs/01preprocess_YY-MM-DD_HHMMSS.log         — 运行日志
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import export_utils as eu

NDArray = npt.NDArray[Any]

# ─── 预处理全局参数 ───────────────────────────────────────────────────────────
SG_WINDOW = 15  # SG 平滑窗口（奇数）
SG_POLY = 2  # SG 多项式阶数
OVERLAP_LOW = 590.0  # 公共重叠波长下限 (nm)
OVERLAP_HIGH = 1001.5  # 公共重叠波长上限 (nm)

# 2018/2019 产地标签映射（Z 列：1=新疆, 2=山东, 3=陕西）
LABEL_MAP_OLD = {1: "XJ", 2: "SD", 3: "SX"}

# 9 元素定义（(年份, 产地) → E1..E9）
ELEMENTS: list[tuple[str, str]] = [
    ("2018", "XJ"),
    ("2018", "SD"),
    ("2018", "SX"),
    ("2019", "XJ"),
    ("2019", "SD"),
    ("2019", "SX"),
    ("2025", "XJ"),
    ("2025", "SD"),
    ("2025", "GS"),
]
ELEMENT_ID: dict[tuple[str, str], str] = {e: f"E{i + 1}" for i, e in enumerate(ELEMENTS)}

# 原始数据文件（位于 03data/ 顶层）
RAW_2018 = os.path.join(eu.DATA_DIR, "01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx")
RAW_2019 = os.path.join(eu.DATA_DIR, "1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx")
RAW_2025 = os.path.join(eu.DATA_DIR, "新疆-山东-甘肃苹果光谱数据2025.csv")


# ─── 逐样本 L1 变换 ───────────────────────────────────────────────────────────
def snv(x: NDArray) -> NDArray:
    """标准正态变量变换（SNV）：逐行（逐样本）标准化，不跨样本统计。"""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return np.asarray((x - mu) / sd)


def sg_smooth(x: NDArray, window: int = SG_WINDOW, poly: int = SG_POLY) -> NDArray:
    """Savitzky-Golay 平滑（逐行固定卷积，不跨样本）。"""
    return np.asarray(savgol_filter(x, window_length=window, polyorder=poly, axis=1))


def align_to_target_wavelengths(x: NDArray, src_wl: NDArray, tgt_wl: NDArray) -> NDArray:
    """将源域光谱逐样本三次样条插值到目标波长网格。"""
    interp = interp1d(src_wl, x, kind="cubic", axis=1, bounds_error=False, fill_value="extrapolate")
    return np.asarray(interp(tgt_wl))


def make_synthetic_apple_ids(origins: NDArray, year_str: str) -> list[str]:
    """2018/2019 每行即一果，按产地顺序生成唯一合成苹果ID（年份_产地_序号）。"""
    counters: dict[str, int] = {}
    ids: list[str] = []
    for o in origins:
        counters[o] = counters.get(o, 0) + 1
        ids.append(f"{year_str}_{o}_{counters[o]:04d}")
    return ids


def build_element_df(
    x: NDArray,
    origins: NDArray,
    year_str: str,
    wl_col_names: list[str],
    apple_ids: list[str] | NDArray | None = None,
    faces: NDArray | None = None,
    ssc: NDArray | None = None,
) -> pd.DataFrame:
    """为单年份数据构建 DataFrame（列序：年份/产地/波段.../element_id/苹果编号/面位/SSC）。"""
    df = pd.DataFrame(x, columns=wl_col_names)
    df.insert(0, "产地", origins)
    df.insert(0, "年份", year_str)
    df["element_id"] = [ELEMENT_ID.get((year_str, o), "UNKNOWN") for o in origins]
    if apple_ids is not None:
        df["苹果编号"] = list(apple_ids)
    else:
        df["苹果编号"] = [f"{year_str}_sample_{i:04d}" for i in range(len(x))]
    df["面位"] = faces if faces is not None else "单面"
    if ssc is not None:
        df["SSC"] = ssc
    return df


def load_old_year(filepath: str, year_str: str, logger: eu.Logger) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """加载 2018/2019 Excel（首行别名行，Z 列产地 1/2/3），返回 (X_raw, wl, origins, ssc)。"""
    logger.log(f"加载 {year_str} 年：{os.path.basename(filepath)}")
    df = pd.read_excel(filepath, header=0)
    z_col = df.columns[-1]
    y_col = df.columns[-2]
    spec_cols = df.columns[:-2]
    valid_mask = df[z_col].isin([1, 2, 3])
    df_valid = df[valid_mask].reset_index(drop=True)
    wl = np.array([float(c) for c in spec_cols], dtype=np.float32)
    x = df_valid[spec_cols].to_numpy().astype(np.float32)
    origins = df_valid[z_col].map(LABEL_MAP_OLD).to_numpy()
    ssc = df_valid[y_col].to_numpy().astype(np.float32)
    logger.log(f"  有效样本={len(df_valid)} 波段={len(wl)} 范围={wl[0]:.1f}–{wl[-1]:.1f}nm")
    return x, wl, origins, ssc


def preprocess_old(x_raw: NDArray, wl_src: NDArray, common_wl: NDArray, label: str, logger: eu.Logger) -> NDArray:
    """2018/2019：裁剪重叠区 → SG → 插值到 common_wl → SNV（全逐样本）。"""
    mask = (wl_src >= OVERLAP_LOW - 1) & (wl_src <= OVERLAP_HIGH + 1)
    x_crop = x_raw[:, mask]
    wl_crop = wl_src[mask]
    x_sg = sg_smooth(x_crop)
    x_aligned = align_to_target_wavelengths(x_sg, wl_crop, common_wl)
    x_snv = snv(x_aligned)
    logger.log(f"{label}: 裁剪{x_crop.shape[1]}带 → 对齐{x_aligned.shape[1]}带 → SNV 均值{x_snv.mean():.4f}")
    return x_snv


def rebuild_processed(logger: eu.Logger) -> None:
    """从原始数据再生 processed/ 三件 CSV（覆盖前打印与旧文件的数值差异报告）。"""
    for p in (RAW_2018, RAW_2019, RAW_2025):
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺原始数据：{p}")

    # 1. 2025 确定公共波长网格
    df25_raw = pd.read_csv(RAW_2025)
    spec_cols_25 = [c for c in df25_raw.columns if "nm" in str(c)]
    wl_25 = np.array([float(c.replace("nm", "")) for c in spec_cols_25])
    overlap_mask_25 = (wl_25 >= OVERLAP_LOW) & (wl_25 <= OVERLAP_HIGH)
    common_wl = wl_25[overlap_mask_25]
    common_cols_25 = np.array(spec_cols_25)[overlap_mask_25].tolist()
    wl_col_names = [f"{w:.2f}nm" for w in common_wl]
    logger.log(f"公共波长网格：{len(common_wl)} 带 {common_wl[0]:.2f}–{common_wl[-1]:.2f}nm")

    x25_raw = df25_raw[common_cols_25].to_numpy().astype(np.float32)
    origin_25 = df25_raw["图像名称"].str[:2].to_numpy()
    apple_id_25 = df25_raw["实际苹果编号"].str.split("-").str[0].to_numpy()
    face_25 = df25_raw["实际苹果编号"].str.split("-").str[1].to_numpy()

    # 2. 2018/2019 加载 + 预处理
    x18_raw, wl_18, orig_18, ssc_18 = load_old_year(RAW_2018, "2018", logger)
    x19_raw, wl_19, orig_19, ssc_19 = load_old_year(RAW_2019, "2019", logger)
    x18 = preprocess_old(x18_raw, wl_18, common_wl, "2018", logger)
    x19 = preprocess_old(x19_raw, wl_19, common_wl, "2019", logger)

    # 3. 2025 预处理（SG + SNV，已在公共网格上）
    x25 = snv(sg_smooth(x25_raw))
    logger.log(f"2025: SG+SNV 均值{x25.mean():.4f}")

    # 4. 构建 9 元素数据集
    df18 = build_element_df(x18, orig_18, "2018", wl_col_names, make_synthetic_apple_ids(orig_18, "2018"), ssc=ssc_18)
    df19 = build_element_df(x19, orig_19, "2019", wl_col_names, make_synthetic_apple_ids(orig_19, "2019"), ssc=ssc_19)
    df25 = build_element_df(x25, origin_25, "2025", wl_col_names, apple_ids=apple_id_25, faces=face_25)
    df_all = pd.concat([df18, df19, df25], axis=0, ignore_index=True)

    # 5. 2025 苹果级聚合
    df25_apple = df25.groupby(["苹果编号", "产地", "年份", "element_id"])[wl_col_names].mean().reset_index()

    # 6. 苹果粒度索引
    apple_index = df_all.groupby(["苹果编号", "年份", "产地", "element_id"]).size().reset_index(name="面位数")

    # 7. 覆盖前差异报告（对 all_9elements）
    if os.path.exists(eu.ALL_ELEMENTS_FILE):
        old = pd.read_csv(eu.ALL_ELEMENTS_FILE)
        old_cols = eu.get_spectrum_columns(old)
        if old.shape[0] == df_all.shape[0] and len(old_cols) == len(wl_col_names):
            diff = np.abs(old[old_cols].to_numpy() - df_all[wl_col_names].to_numpy())
            logger.log(f"[差异报告] 与现有冻结文件 max|Δ|={float(diff.max()):.3e} mean|Δ|={float(diff.mean()):.3e}")
        else:
            logger.log(f"[差异报告] 形状不一致 旧{old.shape} 新{df_all.shape}，按新结构覆盖")

    # 8. 落盘
    df_all.to_csv(eu.ALL_ELEMENTS_FILE, index=False, encoding="utf-8-sig")
    df25_apple.to_csv(eu.APPLE_LEVEL_2025_FILE, index=False, encoding="utf-8-sig")
    apple_index.to_csv(eu.APPLE_INDEX_FILE, index=False, encoding="utf-8-sig")
    logger.log(
        f"再生完成：all_9elements={df_all.shape} apple_level_2025={df25_apple.shape} apple_index={apple_index.shape}"
    )


# ─── 数据集卡（§4.6） ─────────────────────────────────────────────────────────
def _counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    """返回某列取值计数（按取值字符串排序），列不存在时返回空字典。"""
    if col not in df.columns:
        return {}
    vc = df[col].value_counts()
    return {str(k): int(v) for k, v in sorted(vc.items(), key=lambda kv: str(kv[0]))}


def _datasheet_all_elements(path: str) -> str:
    df = pd.read_csv(path)
    spectrum_cols = eu.get_spectrum_columns(df)
    wl = eu.get_spectrum_wavelengths(df, spectrum_cols)
    id_col = "苹果编号" if "苹果编号" in df.columns else None
    stats: dict[str, object] = {
        "面位级光谱样本数": int(df.shape[0]),
        "苹果数(唯一编号)": int(df[id_col].nunique()) if id_col else "未知",
        "波段数": len(spectrum_cols),
        "波长范围(nm)": f"{float(wl.min()):.2f}–{float(wl.max()):.2f}",
        "按年份": _counts(df, "年份"),
        "按产地": _counts(df, "产地"),
        "按元素element_id": _counts(df, "element_id"),
    }
    return eu.write_datasheet(
        data_path=path,
        source_and_license=(
            "新疆农业大学（XJAU）自采可见/近红外光谱，2018/2019/2025 三个采集年份的商品苹果："
            "2018/2019 年为光纤光谱仪采集的透射光谱（2019 年一批的样品与采集过程见黄华等 2022、马永杰等 2020），"
            "2025 年为推扫式高光谱成像系统采集的反射光谱。"
            "数据受限获取（managed access），非公开，可向通讯作者申请（详见稿件 Data availability）。"
        ),
        granularity=(
            "粒度=单面位光谱（每行一条光谱）。2025 每果 5 面位、2018/2019 每果 1 面位。"
            "单位：SNV 标准化后的无量纲强度值（2018/2019 源自透射强度、2025 源自反射率）。"
        ),
        sample_stats=stats,
        split_logic=(
            "LOEO-Year（留一年份）3 折，时序外推；分组单位=苹果编号 apple_id，"
            "同果多面位严禁跨 support/query 或 train/test 泄漏。详见 01STATISTICAL_PROTOCOL.md。"
        ),
        known_bias=(
            "标签空间随年份变化（陕西仅历史年、甘肃仅 2025）；2025 面位数与历史年不一致，"
            "构成测量协议混杂；山东在 2018/2019 与陕西光谱高度相似，存在系统性混淆。"
        ),
        label_reliability=(
            "标签=采样登记的声明产地（origin label，非化学金标准）。商品苹果场景下"
            "光谱另含品种/储藏/成熟度/供应链因素，'产地溯源'定义为商品批次条件下的产地标签判别。"
        ),
        preprocessing_pointer=(
            "逐样本 L1：波长裁剪(590–1001nm) → SG 平滑(window=15,poly=2,axis=1) → "
            "三次样条插值(对齐 229 波段) → SNV(逐样本)。无跨样本 fit 型 L2。"
            "z-score 等 L2 仅在折内源数据拟合（strict source-only）。生成入口=01preprocess.py。"
            "可复现性：本文件为 SHA-256 锁定的冻结正典；以 scipy 1.15 重跑 01preprocess.py --rebuild "
            "可在 4524/4529 行复现到浮点噪声(mean|Δ|≈2e-4)，余 5 行(2025-SD-E8: SD004/SD022/SD058)"
            "因原始光谱各含 1 个零值波段(传感器 dropout)，其 SG/三次样条平滑随 scipy 版本而异。"
            "正式结果一律以本冻结正典为输入，下游读取经 load_all_elements_df(verify_sha256=True) 强制校验。"
        ),
        ethics=("植物样本，无人类受试者/个人隐私数据；不涉及伦理审查豁免之外的敏感属性。"),
    )


def _datasheet_apple_index(path: str) -> str:
    df = pd.read_csv(path)
    stats: dict[str, object] = {
        "苹果数": int(df.shape[0]),
        "列": ", ".join(str(c) for c in df.columns),
        "按年份": _counts(df, "年份"),
    }
    return eu.write_datasheet(
        data_path=path,
        source_and_license="同 all_9elements（XJAU 受限获取）。苹果粒度索引表。",
        granularity="粒度=单个苹果（每行一果），含面位数等果级元数据。",
        sample_stats=stats,
        split_logic="作为 apple_id 分组划分的索引，配合 LOEO-Year 使用。",
        known_bias="2018/2019 面位数=1，2025 面位数=5，年份间测量密度不均衡。",
        label_reliability="果级产地标签同 all_9elements。",
        preprocessing_pointer="由 01preprocess.py 对齐生成，详见 01STATISTICAL_PROTOCOL.md。",
        ethics="植物样本，无敏感个人数据。",
    )


def _datasheet_apple_level_2025(path: str) -> str:
    df = pd.read_csv(path)
    spectrum_cols = eu.get_spectrum_columns(df)
    stats: dict[str, object] = {
        "苹果数(2025聚合)": int(df.shape[0]),
        "波段数": len(spectrum_cols),
        "按产地": _counts(df, "产地"),
        "按元素element_id": _counts(df, "element_id"),
    }
    return eu.write_datasheet(
        data_path=path,
        source_and_license="同 all_9elements（XJAU 受限获取）。2025 苹果级（5 面位均值）聚合。",
        granularity="粒度=单个苹果（2025，5 面位均值），仅 E7/E8/E9 元素、XJ/SD/GS 产地。",
        sample_stats=stats,
        split_logic="用于面位均值消融（§13 异质性解耦），不替代面位级正式评估。",
        known_bias="仅 2025 年、仅三产地，覆盖面窄；为面位聚合，丢失面位级方差。",
        label_reliability="果级产地标签同 all_9elements。",
        preprocessing_pointer="5 面位逐果求均值（01preprocess.py），详见 01STATISTICAL_PROTOCOL.md。",
        ethics="植物样本，无敏感个人数据。",
    )


# ─── 预处理前后对比可视化（--plot；提供原 02_preprocess-图a 的溯源来源）──────────
_PLOT_COLORS = {"XJ": "#0072B2", "SD": "#D55E00", "SX": "#009E73", "GS": "#CC79A7"}
_ORIGIN_TXT = {
    "zh": {"XJ": "新疆", "SD": "山东", "SX": "陕西", "GS": "甘肃"},
    "en": {"XJ": "Xinjiang", "SD": "Shandong", "SX": "Shaanxi", "GS": "Gansu"},
}


def _load_years_for_plot(
    logger: eu.Logger,
) -> tuple[NDArray, list[tuple[str, NDArray, NDArray, NDArray]]]:
    """加载三年 (对齐未SNV, SNV后, 产地code) 供预处理对比图；复用 rebuild 的加载逻辑。"""
    df25 = pd.read_csv(RAW_2025)
    spec25 = [c for c in df25.columns if "nm" in str(c)]
    wl25 = np.array([float(str(c).replace("nm", "")) for c in spec25])
    m25 = (wl25 >= OVERLAP_LOW) & (wl25 <= OVERLAP_HIGH)
    common_wl = wl25[m25]
    cols25 = np.array(spec25)[m25].tolist()
    x25_raw = df25[cols25].to_numpy().astype(np.float32)
    zh2code = {"新疆": "XJ", "山东": "SD", "甘肃": "GS"}
    orig25 = np.array([zh2code.get(str(s)[:2], str(s)[:2]) for s in df25["图像名称"]])
    x25a = sg_smooth(x25_raw)

    def _prep_old(path: str, yr: str) -> tuple[NDArray, NDArray, NDArray]:
        xr, wl, org, _ = load_old_year(path, yr, logger)
        mm = (wl >= OVERLAP_LOW - 1) & (wl <= OVERLAP_HIGH + 1)
        xa = align_to_target_wavelengths(sg_smooth(xr[:, mm]), wl[mm], common_wl)
        return xa, snv(xa), org

    x18a, x18s, o18 = _prep_old(RAW_2018, "2018")
    x19a, x19s, o19 = _prep_old(RAW_2019, "2019")
    return common_wl, [
        ("2018", x18a, x18s, o18),
        ("2019", x19a, x19s, o19),
        ("2025", x25a, snv(x25a), orig25),
    ]


def plot_preprocess_overview(logger: eu.Logger) -> None:
    """--plot：三年 × 产地 原始(对齐,未SNV) vs SNV 前后对比（每产地 5 条代表样本）。"""
    eu.setup_chinese_font()
    common_wl, years = _load_years_for_plot(logger)
    rng = np.random.default_rng(eu.PILOT_SEED)
    picks: list[dict[str, NDArray]] = []
    for _yr, _xa, _xs, org in years:
        reg_idx: dict[str, NDArray] = {}
        for reg in dict.fromkeys(org.tolist()):
            ridx = np.where(org == reg)[0]
            if ridx.size:
                reg_idx[str(reg)] = rng.choice(ridx, size=min(5, ridx.size), replace=False)
        picks.append(reg_idx)

    def plot_fn(lang: str) -> Any:
        import matplotlib.pyplot as plt

        zh = lang == "zh"
        col_t = ("原始反射率（对齐后）", "SNV 变换后") if zh else ("Raw reflectance (aligned)", "After SNV")
        otxt = _ORIGIN_TXT[lang]
        fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
        for row, (yr, xa, xs, _org) in enumerate(years):
            for col, data in ((0, xa), (1, xs)):
                ax = axes[row][col]
                for reg, idx in picks[row].items():
                    ax.plot(common_wl, data[idx].T, color=_PLOT_COLORS.get(reg, "gray"), lw=0.8, alpha=0.75)
                if row == 0:
                    ax.set_title(col_t[col], fontsize=12)
                ax.grid(alpha=0.3)
            axes[row][0].set_ylabel(f"{yr}", fontsize=11)
            handles = [plt.Line2D([0], [0], color=_PLOT_COLORS.get(r, "gray"), lw=2) for r in picks[row]]  # type: ignore[attr-defined]
            axes[row][1].legend(handles, [otxt.get(r, r) for r in picks[row]], fontsize=8, loc="upper right")
        xlabel = "波长 (nm)" if zh else "Wavelength (nm)"
        for ax in axes[2]:
            ax.set_xlabel(xlabel, fontsize=11)
        sup = (
            "三年 × 产地 NIR 光谱预处理前后对比（每产地 5 条代表样本）"
            if zh
            else "NIR preprocessing comparison across years x origins (5 spectra each)"
        )
        fig.suptitle(sup, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig

    p_zh, p_en = eu.save_figure_bilingual(plot_fn, "01preprocess", 0)
    logger.log(f"预处理前后对比图 → {os.path.basename(p_zh)} / {os.path.basename(p_en)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="01preprocess：苹果 NIR 预处理 + 数据集卡（016 唯一入口）")
    parser.add_argument("--rebuild", action="store_true", help="从原始数据再生 processed/（默认仅校验+数据集卡）")
    parser.add_argument("--plot", action="store_true", help="生成预处理前后对比图 01preprocess-图a/-fig-a")
    args = parser.parse_args()

    logger = eu.get_logger("01preprocess")
    mode = "rebuild" if args.rebuild else "verify"
    eu.log_experiment_header(logger, {"任务": "数据预处理 + 数据集卡(§4.5/§4.6)", "阶段": "data-card", "模式": mode})
    if args.rebuild:
        logger.log("=== --rebuild：从原始数据再生 processed/ ===")
        rebuild_processed(logger)
    if args.plot:
        logger.log("=== --plot：生成预处理前后对比图 ===")
        plot_preprocess_overview(logger)

    targets = [
        (eu.ALL_ELEMENTS_FILE, _datasheet_all_elements),
        (eu.APPLE_INDEX_FILE, _datasheet_apple_index),
        (eu.APPLE_LEVEL_2025_FILE, _datasheet_apple_level_2025),
    ]
    for path, fn in targets:
        if not os.path.exists(path):
            logger.log(f"[跳过] 未找到 {path}")
            continue
        out = fn(path)
        sha = eu.compute_sha256(path)
        logger.log(f"[完成] {os.path.basename(path)}  SHA-256={sha[:16]}…  → {os.path.basename(out)}")
    logger.log("数据集卡生成完毕。下游可用 load_all_elements_df(verify_sha256=True) 校验完整性。")
    logger.close()


if __name__ == "__main__":
    main()
