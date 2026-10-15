"""41_confident_learning_audit.py: Confident Learning 可疑标签检出（L3-3）。

要回答的问题:
    三年的产地标签全部是商户声明。用 Confident Learning（cleanlab）报**各产地的可疑标签率**，
    看是否存在**针对山东的系统性错标**。

预设判读口径（**先写死再看结果**）:
    - 若 SD 的可疑率与同年其他产地**相当** → 不存在针对山东的系统性错标。
    - 若 SD **显著更高** → 如实报，交第 3 期在局限里正面写。
    - 「显著更高」= 同年两两比较 Fisher 精确检验 **Holm 校正后 P < 0.05** 且**率差 > 5 个百分点**，
      两条同时满足才算；只满足一条记为「方向性偏高，不构成结论」。

三条必须先讲清的边界（决定这个数能读到什么程度）:
    1. **只做年份内 CV**。跨年份训练→预测的路线已被 L3-1（`29INSTRUMENT_LABEL_DECOMP.md`）
       证明被域漂移支配（源→目标位移的 72–74% 是类别无关公共分量），检出率会退化成
       「哪一年被漂移打得最惨」而不是「哪个标签可疑」。本脚本仍把跨年口径作为**诊断对照**
       跑一遍，用来量化这条路有多不可用，而不是用它下结论。
    2. **年份内 CL 检出的是「这个果不像它同年同产地的同伴」，不是「这个果不是山东的」。**
       商户按批次报产地，**整批误报**时该批内部完全自洽，年份内 CL 对此**结构性失明**。
       2025 年三个产地在同一市场买齐，恰恰是最可能整批误报的情形。
    3. **检出率不能跨类直接比大小**：本身更难分的类（山东与陕西光谱高度重叠，数据卡已记录）
       即使标签全对也会拿到更高的检出率。故本脚本附**注入校准**——按已知比例注入翻转，
       测每个产地的**检出灵敏度**与**未注入果的误报底噪**，没有灵敏度就没有解读权。

可比性（任务书点名）:
    GS 只在 2025 出现、SX 只在 2018/2019 出现，故**一切检出率按年份分开报**，
    跨年份的产地对比只有 SD 与 XJ 三年都在，其余不可比。

阶段与硬件:
    <MPS_PILOT> —— CPU，无 GPU。分类器为多项逻辑回归（不依赖任何编码器），
    故本节结论不受「共享/折限定编码器」之争影响。

运行方式:
    cd <project root>
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python 02code/41_confident_learning_audit.py

输出文件:
    04outputs/41_confident_learning_audit-图a.pdf / -fig-a.pdf  — 各年份×产地可疑标签率
    04outputs/41_confident_learning_audit-图b.pdf / -fig-b.pdf  — 注入校准：检出灵敏度与误报底噪
    04outputs/41_confident_learning_audit-图c.pdf / -fig-c.pdf  — 年份内 confident joint 热力图
    04outputs/41_confident_learning_audit.xlsx                   — 表a–表f + 算力
    07checkpoints/41_confident_learning_probe/                   — CL 用分类器权重 + 模型卡（F1）
    05logs/41_confident_learning_audit_YY-MM-DD_HHMMSS.log       — 运行日志
"""

from __future__ import annotations

import argparse
import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from cleanlab.count import compute_confident_joint
from cleanlab.filter import find_label_issues
from scipy import stats as sstats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import export_utils as eu

NDArray = npt.NDArray[Any]

YEARS = (2018, 2019, 2025)
YEAR_CLASSES: dict[int, list[str]] = {2018: ["SD", "SX", "XJ"], 2019: ["SD", "SX", "XJ"], 2025: ["GS", "SD", "XJ"]}
ORIGIN_ZH = {"XJ": "新疆", "SD": "山东", "SX": "陕西", "GS": "甘肃"}
INJECT_EPS = (0.10, 0.20, 0.30)
N_CV = 5
MAJORITY = 3  # 5 个种子里被标 ≥3 次才算「可疑」，避免单次 CV 划分的偶然
RATE_DIFF_MARGIN = 0.05
ALPHA_LEVEL = 0.05
C_ORIGIN = {"XJ": "#0072B2", "SD": "#C1272D", "SX": "#E69F00", "GS": "#009E73"}


def build_apple_table() -> tuple[pd.DataFrame, NDArray]:
    """苹果粒度表 + 苹果均值光谱（同果全部面位取均值，标签本就是果级的）。"""
    df = eu.load_all_elements_df(verify_sha256=True)
    wl = eu.get_spectrum_columns(df)
    grouped = df.groupby("苹果编号", sort=True)
    spec = grouped[wl].mean()
    meta = grouped[["年份", "产地"]].first().loc[spec.index]
    table = pd.DataFrame(
        {"苹果编号": spec.index, "年份": meta["年份"].to_numpy(), "产地": meta["产地"].to_numpy()}
    ).reset_index(drop=True)
    return table, spec.to_numpy().astype(np.float64)


def cv_pred_probs(x: NDArray, y: NDArray, n_cls: int, seed: int) -> NDArray:
    """年份内分层 K 折交叉验证的样本外预测概率（苹果为行，无同果跨折问题）。"""
    probs = np.zeros((len(y), n_cls), dtype=float)
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=seed)
    for tr, te in skf.split(x, y):
        clf = make_clf()
        clf.fit(x[tr], y[tr])
        p = clf.predict_proba(x[te])
        # 训练折可能缺类：按 clf.classes_ 回填到全类列
        probs[np.ix_(te, clf.named_steps["lr"].classes_)] = p
    row = probs.sum(axis=1, keepdims=True)
    return probs / np.where(row > 0, row, 1.0)


def make_clf() -> Pipeline:
    return Pipeline(
        # multi_class 参数在 sklearn 1.5 起弃用；多类下默认即多项 softmax，故不再显式传
        [("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=3000, C=1.0))]
    )


def flag_matrix(x: NDArray, y: NDArray, n_cls: int, seeds: tuple[int, ...]) -> tuple[NDArray, NDArray]:
    """跨种子重复 CV → (每果被标次数, 末次 confident joint)。"""
    counts = np.zeros(len(y), dtype=int)
    cj = np.zeros((n_cls, n_cls), dtype=float)
    for s in seeds:
        probs = cv_pred_probs(x, y, n_cls, int(s) % (2**31 - 1))
        idx = find_label_issues(labels=y, pred_probs=probs, filter_by="confident_learning")
        counts += np.asarray(idx, dtype=int)
        cj += np.asarray(compute_confident_joint(labels=y, pred_probs=probs), dtype=float)
    return counts, cj / len(seeds)


def inject(y: NDArray, target_cls: int, eps: float, n_cls: int, seed: int) -> tuple[NDArray, NDArray]:
    """把 target_cls 的果按 eps 翻到同年其它类（与 40 号脚本同一噪声模型：果级、嵌套）。"""
    rng = np.random.default_rng([int(seed), int(target_cls), 41])
    u = rng.random(len(y))
    alt = rng.integers(0, max(1, n_cls - 1), len(y))
    out = y.copy()
    hit = (y == target_cls) & (u < eps)
    for i in np.where(hit)[0]:
        others = [c for c in range(n_cls) if c != target_cls]
        out[i] = others[int(alt[i]) % len(others)]
    return out, hit


def main() -> None:
    p = argparse.ArgumentParser(description="41_confident_learning_audit：Confident Learning 可疑标签检出（L3-3）")
    p.add_argument("--stage", default="pilot", choices=["pilot", "formal"])
    args = p.parse_args()

    logger = eu.get_logger("41_confident_learning_audit")
    eu.set_random_seed(int(eu.RECOMMENDED_SEEDS[0]), logger)
    eu.log_experiment_header(
        logger,
        {
            "任务": "Confident Learning 可疑标签检出（L3-3）",
            "阶段": args.stage,
            "设备": "cpu",
            "分类器": "StandardScaler + 多项逻辑回归（不依赖编码器）",
            "CV": f"年份内分层{N_CV}折 × {eu.n_distinct_seeds()}种子，多数票阈值{MAJORITY}",
            "注入校准ε": list(INJECT_EPS),
            "种子": list(eu.RECOMMENDED_SEEDS),
        },
    )
    table, x_all = build_apple_table()
    logger.log(f"数据：{len(table)} 苹果 / {x_all.shape[1]} 波段")
    seeds = tuple(int(s) for s in eu.RECOMMENDED_SEEDS)

    rate_rows: list[dict[str, Any]] = []
    cj_rows: list[dict[str, Any]] = []
    calib_rows: list[dict[str, Any]] = []
    per_apple: list[dict[str, Any]] = []

    with eu.track_compute("Confident Learning 审计", logger, None):
        for year in YEARS:
            sel = table["年份"].to_numpy() == year
            classes = YEAR_CLASSES[year]
            c2i = {c: i for i, c in enumerate(classes)}
            y = np.array([c2i[c] for c in table["产地"].to_numpy()[sel]], dtype=int)
            x = x_all[sel]
            ids = table["苹果编号"].to_numpy()[sel]
            counts, cj = flag_matrix(x, y, len(classes), seeds)
            flagged = counts >= MAJORITY
            logger.log(f"\n【{year}】{len(y)} 果，{len(classes)} 类：整体可疑率 {flagged.mean():.4f}")
            for c, ci in c2i.items():
                m = y == ci
                rate_rows.append(
                    {
                        "年份": year,
                        "产地": c,
                        "产地中文": ORIGIN_ZH[c],
                        "苹果数": int(m.sum()),
                        "可疑数": int(flagged[m].sum()),
                        "可疑率": float(flagged[m].mean()),
                        "跨种子稳定性_平均被标次数": float(counts[m].mean()),
                    }
                )
                logger.log(
                    f"  {c}({ORIGIN_ZH[c]}) n={int(m.sum()):4d}  可疑 {int(flagged[m].sum()):3d}  "
                    f"率 {flagged[m].mean():.4f}"
                )
            for i, c in enumerate(classes):
                for j, c2 in enumerate(classes):
                    cj_rows.append({"年份": year, "声明产地": c, "CL判定产地": c2, "计数": float(cj[i, j])})
            for aid, yy, cc, kk in zip(ids, y, counts, flagged, strict=False):
                per_apple.append(
                    {
                        "年份": year,
                        "苹果编号": aid,
                        "声明产地": classes[int(yy)],
                        "被标次数": int(cc),
                        "判可疑": bool(kk),
                    }
                )

            # 注入校准：每个产地按已知比例翻转 → 灵敏度 + 未注入果的误报底噪
            for c, ci in c2i.items():
                for eps in INJECT_EPS:
                    y_inj, hit = inject(y, ci, eps, len(classes), seeds[0])
                    cnt_i, _ = flag_matrix(x, y_inj, len(classes), seeds[:3])
                    fl = cnt_i >= 2  # 3 个种子里 ≥2 次
                    clean_same = (y_inj == ci) & (~hit)
                    calib_rows.append(
                        {
                            "年份": year,
                            "注入产地": c,
                            "注入ε": eps,
                            "注入果数": int(hit.sum()),
                            "检出灵敏度": float(fl[hit].mean()) if hit.any() else np.nan,
                            "未注入同类误报率": float(fl[clean_same].mean()) if clean_same.any() else np.nan,
                        }
                    )
                    logger.log(
                        f"  [校准] {year} 注入{c} ε={eps:.2f} n={int(hit.sum()):3d} "
                        f"灵敏度={calib_rows[-1]['检出灵敏度']:.3f} 底噪={calib_rows[-1]['未注入同类误报率']:.3f}"
                    )

        # 跨年份诊断对照：训 A 年、测 B 年（预期被域漂移支配，只用来量化这条路不可用）
        cross_rows: list[dict[str, Any]] = []
        for tr_y, te_y in itertools.permutations(YEARS, 2):
            common = [c for c in YEAR_CLASSES[tr_y] if c in YEAR_CLASSES[te_y]]
            if len(common) < 2:
                continue
            c2i = {c: i for i, c in enumerate(common)}
            m_tr = (table["年份"].to_numpy() == tr_y) & np.isin(table["产地"].to_numpy(), common)
            m_te = (table["年份"].to_numpy() == te_y) & np.isin(table["产地"].to_numpy(), common)
            y_tr = np.array([c2i[c] for c in table["产地"].to_numpy()[m_tr]])
            y_te = np.array([c2i[c] for c in table["产地"].to_numpy()[m_te]])
            clf = make_clf()
            clf.fit(x_all[m_tr], y_tr)
            probs = clf.predict_proba(x_all[m_te])
            full = np.zeros((len(y_te), len(common)))
            full[:, clf.named_steps["lr"].classes_] = probs
            idx = find_label_issues(labels=y_te, pred_probs=full, filter_by="confident_learning")
            for c, ci in c2i.items():
                mm = y_te == ci
                cross_rows.append(
                    {
                        "训练年": tr_y,
                        "测试年": te_y,
                        "产地": c,
                        "苹果数": int(mm.sum()),
                        "可疑率": float(np.asarray(idx)[mm].mean()),
                    }
                )
            logger.log(
                f"[跨年诊断] 训{tr_y}→测{te_y}（{'/'.join(common)}）整体可疑率 {np.asarray(idx).mean():.4f}"
            )

    rate_df = pd.DataFrame(rate_rows)
    calib_df = pd.DataFrame(calib_rows)
    cj_df = pd.DataFrame(cj_rows)
    cross_df = pd.DataFrame(cross_rows)
    apple_df = pd.DataFrame(per_apple)

    # ── 预设判据：同年两两 Fisher + Holm ────────────────────────────────────────
    cmp_rows: list[dict[str, Any]] = []
    for year in YEARS:
        sub = rate_df[rate_df["年份"] == year]
        for a, b in itertools.combinations(sorted(sub["产地"]), 2):
            ra = sub[sub["产地"] == a].iloc[0]
            rb = sub[sub["产地"] == b].iloc[0]
            tab = [
                [int(ra["可疑数"]), int(ra["苹果数"] - ra["可疑数"])],
                [int(rb["可疑数"]), int(rb["苹果数"] - rb["可疑数"])],
            ]
            _, pv = sstats.fisher_exact(tab)
            cmp_rows.append(
                {
                    "年份": year,
                    "产地A": a,
                    "产地B": b,
                    "率A": float(ra["可疑率"]),
                    "率B": float(rb["可疑率"]),
                    "率差A减B": float(ra["可疑率"] - rb["可疑率"]),
                    "P原始": float(pv),
                }
            )
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df["P_Holm"] = eu.apply_multi_comp_correction(cmp_df["P原始"].tolist(), method="holm")["p_adjusted"]
    cmp_df["达显著且率差>5pp"] = (cmp_df["P_Holm"] < ALPHA_LEVEL) & (cmp_df["率差A减B"].abs() > RATE_DIFF_MARGIN)
    logger.log("\n【同年两两比较】\n" + cmp_df.round(4).to_string(index=False))

    # ── 由「实测检出率 + 实测灵敏度」反推 iid 果级错标率的单侧上界 ──────────────
    # p = ε·s + (1−ε)·f。取 f=0（忽略误报底噪）会**高估** ε，故所得为保守上界：
    # ε_upper = Clopper-Pearson 97.5% 上限(k/n) / s。**只对 iid 果级噪声有效**，
    # 整批误报不在此模型内（见文件头边界 2）。
    bound_rows: list[dict[str, Any]] = []
    for _, r in rate_df.iterrows():
        k, n = int(r["可疑数"]), int(r["苹果数"])
        p_hi = float(sstats.beta.ppf(1 - ALPHA_LEVEL / 2, k + 1, n - k)) if k < n else 1.0
        s = float(
            calib_df[(calib_df["年份"] == r["年份"]) & (calib_df["注入产地"] == r["产地"])]["检出灵敏度"].mean()
        )
        bound_rows.append(
            {
                "年份": r["年份"],
                "产地": r["产地"],
                "可疑数": k,
                "苹果数": n,
                "检出率97.5%上限": round(p_hi, 4),
                "实测检出灵敏度均值": round(s, 4),
                "iid错标率95%单侧上界": round(min(1.0, p_hi / s), 4) if s > 0 else np.nan,
            }
        )
    bound_df = pd.DataFrame(bound_rows)
    logger.log("\n【iid 果级错标率上界（整批误报不适用）】\n" + bound_df.to_string(index=False))

    verdict: list[dict[str, Any]] = []
    for year in YEARS:
        rel = cmp_df[(cmp_df["年份"] == year) & ((cmp_df["产地A"] == "SD") | (cmp_df["产地B"] == "SD"))]
        hits = []
        for _, r in rel.iterrows():
            d = r["率差A减B"] if r["产地A"] == "SD" else -r["率差A减B"]
            other = r["产地B"] if r["产地A"] == "SD" else r["产地A"]
            if bool(r["达显著且率差>5pp"]) and d > 0:
                hits.append(f"高于{other}")
            elif d > RATE_DIFF_MARGIN or r["P_Holm"] < ALPHA_LEVEL:
                hits.append(f"方向性高于{other}(未双达标)")
        sd_rate = float(rate_df[(rate_df["年份"] == year) & (rate_df["产地"] == "SD")]["可疑率"].iloc[0])
        verdict.append(
            {
                "年份": year,
                "SD可疑率": round(sd_rate, 4),
                "同年其它产地可疑率": ", ".join(
                    f"{r['产地']}={r['可疑率']:.4f}"
                    for _, r in rate_df[(rate_df["年份"] == year) & (rate_df["产地"] != "SD")].iterrows()
                ),
                "预设判据结论": "SD 显著更高：" + "；".join(hits) if hits else "SD 与同年其它产地相当（未达预设判据）",
            }
        )
    verdict_df = pd.DataFrame(verdict)
    logger.log("\n【预设判据下的结论】\n" + verdict_df.to_string(index=False))

    # ── 图 ──────────────────────────────────────────────────────────────────
    txt = {
        "zh": {
            "ya": "可疑标签率（Confident Learning）",
            "ta": f"各采收年份×声明产地的可疑标签率（年份内 {N_CV} 折 CV，{eu.n_distinct_seeds()} 种子多数票）",
            "x": "声明产地",
            "tb": "注入校准：已知翻转的检出灵敏度与未注入果的误报底噪",
            "yb": "比例",
            "sens": "检出灵敏度",
            "fp": "误报底噪",
            "eps": "注入比例 ε",
            "tc": "年份内 confident joint（行=声明产地，列=CL 判定产地）",
            "cd": "声明产地",
            "cl": "CL 判定产地",
        },
        "en": {
            "ya": "Suspicious-label rate (confident learning)",
            "ta": (
                f"Suspicious-label rate by harvest year x declared origin "
                f"({N_CV}-fold within-year CV, majority of {eu.n_distinct_seeds()} seeds)"
            ),
            "x": "Declared origin",
            "tb": "Injection calibration: detection sensitivity vs false-positive floor",
            "yb": "Proportion",
            "sens": "Sensitivity",
            "fp": "False-positive floor",
            "eps": "Injected flip rate ε",
            "tc": "Within-year confident joint (row = declared, col = CL-assigned)",
            "cd": "Declared origin",
            "cl": "CL-assigned origin",
        },
    }

    def plot_a(lang: str) -> Any:
        t = txt[lang]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, year in zip(axes, YEARS, strict=False):
            sub = rate_df[rate_df["年份"] == year].sort_values("产地")
            labels = [c if lang == "en" else f"{c}\n{ORIGIN_ZH[c]}" for c in sub["产地"]]
            bars = ax.bar(labels, sub["可疑率"], color=[C_ORIGIN[c] for c in sub["产地"]], width=0.6)
            for b, n, k in zip(bars, sub["苹果数"], sub["可疑数"], strict=False):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{k}/{n}", ha="center", fontsize=8)
            ax.set_title(str(year))
            ax.set_xlabel(t["x"])
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel(t["ya"])
        axes[0].set_ylim(0, max(0.25, float(rate_df["可疑率"].max()) * 1.35))
        fig.suptitle(t["ta"], fontsize=11)
        fig.tight_layout()
        return fig

    def plot_b(lang: str) -> Any:
        t = txt[lang]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, year in zip(axes, YEARS, strict=False):
            sub = calib_df[calib_df["年份"] == year]
            for c in sorted(sub["注入产地"].unique()):
                s = sub[sub["注入产地"] == c].sort_values("注入ε")
                ax.plot(s["注入ε"] * 100, s["检出灵敏度"], marker="o", lw=2,
                        color=C_ORIGIN[c], label=f"{c} {t['sens']}")
                ax.plot(s["注入ε"] * 100, s["未注入同类误报率"], marker="^", ls="--", lw=1.2,
                        color=C_ORIGIN[c], alpha=0.6, label=f"{c} {t['fp']}")
            ax.set_title(str(year))
            ax.set_xlabel(t["eps"] + " (%)")
            ax.set_xticks([e * 100 for e in INJECT_EPS])
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(t["yb"])
        axes[0].set_ylim(0, 1.02)
        axes[2].legend(fontsize=7, ncol=2)
        fig.suptitle(t["tb"], fontsize=11)
        fig.tight_layout()
        return fig

    def plot_c(lang: str) -> Any:
        t = txt[lang]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, year in zip(axes, YEARS, strict=False):
            classes = YEAR_CLASSES[year]
            mat = (
                cj_df[cj_df["年份"] == year]
                .pivot(index="声明产地", columns="CL判定产地", values="计数")
                .reindex(index=classes, columns=classes)
                .to_numpy()
            )
            frac = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1e-9)
            im = ax.imshow(frac, cmap="magma_r", vmin=0, vmax=1)
            ax.set_xticks(range(len(classes)), classes)
            ax.set_yticks(range(len(classes)), classes)
            ax.set_xlabel(t["cl"])
            ax.set_title(str(year))
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if frac[i, j] > 0.5 else "black")
            plt.colorbar(im, ax=ax, shrink=0.85)
        axes[0].set_ylabel(t["cd"])
        fig.suptitle(t["tc"], fontsize=11)
        fig.tight_layout()
        return fig

    for idx, fn in enumerate((plot_a, plot_b, plot_c)):
        zh, en = eu.save_figure_bilingual(fn, "41_confident_learning_audit", idx)
        logger.log(f"图{chr(97 + idx)} → {zh.split('/')[-1]} / {en.split('/')[-1]}")

    # 参照分类器落盘（F1）：2025 折的年份内 CL 探针
    sel25 = table["年份"].to_numpy() == 2025
    cls25 = YEAR_CLASSES[2025]
    y25 = np.array([cls25.index(c) for c in table["产地"].to_numpy()[sel25]])
    probe = make_clf()
    probe.fit(x_all[sel25], y25)
    ck = eu.save_checkpoint(
        probe,
        "41_confident_learning_probe",
        {
            "stage": args.stage,
            "training_hardware": "cpu",
            "deterministic_guarantee": "full",
            "architecture": "StandardScaler + multinomial LogisticRegression",
            "task": "Confident Learning 年份内可疑标签探针（2025 折全量拟合版）",
            "note": "仅作检出探针，不是论文的产地判别主模型；可疑率数值来自 5 折 CV 的样本外概率，非本权重",
            "classes": ",".join(cls25),
            "n_apples": int(sel25.sum()),
        },
    )
    logger.log(f"参照分类器 → {ck['ckpt_path']}  sha256={ck['sha256'][:16]}…")

    prespec = pd.DataFrame(
        [
            {"项": "检出算法", "值": "cleanlab 2.9.0 find_label_issues(filter_by='confident_learning')"},
            {"项": "分析单位", "值": "苹果（同果面位取均值光谱）；标签本就是果级"},
            {"项": "概率来源", "值": f"年份内分层 {N_CV} 折 CV 的样本外概率，分类器=标准化+多项逻辑回归"},
            {"项": "多数票", "值": f"{eu.n_distinct_seeds()} 个正式种子各跑一次，被标 ≥{MAJORITY} 次才算可疑"},
            {
                "项": "「显著更高」判据",
                "值": f"同年两两 Fisher + Holm 后 P<{ALPHA_LEVEL} 且率差>{RATE_DIFF_MARGIN:.0%}，两条同时满足",
            },
            {"项": "可比性", "值": "GS 仅 2025、SX 仅 2018/2019；一切率按年份分开报，跨年只有 SD 与 XJ 可比"},
            {"项": "注入校准", "值": f"每产地按 ε∈{list(INJECT_EPS)} 注入已知翻转，测灵敏度与未注入果误报底噪"},
            {"项": "跨年口径", "值": "仅作诊断对照，不用于下结论（理由见 29INSTRUMENT_LABEL_DECOMP.md）"},
            {"项": "结构性盲区", "值": "整批误报时批内自洽，年份内 CL 对此失明——2025 三产地同市场买齐正属此情形"},
        ]
    )
    xlsx = eu.write_script_workbook(
        __file__,
        {
            0: ("预设判据", prespec),
            1: ("各年份产地可疑率", rate_df.round(6)),
            2: ("同年两两比较", cmp_df.round(6)),
            3: ("预设判据结论", verdict_df),
            4: ("注入校准", calib_df.round(6)),
            8: ("iid错标率上界", bound_df),
            5: ("年份内confident_joint", cj_df.round(4)),
            6: ("跨年诊断对照", cross_df.round(6)),
            7: ("逐苹果检出明细", apple_df),
        },
    )
    logger.log(f"结果表 → {xlsx}")
    logger.close()


if __name__ == "__main__":
    main()
