"""92_sx_tables.py: 补充实验轮（81–91 号）的补充材料表 S20–S30 表体与数字宏 \\Sx…，读各脚本的汇总工作簿，写回生成块。

    表 S20   年内随机 7:3 与跨年评估（81 号）：三种学习器与按已发表流程重建的一臂，准确率；重建臂有两种输入，
             盘上原始的 2048 个点（81 号缺省）与相邻 4 点取均值的 512 个波段（81 号 --recon_bin 4 --tag bin512
             的工作簿，只取其表 d）
    表 S21   化学计量学流程里的暴露（82 号）：标度、波长选择、波长选择（半分）与交叉验证成绩重用，逐 K 的留出域均值
    表 S22   基线的对等暴露（83 号）：池化 SVM、池化 PLS-DA 与原型网络在 B 半上的无暴露值、暴露值与增量，逐数据集
    表 S23   同一条训练轨迹上的检查点选择（85 号）与编码器超参数选择（86 号）：留出域选的减源域选的，逐编码器族
    表 S24   各编码器族 S_src−R 随 K 的变化（87 号）
    表 S25   编码器的波段遮蔽（87 号）：上半（生成块 S25A）R 与 S_src 两条下降量曲线的秩相关，
             下半（S25B）苹果两折 606–699 nm 的份额
    表 S26   监督式（非情节）训练下的三项对照（生成块 S26A）与无标签暴露（S26B，苹果两种设计与猕猴桃）（88 号）
    表 S27   跨数据集推断的稳健性（84 号）：留一数据集、数据集簇符号翻转检验、逐域种子区间
    表 S28   苹果两年设计的重复半分割（89 号）
    表 S29   芒果半分的划分单位（91 号）
    表 S30   芒果产区任务（90 号）
    数字宏 \\Sx…  正文 3.2–3.6 节、摘要、结论与补充材料 S9 节引用；中英文稿共用同一份宏
                 （S9.3 节池化 SVM 迭代上限的灵敏度 \\SxCap… 取自 93 号）
    定性断言   check_claims 逐条核对正文与补充材料 S9 节的定性表述；正式结果下任何一条不成立或无法核对即报错，须改写文字

口径与各脚本的汇总一致：对照在种子内配对作差、再对种子求均值；跨留出域的汇总取主口径
（苹果两年设计 2 个与公开 16 个留出域）的留出域均值与为正留出域数。
某个脚本的汇总不是五个正式种子齐全的正式结果（--stage pilot，或正式汇总不全）时，
取自它的数值一律包 \\pend{}，表示预览、非投稿数字。宏名全部预先声明，缺数据的给「--」，稿件因此在任何阶段都能编译。

运行方式:
    cd <项目根>
    python 02code/92_sx_tables.py                 # 读正式汇总，写回生成块
    python 02code/92_sx_tables.py --stage pilot   # 读预跑汇总（数值包 \\pend{}）
    python 02code/92_sx_tables.py --check         # 只核对，生成块与结果文件不一致时退出码 1

输出文件:
    06doc/01manuscript/Elsevier_en.tex、Elsevier_zh.tex — 生成块 SX（数字宏）
    06doc/01manuscript/Supplementary_en.tex、Supplementary_zh.tex — 生成块 S20–S30（S25、S26 各分 A、B 两块）与 SX
    04outputs/92_sx_tables.xlsx — 宏值、定性断言与各来源是否正式
    05logs/92_sx_tables_YY-MM-DD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

import export_utils as eu

m68: Any = importlib.import_module("68si_tables")
m71: Any = importlib.import_module("71_source_only_tuning")
m77: Any = importlib.import_module("77_generality_tables")
num = m68.num
Wrapper = Callable[[str], str]
FAMILIES: list[str] = m77.FAMILIES
DEEP: list[str] = m77.DEEP
FAM_KEY: dict[str, str] = m77.FAM_KEY
FAM_TEX: dict[str, str] = m77.FAM_TEX
MAIN_SETS: list[str] = m77.MAIN_SETS
DS_TEX: dict[str, dict[str, str]] = m77.DS_TEX
held_tex: Callable[[str, str], str] = m77._held_tex

STEMS = {
    81: "81_within_year_split",
    82: "82_chemometric_paths",
    83: "83_symmetric_exposure",
    84: "84_inference_robustness",
    85: "85_checkpoint_paired",
    86: "86_encoder_hparam_exposure",
    87: "87_kcurve_occlusion",
    88: "88_supervised_unlabelled",
    89: "89_repeated_half_split",
    90: "90_mango_region_task",
    91: "91_mango_half_unit",
    93: "93_svm_cap_sensitivity",
}
BIN81_TAG = "bin512"  # 81 号重建臂 512 个波段那次运行的 --tag
WITHIN_SCENE = "年内 7:3"  # 81 号表 d 的场景名
N_SEEDS = len(eu.RECOMMENDED_SEEDS)
MAIN_SCOPE = "主口径（苹果两年设计 + 公开数据集）"
THREE_SCOPE = "苹果三年设计"  # 87、88 号表里苹果三年设计的口径名
K_KEY = {1: "A", 5: "B", 10: "C", 20: "D"}
CON_KEY = {"Δ_rep": "Rep", "Δ_half": "Half", "Δ_lab": "Lab"}
CON_ORDER = ("Δ_rep", "Δ_lab", "Δ_half")  # 表里的排列次序，同表 S16
CON_TEX: dict[str, str] = {k: m77.CON_TEX[k] for k in CON_KEY}
Book = dict[str, pd.DataFrame]
EMPTY = pd.DataFrame()


def wrap(formal: bool) -> Wrapper:
    """预览时把数值包进 \\pend{}（77 号的 Wrap）。"""
    w: Wrapper = m77.Wrap(formal)
    return w


# ── 读表 ─────────────────────────────────────────────────────────────────────
def load_book(n: int, stage: str) -> Book:
    """读某个脚本的汇总工作簿；表按「表a」「表e1」的字母数字作键，其余表按表名作键。缺文件返回空字典。

    84、93 号是对已有逐种子表的后处理，只有一份工作簿，不分预跑与正式。
    """
    suffix = "" if stage == "formal" or n in (84, 93) else "__pilot"
    path = os.path.join(eu.OUTPUT_DIR, f"{STEMS[n]}{suffix}.xlsx")
    if not os.path.exists(path):
        return {}
    out = _read_sheets(path)
    if n == 81:  # 重建臂 512 个波段的一次运行（--tag bin512）：其表 d 记作 d512
        tagged = os.path.join(eu.OUTPUT_DIR, f"{STEMS[n]}__{BIN81_TAG}{suffix}.xlsx")
        out["d512"] = _read_sheets(tagged).get("d", EMPTY) if os.path.exists(tagged) else EMPTY
    return out


def _read_sheets(path: str) -> Book:
    xl = pd.ExcelFile(path)
    out: Book = {}
    for sh in xl.sheet_names:
        name = str(sh)
        key = name[1:].split("：")[0] if name.startswith("表") and "：" in name else name
        out[key] = xl.parse(sh)
    return out


def _col_min(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df or df[col].dropna().empty:
        return float("nan")
    return float(df[col].dropna().min())


def is_formal(n: int, b: Book, stage: str) -> bool:
    """某个脚本的汇总是否为五个正式种子齐全的正式结果（81 号看划分次数，84 号看自助次数）。"""
    if stage != "formal" or not b:
        return False
    t = b.get
    if n == 81:
        d512 = t("d512", EMPTY)
        within512 = d512[d512["场景"] == WITHIN_SCENE] if not d512.empty else d512
        return _col_min(t("a", EMPTY), "次数") >= 200 and _col_min(within512, "次数") >= 200
    if n == 82:
        return _col_min(t("a", EMPTY), "种子数") >= N_SEEDS and _col_min(t("c", EMPTY), "种子数") >= N_SEEDS
    if n == 83:
        a = t("a", EMPTY)
        dom = a[a["层级"] == "留出域"] if not a.empty else a
        return _col_min(dom, "种子数") >= N_SEEDS and _col_min(t("b", EMPTY), "种子数") >= N_SEEDS
    if n == 84:
        return _col_min(t("compute", EMPTY), "n_boot") >= 2000
    if n in (85, 86):
        s = t("b", EMPTY)
        full = (not s.empty) and bool((s["留出域数"] == s["应有留出域数"]).all())
        return full and _col_min(t("a", EMPTY), "种子数") >= N_SEEDS
    if n == 87:
        a = t("a", EMPTY)
        tot = (a["种子数"] + a["跳过种子数"]).min() if not a.empty else 0
        return tot >= N_SEEDS and _col_min(t("e", EMPTY), "种子数") >= N_SEEDS
    if n == 88:
        return (
            _col_min(t("a", EMPTY), "Δ_rep_种子数") >= N_SEEDS and _col_min(t("c", EMPTY), "Δ_unsup_种子数") >= N_SEEDS
        )
    if n == 89:
        c = t("c", EMPTY)
        return _col_min(c, "种子数") >= N_SEEDS and _col_min(c, "划分数") >= 5
    if n == 90:
        return _col_min(t("b", EMPTY), "种子数") >= N_SEEDS
    if n == 91:
        return _col_min(t("b", EMPTY), "Δ_half_配对种子数") >= N_SEEDS
    if n == 93:
        c = t("compute", EMPTY)
        return (not c.empty) and bool((c["自检"] == "通过").all()) and one(t("b", EMPTY), 口径=MAIN_SCOPE) is not None
    return False


def one(df: pd.DataFrame, **kv: Any) -> pd.Series | None:
    """满足全部等式条件的唯一一行；没有或不止一行时返回 None。"""
    if df.empty:
        return None
    m = df
    for k, v in kv.items():
        if k not in m:
            return None
        m = m[m[k].astype(str) == str(v)]
    return m.iloc[0] if len(m) == 1 else None


def val(r: pd.Series | None, col: str) -> float:
    if r is None or col not in r or pd.isna(r[col]):
        return float("nan")
    return float(r[col])


def txt(r: pd.Series | None, col: str) -> str:
    if r is None or col not in r or pd.isna(r[col]):
        return ""
    return str(r[col])


def pct(v: float, nd: int = 1) -> str:
    return f"{100 * v:.{nd}f}" if np.isfinite(v) else "--"


def n3(v: float) -> str:
    return num(v) if np.isfinite(v) else "--"


def of(frac: str) -> str:
    """「8/18」→「8 of 18」（宏里用；表里保留斜线写法）。"""
    return frac.replace("/", " of ") if frac else "--"


def pos_frac(values: pd.Series) -> str:
    v = values.dropna()
    return f"{int((v > 0).sum())}/{len(v)}" if len(v) else ""


class Macros:
    """宏表：先声明、后赋值；按来源脚本决定是否包 \\pend{}。"""

    def __init__(self, formal: dict[int, bool]) -> None:
        self.formal = formal
        self.vals: dict[str, str] = {}
        self.src: dict[str, int] = {}

    def put(self, key: str, src: int, text: str = "--") -> None:
        if not key.isalpha():
            raise ValueError(f"宏名只能含字母：{key}")
        self.vals[key] = text if text else "--"
        self.src[key] = src

    def block(self) -> str:
        lines = []
        for k, v in self.vals.items():
            shown = v if v == "--" or self.formal.get(self.src[k], False) else rf"\pend{{{v}}}"
            lines.append(rf"\newcommand{{\{k}}}{{{shown}}}")
        return "\n".join(lines)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "宏": list(self.vals),
                "值": list(self.vals.values()),
                "来源脚本": [self.src[k] for k in self.vals],
                "正式": [self.formal.get(self.src[k], False) for k in self.vals],
            }
        )


# ── 表 S20：年内随机划分与跨年（81 号）───────────────────────────────────────
LEARN81 = {"最近质心": "Nc", "SVM": "Svm", "PLS-DA": "Pls"}
LEARN81_TEX = {
    "en": {"最近质心": "Nearest centroid", "SVM": "Support vector machine", "PLS-DA": "PLS-DA"},
    "zh": {"最近质心": "最近质心", "SVM": "支持向量机", "PLS-DA": "PLS-DA"},
}
WITHIN81 = (("A", 2018, "按光谱"), ("B", 2019, "按光谱"), ("C", 2025, "按光谱"), ("Cf", 2025, "按苹果编号"))
RECON = "FD-PCA-SRDA 融合"
S20_TEX = {
    "en": {
        "recon": "Rebuilt pipeline, 2048 points",
        "recon512": "Rebuilt pipeline, 512 bands",
        "pub": "Published value",
    },
    "zh": {"recon": "重建流程，2048 个点", "recon512": "重建流程，512 个波段", "pub": "已发表值"},
}
REC_KEYS = {"": "d", "Bin": "d512"}  # 宏名里的段 → 重建臂的表（2048 个点、512 个波段）


def data_s20(b: Book) -> dict[str, Any]:
    a, cross = b.get("a", EMPTY), b.get("b", EMPTY)
    out: dict[str, Any] = {
        "within": {},
        "cross": {},
        "edge": {},
        "cross_edge": {},
        "pub": (np.nan, np.nan),
        "n": np.nan,
    }
    for learner in LEARN81:
        for key, year, unit in WITHIN81:
            r = one(a, 年份=year, 划分方式=unit, 学习器=learner)
            out["within"][learner, key] = (val(r, "准确率均值"), val(r, "准确率SD"), val(r, "宏F1均值"))
            out["edge"][learner, key] = val(r, "落在网格外沿次数")
        for key, held in (("A", 2018), ("B", 2019)):
            r = one(cross, 设计="two_year", 留出年=held, 学习器=learner)
            out["cross"][learner, key] = (val(r, "准确率"), val(r, "宏F1"))
            out["cross_edge"][learner, key] = val(r, "落在网格外沿")
    out["n"] = _col_min(a, "次数")
    for seg, sheet in REC_KEYS.items():
        d = b.get(sheet, EMPTY)
        rec_w: dict[str, tuple[float, float, float]] = {}
        rec_c: dict[str, tuple[float, float, float]] = {}
        for key, year in (("A", 2018), ("B", 2019)):
            r = one(d, 场景=WITHIN_SCENE, 训练年份=year, 测试年份=year, 学习器=RECON)
            rec_w[key] = (val(r, "测试集准确率均值"), val(r, "测试集准确率SD"), val(r, "测试集宏F1均值"))
            if key == "B" and not seg:
                out["pub"] = (val(r, "已发表测试集准确率均值"), val(r, "已发表测试集准确率SD"))
        for key, train, test in (("A", 2019, 2018), ("B", 2018, 2019)):
            r = one(d, 场景="跨年", 训练年份=train, 测试年份=test, 学习器=RECON)
            rec_c[key] = (val(r, "测试集准确率均值"), val(r, "测试集宏F1均值"), val(r, "训练集准确率均值"))
        out[f"rec_within{seg}"], out[f"rec_cross{seg}"] = rec_w, rec_c
    return out


def _mean_sd(m: float, s: float, nd: int = 1) -> str:
    return f"{pct(m, nd)} ({pct(s, nd)})" if np.isfinite(m) and np.isfinite(s) else "--"


def tex_s20(s: dict[str, Any], w: Wrapper, lang: str) -> str:
    lines: list[str] = []
    for learner in LEARN81:
        cells = [_mean_sd(*s["within"][learner, key][:2]) for key, _, _ in WITHIN81]
        cells += [pct(s["cross"][learner, key][0]) for key in ("A", "B")]
        lines.append(rf"{LEARN81_TEX[lang][learner]} & {' & '.join(w(c) for c in cells)} \\")
    for seg, label in (("", "recon"), ("Bin", "recon512")):
        rec_w, rec_c = s[f"rec_within{seg}"], s[f"rec_cross{seg}"]
        rec = [_mean_sd(*rec_w["A"][:2]), _mean_sd(*rec_w["B"][:2]), "--", "--"]
        rec += [pct(rec_c[key][0]) for key in ("A", "B")]
        lines.append(rf"{S20_TEX[lang][label]} & {' & '.join(w(c) for c in rec)} \\")
    pub = _mean_sd(float(s["pub"][0]), float(s["pub"][1]), 2)
    lines.append(rf"{S20_TEX[lang]['pub']} & -- & {pub} & -- & -- & -- & -- \\")
    return "\n".join(lines)


def macros_s20(s: dict[str, Any], mc: Macros) -> None:
    for learner, lk in LEARN81.items():
        for key, _, _ in WITHIN81:
            m, sd, f1 = s["within"][learner, key]
            mc.put(f"SxWy{lk}{key}", 81, pct(m) + r"\%" if np.isfinite(m) else "")
            mc.put(f"SxWy{lk}{key}Sd", 81, pct(sd) if np.isfinite(sd) else "")
            mc.put(f"SxWyF{lk}{key}", 81, n3(f1))
        for key in ("A", "B"):
            acc, f1 = s["cross"][learner, key]
            mc.put(f"SxCy{lk}{key}", 81, pct(acc) + r"\%" if np.isfinite(acc) else "")
            mc.put(f"SxCyF{lk}{key}", 81, n3(f1))
    for seg in REC_KEYS:
        for key in ("A", "B"):
            m, sd, _ = s[f"rec_within{seg}"][key]
            mc.put(f"SxWyRec{seg}{key}", 81, pct(m) + r"\%" if np.isfinite(m) else "")
            mc.put(f"SxWyRec{seg}{key}Sd", 81, pct(sd) if np.isfinite(sd) else "")
            acc, _, train = s[f"rec_cross{seg}"][key]
            mc.put(f"SxCyRec{seg}{key}", 81, pct(acc) + r"\%" if np.isfinite(acc) else "")
            mc.put(f"SxCyRecTrain{seg}{key}", 81, pct(train) + r"\%" if np.isfinite(train) else "")
    mc.put("SxWyN", 81, str(int(s["n"])) if np.isfinite(s["n"]) else "")
    # 所选超参数落在网格外沿的次数：年内四种划分各 n 次中的最少与最多（SVM、PLS-DA），两年设计跨年两次中的次数
    for learner, lk in (("SVM", "Svm"), ("PLS-DA", "Pls")):
        edges = [s["edge"][learner, key] for key, _, _ in WITHIN81 if np.isfinite(s["edge"][learner, key])]
        mc.put(f"SxWyEdge{lk}Min", 81, str(int(min(edges))) if edges else "")
        mc.put(f"SxWyEdge{lk}Max", 81, str(int(max(edges))) if edges else "")
        cross_e = [s["cross_edge"][learner, key] for key in ("A", "B")]
        ok = all(np.isfinite(v) for v in cross_e)
        mc.put(f"SxCyEdge{lk}", 81, str(int(sum(cross_e))) if ok else "")
    # 年内减跨年的最小差（百分点）：三种学习器 × 2018、2019 两个留出年
    gaps = [
        s["within"][learner, key][0] - s["cross"][learner, key][0]
        for learner in LEARN81
        for key in ("A", "B")
        if np.isfinite(s["within"][learner, key][0]) and np.isfinite(s["cross"][learner, key][0])
    ]
    mc.put("SxWyGapMin", 81, num(100 * min(gaps), 1) if gaps else "")
    mc.put("SxWyGapMax", 81, num(100 * max(gaps), 1) if gaps else "")
    by_fruit = [
        s["within"][learner, "C"][0] - s["within"][learner, "Cf"][0]
        for learner in LEARN81
        if np.isfinite(s["within"][learner, "C"][0]) and np.isfinite(s["within"][learner, "Cf"][0])
    ]
    mc.put("SxWyFacetMin", 81, num(100 * min(by_fruit), 1) if by_fruit else "")
    mc.put("SxWyFacetMax", 81, num(100 * max(by_fruit), 1) if by_fruit else "")


# ── 表 S21：化学计量学流程里的暴露（82 号）───────────────────────────────────
PATH82 = {"标度": "Sc", "波长选择": "Wl", "波长选择(半分)": "Wh"}
LEARN82 = {"Pooled-SVM": "Svm", "Pooled-PLS-DA": "Pls"}
LEARN82_CV = {"SVM-Kshot": "Svm", "PLS-DA-Kshot": "Pls"}
S21_TEX = {
    "en": {
        "标度": "Scaling statistics, unlabelled",
        "波长选择": "Wavelength selection, labelled",
        "波长选择(半分)": "Wavelength selection, disjoint half",
        "交叉验证重用": "Reuse of the selection score",
        "Pooled-SVM": "Pooled SVM",
        "Pooled-PLS-DA": "Pooled PLS-DA",
        "SVM-Kshot": "SVM",
        "PLS-DA-Kshot": "PLS-DA",
    },
    "zh": {
        "标度": "标度统计量（无标签）",
        "波长选择": "波长选择（带标签）",
        "波长选择(半分)": "波长选择（不相交的一半）",
        "交叉验证重用": "选参成绩的重用",
        "Pooled-SVM": "池化 SVM",
        "Pooled-PLS-DA": "池化 PLS-DA",
        "SVM-Kshot": "SVM",
        "PLS-DA-Kshot": "PLS-DA",
    },
}


def data_s21(b: Book) -> pd.DataFrame:
    """主口径下各路径 × 学习器 × K 的留出域均值、范围与为正留出域。"""
    rows: list[dict[str, Any]] = []
    for sheet, col, paths, learners in (
        ("b", "暴露−无暴露", list(PATH82), list(LEARN82)),
        ("d", "选参最高分−评估成绩", ["交叉验证重用"], list(LEARN82_CV)),
    ):
        t = b.get(sheet, EMPTY)
        for path in paths:
            for learner in learners:
                for k in K_KEY:
                    r = one(t, 口径=MAIN_SCOPE, 路径=path, 学习器=learner, K=k)
                    rows.append(
                        {
                            "路径": path,
                            "学习器": learner,
                            "K": k,
                            "均值": val(r, f"{col}_均值"),
                            "最小": val(r, f"{col}_最小"),
                            "最大": val(r, f"{col}_最大"),
                            "为正留出域": txt(r, "为正留出域"),
                        }
                    )
    return pd.DataFrame(rows)


def tex_s21(d: pd.DataFrame, w: Wrapper, lang: str) -> str:
    lab = S21_TEX[lang]
    lines: list[str] = []
    for path in [*PATH82, "交叉验证重用"]:
        sub = d[d["路径"] == path]
        for i, learner in enumerate(dict.fromkeys(sub["学习器"])):
            cells = [w(n3(val(one(sub, 学习器=learner, K=k), "均值"))) for k in K_KEY]
            r5 = one(sub, 学习器=learner, K=5)
            cells += [w(txt(r5, "为正留出域") or "--"), w(n3(val(r5, "最小"))), w(n3(val(r5, "最大")))]
            lines.append(rf"{lab[path] if i == 0 else ''} & {lab[str(learner)]} & {' & '.join(cells)} \\")
        if path != "交叉验证重用":
            lines.append(r"\addlinespace")
    return "\n".join(lines)


def macros_s21(d: pd.DataFrame, mc: Macros) -> None:
    keys = {**{p: (k, LEARN82) for p, k in PATH82.items()}, "交叉验证重用": ("Cv", LEARN82_CV)}
    for path, (pk, learners) in keys.items():
        for learner, lk in learners.items():
            means: list[float] = []
            for k, kk in K_KEY.items():
                r = one(d, 路径=path, 学习器=learner, K=k)
                mc.put(f"Sx{pk}{lk}{kk}", 82, n3(val(r, "均值")))
                mc.put(f"Sx{pk}{lk}Pos{kk}", 82, of(txt(r, "为正留出域")))
                if np.isfinite(val(r, "均值")):
                    means.append(val(r, "均值"))
            r5 = one(d, 路径=path, 学习器=learner, K=5)
            mc.put(f"Sx{pk}{lk}MinB", 82, n3(val(r5, "最小")))
            mc.put(f"Sx{pk}{lk}MaxB", 82, n3(val(r5, "最大")))
            mc.put(f"Sx{pk}{lk}KMin", 82, n3(min(means)) if means else "")
            mc.put(f"Sx{pk}{lk}KMax", 82, n3(max(means)) if means else "")


# ── 表 S22：基线的对等暴露（83 号）───────────────────────────────────────────
LEARN83 = {"池化SVM": "Svm", "池化PLS-DA": "Pls"}
PN_FREE, PN_EXP = "原型网络(R^h)", "原型网络(S_half)"
S22_TEX = {
    "en": {"mean": "Mean over 18 domains", "pos": "Positive domains"},
    "zh": {"mean": "18 个留出域的均值", "pos": "为正的留出域"},
}


def data_s22(b: Book) -> dict[str, Any]:
    """逐留出域（主口径）的无暴露值、暴露值与增量：两个池化学习器取 83 号表a，原型网络（CNN）取表b。"""
    a, rank, cnt = b.get("a", EMPTY), b.get("b", EMPTY), b.get("c", EMPTY)
    dom = a[(a["层级"] == "留出域") & a["dataset"].isin(MAIN_SETS)] if not a.empty else a
    rows: list[dict[str, Any]] = []
    for learner in LEARN83:
        sub = dom[dom["学习器"] == learner] if not dom.empty else dom
        for _, r in sub.iterrows():
            rows.append(
                {
                    "dataset": r["dataset"],
                    "held": r["held"],
                    "路线": learner,
                    "无暴露": float(r["无暴露"]),
                    "暴露": float(r["暴露"]),
                    "增量": float(r["暴露增量"]),
                }
            )
    pn = rank[(rank["编码器"] == "cnn") & rank["dataset"].isin(MAIN_SETS)] if not rank.empty else rank
    for _, r in pn.iterrows():
        free, exp = float(r[PN_FREE]), float(r[PN_EXP])
        rows.append(
            {
                "dataset": r["dataset"],
                "held": r["held"],
                "路线": "原型网络",
                "无暴露": free,
                "暴露": exp,
                "增量": exp - free,
            }
        )
    return {"dom": pd.DataFrame(rows), "rank": rank, "cnt": cnt}


def tex_s22(s: dict[str, Any], w: Wrapper, lang: str) -> str:
    dom: pd.DataFrame = s["dom"]
    routes = [*LEARN83, "原型网络"]
    if dom.empty:
        return r"\multicolumn{10}{@{}l}{--} \\"
    lines: list[str] = []
    for ds in MAIN_SETS:
        cells: list[str] = []
        for route in routes:
            sub = dom[(dom["dataset"] == ds) & (dom["路线"] == route)]
            cells += [w(n3(float(sub[c].mean()))) if len(sub) else "--" for c in ("无暴露", "暴露", "增量")]
        lines.append(rf"{DS_TEX[lang][ds]} & {' & '.join(cells)} \\")
    lines.append(r"\midrule")
    cells = []
    for route in routes:
        sub = dom[dom["路线"] == route]
        cells += [w(n3(float(sub[c].mean()))) if len(sub) else "--" for c in ("无暴露", "暴露", "增量")]
    lines.append(rf"{S22_TEX[lang]['mean']} & {' & '.join(cells)} \\")
    cells = []
    for route in routes:
        sub = dom[dom["路线"] == route]
        cells += ["", "", w(pos_frac(sub["增量"]) or "--")]
    lines.append(rf"{S22_TEX[lang]['pos']} & {' & '.join(cells)} \\")
    return "\n".join(lines)


def macros_s22(s: dict[str, Any], mc: Macros) -> None:
    dom: pd.DataFrame = s["dom"]
    for route, rk in (*LEARN83.items(), ("原型网络", "Pn")):
        sub = dom[dom["路线"] == route] if not dom.empty else dom
        has = len(sub) > 0
        mc.put(f"SxSym{rk}", 83, n3(float(sub["增量"].mean())) if has else "")
        mc.put(f"SxSym{rk}Pos", 83, of(pos_frac(sub["增量"])) if has else "")
        mc.put(f"SxSym{rk}Min", 83, n3(float(sub["增量"].min())) if has else "")
        mc.put(f"SxSym{rk}Max", 83, n3(float(sub["增量"].max())) if has else "")
        mc.put(f"SxSym{rk}Free", 83, n3(float(sub["无暴露"].mean())) if has else "")
        mc.put(f"SxSym{rk}Exp", 83, n3(float(sub["暴露"].mean())) if has else "")
    rank: pd.DataFrame = s["rank"]
    cnt: pd.DataFrame = s["cnt"]
    for fam in FAMILIES:
        fk = FAM_KEY[fam]
        r = one(cnt, 编码器=fam, 口径=MAIN_SCOPE, K=5)
        same, total = val(r, "第一名相同数"), val(r, "留出域数")
        mc.put(f"SxSymSame{fk}", 83, f"{int(same)} of {int(total)}" if np.isfinite(same) and np.isfinite(total) else "")
        sub = rank[(rank["编码器"] == fam) & rank["dataset"].isin(MAIN_SETS)] if not rank.empty else rank
        has = len(sub) > 0
        mc.put(f"SxSymFreePn{fk}", 83, str(int((sub["无泄漏第一"] == PN_FREE).sum())) if has else "")
        mc.put(f"SxSymExpPn{fk}", 83, str(int((sub["对等暴露第一"] == PN_EXP).sum())) if has else "")
        pooled = sub["无泄漏第一"].astype(str).str.startswith("池化") if has else pd.Series(dtype=bool)
        mc.put(f"SxSymFreePool{fk}", 83, str(int(pooled.sum())) if has else "")
        pooled_e = sub["对等暴露第一"].astype(str).str.startswith("池化") if has else pd.Series(dtype=bool)
        mc.put(f"SxSymExpPool{fk}", 83, str(int(pooled_e.sum())) if has else "")
    # 池化 SVM 在选参或评估中出现过停在迭代上限的拟合的留出域数（主口径，任一种子出现即计；口径同 93 号）；
    # 其中每个种子的全部评估拟合都停在上限的留出域数
    info = s.get("info", EMPTY)
    long: pd.DataFrame = s.get("long", EMPTY)
    mc.put("SxSymCap", 83, "")
    mc.put("SxCapFullN", 83, "")
    if isinstance(info, pd.DataFrame) and not info.empty and all(c in info for c in CAP_COLS):
        main_info = info[info["dataset"].isin(MAIN_SETS)]
        hit = (main_info.groupby(["dataset", "held"])[list(CAP_COLS)].max() > 0).any(axis=1)
        mc.put("SxSymCap", 83, f"{int(hit.sum())} of {len(hit)}")
        fits = svm_eval_fits(long)
        if fits:
            full = [
                all(
                    int(n) == fits.get((str(ds), str(held), int(seed)), -1)
                    for n, seed in zip(g[CAP_COLS[1]], g["种子"], strict=True)
                )
                for (ds, held), g in main_info.groupby(["dataset", "held"])
            ]
            mc.put("SxCapFullN", 83, str(sum(full)))
    # 池化 PLS-DA 所选潜变量数取到候选最大值的格数（主口径，留出域 × 种子；两臂共用所选值，取无暴露臂的行）
    pls = long[(long["学习器"] == "池化PLS-DA") & (long["臂"] == "无暴露")] if not long.empty else long
    pls = pls[pls["dataset"].isin(MAIN_SETS)] if not pls.empty else pls
    top = str(max(m71.PLS_GRID))
    mc.put("SxSymPlsTop", 83, top)
    mc.put("SxSymPlsTopN", 83, str(int((pls["所选超参数"].astype(str) == top).sum())) if len(pls) else "")
    mc.put("SxSymPlsCells", 83, str(len(pls)) if len(pls) else "")


def svm_eval_fits(long: pd.DataFrame) -> dict[tuple[str, str, int], int]:
    """83 号表 d：每个（数据集, 留出域, 种子）池化 SVM 在 B 半评估中的拟合次数（两臂的情节数之和）。"""
    if long.empty:
        return {}
    sub = long[long["学习器"] == "池化SVM"]
    n = sub.groupby(["dataset", "held", "种子"])["情节数"].sum()
    return {(str(ds), str(held), int(seed)): int(v) for (ds, held, seed), v in n.items()}


# ── S9.3 节：池化 SVM 迭代上限的灵敏度（93 号）──────────────────────────────
CAP_COLS = ("停在上限的拟合数(选参)", "停在上限的拟合数(评估)")
CAP_HIT = "有拟合停在上限的留出域_"  # 93 号表b 的列名前缀，后接上限


def macros_cap(b: Book, long83: pd.DataFrame, mc: Macros) -> None:
    """93 号表b 主口径：重跑的那个种子在两个上限下有拟合停在上限的留出域数，暴露增量之差的最大绝对值与均值；
    93 号表a：低上限下全部评估拟合都停在上限的留出域数、这些留出域在高上限下仍停在上限的评估拟合数的范围
    （每格的评估拟合次数取 83 号表 d 同一种子的情节数）。"""
    r = one(b.get("b", EMPTY), 口径=MAIN_SCOPE)
    caps = sorted(int(str(c)[len(CAP_HIT) :]) for c in (r.index if r is not None else []) if str(c).startswith(CAP_HIT))
    for key, cap in (("Lo", caps[0] if len(caps) == 2 else None), ("Hi", caps[-1] if len(caps) == 2 else None)):
        n = val(r, f"{CAP_HIT}{cap}") if cap is not None else float("nan")
        mc.put(f"SxCap{key}", 93, str(int(n)) if np.isfinite(n) else "")
    mc.put("SxCapDiffMax", 93, n3(val(r, "增量差_最大绝对值")))
    mc.put("SxCapDiffMean", 93, n3(val(r, "增量差_均值")))
    for key in ("SxCapFits", "SxCapLoFullN", "SxCapHiEvalMin", "SxCapHiEvalMax"):
        mc.put(key, 93, "")
    a, seed = b.get("a", EMPTY), val(r, "种子")
    fits = svm_eval_fits(long83)
    if a.empty or len(caps) != 2 or not np.isfinite(seed) or not fits:
        return
    lo_col, hi_col = f"{CAP_COLS[1]}_{caps[0]}", f"{CAP_COLS[1]}_{caps[1]}"
    main_a = a[a["主口径"].astype(bool)]
    doms = zip(main_a["dataset"], main_a["held"], strict=True)
    n_fit = [fits.get((str(ds), str(held), int(seed)), -1) for ds, held in doms]
    if len(set(n_fit)) != 1 or n_fit[0] <= 0:
        return  # 各留出域的评估拟合次数不同或缺，宏留空
    full = main_a[main_a[lo_col].astype(int) == n_fit[0]]
    mc.put("SxCapFits", 93, str(n_fit[0]))
    mc.put("SxCapLoFullN", 93, str(len(full)))
    if len(full):
        mc.put("SxCapHiEvalMin", 93, str(int(full[hi_col].min())))
        mc.put("SxCapHiEvalMax", 93, str(int(full[hi_col].max())))


# ── 表 S23：检查点与编码器超参数（85、86 号）─────────────────────────────────
def _family_summary(b: Book, name: str) -> pd.DataFrame:
    """85、86 号「表b：按族汇总」主口径的均值与为正留出域，另加苹果两种设计的折均值与逐格计数（表a、表c）。"""
    summ, per, cell = b.get("b", EMPTY), b.get("a", EMPTY), b.get("c", EMPTY)
    rows: list[dict[str, Any]] = []
    for fam in FAMILIES:
        r = one(summ, 口径=MAIN_SCOPE, 编码器=fam)
        row: dict[str, Any] = {
            "编码器": fam,
            "均值": val(r, f"{name}_均值"),
            "为正留出域": txt(r, f"{name}_为正留出域"),
            "半分均值": val(r, f"{name}_half_均值"),
            "半分为正留出域": txt(r, f"{name}_half_为正留出域"),
        }
        for design, tag in (("apple_two_year", "两年"), ("apple_three_year", "三年")):
            sub = per[(per["dataset"] == design) & (per["编码器"] == fam)] if not per.empty else per
            row[f"{tag}均值"] = float(sub[name].mean()) if len(sub) else float("nan")
            row[f"{tag}半分均值"] = float(sub[f"{name}_half"].mean()) if len(sub) else float("nan")
            for _, fr in sub.iterrows():
                row[f"{tag}{fr['held']}"] = float(fr[name])
            cs = cell[(cell["dataset"] == design) & (cell["编码器"] == fam)] if not cell.empty else cell
            row[f"{tag}为正格"] = f"{int((cs[name] > 0).sum())}/{len(cs)}" if len(cs) else ""
            row[f"{tag}同选格"] = f"{int(cs[f'{name}_同选'].astype(bool).sum())}/{len(cs)}" if len(cs) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def data_s23(b85: Book, b86: Book) -> dict[str, pd.DataFrame]:
    return {"ckpt": _family_summary(b85, "Δ_ckpt"), "hp": _family_summary(b86, "Δ_hp")}


def tex_s23(s: dict[str, pd.DataFrame], w85: Wrapper, w86: Wrapper, lang: str) -> str:
    lines: list[str] = []
    for fam in FAMILIES:
        cells: list[str] = []
        for key, w in (("ckpt", w85), ("hp", w86)):
            r = one(s[key], 编码器=fam)
            for c_mean, c_pos in (("均值", "为正留出域"), ("半分均值", "半分为正留出域")):
                v = val(r, c_mean)
                cells += [w(n3(v)) if np.isfinite(v) else "--", w(txt(r, c_pos)) if txt(r, c_pos) else "--"]
        lines.append(rf"{FAM_TEX[fam]} & {' & '.join(cells)} \\")
    return "\n".join(lines)


def macros_s23(s: dict[str, pd.DataFrame], mc: Macros) -> None:
    for key, pre, src, fams in (("ckpt", "Ck", 85, DEEP), ("hp", "Hp", 86, FAMILIES)):
        t = s[key]
        means: list[float] = []
        halves: list[float] = []
        for fam in FAMILIES:
            fk = FAM_KEY[fam]
            r = one(t, 编码器=fam) if fam in fams else None
            mc.put(f"Sx{pre}{fk}", src, n3(val(r, "均值")))
            mc.put(f"Sx{pre}{fk}Pos", src, of(txt(r, "为正留出域")))
            mc.put(f"Sx{pre}H{fk}", src, n3(val(r, "半分均值")))
            mc.put(f"Sx{pre}H{fk}Pos", src, of(txt(r, "半分为正留出域")))
            for tag, dk in (("两年", "Tw"), ("三年", "Th")):
                mc.put(f"Sx{pre}{dk}{fk}", src, n3(val(r, f"{tag}均值")))
                mc.put(f"Sx{pre}{dk}H{fk}", src, n3(val(r, f"{tag}半分均值")))
                mc.put(f"Sx{pre}{dk}{fk}Cells", src, of(txt(r, f"{tag}为正格")))
                mc.put(f"Sx{pre}{dk}{fk}Same", src, of(txt(r, f"{tag}同选格")))
                for letter, held in zip("ABC", ("Y2018", "Y2019", "Y2025"), strict=True):
                    if tag == "两年" and letter == "C":
                        continue
                    mc.put(f"Sx{pre}{dk}{fk}{letter}", src, n3(val(r, f"{tag}{held}")))
            if np.isfinite(val(r, "均值")):
                means.append(val(r, "均值"))
            if np.isfinite(val(r, "半分均值")):
                halves.append(val(r, "半分均值"))
        mc.put(f"Sx{pre}Min", src, n3(min(means)) if means else "")
        mc.put(f"Sx{pre}Max", src, n3(max(means)) if means else "")
        mc.put(f"Sx{pre}HMin", src, n3(min(halves)) if halves else "")
        mc.put(f"Sx{pre}HMax", src, n3(max(halves)) if halves else "")


# ── 表 S24、S25：K 曲线与波段遮蔽（87 号）────────────────────────────────────
def tex_s24(b: Book, w: Wrapper) -> str:
    t = b.get("b", EMPTY)
    lines: list[str] = []
    for fam in FAMILIES:
        cells: list[str] = []
        for k in K_KEY:
            r = one(t, 口径=MAIN_SCOPE, 编码器族=fam, K=k)
            v = val(r, "Δ_rep均值")
            cells += [w(n3(v)) if np.isfinite(v) else "--", w(txt(r, "为正/可评")) if txt(r, "为正/可评") else "--"]
        lines.append(rf"{FAM_TEX[fam]} & {' & '.join(cells)} \\")
    return "\n".join(lines)


def macros_s24(b: Book, mc: Macros) -> None:
    t = b.get("b", EMPTY)
    for fam in FAMILIES:
        fk = FAM_KEY[fam]
        vals: list[float] = []
        for k, kk in K_KEY.items():
            r = one(t, 口径=MAIN_SCOPE, 编码器族=fam, K=k)
            mc.put(f"SxK{fk}{kk}", 87, n3(val(r, "Δ_rep均值")))
            mc.put(f"SxK{fk}Pos{kk}", 87, of(txt(r, "为正/可评")))
            if np.isfinite(val(r, "Δ_rep均值")):
                vals.append(val(r, "Δ_rep均值"))
        mc.put(f"SxK{fk}Min", 87, n3(min(vals)) if vals else "")
        mc.put(f"SxK{fk}Max", 87, n3(max(vals)) if vals else "")
    r20 = one(t, 口径=MAIN_SCOPE, 编码器族="cnn", K=20)
    n_ok = val(r20, "可评留出域数")
    mc.put("SxKEvalD", 87, str(int(n_ok)) if np.isfinite(n_ok) else "")
    for ds, dk in K5_SETS.items():
        mc.put(f"SxKFive{dk}Cnn", 87, n3(k5_ds_mean(b.get("a", EMPTY), ds)))


# 正文 3.3 节、S9.7 节与无标签暴露并列的三个数据集（苹果两种设计与猕猴桃）
K5_SETS = {"apple_two_year": "Ap", "apple_three_year": "Th", "kiwifruit_season": "Ki"}


def _k5_rows(a: pd.DataFrame, ds: str, fam: str) -> pd.DataFrame:
    """87 号表 a 中一个数据集、一个编码器族在 K=5 时的逐留出域行；苹果三年设计取其单列的口径，其余取主口径。"""
    if a.empty:
        return a
    scope = THREE_SCOPE if ds == "apple_three_year" else MAIN_SCOPE
    return a[(a["口径"] == scope) & (a["数据集"] == ds) & (a["编码器族"] == fam) & (a["K"].astype(int) == 5)]


def k5_ds_mean(a: pd.DataFrame, ds: str, fam: str = "cnn") -> float:
    """87 号表 a 中一个数据集在 K=5 时 S_src−R 的留出域均值。

    K=5 的逐域值与主分析的逐种子表逐格核对过（87 号自检），所以主口径下这就是正文表 2 的那一格。
    """
    sub = _k5_rows(a, ds, fam)
    return float(sub["Δ_rep"].mean()) if len(sub) else float("nan")


def k5_dom(a: pd.DataFrame, ds: str, held: str, fam: str = "cnn") -> float:
    """87 号表 a 中一个留出域在 K=5 时的 S_src−R（种子均值）。"""
    sub = _k5_rows(a, ds, fam)
    sub = sub[sub["留出域"].astype(str) == held] if len(sub) else sub
    return float(sub["Δ_rep"].iloc[0]) if len(sub) == 1 else float("nan")


OCC_SETS = ["apple_two_year", "mango_season", "kiwifruit_season"]
PIG_COL = {"R": "606–699nm正下降占比_R", "S_src": "606–699nm正下降占比_S_src"}
S25_ARM = {"R": "R", "S_src": r"S\textsubscript{src}"}
BLANK8 = r"\multicolumn{8}{@{}l}{--} \\"


def tex_s25(b: Book, w: Wrapper, lang: str) -> tuple[str, str]:
    """表 S25 的两块表体：(S25A 秩相关，逐留出域；S25B 苹果两折 606–699 nm 的份额，逐折逐臂)。"""
    t = b.get("e", EMPTY)
    if t.empty:
        return BLANK8, BLANK8
    rho: list[str] = []
    for ds in OCC_SETS:
        sub = t[t["数据集"] == ds]
        for held in sorted(sub["留出域"].astype(str).unique()):
            cells = []
            for fam in FAMILIES:
                v = val(one(sub, 留出域=held, 编码器族=fam), "秩相关_种子均值曲线")
                cells.append(w(num(v, 2)) if np.isfinite(v) else "--")
            rho.append(rf"{DS_TEX[lang][ds]} & {held_tex(ds, held)} & {' & '.join(cells)} \\")
    pig: list[str] = []
    sub = t[t["数据集"] == "apple_two_year"]
    for held in sorted(sub["留出域"].astype(str).unique()):
        for arm, col in PIG_COL.items():
            cells = []
            for fam in FAMILIES:
                v = val(one(sub, 留出域=held, 编码器族=fam), col)
                cells.append(w(pct(v, 0)) if np.isfinite(v) else "--")
            pig.append(rf"{held_tex('apple_two_year', held)} & {S25_ARM[arm]} & {' & '.join(cells)} \\")
    return "\n".join(rho) or BLANK8, "\n".join(pig) or BLANK8


def macros_s25(b: Book, mc: Macros) -> None:
    t = b.get("e", EMPTY)
    main = t[t["数据集"].isin(OCC_SETS)] if not t.empty else t
    rho = main["秩相关_种子均值曲线"].dropna() if not main.empty else pd.Series(dtype=float)
    mc.put("SxOccRhoMin", 87, num(float(rho.min()), 2) if len(rho) else "")
    mc.put("SxOccRhoMax", 87, num(float(rho.max()), 2) if len(rho) else "")
    mc.put("SxOccRhoMed", 87, num(float(rho.median()), 2) if len(rho) else "")
    mc.put("SxOccN", 87, str(len(main.drop_duplicates(["数据集", "留出域"]))) if len(main) else "")
    ap = main[main["数据集"] == "apple_two_year"] if not main.empty else main
    for fam in FAMILIES:
        fk = FAM_KEY[fam]
        sub = ap[ap["编码器族"] == fam] if not ap.empty else ap
        has = len(sub) == 2
        mc.put(f"SxOccRho{fk}", 87, num(float(sub["秩相关_种子均值曲线"].mean()), 2) if has else "")
        for arm, col in PIG_COL.items():
            ak = "R" if arm == "R" else "S"
            mc.put(f"SxOccPig{ak}{fk}", 87, pct(float(sub[col].mean()), 0) + r"\%" if has else "")
            for letter, held in (("A", "Y2018"), ("B", "Y2019")):
                v = val(one(sub, 留出域=held), col)
                mc.put(f"SxOccPig{ak}{fk}{letter}", 87, pct(v, 0) + r"\%" if np.isfinite(v) else "")


# ── 表 S26：监督式训练与无标签暴露（88 号）───────────────────────────────────
SUP_FAMS = ["cnn", "resnet"]
UNLAB = {"掩码重建": "Mask", "对比": "Con"}
UNLAB_SETS = {"apple_two_year": "Ap", "apple_three_year": "Th", "kiwifruit_season": "Kw"}
UNLAB_MODES = [f"{mode}{tail}" for mode in UNLAB for tail in ("", "+源域情节训练")]  # 两种目标 × 两种读法
# S9.7 节点名的两格：逐留出域增量最大的一格；无标签增量超过同一留出域 CNN 的 S_src−R（K=5）的格
UNLAB_TOP = ("apple_three_year", "Y2025", "对比")
UNLAB_OVER = [("kiwifruit_season", "Y2018", "对比+源域情节训练")]
S26_TEX = {
    "en": {"掩码重建": "Masked reconstruction", "对比": "Contrastive"},
    "zh": {"掩码重建": "掩码重建", "对比": "对比"},
}


def tex_s26(b: Book, w: Wrapper, lang: str) -> tuple[str, str]:
    """表 S26 的两块表体：(S26A 监督式对情节式，逐对照逐族；S26B 无标签暴露，逐数据集逐预训练方式)。"""
    sup, un = b.get("b", EMPTY), b.get("d", EMPTY)
    lab = S26_TEX[lang]
    top: list[str] = []
    for contrast in CON_ORDER:
        for i, fam in enumerate(SUP_FAMS):
            r = one(sup, 口径=MAIN_SCOPE, 编码器族=fam, 对照=contrast)
            cells = []
            for kind in ("监督式", "情节式"):
                v = val(r, f"{kind}_均值")
                cells += [
                    w(n3(v)) if np.isfinite(v) else "--",
                    w(txt(r, f"{kind}_为正留出域")) if txt(r, f"{kind}_为正留出域") else "--",
                ]
            top.append(rf"{CON_TEX[contrast] if i == 0 else ''} & {FAM_TEX[fam]} & {' & '.join(cells)} \\")
    low: list[str] = []
    for ds in UNLAB_SETS:
        for i, mode in enumerate(UNLAB):
            cells = []
            for full in (mode, f"{mode}+源域情节训练"):
                r = one(un, 数据集=ds, 训练或预训练方式=full)
                v = val(r, "Δ_unsup_均值")
                cells += [
                    w(n3(v)) if np.isfinite(v) else "--",
                    w(txt(r, "Δ_unsup_为正留出域")) if txt(r, "Δ_unsup_为正留出域") else "--",
                ]
            low.append(rf"{DS_TEX[lang][ds] if i == 0 else ''} & {lab[mode]} & {' & '.join(cells)} \\")
    return "\n".join(top), "\n".join(low)


def macros_s26(b: Book, a87: pd.DataFrame, mc: Macros) -> None:
    sup, un = b.get("b", EMPTY), b.get("d", EMPTY)
    for contrast, ck in CON_KEY.items():
        for fam in SUP_FAMS:
            fk = FAM_KEY[fam]
            # 主口径三项对照都给；苹果三年设计只给 S_src−R（S9.7 节正文引用）
            for scope, sk in ((MAIN_SCOPE, ""), (THREE_SCOPE, "Th")):
                if sk and contrast != "Δ_rep":
                    continue
                r = one(sup, 口径=scope, 编码器族=fam, 对照=contrast)
                mc.put(f"SxSup{ck}{fk}{sk}", 88, n3(val(r, "监督式_均值")))
                mc.put(f"SxSup{ck}{fk}{sk}Pos", 88, of(txt(r, "监督式_为正留出域")))
                mc.put(f"SxEpi{ck}{fk}{sk}", 88, n3(val(r, "情节式_均值")))
                mc.put(f"SxEpi{ck}{fk}{sk}Pos", 88, of(txt(r, "情节式_为正留出域")))
    for ds, dk in UNLAB_SETS.items():
        for mode, mk in UNLAB.items():
            for full, ek in ((mode, ""), (f"{mode}+源域情节训练", "E")):
                r = one(un, 数据集=ds, 训练或预训练方式=full)
                mc.put(f"SxUn{mk}{ek}{dk}", 88, n3(val(r, "Δ_unsup_均值")))
                mc.put(f"SxUn{mk}{ek}{dk}Pos", 88, of(txt(r, "Δ_unsup_为正留出域")))
    allv = unlab_values(un)
    mc.put("SxUnMin", 88, n3(min(allv)) if allv else "")
    mc.put("SxUnMax", 88, n3(max(allv)) if allv else "")
    cells = unlab_cells(b.get("c", EMPTY))
    mc.put("SxUnDomMin", 88, n3(min(cells.values())) if cells else "")
    mc.put("SxUnDomMax", 88, n3(max(cells.values())) if cells else "")
    mc.put("SxUnTop", 88, n3(cells.get(UNLAB_TOP, float("nan"))))
    mc.put("SxUnTopRep", 87, n3(k5_dom(a87, UNLAB_TOP[0], UNLAB_TOP[1])))
    over = UNLAB_OVER[0]
    mc.put("SxUnOver", 88, n3(cells.get(over, float("nan"))))
    mc.put("SxUnOverRep", 87, n3(k5_dom(a87, over[0], over[1])))


def unlab_values(un: pd.DataFrame) -> list[float]:
    """88 号表 d 中无标签暴露的全部均值（三个数据集 × 两种目标 × 两种读法），缺的格不计。"""
    allv: list[float] = []
    for ds in UNLAB_SETS:
        for full in UNLAB_MODES:
            v = val(one(un, 数据集=ds, 训练或预训练方式=full), "Δ_unsup_均值")
            if np.isfinite(v):
                allv.append(v)
    return allv


def unlab_cells(dom: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    """88 号表 c 中 CNN 的逐留出域无标签增量：(数据集, 留出域, 预训练方式与读法) → 种子均值。"""
    out: dict[tuple[str, str, str], float] = {}
    if dom.empty:
        return out
    sub = dom[dom["数据集"].isin(list(UNLAB_SETS)) & (dom["编码器族"] == "cnn")]
    for ds, held, full, v in zip(sub["数据集"], sub["留出域"], sub["训练或预训练方式"], sub["Δ_unsup"], strict=True):
        if str(full) in UNLAB_MODES and np.isfinite(float(v)):
            out[str(ds), str(held), str(full)] = float(v)
    return out


# ── 表 S27：推断稳健性（84 号）───────────────────────────────────────────────
ALL_SETS = "无（全部数据集）"
ALL_SEEDS = "无（全部种子）"
MAIN_SCOPE_84 = "主口径（苹果两年设计）"


def _p_text(p: float) -> str:
    return f"{p:.3f}" if np.isfinite(p) else "--"


def tex_s27(b: Book, w: Wrapper) -> str:
    a, sign, dom = b.get("a", EMPTY), b.get("b", EMPTY), b.get("c", EMPTY)
    lines: list[str] = []
    for i, contrast in enumerate(CON_ORDER):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{{CON_TEX[contrast]}}} \\")
        for fam in FAMILIES:
            r = one(a, 编码器=fam, 对照=contrast, 去掉的数据集=ALL_SETS)
            if r is None:
                continue
            d = dom[(dom["编码器"] == fam) & (dom["对照"] == contrast)] if not dom.empty else dom
            above = str(d["其中区间在0以上"].iloc[0]) if len(d) else "--"
            cells = [
                w(n3(val(r, "估计值"))),
                w(n3(val(r, "留一估计最小"))),
                w(n3(val(r, "留一估计最大"))),
                w(txt(r, "留一后区间不含0") or "--"),
                w(_p_text(val(one(sign, 编码器=fam, 对照=contrast), "单侧p"))),
                w(above),
            ]
            lines.append(rf"\quad {FAM_TEX[fam]} & {' & '.join(cells)} \\")
    return "\n".join(lines)


def macros_s27(b: Book, mc: Macros) -> None:
    a, sign, dom, flip, seed = (b.get(k, EMPTY) for k in ("a", "b", "c", "d", "e1"))
    for contrast, ck in CON_KEY.items():
        for fam in FAMILIES:
            fk = FAM_KEY[fam]
            r = one(a, 编码器=fam, 对照=contrast, 去掉的数据集=ALL_SETS)
            mc.put(f"SxLo{ck}{fk}Min", 84, n3(val(r, "留一估计最小")))
            mc.put(f"SxLo{ck}{fk}Max", 84, n3(val(r, "留一估计最大")))
            mc.put(f"SxLo{ck}{fk}Keep", 84, of(txt(r, "留一后区间不含0")))
            mc.put(f"SxFp{ck}{fk}", 84, _p_text(val(one(sign, 编码器=fam, 对照=contrast), "单侧p")).replace("--", ""))
            d = dom[(dom["编码器"] == fam) & (dom["对照"] == contrast)] if not dom.empty else dom
            mc.put(f"SxDom{ck}{fk}", 84, of(str(d["其中区间在0以上"].iloc[0])) if len(d) else "")
            rs = one(seed, 编码器=fam, 对照=contrast, 去掉的种子=ALL_SEEDS)
            mc.put(f"SxSeed{ck}{fk}Min", 84, n3(val(rs, "留一估计最小")))
            mc.put(f"SxSeed{ck}{fk}Max", 84, n3(val(rs, "留一估计最大")))
            mc.put(f"SxSeed{ck}{fk}Keep", 84, of(txt(rs, "留一后区间不含0")))
    r = one(sign, 编码器="cnn", 对照="Δ_rep")
    floor = val(r, "p下限")
    mc.put("SxFpFloor", 84, _p_text(floor).replace("--", ""))
    rf_main = one(flip, 口径=MAIN_SCOPE_84)
    mc.put("SxFlipShare", 84, pct(val(rf_main, "翻转比例"), 0) + r"\%" if rf_main is not None else "")
    mc.put("SxFlipLo", 84, pct(val(rf_main, "比例95%下限"), 0) + r"\%" if rf_main is not None else "")
    mc.put("SxFlipHi", 84, pct(val(rf_main, "比例95%上限"), 0) + r"\%" if rf_main is not None else "")


# ── 表 S28：重复半分割（89 号）───────────────────────────────────────────────
POOLED = "各折合并"


def _sd(var: float) -> float:
    return float(np.sqrt(max(var, 0.0))) if np.isfinite(var) else float("nan")


def tex_s28(b: Book, w: Wrapper) -> str:
    c, d = b.get("c", EMPTY), b.get("d", EMPTY)
    lines: list[str] = []
    for fam in DEEP:
        r = one(c, 设计="two_year", 留出年=POOLED, 编码器族=fam)
        v = one(d, 设计="two_year", 留出年=POOLED, 编码器族=fam, 量="Δ_half")
        cells = [
            w(n3(val(r, "Δ_half_r0"))),
            w(n3(val(r, "Δ_half_均值"))),
            w(n3(val(r, "Δ_half_划分间最小"))),
            w(n3(val(r, "Δ_half_划分间最大"))),
            w(pct(val(r, "份额_均值之比_r0"), 0)),
            w(pct(val(r, "份额_均值之比_划分间最小"), 0)),
            w(pct(val(r, "份额_均值之比_划分间最大"), 0)),
            w(n3(_sd(val(v, "嵌套_种子间方差")))),
            w(n3(_sd(val(v, "嵌套_种子内划分间方差")))),
        ]
        lines.append(rf"{FAM_TEX[fam]} & {' & '.join(cells)} \\")
    return "\n".join(lines)


def macros_s28(b: Book, mc: Macros) -> None:
    c, d = b.get("c", EMPTY), b.get("d", EMPTY)
    for fam in DEEP:
        fk = FAM_KEY[fam]
        r = one(c, 设计="two_year", 留出年=POOLED, 编码器族=fam)
        v = one(d, 设计="two_year", 留出年=POOLED, 编码器族=fam, 量="Δ_half")
        mc.put(f"SxSp{fk}Half", 89, n3(val(r, "Δ_half_均值")))
        mc.put(f"SxSp{fk}HalfZ", 89, n3(val(r, "Δ_half_r0")))
        mc.put(f"SxSp{fk}HalfMin", 89, n3(val(r, "Δ_half_划分间最小")))
        mc.put(f"SxSp{fk}HalfMax", 89, n3(val(r, "Δ_half_划分间最大")))
        for name, col in (("Share", "份额_均值之比"), ("ShareZ", "份额_均值之比_r0")):
            x = val(r, col)
            mc.put(f"SxSp{fk}{name}", 89, pct(x, 0) + r"\%" if np.isfinite(x) else "")
        for name, col in (("ShareMin", "份额_均值之比_划分间最小"), ("ShareMax", "份额_均值之比_划分间最大")):
            x = val(r, col)
            mc.put(f"SxSp{fk}{name}", 89, pct(x, 0) + (r"\%" if name == "ShareMax" else "") if np.isfinite(x) else "")
        mc.put(f"SxSp{fk}SdSeed", 89, n3(_sd(val(v, "嵌套_种子间方差"))))
        mc.put(f"SxSp{fk}SdSplit", 89, n3(_sd(val(v, "嵌套_种子内划分间方差"))))
        mc.put(
            f"SxSp{fk}Cells", 89, f"{int(val(r, 'Δ_half_为正格数'))} of {int(val(r, '格数'))}" if r is not None else ""
        )
    r = one(c, 设计="two_year", 留出年=POOLED, 编码器族="cnn")
    mc.put("SxSpN", 89, str(int(val(r, "划分数"))) if r is not None else "")


# ── 表 S29：芒果半分的划分单位（91 号）───────────────────────────────────────
UNITS = {"row": "Row", "pair": "Pair", "pop": "Pop"}
ALL_FAMS = "各族合并"
S29_TEX = {"en": {"all": "All families"}, "zh": {"all": "各族合并"}}


def tex_s29(b: Book, w: Wrapper, lang: str) -> str:
    t = b.get("b", EMPTY)
    lines: list[str] = []
    for fam in [*FAMILIES, ALL_FAMS]:
        if fam == ALL_FAMS:
            lines.append(r"\midrule")
        cells: list[str] = []
        for unit in UNITS:
            r = one(t, 半分单位=unit, 编码器=fam)
            v = val(r, "Δ_half")
            cells += [
                w(n3(v)) if np.isfinite(v) else "--",
                w(txt(r, "Δ_half_为正留出季")) if txt(r, "Δ_half_为正留出季") else "--",
            ]
        label = S29_TEX[lang]["all"] if fam == ALL_FAMS else FAM_TEX[fam]
        lines.append(rf"{label} & {' & '.join(cells)} \\")
    return "\n".join(lines)


def macros_s29(b: Book, mc: Macros) -> None:
    t, a, diff = b.get("b", EMPTY), b.get("a", EMPTY), b.get("c", EMPTY)
    for unit, uk in UNITS.items():
        for fam in [*FAMILIES, ALL_FAMS]:
            fk = "All" if fam == ALL_FAMS else FAM_KEY[fam]
            r = one(t, 半分单位=unit, 编码器=fam)
            mc.put(f"SxMu{uk}{fk}", 91, n3(val(r, "Δ_half")))
            mc.put(f"SxMu{uk}{fk}Pos", 91, of(txt(r, "Δ_half_为正留出季")))
    row = a[a["半分单位"] == "row"] if not a.empty else a
    mate = float(row["B半同点谱在A半占比_种子均值"].mean()) if len(row) else float("nan")
    mc.put("SxMuMate", 91, pct(mate, 0) + r"\%" if np.isfinite(mate) else "")
    for name, cmp in (("Pair", "pair − row"), ("Pop", "pop − row")):
        r = one(diff, 对比=cmp, 留出季="各季合并", 编码器=ALL_FAMS)
        mc.put(f"SxMuDiff{name}", 91, n3(val(r, "Δ_half之差")))
        mc.put(f"SxMuDiff{name}Lo", 91, n3(val(r, "Δ_half之差_种子t区间下限")))
        mc.put(f"SxMuDiff{name}Hi", 91, n3(val(r, "Δ_half之差_种子t区间上限")))


# ── 表 S30：芒果产区任务（90 号）─────────────────────────────────────────────
VARIANTS = {"all": "", "shared": "S"}
S30_TEX = {
    "en": {"all": "All cultivars", "shared": "Cultivars grown in both regions"},
    "zh": {"all": "全部品种", "shared": "两个产区都种的品种"},
}
CON90 = {"Δ_rep": "Rep", "Δ_lab": "Lab", "Δ_half": "Half"}
MATE_COL = "B半同点谱在A半占比"  # 90 号长格式表的列


def tex_s30(b: Book, w: Wrapper, lang: str) -> str:
    t, lev = b.get("b", EMPTY), b.get("c", EMPTY)
    lines: list[str] = []
    for i, variant in enumerate(VARIANTS):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{{S30_TEX[lang][variant]}}} \\")
        for fam in FAMILIES:
            r = one(t, 版本=variant, 编码器=fam)
            base = val(one(lev, 版本=variant, 编码器=fam, 留出季="各季均值"), "R_宏F1")
            cells = [w(n3(base)) if np.isfinite(base) else "--"]
            for contrast in CON90:
                v = val(r, f"{contrast}_跨季均值")
                p = txt(r, f"{contrast}_为正留出季")
                cells += [w(n3(v)) if np.isfinite(v) else "--", w(p) if p else "--"]
            lines.append(rf"\quad {FAM_TEX[fam]} & {' & '.join(cells)} \\")
    return "\n".join(lines)


def macros_s30(b: Book, mc: Macros) -> None:
    t, lev = b.get("b", EMPTY), b.get("c", EMPTY)
    for variant, vk in VARIANTS.items():
        deep_rep: list[float] = []
        for fam in FAMILIES:
            fk = FAM_KEY[fam]
            r = one(t, 版本=variant, 编码器=fam)
            for contrast, ck in CON90.items():
                mc.put(f"SxRg{vk}{ck}{fk}", 90, n3(val(r, f"{contrast}_跨季均值")))
                mc.put(f"SxRg{vk}{ck}{fk}Pos", 90, of(txt(r, f"{contrast}_为正留出季")))
            base = val(one(lev, 版本=variant, 编码器=fam, 留出季="各季均值"), "R_宏F1")
            mc.put(f"SxRg{vk}R{fk}", 90, n3(base))
            if fam in DEEP and np.isfinite(val(r, "Δ_rep_跨季均值")):
                deep_rep.append(val(r, "Δ_rep_跨季均值"))
        mc.put(f"SxRg{vk}RepDeepMin", 90, n3(min(deep_rep)) if deep_rep else "")
        mc.put(f"SxRg{vk}RepDeepMax", 90, n3(max(deep_rep)) if deep_rep else "")
        # 按行对半时 B 半样本里同一取样点的另一条谱在 A 半的占比：每个（留出季, 种子）一个值，再取均值
        long90 = b.get("e", EMPTY)
        sub = long90[long90["版本"] == variant] if MATE_COL in long90 else EMPTY
        mate = sub.groupby(["留出季", "种子"])[MATE_COL].first().mean() if len(sub) else float("nan")
        mc.put(f"SxRg{vk}Mate", 90, pct(float(mate), 0) + r"\%" if np.isfinite(mate) else "")


# ── 定性断言 ─────────────────────────────────────────────────────────────────
def check_claims(books: dict[int, Book], data: dict[str, Any]) -> pd.DataFrame:
    """正文与补充材料 S9 节里与补充实验有关的定性表述，逐条对结果核对；不成立即须改写文字。

    返回逐条结果表：成立 = True / False；缺数据无法核对时为 None。
    """
    rows: list[dict[str, Any]] = []

    def claim(cid: str, text: str, vals: list[float], ok: Any) -> None:
        known = len(vals) > 0 and all(np.isfinite(v) for v in vals)
        rows.append({"编号": cid, "表述": text, "成立": bool(ok) if known else None})

    s20 = data["s20"]
    within = [s20["within"][learner, key][0] for learner in ("SVM", "PLS-DA") for key in ("A", "B")]
    cross = [s20["cross"][learner, key][0] for learner in ("SVM", "PLS-DA") for key in ("A", "B")]
    claim(
        "X1",
        "3.6 节、S9.1：同一学习器年内随机划分的准确率高于跨年（SVM、PLS-DA，两个留出年）",
        within + cross,
        all(a > b for a, b in zip(within, cross, strict=True)),
    )
    rec = [(s20[f"rec_within{seg}"]["B"][0], s20[f"rec_cross{seg}"]["B"][0]) for seg in REC_KEYS]
    pub = s20["pub"][0]
    claim(
        "X2",
        "3.6 节、S9.1：重建流程（2048 个点与 512 个波段两种输入）在 2019 年内随机划分上都与已发表值相差不到"
        " 2 个百分点，跨年（2018→2019）都低于年内",
        [v for pair in rec for v in pair] + [pub],
        all(abs(rw - pub) < 0.02 and rc < rw for rw, rc in rec),
    )

    def frac(text: str) -> tuple[float, float]:
        """「11/18」→ (11.0, 18.0)；不是这种写法时返回两个 nan。"""
        parts = text.split("/")
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            return float("nan"), float("nan")
        return float(parts[0]), float(parts[1])

    # within、cross 的次序：SVM 的 2018、2019，再 PLS-DA 的 2018、2019；最近质心按 2018、2019
    nc_w = [s20["within"]["最近质心", key][0] for key in ("A", "B")]
    nc_c = [s20["cross"]["最近质心", key][0] for key in ("A", "B")]
    weakest = all(nc_w[i % 2] < within[i] for i in range(4))
    least_drop = all(nc_w[i % 2] - nc_c[i % 2] < within[i] - cross[i] for i in range(4))
    first_2019 = nc_c[1] > max(cross[1], cross[3])
    claim(
        "X3",
        "S9.1：最近质心在 2018、2019 年内都是三种学习器里最低的，跨年时下降最少，留出 2019 年时排第一",
        nc_w + nc_c + within + cross,
        weakest and least_drop and first_2019,
    )

    r_sign = one(books[84].get("b", EMPTY), 编码器="cnn", 对照="Δ_rep")
    p_cnn, p_floor = val(r_sign, "单侧p"), val(r_sign, "p下限")
    claim(
        "X4",
        "3.2 节：CNN 的 S_src−R 在数据集簇符号翻转检验里取到 p 的下限",
        [p_cnn, p_floor],
        abs(p_cnn - p_floor) < 1e-9,
    )

    sup = books[88].get("b", EMPTY)
    sup_vals: list[float] = []
    sup_ok = True
    for fam in SUP_FAMS:
        r_sup = one(sup, 口径=MAIN_SCOPE, 编码器族=fam, 对照="Δ_rep")
        mean = val(r_sup, "监督式_均值")
        pos, total = frac(txt(r_sup, "监督式_为正留出域"))
        sup_vals += [mean, pos, total]
        sup_ok = sup_ok and mean > 0 and 2 * pos > total
    claim(
        "X5",
        "3.2 节：监督式训练下 CNN 与 ResNet 的 S_src−R 均值为正，且过半留出域为正",
        sup_vals,
        sup_ok,
    )

    dom22: pd.DataFrame = data["s22"]["dom"]
    gains = [
        float(dom22.loc[dom22["路线"] == route, "增量"].mean()) if not dom22.empty else float("nan")
        for route in (*LEARN83, "原型网络")
    ]
    claim(
        "X6",
        "3.4 节、S9.3：参数级暴露使池化 SVM、池化 PLS-DA 与原型网络在 B 半上的留出域均值都提高",
        gains,
        all(g > 0 for g in gains),
    )

    r_ck = one(data["s23"]["ckpt"], 编码器="cnn")
    ck_mean = val(r_ck, "两年均值")
    ck_pos, ck_total = frac(txt(r_ck, "两年为正格"))
    claim(
        "X7",
        "摘要、3.2、3.3 节与结论：在留出年份上选检查点使仅源域的 CNN 提高"
        "（同一条轨迹上配对，两年设计的均值为正且过半折×种子格为正）",
        [ck_mean, ck_pos, ck_total],
        ck_mean > 0 and 2 * ck_pos > ck_total,
    )

    rank22: pd.DataFrame = data["s22"]["rank"]
    cnn22 = rank22[(rank22["编码器"] == "cnn") & rank22["dataset"].isin(MAIN_SETS)] if not rank22.empty else rank22
    n_rank = float(len(cnn22))
    n_pn_first = float((cnn22["对等暴露第一"] == PN_EXP).sum()) if len(cnn22) else float("nan")
    claim(
        "X8",
        "3.4 节与结论：池化基线也用有标签的留出样本训练时，两种池化学习器的提高都多于原型网络，"
        "且对等暴露的排名里原型网络一个留出域都没排第一",
        [*gains, n_rank, n_pn_first],
        gains[0] > gains[2] and gains[1] > gains[2] and n_rank > 0 and n_pn_first == 0,
    )

    hp_changed = val(one(books[93].get("b", EMPTY), 口径=MAIN_SCOPE), "所选超参数改变的留出域")
    claim(
        "X10",
        "S9.3 节：把池化 SVM 的迭代上限提高后重跑一个种子，主口径各留出域所选的超参数都不变",
        [hp_changed],
        hp_changed == 0,
    )

    un88, a87 = books[88].get("d", EMPTY), books[87].get("a", EMPTY)
    vals9: list[float] = []
    ok9 = True
    for ds in UNLAB_SETS:
        unl = [val(one(un88, 数据集=ds, 训练或预训练方式=full), "Δ_unsup_均值") for full in UNLAB_MODES]
        lab = k5_ds_mean(a87, ds)
        vals9 += [*unl, lab]
        ok9 = ok9 and max(abs(v) for v in unl) < lab
    claim(
        "X9",
        "3.3 节、S9.7：苹果两种设计与猕猴桃上，无标签暴露的留出域均值（两种目标 × 两种读法）在每个数据集上的"
        "幅度都低于该数据集 CNN 的 S_src−R（K=5，留出域均值）",
        vals9,
        ok9,
    )

    cells = unlab_cells(books[88].get("c", EMPTY))
    reps = {key: k5_dom(a87, key[0], key[1]) for key in cells}
    top = max(cells, key=lambda key: cells[key]) if cells else None
    over = sorted(key for key in cells if cells[key] > reps[key])
    claim(
        "X11",
        "S9.7：逐留出域的无标签增量最大的一格是苹果三年设计留出 2025 年、对比预训练后冻结读出；超过同一留出域"
        " CNN 的 S_src−R（K=5）的只有猕猴桃 2018 年、对比预训练再在源域上情节训练这一格",
        [*cells.values(), *reps.values()],
        top == UNLAB_TOP and over == sorted(UNLAB_OVER),
    )
    return pd.DataFrame(rows).sort_values("编号", key=lambda s: s.str[1:].astype(int), ignore_index=True)


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="92_sx_tables：补充实验轮的补充材料表与数字宏")
    parser.add_argument("--stage", default="formal", choices=["formal", "pilot"])
    parser.add_argument("--check", action="store_true", help="只核对，不写回")
    args = parser.parse_args()
    logger = eu.get_logger("92_sx_tables")
    books = {n: load_book(n, args.stage) for n in STEMS}
    formal = {n: is_formal(n, books[n], args.stage) for n in STEMS}
    for n in STEMS:
        state = "正式" if formal[n] else ("预览" if books[n] else "缺表")
        logger.log(f"  {STEMS[n]}：{state}")
    w = {n: wrap(formal[n]) for n in STEMS}
    mc = Macros(formal)

    s20 = data_s20(books[81])
    s21 = data_s21(books[82])
    s22 = data_s22(books[83])
    s22["info"] = books[83].get("e", EMPTY)
    s22["long"] = books[83].get("d", EMPTY)
    s23 = data_s23(books[85], books[86])
    macros_s20(s20, mc)
    macros_s21(s21, mc)
    macros_s22(s22, mc)
    macros_s23(s23, mc)
    macros_s24(books[87], mc)
    macros_s25(books[87], mc)
    macros_s26(books[88], books[87].get("a", EMPTY), mc)
    macros_s27(books[84], mc)
    macros_s28(books[89], mc)
    macros_s29(books[91], mc)
    macros_s30(books[90], mc)
    macros_cap(books[93], s22["long"], mc)
    sx = mc.block()

    claims = check_claims(books, {"s20": s20, "s21": s21, "s22": s22, "s23": s23})
    for _, r in claims.iterrows():
        state = {True: "成立", False: "不成立", None: "缺数据"}[r["成立"]]
        logger.log(f"  断言 {r['编号']}：{state}　{r['表述']}")
    if all(formal.values()) and not claims["成立"].eq(True).all():
        bad = claims[~claims["成立"].eq(True)]
        raise ValueError(f"正式结果下有定性表述不成立或无法核对，须改写文字：{list(bad['编号'])}")

    ok = True
    for _lang, path in m68.MAIN.items():
        ok &= m68.replace_blocks(path, {"SX": sx}, args.check, logger)
    for lang, path in m68.TEX.items():
        s25a, s25b = tex_s25(books[87], w[87], lang)
        s26a, s26b = tex_s26(books[88], w[88], lang)
        blocks = {
            "S20": tex_s20(s20, w[81], lang),
            "S21": tex_s21(s21, w[82], lang),
            "S22": tex_s22(s22, w[83], lang),
            "S23": tex_s23(s23, w[85], w[86], lang),
            "S24": tex_s24(books[87], w[87]),
            "S25A": s25a,
            "S25B": s25b,
            "S26A": s26a,
            "S26B": s26b,
            "S27": tex_s27(books[84], w[84]),
            "S28": tex_s28(books[89], w[89]),
            "S29": tex_s29(books[91], w[91], lang),
            "S30": tex_s30(books[90], w[90], lang),
            "SX": sx,
        }
        ok &= m68.replace_blocks(path, blocks, args.check, logger)
    if not args.check:
        status = pd.DataFrame({"脚本": [STEMS[n] for n in STEMS], "正式": [formal[n] for n in STEMS]})
        out = eu.write_script_workbook(
            __file__,
            {
                eu.table_sheet_name(0, "数字宏"): mc.frame(),
                eu.table_sheet_name(1, "定性断言"): claims,
                eu.table_sheet_name(2, "来源是否正式"): status,
                eu.table_sheet_name(3, "S21逐K"): s21,
                eu.table_sheet_name(4, "S22逐留出域"): s22["dom"],
                eu.table_sheet_name(5, "S23检查点"): s23["ckpt"],
                eu.table_sheet_name(6, "S23超参数"): s23["hp"],
            },
        )
        logger.log(f"已写 {out}")
    logger.close()
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
