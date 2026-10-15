#!/usr/bin/env python3
"""
120 — C5 稿件数据卡与 SHA-256 登记（补齐项目约定 §4.6 / §202 的强制登记）

背景
----
2026-07-25 核查发现：本项目**从未**对 C5 稿件所用数据做过 SHA-256 登记——
`.datalock-manifest.tsv` 不存在，`08github/data/DATA.md` 仍是含 {{占位符}} 的模板，
`08github/docs/` 无任何数据卡，而 `03data/processed/*-meta.json` 不含 SHA 字段。
但稿件表 tab:datacard 的表注声称各数据集"经确定性 L1 预处理并附 SHA-256 校验"。
本脚本把该声明做实：对稿件实际依赖的每个文件计算 SHA-256 并落盘登记。

诚实注记（2026-07-26 修正）
--------------------------------
本清单最初写在 `03data/.datalock-manifest.tsv`。那是错的：该路径由 `datalock.sh`
独占，其 schema 为 `sha256⇥size⇥ts⇥相对 03data 的路径`，而本脚本写的是
`kind⇥id⇥sha256⇥bytes⇥绝对路径`。撞名的后果是 `datalock.sh verify` 把本清单
15 条连同表头全判为 MISSING，`checkpoint.sh` 因「数据完整性校验失败」而永久拒绝提交。
本清单的覆盖范围本来就与 datalock 不同（跨 03data / 04outputs / 只读的 05data），
无法用相对 DATA_DIR 的路径表达，故改放到数据卡旁边，两套登记各管各的。

覆盖范围（只登记 C5 稿件真正用到的文件，不含 C6 遗留产物）
----------------------------------------------------------
A. 源数据（他组已发表表格的提取件 + 第三体系原始数据）
B. 派生产物（正文与补充材料每个数字的来源 JSON/CSV）

输出
----
- `04outputs/0NDATACARD_C5_manifest.tsv`  逐文件 SHA-256 清单（TSV）
- `refine-logs/0NDATACARD_C5.md`         中文数据卡（§1 项目内部工作区中文强制）
- `04outputs/120datacard_sha256.json`

诚实边界
--------
SHA-256 仅证明**自登记之时起**文件字节未变；它不追认登记之前的历史，也不构成
对提取步骤本身正确性的证明（提取正确性另由脚本 55 的 read_html + 人工逐条核验承担）。
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# 【2026-08-10 订正，RYA-105】仓库外只读原始数据根目录。取证全扫开出义务
# 「Real-kinetics source tables are registered only at unavailable absolute paths」：
# 此前源表按【作者机绝对路径】登记，别的机器上既打不开也无从判断它该在哪。
# 现改为：仓库内文件一律登记【仓库相对路径】；仓库外的只读原始数据登记为
# `${ARIS_RAW_ROOT}/…` 占位形式，并允许用同名环境变量指向本机副本。
RAW_ROOT = Path(os.environ.get("ARIS_RAW_ROOT", str(ROOT.parents[1] / "05data")))
RAW_ROOT_TOKEN = "${ARIS_RAW_ROOT}"


def display_path(p: Path) -> str:
    """登记用路径：仓库内 → 仓库相对；仓库外只读原始数据 → ${ARIS_RAW_ROOT}/… 占位。
    绝不写出作者机绝对路径（那既不可移植，也会把私有目录结构带进发布件）。"""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        pass
    try:
        return f"{RAW_ROOT_TOKEN}/{p.resolve().relative_to(RAW_ROOT.resolve()).as_posix()}"
    except ValueError:
        return p.name          # 兜底：只留文件名，仍不泄露绝对路径

# --- A. 源数据 -------------------------------------------------------------
# 显式标注：值混了 str 与 Path，不标则被推成 dict[str, object]，后续 Path(s["path"]) 报错
SOURCES: list[dict[str, Any]] = [
    {
        "id": "apple_firmness",
        "name": "苹果硬度温度梯度（4 品种 × 5 贮温 × 5 时间点）",
        "role": "C5 真实动力学 / 体系 1",
        "path": ROOT / "03data/processed/pmc_real/PMC10253207_T16.csv",
        "origin": "Foods 2023, doi:10.3390/foods12112113 (PMC10253207) 表 A1（开放获取）",
        "extraction": "pandas.read_html 提取（脚本 55）+ 人工逐条核验；非本组采集",
        "license": "CC BY 4.0（Foods 为全开放获取）",
    },
    {
        "id": "food_ammonia_tbars",
        "name": "长货架期食品 氨/TBARS 累积动力学（5 食品 × 4 贮温，10 序列）",
        "role": "C5 真实动力学 / 体系 2",
        "path": ROOT / "03data/processed/pmc_real/PMC9319022_T3.csv",
        "origin": "Foods 2022 11(14):2004, doi:10.3390/foods11142004 (PMC9319022) 表 3",
        "extraction": "pandas.read_html 提取（脚本 55）+ Europe PMC fullTextXML 独立重抓逐值核对（132 行 0 处不一致）",
        "license": "CC BY 4.0",
    },
    {
        "id": "hass_avocado",
        "name": "Hass 牛油果采后成熟（478 果 × 3 组贮藏，14,722 条记录）",
        "role": "C5 真实动力学 / 体系 3（真统计估计误差）",
        "path": RAW_ROOT / "004_hass_avocado_rgb_ripening"
                / "Hass Avocado Ripening Photographic Dataset" / "Avocado Ripening Dataset.xlsx",
        "origin": "Mendeley Data, doi:10.17632/3xd9n945v8.1",
        "extraction": "原始 xlsx 直接读取（未做 L1 变换）；按 4.5 只读引用，未复制进项目",
        "license": "CC BY 4.0（Mendeley Data 默认）",
    },
]

# --- B. 派生产物（稿件每个数字的来源） -------------------------------------
DERIVED: list[tuple[str, str]] = [
    ("04outputs/43trichotomy_formal.json", "半合成三分律：Ψ 终值、塌缩斜率、Krug 匹配"),
    ("04outputs/54c5_realkinetics.json", "苹果体系：品种 Ea/R²、设计扫描、超定子集"),
    ("04outputs/54c5_realkinetics-designsweep.csv", "苹果 26 温度子集逐条（补充表 S2 数据源）"),
    ("04outputs/91c5_realkinetics_killer.json", "Ψ vs 廉价启发式判别力、Ψ_het 跨体系"),
    ("04outputs/99_realkinetics2.json", "食品体系：Ea 范围、同质/异质 Ψ 复现"),
    ("04outputs/102_avocado_c5_pilot.json", "牛油果体系：真统计估计误差、n 轴扫描"),
    ("04outputs/119pazman_pronzato_bridge.json", "与 Pázman–Pronzato 2006 算例的坐标对照"),
    ("04outputs/122criterion_bakeoff.json", "准则对决（含对自身假设的证伪：塌缩=代数、D 对比未建立）"),
    ("04outputs/123montecarlo_criterion_study.json", "1000 独立设计 Monte-Carlo：两轴速率、饱和、决策 regret"),
    ("04outputs/123montecarlo_criterion_study-designs.csv", "逐设计表：准则值 + 真值 RMSE（正文 §sec:mc 数据源）"),
    ("04outputs/124coverage_study.json", "置信区间覆盖率研究：Ψ 作为标定诊断被证伪 + 区间口径警告"),
    ("04outputs/124coverage_study-designs.csv", "逐设计覆盖率表"),
    ("04outputs/126trichotomy_plugin_psi.json", "去 oracle 的 plug-in Ψ(θ̂) 三分律复现 + 预拟合收敛守卫"),
    ("04outputs/126trichotomy_plugin_psi-percell.csv", "plug-in Ψ 逐 (regime,n,seed) 明细"),
    ("04outputs/127budget_matched_sensitivity.json", "条件数劣势对预算配平容差的敏感性（收紧到严格同预算）"),
    ("04outputs/128montecarlo_runtime_receipt.json", "§sec:mc 墙钟时间凭据 + 123 的字节级可复现性核验"),
    ("04outputs/129kgrid_oracle_ablation.json", "k 搜索网格去 oracle 消融：162 键无一变动 >2%"),
    # --- 2026-08-10 RYA-104 补登（EF-001 清偿）：以下各件此前漏登，但确撑着稿面数字 ---
    ("04outputs/43trichotomy_formal-agg.csv", "图 fig:tri 面板 a/b 的逐 regime 聚合数据源（42paper_figs_v8.fig2 直读）"),
    ("04outputs/43trichotomy_formal-percell.csv", "图 fig:krug 的逐格经验/Krug 斜率数据源（42paper_figs_v8.fig3 直读）"),
    ("04outputs/105multiseed_realdata.json", "真实数据四脚本 × 5 正典种子的跨种子聚合：正文各处「5 种子均值」与品种表 ±std 的来源"),
    # --- 2026-08-10 RYA-105 补登 ---
    ("04outputs/130aggregate_pilot_robustness.json",
     "正文 ±30% 档「96% 保持负号 / 87% 保持 ≤−0.5」两个比例的【现行生成步骤与结果】（脚本 130）"),
    ("04outputs/104ablation_psi_aggregation.json", "Ψ_n 聚合判据与 N 因子消融（订正后 Ψ_n=λmin(I_total) 口径）"),
    ("04outputs/107design_multivariety.json", "三单调品种 48 子集的（品种×ΔT）分层置换检验"),
    # --- 2026-08-10 RYA-108 补登 ---
    ("04outputs/131aggregate_c5_partial_tally.json",
     "正文「逐次抽样 5 次 Ψ_n 严格大于 cond、1 次数值持平」这一【跨 6 次抽样】汇总的现行生成步骤与结果"
     "（脚本 131）；此前该汇总只由人手从 6 个 91c5 产物数出，仓库内无聚合器"),
    # --- 2026-08-19 RYA-526 补登（取证义务 728f003f）：LAQ 消融派生件此前漏登 ---
    ("04outputs/133ablation_innovation_map.json",
     "创新点-消融映射与 LAQ 消融：tab:claims「仿射时斜率 −0.508 至 −1.288 / ρ=1/2 处 +0.077」"
     "与增项闭式相对误差 <5e−7 的直接来源（脚本 133）"),
    ("04outputs/133ablation_innovation_map-laq.csv",
     "LAQ 消融逐 (种子, n, ρ) 明细表，即上一行各斜率的被聚合件"),
    # --- 由本轮新增的 C2 前缀级反向闸门当场抓出的 4 件（均直接支撑稿面数字）---
    ("04outputs/124coverage_study-bins.csv",
     "覆盖率按 Ψ 十分位分箱表：正文「最低/最高十分位覆盖」「$z$ 分位 0.78--0.81」"
     "「无自由度修正 0.66--0.71」「报出 SE 为真实 SD 的 1.32--1.70 倍」四组数的直接来源"),
    ("04outputs/122criterion_bakeoff-percriterion.csv",
     "准则对决逐判据明细：tab:bakeoff 各行边际/偏 Spearman 与成对 Spearman 的被聚合件"),
    ("04outputs/99_realkinetics2-overdet.csv",
     "食品体系超定子集逐条明细：正文该体系同质/异质 Ψ 复现数字的被聚合件"),
    ("04outputs/107design_multivariety-sweep.csv",
     "三单调品种 48 子集的逐子集扫描表：（品种×ΔT）分层置换检验的输入"),
    ("04outputs/104ablation_psi_aggregation_regtest.json",
     "Ψ_n 聚合消融的回归测试基准件（133 的 REGTEST_EXCLUSION 按文件名点名引用它做取证依据，"
     "故它本身必须可核；见 §9.0 裁决『保留、不删、不改名』）"),
    # --- RYA-537 新增：清偿取证义务 936462b7 时新写的核验件 ---
    ("04outputs/137design_variance_scale_invariance.json",
     "经验/总体设计方差相对误差的尺度不变性核验：§sec:converse 证明第 (2) 步「总体矩读法不额外要价」"
     "所依据者（Weyl 上界对本设计族是松的）；正文不引用其数字，但审计 §L.14.3 引用，故须可核"),
    # --- RYA-569 新增：清偿取证义务 1f9bcf45 / 24e3754f ---
    ("04outputs/138eigendirection_angle_check.json",
     "同方差代理与异方差定义量的最小特征方向夹角预检（义务 1f9bcf45 的预注册预测依据）："
     "正文不引用其数字，但审计 §L.16 引用，故须可核"),
    # --- RYA-580 新增：141/142 为稿面 \texttt{} 点名引用的证据件，此前未登记 ---
    ("04outputs/141rya579_seedscope_receipt.json",
     "「5 种子聚合 vs 原生抽样」的产物指向分界：稿面 §统计检验与误差报告按文件名点名引用它"
     "（设计侧逐字节相同的 8 列/122 键 vs 随种子变的 5 列/24 键），故须可核"),
    ("04outputs/142rya579_method_identity_receipt.json",
     "两阶段管线与正过程对应的三条恒等式核验（合同残差 0、delta 恒等式残差 1.97e-16）："
     "稿面方法节按文件名点名引用其数字，故须可核"),
    ("04outputs/143rya579_ledger_receipts.json",
     "取证台账收据：a9930fd7 的「刻意声明」定位与重复组两步判据的实测结果；"
     "正文不引用其数字，但审计 §L.20.6/§L.20.7 引用，故须可核"),
    ("04outputs/144rya579_critical_kl_check.json",
     "LAN 临界档精确 KL 恒等式（eq:klexact）的数值核验：二次近似最大相对误差 6.04e-5；"
     "正文不引用其数字，但审计 §L.20.1 引用，故须可核"),
    ("04outputs/145citation_field_diff.json",
     "参考文献逐字段机械比对（稿面 \\bibitem vs Crossref，DOI 直查兜底）：57 条 → AGREE 53 / "
     "CHECK 1（donoho2009，见审计 §L.21.8）/ 无注册记录 3（预印本与专著）；"
     "正文不引用其数字，但审计 §L.21.8/§L.22.3 引用，故须可核"),
    ("04outputs/147oracle_plugin_error_pairing.json",
     "oracle 层(43) 对 plug-in 层(126) 两列退化方向误差的【配对】差（RYA-583 新增）："
     "140/140 全不同、最大绝对差 3.6418e-4、最大相对差 8.3567e-4、两列 Spearman 1.000；"
     "即稿面 §sec:relate-id 那三个数此前无产物字段承载的那一处（外审核对判 real_gap）。"
     "由 126 脚本产出但【独立成件】，理由见该文件 why_separate：126 的 json 是取证台账义务 "
     "c13bd67a3388 的 resolve 收据所指证据，收据每次门禁重验哈希，改动它会让收据当场自失效"),
    ("04outputs/155relabel_release_fields.json",
     "已登记产物里几处评审写法的字段改写台账（2026-10-15）：103 的 10 份逐种子产物（105_seeds 与 "
     "RYA-103 复跑档案各 5 份）、143 收据与 145 比对产物，逐文件记改写前登记的 SHA-256、改写前后的 "
     "SHA-256 与各规则命中数；只改目标字段，其余字节不动，不重跑。正文不引用其数字，但审计 §L.34 引用，故须可核"),
    ("08github/requirements.lock.txt",
     "逐包精确到版本的依赖锁（含解释器 3.13.5 与平台），即 §sec:repro 明写的 requirements.lock.txt。"
     "取证义务 24e3754f 指出「稿面声明了依赖锁但清单里没有它」——锁文件本身在盘（发布树根目录），"
     "缺的是它未进 SHA-256 登记，故自本轮起登记，使读者可核到与本文数字同一份锁"),
]

# --- B2. 逐种子明细：**按 glob 登记**（EF-001 的成因即硬编码清单漏件） -------
# 教训：固定清单只能核对「清单内的文件是否在盘」，清单本身漏了谁它不会知道。
# 凡属「同一族、会随种子/脚本增删」的产物一律走 glob，使漏登在结构上不可能发生。
DERIVED_GLOBS: list[tuple[str, str]] = [
    ("04outputs/91c5_realkinetics_killer_seed*.json",
     "苹果/食品体系逐种子原始产物：正文「5 种子均值 −0.70 与 −0.37」的直接来源"),
    ("04outputs/105_seeds/*.json",
     "105 复跑的逐种子原始产物（54c5 / 99 / 102 / 103），跨种子均值与 ±std 的被聚合件"),
    ("04outputs/105_seeds/*.csv",
     "105 复跑的逐种子明细表（苹果设计扫描 / 初值敏感性网格）"),
    ("04outputs/104ablation_psi_aggregation_seed*.json",
     "104 聚合消融的逐种子原始产物（5 个预注册种子）：正文「增量恒为负但跨度近三倍"
     "（−0.0765 至 −0.2353）」这一跨种子区间的被聚合件"),
    ("05logs/rya103_verify_20260810/*",
     "RYA-103 独立复跑档案（逐种子件 + 网格 + 聚合结果 + runner + 审计 trace）：**历史留痕**。"
     "⚠️ 2026-08-10 RYA-105 订正 Ψ 装配与自助内层估计量后，该档案对应的 96%/92% 已被"
     "脚本 130 重算的 96%/87% 取代；本目录保留以追溯，**不再是正文数字的现行依据**"),
]

# --- C. 覆盖闸门：对已声明覆盖的目录反向再扫一遍 ---------------------------
# `missing: []` 是**假安心**：它只回答「清单里的文件在不在盘」，回答不了
# 「在盘的该进清单的文件有没有进清单」。故对下列目录做反向核对，任何在盘
# 但未登记的文件都必须显式报出来（AppleDouble `._*` 与 .DS_Store 除外）。
GUARD_DIRS: list[str] = [
    "04outputs/105_seeds",
    "05logs/rya103_verify_20260810",
]

# --- C2. 反向闸门（按脚本族前缀）: 2026-08-19 RYA-526 新增，取证义务 728f003f ---
# 成因：C 的目录级反向扫描只覆盖两个子目录，而 C5 的绝大多数产物直接躺在 `04outputs/`
# 根下——133 的两件与 104 的六件即因此漏登多轮而未被任何闸门发现。
# 为什么不直接把 `04outputs` 整个纳入 GUARD_DIRS：该目录含 393 个文件，其中绝大多数
# 属已废弃研究线（BRIEF B 区），全量反扫会产出约 350 条与本稿无关的「未登记」噪声，
# 反使真信号淹没。故改按【本稿实际依赖的脚本族前缀】反扫：凡文件名以下列前缀开头者，
# 必须已被 DERIVED / DERIVED_GLOBS 登记，否则报出。覆盖面窄于「整个目录」但对本稿是完备的，
# 且新增同族产物（如再加一个种子）会自动被抓，不依赖任何人记得改清单。
GUARD_PREFIXES: list[str] = [
    "43trichotomy_formal", "54c5_realkinetics", "91c5_realkinetics_killer",
    "99_realkinetics2", "102_avocado_c5_pilot", "104ablation_psi_aggregation",
    "105multiseed_realdata", "107design_multivariety", "119pazman_pronzato_bridge",
    "122criterion_bakeoff", "123montecarlo_criterion_study", "124coverage_study",
    "126trichotomy_plugin_psi", "127budget_matched_sensitivity",
    "128montecarlo_runtime_receipt", "129kgrid_oracle_ablation",
    "130aggregate_pilot_robustness", "131aggregate_c5_partial_tally",
    "133ablation_innovation_map", "137design_variance_scale_invariance",
    "138eigendirection_angle_check", "141rya579_seedscope_receipt",
    "142rya579_method_identity_receipt", "143rya579_ledger_receipts",
    "144rya579_critical_kl_check", "145citation_field_diff", "155relabel_release_fields",
]
# 明示豁免：本稿不引用、但与上列前缀同名的旁支产物。豁免必须逐条写明理由，
# 空豁免表是本闸门的正常状态。
GUARD_PREFIX_EXEMPT: dict[str, str] = {
    "04outputs/104ablation_psi_aggregation.xlsx":
        "同名工作簿是 json 的 Excel 呈现（export_utils 工作簿通道副产物），"
        "稿面不引用其中任何数字，其内容逐值来自已登记的 104…json",
    "04outputs/105multiseed_realdata.xlsx":
        "同上：105…json 的 Excel 呈现，稿面引用的是 json 的键，内容逐值同源",
    "04outputs/107design_multivariety.xlsx":
        "同上：107…json 的 Excel 呈现，稿面引用的是 json 的键，内容逐值同源",
}


def _expand(pattern: str) -> list[Path]:
    """glob 展开并剔除 macOS AppleDouble 与 .DS_Store（非产物，登记它们只会制造噪声）。"""
    return sorted(p for p in ROOT.glob(pattern)
                  if p.is_file() and not p.name.startswith("._") and p.name != ".DS_Store")


def sha256(p: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def main() -> None:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for s in SOURCES:
        p = Path(s["path"])
        if not p.is_file():
            missing.append(display_path(p))
            continue
        d, n = sha256(p)
        meta = {k: s[k] for k in ("name", "role", "origin", "extraction", "license")}
        rows.append({"kind": "source", "id": s["id"], "path": display_path(p), "abs": str(p),
                     "sha256": d, "bytes": n, **meta})

    for rel, desc in DERIVED:
        p = ROOT / rel
        if not p.is_file():
            missing.append(display_path(p))
            continue
        d, n = sha256(p)
        rows.append({"kind": "derived", "id": Path(rel).stem, "path": display_path(p), "abs": str(p),
                     "sha256": d, "bytes": n, "name": desc})

    for pattern, desc in DERIVED_GLOBS:
        hits = _expand(pattern)
        if not hits:
            missing.append(f"{pattern}（glob 无命中）")
            continue
        for p in hits:
            d, n = sha256(p)
            rows.append({"kind": "derived", "id": p.stem, "path": display_path(p), "abs": str(p),
                         "sha256": d, "bytes": n, "name": desc})

    # 覆盖闸门：登记路径集合 vs 已声明覆盖目录的在盘实况，反向核对
    registered = {r["abs"] for r in rows}          # 反向闸门比对用【绝对】路径，登记输出用相对路径
    unregistered: list[str] = []
    for rel in GUARD_DIRS:
        base = ROOT / rel
        if not base.is_dir():
            unregistered.append(f"{rel}（目录不存在）")
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.name.startswith("._") or p.name == ".DS_Store":
                continue
            if str(p) not in registered:
                unregistered.append(display_path(p))

    # C2 前缀级反向闸门：04outputs 根下凡属本稿脚本族的产物，必须已登记
    for p in sorted((ROOT / "04outputs").glob("*")):
        if not p.is_file() or p.name.startswith("._") or p.name == ".DS_Store":
            continue
        if not any(p.name.startswith(pre) for pre in GUARD_PREFIXES):
            continue
        if str(p) in registered:
            continue
        rel_disp = display_path(p)
        if rel_disp in GUARD_PREFIX_EXEMPT:
            continue
        unregistered.append(rel_disp)

    # 1) manifest TSV
    man = ROOT / "04outputs/0NDATACARD_C5_manifest.tsv"
    with man.open("w", encoding="utf-8") as f:
        f.write("kind\tid\tsha256\tbytes\tpath\n")
        for r in rows:
            f.write(f"{r['kind']}\t{r['id']}\t{r['sha256']}\t{r['bytes']}\t{r['path']}\n")

    # 2) 中文数据卡（§1 项目内部工作区中文强制）
    card = ROOT / "refine-logs/0NDATACARD_C5.md"
    lines = [
        "# 0NDATACARD_C5.md — C5 稿件数据卡（SHA-256 登记）",
        f"\n生成时间：{stamp} · 生成脚本：`02code/120datacard_sha256.py`",
        "\n> 依据项目约定 §4.6 / §202。本卡只登记 **C5 稿件"
        "（`Elsevier_zh.tex` / `Elsevier_en.tex`）实际依赖**的文件，",
        "> 不含 C6/光谱方向的历史产物（`03data/processed/` 下 35 项与本稿无关）。",
        "\n> **诚实边界**：SHA-256 仅证明**自本次登记起**文件字节未变，不追认登记之前的历史，",
        "> 也不构成对提取步骤正确性的证明——提取正确性由脚本 55 的 `read_html` 与人工/Europe PMC 逐条核验承担。",
        "\n---\n\n## 一、源数据",
    ]
    for r in [x for x in rows if x["kind"] == "source"]:
        lines += [
            f"\n### {r['id']} — {r['name']}",
            f"- **角色**：{r['role']}",
            f"- **来源**：{r['origin']}",
            f"- **获取/提取方式**：{r['extraction']}",
            f"- **许可**：{r['license']}",
            f"- **在盘路径**：`{r['path']}`",
            f"- **大小**：{r['bytes']:,} bytes",
            f"- **SHA-256**：`{r['sha256']}`",
        ]
    lines += ["\n---\n\n## 二、派生产物（稿件每个数字的来源）", "",
          "| 产物 | 内容 | bytes | SHA-256 (前 16) |", "| --- | --- | ---: | --- |"]
    for r in [x for x in rows if x["kind"] == "derived"]:
        lines.append(f"| `{Path(r['path']).name}` | {r['name']} | {r['bytes']:,} | `{r['sha256'][:16]}` |")
    lines += [
        "\n完整 64 位摘要见 `04outputs/0NDATACARD_C5_manifest.tsv`。",
        "\n---\n\n## 三、与稿件的对应",
        "- 稿件表 `tab:datacard`（数据卡表）列出的三个真实体系即本卡第一节三项。",
        "- 稿件正文与补充材料的每个数字，其来源产物见本卡第二节，并在 `06doc/02sub/00SUBMISSION_AUDIT.md` §一"
        "（原 `claim_evidence_map.md`，2026-07-27 原样并入）中逐条绑定。",
        f"\n## 四、缺失项\n\n{'无。' if not missing else chr(10).join('- 未找到：`' + m + '`' for m in missing)}",
        "\n## 五、覆盖闸门（反向核对）\n",
        "> `缺失项` 只回答「清单里的文件在不在盘」，回答不了「在盘的该进清单的文件有没有进清单」。"
        "本节对已声明覆盖的目录反向再扫一遍：\n",
        "\n".join(f"- `{d}`" for d in GUARD_DIRS),
        f"\n**在盘但未登记**：{'无。' if not unregistered else chr(10).join('- `' + m + '`' for m in unregistered)}",
    ]
    card.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = ROOT / "04outputs/120datacard_sha256.json"
    out.write_text(json.dumps({
        "script": Path(__file__).name, "generated": stamp,
        "n_source": sum(1 for r in rows if r["kind"] == "source"),
        "n_derived": sum(1 for r in rows if r["kind"] == "derived"),
        "missing": missing,
        "guarded_dirs": GUARD_DIRS,
        "unregistered_in_guarded_dirs": unregistered,
        # 剥掉内部反向闸门用的绝对路径键：发布件不得带作者机目录结构
        "rows": [{k: v for k, v in r.items() if k != "abs"} for r in rows],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    n_src = sum(1 for r in rows if r["kind"] == "source")
    print(f"登记 {len(rows)} 个文件（源 {n_src} / 派生 {len(rows) - n_src}）")
    for r in rows:
        print(f"  [{r['kind']:7s}] {r['sha256'][:16]}  {r['bytes']:>10,}  {Path(r['path']).name}")
    print(f"缺失: {missing or '无'}")
    print(f"覆盖闸门（{len(GUARD_DIRS)} 个目录）在盘但未登记: {unregistered or '无'}")
    print(f"写出 {man}\n写出 {card}\n写出 {out}")


if __name__ == "__main__":
    main()
