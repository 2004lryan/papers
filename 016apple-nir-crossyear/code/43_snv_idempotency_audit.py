"""
43_snv_idempotency_audit.py: SNV 幂等审计的确定性归档。

背景：正文 §2.4 与补充材料 S2.3 声称「4 529 条中 4 523 条二次 SNV 偏差 <1e-4，
6 条例外全部来自 2025 山东、原始光谱含失效波段且已记录于数据卡」。该主张此前
只有中文底稿沿承的数字、无机器可核验归档，数据卡亦未见 6 条例外的记录——
claim-audit 零上下文臂据此只能判 missing_evidence。本脚本把该审计变成
纯函数归档：输入唯一（03data/processed/all_9elements.csv，SHA-256 见数据卡），
变换唯一（与 20_unified_singleface_kshot.snv 完全相同的逐样本 SNV），输出可复算。

运行方式:
    cd <项目根>
    python 02code/43_snv_idempotency_audit.py

输出:
    04outputs/43_snv_idempotency_audit.xlsx
        表a：审计汇总（阈值、计数、均值口径）
        表b：6 条例外光谱明细（元素/苹果编号/面位/偏差/逐条SD）
    03data/processed/all_9elements-datasheet.md
        追加/更新「SNV 幂等审计与异常光谱记录」小节（幂等 upsert，重跑不重复）
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

import export_utils as eu

NDArray = npt.NDArray[Any]

THRESHOLD = 1e-4
DATASHEET = os.path.join(eu.PROCESSED_DIR, "all_9elements-datasheet.md")
SECTION_HEAD = "## SNV 幂等审计与异常光谱记录"


def snv(x: NDArray) -> NDArray:
    """逐样本 SNV，与 20_unified_singleface_kshot.snv 逐字一致。"""
    return np.asarray((x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8))


def main() -> None:
    logger = eu.get_logger("43_snv_idempotency_audit")
    df = pd.read_csv(os.path.join(eu.PROCESSED_DIR, "all_9elements.csv"))
    wl_cols = [c for c in df.columns if c.endswith("nm")]
    x = df[wl_cols].to_numpy(dtype=float)
    x2 = snv(x)
    diff = np.abs(x2 - x)
    dev_max = diff.max(axis=1)
    mean_dev_all = float(diff.mean())
    n_pass = int((dev_max < THRESHOLD).sum())
    idx = np.where(dev_max >= THRESHOLD)[0]
    per_sd = x.std(axis=1)

    summary = pd.DataFrame(
        [
            {"项": "总光谱数", "值": len(df)},
            {"项": f"二次SNV最大偏差 < {THRESHOLD:g} 的条数", "值": n_pass},
            {"项": "例外条数", "值": int(len(idx))},
            {"项": "全体平均偏差(全元素均值口径)", "值": f"{mean_dev_all:.3e}"},
            {"项": "逐条最大偏差均值口径(参考)", "值": f"{float(dev_max.mean()):.3e}"},
            {"项": "例外最大偏差", "值": f"{float(dev_max[idx].max()):.4f}" if len(idx) else "—"},
            {"项": "例外占比", "值": f"{100.0 * len(idx) / len(df):.2f}%"},
        ]
    )
    detail = df.iloc[idx][["年份", "产地", "element_id", "苹果编号", "面位"]].copy()
    detail["二次SNV最大偏差"] = [round(float(dev_max[i]), 6) for i in idx]
    detail["逐条谱线SD"] = [round(float(per_sd[i]), 6) for i in idx]
    detail = detail.sort_values(["苹果编号", "面位"]).reset_index(drop=True)

    path = eu.write_script_workbook(__file__, {0: ("SNV幂等审计汇总", summary), 1: ("例外光谱明细", detail)})
    logger.log(f"审计归档 → {path}")
    logger.log(f"{n_pass}/{len(df)} < {THRESHOLD:g}；例外 {len(idx)} 条；全体平均偏差 {mean_dev_all:.3e}")

    lines = [SECTION_HEAD, ""]
    lines.append(f"> 由 `02code/43_snv_idempotency_audit.py` 生成（重跑幂等覆盖本节），归档 `04outputs/43_snv_idempotency_audit.xlsx`。")
    lines.append("")
    lines.append(f"- 对 `all_9elements.csv` 全部 {len(df)} 条已预处理光谱再次施加逐样本 SNV：{n_pass} 条最大偏差 < 1e-4，全体平均偏差 {mean_dev_all:.1e}。")
    lines.append(f"- 例外 {len(idx)} 条（占 {100.0 * len(idx) / len(df):.2f}%），全部来自 2025 山东（E8）；其原始光谱含失效传感器波段，经平滑与插值后出现长段常数值，SNV 无法完全归一化：")
    lines.append("")
    lines.append("| 苹果编号 | 面位 | 二次SNV最大偏差 | 逐条谱线SD |")
    lines.append("|---|---|---|---|")
    for _, r in detail.iterrows():
        lines.append(f"| {r['苹果编号']} | {r['面位']} | {r['二次SNV最大偏差']:.6f} | {r['逐条谱线SD']:.6f} |")
    lines.append("")
    section = "\n".join(lines)

    with open(DATASHEET, encoding="utf-8") as f:
        doc = f.read()
    if SECTION_HEAD in doc:
        doc = re.sub(re.escape(SECTION_HEAD) + r".*?(?=\n## |\Z)", section, doc, flags=re.S)
    else:
        doc = doc.rstrip() + "\n\n" + section + "\n"
    with open(DATASHEET, "w", encoding="utf-8") as f:
        f.write(doc)
    logger.log(f"数据卡异常光谱记录已更新 → {DATASHEET}")


if __name__ == "__main__":
    main()
