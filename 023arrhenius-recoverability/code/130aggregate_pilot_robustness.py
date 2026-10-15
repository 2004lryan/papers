"""
130aggregate_pilot_robustness.py: 把 103sensitivity_pilot 的 5 种子产物聚合成稿件 §sec:relate-id
（zh L454 / en 对应句）所引的「最差档 ±30% 下 96% 保持负号、92% 保持 ≤−0.5」两个比例。

为什么需要本脚本（RYA-105，2026-08-10）：
  取证全扫开出义务「Current 96%/92% sensitivity aggregate lacks its stated generating step」——
  这两个数字此前只存在于 `05logs/rya103_verify_20260810/aggregate_result.json` 这份人工产物里，
  其同目录 runner (`rya103_verify_runner.sh`) 只负责【逐种子复跑】，【没有】那一步把 5 个
  per-seed JSON 聚合成 96/92 的代码。本脚本补齐该步骤，使这两个稿面数字可被独立重算。

口径（与 aggregate_result.json 自述一致，不做任何改动）：
  - 统计单位 = 5 个预注册种子（§5.3 RECOMMENDED_SEEDS），逐种子等权、不按 n 加权；
  - 对每个扰动档 pm10/pm20/pm30，取各种子的 frac_same_sign_as_baseline 与 frac_strong 的算术均值；
  - 报百分比时用 ROUND_HALF_UP（Python 内建 round() 是 banker's rounding，不可用）。

输入：04outputs/105_seeds/103sensitivity_pilot-seed<种子>.json（由 105multiseed_realdata.py 产出）
输出：04outputs/130aggregate_pilot_robustness.json

运行: python 02code/130aggregate_pilot_robustness.py
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from export_utils import RECOMMENDED_SEEDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "04outputs"
SEED_DIR = OUT / "105_seeds"
STEM = Path(__file__).stem
LEVELS = ("pm10pct", "pm20pct", "pm30pct")


def half_up_pct(x: float) -> int:
    """百分比取整，ROUND_HALF_UP（内建 round() 为 banker's rounding，会把 0.5 抹成偶数）。"""
    return int(Decimal(str(x * 100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def main() -> None:
    seeds = list(RECOMMENDED_SEEDS)
    per_seed: dict[int, dict[str, Any]] = {}
    for s in seeds:
        f = SEED_DIR / f"103sensitivity_pilot-seed{s}.json"
        if not f.exists():
            raise FileNotFoundError(f"IRON RULE 违反：{f} 不在盘（先跑 105multiseed_realdata.py）")
        per_seed[s] = json.loads(f.read_text(encoding="utf-8"))["L2_gweighted_psi"]["perturbation"]

    agg: dict[str, dict[str, Any]] = {}
    for lv in LEVELS:
        ss = [float(per_seed[s][lv]["frac_same_sign_as_baseline"]) for s in seeds]
        st = [float(per_seed[s][lv]["frac_strong"]) for s in seeds]
        ns = [int(per_seed[s][lv]["n"]) for s in seeds]
        agg[lv] = {
            "n_per_seed": ns,
            "per_seed_same_sign": ss,
            "per_seed_strong": st,
            "same_sign_mean": sum(ss) / len(ss),
            "strong_mean": sum(st) / len(st),
            "same_sign_pct_half_up": half_up_pct(sum(ss) / len(ss)),
            "strong_pct_half_up": half_up_pct(sum(st) / len(st)),
        }

    worst = agg["pm30pct"]
    result = {
        "script": Path(__file__).name,
        "purpose": "稿件 §sec:relate-id「最差档 ±30% 下 X% 保持负号、Y% 保持 ≤−0.5」两个比例的生成步骤",
        "inputs": [f"04outputs/105_seeds/103sensitivity_pilot-seed{s}.json" for s in seeds],
        "seeds": seeds,
        "n_primary": len(seeds),
        "口径": ("逐种子等权算术均值（不按 n 加权），百分比 ROUND_HALF_UP；"
                 "frac_same_sign_as_baseline=扰动后 Spearman 与 baseline 同号的比例，"
                 "frac_strong=扰动后 Spearman ≤ −0.5 的比例"),
        "aggregate": agg,
        "paper_numbers": {
            "worst_level": "pm30pct",
            "same_sign_pct": worst["same_sign_pct_half_up"],
            "strong_pct": worst["strong_pct_half_up"],
        },
    }
    (OUT / f"{STEM}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for lv in LEVELS:
        a = agg[lv]
        print(f"  {lv}: 同号 {a['same_sign_mean']*100:.2f}% → {a['same_sign_pct_half_up']}%  |  "
              f"强 {a['strong_mean']*100:.2f}% → {a['strong_pct_half_up']}%  (n/seed={a['n_per_seed']})")
    print(f"稿面数字（最差档 pm30）：{worst['same_sign_pct_half_up']}% / {worst['strong_pct_half_up']}%")
    print(f"已写出 {OUT / (STEM + '.json')}")


if __name__ == "__main__":
    main()
