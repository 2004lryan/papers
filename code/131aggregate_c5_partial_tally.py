"""
131aggregate_c5_partial_tally.py: 把 91c5 族 6 个产物的偏相关逐次胜负【聚合成一个可复核的产物】。

为什么需要本脚本（RYA-108，2026-08-10）：
  稿面 §sec:res-c5 与主张矩阵报的是【跨 6 次抽样】的汇总判断（原文「$|$偏相关$|$ 在 6/6 次中 Ψ_n
  均大于 cond」），而 91c5 每个产物只承载【本次抽样】，仓库内【没有】做这一步聚合的脚本——
  这个 6/6 是人手从 6 个 JSON 里数出来的，与 RYA-105 清偿的 B3（96%/92% 只存在于人工产物、
  无聚合器）是同一类缺陷。逐条复算实测：标准口径下为【5 次严格大于 + 1 次数值持平】
  （seed2024 的 |偏 Ψ_n| 与 |偏 cond| 相差 2.8e-17，双精度噪声量级，不构成大小关系），
  故 6/6 是一处把浮点噪声当胜负的溢出。本脚本补齐聚合器，使该汇总此后随 91c5 一起可被任何人重出。

做法：
  1) 读 04outputs/91c5_realkinetics_killer.json（原生抽样）与 5 个预注册种子件。
  2) 对【两个偏相关口径】各自逐次比较 |偏 Ψ_n| 与 |偏 cond|，以 1e-12 为持平容差（双精度下
     两个 O(1) 量的差小于此值即无法区分大小），统计 strict / tie / lose 三分。
  3) 同时给出稿面引用的 5 种子均值与范围（Ψ_n、cond），使这些数字亦有现算来源。

诚实边界：
  - 持平容差 1e-12 是【口径选择】不是实测量：seed2024 的实测差 2.8e-17 比它小 5 个数量级，
    换任何 1e-15~1e-9 之间的容差结论不变；本产物把实测差原样写出，供任何人自行判定。
  - 6 次抽样共用同一批 16 个非独立子集，故这是【描述性】计数，不是 6 个独立重复的检验。
  - 顶层键那一路是 91c5 的原口径、非稿面依据，此处并报只为回答「换口径会不会翻结论」。

运行: python 02code/131aggregate_c5_partial_tally.py
输出: 04outputs/131aggregate_c5_partial_tally.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "04outputs"
OUT = OUT_DIR / "131aggregate_c5_partial_tally.json"

FORMAL_SEEDS = [20060515, 20041210, 19810915, 2023, 2024]
TIE_TOL = 1e-12


def draws() -> list[tuple[str, Path]]:
    """(抽样名, 产物路径)。原生抽样在前，5 个预注册种子按正典次序在后。"""
    items = [("native", OUT_DIR / "91c5_realkinetics_killer.json")]
    items += [(f"seed{s}", OUT_DIR / f"91c5_realkinetics_killer_seed{s}.json") for s in FORMAL_SEEDS]
    for _, p in items:
        if not p.exists():
            raise FileNotFoundError(f"IRON RULE 违反：被聚合件 {p} 不在盘，拒绝以缺件的子集冒充 6 次抽样")
    return items


def cmp_abs(a: float, b: float) -> str:
    d = abs(a) - abs(b)
    if abs(d) <= TIE_TOL:
        return "tie"
    return "Psi_n" if d > 0 else "cond"


def main() -> None:
    per_draw: list[dict[str, Any]] = []
    for name, path in draws():
        blk = json.loads(path.read_text(encoding="utf-8"))["partial_spearman_control_deltaT"]
        std, var = blk["_standard_definition"], blk
        row = {
            "draw": name,
            "artifact": path.relative_to(ROOT).as_posix(),
            "paper_convention": {
                "Psi_n": std["Psi_n_given_deltaT"],
                "cond": std["cond_given_deltaT"],
                "abs_gap_Psi_n_minus_cond": abs(std["Psi_n_given_deltaT"]) - abs(std["cond_given_deltaT"]),
                "larger": cmp_abs(std["Psi_n_given_deltaT"], std["cond_given_deltaT"]),
            },
            "script_original_convention": {
                "Psi_n": var["Psi_n_given_deltaT"],
                "cond": var["cond_given_deltaT"],
                "abs_gap_Psi_n_minus_cond": abs(var["Psi_n_given_deltaT"]) - abs(var["cond_given_deltaT"]),
                "larger": cmp_abs(var["Psi_n_given_deltaT"], var["cond_given_deltaT"]),
            },
        }
        per_draw.append(row)

    def tally(key: str) -> dict[str, Any]:
        got = [r[key]["larger"] for r in per_draw]
        return {"n_draws": len(got),
                "Psi_n_strictly_larger": got.count("Psi_n"),
                "numeric_tie": got.count("tie"),
                "cond_larger": got.count("cond"),
                "per_draw": {r["draw"]: r[key]["larger"] for r in per_draw}}

    seeds_only = [r for r in per_draw if r["draw"] != "native"]

    def stats(key: str, which: str) -> dict[str, Any]:
        v = np.array([r[key][which] for r in seeds_only], dtype=float)
        return {"n_seeds": int(v.size), "mean": float(v.mean()),
                "min": float(v.min()), "max": float(v.max()),
                "per_seed": {r["draw"]: r[key][which] for r in seeds_only}}

    ties = [r for r in per_draw if r["paper_convention"]["larger"] == "tie"]
    res: dict[str, Any] = {
        "study": "91c5 族 6 次抽样的偏 Spearman 逐次胜负聚合（控制 ΔT 后 |偏 Ψ_n| vs |偏 cond|）",
        "inputs": [r["artifact"] for r in per_draw],
        "tie_tolerance": TIE_TOL,
        "tie_tolerance_note": "双精度下两个 O(1) 量之差小于该值即无法区分大小，判「数值持平」而非判胜负",
        "paper_convention_key": "_standard_definition（稿面 §sec:res-c5 明写的教科书标准偏 Spearman）",
        "tally_paper_convention": tally("paper_convention"),
        "tally_script_original_convention": tally("script_original_convention"),
        "five_seed_stats_paper_convention": {
            "Psi_n": stats("paper_convention", "Psi_n"),
            "cond": stats("paper_convention", "cond"),
        },
        "per_draw": per_draw,
        "numeric_ties_detail": [
            {"draw": r["draw"], "Psi_n": r["paper_convention"]["Psi_n"], "cond": r["paper_convention"]["cond"],
             "abs_gap": r["paper_convention"]["abs_gap_Psi_n_minus_cond"],
             "note": "两值之差在双精度噪声量级，稿面不得据此判胜负"}
            for r in ties
        ],
        "honest_boundaries": (
            "(1) 6 次抽样共用同一批 16 个相互嵌套的非独立子集，本表为【描述性计数】、非 6 个独立重复的检验；"
            "(2) 持平容差 1e-12 属口径选择，实测最小非零差与它相差数个量级，逐次实测差已原样写出供自查；"
            "(3) `tally_script_original_convention` 那一路是 91c5 的原口径、【非】稿面依据，"
            "并报只为回答「换口径会不会翻结论」；两口径结论不一致时以稿面口径为准并如实披露差异。"
        ),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    t = res["tally_paper_convention"]
    tv = res["tally_script_original_convention"]
    print("=== 91c5 六次抽样偏相关逐次胜负聚合 ===")
    print(f"  稿面口径(_standard_definition)：Ψ_n 严格更大 {t['Psi_n_strictly_larger']}/{t['n_draws']}、"
          f"数值持平 {t['numeric_tie']}、cond 更大 {t['cond_larger']}")
    print(f"  脚本原口径(顶层键)：            Ψ_n 严格更大 {tv['Psi_n_strictly_larger']}/{tv['n_draws']}、"
          f"数值持平 {tv['numeric_tie']}、cond 更大 {tv['cond_larger']}")
    for r in res["numeric_ties_detail"]:
        print(f"  ⚠ 持平：{r['draw']}  Ψ_n={r['Psi_n']!r}  cond={r['cond']!r}  |差|={abs(r['abs_gap']):.3e}")
    s = res["five_seed_stats_paper_convention"]
    print(f"  5 种子均值(稿面口径)：Ψ_n={s['Psi_n']['mean']:+.4f} [{s['Psi_n']['min']:+.4f},{s['Psi_n']['max']:+.4f}]、"
          f"cond={s['cond']['mean']:+.4f} [{s['cond']['min']:+.4f},{s['cond']['max']:+.4f}]")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
