"""
105multiseed_realdata.py: 真实数据四脚本的 Formal 5 种子复跑与跨种子聚合 (§5.3)

为什么必须做：
  论文 §实验协议 写「统计单位为 5 个固定种子（n_primary=5）」，但核查发现该陈述**只对半合成部分成立**
  （43trichotomy_formal.py 确用 RECOMMENDED_SEEDS）。而按 13.2 应用型档位属**主体**的真实数据部分，
  四个脚本全部是单种子、且无一取自预注册集合：
      54c5_realkinetics      20240530 / 20240531
      99_realkinetics2       20260531
      102_avocado_c5_pilot   SEED = 0      <- 支撑全文唯一「真统计估计误差」支柱
      103sensitivity_pilot   20240530
  §5.3 要求正式实验 ≥5 固定种子 [20060515, 20041210, 19810915, 2023, 2024]，并报 mean/std/95%CI
  （cluster bootstrap，种子为聚类单位）。本脚本补齐这一缺口。

改动方式（最小侵入、可审计）：
  四个被复跑脚本各加一个 SEED_OVERRIDE 环境变量入口，**默认值即原字面值**；未设置时行为逐字节不变。
  该性质已由回归闸门实测确认（54c5 与改动前产物最大绝对偏差 0.000e+00）。本脚本通过 env 逐种子驱动。

依赖顺序（不可并行）：
  103sensitivity_pilot 读取 54c5 产出的 -designsweep.csv，故同一种子内必须 54c5 先于 103。
  99 与 102 相互独立。为稳妥，全部按 54c5 -> 103 -> 99 -> 102 串行。

聚合口径：
  逐 JSON 扁平化为「点分键 -> 数值」，跨 5 种子对每个键报 mean/std/CV/min/max，
  并以**种子为聚类单位**做 percentile cluster bootstrap 95% CI（§7.2 禁止 i.i.d. bootstrap）。
  n_primary = 5（种子数），不冒充为独立样本。

运行方式:
    cd <项目根目录绝对路径>
    python 02code/105multiseed_realdata.py --device auto

输出文件:
    04outputs/105multiseed_realdata.xlsx  — sheets: summary / per_seed / headline / runs
    04outputs/105multiseed_realdata.json  — 机读汇总
    04outputs/105_seeds/<脚本>-seed<种子>.json  — 每种子原始产物（流式明细，不计入一脚本一 xlsx）
    05logs/105multiseed_realdata_YY-MM-DD_HHMMSS.log — 运行日志

方法论证锚点: refine-logs/76MULTISEED_REALDATA_DESIGN.md（§1.9 四段式），台账 M54。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from export_utils import RECOMMENDED_SEEDS, get_logger, table_sheet_name, write_script_workbook

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "02code"
OUT = ROOT / "04outputs"
SEED_DIR = OUT / "105_seeds"
STEM = Path(__file__).stem
N_BOOT_CLUSTER = 2000

# (脚本 stem, 该脚本产出的需归档文件名列表)。顺序即执行顺序，见 docstring「依赖顺序」。
SCRIPTS: list[tuple[str, list[str]]] = [
    ("54c5_realkinetics", ["54c5_realkinetics.json", "54c5_realkinetics-designsweep.csv"]),
    ("103sensitivity_pilot", ["103sensitivity_pilot.json", "103sensitivity_pilot-grid.csv"]),
    ("99_realkinetics2", ["99_realkinetics2.json"]),
    ("102_avocado_c5_pilot", ["102_avocado_c5_pilot.json"]),
]

# 论文正文直接引用的指标（点分键），单列一张 headline 表便于逐条核对
HEADLINE = {
    "54c5_realkinetics": [
        "overdetermined_real_error.spearman",
        "overdetermined_real_error.logPsi_vs_logRealCV_slope",
        "overdetermined_real_error.r2",
        "design_sweep.logPsi_vs_logCV_slope",
        "design_sweep.r2",
        "full.Ea_kJmol",
        "full.sigma_lnk",
    ],
    "99_realkinetics2": [
        "overdetermined_real_error.spearman",
        "overdetermined_real_error.het_spearman",
    ],
}


def flatten(obj: object, prefix: str = "") -> dict[str, float]:
    """把嵌套 JSON 压成「点分键 -> 数值」；只保留有限数值叶子，bool 视为非数值丢弃。"""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}{i}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        val = float(obj)
        if np.isfinite(val):
            out[prefix.rstrip(".")] = val
    return out


def cluster_bootstrap_ci(vals: np.ndarray, rng: np.random.Generator,
                         n_boot: int = N_BOOT_CLUSTER) -> tuple[float, float]:
    """以种子为聚类单位的 percentile bootstrap 95% CI（§7.2：禁止 i.i.d. bootstrap）。"""
    if len(vals) < 2 or not np.all(np.isfinite(vals)):
        return float("nan"), float("nan")
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="被复跑脚本均为 CPU scipy 计算，device 仅记录")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(RECOMMENDED_SEEDS),
                    help="默认取 export_utils.RECOMMENDED_SEEDS（§5.3 Formal 固定集合）")
    args = ap.parse_args()

    log = get_logger(STEM)
    seeds = list(args.seeds)
    log.log(f"=== {STEM} 启动 | device={args.device} | 种子 {seeds} ===")
    if list(RECOMMENDED_SEEDS) != seeds:
        log.log(f"⚠ 注意：所用种子与 §5.3 预注册集合 {list(RECOMMENDED_SEEDS)} 不一致，结果不得作为 Formal 证据")
    SEED_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- (1) 逐种子逐脚本复跑 ----------
    runs: list[dict] = []
    for seed in seeds:
        for stem, artifacts in SCRIPTS:
            env = dict(os.environ, SEED_OVERRIDE=str(seed))
            t0 = time.time()
            proc = subprocess.run([sys.executable, str(CODE / f"{stem}.py")],
                                  cwd=str(ROOT), env=env, capture_output=True, text=True)
            dt = time.time() - t0
            ok = proc.returncode == 0
            log.log(f"  seed={seed:<9d} {stem:<22s} rc={proc.returncode} {dt:6.1f}s "
                    f"{'OK' if ok else 'FAIL: ' + proc.stderr.strip().splitlines()[-1][:120]}")
            archived = []
            if ok:
                for art in artifacts:
                    src = OUT / art
                    if not src.exists():
                        raise FileNotFoundError(f"IRON RULE 违反：{stem} seed={seed} 声称成功但 {src} 不在盘")
                    dst = SEED_DIR / f"{Path(art).stem}-seed{seed}{Path(art).suffix}"
                    shutil.copy2(src, dst)
                    archived.append(dst.name)
            runs.append(dict(seed=seed, script=stem, returncode=proc.returncode,
                             seconds=round(dt, 2), ok=ok, archived=";".join(archived),
                             stderr_tail=proc.stderr.strip()[-300:] if not ok else ""))
    if not all(r["ok"] for r in runs):
        bad = [(r["seed"], r["script"]) for r in runs if not r["ok"]]
        raise RuntimeError(f"存在失败的复跑，拒绝聚合：{bad}")
    log.log(f"复跑完成：{len(runs)} 次全部成功（{len(seeds)} 种子 × {len(SCRIPTS)} 脚本）")

    # ---------- (2) 跨种子聚合 ----------
    rng = np.random.default_rng(20060515)  # 聚合层自身的 bootstrap 种子，取 RECOMMENDED_SEEDS[0]
    per_seed_rows: list[dict] = []
    for stem, _ in SCRIPTS:
        for seed in seeds:
            f = SEED_DIR / f"{stem}-seed{seed}.json"
            flat = flatten(json.loads(f.read_text(encoding="utf-8")))
            for k, v in flat.items():
                per_seed_rows.append(dict(script=stem, seed=seed, key=k, value=v))
    per_seed = pd.DataFrame(per_seed_rows)
    log.log(f"扁平化得 {per_seed.key.nunique()} 个不同指标键，共 {len(per_seed)} 条 (脚本×种子×键)")

    sum_rows = []
    for (stem, key), g in per_seed.groupby(["script", "key"], sort=False):
        if len(g) != len(seeds):
            continue  # 某些键并非每个种子都出现（如条件分支产物），跳过以免口径不齐
        v = g.sort_values("seed").value.to_numpy(float)
        lo, hi = cluster_bootstrap_ci(v, rng)
        mean, std = float(v.mean()), float(v.std(ddof=1))
        sum_rows.append(dict(script=stem, key=key, n_seeds=len(seeds),
                             mean=mean, std=std,
                             CV_pct=float(abs(std / mean) * 100) if mean != 0 else float("nan"),
                             ci_lo=lo, ci_hi=hi, vmin=float(v.min()), vmax=float(v.max()),
                             spread_x=float(v.max() / v.min()) if v.min() > 0 else float("nan"),
                             seed_values=";".join(f"{x:.6g}" for x in v)))
    summary = pd.DataFrame(sum_rows)

    head_rows = []
    for stem, keys in HEADLINE.items():
        for k in keys:
            hit = summary[(summary.script == stem) & (summary.key == k)]
            if hit.empty:
                log.log(f"⚠ headline 指标缺失：{stem}.{k}（该键未在全部种子中出现）")
                continue
            r = hit.iloc[0]
            head_rows.append(dict(脚本=stem, 指标=k, n_seeds=int(r.n_seeds),
                                  mean=round(float(r["mean"]), 6), std=round(float(r["std"]), 6),
                                  CI95=f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}]",
                                  各种子值=r.seed_values))
    headline = pd.DataFrame(head_rows)

    log.log("=== 论文正文指标的 5 种子稳健性（mean ± std，CI 为种子聚类 bootstrap）===")
    for r in headline.itertuples():
        log.log(f"  {r.脚本:<22s} {r.指标:<44s} {r.mean:+.4f} ± {r.std:.4f}  CI95={r.CI95}")

    (OUT / f"{STEM}.json").write_text(json.dumps(
        {"seeds": seeds, "n_primary": len(seeds),
         "runs": runs, "headline": head_rows,
         "summary": summary.to_dict("records"),
         "口径": "n_primary=种子数；CI 为以种子为聚类单位的 percentile bootstrap；不冒充独立样本"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    write_script_workbook(__file__, {
        table_sheet_name(1, "headline"): headline,
        table_sheet_name(2, "summary"): summary,
        table_sheet_name(3, "per_seed"): per_seed,
        table_sheet_name(4, "runs"): pd.DataFrame(runs),
    })
    log.log(f"DONE {STEM}")


if __name__ == "__main__":
    main()
