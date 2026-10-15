"""
128montecarlo_runtime_receipt.py: 为 123montecarlo_criterion_study.py 出具【可复跑的】运行时收据。

为什么需要本脚本（RYA-105，2026-08-10）：
  稿件 §sec:mc 的算力披露句引用了 123 的墙钟时间，此前只有一份产物
  `04outputs/128montecarlo_runtime_receipt.json` 而【仓库内没有生成它的脚本】——
  取证全扫据此开出义务「Runtime reproducibility receipt hashes a different Monte Carlo artifact」：
  该收据记 frozen_sha256=253fde5d…，而在盘 123 产物实算为另一值，收据已过期且【无生成器可复跑】。
  本脚本补齐生成器，使该收据此后随 123 一起可被任何人重新出具与核验。

做法（与旧收据自述的方法一致，但这次是可执行的）：
  1) 在隔离沙箱重跑 123：沙箱内 02code/ 只放 123 脚本本体，03data 以【只读符号链接】接入，
     输出写进沙箱自己的 04outputs/，全程不触碰项目 04outputs 的冻结件。
  2) 重复 N_REPEAT 次（默认 3，可用 RYA_RUNTIME_REPEATS 覆盖），逐次用 resource.getrusage
     量子进程 user/sys CPU 时间、用 time.perf_counter 量墙钟，全部落盘。
  3) 对【沙箱重跑产物】与【在盘冻结产物】各自实算 SHA-256，并逐键比对全部数值叶子，
     报 n_keys_compared / n_keys_differing / byte_identical。
  4) 写 04outputs/128montecarlo_runtime_receipt.json。

诚实边界：
  - 墙钟随机器负载波动；收据记录的是本次 N_REPEAT 次测量的逐次值与中位/极值，
    不是唯一正确值。稿面只许引用落在这些实测值之内的区间。
  - 本收据只覆盖 123 一个脚本；它不是「全部正式实验」的算力总账，稿面不得据此推广。
  - byte_identical 为真只说明在同一 numpy/scipy 环境下该研究确定性可复现（123 全程 default_rng(SEED)）。
  - 若 123 或其依赖被改动，本收据即失效，须重跑本脚本重出。

运行: python 02code/128montecarlo_runtime_receipt.py
输出: 04outputs/128montecarlo_runtime_receipt.json
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "02code"
OUT = ROOT / "04outputs"
TARGET = "123montecarlo_criterion_study"
FROZEN = OUT / f"{TARGET}.json"
RECEIPT = OUT / "128montecarlo_runtime_receipt.json"
SANDBOX = Path.home() / ".cache" / "rya105_mc_runtime_receipt"
HOME_TOKEN = "${HOME}"

# 【2026-08-19 RYA-526，取证义务 737748de】重复测量次数。
# 成因：原实现只跑一次 subprocess.run、只落一个 wall_clock_sec，而稿面写的是
# 「多次实测落在 500--550 秒」——收据撑不起复数。改为默认重复 N_REPEAT 次，
# 逐次墙钟与 CPU 时间全部落盘，稿面只许引用本收据实际给出的区间。
N_REPEAT = int(os.environ.get("RYA_RUNTIME_REPEATS", "3"))


def display_path(p: Path) -> str:
    """登记用路径：仓库内 → 仓库相对；家目录下 → ${HOME}/… 占位（同 120 的口径）。

    【2026-08-10 订正，RYA-108】本脚本原样写出 str(SANDBOX)，即作者机绝对路径
    `${HOME}/.cache/rya105_mc_runtime_receipt`（原文为作者机绝对路径），随产物一路带进发布树 08github/。
    发布件单独脱敏是上一轮 55 件漂移的成因，故一律在正典侧脱敏，使正典与发布件可逐字节相同。"""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        pass
    try:
        return f"{HOME_TOKEN}/{p.resolve().relative_to(Path.home().resolve()).as_posix()}"
    except ValueError:
        return p.name          # 兜底：只留文件名，仍不泄露绝对路径


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_leaves(obj: object, prefix: str = "") -> dict[str, float]:
    """把嵌套 JSON 压成「点分键 -> 数值」，只保留数值叶子（bool 视为非数值丢弃）。"""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(numeric_leaves(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(numeric_leaves(v, f"{prefix}{i}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = float(obj)
    return out


def main() -> None:
    if not FROZEN.exists():
        raise FileNotFoundError(f"IRON RULE 违反：在盘冻结产物 {FROZEN} 不存在，无从出具收据")
    src = CODE / f"{TARGET}.py"
    if not src.exists():
        raise FileNotFoundError(f"IRON RULE 违反：被计时脚本 {src} 不在盘")

    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    (SANDBOX / "02code").mkdir(parents=True)
    (SANDBOX / "04outputs").mkdir(parents=True)
    shutil.copy2(src, SANDBOX / "02code" / src.name)
    (SANDBOX / "03data").symlink_to(ROOT / "03data")     # 只读接入，不复制原始数据

    runs: list[dict[str, float]] = []
    rerun = SANDBOX / "04outputs" / f"{TARGET}.json"
    rerun_shas: list[str] = []
    for i in range(N_REPEAT):
        if rerun.exists():
            rerun.unlink()                       # 每次都必须由本次重跑真正重新生成
        load_at_start = os.getloadavg()[0]
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(SANDBOX / "02code" / src.name)],
                              cwd=str(SANDBOX), capture_output=True, text=True)
        wall = time.perf_counter() - t0
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        if proc.returncode != 0:
            raise RuntimeError(f"沙箱第 {i+1} 次重跑失败 rc={proc.returncode}\n{proc.stderr[-2000:]}")
        if not rerun.exists():
            raise FileNotFoundError(f"IRON RULE 违反：第 {i+1} 次沙箱声称成功但 {rerun} 不在盘")
        runs.append({
            "repeat": i + 1,
            "wall_clock_sec": round(wall, 2),
            "user_sec": round(after.ru_utime - before.ru_utime, 2),
            "sys_sec": round(after.ru_stime - before.ru_stime, 2),
            "cpu_sec": round((after.ru_utime - before.ru_utime)
                             + (after.ru_stime - before.ru_stime), 2),
            # 本机为多用户共享工作站，墙钟受同机其他负载支配（实测 1-min load average 可达 100+）。
            # 【2026-08-21 订正，RYA-569】原注写「CPU 时间不受此影响」——本轮实测**推翻**了这一点：
            # 三次重跑的 user_sec 稳定在 386–567 s，而 sys_sec 跨 338–1457 s，总 CPU 因此从 725 s
            # 涨到 2024 s。系统态时间来自同机任务耗尽物理内存后的换页开销，它确实进入 CPU 时间。
            # 故不受负载影响的是【用户态】CPU 时间，稿面应引用 user_sec；总 CPU 与墙钟只作参考。
            # 另：起测 1-min 负载不预测该代价——起测负载最高（106.86）的那次反而最快。
            "loadavg_1min_at_start": round(load_at_start, 2),
        })
        rerun_shas.append(sha256(rerun))
        print(f"  重复 {i+1}/{N_REPEAT}: wall={wall:.2f}s")

    walls = sorted(r["wall_clock_sec"] for r in runs)
    cpus = sorted(r["cpu_sec"] for r in runs)
    wall = walls[len(walls) // 2]                # 收据的代表值取中位数，不取首次
    cpu_med = cpus[len(cpus) // 2]

    a = numeric_leaves(json.loads(FROZEN.read_text(encoding="utf-8")))
    b = numeric_leaves(json.loads(rerun.read_text(encoding="utf-8")))
    keys = sorted(set(a) | set(b))
    differing = [k for k in keys if a.get(k) != b.get(k)]
    frozen_sha, rerun_sha = sha256(FROZEN), sha256(rerun)

    receipt = {
        "script": Path(__file__).name,
        "target_script": src.name,
        "purpose": "稿件 §sec:mc 算力披露句所引墙钟时间的可核查凭据；本收据由本脚本生成，可随时重跑重出。",
        "measured": {
            "n_repeats": N_REPEAT,
            "per_run": runs,
            "cpu_sec_median": cpu_med,
            "cpu_sec_min": cpus[0],
            "cpu_sec_max": cpus[-1],
            "wall_clock_sec_median": wall,
            "wall_clock_sec_min": walls[0],
            "wall_clock_sec_max": walls[-1],
            "user_sec_total": round(sum(r["user_sec"] for r in runs), 2),
            "cpu_hours_total_this_receipt": round(
                sum(r["user_sec"] + r["sys_sec"] for r in runs) / 3600.0, 4),
            "threads": "single process (no BLAS fan-out configured)",
            "note": ("稿面引用的时间区间必须落在 per_run 实测值之内。"
                     "【2026-08-21 订正，RYA-569】本机为共享工作站；实测表明受同机负载影响的不只是墙钟——"
                     "同机任务耗尽物理内存时换页开销进入 sys_sec，总 CPU 随之数倍上涨，而 user_sec 相对稳定。"
                     "故刻画算法本身的量是 user_sec，稿面引用它，总 CPU 与墙钟一并报出仅作参考；"
                     "本收据只覆盖 %s 一个脚本，不构成全部正式实验的算力总账。" % TARGET),
        },
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "reproducibility": {
            "method": "脚本原样在隔离沙箱重跑（03data 只读符号链接，输出改写到沙箱 04outputs），未触碰项目 04outputs 冻结件",
            "sandbox": display_path(SANDBOX),
            "frozen_artifact": str(FROZEN.relative_to(ROOT)),
            "frozen_sha256": frozen_sha,
            "rerun_sha256": rerun_sha,
            "rerun_sha256_all_repeats": rerun_shas,
            "rerun_deterministic_across_repeats": len(set(rerun_shas)) == 1,
            "byte_identical": frozen_sha == rerun_sha,
            "n_keys_compared": len(keys),
            "n_keys_differing": len(differing),
            "differing_keys": differing[:50],
        },
        "honesty": [
            "墙钟随机器负载波动；本收据记录的是本次测量值，不是唯一正确值。",
            "byte_identical 为真只说明同一 numpy/scipy 环境下确定性可复现（123 全程 default_rng(SEED)）。",
            "123 或其依赖一经改动本收据即失效，须重跑 128 重出。",
        ],
    }
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"收据已写出 {RECEIPT}")
    print(f"  wall(median of {N_REPEAT})={wall:.2f}s  range=[{walls[0]:.2f},{walls[-1]:.2f}]  frozen={frozen_sha[:16]}…  rerun={rerun_sha[:16]}…  "
          f"identical={frozen_sha == rerun_sha}  keys={len(keys)} diff={len(differing)}")
    if differing:
        print(f"  ⚠ 差异键（前 10）：{differing[:10]}")


if __name__ == "__main__":
    main()
