"""
export_utils.py: 项目全局输入输出与科研合规收口模块。

本模块为结果导出的"全局收口"，所有实验脚本必须
``from export_utils import ...``。涵盖：设备/种子/确定性、日志与算力记录、
wandb 接入、checkpoint 与模型卡、统计检验（Nadeau-Bengio + cluster bootstrap +
多比较校正）、LLH-GBID 多模型综合评价（第十五章）、中英双语图、一脚本一 xlsx、
数据集卡与 SHA-256。

运行方式:
    作为模块导入使用，不单独运行。

输出文件:
    本模块不直接产出正式结果，仅提供被各实验脚本调用的工具函数。
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import os
import platform
import random
import re
import string
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

# matplotlib.pyplot 只是把 Figure/Axes 当运行时属性挂着，没有 py.typed 级的类型再导出，
# `plt.Figure` 在注解里会被 mypy 判 name-defined（2026-07-27 迁移新模板时暴露）。
# 与模板版一致：注解直接引用 matplotlib.figure.Figure / matplotlib.axes.Axes。
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# numpy 数组类型别名（满足 mypy --strict 的 disallow_any_generics）
NDArray = npt.NDArray[Any]

# 2026-07-27 迁移新模板时补入：模板版 export_utils 的类型别名，供移植进来的
# _rank_ascending 等函数使用。016 原有代码一律沿用上面的 NDArray，不受影响。
FloatArray = npt.NDArray[np.floating[Any]]
"""浮点数组：效用值、GBID 分解量等一切实值向量/矩阵。"""

IntArray = npt.NDArray[np.integer[Any]]
"""整数数组：排名矩阵、维度分组索引等。"""

# ─── 路径与项目常量 ───────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # 02code/ → 项目根目录
DATA_DIR = os.path.join(BASE_DIR, "03data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "04outputs")
LOG_DIR = os.path.join(BASE_DIR, "05logs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "07checkpoints")

for _d in (OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR):
    os.makedirs(_d, exist_ok=True)

PROJECT_CODE = "016"

# 多种子重复（全局口径）。Formal 阶段强制全 5 种子。
RECOMMENDED_SEEDS: tuple[int, ...] = (20060515, 20041210, 19810915, 2023, 2024)
# Pilot dev 种子：仅用于 MPS 可行性验证，严禁混入 Formal 5 种子。
PILOT_SEED = 2004

# 真实处理后数据（由 01XX.py 写入，下游只读）。
ALL_ELEMENTS_FILE = os.path.join(PROCESSED_DIR, "all_9elements.csv")
APPLE_INDEX_FILE = os.path.join(PROCESSED_DIR, "apple_index.csv")
APPLE_LEVEL_2025_FILE = os.path.join(PROCESSED_DIR, "2025_apple_level.csv")

AVAILABLE_YEARS: tuple[int, ...] = (2018, 2019, 2025)
AVAILABLE_REGIONS: tuple[str, ...] = ("XJ", "SD", "SX", "GS")

_WAVELENGTH_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*nm")
_METADATA_COLUMNS: tuple[str, ...] = (
    "年份",
    "产地",
    "element_id",
    "苹果编号",
    "面位",
    "面位数",
    "SSC",
    "SSC含量",
    "SSC含量(°Brix)",
)

# 中国电网平均碳排放因子（kg CO2eq / kWh），用于算力碳足迹估算（§12.1）。
DEFAULT_GRID_CO2 = 0.581


# ═══════════════════════════════════════════════════════════════════════════════
# 一、设备、种子与确定性
# ═══════════════════════════════════════════════════════════════════════════════
def get_device(arg_device: str = "auto") -> torch.device:
    """统一设备解析：auto 优先 CUDA > MPS > CPU；亦可显式 cuda/mps/cpu。"""
    if arg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if arg_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg_device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def set_random_seed(seed: int, logger: Logger | None = None) -> None:
    """六路种子同步：PYTHONHASHSEED / random / numpy / torch / cuda / cudnn。

    CUDA 通路开启确定性算法；MPS/CPU 自动跳过 cudnn 设置并给出 WARN（§5.3）。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        msg = f"[WARN] 非 CUDA 设备（{_active_device_type()}）：跳过 cudnn 确定性设置，结果仅供 Pilot 复现。"
        if logger is not None:
            logger.log(msg)
        else:
            print(msg)


def _active_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def enforce_publication_grade_device(
    device: torch.device | str,
    stage: str,
    logger: Logger | None = None,
) -> None:
    """Formal 阶段 CUDA guard（§5.3，不可绕过）。

    stage == 'formal' 时设备必须为 CUDA，否则抛出 RuntimeError。
    stage == 'pilot' 时允许 MPS/CPU，仅记录 WARN。
    """
    dev_type = device.type if isinstance(device, torch.device) else str(device)
    stage_norm = stage.strip().lower()
    if stage_norm == "formal" and dev_type != "cuda":
        raise RuntimeError(
            f"Formal 阶段必须使用 CUDA，当前设备={dev_type}。"
            f"请在 GPU 节点复跑，或将 stage 显式声明为 'pilot'（结果须标注 <MPS_PILOT>）。"
        )
    if stage_norm == "pilot" and dev_type != "cuda":
        msg = f"[WARN] Pilot 阶段设备={dev_type}：产出仅供可行性验证，论文数字须标注 <MPS_PILOT> 或 <待CUDA复跑>。"
        if logger is not None:
            logger.log(msg)
        else:
            print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# 二、日志、环境信息与算力记录
# ═══════════════════════════════════════════════════════════════════════════════
class Logger:
    """同时向终端和带时间戳的 .log 文件写出日志（§3 命名规范）。"""

    def __init__(self, file_stem: str) -> None:
        ts = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
        # 同一秒启动的同名脚本会得到同一个文件名：以独占方式新建，已存在就依次加 _2、_3 … 后缀，
        # 两个进程不会写进同一个文件
        log_path = os.path.join(LOG_DIR, f"{file_stem}_{ts}.log")
        n = 1
        while True:
            try:
                self._log_file = open(log_path, "x", encoding="utf-8", buffering=1)  # noqa: SIM115  句柄随 Logger 生命周期常驻
                break
            except FileExistsError:
                n += 1
                log_path = os.path.join(LOG_DIR, f"{file_stem}_{ts}_{n}.log")
        self._stdout = sys.stdout
        self.log_path = log_path
        self.file_stem = file_stem
        start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"# 实验日志：{file_stem}  启动时间：{start}\n")

    def _write(self, msg: str) -> None:
        self._log_file.write(msg)
        self._log_file.flush()

    def log(self, *args: object, sep: str = " ", end: str = "\n") -> None:
        """写一行日志（同时输出到终端）。"""
        msg = sep.join(str(a) for a in args) + end
        self._stdout.write(msg)
        self._stdout.flush()
        self._write(msg)

    def close(self) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with contextlib.suppress(Exception):
            self._write(f"\n# 实验结束时间：{ts}\n")
            self._log_file.close()

    def __del__(self) -> None:
        self.close()


def get_logger(file_stem: str) -> Logger:
    """获取一个绑定到 05logs/<file_stem>_YY-MM-DD_HHMMSS.log 的日志实例。"""
    return Logger(file_stem)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def capture_env_info() -> dict[str, str]:
    """采集可复现性所需的环境信息（§5.1）。"""
    info: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "mps_available": str(torch.backends.mps.is_available()),
        "active_device": _active_device_type(),
        "git_commit": _git_commit(),
    }
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = str(torch.cuda.device_count())
    return info


def log_experiment_header(logger: Logger, header_dict: Mapping[str, object]) -> None:
    """统一日志头部，内部强制追加环境信息（§4.1 / §5.1）。"""
    logger.log("=" * 78)
    logger.log("实验配置")
    logger.log("-" * 78)
    for key, val in header_dict.items():
        logger.log(f"  {key}: {val}")
    logger.log("-" * 78)
    logger.log("环境信息")
    logger.log("-" * 78)
    for key, val in capture_env_info().items():
        logger.log(f"  {key}: {val}")
    logger.log("=" * 78)


@contextlib.contextmanager
def track_compute(
    stage_name: str,
    logger: Logger | None = None,
    wandb_run: Any | None = None,
    gpu_tdp_watts: float = 400.0,
    pue: float = 1.5,
    grid_co2_kg_per_kwh: float = DEFAULT_GRID_CO2,
) -> Iterator[dict[str, float]]:
    """算力上下文管理器（§12.1）：记录 wall-clock / GPU 峰值显存 / kWh / CO2eq。

    用法::

        with track_compute("主实验", logger, run) as budget:
            ...训练...
        # budget 内含 wall_clock_s / gpu_hours / peak_mem_gb / kwh / co2_kg
    """
    result: dict[str, float] = {}
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        wall = time.perf_counter() - t0
        peak_mem_gb = 0.0
        if use_cuda:
            with contextlib.suppress(Exception):
                peak_mem_gb = float(torch.cuda.max_memory_allocated()) / (1024**3)
        gpu_hours = wall / 3600.0 if use_cuda else 0.0
        kwh = (gpu_tdp_watts / 1000.0) * gpu_hours * pue
        co2_kg = kwh * grid_co2_kg_per_kwh
        result.update(
            {
                "wall_clock_s": round(wall, 3),
                "gpu_hours": round(gpu_hours, 6),
                "peak_mem_gb": round(peak_mem_gb, 4),
                "kwh": round(kwh, 6),
                "co2_kg": round(co2_kg, 6),
            }
        )
        line = (
            f"[算力] {stage_name}: wall={wall:.1f}s gpu_hours={gpu_hours:.4f} "
            f"peak_mem={peak_mem_gb:.3f}GB kWh={kwh:.5f} CO2={co2_kg:.5f}kg"
        )
        if logger is not None:
            logger.log(line)
        else:
            print(line)
        if wandb_run is not None:
            with contextlib.suppress(Exception):
                wandb_run.log({f"compute/{k}": v for k, v in result.items()})


def log_baseline_budget(
    method_name: str,
    budget: Mapping[str, float],
    hp_search_trials: int,
    seeds: Sequence[int],
    logger: Logger | None = None,
) -> dict[str, object]:
    """记录 baseline 算力预算（§7.4），供以 ``budget`` sheet 汇入主 xlsx。"""
    row: dict[str, object] = {
        "method": method_name,
        "hp_search_trials": hp_search_trials,
        "n_seeds": len(seeds),
        "seeds": ",".join(str(s) for s in seeds),
    }
    row.update({k: budget[k] for k in budget})
    if logger is not None:
        logger.log(f"[baseline budget] {row}")
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# 三、wandb 接入与 checkpoint / 模型卡
# ═══════════════════════════════════════════════════════════════════════════════
def init_wandb(
    file_path: str,
    seed: int,
    project: str,
    stage: str,
    config: Mapping[str, object],
) -> Any:
    """wandb 强制入口（§5.5）。run name=<脚本>_<YY-MM-DD_HHMMSS>_seed<X>。

    远端 project 名为 ``<project>-<stage>``。wandb 不可用时回退离线模式并返回 None。
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    ts = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
    run_name = f"{stem}_{ts}_seed{seed}"
    project_name = f"{project}-{stage}"
    try:
        import wandb
    except Exception:
        print(f"[WARN] 未安装 wandb，跳过实验跟踪：{run_name}")
        return None
    full_config = dict(config)
    full_config.update({"seed": seed, "stage": stage, "script": stem})
    full_config.update(capture_env_info())
    try:
        return wandb.init(project=project_name, name=run_name, config=full_config, reinit=True)
    except Exception:
        os.environ["WANDB_MODE"] = "offline"
        with contextlib.suppress(Exception):
            return wandb.init(project=project_name, name=run_name, config=full_config, reinit=True)
    return None


def _sanitize_run_name(name: str) -> str:
    """把 run_name 清成安全目录名（2026-07-27 随 save_checkpoint 一并补入）。

    016 原本没有此函数；模板版 save_sklearn_checkpoint 依赖它防止 run_name 里的
    路径分隔符逃逸出 07checkpoints/。只替换危险字符，不改变可读性。
    """
    safe = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", name).strip("._")
    return safe or "unnamed"


def _rank_ascending(values: FloatArray) -> IntArray:
    """并列感知的升序排名：GBID 越小名次越靠前，并列一律同取最小名次。

    §15.5 / §15.8 的「第一名频率」「平均名次」「排名区间」「成对胜率」全部建立在
    本函数上，因此它的并列语义直接决定印进论文表格的数字。

    **为什么不能用 `argsort(argsort(x)) + 1`**：该写法在出现并列时按数组下标强行
    分出先后——两个 GBID 完全相同的模型会被报成「第一名频率 100% / 0%」，系统性
    偏袒索引靠前者。消融变体与完整方法在某些指标上取值一致、或权重扰动恰好扫到
    等值点，都会触发这一情形，且不抛任何异常、不留任何痕迹。

    **为什么取 `method='min'` 而非 `'average'`**：并列第一必须两边都记为第 1 名，
    下游的 `(rankings == 1)` 才能同时命中；`'average'` 会把并列第一记成 1.5，
    使「第一名频率」塌成 0% / 0%，与 §15.8 表格的语义不符。代价是并列时平均名次
    偏乐观（记作 1, 1, 3 而非 1.5, 1.5, 3），报告平均名次时须知情。
    """
    from scipy.stats import rankdata

    return np.asarray(rankdata(np.asarray(values, dtype=float), method="min"), dtype=int)


def compute_sha256(path: str) -> str:
    """计算文件 SHA-256（§4.6 / §11.3）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_model_card(
    card_path: str,
    model_card_fields: Mapping[str, object],
    checkpoint_sha256: str,
) -> None:
    """写出模型卡 model_card.md（§11.3）。"""
    stage = str(model_card_fields.get("stage", "pilot")).lower()
    hardware = str(model_card_fields.get("training_hardware", _active_device_type()))
    determinism = str(model_card_fields.get("deterministic_guarantee", "partial"))
    lines = [
        f"# 模型卡：{model_card_fields.get('run_name', 'unknown')}",
        "",
        "## 训练硬件与确定性",
        f"- training_hardware: {hardware}",
        f"- deterministic_guarantee: {determinism}",
        f"- stage: {stage}",
        "",
        "## 权重与追溯",
        f"- checkpoint_sha256: {checkpoint_sha256}",
        f"- git_commit: {model_card_fields.get('git_commit', _git_commit())}",
        "",
        "## 其它字段",
    ]
    reserved = {"run_name", "training_hardware", "deterministic_guarantee", "stage", "git_commit"}
    for key, val in model_card_fields.items():
        if key not in reserved:
            lines.append(f"- {key}: {val}")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_best_checkpoint(
    model: torch.nn.Module,
    run_name: str,
    model_card_fields: Mapping[str, object],
    tag: str = "best",
    extra_state: Mapping[str, object] | None = None,
) -> str:
    """checkpoint 落盘（§5.6）：torch.save → SHA-256 → write_model_card 联动。

    Formal 模型卡必须 cuda+full+formal，否则拒绝落盘。严禁裸 torch.save 绕过本函数。
    extra_state 非空时落盘为 ``{"state_dict": ..., **extra_state}``（保留 feat_dim/n_bands 等重建元数据）。
    """
    stage = str(model_card_fields.get("stage", "pilot")).lower()
    hardware = str(model_card_fields.get("training_hardware", _active_device_type())).lower()
    if stage == "formal" and hardware != "cuda":
        raise RuntimeError(f"Formal 模型卡要求 training_hardware=cuda，当前={hardware}。拒绝落盘以防污染正式权重。")
    run_dir = os.path.join(CHECKPOINT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, f"{tag}.pt")
    if extra_state is None:
        torch.save(model.state_dict(), ckpt_path)
    else:
        torch.save({"state_dict": model.state_dict(), **extra_state}, ckpt_path)
    sha = compute_sha256(ckpt_path)
    fields = dict(model_card_fields)
    fields.setdefault("run_name", run_name)
    fields.setdefault("git_commit", _git_commit())
    write_model_card(os.path.join(run_dir, "model_card.md"), fields, sha)
    return ckpt_path


def save_sklearn_checkpoint(
    model: Any,
    run_name: str,
    model_card_fields: dict[str, Any],
    *,
    tag: str = "best",
    extra_state: dict[str, Any] | None = None,
) -> dict[str, str]:
    """非 torch 模型（sklearn / imblearn Pipeline / statsmodels 等）的 checkpoint 落盘入口。

    与 `save_best_checkpoint` 完全同权：单次调用完成 joblib.dump → SHA-256 →
    write_model_card，落到 `07checkpoints/<run_name>/<tag>.joblib`。

    **为什么必须有这个函数**（本条源自项目 020 的实战事故，回流至模板）：
    `save_best_checkpoint` 的签名写死 `torch.nn.Module`，而光谱 / 表格类课题的主力
    模型往往是 PLSR / SVR / RandomForest 这类 sklearn 估计器。缺了本通道的后果不是
    "换个方式存"，而是**整个项目一个权重都不存**——020 跑到投稿阶段 `07checkpoints/`
    仍然全空、零 model_card，第十一章 Item 3 整章空转，且没有任何报错。
    附带事故：模型缓存被随手 pickle 到了 `03data/processed/`，污染数据区。

    落盘格式用 `joblib` 而非 `pickle`：sklearn 官方推荐，对模型内部的大 numpy 数组
    做专门优化，体积与读写速度都显著优于裸 pickle。

    模型卡字段口径（`_MODELCARD_REQUIRED` 全部必填，与 torch 通道一致）：
        - `arch`   : 估计器类名 + 关键结构（如 "PLSRegression(n_components=12)"）
        - `params` : 可学习参数量；sklearn 填有意义的规模量（如 "coef_ shape=(215,3)"）
        - `flops`  : sklearn 估计器通常无意义，填 "N/A（sklearn 估计器，非逐层前向）"
        - `stage='formal'` 时同样强制 `training_hardware` 以 'cuda' 开头且
          `deterministic_guarantee='full'`。**纯 CPU 的 sklearn 主实验应标
          `stage='pilot'`，或在 `01PROJECT_PROTOCOL.md` B2 中具名声明该 stage
          不受 CUDA 强制**（第五章 Item 3 的 CUDA 强制针对深度模型，见 FORMAL_STAGES）。

    参数：
        model: 任意可被 joblib 序列化的估计器对象（sklearn Estimator / Pipeline / 自定义类）。
        run_name: 与 wandb run name / 日志前缀一致的字符串。
        model_card_fields: 模型卡字段；'ckpt_sha' 会被本函数覆盖。
        tag: "best" 或 "last"。
        extra_state: 可选附加状态（如折号 / 超参 / 训练集统计），与模型一起打包落盘。

    返回：{ckpt_path, sha256, run_dir, model_card_path}
    """
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("save_sklearn_checkpoint 需要 `pip install joblib`（sklearn 的官方序列化依赖）") from exc
    if tag not in ("best", "last"):
        raise ValueError(f"tag 必须是 'best' 或 'last'，收到: {tag!r}")
    run_dir = os.path.join(CHECKPOINT_DIR, _sanitize_run_name(run_name))
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, f"{tag}.joblib")
    payload: dict[str, Any] = {
        "model": model,
        "run_name": run_name,
        "tag": tag,
        "estimator_class": type(model).__name__,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    joblib.dump(payload, ckpt_path)
    sha256 = compute_sha256(ckpt_path)
    fields = dict(model_card_fields)
    fields.setdefault("run_name", run_name)
    fields.setdefault("git_commit", _git_commit())
    # 016 的 write_model_card 签名是 (card_path, fields, sha) → None，
    # 与模板版 (run_name, fields) → path 不同；此处按 016 的签名调用。
    model_card_path = os.path.join(run_dir, "model_card.md")
    write_model_card(model_card_path, fields, sha256)
    return {
        "ckpt_path": ckpt_path,
        "sha256": sha256,
        "run_dir": run_dir,
        "model_card_path": model_card_path,
    }


def save_checkpoint(
    model: Any,
    run_name: str,
    model_card_fields: dict[str, Any],
    *,
    tag: str = "best",
    **kwargs: Any,
) -> dict[str, str]:
    """checkpoint 落盘的**统一分发入口**——按模型类型自动选 torch / joblib 通道。

    脚本作者不需要记住"torch 走哪个、sklearn 走哪个"：一律调本函数。
    `torch.nn.Module` → `save_best_checkpoint`；其余一切 → `save_sklearn_checkpoint`。

    这层分发存在的意义是消除"选错通道就整个项目不存权重"这一失败模式（见
    `save_sklearn_checkpoint` 的事故说明）。两条下游通道各自的强校验（模型卡必填
    字段、Formal 阶段 cuda + full 校验）都不因走本入口而放松。

    `optimizer` 仅 torch 通道支持；传给 sklearn 通道会因签名不匹配直接 TypeError，
    这是有意的——它意味着调用方对模型类型的判断本身就错了。
    """
    if isinstance(model, torch.nn.Module):
        # 016 的 save_best_checkpoint 返回 str 路径且 tag 是位置参数（非 kw-only），
        # 与模板版返回 dict 不同；这里补齐成与 sklearn 通道一致的 dict 形状。
        ckpt_path = save_best_checkpoint(model, run_name, model_card_fields, tag, **kwargs)
        run_dir = os.path.dirname(ckpt_path)
        return {
            "ckpt_path": ckpt_path,
            "sha256": compute_sha256(ckpt_path),
            "run_dir": run_dir,
            "model_card_path": os.path.join(run_dir, "model_card.md"),
        }
    return save_sklearn_checkpoint(model, run_name, model_card_fields, tag=tag, **kwargs)


def n_distinct_seeds(seeds: Sequence[int] = RECOMMENDED_SEEDS) -> int:
    """上报用的种子数：去重后的数量。

    表格/正文里的 `n_seeds=` 一律取本函数，禁止写 `len(seeds)`。集合内允许重复
    （见 RECOMMENDED_SEEDS 说明），但重复的那一次不提供独立信息。
    """
    return len(set(seeds))


DEV_PILOT_SEED: int = 2004
"""Pilot 阶段单 dev 种子；严禁出现在 Formal 5 种子集合中。"""

# ─── Formal 阶段 stage 集合（第五章 Item 5 MPS 豁免子条款）─────────────────────
FORMAL_STAGES: frozenset[str] = frozenset(
    {
        "main_experiment",
        "baseline",
        "external_validation",
        "ablation_final",
        "negative_control",
        "sensitivity_main_figure",
    }
)
"""Formal 阶段 stage 白名单；enforce_publication_grade_device 用其判定是否强制 CUDA。"""


# ─── 路径管理 ─────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
CODE_DIR = os.path.join(BASE_DIR, "02code")
DATA_DIR = os.path.join(BASE_DIR, "03data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "04outputs")
LOG_DIR = os.path.join(BASE_DIR, "05logs")
DOC_DIR = os.path.join(BASE_DIR, "06doc")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "07checkpoints")
REFINE_LOGS_DIR = os.path.join(BASE_DIR, "refine-logs")
WIKI_DIR = os.path.join(BASE_DIR, "research-wiki")


# ═══════════════════════════════════════════════════════════════════════════════
# 四、绘图：中文字体、中英双语图、命名规范
# ═══════════════════════════════════════════════════════════════════════════════
def setup_chinese_font(serif: bool = False) -> None:
    """中英混排字体：西文比例字体 + 中文字体逐字回退（matplotlib >= 3.6）。

    关键：``font.family`` 必须是【具体字体名列表】。若写成泛型 "sans-serif"，
    matplotlib 的 _find_fonts_by_props 只会解析出单个字体，逐字回退（per-glyph
    fallback）失效——中文字体（如 SimHei）会用其自带的**半角等宽**拉丁字形去渲染
    英文与数字，图中英文遂显示为等宽体。仅调整 font.sans-serif 的顺序无法修复：
    把 Arial 提到最前会导致中文变豆腐块。

    serif=True  — 中文宋体 + 英文 Times New Roman（《农业工程学报》图件模板口径）
    serif=False — 中文黑体/思源黑体 + 英文 Arial/Helvetica（默认，用于汇报/预印）
    """
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    if serif:
        latin = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
        cjk = ["SimSun", "Source Han Serif CN", "STSong", "Songti SC", "Noto Serif CJK SC", "Arial Unicode MS"]
    else:
        latin = ["Arial", "Helvetica", "Helvetica Neue", "Liberation Sans", "DejaVu Sans"]
        cjk = [
            "Source Han Sans CN",  # 全字重；注意 "Source Han Sans SC" 在部分机器上只装了 Heavy
            "Noto Sans CJK SC",
            "PingFang SC",
            "Hiragino Sans GB",
            "Microsoft YaHei",
            "STHeiti",
            "Heiti SC",
            "Arial Unicode MS",
            "SimHei",  # 兜底：只保证不缺字，其拉丁字形为半角等宽，故置于末位
        ]
    stack = [f for f in latin if f in available] or [latin[-1]]
    stack += [f for f in cjk if f in available]
    plt.rcParams["font.family"] = stack  # 具体字体名列表 → 开启逐字回退
    plt.rcParams["font.sans-serif"] = stack  # 双保险，防第三方库改回泛型
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42  # PDF 内嵌 TrueType，文字可选可编辑
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def _index_to_letter(index: int) -> str:
    """0→a, 1→b, ..., 25→z, 26→aa。"""
    letters = string.ascii_lowercase
    result = ""
    n = index
    while True:
        result = letters[n % 26] + result
        n = n // 26 - 1
        if n < 0:
            break
    return result


def figure_output_path(file_stem: str, index: int, lang: str = "zh") -> str:
    """单语种图路径。zh→<stem>-图a.pdf；en→<stem>-fig-a.pdf（§4.2）。"""
    letter = _index_to_letter(index)
    if lang == "zh":
        filename = f"{file_stem}-图{letter}.pdf"
    elif lang == "en":
        filename = f"{file_stem}-fig-{letter}.pdf"
    else:
        raise ValueError(f"lang 仅支持 'zh' / 'en'，收到 {lang!r}")
    return os.path.join(OUTPUT_DIR, filename)


def figure_output_paths(file_stem: str, index: int) -> tuple[str, str]:
    """同时返回中英双路径 (zh_path, en_path)（§4.1）。"""
    return figure_output_path(file_stem, index, "zh"), figure_output_path(file_stem, index, "en")


def save_figure_bilingual(
    plot_fn: Callable[[str], Figure],
    file_stem: str,
    index: int,
    dpi: int = 300,
) -> tuple[str, str]:
    """高级封装（§4.1/§4.2）：plot_fn(lang) 按 lang∈{'zh','en'} 返回对应语言 Figure。

    两版数据/配色/字号/版式完全一致，差异仅限文本。返回 (zh_path, en_path)。
    """
    zh_path, en_path = figure_output_paths(file_stem, index)
    setup_chinese_font()  # 集中加载中文字体，避免各脚本遗漏导致 CJK 缺字
    for lang, path in (("zh", zh_path), ("en", en_path)):
        fig = plot_fn(lang)
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    return zh_path, en_path


# ═══════════════════════════════════════════════════════════════════════════════
# 五、一脚本一 xlsx
# ═══════════════════════════════════════════════════════════════════════════════
def table_sheet_name(index: int, title: str) -> str:
    """生成规范 Sheet 名 ``表a：标题``，清洗 Excel 非法字符并截断至 31 字符。"""
    letter = _index_to_letter(index)
    raw = f"表{letter}：{title}"
    cleaned = re.sub(r"[\[\]:*?/\\]", "_", raw)
    return cleaned[:31]


def _excel_output_path(file_stem: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{file_stem}.xlsx")


def write_script_workbook(file_path: str, sheets: Mapping[object, object]) -> str:
    """一脚本一 xlsx 唯一合法入口（§4.1/§4.2）。

    file_path 传 ``__file__``；sheets 支持两种写法::

        {0: ("标题", df), 1: ("标题2", df2)}   # 自动编号为 表a/表b
        {"compute": df, "ablation": df2}        # 直接用作 sheet 名

    所有 dataset×task×protocol×ablation 结果折叠为本 workbook 内多个 sheet。
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    path = _excel_output_path(stem)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for key, val in sheets.items():
            if isinstance(val, tuple):
                title, df = val[0], val[1]
                sheet_name = table_sheet_name(int(key), str(title)) if isinstance(key, int) else str(key)[:31]
            else:
                sheet_name = str(key)[:31]
                df = val
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 六、数据加载、波长解析与数据集卡
# ═══════════════════════════════════════════════════════════════════════════════
def parse_wavelength_from_column(col_name: str) -> float:
    """从列名解析波长，例如 ``591.19nm`` → 591.19；非波长列返回 nan。"""
    if not isinstance(col_name, str):
        return float("nan")
    match = _WAVELENGTH_PATTERN.search(col_name)
    return float(match.group(1)) if match else float("nan")


def get_spectrum_columns(df: pd.DataFrame) -> list[str]:
    """返回光谱波段列名（列名含 ``nm`` 者）。"""
    return [c for c in df.columns if _WAVELENGTH_PATTERN.search(str(c))]


def get_spectrum_wavelengths(df: pd.DataFrame, spectrum_cols: list[str] | None = None) -> NDArray:
    """返回光谱列对应波长数组。"""
    cols = spectrum_cols if spectrum_cols is not None else get_spectrum_columns(df)
    return np.array([parse_wavelength_from_column(c) for c in cols], dtype=np.float32)


def load_all_elements_df(verify_sha256: bool = True) -> pd.DataFrame:
    """读取 03data/processed/all_9elements.csv（下游只读），可选 SHA-256 核对（§4.6）。"""
    if not os.path.exists(ALL_ELEMENTS_FILE):
        raise FileNotFoundError(ALL_ELEMENTS_FILE)
    if verify_sha256:
        _verify_against_datasheet(ALL_ELEMENTS_FILE)
    return pd.read_csv(ALL_ELEMENTS_FILE)


def _datasheet_path(data_path: str) -> str:
    stem = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(PROCESSED_DIR, f"{stem}-datasheet.md")


def _verify_against_datasheet(data_path: str) -> None:
    """若存在数据集卡且记录了 SHA-256，则核对当前文件哈希（§4.6 前提）。"""
    sheet = _datasheet_path(data_path)
    if not os.path.exists(sheet):
        return
    with open(sheet, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"SHA-?256[^0-9a-fA-F]*([0-9a-fA-F]{64})", text)
    if m is None:
        return
    recorded = m.group(1).lower()
    actual = compute_sha256(data_path).lower()
    if recorded != actual:
        raise ValueError(
            f"数据 SHA-256 与数据集卡不一致：{os.path.basename(data_path)}\n"
            f"  记录值={recorded}\n  实际值={actual}\n数据可能被改动，下游读取已阻断。"
        )


def write_datasheet(
    data_path: str,
    source_and_license: str,
    granularity: str,
    sample_stats: Mapping[str, object],
    split_logic: str,
    known_bias: str,
    label_reliability: str,
    preprocessing_pointer: str,
    ethics: str,
) -> str:
    """生成数据集卡 03data/processed/<name>-datasheet.md（§4.6），含 SHA-256。"""
    sha = compute_sha256(data_path)
    name = os.path.basename(data_path)
    lines = [
        f"# 数据集卡：{name}",
        "",
        "## 数据来源与许可",
        source_and_license,
        "",
        "## 完整性校验",
        f"- SHA-256: {sha}",
        f"- 字节数: {os.path.getsize(data_path)}",
        "",
        "## 粒度与单位",
        granularity,
        "",
        "## 样本统计",
    ]
    lines.extend(f"- {k}: {v}" for k, v in sample_stats.items())
    lines.extend(
        [
            "",
            "## 划分逻辑",
            split_logic,
            "",
            "## 已知偏差",
            known_bias,
            "",
            "## 标签可靠性",
            label_reliability,
            "",
            "## 预处理流程指针",
            preprocessing_pointer,
            "",
            "## 伦理标签",
            ethics,
            "",
        ]
    )
    out = _datasheet_path(data_path)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 七、统计检验（Nadeau-Bengio + cluster bootstrap + 效应量 + 多比较校正）
# ═══════════════════════════════════════════════════════════════════════════════
def _cohens_d_paired(diff: NDArray) -> float:
    sd = float(np.std(diff, ddof=1))
    return float(np.mean(diff)) / sd if sd > 0 else 0.0


def _cliffs_delta(a: NDArray, b: NDArray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = int(np.sum(a[:, None] > b[None, :]))
    lt = int(np.sum(a[:, None] < b[None, :]))
    n = a.size * b.size
    return (gt - lt) / n if n > 0 else 0.0


def _cluster_bootstrap_ci(
    diff: NDArray,
    clusters: NDArray | None,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """cluster bootstrap CI（按聚类单位重采样，≥1000 次，§7.2）。"""
    diff = np.asarray(diff, dtype=float)
    if clusters is None:
        clusters = np.arange(diff.size)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        vals = np.concatenate([diff[clusters == c] for c in sampled])
        means[i] = float(np.mean(vals))
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def compare_methods(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_train: int | None = None,
    n_test: int | None = None,
    clusters: Sequence[int] | None = None,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 20060515,
) -> dict[str, float]:
    """配对比较两方法（§7.2）。

    默认 Nadeau-Bengio corrected resampled t-test（提供 n_train/n_test 时启用方差修正），
    否则退化为标准配对 t-test；附 cluster bootstrap CI、Cohen's d 与 Cliff's δ。

    返回字段：mean_a, mean_b, mean_diff, t_stat, p_value, ci_low, ci_high,
    cohens_d, cliffs_delta, corrected。
    """
    from scipy import stats as _stats  # 局部导入，避免无 scipy 时整模块失败

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"scores_a/scores_b 形状不一致：{a.shape} vs {b.shape}")
    diff = a - b
    j = diff.size
    mean_diff = float(np.mean(diff))
    var = float(np.var(diff, ddof=1)) if j > 1 else 0.0

    corrected = n_train is not None and n_test is not None
    if n_train is not None and n_test is not None and var > 0:
        factor = (1.0 / j) + (n_test / n_train)
        t_stat = mean_diff / float(np.sqrt(factor * var))
    elif var > 0:
        t_stat = mean_diff / float(np.sqrt(var / j))
    else:
        t_stat = 0.0
    df = max(j - 1, 1)
    p_value = float(2.0 * _stats.t.sf(abs(t_stat), df)) if var > 0 else 1.0

    rng = np.random.default_rng(seed)
    clusters_arr = None if clusters is None else np.asarray(clusters)
    ci_low, ci_high = _cluster_bootstrap_ci(diff, clusters_arr, n_bootstrap, alpha, rng)

    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": mean_diff,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_d": _cohens_d_paired(diff),
        "cliffs_delta": _cliffs_delta(a, b),
        "corrected": 1.0 if corrected else 0.0,
    }


def apply_multi_comp_correction(
    p_values: Sequence[float],
    method: str = "holm",
    alpha: float = 0.05,
) -> dict[str, list[float]]:
    """多比较校正（§7.2），method ∈ {'holm', 'bh'}。返回校正后 p 值与 reject 标记。"""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    method_norm = method.strip().lower()
    if method_norm == "holm":
        running = 0.0
        for rank, idx in enumerate(order):
            val = (m - rank) * p[idx]
            running = max(running, val)
            adjusted[idx] = min(running, 1.0)
    elif method_norm in ("bh", "fdr"):
        running = 1.0
        for rank in range(m - 1, -1, -1):
            idx = order[rank]
            val = p[idx] * m / (rank + 1)
            running = min(running, val)
            adjusted[idx] = min(running, 1.0)
    else:
        raise ValueError(f"method 仅支持 'holm' / 'bh'，收到 {method!r}")
    reject = [bool(x < alpha) for x in adjusted]
    return {"p_adjusted": [float(x) for x in adjusted], "reject": [float(x) for x in reject]}


def annotate_significance(
    ax: Axes,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    *,
    height: float = 0.02,
) -> None:
    """图表显著性标注（§8.2）：* p<0.05, ** p<0.01, *** p<0.001, ns。"""
    if p_value < 0.001:
        star = "***"
    elif p_value < 0.01:
        star = "**"
    elif p_value < 0.05:
        star = "*"
    else:
        star = "ns"
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=1.2, c="black")
    ax.text((x1 + x2) / 2, y + height, star, ha="center", va="bottom")


# ═══════════════════════════════════════════════════════════════════════════════
# 八、LLH-GBID 多模型综合评价（第十五章 CRITICAL）
# ═══════════════════════════════════════════════════════════════════════════════
def compute_utility(
    x: float,
    ideal: float,
    baseline: float,
    direction: str = "benefit",
) -> float:
    """单指标效用化（§15.3）：u = clip((x - L)/(I - L), 0, 1)。

    direction='benefit' 时 I>L；'cost' 时 I<L（理想小、基准大）。
    """
    if direction not in ("benefit", "cost"):
        raise ValueError(f"direction 仅支持 'benefit' / 'cost'，收到 {direction!r}")
    denom = ideal - baseline
    if denom == 0:
        raise ValueError("ideal 与 baseline 不能相等（效用化分母为 0）")
    u = (x - baseline) / denom
    return float(min(max(u, 0.0), 1.0))


def compute_target_utility(
    x: float,
    target: float,
    tau_minus: float,
    tau_plus: float,
    delta_minus: float = 0.0,
    delta_plus: float = 0.0,
) -> float:
    """目标型指标效用（§15.3）：统一非对称带死区形式。

    [T-δ⁻, T+δ⁺] 内 u=1；之外按 τ⁻/τ⁺ 线性衰减并截断到 [0,1]。
    """
    if tau_minus <= 0 or tau_plus <= 0:
        raise ValueError("tau_minus / tau_plus 必须为正")
    lo, hi = target - delta_minus, target + delta_plus
    if lo <= x <= hi:
        return 1.0
    u = 1.0 - ((x - hi) / tau_plus if x > hi else (lo - x) / tau_minus)
    return float(min(max(u, 0.0), 1.0))


def compute_gbid(
    u: Sequence[float],
    weights_dim: Sequence[float],
    weights_within: Sequence[Sequence[float]],
    dim_groups: Sequence[Sequence[int]],
    q_b: float = 2.0,
    q_w: float = 2.0,
) -> dict[str, Any]:
    """GBID 主计算 + 分解（§15.2）。

    参数:
        u: 各指标效用值 u_j（长度 = 指标数）
        weights_dim: 维度权重 W_d（长度 = 维度数，自动归一化）
        weights_within: 每维度内指标权重 v_{j|d}（与 dim_groups 对齐，自动归一化）
        dim_groups: 每维度包含的指标在 u 中的下标
        q_b, q_w: 维度间/维度内阶数；主分析强制 (2,2)

    返回: gbid, gbids, u_bar, C_B, C_W, D_d（各维度缺口距离）。
    q_b=q_w=2 时满足 gbid² = (1-ū)² + C_B² + C_W²。
    """
    u_arr = np.asarray(u, dtype=float)
    d = 1.0 - u_arr  # 指标缺口 d_j
    w_dim = np.asarray(weights_dim, dtype=float)
    w_dim = w_dim / w_dim.sum()
    n_dim = len(dim_groups)
    if len(w_dim) != n_dim:
        raise ValueError("weights_dim 长度须等于维度数")

    d_dist = np.empty(n_dim, dtype=float)  # D_d^{(q_W)}
    dbar_dim = np.empty(n_dim, dtype=float)  # 维度内线性加权平均缺口
    for di, (group, v_raw) in enumerate(zip(dim_groups, weights_within, strict=True)):
        v = np.asarray(v_raw, dtype=float)
        v = v / v.sum()
        dj = d[list(group)]
        d_dist[di] = float(np.power(np.sum(v * np.power(dj, q_w)), 1.0 / q_w))
        dbar_dim[di] = float(np.sum(v * dj))

    gbid = float(np.power(np.sum(w_dim * np.power(d_dist, q_b)), 1.0 / q_b))
    gbids = 100.0 * (1.0 - gbid)

    # 分解（基于线性加权量，q=2 时精确成立）
    dbar = float(np.sum(w_dim * dbar_dim))  # 平均缺口 = 1 - ū
    u_bar = 1.0 - dbar
    c_b = float(np.sqrt(max(np.sum(w_dim * (dbar_dim - dbar) ** 2), 0.0)))
    c_w_sq = 0.0
    for di, (group, v_raw) in enumerate(zip(dim_groups, weights_within, strict=True)):
        v = np.asarray(v_raw, dtype=float)
        v = v / v.sum()
        dj = d[list(group)]
        c_w_sq += w_dim[di] * float(np.sum(v * (dj - dbar_dim[di]) ** 2))
    c_w = float(np.sqrt(max(c_w_sq, 0.0)))

    return {
        "gbid": gbid,
        "gbids": gbids,
        "u_bar": u_bar,
        "C_B": c_b,
        "C_W": c_w,
        "D_d": [float(x) for x in d_dist],
    }


def gbid_sensitivity_q(
    u: Sequence[float],
    weights_dim: Sequence[float],
    weights_within: Sequence[Sequence[float]],
    dim_groups: Sequence[Sequence[int]],
    q_grid: Sequence[tuple[float, float]] = ((2, 2), (3, 2), (2, 3), (3, 3)),
) -> dict[str, float]:
    """短板敏感性（§15.5）：不同 (q_B,q_W) 下的 GBIDS。"""
    out: dict[str, float] = {}
    for q_b, q_w in q_grid:
        res = compute_gbid(u, weights_dim, weights_within, dim_groups, q_b=q_b, q_w=q_w)
        out[f"GBIDS_qB{q_b}_qW{q_w}"] = float(res["gbids"])
    return out


def gbid_sensitivity_weights(
    u_by_model: Mapping[str, Sequence[float]],
    weights_dim: Sequence[float],
    weights_within: Sequence[Sequence[float]],
    dim_groups: Sequence[Sequence[int]],
    n_draws: int = 1000,
    seed: int = 20060515,
) -> dict[str, dict[str, float]]:
    """权重敏感性（§15.5）：两级 Dirichlet 随机权重下各模型第一名概率与平均名次。"""
    rng = np.random.default_rng(seed)
    models = list(u_by_model.keys())
    n_dim = len(dim_groups)
    first_cnt = dict.fromkeys(models, 0.0)
    rank_sum = dict.fromkeys(models, 0.0)
    base_w = np.asarray(weights_dim, dtype=float)
    base_w = base_w / base_w.sum()
    for _ in range(n_draws):
        w_dim_list = rng.dirichlet(base_w * n_dim).tolist()
        w_within_list = [rng.dirichlet(np.ones(len(g))).tolist() for g in dim_groups]
        gbids = {
            mdl: float(compute_gbid(u_by_model[mdl], w_dim_list, w_within_list, dim_groups)["gbids"]) for mdl in models
        }
        ranked = sorted(models, key=lambda m: gbids[m], reverse=True)
        first_cnt[ranked[0]] += 1.0
        for r, mdl in enumerate(ranked):
            rank_sum[mdl] += r + 1
    return {
        mdl: {
            "first_place_prob": first_cnt[mdl] / n_draws,
            "mean_rank": rank_sum[mdl] / n_draws,
        }
        for mdl in models
    }


def gbid_bootstrap(
    u_reps: Sequence[Sequence[float]],
    weights_dim: Sequence[float],
    weights_within: Sequence[Sequence[float]],
    dim_groups: Sequence[Sequence[int]],
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 20060515,
) -> dict[str, float]:
    """重采样统计推断（§15.6）：R 次重复的 GBIDS 均值、SD 与 percentile bootstrap CI。"""
    rng = np.random.default_rng(seed)
    gbids = np.array(
        [float(compute_gbid(u, weights_dim, weights_within, dim_groups)["gbids"]) for u in u_reps],
        dtype=float,
    )
    r = gbids.size
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, r, size=r)
        boot_means[i] = float(np.mean(gbids[idx]))
    return {
        "gbids_mean": float(np.mean(gbids)),
        "gbids_sd": float(np.std(gbids, ddof=1)) if r > 1 else 0.0,
        "ci_low": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "n_reps": float(r),
    }


def gbid_paper_paragraph(model_name: str, gbid_result: Mapping[str, Any], passed: bool) -> str:
    """论文中文表述自动生成（§15.10），含红线措辞约束（§15.9）。"""
    gbids = float(gbid_result["gbids"])
    u_bar = float(gbid_result["u_bar"])
    c_b = float(gbid_result["C_B"])
    c_w = float(gbid_result["C_W"])
    gate = "通过非补偿性准入门槛" if passed else "未通过非补偿性准入门槛（禁止标注为综合最优）"
    return (
        f"模型 {model_name} 的 0—100 综合指数得分（GBIDS）为 {gbids:.2f}，"
        f"平均效用 ū={u_bar:.3f}，维度间不均衡 C_B={c_b:.3f}，维度内不均衡 C_W={c_w:.3f}；"
        f"该模型{gate}。综合指数为条件相对评价，依赖既定指标体系/权重/基准设定，"
        f"不构成跨研究或跨数据集的绝对可比排序。"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 九、依赖锁定
# ═══════════════════════════════════════════════════════════════════════════════
def freeze_requirements() -> str:
    """将当前环境依赖冻结到 08github/requirements.txt（§5.4）。"""
    req_path = os.path.join(BASE_DIR, "08github", "requirements.txt")
    os.makedirs(os.path.dirname(req_path), exist_ok=True)
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except Exception as exc:
        print(f"[WARN] freeze_requirements 失败：{exc}")
        return req_path
    header = (
        "# requirements.txt — 由 export_utils.freeze_requirements 自动锁定（§5.4）\n"
        f"# 冻结时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# git_commit：{_git_commit()}\n"
    )
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(header + out.stdout)
    return req_path
