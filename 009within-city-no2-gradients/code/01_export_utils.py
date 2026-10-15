"""
01_export_utils.py: central management of the project's data, figures, tables, logs and device selection

Usage:
    python code/01_export_utils.py

Running this module only creates outputs/ and logs/ if they are missing and prints the directories; it writes no
file. Paths returned by the helpers for the stem "example" (图a = figure a):
    outputs/example-图a.pdf  — figure_output_path("example", 1)
    outputs/example.xlsx      — table_output_path("example"); write_excel_workbook() writes it and prefixes each
                                sheet name with its position (01_, 02_, …)
    logs/example.log          — log_output_path("example"); configure_file_logger() writes it
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from playwright.sync_api import Page

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
DOC_DIR = PROJECT_ROOT / "docs"


def ensure_project_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def data_path(filename: str) -> Path:
    return DATA_DIR / filename


def output_path(filename: str) -> Path:
    ensure_project_dirs()
    return OUTPUT_DIR / filename


def log_output_path(file_stem: str) -> Path:
    ensure_project_dirs()
    return LOG_DIR / f"{sanitize_name(file_stem)}.log"


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()
    return sanitized or "output"


def figure_output_path(file_stem: str, index: int | str, suffix: str = "pdf") -> Path:
    ensure_project_dirs()
    letter = chr(ord("a") + index - 1) if isinstance(index, int) else str(index)
    return OUTPUT_DIR / f"{sanitize_name(file_stem)}-图{letter}.{suffix}"


def table_output_path(file_stem: str, suffix: str = "xlsx") -> Path:
    ensure_project_dirs()
    return OUTPUT_DIR / f"{sanitize_name(file_stem)}.{suffix}"


def table_sheet_name(index: int, title: str) -> str:
    prefix = f"{index:02d}_"
    clean_title = re.sub(r"[:\\\\/*?\\[\\]]+", "_", title).strip()
    return (prefix + clean_title)[:31]


def write_excel_workbook(file_stem: str, sheets: Iterable[tuple[str, pd.DataFrame]]) -> Path:
    output_file = table_output_path(file_stem, suffix="xlsx")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for index, (title, dataframe) in enumerate(sheets, start=1):
            dataframe.to_excel(
                writer,
                sheet_name=table_sheet_name(index, title),
                index=False,
            )
    return output_file


def setup_chinese_font() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "PingFang SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def locate_local_browser_executable(browser_executable: str | None = None) -> str:
    candidates: list[str] = []
    if browser_executable:
        candidates.append(browser_executable)

    env_browser = os.getenv("PLAYWRIGHT_BROWSER_EXECUTABLE")
    if env_browser:
        candidates.append(env_browser)

    candidates.extend(
        [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]
    )

    checked_paths: list[str] = []
    for candidate in candidates:
        candidate_path = Path(candidate).expanduser()
        checked_paths.append(str(candidate_path))
        if candidate_path.exists():
            return str(candidate_path)

    checked_display = "\n".join(f"  - {path}" for path in checked_paths)
    raise FileNotFoundError(
        "未找到可用的本机 Chrome / Chromium / Edge 浏览器可执行文件。\n"
        "请安装浏览器，或设置环境变量 PLAYWRIGHT_BROWSER_EXECUTABLE。\n"
        f"已检查路径:\n{checked_display}"
    )


def build_playwright_launch_kwargs(
    headless: bool = True,
    browser_executable: str | None = None,
) -> tuple[dict[str, object], str]:
    executable = locate_local_browser_executable(browser_executable=browser_executable)
    return {"headless": headless, "executable_path": executable}, executable


def page_dsf_http_get(page: Page, url: str, params: dict[str, Any] | None = None) -> Any:
    return page.evaluate(
        """async ({ url, params }) => {
            if (!window.dsf || !window.dsf.http || !window.dsf.http.get) {
                throw new Error("window.dsf.http.get 不可用。");
            }
            const response = await window.dsf.http.get(url, params || {});
            return response.data;
        }""",
        {"url": url, "params": params or {}},
    )


def get_device(device_str: str = "auto") -> str:
    device_str = (device_str or "auto").lower()
    if device_str in {"cpu", "cuda", "mps"}:
        return device_str

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def configure_file_logger(file_stem: str, logger_name: str | None = None) -> tuple[logging.Logger, Path]:
    ensure_project_dirs()
    log_path = log_output_path(file_stem)
    logger = logging.getLogger(logger_name or sanitize_name(file_stem))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger, log_path


if __name__ == "__main__":
    ensure_project_dirs()
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"日志目录: {LOG_DIR}")
