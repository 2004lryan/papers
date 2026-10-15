"""export_utils 单元测试：纯函数与 LLH-GBID 数学一致性。"""

import datetime
import os
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

import export_utils as eu


def test_compute_utility_benefit() -> None:
    assert eu.compute_utility(0.5, ideal=1.0, baseline=0.0, direction="benefit") == 0.5
    assert eu.compute_utility(2.0, ideal=1.0, baseline=0.0) == 1.0  # 上截断
    assert eu.compute_utility(-1.0, ideal=1.0, baseline=0.0) == 0.0  # 下截断


def test_compute_utility_cost() -> None:
    # 成本型：理想小、基准大（如 ECE 理想=0，基准=0.05）
    assert eu.compute_utility(0.0, ideal=0.0, baseline=0.05, direction="cost") == 1.0
    assert eu.compute_utility(0.05, ideal=0.0, baseline=0.05, direction="cost") == 0.0


def test_target_utility_deadzone() -> None:
    assert eu.compute_target_utility(0.0, target=0.0, tau_minus=1.0, tau_plus=1.0) == 1.0
    assert eu.compute_target_utility(0.5, target=0.0, tau_minus=1.0, tau_plus=1.0) == 0.5


def test_gbid_perfect() -> None:
    res = eu.compute_gbid([1.0, 1.0, 1.0], [1.0, 1.0], [[1.0], [1.0, 1.0]], [[0], [1, 2]])
    assert abs(float(res["gbids"]) - 100.0) < 1e-9
    assert abs(float(res["gbid"])) < 1e-9


def test_gbid_decomposition_identity() -> None:
    # q_B=q_W=2 时 gbid² = (1-ū)² + C_B² + C_W²（§15.2 必报分解）
    u = [0.9, 0.6, 0.3, 0.8]
    res = eu.compute_gbid(u, [0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]], [[0, 1], [2, 3]], q_b=2.0, q_w=2.0)
    lhs = float(res["gbid"]) ** 2
    rhs = (1.0 - float(res["u_bar"])) ** 2 + float(res["C_B"]) ** 2 + float(res["C_W"]) ** 2
    assert abs(lhs - rhs) < 1e-9


def test_gbid_sensitivity_q_keys() -> None:
    out = eu.gbid_sensitivity_q([0.8, 0.6, 0.7], [1.0, 1.0], [[1.0], [1.0, 1.0]], [[0], [1, 2]])
    assert len(out) == 4


def test_multi_comp_holm_range() -> None:
    out = eu.apply_multi_comp_correction([0.01, 0.02, 0.03], method="holm")
    assert all(0.0 <= x <= 1.0 for x in out["p_adjusted"])


def test_multi_comp_bh_range() -> None:
    out = eu.apply_multi_comp_correction([0.01, 0.2, 0.5], method="bh")
    assert all(0.0 <= x <= 1.0 for x in out["p_adjusted"])


def test_sha256_length() -> None:
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(b"apple-origin")
        path = f.name
    try:
        assert len(eu.compute_sha256(path)) == 64
    finally:
        os.remove(path)


def test_figure_naming_bilingual() -> None:
    zh, en = eu.figure_output_paths("05_demo", 0)
    assert zh.endswith("05_demo-图a.pdf")
    assert en.endswith("05_demo-fig-a.pdf")


def test_sheet_name_limit() -> None:
    name = eu.table_sheet_name(0, "x" * 50)
    assert len(name) <= 31


def test_set_seed_reproducible() -> None:
    eu.set_random_seed(20060515)
    a = np.random.rand(5)
    eu.set_random_seed(20060515)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_compare_methods_identical() -> None:
    res = eu.compare_methods([0.80, 0.81, 0.79], [0.80, 0.81, 0.79], n_bootstrap=200)
    assert abs(res["mean_diff"]) < 1e-12
    assert res["p_value"] >= 0.0


def test_logger_same_second_gets_distinct_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz: datetime.tzinfo | None = None) -> "_FixedNow":
            return cls(2026, 9, 26, 6, 15, 43)

    monkeypatch.setattr(eu, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(eu, "datetime", types.SimpleNamespace(datetime=_FixedNow))
    a = eu.Logger("demo")
    b = eu.Logger("demo")
    a.log("from a")
    b.log("from b")
    a.close()
    b.close()
    assert sorted(os.listdir(tmp_path)) == ["demo_26-09-26_061543.log", "demo_26-09-26_061543_2.log"]
    first = (tmp_path / "demo_26-09-26_061543.log").read_text(encoding="utf-8")
    second = (tmp_path / "demo_26-09-26_061543_2.log").read_text(encoding="utf-8")
    assert "from a" in first
    assert "from b" not in first
    assert "from b" in second
    assert "from a" not in second
