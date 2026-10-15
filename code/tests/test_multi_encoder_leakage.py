"""74_multi_encoder_leakage 训练池构造单元测试（CPU 快速，合成数据）。"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "multi_encoder_leakage",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "74_multi_encoder_leakage.py"),
)
assert _SPEC is not None
assert _SPEC.loader is not None
_ME: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ME)


def _doms(seed: int = 0) -> dict[str, tuple[Any, Any]]:
    """三个域、连续目标（按源域三分位分类）。"""
    rng = np.random.default_rng(seed)
    out = {}
    for i, d in enumerate(["A", "B", "C"]):
        n = 60 + 10 * i
        y = rng.uniform(0, 10, size=n)
        x = rng.normal(size=(n, 30)) + y[:, None] * 0.1 + i
        out[d] = (x, y)
    return out


def test_fold_rows_align_with_prepare_fold() -> None:
    doms = _doms()
    got = _ME.fold_arrays(doms, True, "B", 5)
    assert got is not None
    fa, dom_of_row = got
    assert len(dom_of_row) == len(fa.yall)
    assert np.array_equal(fa.xall[dom_of_row == "B"], fa.xh)
    assert np.array_equal(fa.xall[dom_of_row != "B"], fa.xs)


def test_shuffled_pool_permutes_only_held_labels() -> None:
    doms = _doms()
    fa, dom_of_row = _ME.fold_arrays(doms, True, "C", 5)
    x, y = _ME.shuffled_pool(fa, dom_of_row, "C", 2024)
    held = dom_of_row == "C"
    assert np.array_equal(x, fa.xall)
    assert np.array_equal(y[~held], fa.yall[~held])
    assert sorted(y[held].tolist()) == sorted(fa.yall[held].tolist())
    assert not np.array_equal(y[held], fa.yall[held])


def test_half_split_disjoint_and_complete() -> None:
    doms = _doms()
    fa, dom_of_row = _ME.fold_arrays(doms, True, "A", 5)
    x, y, b_idx = _ME.half_split(fa, dom_of_row, "A", 7)
    n_src = int((dom_of_row != "A").sum())
    n_a = len(y) - n_src
    assert n_a + len(b_idx) == len(fa.yh)
    # B 半里的样本不在训练池里
    train_rows = {tuple(r) for r in x}
    assert all(tuple(fa.xh[i]) not in train_rows for i in b_idx)
    for c in fa.classes:
        n_c = int((fa.yh == c).sum())
        assert int((fa.yh[b_idx] == c).sum()) == n_c - n_c // 2
