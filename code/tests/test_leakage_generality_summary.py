"""75_leakage_generality_summary 的跨数据集推断单元测试（合成数据，CPU 快速）。"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

m75 = importlib.import_module("75_leakage_generality_summary")


def test_infer_skips_mixed_model_with_too_few_datasets() -> None:
    res = m75.infer(pd.Series([0.10, 0.20, 0.15]), pd.Series(["a", "a", "b"]), n_boot=200)
    assert "不拟合" in res["混合效应"]
    assert "混合效应截距" not in res
    assert res["为正留出域"] == "3/3"
    assert res["数据集数"] == 2


def test_infer_balanced_intercept_equals_grand_mean() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(list("abcdef"), 3)
    ds_effect = dict(zip("abcdef", rng.normal(0.0, 0.05, 6), strict=True))
    y = 0.2 + np.array([ds_effect[g] for g in groups]) + rng.normal(0.0, 0.02, len(groups))
    res = m75.infer(pd.Series(y), pd.Series(groups), n_boot=500)
    # 各数据集留出域数相同，GLS 截距等于总均值
    assert abs(res["混合效应截距"] - float(y.mean())) < 1e-6
    assert res["截距95%下限"] > 0
    assert res["簇自助95%下限"] > 0
    assert res["为正数据集"] == "6/6"
    assert abs(res["数据集级符号检验p"] - 0.5**6) < 1e-12
    assert isinstance(res["混合效应"], str)


def test_fit_note_categories() -> None:
    assert m75._fit_note([], True, 0.01) == "正常"
    note = m75._fit_note(["The MLE may be on the boundary of the parameter space."], True, float("nan"))
    assert "边界" in note
    assert "SE 无效" in note
    assert "未完全收敛" in m75._fit_note([], False, 0.01)
