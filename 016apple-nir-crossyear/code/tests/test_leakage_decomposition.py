"""31_leakage_decomposition + crossdataset_lib 单元测试（CPU 快速，无需数据文件/GPU）。

覆盖：SNV/三分位/MMD 纯函数、prepare_fold 类选择与形状、掩码谱段生成、掩码自监督解码头前向、
masked_ssm_pretrain 在合成数据上产出冻结 ProtoEncoder（隔离 Δ_unsup 的无标签表示学习路径）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch

import crossdataset_lib as cdl

NDArray = npt.NDArray[Any]


def test_snv_zero_mean_unit_std() -> None:
    x = np.random.default_rng(0).normal(5.0, 3.0, size=(4, 50))
    out = cdl.snv(x)
    assert np.allclose(out.mean(1), 0.0, atol=1e-6)
    assert np.allclose(out.std(1), 1.0, atol=1e-3)


def test_tertile_labels_thresholds_from_source_only() -> None:
    y_src = np.arange(90, dtype=float)
    lab = cdl.tertile_labels(y_src, np.array([0.0, 45.0, 89.0]))
    assert lab.tolist() == [0, 1, 2]  # 三分位映射到 0/1/2


def test_mmd2_linear_zero_for_identical() -> None:
    a = np.ones((10, 8))
    assert cdl.mmd2_linear(a, a.copy()) == 0.0
    assert cdl.mmd2_linear(a, a + 1.0) > 0.0


def _toy_doms(seed: int = 0) -> dict[str, tuple[NDArray, NDArray]]:
    rng = np.random.default_rng(seed)
    doms = {}
    for d in ["A", "B", "C"]:
        x = rng.normal(0, 1, size=(120, 30))
        y = rng.integers(0, 4, size=120)  # 4 类
        doms[d] = (x, y)
    return doms


def test_prepare_fold_shapes_and_classes() -> None:
    fa = cdl.prepare_fold(_toy_doms(), is_reg=False, held="C", k=5)
    assert fa is not None
    assert fa.in_dim == 30
    assert len(fa.classes) >= 3
    assert fa.xall.shape[0] == fa.xs.shape[0] + fa.xh.shape[0]  # 源+目标=全量
    assert fa.xh.shape[1] == 30


def test_prepare_fold_none_when_too_few_classes() -> None:
    rng = np.random.default_rng(1)
    doms: dict[str, tuple[NDArray, NDArray]] = {
        d: (rng.normal(0, 1, (40, 10)), np.zeros(40, dtype=int)) for d in ["A", "B"]
    }  # 仅 1 类
    assert cdl.prepare_fold(doms, is_reg=False, held="B", k=5) is None


def test_span_mask_shape_and_binary() -> None:
    rng = np.random.default_rng(2)
    m = cdl.span_mask(6, 40, mask_ratio=0.30, span=5, rng=rng)
    assert m.shape == (6, 40)
    assert set(np.unique(m)).issubset({0.0, 1.0})
    assert m.sum() > 0  # 确有掩码


def test_spectral_decoder_forward_shape() -> None:
    dec = cdl.SpectralDecoder(64, 30)
    z = torch.randn(8, 64)
    assert dec(z).shape == (8, 30)


def test_masked_ssm_pretrain_returns_frozen_encoder() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, size=(80, 30)).astype(np.float64)
    enc = cdl.masked_ssm_pretrain(x, 30, torch.device("cpu"), epochs=3, seed=2004)
    assert not enc.training  # eval 模式（冻结）
    with torch.no_grad():
        feat = enc(torch.from_numpy(x[:5]).float())
    assert feat.shape == (5, 64)
    assert np.allclose(np.linalg.norm(feat.numpy(), axis=1), 1.0, atol=1e-4)  # L2 归一化


def test_apply_preprocess_modes() -> None:
    x = np.array([[3.0, 4.0], [1.0, 0.0]])
    l2 = cdl._apply_preprocess(x, "l2")
    assert np.allclose(np.linalg.norm(l2, axis=1), 1.0)
    assert np.array_equal(cdl._apply_preprocess(x, "none"), x)


def _decomp_doms(seed: int = 0) -> dict[str, tuple[NDArray, NDArray]]:
    rng = np.random.default_rng(seed)
    doms = {}
    for j, d in enumerate(["A", "B", "C"]):
        x = rng.normal(j * 0.2, 1.0, size=(150, 24))  # 轻微域偏移
        y = rng.integers(0, 4, size=150)
        doms[d] = (x, y)
    return doms


def test_run_fold_decomp_snv_returns_four_arms() -> None:
    rec = cdl.run_fold_decomp(
        _decomp_doms(), False, "C", torch.device("cpu"), episodes=20, ssm_epochs=3, reps=3, k=5, seed=2004
    )
    assert rec is not None
    for key in ("e0_sup_fold_restricted", "e2_label_aware", "u_src_selfsup_fold_restricted", "u_tgt_selfsup_exposure"):
        assert np.isfinite(rec[key])
    assert abs(rec["delta_sup"] - (rec["e2_label_aware"] - rec["e0_sup_fold_restricted"])) < 1e-9
    assert abs(rec["delta_unsup"] - (rec["u_tgt_selfsup_exposure"] - rec["u_src_selfsup_fold_restricted"])) < 1e-9


def test_robust_intercept_ci_normal() -> None:
    vals = np.array([0.1, 0.12, 0.09, 0.2, 0.22, 0.18])
    grp = np.array(["a", "a", "a", "b", "b", "b"])
    r = cdl.robust_intercept_ci(vals, grp, n_boot=500)
    assert 0.05 < r["intercept"] < 0.25
    assert r["ci_low"] <= r["intercept"] <= r["ci_high"]
    assert r["method"] in {"mixedlm", "cluster_bootstrap"}


def test_robust_intercept_ci_singular_falls_back() -> None:
    # 每组内零方差 + 组间零方差 → MixedLM 奇异，应回退簇自助不崩
    vals = np.array([0.1, 0.1, 0.1, 0.1])
    grp = np.array(["a", "a", "b", "b"])
    r = cdl.robust_intercept_ci(vals, grp, n_boot=200)
    assert abs(r["intercept"] - 0.1) < 1e-9
    assert np.isfinite(r["ci_low"])
    assert np.isfinite(r["ci_high"])


def test_run_fold_decomp_l2_features_path() -> None:
    # 模拟 foundation 特征（高维、无谱结构）走 l2 预处理路径（32 的核心）
    rng = np.random.default_rng(1)
    doms = {d: (rng.normal(0, 1, (120, 64)), rng.integers(0, 3, 120)) for d in ["A", "B", "C"]}
    rec = cdl.run_fold_decomp(
        doms, False, "B", torch.device("cpu"), episodes=15, ssm_epochs=2, reps=3, k=5, seed=2004, preprocess="l2"
    )
    assert rec is not None
    assert "delta_unsup" in rec
    assert "delta_sup" in rec
