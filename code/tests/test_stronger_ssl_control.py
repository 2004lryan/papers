"""35_stronger_ssl_control + crossdataset_lib 对比自监督单元测试（CPU 快速，无需数据/GPU）。

覆盖：谱增强形状、NT-Xent 损失非负标量、contrastive_ssm_pretrain 产出冻结 L2 归一编码器、
run_fold_contrastive 在合成数据上仅产出对比无监督臂三键且 Δ_unsup^ctr=U_tgt2−U_src2。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

import crossdataset_lib as cdl

NDArray = npt.NDArray[Any]

_MOD_PATH = Path(__file__).resolve().parents[1] / "35_stronger_ssl_control.py"
_spec = importlib.util.spec_from_file_location("_mod35", _MOD_PATH)
assert _spec is not None
assert _spec.loader is not None
mod35 = importlib.util.module_from_spec(_spec)
sys.modules["_mod35"] = mod35
_spec.loader.exec_module(mod35)


def test_spectral_augment_shape() -> None:
    xb = torch.randn(6, 30)
    rng = np.random.default_rng(0)
    aug = cdl._spectral_augment(xb, 30, rng, torch.device("cpu"))
    assert aug.shape == (6, 30)
    assert not torch.equal(aug, xb)  # 增强改变了输入


def test_nt_xent_nonneg_scalar() -> None:
    z1 = torch.randn(8, 64)
    z2 = torch.randn(8, 64)
    loss = cdl._nt_xent(z1, z2, temp=0.5)
    assert loss.dim() == 0
    assert float(loss) >= 0.0


def test_contrastive_pretrain_returns_frozen_l2_encoder() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, size=(80, 30)).astype(np.float64)
    enc = cdl.contrastive_ssm_pretrain(x, 30, torch.device("cpu"), epochs=3, seed=2004)
    assert not enc.training
    with torch.no_grad():
        feat = enc(torch.from_numpy(x[:5]).float())
    assert feat.shape == (5, 64)
    assert np.allclose(np.linalg.norm(feat.numpy(), axis=1), 1.0, atol=1e-4)


def _toy_doms(seed: int = 0) -> dict[str, tuple[NDArray, NDArray]]:
    rng = np.random.default_rng(seed)
    doms: dict[str, tuple[NDArray, NDArray]] = {}
    for j, d in enumerate(["A", "B", "C"]):
        x = rng.normal(j * 0.2, 1.0, size=(150, 24))
        y = rng.integers(0, 4, size=150)
        doms[d] = (x, y)
    return doms


def test_run_fold_contrastive_only_unsup_arm() -> None:
    rec = mod35.run_fold_contrastive(_toy_doms(), False, "C", torch.device("cpu"), ssm_epochs=3, reps=3, k=5, seed=2004)
    assert rec is not None
    assert set(rec) == {"held", "u_src_ctr", "u_tgt_ctr", "delta_unsup_ctr", "mmd2"}
    assert abs(rec["delta_unsup_ctr"] - (rec["u_tgt_ctr"] - rec["u_src_ctr"])) < 1e-9
