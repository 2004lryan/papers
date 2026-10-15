"""encoders.py 编码器族单元测试（CPU 快速，无需数据/GPU）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from sklearn.cross_decomposition import PLSRegression

import encoders as enc_lib


@pytest.mark.parametrize("name", enc_lib.DEEP_ENCODERS)
@pytest.mark.parametrize("in_dim", [200, 229, 700])
def test_deep_encoder_shape_and_unit_norm(name: str, in_dim: int) -> None:
    torch.manual_seed(0)
    enc = enc_lib.build_encoder(name, in_dim, 64).eval()
    with torch.no_grad():
        out = enc(torch.randn(7, in_dim))
    assert out.shape == (7, 64)
    assert torch.allclose(out.norm(dim=1), torch.ones(7), atol=1e-5)


@pytest.mark.parametrize("name", enc_lib.DEEP_ENCODERS)
def test_deep_encoder_reproducible_init(name: str) -> None:
    x = torch.randn(4, 229, generator=torch.Generator().manual_seed(1))
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        enc = enc_lib.build_encoder(name, 229, 32).eval()
        with torch.no_grad():
            outs.append(enc(x))
    assert torch.equal(outs[0], outs[1])


def test_deep_encoder_trains_one_step() -> None:
    for name in enc_lib.DEEP_ENCODERS:
        torch.manual_seed(0)
        enc = enc_lib.build_encoder(name, 229, 32)
        opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
        loss = enc(torch.randn(8, 229)).sum()
        loss.backward()
        opt.step()
        assert enc_lib.n_params(enc) > 0


def _toy(seed: int = 0) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    y = np.repeat(np.array(["a", "b", "c"]), 30)
    means = {"a": 0.0, "b": 1.0, "c": -1.0}
    x = rng.normal(size=(90, 50)) + np.array([means[v] for v in y])[:, None] * np.linspace(0, 1, 50)
    return x, y


def test_pca_matches_svd_projection() -> None:
    x, _ = _toy()
    enc, n = enc_lib.fit_linear_encoder("pca", x, None, 5)
    assert n == 5
    with torch.no_grad():
        z = enc(torch.as_tensor(x, dtype=torch.float32)).numpy()
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    ref = (x - mean) @ vt[:5].T
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    # 各列符号约定可能不同：逐列比较绝对值
    assert np.allclose(np.abs(z), np.abs(ref), atol=1e-4)
    assert enc_lib.n_params(enc) == 0


def test_pls_matches_sklearn_scores() -> None:
    x, y = _toy()
    enc, n = enc_lib.fit_linear_encoder("pls", x, y, 4)
    ind = (y[:, None] == np.unique(y)[None, :]).astype(float)
    ref = PLSRegression(n_components=4, scale=True).fit(x, ind).transform(x)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    with torch.no_grad():
        z = enc(torch.as_tensor(x, dtype=torch.float32)).numpy()
    assert n == 4
    assert np.allclose(np.abs(z), np.abs(ref), atol=1e-4)


def test_linear_component_cap_and_label_requirement() -> None:
    x, _ = _toy()
    _, n = enc_lib.fit_linear_encoder("pca", x[:6], None, 16)
    assert n == 5
    with pytest.raises(ValueError, match="pls 需要类别标签"):
        enc_lib.fit_linear_encoder("pls", x, None, 4)
    with pytest.raises(ValueError, match="未知深度编码器"):
        enc_lib.build_encoder("pls", 10, 8)
