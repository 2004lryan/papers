"""foundation_lib 纯逻辑单元测试（不下载权重/数据；仅校验注册表与错误分支）。"""

from __future__ import annotations

import pytest
import torch

import foundation_lib as fl


def test_registry_consistency() -> None:
    # 每个登记编码器都有维度；PACS/OfficeHome 各 4 域，DomainNet 6 域
    assert set(fl.FOUNDATION_DIMS) == {"resnet50", "dinov2_vits14", "clip_vitb32"}
    assert len(fl.DG_BENCHMARKS["PACS"]) == 4
    assert len(fl.DG_BENCHMARKS["OfficeHome"]) == 4
    assert len(fl.DG_BENCHMARKS["DomainNet"]) == 6


def test_build_encoder_unknown_raises() -> None:
    with pytest.raises(ValueError, match="未知 foundation 编码器"):
        fl._build_encoder("does_not_exist", torch.device("cpu"))
