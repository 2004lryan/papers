"""foundation_lib.py: DG benchmark 图像加载 + 冻结 foundation 编码器特征提取。

作为模块导入使用，不单独运行。为 32_dg_benchmark_leakage 提供"把主流域泛化 benchmark 的图像经
frozen foundation 编码器(ResNet-50/DINOv2/CLIP)转成每域特征向量"的接口，随后复用 crossdataset_lib
的 E0/E2/U_src/U_tgt 分解机制在特征上量化 Δ_unsup/Δ_sup。

重型依赖(torchvision/timm/open_clip/PIL)按需 lazy import：本机仅装 torchvision，DINOv2/CLIP 权重与
PACS/OfficeHome/DomainNet 数据须在 GPU 服务器就绪后下载。benchmark 目录结构约定：
    <root>/<benchmark>/<domain>/<class>/<image>.jpg
    PACS 域: art_painting cartoon photo sketch ; OfficeHome 域: Art Clipart Product Real_World
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

NDArray = npt.NDArray[Any]

# benchmark → 域名列表（目录名）
DG_BENCHMARKS: dict[str, list[str]] = {
    "PACS": ["art_painting", "cartoon", "photo", "sketch"],
    "OfficeHome": ["Art", "Clipart", "Product", "Real_World"],
    "DomainNet": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
}

# foundation 编码器 → 输出特征维度（用于日志/校验）
FOUNDATION_DIMS: dict[str, int] = {"resnet50": 2048, "dinov2_vits14": 384, "clip_vitb32": 512}


def _build_encoder(name: str, device: torch.device) -> tuple[Any, Callable[[Any], Any], int]:
    """按名加载冻结 foundation 编码器，返回 (前向函数, 图像预处理 transform, 特征维度)。

    前向函数输入一批已 transform 的图像张量 (B,C,H,W)，返回 (B,feat_dim) 特征张量。
    """
    if name == "resnet50":
        import torchvision.transforms as tvt  # lazy
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()  # 去分类头 → 2048 维特征
        model.eval().to(device)
        transform = weights.transforms()

        def fwd_rn(batch: Any) -> Any:
            return model(batch)

        return fwd_rn, transform, 2048

    if name == "dinov2_vits14":
        import torchvision.transforms as tvt  # lazy

        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")  # type: ignore[no-untyped-call]
        model.eval().to(device)
        transform = tvt.Compose(
            [
                tvt.Resize(256),
                tvt.CenterCrop(224),
                tvt.ToTensor(),
                tvt.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        def fwd_dino(batch: Any) -> Any:
            return model(batch)

        return fwd_dino, transform, 384

    if name == "clip_vitb32":
        import open_clip  # lazy

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model.eval().to(device)

        def fwd_clip(batch: Any) -> Any:
            return model.encode_image(batch)

        return fwd_clip, preprocess, 512

    raise ValueError(f"未知 foundation 编码器: {name}（可选 {list(FOUNDATION_DIMS)}）")


def extract_domain_features(
    benchmark_root: str,
    benchmark: str,
    encoder_name: str,
    device: torch.device,
    batch_size: int = 128,
    max_per_domain: int = 2000,
    seed: int = 0,
) -> dict[str, tuple[NDArray, NDArray]]:
    """对 benchmark 各域图像用冻结 foundation 编码器提特征，返回 {域: (特征[N,D], 标签[N])}。

    每域至多采样 max_per_domain 张以控算力；标签为该域 ImageFolder 的类索引（跨域类名一致时可比）。
    """
    import torchvision.transforms as tvt  # noqa: F401  (确保 torchvision 在此环境可用)
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    fwd, transform, feat_dim = _build_encoder(encoder_name, device)
    if feat_dim != FOUNDATION_DIMS.get(encoder_name, feat_dim):
        raise ValueError(f"{encoder_name} 特征维度 {feat_dim} 与登记值 {FOUNDATION_DIMS[encoder_name]} 不符")
    rng = np.random.default_rng(seed)
    doms: dict[str, tuple[NDArray, NDArray]] = {}
    for dom in DG_BENCHMARKS[benchmark]:
        ds = ImageFolder(f"{benchmark_root}/{benchmark}/{dom}", transform=transform)
        idx: NDArray = np.arange(len(ds))
        if len(idx) > max_per_domain:
            idx = rng.choice(idx, max_per_domain, replace=False)
        loader = DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=batch_size, shuffle=False)
        feats_list: list[NDArray] = []
        labs_list: list[NDArray] = []
        with torch.no_grad():
            for xb, yb in loader:
                out = fwd(xb.to(device)).float().cpu().numpy()
                feats_list.append(out)
                labs_list.append(yb.numpy())
        doms[dom] = (np.vstack(feats_list).astype(np.float64), np.concatenate(labs_list).astype(int))
    return doms
