"""encoders.py: 评估泄漏实验用的编码器族（作为模块导入使用，不单独运行）。

    深度族  cnn         protonet.ProtoEncoder（一维卷积 + 全连接，现有正式结果所用）
            mlp         三层全连接
            resnet      一维残差网络（步长卷积下采样，三个残差块，全局均值池化）
            transformer 波段分块嵌入 + 两层自注意力（注意力按矩阵乘法写出）
    线性族  pls         对训练池的类别指示矩阵做 PLS2，取潜变量得分（有监督）
            pca         训练池光谱的主成分得分（无监督，不看标签）

所有编码器输入 (B, D) 光谱，输出 L2 归一化嵌入，可直接替换原型网络的编码器：深度族按原型网络
元训练，线性族在训练池上直接拟合后冻结为线性映射 ((x − 均值) / 尺度) · W。

深度族只用 CUDA 确定性模式下有确定实现的算子（export_utils.set_random_seed 在 CUDA 上开启
torch.use_deterministic_algorithms）：不用自适应池化、最大池化与融合注意力核。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression

from protonet import ProtoEncoder

NDArray = npt.NDArray[Any]

DEEP_ENCODERS = ("cnn", "mlp", "resnet", "transformer")
LINEAR_ENCODERS = ("pls", "pca")
ENCODERS = (*DEEP_ENCODERS, *LINEAR_ENCODERS)
ARCH_DESC = {
    "cnn": "ProtoEncoder(1D-CNN+MLP)",
    "mlp": "MLP(2x256 hidden, BN, dropout 0.3)",
    "resnet": "ResNet1D(stride-2 stem, 3 residual blocks x 64 ch, global mean pool)",
    "transformer": "Transformer(8-band patches, d=64, 2 layers, 4 heads, pre-norm)",
    "pls": "PLS2 latent scores on class indicators (linear, supervised)",
    "pca": "PCA scores (linear, unsupervised)",
}


class MLPEncoder(nn.Module):
    """三层全连接编码器。"""

    def __init__(self, in_dim: int, feat_dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = nn.functional.normalize(self.net(x), dim=1)
        return out


class _ResBlock(nn.Module):
    """两层 3 核卷积 + 跳连。"""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        out: torch.Tensor = nn.functional.relu(x + h)
        return out


class ResNetEncoder(nn.Module):
    """一维残差网络编码器：步长 2 的卷积下采样两次，三个残差块，沿波段取均值。"""

    def __init__(self, in_dim: int, feat_dim: int = 128, ch: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 7, stride=2, padding=3),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, 3, stride=2, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(_ResBlock(ch), _ResBlock(ch), _ResBlock(ch))
        self.head = nn.Linear(ch, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x.unsqueeze(1))).mean(dim=-1)
        out: torch.Tensor = nn.functional.normalize(self.head(h), dim=1)
        return out


class _AttnBlock(nn.Module):
    """前置层归一化的自注意力块；注意力写成矩阵乘法与 softmax。"""

    def __init__(self, d_model: int, n_heads: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        dh = d // self.n_heads
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.n_heads, dh).permute(2, 0, 3, 1, 4)
        att = self.drop(torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(dh), dim=-1))
        h = (att @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.drop(self.proj(h))
        out: torch.Tensor = x + self.drop(self.ff(self.norm2(x)))
        return out


class TransformerEncoder(nn.Module):
    """波段分块嵌入的 Transformer 编码器：每 patch 个相邻波段为一个词元，末尾补零到整块。"""

    def __init__(
        self,
        in_dim: int,
        feat_dim: int = 128,
        patch: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.n_tok = math.ceil(in_dim / patch)
        self.pad = self.n_tok * patch - in_dim
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.Sequential(*[_AttnBlock(d_model, n_heads, ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad:
            x = nn.functional.pad(x, (0, self.pad))
        h = self.embed(x.view(x.size(0), self.n_tok, self.patch)) + self.pos
        h = self.norm(self.blocks(h)).mean(dim=1)
        out: torch.Tensor = nn.functional.normalize(self.head(h), dim=1)
        return out


class LinearProjEncoder(nn.Module):
    """冻结的线性映射 ((x − mean) / scale) · W，输出 L2 归一化得分。"""

    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor

    def __init__(self, mean: NDArray, scale: NDArray, weight: NDArray) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(np.asarray(mean, dtype=np.float32)))
        self.register_buffer("scale", torch.as_tensor(np.asarray(scale, dtype=np.float32)))
        self.register_buffer("weight", torch.as_tensor(np.asarray(weight, dtype=np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = ((x - self.mean) / self.scale) @ self.weight
        out: torch.Tensor = nn.functional.normalize(z, dim=1)
        return out


def build_encoder(name: str, in_dim: int, feat_dim: int) -> nn.Module:
    """按名称构造未训练的深度编码器。"""
    if name == "cnn":
        return ProtoEncoder(in_dim, feat_dim=feat_dim)
    if name == "mlp":
        return MLPEncoder(in_dim, feat_dim=feat_dim)
    if name == "resnet":
        return ResNetEncoder(in_dim, feat_dim=feat_dim)
    if name == "transformer":
        return TransformerEncoder(in_dim, feat_dim=feat_dim)
    raise ValueError(f"未知深度编码器：{name}（可选 {DEEP_ENCODERS}）")


def _fix_sign(w: NDArray) -> NDArray:
    """各列按绝对值最大的元素取正号，使分解结果与求解器的符号约定无关。"""
    idx = np.abs(w).argmax(axis=0)
    signs = np.sign(w[idx, np.arange(w.shape[1])])
    signs[signs == 0] = 1.0
    out: NDArray = w * signs
    return out


def fit_linear_encoder(name: str, x: NDArray, y: NDArray | None, n_comp: int) -> tuple[LinearProjEncoder, int]:
    """在训练池 (x, y) 上拟合线性编码器，返回 (编码器, 实际成分数)。

    pca：中心化后做奇异值分解，取前 n_comp 个右奇异向量；不看 y。
    pls：y 为类别编号，转成类别指示矩阵后做 PLS2（sklearn PLSRegression，scale=True），
         取 x 的旋转矩阵；拟合后用 transform 核对 ((x − 均值) / 尺度) · W 与 sklearn 的得分一致。
    成分数不超过 min(样本数 − 1, 波段数)。
    """
    x = np.asarray(x, dtype=np.float64)
    n_eff = int(min(n_comp, x.shape[0] - 1, x.shape[1]))
    if n_eff < 1:
        raise ValueError(f"训练池太小，无法拟合 {name}：{x.shape}")
    if name == "pca":
        mean = x.mean(axis=0)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        w = _fix_sign(vt[:n_eff].T)
        return LinearProjEncoder(mean, np.ones_like(mean), w), n_eff
    if name == "pls":
        if y is None:
            raise ValueError("pls 需要类别标签")
        classes = np.unique(y)
        ind = (np.asarray(y)[:, None] == classes[None, :]).astype(np.float64)
        pls = PLSRegression(n_components=n_eff, scale=True).fit(x, ind)
        mean = x.mean(axis=0)
        scale = x.std(axis=0, ddof=1)
        scale[scale == 0] = 1.0
        w = np.asarray(pls.x_rotations_, dtype=np.float64)
        ref = np.asarray(pls.transform(x), dtype=np.float64)
        mine = ((x - mean) / scale) @ w
        if not np.allclose(ref, mine, rtol=1e-6, atol=1e-6 * max(1.0, float(np.abs(ref).max()))):
            raise RuntimeError("PLS 得分与 sklearn transform 不一致，中心化或尺度约定已变")
        signs = np.sign(w[np.abs(w).argmax(axis=0), np.arange(w.shape[1])])
        signs[signs == 0] = 1.0
        return LinearProjEncoder(mean, scale, w * signs), n_eff
    raise ValueError(f"未知线性编码器：{name}（可选 {LINEAR_ENCODERS}）")


def n_params(module: nn.Module) -> int:
    """可训练参数个数（线性族为 0，其映射存为缓冲区）。"""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
