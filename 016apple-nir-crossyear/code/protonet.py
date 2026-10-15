"""
protonet.py: ProtoNet 共享组件（编码器 / 损失 / episode 采样 / 苹果粒度索引）。

作为模块导入使用，不单独运行。供 05_meta_train / 10_loeo_year_foldwise / 11_kshot_baselines
等脚本统一引用，避免 ProtoEncoder、proto_loss、采样器在多脚本重复实现导致漂移。
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

NDArray = npt.NDArray[Any]

# 9 个年份-产地"元素"定义（全项目统一）
ELEM_INFO: dict[str, tuple[int, str]] = {
    "E1": (2018, "XJ"),
    "E2": (2018, "SD"),
    "E3": (2018, "SX"),
    "E4": (2019, "XJ"),
    "E5": (2019, "SD"),
    "E6": (2019, "SX"),
    "E7": (2025, "XJ"),
    "E8": (2025, "SD"),
    "E9": (2025, "GS"),
}


def build_apple_data(
    df: Any,
    wl_cols: list[str],
    elem_ids: list[str] | None = None,
) -> dict[str, dict[Any, dict[str, Any]]]:
    """构建苹果粒度索引 apple_data[eid][apple_id] = {'X': (n_faces, bands), 'region': str}。"""
    ids = elem_ids if elem_ids is not None else list(ELEM_INFO.keys())
    apple_data: dict[str, dict[Any, dict[str, Any]]] = {}
    for eid in ids:
        yr, reg = ELEM_INFO[eid]
        sub = df[(df["年份"] == yr) & (df["产地"] == reg)]
        aids = sub["苹果编号"].to_numpy()
        x_arr = sub[wl_cols].to_numpy().astype(np.float32)
        apple_data[eid] = {aid: {"X": x_arr[np.where(aids == aid)[0]], "region": reg} for aid in np.unique(aids)}
    return apple_data


def build_year_region_data(df: Any, wl_cols: list[str]) -> dict[int, dict[str, dict[Any, NDArray]]]:
    """构建 year_data[year][region][apple_id] = X(n_faces, bands)，供 K-shot 目标域评估。"""
    year_data: dict[int, dict[str, dict[Any, NDArray]]] = {}
    for year in (2018, 2019, 2025):
        sub_yr = df[df["年份"] == year]
        year_data[year] = {}
        for reg in sub_yr["产地"].unique():
            sub_reg = sub_yr[sub_yr["产地"] == reg]
            aids = sub_reg["苹果编号"].to_numpy()
            x_arr = sub_reg[wl_cols].to_numpy().astype(np.float32)
            year_data[year][str(reg)] = {aid: x_arr[np.where(aids == aid)[0]] for aid in np.unique(aids)}
    return year_data


class EpisodeSampler:
    """从给定元素集采样 N-way K-shot Q-query episode（苹果粒度，跨元素同产地合并）。

    目标年份隔离由调用方通过 elem_ids 控制（fold-restricted 时只传源域元素）。
    """

    def __init__(
        self,
        elem_ids: list[str],
        apple_data: dict[str, dict[Any, dict[str, Any]]],
        n_way: int,
        k_support: int,
        q_query: int,
    ) -> None:
        self.apple_data = apple_data
        self.n_way = n_way
        self.k_support = k_support
        self.q_query = q_query
        self.region_apples: dict[str, list[tuple[str, Any]]] = {}
        for eid in elem_ids:
            for aid, info in apple_data[eid].items():
                self.region_apples.setdefault(info["region"], []).append((eid, aid))
        self.all_regions = list(self.region_apples.keys())

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """返回 (support_X, support_y, query_X, query_y, n_way)。"""
        n_way = min(self.n_way, len(self.all_regions))
        chosen_regions = random.sample(self.all_regions, n_way)
        sup_x: list[Any] = []
        sup_y: list[int] = []
        qry_x: list[Any] = []
        qry_y: list[int] = []
        for label, reg in enumerate(chosen_regions):
            apples = self.region_apples[reg].copy()
            random.shuffle(apples)
            n_total = self.k_support + self.q_query
            if len(apples) < n_total:
                apples = apples * (n_total // len(apples) + 1)
            chosen = apples[:n_total]
            for eid, aid in chosen[: self.k_support]:
                sup_x.append(self.apple_data[eid][aid]["X"].mean(axis=0))  # 苹果=多面位均值
                sup_y.append(label)
            for eid, aid in chosen[self.k_support : n_total]:
                faces = self.apple_data[eid][aid]["X"]
                qry_x.append(faces[random.randint(0, len(faces) - 1)])  # 查询=随机单面位
                qry_y.append(label)
        return (
            torch.from_numpy(np.stack(sup_x)).float(),
            torch.tensor(sup_y, dtype=torch.long),
            torch.from_numpy(np.stack(qry_x)).float(),
            torch.tensor(qry_y, dtype=torch.long),
            n_way,
        )


class ProtoEncoder(nn.Module):
    """1D-CNN + MLP 编码器，输出 L2 归一化特征（适配 NIR 光谱）。"""

    def __init__(self, in_dim: int, feat_dim: int = 128) -> None:
        super().__init__()
        pool_stride = max(1, in_dim // 32)
        pooled_len = 1 + (in_dim - pool_stride) // pool_stride
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # MPS 不支持非整除 AdaptiveAvgPool1d，用固定步长平均池化稳定压缩
            nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64 * pooled_len, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))
        h = h.view(h.size(0), -1)
        feat: torch.Tensor = nn.functional.normalize(self.mlp(h), dim=1)
        return feat


def proto_loss(
    encoder: nn.Module,
    sup_x: torch.Tensor,
    sup_y: torch.Tensor,
    qry_x: torch.Tensor,
    qry_y: torch.Tensor,
    n_way: int,
    device: torch.device,
    temp: float = 10.0,
) -> tuple[torch.Tensor, float]:
    """ProtoNet episode 损失 + 准确率（余弦原型 + 温度缩放交叉熵）。"""
    sup_x, sup_y = sup_x.to(device), sup_y.to(device)
    qry_x, qry_y = qry_x.to(device), qry_y.to(device)
    all_feat = encoder(torch.cat([sup_x, qry_x], dim=0))
    sup_feat, qry_feat = all_feat[: len(sup_x)], all_feat[len(sup_x) :]
    prototypes = []
    for c in range(n_way):
        mask = sup_y == c
        if mask.sum() == 0:
            prototypes.append(torch.zeros(sup_feat.size(1), device=device))
        else:
            prototypes.append(nn.functional.normalize(sup_feat[mask].mean(0), dim=0))
    sim = qry_feat @ torch.stack(prototypes).T * temp
    loss = nn.functional.cross_entropy(sim, qry_y)
    acc = float((sim.argmax(dim=1) == qry_y).float().mean().item())
    return loss, acc


def encode_batch(encoder: nn.Module, x_np: NDArray, device: torch.device, batch_size: int = 512) -> NDArray:
    """分批编码并返回 numpy 特征矩阵（L2 归一化由编码器内部完成）。"""
    feats: list[NDArray] = []
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb = torch.from_numpy(x_np[i : i + batch_size]).to(device)
            feats.append(encoder(xb).cpu().numpy())
    return np.vstack(feats)


def apple_feature(encoder: nn.Module, faces_x: NDArray, device: torch.device) -> NDArray:
    """单个苹果的全面位均值特征（再次 L2 归一化）。"""
    f: NDArray = encode_batch(encoder, faces_x, device).mean(0)
    return f / (float(np.linalg.norm(f)) + 1e-8)
