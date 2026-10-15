"""crossdataset_lib.py: 跨数据集泄漏实验共享 harness（loaders / 协议 / 评估）。

作为模块导入使用，不单独运行。抽取自 24_crossdataset_leakage.py 的稳定组件，供
24（共享 vs 折限定）与 31_leakage_decomposition（E0/E1/E2 三协议 Δ_unsup/Δ_sup 分解）
共用，保证 loader、预处理、类选择、元训练、K-shot 评估口径完全一致（单一真值源）。

覆盖数据集（4 个真实跨域，13 留一域折）：corn(仪器) / tablet(仪器) / mango(季) / indianpines(空间)。
apple(年份 3 折) 与 OliveOil(国别 1 折) 因流水线不同，其分解由各自章节脚本平行产出后并入。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import encoders as enc_lib
import export_utils as eu
from protonet import ProtoEncoder, proto_loss

NDArray = npt.NDArray[Any]
LoaderResult = tuple[dict[str, tuple[NDArray, NDArray]], bool]


# ─────────────────────────── 预处理工具 ───────────────────────────
def snv(spec: NDArray) -> NDArray:
    """逐光谱标准正态变量变换（L1 安全：仅用该光谱自身统计）。"""
    out: NDArray = (spec - spec.mean(1, keepdims=True)) / (spec.std(1, keepdims=True) + 1e-8)
    return out


def tertile_labels(y_src: NDArray, y_all: NDArray) -> NDArray:
    """按源域三分位把连续目标分 3 类（阈值仅用源域，防泄漏）。"""
    q1, q2 = np.quantile(y_src, [1 / 3, 2 / 3])
    return np.digitize(y_all, [q1, q2]).astype(int)


def mmd2_linear(a: NDArray, b: NDArray) -> float:
    """线性核 MMD^2（原始光谱均值差平方范数）。"""
    return float(np.sum((a.mean(0) - b.mean(0)) ** 2))


# ─────────────────────────── 数据集加载 ───────────────────────────
def load_corn(root: str, seed: int) -> LoaderResult:
    mat = sio.loadmat(f"{root}/036_corn_nir_3instruments/corn.mat")
    prot = np.asarray(mat["propvals"][0, 0]["data"], dtype=np.float64)[:, 2]
    doms = {
        key: (np.asarray(mat[key][0, 0]["data"], dtype=np.float64), prot) for key in ["m5spec", "mp5spec", "mp6spec"]
    }
    return doms, True


def load_tablet(root: str, seed: int) -> LoaderResult:
    mat = sio.loadmat(f"{root}/037_tablet_nir_shootout2002/nir_shootout_2002.mat")
    parts = ["calibrate", "test", "validate"]
    doms = {}
    for inst in ["1", "2"]:
        spec = np.vstack([np.asarray(mat[f"{p}_{inst}"][0, 0]["data"], dtype=np.float64) for p in parts])
        yval = np.vstack([np.asarray(mat[f"{p}_Y"][0, 0]["data"], dtype=np.float64) for p in parts])
        doms[f"instr{inst}"] = (spec, yval[:, 0])
    return doms, True


def load_mango(root: str, seed: int) -> LoaderResult:
    df = pd.read_csv(f"{root}/041_mango_anderson_crossseason/NAnderson2020MendeleyMangoNIRData.csv")
    wl = [c for c in df.columns if c.isdigit()]
    doms: dict[str, tuple[NDArray, NDArray]] = {}
    for season in sorted(df["Season"].unique()):
        sub = df[df["Season"] == season]
        if len(sub) < 60:
            continue
        sub = sub.sample(min(500, len(sub)), random_state=seed)
        doms[f"S{season}"] = (sub[wl].to_numpy(np.float64), sub["DM"].to_numpy(np.float64))
    return doms, True


def load_indianpines(root: str, seed: int) -> LoaderResult:
    base = f"{root}/034_hsi_classification_benchmarks"
    cube = sio.loadmat(f"{base}/Indian_pines_corrected.mat")["indian_pines_corrected"].astype(np.float64)
    gt = sio.loadmat(f"{base}/Indian_pines_gt.mat")["indian_pines_gt"].astype(int)
    height, width, n_band = cube.shape
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dom = (rows >= height // 2).astype(int) * 2 + (cols >= width // 2).astype(int)
    spec, lab, dvec = cube.reshape(-1, n_band), gt.reshape(-1), dom.reshape(-1)
    keep = lab > 0
    spec, lab, dvec = spec[keep], lab[keep], dvec[keep]
    doms = {f"Q{q}": (spec[dvec == q], lab[dvec == q]) for q in sorted(np.unique(dvec))}
    return doms, False  # native 分类


def load_kiwifruit(root: str, seed: int, device_name: str = "KK1") -> LoaderResult:
    """猕猴桃 NIR（Figshare 21747983）：只取一台仪器（默认 KK1），按采集年份（2017/2018/2019）分域，目标为干物质（%）。

    同一果在同一台仪器上只有一条光谱：以「采集日期 + 样品号」为果的编号核对，出现重复即报错。
    波段取 402–1065 nm（222 个）：数据集自带的协方差矩阵即为这一窗口；1068 nm 以上 2017 年有 90 条光谱缺值。
    不在 DATASETS 里登记（24/31/33/35 号的数据集清单不变），由 74 号脚本单独登记。
    """
    df = pd.read_csv(f"{root}/012_kiwifruit_nir_drymatter/kiwifruit_dat.csv")
    df = df[df["device"] == device_name].reset_index(drop=True)
    key = df["Date"].astype(str) + "#" + df["sample_id"].astype(str)
    if key.duplicated().any():
        raise ValueError(f"猕猴桃 {device_name}：同一果出现多条光谱（{int(key.duplicated().sum())} 条）")
    wl = [c for c in df.columns if c.startswith("X") and int(c[1:]) <= 1065]
    spec = df[wl].to_numpy(np.float64)
    if not np.isfinite(spec).all() or df["DM"].isna().any():
        raise ValueError(f"猕猴桃 {device_name}：光谱或干物质含缺失值")
    year = pd.to_datetime(df["Date"], dayfirst=True).dt.year.to_numpy()
    doms = {f"Y{y}": (spec[year == y], df["DM"].to_numpy(np.float64)[year == y]) for y in sorted(np.unique(year))}
    return doms, True


DATASETS: dict[str, dict[str, Any]] = {
    "corn_instr": {"loader": load_corn, "drift": "仪器", "modality": "NIR点谱"},
    "tablet_instr": {"loader": load_tablet, "drift": "仪器", "modality": "NIR点谱"},
    "mango_season": {"loader": load_mango, "drift": "采收季", "modality": "NIR点谱"},
    "indianpines_spatial": {"loader": load_indianpines, "drift": "空间域", "modality": "HSI成像"},
}


# ─────────────────────────── episode 采样与元训练 ───────────────────────────
class ArrSampler:
    """通用 (X,y) episode 采样器（最多 max_way 类）。"""

    def __init__(self, feats: NDArray, labels: NDArray, k: int, q: int, max_way: int = 6) -> None:
        self.by_class = {c: np.where(labels == c)[0] for c in np.unique(labels)}
        self.feats, self.k, self.q, self.max_way = feats, k, q, max_way
        self.classes = list(np.unique(labels))

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if len(self.classes) <= self.max_way:
            chosen = self.classes
        else:
            chosen = list(np.random.choice(self.classes, self.max_way, replace=False))
        sup_x, sup_y, qry_x, qry_y = [], [], [], []
        for label, cls in enumerate(chosen):
            idx = self.by_class[cls].copy()
            np.random.shuffle(idx)
            need = self.k + self.q
            if len(idx) < need:
                idx = np.tile(idx, need // len(idx) + 1)
            ch = idx[:need]
            for i in ch[: self.k]:
                sup_x.append(self.feats[i])
                sup_y.append(label)
            for i in ch[self.k :]:
                qry_x.append(self.feats[i])
                qry_y.append(label)
        return (
            torch.from_numpy(np.stack(sup_x)).float(),
            torch.tensor(sup_y, dtype=torch.long),
            torch.from_numpy(np.stack(qry_x)).float(),
            torch.tensor(qry_y, dtype=torch.long),
            len(chosen),
        )


def meta_train(
    feats: NDArray,
    labels: NDArray,
    in_dim: int,
    device: torch.device,
    episodes: int,
    seed: int,
    encoder: str = "cnn",
) -> nn.Module:
    """有监督元训练编码器（E0 折限定 / E2 label-aware 共享，由传入样本集决定）。

    encoder 取 encoders.DEEP_ENCODERS 之一；默认 cnn 即 ProtoEncoder(in_dim, feat_dim=64)，与此前各脚本相同。
    """
    eu.set_random_seed(seed)
    enc = enc_lib.build_encoder(encoder, in_dim, 64).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    smp = ArrSampler(feats, labels, 5, 10)
    enc.train()
    for _ in range(episodes):
        sup_x, sup_y, qry_x, qry_y, n_way = smp.sample()
        opt.zero_grad()
        loss, _ = proto_loss(enc, sup_x, sup_y, qry_x, qry_y, n_way, device)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    return enc


def eval_kshot(
    enc: nn.Module,
    feats: NDArray,
    labels: NDArray,
    classes: list[int],
    device: torch.device,
    k: int,
    reps: int,
    seed: int,
) -> float:
    """留出域 K-shot ProtoNet(alpha=0) 评估，返回 reps 次重复的宏 F1 均值。"""
    np.random.seed(seed)
    enc.eval()
    f1s: list[float] = []
    with torch.no_grad():
        for _ in range(reps):
            proto: dict[int, NDArray] = {}
            used: set[int] = set()
            ok = True
            for cls in classes:
                idx = np.where(labels == cls)[0].copy()
                np.random.shuffle(idx)
                if len(idx) < k + 2:
                    ok = False
                    break
                sup = idx[:k]
                used.update(sup.tolist())
                feat = enc(torch.from_numpy(feats[sup]).float().to(device)).cpu().numpy().mean(0)
                proto[cls] = feat / (np.linalg.norm(feat) + 1e-8)
            if not ok:
                continue
            qidx = [i for i in range(len(labels)) if i not in used and labels[i] in classes]
            if not qidx:
                continue
            proto_mat = np.stack([proto[c] for c in classes])
            feat_q = enc(torch.from_numpy(feats[qidx]).float().to(device)).cpu().numpy()
            pred = np.array(classes)[(feat_q @ proto_mat.T).argmax(1)]
            f1s.append(f1_score(np.array(labels)[qidx], pred, average="macro", labels=classes))
    return float(np.mean(f1s)) if f1s else float("nan")


# ─────────────────────────── 掩码自监督（无标签目标暴露 U_src/U_tgt 共用）───────────────────────────
class SpectralDecoder(nn.Module):
    """从 ProtoEncoder 的 64 维特征重构完整输入向量（掩码自监督解码头，谱/特征通用）。"""

    def __init__(self, feat_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(z)
        return out


def span_mask(n: int, in_dim: int, mask_ratio: float, span: int, rng: np.random.Generator) -> NDArray:
    """连续 span 掩码：每样本随机选若干起点，掩掉长度 span 的段，直至达到 mask_ratio。"""
    mask = np.zeros((n, in_dim), dtype=np.float32)
    n_span = max(1, int(np.ceil(mask_ratio * in_dim / span)))
    for i in range(n):
        for _ in range(n_span):
            start = int(rng.integers(0, max(1, in_dim - span)))
            mask[i, start : start + span] = 1.0
    return mask


def masked_ssm_pretrain(
    x_unlabeled: NDArray,
    in_dim: int,
    device: torch.device,
    epochs: int,
    seed: int,
    mask_ratio: float = 0.30,
    span: int = 5,
    batch_size: int = 128,
    patience: int = 15,
) -> ProtoEncoder:
    """在无标签 x（源 or 源+目标）上做掩码段重构预训练，返回冻结特征编码器。

    编码器「绝不接触 y」；仅从无标签分布几何学表示，供 U_src(仅源)/U_tgt(源+目标) 隔离 Δ_unsup。
    """
    eu.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    enc = ProtoEncoder(in_dim, feat_dim=64).to(device)
    dec = SpectralDecoder(64, in_dim).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)

    x_all = torch.from_numpy(x_unlabeled.astype(np.float32)).to(device)
    n = x_all.shape[0]
    n_val = max(1, int(0.1 * n))
    perm = rng.permutation(n)
    x_tr, x_val = x_all[perm[n_val:]], x_all[perm[:n_val]]

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        enc.train()
        dec.train()
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), batch_size):
            bidx = order[s : s + batch_size]
            if len(bidx) < 2:  # BatchNorm 需 ≥2
                continue
            xb = x_tr[bidx]
            m = torch.from_numpy(span_mask(len(bidx), in_dim, mask_ratio, span, rng)).to(device)
            recon = dec(enc(xb * (1.0 - m)))
            loss = ((recon - xb) ** 2 * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        enc.eval()
        dec.eval()
        with torch.no_grad():
            mv = torch.from_numpy(span_mask(len(x_val), in_dim, mask_ratio, span, np.random.default_rng(seed))).to(
                device
            )
            rv = dec(enc(x_val * (1.0 - mv)))
            vloss = float((((rv - x_val) ** 2 * mv).sum() / mv.sum().clamp_min(1.0)).item())
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {kk: vv.detach().clone() for kk, vv in enc.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        enc.load_state_dict(best_state)
    enc.eval()
    return enc


class _ProjHead(nn.Module):
    """SimCLR 投影头（对比预训练用，评估前丢弃）。"""

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(z)
        return out


def _spectral_augment(xb: torch.Tensor, in_dim: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    """谱增强（对比学习正样本对生成）：波段段掩码 + 高斯噪声 + 随机幅度缩放。"""
    n = xb.shape[0]
    m = torch.from_numpy(span_mask(n, in_dim, 0.20, 5, rng)).to(device)
    noise = torch.from_numpy(rng.normal(0.0, 0.05, size=(n, in_dim)).astype(np.float32)).to(device)
    scale = torch.from_numpy(rng.uniform(0.9, 1.1, size=(n, 1)).astype(np.float32)).to(device)
    out: torch.Tensor = (xb * (1.0 - m) + noise) * scale
    return out


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    """NT-Xent 对比损失（2N 视图，正样本为对应增强视图）。"""
    z = torch.cat([z1, z2], dim=0)
    z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
    n2 = z.shape[0]
    sim = (z @ z.t()) / temp
    sim.fill_diagonal_(float("-inf"))
    n = n2 // 2
    targets = torch.arange(n2, device=z.device)
    targets = (targets + n) % n2
    return torch.nn.functional.cross_entropy(sim, targets)


def contrastive_ssm_pretrain(
    x_unlabeled: NDArray,
    in_dim: int,
    device: torch.device,
    epochs: int,
    seed: int,
    temp: float = 0.5,
    batch_size: int = 128,
    patience: int = 15,
) -> ProtoEncoder:
    """在无标签 x 上做 SimCLR 式对比自监督预训练，返回冻结特征编码器。

    比掩码重构更强的 target-aware 无监督对照（回应审稿人：Δ_unsup 是否仅因掩码重构目标弱）。
    编码器「绝不接触 y」；谱增强=波段掩码+高斯噪声+随机缩放，NT-Xent 损失，投影头评估前丢弃。
    """
    eu.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    enc = ProtoEncoder(in_dim, feat_dim=64).to(device)
    head = _ProjHead(64).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)

    x_all = torch.from_numpy(x_unlabeled.astype(np.float32)).to(device)
    n = x_all.shape[0]
    n_val = max(2, int(0.1 * n))
    perm = rng.permutation(n)
    x_tr, x_val = x_all[perm[n_val:]], x_all[perm[:n_val]]

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        enc.train()
        head.train()
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), batch_size):
            bidx = order[s : s + batch_size]
            if len(bidx) < 2:  # NT-Xent / BatchNorm 需 ≥2
                continue
            xb = x_tr[bidx]
            v1 = head(enc(_spectral_augment(xb, in_dim, rng, device)))
            v2 = head(enc(_spectral_augment(xb, in_dim, rng, device)))
            loss = _nt_xent(v1, v2, temp)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        enc.eval()
        head.eval()
        with torch.no_grad():
            vrng = np.random.default_rng(seed)
            vv1 = head(enc(_spectral_augment(x_val, in_dim, vrng, device)))
            vv2 = head(enc(_spectral_augment(x_val, in_dim, vrng, device)))
            vloss = float(_nt_xent(vv1, vv2, temp).item())
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {kk: vv.detach().clone() for kk, vv in enc.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        enc.load_state_dict(best_state)
    enc.eval()
    return enc


class FoldArrays:
    """单个留一域折准备好的数组与元信息（供三协议共用同一输入）。"""

    def __init__(
        self,
        classes: list[int],
        xs: NDArray,
        ys: NDArray,
        xall: NDArray,
        yall: NDArray,
        xh: NDArray,
        yh: NDArray,
        in_dim: int,
        mmd2: float,
    ) -> None:
        self.classes = classes
        self.xs, self.ys = xs, ys  # 源域（E0 有监督元训练输入）
        self.xall, self.yall = xall, yall  # 源+目标（E2 label-aware 元训练；E1 仅用 xall 无标签）
        self.xh, self.yh = xh, yh  # 留出目标域（评估）
        self.in_dim = in_dim
        self.mmd2 = mmd2


def _apply_preprocess(x: NDArray, mode: str) -> NDArray:
    """折内预处理（L1 安全，逐样本）：snv=光谱标准正态；l2=特征向量 L2 归一；none=不变。"""
    if mode == "snv":
        return snv(x)
    if mode == "l2":
        out: NDArray = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return out
    return x


def prepare_fold(
    doms: dict[str, tuple[NDArray, NDArray]], is_reg: bool, held: str, k: int, preprocess: str = "snv"
) -> FoldArrays | None:
    """构造单折三协议共用的数组（类不足返回 None）；与 24 的类选择口径一致。

    preprocess: 光谱用 "snv"（默认，与 24/31/33 一致）；foundation 图像特征用 "l2"（32）。
    """
    names = list(doms.keys())
    src = [d for d in names if d != held]
    in_dim = doms[names[0]][0].shape[1]
    y_src_cont = np.concatenate([doms[d][1] for d in src])

    def lab(v: NDArray) -> NDArray:
        return tertile_labels(y_src_cont, v) if is_reg else v.astype(int)

    ys_all = lab(np.concatenate([doms[d][1] for d in src]))
    yh_all = lab(doms[held][1])
    cs_src = {c for c in np.unique(ys_all) if (ys_all == c).sum() >= 15}
    cs_held = {c for c in np.unique(yh_all) if (yh_all == c).sum() >= k + 2}
    classes = sorted(cs_src & cs_held)[:6]
    if len(classes) < 3:
        return None

    def build(dom_list: list[str]) -> tuple[NDArray, NDArray]:
        feats = _apply_preprocess(np.vstack([doms[d][0] for d in dom_list]), preprocess)
        labels = np.concatenate([lab(doms[d][1]) for d in dom_list])
        mask = np.isin(labels, classes)
        return feats[mask], labels[mask]

    xs, ys = build(src)
    xall, yall = build(names)
    xh, yh = build([held])
    return FoldArrays(classes, xs, ys, xall, yall, xh, yh, in_dim, mmd2_linear(xs, xh))


def run_fold_decomp(
    doms: dict[str, tuple[NDArray, NDArray]],
    is_reg: bool,
    held: str,
    device: torch.device,
    episodes: int,
    ssm_epochs: int,
    reps: int,
    k: int,
    seed: int,
    preprocess: str = "snv",
) -> dict[str, Any] | None:
    """单折监督/无监督四协议分解，返回 Δ_unsup/Δ_sup 记录（类不足则 None）；31(谱)/32(图像特征)共用。

    监督臂 E0(源域x,y有监督)/E2(源+目标x,y有监督,label-aware)→Δ_sup=E2−E0；
    无监督臂 U_src(自监督仅源x)/U_tgt(自监督源+目标x,均不碰标签)→Δ_unsup=U_tgt−U_src(干净隔离)。
    """
    fa = prepare_fold(doms, is_reg, held, k, preprocess)
    if fa is None:
        return None
    enc_e0 = meta_train(fa.xs, fa.ys, fa.in_dim, device, episodes, seed)
    enc_e2 = meta_train(fa.xall, fa.yall, fa.in_dim, device, episodes, seed)
    enc_usrc = masked_ssm_pretrain(fa.xs, fa.in_dim, device, ssm_epochs, seed)
    enc_utgt = masked_ssm_pretrain(fa.xall, fa.in_dim, device, ssm_epochs, seed)
    f_e0 = eval_kshot(enc_e0, fa.xh, fa.yh, fa.classes, device, k, reps, seed)
    f_e2 = eval_kshot(enc_e2, fa.xh, fa.yh, fa.classes, device, k, reps, seed)
    f_usrc = eval_kshot(enc_usrc, fa.xh, fa.yh, fa.classes, device, k, reps, seed)
    f_utgt = eval_kshot(enc_utgt, fa.xh, fa.yh, fa.classes, device, k, reps, seed)
    return {
        "held": held,
        "e0_sup_fold_restricted": f_e0,
        "e2_label_aware": f_e2,
        "u_src_selfsup_fold_restricted": f_usrc,
        "u_tgt_selfsup_exposure": f_utgt,
        "delta_sup": f_e2 - f_e0,
        "delta_unsup": f_utgt - f_usrc,
        "mmd2": fa.mmd2,
        "n_classes": len(fa.classes),
    }


def robust_intercept_ci(values: NDArray, groups: NDArray, n_boot: int = 2000, boot_seed: int = 2024) -> dict[str, Any]:
    """总体截距 + 95%CI（折级观测，数据集随机效应）：优先 MixedLM，奇异/不收敛则簇自助回退。

    回退用按数据集的 cluster bootstrap（§7.2 预注册方法之一），对 MixedLM 奇异矩阵稳健。
    返回 {intercept, ci_low, ci_high, method}。
    """
    vals = np.asarray(values, dtype=np.float64)
    grp = np.asarray(groups)
    uniq = list(dict.fromkeys(grp.tolist()))
    if len(uniq) >= 2 and len(vals) >= 4:
        try:
            import statsmodels.formula.api as smf

            df = pd.DataFrame({"y": vals, "g": grp})
            res = smf.mixedlm("y ~ 1", df, groups=df["g"]).fit(reml=True, method="lbfgs")
            coef = float(res.fe_params["Intercept"])
            se = float(res.bse_fe["Intercept"])
            if np.isfinite(se) and se > 0:
                return {"intercept": coef, "ci_low": coef - 1.96 * se, "ci_high": coef + 1.96 * se, "method": "mixedlm"}
        except Exception:  # MixedLM 奇异/不收敛 → 簇自助回退
            pass
    rng = np.random.default_rng(boot_seed)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(np.array(uniq), len(uniq), replace=True)
        pool = np.concatenate([vals[grp == g] for g in chosen])
        boots.append(float(pool.mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"intercept": float(vals.mean()), "ci_low": float(lo), "ci_high": float(hi), "method": "cluster_bootstrap"}


def iter_datasets(
    data_root: str, seed: int
) -> list[tuple[str, dict[str, Any], dict[str, tuple[NDArray, NDArray]], bool]]:
    """按 DATASETS 顺序加载全部数据集，返回 (名称, 元信息, 域字典, 是否回归)。"""
    out = []
    for dsname, meta in DATASETS.items():
        loader: Callable[[str, int], LoaderResult] = meta["loader"]
        doms, is_reg = loader(data_root, seed)
        out.append((dsname, meta, doms, is_reg))
    return out
