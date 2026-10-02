"""
模式嵌入网络 Psi(X, S_rel) -> Z

坐标系设计（跨切片对齐）：
    输入的原始坐标 S（任意坐标系）在进入 encoder 前被规范化为 S_rel：
        1. 质心归一化：S_centered = S - mean(S)          消除平移
        2. PCA 主轴对齐：S_rel = S_centered @ V.T        消除旋转/翻折
           符号约定：每个主轴的最大绝对值方向为正，消除翻折歧义
    这样不同切片的 S_rel 在同一规范坐标系里，跨时间点可比，
    ODE 在 Z 空间积分后，SpatialDecoder 解码回 S_rel，与真实 S_rel 做 Wasserstein。
    评估时调用 canonical_to_original() 还原回原始坐标系。

空间编码模式：
    1. 离散 kNN（use_continuous_kernel=False）
    2. 连续 Gaussian kernel（use_continuous_kernel=True，推荐）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================
# PCA 规范化坐标变换
# ============================================================

def pca_canonicalize(
    coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将原始空间坐标规范化为 PCA 对齐的相对坐标。

    变换：S_rel = (S - centroid) @ V.T
    其中 V 是 PCA 旋转矩阵，符号固定（每轴最大绝对值方向为正）。

    Args:
        coords: (N, 2) 原始空间坐标（任意坐标系）

    Returns:
        s_rel:    (N, 2) 规范化相对坐标（跨切片可比）
        centroid: (2,)   质心，还原时需要
        V:        (2, 2) PCA 旋转矩阵，还原时需要
    """
    centroid = coords.mean(dim=0)               # (2,)
    s_centered = coords - centroid              # (N, 2)

    # SVD 计算主轴（等价于 PCA）
    # s_centered = U @ diag(sigma) @ V^T
    # V 的行是主成分方向
    try:
        _, _, Vh = torch.linalg.svd(s_centered, full_matrices=False)  # Vh: (2, 2)
    except Exception:
        # 退化情况（所有点重合）：返回单位变换
        Vh = torch.eye(2, device=coords.device, dtype=coords.dtype)

    # 消除翻折歧义：每个主轴方向的最大绝对值元素强制为正
    # Vh 的每一行是一个主轴方向向量
    signs = torch.sign(
        Vh[torch.arange(2, device=coords.device),
           Vh.abs().argmax(dim=1)]
    )
    Vh = Vh * signs.unsqueeze(1)                # (2, 2)

    s_rel = s_centered @ Vh.T                  # (N, 2)
    return s_rel, centroid, Vh


def canonical_to_original(
    s_rel: torch.Tensor,
    centroid: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """
    将规范化相对坐标还原回原始坐标系。

    逆变换：S = S_rel @ V + centroid

    Args:
        s_rel:    (N, 2) 规范化相对坐标
        centroid: (2,)   编码时保存的质心
        V:        (2, 2) 编码时保存的 PCA 旋转矩阵

    Returns:
        coords: (N, 2) 原始坐标系下的坐标
    """
    return s_rel @ V + centroid


# ============================================================
# 空间编码器（相对位置）
# ============================================================

class RelativeSpatialEncoder(nn.Module):
    """
    空间编码器：将规范化相对坐标 S_rel 编码为空间边特征。

    Anchor 设计：确定性均匀网格
    ────────────────────────────
    PCA 规范化后 S_rel 近似服从标准正态，3σ 覆盖 99.7% 的数据点。
    在 [-3, 3] × [-3, 3] 上铺 k×k 均匀网格作为 anchor（k = ceil(sqrt(num_anchors))）。

    优点：
    - 确定性：不依赖任何数据采样，所有 batch / 切片 / epoch 使用完全相同的 anchor
    - 均匀覆盖：网格保证空间各区域等密度覆盖，无聚集问题
    - 无需初始化：__init__ 时直接生成，register_buffer 永久固定
    - 批量无关：batch 大小变化不影响 anchor，彻底解决 Z 编码不稳定问题

    支持两种模式：
        连续 Gaussian kernel（推荐）：完全可微
        离散 kNN：近似但快速
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.k = config.relative_k
        self.use_continuous = getattr(config, 'use_continuous_kernel', False)

        # RBF 参数
        self.n_rbf = 16
        self.register_buffer('rbf_centers', torch.linspace(0, 2, self.n_rbf))

        if self.use_continuous:
            init_bw = getattr(config, 'kernel_bandwidth', 0.3)
            self.log_bandwidth = nn.Parameter(torch.tensor(math.log(init_bw)))

            # 输入：加权相对位置 (2) + 加权 RBF (16) = 18
            input_dim = 18

            if config.anchor_ratio is not None:
                num_anchors = int(config.anchor_ratio * 100)
            else:
                num_anchors = getattr(config, "num_anchors", 256)

            # 确定性均匀网格 anchor，在 [-3, 3]^2 上铺 k×k 个点
            # PCA 规范化后 S_rel ~ N(0,1)，3σ 覆盖 99.7%
            k_grid = math.ceil(math.sqrt(num_anchors))
            lin = torch.linspace(-3.0, 3.0, k_grid)
            gx, gy = torch.meshgrid(lin, lin, indexing="ij")
            anchors = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (k²,2)
            self.register_buffer("grid_anchors", anchors)   # 永远固定，不随数据变化
        else:
            self.rbf_width = 0.3
            input_dim = self.n_rbf + 2 if config.use_edge_features else 1

        self.edge_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, s_rel: torch.Tensor, knn_indices: Optional[torch.Tensor] = None):
        """
        Args:
            s_rel:       (N, 2) 规范化相对坐标（已经过 pca_canonicalize）
            knn_indices: (N, k) 仅离散模式需要

        Returns:
            连续模式: (N, 64)
            离散模式: (N, k, 64)
        """
        if self.use_continuous:
            return self._forward_continuous(s_rel)
        else:
            return self._forward_discrete(s_rel, knn_indices)

    def _forward_continuous(self, s_rel: torch.Tensor) -> torch.Tensor:
        # anchor 是确定性网格，直接使用，无需任何初始化逻辑
        anchors = self.grid_anchors                            # (M, 2)，固定常量
        bw = torch.exp(self.log_bandwidth)

        dists = torch.cdist(s_rel, anchors)                   # (N, M)

        weights = torch.exp(-dists ** 2 / (2 * bw ** 2))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        rel_pos = anchors.unsqueeze(0) - s_rel.unsqueeze(1)   # (N, M, 2)
        weighted_pos = (weights.unsqueeze(-1) * rel_pos).sum(1)  # (N, 2)

        rbf = self._rbf(dists)                                 # (N, M, 16)
        weighted_rbf = (weights.unsqueeze(-1) * rbf).sum(1)   # (N, 16)

        features = torch.cat([weighted_pos, weighted_rbf], dim=-1)  # (N, 18)
        return self.edge_encoder(features)                           # (N, 64)

    def _forward_discrete(self, s_rel: torch.Tensor, knn_indices: torch.Tensor) -> torch.Tensor:
        if knn_indices is None:
            raise ValueError("knn_indices cannot be None in discrete mode")

        neighbor_coords = s_rel[knn_indices]           # (N, k, 2)
        rel_vec = neighbor_coords - s_rel.unsqueeze(1) # (N, k, 2)
        dists   = torch.norm(rel_vec, dim=-1)          # (N, k)

        if self.config.use_edge_features:
            rbf_dist   = self._rbf_knn(dists)          # (N, k, 16)
            angles     = torch.atan2(rel_vec[..., 1], rel_vec[..., 0])
            angle_feat = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
            features   = torch.cat([rbf_dist, angle_feat], dim=-1)  # (N, k, 18)
        else:
            features = dists.unsqueeze(-1)             # (N, k, 1)

        return self.edge_encoder(features)             # (N, k, 64)

    def _rbf(self, dists: torch.Tensor) -> torch.Tensor:
        centers = self.rbf_centers.view(1, 1, -1)
        d = dists.unsqueeze(-1)
        return torch.exp(-(d - centers) ** 2 / (2 * 0.3 ** 2))

    def _rbf_knn(self, dists: torch.Tensor) -> torch.Tensor:
        centers = self.rbf_centers.view(1, 1, -1)
        d = dists.unsqueeze(-1)
        return torch.exp(-(d - centers) ** 2 / (2 * self.rbf_width ** 2))


# ============================================================
# GNN 层（不变）
# ============================================================

class SpatialGraphConv(nn.Module):
    """空间图卷积层，支持连续/离散两种模式。"""
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, use_continuous: bool = False):
        super().__init__()
        self.use_continuous = use_continuous
        self.node_mlp = nn.Linear(in_dim, out_dim)

        if self.use_continuous:
            self.update_mlp = nn.Sequential(
                nn.Linear(out_dim + edge_dim, out_dim),
                nn.ReLU(),
                nn.LayerNorm(out_dim),
            )
        else:
            self.message_mlp = nn.Sequential(
                nn.Linear(in_dim + edge_dim, out_dim),
                nn.ReLU(),
            )
            self.update_mlp = nn.Sequential(
                nn.Linear(out_dim + out_dim, out_dim),
                nn.ReLU(),
                nn.LayerNorm(out_dim),
            )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        knn_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_feat = self.node_mlp(node_features)

        if self.use_continuous:
            combined = torch.cat([self_feat, edge_features], dim=-1)
            return self.update_mlp(combined)
        else:
            if knn_indices is None:
                raise ValueError("Discrete mode requires knn_indices.")
            neighbor_feat = node_features[knn_indices]
            messages  = self.message_mlp(torch.cat([neighbor_feat, edge_features], dim=-1))
            aggregated = messages.mean(dim=1)
            return self.update_mlp(torch.cat([self_feat, aggregated], dim=-1))


# ============================================================
# PatternEmbedding：主编码器  Psi(X, S) -> Z
# ============================================================

class PatternEmbedding(nn.Module):
    """
    模式嵌入网络 Psi(X, S) -> Z

    内部自动完成 S -> S_rel 的 PCA 规范化，外部调用者无需预处理坐标。
    PCA 变换参数（centroid, V）在 forward 时计算并缓存，供 decode_spatial() 还原使用。

    API 与旧版完全兼容：forward(gene_expr, spatial_coords, knn_indices=None)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 基因编码器
        self.gene_encoder = nn.Sequential(
            nn.Linear(config.n_genes, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # 空间编码器（输入为 S_rel，内部规范化由 forward 完成）
        if config.spatial_encoding == "relative":
            self.spatial_encoder = RelativeSpatialEncoder(config)
        else:
            self.spatial_encoder = nn.Sequential(
                nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64)
            )

        # GNN
        if config.use_spatial_gnn:
            self.gnn_layers = nn.ModuleList([
                SpatialGraphConv(
                    128 if i == 0 else config.gnn_hidden_dim,
                    config.gnn_hidden_dim,
                    64,
                    use_continuous=config.use_continuous_kernel,
                )
                for i in range(config.gnn_num_layers)
            ])
            final_dim = config.gnn_hidden_dim
        else:
            final_dim = 128

        self.latent_projection = nn.Sequential(
            nn.Linear(final_dim, config.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )
        self.layer_norm = nn.LayerNorm(config.latent_dim)

        # PCA 变换缓存（每次 forward 更新，供 decode_spatial 使用）
        # 不用 register_buffer 是因为这是运行时的 batch 级统计，不需要存入 state_dict
        self._last_centroid: Optional[torch.Tensor] = None
        self._last_V: Optional[torch.Tensor] = None

    def forward(
        self,
        gene_expr: torch.Tensor,
        spatial_coords: torch.Tensor,
        knn_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            gene_expr:     (N, n_genes)  z-scored 基因表达
            spatial_coords:(N, 2)        原始坐标（任意坐标系，内部自动 PCA 规范化）
            knn_indices:   (N, k) 可选，离散模式需要

        Returns:
            Z: (N, latent_dim)
        """
        # ---- 1. 空间坐标 PCA 规范化 ----
        s_rel, centroid, V = pca_canonicalize(spatial_coords)
        # 缓存 PCA 参数，供 decode_spatial() 还原
        self._last_centroid = centroid.detach()
        self._last_V        = V.detach()

        # ---- 2. 基因编码 ----
        gene_features = self.gene_encoder(gene_expr)   # (N, 128)

        # ---- 3. kNN 统一入口（离散模式且未提供时自动计算）----
        if (
            self.config.spatial_encoding == "relative"
            and not self.spatial_encoder.use_continuous
            and knn_indices is None
        ):
            knn_indices = compute_knn_kdtree(
                s_rel.detach().cpu(), self.config.relative_k
            ).to(spatial_coords.device)

        # ---- 4. 空间编码（输入 S_rel）----
        if self.config.spatial_encoding == "relative":
            edge_features = self.spatial_encoder(s_rel, knn_indices)
        else:
            edge_features = self.spatial_encoder(s_rel)

        # ---- 5. GNN 融合 ----
        if self.config.use_spatial_gnn:
            node_features = gene_features
            for gnn_layer in self.gnn_layers:
                node_features = F.relu(
                    gnn_layer(node_features, edge_features, knn_indices)
                )
            final_features = node_features
        else:
            final_features = gene_features

        # ---- 6. 投影到 latent 空间 ----
        Z = self.latent_projection(final_features)
        return self.layer_norm(Z)

    def get_canonical_coords(
        self,
        spatial_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        只做 PCA 规范化，不做编码。
        用于在 model.forward() 中获取 S_rel_true（作为 loss 的 ground truth）。

        Returns:
            s_rel:    (N, 2)
            centroid: (2,)
            V:        (2, 2)
        """
        return pca_canonicalize(spatial_coords)

    def decode_spatial(
        self,
        s_rel_pred: torch.Tensor,
        centroid: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        将 SpatialDecoder 输出的规范相对坐标还原回原始坐标系。

        Args:
            s_rel_pred: (N, 2)  SpatialDecoder 的输出
            centroid:   (2,)    若为 None，使用最近一次 forward 缓存的值
            V:          (2, 2)  若为 None，使用最近一次 forward 缓存的值

        Returns:
            s_original: (N, 2) 原始坐标系下的预测坐标
        """
        c = centroid if centroid is not None else self._last_centroid
        v = V        if V        is not None else self._last_V
        if c is None or v is None:
            raise RuntimeError(
                "No cached PCA params. Call PatternEmbedding.forward() first, "
                "or pass centroid and V explicitly."
            )
        return canonical_to_original(s_rel_pred, c, v)


# ============================================================
# kNN 工具
# ============================================================

from sklearn.neighbors import NearestNeighbors

def compute_knn_kdtree(coords: torch.Tensor, k: int) -> torch.Tensor:
    """
    基于 kd-tree 的 kNN 计算（CPU，sklearn）。

    Args:
        coords: (N, 2) CPU tensor
        k:      邻居数

    Returns:
        indices: (N, k) LongTensor
    """
    coords_np = coords.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(coords_np)
    _, indices = nbrs.kneighbors(coords_np)
    return torch.from_numpy(indices[:, 1:]).long()


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("Testing PatternEmbedding with PCA canonical coords...")

    from config import Config
    config = Config()
    config.n_genes     = 100
    config.latent_dim = 64
    config.use_spatial_gnn = True

    N = 200
    X = torch.randn(N, config.n_genes)
    # 模拟两个坐标系不同的切片（旋转 45 度）
    theta = 0.785
    S1 = torch.rand(N, 2) * 10
    R  = torch.tensor([[math.cos(theta), -math.sin(theta)],
                        [math.sin(theta),  math.cos(theta)]])
    S2 = S1 @ R.T + torch.tensor([100.0, 50.0])  # 旋转 + 平移

    print("\n1. 测试 PCA 规范化的旋转不变性:")
    s_rel1, c1, V1 = pca_canonicalize(S1)
    s_rel2, c2, V2 = pca_canonicalize(S2)
    diff = (s_rel1 - s_rel2).norm()
    print(f"   旋转+平移前后 S_rel 差异: {diff.item():.6f}（应接近 0）")

    print("\n2. 测试 PCA 逆变换还原精度:")
    S1_restored = canonical_to_original(s_rel1, c1, V1)
    err = (S1 - S1_restored).norm()
    print(f"   还原误差: {err.item():.8f}（应接近 0）")

    print("\n3. 测试连续 kernel 模式:")
    config.use_continuous_kernel = True
    model = PatternEmbedding(config)
    Z = model(X, S1)
    print(f"   Output shape: {Z.shape}")

    print("\n4. 测试 get_canonical_coords + decode_spatial:")
    s_rel_true, centroid, V = model.get_canonical_coords(S1)
    s_restored = model.decode_spatial(s_rel_true, centroid, V)
    err2 = (S1 - s_restored).norm()
    print(f"   get_canonical_coords -> decode_spatial 还原误差: {err2.item():.8f}")

    print("\n5. 测试可微性:")
    S1.requires_grad = True
    Z = model(X, S1)
    Z.sum().backward()
    print(f"   Gradient exists: {S1.grad is not None}")
    print(f"   Gradient norm: {S1.grad.norm().item():.4f}")

    print("\n[OK] All tests passed!")
