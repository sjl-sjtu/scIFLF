"""
Configuration for the feedback-control developmental model.

Preprocessing assumptions:
  X: CPM -> log1p -> z-score, so model inputs have mean near 0 and std near 1.
  S: global min-max scaling -> [0, 1] -> z-score, so model inputs have mean near 0 and std near 1.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:

    # ================================================================
    # Data
    # ================================================================
    n_genes: Optional[int] = None  # None uses all genes; an integer selects top-N HVGs.
    num_workers: int = 4
    lag_knn: int = 10  # Number of kNN neighbors used by _compute_knn_kdtree.
    batch_size: int = 512  # Shared mini-batch size for warmup and random subsampling.
    steps_per_epoch: Optional[int] = None  # None uses N_max // batch_size.

    # Sinkhorn OT parameters.
    sinkhorn_blur: Optional[float] = 0.1  # Entropy regularization; smaller is sharper, larger is smoother.
    sinkhorn_reach: Optional[float] = 1.0 # 0.5
    # Unbalanced OT marginal relaxation. None gives balanced OT; smaller values allow more mass variation.

    # ================================================================
    # Model Architecture
    # ================================================================

    # ---------- Pattern Embedding Psi(X, S) -> Z ----------
    latent_dim: int = 128  # Latent pattern-space dimension.

    # Spatial encoder
    spatial_encoding: str = "relative"     # "relative" | "absolute"
    relative_k: int = 20  # Neighbor count for discrete kNN mode.
    use_continuous_kernel: bool = True
    kernel_bandwidth: float = 0.1  # Initial learnable continuous-kernel bandwidth.
    anchor_ratio: Optional[float] = 0.1  # Anchor count is anchor_ratio x 100; None uses num_anchors.
    num_anchors: int = 256
    use_edge_features: bool = True  # Discrete mode concatenates RBF and angular edge features.

    # GNN
    use_spatial_gnn: bool = True
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3

    # Dynamics network in Z space: dZ/dt = g(Z) + B(M - Z).
    dynamics_hidden_dim: int = 256
    dynamics_num_layers: int = 4
    use_residual: bool = True

    # Decoders form the main prediction path from latent states back to genes and space.
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 3

    # Prototype attractor target predictor.
    n_prototypes: int = 64  # Number of cell-fate prototypes.
    prototype_temp: float = 1.0  # Softmax temperature for prototype assignment.

    M_init: str = "final"  # "final" | "random" | "zeros"

    # ================================================================
    # Training
    # ================================================================
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 200
    scheduler: str = "cosine"       # "cosine" | "step" | "plateau"
    gradient_clip: float = 1.0

    decoder_warmup_epochs: int = 40
    # Phase 1 freezes PatternDynamics and TargetPredictor while training encoder/decoder.
    # Set to 0 to skip warmup and train end-to-end immediately.

    feedback_warmup_epochs: int = 20
    # Phase 2 linearly ramps feedback_gain from 0 to 1 after dynamics are unfrozen.
    # Set to 0 to use feedback_gain=1 immediately.

    early_stopping: bool = False
    early_stopping_patience: int = 20  # Allowed epochs without validation improvement after warmup.
    early_stopping_min_delta: float = 0  # Minimum validation loss drop counted as improvement.

    # ================================================================
    # Loss Weights
    # ================================================================
    lambda_trajectory: float = 1.0
    # Main endpoint-matching signal in latent Z space.

    lambda_equilibrium: float = 0.3
    # Terminal-time equilibrium penalty M_theta(Z_T) ~= Z_T.

    lambda_recon: float = 1.0
    # 0.5 x [MSE(X_recon_t0, X_t0) + MSE(X_recon_t1, X_t1)]
    #     + 0.5 x [MSE(S_rel_recon_t0, S_rel_t0) + MSE(S_rel_recon_t1, S_rel_t1)]

    lambda_diversity: float = 0.05
    # Negative std(Z_t0) term that discourages latent collapse.

    lambda_isometric: float = 0.01
    # Local decoder isometry regularization.
    isometric_sigma: float = 0.05
    isometric_gamma: float = 5.0

    # Optional label-aware training losses, disabled by default for fair baseline comparisons.
    use_label_aware_losses: bool = False
    label_min_cells: int = 8
    label_distance_metric: str = "feat"
    lambda_label_trajectory: float = 0.0
    lambda_label_proportion: float = 0.0
    lambda_label_centroid: float = 0.0

    # ================================================================
    # ODE Solver
    # ================================================================
    ode_solver: str = "rk4"         # "euler" | "rk4" | "dopri5"
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4
    ode_adjoint: bool = False

    # ================================================================
    # Data Augmentation
    # ================================================================
    use_augmentation: bool = True
    augment_rotation: bool = True
    augment_flip: bool = True
    augment_noise: float = 0.01

    # ================================================================
    # Experiment Management
    # ================================================================
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    save_interval: int = 20
    save_checkpoints: bool = True
    save_final_model: bool = True
    save_tensorboard: bool = True

    # ================================================================
    # Validation
    # ================================================================
    def __post_init__(self):
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.spatial_encoding in ("relative", "absolute"), \
            f"spatial_encoding must be 'relative' or 'absolute', got {self.spatial_encoding!r}"
        assert self.M_init in ("final", "random", "zeros"), \
            f"M_init must be 'final'/'random'/'zeros', got {self.M_init!r}"
        assert self.ode_solver in ("euler", "rk4", "dopri5"), \
            f"ode_solver must be 'euler'/'rk4'/'dopri5', got {self.ode_solver!r}"
        assert self.label_distance_metric in ("feat",), \
            f"label_distance_metric must be 'feat', got {self.label_distance_metric!r}"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.steps_per_epoch is None or self.steps_per_epoch > 0, \
            "steps_per_epoch must be positive or None"
        assert self.sinkhorn_blur is None or 0.0 < self.sinkhorn_blur < 1.0, \
            f"sinkhorn_blur must be in (0, 1) or None, got {self.sinkhorn_blur}"

    def load(self, path: str):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value)
