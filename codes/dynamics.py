"""
Dynamics module - Z-space feedback control model

Core ODE (integrated in low-dimensional latent space Z):

    Z_i(t0)  = Psi(X_i, S_rel_i)              Encoding: genes + relative spatial position -> latent
    M_i      = M_theta(Z_i)                    Per-cell terminal state prediction
    dZ_i/dt  = g(Z_i) + B(M_i - Z_i)          Z-space dynamics (core of this file)
    X_i_pred = Phi_X(Z_i(t1))                 Decoding: latent -> gene expression
    S_rel_pred = Phi_S(Z_i(t1))               Decoding: latent -> relative spatial coordinates

Key differences from old architecture:
    Old: ODE integrates in joint (X, S) space, dimension G+2 (thousands), requires calling PatternEmbedding at each step
    New: ODE integrates in Z space, dimension d (64), only requires calling PatternDynamics MLP at each step
         Advantages: computation greatly reduced, shorter gradient paths, coordinate system issues resolved via relative encoding inside the encoder

Coordinate system design:
    S_rel = PCA-canonicalized relative coordinates (offset from within-slice centroid, principal axis aligned)
    - Translation invariant: absolute position differences across slices are eliminated
    - Rotation/flip invariant: PCA principal axis directions are fixed (element with largest absolute value is positive)
    - Cross-timepoint comparable: S_rel of two slices are in the same canonical space, allowing direct Wasserstein computation
    Restored to original coordinate system via inverse transform of SpatialDecoder during evaluation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================
# Helper blocks
# ============================================================

class ResidualBlock(nn.Module):
    """Residual connection block with LayerNorm"""
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(x + self.block(x)))


# ============================================================
# Z-space dynamics  dZ/dt = g(Z) + B(M - Z)
# ============================================================

class PatternDynamics(nn.Module):
    """
    dZ_i/dt = g(Z_i) + B(M_i - Z_i)

    Strict linear decomposition (same structure as old architecture GeneExpressionDynamics, but input/output both d-dimensional):
        g: MLP(Z) -> R^d     intrinsic velocity field, depends only on current latent
        B: Linear(d -> d, bias=False)  feedback gain matrix

    Advantages (compared to old integration in (X,S) space):
        - ODE dimension reduced from G+2 (thousands) to d (64), per-step cost O(d^2) vs O(G*d)
        - No need to re-call full PatternEmbedding at each integration step
        - Gradient path: trajectory loss -> decoder -> Z_pred -> PatternDynamics
          Path is clear, avoids numerical instability of high-dimensional (X,S) space

    Identifiability:
        At Z(i,t1) = Z(j,t2) = z with M_i != M_j:
            dZ_i/dt - dZ_j/dt = B(M_i - M_j)
        -> B can be uniquely determined from velocity differences; substituting back, g(z) is uniquely determined
    """
    def __init__(self, config):
        super().__init__()
        d = config.latent_dim
        h = config.dynamics_hidden_dim

        # Intrinsic velocity field g(Z): MLP, depends only on Z
        layers: list = [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
        for _ in range(config.dynamics_num_layers - 1):
            if getattr(config, "use_residual", True):
                layers.append(ResidualBlock(h))
            else:
                layers.extend([nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU()])
        layers.append(nn.Linear(h, d))
        self.g = nn.Sequential(*layers)

        # Feedback gain matrix B: R^{dxd}, no bias (theory requires pure linearity)
        self.B = nn.Linear(d, d, bias=False)

        # Initialization: feedback term starts weak to prevent ODE from blowing up early in training
        nn.init.normal_(self.B.weight, std=0.01)

    def forward(self, Z: torch.Tensor, M_i: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """
        Args:
            Z:    (N, d) current latent
            M_i:  (N, d) terminal latent prediction
            gain: feedback gain warmup coefficient (linearly increases from 0 to 1 at training start)
        Returns:
            dZ/dt: (N, d)
        """
        dZ = self.g(Z) + gain * self.B(M_i - Z)
        # Velocity magnitude protection: prevents ODE divergence early in training; placed in dynamics layer rather than solver
        return dZ.clamp(-1e2, 1e2)


class TargetPredictor(nn.Module):
    """
    Prototype attractor version of TargetPredictor.

    K learnable prototypes P_k in R^d; the target M_i for each cell Z_i is a softmax-weighted combination of prototypes:

        a_k(Z_i) = softmax(-||Z_i - P_k||^2 / tau)_k
        M_i      = sum_k a_k(Z_i) * P_k

    Advantages:
      - K prototypes correspond to K cell fates; each cell is pulled toward the nearest prototype (multi-attractor structure)
      - equilibrium loss directly optimizes positions of K prototypes, denser and more stable gradients than a free MLP
      - Different cell types are naturally assigned to different prototypes, improving ct_centroid_corr

    tau (temperature) controls soft/hard assignment: tau->0 is hard assignment (nearest prototype), tau->infinity is uniform mixture.
    """
    def __init__(self, config):
        super().__init__()
        d = config.latent_dim
        K = getattr(config, "n_prototypes", 16)
        tau = getattr(config, "prototype_temp", 1.0)
        self.register_buffer("log_tau", torch.tensor(tau).log())
        self.prototypes = nn.Parameter(torch.randn(K, d) * 0.1)  # (K, d)

    @property
    def tau(self):
        return self.log_tau.exp()

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (N, d)  prototypes: (K, d)
        # dist2[i, k] = ||Z_i - P_k||^2  (squared distance)
        dist2 = torch.cdist(Z, self.prototypes, p=2).pow(2)       # (N, K)
        weights = F.softmax(-dist2 / self.tau, dim=1)              # (N, K)
        return weights @ self.prototypes                           # (N, d)


# ============================================================
# Z-space ODE function
# ============================================================

class PatternODEFunc(nn.Module):
    """
    ODE right-hand side: dZ/dt = g(Z) + B(M_theta(Z) - Z)

    state: (N, d)  pure Z space, no need to call PatternEmbedding at each step

    Compared to old CoupledODEFunc (in (X,S) space):
        Old: state (N, G+2), each forward step requires PatternEmbedding(X, S) + two dynamics networks
        New: state (N, d), each forward step only requires PatternDynamics, greatly reducing computation
    """
    def __init__(
        self,
        latent_dynamics: PatternDynamics,
        target_predictor: TargetPredictor,
        feedback_gain: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.latent_dynamics = latent_dynamics
        self.target_predictor = target_predictor
        self.feedback_gain    = feedback_gain
        self.M_i              = None

    def forward(self, t: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        if self.M_i is None:
            M_i = self.target_predictor(Z)
        else:
            M_i = self.M_i
        gain = self.feedback_gain if self.feedback_gain is not None else 1.0
        return self.latent_dynamics(Z, M_i, gain=gain)


# ============================================================
# Decoder: Z -> X (main path)
# ============================================================

class GeneDecoder(nn.Module):
    """
    Phi_X: R^d -> R^G

    Upgraded to the main prediction path in the new architecture (was auxiliary supervision in the old architecture).
    Gradients from the trajectory loss are backpropagated through this decoder to PatternDynamics.
    """
    def __init__(self, config):
        super().__init__()
        d = config.latent_dim
        h = config.decoder_hidden_dim
        G = config.n_genes

        layers: list = [nn.Linear(d, h), nn.ReLU()]
        for _ in range(config.decoder_num_layers - 1):
            layers.extend([nn.Linear(h, h), nn.ReLU()])
        layers.append(nn.Linear(h, G))
        self.net = nn.Sequential(*layers)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.net(Z)


# ============================================================
# Decoder: Z -> S_rel (main path)
# ============================================================

class SpatialDecoder(nn.Module):
    """
    Phi_S: R^d -> R^2

    Output is canonical relative coordinates S_rel (PCA-aligned coordinates, comparable across slices).
    No Sigmoid - S_rel range after PCA canonicalization is approximately (-3, 3), linear output covers this range.

    Restoration pipeline during evaluation:
        S_rel_pred -> S_original = S_rel_pred * pca_scale + centroid
        See PatternEmbedding.decode_spatial() for implementation details
    """
    def __init__(self, config):
        super().__init__()
        d = config.latent_dim
        h = config.decoder_hidden_dim // 2

        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 2),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.net(Z)



# ============================================================
# test
# ============================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from config import Config

    print("Testing dynamics module (Z-space formulation)...")
    config = Config()
    config.n_genes = 100
    config.latent_dim = 32
    N = 50

    dyn      = PatternDynamics(config)
    tgt_pred = TargetPredictor(config)
    ode_fn   = PatternODEFunc(dyn, tgt_pred)
    gene_dec = GeneDecoder(config)
    spat_dec = SpatialDecoder(config)

    Z   = torch.randn(N, config.latent_dim)
    M_i = tgt_pred(Z)
    dZ  = dyn(Z, M_i)
    print(f"  PatternDynamics:  Z{list(Z.shape)}, M{list(M_i.shape)} -> dZ{list(dZ.shape)}")

    t   = torch.tensor(0.0)
    dZ2 = ode_fn(t, Z)
    print(f"  PatternODEFunc:   Z{list(Z.shape)} -> dZ{list(dZ2.shape)}")

    X_pred = gene_dec(Z)
    S_pred = spat_dec(Z)
    print(f"  GeneDecoder:      Z{list(Z.shape)} -> X{list(X_pred.shape)}")
    print(f"  SpatialDecoder:   Z{list(Z.shape)} -> S_rel{list(S_pred.shape)}")

    loss = dZ.sum() + X_pred.sum() + S_pred.sum()
    loss.backward()
    print("  backward() OK")
    print("[OK] All dynamics tests passed!")
