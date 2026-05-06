"""
Main model: Z-space feedback control developmental system

Core equations of the new architecture:
    Encoding:  Z_i(t0) = Psi(X_i, S_rel_i)           S_rel = PCA-canonicalized relative coordinates
    Dynamics:  dZ_i/dt = g(Z_i) + B(M_i - Z_i)       purely in d-dimensional Z space (d=64)
    Decoding:  X_pred  = Phi_X(Z_i(t1))               gene decoding (main path)
               S_rel_pred = Phi_S(Z_i(t1))            spatial decoding (main path, canonical coordinates)


"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

from pattern_embedding import PatternEmbedding, pca_canonicalize, canonical_to_original
from dynamics import (
    PatternDynamics,
    PatternODEFunc,
    TargetPredictor,
    GeneDecoder,
    SpatialDecoder,
    ResidualBlock,
)


class FeedbackControlModel(nn.Module):

    def __init__(self, config, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.device = device

        # 1. Pattern embedding Psi: (X, S) -> Z (PCA canonicalization applied internally)
        self.latent_embedding = PatternEmbedding(config)

        # 2. Terminal state prediction M_theta: Z -> M_i
        self.target_predictor = TargetPredictor(config)

        # 3. Z-space dynamics: dZ/dt = g(Z) + B(M_i - Z)
        self.latent_dynamics = PatternDynamics(config)

        # 4. Decoders
        self.gene_decoder    = GeneDecoder(config)     # Z -> X
        self.spatial_decoder = SpatialDecoder(config)  # Z -> S_rel

        # 5. Feedback gain warmup (0.0 -> 1.0)
        self.register_buffer("feedback_gain", torch.tensor(1.0))

    # ------------------------------------------------------------------
    # Compatibility property
    # ------------------------------------------------------------------

    @property
    def beta_val(self) -> torch.Tensor:
        return torch.tensor(1.0, device=self.device or torch.device("cpu"))

    # ------------------------------------------------------------------
    # ODE function construction
    # ------------------------------------------------------------------

    def _get_ode_func(self) -> PatternODEFunc:
        return PatternODEFunc(
            latent_dynamics=self.latent_dynamics,
            target_predictor=self.target_predictor,
            feedback_gain=self.feedback_gain,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data_t0: Dict, data_t1: Dict,
                pi_cache: "Optional[Dict]" = None) -> Dict:
        device = data_t0["x"].device
        t0 = data_t0["t"]
        t1 = data_t1["t"]

        # ---- 1. Obtain canonical relative coordinates (used as ground truth for losses) ----
        # pca_canonicalize is also called inside PatternEmbedding.forward;
        # calling it here separately is to obtain S_rel_true for loss computation and recon supervision.
        S_rel_t0, _, _   = pca_canonicalize(data_t0["s"])  # (N0, 2)
        S_rel_t1, _, _   = pca_canonicalize(data_t1["s"])  # (N1, 2)

        # ---- 2. Encode pattern Z for t0/t1 ----
        Z_t0 = self.latent_embedding(
            data_t0["x"], data_t0["s"], data_t0.get("knn_indices")
        )  # (N0, d)
        Z_t1_true = self.latent_embedding(
            data_t1["x"], data_t1["s"], data_t1.get("knn_indices")
        )  # (N1, d)

        # ---- 3. Terminal state prediction ----
        M_i_t0 = self.target_predictor(Z_t0)   # (N0, d)

        # ---- 4. Z-space ODE integration (main prediction path) ----
        # New architecture: ODE integrates in d-dimensional Z space; no longer needs integration in high-dimensional (X,S) space
        Z_t1_pred = self._ode_integrate_Z(Z_t0, t0, t1)  # (N0, d)

        # ---- 6. Decode predictions (decoder is the main path) ----
        X_t1_pred     = self.gene_decoder(Z_t1_pred)     # (N0, G)
        S_rel_t1_pred = self.spatial_decoder(Z_t1_pred)  # (N0, 2)

        # ---- 7. Compute all losses ----
        losses = {}
        # t0, t1 are normalized times in [0,1]; the final time point corresponds to normalized value 1.0
        is_final = abs(t1.item() - 1.0) < 1e-3

        # (a) Main trajectory matching: Sinkhorn OT(Z_t1_pred, Z_t1_true) - geomloss entropy-regularized OT in latent space
        # BB endpoint condition: ODE transports mu_Z0 to mu_Z1; each predicted cell is pulled toward its matched true cell (cell-level gradient).
        #
        # Key design: Z_t0 is detached here so that Sinkhorn gradients only train ODE parameters, not the encoder.
        # Otherwise the encoder would take a shortcut: setting Z_t0 ~= Z_t1_true (ODE velocity->0),
        # creating adversarial oscillation with the diversity loss, causing trajectory to converge to ~0 while the ODE learns no dynamics.
        # encoder(X_t0) is indirectly aligned with the ODE via the recon_x_t1 path (encoder->ODE->decoder->X_t1).
        Z_t1_pred_for_traj = self._ode_integrate_Z(Z_t0.detach(), t0, t1)
        losses["trajectory"] = self._sinkhorn_latent(
            Z_t1_pred_for_traj,
            Z_t1_true.detach(),
            blur=getattr(self.config, "sinkhorn_blur", 0.05),
            reach=getattr(self.config, "sinkhorn_reach", 0.5),
        )

        # (b) Encoder-decoder cycle consistency (recon)
        # Denoising AE: inject random noise to train decoder smoothness, improving LOTO interpolation capability
        Z_t0_noisy = Z_t0 + torch.randn_like(Z_t0) * 0.05
        X_recon_t0          = self.gene_decoder(Z_t0_noisy)
        S_rel_recon_t0      = self.spatial_decoder(Z_t0_noisy)

        losses["recon_x"] = F.mse_loss(X_recon_t0, data_t0["x"])
        losses["recon_s"] = F.mse_loss(S_rel_recon_t0, S_rel_t0)

        # Avoid double-weighting intermediate time points: normally only supervise start point t0; only supervise end point t1 for the last time pair
        if is_final:
            Z_t1_noisy = Z_t1_true + torch.randn_like(Z_t1_true) * 0.05
            X_recon_t1_true     = self.gene_decoder(Z_t1_noisy)
            S_rel_recon_t1_true = self.spatial_decoder(Z_t1_noisy)

            losses["recon_x"] = 0.5 * (losses["recon_x"] + F.mse_loss(X_recon_t1_true, data_t1["x"]))
            losses["recon_s"] = 0.5 * (losses["recon_s"] + F.mse_loss(S_rel_recon_t1_true, S_rel_t1))

        losses["latent_smooth"] = torch.tensor(0.0, device=device)

        # (d) Equilibrium: at terminal time M_theta(Z_T) ~= Z_T
        if is_final:
            M_i_t1 = self.target_predictor(Z_t1_true)
            losses["equilibrium"] = F.mse_loss(M_i_t1, Z_t1_true.detach())
        else:
            losses["equilibrium"] = torch.tensor(0.0, device=device)

        # (g) Isometric embedding regularization
        # t0: true z, supervises overall isometry of encoder+decoder
        # t1: pred z (ODE output, already detached), only supervises isometry of decoder on ODE outputs
        # Symmetric with recon: true z covers encoder, pred z covers decoder performance on ODE distribution
        lam_iso = getattr(self.config, "lambda_isometric", 0.0)
        if lam_iso > 0:
            losses["isometric"] = 0.5 * (
                self._isometric_loss(Z_t0) +
                self._isometric_loss(Z_t1_pred_for_traj.detach())
            )
        else:
            losses["isometric"] = torch.tensor(0.0, device=device)

        losses["velocity"] = torch.tensor(0.0, device=device)
        losses["spatial_smooth"] = torch.tensor(0.0, device=device)

        # (e) Diversity: prevent latent collapse
        losses["diversity"] = (
            -Z_t0.std(dim=0).mean().clamp(max=10.0)
            if Z_t0.shape[0] > 1
            else torch.tensor(0.0, device=device)
        )

        # (h) Optional label-aware losses
        label_losses = self._compute_label_aware_losses(
            Z_t1_pred=Z_t1_pred,
            X_t1_pred=X_t1_pred,
            S_rel_t1_pred=S_rel_t1_pred,
            Z_t1_true=Z_t1_true.detach(),
            X_t1_true=data_t1["x"],
            S_rel_t1=S_rel_t1,
            labels_t1=data_t1.get("annotations"),
        )
        losses.update(label_losses)

        # ---- 8. Weighted total loss ----
        # Phase 1 (dynamics frozen, encoder trainable): recon / diversity / isometric active
        # Phase 2 (dynamics unfrozen, encoder trainable): all losses active, end-to-end joint training
        dynamics_on = float(next(self.latent_dynamics.parameters()).requires_grad)
        encoder_on  = float(next(self.latent_embedding.parameters()).requires_grad)

        total_loss = (
            # Phase 1 active (when encoder is trainable)
            encoder_on  * self.config.lambda_recon          * (losses["recon_x"] + losses["recon_s"]) +
            encoder_on  * self.config.lambda_diversity      * losses["diversity"]      +
            encoder_on  * getattr(self.config, "lambda_isometric", 0.0) * losses["isometric"] +
            # Phase 2 active (after dynamics unfrozen, depends on ODE)
            dynamics_on * self.config.lambda_trajectory      * losses["trajectory"]     +
            dynamics_on * self.config.lambda_equilibrium     * losses["equilibrium"]    +
            dynamics_on * getattr(self.config, "lambda_label_trajectory", 0.0) * losses["label_trajectory"] +
            dynamics_on * getattr(self.config, "lambda_label_proportion", 0.0) * losses["label_proportion"] +
            dynamics_on * getattr(self.config, "lambda_label_centroid",   0.0) * losses["label_centroid"]
        )

        # NaN/Inf detection (automatically skip anomalous batches during training)
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            sub = {k: f"{v.item():.4f}" for k, v in losses.items()
                   if isinstance(v, torch.Tensor)}
            import warnings
            warnings.warn(f"NaN/Inf loss detected: {sub}")
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        losses["total"] = total_loss

        return {
            "loss":   total_loss,
            "losses": losses,
            "predictions": {
                "X_t1":     X_t1_pred,
                "S_t1":     S_rel_t1_pred,   # canonical coordinates; call decode_spatial() to restore during evaluation
                "S_rel_t1": S_rel_t1_pred,
                "Z_t1":     Z_t1_pred,
            },
            "latents": {
                "Z_t0":      Z_t0.mean(dim=0),
                "Z_t1_true": Z_t1_true.mean(dim=0),
                "Z_t1_pred": Z_t1_pred.mean(dim=0),
                "M_i_mean":  M_i_t0.mean(dim=0),
                "beta":      self.beta_val,
            },
        }

    # ------------------------------------------------------------------
    # Phase 1 warmup forward (single time point, only per-timepoint losses)
    # ------------------------------------------------------------------

    def forward_warmup(self, data_t: Dict) -> Dict:
        """Phase 1 warmup: compute recon + isometric + diversity for a single time point.

        The training loop iterates over all time points (not time pairs) during warmup,
        with each time point appearing exactly once to avoid double-counting intermediate points.
        Dynamics parameters are frozen at this stage; ODE-related losses are not needed.
        """
        device = data_t["x"].device
        S_rel, _, _ = pca_canonicalize(data_t["s"])
        Z = self.latent_embedding(data_t["x"], data_t["s"], data_t.get("knn_indices"))

        losses: Dict[str, torch.Tensor] = {}

        # Denoising AE: inject pseudo-noise to improve decoder smoothness and generalization
        Z_noisy = Z + torch.randn_like(Z) * 0.05
        losses["recon_x"] = F.mse_loss(self.gene_decoder(Z_noisy), data_t["x"])
        losses["recon_s"] = F.mse_loss(self.spatial_decoder(Z_noisy), S_rel)

        lam_iso = getattr(self.config, "lambda_isometric", 0.0)
        losses["isometric"] = (
            self._isometric_loss(Z) if lam_iso > 0
            else torch.tensor(0.0, device=device)
        )

        losses["diversity"] = (
            -Z.std(dim=0).mean().clamp(max=10.0)
            if Z.shape[0] > 1
            else torch.tensor(0.0, device=device)
        )

        # Set all dynamic losses to 0 (not computed during warmup; keeps logging keys consistent)
        for key in ("trajectory", "velocity", "equilibrium",
                    "latent_smooth", "spatial_smooth"):
            losses[key] = torch.tensor(0.0, device=device)

        total_loss = (
            self.config.lambda_recon     * (losses["recon_x"] + losses["recon_s"]) +
            self.config.lambda_diversity * losses["diversity"] +
            lam_iso                      * losses["isometric"]
        )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            sub = {k: f"{v.item():.4f}" for k, v in losses.items()
                   if isinstance(v, torch.Tensor)}
            import warnings
            warnings.warn(f"NaN/Inf loss in warmup: {sub}")
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        losses["total"] = total_loss
        return {"loss": total_loss, "losses": losses}

    # ------------------------------------------------------------------
    # Z-space ODE integration
    # ------------------------------------------------------------------

    def _ode_integrate_Z(self, Z0: torch.Tensor, t0, t1) -> torch.Tensor:
        """
        Integrate ODE in d-dimensional Z space: dZ/dt = g(Z) + B(M(Z(t)) - Z(t))

        M is dynamically predicted at each ODE step by target_predictor(Z(t)),
        allowing different cells to follow different fates (per-cell attractors), preserving diversity in Z space.
        The ODE dimensionality in the new architecture is d (64), not G+2 (thousands) as in the old architecture.
        """
        ode_fn = self._get_ode_func()
        solver = self.config.ode_solver
        if solver == "euler":
            return self._euler_step(ode_fn, Z0, t0, t1)
        elif solver == "rk4":
            return self._rk4_step(ode_fn, Z0, t0, t1)
        else:
            return self._dopri5_step(ode_fn, Z0, t0, t1)

    # ------------------------------------------------------------------
    # ODE Solvers (general purpose, state can be a tensor of any shape)
    # ------------------------------------------------------------------

    def _euler_step(self, fn, state0, t0, t1):
        dt = self._get_dt(t0, t1)
        return state0 + dt * fn(self._to_tensor(t0, state0.device), state0)

    def _rk4_step(self, fn, state0, t0, t1):
        dt  = self._get_dt(t0, t1)
        _t0 = self._to_tensor(t0, state0.device)
        k1 = fn(_t0,          state0)
        k2 = fn(_t0 + dt / 2, state0 + dt / 2 * k1)
        k3 = fn(_t0 + dt / 2, state0 + dt / 2 * k2)
        k4 = fn(_t0 + dt,     state0 + dt     * k3)
        return state0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _dopri5_step(self, fn, state0, t0, t1):
        try:
            from torchdiffeq import odeint, odeint_adjoint
        except ImportError:
            return self._rk4_step(fn, state0, t0, t1)
        solver_fn = odeint_adjoint if self.config.ode_adjoint else odeint
        t = torch.stack([self._to_tensor(t0, state0.device),
                         self._to_tensor(t1, state0.device)])
        states = solver_fn(fn, state0, t,
                           rtol=self.config.ode_rtol,
                           atol=self.config.ode_atol,
                           method="dopri5")
        return states[-1]

    def _get_dt(self, t0, t1) -> float:
        if isinstance(t0, torch.Tensor): t0 = t0.item()
        if isinstance(t1, torch.Tensor): t1 = t1.item()
        return float(t1 - t0)

    def _to_tensor(self, t, device):
        if isinstance(t, torch.Tensor):
            return t.to(device)
        return torch.tensor(t, dtype=torch.float32, device=device)

    def _isometric_loss(self, Z: torch.Tensor) -> torch.Tensor:
        """Isometric embedding regularization: penalizes inconsistent local scaling ratios of the decoder across directions/positions.

        Goal: J_Psi^T J_Psi = lambda^2 * I_d (isotropic scaling).
        Measured by the coefficient of variation of per-sample local scaling ratios r_i = ||Psi(Z+eps)-Psi(Z)||^2 / ||eps||^2:
          CV = std(r) / mean(r)
        CV=0 indicates isotropy (consistent scaling across all samples/directions), which cannot be trivially satisfied by global rescaling.

        Old implementation estimated lambda^2 using batch mean then subtracted; mean can be arbitrarily small, making loss trivially 0.
        New implementation directly measures dispersion of scaling ratios, scale-invariant, unaffected by global scaling.
        """
        sigma = getattr(self.config, "isometric_sigma", 0.05)
        gamma = getattr(self.config, "isometric_gamma", 1.0)

        eps       = torch.randn_like(Z) * sigma     # (N, d)
        Z_perturb = Z + eps

        X_orig    = self.gene_decoder(Z)
        X_perturb = self.gene_decoder(Z_perturb)
        S_orig    = self.spatial_decoder(Z)
        S_perturb = self.spatial_decoder(Z_perturb)

        dx2       = ((X_perturb - X_orig) ** 2).sum(dim=-1)        # (N,)
        ds2       = ((S_perturb - S_orig) ** 2).sum(dim=-1)        # (N,)
        psi_dist2 = dx2 + (gamma ** 2) * ds2                       # (N,)
        eps_norm2 = (eps ** 2).sum(dim=-1)                          # (N,)

        # Local scaling ratio per sample (should be equal everywhere under isotropy)
        ratio     = psi_dist2 / eps_norm2.clamp(min=1e-8)          # (N,)

        # Coefficient of variation: std/mean, scale-invariant, measures degree of anisotropy
        mean_r    = ratio.mean().clamp(min=1e-8)
        return ratio.var() / (mean_r ** 2)

    @staticmethod
    def _sinkhorn_latent(Z_pred: torch.Tensor, Z_true: torch.Tensor,
                         blur: Optional[float] = 0.05,
                         reach: Optional[float] = 0.5) -> torch.Tensor:
        """Sinkhorn (unbalanced) OT in latent Z space (geomloss).

        Gradient approximation: grad_{Z_i} L ~= 2(Z_i - sum_j T_ij Z_j_true)
        Each predicted cell is pulled toward its matched true cell, providing cell-level fate guidance.

        reach=tau (unbalanced strength): None = balanced OT; float = unbalanced OT, allows mass creation/annihilation.
        In cell differentiation settings, t1 cell type proportions differ from t0; unbalanced OT amplifies gradient signals for rare types.
        blur = epsilon (entropy regularization strength): smaller means more precise matching, larger means smoother; recommended range 0.03~0.1.
        """
        # Input contains NaN/Inf (occasionally occurs during gradient explosion) -> clamp and continue
        Z_pred = torch.nan_to_num(Z_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        Z_true = torch.nan_to_num(Z_true, nan=0.0, posinf=1e4, neginf=-1e4)

        # geomloss internally uses log(diameter) to compute epsilon schedule;
        # diameter=0 (when Z_pred~=Z_true early in ODE training) or blur=None both cause log(0)=-inf
        # -> np.arange(-inf, ...) -> "cannot compute length"
        with torch.no_grad():
            diameter = torch.cdist(Z_pred.detach(), Z_true.detach()).max().item()
        if diameter < 1e-6:
            # Two distributions nearly identical, treat loss as 0, skip geomloss
            return torch.tensor(0.0, device=Z_pred.device, requires_grad=True)

        if blur is None:
            blur = max(diameter * 0.1, 1e-3)

        from geomloss import SamplesLoss
        loss_fn = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=0.9,
                              backend="auto", debias=True, reach=reach)
        return loss_fn(Z_pred, Z_true)

    # ------------------------------------------------------------------
    # Decoder warmup: freeze / unfreeze dynamics network
    # ------------------------------------------------------------------

    def freeze_dynamics(self):
        """Phase 1: freeze the dynamics network, train only the encoder-decoder."""
        for p in self.latent_dynamics.parameters():
            p.requires_grad_(False)
        for p in self.target_predictor.parameters():
            p.requires_grad_(False)

    def unfreeze_dynamics(self):
        """Phase 2: unfreeze the dynamics network (latent_dynamics + target_predictor)."""
        for p in self.latent_dynamics.parameters():
            p.requires_grad_(True)
        for p in self.target_predictor.parameters():
            p.requires_grad_(True)

    def freeze_encoder_decoder(self):
        """Phase 2: freeze encoder + decoder, train only the ODE.
        Fixes the latent space coordinate system, eliminating the moving-target problem."""
        for p in self.latent_embedding.parameters():
            p.requires_grad_(False)
        for p in self.gene_decoder.parameters():
            p.requires_grad_(False)
        for p in self.spatial_decoder.parameters():
            p.requires_grad_(False)

    def is_dynamics_frozen(self) -> bool:
        return not next(self.latent_dynamics.parameters()).requires_grad

    def is_encoder_frozen(self) -> bool:
        return not next(self.latent_embedding.parameters()).requires_grad

    def _compute_label_aware_losses(
        self,
        Z_t1_pred: torch.Tensor,
        X_t1_pred: torch.Tensor,
        S_rel_t1_pred: torch.Tensor,
        Z_t1_true: torch.Tensor,
        X_t1_true: torch.Tensor,
        S_rel_t1: torch.Tensor,
        labels_t1,
    ) -> Dict[str, torch.Tensor]:
        device = Z_t1_pred.device
        zero = torch.tensor(0.0, device=device)
        out = {
            "label_trajectory": zero,
            "label_proportion": zero,
            "label_centroid": zero,
        }

        if not getattr(self.config, "use_label_aware_losses", False):
            return out
        if labels_t1 is None:
            return out
        if len(labels_t1) == 0 or Z_t1_true.shape[0] == 0 or Z_t1_pred.shape[0] == 0:
            return out

        labels_true_np = np.asarray(labels_t1)
        min_cells = int(getattr(self.config, "label_min_cells", 8))

        with torch.no_grad():
            nn_idx = torch.cdist(Z_t1_pred.detach(), Z_t1_true.detach(), p=2).argmin(dim=1)
            pseudo_labels_np = labels_true_np[nn_idx.detach().cpu().numpy()]

        valid_labels = []
        for label in np.unique(labels_true_np):
            true_count = int(np.sum(labels_true_np == label))
            pred_count = int(np.sum(pseudo_labels_np == label))
            if true_count >= min_cells and pred_count >= min_cells:
                valid_labels.append(label)

        if not valid_labels:
            return out

        traj_terms = []
        centroid_terms = []
        true_props = []
        pred_props = []
        metric = getattr(self.config, "label_distance_metric", "fgw")

        for label in valid_labels:
            true_mask_np = labels_true_np == label
            pred_mask_np = pseudo_labels_np == label

            true_mask = torch.from_numpy(true_mask_np).to(device=device, dtype=torch.bool)
            pred_mask = torch.from_numpy(pred_mask_np).to(device=device, dtype=torch.bool)

            X_true_g = X_t1_true[true_mask]
            S_true_g = S_rel_t1[true_mask]
            Z_true_g = Z_t1_true[true_mask]
            X_pred_g = X_t1_pred[pred_mask]
            S_pred_g = S_rel_t1_pred[pred_mask]
            Z_pred_g = Z_t1_pred[pred_mask]

            traj_terms.append(F.mse_loss(X_pred_g.mean(dim=0), X_true_g.mean(dim=0)))

            centroid_terms.append(F.mse_loss(Z_pred_g.mean(dim=0), Z_true_g.mean(dim=0)))
            true_props.append(float(true_mask_np.mean()))
            pred_props.append(float(pred_mask_np.mean()))

        if traj_terms:
            out["label_trajectory"] = torch.stack(traj_terms).mean()
        if centroid_terms:
            out["label_centroid"] = torch.stack(centroid_terms).mean()

        true_props_t = torch.tensor(true_props, device=device, dtype=Z_t1_pred.dtype)
        pred_props_t = torch.tensor(pred_props, device=device, dtype=Z_t1_pred.dtype)
        out["label_proportion"] = torch.abs(true_props_t - pred_props_t).mean()
        return out

    def initialize_M_from_data(self, final_data: Dict):
        """Initialize prototype attractors from the Z distribution at the final time point (K-means clustering).

        Encodes all cells at the final time point to Z and runs K-means,
        then initializes target_predictor.prototypes with the K cluster centers.
        Converges faster than random initialization since prototypes start distributed on the true data manifold.
        """
        self.eval()
        n_total    = final_data["x"].shape[0]
        batch_size = 256
        Z_list     = []

        with torch.no_grad():
            for i in range(0, n_total, batch_size):
                bx   = final_data["x"][i:i+batch_size].to(self.device)
                bs   = final_data["s"][i:i+batch_size].to(self.device)
                bknn = (final_data["knn_indices"][i:i+batch_size].to(self.device)
                        if final_data.get("knn_indices") is not None else None)
                Z_list.append(self.latent_embedding(bx, bs, bknn))

        Z_all = torch.cat(Z_list, dim=0).cpu()   # (N, d)
        K     = self.target_predictor.prototypes.shape[0]

        # K-means prototype initialization (sklearn, runs once on CPU)
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=0,
                             batch_size=min(4096, n_total))
        km.fit(Z_all.numpy())
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)

        with torch.no_grad():
            self.target_predictor.prototypes.copy_(centers.to(self.device))
        print(f"  PrototypeTargetPredictor initialized: K={K}, "
              f"center_mean={centers.mean().item():.4f}  std={centers.std().item():.4f}")

    # ------------------------------------------------------------------
    # Trajectory prediction (inference)
    # ------------------------------------------------------------------

    def predict_trajectory(
        self,
        X0: torch.Tensor,
        S0: torch.Tensor,
        time_points: List[float],
        knn_indices=None,
    ) -> List[Dict]:
        """
        Predict trajectory at multiple time points starting from (X0, S0).

        Returns:
            A list of dicts, one per time point, containing:
                t, Z (latent), X (genes), S (original coordinates), S_rel (canonical coordinates)
        """
        self.eval()
        with torch.no_grad():
            trajectory = []
            Z_curr = self.latent_embedding(X0, S0, knn_indices)
            # Save t0 PCA parameters for coordinate restoration
            S_rel_0, centroid_0, V_0 = pca_canonicalize(S0)

            trajectory.append({
                "t":     time_points[0],
                "Z":     Z_curr.mean(0),
                "X":     X0,
                "S":     S0,
                "S_rel": S_rel_0,
            })

            t_curr = time_points[0]
            for t_next in time_points[1:]:
                Z_next = self._ode_integrate_Z(
                    Z_curr,
                    self._to_tensor(t_curr, X0.device),
                    self._to_tensor(t_next, X0.device),
                )
                X_next     = self.gene_decoder(Z_next)
                S_rel_next = self.spatial_decoder(Z_next)
                # Restore coordinates using t0 slice PCA parameters (maintains coordinate system continuity)
                S_next     = canonical_to_original(S_rel_next, centroid_0, V_0)

                trajectory.append({
                    "t":     t_next,
                    "Z":     Z_next.mean(0),
                    "X":     X_next,
                    "S":     S_next,
                    "S_rel": S_rel_next,
                })
                Z_curr = Z_next
                t_curr = t_next

        return trajectory

    # ------------------------------------------------------------------
    # Single-step prediction (for evaluate.py / LOTO)
    # ------------------------------------------------------------------

    def predict_step(
        self,
        X0: torch.Tensor,
        S0: torch.Tensor,
        t0: float,
        t1: float,
        knn_indices=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step prediction: (X0, S0, t0) -> (Z1, X1, S1)

        Returns:
            Z1:  (N, d) latent
            X1:  (N, G) gene expression (in original coordinate scale)
            S1:  (N, 2) spatial coordinates (restored to t0 slice coordinate system)
        """
        self.eval()
        with torch.no_grad():
            Z0 = self.latent_embedding(X0, S0, knn_indices)
            _, centroid, V = pca_canonicalize(S0)
            Z1         = self._ode_integrate_Z(
                Z0,
                self._to_tensor(t0, X0.device),
                self._to_tensor(t1, X0.device),
            )
            X1     = self.gene_decoder(Z1)
            S_rel1 = self.spatial_decoder(Z1)
            S1     = canonical_to_original(S_rel1, centroid, V)
        return Z1, X1, S1

    # ------------------------------------------------------------------
    # k-NN helper
    # ------------------------------------------------------------------

    def _compute_knn_kdtree(self, coords: torch.Tensor, k: int = None):
        from sklearn.neighbors import NearestNeighbors
        if k is None:
            k = self.config.lag_knn
        coords_np = coords.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(coords_np)
        _, idx = nbrs.kneighbors(coords_np)
        return torch.from_numpy(idx[:, 1:]).long()
