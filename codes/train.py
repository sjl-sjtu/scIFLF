"""
Training script - new feedback control model
Structure consistent with control_development/train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional

from config import Config
from model import FeedbackControlModel
from data_loader import SpatialTranscriptomicsLoader
from pattern_embedding import pca_canonicalize


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = SpatialTranscriptomicsLoader(config)

        self._set_seed(config.seed)
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.save_checkpoints = config.save_checkpoints
        self.save_final_model = config.save_final_model
        self.save_tensorboard = config.save_tensorboard
        print(f"Using device: {self.device}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self,
        slice_paths: List[str],
        time_points: List[float],
        annotation_key: Optional[str] = None,
        layer: Optional[str] = None,
        need_normalize: bool = False,
        need_log_transform: bool = False,
        need_scale: bool = False,
        spatial_layer: Optional[str] = None,
        n_genes: Optional[int] = None,
    ):
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        self.data_loader.load_from_anndata(
            slice_paths, time_points,
            annotation_key=annotation_key,
            layer=layer,
            need_normalize=need_normalize,
            need_log_transform=need_log_transform,
            need_scale=need_scale,
            spatial_layer=spatial_layer,
            n_genes=n_genes,
        )

        sample_data = self.data_loader.get_full_data(0)
        self.data_loader.n_genes = sample_data["x"].shape[1]
        self.config.n_genes = self.data_loader.n_genes
        self.config.time_points = sorted(time_points)

        # Time normalization parameters: [t_min, t_max] -> [0, 1]
        # The ODE integration step dt is directly determined by raw time differences, which may range from 1 to 100.
        # After normalization all dt values are in (0,1], ODE numerical behavior is consistent, gradients are more stable.
        # time_point in data_loader keeps the original value (used for logging/evaluation);
        # normalization is applied only before passing to model.forward.
        t_all = sorted(time_points)
        self.t_min = float(min(t_all))
        self.t_max = float(max(t_all))
        t_range = self.t_max - self.t_min
        self.t_scale = t_range if t_range > 1e-8 else 1.0
        # Normalized time point list (used by is_final and similar logic in model.forward)
        self.config.time_points_norm = [
            (t - self.t_min) / self.t_scale for t in t_all
        ]
        print(f"[OK] Time normalization: [{self.t_min}, {self.t_max}] -> [0, 1] "
              f"(scale={self.t_scale:.3f})")

        self._initialize_models()

        print(f"\n[OK] Data loaded: {len(self.data_loader.datasets)} slices")
        print(f"[OK] n_genes = {self.config.n_genes}")
        print(f"[OK] Models initialized")

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def _initialize_models(self):
        self.model = FeedbackControlModel(self.config, self.device).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model created with {n_params:,} parameters")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = self._create_scheduler()

        if self.save_tensorboard:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.config.output_dir, "logs"))
        if self.save_checkpoints or self.save_final_model:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.epoch = 0
        self.best_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "learning_rate": []}

        # Initialize M from the final time point
        if self.config.M_init == "final":
            final_slice_id = max(
                self.data_loader.datasets.keys(),
                key=lambda k: self.data_loader.datasets[k].time_point,
            )
            final_data = self.data_loader.get_full_data(final_slice_id)
            final_data = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in final_data.items()
            }
            self.model.initialize_M_from_data(final_data)
            t_final = self.data_loader.datasets[final_slice_id].time_point
            print(f"[OK] M initialized from final time point (t={t_final})")

    def _create_scheduler(self):
        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs, eta_min=1e-6
            )
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.5
            )
        elif self.config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10
            )
        return None

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()

        # loss_keys collected dynamically - all keys discovered automatically after the first batch;
        # no longer hardcoded, newly added loss terms are included in statistics automatically
        epoch_losses: Dict[str, list] = {"total": []}

        if self.model.is_dynamics_frozen():
            # -- Phase 1 (warmup): iterate over single time points, shuffled in batch order --
            # Each time point appears exactly once, avoiding double-counting of intermediate points caused by time-pair iteration.
            # Only per-timepoint losses (recon + isometric + diversity) are computed; ODE is not involved.
            sorted_ids = sorted(
                self.data_loader.datasets.keys(),
                key=lambda k: self.data_loader.datasets[k].time_point,
            )
            for slice_id in sorted_ids:
                full_t = self.data_loader.get_full_data(slice_id)
                t_raw  = self.data_loader.datasets[slice_id].time_point
                t_norm = (t_raw - self.t_min) / self.t_scale

                n_total    = full_t["x"].shape[0]
                batch_size = self.config.batch_size
                perm       = torch.randperm(n_total)

                pbar = tqdm(
                    range(0, n_total, batch_size),
                    desc=f"Epoch {self.epoch} [warmup t={t_raw}]",
                    leave=False,
                )
                for start in pbar:
                    idx  = perm[start: start + batch_size]
                    data_t: Dict = {
                        "x": full_t["x"][idx],
                        "s": full_t["s"][idx],
                        "t": torch.tensor(t_norm, dtype=torch.float32),
                    }
                    if "annotations" in full_t:
                        data_t["annotations"] = full_t["annotations"][idx.cpu().numpy()]
                    if not self.config.use_continuous_kernel:
                        data_t["knn_indices"] = self.model._compute_knn_kdtree(
                            data_t["s"], k=self.config.relative_k
                        )
                    data_t = self._to_device(data_t)

                    outputs = self.model.forward_warmup(data_t)
                    loss    = outputs["loss"]

                    if torch.isnan(loss) or torch.isinf(loss):
                        sub = {k: f"{v.item():.4f}" for k, v in outputs["losses"].items()
                               if isinstance(v, torch.Tensor)}
                        print(f"\n[WARNING] NaN/Inf in warmup, skipping. losses: {sub}")
                        self.optimizer.zero_grad()
                        continue

                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.config.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip
                        )
                    self.optimizer.step()

                    epoch_losses["total"].append(loss.item())
                    for name, val in outputs["losses"].items():
                        if name == "total":
                            continue
                        if isinstance(val, torch.Tensor):
                            if name not in epoch_losses:
                                epoch_losses[name] = []
                            epoch_losses[name].append(val.item())

                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        else:
            # -- Phase 2 (dynamics unfrozen): iterate over time pairs --
            # recon/isometric computed only for t0 (true z) and t1 (pred z), not t1's true z,
            # ensuring each time point's per-timepoint loss is computed only once per epoch.
            time_pairs = self.data_loader.get_time_pairs()

            for id_t0, id_t1, t0, t1 in time_pairs:
                full_t0 = self.data_loader.get_full_data(id_t0)
                full_t1 = self.data_loader.get_full_data(id_t1)

                t0_norm = (t0 - self.t_min) / self.t_scale
                t1_norm = (t1 - self.t_min) / self.t_scale

                n_spots = max(full_t0["x"].shape[0], full_t1["x"].shape[0])
                steps = self.config.steps_per_epoch
                if steps is None:
                    steps = max(1, n_spots // self.config.batch_size)

                pbar = tqdm(range(steps), desc=f"Epoch {self.epoch} [{t0}->{t1}]", leave=False)

                for _ in pbar:
                    data_t0 = self._get_batch(full_t0)
                    data_t1 = self._get_batch(full_t1)

                    data_t0 = self._to_device(data_t0)
                    data_t1 = self._to_device(data_t1)

                    data_t0["t"] = torch.tensor(t0_norm, dtype=torch.float32,
                                                device=data_t0["x"].device)
                    data_t1["t"] = torch.tensor(t1_norm, dtype=torch.float32,
                                                device=data_t1["x"].device)

                    if self.config.use_augmentation and np.random.rand() < 0.5:
                        data_t0, data_t1 = self._augment_data(data_t0, data_t1)

                    outputs = self.model(data_t0, data_t1)
                    loss    = outputs["loss"]

                    if torch.isnan(loss) or torch.isinf(loss):
                        sub = {
                            k: f"{v.item():.4f}" if isinstance(v, torch.Tensor) else str(v)
                            for k, v in outputs["losses"].items()
                        }
                        print(f"\n[WARNING] NaN/Inf loss detected, skipping batch. sub-losses: {sub}")
                        self.optimizer.zero_grad()
                        continue

                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.config.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip
                        )

                    self.optimizer.step()

                    epoch_losses["total"].append(loss.item())
                    for name, val in outputs["losses"].items():
                        if name == "total":
                            continue
                        if isinstance(val, torch.Tensor):
                            if name not in epoch_losses:
                                epoch_losses[name] = []
                            epoch_losses[name].append(val.item())

                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_losses: Dict[str, list] = {"total": []}
        time_pairs = self.data_loader.get_time_pairs()

        with torch.no_grad():
            # Fix random seed to ensure val loss is comparable across epochs (landmark selection
            # has randomness in approx/lowrank mode; fixing seed eliminates inter-batch noise)
            rng_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            torch.manual_seed(0)

            for id_t0, id_t1, t0, t1 in time_pairs:
                full_t0 = self.data_loader.get_full_data(id_t0)
                full_t1 = self.data_loader.get_full_data(id_t1)
                t0_norm = (t0 - self.t_min) / self.t_scale
                t1_norm = (t1 - self.t_min) / self.t_scale

                data_t0 = self._get_batch(full_t0)
                data_t1 = self._get_batch(full_t1)
                data_t0 = self._to_device(data_t0)
                data_t1 = self._to_device(data_t1)
                data_t0["t"] = torch.tensor(t0_norm, dtype=torch.float32,
                                            device=data_t0["x"].device)
                data_t1["t"] = torch.tensor(t1_norm, dtype=torch.float32,
                                            device=data_t1["x"].device)

                outputs = self.model(data_t0, data_t1)
                val_losses["total"].append(outputs["loss"].item())
                for k, v in outputs["losses"].items():
                    if k == "total":
                        continue
                    if isinstance(v, torch.Tensor):
                        if k not in val_losses:
                            val_losses[k] = []
                        val_losses[k].append(v.item())

            # Restore training random state
            torch.set_rng_state(rng_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state)

        return {k: float(np.mean(v)) if v else 0.0 for k, v in val_losses.items()}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print(f"\n{'='*50}")
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"{'='*50}\n")

        try:
            import wandb
            _use_wandb = wandb.run is not None
        except ImportError:
            _use_wandb = False

        # -- Early stopping state ----------------------------------
        es_enabled  = getattr(self.config, "early_stopping", True)
        es_patience = getattr(self.config, "early_stopping_patience", 20)
        es_delta    = getattr(self.config, "early_stopping_min_delta", 1e-4)
        # Epoch at which warmup ends (includes both decoder and feedback phases)
        warmup_end  = (getattr(self.config, "decoder_warmup_epochs",  0)
                       + getattr(self.config, "feedback_warmup_epochs", 0))
        es_best     = float("inf")
        es_counter  = 0

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.config._current_epoch = epoch  # ODE integration and decoding.

            # -- Decoder warmup (optional two-phase training) ----------------------
            # decoder_warmup_epochs=0 (default): pure end-to-end, feedback_gain=1 throughout
            # decoder_warmup_epochs=N > 0:
            #   Phase 1 (epoch < N): freeze dynamics network, train only encoder+decoder
            #     - Active:   recon / diversity / isometric
            #     - Disabled: trajectory / velocity / latent_smooth / equilibrium / spatial_smooth
            #     - feedback_gain=0 (ODE frozen, meaningless so set to 0)
            #   Phase 2 (epoch >= N): unfreeze all parameters, end-to-end joint training
            #     - All losses active, feedback_gain=1
            decoder_warmup  = getattr(self.config, "decoder_warmup_epochs",  0)
            feedback_warmup = getattr(self.config, "feedback_warmup_epochs", 0)

            if decoder_warmup > 0 and epoch < decoder_warmup:
                # -- Phase 1: freeze dynamics, train only encoder+decoder --
                if not self.model.is_dynamics_frozen():
                    self.model.freeze_dynamics()
                    print(f"\n[Phase 1] Epoch {epoch}: encoder/decoder warmup "
                          f"(dynamics frozen until epoch {decoder_warmup})")
                self.model.feedback_gain.fill_(0.0)
            else:
                # -- Phase 2: unfreeze ODE, end-to-end joint training --
                # encoder/decoder remain trainable, all losses (recon + trajectory + velocity + equilibrium)
                # activated simultaneously, feedback gain ramps linearly from 0 to 1.
                if self.model.is_dynamics_frozen():
                    self.model.unfreeze_dynamics()
                    msg = (f"feedback_gain ramp 0->1 over {feedback_warmup} epochs"
                           if feedback_warmup > 0 else "feedback_gain=1 immediately")
                    print(f"\n[Phase 2] Epoch {epoch}: dynamics unfrozen, joint training. {msg}")
                if feedback_warmup > 0:
                    gain = min(1.0, (epoch - decoder_warmup + 1) / feedback_warmup)
                    self.model.feedback_gain.fill_(gain)
                else:
                    self.model.feedback_gain.fill_(1.0)

            train_losses = self.train_epoch()
            val_losses   = self.validate()

            if self.config.scheduler == "plateau":
                self.scheduler.step(val_losses["total"])
            elif self.scheduler is not None:
                self.scheduler.step()

            lr   = self.optimizer.param_groups[0]["lr"]
            beta = self.model.beta_val.item()

            self.history["train_loss"].append(train_losses["total"])
            self.history["val_loss"].append(val_losses["total"])
            self.history["learning_rate"].append(lr)

            # ---- TensorBoard ----
            if self.save_tensorboard:
                self.writer.add_scalar("Loss/train", train_losses["total"], epoch)
                self.writer.add_scalar("Loss/val",   val_losses["total"],   epoch)
                self.writer.add_scalar("LR",          lr,                   epoch)
                self.writer.add_scalar("Beta",        beta,                 epoch)
                for name, val in train_losses.items():
                    if name != "total":
                        self.writer.add_scalar(f"Train/{name}", val, epoch)
                for name, val in val_losses.items():
                    if name != "total":
                        self.writer.add_scalar(f"Val/{name}", val, epoch)

            # ---- WandB ----
            if _use_wandb:
                log_dict = {
                    "epoch":          epoch,
                    "lr":             lr,
                    "beta":           beta,
                    "train/loss":     train_losses["total"],
                    "val/loss":       val_losses["total"],
                }
                # All training sub-losses
                for name, val in train_losses.items():
                    if name != "total":
                        log_dict[f"train/{name}"] = val
                # All validation sub-losses
                for name, val in val_losses.items():
                    if name != "total":
                        log_dict[f"val/{name}"] = val
                wandb.log(log_dict, step=epoch)

            if epoch % self.config.log_interval == 0:
                print(f"\nEpoch {epoch}/{self.config.num_epochs}")
                # Dynamically print all sub-losses
                train_sub = "  ".join(
                    f"{k}={v:.4f}"
                    for k, v in sorted(train_losses.items())
                    if k != "total"
                )
                val_sub = "  ".join(
                    f"{k}={v:.4f}"
                    for k, v in sorted(val_losses.items())
                    if k != "total"
                )
                print(f"  Train: total={train_losses['total']:.4f}  {train_sub}")
                print(f"  Val:   total={val_losses['total']:.4f}  {val_sub}")
                print(f"  LR: {lr:.6f}  |  beta: {beta:.4f}")

            if epoch % self.config.save_interval == 0 or epoch == self.config.num_epochs - 1:
                if self.save_checkpoints:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            if val_losses["total"] < self.best_loss:
                self.best_loss = val_losses["total"]
                if self.save_final_model:
                    self.save_checkpoint("best_model.pt")
                print(f"  [Best] New best model! Val loss: {self.best_loss:.4f}")

            # -- Early stopping (counting starts only after all warmup phases complete) ----------
            if es_enabled and epoch >= warmup_end:
                if val_losses["total"] < es_best - es_delta:
                    es_best    = val_losses["total"]
                    es_counter = 0
                else:
                    es_counter += 1
                    if es_counter >= es_patience:
                        print(f"\n[Early Stopping] No improvement for {es_patience} epochs "
                              f"after warmup (epoch {epoch}). Stopping.")
                        break

        print(f"\n{'='*50}")
        print("Training completed!")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"{'='*50}\n")

        if self.save_tensorboard:
            self.save_history()
            self.writer.close()


    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str):
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_loss": self.best_loss,
            "config": self.config,
            "history": self.history,
            # Time normalization parameters: needed when predicting after loading checkpoint
            "t_min": getattr(self, "t_min", 0.0),
            "t_max": getattr(self, "t_max", 1.0),
            "t_scale": getattr(self, "t_scale", 1.0),
        }
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        path = os.path.join(self.config.checkpoint_dir, filename)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch     = ckpt["epoch"]
        self.best_loss = ckpt["best_loss"]
        self.history   = ckpt["history"]
        # Restore time normalization parameters
        self.t_min   = ckpt.get("t_min",   getattr(self, "t_min",   0.0))
        self.t_max   = ckpt.get("t_max",   getattr(self, "t_max",   1.0))
        self.t_scale = ckpt.get("t_scale", getattr(self, "t_scale", 1.0))
        print(f"Checkpoint loaded: {path}  (epoch {self.epoch}, "
              f"t_norm: [{self.t_min},{self.t_max}]->[0,1])")

    def normalize_time(self, t: float) -> float:
        """Normalize raw time to [0, 1], for use by external callers of predict_step"""
        return (t - self.t_min) / self.t_scale

    def save_history(self):
        path = os.path.join(self.config.output_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved: {path}")

    def save_config(self, config_path: Optional[str] = None):
        import yaml
        if config_path is None:
            config_path = os.path.join(self.config.output_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.config.__dict__, f)
        print(f"[OK] Config saved: {config_path}")

    # ------------------------------------------------------------------
    # Batch sampling
    # ------------------------------------------------------------------

    def _get_batch(self, data: Dict) -> Dict:
        return self._global_subset(data)

    def _global_subset(self, data: Dict) -> Dict:
        """Global uniform random subsampling: no spatial bias, captures global distribution.

        Each step randomly samples batch_size cells from the full set (or all cells if fewer available).
        Uses the same batch_size as Phase 1, unifying sampling scale across both phases.
        """
        n_total  = data["x"].shape[0]
        n_sample  = min(n_total, self.config.batch_size)

        idx = torch.randperm(n_total)[:n_sample]

        new_data: Dict = {
            "x": data["x"][idx],
            "s": data["s"][idx],
            "t": data["t"],
        }
        if "annotations" in data:
            new_data["annotations"] = data["annotations"][idx.cpu().numpy()]
        if not self.config.use_continuous_kernel:
            new_data["knn_indices"] = self.model._compute_knn_kdtree(
                new_data["s"], k=self.config.relative_k
            ).to(self.device)
        return new_data

    # ------------------------------------------------------------------
    # Data augmentation
    # ------------------------------------------------------------------

    def _augment_data(self, data_t0: Dict, data_t1: Dict):
        if self.config.augment_rotation:
            theta = np.random.rand() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=self.device)
            for data in [data_t0, data_t1]:
                center = data["s"].mean(dim=0, keepdim=True)
                data["s"] = torch.matmul(data["s"] - center, R.T) + center

        if self.config.augment_flip and np.random.rand() < 0.5:
            axis = np.random.choice([0, 1])
            for data in [data_t0, data_t1]:
                # For z-scored data (mean=0), mirror symmetry should be around 0, i.e., negate
                s = data["s"].clone()
                s[:, axis] = -s[:, axis]
                data["s"] = s

        if self.config.augment_noise > 0:
            for data in [data_t0, data_t1]:
                data["x"] = data["x"] + torch.randn_like(data["x"]) * self.config.augment_noise

        return data_t0, data_t1

    def _to_device(self, data: Dict) -> Dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
