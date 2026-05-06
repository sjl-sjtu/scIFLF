"""
hparam_loto_sweep.py
====================
Hyperparameter search for leave-one-timepoint-out metrics.

Features:
  1. Read experiment parameter combinations from hparam_search_config.csv.
  2. Run LOTO training and evaluation for each parameter set.
  3. Write hyperparameters and evaluation metrics to a summary CSV.

Notes:
  - Checkpoints, model files, TensorBoard logs, and wandb logs are not saved.
  - The metric logic is based on compute_loto_metrics from the existing LOTO benchmark scripts.

Example:
  python hparam_loto_sweep.py --data mouse_embryo --holdout_idx 3 --config_csv hparam_search_config.csv --output_csv sweep_results.csv --num_epochs 100
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_DISABLE_CACHE"] = "1"

import sys
import gc
import time
import random
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from train import Trainer
from data_loader import SpatialTranscriptomicsLoader
from benchmark_utils import (
    DATASET_CONFIG,
    compute_all_metrics,
    load_dataset_params,
)

try:
    import wandb
    wandb.init(mode="disabled")
except Exception:
    pass

# ============================================================
# Dataset configuration
# ============================================================
BASE_DATA_DIR = "./spatiotemporal"


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter sweep for LOTO evaluation (no saving)"
    )
    p.add_argument("--data", type=str, default="mouse_embryo",
                   choices=list(DATASET_CONFIG.keys()))
    p.add_argument("--slice_paths", type=str, nargs="+", default=None,
                   help="Override the dataset default paths")
    p.add_argument("--time_points", type=float, nargs="+", default=None,
                   help="Override the dataset default time points")
    p.add_argument("--annotation_key", type=str, default=None)
    p.add_argument("--layer", type=str, default=None)
    p.add_argument("--n_genes", type=int, default=3000)
    p.add_argument("--evaluate_top_k", type=int, default=100,
                   help="Number of top HVGs used for metric computation")
    p.add_argument("--need_normalize", action="store_true")
    p.add_argument("--need_log_transform", action="store_true")
    p.add_argument("--need_scale", action="store_true")

    p.add_argument("--holdout_idx", type=int, default=None,
                   help="Held-out time point index (0-based; None selects one randomly)")
    p.add_argument("--seed", type=int, default=42)

    # Fixed training hyperparameters, optionally overridden by config_csv rows.
    p.add_argument("--num_epochs", type=int, default=200,
                   help="Default training epochs; config_csv num_epochs overrides this value")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ode_solver", type=str, default="rk4",
                   choices=["euler", "rk4", "dopri5"])
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--decoder_chunk_size", type=int, default=2048,
                   help="Chunk size for full-slice decoding to avoid GPU OOM")
    p.add_argument("--config_json", type=str, default=None,
                   help="Dataset-specific hyperparameter JSON")
    p.add_argument("--use_label_aware_losses", action="store_true")
    p.add_argument("--label_distance_metric", type=str, default="feat",
                   choices=["feat"])
    p.add_argument("--lambda_label_trajectory", type=float, default=0.0)
    p.add_argument("--lambda_label_proportion", type=float, default=0.0)
    p.add_argument("--lambda_label_centroid", type=float, default=0.0)
    p.add_argument("--label_min_cells", type=int, default=8)

    p.add_argument("--config_csv", type=str, default="hparam_search_config.csv",
                   help="Path to the CSV file containing experiment parameter combinations")
    p.add_argument("--output_csv", type=str, default="sweep_results.csv",
                   help="Path to the summary output CSV file")

    p.add_argument("--exp_ids", type=str, default=None,
                   help="Run only selected exp_id values, comma-separated, such as '1,3,5'")
    return p.parse_args()


# (Removed _vf_cosine_sim_xspace as it's now part of unified metrics)


# ============================================================
# LOTO evaluation step.
# ============================================================
def compute_loto_metrics(
    trainer, held_out_loader, source_id, next_id,
    held_out_time, annotation_key=None, hvg_k=100,
    decoder_chunk_size=2048,
):
    """
    Tier 3 LOTO evaluation: predict gene expression for the held-out t1.
    Synchronized with benchmark_loto_ours.py and benchmark_utils.
    """
    device = trainer.device

    src  = trainer.data_loader.get_full_data(source_id)
    x_t0 = src["x"].to(device); s_t0 = src["s"].to(device)
    knn0 = src.get("knn_indices")
    if knn0 is not None: knn0 = knn0.to(device)
    t0 = float(src["t"])

    true_data = held_out_loader.get_full_data(0)
    X_true_sc = true_data["x"].to(device)
    S_true_sc = true_data["s"].to(device)

    def _forward_in_chunks(fn, x, chunk_size):
        if x.shape[0] <= chunk_size:
            return fn(x)
        outs = []
        for start in range(0, x.shape[0], chunk_size):
            outs.append(fn(x[start:start + chunk_size]))
        return torch.cat(outs, dim=0)

    trainer.model.eval()
    with torch.no_grad():
        Z_t0 = trainer.model.latent_embedding(x_t0, s_t0, knn0)
        Z_t1_pred = trainer.model._ode_integrate_Z(
            Z_t0,
            trainer.model._to_tensor(trainer.normalize_time(t0), device),
            trainer.model._to_tensor(trainer.normalize_time(held_out_time), device),
        )
        X_pred_sc = _forward_in_chunks(
            trainer.model.gene_decoder,
            Z_t1_pred,
            decoder_chunk_size,
        )

    # 4. Unnormalize to Level 1 (LogCPM)
    X_pred_np, _ = trainer.data_loader.unnormalize(x_scaled=X_pred_sc)
    X_true_np, S_true_np = trainer.data_loader.unnormalize(
        x_scaled=X_true_sc, s_scaled=S_true_sc
    )
    # Annotation handling.
    labels_t1 = true_data.get("annotations")
    labels_t0 = src.get("annotations")

    metrics = compute_all_metrics(
        X_pred=X_pred_np, X_true_t1=X_true_np,
        S_true_t1=S_true_np, labels_t1=labels_t1, labels_t0=labels_t0,
        hvg_k=hvg_k,
    )

    model_metrics = {
        **metrics,
    }
    return model_metrics
# ============================================================
# ============================================================
def run_one_experiment(
    config,
    exp_row,
    args,
    train_paths, train_times,
    held_out_path, held_out_time,
    source_idx, next_idx,
    annotation_key,
):
    """
    Train one model from exp_row hyperparameters and return a metric dict.
    Checkpoints, model files, TensorBoard logs, and wandb logs are not saved.
    """

    # -- Trainer ------------------------------------------------------
    trainer = Trainer(config)
    trainer.load_data(
        slice_paths=train_paths, time_points=train_times,
        annotation_key=annotation_key, layer=args.layer,
        n_genes=args.n_genes,
        need_normalize=args.need_normalize,
        need_log_transform=args.need_log_transform,
        need_scale=args.need_scale,
    )
    trainer.train()

    train_stats = {
        "x_stats": trainer.data_loader.x_stats,
        "s_stats": trainer.data_loader.s_stats
    }
    held_out_loader = SpatialTranscriptomicsLoader(config)
    held_out_loader.load_from_anndata(
        slice_paths=[held_out_path], time_points=[held_out_time],
        annotation_key=annotation_key, layer=args.layer,
        selected_genes=trainer.data_loader.selected_genes,
        need_normalize=args.need_normalize,
        need_log_transform=args.need_log_transform,
        need_scale=args.need_scale,
        external_stats=train_stats
    )

    source_slice_id = next_slice_id = None
    for sid, ds in trainer.data_loader.datasets.items():
        if abs(ds.time_point - train_times[source_idx]) < 1e-6:
            source_slice_id = sid
        if next_idx is not None and abs(ds.time_point - train_times[next_idx]) < 1e-6:
            next_slice_id = sid

    metrics = compute_loto_metrics(
        trainer, held_out_loader,
        source_slice_id, next_slice_id,
        held_out_time, annotation_key=annotation_key,
        hvg_k=args.evaluate_top_k,
        decoder_chunk_size=args.decoder_chunk_size,
    )
    return metrics


def build_experiment_config(args, exp_row):
    loss_weight_fields = [
        "lambda_trajectory", "lambda_equilibrium",
        "lambda_diversity", "lambda_recon",
        "lambda_isometric",
    ]

    config = Config(
        learning_rate=args.lr,
        ode_solver=args.ode_solver,
        latent_dim=args.latent_dim,
        num_epochs=args.num_epochs,
        output_dir="/tmp/sweep_tmp",
        checkpoint_dir="/tmp/sweep_tmp",
        save_checkpoints=False,
        save_final_model=False,
        save_tensorboard=False,
        use_label_aware_losses=args.use_label_aware_losses,
        label_distance_metric=args.label_distance_metric,
        lambda_label_trajectory=args.lambda_label_trajectory,
        lambda_label_proportion=args.lambda_label_proportion,
        lambda_label_centroid=args.lambda_label_centroid,
        label_min_cells=args.label_min_cells,
    )

    # LOTO evaluation step.
    # Sinkhorn OT parameters.
    if args.config_json is not None:
        load_dataset_params(args.data, config, json_path=args.config_json, holdout_idx=None)

    for field in loss_weight_fields:
        if field in exp_row.index and pd.notna(exp_row[field]):
            setattr(config, field, float(exp_row[field]))

    if "latent_dim" in exp_row.index and pd.notna(exp_row["latent_dim"]):
        config.latent_dim = int(exp_row["latent_dim"])
    if "num_epochs" in exp_row.index and pd.notna(exp_row["num_epochs"]):
        config.num_epochs = int(exp_row["num_epochs"])
    if "decoder_warmup_epochs" in exp_row.index and pd.notna(exp_row["decoder_warmup_epochs"]):
        config.decoder_warmup_epochs = int(exp_row["decoder_warmup_epochs"])
    if "feedback_warmup_epochs" in exp_row.index and pd.notna(exp_row["feedback_warmup_epochs"]):
        config.feedback_warmup_epochs = int(exp_row["feedback_warmup_epochs"])

    # Sinkhorn OT parameters.
    if "sinkhorn_blur" in exp_row.index and pd.notna(exp_row["sinkhorn_blur"]):
        val = str(exp_row["sinkhorn_blur"]).strip()
        config.sinkhorn_blur = None if val.lower() == "none" else float(val)
    if "sinkhorn_reach" in exp_row.index and pd.notna(exp_row["sinkhorn_reach"]):
        val = str(exp_row["sinkhorn_reach"]).strip()
        config.sinkhorn_reach = None if val.lower() == "none" else float(val)

    if "n_prototypes" in exp_row.index and pd.notna(exp_row["n_prototypes"]):
        config.n_prototypes = int(exp_row["n_prototypes"])
    if "prototype_temp" in exp_row.index and pd.notna(exp_row["prototype_temp"]):
        config.prototype_temp = float(exp_row["prototype_temp"])

    # isometric_gamma
    if "isometric_gamma" in exp_row.index and pd.notna(exp_row["isometric_gamma"]):
        config.isometric_gamma = float(exp_row["isometric_gamma"])

    # CSV output handling.
    if "learning_rate" in exp_row.index and pd.notna(exp_row["learning_rate"]):
        config.learning_rate = float(exp_row["learning_rate"])
    if "batch_size" in exp_row.index and pd.notna(exp_row["batch_size"]):
        config.batch_size = int(exp_row["batch_size"])

    return config


# ============================================================
# ============================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    output_csv = Path(args.output_csv)

    # CSV output handling.
    config_csv_path = Path(args.config_csv)
    if not config_csv_path.exists():
        print(f"[ERROR] config_csv not found: {config_csv_path}")
        sys.exit(1)
    # Sinkhorn OT parameters.
    def _keep_none_str(val):
        s = str(val).strip()
        return s if s != "" else float("nan")

    exp_df = pd.read_csv(
        config_csv_path,
        converters={"sinkhorn_reach": _keep_none_str, "sinkhorn_blur": _keep_none_str},
    )
    print(f"[INFO] Loaded {len(exp_df)} experiment configs from {config_csv_path}")

    exp_df = exp_df.copy()
    if "run_id" in exp_df.columns:
        exp_df["_run_id"] = exp_df["run_id"].astype(int)
    elif "exp_id" in exp_df.columns:
        exp_df["_run_id"] = exp_df["exp_id"].astype(int)
    else:
        exp_df["_run_id"] = np.arange(1, len(exp_df) + 1)

    exp_df["_config_row_id"] = np.arange(len(exp_df))

    if args.exp_ids is not None:
        wanted = set(int(x.strip()) for x in args.exp_ids.split(","))
        exp_df = exp_df[exp_df["_run_id"].isin(wanted)].reset_index(drop=True)
        print(f"[INFO] Filtered to {len(exp_df)} experiments: {sorted(wanted)}")

    # CSV output handling.
    if output_csv.exists():
        done_df = pd.read_csv(output_csv)
        if "exp_id" in done_df.columns:
            done_keys = set(done_df["exp_id"].dropna().astype(int).tolist())
            keep_mask = ~exp_df["_run_id"].isin(done_keys)
            skip_desc = "run_id/exp_id"
        elif "_config_row_id" in done_df.columns:
            done_keys = set(done_df["_config_row_id"].dropna().astype(int).tolist())
            keep_mask = ~exp_df["_config_row_id"].isin(done_keys)
            skip_desc = "_config_row_id"
        else:
            keep_mask = pd.Series([True] * len(exp_df))
            skip_desc = "none"

        skipped_count = int((~keep_mask).sum())
        exp_df = exp_df[keep_mask].reset_index(drop=True)
        print(
            f"[INFO] Found existing output CSV {output_csv}, "
            f"skipped {skipped_count} finished configs by {skip_desc}, "
            f"remaining {len(exp_df)} configs to run"
        )

    ds_cfg         = DATASET_CONFIG[args.data]
    data_dir       = f"{BASE_DATA_DIR}/{args.data}"
    slice_paths    = args.slice_paths or [f"{data_dir}/{s}" for s in ds_cfg["suffixes"]]
    time_points    = args.time_points or ds_cfg["time_points"]
    annotation_key = args.annotation_key or ds_cfg["annotation_key"]
    assert len(slice_paths) == len(time_points)

    n_slices = len(slice_paths)
    assert n_slices >= 3

    order       = sorted(range(n_slices), key=lambda i: time_points[i])
    slice_paths = [slice_paths[i] for i in order]
    time_points = [time_points[i] for i in order]

    if args.holdout_idx is not None:
        ho_idx = args.holdout_idx
        assert 0 < ho_idx < n_slices - 1, "holdout_idx ()"
    else:
        ho_idx = random.Random(args.seed).choice(range(1, n_slices - 1))

    held_out_time = time_points[ho_idx]
    held_out_path = slice_paths[ho_idx]
    train_paths   = [p for i, p in enumerate(slice_paths) if i != ho_idx]
    train_times   = [t for i, t in enumerate(time_points) if i != ho_idx]

    # source_idx and next_idx are indices within train_times, not the original slice indices.
    source_idx = max(i for i, t in enumerate(train_times) if t < held_out_time)
    next_candidates = [i for i, t in enumerate(train_times) if t > held_out_time]
    next_idx = next_candidates[0] if next_candidates else None

    print("=" * 70)
    print("  FeedbackControlModel - Hyperparameter LOTO Sweep")
    print(f"  Dataset     : {args.data}")
    print(f"  Held-out    : idx={ho_idx}, t={held_out_time}")
    print(f"  Train times : {train_times}")
    print(f"  Experiments : {len(exp_df)}")
    print(f"  Output CSV  : {args.output_csv}")
    print("=" * 70)

    all_rows = []

    for row_idx, exp_row in exp_df.iterrows():
        exp_id = int(exp_row["_run_id"])
        note   = exp_row.get("note", "")
        print(f"\n{'=' * 70}")
        print(f"  [Exp {exp_id}] {note}")

        loss_fields = [
            "lambda_trajectory", "lambda_equilibrium",
            "lambda_diversity", "lambda_recon",
            "lambda_isometric",
        ]
        for f in loss_fields:
            if f in exp_row.index and pd.notna(exp_row[f]):
                print(f"    {f:30s} = {exp_row[f]}")
        for f in ["sinkhorn_blur", "sinkhorn_reach", "n_prototypes", "prototype_temp",
                   "latent_dim", "num_epochs", "learning_rate", "batch_size",
                   "decoder_warmup_epochs", "feedback_warmup_epochs"]:
            if f in exp_row.index and pd.notna(exp_row[f]):
                print(f"    {f:30s} = {exp_row[f]}")
        print(f"{'=' * 70}")

        t_start = time.time()
        status  = "success"
        metrics = {}
        used_config = build_experiment_config(args, exp_row)

        try:
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            metrics = run_one_experiment(
                config=used_config,
                exp_row=exp_row, args=args,
                train_paths=train_paths, train_times=train_times,
                held_out_path=held_out_path, held_out_time=held_out_time,
                source_idx=source_idx, next_idx=next_idx,
                annotation_key=annotation_key,
            )
            elapsed = time.time() - t_start
            print(f"  [Exp {exp_id}] Done in {elapsed:.1f}s  "
                  f"x_hvg_corr={metrics.get('x_hvg_corr', float('nan')):.4f}  "
                  f"x_wasserstein={metrics.get('x_wasserstein', float('nan')):.4f}  "
                  f"")
        except Exception as e:
            elapsed = time.time() - t_start
            status  = f"error: {type(e).__name__}: {e}"
            print(f"  [Exp {exp_id}] FAILED in {elapsed:.1f}s: {e}")
            traceback.print_exc()

        result_row = {
            "exp_id":       exp_id,
            "note":         note,
            "_config_row_id": int(exp_row["_config_row_id"]),
            "data":         args.data,
            "holdout_idx":  ho_idx,
            "held_out_time": held_out_time,
            "seed":         args.seed,
            "lr":                    args.lr,
            "ode_solver":            args.ode_solver,
            "latent_dim":            used_config.latent_dim,
            "learning_rate":         used_config.learning_rate,
            "batch_size":            used_config.batch_size,
            "decoder_warmup_epochs": used_config.decoder_warmup_epochs,
            "feedback_warmup_epochs":used_config.feedback_warmup_epochs,
            "sinkhorn_blur":         used_config.sinkhorn_blur,
            "sinkhorn_reach":        str(used_config.sinkhorn_reach),
            "n_prototypes":          used_config.n_prototypes,
            "prototype_temp":        used_config.prototype_temp,
            "elapsed_sec":           round(time.time() - t_start, 1),
            "status":                status,
        }

        for f in loss_fields + ["num_epochs"]:
            if f in exp_row.index and pd.notna(exp_row[f]):
                result_row[f] = exp_row[f]
            else:
                result_row[f] = getattr(used_config, f, None)

        result_row.update(metrics)

        all_rows.append(result_row)

        # CSV output handling.
        df_cur = pd.DataFrame([result_row])
        if output_csv.exists():
            df_cur.to_csv(output_csv, mode="a", header=False, index=False)
        else:
            df_cur.to_csv(output_csv, mode="w", header=True, index=False)
        print(f"  [Exp {exp_id}] Result appended to {output_csv}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("  Sweep Complete - Summary")
    print("=" * 70)
    final_df = pd.read_csv(output_csv)
    summary_cols = ["exp_id", "note",
                    "sinkhorn_blur", "sinkhorn_reach", "n_prototypes", "prototype_temp",
                    "lambda_trajectory", "lambda_equilibrium", "lambda_isometric",
                    "x_hvg_corr", "x_wasserstein",
                    "spatial_domain_ari", "spatial_coherence_pc1",
                    "status"]
    avail_cols = [c for c in summary_cols if c in final_df.columns]
    print(final_df[avail_cols].to_string(index=False))
    print(f"\n  Full results saved to: {output_csv}")


if __name__ == "__main__":
    main()
