"""
Tier 3 leave-one-timepoint-out benchmark for FeedbackControlModel.

Prediction strategy:
  1. Encode t0 (X, S) into Z_t0.
  2. Integrate the ODE to obtain Z_t1_pred.
  3. Decode X_t1_pred with GeneDecoder.

Strict LOTO setting: prediction never reads the held-out slice. The held-out slice is used only after prediction for evaluation.

Example:
  python benchmark_loto_ours.py --data mouse_embryo --holdout_idx 3
"""
import os, sys, time, json, argparse, random
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from train import Trainer
import scanpy as sc
from benchmark_utils import (
    DATASET_CONFIG, CANONICAL_COLS, compute_all_metrics, save_results,
    plot_metrics_bar, print_metrics_summary,
    load_dataset_params,
    load_spatial_coords, normalize_coords_per_slice,
)

BASE_DATA_DIR      = "./spatiotemporal"
DEFAULT_OUTPUT_DIR = "./spatiotemporal/outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Tier 3 LOTO - FeedbackControlModel (given S_t1)")
    p.add_argument("--data",         type=str,   default="mouse_embryo",
                   choices=list(DATASET_CONFIG.keys()))
    p.add_argument("--output_dir",   type=str,   default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--slice_paths",  type=str,   nargs="+", default=None)
    p.add_argument("--time_points",  type=float, nargs="+", default=None)
    p.add_argument("--annotation_key", type=str, default=None)
    p.add_argument("--layer",        type=str,   default=None)
    p.add_argument("--n_genes",      type=int,   default=3000)
    p.add_argument("--need_normalize",     action="store_true")
    p.add_argument("--need_log_transform", action="store_true")
    p.add_argument("--need_scale",         action="store_true")
    p.add_argument("--evaluate_top_k", type=int, default=100)
    p.add_argument("--holdout_idx",  type=int,   default=None,
                   help="Held-out time point index (0-based, must be an intermediate time point)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--use_continuous_kernel",  type=bool, default=True,
                   help="Use a continuous kernel such as Gaussian RBF; if False, use a binary KNN kernel")
    p.add_argument("--num_epochs",   type=int,   default=200)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--ode_solver",   type=str,   default="rk4")
    p.add_argument("--latent_dim",  type=int,   default=128)
    p.add_argument("--checkpoint",   type=str,   default=None)
    p.add_argument("--use_wandb",    action="store_true")
    p.add_argument("--wandb_project", type=str,  default="loto-ours")
    p.add_argument("--decoder_chunk_size", type=int, default=2048,
                   help="Chunk size for full-slice decoder inference to avoid GPU OOM")
    p.add_argument("--metrics_output_csv", type=str, default=None,
                   help="If specified, also save metrics to this CSV file (appending if exists)")
    p.add_argument("--config_json", type=str, default=None,
                   help="Path to JSON file with dataset-specific hyperparameters (overrides defaults)")
    p.add_argument("--use_label_aware_losses", action="store_true")
    p.add_argument("--label_distance_metric", type=str, default="feat", choices=["feat"])
    p.add_argument("--lambda_label_trajectory", type=float, default=0.0)
    p.add_argument("--lambda_label_proportion", type=float, default=0.0)
    p.add_argument("--lambda_label_centroid", type=float, default=0.0)
    p.add_argument("--label_min_cells", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds_cfg      = DATASET_CONFIG[args.data]
    data_dir    = f"{BASE_DATA_DIR}/{args.data}"
    slice_paths = args.slice_paths or [f"{data_dir}/{s}" for s in ds_cfg["suffixes"]]
    time_points = args.time_points or ds_cfg["time_points"]
    ann_key     = args.annotation_key or ds_cfg["annotation_key"]

    order       = sorted(range(len(slice_paths)), key=lambda i: time_points[i])
    slice_paths = [slice_paths[i] for i in order]
    time_points = [time_points[i] for i in order]
    n_slices    = len(slice_paths)
    assert n_slices >= 3

    ho_idx = args.holdout_idx
    if ho_idx is None:
        ho_idx = random.Random(args.seed).choice(range(1, n_slices - 1))
    assert 0 < ho_idx < n_slices - 1, ""

    held_out_time = time_points[ho_idx]
    held_out_path = slice_paths[ho_idx]
    train_paths   = [p for i, p in enumerate(slice_paths) if i != ho_idx]
    train_times   = [t for i, t in enumerate(time_points) if i != ho_idx]
    source_train_idx = ho_idx - 1

    run_name   = f"loto_ours_{args.data}_t{held_out_time}_{time.strftime('%m%d-%H%M')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Tier 3 LOTO - FeedbackControlModel")
    print(f"  Held-out: idx={ho_idx}, t={held_out_time}")
    print(f"  Training: {len(train_paths)} slices, t={train_times}")
    print("=" * 70)

    if args.use_wandb:
        import wandb; wandb.login()
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    config = Config(
        learning_rate=args.lr, ode_solver=args.ode_solver,
        latent_dim=args.latent_dim, num_epochs=args.num_epochs,
        output_dir=str(output_dir / "results"),
        checkpoint_dir=str(output_dir / "checkpoints"),
        use_label_aware_losses=args.use_label_aware_losses,
        label_distance_metric=args.label_distance_metric,
        lambda_label_trajectory=args.lambda_label_trajectory,
        lambda_label_proportion=args.lambda_label_proportion,
        lambda_label_centroid=args.lambda_label_centroid,
        label_min_cells=args.label_min_cells,
    )
    trainer = Trainer(config)

    if args.config_json is not None:
        print(f"\n[INFO] Loading dataset-specific hyperparameters from {args.config_json}")
        json_path = args.config_json
        load_dataset_params(args.data, config, json_path=json_path, holdout_idx=ho_idx)

    trainer.load_data(
        slice_paths=train_paths, time_points=train_times,
        annotation_key=ann_key, layer=args.layer, n_genes=args.n_genes,
        need_normalize=args.need_normalize,
        need_log_transform=args.need_log_transform,
        need_scale=args.need_scale,
    )

    if args.checkpoint:
        print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    else:
        print("\n[1/3] Training...")
        trainer.train()
        trainer.save_checkpoint("final_model.pt")

    source_slice_id = None
    for sid, ds in trainer.data_loader.datasets.items():
        if abs(ds.time_point - train_times[source_train_idx]) < 1e-6:
            source_slice_id = sid; break
    assert source_slice_id is not None

    src_data = trainer.data_loader.get_full_data(source_slice_id)
    t0 = float(src_data["t"])

    print("\n[2/3] Predicting...")
    trainer.model.eval()

    def _forward_in_chunks(fn, x, chunk_size):
        if x.shape[0] <= chunk_size:
            return fn(x)
        outs = []
        for start in range(0, x.shape[0], chunk_size):
            outs.append(fn(x[start:start + chunk_size]))
        return torch.cat(outs, dim=0)

    x_t0 = src_data["x"].to(trainer.device)
    s_t0 = src_data["s"].to(trainer.device)
    knn  = src_data.get("knn_indices")
    if knn is not None: knn = knn.to(trainer.device)

    # ODE integration and decoding.
    with torch.no_grad():
        Z_t0 = trainer.model.latent_embedding(x_t0, s_t0, knn)
        Z_t1_pred = trainer.model._ode_integrate_Z(
            Z_t0,
            trainer.model._to_tensor(trainer.normalize_time(t0), trainer.device),
            trainer.model._to_tensor(trainer.normalize_time(held_out_time), trainer.device),
        )
        X_pred_sc = _forward_in_chunks(
            trainer.model.gene_decoder,
            Z_t1_pred,
            args.decoder_chunk_size,
        )

    print("\n[3/3] Loading held-out slice for evaluation...")
    # Load held-out slice directly (same as spateo/moscot/stories) for
    # identical gene selection and preprocessing across methods.
    ho_raw = sc.read_h5ad(held_out_path)
    if args.layer and args.layer in ho_raw.layers:
        ho_raw.X = ho_raw.layers[args.layer]
    if args.need_normalize:
        sc.pp.normalize_total(ho_raw, target_sum=1e6)
    if args.need_log_transform:
        sc.pp.log1p(ho_raw)
    selected = trainer.data_loader.selected_genes or []
    common = [g for g in selected if g in ho_raw.var_names]
    ho_sub = ho_raw[:, common]
    X_true_np = (ho_sub.X.toarray() if hasattr(ho_sub.X, "toarray")
                 else np.array(ho_sub.X))
    ho_S_raw = load_spatial_coords(ho_raw)
    S_true_np = normalize_coords_per_slice(ho_S_raw) if ho_S_raw is not None else None
    labels_t1 = (ho_raw.obs[ann_key].astype(str).values
                 if ann_key and ann_key in ho_raw.obs.columns else None)
    labels_t0 = src_data.get("annotations")

    X_pred_np, _ = trainer.data_loader.unnormalize(x_scaled=X_pred_sc)

    print(f"  X_pred: {X_pred_np.shape}, X_true: {X_true_np.shape}, "
          f"S_true: {S_true_np.shape}")

    eval_runs = [
        (X_pred_np, "FeedbackControlModel", "base"),
    ]

    X_src_mean_np, _ = trainer.data_loader.unnormalize(x_scaled=src_data["x"])
    X_prevcopy = np.tile(X_src_mean_np.mean(0), (len(X_true_np), 1))
    baseline_metrics = compute_all_metrics(
        X_pred=X_prevcopy, X_true_t1=X_true_np,
        S_true_t1=S_true_np, labels_t1=labels_t1,
        hvg_k=args.evaluate_top_k,
    )
    print_metrics_summary(baseline_metrics, "PrevCopy (baseline)", t0, held_out_time)

    def make_row(method_name, metrics_dict, t0, held_out_time):
        return {
            "method":                method_name,
            "from_time":             t0,
            "to_time":               held_out_time,
            "time_span":             f"{t0}->{held_out_time}",
            "x_hvg_corr":            metrics_dict.get("x_hvg_corr",           float("nan")),
            "x_wasserstein":         metrics_dict.get("x_wasserstein",        float("nan")),
            "spatial_domain_ari":    metrics_dict.get("spatial_domain_ari",   float("nan")),
            "spatial_coherence_pc1": metrics_dict.get("spatial_coherence_pc1", float("nan")),
        }

    rows = []
    for X_pred_cur, method_name, tag in eval_runs:
        metrics = compute_all_metrics(
            X_pred=X_pred_cur, X_true_t1=X_true_np,
            S_true_t1=S_true_np, labels_t1=labels_t1, labels_t0=labels_t0,
            hvg_k=args.evaluate_top_k,
        )
        print_metrics_summary(metrics, method_name, t0, held_out_time)
        plot_metrics_bar(metrics, method_name, t0, held_out_time,
                         str(output_dir / f"loto_metrics_{tag}.png"))
        rows.append(make_row(method_name, metrics, t0, held_out_time))

    rows.append(make_row("PrevCopy", baseline_metrics, t0, held_out_time))

    df = pd.DataFrame(rows, columns=CANONICAL_COLS)
    df.to_csv(output_dir / "loto_results.csv", index=False)

    import os
    if args.metrics_output_csv:
        results_csv = args.metrics_output_csv
    else:
        results_csv = f"metrics/comparison_results_loto_{args.data}.csv"
    os.makedirs("metrics", exist_ok=True)
    df.to_csv(results_csv, mode="a" if os.path.exists(results_csv) else "w",
              header=not os.path.exists(results_csv), index=False)
    print(f"\n  [CSV] {len(rows)} rows saved to {results_csv}")

    if args.use_wandb:
        try:
            import wandb
            if wandb.run:
                for r in rows:
                    prefix = "eval"
                    wandb.log({f"{prefix}/{k}": v for k, v in r.items()
                               if isinstance(v, (int, float, np.floating))})
                wandb.finish()
        except Exception: pass


if __name__ == "__main__":
    main()
