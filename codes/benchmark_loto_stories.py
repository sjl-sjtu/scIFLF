"""
Tier 3 leave-one-timepoint-out benchmark for STORIES.

Strict LOTO setting: STORIES trains only on slices outside the held-out time point and transforms from t0 to the held-out t1 during prediction. The held-out slice is used only after prediction for evaluation.

Example:
  python benchmark_loto_stories.py --data mouse_embryo --holdout_idx 3
"""
import os, sys, time, json, argparse, random, warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib; matplotlib.use("Agg")
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_utils import (
    DATASET_CONFIG, CANONICAL_COLS, compute_all_metrics,
    plot_metrics_bar, print_metrics_summary,
    load_spatial_coords, normalize_coords_per_slice,
)

try:
    from stories import SpaceTime
    import optax
    HAS_STORIES = True
except ImportError:
    HAS_STORIES = False
    print("WARNING: stories-jax not installed.")

BASE_DATA_DIR      = "./spatiotemporal"
DEFAULT_OUTPUT_DIR = "./spatiotemporal/outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Tier 3 LOTO - STORIES")
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
    p.add_argument("--n_pca",        type=int,   default=50)
    p.add_argument("--holdout_idx",  type=int,   default=None)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--max_iter",     type=int,   default=10000)
    p.add_argument("--batch_size",   type=int,   default=1000)
    p.add_argument("--lr",           type=float, default=1e-2)
    p.add_argument("--epsilon",      type=float, default=0.01)
    p.add_argument("--quadratic_weight", type=float, default=5e-3)
    p.add_argument("--patience",     type=int,   default=150)
    p.add_argument("--use_wandb",    action="store_true")
    p.add_argument("--wandb_project", type=str,  default="loto-stories")
    p.add_argument("--metrics_output_csv", type=str, default=None,
                   help="If specified, also save metrics to this CSV file (appending if exists)")
    return p.parse_args()


def load_slices(slice_paths, time_points, indices, layer,
                annotation_key, need_normalize, need_log_transform, need_scale,
                n_genes, n_pca, fit_pca=True, pca_model=None):
    adatas = []
    for i in indices:
        ad_i = sc.read_h5ad(slice_paths[i])
        if layer and layer in ad_i.layers: ad_i.X = ad_i.layers[layer]
        ad_i.obs["time"]     = float(time_points[i])
        ad_i.obs["slice_id"] = i
        adatas.append(ad_i)

    combined = ad.concat(adatas, join="inner")
    combined.obs_names_make_unique()

    if need_normalize: sc.pp.normalize_total(combined, target_sum=1e6)
    if need_log_transform: sc.pp.log1p(combined)

    if fit_pca:
        try:
            sc.pp.highly_variable_genes(combined, n_top_genes=n_genes, flavor="seurat")
        except Exception:
            sc.pp.highly_variable_genes(combined, n_top_genes=n_genes, flavor="seurat_v3")
        hvg_mask = combined.var["highly_variable"].values
    else:
        hvg_mask = np.ones(combined.shape[1], dtype=bool)

    combined_hvg = combined[:, hvg_mask].copy()
    if need_scale:
        import scipy.sparse as sp_mod
        if sp_mod.issparse(combined_hvg.X): combined_hvg.X = combined_hvg.X.toarray()
        for sid in combined_hvg.obs["slice_id"].unique():
            mask = (combined_hvg.obs["slice_id"] == sid).values
            sl = combined_hvg[mask].copy(); sc.pp.scale(sl)
            combined_hvg.X[mask] = sl.X

    X_raw = (combined_hvg.X.toarray() if hasattr(combined_hvg.X, "toarray")
             else np.array(combined_hvg.X))

    if fit_pca:
        pca_model = PCA(n_components=n_pca, random_state=42, whiten=True)
        X_pca = pca_model.fit_transform(X_raw)
    else:
        X_pca = pca_model.transform(X_raw)

    spatial_all = np.zeros((len(combined_hvg), 2))
    for j, i in enumerate(indices):
        mask = (combined_hvg.obs["slice_id"] == i).values
        raw  = load_spatial_coords(combined[mask])
        if raw is not None:
            spatial_all[mask] = normalize_coords_per_slice(raw)
    combined_hvg.obsm["X_pca"]   = X_pca
    combined_hvg.obsm["spatial"] = spatial_all

    slices = {}
    for i in indices:
        mask = (combined_hvg.obs["slice_id"] == i).values
        labels = None
        if annotation_key and annotation_key in combined_hvg.obs.columns:
            labels = combined_hvg.obs[annotation_key].values[mask].astype(str)
        slices[i] = {
            "X_raw": X_raw[mask], "X_pca": X_pca[mask],
            "S": spatial_all[mask], "labels": labels, "time": time_points[i],
        }
    return combined_hvg, pca_model, slices, hvg_mask if fit_pca else None


def main():
    args = parse_args()
    np.random.seed(args.seed)

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
    assert 0 < ho_idx < n_slices - 1

    held_out_time = time_points[ho_idx]
    train_indices = [i for i in range(n_slices) if i != ho_idx]
    source_idx    = max([i for i in train_indices if time_points[i] < held_out_time],
                        key=lambda i: time_points[i])

    run_name   = f"loto_stories_{args.data}_t{held_out_time}_{time.strftime('%m%d-%H%M')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Tier 3 LOTO - STORIES")
    print(f"  Held-out: idx={ho_idx}, t={held_out_time}")
    print(f"  Source:   idx={source_idx}, t={time_points[source_idx]}")
    print("=" * 70)

    if args.use_wandb:
        import wandb; wandb.login()
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    print("\n[1/3] Loading training slices...")
    train_adata, pca_model, train_slices, hvg_mask = load_slices(
        slice_paths, time_points, train_indices, args.layer, ann_key,
        args.need_normalize, args.need_log_transform, args.need_scale,
        args.n_genes, args.n_pca, fit_pca=True,
    )

    print("\n[2/3] Training STORIES on training slices...")
    if not HAS_STORIES:
        raise ImportError("stories-jax not installed")

    tau = float(held_out_time - time_points[source_idx])
    model = SpaceTime(
        n_steps=1, epsilon=args.epsilon, quadratic=True,
        debias=True, quadratic_weight=args.quadratic_weight,
    )
    t_train = time.time()
    model.fit(
        adata=train_adata, time_key="time", omics_key="X_pca", space_key="spatial",
        optimizer=optax.adamw(args.lr),
        max_iter=args.max_iter, batch_size=args.batch_size,
        patience=args.patience, checkpoint_manager=None,
    )
    print(f"  Training: {time.time()-t_train:.1f}s")

    src_adata = ad.AnnData(X=train_slices[source_idx]["X_raw"])
    src_adata.obsm["X_pca"]   = train_slices[source_idx]["X_pca"]
    src_adata.obsm["spatial"] = train_slices[source_idx]["S"]

    n_src = len(train_slices[source_idx]["X_raw"])
    transform_batch = min(args.batch_size, n_src)
    X_pred_pca = model.transform(
        adata=src_adata, omics_key="X_pca", tau=tau, batch_size=transform_batch
    )
    X_pred = pca_model.inverse_transform(X_pred_pca)

    t0 = time_points[source_idx]
    print(f"  X_pred: {X_pred.shape}")

    print("\n[3/3] Loading held-out slice for evaluation...")
    import scanpy as sc_ho
    ho_adata_raw = sc_ho.read_h5ad(slice_paths[ho_idx])
    if args.layer and args.layer in ho_adata_raw.layers:
        ho_adata_raw.X = ho_adata_raw.layers[args.layer]
    if args.need_normalize:
        sc_ho.pp.normalize_total(ho_adata_raw, target_sum=1e6)
    if args.need_log_transform:
        sc_ho.pp.log1p(ho_adata_raw)

    common_genes = [g for g in train_adata.var_names if g in ho_adata_raw.var_names]
    ho_X_hvg = (ho_adata_raw[:, common_genes].X.toarray()
                if hasattr(ho_adata_raw[:, common_genes].X, "toarray")
                else np.array(ho_adata_raw[:, common_genes].X))
    ho_S_raw = load_spatial_coords(ho_adata_raw)
    ho_S = normalize_coords_per_slice(ho_S_raw) if ho_S_raw is not None else None
    ho_labels = None
    if ann_key and ann_key in ho_adata_raw.obs.columns:
        ho_labels = ho_adata_raw.obs[ann_key].astype(str).values
    print(f"  X_true: {ho_X_hvg.shape}")

    metrics = compute_all_metrics(
        X_pred=X_pred, X_true_t1=ho_X_hvg,
        S_true_t1=ho_S, labels_t1=ho_labels,
        labels_t0=train_slices[source_idx]["labels"],
        hvg_k=args.evaluate_top_k,
    )
    print_metrics_summary(metrics, "STORIES (LOTO)", t0, held_out_time)
    plot_metrics_bar(metrics, "STORIES_LOTO", t0, held_out_time,
                     str(output_dir / "loto_metrics.png"))

    rows = [{"method": "STORIES_LOTO", "from_time": t0,
             "to_time": held_out_time, "time_span": f"{t0}->{held_out_time}", **metrics}]
    df = pd.DataFrame(rows).reindex(columns=CANONICAL_COLS)
    df.to_csv(output_dir / "loto_results.csv", index=False)

    import os
    if args.metrics_output_csv:
        results_csv = args.metrics_output_csv
    else:
        results_csv = f"metrics/comparison_results_loto_{args.data}.csv"
    os.makedirs("metrics", exist_ok=True)
    df.to_csv(results_csv, mode="a" if os.path.exists(results_csv) else "w",
              header=not os.path.exists(results_csv), index=False)
    print(f"\n  [CSV] Saved to {results_csv}")

    if args.use_wandb:
        try:
            import wandb
            if wandb.run:
                wandb.log({f"eval/{k}": v for k, v in metrics.items()
                           if isinstance(v, (int, float, np.floating))})
                wandb.finish()
        except Exception: pass


if __name__ == "__main__":
    main()
