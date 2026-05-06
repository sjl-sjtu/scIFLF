"""
Tier 3 leave-one-timepoint-out benchmark for Spateo/Dynamo.

Strict LOTO setting: prediction never reads the held-out slice. Spateo estimates pseudo-velocity from training slices, reconstructs a SparseVFC vector field, and integrates source cells from t0 in PCA space. The held-out slice is used only after prediction for evaluation.

Example:
  python benchmark_loto_spateo.py --data mouse_embryo --holdout_idx 3
"""
import os, sys, time, json, argparse, random, warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
if not hasattr(np, "bool8"): np.bool8 = np.bool_

import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib; matplotlib.use("Agg")
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.integrate import solve_ivp
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_utils import (
    DATASET_CONFIG, CANONICAL_COLS, compute_all_metrics,
    plot_metrics_bar, print_metrics_summary,
    load_spatial_coords, normalize_coords_per_slice,
)

try:
    import dynamo as dyn
    HAS_DYNAMO = True
except ImportError:
    HAS_DYNAMO = False
    print("WARNING: dynamo not installed.")

BASE_DATA_DIR      = "./spatiotemporal"
DEFAULT_OUTPUT_DIR = "./spatiotemporal/outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Tier 3 LOTO - Spateo/Dynamo")
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
    p.add_argument("--tau",          type=float, default=0.0)
    p.add_argument("--ode_steps",    type=int,   default=100)
    p.add_argument("--ode_method",   type=str,   default="RK45")
    p.add_argument("--vf_method",    type=str,   default="SparseVFC")
    p.add_argument("--subsample_n",  type=int,   default=20000)
    p.add_argument("--M_ctrl",       type=int,   default=500)
    p.add_argument("--use_wandb",    action="store_true")
    p.add_argument("--wandb_project", type=str,  default="loto-spateo")
    p.add_argument("--metrics_output_csv", type=str, default=None,
                   help="If specified, also save metrics to this CSV file (appending if exists)")
    return p.parse_args()


def load_train_slices(slice_paths, time_points, train_indices, layer, annotation_key,
                      need_normalize, need_log_transform, need_scale, n_genes, n_pca):
    adatas = []
    for i in train_indices:
        ad_i = sc.read_h5ad(slice_paths[i])
        if layer and layer in ad_i.layers: ad_i.X = ad_i.layers[layer]
        ad_i.obs["time"]     = float(time_points[i])
        ad_i.obs["slice_id"] = i
        adatas.append(ad_i)

    combined = ad.concat(adatas, join="inner")
    combined.obs_names_make_unique()

    if need_normalize: sc.pp.normalize_total(combined, target_sum=1e6)
    if need_log_transform: sc.pp.log1p(combined)
    try:
        sc.pp.highly_variable_genes(combined, n_top_genes=n_genes, flavor="seurat")
    except Exception:
        sc.pp.highly_variable_genes(combined, n_top_genes=n_genes, flavor="seurat_v3")

    combined_hvg = combined[:, combined.var["highly_variable"].values].copy()
    if need_scale:
        import scipy.sparse as sp_mod
        if sp_mod.issparse(combined_hvg.X): combined_hvg.X = combined_hvg.X.toarray()
        for sid in combined_hvg.obs["slice_id"].unique():
            mask = (combined_hvg.obs["slice_id"] == sid).values
            sl = combined_hvg[mask].copy(); sc.pp.scale(sl)
            combined_hvg.X[mask] = sl.X

    X_raw = (combined_hvg.X.toarray() if hasattr(combined_hvg.X, "toarray")
             else np.array(combined_hvg.X))
    pca_model = PCA(n_components=n_pca, random_state=42, whiten=True)
    X_pca_all = pca_model.fit_transform(X_raw)

    spatial_all = np.zeros((len(combined_hvg), 2))
    for i in train_indices:
        mask = (combined_hvg.obs["slice_id"] == i).values
        raw  = load_spatial_coords(combined[mask])
        if raw is not None:
            spatial_all[mask] = normalize_coords_per_slice(raw)
    combined_hvg.obsm["X_pca"]   = X_pca_all
    combined_hvg.obsm["spatial"] = spatial_all

    slices = {}
    for i in train_indices:
        mask = (combined_hvg.obs["slice_id"] == i).values
        labels = None
        if annotation_key and annotation_key in combined_hvg.obs.columns:
            labels = combined_hvg.obs[annotation_key].values[mask].astype(str)
        slices[i] = {
            "X_raw": X_raw[mask], "X_pca": X_pca_all[mask],
            "S": spatial_all[mask], "labels": labels,
        }
    return combined_hvg, pca_model, slices


def estimate_velocity_all(combined_adata, X_pca_all, slices, train_indices, time_points):
    times        = combined_adata.obs["time"].values
    unique_times = sorted(np.unique(times))
    velocity     = np.zeros_like(X_pca_all)

    time_mean = {t: X_pca_all[times == t].mean(0) for t in unique_times}

    for j, t in enumerate(unique_times):
        mask = times == t
        cells = X_pca_all[mask]
        mean_t = time_mean[t]

        if j < len(unique_times) - 1:
            t_next = unique_times[j + 1]
            dt = t_next - t
            base_dir = (time_mean[t_next] - mean_t) / (dt + 1e-8)
            dev = cells - mean_t
            velocity[mask] = base_dir + 0.5 * dev / (dt + 1e-8)
        else:
            t_prev = unique_times[j - 1]
            dt = t - t_prev
            velocity[mask] = (mean_t - time_mean[t_prev]) / (dt + 1e-8)

    return velocity


def run_vf_ode(X_train, src_X, velocity, tau, ode_steps, ode_method,
               vf_method, subsample_n, M_ctrl, seed):
    if not HAS_DYNAMO:
        mean_v = velocity.mean(0)
        return (src_X + tau * mean_v).astype(np.float64)

    n_dims = X_train.shape[1]
    rng    = np.random.default_rng(seed)
    if subsample_n > 0 and len(X_train) > subsample_n:
        idx   = rng.choice(len(X_train), subsample_n, replace=False)
        X_sub = X_train[idx]; V_sub = velocity[idx]
    else:
        X_sub = X_train; V_sub = velocity

    dyn_adata = ad.AnnData(X=X_sub.astype(np.float32))
    dyn_adata.obsm["X_pca"]        = X_sub.astype(np.float64)
    dyn_adata.obsm["velocity_pca"] = V_sub.astype(np.float64)
    dyn_adata.var_names = [f"dim{j+1}" for j in range(n_dims)]
    dyn_adata.var["use_for_transition"] = True
    dyn_adata.uns["pp"] = {"has_splicing": False, "has_labeling": False,
                           "has_protein": False, "has_velocity": True, "norm_method": "None"}
    dyn_adata.uns["dynamics"] = {
        "filter_gene_mode": "final", "t": None, "group": None,
        "X_data": None, "X_fit_data": None, "asspt_mRNA": "ss",
        "experiment_type": "conventional", "normalized": True,
        "model": "deterministic", "est_method": "tweedieDecon",
    }

    try:
        dyn.tl.neighbors(dyn_adata, basis="pca", result_prefix="pca")
    except Exception:
        sc.pp.neighbors(dyn_adata, use_rep="X_pca", n_neighbors=30)

    try:
        dyn.vf.VectorField(dyn_adata, basis="pca", method=vf_method,
                           restart_num=3, restart_seed=[0, 42, 123],
                           min_vel_corr=0.3, M=M_ctrl)
        vf_params = dyn_adata.uns["VecFld_pca"]
        X_ctrl = vf_params["X_ctrl"]; C = vf_params["C"]
        beta   = float(vf_params["beta"])

        def vf_func(x):
            x = np.atleast_1d(x).reshape(1, -1)
            K = np.exp(-beta * np.sum((x - X_ctrl) ** 2, axis=1))
            return K @ C
    except Exception as e:
        print(f"  [WARN] VF failed ({e}), Euler fallback")
        mean_v = velocity.mean(0)
        return (src_X + tau * mean_v).astype(np.float64)

    n_src = src_X.shape[0]
    X_pred = np.zeros_like(src_X, dtype=np.float64)
    batch_size = 500
    for start in range(0, n_src, batch_size):
        end = min(start + batch_size, n_src)
        nb  = end - start
        x0  = src_X[start:end].astype(np.float64).ravel()

        def batch_ode(t, y_flat, _nb=nb, _nd=n_dims):
            Y  = y_flat.reshape(_nb, _nd)
            dY = np.vstack([np.atleast_1d(vf_func(Y[j])) for j in range(_nb)])
            return dY.ravel()

        sol = solve_ivp(batch_ode, (0.0, tau), x0, method=ode_method,
                        t_eval=[tau], max_step=tau / max(ode_steps, 1))
        if sol.success:
            X_pred[start:end] = sol.y[:, -1].reshape(nb, n_dims)
        else:
            v_batch = np.vstack([np.atleast_1d(vf_func(src_X[j])) for j in range(start, end)])
            X_pred[start:end] = src_X[start:end] + tau * v_batch
    return X_pred


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

    tau = args.tau if args.tau > 0 else (held_out_time - time_points[source_idx])

    run_name   = f"loto_spateo_{args.data}_t{held_out_time}_{time.strftime('%m%d-%H%M')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Tier 3 LOTO - Spateo/Dynamo")
    print(f"  Held-out: idx={ho_idx}, t={held_out_time}")
    print(f"  Source: t={time_points[source_idx]}, tau={tau}")
    print("=" * 70)

    if args.use_wandb:
        import wandb; wandb.login()
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    print("\n[1/3] Loading training slices...")
    train_adata, pca_model, train_slices = load_train_slices(
        slice_paths, time_points, train_indices, args.layer, ann_key,
        args.need_normalize, args.need_log_transform, args.need_scale,
        args.n_genes, args.n_pca,
    )

    print("\n[2/3] Estimating velocity and running ODE...")
    t_start = time.time()

    X_pca_all = train_adata.obsm["X_pca"]
    velocity  = estimate_velocity_all(train_adata, X_pca_all, train_slices,
                                      train_indices, time_points)

    src_mask  = (train_adata.obs["slice_id"] == source_idx).values
    src_X_pca = X_pca_all[src_mask]

    X_pred_pca = run_vf_ode(
        X_train=X_pca_all, src_X=src_X_pca,
        velocity=velocity, tau=tau,
        ode_steps=args.ode_steps, ode_method=args.ode_method,
        vf_method=args.vf_method, subsample_n=args.subsample_n,
        M_ctrl=args.M_ctrl, seed=args.seed,
    )
    X_pred = pca_model.inverse_transform(X_pred_pca)
    print(f"  Done in {time.time()-t_start:.1f}s  X_pred: {X_pred.shape}")

    print("\n[3/3] Loading held-out slice for evaluation...")
    ho_raw = sc.read_h5ad(slice_paths[ho_idx])
    if args.layer and args.layer in ho_raw.layers:
        ho_raw.X = ho_raw.layers[args.layer]
    if args.need_normalize:
        sc.pp.normalize_total(ho_raw, target_sum=1e6)
    if args.need_log_transform:
        sc.pp.log1p(ho_raw)
    common = [g for g in train_adata.var_names if g in ho_raw.var_names]
    ho_X = (ho_raw[:, common].X.toarray() if hasattr(ho_raw[:, common].X, "toarray")
            else np.array(ho_raw[:, common].X))
    ho_S_raw = load_spatial_coords(ho_raw)
    ho_S = normalize_coords_per_slice(ho_S_raw) if ho_S_raw is not None else None
    ho_labels = None
    if ann_key and ann_key in ho_raw.obs.columns:
        ho_labels = ho_raw.obs[ann_key].astype(str).values

    t0 = time_points[source_idx]
    metrics = compute_all_metrics(
        X_pred=X_pred, X_true_t1=ho_X,
        S_true_t1=ho_S, labels_t1=ho_labels,
        labels_t0=train_slices[source_idx]["labels"],
        hvg_k=args.evaluate_top_k,
    )
    print_metrics_summary(metrics, "Spateo/Dynamo (LOTO)", t0, held_out_time)
    plot_metrics_bar(metrics, "Spateo_LOTO", t0, held_out_time,
                     str(output_dir / "loto_metrics.png"))

    rows = [{"method": "Spateo_LOTO", "from_time": t0,
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
