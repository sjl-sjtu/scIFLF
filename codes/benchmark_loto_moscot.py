"""
Tier 3 leave-one-timepoint-out benchmark for Moscot.

Strict LOTO setting: prediction never reads the held-out slice. Moscot estimates a transport-induced velocity field in PCA space using only training slices, then extrapolates from t0 to the held-out t1. The held-out slice is used only after prediction for evaluation.

Example:
  python benchmark_loto_moscot.py --data mouse_embryo --holdout_idx 3
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
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_utils import (
    DATASET_CONFIG, CANONICAL_COLS, compute_all_metrics,
    plot_metrics_bar, print_metrics_summary,
    load_spatial_coords, normalize_coords_per_slice,
)

try:
    from moscot.problems.spatiotemporal import SpatioTemporalProblem
    HAS_MOSCOT = True
except ImportError:
    try:
        from moscot.problems.spatio_temporal import SpatioTemporalProblem
        HAS_MOSCOT = True
    except ImportError:
        HAS_MOSCOT = False
        print("WARNING: moscot not installed.")

BASE_DATA_DIR      = "./spatiotemporal"
DEFAULT_OUTPUT_DIR = "./spatiotemporal/outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Tier 3 LOTO - Moscot")
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
    p.add_argument("--alpha",        type=float, default=0.5)
    p.add_argument("--epsilon",      type=float, default=0.1)
    p.add_argument("--tau_a",        type=float, default=0.95)
    p.add_argument("--tau_b",        type=float, default=0.95)
    p.add_argument("--rank",         type=int,   default=1000,
                   help="Low-rank approximation rank; recommended 500-2000 for large datasets, -1 means full rank")
    p.add_argument("--subsample_n",  type=int,   default=10000,
                   help="Maximum number of cells before OT solving; -1 disables subsampling")
    p.add_argument("--ode_steps",    type=int,   default=100)
    p.add_argument("--ode_method",   type=str,   default="RK45")
    p.add_argument("--vf_ctrl_n",    type=int,   default=8000,
                   help="Maximum number of RBF velocity-field control points; -1 disables subsampling")
    p.add_argument("--use_wandb",    action="store_true")
    p.add_argument("--wandb_project", type=str,  default="loto-moscot")
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


def solve_pair_transport(sub_src_X, sub_src_S, sub_tgt_X, sub_tgt_S,
                         t_src, t_tgt, args):
    pair_adata = ad.AnnData(X=np.vstack([sub_src_X, sub_tgt_X]))
    pair_adata.obs["time"] = np.array([t_src] * len(sub_src_X) + [t_tgt] * len(sub_tgt_X))
    pair_adata.obsm["X_pca"] = np.vstack([sub_src_X, sub_tgt_X])
    pair_adata.obsm["spatial"] = np.vstack([sub_src_S, sub_tgt_S])
    pair_adata.obs_names = [f"cell_{j}" for j in range(len(sub_src_X) + len(sub_tgt_X))]

    stp = SpatioTemporalProblem(adata=pair_adata)
    stp = stp.prepare(
        time_key="time",
        spatial_key="spatial",
        joint_attr={"attr": "obsm", "key": "X_pca"},
    )
    solve_kwargs = dict(
        alpha=args.alpha,
        epsilon=args.epsilon,
        tau_a=args.tau_a,
        tau_b=args.tau_b,
        scale_cost="mean",
    )
    if args.rank > 0:
        solve_kwargs["rank"] = args.rank
    stp = stp.solve(**solve_kwargs)
    T = np.array(stp[(t_src, t_tgt)].solution.transport_matrix)
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


def estimate_velocity_field(train_slices, train_indices, time_points, args):
    """
    Estimate a transport-induced velocity field from training slices only.

    For each adjacent training time pair, solve OT, use barycentric projection
    to estimate source-cell states at the next time point, and treat
    displacement divided by dt as instantaneous velocity.
    """
    rng = np.random.default_rng(args.seed)
    ordered = sorted(train_indices, key=lambda i: time_points[i])
    ctrl_X, ctrl_V = [], []

    for left, right in zip(ordered[:-1], ordered[1:]):
        t_src = time_points[left]
        t_tgt = time_points[right]
        dt = float(t_tgt - t_src)
        if dt <= 1e-8:
            continue

        src_X = train_slices[left]["X_pca"]
        src_S = train_slices[left]["S"]
        tgt_X = train_slices[right]["X_pca"]
        tgt_S = train_slices[right]["S"]

        n_src = len(src_X)
        n_tgt = len(tgt_X)
        subsample_n = getattr(args, "subsample_n", 10000)
        do_subsample = subsample_n > 0 and (n_src > subsample_n or n_tgt > subsample_n)

        if do_subsample:
            src_ids = rng.choice(n_src, min(subsample_n, n_src), replace=False)
            tgt_ids = rng.choice(n_tgt, min(subsample_n, n_tgt), replace=False)
            sub_src_X = src_X[src_ids]
            sub_src_S = src_S[src_ids]
            sub_tgt_X = tgt_X[tgt_ids]
            sub_tgt_S = tgt_S[tgt_ids]
            print(f"  [Moscot] Pair {t_src}->{t_tgt} src {n_src}->{len(src_ids)}, "
                  f"tgt {n_tgt}->{len(tgt_ids)}")
        else:
            src_ids = np.arange(n_src)
            tgt_ids = np.arange(n_tgt)
            sub_src_X = src_X
            sub_src_S = src_S
            sub_tgt_X = tgt_X
            sub_tgt_S = tgt_S

        T_norm = solve_pair_transport(
            sub_src_X=sub_src_X, sub_src_S=sub_src_S,
            sub_tgt_X=sub_tgt_X, sub_tgt_S=sub_tgt_S,
            t_src=t_src, t_tgt=t_tgt, args=args,
        )
        proj_sub = T_norm @ sub_tgt_X

        if do_subsample:
            from sklearn.neighbors import NearestNeighbors as _NN
            nn = _NN(n_neighbors=1).fit(sub_src_X)
            _, nn_ids = nn.kneighbors(src_X)
            proj_full = proj_sub[nn_ids.ravel()]
        else:
            proj_full = proj_sub

        vel = (proj_full - src_X) / dt
        ctrl_X.append(src_X)
        ctrl_V.append(vel)

    if not ctrl_X:
        raise RuntimeError("Failed to estimate Moscot velocity field from training pairs.")

    ctrl_X = np.vstack(ctrl_X).astype(np.float64)
    ctrl_V = np.vstack(ctrl_V).astype(np.float64)

    vf_ctrl_n = getattr(args, "vf_ctrl_n", -1)
    if vf_ctrl_n > 0 and len(ctrl_X) > vf_ctrl_n:
        keep = rng.choice(len(ctrl_X), vf_ctrl_n, replace=False)
        ctrl_X = ctrl_X[keep]
        ctrl_V = ctrl_V[keep]

    return ctrl_X, ctrl_V


def build_rbf_velocity_fn(ctrl_X, ctrl_V):
    if len(ctrl_X) == 1:
        mean_v = ctrl_V[0:1]

        def vf_func(x):
            x = np.atleast_2d(x)
            return np.repeat(mean_v, len(x), axis=0)

        return vf_func

    sample_n = min(2048, len(ctrl_X))
    sample = ctrl_X[:sample_n]
    diff = sample[:, None, :] - sample[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    upper = dist2[np.triu_indices_from(dist2, k=1)]
    median_dist2 = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    beta = 1.0 / max(median_dist2, 1e-6)

    def vf_func(x):
        x = np.atleast_2d(x).astype(np.float64)
        diff = x[:, None, :] - ctrl_X[None, :, :]
        weights = np.exp(-beta * np.sum(diff * diff, axis=2))
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_sum[weights_sum == 0] = 1.0
        weights /= weights_sum
        return weights @ ctrl_V

    return vf_func


def run_vf_ode(src_X, tau, vf_func, ode_steps, ode_method):
    tau = float(tau)
    if abs(tau) < 1e-8:
        return src_X.astype(np.float64)

    n_src, n_dims = src_X.shape
    X_pred = np.zeros_like(src_X, dtype=np.float64)
    batch_size = 500
    for start in range(0, n_src, batch_size):
        end = min(start + batch_size, n_src)
        nb = end - start
        x0 = src_X[start:end].astype(np.float64).ravel()

        def batch_ode(_t, y_flat, _nb=nb, _nd=n_dims):
            Y = y_flat.reshape(_nb, _nd)
            return vf_func(Y).ravel()

        sol = solve_ivp(
            batch_ode, (0.0, tau), x0,
            method=ode_method, t_eval=[tau],
            max_step=max(abs(tau) / max(ode_steps, 1), 1e-3),
        )
        if sol.success:
            X_pred[start:end] = sol.y[:, -1].reshape(nb, n_dims)
        else:
            X_pred[start:end] = src_X[start:end] + tau * vf_func(src_X[start:end])
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

    run_name   = f"loto_moscot_{args.data}_t{held_out_time}_{time.strftime('%m%d-%H%M')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Tier 3 LOTO - Moscot")
    print(f"  Held-out: idx={ho_idx}, t={held_out_time}")
    print(f"  Source: t={time_points[source_idx]}")
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

    print("\n[2/3] Estimating OT-induced velocity field from training slices...")
    if not HAS_MOSCOT:
        raise ImportError("moscot not installed")

    t_src      = time_points[source_idx]
    t_holdout  = held_out_time

    t_start = time.time()
    ctrl_X, ctrl_V = estimate_velocity_field(train_slices, train_indices, time_points, args)
    vf_func = build_rbf_velocity_fn(ctrl_X, ctrl_V)

    print("\n[3/3] Predicting held-out timepoint from source slice...")
    src_X_pca = train_slices[source_idx]["X_pca"]
    tau = float(t_holdout - t_src)
    X_pred_pca = run_vf_ode(
        src_X=src_X_pca, tau=tau, vf_func=vf_func,
        ode_steps=args.ode_steps, ode_method=args.ode_method,
    )
    X_pred = pca_model.inverse_transform(X_pred_pca)
    print(f"  Prediction done in {time.time()-t_start:.1f}s  X_pred: {X_pred.shape}")

    print("\n[Eval] Loading held-out slice after prediction...")
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

    metrics = compute_all_metrics(
        X_pred=X_pred, X_true_t1=ho_X,
        S_true_t1=ho_S, labels_t1=ho_labels,
        labels_t0=train_slices[source_idx]["labels"],
        hvg_k=args.evaluate_top_k,
    )
    print_metrics_summary(metrics, "Moscot (LOTO)", t_src, t_holdout)
    plot_metrics_bar(metrics, "Moscot_LOTO", t_src, t_holdout,
                     str(output_dir / "loto_metrics.png"))

    rows = [{"method": "Moscot_LOTO", "from_time": t_src,
             "to_time": t_holdout, "time_span": f"{t_src}->{t_holdout}", **metrics}]
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
