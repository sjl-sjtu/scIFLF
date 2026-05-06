"""
Shared evaluation utilities for benchmark scripts.

Metrics:
  x_hvg_corr: Pearson correlation of Top-K HVG mean vectors (higher is better)
  x_wasserstein: median per-gene Wasserstein distance (lower is better)
  spatial_domain_ari: Leiden clusters from predicted expression on true t1 coordinates vs. true annotations (higher is better)
  spatial_coherence_pc1: Moran's I for predicted-expression PC1 on true t1 coordinates (higher is better)
"""
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import adjusted_rand_score
from scipy.stats import wasserstein_distance
from scipy.spatial import procrustes as scipy_procrustes
import scipy.sparse as sp
import json
import os
from typing import Optional


# ============================================================
# Basic helper functions
# ============================================================

def safe_corr_np(pred: np.ndarray, actual: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute row-wise Pearson correlation, mapping (N, D) arrays to (N,)."""
    corrs = []
    for i in range(pred.shape[0]):
        p = pred[i] - pred[i].mean()
        a = actual[i] - actual[i].mean()
        denom = np.linalg.norm(p) * np.linalg.norm(a) + eps
        corrs.append(float((p * a).sum() / denom))
    return np.array(corrs)


def compute_hvg_indices(x_np: np.ndarray, k: int = 100) -> np.ndarray:
    """
    Return Top-K HVG column indices based on the true t1 data.

    For positive log-CPM inputs, use Scanpy's Seurat flavor. For z-scored
    inputs with negative values, fall back to variance ranking because Seurat
    expects non-negative data.
    """
    import scanpy as sc
    import anndata as ad
    if k >= x_np.shape[1]:
        return np.arange(x_np.shape[1])

    has_negatives = bool(x_np.min() < -0.5)

    if not has_negatives:
        adata = ad.AnnData(X=x_np + np.random.uniform(0, 1e-8, size=x_np.shape))
        sc.pp.highly_variable_genes(adata, n_top_genes=k, flavor="seurat")
        return np.where(adata.var["highly_variable"])[0]

    return np.argsort(x_np.var(axis=0))[::-1][:k]


def wasserstein_median(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute the median per-gene 1D Wasserstein distance."""
    n = min(pred.shape[1], true.shape[1])
    return float(np.median([wasserstein_distance(pred[:, g], true[:, g]) for g in range(n)]))


def morans_I_sparse(coords: np.ndarray, values: np.ndarray, k: int = 10) -> float:
    """Compute sparse KNN Moran's I."""
    N = len(values)
    if N < k + 2:
        return float("nan")
    W = kneighbors_graph(coords, n_neighbors=min(k, N - 1),
                         mode="connectivity", include_self=False).astype(float)
    row_sums = np.array(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W = sp.diags(1.0 / row_sums) @ W
    z = values - values.mean()
    z = z / (z.std() + 1e-8)
    return float(z @ (W @ z)) / N


# ============================================================
# Core spatial-coherence metric, following the STORIES definition.
# ============================================================

def compute_spatial_coherence(
    X_pred: np.ndarray,      # (N_pred, n_genes) predicted gene expression
    S_true_t1: np.ndarray,   # (N_true, 2) true t1 spatial coordinates
    X_true_t1: np.ndarray,   # (N_true, n_genes) true t1 gene expression
    n_pc: int = 1,
    k_moran: int = 10,
) -> dict:
    """Compute Moran's I for predicted-expression PC1 on true t1 coordinates."""
    pca = SklearnPCA(n_components=n_pc, random_state=42)
    pca.fit(X_pred)
    proj_true = X_true_t1 @ pca.components_[0]
    return {"spatial_coherence_pc1": morans_I_sparse(S_true_t1, proj_true, k=k_moran)}


# ============================================================
# Tier 2 cell-type-level metrics
# ============================================================

def compute_celltype_metrics(
    X_pred: np.ndarray,             # (N_pred, n_genes) predicted gene expression
    X_true_t1: np.ndarray,          # (N_true, n_genes) true t1 gene expression
    S_true_t1: np.ndarray,          # (N_true, 2) true t1 coordinates
    labels_t1: np.ndarray,          # (N_true,) true t1 cell-type labels
    labels_t0: Optional[np.ndarray] = None,
    n_pca_ct: int = 30,
    n_leiden_neighbors: int = 15,
    leiden_resolution: float = 1.0,
) -> dict:
    """Compute spatial_domain_ari from predicted-expression spatial clusters."""
    import scanpy as sc
    import anndata as ad

    n_pc = min(n_pca_ct, X_true_t1.shape[1], X_true_t1.shape[0] - 1,
               X_pred.shape[1], X_pred.shape[0] - 1)
    pca  = SklearnPCA(n_components=n_pc, random_state=42)
    pca.fit(X_pred)
    true_in_pred_space = pca.transform(X_true_t1)

    adata_eval = ad.AnnData(X=true_in_pred_space)
    adata_eval.obsm["spatial"] = S_true_t1
    sc.pp.neighbors(adata_eval, n_neighbors=n_leiden_neighbors, use_rep="X")
    sc.tl.leiden(adata_eval, resolution=leiden_resolution, key_added="leiden_pred")

    pred_clusters = adata_eval.obs["leiden_pred"].values.astype(str)
    return {"spatial_domain_ari": float(adjusted_rand_score(
        labels_t1.astype(str), pred_clusters
    ))}


# ============================================================
# Main evaluation entry point
# ============================================================

def compute_all_metrics(
    X_pred: np.ndarray,             # (N_pred, n_genes) predicted expression
    X_true_t1: np.ndarray,          # (N_true, n_genes) true t1 expression
    S_true_t1: np.ndarray,          # (N_true, 2) true t1 spatial coordinates
    labels_t1: Optional[np.ndarray] = None,  # (N_true,) cell-type labels
    labels_t0: Optional[np.ndarray] = None,  # (N_src,) source cell-type labels
    hvg_k: int = 100,
    k_moran: int = 10,
    compute_tier2: bool = True,
) -> dict:
    """
    Compute the shared benchmark metrics used by all benchmark scripts.

    Metrics: x_hvg_corr, x_wasserstein, spatial_coherence_pc1, and
    spatial_domain_ari.
    """
    result = {}

    # -- x_hvg_corr, x_wasserstein -------------------------------
    hvg_idx = compute_hvg_indices(X_true_t1, k=hvg_k)
    X_pred_hvg  = X_pred[:, hvg_idx]    if X_pred.shape[1] > max(hvg_idx) else X_pred
    X_true_hvg  = X_true_t1[:, hvg_idx] if X_true_t1.shape[1] > max(hvg_idx) else X_true_t1

    p_mean = X_pred_hvg.mean(axis=0);  a_mean = X_true_hvg.mean(axis=0)
    p_ = p_mean - p_mean.mean();       a_ = a_mean - a_mean.mean()
    denom = np.linalg.norm(p_) * np.linalg.norm(a_) + 1e-8
    result["x_hvg_corr"] = float((p_ * a_).sum() / denom)
    result["x_wasserstein"] = wasserstein_median(X_pred_hvg, X_true_hvg)

    # -- spatial_coherence_pc1 ------------------------------------
    result.update(compute_spatial_coherence(X_pred, S_true_t1, X_true_t1, k_moran=k_moran))

    # -- spatial_domain_ari ---------------------------------------
    if compute_tier2 and labels_t1 is not None and len(labels_t1) > 0:
        result.update(compute_celltype_metrics(
            X_pred=X_pred, X_true_t1=X_true_t1,
            S_true_t1=S_true_t1, labels_t1=labels_t1, labels_t0=labels_t0,
        ))
    else:
        result["spatial_domain_ari"] = float("nan")

    return result


# ============================================================
# Result saving
# ============================================================

def save_results(
    metrics: dict,
    method: str,
    from_time: float,
    to_time: float,
    output_dir,
    data_name: str,
    extra_info: dict = None,
):
    """Save evaluation results to CSV and JSON."""
    import json, os
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "method":    method,
        "from_time": from_time,
        "to_time":   to_time,
        "time_span": f"{from_time}->{to_time}",
        **{k: (float(v) if isinstance(v, (float, np.floating, int)) else v)
           for k, v in metrics.items()},
    }
    df = pd.DataFrame([row])
    cols = ["method", "from_time", "to_time", "time_span"] + \
           [c for c in df.columns if c not in ("method", "from_time", "to_time", "time_span")]
    df = df[cols]

    results_csv = f"metrics/comparison_results_{data_name}.csv"
    os.makedirs("metrics", exist_ok=True)
    df.to_csv(results_csv, mode="a" if os.path.exists(results_csv) else "w",
              header=not os.path.exists(results_csv), index=False)
    print(f"  [CSV] Saved to {results_csv}")

    report = {"method": method, "from_time": from_time, "to_time": to_time,
              **(extra_info or {}),
              "metrics": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                          for k, v in metrics.items()}}
    json_path = output_dir / f"{method}_results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [JSON] Saved to {json_path}")
    return str(results_csv)


# ============================================================
# Visualization
# ============================================================

def plot_metrics_bar(metrics: dict, method: str, from_time: float, to_time: float,
                     save_path: str):
    """Plot a standard metric bar chart."""
    import matplotlib.pyplot as plt

    keys   = ["x_hvg_corr", "x_wasserstein", "spatial_domain_ari", "spatial_coherence_pc1"]
    keys   = [k for k in keys if k in metrics]
    vals   = [metrics[k] for k in keys]
    colors = ["steelblue"] * len(keys)

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.5), 5))
    ax.bar(keys, [v if not np.isnan(v) else 0 for v in vals],
           color=colors, edgecolor="k", linewidth=0.5)
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, max(v, 0) + 0.005, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_title(f"{method}  |  t={from_time}->{to_time}", fontsize=11)
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Plot] Saved to {save_path}")


def print_metrics_summary(metrics: dict, method: str, from_time: float, to_time: float):
    print(f"\n{'='*70}")
    print(f"  {method}  |  t={from_time} -> {to_time}")
    print(f"{'='*70}")
    print(f"  x_hvg_corr          : {metrics.get('x_hvg_corr', float('nan')):.4f}  (higher is better)")
    print(f"  x_wasserstein       : {metrics.get('x_wasserstein', float('nan')):.4f}  (lower is better)")
    print(f"  spatial_domain_ari  : {metrics.get('spatial_domain_ari', float('nan')):.4f}  (higher is better)")
    print(f"  spatial_coherence_pc1: {metrics.get('spatial_coherence_pc1', float('nan')):.4f}  (higher is better)")
    print(f"{'='*70}")


# ============================================================
# Shared parameter loading utilities
# ============================================================

def load_dataset_params(data_name: str, config_obj, json_path: str = "dataset_params.json",
                        holdout_idx: "int | None" = None) -> bool:
    """
    Load dataset-specific optimized hyperparameters from JSON into Config.

    Supports two JSON formats:

    1. Flat format, with one parameter group per dataset:
    {
        "axolotl_telencephalon_regeneration": {
            "lambda_recon": 1.0,
            "lambda_pattern_smooth": 0.01
        }
    }

    2. Nested format keyed by holdout_idx:
    {
        "axolotl_telencephalon_regeneration": {
            "2": {"lambda_recon": 0.8, ...},
            "3": {"lambda_recon": 0.5, ...}
        }
    }

    When holdout_idx is not None, a matching nested fold is preferred. If no
    matching fold is found, the flat format is used when available.

    Args:
        data_name: Dataset name.
        config_obj: Config instance to modify in place.
        json_path: JSON file path. Defaults to "dataset_params.json".
        holdout_idx: Optional LOTO fold index for nested JSON entries.

    Returns:
        True if parameters were loaded and applied; otherwise False.
    """
    if not os.path.exists(json_path):
        print(f"  [Config] {json_path} not found, using default parameters.")
        return False

    with open(json_path, "r") as f:
        all_params = json.load(f)

    if data_name not in all_params:
        print(f"  [Config] No specific parameters for '{data_name}' in {json_path}, using defaults.")
        return False

    ds_entry = all_params[data_name]

    is_nested = isinstance(ds_entry, dict) and all(
        isinstance(v, dict) for v in ds_entry.values()
    )

    ds_params = None
    if is_nested and holdout_idx is not None:
        key = str(holdout_idx)
        if key in ds_entry:
            print(f"  [Config] Loading per-holdout parameters for '{data_name}' loto{holdout_idx} from {json_path}")
            ds_params = ds_entry[key]
        else:
            print(f"  [Config] No per-holdout entry for loto{holdout_idx} in '{data_name}', trying flat fallback.")

    if ds_params is None:
        if is_nested:
            print(f"  [Config] Nested format but no matching holdout_idx, using defaults.")
            return False
        print(f"  [Config] Loading optimized parameters for '{data_name}' from {json_path}")
        ds_params = ds_entry

    for k, v in ds_params.items():
        if hasattr(config_obj, k):
            old_val = getattr(config_obj, k)
            setattr(config_obj, k, v)
            print(f"    - {k}: {old_val} -> {v}")
        else:
            print(f"    - [Warning] Config has no attribute '{k}', skipping.")
    return True


# ============================================================
# Shared data loading utilities
# ============================================================

def load_spatial_coords(adata) -> Optional[np.ndarray]:
    """Extract spatial coordinates using several common AnnData obsm keys."""
    for key in ("spatial", "X_spatial", "Spatial", "X_Spatial"):
        if key in adata.obsm:
            return adata.obsm[key].astype(float)
    return None


def normalize_coords_per_slice(coords: np.ndarray) -> np.ndarray:
    """Apply per-slice min-max normalization to [0, 1]."""
    c_min = coords.min(axis=0)
    c_max = coords.max(axis=0)
    c_range = c_max - c_min
    c_range[c_range < 1e-6] = 1.0
    return (coords - c_min) / c_range


CANONICAL_COLS = [
    "method", "from_time", "to_time", "time_span",
    "x_hvg_corr", "x_wasserstein",
    "spatial_domain_ari", "spatial_coherence_pc1",
]

DATASET_CONFIG = {
    "mouse_embryo": {
        "time_points": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5],
        "suffixes": [
            "Mouse_embryo_E9.5_processed.h5ad",
            "Mouse_embryo_E10.5_processed.h5ad",
            "Mouse_embryo_E11.5_processed.h5ad",
            "Mouse_embryo_E12.5_processed.h5ad",
            "Mouse_embryo_E13.5_processed.h5ad",
            "Mouse_embryo_E14.5_processed.h5ad",
            "Mouse_embryo_E15.5_processed.h5ad",
            "Mouse_embryo_E16.5_processed.h5ad",
        ],
        "annotation_key": "annotation",
    },
    "zebrafish_embryo": {
        "time_points": [3.3, 5.25, 10, 12, 18, 24],
        "suffixes": [
            "Zebrafish_embryo_3.3hpf_processed.h5ad",
            "Zebrafish_embryo_5.25hpf_processed.h5ad",
            "Zebrafish_embryo_10hpf_processed.h5ad",
            "Zebrafish_embryo_12hpf_processed.h5ad",
            "Zebrafish_embryo_18hpf_processed.h5ad",
            "Zebrafish_embryo_24hpf_processed.h5ad",
        ],
        "annotation_key": "layer_annotation",
    },
    "axolotl_telencephalon_regeneration": {
        "time_points": [2, 5, 10, 15, 20, 30, 60, 100],
        "suffixes": [
            "Axolotl_telencephalon_2DPI_processed.h5ad",
            "Axolotl_telencephalon_5DPI_processed.h5ad",
            "Axolotl_telencephalon_10DPI_processed.h5ad",
            "Axolotl_telencephalon_15DPI_processed.h5ad",
            "Axolotl_telencephalon_20DPI_processed.h5ad",
            "Axolotl_telencephalon_30DPI_processed.h5ad",
            "Axolotl_telencephalon_60DPI_processed.h5ad",
            "Axolotl_telencephalon_control_processed.h5ad",
        ],
        "annotation_key": "Annotation",
    },
}
