"""
full_analysis.py
================
Full-dataset training plus biological analysis.

Workflow:
  1. Train FeedbackControlModel on the full dataset.
  2. Run standard benchmark evaluation on all adjacent time pairs.
  3. Run biological analyses: driver gene identification, GRN Jacobian estimation, B-matrix heatmaps, gene velocity heatmaps, and feedback convergence curves.
  4. Visualize latent space with PCA and optionally UMAP.

Examples:
  python full_analysis.py --data mouse_embryo --num_epochs 200
  python full_analysis.py --data axolotl_telencephalon_regeneration --checkpoint /path/to/final_model.pt
  python full_analysis.py --data mouse_embryo --config_json dataset_params_full.json
"""
import os, sys, time, json, argparse
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA

sc = None
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False
    print("[Warning] scanpy not installed - UMAP plots will be skipped.")

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from train import Trainer
from benchmark_utils import (
    DATASET_CONFIG, compute_all_metrics,
    plot_metrics_bar, print_metrics_summary,
    load_dataset_params,
)

BASE_DATA_DIR      = "./spatiotemporal"
DEFAULT_OUTPUT_DIR = "./spatiotemporal/outputs"


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Full training + biological analysis for FeedbackControlModel"
    )
    p.add_argument("--data",         type=str,  default="mouse_embryo",
                   choices=list(DATASET_CONFIG.keys()))
    p.add_argument("--output_dir",   type=str,  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--slice_paths",  type=str,  nargs="+", default=None)
    p.add_argument("--time_points",  type=float, nargs="+", default=None)
    p.add_argument("--annotation_key", type=str, default=None)
    p.add_argument("--layer",        type=str,  default=None)
    p.add_argument("--n_genes",      type=int,  default=3000)
    p.add_argument("--need_normalize",     action="store_true")
    p.add_argument("--need_log_transform", action="store_true")
    p.add_argument("--need_scale",         action="store_true")
    p.add_argument("--evaluate_top_k", type=int, default=100)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--num_epochs",   type=int,  default=200)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--ode_solver",   type=str,  default="rk4")
    p.add_argument("--latent_dim",   type=int,  default=128)
    p.add_argument("--checkpoint",   type=str,  default=None,
                   help="Existing checkpoint path; skip training and run analysis directly")
    p.add_argument("--config_json",  type=str,  default=None,
                   help="JSON file used to override default hyperparameters")
    p.add_argument("--use_wandb",    action="store_true")
    p.add_argument("--wandb_project", type=str, default="full-analysis")
    p.add_argument("--metrics_output_csv", type=str, default=None)
    p.add_argument("--decoder_chunk_size", type=int, default=2048)
    p.add_argument("--max_cells_bio", type=int, default=500,
                   help="Maximum sampled cells per slice for biological analysis")
    p.add_argument("--top_k_genes",  type=int,  default=50,
                   help="Number of top driver genes shown in biological analysis")
    p.add_argument("--use_label_aware_losses", action="store_true")
    p.add_argument("--label_distance_metric", type=str, default="feat", choices=["feat"])
    p.add_argument("--lambda_label_trajectory", type=float, default=0.0)
    p.add_argument("--lambda_label_proportion", type=float, default=0.0)
    p.add_argument("--lambda_label_centroid",   type=float, default=0.0)
    p.add_argument("--label_min_cells", type=int, default=8)
    return p.parse_args()


# ============================================================
# ============================================================

def _forward_in_chunks(fn, x, chunk_size):
    if x.shape[0] <= chunk_size:
        return fn(x)
    outs = []
    for start in range(0, x.shape[0], chunk_size):
        outs.append(fn(x[start:start + chunk_size]))
    return torch.cat(outs, dim=0)


# ============================================================
# ============================================================

def extract_latent_embeddings(trainer: Trainer, max_cells: int = 2000) -> dict:
    trainer.model.eval()
    device = trainer.device
    Z_list, X_list, S_list, tp_list, ann_list = [], [], [], [], []

    with torch.no_grad():
        for sid in sorted(trainer.data_loader.datasets.keys()):
            data  = trainer.data_loader.get_full_data(sid)
            t_val = trainer.data_loader.time_points[sid]
            x     = data["x"].to(device)
            s     = data["s"].to(device)
            knn   = data.get("knn_indices")
            if knn is not None:
                knn = knn.to(device)
            ann = data.get("annotations", None)

            if x.shape[0] > max_cells:
                idx = torch.randperm(x.shape[0])[:max_cells]
                x   = x[idx]; s = s[idx]
                if knn is not None: knn = knn[idx]
                if ann is not None: ann = np.asarray(ann)[idx.cpu().numpy()]

            Z = trainer.model.latent_embedding(x, s, knn)
            Z_list.append(Z.cpu().numpy())
            X_list.append(x.cpu().numpy())
            S_list.append(s.cpu().numpy())
            tp_list.append(np.full(Z.shape[0], t_val))
            ann_list.append(
                np.asarray(ann).astype(str) if ann is not None
                else np.array(["Unknown"] * Z.shape[0])
            )
            print(f"  t={t_val}: {Z.shape[0]} cells, Z_mean={Z.mean().item():.4f}")

    return {
        "Z":           np.vstack(Z_list),
        "X":           np.vstack(X_list),
        "S":           np.vstack(S_list),
        "timepoints":  np.concatenate(tp_list),
        "annotations": np.concatenate(ann_list),
    }


# ============================================================
# ============================================================

def viz_feedback_convergence(trainer: Trainer, save_dir: Path) -> dict:
    trainer.model.eval()
    device = trainer.device
    times, fb_means, fb_stds = [], [], []

    with torch.no_grad():
        for sid in sorted(trainer.data_loader.datasets.keys()):
            data = trainer.data_loader.get_full_data(sid)
            x    = data["x"].to(device)
            s    = data["s"].to(device)
            knn  = data.get("knn_indices")
            if knn is not None: knn = knn.to(device)

            Z   = trainer.model.latent_embedding(x, s, knn)
            M_i = trainer.model.target_predictor(Z)
            delta = (M_i - Z).norm(dim=-1).cpu().numpy()
            times.append(float(data["t"]))
            fb_means.append(float(delta.mean()))
            fb_stds.append(float(delta.std()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(times, fb_means, "D--", color="darkgreen", ms=8, lw=2)
    axes[0].fill_between(
        times,
        np.array(fb_means) - np.array(fb_stds),
        np.array(fb_means) + np.array(fb_stds),
        alpha=0.15, color="green",
    )
    for t, d in zip(times, fb_means):
        axes[0].annotate(f"t={t}", (t, d), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axes[0].set_xlabel("Developmental Time")
    axes[0].set_ylabel(r"Mean $\|M_i - Z_i\|$")
    axes[0].set_title(
        r"Feedback Signal $\|M_i - Z_i\|$ over Time"
        "\n(should decrease toward terminal if model converges)"
    )
    axes[0].grid(True, alpha=0.3)

    n_params = sum(p.numel() for p in trainer.model.parameters())
    info = (
        f"Architecture\n\n"
        f"$dZ/dt = g(Z) + B(M_i - Z)$\n"
        f"$X_{{pred}} = \\Phi_X(Z)$\n"
        f"$S_{{rel,pred}} = \\Phi_S(Z)$\n\n"
        f"$M_i = M_\\theta(Z_i)$ [prototype attractor]\n\n"
        f"Parameters: {n_params:,}\n"
        f"Latent dim (Z): {trainer.config.latent_dim}\n"
        f"n_genes (X): {trainer.config.n_genes}"
    )
    axes[1].text(0.5, 0.5, info, ha="center", va="center", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.6",
                           facecolor="lightyellow", edgecolor="orange"))
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1); axes[1].axis("off")
    axes[1].set_title("Model Architecture")

    plt.tight_layout()
    p = save_dir / "feedback_convergence.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved: feedback_convergence.png")

    return {"times": times, "fb_means": fb_means, "fb_stds": fb_stds}


# ============================================================
# ============================================================

def viz_latent_space(embeddings: dict, save_dir: Path) -> np.ndarray:
    Z   = embeddings["Z"]
    tp  = embeddings["timepoints"]
    ann = embeddings["annotations"]

    n_comp = min(50, Z.shape[1], Z.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=42)
    Z_pca  = pca.fit_transform(Z)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    eff_dim = int(np.searchsorted(cumvar, 0.95)) + 1

    # PCA-related step.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(min(30, n_comp)),
           pca.explained_variance_ratio_[:30], alpha=0.7, color="steelblue")
    ax2 = ax.twinx()
    ax2.plot(range(min(30, n_comp)), cumvar[:30], "r-o", ms=4, lw=2)
    ax2.axhline(0.95, color="green", linestyle="--", label="95%")
    ax.set_xlabel("Principal Component"); ax.set_ylabel("Individual Variance")
    ax2.set_ylabel("Cumulative Variance"); ax2.set_ylim(0, 1.1)
    ax.set_title(f"PCA Variance in Latent Space Z  (eff_dim@95% = {eff_dim})")
    ax2.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "pca_variance.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: pca_variance.png")

    # PCA-related step.
    unique_tps = sorted(np.unique(tp))
    cmap       = mpl_cm.get_cmap("viridis", len(unique_tps))
    unique_ann = np.unique(ann)
    fig, axes  = plt.subplots(1, 2, figsize=(14, 6))

    for i, t in enumerate(unique_tps):
        m = tp == t
        axes[0].scatter(Z_pca[m, 0], Z_pca[m, 1],
                        c=[cmap(i)], s=4, alpha=0.5, label=f"t={t}")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].set_title("Latent Space Z: PCA (by time)")
    axes[0].legend(fontsize=7, markerscale=3)

    if len(unique_ann) > 1:
        cmap2   = mpl_cm.get_cmap("tab20", len(unique_ann))
        ann_idx = {a: i for i, a in enumerate(unique_ann)}
        axes[1].scatter(Z_pca[:, 0], Z_pca[:, 1],
                        c=[cmap2(ann_idx[a]) for a in ann], s=4, alpha=0.5)
        for i, a in enumerate(unique_ann[:15]):
            axes[1].scatter([], [], c=[cmap2(i)], label=a, s=20)
        axes[1].legend(fontsize=7, markerscale=2, title="Cell Type", ncol=2)
    else:
        axes[1].scatter(Z_pca[:, 0], Z_pca[:, 1],
                        c=tp, cmap="viridis", s=4, alpha=0.5)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].set_title("Latent Space Z: PCA (by cell type)")
    plt.tight_layout()
    plt.savefig(save_dir / "pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: pca_scatter.png")

    # Optional UMAP step.
    if HAS_SCANPY:
        adata = sc.AnnData(X=Z)
        adata.obs["timepoint"]     = [str(t) for t in tp]
        adata.obs["timepoint_num"] = tp.astype(float)
        adata.obs["cell_type"]     = ann
        sc.settings.figdir = str(save_dir)
        sc.settings.set_figure_params(dpi=150, facecolor="white")
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")
        sc.tl.umap(adata)
        for col, fname in [("timepoint", "umap_time.png"),
                           ("cell_type",  "umap_celltype.png")]:
            try:
                sc.pl.umap(adata, color=col,
                           title=f"Latent Z UMAP - {col}",
                           save=f"_{fname}", show=False)
                print(f"  Saved: umap_{fname}")
            except Exception as e:
                print(f"  [Warning] UMAP {col} failed: {e}")

        if len(unique_ann) > 1:
            try:
                ct_prop = pd.crosstab(
                    adata.obs["timepoint"], adata.obs["cell_type"],
                    normalize="index"
                )
                plt.figure(figsize=(max(12, len(unique_ann) * 0.5), 5))
                sns.heatmap(ct_prop.T, cmap="YlOrRd",
                            cbar_kws={"label": "Proportion"})
                plt.title("Cell Type Composition across Timepoints")
                plt.tight_layout()
                plt.savefig(save_dir / "celltype_heatmap.png",
                            dpi=150, bbox_inches="tight")
                plt.close(); print("  Saved: celltype_heatmap.png")
            except Exception as e:
                print(f"  [Warning] celltype heatmap failed: {e}")

    return Z_pca


# ============================================================
#    Theory: spatiotemporal_identification_new.md Section 4
# ============================================================

def compute_biological_analysis(
    trainer: Trainer,
    max_cells_per_slice: int = 500,
    top_k: int = 50,
    eps_fd: float = 1e-3,
) -> dict:
    """
    Section 4 "Biological Meaning":

       - psix = gene_decoder(Z -> X)
       - v_Z = dZ/dt = latent_dynamics(Z, M_i)

    2. GRN Jacobian  dv_X/dX ~= J_psix * (-B) * (J_psix^T J_psix)^{-1} * J_psix^T
       - B = latent_dynamics.B.weight(dxd)

    """
    model  = trainer.model
    device = trainer.device
    model.eval()

    B_np = model.latent_dynamics.B.weight.detach().cpu().numpy()  # (d, d)
    d_dim = B_np.shape[0]

    all_vX, all_times = [], []

    with torch.no_grad():
        for sid in sorted(trainer.data_loader.datasets.keys()):
            data  = trainer.data_loader.get_full_data(sid)
            t_val = trainer.data_loader.time_points[sid]
            x     = data["x"].to(device)
            s     = data["s"].to(device)
            knn   = data.get("knn_indices")
            if knn is not None: knn = knn.to(device)

            if x.shape[0] > max_cells_per_slice:
                idx = torch.randperm(x.shape[0])[:max_cells_per_slice]
                x   = x[idx]; s = s[idx]
                if knn is not None: knn = knn[idx]

            Z   = model.latent_embedding(x, s, knn)       # (N, d)
            M_i = model.target_predictor(Z)                # (N, d)
            dZ  = model.latent_dynamics(Z, M_i)            # (N, d)  = v_Z

            # v_X = J_psix * v_Z  ~= decoder(Z + dZ) - decoder(Z)
            X_cur  = _forward_in_chunks(model.gene_decoder, Z,      chunk_size=1024)
            X_next = _forward_in_chunks(model.gene_decoder, Z + dZ, chunk_size=1024)
            vX     = (X_next - X_cur).mean(dim=0).cpu().numpy()    # (G,)

            all_vX.append(vX)
            all_times.append(t_val)

    sids = sorted(trainer.data_loader.datasets.keys())
    mid_sid = sids[len(sids) // 2]
    data_mid = trainer.data_loader.get_full_data(mid_sid)
    x_mid = data_mid["x"].to(device)
    s_mid = data_mid["s"].to(device)
    knn_mid = data_mid.get("knn_indices")
    if knn_mid is not None: knn_mid = knn_mid.to(device)

    n_sample = min(200, x_mid.shape[0])
    idx_s    = torch.randperm(x_mid.shape[0])[:n_sample]

    with torch.no_grad():
        Z_sample = model.latent_embedding(
            x_mid[idx_s], s_mid[idx_s],
            knn_mid[idx_s] if knn_mid is not None else None
        )
        Z_mean = Z_sample.mean(dim=0, keepdim=True)  # (1, d)
        X_base = model.gene_decoder(Z_mean).squeeze(0)  # (G,)
        G_dim  = X_base.shape[0]

        J_psi_x = torch.zeros(G_dim, d_dim, device=device)
        for j in range(d_dim):
            Y_pert    = Z_mean.clone()
            Y_pert[0, j] += eps_fd
            X_pert    = model.gene_decoder(Y_pert).squeeze(0)
            J_psi_x[:, j] = (X_pert - X_base) / eps_fd

    J_np  = J_psi_x.cpu().numpy()                      # (G, d)
    JtJ   = J_np.T @ J_np                               # (d, d)
    try:
        JtJ_inv = np.linalg.inv(JtJ + 1e-6 * np.eye(d_dim))
    except np.linalg.LinAlgError:
        JtJ_inv = np.linalg.pinv(JtJ)

    # GRN = J * (-B) * (J^T J)^{-1} * J^T  ->  (G, G)
    GRN       = J_np @ (-B_np) @ JtJ_inv @ J_np.T
    grn_diag  = np.diag(GRN)
    grn_rownorm = np.linalg.norm(GRN, axis=1)

    gene_names = getattr(trainer.data_loader, "selected_genes", None)
    if gene_names is None or len(gene_names) != G_dim:
        gene_names = [f"Gene_{i}" for i in range(G_dim)]
    else:
        gene_names = list(gene_names)

    top_idx        = np.argsort(grn_rownorm)[::-1][:top_k]
    top_gene_names = [gene_names[i] for i in top_idx]

    print(f"\n  [Bio] B matrix shape: {B_np.shape}")
    print(f"  [Bio] J_psix shape: {J_np.shape}   GRN shape: {GRN.shape}")
    print(f"  [Bio] Top-10 driver genes (by regulatory influence):")
    for rank, gi in enumerate(top_idx[:10]):
        print(f"    {rank+1:3d}. {gene_names[gi]:20s}  "
              f"self_reg={grn_diag[gi]:+.4f}  influence={grn_rownorm[gi]:.4f}")

    return {
        "B_matrix":      B_np,
        "J_psi_x":       J_np,
        "GRN":           GRN,
        "grn_diag":      grn_diag,
        "grn_rownorm":   grn_rownorm,
        "top_gene_idx":  top_idx,
        "top_gene_names": top_gene_names,
        "gene_names":    gene_names,
        "vX_per_time":   np.array(all_vX),   # (T, G)
        "times":         all_times,
    }


# ============================================================
# ============================================================

def viz_biological_analysis(bio: dict, save_dir: Path, top_k: int = 30):
    """
    """
    bio_dir = save_dir / "bio_analysis"
    bio_dir.mkdir(parents=True, exist_ok=True)

    B_np        = bio["B_matrix"]
    GRN         = bio["GRN"]
    grn_diag    = bio["grn_diag"]
    grn_rownorm = bio["grn_rownorm"]
    top_idx     = bio["top_gene_idx"][:top_k]
    top_names   = bio["top_gene_names"][:top_k]
    vX_per_time = bio["vX_per_time"]   # (T, G)
    times       = bio["times"]

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(abs(B_np.min()), abs(B_np.max()), 1e-8)
    sns.heatmap(B_np, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                ax=ax, square=True, cbar_kws={"label": "B weight"})
    ax.set_title(
        "Latent Modular Regulatory Network (B matrix)\n"
        r"$dZ/dt = g(Z) + B(M_i - Z)$"
        "\nB encodes latent module interactions (GRN in Z-space)",
        fontsize=11,
    )
    ax.set_xlabel("Latent dim (source)")
    ax.set_ylabel("Latent dim (target)")
    plt.tight_layout()
    plt.savefig(bio_dir / "B_matrix_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: bio_analysis/B_matrix_heatmap.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, top_k * 0.25)))

    diag_vals = grn_diag[top_idx]
    colors_d  = ["crimson" if v < 0 else "steelblue" for v in diag_vals]
    axes[0].barh(range(len(top_names)), diag_vals, color=colors_d,
                 edgecolor="k", linewidth=0.3)
    axes[0].set_yticks(range(len(top_names)))
    axes[0].set_yticklabels(top_names, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"Self-regulation $[\partial v_{X_i}/\partial X_i]$")
    axes[0].set_title(
        "Gene Self-Regulation Strength\n"
        r"(diag of $\partial v_X/\partial X$; negative = self-inhibiting)",
        fontsize=10,
    )
    axes[0].axvline(0, color="gray", linewidth=0.5)
    axes[0].grid(True, alpha=0.3, axis="x")

    rownorm_vals = grn_rownorm[top_idx]
    axes[1].barh(range(len(top_names)), rownorm_vals,
                 color="darkorange", edgecolor="k", linewidth=0.3)
    axes[1].set_yticks(range(len(top_names)))
    axes[1].set_yticklabels(top_names, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(
        r"Regulatory Influence $\|[\partial v_X/\partial X]_{i,:}\|$"
    )
    axes[1].set_title(
        "Gene Regulatory Influence\n"
        r"(row-norm of $\partial v_X/\partial X$; identifiable, invariant to $Q,m,\lambda$)",
        fontsize=10,
    )
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.suptitle(
        r"Driver Gene Identification: $\partial v_X/\partial X \approx "
        r"J_{\psi_x} \cdot (-B) \cdot (J_{\psi_x}^T J_{\psi_x})^{-1} \cdot J_{\psi_x}^T$",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(bio_dir / "driver_genes_ranking.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: bio_analysis/driver_genes_ranking.png")

    top_k_show = min(top_k, 25)
    sub_idx    = top_idx[:top_k_show]
    sub_names  = top_names[:top_k_show]
    GRN_sub    = GRN[np.ix_(sub_idx, sub_idx)]

    fig, ax = plt.subplots(figsize=(11, 10))
    vmax_grn = max(abs(GRN_sub.min()), abs(GRN_sub.max()), 1e-8)
    sns.heatmap(
        GRN_sub,
        xticklabels=sub_names, yticklabels=sub_names,
        cmap="RdBu_r", center=0, vmin=-vmax_grn, vmax=vmax_grn,
        ax=ax, square=True,
        cbar_kws={"label": r"$\partial v_{X_i}/\partial X_j$"},
    )
    ax.set_title(
        r"Gene Regulatory Network (top drivers): $\partial v_X/\partial X$"
        "\n"
        r"$\approx J_{\psi_x} \cdot (-B) \cdot (J_{\psi_x}^T J_{\psi_x})^{-1} \cdot J_{\psi_x}^T$",
        fontsize=11,
    )
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout()
    plt.savefig(bio_dir / "GRN_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: bio_analysis/GRN_heatmap.png")

    if len(times) > 1:
        vX_top = vX_per_time[:, top_idx[:top_k_show]]   # (T, top_k_show)
        fig, ax = plt.subplots(figsize=(max(8, len(times) * 1.2),
                                        max(8, top_k_show * 0.35)))
        sns.heatmap(
            vX_top.T,
            xticklabels=[f"t={t}" for t in times],
            yticklabels=sub_names,
            cmap="RdBu_r", center=0, ax=ax,
            cbar_kws={"label": r"Mean $v_X$ (gene velocity)"},
        )
        ax.set_title(
            "Identified Gene Velocity across Developmental Time\n"
            r"$v_X = J_{\psi_x}(Z) \cdot v_Z$ (identifiable, invariant to $Q, m, \lambda$)",
            fontsize=11,
        )
        ax.tick_params(axis="y", rotation=0, labelsize=7)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        plt.tight_layout()
        plt.savefig(bio_dir / "gene_velocity_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: bio_analysis/gene_velocity_heatmap.png")

    # CSV output handling.
    gene_names_all = bio["gene_names"]
    df_genes = pd.DataFrame({
        "gene":                gene_names_all,
        "self_regulation":     grn_diag,
        "regulatory_influence": grn_rownorm,
        "rank":                np.argsort(np.argsort(-grn_rownorm)) + 1,
    }).sort_values("rank")
    df_genes.to_csv(bio_dir / "driver_genes_ranking.csv", index=False)
    print(f"  Saved: bio_analysis/driver_genes_ranking.csv ({len(df_genes)} genes)")

    np.save(bio_dir / "B_matrix.npy",  B_np)
    np.save(bio_dir / "J_psi_x.npy",   bio["J_psi_x"])
    np.save(bio_dir / "GRN_matrix.npy", GRN)
    print("  Saved: bio_analysis/{B_matrix, J_psi_x, GRN_matrix}.npy")


# ============================================================
# ============================================================

def viz_summary_panel(
    embeddings: dict,
    conv_data: dict,
    Z_pca: np.ndarray,
    benchmark_rows: list,
    trainer: Trainer,
    bio: dict,
    save_dir: Path,
):
    fig = plt.figure(figsize=(22, 14))
    gs  = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.4, figure=fig)

    # 1. PCA colored by time
    ax1 = fig.add_subplot(gs[0, 0:2])
    tp  = embeddings["timepoints"]
    unique_tps = sorted(np.unique(tp))
    cmap = mpl_cm.get_cmap("viridis", len(unique_tps))
    for i, t in enumerate(unique_tps):
        m = tp == t
        ax1.scatter(Z_pca[m, 0], Z_pca[m, 1],
                    c=[cmap(i)], s=3, alpha=0.5, label=f"t={t}")
    ax1.set_title("Latent Space Z: PCA (by timepoint)", fontsize=11)
    ax1.legend(fontsize=6, markerscale=4, ncol=2)
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(conv_data["times"], conv_data["fb_means"], "D--",
             color="darkgreen", ms=7)
    ax2.fill_between(
        conv_data["times"],
        np.array(conv_data["fb_means"]) - np.array(conv_data["fb_stds"]),
        np.array(conv_data["fb_means"]) + np.array(conv_data["fb_stds"]),
        alpha=0.15, color="green",
    )
    ax2.set_title(r"Feedback $\|M_i-Z_i\|$ over Time", fontsize=11)
    ax2.set_xlabel("Time"); ax2.set_ylabel(r"Mean $\|M_i - Z_i\|$")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 3])
    n_params = sum(p.numel() for p in trainer.model.parameters())
    arch_info = (
        f"Architecture\n\n"
        f"$dZ/dt = g(Z) + B(M_i-Z)$\n"
        f"$X = \\Phi_X(Z)$,  $S_{{rel}} = \\Phi_S(Z)$\n\n"
        f"$M_i = M_\\theta(Z_i)$ [prototypes]\n\n"
        f"Params: {n_params:,}\n"
        f"Z-dim: {trainer.config.latent_dim}\n"
        f"n_genes: {trainer.config.n_genes}\n"
        f"Solver: {trainer.config.ode_solver}"
    )
    ax3.text(0.5, 0.5, arch_info, ha="center", va="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="lightyellow", edgecolor="orange"))
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")
    ax3.set_title("Model Info", fontsize=11)

    # HVG-based metric calculation.
    ax4 = fig.add_subplot(gs[1, 0:2])
    if benchmark_rows:
        spans  = [r["time_span"]   for r in benchmark_rows]
        corrs  = [r["x_hvg_corr"]  for r in benchmark_rows]
        ax4.bar(spans, corrs, color="steelblue", edgecolor="k", lw=0.5)
        mean_c = float(np.nanmean(corrs))
        ax4.axhline(mean_c, color="red", linestyle="--",
                    label=f"Mean={mean_c:.3f}")
        ax4.set_ylim(0, 1.1)
        ax4.legend()
    ax4.set_title("X HVG Pearson Correlation per Time Pair", fontsize=11)
    ax4.set_ylabel("Pearson Corr")
    ax4.tick_params(axis="x", rotation=45, labelsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    ax5 = fig.add_subplot(gs[1, 2:])
    top_n_show = min(15, len(bio["top_gene_names"]))
    top_names  = bio["top_gene_names"][:top_n_show]
    top_infl   = bio["grn_rownorm"][bio["top_gene_idx"][:top_n_show]]
    ax5.barh(range(top_n_show), top_infl,
             color="darkorange", edgecolor="k", lw=0.3)
    ax5.set_yticks(range(top_n_show))
    ax5.set_yticklabels(top_names, fontsize=8)
    ax5.invert_yaxis()
    ax5.set_xlabel(r"Regulatory Influence $\|[\partial v_X/\partial X]_{i,:}\|$")
    ax5.set_title("Top Driver Genes\n"
                  r"(row-norm of $\partial v_X/\partial X$, invariant to $Q,m,\lambda$)",
                  fontsize=10)
    ax5.grid(True, alpha=0.3, axis="x")

    ax6 = fig.add_subplot(gs[2, :])
    fm   = conv_data["fb_means"]
    mean_hvg = float(np.nanmean([r["x_hvg_corr"]    for r in benchmark_rows])) if benchmark_rows else float("nan")
    mean_w   = float(np.nanmean([r["x_wasserstein"]  for r in benchmark_rows])) if benchmark_rows else float("nan")
    summary_txt = (
        f"FeedbackControlModel - Full Analysis Summary\n\n"
        f"Dataset: {trainer.config.output_dir.split('/')[-2] if hasattr(trainer.config, 'output_dir') else ''}  |  "
        f"Params: {n_params:,}  |  Z-dim: {trainer.config.latent_dim}  |  "
        f"Epochs: {trainer.config.num_epochs}  |  Solver: {trainer.config.ode_solver}\n\n"
        f"Feedback ||M_i-Z_i||: t={conv_data['times'][0]} -> {fm[0]:.4f}  |  "
        f"t={conv_data['times'][-1]} -> {fm[-1]:.4f}  "
        f"(Delta {100*(fm[-1]-fm[0])/(fm[0]+1e-8):+.1f}%)\n\n"
        f"Benchmark  x_hvg_corr (mean): {mean_hvg:.4f}  |  "
        f"x_wasserstein (mean): {mean_w:.4f}\n\n"
        f"Top driver genes: {', '.join(bio['top_gene_names'][:10])}"
    )
    ax6.text(0.5, 0.5, summary_txt, ha="center", va="center", fontsize=10,
             transform=ax6.transAxes,
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="lightcyan", edgecolor="gray"))
    ax6.axis("off")

    fig.suptitle("FeedbackControlModel - Full Training & Biological Analysis",
                 fontsize=16, y=1.01)
    plt.savefig(save_dir / "summary_panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: summary_panel.png")


# ============================================================
# ============================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds_cfg      = DATASET_CONFIG[args.data]
    data_dir    = f"{BASE_DATA_DIR}/{args.data}"
    slice_paths = args.slice_paths or [f"{data_dir}/{s}" for s in ds_cfg["suffixes"]]
    time_points = args.time_points or ds_cfg["time_points"]
    ann_key     = args.annotation_key or ds_cfg["annotation_key"]

    order       = sorted(range(len(slice_paths)), key=lambda i: time_points[i])
    slice_paths = [slice_paths[i] for i in order]
    time_points = [time_points[i] for i in order]

    run_name   = f"full_analysis_{args.data}_{time.strftime('%m%d-%H%M')}"
    output_dir = Path(args.output_dir) / run_name
    viz_dir    = output_dir / "visualizations"
    bio_base   = output_dir
    eval_dir   = output_dir / "evaluation"
    for d in [viz_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FeedbackControlModel - Full Training & Biological Analysis")
    print("  dZ/dt = g(Z) + B(M_i - Z)")
    print(f"  Data   : {args.data}  ({len(slice_paths)} slices)")
    print(f"  Output : {output_dir}")
    print("=" * 70)

    # -- W&B ---------------------------------------------------------
    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # -- Config & Trainer --------------------------------------------
    config = Config(
        learning_rate=args.lr,
        ode_solver=args.ode_solver,
        latent_dim=args.latent_dim,
        num_epochs=args.num_epochs,
        output_dir=str(output_dir / "results"),
        checkpoint_dir=str(output_dir / "checkpoints"),
        use_label_aware_losses=args.use_label_aware_losses,
        label_distance_metric=args.label_distance_metric,
        lambda_label_trajectory=args.lambda_label_trajectory,
        lambda_label_proportion=args.lambda_label_proportion,
        lambda_label_centroid=args.lambda_label_centroid,
        label_min_cells=args.label_min_cells,
    )

    if args.config_json is not None:
        print(f"\n[INFO] Loading hyperparameters from {args.config_json}")
        load_dataset_params(args.data, config, json_path=args.config_json)

    trainer = Trainer(config)
    trainer.load_data(
        slice_paths=slice_paths,
        time_points=time_points,
        annotation_key=ann_key,
        layer=args.layer,
        n_genes=args.n_genes,
        need_normalize=args.need_normalize,
        need_log_transform=args.need_log_transform,
        need_scale=args.need_scale,
    )

    # Checkpoint handling.
    if args.checkpoint:
        print(f"\n[INFO] Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    else:
        print(f"\n[1/4] Training ({args.num_epochs} epochs)...")
        trainer.train()
        trainer.save_checkpoint("final_model.pt")

    trainer.model.eval()

    # ================================================================
    # ================================================================
    print("\n" + "=" * 70)
    print("  [2/4] Benchmark evaluation on all adjacent time pairs")
    print("=" * 70)

    benchmark_rows = []

    for id_t0, id_t1, t0, t1 in trainer.data_loader.get_time_pairs():
        print(f"\n  --- t={t0} -> t={t1} ---")
        data_t0 = trainer.data_loader.get_full_data(id_t0)
        data_t1 = trainer.data_loader.get_full_data(id_t1)

        x_t0 = data_t0["x"].to(trainer.device)
        s_t0 = data_t0["s"].to(trainer.device)
        knn  = data_t0.get("knn_indices")
        if knn is not None: knn = knn.to(trainer.device)

        with torch.no_grad():
            Z_t0 = trainer.model.latent_embedding(x_t0, s_t0, knn)
            Z_t1_pred = trainer.model._ode_integrate_Z(
                Z_t0,
                trainer.model._to_tensor(trainer.normalize_time(t0), trainer.device),
                trainer.model._to_tensor(trainer.normalize_time(t1), trainer.device),
            )
            X_pred_sc = _forward_in_chunks(
                trainer.model.gene_decoder, Z_t1_pred, args.decoder_chunk_size
            )

        X_pred_np, _ = trainer.data_loader.unnormalize(x_scaled=X_pred_sc)
        X_true_np, _ = trainer.data_loader.unnormalize(x_scaled=data_t1["x"])
        _, S_true_np  = trainer.data_loader.unnormalize(s_scaled=data_t1["s"])
        labels_t1     = data_t1.get("annotations")
        labels_t0     = data_t0.get("annotations")

        metrics = compute_all_metrics(
            X_pred=X_pred_np, X_true_t1=X_true_np,
            S_true_t1=S_true_np, labels_t1=labels_t1, labels_t0=labels_t0,
            hvg_k=args.evaluate_top_k,
        )
        print_metrics_summary(metrics, "FeedbackControlModel", t0, t1)
        plot_metrics_bar(metrics, "FeedbackControlModel", t0, t1,
                         str(eval_dir / f"metrics_{t0}_to_{t1}.png"))

        row = {
            "method":    "FeedbackControlModel",
            "from_time": t0, "to_time": t1,
            "time_span": f"{t0}->{t1}",
            **{k: metrics.get(k, float("nan")) for k in [
                "x_hvg_corr", "x_wasserstein",
                "spatial_domain_ari", "spatial_coherence_pc1",
            ]},
        }
        benchmark_rows.append(row)

    df_bm = pd.DataFrame(benchmark_rows)
    df_bm.to_csv(eval_dir / "benchmark_metrics.csv", index=False)
    print(f"\n  Saved: evaluation/benchmark_metrics.csv")

    # CSV output handling.
    results_csv = (args.metrics_output_csv
                   or f"metrics/comparison_results_{args.data}.csv")
    os.makedirs("metrics", exist_ok=True)
    df_bm.to_csv(results_csv,
                 mode="a" if os.path.exists(results_csv) else "w",
                 header=not os.path.exists(results_csv), index=False)
    print(f"  [CSV] Appended to {results_csv}")

    print(f"\n{'='*70}")
    print(f"  SUMMARY (mean over {len(benchmark_rows)} time pairs)")
    print(f"{'='*70}")
    for col in ["x_hvg_corr", "x_wasserstein", "spatial_coherence_pc1", "spatial_domain_ari"]:
        vals = [r[col] for r in benchmark_rows
                if col in r and not np.isnan(r[col])]
        if vals:
            print(f"  {col:35s}: {np.mean(vals):.4f}")

    # ================================================================
    # ================================================================
    print("\n" + "=" * 70)
    print("  [3/4] Visualization")
    print("=" * 70)

    print("\n  Extracting latent embeddings Z for PCA/UMAP...")
    embeddings = extract_latent_embeddings(trainer, max_cells=2000)

    print("\n  Feedback signal convergence...")
    conv_data = viz_feedback_convergence(trainer, viz_dir)

    print("\n  Latent space PCA + UMAP...")
    Z_pca = viz_latent_space(embeddings, viz_dir)

    # ================================================================
    # ================================================================
    print("\n" + "=" * 70)
    print("  [3.5/4] Biological Analysis - Driver Gene Identification")
    print("  Theory: spatiotemporal_identification_new.md Section 4")
    print("  v_X = J_psix * v_Z  (identifiable, invariant to Q, m, lambda)")
    print("  dv_X/dX ~= J_psix * (-B) * (J_psix^T J_psix)^{-1} * J_psix^T")
    print("=" * 70)

    bio = compute_biological_analysis(
        trainer,
        max_cells_per_slice=args.max_cells_bio,
        top_k=args.top_k_genes,
    )
    print("\n  Visualizing biological analysis...")
    viz_biological_analysis(bio, bio_base, top_k=args.top_k_genes)

    # ================================================================
    # ================================================================
    print("\n  Summary panel...")
    viz_summary_panel(
        embeddings=embeddings,
        conv_data=conv_data,
        Z_pca=Z_pca,
        benchmark_rows=benchmark_rows,
        trainer=trainer,
        bio=bio,
        save_dir=viz_dir,
    )

    # ================================================================
    # ================================================================
    print("\n" + "=" * 70)
    print("  [4/4] Saving JSON report...")
    print("=" * 70)

    def _safe_mean(lst):
        v = [x for x in lst if not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    bm_summary = {
        col: _safe_mean([r.get(col, float("nan")) for r in benchmark_rows])
        for col in ["x_hvg_corr", "x_wasserstein", "spatial_coherence_pc1", "spatial_domain_ari"]
    }

    bio_summary = {
        "top_10_driver_genes":     bio["top_gene_names"][:10],
        "B_matrix_frobenius_norm": float(np.linalg.norm(bio["B_matrix"])),
        "GRN_max_influence":       float(np.max(bio["grn_rownorm"])),
        "GRN_mean_influence":      float(np.mean(bio["grn_rownorm"])),
        "GRN_max_self_regulation": float(np.max(np.abs(bio["grn_diag"]))),
    }

    fb_means = conv_data["fb_means"]
    report = {
        "metadata": {
            "run_name":   run_name,
            "dataset":    args.data,
            "n_slices":   len(slice_paths),
            "n_genes":    config.n_genes,
            "latent_dim": config.latent_dim,
            "ode_solver": config.ode_solver,
            "num_epochs": config.num_epochs,
        },
        "benchmark_metrics": bm_summary,
        "feedback_convergence": {
            "times":      conv_data["times"],
            "fb_means":   fb_means,
            "fb_change_pct": float(100 * (fb_means[-1] - fb_means[0]) / (fb_means[0] + 1e-8)),
        },
        "biological_analysis": bio_summary,
    }

    with open(output_dir / "full_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if args.use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({f"bm/{k}": v for k, v in bm_summary.items()})
                wandb.log({f"bio/{k}": v for k, v in bio_summary.items()
                           if isinstance(v, float)})
                for fname in ["feedback_convergence.png", "pca_scatter.png",
                              "summary_panel.png"]:
                    p = viz_dir / fname
                    if p.exists():
                        wandb.log({f"viz/{fname.replace('.png', '')}": wandb.Image(str(p))})
                bio_dir = bio_base / "bio_analysis"
                for fname in ["B_matrix_heatmap.png", "driver_genes_ranking.png",
                              "GRN_heatmap.png", "gene_velocity_heatmap.png"]:
                    p = bio_dir / fname
                    if p.exists():
                        wandb.log({f"bio/{Path(fname).stem}": wandb.Image(str(p))})
                wandb.finish()
        except Exception as e:
            print(f"  [WandB] Warning: {e}")

    print(f"\n{'='*70}")
    print("  Full analysis complete!")
    print(f"  Output dir   : {output_dir}")
    print(f"  Evaluation   : {eval_dir}")
    print(f"  Bio analysis : {bio_base / 'bio_analysis'}")
    print(f"  Visualizations: {viz_dir}")
    print(f"\n  Key results:")
    print(f"    x_hvg_corr (mean)            = {bm_summary['x_hvg_corr']:.4f}")
    print(f"    x_wasserstein (mean)         = {bm_summary['x_wasserstein']:.4f}")
    print(f"    spatial_domain_ari (mean)    = {bm_summary['spatial_domain_ari']:.4f}")
    print(f"    spatial_coherence_pc1 (mean) = {bm_summary['spatial_coherence_pc1']:.4f}")
    print(f"    Top driver genes: {', '.join(bio_summary['top_10_driver_genes'])}")
    print(f"    B ||F||                      = {bio_summary['B_matrix_frobenius_norm']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
