"""
Data loading and preprocessing utilities.

Supports Stereo-seq spatial transcriptomics data in AnnData (h5ad) format.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.neighbors import NearestNeighbors
import warnings
import scanpy as sc
class SpatialTranscriptomicsDataset(Dataset):
    """
    Spatial transcriptomics dataset.

    Each sample contains gene expression, spatial coordinates, a time point,
    the slice id, and optionally annotation metadata.
    """

    def __init__(
        self,
        gene_expression: np.ndarray,  # (n_spots, n_genes)
        spatial_coords: np.ndarray,   # (n_spots, 2)
        time_point: float,
        slice_id: int,
        annotations: Optional[np.ndarray] = None  # Optional cell-type annotations.
    ):
        assert gene_expression.shape[0] == spatial_coords.shape[0], \
            "Gene expression and spatial coords must have same length"

        self.gene_expression = torch.from_numpy(gene_expression).float()
        self.spatial_coords = torch.from_numpy(spatial_coords).float()
        self.time_point = time_point
        self.slice_id = slice_id
        self.n_spots = len(self.gene_expression)
        self.annotations = annotations

    def __len__(self):
        return self.n_spots

    def __getitem__(self, idx):
        return {
            'x': self.gene_expression[idx],
            's': self.spatial_coords[idx],
            't': torch.tensor(self.time_point, dtype=torch.float32),
            'slice_id': self.slice_id,
            'idx': idx
        }


class SpatialTranscriptomicsLoader:
    """
    Loader manager for multiple time-point slices.
    """

    def __init__(self, config):
        self.config = config
        self.datasets: Dict[int, SpatialTranscriptomicsDataset] = {}
        self.time_points: List[float] = []
        self.knn_graphs: Dict[int, np.ndarray] = {}

        # Normalization statistics used for later inverse transforms.
        self.x_stats = {"mean": 0.0, "std": 1.0}
        self.s_stats = {"mean": 0.0, "std": 1.0}
        self.selected_genes = None

    def load_from_anndata(
        self,
        slice_paths: List[str],
        time_points: List[float],
        annotation_key: Optional[str] = None,  # default: None
        layer: Optional[str] = None,  # Expression layer to use, e.g. "counts".
        need_normalize: bool = False,  # Apply CPM normalization.
        need_log_transform: bool = False,  # Apply log1p transform.
        need_scale: bool = False,  # Apply z-score scaling to expression.
        spatial_layer: Optional[str] = None,  # obsm key for spatial coordinates.
        n_genes: Optional[int] = None,  # Number of HVGs to select; None uses all genes.
        selected_genes: Optional[List[str]] = None,  # Pre-selected genes, overriding n_genes.
        external_stats: Optional[Dict] = None  # Train-set stats for validation/test slices.
    ):
        """
        Load slices from AnnData (.h5ad) files.

        Args:
            slice_paths: List of h5ad file paths.
            time_points: Matching time points, for example [4, 12, 28].
            annotation_key: Optional obs column containing annotations.
        """
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("Please install scanpy: pip install scanpy")

        assert len(slice_paths) == len(time_points), \
            "Number of slices must match number of time points"

        # highly variable genes should be the same across slices. Should be selected across all slices before this step.
        if selected_genes is not None:
            print(f"Using {len(selected_genes)} pre-selected genes.")
            self.selected_genes = list(selected_genes)
        elif n_genes is not None:
            adata_all = sc.concat([sc.read_h5ad(p) for p in slice_paths], join='inner', label='batch')
            if adata_all.n_vars < n_genes:
                warnings.warn(f"Requested {n_genes} highly variable genes, but only {adata_all.n_vars} genes available. Using all genes.")
                n_genes = adata_all.n_vars
                selected_genes = adata_all.var_names.tolist()
            elif adata_all.n_vars == n_genes:
                selected_genes = adata_all.var_names.tolist()
                print(f"Number of genes equals requested highly variable genes ({n_genes}). Skipping HVG selection.")
            else:
                print(f"Selecting top {n_genes} highly variable genes across all slices...")
                text_idx = min(1000, adata_all.n_obs)
                if np.max(adata_all[:text_idx,:].X.data) > 50 and (not np.allclose(adata_all[:text_idx,:].X.sum(axis=1).A1, adata_all[:text_idx,:].X.sum(axis=1).A1[0], rtol=0.5)):
                    warnings.warn("Data does not seem to be normalized. Applying CPM normalization for HVG selection.")
                    sc.pp.normalize_total(adata_all, target_sum=1e6)
                if not np.all(adata_all[:text_idx,:].X.data >= 0) or np.max(adata_all[:text_idx,:].X.data) > 50:
                    warnings.warn("Data does not seem to be log-transformed. Applying log1p for HVG selection.")
                    sc.pp.log1p(adata_all)
                sc.pp.highly_variable_genes(adata_all, n_top_genes=n_genes, subset=True)
                selected_genes = adata_all.var_names.tolist()
                print(f"  Selected top {n_genes} highly variable genes")
            del adata_all  # Release memory before per-slice loading.
            self.selected_genes = selected_genes
        else:
            self.selected_genes = None

        self.time_points = sorted(time_points)

        # Stage 1: load all slices and apply expression preprocessing.
        all_adatas = []
        all_annotations = []
        for slice_id, (path, t) in enumerate(zip(slice_paths, time_points)):
            print(f"Loading slice {slice_id} (t={t}) from {path}...")
            adata = sc.read_h5ad(path)

            # Annotation handling.
            annotations = None
            if annotation_key and annotation_key in adata.obs.columns:
                annotations = adata.obs[annotation_key].values

            # Gene selection
            genes_to_use = self.selected_genes if hasattr(self, 'selected_genes') and self.selected_genes is not None else (selected_genes if 'selected_genes' in locals() else None)
            if genes_to_use is not None:
                common_genes = [g for g in genes_to_use if g in adata.var_names]
                adata = adata[:, common_genes].copy()

            # Level-1 expression preprocessing.
            if need_normalize:
                sc.pp.normalize_total(adata, target_sum=1e6)
            if need_log_transform:
                sc.pp.log1p(adata)

            all_adatas.append(adata)
            all_annotations.append(annotations)

        # Stage 2: collect raw spatial coordinates.
        S_raw_list = []
        for ad in all_adatas:
            for key in (spatial_layer, 'spatial', 'X_spatial', 'X_Spatial', 'Spatial'):
                if key and key in ad.obsm:
                    S_raw_list.append(ad.obsm[key][:, :2].astype(float))
                    break
            else:
                raise ValueError("No spatial coordinates found in obsm")

        # Stage 3: compute global normalization statistics.
        X_all  = np.vstack([ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X
                            for ad in all_adatas])
        S_all  = np.vstack(S_raw_list)

        if external_stats:
            # Validation/test loaders reuse training statistics to keep scales aligned.
            self.x_stats.update(external_stats.get("x_stats", {}))
            self.s_stats.update(external_stats.get("s_stats", {}))
            print("  [Stats] Using external statistics (train set mean/std)")
        else:
            # Spatial coordinates always use global z-score normalization.
            self.s_stats["mean"] = np.mean(S_all, axis=0)
            self.s_stats["std"]  = np.std(S_all,  axis=0) + 1e-8

            # Expression is z-scored only when need_scale=True.
            if need_scale:
                self.x_stats["mean"] = np.mean(X_all, axis=0)
                self.x_stats["std"]  = np.std(X_all,  axis=0) + 1e-8
            else:
                self.x_stats["mean"] = np.zeros(X_all.shape[1])
                self.x_stats["std"]  = np.ones(X_all.shape[1])

            print(f"  [Stats] X  need_scale={need_scale}  "
                  f"mean={np.mean(self.x_stats['mean']):.4f}, "
                  f"std={np.mean(self.x_stats['std']):.4f}")
            print(f"  [Stats] S  global zscore  "
                  f"mean={self.s_stats['mean'].round(2)}, "
                  f"std={self.s_stats['std'].round(2)}")

        # Stage 4: apply scaling and create Dataset objects.
        for i, (adata, S_raw, annotations, t) in enumerate(
                zip(all_adatas, S_raw_list, all_annotations, time_points)):
            X_l1 = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

            gene_expr      = (X_l1  - self.x_stats["mean"]) / self.x_stats["std"]
            spatial_coords = (S_raw - self.s_stats["mean"]) / self.s_stats["std"]

            dataset = SpatialTranscriptomicsDataset(
                gene_expression=gene_expr,
                spatial_coords=spatial_coords,
                time_point=t,
                slice_id=i,
                annotations=annotations
            )
            self.datasets[i] = dataset
            self.knn_graphs[i] = self._build_knn_graph(spatial_coords, k=self.config.lag_knn)
            print(f"  Slice {i} ready: {dataset.n_spots} spots")

    def load_from_numpy(
        self,
        data_dict: Dict[int, Tuple[np.ndarray, np.ndarray, float]]
    ):
        """
        Load data from numpy arrays using the same Level 1 -> Level 2 logic
        as load_from_anndata.

        Args:
            data_dict: {slice_id: (gene_expr, spatial_coords, time_point)}
        """
        self.time_points = sorted([v[2] for v in data_dict.values()])

        X_all = np.vstack([v[0] for v in data_dict.values()])
        S_all = np.vstack([v[1] for v in data_dict.values()])

        # Spatial coordinates use global z-score normalization.
        self.s_stats["mean"] = np.mean(S_all, axis=0)
        self.s_stats["std"]  = np.std(S_all,  axis=0) + 1e-8
        # Expression uses global z-score normalization.
        self.x_stats["mean"] = np.mean(X_all, axis=0)
        self.x_stats["std"]  = np.std(X_all,  axis=0) + 1e-8

        for i, (slice_id, (X_raw, S_raw, t)) in enumerate(data_dict.items()):
            gene_expr      = (X_raw - self.x_stats["mean"]) / self.x_stats["std"]
            spatial_coords = (S_raw - self.s_stats["mean"]) / self.s_stats["std"]

            dataset = SpatialTranscriptomicsDataset(
                gene_expression=gene_expr,
                spatial_coords=spatial_coords,
                time_point=t,
                slice_id=slice_id
            )

            self.datasets[slice_id] = dataset
            self.knn_graphs[slice_id] = self._build_knn_graph(
                spatial_coords,
                k=self.config.lag_knn
            )

    def _normalize_spatial_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalize spatial coordinates to [0, 1] as a fallback."""
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        coords_range = coords_max - coords_min
        coords_range[coords_range < 1e-6] = 1.0
        return (coords - coords_min) / coords_range

    def unnormalize(
        self,
        x_scaled: Optional[torch.Tensor] = None,
        s_scaled: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Restore standardized predictions to original expression/spatial scales."""
        x_orig = s_orig = None

        if x_scaled is not None:
            if isinstance(x_scaled, torch.Tensor):
                x_scaled = x_scaled.detach().cpu().numpy()
            x_orig = x_scaled * self.x_stats["std"] + self.x_stats["mean"]

        if s_scaled is not None:
            if isinstance(s_scaled, torch.Tensor):
                s_scaled = s_scaled.detach().cpu().numpy()
            s_orig = s_scaled * self.s_stats["std"] + self.s_stats["mean"]

        return x_orig, s_orig

    def _build_knn_graph(self, coords: np.ndarray, k: int) -> np.ndarray:
        """
        Build a k-NN graph.

        Args:
            coords: (n_spots, 2)
            k: Number of neighbors.

        Returns:
            knn_indices: (n_spots, k) nearest-neighbor indices.
        """
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords)
        _, indices = nbrs.kneighbors(coords)

        # Drop the self-neighbor, which is returned first.
        knn_indices = indices[:, 1:]

        return knn_indices

    def get_dataloader(
        self,
        slice_id: int,
        batch_size: Optional[int] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """Return a DataLoader for one slice."""
        if batch_size is None:
            batch_size = self.config.batch_size

        return DataLoader(
            self.datasets[slice_id],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def get_full_data(self, slice_id: int) -> dict:
        """Return all data for one slice, used by training and evaluation."""
        dataset = self.datasets[slice_id]

        data_dict = {
            'x': dataset.gene_expression,
            's': dataset.spatial_coords,
            't': torch.tensor(dataset.time_point, dtype=torch.float32),
            'knn_indices': torch.from_numpy(self.knn_graphs[slice_id]).long()
        }

        # Annotation handling.
        if dataset.annotations is not None:
            data_dict['annotations'] = dataset.annotations

        return data_dict

    def get_time_pairs(self) -> List[Tuple[int, int, float, float]]:
        """
        Return all adjacent time-point pairs.

        Returns:
            pairs: [(slice_id_1, slice_id_2, t1, t2), ...]
        """
        sorted_slices = sorted(self.datasets.keys(),
                              key=lambda k: self.datasets[k].time_point)

        pairs = []
        for i in range(len(sorted_slices) - 1):
            id1, id2 = sorted_slices[i], sorted_slices[i+1]
            t1 = self.datasets[id1].time_point
            t2 = self.datasets[id2].time_point
            pairs.append((id1, id2, t1, t2))

        return pairs


def preprocess_gene_expression(
    adata: sc.AnnData,
    layer: Optional[str] = None,
    normalize: bool = True,
    log_transform: bool = True,
    scale: bool = True
) -> sc.AnnData:
    """
    Preprocess gene expression stored in an AnnData object.

    Args:
        X: (n_spots, n_genes) raw counts.
        log_transform: Whether to apply log(1+x).
        scale: Whether to apply z-score scaling.

    Returns:
        The processed AnnData object.
    """
    # create adata for processing
    import scipy.sparse as sp
    import numpy as np
    import scanpy as sc

    if layer is not None:
        if layer in adata.layers:
            adata.layers['original_X'] = adata.X.copy()
            adata.X = adata.layers[layer].copy()
            print(f"  Using layer '{layer}' for gene expression")
        else:
            raise ValueError(f"Layer '{layer}' not found in AnnData layers.")

    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e6)  # CPM normalization
    if log_transform:
        sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata)

    return adata


if __name__ == "__main__":
    print("Testing Data Loader...")

    from config import Config

    config = Config()
    loader = SpatialTranscriptomicsLoader(config)

    # Generate synthetic data.
    print("\nGenerating synthetic data...")
    np.random.seed(42)

    data_dict = {}
    for slice_id, t in enumerate([4, 12, 28]):
        n_spots = 1000
        n_genes = 2000

        # Simulate gene expression.
        gene_expr = np.random.randn(n_spots, n_genes) + 5
        gene_expr = np.exp(gene_expr)  # Simulated count data.

        # Simulate circular spatial coordinates.
        theta = np.random.rand(n_spots) * 2 * np.pi
        r = np.random.rand(n_spots) * 10
        spatial_coords = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta)
        ])

        data_dict[slice_id] = (gene_expr, spatial_coords, float(t))

    # Load synthetic data.
    loader.load_from_numpy(data_dict)
    print(f"Loaded {len(loader.datasets)} slices")
    print(f"Time points: {loader.time_points}")

    # Test DataLoader.
    print("\nTesting DataLoader...")
    dataloader = loader.get_dataloader(slice_id=0, batch_size=64)
    batch = next(iter(dataloader))

    print(f"Batch keys: {batch.keys()}")
    print(f"Gene expression shape: {batch['x'].shape}")
    print(f"Spatial coords shape: {batch['s'].shape}")
    print(f"Time: {batch['t'][0]}")

    # Test full-data access.
    print("\nTesting full-data access...")
    full_data = loader.get_full_data(slice_id=0)
    print(f"Full data x shape: {full_data['x'].shape}")
    print(f"Full data s shape: {full_data['s'].shape}")
    print(f"k-NN indices shape: {full_data['knn_indices'].shape}")

    # Test adjacent time-pair generation.
    print("\nTesting time pairs...")
    pairs = loader.get_time_pairs()
    for id1, id2, t1, t2 in pairs:
        print(f"Pair: slice {id1} (t={t1}) -> slice {id2} (t={t2})")

    print("\n[OK] All tests passed!")
