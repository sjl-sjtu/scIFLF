# scIFLF: Latent Feedback-Control Neural ODE for Spatiotemporal Transcriptomics

Implementation of **scIFLF**, a target-guided feedback-control dynamical model designed for reconstructing and predicting developmental trajectories in spatiotemporal transcriptomics (ST).

scIFLF maps high-dimensional gene expression and canonicalized spatial coordinates into a low-dimensional latent space ($\mathcal{Z}$), tracks continuous developmental progression via a feedback-driven Neural Ordinary Differential Equation (Neural ODE)
$$\frac{dZ_i}{dt} = g(Z_i) + B(M_i - Z_i)$$
and decodes future gene expression ($\hat{X}$) and spatial distributions ($\hat{S}_{\text{rel}}$).

---

## Repository Structure

```text
codes/
├── config.py                 # Configuration dataclass (data, model dimensions, loss weights, ODE solver)
├── data_loader.py            # AnnData/Stereo-seq dataset loaders and preprocessing pipelines
├── pattern_embedding.py      # Spatial-transcriptomic encoder with PCA-canonicalized coordinates
├── dynamics.py               # Latent-space Neural ODE dynamics: dz/dt = g(z) + B(m - z)
├── model.py                  # Core FeedbackControlModel integrating encoder, ODE, and decoders
├── train.py                  # PyTorch Trainer implementation with multi-task loss functions
├── full_analysis.py          # End-to-end training, trajectory metrics, and biological downstream analyses
├── grn_analysis.ipynb        # Jupyter notebook for Gene Regulatory Network (GRN) Jacobian analysis
├── benchmark_utils.py        # Dataset presets and metric computations (HVG corr, Wasserstein, ARI, Moran's I)
├── benchmark_loto_ours.py    # Leave-One-Timepoint-Out (LOTO) benchmark for scIFLF
├── benchmark_loto_stories.py # Baseline benchmark: STORIES (SpaceTime based on JAX)
├── benchmark_loto_moscot.py  # Baseline benchmark: moscot (Optimal Transport)
├── benchmark_loto_spateo.py  # Baseline benchmark: Spateo / Dynamo (Vector field)
└── hparam_loto_sweep.py      # Hyperparameter sweep script across holdout timepoints
```

---

## Installation & Dependencies

```bash
# Core requirements (for scIFLF)
pip install torch torchvision torchaudio
pip install geomloss scanpy anndata scikit-learn scipy pandas numpy pyyaml tqdm matplotlib

# Optional packages
pip install torchdiffeq                                     # if you want to use the dopri5 ODE solver (rk4 is default and built-in)
pip install wandb                                          # if you want experiment tracking with Weights & Biases (--use_wandb)
pip install networkx adjustText gprofiler-official seaborn # for downstream GRN and bio analysis (grn_analysis.ipynb)

# Baseline packages (only needed if running comparative benchmarks)
pip install stories-jax jax optax                           # for STORIES (benchmark_loto_stories.py)
pip install moscot                                         # for Moscot (benchmark_loto_moscot.py)
pip install dynamo-release spateo-release                   # for Spateo / Dynamo (benchmark_loto_spateo.py)
```

---

## Datasets

The repository provides built-in dataset presets configured in `benchmark_utils.py` for three landmark Stereo-seq datasets:

1. **`mouse_embryo`** (MOSTA): 8 time points from E9.5 to E16.5 (`annotation_key="annotation"`).
2. **`zebrafish_embryo`** (ZESTA): 6 time points from 3.3 to 24 hpf (`annotation_key="layer_annotation"`).
3. **`axolotl_telencephalon_regeneration`** (ARTISTA): 8 snapshots from 2 DPI to 100 DPI (`annotation_key="annotation"`).

Data files should be formatted as `.h5ad` AnnData objects. You can use the built-in presets via `--data <name>` or specify custom paths using `--slice_paths`, `--time_points`, and `--annotation_key`.

---

## Usage

### 1. Full Dataset Training & Biological Analysis (`full_analysis.py`)

Train `FeedbackControlModel` on the full developmental trajectory, calculate time-step transitions, and conduct biological discovery (driver genes, Jacobian GRN, feedback matrix heatmaps, velocity trajectories, and latent PCA):

```bash
# Train and run analysis on mouse embryo
python full_analysis.py --data mouse_embryo --num_epochs 200

# Run analysis using an existing checkpoint
python full_analysis.py --data axolotl_telencephalon_regeneration --checkpoint /path/to/checkpoint.pt
```

### 2. Tier-3 Leave-One-Timepoint-Out (LOTO) Benchmark (`benchmark_loto_ours.py`)

Evaluate temporal extrapolation by completely holding out an intermediate timepoint slice during training. The model encodes $t_0$, integrates forward to $t_1$ in latent space using the native RK4 solver, and decodes to predict $t_1$:

```bash
# Hold out time point index 3 (e.g., E12.5 in mouse embryo)
python benchmark_loto_ours.py --data mouse_embryo --holdout_idx 3 --num_epochs 200
```

### 3. Running Comparison Baselines

Run baseline methods under the exact same LOTO protocol:

```bash
# STORIES (JAX-based spatiotemporal alignment & interpolation)
python benchmark_loto_stories.py --data mouse_embryo --holdout_idx 3

# Moscot (Optimal Transport)
python benchmark_loto_moscot.py --data mouse_embryo --holdout_idx 3

# Spateo / Dynamo (Vector field)
python benchmark_loto_spateo.py --data mouse_embryo --holdout_idx 3
```

### 4. Hyperparameter Sweeping (`hparam_loto_sweep.py`)

Explore hyperparameter combinations (e.g. latent dimension, learning rate, loss weights) across held-out timepoints:

```bash
python hparam_loto_sweep.py --data mouse_embryo --n_genes 3000
```

### 5. Gene Regulatory Network (GRN) Analysis (`grn_analysis.ipynb`)

Interactive notebook to compute:
- Jacobian sensitivity matrix across cells for cell-type-specific gene-gene interactions.
- Regulatory network degree centrality and key hub driver identification.
- GO functional enrichment via `gprofiler`.

---

## Evaluation Metrics

All benchmark scripts calculate standardized metrics defined in `benchmark_utils.py`:

| Metric | Target | Description |
| :--- | :---: | :--- |
| **`x_hvg_corr`** | $\uparrow$ Higher is better | Pearson correlation between predicted and true mean expression across Top-K HVGs. |
| **`x_wasserstein`** | $\downarrow$ Lower is better | Median 1D Wasserstein distance across genes measuring distribution fidelity. |
| **`spatial_domain_ari`** | $\uparrow$ Higher is better | Adjusted Rand Index comparing predicted Leiden spatial clusters with ground truth annotations. |
| **`spatial_coherence_pc1`** | $\uparrow$ Higher is better | Moran's I of the 1st principal component across spatial coordinates to quantify pattern coherence. |
