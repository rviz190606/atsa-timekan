# TimeKAN

An enhanced, high-performance R implementation (using the [`torch`](https://torch.mlverse.org/) package) of **TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting** (Huang et al., ICLR 2025, [arXiv:2502.06910](https://arxiv.org/abs/2502.06910)).

This version includes architectural extensions and extensive CPU-oriented performance optimizations designed specifically for fast, accurate training on single PCs without GPUs.

---

## Architecture Overview

TimeKAN decomposes multi-frequency time series into distinct frequency bands, models temporal dependencies and nonlinear representations separately for each band, and recombines them hierarchically:

1. **Hierarchical Preprocessing**: Moving averages progressively smooth the input sequence into $k$ levels ($x_1, \dots, x_k$), each embedded into $D$ hidden dimensions.
2. **Cascaded Frequency Decomposition (CFD)**: Band residuals are extracted bottom-up: $f_i = x_i - \text{FreqUp}(x_{i+1})$, where $\text{FreqUp}$ upsamples via $\text{IFFT}(\text{ZeroPad}(\text{FFT}(x)))$.
3. **Multi-order KAN Representation Learning (M-KAN)**: Each band $f_i$ is processed by two parallel branches:
   - Temporal Branch: Depthwise 1D Convolution along the time dimension.
   - Channel Branch: Multi-order Chebyshev KAN along channel dimensions with order $b + k - i$.
4. **Frequency Mixing**: Bands are recombined bottom-up: $x_i = \text{FreqUp}(x_{i+1}) + \hat{f}_i$.
5. **Prediction**: Stacked across $L$ layers; a linear forecasting head projects $x_1$ into future horizon $F$.

---

## Extensions Implemented

### 1. Learnable M-KAN Branch Fusion (`--mkan_fusion`)
Instead of fixed addition $\hat{f}_i = f_{i, 1} + f_{i, 2}$, the outputs of the Depthwise Conv and Chebyshev KAN branches are adaptively balanced via learnable scalars:
- **`learnable` (default)**: Independent learnable scalars $\alpha$ and $\beta$:
  $$\hat{f}_i = \alpha f_{i, 1} + \beta f_{i, 2}$$
  Initialized to $\alpha = 0.5, \beta = 0.5$ for balanced starting dynamics.
- **`convex`**: Normalized convex combination bounded in $(0, 1)$ via sigmoid gating:
  $$\alpha = \sigma(w_\alpha), \quad \hat{f}_i = \alpha f_{i, 1} + (1 - \alpha) f_{i, 2}$$
  Initialized to $w_\alpha = 0 \implies \alpha = 0.5$.
- **`add`**: Fixed direct sum $f_{i, 1} + f_{i, 2}$ as in the vanilla paper.

### 2. Chebyshev Polynomial Complexity Regularization (`--poly_reg`, `--poly_power`)
Chebyshev polynomials of order $k$ oscillate with increasing frequency $T_k(x) = \cos(k \arccos x)$. Higher-order polynomials are susceptible to overfitting and boundary oscillation (Runge phenomenon). We add an order-dependent penalty to the loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{poly}} \sum_{l=1}^L \sum_{i=1}^k \sum_{d=0}^{\text{order}} d^p \|\Theta_{:, :, d}\|_2^2$$
- Penalty is zero for degree 0 (constant DC component).
- Penalty scales with $d^p$ (default $p = 1$ linear, or $p = 2$ quadratic), discouraging unwarranted polynomial complexity.

### 3. High-Dimensional PCA Acceleration with Inverse PCA Reconstruction (`--pca_dim`, `--pca_var`)
TimeKAN forecasts variables independently ($B \times N$). For datasets with hundreds of variables (e.g., `electricity.csv` with 321 sensors), vanilla training processes thousands of sequences per batch, causing severe CPU slowdowns.
- **PCA Decomposition**: PCA is fitted strictly on the training set (preventing data leakage). The time series is projected into $K_{\text{pca}} \ll N$ orthogonal principal components:
  $$Z = (X_{\text{norm}}) W, \quad W \in \mathbb{R}^{N \times K_{\text{pca}}}$$
- **Fast Latent Training**: TimeKAN is trained on $Z$ (e.g. 16 channels instead of 321), speeding up training by **10x–20x** while capturing cross-variable correlations.
- **Inverse PCA Transform**: Future forecasts $\hat{Z} \in \mathbb{R}^{B \times F \times K_{\text{pca}}}$ are mapped back to all $N$ original variables via:
  $$\hat{X}_{\text{orig}} = \hat{Z} W^T \in \mathbb{R}^{B \times F \times N}$$
- **Full-Variable Evaluation**: Test MSE and MAE are evaluated directly on the reconstructed forecasts against the ground truth of all $N$ original variables.

---

## CPU-Efficiency Optimizations (No-GPU Setup)

- **Vectorized GEMM in ChebyKAN**: Replaced PyTorch `einsum` with flattened tensor matrix multiplication (`torch_matmul`), yielding an immediate **~2x speedup** per KAN layer using CPU BLAS/AVX routines.
- **Multi-Threading**: Automatically detects physical CPU cores and sets `torch_set_num_threads()` to utilize full multi-core parallel performance.
- **Sliding Window Stride (`--train_stride`)**: For dense long series (e.g., 15-minute sampling in ETTm), training stride allows sampling non-overlapping or lightly overlapping windows, reducing epoch duration by 2x–4x without sacrificing generalization.
- **Early Stopping (`--patience`)**: Monitors validation MSE and stops automatically when validation loss plateaus, eliminating wasted CPU epochs.

---

## Setup

```r
# Install torch and coro in R:
install.packages("torch")
install.packages("coro")
torch::install_torch() # downloads LibTorch CPU binaries on first use
```

---

## Usage

```bash
# 1. Train on synthetic multi-frequency series:
Rscript timekan.R --epochs 5

# 2. Train on ETTh1 benchmark dataset:
Rscript timekan.R --csv data/ETTh1.csv --epochs 5

# 3. Train on high-dimensional Electricity dataset (321 variables) with PCA:
# Reduces 321 variables -> 16 principal components (20x training speedup!)
Rscript timekan.R --csv data/electricity.csv --pca_dim 16 --train_stride 2 --epochs 5

# 4. Automatically select PCA components to retain 90% variance:
Rscript timekan.R --csv data/weather.csv --pca_var 0.90 --epochs 5

# 5. Experiment with convex M-KAN fusion and polynomial regularization:
Rscript timekan.R --csv data/ETTh1.csv --mkan_fusion convex --poly_reg 1e-4 --poly_power 2
```

You can also `source("timekan.R")` in an R session to import `timekan()` and use it programmatically.

---

## Options & Flags

| Flag | Default | Description |
|---|---|---|
| `--csv` | `""` | Path to CSV data file (empty = synthetic multi-frequency series). |
| `--seq_len` | `96` | Input look-back window length $T$. |
| `--pred_len` | `96` | Forecast horizon length $F$. |
| `--d_model` | `16` | Hidden embedding dimension $D$. |
| `--levels` | `3` | Number of frequency decomposition levels $k$. |
| `--window` | `2` | Moving-average downsampling factor $d$. |
| `--layers` | `2` | Number of stacked TimeKAN blocks $L$. |
| `--base_order` | `1` | Lowest Chebyshev polynomial order $b$. |
| `--epochs` | `10` | Maximum number of training epochs. |
| `--batch_size` | `64` | Batch size (optimized for CPU cache and vectorization). |
| `--lr` | `1e-3` | Adam optimizer learning rate. |
| `--mkan_fusion` | `"learnable"` | M-KAN branch fusion: `"learnable"` ($\alpha f_1 + \beta f_2$), `"convex"` ($\alpha f_1 + (1-\alpha)f_2$), or `"add"`. |
| `--poly_reg` | `1e-4` | Chebyshev polynomial complexity regularization coefficient $\lambda_{\text{poly}}$ ($0$ to disable). |
| `--poly_power` | `1` | Exponent for polynomial order penalty ($1$ for linear $d$, $2$ for quadratic $d^2$). |
| `--pca_dim` | `0` | Number of PCA components ($0$ to disable). |
| `--pca_var` | `0.0` | Target explained variance ratio for PCA (e.g. `0.90` for 90%; $0$ to disable). |
| `--split` | `"auto"` | Split ratio: `"auto"` (6:2:2 for ETT, 7:1:2 for others), `"6:2:2"`, or `"7:1:2"`. |
| `--train_stride` | `1` | Window stride during training (set $\ge 2$ to speed up large datasets). |
| `--patience` | `3` | Early stopping patience (stop if no val improvement for $P$ epochs; $0$ to disable). |
| `--threads` | `0` | PyTorch CPU threads ($0$ = auto-detect all CPU cores). |

`seq_len` must be divisible by `window^(levels − 1)`.
