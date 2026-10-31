# TimeKAN

A minimal R implementation (using the [`torch`](https://torch.mlverse.org/) package) of **TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting** (Huang et al., ICLR 2025, [arXiv:2502.06910](https://arxiv.org/abs/2502.06910)).

## How it works

TimeKAN splits a time series into frequency bands, learns each band separately, then recombines them:

1. **Hierarchical preprocessing**: moving averages build progressively smoother copies of the input, and each copy is embedded into `d_model` dimensions.
2. **Cascaded Frequency Decomposition**: each band is `f_i = x_i − FreqUp(x_{i+1})`, where `FreqUp` upsamples with FFT → zero-pad → inverse FFT.
3. **Multi-order KAN learning**: each band goes through a depthwise conv plus a Chebyshev KAN. Higher-frequency bands get a higher polynomial order.
4. **Frequency Mixing**: bands are added back together from the lowest up, `x_i = FreqUp(x_{i+1}) + f̂_i`.

Steps 2–4 repeat `--layers` times, then a linear head produces the forecast. Each variable is forecast independently.

## Setup

```r
install.packages("torch")
torch::install_torch()   # downloads libtorch on first use (also prompted automatically)
```

For an NVIDIA GPU, install the CUDA build instead (a ~3.6 GB download):

```bash
CUDA=12.8 Rscript -e 'torch::install_torch(reinstall = TRUE)'
```

If `/tmp` is small (for example an in-memory tmpfs), the unzip can fail. Point `TMPDIR` at a folder on a larger disk first.
If loading torch then reports `libcudart.so.12: cannot open shared object file`, create that name as a symlink to the bundled
`libcudart-*.so.12` file in torch's `lib/` folder.

## Usage

```bash
Rscript timekan.R                       # train on a synthetic multi-frequency series
Rscript timekan.R --csv ETTh1.csv       # train on a CSV: first column is the date, the rest are variables
Rscript timekan.R --epochs 5 --levels 4 # override any option below
```

You can also `source("timekan.R")` in an R session to use the `timekan()` model without running training.

The data is split 7:1:2 (train/val/test) in time order. The script reports test MSE and MAE for the checkpoint with the best validation MSE.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--seq_len` | 96 | look-back window T |
| `--pred_len` | 96 | forecast horizon F |
| `--d_model` | 16 | embedding dimension D |
| `--levels` | 3 | number of frequency bands k |
| `--window` | 2 | moving-average window d |
| `--layers` | 2 | number of stacked blocks L |
| `--base_order` | 1 | lowest Chebyshev order b |
| `--epochs` | 10 | training epochs |
| `--batch_size` | 32 | batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--device` | auto | `cuda`, `cpu`, or `auto` (GPU if available) |

`seq_len` must be divisible by `window^(levels − 1)`.

## Notes

- RevIN (per-window normalisation) is added on top of the paper's method, as is standard for these models.
- The default hyperparameters are reasonable choices, not values from the paper.
