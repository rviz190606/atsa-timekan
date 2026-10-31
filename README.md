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
Rscript timekan.R                         # train on a synthetic multi-frequency series
Rscript timekan.R --csv data/ETTh1.csv    # train on a CSV: first column is the date, the rest are variables
Rscript timekan.R --epochs 5 --levels 4   # override any option below
```

You can also `source("timekan.R")` in an R session to use the model and data helpers without training.

The data split depends on `--split`. `auto`, the default, uses the standard ETT split for files named `ETTh*`/`ETTm*`:
12/4/4 months for train/val/test, as in the paper. Any other file gets 7:1:2. The script reports test MSE and MAE for the
checkpoint with the best validation MSE.

## Reproducing the paper (ETT datasets)

```bash
mkdir -p data
for f in ETTh1 ETTh2 ETTm1 ETTm2; do
  curl -sSL -o data/$f.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/$f.csv
done
./run_experiments.sh   # main results + ablations; about 1 hour on a laptop GPU
Rscript plots.R        # figures/ and results/comparison.csv (works on partial results too)
```

- `results/main.csv` holds the paper's Table 1 runs: look-back 96, horizons 96/192/336/720.
- `results/ablation.csv` holds the number of frequency bands (Table 9) and fixed-order KANs (Table 3), at horizon 96.
- `results/comparison.csv` puts our MSE/MAE next to the paper's.
- `figures/` holds the decomposition, an example forecast, results vs. the paper, and the ablations.

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
| `--fixed_order` | 0 | if > 0, same KAN order at every level (ablation) |
| `--split` | auto | `auto`, `etth`, `ettm` or `712` |
| `--epochs` | 10 | maximum training epochs |
| `--patience` | 3 | early stopping: epochs without a better validation MSE |
| `--batch_size` | 32 | batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--seed` | 0 | random seed |
| `--device` | auto | `cuda`, `cpu`, or `auto` (GPU if available) |
| `--out` | | append a results row to this CSV |
| `--save` | | save the trained model to this file |
| `--tag` | | label stored in the results row |

`seq_len` must be divisible by `window^(levels − 1)`.

## Notes

- RevIN (per-window normalisation) is added on top of the paper's method, as is standard for these models.
- The default hyperparameters are reasonable choices, not values from the paper. `run_experiments.sh` uses
  `--lr 0.01 --batch_size 128`, picked from a quick check on ETTh1.
