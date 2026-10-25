# TimeKAN

A minimal PyTorch implementation of **TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting** (Huang et al., ICLR 2025, [arXiv:2502.06910](https://arxiv.org/abs/2502.06910)).

## How it works

TimeKAN splits a time series into frequency bands, learns each band separately, then recombines them:

1. **Hierarchical preprocessing**: moving averages build progressively smoother copies of the input, and each copy is embedded into `d_model` dimensions.
2. **Cascaded Frequency Decomposition**: each band is `f_i = x_i − FreqUp(x_{i+1})`, where `FreqUp` upsamples with FFT → zero-pad → inverse FFT.
3. **Multi-order KAN learning**: each band goes through a depthwise conv plus a Chebyshev KAN. Higher-frequency bands get a higher polynomial order.
4. **Frequency Mixing**: bands are added back together from the lowest up, `x_i = FreqUp(x_{i+1}) + f̂_i`.

Steps 2–4 repeat `--layers` times, then a linear head produces the forecast. Each variable is forecast independently.

## Setup


```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas
```

Without a GPU, the CPU-only PyTorch build is a much smaller download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas
```

## Usage

```bash
python timekan.py                    # train on a synthetic multi-frequency series
python timekan.py --csv ETTh1.csv    # train on a CSV: first column is the date, the rest are variables
python timekan.py --help             # list all options
```

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

`seq_len` must be divisible by `window^(levels − 1)`.

## Notes

- RevIN (per-window normalisation) is added on top of the paper's method, as is standard for these models.
- The default hyperparameters are reasonable choices, not values from the paper.
