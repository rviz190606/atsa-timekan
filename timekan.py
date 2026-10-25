"""
TimeKAN: KAN-based Frequency Decomposition Learning Architecture
for Long-term Time Series Forecasting (Huang et al., ICLR 2025).

Minimal PyTorch implementation of the paper's Decomposition-Learning-Mixing design:

  1. Hierarchical Sequence Preprocessing (Sec 3.2): moving averages build k
     progressively smoother sequences, each embedded to D dims.
  2. Cascaded Frequency Decomposition (Sec 3.3): f_i = x_i - FreqUp(x_{i+1}).
  3. Multi-order KAN Representation Learning (Sec 3.4):
     f̂_i = DepthwiseConv(f_i) + ChebyKAN(f_i, order = b + k - i).
  4. Frequency Mixing (Sec 3.5): x_i = FreqUp(x_{i+1}) + f̂_i.
  Steps 2-4 are repeated L times, then a linear head produces the forecast.

Usage:
  python timekan.py                       # trains on a synthetic multi-frequency series
  python timekan.py --csv ETTh1.csv       # trains on a CSV (first column = date, rest = variates)
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def frequency_upsample(x, target_len):
    """Frequency Upsampling, Eq. (3): IFFT(ZeroPad(FFT(x))).

    Stretches a short sequence to `target_len` without adding new frequencies.
    x: [B, L_short, D]  ->  [B, target_len, D]
    """
    short_len = x.shape[1]
    spec = torch.fft.rfft(x, dim=1)                      # [B, L_short//2+1, D]
    # irfft with a larger n zero-pads the high-frequency bins for us.
    out = torch.fft.irfft(spec, n=target_len, dim=1)
    # Rescale so amplitudes match (FFT normalisation depends on length).
    return out * (target_len / short_len)


class ChebyKAN(nn.Module):
    """1-layer Chebyshev KAN over the channel dimension, Eqs. (6)-(8).

    phi_o(x) = sum_j sum_{i=0..n} Theta[o, j, i] * T_i(tanh(x_j))
    """

    def __init__(self, dim, order):
        super().__init__()
        self.order = order
        self.coeffs = nn.Parameter(torch.empty(dim, dim, order + 1))   # Theta: [out, in, n+1]
        nn.init.normal_(self.coeffs, std=1.0 / (dim * (order + 1)))

    def forward(self, x):                      # x: [..., D]
        x = torch.tanh(x)                      # squash into [-1, 1], Chebyshev domain
        # T_n(x) = cos(n arccos x), computed with the stable recurrence
        # T_0 = 1, T_1 = x, T_{n+1} = 2x T_n - T_{n-1}.
        polys = [torch.ones_like(x), x]
        for _ in range(2, self.order + 1):
            polys.append(2 * x * polys[-1] - polys[-2])
        polys = torch.stack(polys[: self.order + 1], dim=-1)          # [..., D_in, n+1]
        return torch.einsum("...ji,oji->...o", polys, self.coeffs)    # [..., D_out]


class MKAN(nn.Module):
    """M-KAN block for one frequency band, Eqs. (5), (9), (10).

    Two parallel branches:
      - depthwise conv along time   (temporal dependency)
      - Chebyshev KAN along channels (representation learning)
    """

    def __init__(self, dim, order, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.kan = ChebyKAN(dim, order)

    def forward(self, f):                                      # f: [B, L, D]
        f1 = self.dwconv(f.transpose(1, 2)).transpose(1, 2)    # Eq. (5)
        f2 = self.kan(f)                                       # Eq. (9)
        return f1 + f2                                         # Eq. (10)


class TimeKANLayer(nn.Module):
    """One Decomposition -> Learning -> Mixing pass over k levels."""

    def __init__(self, dim, num_levels, base_order):
        super().__init__()
        k = num_levels
        # Level i (1-indexed) gets order b + k - i: highest frequency -> highest order.
        self.mkans = nn.ModuleList(MKAN(dim, base_order + k - i) for i in range(1, k + 1))

    def forward(self, xs):                     # xs: list of k tensors, xs[0] = highest level
        k = len(xs)

        # --- Cascaded Frequency Decomposition, Eq. (4) ---
        # f_i = x_i - FreqUp(x_{i+1}); the lowest level keeps the remaining low frequencies.
        fs = [xs[i] - frequency_upsample(xs[i + 1], xs[i].shape[1]) for i in range(k - 1)]
        fs.append(xs[-1])

        # --- Multi-order KAN Representation Learning ---
        fs = [mkan(f) for mkan, f in zip(self.mkans, fs)]

        # --- Frequency Mixing, Eq. (11) ---
        # Rebuild multi-level sequences bottom-up: x_i = FreqUp(x_{i+1}) + f̂_i.
        out = [None] * k
        out[-1] = fs[-1]
        for i in range(k - 2, -1, -1):
            out[i] = frequency_upsample(out[i + 1], fs[i].shape[1]) + fs[i]
        return out


class RevIN(nn.Module):
    """Reversible instance normalisation (standard in this line of work / official code)."""

    def forward(self, x, mode, eps=1e-5):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = (x.var(dim=1, keepdim=True, unbiased=False) + eps).sqrt().detach()
            return (x - self.mean) / self.std
        return x * self.std + self.mean        # "denorm"


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TimeKAN(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=16, num_levels=3, window=2,
                 num_layers=2, base_order=1):
        super().__init__()
        assert seq_len % (window ** (num_levels - 1)) == 0, "seq_len must be divisible by window^(k-1)"
        self.num_levels = num_levels
        self.revin = RevIN()
        # Moving average used to remove high frequencies, Eq. (1).
        self.pool = nn.AvgPool1d(window, stride=window)
        # Embedding each scalar time step into D dims, Eq. (2) (one Linear per level).
        self.embeds = nn.ModuleList(nn.Linear(1, d_model) for _ in range(num_levels))
        self.layers = nn.ModuleList(TimeKANLayer(d_model, num_levels, base_order)
                                    for _ in range(num_layers))
        # Forecast head, Eq. (12): project D -> 1, then time T -> F.
        self.proj_dim = nn.Linear(d_model, 1)
        self.proj_time = nn.Linear(seq_len, pred_len)

    def forward(self, x):                      # x: [B, T, N]
        B, T, N = x.shape
        x = self.revin(x, "norm")

        # Variate independence: treat each variate as its own univariate series.
        x = x.permute(0, 2, 1).reshape(B * N, 1, T)          # [B*N, 1, T]

        # --- Hierarchical Sequence Preprocessing, Eqs. (1)-(2) ---
        levels = [x]
        for _ in range(self.num_levels - 1):
            levels.append(self.pool(levels[-1]))             # lengths T, T/d, T/d^2, ...
        xs = [emb(lv.transpose(1, 2)) for emb, lv in zip(self.embeds, levels)]  # each [B*N, L_i, D]

        # --- L stacked Decomposition-Learning-Mixing layers ---
        for layer in self.layers:
            xs = layer(xs)

        # --- Forecast from the highest-level sequence x_1, Eq. (12) ---
        y = self.proj_dim(xs[0]).squeeze(-1)                 # [B*N, T]
        y = self.proj_time(y)                                # [B*N, F]
        y = y.reshape(B, N, -1).permute(0, 2, 1)             # [B, F, N]  (stack variates back)
        return self.revin(y, "denorm")


# ---------------------------------------------------------------------------
# Data + training
# ---------------------------------------------------------------------------

def load_series(csv_path):
    """Return a [time, variates] float array (synthetic if no CSV given)."""
    if csv_path:
        import pandas as pd
        return pd.read_csv(csv_path).iloc[:, 1:].values.astype(np.float32)
    # Synthetic: 3 variates, each a mix of low/mid/high frequency sines + noise.
    t = np.arange(6000)
    rng = np.random.default_rng(0)
    cols = [np.sin(2 * np.pi * t / 200 + p) + 0.5 * np.sin(2 * np.pi * t / 24 + 2 * p)
            + 0.2 * np.sin(2 * np.pi * t / 6) + 0.1 * rng.standard_normal(len(t))
            for p in (0.0, 1.0, 2.0)]
    return np.stack(cols, axis=1).astype(np.float32)


class WindowDataset(torch.utils.data.Dataset):
    """Sliding windows: (past seq_len steps, next pred_len steps)."""

    def __init__(self, data, seq_len, pred_len):
        self.data, self.seq_len, self.pred_len = torch.tensor(data), seq_len, pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, i):
        s = i + self.seq_len
        return self.data[i:s], self.data[s:s + self.pred_len]


def evaluate(model, loader):
    model.eval()
    mse = mae = n = 0.0
    with torch.no_grad():
        for x, y in loader:
            p = model(x)
            mse += F.mse_loss(p, y, reduction="sum").item()
            mae += F.l1_loss(p, y, reduction="sum").item()
            n += y.numel()
    return mse / n, mae / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="CSV file (first column = timestamp)")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--d_model", type=int, default=16)
    ap.add_argument("--levels", type=int, default=3, help="number of frequency bands k")
    ap.add_argument("--window", type=int, default=2, help="moving average window d")
    ap.add_argument("--layers", type=int, default=2, help="number of stacked blocks L")
    ap.add_argument("--base_order", type=int, default=1, help="lower-bound Chebyshev order b")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(0)

    # Chronological 7:1:2 split, z-scored with training statistics.
    data = load_series(args.csv)
    n_train, n_val = int(0.7 * len(data)), int(0.1 * len(data))
    mean, std = data[:n_train].mean(0), data[:n_train].std(0) + 1e-8
    data = (data - mean) / std
    splits = {
        "train": data[:n_train],
        "val": data[n_train - args.seq_len:n_train + n_val],
        "test": data[n_train + n_val - args.seq_len:],
    }
    loaders = {k: torch.utils.data.DataLoader(WindowDataset(v, args.seq_len, args.pred_len),
                                              batch_size=args.batch_size, shuffle=(k == "train"))
               for k, v in splits.items()}

    model = TimeKAN(args.seq_len, args.pred_len, args.d_model, args.levels,
                    args.window, args.layers, args.base_order)
    print(f"TimeKAN parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = F.mse_loss(model(x), y)     # L2 loss, as in the paper
            loss.backward()
            opt.step()
        val_mse, _ = evaluate(model, loaders["val"])
        print(f"epoch {epoch + 1:2d} | train loss {loss.item():.4f} | val MSE {val_mse:.4f}")
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_mse, test_mae = evaluate(model, loaders["test"])
    print(f"test MSE {test_mse:.4f} | test MAE {test_mae:.4f}")


if __name__ == "__main__":
    main()
