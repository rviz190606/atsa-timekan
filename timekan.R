# TimeKAN: KAN-based Frequency Decomposition Learning Architecture
# for Long-term Time Series Forecasting (Huang et al., ICLR 2025).
#
# Minimal R (torch) implementation of the paper's Decomposition-Learning-Mixing design:
#
#   1. Hierarchical Sequence Preprocessing (Sec 3.2): moving averages build k
#      progressively smoother sequences, each embedded to D dims.
#   2. Cascaded Frequency Decomposition (Sec 3.3): f_i = x_i - FreqUp(x_{i+1}).
#   3. Multi-order KAN Representation Learning (Sec 3.4):
#      f_hat_i = DepthwiseConv(f_i) + ChebyKAN(f_i, order = b + k - i).
#   4. Frequency Mixing (Sec 3.5): x_i = FreqUp(x_{i+1}) + f_hat_i.
#   Steps 2-4 are repeated L times, then a linear head produces the forecast.
#
# Usage:
#   Rscript timekan.R                         # synthetic multi-frequency series
#   Rscript timekan.R --csv ETTh1.csv         # CSV: first column = date, rest = variates
#   Rscript timekan.R --epochs 5 --levels 4   # any option in default_options can be overridden
#
# Tensor shapes in comments use: B = batch, T = input length, F = forecast length,
# N = number of variates, D = embedding dim, L = sequence length at a given level.

library(torch)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

# Frequency Upsampling, Eq. (3): IFFT(ZeroPad(FFT(x))).
# Stretches a short sequence to `target_len` without adding new frequencies.
# x: [B, L_short, D]  ->  [B, target_len, D]
frequency_upsample <- function(x, target_len) {
  short_len <- x$size(2)
  spec <- torch_fft_rfft(x, dim = 2)
  # irfft with a larger n zero-pads the high-frequency bins for us.
  out <- torch_fft_irfft(spec, n = target_len, dim = 2)
  # Rescale so amplitudes match (FFT normalisation depends on length).
  out * (target_len / short_len)
}


# Basis functions g_0..g_n evaluated at x, stacked on a new last dim: [..., n+1].
# The paper uses Chebyshev polynomials; the others are for the basis-swap extension.
# Every basis returns n+1 functions for "order" n, so swapping keeps the parameter count.
#   chebyshev: T_0..T_n of tanh(x)            T_0 = 1, T_1 = u, T_{k+1} = 2u T_k - T_{k-1}
#   legendre:  P_0..P_n of tanh(x)            P_0 = 1, P_1 = u, (k+1) P_{k+1} = (2k+1) u P_k - k P_{k-1}
#   fourier:   1, sin(pi u/2), cos(pi u/2), sin(pi u), cos(pi u), ...   (first n+1 of these)
#   rbf:       n+1 Gaussian bumps on u, centres evenly spaced over [-1, 1]
#   linear:    1, x  (control: an ordinary linear layer, ignores n)
# Here u = tanh(x) squashes inputs into [-1, 1], the natural domain of these bases.
kan_basis <- function(x, order, basis = "chebyshev") {
  if (basis == "linear") return(torch_stack(list(torch_ones_like(x), x), dim = -1))
  u <- torch_tanh(x)
  n <- order
  g <- switch(basis,
    chebyshev = {
      g <- list(torch_ones_like(u), u)
      if (n >= 2) for (k in 1:(n - 1)) g[[k + 2]] <- 2 * u * g[[k + 1]] - g[[k]]
      g
    },
    legendre = {
      g <- list(torch_ones_like(u), u)
      if (n >= 2) for (k in 1:(n - 1)) g[[k + 2]] <- ((2 * k + 1) * u * g[[k + 1]] - k * g[[k]]) / (k + 1)
      g
    },
    fourier = {
      g <- list(torch_ones_like(u))
      m <- 1
      while (length(g) < n + 1) {
        g[[length(g) + 1]] <- torch_sin(m * pi * u / 2)
        g[[length(g) + 1]] <- torch_cos(m * pi * u / 2)
        m <- m + 1
      }
      g
    },
    rbf = {
      centres <- seq(-1, 1, length.out = n + 1)
      width <- 2 / max(n, 1)                    # neighbouring bumps overlap
      lapply(centres, function(c) torch_exp(-((u - c) / width)^2))
    },
    stop("unknown basis: ", basis))
  torch_stack(g[1:(n + 1)], dim = -1)
}

# 1-layer KAN over the channel dimension, Eqs. (6)-(8):
#   phi_o(x) = sum_j sum_{i=0..n} Theta[o, j, i] * g_i(x_j)
# Each input->output edge has its own learnable function instead of a fixed weight.
# With the default Chebyshev basis this is the paper's ChebyKAN.
cheby_kan <- nn_module(
  "KAN",
  initialize = function(dim, order, basis = "chebyshev") {
    self$order <- order
    self$basis <- basis
    n_funcs <- if (basis == "linear") 2 else order + 1
    # Theta: [D_out, D_in, number of basis functions]
    self$coeffs <- nn_parameter(torch_randn(dim, dim, n_funcs) / (dim * n_funcs))
  },
  forward = function(x) {                       # x: [..., D]
    g <- kan_basis(x, self$order, self$basis)   # [..., D_in, n_funcs]
    # Sum over input channel j and basis function i for each output channel o.
    torch_einsum("...ji,oji->...o", list(g, self$coeffs))   # [..., D_out]
  }
)


# M-KAN block for one frequency band, Eqs. (5), (9), (10). Two parallel branches:
#   - depthwise conv along time    (temporal dependency)
#   - KAN along channels           (representation learning; Chebyshev in the paper)
mkan <- nn_module(
  "MKAN",
  initialize = function(dim, order, kernel_size = 3, basis = "chebyshev") {
    # groups = dim makes it depthwise: each channel gets its own 1D filter.
    self$dwconv <- nn_conv1d(dim, dim, kernel_size, padding = kernel_size %/% 2, groups = dim)
    self$kan <- cheby_kan(dim, order, basis)
  },
  forward = function(f) {                                   # f: [B, L, D]
    f1 <- self$dwconv(f$transpose(2, 3))$transpose(2, 3)    # Eq. (5); conv wants [B, D, L]
    f2 <- self$kan(f)                                       # Eq. (9)
    f1 + f2                                                 # Eq. (10)
  }
)


# Chebyshev order of the KAN for each band (band 1 = highest frequency).
#   "paper":    b + k - i, so the highest-frequency band gets the highest order (k=3, b=1: 3,2,1)
#   "reversed": b + i - 1, the opposite assignment                         (k=3, b=1: 1,2,3)
#   fixed_order > 0 overrides both and gives every band that order (ablation, paper Table 3).
band_orders <- function(k, base_order, fixed_order = 0, order_mode = "paper") {
  if (fixed_order > 0) return(rep(fixed_order, k))
  switch(order_mode,
         paper    = base_order + k - (1:k),
         reversed = base_order + (1:k) - 1,
         stop("unknown order_mode: ", order_mode))
}


# One Decomposition -> Learning -> Mixing pass over k levels.
# Input/output: a list of k tensors; xs[[1]] is the finest (longest) level.
timekan_layer <- nn_module(
  "TimeKANLayer",
  initialize = function(dim, num_levels, base_order, fixed_order = 0, order_mode = "paper",
                        basis = "chebyshev") {
    self$orders <- band_orders(num_levels, base_order, fixed_order, order_mode)
    self$mkans <- nn_module_list(lapply(self$orders, function(o) mkan(dim, o, basis = basis)))
  },

  # --- Cascaded Frequency Decomposition, Eq. (4) ---
  # f_i = x_i - FreqUp(x_{i+1}): what level i has that the smoother level i+1 lacks.
  # The last (smoothest) level keeps the remaining low frequencies as-is.
  decompose = function(xs) {
    k <- length(xs)
    fs <- vector("list", k)
    for (i in seq_len(k - 1)) {
      fs[[i]] <- xs[[i]] - frequency_upsample(xs[[i + 1]], xs[[i]]$size(2))
    }
    fs[[k]] <- xs[[k]]
    fs
  },

  forward = function(xs) {
    k <- length(xs)
    fs <- self$decompose(xs)

    # --- Multi-order KAN Representation Learning (one M-KAN per band) ---
    for (i in 1:k) fs[[i]] <- self$mkans[[i]](fs[[i]])

    # --- Frequency Mixing, Eq. (11) ---
    # Rebuild multi-level sequences bottom-up: x_i = FreqUp(x_{i+1}) + f_hat_i.
    out <- vector("list", k)
    out[[k]] <- fs[[k]]
    for (i in rev(seq_len(k - 1))) {
      out[[i]] <- frequency_upsample(out[[i + 1]], fs[[i]]$size(2)) + fs[[i]]
    }
    out
  }
)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

timekan <- nn_module(
  "TimeKAN",
  initialize = function(seq_len, pred_len, d_model = 16, num_levels = 3, window = 2,
                        num_layers = 2, base_order = 1, fixed_order = 0,
                        order_mode = "paper", basis = "chebyshev") {
    stopifnot("seq_len must be divisible by window^(levels - 1)" =
                seq_len %% (window^(num_levels - 1)) == 0)
    self$num_levels <- num_levels
    # Moving average used to remove high frequencies, Eq. (1).
    self$pool <- nn_avg_pool1d(window, stride = window)
    # Embed each scalar time step into D dims, Eq. (2) (one Linear per level).
    self$embeds <- nn_module_list(lapply(1:num_levels, function(i) nn_linear(1, d_model)))
    self$layers <- nn_module_list(lapply(1:num_layers, function(i)
      timekan_layer(d_model, num_levels, base_order, fixed_order, order_mode, basis)))
    # Forecast head, Eq. (12): project D -> 1, then time T -> F.
    self$proj_dim <- nn_linear(d_model, 1)
    self$proj_time <- nn_linear(seq_len, pred_len)
  },

  # Normalise and build the embedded multi-level sequences (everything before the layers).
  encode = function(x) {                        # x: [B, T, N]
    B <- x$size(1); T <- x$size(2); N <- x$size(3)

    # RevIN (reversible instance norm): normalise each window, undo it at the end.
    # Not part of the paper's method, but standard in this line of work.
    mu <- x$mean(dim = 2, keepdim = TRUE)$detach()
    sigma <- torch_sqrt(x$var(dim = 2, keepdim = TRUE, unbiased = FALSE) + 1e-5)$detach()
    x <- (x - mu) / sigma

    # Variate independence: treat each variate as its own univariate series.
    x <- x$permute(c(1, 3, 2))$reshape(c(B * N, 1, T))       # [B*N, 1, T]

    # --- Hierarchical Sequence Preprocessing, Eqs. (1)-(2) ---
    # Lengths T, T/d, T/d^2, ...; each level is a smoother copy of the one before.
    levels <- list(x)
    for (i in seq_len(self$num_levels - 1)) {
      levels[[i + 1]] <- self$pool(levels[[i]])
    }
    xs <- lapply(seq_along(levels), function(i)
      self$embeds[[i]](levels[[i]]$transpose(2, 3)))         # each [B*N, L_i, D]
    list(xs = xs, mu = mu, sigma = sigma, B = B, N = N)
  },

  forward = function(x) {                       # x: [B, T, N]
    e <- self$encode(x)
    xs <- e$xs

    # --- L stacked Decomposition-Learning-Mixing layers ---
    for (i in seq_along(self$layers)) xs <- self$layers[[i]](xs)

    # --- Forecast from the finest-level sequence x_1, Eq. (12) ---
    y <- self$proj_dim(xs[[1]])$squeeze(-1)                  # [B*N, T]
    y <- self$proj_time(y)                                   # [B*N, F]
    y <- y$reshape(c(e$B, e$N, -1))$permute(c(1, 3, 2))      # [B, F, N] (stack variates back)

    y * e$sigma + e$mu                                       # undo RevIN
  }
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Return a [time, variates] numeric matrix (synthetic if no CSV given).
load_series <- function(csv_path = "") {
  if (nzchar(csv_path)) {
    df <- read.csv(csv_path)
    return(as.matrix(df[, -1]))                 # drop the date column
  }
  # Synthetic: 3 variates, each a mix of low/mid/high frequency sines + noise.
  set.seed(0)
  t <- 0:5999
  sapply(c(0, 1, 2), function(p)
    sin(2 * pi * t / 200 + p) + 0.5 * sin(2 * pi * t / 24 + 2 * p) +
      0.2 * sin(2 * pi * t / 6) + 0.1 * rnorm(length(t)))
}

# Row ranges [start, end] of the train/val/test parts.
#   "etth"/"ettm": the standard ETT benchmark split (Zhou et al. 2021, used by the paper):
#                  12 months train, 4 months val, 4 months test (months of 30 days).
#                  The last ~6 months of the file are not used.
#   "712":         70% / 10% / 20% of the whole series.
#   "auto":        "etth"/"ettm" if the file name starts with ETTh/ETTm, else "712".
split_borders <- function(n_rows, split, csv_path) {
  if (split == "auto") {
    name <- basename(csv_path)
    split <- if (grepl("^ETTh", name)) "etth" else if (grepl("^ETTm", name)) "ettm" else "712"
  }
  if (split %in% c("etth", "ettm")) {
    month <- 30 * 24 * (if (split == "ettm") 4 else 1)   # rows per month (hourly or 15-min)
    ends <- c(12, 16, 20) * month
  } else {
    ends <- c(floor(0.7 * n_rows), floor(0.7 * n_rows) + floor(0.1 * n_rows), n_rows)
  }
  list(name = split, train = c(1, ends[1]), val = c(ends[1] + 1, ends[2]),
       test = c(ends[2] + 1, ends[3]))
}

# Load, split and z-score the data (using training statistics only).
# Returns the three parts plus the scaling stats (needed to plot in original units).
prepare_data <- function(opt) {
  data <- load_series(opt$csv)
  b <- split_borders(nrow(data), opt$split, opt$csv)
  train_rows <- b$train[1]:b$train[2]
  mu <- colMeans(data[train_rows, , drop = FALSE])
  sdev <- apply(data[train_rows, , drop = FALSE], 2, sd) + 1e-8
  data <- scale(data, center = mu, scale = sdev)

  # Val/test parts start seq_len rows early, so their first forecast begins exactly
  # at the split boundary (the look-back window may reach into the previous part).
  part <- function(r, back) data[(r[1] - back):r[2], , drop = FALSE]
  list(split = b$name,
       train = part(b$train, 0),
       val   = part(b$val, opt$seq_len),
       test  = part(b$test, opt$seq_len),
       mu = mu, sdev = sdev, names = colnames(data))
}

# Sliding windows: (past seq_len steps, next pred_len steps).
# Window i covers rows i .. i+seq_len-1 (input) and the next pred_len rows (target).
window_dataset <- dataset(
  "WindowDataset",
  initialize = function(data, seq_len, pred_len) {
    self$data <- unname(as.matrix(data))        # plain [time, variates] R matrix
    self$seq_len <- seq_len
    self$pred_len <- pred_len
  },
  .length = function() {
    nrow(self$data) - self$seq_len - self$pred_len + 1
  },
  # Build a whole batch at once (much faster than one window at a time).
  # `index` holds the 1-based start rows of the windows in this batch.
  .getbatch = function(index) {
    gather <- function(offset, len) {
      rows <- as.vector(outer(0:(len - 1), index + offset, "+"))   # window by window
      torch_tensor(self$data[rows, , drop = FALSE], dtype = torch_float())$
        reshape(c(length(index), len, ncol(self$data)))             # [batch, len, N]
    }
    list(x = gather(0, self$seq_len), y = gather(self$seq_len, self$pred_len))
  }
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Mean squared and absolute error over every forecast value in the loader.
evaluate <- function(model, loader, device) {
  model$eval()
  mse <- 0; mae <- 0; n <- 0
  with_no_grad({
    coro::loop(for (b in loader) {
      x <- b$x$to(device = device); y <- b$y$to(device = device)
      p <- model(x)
      mse <- mse + nnf_mse_loss(p, y, reduction = "sum")$item()
      mae <- mae + nnf_l1_loss(p, y, reduction = "sum")$item()
      n <- n + y$numel()
    })
  })
  c(mse = mse / n, mae = mae / n)
}

# All options with their defaults. Any of them can be overridden as "--name value".
default_options <- list(
  csv = "",              # CSV file (first column = timestamp); empty = synthetic data
  split = "auto",        # "auto", "etth", "ettm" or "712" (see split_borders)
  seq_len = 96,          # look-back window T
  pred_len = 96,         # forecast horizon F
  d_model = 16,          # embedding dimension D
  levels = 3,            # number of frequency bands k
  window = 2,            # moving-average window d
  layers = 2,            # number of stacked blocks L
  base_order = 1,        # lowest Chebyshev order b
  fixed_order = 0,       # if > 0, use this KAN order at every level (ablation)
  order_mode = "paper",  # "paper" (high freq -> high order) or "reversed" (see band_orders)
  basis = "chebyshev",   # KAN function family: chebyshev, legendre, fourier, rbf, linear
  epochs = 10,
  patience = 3,          # stop after this many epochs without a better val MSE
  batch_size = 32,
  lr = 1e-3,
  seed = 0,
  device = "auto",       # "cuda", "cpu", or "auto" (GPU if available)
  out = "",              # if set, append a row of results to this CSV
  save = "",             # if set, save the trained model to this file
  tag = ""               # free-text label stored in the results row (e.g. "ablation")
)

# Minimal "--flag value" parser; returns `defaults` with any overrides applied.
parse_args <- function(defaults, args = commandArgs(trailingOnly = TRUE)) {
  for (i in seq_len(length(args) %/% 2) * 2 - 1) {
    key <- sub("^--", "", args[i])
    if (!key %in% names(defaults)) stop("Unknown option: ", args[i])
    val <- args[i + 1]
    defaults[[key]] <- if (is.numeric(defaults[[key]])) as.numeric(val) else val
  }
  defaults
}

# Train one model with options `opt`, report test MSE/MAE, optionally save results/model.
run_experiment <- function(opt) {
  set.seed(opt$seed)
  torch_manual_seed(opt$seed)
  device <- if (opt$device == "auto") {
    if (cuda_is_available()) "cuda" else "cpu"
  } else opt$device

  d <- prepare_data(opt)
  cat(sprintf("data: %s | split: %s | %d variates | seq_len %d -> pred_len %d | device: %s\n",
              if (nzchar(opt$csv)) basename(opt$csv) else "synthetic", d$split,
              ncol(d$train), opt$seq_len, opt$pred_len, device))
  loaders <- lapply(c(train = "train", val = "val", test = "test"), function(k)
    dataloader(window_dataset(d[[k]], opt$seq_len, opt$pred_len),
               batch_size = opt$batch_size, shuffle = (k == "train")))

  model <- timekan(opt$seq_len, opt$pred_len, opt$d_model, opt$levels,
                   opt$window, opt$layers, opt$base_order, opt$fixed_order,
                   opt$order_mode, opt$basis)
  model$to(device = device)
  n_params <- sum(sapply(model$parameters, function(p) p$numel()))
  cat(sprintf("TimeKAN parameters: %s\n", format(n_params, big.mark = ",")))
  optimizer <- optim_adam(model$parameters, lr = opt$lr)

  # Keep the weights with the best validation MSE; stop early if it stops improving.
  best_val <- Inf
  best_state <- NULL
  bad_epochs <- 0
  t0 <- Sys.time()
  for (epoch in 1:opt$epochs) {
    start <- Sys.time()
    model$train()
    coro::loop(for (b in loaders$train) {
      optimizer$zero_grad()
      x <- b$x$to(device = device); y <- b$y$to(device = device)
      loss <- nnf_mse_loss(model(x), y)        # L2 loss, as in the paper
      loss$backward()
      optimizer$step()
    })
    val <- evaluate(model, loaders$val, device)
    secs <- as.numeric(difftime(Sys.time(), start, units = "secs"))
    cat(sprintf("epoch %2d | train loss %.4f | val MSE %.4f | %.1fs\n",
                epoch, loss$item(), val[["mse"]], secs))
    if (val[["mse"]] < best_val) {
      best_val <- val[["mse"]]
      best_state <- lapply(model$state_dict(), function(t) t$clone())
      bad_epochs <- 0
    } else {
      bad_epochs <- bad_epochs + 1
      if (bad_epochs >= opt$patience) { cat("early stopping\n"); break }
    }
  }

  model$load_state_dict(best_state)
  test <- evaluate(model, loaders$test, device)
  minutes <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  cat(sprintf("test MSE %.4f | test MAE %.4f\n", test[["mse"]], test[["mae"]]))

  if (nzchar(opt$out)) {
    row <- data.frame(
      dataset = if (nzchar(opt$csv)) sub("\\.csv$", "", basename(opt$csv)) else "synthetic",
      tag = opt$tag, seq_len = opt$seq_len, pred_len = opt$pred_len, levels = opt$levels,
      base_order = opt$base_order, fixed_order = opt$fixed_order,
      order_mode = opt$order_mode, basis = opt$basis, d_model = opt$d_model, seed = opt$seed,
      params = n_params, epochs_run = epoch, minutes = round(minutes, 2),
      val_mse = best_val, test_mse = test[["mse"]], test_mae = test[["mae"]])
    write.table(row, opt$out, sep = ",", row.names = FALSE,
                col.names = !file.exists(opt$out), append = file.exists(opt$out))
  }
  if (nzchar(opt$save)) torch_save(model, opt$save)
  invisible(list(model = model, test = test, data = d, device = device))
}

# Run only when executed as a script (not when source()'d, e.g. by plots.R).
if (sys.nframe() == 0) run_experiment(parse_args(default_options))
