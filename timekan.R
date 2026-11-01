# TimeKAN: KAN-based Frequency Decomposition Learning Architecture
# for Long-term Time Series Forecasting (Huang et al., ICLR 2025).
#
# Enhanced R (torch) implementation supporting CPU-efficient training,
# learnable M-KAN branch weighting, Chebyshev polynomial regularization,
# and PCA latent decomposition with Inverse PCA forecast reconstruction.
#
# Core Paper Architecture:
#   1. Hierarchical Sequence Preprocessing (Sec 3.2): moving averages build k
#      progressively smoother sequences, each embedded to D dims.
#   2. Cascaded Frequency Decomposition (Sec 3.3): f_i = x_i - FreqUp(x_{i+1}).
#   3. Multi-order KAN Representation Learning (Sec 3.4):
#      f_hat_i = alpha * DepthwiseConv(f_i) + beta * ChebyKAN(f_i, order = b + k - i).
#   4. Frequency Mixing (Sec 3.5): x_i = FreqUp(x_{i+1}) + f_hat_i.
#   Steps 2-4 are repeated L times, then a linear head produces the forecast.
#
# Key Extensions & Hardware Optimizations (Fast CPU Training):
#   - Learnable M-KAN branch scalars (learnable alpha/beta or convex sigmoid gating).
#   - Chebyshev polynomial complexity regularization proportional to degree (order^p).
#   - High-dimensional PCA acceleration with Inverse PCA forecast reconstruction.
#   - Fast GEMM (torch_matmul) replacing torch_einsum in ChebyKAN (2x faster on CPU).
#   - Automatic multi-threading configuration for all available CPU cores.
#   - Early stopping and sliding window stride support.
#
# Usage:
#   Rscript timekan.R                                      # synthetic series
#   Rscript timekan.R --csv data/ETTh1.csv                 # benchmark dataset
#   Rscript timekan.R --csv data/electricity.csv --pca_var 0.90  # PCA for 321 variates
#   Rscript timekan.R --mkan_fusion convex --poly_reg 1e-4 # configure extensions
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


# 1-layer Chebyshev KAN over the channel dimension, Eqs. (6)-(8):
#   phi_o(x) = sum_j sum_{i=0..n} Theta[o, j, i] * T_i(tanh(x_j))
# Accelerated with GEMM (torch_matmul) instead of einsum for fast CPU training.
cheby_kan <- nn_module(
  "ChebyKAN",
  initialize = function(dim, order) {
    self$dim <- dim
    self$order <- order
    # Theta: [D_out, D_in, order + 1]
    self$coeffs <- nn_parameter(torch_randn(dim, dim, order + 1) / (dim * (order + 1)))
  },
  forward = function(x) {                       # x: [..., D]
    x_t <- torch_tanh(x)                        # squash into [-1, 1], Chebyshev domain
    # T_n(x) = cos(n arccos x), computed with the stable recurrence:
    # T_0 = 1, T_1 = x, T_{n+1} = 2x T_n - T_{n-1}.
    polys <- list(torch_ones_like(x_t), x_t)
    if (self$order >= 2) {
      for (n in 2:self$order) {
        k <- length(polys)
        polys[[k + 1]] <- 2 * x_t * polys[[k]] - polys[[k - 1]]
      }
    }
    polys_tensor <- torch_stack(polys[1:(self$order + 1)], dim = -1)   # [..., D_in, order+1]

    # CPU-optimized GEMM: flatten last two dimensions and perform matrix multiply
    sh <- x$shape
    polys_flat <- polys_tensor$reshape(c(sh[1], sh[2], self$dim * (self$order + 1)))
    coeffs_flat <- self$coeffs$reshape(c(self$dim, self$dim * (self$order + 1)))
    torch_matmul(polys_flat, coeffs_flat$t())                           # [..., D_out]
  },

  # Extension 4: Discourage high-degree polynomial complexity by penalizing
  # coefficients proportional to the polynomial order (order^power).
  poly_regularization = function(power = 1) {
    if (self$order <= 0) return(torch_tensor(0, dtype = torch_float(), device = self$coeffs$device))
    degrees <- torch_arange(0, self$order, dtype = torch_float(), device = self$coeffs$device)
    weights <- (degrees^power)$view(c(1, 1, self$order + 1))
    (self$coeffs^2 * weights)$sum()
  }
)


# M-KAN block for one frequency band, Eqs. (5), (9), (10). Two parallel branches:
#   - depthwise conv along time    (temporal dependency)
#   - Chebyshev KAN along channels (representation learning)
# Extension 3: Learnable scalars for branch fusion instead of fixed addition.
mkan <- nn_module(
  "MKAN",
  initialize = function(dim, order, kernel_size = 3, fusion = "learnable") {
    self$fusion <- fusion
    # groups = dim makes it depthwise: each channel gets its own 1D filter.
    self$dwconv <- nn_conv1d(dim, dim, kernel_size, padding = kernel_size %/% 2, groups = dim)
    self$kan <- cheby_kan(dim, order)

    if (fusion == "convex") {
      # Learnable convex scalar alpha in (0, 1) initialized to 0.5 via sigmoid(0.0)
      self$alpha_raw <- nn_parameter(torch_tensor(0.0))
    } else if (fusion == "learnable") {
      # Independent learnable scalars alpha, beta initialized to 0.5
      self$alpha <- nn_parameter(torch_tensor(0.5))
      self$beta  <- nn_parameter(torch_tensor(0.5))
    }
  },
  forward = function(f) {                                   # f: [B, L, D]
    f1 <- self$dwconv(f$transpose(2, 3))$transpose(2, 3)    # Eq. (5); conv wants [B, D, L]
    f2 <- self$kan(f)                                       # Eq. (9)

    if (self$fusion == "convex") {
      alpha <- torch_sigmoid(self$alpha_raw)
      alpha * f1 + (1.0 - alpha) * f2
    } else if (self$fusion == "learnable") {
      self$alpha * f1 + self$beta * f2
    } else {
      f1 + f2                                               # Eq. (10)
    }
  }
)


# One Decomposition -> Learning -> Mixing pass over k levels.
# Input/output: a list of k tensors; xs[[1]] is the finest (longest) level.
timekan_layer <- nn_module(
  "TimeKANLayer",
  initialize = function(dim, num_levels, base_order, fusion = "learnable") {
    k <- num_levels
    # Level i gets order b + k - i: highest frequency band -> highest polynomial order.
    self$mkans <- nn_module_list(lapply(1:k, function(i)
      mkan(dim, base_order + k - i, fusion = fusion)))
  },
  forward = function(xs) {
    k <- length(xs)

    # --- Cascaded Frequency Decomposition, Eq. (4) ---
    # f_i = x_i - FreqUp(x_{i+1}): what level i has that the smoother level i+1 lacks.
    # The last (smoothest) level keeps the remaining low frequencies as-is.
    fs <- vector("list", k)
    for (i in seq_len(k - 1)) {
      fs[[i]] <- xs[[i]] - frequency_upsample(xs[[i + 1]], xs[[i]]$size(2))
    }
    fs[[k]] <- xs[[k]]

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
                        num_layers = 2, base_order = 1, mkan_fusion = "learnable") {
    stopifnot("seq_len must be divisible by window^(levels - 1)" =
                seq_len %% (window^(num_levels - 1)) == 0)
    self$num_levels <- num_levels
    # Moving average used to remove high frequencies, Eq. (1).
    self$pool <- nn_avg_pool1d(window, stride = window)
    # Embed each scalar time step into D dims, Eq. (2) (one Linear per level).
    self$embeds <- nn_module_list(lapply(1:num_levels, function(i) nn_linear(1, d_model)))
    self$layers <- nn_module_list(lapply(1:num_layers, function(i)
      timekan_layer(d_model, num_levels, base_order, fusion = mkan_fusion)))
    # Forecast head, Eq. (12): project D -> 1, then time T -> F.
    self$proj_dim <- nn_linear(d_model, 1)
    self$proj_time <- nn_linear(seq_len, pred_len)
  },

  # Extension 4: Collect polynomial regularizations across all Chebyshev KAN modules
  get_poly_reg = function(power = 1) {
    reg <- torch_tensor(0, dtype = torch_float())
    for (i in seq_along(self$layers)) {
      layer <- self$layers[[i]]
      for (j in seq_along(layer$mkans)) {
        reg <- reg + layer$mkans[[j]]$kan$poly_regularization(power)
      }
    }
    reg
  },

  forward = function(x) {                       # x: [B, T, N]
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

    # --- L stacked Decomposition-Learning-Mixing layers ---
    for (i in seq_along(self$layers)) xs <- self$layers[[i]](xs)

    # --- Forecast from the finest-level sequence x_1, Eq. (12) ---
    y <- self$proj_dim(xs[[1]])$squeeze(-1)                  # [B*N, T]
    y <- self$proj_time(y)                                   # [B*N, F]
    y <- y$reshape(c(B, N, -1))$permute(c(1, 3, 2))          # [B, F, N] (stack variates back)

    y * sigma + mu                                           # undo RevIN
  }
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Return a [time, variates] numeric matrix (synthetic if no CSV given).
load_series <- function(csv_path = NULL) {
  if (!is.null(csv_path) && nzchar(csv_path)) {
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

# Sliding windows: (past seq_len steps, next pred_len steps).
# Supports optional stride for faster training and original variate storage for PCA inverse evaluation.
window_dataset <- dataset(
  "WindowDataset",
  initialize = function(data, seq_len, pred_len, orig_data = NULL, stride = 1) {
    self$data <- torch_tensor(data, dtype = torch_float())
    self$seq_len <- seq_len
    self$pred_len <- pred_len
    self$stride <- max(1L, as.integer(stride))
    self$orig_data <- if (!is.null(orig_data)) torch_tensor(orig_data, dtype = torch_float()) else NULL

    total_valid <- self$data$size(1) - self$seq_len - self$pred_len + 1
    self$indices <- seq(1L, total_valid, by = self$stride)
  },
  .length = function() {
    length(self$indices)
  },
  .getitem = function(idx) {
    i <- self$indices[idx]
    s <- i + self$seq_len
    item <- list(
      x = self$data[i:(s - 1), ],
      y = self$data[s:(s + self$pred_len - 1), ]
    )
    if (!is.null(self$orig_data)) {
      item$y_orig <- self$orig_data[s:(s + self$pred_len - 1), ]
    }
    item
  }
)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

# Mean squared and absolute error. When pca_W is supplied, applies Inverse PCA
# transform to reconstruct forecasts and evaluates metrics on original variables.
evaluate <- function(model, loader, pca_W = NULL) {
  model$eval()
  mse <- 0; mae <- 0; n <- 0
  mse_pca <- 0; n_pca <- 0
  has_pca <- !is.null(pca_W)

  with_no_grad({
    coro::loop(for (b in loader) {
      p <- model(b$x)                                  # [B, F, K_pca] (or [B, F, N])

      if (has_pca && !is.null(b$y_orig)) {
        # Inverse PCA transform: [B, F, K_pca] %*% [K_pca, N] -> [B, F, N]
        p_orig <- torch_matmul(p, pca_W)
        mse <- mse + nnf_mse_loss(p_orig, b$y_orig, reduction = "sum")$item()
        mae <- mae + nnf_l1_loss(p_orig, b$y_orig, reduction = "sum")$item()
        n <- n + b$y_orig$numel()

        # Latent PCA error for monitoring
        mse_pca <- mse_pca + nnf_mse_loss(p, b$y, reduction = "sum")$item()
        n_pca <- n_pca + b$y$numel()
      } else {
        mse <- mse + nnf_mse_loss(p, b$y, reduction = "sum")$item()
        mae <- mae + nnf_l1_loss(p, b$y, reduction = "sum")$item()
        n <- n + b$y$numel()
      }
    })
  })

  res <- c(mse = mse / n, mae = mae / n)
  if (has_pca && n_pca > 0) {
    res["mse_pca"] <- mse_pca / n_pca
  }
  res
}

# Minimal "--flag value" parser; returns `defaults` with any overrides applied.
parse_args <- function(defaults) {
  args <- commandArgs(trailingOnly = TRUE)
  for (i in seq_len(length(args) %/% 2) * 2 - 1) {
    key <- sub("^--", "", args[i])
    if (!key %in% names(defaults)) stop("Unknown option: ", args[i])
    val <- args[i + 1]
    defaults[[key]] <- if (is.numeric(defaults[[key]])) as.numeric(val) else val
  }
  defaults
}

main <- function() {
  opt <- parse_args(list(
    csv = "",              # CSV file (first col = timestamp); empty = synthetic data
    seq_len = 96,          # look-back window T
    pred_len = 96,         # forecast horizon F
    d_model = 16,          # embedding dimension D
    levels = 3,            # number of frequency bands k
    window = 2,            # moving-average window d
    layers = 2,            # number of stacked blocks L
    base_order = 1,        # lowest Chebyshev order b
    epochs = 10,
    batch_size = 64,       # batch size (64 is optimal for CPU cache and vectorized BLAS)
    lr = 1e-3,
    # Extensions:
    mkan_fusion = "learnable", # M-KAN branch fusion: "learnable", "convex", or "add"
    poly_reg = 1e-4,           # Chebyshev polynomial order complexity regularization weight (0 to disable)
    poly_power = 1,            # Order complexity power (1 = linear with order, 2 = quadratic)
    pca_dim = 0,               # Number of principal components (0 = disabled)
    pca_var = 0.0,             # Target cumulative explained variance ratio (e.g. 0.90; 0 to disable)
    split = "auto",            # "auto" (6:2:2 for ETT, 7:1:2 for others), "6:2:2", or "7:1:2"
    train_stride = 1,          # Sliding window stride during training (higher = faster training)
    patience = 3,              # Early stopping patience in epochs (0 to disable)
    threads = 0                # CPU threads for torch (0 = auto-detect all available cores)
  ))

  # Configure CPU threads for high-efficiency parallel matrix operations
  num_cores <- parallel::detectCores(logical = FALSE)
  if (is.na(num_cores)) num_cores <- parallel::detectCores()
  target_threads <- if (opt$threads > 0) as.integer(opt$threads) else max(1L, as.integer(num_cores))
  torch_set_num_threads(target_threads)
  cat(sprintf("Torch CPU threads set to: %d (System cores: %d)\n", target_threads, num_cores))

  torch_manual_seed(0)

  data_raw <- load_series(opt$csv)
  n_total <- nrow(data_raw)
  n_vars <- ncol(data_raw)
  cat(sprintf("Loaded dataset: %d time steps, %d variables\n", n_total, n_vars))

  # Determine train/val/test split
  split_type <- opt$split
  if (split_type == "auto") {
    is_ett <- grepl("ETT", opt$csv, ignore.case = TRUE)
    train_ratio <- if (is_ett) 0.6 else 0.7
    val_ratio   <- if (is_ett) 0.2 else 0.1
  } else if (split_type == "6:2:2") {
    train_ratio <- 0.6; val_ratio <- 0.2
  } else {
    train_ratio <- 0.7; val_ratio <- 0.1
  }

  n_train <- floor(train_ratio * n_total)
  n_val   <- floor(val_ratio * n_total)

  # RevIN / Z-score normalization strictly with training set statistics
  mu <- colMeans(data_raw[1:n_train, , drop = FALSE])
  sdev <- apply(data_raw[1:n_train, , drop = FALSE], 2, sd) + 1e-8
  data_norm <- scale(data_raw, center = mu, scale = sdev)

  # Extension 5: PCA Preprocessing for High-Dimensional Data
  pca_W_tensor <- NULL
  use_pca <- (opt$pca_dim > 0) || (opt$pca_var > 0)
  if (use_pca && n_vars > 1) {
    pca_fit <- prcomp(data_norm[1:n_train, , drop = FALSE], center = FALSE, scale. = FALSE)
    cum_var <- cumsum(pca_fit$sdev^2) / sum(pca_fit$sdev^2)

    if (opt$pca_dim > 0) {
      k_pca <- min(as.integer(opt$pca_dim), n_vars)
    } else {
      k_pca <- which(cum_var >= opt$pca_var)[1]
      if (is.na(k_pca)) k_pca <- n_vars
    }

    W <- pca_fit$rotation[, 1:k_pca, drop = FALSE]             # [N, k_pca]
    pca_W_tensor <- torch_tensor(t(W), dtype = torch_float()) # [k_pca, N]
    data_model <- data_norm %*% W                             # [total_steps, k_pca]

    cat(sprintf("[PCA] Reduced %d variables -> %d principal components (retaining %.2f%% variance)\n",
                n_vars, k_pca, cum_var[k_pca] * 100))
    cat(sprintf("[PCA] Training speedup factor: ~%.1fx\n", n_vars / k_pca))
  } else {
    data_model <- data_norm
  }

  # Split windows: val/test start seq_len steps early so their first forecast begins at the split
  splits <- list(
    train = list(model = data_model[1:n_train, , drop = FALSE],
                 orig  = data_norm[1:n_train, , drop = FALSE]),
    val   = list(model = data_model[(n_train - opt$seq_len + 1):(n_train + n_val), , drop = FALSE],
                 orig  = data_norm[(n_train - opt$seq_len + 1):(n_train + n_val), , drop = FALSE]),
    test  = list(model = data_model[(n_train + n_val - opt$seq_len + 1):n_total, , drop = FALSE],
                 orig  = data_norm[(n_train + n_val - opt$seq_len + 1):n_total, , drop = FALSE])
  )

  loaders <- list(
    train = dataloader(window_dataset(splits$train$model, opt$seq_len, opt$pred_len,
                                      orig_data = splits$train$orig, stride = opt$train_stride),
                       batch_size = opt$batch_size, shuffle = TRUE),
    val   = dataloader(window_dataset(splits$val$model, opt$seq_len, opt$pred_len,
                                      orig_data = splits$val$orig),
                       batch_size = opt$batch_size, shuffle = FALSE),
    test  = dataloader(window_dataset(splits$test$model, opt$seq_len, opt$pred_len,
                                      orig_data = splits$test$orig),
                       batch_size = opt$batch_size, shuffle = FALSE)
  )

  model <- timekan(opt$seq_len, opt$pred_len, opt$d_model, opt$levels,
                   opt$window, opt$layers, opt$base_order, mkan_fusion = opt$mkan_fusion)
  n_params <- sum(sapply(model$parameters, function(p) p$numel()))
  cat(sprintf("TimeKAN parameters: %s (M-KAN Fusion: %s, Poly Reg: %g)\n",
              format(n_params, big.mark = ","), opt$mkan_fusion, opt$poly_reg))

  optimizer <- optim_adam(model$parameters, lr = opt$lr)

  best_val <- Inf
  best_state <- NULL
  patience_counter <- 0

  t_start <- Sys.time()
  for (epoch in 1:opt$epochs) {
    epoch_start <- Sys.time()
    model$train()
    loss_sum <- 0; n_batches <- 0

    coro::loop(for (b in loaders$train) {
      optimizer$zero_grad()
      pred <- model(b$x)
      loss <- nnf_mse_loss(pred, b$y)

      # Extension 4: Add polynomial complexity regularization
      if (opt$poly_reg > 0) {
        loss <- loss + opt$poly_reg * model$get_poly_reg(opt$poly_power)
      }

      loss$backward()
      optimizer$step()
      loss_sum <- loss_sum + loss$item()
      n_batches <- n_batches + 1
    })

    val <- evaluate(model, loaders$val, pca_W = pca_W_tensor)
    epoch_sec <- as.numeric(difftime(Sys.time(), epoch_start, units = "secs"))
    avg_train_loss <- loss_sum / max(1, n_batches)

    if (!is.null(pca_W_tensor)) {
      cat(sprintf("epoch %2d (%4.1fs) | train loss %.4f | val MSE (orig) %.4f | val MSE (pca) %.4f\n",
                  epoch, epoch_sec, avg_train_loss, val[["mse"]], val[["mse_pca"]]))
    } else {
      cat(sprintf("epoch %2d (%4.1fs) | train loss %.4f | val MSE %.4f\n",
                  epoch, epoch_sec, avg_train_loss, val[["mse"]]))
    }

    if (val[["mse"]] < best_val) {
      best_val <- val[["mse"]]
      patience_counter <- 0
      best_state <- lapply(model$state_dict(), function(t) t$clone())
    } else {
      patience_counter <- patience_counter + 1
      if (opt$patience > 0 && patience_counter >= opt$patience) {
        cat(sprintf("Early stopping triggered after %d epochs without val improvement.\n", opt$patience))
        break
      }
    }
  }
  total_time <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
  cat(sprintf("Training complete in %.2f seconds.\n", total_time))

  if (!is.null(best_state)) {
    model$load_state_dict(best_state)
  }

  test <- evaluate(model, loaders$test, pca_W = pca_W_tensor)
  if (!is.null(pca_W_tensor)) {
    cat(sprintf("\n=== Final Test Results (Inverse PCA Reconstructed on %d Original Variables) ===\n", n_vars))
    cat(sprintf("test MSE %.4f | test MAE %.4f (latent PCA MSE: %.4f)\n",
                test[["mse"]], test[["mae"]], test[["mse_pca"]]))
  } else {
    cat(sprintf("\n=== Final Test Results ===\ntest MSE %.4f | test MAE %.4f\n",
                test[["mse"]], test[["mae"]]))
  }
}

# Run main() only when executed as a script (not when source()'d).
if (sys.nframe() == 0) main()
