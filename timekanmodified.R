#!/usr/bin/env Rscript
# =============================================================================
# TimeKAN (Huang et al., ICLR 2025) in R torch  --  CPU-friendly extended version
# =============================================================================
#
# Original method (variant "base"):
#   Decomposition-Learning-Mixing: moving-average pyramid -> cascaded frequency
#   decomposition (FFT upsampling) -> multi-order Chebyshev-KAN + depthwise conv
#   per frequency band -> frequency mixing -> linear head.
#
# Extensions added on top (each is a "variant token", combine with "+"):
#   ab    learnable scalars alpha, beta:  alpha * DWConv + beta * KAN   (init 1, 1)
#   reg   penalty  lambda * mean(n * theta_n^2)  on Chebyshev coefficients
#         (higher polynomial order = higher cost); uses the *effective* coefficients
#         beta * theta when `ab` is on, so beta cannot dodge the penalty
#   pca   fit PCA on the training split, forecast the r principal-component series,
#         map forecasts back with the inverse PCA transform, score on ORIGINAL
#         variables (big speed-up for wide data such as Electricity)
#   hi    raise every Chebyshev order by 2 (use with `reg` to test whether the
#         penalty lets high orders help instead of hurt)
# Reference models (for context): naive (repeat last value), snaive (seasonal
#   naive), linear (RevIN + one shared linear layer, DLinear-like).
# Examples of variant strings:  base | ab | reg | ab+reg | ab+reg+pca | hi+reg
#
# Other features:
#   * fast CPU training: windows are indexed directly (no dataloader), one matmul
#     per Chebyshev layer, random fraction of windows per epoch, early stopping,
#     configurable threads, resumable benchmark (finished runs are skipped)
#   * paper-style ETT protocol: 6:2:2 split on the first 14,400 (ETTh) / 57,600
#     (ETTm) rows; 7:1:2 for everything else
#   * univariate runs (--target OT), generic CSVs with missing-value imputation
#   * plots (base R graphics, PNG): training curves, old-vs-new comparison bars,
#     % improvement, error vs horizon, forecast examples, error by forecast step,
#     per-channel scatter, predicted-vs-actual, residuals, alpha/beta, PCA, speed
#
# Quick start (run from the folder that contains this file):
#   Rscript timekan.R --mode selftest
#   Rscript timekan.R --csv data/ETTh1.csv --pred_len 96
#   Rscript timekan.R --csv data/ETTh1.csv,data/ETTh2.csv --pred_len 96,336 --seeds 0,1,2
#   Rscript timekan.R --csv data/electricity.csv --variants base,ab+reg+pca --train_frac 0.5
#   Rscript timekan.R --csv data/ETTh1.csv --target OT --variants snaive,linear,base,ab+reg
#   Rscript timekan.R --mode plots --out_dir results      # (re)draw plots only
#   Rscript timekan.R --help
# =============================================================================


USAGE <- "
TimeKAN (R torch) - options are given as '--name value' pairs

  --mode        benchmark (default) | plots | selftest
  --csv         one or more CSV paths, comma separated (empty = synthetic data)
  --target      keep only this column (univariate run), e.g. OT
  --drop_cols   columns to ignore, comma separated (e.g. Year,Month)
  --max_rows    -1 = auto (ETT protocol), 0 = use all rows, N = first N rows
  --split       auto | 721 | 622   (train/val/test in tenths)
  --variants    comma list, default snaive,linear,base,ab+reg
                tokens: naive snaive linear base ab reg pca hi  (join with +)
  --pred_len    forecast horizon(s), comma separated, default 96
  --seeds       comma separated, default 0
  --seq_len 96  --d_model 16  --levels 3  --window 2  --layers 2  --base_order 1
  --lambda 0.01 order-penalty weight (variants with 'reg')
  --pca_var 0.95  --pca_r 0   PCA: variance target, or fixed number of components
  --season 0    seasonal-naive period (0 = auto from file name)
  --epochs 12  --patience 3  --batch_size 32  --lr 0.001
  --train_frac 1    fraction of training windows used per epoch (0.5 is faster)
  --val_stride 1    use every k-th validation window
  --threads 0       torch CPU threads (0 = auto)
  --out_dir results  --plots 1  --force 0 (1 = rerun finished runs)  --quiet 0
"

DEFAULTS <- list(
  mode = "benchmark",
  csv = "", target = "", drop_cols = "", max_rows = -1, split = "auto",
  variants = "snaive,linear,base,ab+reg", pred_len = "96", seeds = "0",
  seq_len = 96, d_model = 16, levels = 3, window = 2, layers = 2, base_order = 1,
  lambda = 0.01, pca_var = 0.95, pca_r = 0, season = 0,
  epochs = 12, patience = 3, batch_size = 32, lr = 1e-3,
  train_frac = 1, val_stride = 1, eval_batch = 256,
  threads = 0, out_dir = "results", plots = 1, force = 0, quiet = 0
)

parse_args <- function(defaults, args = commandArgs(trailingOnly = TRUE)) {
  if (any(args %in% c("--help", "-h"))) { cat(USAGE); quit(save = "no", status = 0) }
  if (length(args) %% 2 != 0) stop("Options must be given as '--name value' pairs (see --help).")
  for (i in seq_len(length(args) %/% 2) * 2 - 1) {
    key <- sub("^--", "", args[i])
    if (!key %in% names(defaults)) stop("Unknown option: ", args[i], "  (see --help)")
    val <- args[i + 1]
    defaults[[key]] <- if (is.numeric(defaults[[key]])) as.numeric(val) else val
  }
  defaults
}

split_csv <- function(x) { x <- trimws(strsplit(x, ",", fixed = TRUE)[[1]]); x[nzchar(x)] }

ALLOWED_TOKENS <- c("base", "naive", "snaive", "linear", "ab", "reg", "pca", "hi")

parse_variant <- function(v) {
  tok <- strsplit(tolower(trimws(v)), "+", fixed = TRUE)[[1]]
  bad <- setdiff(tok, ALLOWED_TOKENS)
  if (length(bad)) stop("Unknown variant token(s): ", paste(bad, collapse = ", "),
                        "  (allowed: ", paste(ALLOWED_TOKENS, collapse = " "), ")")
  fam <- if (tok[1] %in% c("naive", "snaive", "linear")) tok[1] else "timekan"
  list(name = paste(tok, collapse = "+"), family = fam,
       ab  = "ab"  %in% tok && fam == "timekan",
       reg = "reg" %in% tok && fam == "timekan",
       hi  = "hi"  %in% tok && fam == "timekan",
       pca = "pca" %in% tok && !(fam %in% c("naive", "snaive")))
}

run_tag <- function(key, variant, pred_len, seed) {
  paste(key, gsub("[^A-Za-z0-9]+", "-", variant), paste0("H", pred_len), paste0("s", seed), sep = "__")
}

# Numbers reported in the TimeKAN paper (Table 1, look-back 96), for reference markers.
PAPER <- list(
  etth1       = list(mse = c(.367, .414, .445, .444), mae = c(.395, .420, .434, .459)),
  etth2       = list(mse = c(.290, .375, .423, .443), mae = c(.340, .392, .435, .449)),
  ettm1       = list(mse = c(.322, .357, .382, .445), mae = c(.361, .383, .401, .435)),
  ettm2       = list(mse = c(.174, .239, .301, .395), mae = c(.255, .299, .340, .396)),
  weather     = list(mse = c(.162, .207, .263, .338), mae = c(.208, .249, .290, .340)),
  electricity = list(mse = c(.174, .182, .197, .236), mae = c(.266, .273, .286, .320))
)
paper_ref <- function(key, h, metric = c("mse", "mae")) {
  metric <- match.arg(metric)
  i <- match(h, c(96, 192, 336, 720))
  if (is.na(i) || is.null(PAPER[[key]])) return(NA_real_)
  PAPER[[key]][[metric]][i]
}


# =============================================================================
# 1. Data (pure R, no torch needed)
# =============================================================================

make_synthetic <- function() {
  set.seed(0)
  t <- 0:5999
  m <- sapply(c(0, 1, 2), function(p)
    sin(2 * pi * t / 200 + p) + 0.5 * sin(2 * pi * t / 24 + 2 * p) +
      0.2 * sin(2 * pi * t / 6) + 0.1 * rnorm(length(t)))
  colnames(m) <- paste0("V", 1:3)
  m
}

impute_col <- function(v) {
  bad <- !is.finite(v)
  if (!any(bad)) return(v)
  good <- which(!bad)
  if (length(good) == 0) return(v)
  if (length(good) == 1) { v[bad] <- v[good]; return(v) }
  approx(good, v[good], xout = seq_along(v), rule = 2)$y   # linear interpolation
}

# Shared by load_dataset() and the self-test: truncate, split, z-score with train stats.
finalize_dataset <- function(mat, key, base_key, max_rows = -1, split = "auto", quiet = FALSE) {
  is_ett <- grepl("^ett", base_key)
  cap <- if (max_rows > 0) max_rows else if (max_rows == 0) Inf else
    if (grepl("^etth", base_key)) 14400 else if (grepl("^ettm", base_key)) 57600 else Inf
  if (nrow(mat) > cap) {
    if (!quiet) cat(sprintf("[data] using the first %d of %d rows (paper protocol)\n", cap, nrow(mat)))
    mat <- mat[seq_len(cap), , drop = FALSE]
  }
  if (split == "auto") split <- if (is_ett) "622" else "721"
  if (!grepl("^[0-9]{3}$", split)) stop("--split must be 'auto' or three digits such as 721 or 622")
  r <- as.numeric(strsplit(split, "")[[1]]) / 10
  if (abs(sum(r) - 1) > 1e-9) stop("--split digits must sum to 10")
  n <- nrow(mat)
  n_train <- floor(r[1] * n); n_val <- floor(r[2] * n)
  mu <- colMeans(mat[1:n_train, , drop = FALSE])
  sdev <- apply(mat[1:n_train, , drop = FALSE], 2, sd) + 1e-8
  z <- sweep(sweep(mat, 2, mu, "-"), 2, sdev, "/")
  if (!quiet) cat(sprintf("[data] %s: %d rows x %d variates, split %s -> train %d / val %d / test %d\n",
                          key, n, ncol(z), split, n_train, n_val, n - n_train - n_val))
  list(key = key, base_key = base_key, z = z, n = n, n_train = n_train, n_val = n_val,
       cols = colnames(mat), split = split)
}

load_dataset <- function(csv, opt) {
  if (!nzchar(csv)) {
    cat("[data] no --csv given: using the synthetic multi-frequency series\n")
    return(finalize_dataset(make_synthetic(), "synthetic", "synthetic", opt$max_rows, opt$split))
  }
  if (!file.exists(csv)) stop("File not found: ", csv)
  df <- read.csv(csv, check.names = FALSE, stringsAsFactors = FALSE)
  base_key <- tolower(tools::file_path_sans_ext(basename(csv)))
  key <- base_key
  drop <- split_csv(opt$drop_cols)
  if (length(drop)) df <- df[, setdiff(names(df), drop), drop = FALSE]
  # Only numeric columns are variates; date / text columns are ignored.
  is_num <- vapply(df, is.numeric, logical(1))
  if (!any(is_num)) stop("No numeric columns found in ", csv)
  if (any(!is_num)) cat(sprintf("[data] ignoring non-numeric column(s): %s\n", paste(names(df)[!is_num], collapse = ", ")))
  mat <- as.matrix(df[, is_num, drop = FALSE])
  storage.mode(mat) <- "double"
  
  if (nzchar(opt$target)) {
    if (!opt$target %in% colnames(mat))
      stop("Target column '", opt$target, "' not found. Columns: ", paste(colnames(mat), collapse = ", "))
    mat <- mat[, opt$target, drop = FALSE]
    key <- paste0(base_key, "_", tolower(opt$target))
  }
  
  n_bad <- sum(!is.finite(mat))
  if (n_bad > 0) {
    cat(sprintf("[data] %d missing/non-finite values (%.2f%%) -> linear interpolation\n", n_bad, 100 * n_bad / length(mat)))
    all_bad <- colSums(is.finite(mat)) == 0
    if (any(all_bad)) { cat("[data] dropping all-missing column(s): ", paste(colnames(mat)[all_bad], collapse = ", "), "\n"); mat <- mat[, !all_bad, drop = FALSE] }
    mat <- apply(mat, 2, impute_col)
    if (is.null(dim(mat))) mat <- matrix(mat, ncol = 1)
  }
  if (ncol(mat) > 1) {
    const <- apply(mat, 2, function(v) sd(v) < 1e-12)
    if (any(const) && sum(!const) >= 1) {
      cat("[data] dropping constant column(s): ", paste(colnames(mat)[const], collapse = ", "), "\n")
      mat <- mat[, !const, drop = FALSE]
    }
  }
  shown <- head(colnames(mat), 8)
  cat(sprintf("[data] variates (%d): %s%s\n", ncol(mat), paste(shown, collapse = ", "), if (ncol(mat) > 8) ", ..." else ""))
  finalize_dataset(mat, key, base_key, opt$max_rows, opt$split)
}

fit_pca <- function(train_z, var_target = 0.95, r_fixed = 0) {
  p <- prcomp(train_z, center = TRUE, scale. = FALSE)
  ev <- p$sdev^2
  cumvar <- cumsum(ev) / sum(ev)
  r <- if (r_fixed > 0) r_fixed else which(cumvar >= var_target)[1]
  r <- max(1, min(r, ncol(train_z)))
  list(W = p$rotation[, 1:r, drop = FALSE], center = p$center, r = r, cumvar = cumvar)
}
pca_project <- function(p, X) sweep(X, 2, p$center, "-") %*% p$W
pca_inverse <- function(p, Z) sweep(Z %*% t(p$W), 2, p$center, "+")

default_season <- function(base_key) {
  if (grepl("^ettm", base_key)) 96 else if (grepl("^weather", base_key)) 144 else 24
}


# =============================================================================
# 2. Models (need torch; defined inside a function so plotting works without it)
# =============================================================================

load_torch <- function(opt) {
  if (!requireNamespace("torch", quietly = TRUE))
    stop("The 'torch' package is not installed. In R run: install.packages('torch'); library(torch)")
  suppressPackageStartupMessages(library(torch))
  if (!torch::torch_is_installed())
    stop("libtorch is missing. In R run: torch::install_torch()")
  n <- if (opt$threads > 0) opt$threads else max(1, min(4, parallel::detectCores() - 1, na.rm = TRUE))
  tryCatch(torch_set_num_threads(n), error = function(e) NULL)
  cat(sprintf("[torch] using %d CPU thread(s)\n", n))
}

define_models <- function() {
  
  # Frequency Upsampling, Eq. (3): IFFT(ZeroPad(FFT(x))).  x: [B, L_short, D]
  frequency_upsample <- function(x, target_len) {
    short_len <- x$size(2)
    spec <- torch_fft_rfft(x, dim = 2)
    out <- torch_fft_irfft(spec, n = target_len, dim = 2)   # larger n = zero padding
    out * (target_len / short_len)
  }
  
  # 1-layer Chebyshev KAN over channels, Eqs. (6)-(8). One matmul over the stacked
  # polynomial features (BLAS-friendly, much faster on CPU than an einsum).
  cheby_kan <- nn_module(
    "ChebyKAN",
    initialize = function(dim, order) {
      self$order <- order
      self$coeffs <- nn_parameter(torch_randn(c((order + 1) * dim, dim)) / (dim * (order + 1)))
      # weight n for every order-n coefficient (order 0 and 1 are free)
      self$register_buffer("ord_w", torch_tensor(rep(0:order, each = dim), dtype = torch_float())$unsqueeze(2))
    },
    forward = function(x) {                         # x: [..., D]
      x <- torch_tanh(x)
      polys <- list(torch_ones_like(x), x)          # T0, T1; recurrence T_{n+1} = 2x T_n - T_{n-1}
      if (self$order >= 2) {
        for (n in 2:self$order) {
          k <- length(polys)
          polys[[k + 1]] <- 2 * x * polys[[k]] - polys[[k - 1]]
        }
      }
      feats <- torch_cat(polys[1:(self$order + 1)], dim = -1)   # [..., (order+1)*D]
      torch_matmul(feats, self$coeffs)                          # [..., D]
    },
    penalty = function() (self$ord_w * self$coeffs^2)$mean()
  )
  
  # M-KAN block: depthwise conv (time) + Chebyshev KAN (channels), Eqs. (5), (9), (10).
  mkan <- nn_module(
    "MKAN",
    initialize = function(dim, order, kernel_size = 3, learnable_mix = FALSE) {
      self$dwconv <- nn_conv1d(dim, dim, kernel_size, padding = kernel_size %/% 2, groups = dim)
      self$kan <- cheby_kan(dim, order)
      self$learnable_mix <- learnable_mix
      if (learnable_mix) {
        self$alpha <- nn_parameter(torch_ones(1))
        self$beta  <- nn_parameter(torch_ones(1))
      }
    },
    forward = function(f) {                         # f: [B, L, D]
      f1 <- self$dwconv(f$transpose(2, 3))$transpose(2, 3)
      f2 <- self$kan(f)
      if (self$learnable_mix) self$alpha * f1 + self$beta * f2 else f1 + f2
    },
    penalty = function() {
      p <- self$kan$penalty()
      if (self$learnable_mix) p <- self$beta$pow(2)$sum() * p   # penalise beta * theta
      p
    }
  )
  
  timekan_layer <- nn_module(
    "TimeKANLayer",
    initialize = function(dim, num_levels, base_order, learnable_mix) {
      k <- num_levels
      self$mkans <- nn_module_list(lapply(1:k, function(i)
        mkan(dim, base_order + k - i, learnable_mix = learnable_mix)))
    },
    forward = function(xs) {
      k <- length(xs)
      fs <- vector("list", k)                       # Cascaded Frequency Decomposition, Eq. (4)
      for (i in seq_len(k - 1)) fs[[i]] <- xs[[i]] - frequency_upsample(xs[[i + 1]], xs[[i]]$size(2))
      fs[[k]] <- xs[[k]]
      for (i in 1:k) fs[[i]] <- self$mkans[[i]](fs[[i]])   # per-band M-KAN
      out <- vector("list", k)                      # Frequency Mixing, Eq. (11)
      out[[k]] <- fs[[k]]
      for (i in rev(seq_len(k - 1))) out[[i]] <- frequency_upsample(out[[i + 1]], fs[[i]]$size(2)) + fs[[i]]
      out
    }
  )
  
  timekan <- nn_module(
    "TimeKAN",
    initialize = function(seq_len, pred_len, d_model = 16, num_levels = 3, window = 2,
                          num_layers = 2, base_order = 1, learnable_mix = FALSE) {
      if (seq_len %% (window^(num_levels - 1)) != 0)
        stop("seq_len (", seq_len, ") must be divisible by window^(levels-1) = ", window^(num_levels - 1))
      self$num_levels <- num_levels
      self$num_layers <- num_layers
      self$pool <- nn_avg_pool1d(window, stride = window)
      self$embeds <- nn_module_list(lapply(1:num_levels, function(i) nn_linear(1, d_model)))
      self$layers <- nn_module_list(lapply(1:num_layers, function(i)
        timekan_layer(d_model, num_levels, base_order, learnable_mix)))
      self$proj_dim <- nn_linear(d_model, 1)
      self$proj_time <- nn_linear(seq_len, pred_len)
    },
    forward = function(x) {                         # x: [B, T, N]
      B <- x$size(1); Tn <- x$size(2); N <- x$size(3)
      mu <- x$mean(dim = 2, keepdim = TRUE)$detach()                       # RevIN
      sigma <- torch_sqrt(x$var(dim = 2, keepdim = TRUE, unbiased = FALSE) + 1e-5)$detach()
      x <- (x - mu) / sigma
      x <- x$permute(c(1, 3, 2))$reshape(c(B * N, 1, Tn))                  # variate independence
      levels <- list(x)
      for (i in seq_len(self$num_levels - 1)) levels[[i + 1]] <- self$pool(levels[[i]])
      xs <- lapply(seq_along(levels), function(i) self$embeds[[i]](levels[[i]]$transpose(2, 3)))
      for (i in seq_len(self$num_layers)) xs <- self$layers[[i]](xs)
      y <- self$proj_dim(xs[[1]])$squeeze(-1)                              # [B*N, T]
      y <- self$proj_time(y)                                               # [B*N, F]
      y <- y$reshape(c(B, N, -1))$permute(c(1, 3, 2))                      # [B, F, N]
      y * sigma + mu
    },
    penalty = function() {
      total <- torch_zeros(1)
      for (l in seq_len(self$num_layers))
        for (i in seq_len(self$num_levels))
          total <- total + self$layers[[l]]$mkans[[i]]$penalty()
      total$sum()
    }
  )
  
  # DLinear-like reference: RevIN + one linear map T -> F shared by all variates.
  linear_model <- nn_module(
    "LinearRef",
    initialize = function(seq_len, pred_len) { self$fc <- nn_linear(seq_len, pred_len) },
    forward = function(x) {
      mu <- x$mean(dim = 2, keepdim = TRUE)$detach()
      sigma <- torch_sqrt(x$var(dim = 2, keepdim = TRUE, unbiased = FALSE) + 1e-5)$detach()
      y <- self$fc(((x - mu) / sigma)$permute(c(1, 3, 2)))$permute(c(1, 3, 2))
      y * sigma + mu
    }
  )
  
  # No-parameter references: season = 0 -> repeat last value, season > 0 -> seasonal naive.
  naive_model <- nn_module(
    "NaiveRef",
    initialize = function(seq_len, pred_len, season = 0) {
      self$seq_len <- seq_len; self$pred_len <- pred_len
      self$season <- if (season > 0) min(season, seq_len) else 0
    },
    forward = function(x) {
      if (self$season == 0) {
        last <- x[, self$seq_len:self$seq_len, , drop = FALSE]
        return(torch_cat(rep(list(last), self$pred_len), dim = 2))
      }
      s <- self$season
      last <- x[, (self$seq_len - s + 1):self$seq_len, , drop = FALSE]
      out <- torch_cat(rep(list(last), ceiling(self$pred_len / s)), dim = 2)
      out[, 1:self$pred_len, , drop = FALSE]
    }
  )
  
  list(timekan = timekan, linear = linear_model, naive = naive_model)
}

build_model <- function(M, vc, opt, pred_len, season) {
  switch(vc$family,
         timekan = M$timekan(opt$seq_len, pred_len, opt$d_model, opt$levels, opt$window, opt$layers,
                             opt$base_order + if (vc$hi) 2 else 0, vc$ab),
         linear  = M$linear(opt$seq_len, pred_len),
         naive   = M$naive(opt$seq_len, pred_len, 0),
         snaive  = M$naive(opt$seq_len, pred_len, season))
}


# =============================================================================
# 3. Windows, evaluation, training
# =============================================================================

# Rows starts[b] + offset + (0 .. len-1) of X [time, N]  ->  tensor [B, len, N]
slice_windows <- function(X, starts, len, offset = 0L) {
  rows <- as.integer(rep(starts + offset, times = len) + rep(0:(len - 1L), each = length(starts)))
  X[rows, , drop = FALSE]$reshape(c(len, length(starts), X$size(2)))$permute(c(2, 1, 3))
}
get_xy <- function(Xin, Xout, starts, l_in, l_out)
  list(x = slice_windows(Xin, starts, l_in, 0L), y = slice_windows(Xout, starts, l_out, l_in))

# Errors are always measured in the ORIGINAL (z-scored) variable space: PCA models
# decode their forecasts first.
evaluate_model <- function(model, Xin, Xout, starts, l_in, l_out, bs, decode = NULL, detail = FALSE) {
  model$eval()
  sse <- 0; sae <- 0; cnt <- 0; step_sum <- NULL; chan_sum <- NULL
  with_no_grad({
    for (s in seq(1, length(starts), by = bs)) {
      idx <- starts[s:min(s + bs - 1, length(starts))]
      b <- get_xy(Xin, Xout, idx, l_in, l_out)
      p <- model(b$x)
      if (!is.null(decode)) p <- decode(p)
      d <- p - b$y
      sq <- d^2
      sse <- sse + sq$sum()$item()
      sae <- sae + d$abs()$sum()$item()
      cnt <- cnt + b$y$numel()
      if (detail) {
        ss <- as.numeric(as.array(sq$sum(dim = c(1, 3))))   # per forecast step
        cs <- as.numeric(as.array(sq$sum(dim = c(1, 2))))   # per variate
        step_sum <- if (is.null(step_sum)) ss else step_sum + ss
        chan_sum <- if (is.null(chan_sum)) cs else chan_sum + cs
      }
    }
  })
  out <- list(mse = sse / cnt, mae = sae / cnt)
  if (detail) {
    nwin <- length(starts)
    out$step_mse <- step_sum / (nwin * length(chan_sum))
    out$chan_mse <- chan_sum / (nwin * length(step_sum))
  }
  out
}

# A few test windows (history, truth, forecast) for the forecast / scatter plots.
collect_samples <- function(model, Xin, Xorig, starts, l_in, l_out, decode, n_win = 32, max_ch = 8) {
  model$eval()
  idx <- unique(pmax(1, round(seq(1, length(starts), length.out = min(n_win, length(starts))))))
  st <- starts[idx]
  with_no_grad({
    p <- model(slice_windows(Xin, st, l_in, 0L))
    if (!is.null(decode)) p <- decode(p)
  })
  x <- as.array(slice_windows(Xorig, st, l_in, 0L))
  y <- as.array(slice_windows(Xorig, st, l_out, l_in))
  p <- as.array(p)
  N <- dim(x)[3]
  ch <- unique(round(seq(1, N, length.out = min(N, max_ch))))
  list(x = x[, , ch, drop = FALSE], y = y[, , ch, drop = FALSE], p = p[, , ch, drop = FALSE], ch = ch)
}

assemble_result <- function(key, variant, family, pred_len, seq_len, seed, n_channels, n_model_channels,
                            params, hist, best_epoch, train_sec, val_mse, test_mse, test_mae,
                            pca_floor, lambda, d_model, detail) {
  epochs_run <- if (is.null(hist)) 0L else nrow(hist)
  m <- data.frame(key = key, variant = variant, family = family, pred_len = pred_len, seq_len = seq_len,
                  seed = seed, n_channels = n_channels, n_model_channels = n_model_channels, params = params,
                  epochs_run = epochs_run, best_epoch = best_epoch, train_sec = train_sec,
                  sec_per_epoch = if (epochs_run > 0) train_sec / epochs_run else 0,
                  val_mse = val_mse, test_mse = test_mse, test_mae = test_mae,
                  pca_floor = pca_floor, lambda = lambda, d_model = d_model, stringsAsFactors = FALSE)
  h <- NULL
  if (!is.null(hist) && nrow(hist) > 0)
    h <- cbind(data.frame(key = key, variant = variant, pred_len = pred_len, seed = seed, stringsAsFactors = FALSE), hist)
  list(metrics = m, history = h, detail = detail)
}

train_one <- function(ds, vc, opt, seed, pred_len, M) {
  l_in <- opt$seq_len; l_out <- pred_len
  set.seed(seed); torch_manual_seed(seed)
  Z <- ds$z; N <- ncol(Z)
  rows_tr <- 1:ds$n_train
  rows_va <- (ds$n_train - l_in + 1):(ds$n_train + ds$n_val)
  rows_te <- (ds$n_train + ds$n_val - l_in + 1):ds$n
  nw <- function(rows) length(rows) - l_in - l_out + 1
  if (min(nw(rows_tr), nw(rows_va), nw(rows_te)) < 1)
    stop("Dataset too short for seq_len=", l_in, " and pred_len=", l_out, ". Try smaller values.")
  season <- if (opt$season > 0) opt$season else default_season(ds$base_key)
  season <- min(season, l_in)
  
  # ---- optional PCA (fit on the training rows only) ----
  pc <- NULL; decode <- NULL; floor_mse <- NA_real_
  Zp <- Z
  if (vc$pca && N >= 2) {
    pc <- fit_pca(Z[rows_tr, , drop = FALSE], opt$pca_var, opt$pca_r)
    Zp <- pca_project(pc, Z)
    Wt <- torch_tensor(t(pc$W), dtype = torch_float())
    ctr <- torch_tensor(as.numeric(pc$center), dtype = torch_float())
    decode <- function(p) torch_matmul(p, Wt) + ctr
    Xte_rows <- Z[rows_te, , drop = FALSE]
    floor_mse <- mean((Xte_rows - pca_inverse(pc, pca_project(pc, Xte_rows)))^2)
    cat(sprintf("  PCA: %d -> %d components (%.1f%% variance), reconstruction floor MSE %.4f\n",
                N, pc$r, 100 * pc$cumvar[pc$r], floor_mse))
  }
  n_in <- ncol(Zp)
  tens <- function(A, rows) torch_tensor(A[rows, , drop = FALSE], dtype = torch_float())
  Xin_tr <- tens(Zp, rows_tr)
  Xin_va <- tens(Zp, rows_va); Xo_va <- if (is.null(pc)) Xin_va else tens(Z, rows_va)
  Xin_te <- tens(Zp, rows_te); Xo_te <- if (is.null(pc)) Xin_te else tens(Z, rows_te)
  starts_tr <- seq_len(nw(rows_tr))
  starts_va <- seq(1, nw(rows_va), by = max(1, opt$val_stride))
  starts_te <- seq_len(nw(rows_te))
  
  model <- build_model(M, vc, opt, pred_len, season)
  params <- sum(vapply(model$parameters, function(p) p$numel(), numeric(1)))
  use_reg <- vc$reg && opt$lambda > 0
  cat(sprintf("[%s] %-12s H=%-3d seed=%d | channels %d%s | params %s%s\n", ds$key, vc$name, pred_len, seed, N,
              if (!is.null(pc)) sprintf(" -> %d (PCA)", n_in) else "",
              format(params, big.mark = ","), if (use_reg) sprintf(" | lambda %.3g", opt$lambda) else ""))
  
  hist <- NULL; best_epoch <- 0; best_val <- Inf; train_sec <- 0
  if (params > 0) {
    adam <- optim_adam(model$parameters, lr = opt$lr)
    best_state <- NULL; bad <- 0; rows <- list(); t0 <- Sys.time()
    for (ep in seq_len(opt$epochs)) {
      model$train()
      n_use <- min(length(starts_tr), max(opt$batch_size, floor(opt$train_frac * length(starts_tr))))
      perm <- sample.int(length(starts_tr), n_use)
      tr_sum <- 0; tr_n <- 0
      for (s in seq(1, length(perm), by = opt$batch_size)) {
        idx <- starts_tr[perm[s:min(s + opt$batch_size - 1, length(perm))]]
        b <- get_xy(Xin_tr, Xin_tr, idx, l_in, l_out)
        adam$zero_grad()
        mse <- nnf_mse_loss(model(b$x), b$y)
        loss <- if (use_reg) mse + opt$lambda * model$penalty() else mse
        loss$backward()
        adam$step()
        tr_sum <- tr_sum + mse$item() * length(idx); tr_n <- tr_n + length(idx)
      }
      va <- evaluate_model(model, Xin_va, Xo_va, starts_va, l_in, l_out, opt$eval_batch, decode)
      secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      rows[[ep]] <- data.frame(epoch = ep, train_mse = tr_sum / tr_n, val_mse = va$mse, sec = secs)
      improved <- va$mse < best_val - 1e-9
      if (!opt$quiet) cat(sprintf("  epoch %2d | train MSE %.4f | val MSE %.4f | %5.1fs%s\n",
                                  ep, tr_sum / tr_n, va$mse, secs, if (improved) "  *" else ""))
      if (improved) {
        best_val <- va$mse; best_epoch <- ep; bad <- 0
        best_state <- lapply(model$state_dict(), function(t) t$clone())
      } else {
        bad <- bad + 1
        if (bad >= opt$patience) { if (!opt$quiet) cat("  early stop\n"); break }
      }
    }
    train_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    model$load_state_dict(best_state)
    hist <- do.call(rbind, rows)
  } else {
    best_val <- evaluate_model(model, Xin_va, Xo_va, starts_va, l_in, l_out, opt$eval_batch, decode)$mse
  }
  
  te <- evaluate_model(model, Xin_te, Xo_te, starts_te, l_in, l_out, opt$eval_batch, decode, detail = TRUE)
  cat(sprintf("  => test MSE %.4f | test MAE %.4f%s\n", te$mse, te$mae,
              if (params > 0) sprintf(" | %.1fs total (%.1fs/epoch)", train_sec, train_sec / nrow(hist)) else ""))
  
  smp <- collect_samples(model, Xin_te, Xo_te, starts_te, l_in, l_out, decode)
  detail <- list(x = smp$x, y = smp$y, p = smp$p, ch = smp$ch, step_mse = te$step_mse, chan_mse = te$chan_mse,
                 cumvar = if (!is.null(pc)) pc$cumvar else NULL, pca_r = if (!is.null(pc)) pc$r else NULL,
                 pca_floor = floor_mse, ab = NULL)
  if (vc$family == "timekan" && vc$ab) {
    alpha <- beta <- matrix(NA_real_, opt$layers, opt$levels)
    for (l in seq_len(opt$layers)) for (i in seq_len(opt$levels)) {
      m <- model$layers[[l]]$mkans[[i]]
      alpha[l, i] <- m$alpha$item(); beta[l, i] <- m$beta$item()
    }
    detail$ab <- list(alpha = alpha, beta = beta)
  }
  assemble_result(ds$key, vc$name, vc$family, pred_len, l_in, seed, N, n_in, params, hist, best_epoch,
                  train_sec, best_val, te$mse, te$mae, floor_mse, if (use_reg) opt$lambda else 0,
                  opt$d_model, detail)
}

write_run_outputs <- function(res, out_dir) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  m <- res$metrics
  tag <- run_tag(m$key, m$variant, m$pred_len, m$seed)
  if (!is.null(res$history)) write.csv(res$history, file.path(out_dir, paste0("history__", tag, ".csv")), row.names = FALSE)
  saveRDS(res$detail, file.path(out_dir, paste0("detail__", tag, ".rds")))
  write.csv(m, file.path(out_dir, paste0("metrics__", tag, ".csv")), row.names = FALSE)  # written last = "run finished"
  invisible(tag)
}


# =============================================================================
# 4. Plots (base R graphics; no extra packages needed)
# =============================================================================

VARIANT_ORDER <- c("naive", "snaive", "linear", "base", "ab", "reg", "hi", "ab+reg", "hi+reg", "pca", "ab+reg+pca")
order_variants <- function(vs) {
  vs <- unique(vs); rk <- match(vs, VARIANT_ORDER)
  rk[is.na(rk)] <- 100 + seq_len(sum(is.na(rk)))
  vs[order(rk)]
}
variant_colors <- function(vs) {
  pal <- c(naive = "#9ecae1", snaive = "#bdbdbd", linear = "#636363", base = "#1f77b4", ab = "#2ca02c",
           reg = "#ff7f0e", hi = "#8c564b", `ab+reg` = "#d62728", `hi+reg` = "#e377c2", pca = "#9467bd",
           `ab+reg+pca` = "#17becf")
  out <- setNames(rep(NA_character_, length(vs)), vs)
  known <- vs %in% names(pal)
  out[known] <- unname(pal[vs[known]])
  if (any(!known)) out[!known] <- grDevices::hcl.colors(sum(!known), "Dark 3")
  out
}
png_open <- function(file, w, h) grDevices::png(file, width = w, height = h, units = "in", res = 110)
safe_plot <- function(name, expr) {
  tryCatch(expr, error = function(e) {
    message("  [plot skipped: ", name, "] ", conditionMessage(e))
    while (grDevices::dev.cur() > 1) grDevices::dev.off()
  })
}

read_metrics <- function(out_dir) {
  fs <- list.files(out_dir, pattern = "^metrics__.*\\.csv$", full.names = TRUE)
  if (!length(fs)) return(NULL)
  do.call(rbind, lapply(fs, read.csv, stringsAsFactors = FALSE))
}

summarise_results <- function(M) {
  grp <- split(M, list(M$key, M$variant, M$pred_len), drop = TRUE)
  S <- do.call(rbind, lapply(grp, function(g) data.frame(
    key = g$key[1], variant = g$variant[1], family = g$family[1], pred_len = g$pred_len[1], n_seeds = nrow(g),
    test_mse = mean(g$test_mse), mse_sd = if (nrow(g) > 1) sd(g$test_mse) else NA_real_,
    test_mae = mean(g$test_mae), mae_sd = if (nrow(g) > 1) sd(g$test_mae) else NA_real_,
    params = g$params[1], sec_per_epoch = mean(g$sec_per_epoch), best_epoch = mean(g$best_epoch),
    stringsAsFactors = FALSE)))
  rownames(S) <- NULL
  b <- S[S$variant == "base", ]
  S$mse_vs_base_pct <- 100 * (S$test_mse / b$test_mse[match(paste(S$key, S$pred_len), paste(b$key, b$pred_len))] - 1)
  S[order(S$key, S$pred_len, match(S$variant, order_variants(S$variant))), ]
}

# matrix [variants x keys] of column `col` from summary S at horizon h
sum_mat <- function(S, h, col, vs, keys) {
  m <- matrix(NA_real_, length(vs), length(keys), dimnames = list(vs, keys))
  for (v in vs) for (k in keys) {
    x <- S[[col]][S$variant == v & S$key == k & S$pred_len == h]
    if (length(x)) m[v, k] <- x[1]
  }
  m
}

plot_training_curves <- function(out_dir, file) {
  fs <- list.files(out_dir, pattern = "^history__.*\\.csv$", full.names = TRUE)
  if (!length(fs)) stop("no history files")
  H <- do.call(rbind, lapply(fs, read.csv, stringsAsFactors = FALSE))
  keys <- unique(H$key); n <- length(keys); nc <- min(3, n); nr <- ceiling(n / nc)
  cols <- variant_colors(order_variants(H$variant))
  png_open(file, 4.8 * nc, 3.9 * nr)
  par(mfrow = c(nr, nc), mar = c(4, 4, 3, 1), mgp = c(2.4, 0.7, 0))
  for (k in keys) {
    d <- H[H$key == k, ]
    h0 <- min(d$pred_len); d <- d[d$pred_len == h0, ]; d <- d[d$seed == min(d$seed), ]
    vs <- order_variants(d$variant)
    vals <- c(d$train_mse, d$val_mse)
    plot(NA, xlim = range(d$epoch), ylim = c(min(vals), quantile(vals, 0.97) * 1.1),
         xlab = "epoch", ylab = "MSE (z-scored units)", main = sprintf("%s  (horizon %d)", k, h0))
    grid(col = "grey90")
    for (v in vs) {
      dv <- d[d$variant == v, ]
      lines(dv$epoch, dv$val_mse, col = cols[v], lwd = 2)
      lines(dv$epoch, dv$train_mse, col = cols[v], lwd = 1.5, lty = 2)
      i <- which.min(dv$val_mse)
      points(dv$epoch[i], dv$val_mse[i], pch = 19, col = cols[v])
    }
    legend("topright", legend = vs, col = cols[vs], lwd = 2, cex = 0.75, bty = "n")
    mtext("solid = validation, dashed = training, dot = best epoch", side = 3, line = 0, cex = 0.55)
  }
  dev.off()
}

plot_metric_bars <- function(S, out_dir) {
  for (h in sort(unique(S$pred_len))) {
    d <- S[S$pred_len == h, ]
    keys <- unique(d$key); vs <- order_variants(d$variant); cols <- variant_colors(vs)
    png_open(file.path(out_dir, sprintf("plot_compare_metrics_H%d.png", h)), max(10, 3.2 * length(keys)), 5)
    par(mfrow = c(1, 2), mar = c(5, 4, 3, 1), mgp = c(2.4, 0.7, 0))
    for (metric in c("mse", "mae")) {
      mat <- sum_mat(d, h, paste0("test_", metric), vs, keys)
      sdm <- sum_mat(d, h, paste0(metric, "_sd"), vs, keys)
      pap <- vapply(keys, function(k) paper_ref(k, h, metric), numeric(1))
      top <- max(c(mat + ifelse(is.na(sdm), 0, sdm), pap), na.rm = TRUE) * 1.2
      draw <- mat; draw[is.na(draw)] <- 0
      mids <- barplot(draw, beside = TRUE, col = cols[vs], ylim = c(0, top), names.arg = keys, las = 2,
                      ylab = paste("test", toupper(metric), "(z-scored units)"),
                      main = sprintf("Test %s, horizon %d (lower is better)", toupper(metric), h))
      ok <- !is.na(sdm)
      if (any(ok)) suppressWarnings(arrows(mids[ok], (mat - sdm)[ok], mids[ok], (mat + sdm)[ok], angle = 90, code = 3, length = 0.03))
      ib <- match("base", vs)
      has_paper <- FALSE
      if (!is.na(ib)) for (j in seq_along(keys)) if (!is.na(pap[j])) {
        points(mids[ib, j], pap[j], pch = 23, bg = "gold", cex = 1.4); has_paper <- TRUE
      }
      if (metric == "mse")
        legend("topleft", legend = c(vs, if (has_paper) "paper (reported)"), bty = "n", cex = 0.75,
               fill = c(cols[vs], if (has_paper) NA), border = c(rep("black", length(vs)), if (has_paper) NA),
               pch = c(rep(NA, length(vs)), if (has_paper) 23), pt.bg = c(rep(NA, length(vs)), if (has_paper) "gold"))
    }
    dev.off()
  }
}

plot_improvement <- function(S, out_dir) {
  tk <- S[S$family == "timekan" & S$variant != "base" & !is.na(S$mse_vs_base_pct), ]
  if (!nrow(tk)) stop("need 'base' plus at least one other TimeKAN variant")
  keys <- unique(tk$key); vs <- order_variants(tk$variant); cols <- variant_colors(vs)
  mat <- matrix(NA_real_, length(vs), length(keys), dimnames = list(vs, keys))
  for (v in vs) for (k in keys) {
    x <- tk$mse_vs_base_pct[tk$variant == v & tk$key == k]
    if (length(x)) mat[v, k] <- mean(x)
  }
  rng <- range(c(0, mat), na.rm = TRUE); pad <- diff(rng) * 0.2 + 0.2
  png_open(file.path(out_dir, "plot_improvement_vs_base.png"), max(7, 1.5 * length(keys) * length(vs) / 2 + 3), 5)
  par(mar = c(5, 4.5, 3, 1))
  draw <- mat; draw[is.na(draw)] <- 0
  mids <- barplot(draw, beside = TRUE, col = cols[vs], names.arg = keys, las = 2, ylim = c(rng[1] - pad, rng[2] + pad),
                  ylab = "change in test MSE vs original TimeKAN (%)",
                  main = "Effect of the modifications (negative = better than the paper's model)")
  abline(h = 0, lwd = 1.5)
  ok <- !is.na(mat)
  text(mids[ok], mat[ok], sprintf("%+.1f%%", mat[ok]), pos = ifelse(mat[ok] >= 0, 3, 1), cex = 0.7)
  legend("topright", legend = vs, fill = cols[vs], bty = "n", cex = 0.8)
  dev.off()
}

plot_vs_horizon <- function(S, out_dir) {
  hs <- sort(unique(S$pred_len)); if (length(hs) < 2) stop("only one horizon was run")
  keys <- unique(S$key); n <- length(keys); nc <- min(3, n); nr <- ceiling(n / nc)
  cols <- variant_colors(order_variants(S$variant))
  png_open(file.path(out_dir, "plot_mse_vs_horizon.png"), 4.8 * nc, 3.9 * nr)
  par(mfrow = c(nr, nc), mar = c(4, 4, 3, 1), mgp = c(2.4, 0.7, 0))
  for (k in keys) {
    d <- S[S$key == k, ]; vs <- order_variants(d$variant)
    plot(NA, xlim = range(hs), ylim = range(d$test_mse, na.rm = TRUE), xlab = "forecast horizon", ylab = "test MSE", main = k, xaxt = "n")
    axis(1, at = hs)
    for (v in vs) { dv <- d[d$variant == v, ]; dv <- dv[order(dv$pred_len), ]
    lines(dv$pred_len, dv$test_mse, type = "b", pch = 19, col = cols[v], lwd = 2) }
    pp <- vapply(hs, function(h) paper_ref(k, h, "mse"), numeric(1))
    if (any(!is.na(pp))) points(hs, pp, pch = 23, bg = "gold", cex = 1.3)
    legend("topleft", legend = c(vs, if (any(!is.na(pp))) "paper (reported)"), bty = "n", cex = 0.7,
           col = c(cols[vs], if (any(!is.na(pp))) "black"), lwd = c(rep(2, length(vs)), if (any(!is.na(pp))) NA),
           pch = c(rep(19, length(vs)), if (any(!is.na(pp))) 23), pt.bg = c(rep(NA, length(vs)), if (any(!is.na(pp))) "gold"))
  }
  dev.off()
}

plot_efficiency <- function(S, out_dir) {
  S <- S[S$params > 0, ]; if (!nrow(S)) stop("no trained models")
  keys <- unique(S$key); n <- length(keys); nc <- min(3, n); nr <- ceiling(n / nc)
  cols <- variant_colors(order_variants(S$variant))
  png_open(file.path(out_dir, "plot_speed_vs_accuracy.png"), 4.8 * nc, 3.9 * nr)
  par(mfrow = c(nr, nc), mar = c(4, 4, 3, 1), mgp = c(2.4, 0.7, 0))
  for (k in keys) {
    d <- S[S$key == k, ]; d <- d[d$pred_len == min(d$pred_len), ]
    xr <- range(d$sec_per_epoch); xr <- xr + c(-0.15, 0.25) * max(diff(xr), 0.1 * xr[2], 0.1)
    plot(d$sec_per_epoch, d$test_mse, pch = 19, cex = 1.6, col = cols[d$variant], xlim = xr,
         xlab = "seconds per epoch (CPU)", ylab = "test MSE", main = sprintf("%s (horizon %d)", k, min(d$pred_len)))
    text(d$sec_per_epoch, d$test_mse, sprintf("%s\n%.1fK params", d$variant, d$params / 1e3), pos = 3, cex = 0.65)
  }
  dev.off()
}

load_detail <- function(out_dir, key, v, h, seed) {
  f <- file.path(out_dir, paste0("detail__", run_tag(key, v, h, seed), ".rds"))
  if (file.exists(f)) readRDS(f) else NULL
}

plot_forecasts <- function(dets, cmp, key, h, file) {
  d0 <- dets[[cmp[1]]]
  nW <- dim(d0$x)[1]; nC <- dim(d0$x)[3]
  wins <- unique(pmax(1, pmin(nW, round(nW * c(0.15, 0.5, 0.85)))))
  chs <- seq_len(min(2, nC))
  cols <- variant_colors(cmp)
  png_open(file, 5.4 * length(chs), 3 * length(wins))
  par(mfrow = c(length(wins), length(chs)), mar = c(3.8, 4, 2.6, 1), mgp = c(2.3, 0.7, 0))
  first <- TRUE
  for (w in wins) for (cc in chs) {
    hx <- d0$x[w, , cc]; ty <- d0$y[w, , cc]
    Tn <- length(hx); Fn <- length(ty)
    preds <- lapply(cmp, function(v) dets[[v]]$p[w, , cc])
    plot(NA, xlim = c(-Tn + 1, Fn), ylim = range(c(hx, ty, unlist(preds)), finite = TRUE),
         xlab = "time step (0 = last observed)", ylab = "value (z-scored)",
         main = sprintf("%s: test window %d, variate %d", key, w, d0$ch[cc]))
    abline(v = 0.5, lty = 3, col = "grey60")
    lines(seq(-Tn + 1, 0), hx, col = "grey50", lwd = 1.5)
    lines(seq_len(Fn), ty, col = "black", lwd = 2)
    for (i in seq_along(cmp)) lines(seq_len(Fn), preds[[i]], col = cols[i], lwd = 2, lty = 2)
    if (first) { legend("topleft", legend = c("history", "truth", cmp), col = c("grey50", "black", cols),
                        lwd = 2, lty = c(1, 1, rep(2, length(cmp))), bty = "n", cex = 0.7); first <- FALSE }
  }
  dev.off()
}

plot_diagnostics <- function(dets, cmp, key, h, file) {
  vs_all <- order_variants(names(dets)); cols_all <- variant_colors(vs_all); cols <- variant_colors(cmp)
  png_open(file, 11, 9)
  par(mfrow = c(2, 2), mar = c(4.2, 4.2, 3, 1), mgp = c(2.4, 0.7, 0))
  # (a) error vs forecast step
  steps <- lapply(vs_all, function(v) dets[[v]]$step_mse)
  plot(NA, xlim = c(1, max(lengths(steps))), ylim = range(unlist(steps), finite = TRUE),
       xlab = "forecast step", ylab = "test MSE at that step", main = "(a) Error grows with the forecast step")
  grid(col = "grey90")
  for (i in seq_along(vs_all)) lines(seq_along(steps[[i]]), steps[[i]], col = cols_all[i], lwd = 2)
  legend("topleft", legend = vs_all, col = cols_all, lwd = 2, bty = "n", cex = 0.75)
  # (b) per-variate error, first vs second compared model
  c1 <- dets[[cmp[1]]]$chan_mse
  if (length(cmp) >= 2 && length(c1) > 1) {
    c2 <- dets[[cmp[2]]]$chan_mse; lim <- range(c(c1, c2), finite = TRUE)
    plot(c1, c2, pch = 19, col = adjustcolor(cols[2], 0.6), xlim = lim, ylim = lim,
         xlab = paste("per-variate MSE:", cmp[1]), ylab = paste("per-variate MSE:", cmp[2]),
         main = sprintf("(b) Per-variate error (%d variates)", length(c1)))
    abline(0, 1, lty = 2)
    legend("topleft", bty = "n", cex = 0.8, legend = sprintf("%s better on %d/%d variates", cmp[2], sum(c2 < c1), length(c1)))
  } else {
    plot.new(); title("(b) Per-variate error"); text(0.5, 0.5, "needs >1 variate and 2 compared models")
  }
  # (c) predicted vs actual
  set.seed(1)
  d1 <- dets[[cmp[1]]]; n_all <- length(d1$y); idx <- sample.int(n_all, min(4000, n_all))
  truth <- as.vector(d1$y)[idx]; lim <- range(truth, finite = TRUE)
  plot(NA, xlim = lim, ylim = lim, xlab = "actual", ylab = "predicted", main = "(c) Predicted vs actual (test windows)")
  abline(0, 1, lty = 2)
  r2 <- numeric(0)
  for (i in seq_along(cmp)) {
    pr <- as.vector(dets[[cmp[i]]]$p)[idx]
    points(truth, pr, pch = ".", cex = 2.5, col = adjustcolor(cols[i], 0.5))
    r2[i] <- 1 - sum((pr - truth)^2) / sum((truth - mean(truth))^2)
  }
  legend("topleft", bty = "n", cex = 0.8, legend = sprintf("%s  R2 = %.3f", cmp, r2), col = cols, pch = 15)
  # (d) residual densities
  res <- lapply(cmp, function(v) as.vector(dets[[v]]$p) - as.vector(dets[[v]]$y))
  dens <- lapply(res, density)
  xl <- quantile(unlist(res), c(0.005, 0.995))
  plot(NA, xlim = xl, ylim = c(0, max(vapply(dens, function(d) max(d$y), numeric(1)))),
       xlab = "forecast error (prediction - actual)", ylab = "density", main = "(d) Residual distribution")
  abline(v = 0, lty = 2)
  for (i in seq_along(cmp)) lines(dens[[i]], col = cols[i], lwd = 2)
  legend("topright", bty = "n", cex = 0.8, col = cols, lwd = 2,
         legend = sprintf("%s  (bias %+.3f, sd %.3f)", cmp, vapply(res, mean, numeric(1)), vapply(res, sd, numeric(1))))
  dev.off()
}

plot_extras <- function(dets, key, file) {
  has_ab <- Filter(function(d) !is.null(d$ab), dets)
  has_pca <- Filter(function(d) !is.null(d$cumvar), dets)
  np <- (length(has_ab) > 0) + (length(has_pca) > 0)
  if (np == 0) return(invisible(NULL))
  png_open(file, 5.8 * np, 4.6)
  par(mfrow = c(1, np), mar = c(5.5, 4.2, 3, 1), mgp = c(2.4, 0.7, 0))
  if (length(has_ab)) {
    v <- names(has_ab)[length(has_ab)]; ab <- has_ab[[v]]$ab
    nl <- nrow(ab$alpha); nb <- ncol(ab$alpha)
    mat <- rbind(alpha = as.vector(t(ab$alpha)), beta = as.vector(t(ab$beta)))
    labs <- paste0("L", rep(1:nl, each = nb), "-B", rep(1:nb, times = nl))
    barplot(mat, beside = TRUE, names.arg = labs, las = 2, col = c("#4c72b0", "#dd8452"),
            ylab = "learned value (start = 1)", main = sprintf("Learned branch weights (%s)", v),
            ylim = c(0, max(1.2, max(mat, na.rm = TRUE) * 1.15)))
    abline(h = 1, lty = 2)
    legend("topright", legend = c("alpha (depthwise conv)", "beta (Chebyshev KAN)"), fill = c("#4c72b0", "#dd8452"), bty = "n", cex = 0.8)
    mtext("L = layer, B1 = highest-frequency band ... last band = lowest", side = 1, line = 4.3, cex = 0.7)
  }
  if (length(has_pca)) {
    d <- has_pca[[1]]; v <- names(has_pca)[1]
    m <- min(length(d$cumvar), max(40, d$pca_r * 2))
    plot(seq_len(m), d$cumvar[seq_len(m)], type = "b", pch = 19, cex = 0.6, ylim = c(0, 1),
         xlab = "number of principal components", ylab = "cumulative explained variance",
         main = sprintf("PCA on %s (%s)", key, v))
    abline(v = d$pca_r, lty = 2, col = "red"); abline(h = d$cumvar[d$pca_r], lty = 3, col = "grey50")
    legend("bottomright", bty = "n", cex = 0.85, legend = c(sprintf("kept r = %d components", d$pca_r),
                                                            sprintf("%.1f%% variance", 100 * d$cumvar[d$pca_r]),
                                                            sprintf("reconstruction floor MSE = %.4f", d$pca_floor)))
  }
  dev.off()
}

plot_dataset_details <- function(out_dir, M, S, key) {
  Mk <- M[M$key == key, ]
  h0 <- min(Mk$pred_len); Mk <- Mk[Mk$pred_len == h0, ]; s0 <- min(Mk$seed); Mk <- Mk[Mk$seed == s0, ]
  dets <- setNames(lapply(Mk$variant, function(v) load_detail(out_dir, key, v, h0, s0)), Mk$variant)
  dets <- dets[!vapply(dets, is.null, logical(1))]
  if (!length(dets)) stop("no detail files for ", key)
  tk <- Mk[Mk$family == "timekan" & Mk$variant %in% names(dets), ]
  cmp <- character(0)
  if ("base" %in% tk$variant) cmp <- "base"
  others <- tk[tk$variant != "base", ]
  if (nrow(others)) cmp <- c(cmp, others$variant[which.min(others$test_mse)])
  if (!length(cmp)) cmp <- names(dets)[1]
  base <- file.path(out_dir, paste0("plot_%s_", key, ".png"))
  safe_plot(paste("forecasts", key), plot_forecasts(dets, cmp, key, h0, sprintf(base, "forecasts")))
  safe_plot(paste("diagnostics", key), plot_diagnostics(dets, cmp, key, h0, sprintf(base, "diagnostics")))
  safe_plot(paste("extras", key), plot_extras(dets, key, sprintf(base, "alpha_beta_pca")))
}

make_all_plots <- function(out_dir) {
  M <- read_metrics(out_dir)
  if (is.null(M)) stop("No results found in '", out_dir, "'. Run a benchmark first.")
  S <- summarise_results(M)
  write.csv(S, file.path(out_dir, "summary.csv"), row.names = FALSE)
  old_opts <- options(width = 200); on.exit(options(old_opts), add = TRUE)
  cat("\n================ SUMMARY (mean over seeds; z-scored units) ================\n")
  show <- S[, c("key", "variant", "pred_len", "n_seeds", "test_mse", "test_mae", "mse_vs_base_pct", "params", "sec_per_epoch")]
  show$test_mse <- round(show$test_mse, 4); show$test_mae <- round(show$test_mae, 4)
  show$mse_vs_base_pct <- round(show$mse_vs_base_pct, 2); show$sec_per_epoch <- round(show$sec_per_epoch, 2)
  print(show, row.names = FALSE)
  cat("\n[plots] writing PNG files to", normalizePath(out_dir), "\n")
  safe_plot("training curves", plot_training_curves(out_dir, file.path(out_dir, "plot_training_curves.png")))
  safe_plot("metric bars", plot_metric_bars(S, out_dir))
  safe_plot("improvement", plot_improvement(S, out_dir))
  safe_plot("horizon", plot_vs_horizon(S, out_dir))
  safe_plot("efficiency", plot_efficiency(S, out_dir))
  for (k in unique(M$key)) safe_plot(paste("details", k), plot_dataset_details(out_dir, M, S, k))
  cat("[plots] done. Files:\n"); print(basename(list.files(out_dir, pattern = "^plot_.*\\.png$")))
  invisible(S)
}


# =============================================================================
# 5. Orchestration
# =============================================================================

run_benchmark <- function(opt) {
  load_torch(opt)
  M <- define_models()
  dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)
  csvs <- if (nzchar(opt$csv)) split_csv(opt$csv) else ""
  variants <- split_csv(opt$variants)
  vcs <- lapply(variants, parse_variant)             # validates the names early
  hs <- as.integer(split_csv(opt$pred_len)); seeds <- as.integer(split_csv(opt$seeds))
  for (csv in csvs) {
    ds <- load_dataset(csv, opt)
    for (h in hs) for (sd in seeds) for (vc in vcs) {
      tag <- run_tag(ds$key, vc$name, h, sd)
      if (opt$force == 0 && file.exists(file.path(opt$out_dir, paste0("metrics__", tag, ".csv")))) {
        cat(sprintf("[skip] %s already finished (use --force 1 to rerun)\n", tag)); next
      }
      res <- tryCatch(train_one(ds, vc, opt, sd, h, M), error = function(e) {
        message("  FAILED ", tag, ": ", conditionMessage(e)); NULL })
      if (!is.null(res)) write_run_outputs(res, opt$out_dir)
    }
  }
  if (opt$plots == 1) make_all_plots(opt$out_dir)
}

run_selftest <- function(opt) {
  cat("== TimeKAN self-test (tiny synthetic data; checks code paths, not accuracy) ==\n")
  load_torch(opt)
  M <- define_models()
  o <- opt
  o$seq_len <- 32; o$d_model <- 8; o$epochs <- 2; o$batch_size <- 16; o$patience <- 2
  o$season <- 8; o$quiet <- 1; o$out_dir <- file.path(tempdir(), "timekan_selftest")
  set.seed(1); n <- 700; tt <- 0:(n - 1)
  mat <- sapply(1:12, function(p) sin(2 * pi * tt / 48 + p) + 0.5 * sin(2 * pi * tt / 8 + p) + 0.1 * rnorm(n))
  colnames(mat) <- paste0("V", 1:12)
  ds <- finalize_dataset(mat, "selftest", "selftest", 0, "721", quiet = TRUE)
  ok <- TRUE
  for (v in c("naive", "snaive", "linear", "base", "ab", "reg", "ab+reg", "hi+reg", "pca", "ab+reg+pca")) {
    r <- tryCatch(train_one(ds, parse_variant(v), o, 0, 8, M), error = function(e) { message("  ERROR: ", conditionMessage(e)); NULL })
    pass <- !is.null(r) && is.finite(r$metrics$test_mse)
    if (pass) write_run_outputs(r, o$out_dir)
    cat(sprintf("  %-12s %s\n", v, if (pass) "PASS" else "FAIL"))
    ok <- ok && pass
  }
  # univariate path
  ds1 <- finalize_dataset(mat[, 1, drop = FALSE], "selftest_uni", "selftest", 0, "721", quiet = TRUE)
  r1 <- tryCatch(train_one(ds1, parse_variant("ab+reg"), o, 0, 8, M), error = function(e) { message("  ERROR: ", conditionMessage(e)); NULL })
  pass1 <- !is.null(r1) && is.finite(r1$metrics$test_mse)
  if (pass1) write_run_outputs(r1, o$out_dir)
  cat(sprintf("  %-12s %s\n", "univariate", if (pass1) "PASS" else "FAIL"))
  ok <- ok && pass1
  if (ok) { cat("\nAll model paths run. Testing plots...\n"); make_all_plots(o$out_dir) }
  cat(if (ok) "\nSELF-TEST PASSED\n" else "\nSELF-TEST FAILED (see errors above)\n")
}

main <- function() {
  opt <- parse_args(DEFAULTS)
  switch(opt$mode,
         benchmark = run_benchmark(opt),
         plots     = make_all_plots(opt$out_dir),
         selftest  = run_selftest(opt),
         stop("Unknown --mode '", opt$mode, "' (use benchmark, plots or selftest)"))
  invisible(NULL)
}

# Run main() only when executed as a script (not when source()'d for debugging).
if (sys.nframe() == 0) main()