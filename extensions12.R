# Additional univariate and Indian-data experiments for the original R TimeKAN.
# Run from the repository root. The original timekan.R is not modified.
source("timekan.R")

extension_defaults <- modifyList(default_options, list(
  experiment = "ett", target = "OT", test_csv = "", city = "Delhi",
  start_date = "", end_date = "", max_gap = 3, season = 24,
  name = "experiment", results_dir = "results/extensions12", self_test = 0,
  batch_size = 128, lr = 0.01
))

# Retain the original loader/optimizer/model; replace only data preparation and
# window selection so missing Indian observations cannot become training labels.
load_series <- function(csv_path = "") {
  df <- read.csv(csv_path, check.names = FALSE)
  as.matrix(df[, -1, drop = FALSE])
}

causal_fill <- function(x, limit) {
  out <- x
  previous <- NA_real_; gap <- 0L
  for (i in seq_along(x)) {
    if (is.finite(x[i])) {
      previous <- x[i]; gap <- 0L
    } else {
      gap <- gap + 1L
      if (gap <= limit && is.finite(previous)) out[i] <- previous
    }
  }
  out
}

read_extension_data <- function(opt) {
  read_one <- function(path) read.csv(path, check.names = FALSE,
                                     stringsAsFactors = FALSE)
  df <- read_one(opt$csv)
  if (nzchar(opt$test_csv)) df <- rbind(df, read_one(opt$test_csv))
  if (opt$experiment == "air") {
    stopifnot("City" %in% names(df), "Datetime" %in% names(df))
    df <- df[df$City == opt$city, , drop = FALSE]
    date_col <- "Datetime"
    step <- 3600
  } else if (opt$experiment == "climate") {
    date_col <- if ("date" %in% names(df)) "date" else names(df)[1]
    step <- 86400
  } else if (opt$experiment == "ett") {
    date_col <- names(df)[1]
    step <- if (opt$split == "ettm") 900 else 3600
  } else stop("experiment must be ett, air or climate")
  stopifnot(opt$target %in% names(df), nrow(df) > 0)
  # UTC is an indexing convention here: input local clock labels are not shifted.
  tt <- as.POSIXct(df[[date_col]], tz = "UTC",
                  tryFormats = c("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"))
  if (anyNA(tt)) stop("Unparseable timestamps; inspect the source date format")
  xx <- suppressWarnings(as.numeric(df[[opt$target]]))
  xx[!is.finite(xx)] <- NA_real_
  order_id <- order(tt)
  tt <- tt[order_id]; xx <- xx[order_id]
  # Source Delhi train/test files may share a boundary date. Identical duplicate
  # values are harmless; conflicting duplicates require an explicit decision.
  duplicate_times <- unique(tt[duplicated(tt)])
  for (t in duplicate_times) {
    values <- unique(xx[as.numeric(tt) == as.numeric(t)])
    values <- values[is.finite(values)]
    if (length(values) > 1) stop("Conflicting duplicate timestamp: ", t)
  }
  if (anyDuplicated(tt)) {
    cat("Dropping identical duplicate timestamps:", sum(duplicated(tt)), "\n")
    keep <- !duplicated(tt); tt <- tt[keep]; xx <- xx[keep]
  }
  if (nzchar(opt$start_date)) {
    keep <- tt >= as.POSIXct(opt$start_date, tz = "UTC")
    tt <- tt[keep]; xx <- xx[keep]
  }
  if (nzchar(opt$end_date)) {
    # The end date is inclusive, including all hours on that day.
    keep <- tt < as.POSIXct(opt$end_date, tz = "UTC") + 86400
    tt <- tt[keep]; xx <- xx[keep]
  }
  if (length(tt) < 2) stop("Selected interval is empty or too short")
  grid <- seq(min(tt), max(tt), by = step)
  index <- match(tt, grid)
  if (anyNA(index)) stop("Timestamps are not aligned to the requested frequency")
  raw <- rep(NA_real_, length(grid)); raw[index] <- xx
  list(time = grid, raw = raw, step = step,
       inserted_timestamps = length(grid) - length(tt))
}

prepare_data <- function(opt) {
  series <- read_extension_data(opt)
  raw <- matrix(series$raw, ncol = 1,
                dimnames = list(NULL, opt$target))
  b <- split_borders(nrow(raw), opt$split, opt$csv)
  if (b$test[2] > nrow(raw)) stop("Not enough rows for the requested split")
  if (b$train[2] <= opt$seq_len + opt$pred_len) stop("Training interval too short")
  observed <- is.finite(raw)
  train_rows <- b$train[1]:b$train[2]
  mu <- colMeans(raw[train_rows, , drop = FALSE], na.rm = TRUE)
  sdev <- apply(raw[train_rows, , drop = FALSE], 2, sd, na.rm = TRUE)
  if (!all(is.finite(mu)) || !all(is.finite(sdev)) || any(sdev <= 0)) {
    stop("Training observations are missing or constant")
  }
  # Only past observations are carried forward, at most max_gap time steps.
  filled <- matrix(causal_fill(raw[, 1], opt$max_gap), ncol = 1,
                   dimnames = dimnames(raw))
  scaled <- scale(filled, center = mu, scale = sdev)
  part <- function(r, back) {
    idx <- (r[1] - back):r[2]
    ans <- scaled[idx, , drop = FALSE]
    attr(ans, "observed") <- observed[idx, , drop = FALSE]
    ans
  }
  result <- list(split = b$name, train = part(b$train, 0),
                 val = part(b$val, opt$seq_len),
                 test = part(b$test, opt$seq_len), mu = mu, sdev = sdev,
                 names = opt$target, time = series$time, borders = b,
                 raw = raw, observed = observed,
                 inserted_timestamps = series$inserted_timestamps)
  result
}

valid_starts <- function(data, seq_len, pred_len) {
  n <- nrow(data) - seq_len - pred_len + 1L
  if (n <= 0) return(integer())
  observed <- attr(data, "observed")
  if (is.null(observed)) observed <- is.finite(data)
  bad_input <- rowSums(!is.finite(data)) > 0
  bad_target <- rowSums(!observed | !is.finite(data)) > 0
  input_prefix <- c(0, cumsum(bad_input))
  target_prefix <- c(0, cumsum(bad_target))
  s <- seq_len(n)
  count_x <- input_prefix[s + seq_len] - input_prefix[s]
  count_y <- target_prefix[s + seq_len + pred_len] - target_prefix[s + seq_len]
  s[count_x == 0 & count_y == 0]
}

window_dataset <- dataset(
  "ObservedWindowDataset",
  initialize = function(data, seq_len, pred_len) {
    self$starts <- valid_starts(data, seq_len, pred_len)
    if (!length(self$starts)) stop("No valid windows; inspect missingness and horizon")
    self$data <- unname(as.matrix(data))
    self$seq_len <- seq_len; self$pred_len <- pred_len
  },
  .length = function() length(self$starts),
  .getbatch = function(index) {
    starts <- self$starts[index]
    gather <- function(offset, len) {
      rows <- as.vector(outer(0:(len - 1), starts + offset, "+"))
      torch_tensor(self$data[rows, , drop = FALSE], dtype = torch_float())$
        reshape(c(length(index), len, ncol(self$data)))
    }
    list(x = gather(0, self$seq_len), y = gather(self$seq_len, self$pred_len))
  }
)

run_self_tests <- function() {
  stopifnot(identical(causal_fill(c(1, NA, NA, 4), 1), c(1, 1, NA, 4)))
  a <- matrix(seq_len(40), ncol = 1)
  attr(a, "observed") <- matrix(TRUE, nrow(a), 1)
  ds <- window_dataset(a, 4, 2)
  b <- ds$.getbatch(c(1, 3))
  stopifnot(identical(as.numeric(b$x[1, , 1]), c(1, 2, 3, 4)),
            identical(as.numeric(b$x[2, , 1]), c(3, 4, 5, 6)),
            identical(as.numeric(b$y[2, , 1]), c(7, 8)))
  attr(a, "observed")[10, 1] <- FALSE
  starts <- valid_starts(a, 4, 2)
  stopifnot(!any(starts %in% c(5, 6)))
  cat("EXTENSION_SELF_TESTS_OK\n")
}

evaluate_baselines <- function(model, data, opt, device) {
  loader <- dataloader(window_dataset(data, opt$seq_len, opt$pred_len),
                       batch_size = opt$batch_size, shuffle = FALSE)
  totals <- matrix(0, nrow = 3, ncol = 2,
                   dimnames = list(c("TimeKAN", "Persistence", "SeasonalNaive"),
                                   c("mse", "mae")))
  n <- 0
  model$eval()
  with_no_grad({
    coro::loop(for (b in loader) {
      x <- b$x$to(device = device); y <- b$y$to(device = device)
      persistence <- x[, opt$seq_len, , drop = FALSE]$expand_as(y)
      ids <- opt$seq_len - opt$season + 1L + ((seq_len(opt$pred_len) - 1L) %% opt$season)
      seasonal <- x[, ids, , drop = FALSE]
      preds <- list(model(x), persistence, seasonal)
      for (j in seq_along(preds)) {
        totals[j, 1] <- totals[j, 1] + nnf_mse_loss(preds[[j]], y, reduction = "sum")$item()
        totals[j, 2] <- totals[j, 2] + nnf_l1_loss(preds[[j]], y, reduction = "sum")$item()
      }
      n <- n + y$numel()
    })
  })
  totals / n
}

run_extension <- function(opt) {
  stopifnot(opt$season >= 1, opt$season <= opt$seq_len, opt$max_gap >= 0)
  dir.create(opt$results_dir, recursive = TRUE, showWarnings = FALSE)
  prefix <- file.path(opt$results_dir, opt$name)
  if (file.exists(paste0(prefix, "_summary.csv"))) {
    cat("Already completed:", opt$name, "\n"); return(invisible(NULL))
  }
  d <- prepare_data(opt)
  counts <- sapply(c("train", "val", "test"), function(k)
    length(valid_starts(d[[k]], opt$seq_len, opt$pred_len)))
  if (any(counts == 0)) stop("At least one split has no valid windows")
  cat("Valid windows:", paste(names(counts), counts, collapse = "; "), "\n")
  cat("Targets are scored only when originally observed; no imputed labels.\n")
  # Save the complete resolved configuration and preprocessing provenance.
  capture.output(dput(opt), file = paste0(prefix, "_options.R"))
  hashes <- tools::md5sum(c(opt$csv, if (nzchar(opt$test_csv)) opt$test_csv))
  metadata <- data.frame(
    experiment = opt$experiment, target = opt$target, city = opt$city,
    first_timestamp = as.character(d$time[1]),
    last_timestamp = as.character(tail(d$time, 1)), rows = nrow(d$raw),
    inserted_timestamps = d$inserted_timestamps,
    missing_observations = sum(!d$observed),
    train_end_row = d$borders$train[2], val_end_row = d$borders$val[2],
    test_end_row = d$borders$test[2],
    train_mean = d$mu[1], train_sd = d$sdev[1],
    train_windows = counts["train"], val_windows = counts["val"],
    test_windows = counts["test"], source_md5 = paste(hashes, collapse = ";"))
  write.csv(metadata, paste0(prefix, "_data.csv"), row.names = FALSE)
  opt$out <- paste0(prefix, "_training.csv")
  opt$save <- paste0(prefix, ".pt")
  result <- run_experiment(opt)
  metrics <- evaluate_baselines(result$model, result$data$test, opt, result$device)
  summary <- data.frame(experiment = opt$name, target = opt$target,
    horizon = opt$pred_len, seed = opt$seed, model = rownames(metrics),
    mse_standardized = metrics[, "mse"], mae_standardized = metrics[, "mae"],
    mse_original_units = metrics[, "mse"] * d$sdev[1]^2,
    mae_original_units = metrics[, "mae"] * d$sdev[1], row.names = NULL)
  stopifnot(abs(metrics["TimeKAN", "mse"] - result$test["mse"]) < 1e-5)
  write.csv(summary, paste0(prefix, "_summary.csv"), row.names = FALSE)
  print(summary, row.names = FALSE)
  cat("EXTENSION_COMPLETE", opt$name, "\n")
}

if (sys.nframe() == 0L) {
  opt <- parse_args(extension_defaults)
  if (opt$self_test > 0) run_self_tests() else run_extension(opt)
}
