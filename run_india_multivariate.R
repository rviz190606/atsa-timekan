# Ordinary, unchanged TimeKAN on multivariate Indian data. R + torch only.
# Run from the repository root containing timekan.R.
# Rscript run_india_multivariate.R --device cuda --download 1
# Existing raw CSVs: --air_csv path/city_hour.csv --climate_csv path/weather.csv
# Regenerate images without torch/training: --task plots

india_defaults <- list(task = "all", device = "auto", epochs = 10, patience = 3,
  batch_size = 16, lr = 0.001, seed = 0, seq_len = 96, d_model = 16,
  air_csv = "", climate_csv = "", download = 0, backup = "",
  results_dir = "results/india/timekan", min_train_coverage = 0.5)

india_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  opt <- india_defaults
  if (length(args) %% 2) stop("Use --option value pairs")
  if (length(args)) for (i in seq(1, length(args), 2)) {
    key <- sub("^--", "", args[i])
    if (!key %in% names(opt)) stop("Unknown option: ", key)
    opt[[key]] <- if (is.numeric(opt[[key]])) as.numeric(args[i + 1]) else args[i + 1]
  }
  stopifnot(opt$task %in% c("all", "air", "climate", "plots", "self_test"),
    opt$epochs >= 1, opt$patience >= 1, opt$batch_size >= 1,
    opt$seq_len >= 8, opt$seq_len %% 4 == 0,
    opt$min_train_coverage > 0, opt$min_train_coverage <= 1)
  opt
}

india_download <- function(handle, wanted = NULL) {
  directory <- file.path("data/kaggle", gsub("/", "_", handle))
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  existing <- list.files(directory, "[.]csv$", full.names = TRUE, recursive = TRUE)
  if (!is.null(wanted)) existing <- existing[basename(existing) == wanted]
  if (length(existing)) return(existing)
  archive <- file.path(directory, "dataset.zip")
  if (!file.exists(archive)) {
    options(timeout = max(1800, getOption("timeout")))
    download.file(paste0("https://www.kaggle.com/api/v1/datasets/download/", handle),
                  archive, mode = "wb", method = "libcurl")
  }
  entries <- unzip(archive, list = TRUE)$Name
  entries <- entries[grepl("[.]csv$", entries, ignore.case = TRUE)]
  if (!is.null(wanted)) entries <- entries[basename(entries) == wanted]
  if (!length(entries)) stop("Requested CSV is absent from the Kaggle ZIP")
  # Do not extract absolute paths or parent traversals from an archive.
  if (any(grepl("(^[/\\\\]|^[A-Za-z]:|(^|[/\\\\])[.][.]([/\\\\]|$))", entries)))
    stop("Unsafe archive path")
  unzip(archive, files = entries, exdir = directory)
  file.path(directory, entries)
}

india_raw_path <- function(kind, opt) {
  supplied <- opt[[paste0(kind, "_csv")]]
  if (nzchar(supplied)) {
    if (!file.exists(supplied)) stop("CSV not found: ", supplied)
    return(supplied)
  }
  # The old canonical extension CSVs contain ONE column; never reuse those.
  candidates <- list.files("data", "[.]csv$", full.names = TRUE, recursive = TRUE)
  candidates <- candidates[!grepl("/extensions12/", gsub("\\\\", "/", candidates))]
  if (kind == "air") candidates <- candidates[basename(candidates) == "city_hour.csv"]
  else candidates <- candidates[grepl("weather|climate", basename(candidates), ignore.case = TRUE)]
  if (length(candidates)) {
    for (p in candidates) {
      header <- names(read.csv(p, nrows = 1, check.names = FALSE))
      if (kind == "air" && all(c("City", "Datetime", "PM2.5") %in% header)) return(p)
      if (kind == "climate" && (all(c("DATE", "temp") %in% header) ||
          all(c("date", "meantemp") %in% header))) return(p)
    }
  }
  if (!opt$download) stop("Raw multivariate CSV required. Supply --", kind,
                         "_csv or --download 1. The earlier univariate CSV is insufficient.")
  if (kind == "air") return(india_download("rohanrao/air-quality-data-in-india", "city_hour.csv")[1])
  paths <- india_download("yug201/daily-climate-time-series-data-delhi-india")
  eligible <- vapply(paths, function(p) {
    h <- names(read.csv(p, nrows = 1, check.names = FALSE))
    all(c("DATE", "temp") %in% h) || all(c("date", "meantemp") %in% h)
  }, logical(1))
  if (!any(eligible)) stop("No supported climate CSV")
  # Prefer the dataset's full weather table over any ancillary short table.
  paths <- paths[eligible]
  paths[which.max(vapply(paths, function(p) nrow(read.csv(p)), integer(1)))]
}

india_prepare <- function(path, kind, seq_len = 96, min_coverage = 0.5) {
  df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  if (kind == "air") {
    stopifnot(all(c("City", "Datetime") %in% names(df)))
    df <- df[!is.na(df$City) & df$City == "Delhi", , drop = FALSE]
    tt <- as.POSIXct(df$Datetime, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
    keep <- !is.na(tt) & tt >= as.POSIXct("2018-01-01", tz = "UTC") &
      tt < as.POSIXct("2020-01-01", tz = "UTC")
    df <- df[keep, , drop = FALSE]; tt <- tt[keep]
    columns <- intersect(c("PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
                            "SO2", "O3", "Benzene", "Toluene", "Xylene"), names(df))
    step <- 3600; season <- 24
  } else {
    if (all(c("DATE", "temp") %in% names(df))) {
      date <- as.Date(df$DATE, "%m/%d/%Y"); date_col <- "DATE"
    } else if (all(c("date", "meantemp") %in% names(df))) {
      date <- as.Date(df$date); date_col <- "date"
    } else stop("Climate needs DATE/temp or date/meantemp")
    if (anyNA(date)) stop("Invalid climate dates")
    tt <- as.POSIXct(date, tz = "UTC")
    # Measured weather fields only: exclude row IDs, calendar fields, and the
    # supplied dataset's engineered temperature/humidity interaction features.
    columns <- intersect(c("tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin",
      "feelslike", "humidity", "precip", "precipprob", "precipcover", "windspeed",
      "sealevelpressure", "meantemp", "wind_speed", "meanpressure", "wind", "pressure"), names(df))
    columns <- columns[vapply(df[columns], function(x) {
      present <- !is.na(x) & nzchar(trimws(as.character(x)))
      any(present) && mean(is.finite(suppressWarnings(as.numeric(x[present])))) >= .95
    }, logical(1))]
    step <- 86400; season <- 7
  }
  if (anyNA(tt) || length(tt) < seq_len * 2) stop("Invalid/insufficient timestamps")
  if (length(columns) < 2) stop("Need at least two measured numeric variables")
  raw <- sapply(columns, function(k) suppressWarnings(as.numeric(df[[k]])))
  raw[!is.finite(raw)] <- NA_real_
  sorted <- order(tt); tt <- tt[sorted]; raw <- raw[sorted, , drop = FALSE]
  # Merge identical duplicates, retaining any observed value; reject disagreements.
  for (t in unique(tt[duplicated(tt)])) {
    ids <- which(as.numeric(tt) == as.numeric(t))
    for (j in seq_len(ncol(raw))) {
      values <- unique(raw[ids, j]); values <- values[is.finite(values)]
      if (length(values) > 1) stop("Conflicting duplicate timestamp: ", t, ", ", columns[j])
      if (length(values)) raw[ids[1], j] <- values
    }
  }
  keep <- !duplicated(tt); tt <- tt[keep]; raw <- raw[keep, , drop = FALSE]
  grid <- seq(min(tt), max(tt), by = step)
  ids <- match(tt, grid)
  if (anyNA(ids)) stop("Irregular time alignment")
  expanded <- matrix(NA_real_, length(grid), ncol(raw), dimnames = list(NULL, columns))
  expanded[ids, ] <- raw; raw <- expanded
  n <- nrow(raw); ends <- c(floor(.7 * n), floor(.7 * n) + floor(.1 * n), n)
  if (ends[1] <= seq_len || ends[2] - ends[1] < 1) stop("Dataset too short")
  train <- raw[seq_len(ends[1]), , drop = FALSE]
  coverage <- colMeans(is.finite(train))
  sdev <- apply(train, 2, sd, na.rm = TRUE)
  retained <- is.finite(sdev) & sdev > 0 & coverage >= min_coverage
  dropped <- data.frame(variable = columns[!retained], train_coverage = coverage[!retained],
                        reason = ifelse(!is.finite(sdev[!retained]) | sdev[!retained] <= 0,
                                        "constant_or_no_training_values", "low_training_coverage"))
  raw <- raw[, retained, drop = FALSE]; train <- train[, retained, drop = FALSE]
  if (ncol(raw) < 2) stop("Fewer than two usable training variables; inspect data coverage")
  mu <- colMeans(train, na.rm = TRUE); sdev <- sdev[retained]
  medians <- apply(train, 2, median, na.rm = TRUE)
  filled <- raw
  for (j in seq_len(ncol(raw))) {
    last <- medians[j]
    for (i in seq_len(nrow(raw))) {
      if (is.finite(raw[i, j])) last <- raw[i, j] else filled[i, j] <- last
    }
  }
  # Filling is for inputs only; all losses/metrics mask originally missing targets.
  list(x = unname(scale(filled, center = mu, scale = sdev)), raw = raw,
       observed = is.finite(raw), time = grid, ends = ends, mu = mu, sdev = sdev,
       columns = colnames(raw), dropped = dropped, season = season,
       train_coverage = coverage[retained], source_md5 = unname(tools::md5sum(path)))
}

india_starts <- function(d, part, history, horizon) {
  boundaries <- switch(part, train = c(1L, d$ends[1]),
    val = c(d$ends[1] + 1L, d$ends[2]), test = c(d$ends[2] + 1L, d$ends[3]))
  first <- max(boundaries[1], history + 1L); last <- boundaries[2] - horizon + 1L
  if (last < first) stop("Split too short for horizon: ", part)
  target_first <- seq.int(first, last)
  p <- c(0, cumsum(rowSums(d$observed)))
  target_first <- target_first[p[target_first + horizon] - p[target_first] > 0]
  if (!length(target_first)) stop("No observed targets in ", part)
  target_first - history
}

india_dataset_class <- function() dataset("IndiaMultivariateWindows",
  initialize = function(d, starts, history, horizon) {
    self$x <- d$x; self$mask <- d$observed
    self$starts <- starts; self$history <- history; self$horizon <- horizon
  },
  .length = function() length(self$starts),
  .getbatch = function(index) {
    starts <- self$starts[index]
    gather <- function(data, offset, len) {
      ids <- as.vector(outer(0:(len - 1), starts + offset, "+"))
      torch_tensor(unname(data[ids, , drop = FALSE]), dtype = torch_float())$
        reshape(c(length(index), len, ncol(data)))
    }
    list(x = gather(self$x, 0, self$history),
      y = gather(self$x, self$history, self$horizon),
      mask = gather(self$mask, self$history, self$horizon))
  })

india_loss <- function(prediction, target, mask) {
  ((prediction - target)$pow(2) * mask)$sum() / mask$sum()$clamp_min(1)
}

india_torch_test <- function() {
  torch_manual_seed(7)
  # Check tensor layout across multiple channels AND multiple batch windows.
  d <- list(x = cbind(1:50, 101:150, 201:250),
    observed = matrix(TRUE, 50, 3), columns = c("a", "b", "c"),
    season = 2, sdev = rep(1, 3))
  d$observed[10, 2] <- FALSE
  DS <- india_dataset_class(); ds <- DS(d, c(1, 3), 8, 4)
  b <- ds$.getbatch(1:2)
  stopifnot(identical(as.integer(b$x$size()), c(2L, 8L, 3L)),
    identical(as.numeric(b$x[2, , 2]), as.numeric(103:110)),
    b$mask[1, 2, 2]$item() == 0)
  p <- torch_tensor(c(1, 999), requires_grad = TRUE)
  loss <- india_loss(p, torch_zeros(2), torch_tensor(c(1, 0)))
  loss$backward()
  stopifnot(loss$item() == 1, identical(as.numeric(p$grad), c(2, 0)))
  model <- timekan(8, 4, 4, 3, 2, 1, 1, 0)
  model(b$x)$pow(2)$mean()$backward()
  stopifnot(all(vapply(model$parameters, function(p)
    !is.null(p$grad) && torch_isfinite(p$grad)$all()$item(), logical(1))))
  loader <- dataloader(ds, batch_size = 2, shuffle = FALSE)
  metrics <- india_metrics(model, loader, "cpu", d, 8, 4)
  # Persistence errors at each lead are exactly 1,2,3,4 for these linear series.
  # One missing lead-2 target in batch 1/channel 2 is excluded.
  expected <- (6 * sum((1:4)^2) - 2^2) / 23
  stopifnot(abs(metrics$overall$mse_standardized[2] - expected) < 1e-5,
    metrics$overall$observed_targets[1] == 23)
  cat("MULTIVARIATE_TORCH_LAYOUT_MASK_GRADIENT_AND_METRICS_OK\n")
}

india_val <- function(model, loader, device) {
  model$eval(); total <- 0; count <- 0
  with_no_grad({ coro::loop(for (b in loader) {
    x <- b$x$to(device = device); y <- b$y$to(device = device); m <- b$mask$to(device = device)
    total <- total + (((model(x) - y)$pow(2)) * m)$sum()$item()
    count <- count + m$sum()$item()
  }) })
  if (count == 0) stop("No observed validation targets")
  total / count
}

india_metrics <- function(model, loader, device, d, history, horizon) {
  methods <- c("TimeKAN", "Persistence", "SeasonalNaive")
  squared <- absolute <- array(0, c(3, horizon, length(d$columns)))
  counts <- matrix(0, horizon, length(d$columns))
  model$eval()
  with_no_grad({ coro::loop(for (b in loader) {
    x <- b$x$to(device = device)
    seasonal_ids <- history - d$season + 1L + ((seq_len(horizon) - 1L) %% d$season)
    preds <- list(model(x), x[, history, , drop = FALSE]$expand(c(x$size(1), horizon, x$size(3))),
                  x[, seasonal_ids, , drop = FALSE])
    truth <- as_array(b$y); mask <- as_array(b$mask)
    counts <- counts + apply(mask, c(2, 3), sum)
    for (j in seq_along(preds)) {
      error <- as_array(preds[[j]]$to(device = "cpu")) - truth
      squared[j, , ] <- squared[j, , ] + apply(error^2 * mask, c(2, 3), sum)
      absolute[j, , ] <- absolute[j, , ] + apply(abs(error) * mask, c(2, 3), sum)
    }
  }) })
  if (any(colSums(counts) == 0)) stop("A retained variable has no observed test targets")
  overall <- data.frame(model = methods,
    mse_standardized = apply(squared, 1, sum) / sum(counts),
    mae_standardized = apply(absolute, 1, sum) / sum(counts), observed_targets = sum(counts))
  per_variable <- do.call(rbind, lapply(seq_len(3), function(j) {
    mse <- colSums(squared[j, , ]) / colSums(counts)
    mae <- colSums(absolute[j, , ]) / colSums(counts)
    data.frame(model = methods[j], variable = d$columns, mse_standardized = mse,
      mae_standardized = mae, mse_original_units = mse * d$sdev^2,
      mae_original_units = mae * d$sdev, observed_targets = colSums(counts))
  }))
  by_horizon <- do.call(rbind, lapply(seq_len(3), function(j) data.frame(
    model = methods[j], lead = seq_len(horizon),
    mse_standardized = rowSums(squared[j, , ]) / rowSums(counts),
    mae_standardized = rowSums(absolute[j, , ]) / rowSums(counts))))
  list(overall = overall, per_variable = per_variable, by_horizon = by_horizon)
}

india_png <- function(path, draw, width = 1800, height = 1200) {
  # Keep presentation figures separate from numerical experiment records.
  path <- gsub("\\", "/", path, fixed = TRUE)
  for (kind in c("timekan", "learned_baselines")) {
    prefix <- paste0("results/india/", kind, "/")
    if (startsWith(path, prefix))
      path <- paste0("figures/india/", kind, "/", substring(path, nchar(prefix) + 1L))
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  png(path, width = width, height = height, res = 160)
  on.exit(dev.off())
  draw()
}

india_title <- function(record) paste0("Delhi ", if (record$kind == "air") "air quality" else "weather",
  ": ", record$horizon, "-", if (record$kind == "air") "hour" else "day",
  " forecast (", length(record$columns), " variables)")
india_metric_label <- function(metric) if (metric == "mse_standardized") "MSE (standardized)" else "MAE (standardized)"

india_plots <- function(record, prefix) {
  colors <- c("#2563eb", "#64748b", "#d97706")
  methods <- record$metrics$overall$model
  india_png(paste0(prefix, "_comparison.png"), function() {
    par(mfrow = c(1, 2), mar = c(5, 5, 4, 1))
    for (metric in c("mse_standardized", "mae_standardized")) {
      v <- record$metrics$overall[[metric]]
      bp <- barplot(v, names.arg = c("TimeKAN", "Persistence", "Seasonal-naive"), col = colors, las = 1,
        ylim = c(0, max(v) * 1.2), ylab = india_metric_label(metric),
        main = paste(india_title(record), "\nObserved targets only"), cex.main = .85)
      text(bp, v, sprintf("%.4f", v), pos = 3, cex = .85)
    }
  })
  india_png(paste0(prefix, "_per_variable.png"), function() {
    par(mfrow = c(2, 1), mar = c(9, 5, 3, 1))
    for (metric in c("mse_standardized", "mae_standardized")) {
      frame <- record$metrics$per_variable
      values <- t(vapply(methods, function(method) frame[frame$model == method, metric],
                         numeric(length(record$columns))))
      barplot(values, beside = TRUE, col = colors, names.arg = record$columns, las = 2,
        ylim = c(0, max(values) * 1.3), cex.names = .8,
        ylab = india_metric_label(metric), main = india_title(record))
      legend("topright", methods, fill = colors, bty = "n", cex = .8, horiz = TRUE)
    }
  }, height = 1600)
  india_png(paste0(prefix, "_horizon_errors.png"), function() {
    par(mfrow = c(1, 2), mar = c(5, 5, 4, 1))
    frame <- record$metrics$by_horizon
    for (metric in c("mse_standardized", "mae_standardized")) {
      values <- vapply(methods, function(method) frame[frame$model == method, metric],
                       numeric(record$horizon))
      matplot(seq_len(record$horizon), values, type = "l", lty = 1, col = colors,
        xlab = paste("Forecast lead (", record$unit, ")"),
        ylab = india_metric_label(metric), main = india_title(record), cex.main = .85)
      legend("topleft", methods, col = colors, lty = 1, bty = "n", cex = .8)
    }
  })
  india_png(paste0(prefix, "_training.png"), function() {
    matplot(record$history$epoch, record$history[, c("train_mse", "val_mse")],
      type = "l", lty = 1, col = colors[1:2], xlab = "Epoch",
      ylab = "Standardized MSE on observed targets", main = india_title(record))
    abline(v = record$best_epoch, lty = 3)
    legend("topright", c("Training", "Validation", "Selected epoch"),
           col = c(colors[1:2], "black"), lty = c(1, 1, 3), bty = "n")
  })
  for (page in seq_len(ceiling(length(record$columns) / 4))) {
    ids <- seq.int((page - 1) * 4 + 1, min(page * 4, length(record$columns)))
    india_png(paste0(prefix, "_forecasts_", page, ".png"), function() {
      par(mfrow = c(2, 2), mar = c(5, 5, 4, 1))
      sample <- record$sample
      back <- min(48, nrow(sample$past)); xp <- seq.int(-back + 1, 0)
      xf <- seq_len(record$horizon)
      for (j in ids) {
        past <- tail(sample$past[, j], back); actual <- sample$actual[, j]
        predicted <- sample$prediction[, j]
        ylim <- range(c(past, actual, predicted), finite = TRUE)
        plot(c(xp, xf), c(past, actual), type = "l", col = "#111827", ylim = ylim,
          xlab = paste("Steps from first test forecast origin (", record$unit, ")"),
          ylab = paste(record$columns[j], "(original units)"), main = india_title(record), cex.main = .8)
        lines(xf, predicted, col = colors[1], lwd = 2); abline(v = .5, lty = 3)
        legend("topleft", c("Observed", "TimeKAN"), col = c("#111827", colors[1]),
               lty = 1, bty = "n", cex = .8)
      }
    })
  }
}

india_collect <- function(directory) {
  paths <- list.files(directory, "_complete[.]rds$", full.names = TRUE)
  if (!length(paths)) return(invisible(NULL))
  records <- lapply(paths, readRDS)
  records <- records[order(vapply(records, function(r) match(r$kind, c("air", "climate")), integer(1)),
                           vapply(records, function(r) r$horizon, numeric(1)))]
  frame <- do.call(rbind, lapply(records, function(r)
    data.frame(experiment = r$name, variables = length(r$columns), r$metrics$overall)))
  write.csv(frame, file.path(directory, "multivariate_summary.csv"), row.names = FALSE)
  india_png(file.path(directory, "all_datasets_comparison.png"), function() {
    par(mfrow = c(length(records), 2), mar = c(4, 5, 3, 1))
    for (r in records) for (metric in c("mse_standardized", "mae_standardized")) {
      f <- r$metrics$overall
      bp <- barplot(f[[metric]], names.arg = c("TimeKAN", "Persistence", "Seasonal-naive"), las = 1,
        col = c("#2563eb", "#64748b", "#d97706"),
        ylim = c(0, max(f[[metric]]) * 1.18),
        ylab = india_metric_label(metric), main = india_title(r), cex.main = .85)
      text(bp, f[[metric]], sprintf("%.3f", f[[metric]]), pos = 3, cex = .8)
    }
  }, height = max(1000, 500 * length(records)))
  invisible(frame)
}

india_backup <- function(opt) {
  if (!nzchar(opt$backup)) return(invisible(NULL))
  dest <- file.path(opt$backup, "results/india/timekan")
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  figure_source <- "figures/india/timekan"
  if (dir.exists(figure_source)) {
    figure_destination <- file.path(opt$backup, figure_source)
    dir.create(figure_destination, recursive = TRUE, showWarnings = FALSE)
    file.copy(list.files(figure_source, full.names = TRUE), figure_destination, overwrite = TRUE)
  }
  files <- list.files(opt$results_dir, full.names = TRUE)
  if (length(files) && !all(file.copy(files, dest, overwrite = TRUE))) warning("Incomplete backup")
}

india_same_config <- function(a, b) {
  ignored <- c("task", "model", "device", "air_csv", "climate_csv", "download", "backup",
               "results_dir", "figures_dir", "timekan_dir", "reference_dir")
  keys <- union(setdiff(names(a), ignored), setdiff(names(b), ignored))
  all(vapply(keys, function(k) isTRUE(all.equal(a[[k]], b[[k]])), logical(1)))
}

india_train <- function(kind, horizon, opt, d, path) {
  name <- paste0("Delhi_", kind, "_multivariate_F", horizon, "_seed", opt$seed)
  prefix <- file.path(opt$results_dir, name)
  marker <- paste0(prefix, "_complete.rds")
  if (file.exists(marker)) {
    old <- readRDS(marker)
    if (!india_same_config(old$config, opt) || !identical(old$source_md5, d$source_md5))
      stop("Completed run has different options/data; use a new --results_dir")
    cat("Already complete: ", name, "\n"); india_plots(old, prefix); return(invisible(old))
  }
  starts <- lapply(c(train = "train", val = "val", test = "test"), function(part)
    india_starts(d, part, opt$seq_len, horizon))
  DS <- india_dataset_class()
  loaders <- lapply(names(starts), function(part) dataloader(
    DS(d, starts[[part]], opt$seq_len, horizon), batch_size = opt$batch_size,
    shuffle = part == "train"))
  names(loaders) <- names(starts)
  device <- if (opt$device == "auto") if (cuda_is_available()) "cuda" else "cpu" else opt$device
  if (device == "cuda" && !cuda_is_available()) stop("CUDA requested but unavailable")
  set.seed(opt$seed); torch_manual_seed(opt$seed)
  model <- timekan(opt$seq_len, horizon, opt$d_model, 3, 2, 2, 1, 0)
  model$to(device = device); optimizer <- optim_adam(model$parameters, lr = opt$lr)
  best <- Inf; state <- NULL; bad <- 0; best_epoch <- NA_integer_
  history <- data.frame(); start_time <- Sys.time()
  cat("\n", name, " | ", length(d$columns), " variables: ", paste(d$columns, collapse = ", "),
      " | ", device, "\n", sep = "")
  for (epoch in seq_len(opt$epochs)) {
    model$train(); total <- 0; count <- 0
    coro::loop(for (b in loaders$train) {
      optimizer$zero_grad()
      x <- b$x$to(device = device); y <- b$y$to(device = device); m <- b$mask$to(device = device)
      loss <- india_loss(model(x), y, m); loss$backward(); optimizer$step()
      observed <- m$sum()$item(); total <- total + loss$item() * observed; count <- count + observed
    })
    val <- india_val(model, loaders$val, device)
    history <- rbind(history, data.frame(epoch = epoch, train_mse = total / count, val_mse = val))
    cat(sprintf("epoch %d | train MSE %.5f | val MSE %.5f\n", epoch, total / count, val))
    if (!is.finite(val)) stop("Nonfinite validation loss")
    if (val < best) {
      best <- val; state <- lapply(model$state_dict(), function(t) t$clone())
      bad <- 0; best_epoch <- epoch
    } else { bad <- bad + 1; if (bad >= opt$patience) break }
  }
  model$load_state_dict(state)
  metrics <- india_metrics(model, loaders$test, device, d, opt$seq_len, horizon)
  i <- starts$test[1]; past_ids <- i:(i + opt$seq_len - 1)
  future_ids <- (i + opt$seq_len):(i + opt$seq_len + horizon - 1)
  x <- torch_tensor(d$x[past_ids, , drop = FALSE], dtype = torch_float())$
    unsqueeze(1)$to(device = device)
  prediction <- with_no_grad({ as_array(model(x)$to(device = "cpu")) })
  prediction <- matrix(prediction, nrow = horizon, ncol = length(d$columns))
  prediction <- sweep(sweep(prediction, 2, d$sdev, "*"), 2, d$mu, "+")
  record <- list(name = name, kind = kind, horizon = horizon, config = opt,
    unit = if (kind == "air") "hours" else "days", columns = d$columns,
    source_path = path, source_md5 = d$source_md5, metrics = metrics,
    mu = d$mu, sd = d$sdev, train_coverage = d$train_coverage, dropped = d$dropped,
    split_end_times = as.character(d$time[d$ends]), split_end_rows = d$ends,
    missing_fraction = colMeans(!d$observed), windows = lengths(starts),
    history = history, best_epoch = best_epoch,
    minutes = as.numeric(difftime(Sys.time(), start_time, units = "mins")),
    sample = list(past = d$raw[past_ids, , drop = FALSE],
                  actual = d$raw[future_ids, , drop = FALSE], prediction = prediction))
  torch_save(model, paste0(prefix, ".pt"))
  for (key in names(metrics)) write.csv(metrics[[key]], paste0(prefix, "_", key, ".csv"), row.names = FALSE)
  write.csv(history, paste0(prefix, "_training.csv"), row.names = FALSE)
  write.csv(d$dropped, paste0(prefix, "_dropped_variables.csv"), row.names = FALSE)
  capture.output(dput(record[setdiff(names(record), c("sample", "metrics", "history"))]),
                 file = paste0(prefix, "_provenance.R"))
  india_plots(record, prefix)
  saveRDS(record, marker) # Completion marker is written only after evaluation and plotting.
  print(metrics$overall, row.names = FALSE)
  cat("MULTIVARIATE_COMPLETE ", name, "\n", sep = "")
  invisible(record)
}

india_main <- function(opt) {
  dir.create(opt$results_dir, recursive = TRUE, showWarnings = FALSE)
  if (opt$task == "plots") {
    paths <- list.files(opt$results_dir, "_complete[.]rds$", full.names = TRUE)
    if (!length(paths)) stop("No completed multivariate experiments to plot")
    for (p in paths) india_plots(readRDS(p), sub("_complete[.]rds$", "", p))
    india_collect(opt$results_dir); india_backup(opt)
    return(invisible(NULL))
  }
  if (!file.exists("timekan.R")) stop("Run from the repository root containing timekan.R")
  source("timekan.R", local = .GlobalEnv)
  if (opt$task == "self_test") return(india_torch_test())
  if (nzchar(opt$backup) && !dir.exists(dirname(opt$backup)))
    stop("Backup parent does not exist; mount Google Drive first")
  for (kind in if (opt$task == "all") c("air", "climate") else opt$task) {
    path <- india_raw_path(kind, opt)
    d <- india_prepare(path, kind, opt$seq_len, opt$min_train_coverage)
    for (horizon in if (kind == "air") c(24, 96) else c(7, 30)) {
      tryCatch({ india_train(kind, horizon, opt, d, path); india_collect(opt$results_dir) },
        finally = india_backup(opt))
    }
  }
}

if (sys.nframe() == 0L) india_main(india_args())
