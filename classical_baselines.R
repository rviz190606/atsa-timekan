# Compare the completed univariate runs against classical models on identical
# forecast origins, targets and information windows. Uses base R stats only.
source("extensions12.R")

fit_classical <- function(train, family, order) {
  if (family == "AR") {
    p <- order[1]
    design <- embed(train, p + 1)
    design <- design[complete.cases(design), , drop = FALSE]
    x <- cbind(1, design[, -1, drop = FALSE])
    fit <- lm.fit(x, design[, 1])
    if (fit$rank != ncol(x) || any(!is.finite(fit$coefficients))) stop("Singular AR fit")
    return(list(family = family, order = order, coefficients = fit$coefficients))
  }
  fit <- suppressWarnings(stats::arima(train, order = order, method = "ML",
    include.mean = order[2] == 0, optim.control = list(maxit = 200)))
  if (fit$code != 0 || any(!is.finite(fit$coef))) stop("ARIMA-family fit did not converge")
  list(family = family, order = order, coefficients = fit$coef)
}

predict_classical <- function(fit, history, horizon) {
  if (fit$family == "AR") {
    p <- fit$order[1]
    buffer <- history
    predictions <- numeric(horizon)
    for (h in seq_len(horizon)) {
      value <- sum(c(1, rev(tail(buffer, p))) * fit$coefficients)
      predictions[h] <- value
      buffer <- c(buffer, value)
    }
    return(predictions)
  }
  # All coefficients are fixed: this filters the latest history window to infer
  # innovations/state, without re-estimating parameters at each forecast origin.
  filtered <- suppressWarnings(stats::arima(history, order = fit$order,
    fixed = fit$coefficients, transform.pars = FALSE, method = "ML",
    include.mean = fit$order[2] == 0))
  as.numeric(suppressWarnings(predict(filtered, n.ahead = horizon, se.fit = FALSE)))
}

score_classical <- function(fit, part, starts, history, horizon) {
  squared <- 0; absolute <- 0; count <- 0
  for (s in starts) {
    x <- part[s:(s + history - 1), 1]
    y <- part[(s + history):(s + history + horizon - 1), 1]
    pred <- predict_classical(fit, x, horizon)
    if (length(pred) != horizon || any(!is.finite(pred))) stop("Non-finite forecast")
    squared <- squared + sum((pred - y)^2)
    absolute <- absolute + sum(abs(pred - y))
    count <- count + horizon
  }
  c(mse = squared / count, mae = absolute / count)
}

compare_one <- function(options_file, output_dir) {
  opt <- dget(options_file)
  prefix <- file.path(opt$results_dir, opt$name)
  completed <- paste0(prefix, "_summary.csv")
  if (!file.exists(completed)) return(invisible(NULL))
  dest <- file.path(output_dir, paste0(opt$name, "_classical.csv"))
  if (file.exists(dest)) {
    cat("Already compared:", opt$name, "\n"); return(invisible(NULL))
  }
  d <- prepare_data(opt)
  train <- d$train[, 1]
  train[!attr(d$train, "observed")[, 1]] <- NA_real_
  val <- valid_starts(d$val, opt$seq_len, opt$pred_len)
  test <- valid_starts(d$test, opt$seq_len, opt$pred_len)
  # A fixed validation-origin budget controls order-search cost. ALL eligible
  # test origins are scored, matching the completed TimeKAN experiment.
  val <- val[unique(round(seq(1, length(val), length.out = min(64, length(val)))))]
  grids <- list(
    AR = list(c(4, 0, 0), c(12, 0, 0), c(24, 0, 0)),
    MA = list(c(0, 0, 1), c(0, 0, 2), c(0, 0, 4)),
    ARMA = list(c(1, 0, 1), c(2, 0, 1), c(2, 0, 2)),
    ARIMA = list(c(1, 1, 0), c(0, 1, 1), c(1, 1, 1)))
  # Cache fits across horizons with identical data and training intervals.
  cache_key <- gsub("[^A-Za-z0-9_-]", "_", paste(opt$experiment, unname(tools::md5sum(opt$csv)),
    opt$target, opt$start_date, opt$end_date, length(train), sep = "_"))
  cache_file <- file.path(output_dir, paste0(cache_key, "_fits.rds"))
  cache <- if (file.exists(cache_file)) readRDS(cache_file) else list()
  scores <- list(); selection <- list()
  for (family in names(grids)) {
    cat("Selecting", family, "for", opt$name, "\n")
    best <- NULL; best_val <- Inf
    for (order in grids[[family]]) {
      key <- paste(family, paste(order, collapse = ","))
      candidate <- tryCatch({
        if (is.null(cache[[key]])) {
          cache[[key]] <- fit_classical(train, family, order)
          saveRDS(cache, cache_file)
        }
        fit <- cache[[key]]
        metric <- score_classical(fit, d$val, val, opt$seq_len, opt$pred_len)
        list(fit = fit, mse = metric["mse"])
      }, error = function(e) {
        cat("Candidate rejected:", key, conditionMessage(e), "\n"); NULL
      })
      if (!is.null(candidate) && candidate$mse < best_val) {
        best <- candidate$fit; best_val <- candidate$mse
      }
    }
    if (is.null(best)) stop("No valid candidate in family ", family)
    cat("Selected", family, paste(best$order, collapse = ","),
        "validation MSE", best_val, "test origins", length(test), "\n")
    metric <- score_classical(best, d$test, test, opt$seq_len, opt$pred_len)
    scores[[family]] <- data.frame(experiment = opt$name, target = opt$target,
      horizon = opt$pred_len, seed = opt$seed, model = family,
      mse_standardized = metric["mse"], mae_standardized = metric["mae"],
      mse_original_units = metric["mse"] * d$sdev[1]^2,
      mae_original_units = metric["mae"] * d$sdev[1], row.names = NULL)
    selection[[family]] <- data.frame(model = family, p = best$order[1],
      d = best$order[2], q = best$order[3], validation_mse = best_val,
      validation_origins = length(val), test_origins = length(test))
  }
  result <- do.call(rbind, scores)
  write.csv(do.call(rbind, selection), file.path(output_dir,
    paste0(opt$name, "_selected_orders.csv")), row.names = FALSE)
  write.csv(result, dest, row.names = FALSE)
  print(result, row.names = FALSE)
  cat("CLASSICAL_COMPARISON_COMPLETE", opt$name, "\n")
}

classical_self_tests <- function() {
  set.seed(123)
  train <- as.numeric(arima.sim(list(ar = .7), 300))
  for (spec in list(list("AR", c(4, 0, 0)), list("MA", c(0, 0, 1)),
                   list("ARMA", c(1, 0, 1)), list("ARIMA", c(1, 1, 0)))) {
    fit <- fit_classical(train, spec[[1]], spec[[2]])
    before <- fit$coefficients
    pred <- predict_classical(fit, tail(train, 96), 24)
    stopifnot(length(pred) == 24, all(is.finite(pred)), identical(before, fit$coefficients))
    if (spec[[1]] == "MA") stopifnot(max(abs(pred[2:24] - before["intercept"])) < 1e-8)
  }
  cat("CLASSICAL_FIXED_PARAMETERS_AND_FORECAST_TESTS_OK\n")
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if ("--self_test" %in% args) classical_self_tests() else {
    output_dir <- "results/univariate_baselines"
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    files <- list.files("results/extensions12", pattern = "_options[.]R$", full.names = TRUE)
    for (file in files) compare_one(file, output_dir)
  }
}
