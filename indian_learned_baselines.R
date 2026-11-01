# Native R torch ports of DLinear and iTransformer; matched Indian-data evaluation.
# Source architectures: cure-lab/LTSF-Linear/models/DLinear.py and thuml/iTransformer.
# See docs/india/LEARNED_BASELINES.md for attribution, settings, protocol and limitations.
source("run_india_multivariate.R")

learned_defaults <- modifyList(india_defaults, list(model = "both", batch_size = 64,
  d_model = 128, d_ff = 256, heads = 8, encoder_layers = 2, dropout = 0.1,
  results_dir = "results/india/learned_baselines", timekan_dir = "results/india/timekan",
  reference_dir = "results/india/learned_baselines/port_checks"))

learned_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  opt <- learned_defaults
  if (length(args) %% 2) stop("Use --option value pairs")
  if (length(args)) for (i in seq(1, length(args), 2)) {
    key <- sub("^--", "", args[i]); if (!key %in% names(opt)) stop("Unknown option: ", key)
    opt[[key]] <- if (is.numeric(opt[[key]])) as.numeric(args[i + 1]) else args[i + 1]
  }
  stopifnot(opt$task %in% c("all", "air", "climate", "plots", "self_test"),
    opt$model %in% c("both", "DLinear", "iTransformer"), opt$seq_len >= 25,
    opt$d_model %% opt$heads == 0, opt$encoder_layers >= 1,
    opt$epochs >= 1, opt$patience >= 1, opt$dropout >= 0, opt$dropout < 1)
  opt
}

init_learned_models <- function() {
  dlinear_model <<- nn_module("DLinear",
    initialize = function(history, horizon) {
      self$average <- nn_avg_pool1d(25, stride = 1)
      self$seasonal <- nn_linear(history, horizon)
      self$trend <- nn_linear(history, horizon)
    },
    decompose = function(x) {
      B <- x$size(1); T <- x$size(2); N <- x$size(3)
      front <- x[, 1, , drop = FALSE]$expand(c(B, 12, N))
      end <- x[, T, , drop = FALSE]$expand(c(B, 12, N))
      trend <- self$average(torch_cat(list(front, x, end), dim = 2)$transpose(2, 3))$transpose(2, 3)
      list(seasonal = x - trend, trend = trend)
    },
    forward = function(x) {
      parts <- self$decompose(x)
      (self$seasonal(parts$seasonal$transpose(2, 3)) +
        self$trend(parts$trend$transpose(2, 3)))$transpose(2, 3)
    })
  inverted_encoder_layer <<- nn_module("InvertedEncoderLayer",
    initialize = function(dim, ff, heads, dropout) {
      self$heads <- heads; self$width <- dim / heads; self$dim <- dim
      self$q <- nn_linear(dim, dim); self$k <- nn_linear(dim, dim)
      self$v <- nn_linear(dim, dim); self$out <- nn_linear(dim, dim)
      self$conv1 <- nn_conv1d(dim, ff, 1); self$conv2 <- nn_conv1d(ff, dim, 1)
      self$norm1 <- nn_layer_norm(dim); self$norm2 <- nn_layer_norm(dim)
      self$drop <- nn_dropout(dropout); self$attention_drop <- nn_dropout(dropout)
    },
    forward = function(x) {
      B <- x$size(1); N <- x$size(2)
      divide <- function(y) y$reshape(c(B, N, self$heads, self$width))$permute(c(1, 3, 2, 4))
      q <- divide(self$q(x)); k <- divide(self$k(x)); v <- divide(self$v(x))
      a <- self$attention_drop(nnf_softmax(torch_matmul(q, k$transpose(3, 4)) / sqrt(self$width), dim = 4))
      joined <- torch_matmul(a, v)$permute(c(1, 3, 2, 4))$contiguous()$reshape(c(B, N, self$dim))
      x <- self$norm1(x + self$drop(self$out(joined)))
      y <- self$drop(nnf_gelu(self$conv1(x$transpose(2, 3))))
      y <- self$drop(self$conv2(y)$transpose(2, 3))
      self$norm2(x + y)
    })
  itransformer_model <<- nn_module("iTransformer",
    initialize = function(history, horizon, dim = 128, ff = 256, heads = 8, layers = 2, dropout = .1) {
      self$embedding <- nn_linear(history, dim); self$drop <- nn_dropout(dropout)
      self$encoders <- nn_module_list(lapply(seq_len(layers), function(i)
        inverted_encoder_layer(dim, ff, heads, dropout)))
      self$norm <- nn_layer_norm(dim); self$projector <- nn_linear(dim, horizon)
    },
    forward = function(x) {
      means <- x$mean(dim = 2, keepdim = TRUE)$detach()
      centered <- x - means
      sd <- torch_sqrt(centered$var(dim = 2, keepdim = TRUE, unbiased = FALSE) + 1e-5)
      z <- self$drop(self$embedding((centered / sd)$transpose(2, 3)))
      for (i in seq_along(self$encoders)) z <- self$encoders[[i]](z)
      self$projector(self$norm(z))$transpose(2, 3) * sd + means
    })
}

learned_factory <- function(method, history, horizon, opt) {
  if (method == "DLinear") dlinear_model(history, horizon)
  else itransformer_model(history, horizon, opt$d_model, opt$d_ff, opt$heads,
                          opt$encoder_layers, opt$dropout)
}

learned_self_test <- function(directory) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  india_torch_test()
  torch_manual_seed(19)
  opt <- modifyList(learned_defaults, list(d_model = 8, d_ff = 16, heads = 2,
                                         encoder_layers = 2, dropout = 0))
  for (method in c("DLinear", "iTransformer")) {
    model <- learned_factory(method, 8, 4, opt); model$eval()
    x <- torch_randn(2, 8, 3)
    y <- model(x); y$pow(2)$mean()$backward()
    stopifnot(identical(as.integer(y$size()), c(2L, 4L, 3L)),
      all(vapply(model$parameters, function(p)
        !is.null(p$grad) && torch_isfinite(p$grad)$all()$item(), logical(1))))
    changed <- x$clone(); changed[, 1:3, 2] <- changed[, 1:3, 2] + 3
    difference <- (model(changed)[, , 1] - y[, , 1])$abs()$max()$item()
    if (method == "DLinear") {
      stopifnot(difference == 0)
      parts <- model$decompose(x)
      stopifnot(torch_allclose(parts$seasonal + parts$trend, x))
      constant <- torch_ones(2, 8, 3)
      stopifnot(torch_allclose(model$decompose(constant)$trend, constant))
    } else stopifnot(difference > 1e-8)
    # Identical deterministic tensors are assigned in the official Python models.
    # Export input gradients too, so a numerically correct forward alone is insufficient.
    with_no_grad({ for (p in model$parameters)
      p$copy_(torch_linspace(-.1, .1, steps = p$numel())$reshape(p$size())) })
    x <- torch_linspace(-1, 2, steps = 48)$reshape(c(2, 8, 3))$requires_grad_(TRUE)
    y <- model(x); y$pow(2)$sum()$backward()
    write.table(as.numeric(as_array(y)), file.path(directory, paste0(method, "_output.csv")),
      row.names = FALSE, col.names = FALSE, sep = ",")
    write.table(as.numeric(as_array(x$grad)), file.path(directory, paste0(method, "_input_gradient.csv")),
      row.names = FALSE, col.names = FALSE, sep = ",")
  }
  cat("LEARNED_BASELINES_SHAPES_GRADIENTS_DECOMPOSITION_AND_CHANNEL_TESTS_OK\n")
}

learned_match <- function(reference, d, horizon, opt) {
  stopifnot(identical(reference$source_md5, d$source_md5),
    identical(reference$columns, d$columns), identical(reference$split_end_rows, d$ends),
    isTRUE(all.equal(reference$mu, d$mu)), isTRUE(all.equal(reference$sd, d$sdev)))
  # Match the actual TimeKAN budget; model dimensions are architecture-specific.
  for (key in c("seq_len", "seed", "epochs", "patience", "batch_size", "lr", "min_train_coverage"))
    if (!identical(as.numeric(reference$config[[key]]), as.numeric(opt[[key]])))
      stop("TimeKAN configuration mismatch: ", key)
  counts <- sapply(c(train = "train", val = "val", test = "test"), function(part)
    length(india_starts(d, part, opt$seq_len, horizon)))
  stopifnot(identical(unname(counts), unname(reference$windows)))
}

learned_train <- function(method, kind, horizon, opt, d, reference) {
  case <- paste0("Delhi_", kind, "_multivariate_F", horizon, "_seed", opt$seed)
  name <- paste0(case, "_", method); prefix <- file.path(opt$results_dir, name)
  marker <- paste0(prefix, "_complete.rds")
  learned_match(reference, d, horizon, opt)
  if (file.exists(marker)) {
    old <- readRDS(marker)
    if (!india_same_config(old$config, opt) || !identical(old$source_md5, d$source_md5))
      stop("Existing baseline has different settings/data; choose another --results_dir")
    cat("Already complete: ", name, "\n"); return(invisible(old))
  }
  starts <- lapply(c(train = "train", val = "val", test = "test"), function(part)
    india_starts(d, part, opt$seq_len, horizon))
  DS <- india_dataset_class()
  loaders <- lapply(names(starts), function(part) dataloader(DS(d, starts[[part]], opt$seq_len, horizon),
    batch_size = opt$batch_size, shuffle = part == "train")); names(loaders) <- names(starts)
  device <- if (opt$device == "auto") if (cuda_is_available()) "cuda" else "cpu" else opt$device
  if (device == "cuda" && !cuda_is_available()) stop("CUDA requested but unavailable")
  set.seed(opt$seed); torch_manual_seed(opt$seed)
  model <- learned_factory(method, opt$seq_len, horizon, opt); model$to(device = device)
  parameters <- sum(vapply(model$parameters, function(p) p$numel(), numeric(1)))
  optimizer <- optim_adam(model$parameters, lr = opt$lr)
  best <- Inf; state <- NULL; bad <- 0; best_epoch <- NA_integer_
  history <- data.frame(); start_time <- Sys.time()
  cat("\nSTART ", name, " | ", parameters, " parameters | ", length(d$columns), " variables | ", device, "\n", sep = "")
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
  for (key in names(metrics)) metrics[[key]]$model[metrics[[key]]$model == "TimeKAN"] <- method
  stopifnot(metrics$overall$observed_targets[1] == reference$metrics$overall$observed_targets[1])
  # Recomputed naive scores must match the frozen TimeKAN record exactly.
  for (metric in c("mse_standardized", "mae_standardized"))
    stopifnot(max(abs(metrics$overall[[metric]][2:3] - reference$metrics$overall[[metric]][2:3])) < 1e-7)
  i <- starts$test[1]; past <- i:(i + opt$seq_len - 1)
  future <- (i + opt$seq_len):(i + opt$seq_len + horizon - 1)
  x <- torch_tensor(d$x[past, , drop = FALSE], dtype = torch_float())$unsqueeze(1)$to(device = device)
  prediction <- with_no_grad({ as_array(model(x)$to(device = "cpu")) })
  prediction <- matrix(prediction, nrow = horizon, ncol = length(d$columns))
  prediction <- sweep(sweep(prediction, 2, d$sdev, "*"), 2, d$mu, "+")
  record <- list(name = name, case = case, method = method, kind = kind, horizon = horizon,
    config = opt, columns = d$columns, source_md5 = d$source_md5, metrics = metrics,
    mu = d$mu, sd = d$sdev, windows = lengths(starts), split_end_rows = d$ends,
    history = history, best_epoch = best_epoch, parameters = parameters,
    minutes = as.numeric(difftime(Sys.time(), start_time, units = "mins")),
    sample = list(past = d$raw[past, , drop = FALSE], actual = d$raw[future, , drop = FALSE], prediction = prediction))
  torch_save(model, paste0(prefix, ".pt"))
  for (key in names(metrics)) write.csv(metrics[[key]], paste0(prefix, "_", key, ".csv"), row.names = FALSE)
  write.csv(history, paste0(prefix, "_training.csv"), row.names = FALSE)
  saveRDS(record, marker)
  print(metrics$overall[1, ], row.names = FALSE)
  cat("LEARNED_BASELINE_COMPLETE ", name, "\n", sep = "")
  invisible(record)
}

learned_comparison <- function(opt) {
  files <- list.files(opt$results_dir, "_complete[.]rds$", full.names = TRUE)
  if (!length(files)) return(invisible(NULL))
  records <- lapply(files, readRDS)
  cases <- unique(vapply(records, function(r) r$case, character(1)))
  cases <- cases[order(!grepl("_air_", cases), as.numeric(sub(".*_F([0-9]+)_.*", "\\1", cases)))]
  order_models <- c("TimeKAN", "DLinear", "iTransformer", "Persistence", "SeasonalNaive")
  colors <- c("#2563eb", "#16a34a", "#9333ea", "#64748b", "#d97706")
  combined <- list(); per_variable <- list(); per_horizon <- list()
  for (case in cases) {
    reference <- readRDS(file.path(opt$timekan_dir, paste0(case, "_complete.rds")))
    current <- Filter(function(r) r$case == case, records)
    frame <- reference$metrics$overall
    variables <- reference$metrics$per_variable; leads <- reference$metrics$by_horizon
    for (r in current) {
      frame <- rbind(frame, r$metrics$overall[r$metrics$overall$model == r$method, ])
      variables <- rbind(variables, r$metrics$per_variable[r$metrics$per_variable$model == r$method, ])
      leads <- rbind(leads, r$metrics$by_horizon[r$metrics$by_horizon$model == r$method, ])
    }
    frame <- frame[order(match(frame$model, order_models)), ]
    combined[[case]] <- data.frame(experiment = case, frame)
    per_variable[[case]] <- data.frame(experiment = case, variables)
    per_horizon[[case]] <- data.frame(experiment = case, leads)
    short <- sub("_seed[0-9]+$", "", case)
    india_png(file.path(opt$results_dir, paste0(short, "_per_variable.png")), function() {
      par(mfrow = c(2, 1), mar = c(9, 5, 3, 1))
      methods <- frame$model
      for (metric in c("mse_standardized", "mae_standardized")) {
        values <- t(vapply(methods, function(m) variables[variables$model == m, metric], numeric(length(reference$columns))))
        barplot(values, beside = TRUE, names.arg = reference$columns, las = 2, cex.names = .7,
          col = colors[match(methods, order_models)], ylim = c(0, max(values) * 1.3),
          ylab = india_metric_label(metric), main = india_title(reference))
        legend("topright", methods, fill = colors[match(methods, order_models)], bty = "n", horiz = TRUE, cex = .7)
      }
    }, width = 2000, height = 1600)
    india_png(file.path(opt$results_dir, paste0(short, "_horizon_errors.png")), function() {
      par(mfrow = c(1, 2), mar = c(5, 5, 4, 1)); methods <- frame$model
      for (metric in c("mse_standardized", "mae_standardized")) {
        values <- vapply(methods, function(m) leads[leads$model == m, metric], numeric(reference$horizon))
        matplot(seq_len(reference$horizon), values, type = "l", lty = 1,
          col = colors[match(methods, order_models)], xlab = paste("Forecast lead (", reference$unit, ")"),
          ylab = india_metric_label(metric), main = india_title(reference), cex.main = .8)
        legend("topleft", methods, col = colors[match(methods, order_models)], lty = 1, bty = "n", cex = .7)
      }
    })
    for (page in seq_len(ceiling(length(reference$columns) / 4))) {
      ids <- seq.int((page - 1) * 4 + 1, min(page * 4, length(reference$columns)))
      india_png(file.path(opt$results_dir, paste0(short, "_forecast_comparison_", page, ".png")), function() {
        par(mfrow = c(2, 2), mar = c(5, 5, 4, 1))
        for (j in ids) {
          back <- min(48, nrow(reference$sample$past)); xf <- seq_len(reference$horizon)
          past <- tail(reference$sample$past[, j], back); truth <- reference$sample$actual[, j]
          predictions <- c(list(reference$sample$prediction[, j]), lapply(current, function(r) r$sample$prediction[, j]))
          labels <- c("TimeKAN", vapply(current, function(r) r$method, character(1)))
          plot(c(seq.int(-back + 1, 0), xf), c(past, truth), type = "l", col = "#111827",
            ylim = range(c(past, truth, unlist(predictions)), finite = TRUE),
            xlab = paste("Steps from first test origin (", reference$unit, ")"),
            ylab = paste(reference$columns[j], "(original units)"), main = india_title(reference), cex.main = .8)
          for (i in seq_along(predictions)) lines(xf, predictions[[i]], col = colors[match(labels[i], order_models)], lwd = 2)
          abline(v = .5, lty = 3)
          legend("topleft", c("Observed", labels), col = c("#111827", colors[match(labels, order_models)]), lty = 1, bty = "n", cex = .65)
        }
      })
    }
  }
  table <- do.call(rbind, combined)
  write.csv(table, file.path(opt$results_dir, "all_models.csv"), row.names = FALSE)
  write.csv(do.call(rbind, per_variable), file.path(opt$results_dir, "all_models_per_variable.csv"), row.names = FALSE)
  write.csv(do.call(rbind, per_horizon), file.path(opt$results_dir, "all_models_by_horizon.csv"), row.names = FALSE)
  india_png(file.path(opt$results_dir, "learned_baseline_comparison.png"), function() {
    par(mfrow = c(length(cases), 2), mar = c(5, 5, 3, 1))
    for (case in cases) for (metric in c("mse_standardized", "mae_standardized")) {
      frame <- combined[[case]]
      reference <- readRDS(file.path(opt$timekan_dir, paste0(case, "_complete.rds")))
      labels <- c(TimeKAN = "TimeKAN", DLinear = "DLinear", iTransformer = "iTransformer", Persistence = "Persistence", SeasonalNaive = "Seasonal-naive")
      bp <- barplot(frame[[metric]], names.arg = labels[frame$model], cex.names = .7,
        col = colors[match(frame$model, order_models)], ylim = c(0, max(frame[[metric]]) * 1.18),
        ylab = india_metric_label(metric), main = india_title(reference), cex.main = .8)
      text(bp, frame[[metric]], sprintf("%.3f", frame[[metric]]), pos = 3, cex = .7)
    }
  }, width = 2200, height = max(1100, length(cases) * 550))
  for (r in records) india_png(file.path(opt$results_dir, paste0(r$name, "_training.png")), function() {
    matplot(r$history$epoch, r$history[, c("train_mse", "val_mse")], type = "l", lty = 1,
      col = c("#2563eb", "#d97706"), xlab = "Epoch", ylab = "Observed-target standardized MSE", main = r$name)
    abline(v = r$best_epoch, lty = 3)
    legend("topright", c("Train", "Validation", "Selected epoch"), col = c("#2563eb", "#d97706", "black"), lty = c(1, 1, 3), bty = "n")
  })
  print(table, row.names = FALSE)
  invisible(table)
}

learned_backup <- function(opt) {
  if (!nzchar(opt$backup)) return(invisible(NULL))
  target <- file.path(opt$backup, "results/india/learned_baselines")
  dir.create(target, recursive = TRUE, showWarnings = FALSE)
  files <- list.files(opt$results_dir, full.names = TRUE)
  if (length(files) && !all(file.copy(files, target, recursive = TRUE, overwrite = TRUE))) warning("Incomplete backup")
  figure_source <- "figures/india/learned_baselines"
  if (dir.exists(figure_source)) {
    destination <- file.path(opt$backup, figure_source)
    dir.create(destination, recursive = TRUE, showWarnings = FALSE)
    file.copy(list.files(figure_source, full.names = TRUE), destination, overwrite = TRUE)
  }
  file.copy("indian_learned_baselines.R", opt$backup, overwrite = TRUE)
}

learned_main <- function(opt) {
  dir.create(opt$results_dir, recursive = TRUE, showWarnings = FALSE)
  if (opt$task == "plots") { learned_comparison(opt); learned_backup(opt); return(invisible(NULL)) }
  source("timekan.R", local = .GlobalEnv); init_learned_models()
  if (opt$task == "self_test") return(learned_self_test(opt$reference_dir))
  for (kind in if (opt$task == "all") c("air", "climate") else opt$task) {
    path <- india_raw_path(kind, opt)
    d <- india_prepare(path, kind, opt$seq_len, opt$min_train_coverage)
    for (horizon in if (kind == "air") c(24, 96) else c(7, 30)) {
      case <- paste0("Delhi_", kind, "_multivariate_F", horizon, "_seed", opt$seed)
      reference <- readRDS(file.path(opt$timekan_dir, paste0(case, "_complete.rds")))
      for (method in if (opt$model == "both") c("DLinear", "iTransformer") else opt$model)
        tryCatch({ learned_train(method, kind, horizon, opt, d, reference) }, finally = learned_backup(opt))
    }
  }
  learned_comparison(opt); learned_backup(opt)
  cat("ALL_EIGHT_LEARNED_BASELINE_RUNS_COMPLETE\n")
}

if (sys.nframe() == 0L) learned_main(learned_args())
