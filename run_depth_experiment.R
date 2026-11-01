# Single-seed depth sensitivity study for the project's native R TimeKAN.
# Run from the repository root. No changes to the original model are required.
depth_defaults <- list(task = "train", datasets = "ETTh1,ETTh2,ETTm1",
  depths = "1,2,3,4,5,10,20", data_dir = "data", results_dir = "results/depth",
  figures_dir = "figures/depth", backup = "", device = "auto", seed = 0,
  epochs = 10, patience = 3, batch_size = 128, lr = 0.001, seq_len = 96,
  pred_len = 96, d_model = 16, levels = 3, window = 2, base_order = 1,
  cfd_mode = "paper", threads = 2)

depth_args <- function(args = commandArgs(TRUE)) {
  opt <- depth_defaults
  if (length(args) %% 2) stop("Use --option value pairs")
  if (length(args)) for (i in seq(1, length(args), by = 2)) {
    k <- sub("^--", "", args[i])
    if (!k %in% names(opt)) stop("Unknown option: ", k)
    opt[[k]] <- if (is.numeric(opt[[k]])) as.numeric(args[i+1]) else args[i+1]
  }
  stopifnot(opt$task %in% c("train", "plots", "self_test", "benchmark"),
    opt$epochs >= 1, opt$patience >= 1, opt$batch_size >= 1, opt$lr > 0)
  ds <- strsplit(opt$datasets, ",", fixed = TRUE)[[1]]
  ls <- suppressWarnings(as.numeric(strsplit(opt$depths, ",", fixed = TRUE)[[1]]))
  stopifnot(length(ds) > 0, all(ds %in% c("ETTh1", "ETTh2", "ETTm1", "ETTm2")),
    !anyDuplicated(ds), length(ls) > 0, all(is.finite(ls)), all(ls >= 1),
    all(ls == as.integer(ls)), !anyDuplicated(ls))
  opt
}

depth_tables <- function(opt) {
  files <- list.files(file.path(opt$results_dir, "records"), "_complete.rds$", full.names = TRUE)
  if (length(files)) {
    records <- lapply(files, readRDS)
    rows <- do.call(rbind, lapply(records, `[[`, "summary"))
    curves <- do.call(rbind, lapply(records, `[[`, "history"))
  } else {
    if (!all(file.exists(file.path(opt$results_dir, c("summary.csv", "learning_curves.csv"))))) {
      cat("No completed depth results found.\n"); return(invisible(NULL))
    }
    rows <- read.csv(file.path(opt$results_dir, "summary.csv"))
    curves <- read.csv(file.path(opt$results_dir, "learning_curves.csv"))
  }
  rows <- rows[order(rows$dataset, rows$layers), ]
  write.csv(rows, file.path(opt$results_dir, "summary.csv"), row.names = FALSE)
  write.csv(curves, file.path(opt$results_dir, "learning_curves.csv"), row.names = FALSE)
  if (anyDuplicated(paste(rows$dataset, rows$layers))) stop("Duplicate depth results")
  dir.create(opt$figures_dir, recursive = TRUE, showWarnings = FALSE)
  sets <- unique(rows$dataset); colors <- c("#167a9d", "#df7243", "#59974b", "#8563ae")
  png(file.path(opt$figures_dir, "depth_accuracy_cost.png"), 1500, 1050, res = 145)
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
  for (metric in c("test_mse", "test_mae", "minutes", "params")) {
    plot(range(rows$layers), range(rows[[metric]], finite = TRUE), type = "n",
      xlab = "Number of layers L", ylab = switch(metric, test_mse = "Test MSE",
      test_mae = "Test MAE", minutes = "Train + validation + test (minutes)", params = "Parameters"),
      main = switch(metric, test_mse = "Forecast error (standardized)", test_mae = "Absolute error (standardized)",
        minutes = "Measured runtime", params = "Model size"))
    abline(v = 3, lty = 3, col = "grey65")
    for (j in seq_along(sets)) {
      r <- rows[rows$dataset == sets[j], ]
      lines(r$layers, r[[metric]], type = "b", pch = 15+j, col = colors[j], lwd = 2)
    }
    legend("topleft", sets, col = colors[seq_along(sets)], pch = 15+seq_along(sets), lty = 1, bty = "n", cex = .75)
  }
  dev.off()
  png(file.path(opt$figures_dir, "depth_validation_curves.png"), 1500, 430*length(sets), res = 145)
  par(mfrow = c(length(sets), 1), mar = c(4, 4, 3, 1))
  ls <- sort(unique(curves$layers)); palette <- hcl.colors(length(ls), "Dark 3")
  for (ds in sets) {
    r <- curves[curves$dataset == ds, ]
    plot(range(r$epoch), range(r$val_mse), type = "n", xlab = "Epoch", ylab = "Validation MSE",
      main = paste(ds, "- validation curves (single seed)"))
    for (j in seq_along(ls)) {
      z <- r[r$layers == ls[j], ]; lines(z$epoch, z$val_mse, type = "b", col = palette[j], pch = j)
    }
    legend("topright", paste0("L=", ls), col = palette, lty = 1, pch = seq_along(ls), bty = "n", ncol = 4, cex = .7)
  }
  dev.off()
  invisible(rows)
}

depth_backup <- function(opt) {
  if (!nzchar(opt$backup)) return(invisible(NULL))
  for (folder in c(opt$results_dir, opt$figures_dir)) {
    dest <- file.path(opt$backup, folder)
    dir.create(dest, recursive = TRUE, showWarnings = FALSE)
    files <- list.files(folder, recursive = TRUE, full.names = TRUE, all.files = FALSE)
    for (f in files) {
      target <- file.path(dest, substring(f, nchar(folder)+2))
      dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
      if (!file.copy(f, target, overwrite = TRUE)) stop("Backup failed: ", target)
    }
  }
}

depth_self_test <- function() {
  a <- cbind(1:40, 101:140, 201:240)
  ds <- window_dataset(a, 4, 2); b <- ds$.getbatch(c(1, 3))
  for (j in 1:3) for (i in 1:2) {
    s <- c(1, 3)[i]
    stopifnot(isTRUE(all.equal(as.numeric(b$x[i,,j]), as.numeric(a[s:(s+3),j]))),
      isTRUE(all.equal(as.numeric(b$y[i,,j]), as.numeric(a[(s+4):(s+5),j]))))
  }
  counts <- numeric()
  for (L in c(1, 3, 20)) {
    torch_manual_seed(0)
    m <- timekan(16, 8, 4, 3, 2, L)
    stopifnot(length(m$layers) == L)
    x <- torch_randn(2, 16, 3); y <- m(x)
    stopifnot(identical(as.integer(y$size()), c(2L, 8L, 3L)), torch_isfinite(y)$all()$item())
    y$square()$mean()$backward()
    stopifnot(all(vapply(m$parameters, function(p) !is.null(p$grad) && torch_isfinite(p$grad)$all()$item(), logical(1))))
    counts <- c(counts, sum(vapply(m$parameters, function(p) p$numel(), numeric(1))))
  }
  stopifnot(all(diff(counts) > 0))
  cat("DEPTH_MULTIVARIATE_LAYOUT_LAYER_COUNTS_AND_GRADIENTS_OK\n")
}

depth_run <- function(opt, ds, L) {
  csv <- file.path(opt$data_dir, paste0(ds, ".csv"))
  config <- modifyList(default_options, opt[intersect(names(opt), names(default_options))])
  config$csv <- csv; config$layers <- L
  hash <- c(data = unname(tools::md5sum(csv)), model = unname(tools::md5sum("timekan.R")),
    runner = unname(tools::md5sum("run_depth_experiment.R")))
  key <- sprintf("%s_L%d_seed%d", ds, L, opt$seed)
  path <- file.path(opt$results_dir, "records", paste0(key, "_complete.rds"))
  if (file.exists(path)) {
    old <- readRDS(path)
    compare <- setdiff(names(config), c("csv", "device"))
    if (!identical(old$hash, hash) || !identical(old$config[compare], config[compare]))
      stop("Existing record has different source/data/settings; choose a separate --results_dir")
    cat("Already complete:", key, "\n"); return(invisible(old))
  }
  set.seed(opt$seed); torch_manual_seed(opt$seed)
  d <- prepare_data(config)
  stopifnot(ncol(d$train) == 7, all(is.finite(d$train)), all(is.finite(d$val)), all(is.finite(d$test)))
  dev <- if (opt$device == "auto") if (cuda_is_available()) "cuda" else "cpu" else opt$device
  if (dev == "cuda" && !cuda_is_available()) stop("CUDA requested but unavailable")
  loaders <- lapply(c(train = "train", val = "val", test = "test"), function(k)
    dataloader(window_dataset(d[[k]], opt$seq_len, opt$pred_len), batch_size = opt$batch_size, shuffle = k == "train"))
  m <- timekan(opt$seq_len, opt$pred_len, opt$d_model, opt$levels, opt$window,
    L, opt$base_order, 0, "paper", "chebyshev", opt$cfd_mode)
  m$to(device = dev)
  params <- sum(vapply(m$parameters, function(p) p$numel(), numeric(1)))
  optim <- optim_adam(m$parameters, lr = opt$lr)
  best <- Inf; state <- NULL; bad <- 0; best_epoch <- 0; history <- list()
  started <- Sys.time()
  cat(sprintf("START %s | %s | %d parameters\n", key, dev, params))
  for (epoch in seq_len(opt$epochs)) {
    # Reset shuffle RNG independently of how many parameters initialization consumed.
    set.seed(opt$seed + epoch); torch_manual_seed(opt$seed + epoch)
    tick <- Sys.time(); m$train(); total <- 0; n <- 0
    coro::loop(for (b in loaders$train) {
      optim$zero_grad(); x <- b$x$to(device = dev); y <- b$y$to(device = dev)
      loss <- nnf_mse_loss(m(x), y)
      if (!is.finite(loss$item())) stop("Nonfinite training loss: ", key)
      loss$backward(); optim$step()
      total <- total + loss$item()*y$numel(); n <- n + y$numel()
    })
    val <- evaluate(m, loaders$val, dev)[["mse"]]
    if (!is.finite(val)) stop("Nonfinite validation error: ", key)
    history[[epoch]] <- data.frame(dataset = ds, layers = L, seed = opt$seed, epoch,
      train_mse = total/n, val_mse = val, seconds = as.numeric(difftime(Sys.time(), tick, units = "secs")))
    cat(sprintf("L=%d epoch %d | train %.6f | val %.6f | %.1fs\n", L, epoch, total/n, val, history[[epoch]]$seconds))
    if (val < best) {
      best <- val; best_epoch <- epoch; bad <- 0
      state <- lapply(m$state_dict(), function(t) t$detach()$to(device = "cpu")$clone())
    } else bad <- bad + 1
    if (bad >= opt$patience) break
  }
  m$load_state_dict(state)
  scores <- evaluate(m, loaders$test, dev)
  stopifnot(all(is.finite(scores)))
  row <- data.frame(dataset = ds, layers = L, seed = opt$seed, seq_len = opt$seq_len,
    pred_len = opt$pred_len, variables = 7, levels = opt$levels, d_model = opt$d_model,
    lr = opt$lr, batch_size = opt$batch_size, max_epochs = opt$epochs, patience = opt$patience,
    params, epochs_run = epoch, best_epoch, hit_epoch_limit = epoch == opt$epochs,
    best_in_last_two = best_epoch >= epoch-1, val_mse = best,
    test_mse = scores[["mse"]], test_mae = scores[["mae"]],
    minutes = as.numeric(difftime(Sys.time(), started, units = "mins")), device = dev, cfd_mode = opt$cfd_mode)
  record <- list(config = config, hash = hash, summary = row, history = do.call(rbind, history),
    scaling = list(mean = d$mu, sd = d$sdev),
    windows = sapply(d[c("train", "val", "test")], function(a) nrow(a)-opt$seq_len-opt$pred_len+1),
    session = capture.output(sessionInfo()))
  saveRDS(record, paste0(path, ".tmp"))
  if (!file.rename(paste0(path, ".tmp"), path)) stop("Could not commit run record")
  cat(sprintf("COMPLETE %s | MSE %.6f | MAE %.6f | %.2f min\n", key, row$test_mse, row$test_mae, row$minutes))
  invisible(record)
}

depth_main <- function(opt = depth_args()) {
  if (opt$task == "plots") { depth_tables(opt); return(invisible(NULL)) }
  source("timekan.R")
  torch_set_num_threads(as.integer(opt$threads))
  if (opt$task == "self_test") { depth_self_test(); return(invisible(NULL)) }
  dir.create(file.path(opt$results_dir, "records"), recursive = TRUE, showWarnings = FALSE)
  dir.create(opt$data_dir, recursive = TRUE, showWarnings = FALSE)
  ds <- strsplit(opt$datasets, ",", fixed = TRUE)[[1]]
  for (name in ds) {
    path <- file.path(opt$data_dir, paste0(name, ".csv"))
    if (!file.exists(path)) download.file(paste0("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/", name, ".csv"), path, mode = "wb")
  }
  if (opt$task == "benchmark") {
    # Real full-dataset, one-epoch run saved separately, never pooled with the study.
    opt$epochs <- 1; opt$results_dir <- paste0(opt$results_dir, "_benchmark")
    dir.create(file.path(opt$results_dir, "records"), recursive = TRUE, showWarnings = FALSE)
    depth_run(opt, ds[1], 2); return(invisible(NULL))
  }
  depth_self_test()
  study_path <- file.path(opt$results_dir, "study_config.rds")
  science <- list(options = opt[setdiff(names(opt), c("task", "datasets", "depths", "data_dir", "results_dir", "figures_dir", "backup", "device", "threads"))],
    source_md5 = unname(tools::md5sum(c("timekan.R", "run_depth_experiment.R"))))
  if (file.exists(study_path)) {
    if (!identical(readRDS(study_path), science)) stop("Study settings changed; use a different --results_dir")
  } else saveRDS(science, study_path)
  failures <- character()
  for (L in as.numeric(strsplit(opt$depths, ",", fixed = TRUE)[[1]])) for (name in ds) {
    tryCatch({ depth_run(opt, name, L); depth_tables(opt); depth_backup(opt) },
      error = function(e) { failures <<- c(failures, sprintf("%s L=%d: %s", name, L, conditionMessage(e))); message(tail(failures, 1)) })
    gc()
  }
  if (length(failures)) {
    writeLines(failures, file.path(opt$results_dir, "failures.txt"))
    stop("Some runs failed; completed results were preserved. See failures.txt")
  }
  cat("DEPTH_SWEEP_COMPLETE\n")
}
if (sys.nframe() == 0L) depth_main()
