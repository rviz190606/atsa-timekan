# R-only orchestration for the requested experiments. No Python required.
# Run from the repository root: Rscript run_extra.R --task extensions --device cuda

defaults <- list(task = "extensions", device = "auto", epochs = 10, seed = 0,
                 air_csv = "", climate_csv = "", download = 0, backup = "",
                 component_dataset = "ETTh1", lookback_dataset = "ETTm2")
parse_extra <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- defaults
  if (!length(args)) return(out)
  if (length(args) %% 2) stop("Every --option needs a value")
  for (i in seq(1, length(args), by = 2)) {
    key <- sub("^--", "", args[i])
    if (!key %in% names(out)) stop("Unknown option: ", args[i])
    out[[key]] <- if (is.numeric(out[[key]])) as.numeric(args[i + 1]) else args[i + 1]
  }
  out
}

copy_backup <- function(destination) {
  if (!nzchar(destination)) return(invisible(NULL))
  dir.create(destination, recursive = TRUE, showWarnings = FALSE)
  for (folder in c("results/extensions12", "results/missing_ablations",
                   "results/univariate_baselines", "data/extensions12")) {
    if (!dir.exists(folder)) next
    dest <- file.path(destination, folder)
    dir.create(dest, recursive = TRUE, showWarnings = FALSE)
    files <- list.files(folder, full.names = TRUE, all.files = FALSE)
    if (length(files) && !all(file.copy(files, dest, recursive = TRUE, overwrite = TRUE))) {
      warning("Some backup files could not be copied: ", folder)
    }
  }
  files <- intersect(c("run_extra.R", "extensions12.R", "ablations.R",
                       "classical_baselines.R", "EXPERIMENTS.md"), list.files())
  if (length(files)) file.copy(files, destination, overwrite = TRUE)
  invisible(NULL)
}

rscript_path <- function() file.path(R.home("bin"),
  if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")

run_child <- function(script, args, log) {
  dir.create(dirname(log), recursive = TRUE, showWarnings = FALSE)
  cat("START", script, paste(args, collapse = " "), "\n")
  status <- system2(rscript_path(), shQuote(c(script, args)), stdout = log, stderr = log)
  lines <- readLines(log, warn = FALSE)
  cat(paste(tail(lines, 18), collapse = "\n"), "\n")
  if (status != 0) stop("Experiment failed; inspect ", log)
  invisible(status)
}

common_args <- function(opt) c("--device", opt$device, "--epochs", opt$epochs,
  "--seed", opt$seed, "--seq_len", "96", "--pred_len", "96", "--d_model", "16",
  "--layers", "2", "--levels", "3", "--base_order", "1", "--window", "2",
  "--batch_size", "128", "--lr", "0.01", "--patience", "3")

download_kaggle_csv <- function(handle, wanted = NULL) {
  directory <- file.path("data", "kaggle", gsub("/", "_", handle))
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  archive <- file.path(directory, "dataset.zip")
  if (!file.exists(archive)) {
    options(timeout = max(1800, getOption("timeout")))
    url <- paste0("https://www.kaggle.com/api/v1/datasets/download/", handle)
    status <- tryCatch(download.file(url, archive, mode = "wb", method = "libcurl"),
                       error = function(e) stop("Download failed. Supply --air_csv/--climate_csv from Kaggle: ", conditionMessage(e)))
    if (status != 0) stop("Kaggle download failed; supply a downloaded CSV")
  }
  listing <- tryCatch(unzip(archive, list = TRUE)$Name,
    error = function(e) stop("Kaggle did not return a readable ZIP; supply the CSV manually"))
  candidates <- listing[grepl("[.]csv$", listing, ignore.case = TRUE)]
  if (!is.null(wanted)) candidates <- candidates[basename(candidates) == wanted]
  if (!length(candidates)) stop("No requested CSV in the Kaggle archive")
  unzip(archive, files = candidates, exdir = directory)
  file.path(directory, candidates)
}

prepare_indian_inputs <- function(opt) {
  dir.create("data/extensions12", recursive = TRUE, showWarnings = FALSE)
  air <- "data/extensions12/Delhi_city_hour.csv"
  if (!file.exists(air)) {
    source <- opt$air_csv
    if (!nzchar(source)) {
      if (!opt$download) stop("Supply --air_csv city_hour.csv or --download 1")
      source <- download_kaggle_csv("rohanrao/air-quality-data-in-india", "city_hour.csv")[1]
    }
    frame <- read.csv(source, check.names = FALSE, stringsAsFactors = FALSE)
    stopifnot(all(c("City", "Datetime", "PM2.5") %in% names(frame)))
    frame <- frame[frame$City == "Delhi", c("City", "Datetime", "PM2.5"), drop = FALSE]
    if (!nrow(frame)) stop("No Delhi observations in the supplied air CSV")
    write.csv(frame, air, row.names = FALSE, na = "")
  }
  climate <- "data/extensions12/Delhi_daily_climate.csv"
  if (!file.exists(climate)) {
    sources <- opt$climate_csv
    if (!nzchar(sources)) {
      if (!opt$download) stop("Supply --climate_csv downloaded.csv or --download 1")
      sources <- download_kaggle_csv("yug201/daily-climate-time-series-data-delhi-india")
    }
    candidates <- lapply(sources, function(path) {
      frame <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
      if (all(c("DATE", "temp") %in% names(frame))) {
        date <- as.Date(frame$DATE, "%m/%d/%Y"); target <- "temp"
      } else if (all(c("date", "meantemp") %in% names(frame))) {
        date <- as.Date(frame$date); target <- "meantemp"
      } else return(NULL)
      if (anyNA(date)) stop("Invalid climate dates in ", path)
      output <- data.frame(date = format(date, "%Y-%m-%d"))
      output[[target]] <- frame[[target]]
      list(path = path, frame = output, target = target)
    })
    candidates <- Filter(Negate(is.null), candidates)
    if (!length(candidates)) stop("No supported climate schema: need DATE/temp or date/meantemp")
    selected <- candidates[[which.max(sapply(candidates, function(x) nrow(x$frame)))]]
    write.csv(selected$frame, climate, row.names = FALSE, na = "")
    cat("Climate source:", selected$path, "; target:", selected$target, "\n")
  }
  columns <- names(read.csv(climate, nrows = 1, check.names = FALSE))
  target <- intersect(c("temp", "meantemp"), columns)
  if (length(target) != 1) stop("Prepared climate target is ambiguous")
  list(air = air, climate = climate, climate_target = target)
}

collect_extensions <- function() {
  directory <- "results/extensions12"
  files <- list.files(directory, "_summary[.]csv$", full.names = TRUE)
  files <- files[basename(files) != "extension_summary.csv"]
  if (!length(files)) return(invisible(NULL))
  table <- do.call(rbind, lapply(files, read.csv))
  write.csv(table, file.path(directory, "extension_summary.csv"), row.names = FALSE)
  png(file.path(directory, "extension_comparison_R.png"), width = 1600, height = 1600, res = 140)
  on.exit(dev.off())
  groups <- split(table, table$experiment)
  par(mfrow = c(ceiling(length(groups) / 2), 2), mar = c(6, 4, 3, 1))
  for (name in names(groups)) {
    frame <- groups[[name]]
    barplot(frame$mse_standardized, names.arg = frame$model, las = 2,
            main = name, cex.main = .75, ylab = "Standardized test MSE",
            col = ifelse(frame$model == "TimeKAN", "#2563eb", "#94a3b8"))
  }
  print(table, row.names = FALSE)
  invisible(table)
}

run_extensions_r <- function(opt) {
  input <- prepare_indian_inputs(opt)
  plan <- list()
  for (dataset in c("ETTh1", "ETTh2")) for (horizon in c(96, 192)) {
    name <- paste0(dataset, "_OT_F", horizon, "_seed", opt$seed)
    plan[[name]] <- c("--experiment", "ett", "--csv", paste0("data/", dataset, ".csv"),
      "--target", "OT", "--split", "etth", "--season", "24", "--max_gap", "0",
      "--pred_len", horizon)
  }
  for (horizon in c(24, 96)) {
    name <- paste0("India_Delhi_PM25_F", horizon, "_seed", opt$seed)
    plan[[name]] <- c("--experiment", "air", "--csv", input$air, "--target", "PM2.5",
      "--city", "Delhi", "--split", "712", "--start_date", "2018-01-01",
      "--end_date", "2019-12-31", "--season", "24", "--max_gap", "3", "--pred_len", horizon)
  }
  for (horizon in c(7, 30)) {
    name <- paste0("India_Delhi_Temperature_F", horizon, "_seed", opt$seed)
    plan[[name]] <- c("--experiment", "climate", "--csv", input$climate,
      "--target", input$climate_target, "--split", "712", "--season", "7",
      "--max_gap", "1", "--pred_len", horizon)
  }
  for (name in names(plan)) {
    if (file.exists(file.path("results/extensions12", paste0(name, "_summary.csv")))) {
      cat("Already complete:", name, "\n"); next
    }
    tryCatch(run_child("extensions12.R", c(common_args(opt), "--name", name, plan[[name]]),
      file.path("results/extensions12", paste0(name, ".log"))),
      finally = copy_backup(opt$backup))
  }
  collect_extensions(); copy_backup(opt$backup)
}

run_ablations_r <- function(opt) {
  directory <- "results/missing_ablations"
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  variants <- list(linear_mapping = c("--upsample", "linear_map"),
    linear_interpolation = c("--upsample", "linear"),
    transposed_convolution = c("--upsample", "transpose_conv"),
    MLP = c("--representation", "mlp"), no_convolution = c("--temporal", "none"),
    standard_convolution = c("--temporal", "standard"), self_attention = c("--temporal", "attention"))
  studies <- c(linear_mapping = "upsampling", linear_interpolation = "upsampling",
    transposed_convolution = "upsampling", MLP = "representation",
    no_convolution = "temporal", standard_convolution = "temporal", self_attention = "temporal")
  plan <- list()
  for (variant in names(variants)) {
    name <- paste0(opt$component_dataset, "_", studies[[variant]], "_", variant, "_seed", opt$seed)
    plan[[name]] <- c("--csv", paste0("data/", opt$component_dataset, ".csv"),
      "--tag", paste0(studies[[variant]], ":", variant), variants[[variant]])
  }
  for (len in c(48, 192, 336, 512, 720)) {
    name <- paste0(opt$lookback_dataset, "_lookback_T", len, "_seed", opt$seed)
    plan[[name]] <- c("--csv", paste0("data/", opt$lookback_dataset, ".csv"),
      "--seq_len", len, "--tag", paste0("lookback:T=", len))
  }
  for (name in names(plan)) {
    file <- file.path(directory, paste0(name, ".csv"))
    if (file.exists(file)) { cat("Already complete:", name, "\n"); next }
    tryCatch(run_child("ablations.R", c(common_args(opt), plan[[name]], "--out", file,
      "--save", file.path(directory, paste0(name, ".pt"))),
      file.path(directory, paste0(name, ".log"))), finally = copy_backup(opt$backup))
  }
  files <- list.files(directory, "[.]csv$", full.names = TRUE)
  files <- files[!grepl("summary", basename(files), fixed = TRUE)]
  if (length(files)) {
    table <- do.call(rbind, lapply(files, read.csv))
    if (file.exists("results/main.csv")) {
      baseline <- read.csv("results/main.csv")
      baseline <- baseline[baseline$seq_len == 96 & baseline$pred_len == 96 & baseline$seed == opt$seed, ]
      for (study in c("upsampling", "representation", "temporal", "lookback")) {
        dataset <- if (study == "lookback") opt$lookback_dataset else opt$component_dataset
        row <- tail(baseline[baseline$dataset == dataset, ], 1)
        if (nrow(row)) {
          row$tag <- paste0(study, ":existing_baseline")
          table <- rbind(table, row)
        }
      }
    }
    write.csv(table, file.path(directory, "missing_ablation_summary_R.csv"), row.names = FALSE)
    print(table[, c("dataset", "tag", "seq_len", "params", "test_mse", "test_mae")], row.names = FALSE)
    groups <- split(table, sub(":.*", "", table$tag))
    png(file.path(directory, "missing_ablation_comparison_R.png"), width = 1600, height = 1200, res = 140)
    par(mfrow = c(2, 2), mar = c(9, 4, 3, 1))
    for (study in names(groups)) {
      frame <- groups[[study]]
      labels <- if (study == "lookback") paste0("T=", frame$seq_len) else sub("^[^:]*:", "", frame$tag)
      barplot(frame$test_mse, names.arg = labels, las = 2, main = study,
        ylab = "Standardized test MSE", cex.names = .65,
        col = ifelse(grepl("existing_baseline", frame$tag), "#2563eb", "#94a3b8"))
    }
    dev.off()
  }
  copy_backup(opt$backup)
}

collect_baselines_r <- function() {
  directory <- "results/univariate_baselines"
  files <- list.files(directory, "_classical[.]csv$", full.names = TRUE)
  if (!length(files)) return(invisible(NULL))
  rows <- lapply(files, function(file) {
    classical <- read.csv(file)
    original <- file.path("results/extensions12", paste0(classical$experiment[1], "_summary.csv"))
    rbind(read.csv(original), classical)
  })
  table <- do.call(rbind, rows)
  write.csv(table, file.path(directory, "all_models.csv"), row.names = FALSE)
  groups <- split(table, table$experiment)
  overview <- do.call(rbind, lapply(groups, function(frame) {
    model <- frame[frame$model == "TimeKAN", ]
    other <- frame[frame$model != "TimeKAN", ]
    best <- other[which.min(other$mse_standardized), ]
    data.frame(experiment = model$experiment, timekan_mse = model$mse_standardized,
      best_baseline = best$model, best_baseline_mse = best$mse_standardized,
      timekan_mse_reduction_pct = 100 * (best$mse_standardized - model$mse_standardized) / best$mse_standardized)
  }))
  write.csv(overview, file.path(directory, "comparison_overview.csv"), row.names = FALSE)
  png(file.path(directory, "univariate_baseline_comparison_R.png"), width = 1800, height = 1800, res = 150)
  on.exit(dev.off())
  par(mfrow = c(ceiling(length(groups) / 2), 2), mar = c(6, 4, 3, 1))
  for (name in names(groups)) {
    frame <- groups[[name]]
    barplot(frame$mse_standardized, names.arg = frame$model, las = 2,
      main = name, cex.main = .75, ylab = "Standardized test MSE",
      col = ifelse(frame$model == "TimeKAN", "#2563eb", "#94a3b8"))
  }
  print(overview, row.names = FALSE)
  invisible(overview)
}

if (sys.nframe() == 0L) {
  opt <- parse_extra()
  if (!file.exists("timekan.R")) stop("Run from the repository root")
  if (opt$task == "self_test") {
    for (script in c("extensions12.R", "ablations.R", "classical_baselines.R")) {
      flag <- if (script == "classical_baselines.R") "--self_test" else c("--self_test", "1")
      run_child(script, flag, file.path("results/extra_checks", paste0(script, ".log")))
    }
  } else {
    if (!opt$task %in% c("extensions", "ablations", "compare", "all")) stop("Unknown --task")
    if (opt$task %in% c("extensions", "all")) run_extensions_r(opt)
    if (opt$task %in% c("ablations", "all")) run_ablations_r(opt)
    if (opt$task %in% c("compare", "all")) {
      tryCatch({
        run_child("classical_baselines.R", character(), "results/univariate_baselines/comparison.log")
        collect_baselines_r()
      }, finally = copy_backup(opt$backup))
    }
  }
}
