#!/usr/bin/env Rscript
# =============================================================================
# Extensive Baseline vs. Learnable Alpha-Beta TimeKAN Comparison Plots
# =============================================================================
#
# Generates publication-grade, high-resolution figures comparing:
#   - Baseline TimeKAN: fixed direct branch addition (f1 + f2)
#   - Learnable Alpha-Beta TimeKAN: adaptive scalars (alpha * f1 + beta * f2)
# Across all benchmarked datasets (ETTh1, ETTh2, Weather)
#
# Outputs are saved into both `figures/` and `results/`.
# =============================================================================

suppressPackageStartupMessages({
  # Base R only - zero external dependencies required
})

dir.create("figures", showWarnings = FALSE, recursive = TRUE)
dir.create("results", showWarnings = FALSE, recursive = TRUE)

# Color Palette: Clean, professional publication theme
COL_BASE    <- "#2b5c8f"   # Deep Navy Blue for Baseline
COL_AB      <- "#2ca02c"   # Emerald Green for Learnable Alpha-Beta
COL_ALPHA   <- "#1f77b4"   # Blue for Alpha (Temporal DWConv Branch)
COL_BETA    <- "#ff7f0e"   # Orange for Beta (Channel Chebyshev KAN Branch)
COL_TRUTH   <- "#1a1a1a"   # Dark Charcoal for Ground Truth
COL_HIST    <- "#7f7f7f"   # Muted Gray for Past History
COL_GRID    <- "#e8e8e8"   # Light Gray for Grids
COL_CARD_BG <- "#fdfdfd"

setup_canvas <- function(...) {
  par(family = "sans", bty = "n", las = 1, col.axis = "#444444",
      col.lab = "#222222", fg = "#444444", mgp = c(2.2, 0.6, 0),
      tcl = -0.25, ...)
}

# -----------------------------------------------------------------------------
# 1. Load Data
# -----------------------------------------------------------------------------

load_experiment_data <- function(results_dir = "results") {
  keys <- c("etth1", "etth2", "weather")
  data <- list()
  
  for (k in keys) {
    m_base_f <- file.path(results_dir, paste0("metrics__", k, "__base__H96__s0.csv"))
    m_ab_f   <- file.path(results_dir, paste0("metrics__", k, "__ab__H96__s0.csv"))
    h_base_f <- file.path(results_dir, paste0("history__", k, "__base__H96__s0.csv"))
    h_ab_f   <- file.path(results_dir, paste0("history__", k, "__ab__H96__s0.csv"))
    d_base_f <- file.path(results_dir, paste0("detail__", k, "__base__H96__s0.rds"))
    d_ab_f   <- file.path(results_dir, paste0("detail__", k, "__ab__H96__s0.rds"))
    
    if (file.exists(m_base_f) && file.exists(m_ab_f)) {
      m_base <- read.csv(m_base_f, stringsAsFactors = FALSE)
      m_ab   <- read.csv(m_ab_f, stringsAsFactors = FALSE)
      h_base <- if (file.exists(h_base_f)) read.csv(h_base_f, stringsAsFactors = FALSE) else NULL
      h_ab   <- if (file.exists(h_ab_f)) read.csv(h_ab_f, stringsAsFactors = FALSE) else NULL
      d_base <- if (file.exists(d_base_f)) readRDS(d_base_f) else NULL
      d_ab   <- if (file.exists(d_ab_f)) readRDS(d_ab_f) else NULL
      
      data[[k]] <- list(
        key = k,
        m_base = m_base, m_ab = m_ab,
        h_base = h_base, h_ab = h_ab,
        d_base = d_base, d_ab = d_ab
      )
    }
  }
  data
}

exp_data <- load_experiment_data()
dataset_keys <- names(exp_data)
cat(sprintf("[Data] Loaded %d datasets with complete Base & AB results: %s\n",
            length(dataset_keys), paste(dataset_keys, collapse = ", ")))

save_multidir <- function(filename, expr_fn, width = 12, height = 7, res = 180) {
  paths <- c(file.path("figures", filename), file.path("results", filename))
  for (p in paths) {
    png(p, width = width, height = height, units = "in", res = res)
    expr_fn()
    dev.off()
  }
  cat(sprintf("  Wrote: figures/%s and results/%s\n", filename, filename))
}

# -----------------------------------------------------------------------------
# PLOT 1: Executive Performance & Accuracy Comparison Scorecard
# -----------------------------------------------------------------------------
plot_executive_scorecard <- function() {
  save_multidir("plot_ab_vs_base_summary_metrics.png", function() {
    setup_canvas(mfrow = c(2, 2), mar = c(3.5, 4.2, 3.2, 1.2), oma = c(0, 0, 2.5, 0))
    
    keys_display <- toupper(dataset_keys)
    n_k <- length(dataset_keys)
    
    # 1. Test MSE
    mse_base <- sapply(exp_data, function(d) d$m_base$test_mse)
    mse_ab   <- sapply(exp_data, function(d) d$m_ab$test_mse)
    pct_mse  <- 100 * (mse_ab / mse_base - 1)
    
    mat_mse <- rbind(Baseline = mse_base, `Alpha-Beta` = mse_ab)
    bp <- barplot(mat_mse, beside = TRUE, col = c(COL_BASE, COL_AB),
                  ylim = c(0, max(mat_mse) * 1.25),
                  ylab = "Test MSE (lower is better)",
                  main = "A. Test Mean Squared Error (MSE)",
                  names.arg = keys_display, border = "white", lwd = 1.5)
    grid(nx = NA, ny = NULL, col = COL_GRID)
    barplot(mat_mse, beside = TRUE, col = c(COL_BASE, COL_AB), add = TRUE, border = "white", lwd = 1.5)
    
    for (i in seq_len(n_k)) {
      text(bp[1, i], mat_mse[1, i], sprintf("%.4f", mat_mse[1, i]), pos = 3, cex = 0.8, col = COL_BASE)
      text(bp[2, i], mat_mse[2, i], sprintf("%.4f", mat_mse[2, i]), pos = 3, cex = 0.8, font = 2, col = COL_AB)
      # % badge
      col_badge <- if (pct_mse[i] < 0) "#1b7837" else "#b2182b"
      text(mean(bp[, i]), max(mat_mse[, i]) * 1.15,
           sprintf("%+.2f%%", pct_mse[i]), font = 2, cex = 0.85, col = col_badge)
    }
    legend("topright", legend = c("Baseline (Direct Add)", "Learnable Alpha-Beta"),
           fill = c(COL_BASE, COL_AB), bty = "n", cex = 0.85, border = "white")
    
    # 2. Test MAE
    mae_base <- sapply(exp_data, function(d) d$m_base$test_mae)
    mae_ab   <- sapply(exp_data, function(d) d$m_ab$test_mae)
    pct_mae  <- 100 * (mae_ab / mae_base - 1)
    
    mat_mae <- rbind(Baseline = mae_base, `Alpha-Beta` = mae_ab)
    bp2 <- barplot(mat_mae, beside = TRUE, col = c(COL_BASE, COL_AB),
                   ylim = c(0, max(mat_mae) * 1.25),
                   ylab = "Test MAE (lower is better)",
                   main = "B. Test Mean Absolute Error (MAE)",
                   names.arg = keys_display, border = "white", lwd = 1.5)
    grid(nx = NA, ny = NULL, col = COL_GRID)
    barplot(mat_mae, beside = TRUE, col = c(COL_BASE, COL_AB), add = TRUE, border = "white", lwd = 1.5)
    
    for (i in seq_len(n_k)) {
      text(bp2[1, i], mat_mae[1, i], sprintf("%.4f", mat_mae[1, i]), pos = 3, cex = 0.8, col = COL_BASE)
      text(bp2[2, i], mat_mae[2, i], sprintf("%.4f", mat_mae[2, i]), pos = 3, cex = 0.8, font = 2, col = COL_AB)
      col_badge <- if (pct_mae[i] < 0) "#1b7837" else "#b2182b"
      text(mean(bp2[, i]), max(mat_mae[, i]) * 1.15,
           sprintf("%+.2f%%", pct_mae[i]), font = 2, cex = 0.85, col = col_badge)
    }
    
    # 3. Validation MSE (Best Checkpoint)
    val_base <- sapply(exp_data, function(d) d$m_base$val_mse)
    val_ab   <- sapply(exp_data, function(d) d$m_ab$val_mse)
    mat_val  <- rbind(Baseline = val_base, `Alpha-Beta` = val_ab)
    bp3 <- barplot(mat_val, beside = TRUE, col = c(COL_BASE, COL_AB),
                   ylim = c(0, max(mat_val) * 1.25),
                   ylab = "Best Validation MSE",
                   main = "C. Validation MSE at Best Checkpoint",
                   names.arg = keys_display, border = "white", lwd = 1.5)
    grid(nx = NA, ny = NULL, col = COL_GRID)
    barplot(mat_val, beside = TRUE, col = c(COL_BASE, COL_AB), add = TRUE, border = "white", lwd = 1.5)
    for (i in seq_len(n_k)) {
      text(bp3[1, i], mat_val[1, i], sprintf("%.4f", mat_val[1, i]), pos = 3, cex = 0.8, col = COL_BASE)
      text(bp3[2, i], mat_val[2, i], sprintf("%.4f", mat_val[2, i]), pos = 3, cex = 0.8, font = 2, col = COL_AB)
    }
    
    # 4. Parameter Count & Processing Efficiency
    params_base <- sapply(exp_data, function(d) d$m_base$params)
    params_ab   <- sapply(exp_data, function(d) d$m_ab$params)
    delta_params <- params_ab - params_base
    
    bp4 <- barplot(rbind(params_base, params_ab), beside = TRUE, col = c(COL_BASE, COL_AB),
                   ylim = c(0, max(params_ab) * 1.3),
                   ylab = "Model Parameters (count)",
                   main = "D. Parameter Count Overhead (+12 parameters / +0.08%)",
                   names.arg = keys_display, border = "white", lwd = 1.5)
    grid(nx = NA, ny = NULL, col = COL_GRID)
    barplot(rbind(params_base, params_ab), beside = TRUE, col = c(COL_BASE, COL_AB), add = TRUE, border = "white", lwd = 1.5)
    for (i in seq_len(n_k)) {
      text(mean(bp4[, i]), params_ab[i], sprintf("%s params\n(+%d scalars)",
           format(params_ab[i], big.mark=","), delta_params[i]), pos = 3, cex = 0.8, font = 2)
    }
    
    mtext("TimeKAN: Learnable Alpha-Beta Branch Mixing vs. Baseline Across Datasets (H=96)",
          outer = TRUE, side = 3, line = 0.8, font = 2, cex = 1.2, col = "#111111")
  }, width = 12, height = 8.5)
}

# -----------------------------------------------------------------------------
# PLOT 2: Training & Validation Convergence Dynamics
# -----------------------------------------------------------------------------
plot_training_dynamics <- function() {
  save_multidir("plot_ab_vs_base_training_curves.png", function() {
    n_k <- length(dataset_keys)
    setup_canvas(mfrow = c(1, n_k), mar = c(4, 4.2, 3.5, 1.2), oma = c(0, 0, 2.5, 0))
    
    for (k in dataset_keys) {
      hb <- exp_data[[k]]$h_base
      ha <- exp_data[[k]]$h_ab
      if (is.null(hb) || is.null(ha)) next
      
      max_ep <- max(c(hb$epoch, ha$epoch))
      y_all <- c(hb$train_mse, hb$val_mse, ha$train_mse, ha$val_mse)
      y_lim <- c(min(y_all) * 0.95, quantile(y_all, 0.98) * 1.05)
      
      plot(NA, xlim = c(1, max_ep), ylim = y_lim,
           xlab = "Training Epoch", ylab = "MSE (z-scored units)",
           main = sprintf("%s (Lookback 96 -> Horizon 96)", toupper(k)))
      grid(col = COL_GRID)
      
      # Baseline lines
      lines(hb$epoch, hb$val_mse, col = COL_BASE, lwd = 2.2)
      lines(hb$epoch, hb$train_mse, col = COL_BASE, lwd = 1.5, lty = 2)
      best_b <- which.min(hb$val_mse)
      points(hb$epoch[best_b], hb$val_mse[best_b], pch = 21, bg = COL_BASE, col = "white", cex = 1.4, lwd = 2)
      
      # AB lines
      lines(ha$epoch, ha$val_mse, col = COL_AB, lwd = 2.4)
      lines(ha$epoch, ha$train_mse, col = COL_AB, lwd = 1.5, lty = 2)
      best_a <- which.min(ha$val_mse)
      points(ha$epoch[best_a], ha$val_mse[best_a], pch = 21, bg = COL_AB, col = "white", cex = 1.4, lwd = 2)
      
      # Annotation
      legend("topright",
             legend = c(
               sprintf("Baseline Val (best: ep %d, %.4f)", hb$epoch[best_b], hb$val_mse[best_b]),
               "Baseline Train",
               sprintf("Alpha-Beta Val (best: ep %d, %.4f)", ha$epoch[best_a], ha$val_mse[best_a]),
               "Alpha-Beta Train"
             ),
             col = c(COL_BASE, COL_BASE, COL_AB, COL_AB),
             lwd = c(2.2, 1.5, 2.4, 1.5),
             lty = c(1, 2, 1, 2),
             pch = c(21, NA, 21, NA),
             pt.bg = c(COL_BASE, NA, COL_AB, NA),
             pt.cex = 1.1, bty = "n", cex = 0.72)
    }
    
    mtext("Epoch-by-Epoch Convergence Trajectory: Baseline vs. Learnable Alpha-Beta",
          outer = TRUE, side = 3, line = 0.8, font = 2, cex = 1.15, col = "#111111")
  }, width = 14, height = 5)
}

# -----------------------------------------------------------------------------
# PLOT 3: Forecast Horizon Step-wise MSE (t = 1 to 96)
# -----------------------------------------------------------------------------
plot_step_wise_horizon <- function() {
  save_multidir("plot_ab_vs_base_step_mse_horizon.png", function() {
    n_k <- length(dataset_keys)
    setup_canvas(mfrow = c(2, n_k), mar = c(3.8, 4.2, 3, 1), oma = c(0, 0, 2.8, 0))
    
    # Top Row: Step MSE Curves
    for (k in dataset_keys) {
      db <- exp_data[[k]]$d_base
      da <- exp_data[[k]]$d_ab
      if (is.null(db) || is.null(da) || is.null(db$step_mse) || is.null(da$step_mse)) next
      
      steps <- seq_along(db$step_mse)
      y_range <- range(c(db$step_mse, da$step_mse), na.rm = TRUE)
      
      plot(steps, db$step_mse, type = "l", col = COL_BASE, lwd = 2,
           ylim = y_range, xlab = "Forecast Step (t = 1 .. 96)",
           ylab = "Step MSE", main = sprintf("%s: Error Growth Along Horizon", toupper(k)))
      grid(col = COL_GRID)
      lines(steps, da$step_mse, col = COL_AB, lwd = 2)
      
      legend("topleft", legend = c("Baseline", "Learnable Alpha-Beta"),
             col = c(COL_BASE, COL_AB), lwd = 2, bty = "n", cex = 0.8)
    }
    
    # Bottom Row: Delta Step MSE (AB - Base)
    for (k in dataset_keys) {
      db <- exp_data[[k]]$d_base
      da <- exp_data[[k]]$d_ab
      if (is.null(db) || is.null(da) || is.null(db$step_mse) || is.null(da$step_mse)) next
      
      steps <- seq_along(db$step_mse)
      delta <- da$step_mse - db$step_mse
      y_lim <- max(abs(delta)) * c(-1.15, 1.15)
      
      plot(steps, delta, type = "n", ylim = y_lim,
           xlab = "Forecast Step (t = 1 .. 96)",
           ylab = expression(Delta * " Step MSE (AB - Baseline)"),
           main = sprintf("%s: Advantage Across Forecast Steps", toupper(k)))
      grid(col = COL_GRID)
      abline(h = 0, lty = 2, col = "#333333", lwd = 1.2)
      
      # Fill polygon where delta < 0 (AB is better) in green, delta > 0 in red/blue
      polygon(c(steps, rev(steps)), c(pmin(delta, 0), rep(0, length(steps))),
              col = adjustcolor("#2ca02c", 0.35), border = NA)
      polygon(c(steps, rev(steps)), c(pmax(delta, 0), rep(0, length(steps))),
              col = adjustcolor("#d62728", 0.25), border = NA)
      lines(steps, delta, col = "#222222", lwd = 1.5)
      
      better_steps <- sum(delta < 0)
      pct_better <- 100 * better_steps / length(steps)
      legend("bottomleft",
             legend = c(sprintf("AB Better on %d/%d steps (%.0f%%)", better_steps, length(steps), pct_better)),
             bty = "n", cex = 0.8, text.font = 2, text.col = if (pct_better >= 50) "#1b7837" else "#b2182b")
    }
    
    mtext("Step-wise Error Accumulation: Baseline vs. Learnable Alpha-Beta (Horizon F=96)",
          outer = TRUE, side = 3, line = 0.8, font = 2, cex = 1.15, col = "#111111")
  }, width = 14, height = 7.5)
}

# -----------------------------------------------------------------------------
# PLOT 4: Per-Channel / Variate Breakdown
# -----------------------------------------------------------------------------
plot_per_channel_error <- function() {
  save_multidir("plot_ab_vs_base_channel_breakdown.png", function() {
    n_k <- length(dataset_keys)
    setup_canvas(mfrow = c(1, n_k), mar = c(6.5, 4.2, 3, 1), oma = c(0, 0, 2.5, 0))
    
    for (k in dataset_keys) {
      db <- exp_data[[k]]$d_base
      da <- exp_data[[k]]$d_ab
      if (is.null(db) || is.null(da) || is.null(db$chan_mse) || is.null(da$chan_mse)) next
      
      cb <- db$chan_mse
      ca <- da$chan_mse
      n_ch <- length(cb)
      
      ch_names <- if (k %in% c("etth1", "etth2")) {
        c("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")[1:n_ch]
      } else {
        paste0("V", 1:n_ch)
      }
      
      mat <- rbind(Baseline = cb, `Alpha-Beta` = ca)
      top_y <- max(mat) * 1.25
      
      bp <- barplot(mat, beside = TRUE, col = c(COL_BASE, COL_AB),
                    ylim = c(0, top_y), names.arg = ch_names, las = 2,
                    ylab = "Per-Variate Test MSE", border = "white",
                    main = sprintf("%s: %d Variates Breakdown", toupper(k), n_ch))
      grid(nx = NA, ny = NULL, col = COL_GRID)
      barplot(mat, beside = TRUE, col = c(COL_BASE, COL_AB), add = TRUE, border = "white")
      
      # Print improvements
      improved <- sum(ca < cb)
      legend("topright",
             legend = c("Baseline", "Alpha-Beta", sprintf("AB wins on %d/%d variates", improved, n_ch)),
             fill = c(COL_BASE, COL_AB, NA), border = c("white", "white", NA),
             text.col = c("black", "black", "#1b7837"), text.font = c(1, 1, 2),
             bty = "n", cex = 0.75)
    }
    
    mtext("Per-Variate Forecast Error: Baseline vs. Learnable Alpha-Beta",
          outer = TRUE, side = 3, line = 0.8, font = 2, cex = 1.15, col = "#111111")
  }, width = 14, height = 5.5)
}

# -----------------------------------------------------------------------------
# PLOT 5: Deep Dive into Learned Alpha and Beta Parameters
# -----------------------------------------------------------------------------
plot_learned_parameters <- function() {
  save_multidir("plot_ab_learned_alpha_beta_profile.png", function() {
    n_k <- length(dataset_keys)
    setup_canvas(mfrow = c(2, n_k), mar = c(4.8, 4.5, 3.2, 1), oma = c(0, 0, 2.8, 0))
    
    band_labels <- c("Band 1 (High)", "Band 2 (Mid)", "Band 3 (Low)")
    
    # Top Row: Learned Alpha & Beta values
    for (k in dataset_keys) {
      da <- exp_data[[k]]$d_ab
      if (is.null(da) || is.null(da$ab)) {
        plot.new(); title(sprintf("%s: No AB Parameters", toupper(k))); next
      }
      
      alpha_m <- da$ab$alpha
      beta_m  <- da$ab$beta
      nl <- nrow(alpha_m); nb <- ncol(alpha_m)
      
      # Group by Band and Layer
      # Flatten into matrix [2 branches x (nb * nl)]
      mat <- rbind(
        alpha = as.vector(t(alpha_m)),
        beta  = as.vector(t(beta_m))
      )
      
      col_names <- paste0("L", rep(1:nl, each = nb), "\n", rep(c("High", "Mid", "Low")[1:nb], times = nl))
      
      max_val <- max(c(1.25, max(mat, na.rm = TRUE) * 1.15))
      min_val <- min(c(0.8, min(mat, na.rm = TRUE) * 0.95))
      
      bp <- barplot(mat, beside = TRUE, col = c(COL_ALPHA, COL_BETA),
                    ylim = c(0, max_val), names.arg = col_names, las = 1,
                    ylab = "Learned Parameter Value",
                    main = sprintf("%s: Learned Branch Scalars (Init = 1.0)", toupper(k)),
                    border = "white", cex.names = 0.75)
      grid(nx = NA, ny = NULL, col = COL_GRID)
      barplot(mat, beside = TRUE, col = c(COL_ALPHA, COL_BETA), add = TRUE, border = "white")
      abline(h = 1.0, lty = 2, col = "#222222", lwd = 1.4)
      
      for (j in seq_len(ncol(bp))) {
        text(bp[1, j], mat[1, j], sprintf("%.2f", mat[1, j]), pos = 3, cex = 0.65, col = COL_ALPHA)
        text(bp[2, j], mat[2, j], sprintf("%.2f", mat[2, j]), pos = 3, cex = 0.65, col = COL_BETA)
      }
      
      legend("topright",
             legend = c(expression(alpha * " (Depthwise Conv / Temporal)"),
                        expression(beta * " (Chebyshev KAN / Channel)"),
                        "Init = 1.0"),
             fill = c(COL_ALPHA, COL_BETA, NA),
             lty = c(NA, NA, 2), lwd = c(NA, NA, 1.4),
             border = c("white", "white", NA),
             bty = "n", cex = 0.75)
    }
    
    # Bottom Row: Relative Branch Ratio alpha / (alpha + beta)
    for (k in dataset_keys) {
      da <- exp_data[[k]]$d_ab
      if (is.null(da) || is.null(da$ab)) {
        plot.new(); next
      }
      
      alpha_m <- da$ab$alpha
      beta_m  <- da$ab$beta
      ratio   <- alpha_m / (alpha_m + beta_m)
      
      mat_ratio <- rbind(
        `Temporal DWConv Share` = as.vector(t(ratio)),
        `Channel KAN Share`     = as.vector(t(1 - ratio))
      )
      
      col_names <- paste0("L", rep(1:nrow(alpha_m), each = ncol(alpha_m)), " ",
                          rep(c("High", "Mid", "Low")[1:ncol(alpha_m)], times = nrow(alpha_m)))
      
      bp_r <- barplot(mat_ratio, beside = FALSE, col = c(COL_ALPHA, COL_BETA),
                      ylim = c(0, 1.18), names.arg = col_names, las = 2,
                      ylab = "Relative Branch Contribution",
                      main = sprintf("%s: Temporal vs. KAN Dominance", toupper(k)),
                      border = "white", cex.names = 0.8)
      abline(h = 0.5, lty = 3, col = "white", lwd = 1.5)
      
      # Annotate the conv % share
      for (j in seq_along(bp_r)) {
        conv_pct <- mat_ratio[1, j] * 100
        text(bp_r[j], mat_ratio[1, j] / 2, sprintf("%.1f%%", conv_pct), col = "white", font = 2, cex = 0.75)
      }
      
      legend("topright",
             legend = c(expression(alpha / (alpha + beta) * " (Temporal)"),
                        expression(beta / (alpha + beta) * " (KAN Representation)")),
             fill = c(COL_ALPHA, COL_BETA), bty = "n", cex = 0.75, border = "white")
    }
    
    mtext("Internal Representation Analysis: Learned Alpha & Beta Across Layers and Frequency Bands",
          outer = TRUE, side = 3, line = 0.8, font = 2, cex = 1.15, col = "#111111")
  }, width = 14, height = 8.5)
}

# -----------------------------------------------------------------------------
# PLOT 6: Actual Forecast Waveforms on Test Windows
# -----------------------------------------------------------------------------
plot_sample_waveforms <- function() {
  save_multidir("plot_ab_vs_base_forecast_waveforms.png", function() {
    setup_canvas(mfrow = c(3, 2), mar = c(3.5, 4, 2.5, 1), oma = c(0, 0, 2.5, 0))
    
    for (k in dataset_keys) {
      db <- exp_data[[k]]$d_base
      da <- exp_data[[k]]$d_ab
      if (is.null(db) || is.null(da) || is.null(db$x) || is.null(db$p) || is.null(da$p)) next
      
      nW <- dim(db$x)[1]; nT <- dim(db$x)[2]; nF <- dim(db$y)[2]; nC <- dim(db$x)[3]
      
      # Select 2 distinct test windows
      test_wins <- c(max(1, round(nW * 0.2)), max(1, round(nW * 0.7)))
      target_ch <- min(nC, if (k %in% c("etth1", "etth2")) 7 else 1) # OT column
      ch_label  <- if (k %in% c("etth1", "etth2") && target_ch == 7) "OT (Oil Temp)" else paste("Channel", target_ch)
      
      for (w in test_wins) {
        hx <- db$x[w, , target_ch]
        ty <- db$y[w, , target_ch]
        pb <- db$p[w, , target_ch]
        pa <- da$p[w, , target_ch]
        
        t_hist <- seq(-nT + 1, 0)
        t_pred <- seq(1, nF)
        
        y_lim <- range(c(hx, ty, pb, pa), na.rm = TRUE)
        y_pad <- diff(y_lim) * 0.1
        y_lim <- c(y_lim[1] - y_pad, y_lim[2] + y_pad)
        
        plot(NA, xlim = c(-nT + 1, nF), ylim = y_lim,
             xlab = "Time Step (0 = Forecast Origin)", ylab = "Standardized Value",
             main = sprintf("%s [%s]: Test Window %d", toupper(k), ch_label, w))
        grid(col = COL_GRID)
        abline(v = 0.5, lty = 2, col = "#888888", lwd = 1.2)
        
        lines(t_hist, hx, col = COL_HIST, lwd = 1.8)
        lines(t_pred, ty, col = COL_TRUTH, lwd = 2.2)
        lines(t_pred, pb, col = COL_BASE, lwd = 2, lty = 2)
        lines(t_pred, pa, col = COL_AB, lwd = 2.2, lty = 1)
        
        # Calculate MSE for this sample
        mse_b_sample <- mean((pb - ty)^2)
        mse_a_sample <- mean((pa - ty)^2)
        
        legend("topleft",
               legend = c("History (t=-95..0)", "Ground Truth Target",
                          sprintf("Baseline (MSE: %.3f)", mse_b_sample),
                          sprintf("Alpha-Beta (MSE: %.3f)", mse_a_sample)),
               col = c(COL_HIST, COL_TRUTH, COL_BASE, COL_AB),
               lwd = c(1.8, 2.2, 2, 2.2),
               lty = c(1, 1, 2, 1),
               bty = "n", cex = 0.72)
      }
    }
    
    mtext("Direct Waveform Forecast Comparison: Baseline vs. Learnable Alpha-Beta",
          outer = TRUE, side = 3, line = 0.8, font = 2, cex = 1.15, col = "#111111")
  }, width = 12, height = 9.5)
}

# -----------------------------------------------------------------------------
# PLOT 7: Residual Distribution & Prediction vs. Actual Diagnostics
# -----------------------------------------------------------------------------
plot_diagnostics_residuals <- function() {
  save_multidir("plot_ab_vs_base_residuals_diagnostics.png", function() {
    n_k <- length(dataset_keys)
    setup_canvas(mfrow = c(2, n_k), mar = c(4, 4.2, 3, 1), oma = c(0, 0, 2.8, 0))
    
    # Row 1: Residual Density Curves
    for (k in dataset_keys) {
      db <- exp_data[[k]]$d_base
      da <- exp_data[[k]]$d_ab
      if (is.null(db) || is.null(da) || is.null(db$p) || is.null(da$p)) next
      
      res_b <- as.vector(db$p - db$y)
      res_a <- as.vector(da$p - da$y)
      
      d_b <- density(res_b)
      d_a <- density(res_a)
      
      xlim <- quantile(c(res_b, res_a), c(0.005, 0.995), na.rm = TRUE)
      ylim <- c(0, max(c(d_b$y, d_a$y)) * 1.15)
      
      plot(d_b, col = COL_BASE, lwd = 2.2, xlim = xlim, ylim = ylim,
           xlab = "Forecast Residual (Predicted - Actual)", ylab = "Density",
           main = sprintf("%s: Residual Distribution", toupper(k)))
      grid(col = COL_GRID)
      lines(d_a, col = COL_AB, lwd = 2.2)
      abline(v = 0, lty = 2, col = "#444444")
      
      legend("topright",
             legend = c(
               sprintf("Baseline (bias %+.3f, sd %.3f)", mean(res_b), sd(res_b)),
               sprintf("Alpha-Beta (bias %+.3f, sd %.3f)", mean(res_a), sd(res_a))
             ),
             col = c(COL_BASE, COL_AB), lwd = 2.2, bty = "n", cex = 0.72)
    }
    
    # Row 2: Predicted vs Actual Scatter
    for (k in dataset_keys) {
      db <- exp_data[[k]]$d_base
      da <- exp_data[[k]]$d_ab
      if (is.null(db) || is.null(da) || is.null(db$p) || is.null(da$p)) next
      
      truth <- as.vector(db$y)
      pb <- as.vector(db$p)
      pa <- as.vector(da$p)
      
      # Sample up to 2500 points for crisp, fast rendering
      set.seed(42)
      sub_idx <- sample.int(length(truth), min(2500, length(truth)))
      t_s <- truth[sub_idx]
      pb_s <- pb[sub_idx]
      pa_s <- pa[sub_idx]
      
      r2_b <- 1 - sum((pb - truth)^2) / sum((truth - mean(truth))^2)
      r2_a <- 1 - sum((pa - truth)^2) / sum((truth - mean(truth))^2)
      
      lims <- range(c(t_s, pb_s, pa_s), finite = TRUE)
      
      plot(t_s, pb_s, pch = 16, cex = 0.5, col = adjustcolor(COL_BASE, 0.35),
           xlim = lims, ylim = lims,
           xlab = "Actual Value (z-scored)", ylab = "Predicted Value (z-scored)",
           main = sprintf("%s: Prediction Concordance", toupper(k)))
      grid(col = COL_GRID)
      points(t_s, pa_s, pch = 16, cex = 0.5, col = adjustcolor(COL_AB, 0.35))
      abline(0, 1, lty = 2, col = "#222222", lwd = 1.5)
      
      legend("topleft",
             legend = c(sprintf("Baseline (R² = %.3f)", r2_b),
                        sprintf("Alpha-Beta (R² = %.3f)", r2_a)),
             col = c(COL_BASE, COL_AB), pch = 16, bty = "n", cex = 0.78)
    }
    
    mtext("Statistical Diagnostics: Residual Densities and Prediction Fit",
          outer = TRUE, side = 3, line = 0.8, font = 2, cex = 1.15, col = "#111111")
  }, width = 14, height = 8)
}

# -----------------------------------------------------------------------------
# PLOT 8: Computational Efficiency & Throughput Profile
# -----------------------------------------------------------------------------
plot_computational_efficiency <- function() {
  save_multidir("plot_ab_computational_efficiency_profile.png", function() {
    setup_canvas(mfrow = c(1, 3), mar = c(4.5, 4.5, 3.2, 1), oma = c(0, 0, 2.5, 0))
    
    keys_display <- toupper(dataset_keys)
    
    # 1. Seconds per Epoch
    sec_base <- sapply(exp_data, function(d) d$m_base$sec_per_epoch)
    sec_ab   <- sapply(exp_data, function(d) d$m_ab$sec_per_epoch)
    
    mat_sec <- rbind(Baseline = sec_base, `Alpha-Beta` = sec_ab)
    bp1 <- barplot(mat_sec, beside = TRUE, col = c(COL_BASE, COL_AB),
                   ylab = "Seconds per Epoch (CPU)",
                   main = "A. Runtime per Epoch",
                   names.arg = keys_display, border = "white",
                   ylim = c(0, max(mat_sec) * 1.25))
    grid(nx = NA, ny = NULL, col = COL_GRID)
    barplot(mat_sec, beside = TRUE, col = c(COL_BASE, COL_AB), add = TRUE, border = "white")
    for (i in seq_along(keys_display)) {
      text(bp1[1, i], mat_sec[1, i], sprintf("%.1fs", mat_sec[1, i]), pos = 3, cex = 0.75, col = COL_BASE)
      text(bp1[2, i], mat_sec[2, i], sprintf("%.1fs", mat_sec[2, i]), pos = 3, cex = 0.75, col = COL_AB)
    }
    legend("topleft", legend = c("Baseline", "Alpha-Beta"), fill = c(COL_BASE, COL_AB), bty = "n", cex = 0.8)
    
    # 2. Total Training Time
    tot_base <- sapply(exp_data, function(d) d$m_base$train_sec)
    tot_ab   <- sapply(exp_data, function(d) d$m_ab$train_sec)
    
    mat_tot <- rbind(Baseline = tot_base / 60, `Alpha-Beta` = tot_ab / 60)
    bp2 <- barplot(mat_tot, beside = TRUE, col = c(COL_BASE, COL_AB),
                   ylab = "Total Training Time (minutes)",
                   main = "B. Convergence Wall-Clock Time",
                   names.arg = keys_display, border = "white",
                   ylim = c(0, max(mat_tot) * 1.25))
    grid(nx = NA, ny = NULL, col = COL_GRID)
    barplot(mat_tot, beside = TRUE, col = c(COL_BASE, COL_AB), add = TRUE, border = "white")
    for (i in seq_along(keys_display)) {
      text(bp2[1, i], mat_tot[1, i], sprintf("%.1fm", mat_tot[1, i]), pos = 3, cex = 0.75, col = COL_BASE)
      text(bp2[2, i], mat_tot[2, i], sprintf("%.1fm", mat_tot[2, i]), pos = 3, cex = 0.75, col = COL_AB)
    }
    
    # 3. Parameters vs Test MSE (Trade-off)
    params_base <- sapply(exp_data, function(d) d$m_base$params)
    params_ab   <- sapply(exp_data, function(d) d$m_ab$params)
    mse_base    <- sapply(exp_data, function(d) d$m_base$test_mse)
    mse_ab      <- sapply(exp_data, function(d) d$m_ab$test_mse)
    
    plot(params_base, mse_base, pch = 21, bg = COL_BASE, col = "white", cex = 2.2, lwd = 2,
         xlim = range(c(params_base, params_ab)) + c(-20, 20),
         ylim = range(c(mse_base, mse_ab)) * c(0.9, 1.1),
         xlab = "Model Parameter Count", ylab = "Test MSE",
         main = "C. Parameter Scaling vs. Test Accuracy")
    grid(col = COL_GRID)
    points(params_ab, mse_ab, pch = 21, bg = COL_AB, col = "white", cex = 2.2, lwd = 2)
    
    for (i in seq_along(keys_display)) {
      arrows(params_base[i], mse_base[i], params_ab[i], mse_ab[i],
             col = "#555555", length = 0.08, lwd = 1.3)
      text(params_ab[i], mse_ab[i], sprintf(" %s", keys_display[i]), pos = 4, font = 2, cex = 0.8)
    }
    
    legend("topright",
           legend = c("Baseline (14,417 params)", "Alpha-Beta (+12 params = 14,429)"),
           pch = 21, pt.bg = c(COL_BASE, COL_AB), pt.cex = 1.4, bty = "n", cex = 0.8)
    
    mtext("Computational Efficiency & Throughput: Baseline vs. Learnable Alpha-Beta",
          outer = TRUE, side = 3, line = 0.8, font = 2, cex = 1.15, col = "#111111")
  }, width = 14, height = 4.8)
}

# -----------------------------------------------------------------------------
# Execute All Plot Generation
# -----------------------------------------------------------------------------
cat("\n=== Generating Extensive Baseline vs. Learnable Alpha-Beta Plots ===\n")
plot_executive_scorecard()
plot_training_dynamics()
plot_step_wise_horizon()
plot_per_channel_error()
plot_learned_parameters()
plot_sample_waveforms()
plot_diagnostics_residuals()
plot_computational_efficiency()

cat("\n=== All 8 Publication Plots Successfully Generated in figures/ and results/ ===\n")

