# Figures and tables for presenting the TimeKAN reproduction.
# Reads what run_experiments.sh has produced so far (partial results are fine) and writes:
#   figures/decomposition.png   how TimeKAN splits an input window into frequency bands
#   figures/forecast.png        a forecast vs. the true values (ETTh1, horizon 96)
#   figures/main_results.png    our MSE vs. the paper's, per dataset and horizon
#   figures/ablation.png        effect of the number of bands and of fixed-order KANs
#   results/comparison.csv      the numbers behind main_results.png
# Usage:  Rscript plots.R

source("timekan.R")   # model + data helpers (training does not run when sourced)
dir.create("figures", showWarnings = FALSE)

# Colours: blue = this implementation, orange = paper, grays = context.
BLUE <- "#2a78d6"; ORANGE <- "#eb6834"
INK <- "#0b0b0b"; INK2 <- "#52514e"; GRID <- "#e4e3df"; LIGHT <- "#b9b8b2"

# Shared look: no box, light grid, gray axis text.
setup_par <- function(...) {
  par(family = "sans", bty = "n", las = 1, col.axis = INK2, col.lab = INK2,
      fg = INK2, mgp = c(2.4, 0.6, 0), tcl = -0.25, ...)
}
grid_y <- function() abline(h = axTicks(2), col = GRID, lwd = 1)

# TimeKAN's own results from Table 1 of the paper (look-back 96).
paper <- data.frame(
  dataset  = rep(c("ETTh1", "ETTh2", "ETTm1", "ETTm2"), each = 4),
  pred_len = rep(c(96, 192, 336, 720), 4),
  paper_mse = c(0.367, 0.414, 0.445, 0.444,  0.290, 0.375, 0.423, 0.443,
                0.322, 0.357, 0.382, 0.445,  0.174, 0.239, 0.301, 0.395),
  paper_mae = c(0.395, 0.420, 0.434, 0.459,  0.340, 0.392, 0.435, 0.449,
                0.361, 0.383, 0.401, 0.435,  0.255, 0.299, 0.340, 0.396))


# ---------------------------------------------------------------------------
# 1. Frequency decomposition of one real input window (no training needed)
# ---------------------------------------------------------------------------
# Same steps as the model's first stage, applied to the raw series so it is easy to see:
# smooth with moving averages, then each band = level minus the upsampled smoother level.
opt <- parse_args(default_options, c("--csv", "data/ETTh1.csv"))
d <- prepare_data(opt)
ot <- ncol(d$test)                                   # "OT" (oil temperature) column
x <- torch_tensor(d$test[1:96, ot], dtype = torch_float())$view(c(1, 1, 96))
pool <- nn_avg_pool1d(2, stride = 2)
lv <- list(x, pool(x), pool(pool(x)))                 # lengths 96, 48, 24
as_seq <- function(t) t$view(c(1, -1, 1))             # [1, L, 1] for frequency_upsample
bands <- list(
  "High-frequency band  f1 = x1 - up(x2)" = as_seq(lv[[1]]) - frequency_upsample(as_seq(lv[[2]]), 96),
  "Mid-frequency band  f2 = x2 - up(x3)"  = as_seq(lv[[2]]) - frequency_upsample(as_seq(lv[[3]]), 48),
  "Low-frequency band  f3 = x3"           = as_seq(lv[[3]]))

png("figures/decomposition.png", width = 1600, height = 1300, res = 160)
setup_par(mfrow = c(4, 1), mar = c(2.2, 4, 2.2, 1), oma = c(2, 0, 2.5, 0))
panel <- function(v, title, col) {
  v <- as.numeric(v$cpu())
  tt <- seq(1, 96, length.out = length(v))           # stretch shorter levels to 96 steps
  plot(tt, v, type = "n", xlab = "", ylab = "", xlim = c(1, 96))
  grid_y()
  lines(tt, v, col = col, lwd = 2)
  mtext(title, side = 3, adj = 0, line = 0.4, col = INK, cex = 0.85, font = 2)
}
panel(lv[[1]]$view(-1), "Input window x1 (ETTh1, oil temperature, normalised)", INK2)
for (nm in names(bands)) panel(bands[[nm]]$view(-1), nm, BLUE)
mtext("Time step within the 96-step look-back window", side = 1, outer = TRUE, col = INK2, line = 0.5)
mtext("Cascaded frequency decomposition", side = 3, outer = TRUE, adj = 0.02,
      font = 2, cex = 1.1, col = INK, line = 0.8)
dev.off()
cat("wrote figures/decomposition.png\n")


# ---------------------------------------------------------------------------
# 2. A forecast vs. the truth, from the saved ETTh1 horizon-96 model
# ---------------------------------------------------------------------------
if (file.exists("results/ETTh1_96.pt")) {
  model <- torch_load("results/ETTh1_96.pt", device = "cpu")
  model$to(device = "cpu")
  model$eval()
  # Show a typical window, not a cherry-picked one: the test window whose
  # OT-column error is the median over all test windows.
  ds <- window_dataset(d$test, 96, 96)
  errs <- unlist(lapply(split(seq_len(length(ds)), ceiling(seq_len(length(ds)) / 256)), function(idx) {
    b <- ds$.getbatch(idx)
    p <- with_no_grad(model(b$x))
    as.numeric(((p - b$y)[, , ot])$pow(2)$mean(dim = 2))
  }))
  start <- order(errs)[ceiling(length(errs) / 2)]
  cat(sprintf("forecast plot: test window %d (median OT MSE %.3f)\n", start, errs[start]))
  hist_rows <- start:(start + 95); fut_rows <- (start + 96):(start + 191)
  inp <- torch_tensor(d$test[hist_rows, , drop = FALSE], dtype = torch_float())$unsqueeze(1)
  pred <- with_no_grad(as.matrix(model(inp)$squeeze(1)))
  # Back to original units (degrees C) for the OT column.
  unscale <- function(v) v * d$sdev[ot] + d$mu[ot]
  h <- unscale(d$test[hist_rows, ot]); y <- unscale(d$test[fut_rows, ot]); p <- unscale(pred[, ot])

  png("figures/forecast.png", width = 1600, height = 800, res = 160)
  setup_par(mar = c(4, 4.2, 3.5, 7))
  plot(NA, xlim = c(-95, 96), ylim = range(h, y, p), xlab = "Hours relative to forecast start",
       ylab = "Oil temperature (°C)")
  grid_y()
  abline(v = 0.5, col = LIGHT, lty = 2)
  lines(-95:0, h, col = INK2, lwd = 2)
  lines(1:96, y, col = INK2, lwd = 2)
  lines(1:96, p, col = BLUE, lwd = 2)
  # Direct labels at the right edge, in text colour next to a coloured key mark.
  par(xpd = NA)
  segments(100, y[96], 106, y[96], col = INK2, lwd = 2); text(108, y[96], "Actual", adj = 0, col = INK)
  segments(100, p[96], 106, p[96], col = BLUE, lwd = 2); text(108, p[96], "TimeKAN", adj = 0, col = INK)
  par(xpd = FALSE)
  text(-47, par("usr")[4], "look-back (input)", col = INK2, pos = 1, cex = 0.85)
  text(48, par("usr")[4], "forecast", col = INK2, pos = 1, cex = 0.85)
  mtext("ETTh1: 96-hour forecast of oil temperature from the previous 96 hours",
        side = 3, adj = 0, line = 1.5, font = 2, col = INK)
  mtext("Test window with the median error (a typical case)", side = 3, adj = 0,
        line = 0.4, col = INK2, cex = 0.85)
  dev.off()
  cat("wrote figures/forecast.png\n")
}


# ---------------------------------------------------------------------------
# 3. Main results vs. the paper
# ---------------------------------------------------------------------------
if (file.exists("results/main.csv")) {
  ours <- read.csv("results/main.csv")
  ours <- ours[!duplicated(ours[, c("dataset", "pred_len")], fromLast = TRUE), ]   # latest run wins
  cmp <- merge(ours[, c("dataset", "pred_len", "test_mse", "test_mae")], paper)
  cmp <- cmp[order(cmp$dataset, cmp$pred_len), ]
  cmp$mse_diff_pct <- round(100 * (cmp$test_mse / cmp$paper_mse - 1), 1)
  cmp[, c("test_mse", "test_mae")] <- round(cmp[, c("test_mse", "test_mae")], 3)
  write.csv(cmp, "results/comparison.csv", row.names = FALSE)
  cat("wrote results/comparison.csv\n"); print(cmp, row.names = FALSE)

  sets <- unique(cmp$dataset)
  png("figures/main_results.png", width = 1800, height = 520 + 0 * length(sets), res = 160)
  setup_par(mfrow = c(1, length(sets)), mar = c(4, 4, 2.5, 0.5), oma = c(0, 0, 2.8, 0))
  ymax <- max(cmp$test_mse, cmp$paper_mse) * 1.12
  for (s in sets) {
    r <- cmp[cmp$dataset == s, ]
    m <- rbind(r$test_mse, r$paper_mse)
    bp <- barplot(m, beside = TRUE, col = c(BLUE, ORANGE), border = "white", lwd = 2,
                  names.arg = r$pred_len, ylim = c(0, ymax), xlab = "Horizon",
                  ylab = if (s == sets[1]) "Test MSE (lower is better)" else "", space = c(0.15, 0.6))
    mtext(s, side = 3, adj = 0, line = 0.6, font = 2, col = INK)
  }
  par(xpd = NA)
  legend("topright", inset = c(0, -0.3), legend = c("This R implementation", "Paper (Table 1)"),
         fill = c(BLUE, ORANGE), border = NA, bty = "n", horiz = TRUE, text.col = INK)
  mtext("Test MSE: this implementation vs. the paper", side = 3, outer = TRUE, adj = 0.01,
        font = 2, cex = 1.1, col = INK, line = 1)
  dev.off()
  cat("wrote figures/main_results.png\n")
}


# ---------------------------------------------------------------------------
# 4. Ablations (horizon 96)
# ---------------------------------------------------------------------------
if (file.exists("results/ablation.csv") && file.exists("results/main.csv")) {
  ab <- read.csv("results/ablation.csv")
  base <- read.csv("results/main.csv")
  base <- base[base$pred_len == 96, ]
  base$tag <- "default: 3 bands, multi-order KANs"
  ab <- rbind(ab[, c("dataset", "tag", "test_mse")], base[, c("dataset", "tag", "test_mse")])
  ab <- ab[ab$dataset %in% unique(read.csv("results/ablation.csv")$dataset), ]
  order_tags <- c("default: 3 bands, multi-order KANs", "levels=2", "levels=4", "levels=5",
                  "fixed order 2", "fixed order 5")
  labels <- c("Default (3 bands, multi-order)", "2 bands", "4 bands", "5 bands",
              "Fixed KAN order 2", "Fixed KAN order 5")
  sets <- unique(ab$dataset)

  png("figures/ablation.png", width = 1700, height = 750, res = 160)
  setup_par(mfrow = c(1, length(sets)), mar = c(4, 12, 2.5, 3), oma = c(0, 0, 2, 0))
  for (s in sets) {
    r <- ab[ab$dataset == s, ]
    v <- sapply(order_tags, function(t) { x <- r$test_mse[r$tag == t]; if (length(x)) tail(x, 1) else NA })
    keep <- !is.na(v)
    bp <- barplot(rev(v[keep]), horiz = TRUE, names.arg = rev(labels[keep]),
                  col = rev(ifelse(order_tags[keep] == order_tags[1], BLUE, LIGHT)),
                  border = "white", lwd = 2, xlim = c(0, max(v, na.rm = TRUE) * 1.2),
                  xlab = "Test MSE, horizon 96 (lower is better)", cex.names = 0.85)
    par(xpd = NA)   # let value labels extend past the plot edge
    text(rev(v[keep]), bp, sprintf("%.3f", rev(v[keep])), pos = 4, cex = 0.8, col = INK)
    par(xpd = FALSE)
    mtext(s, side = 3, adj = 0, line = 0.6, font = 2, col = INK)
  }
  mtext("Ablations: number of frequency bands and KAN order", side = 3, outer = TRUE,
        adj = 0.01, font = 2, cex = 1.1, col = INK, line = 0.3)
  dev.off()
  cat("wrote figures/ablation.png\n")
}
