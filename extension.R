# Extension analysis: are higher-order KANs really what high-frequency bands need?
#
# Part A (order experiment): summarises results/orders.csv from run_order_experiment.sh.
#   figures/order_results.png, results/order_summary.csv
#
# Part C (basis swap): summarises results/basis.csv from run_basis_experiment.sh.
#   figures/basis_results.png, results/basis_summary.csv
#
# Part B (looking inside the KANs): uses the "fixed3" models, where EVERY band has an
# order-3 KAN, so each band is free to learn curved or nearly-straight functions.
# If the paper's intuition is right, the high-frequency band should use more curvature.
#   figures/kan_functions.png     the most important learned functions in each band
#   figures/kan_nonlinearity.png  how much of each band's KAN output is curvature
#
# How "curvature" is measured. Each KAN edge computes
#   phi(x) = sum_i  theta_i * T_i(tanh(x)),   i = 0..order
# T_0 is a constant and T_1(tanh x) = tanh x is a plain squashing curve. Only the degree >= 2
# terms add bends. On real inputs (test windows) we split every edge's output into
#   simple part  (degree 0-1)   and   curved part (degree >= 2)
# and report  curvature share = var(curved) / (var(simple) + var(curved)), summed over edges.
# 0 means the band's KAN acts like a squashed linear layer; near 1 means it is mostly bends.
#
# Usage:  Rscript extension.R

source("timekan.R")   # model + data helpers (training does not run when sourced)
dir.create("figures", showWarnings = FALSE)

BLUE <- "#2a78d6"; ORANGE <- "#eb6834"
INK <- "#0b0b0b"; INK2 <- "#52514e"; GRID <- "#e4e3df"; LIGHT <- "#b9b8b2"
setup_par <- function(...) {
  par(family = "sans", bty = "n", las = 1, col.axis = INK2, col.lab = INK2,
      fg = INK2, mgp = c(2.4, 0.6, 0), tcl = -0.25, ...)
}
BAND_NAMES <- c("High frequency", "Mid frequency", "Low frequency")


# ---------------------------------------------------------------------------
# Shared: compare variants of the model across seeds (used by Parts A and C)
# ---------------------------------------------------------------------------

# r: results rows with dataset, pred_len, tag, test_mse.  labels: named vector tag -> axis label
# (also sets the order).  highlight: the tag drawn in blue (the paper's choice).
# Writes a mean/sd summary CSV and a figure with one panel per dataset/horizon.
compare_variants <- function(r, labels, highlight, title, fig, summary_csv) {
  s <- aggregate(test_mse ~ dataset + pred_len + tag, r,
                 function(v) c(mean = mean(v), sd = sd(v), n = length(v)))
  s <- data.frame(s[, 1:3], round(s$test_mse, 4))
  s <- s[order(s$dataset, s$pred_len, match(s$tag, names(labels))), ]
  write.csv(s, summary_csv, row.names = FALSE)
  cat("wrote", summary_csv, "\n"); print(s, row.names = FALSE)

  groups <- unique(r[, c("dataset", "pred_len")])
  groups <- groups[order(groups$dataset, groups$pred_len), ]
  png(fig, width = 1900, height = 650, res = 160)
  setup_par(mfrow = c(1, nrow(groups)), mar = c(7.5, 4.2, 3, 0.5), oma = c(0, 0, 2.2, 0))
  for (g in seq_len(nrow(groups))) {
    rg <- r[r$dataset == groups$dataset[g] & r$pred_len == groups$pred_len[g], ]
    tags <- names(labels)[names(labels) %in% rg$tag]
    means <- sapply(tags, function(m) mean(rg$test_mse[rg$tag == m]))
    lo <- min(rg$test_mse) * 0.97; hi <- max(rg$test_mse) * 1.02   # zoomed: differences are small
    bp <- barplot(means - lo, offset = lo, ylim = c(lo, hi), xpd = FALSE, names.arg = rep("", length(tags)),
                  col = ifelse(tags == highlight, BLUE, LIGHT), border = "white", lwd = 2,
                  ylab = if (g == 1) "Test MSE (lower is better)" else "", space = 0.35)
    # One dot per seed, so the run-to-run spread is visible.
    for (k in seq_along(tags)) {
      v <- rg$test_mse[rg$tag == tags[k]]
      points(rep(bp[k], length(v)) + c(-0.12, 0, 0.12)[seq_along(v)], v, pch = 19, cex = 0.7, col = INK)
    }
    text(bp, par("usr")[3] - 0.03 * diff(par("usr")[3:4]), labels[tags], srt = 35, adj = 1,
         xpd = NA, cex = 0.8, col = INK2)
    mtext(sprintf("%s, horizon %d", groups$dataset[g], groups$pred_len[g]),
          side = 3, adj = 0, line = 0.8, font = 2, col = INK, cex = 0.9)
  }
  mtext(paste0(title, " (bars: mean of 3 seeds, dots: each seed; y-axis zoomed)"),
        side = 3, outer = TRUE, adj = 0.01, font = 2, col = INK, line = 0.6)
  dev.off()
  cat("wrote", fig, "\n")
}


# ---------------------------------------------------------------------------
# Part A: order experiment results
# ---------------------------------------------------------------------------
MODES <- c(paper = "Paper  (3,2,1)", reversed = "Reversed  (1,2,3)",
           fixed2 = "All order 2  (2,2,2)", fixed3 = "All order 3  (3,3,3)")

if (file.exists("results/orders.csv")) {
  compare_variants(read.csv("results/orders.csv"), MODES, "paper",
                   "KAN order assignment across frequency bands",
                   "figures/order_results.png", "results/order_summary.csv")
}


# ---------------------------------------------------------------------------
# Part C: basis-function swap (Chebyshev vs. other function families)
# ---------------------------------------------------------------------------
# Chebyshev = the "paper" runs of the order experiment (same settings and seeds).
BASES <- c(chebyshev = "Chebyshev (paper)", legendre = "Legendre", fourier = "Fourier",
           rbf = "Gaussian RBF", linear = "Linear (control)")

if (file.exists("results/basis.csv") && file.exists("results/orders.csv")) {
  cols <- c("dataset", "pred_len", "tag", "test_mse", "minutes")
  cheb_runs <- read.csv("results/orders.csv")
  cheb_runs <- cheb_runs[cheb_runs$tag == "paper", ]
  cheb_runs$tag <- "chebyshev"
  rb <- rbind(cheb_runs[, cols], read.csv("results/basis.csv")[, cols])
  compare_variants(rb, BASES, "chebyshev", "KAN basis functions",
                   "figures/basis_results.png", "results/basis_summary.csv")
}


# ---------------------------------------------------------------------------
# Part B: looking inside the KANs of the fixed-order-3 models
# ---------------------------------------------------------------------------

# Chebyshev polynomials T_0..T_order of tanh(x), stacked on a new last dim (same as cheby_kan).
cheb <- function(x, order) {
  x <- torch_tanh(x)
  p <- list(torch_ones_like(x), x)
  if (order >= 2) for (n in 2:order) p[[n + 1]] <- 2 * x * p[[n]] - p[[n - 1]]
  torch_stack(p[1:(order + 1)], dim = -1)
}

# The real inputs reaching each band's KAN, for every layer: list[layer][band] -> [rows, D].
band_inputs <- function(model, x, max_rows = 20000) {
  with_no_grad({
    xs <- model$encode(x)$xs
    out <- list()
    for (l in seq_along(model$layers)) {
      fs <- model$layers[[l]]$decompose(xs)
      out[[l]] <- lapply(fs, function(f) {
        f <- f$reshape(c(-1, f$size(3)))
        if (f$size(1) > max_rows) f <- f[sample.int(f$size(1), max_rows), ]
        f
      })
      xs <- model$layers[[l]](xs)
    }
  })
  out
}

# Per-edge output on real inputs, split into simple (degree 0-1) and curved (degree >= 2) parts.
edge_parts <- function(kan, inp) {
  C <- kan$coeffs$detach()                          # [D_out, D_in, order+1]
  Tm <- cheb(inp, kan$order)                        # [n, D_in, order+1]
  k <- kan$order + 1
  simple <- torch_einsum("nji,oji->noj", list(Tm[, , 1:2], C[, , 1:2]))
  curved <- torch_einsum("nji,oji->noj", list(Tm[, , 3:k], C[, , 3:k]))
  list(simple = simple, curved = curved)
}

curvature_share <- function(kan, inp) {
  p <- edge_parts(kan, inp)
  vs <- p$simple$var(dim = 1)$sum()$item(); vc <- p$curved$var(dim = 1)$sum()$item()
  vc / (vs + vc)
}

test_windows <- function(csv, n = 256) {
  d <- prepare_data(parse_args(default_options, c("--csv", csv)))
  ds <- window_dataset(d$test, 96, 96)
  ds$.getbatch(round(seq(1, length(ds), length.out = n)))$x   # evenly spaced test windows
}

model_files <- Sys.glob("results/orders/*_96_fixed3_s*.pt")
if (length(model_files)) {
  set.seed(0)
  rows <- list()
  for (f in model_files) {
    ds_name <- sub("_96_.*", "", basename(f)); seed <- sub(".*_s(\\d+)\\.pt", "\\1", basename(f))
    model <- torch_load(f, device = "cpu"); model$to(device = "cpu"); model$eval()
    inp <- band_inputs(model, test_windows(sprintf("data/%s.csv", ds_name)))
    for (l in seq_along(inp)) for (b in seq_along(inp[[l]])) {
      kan <- model$layers[[l]]$mkans[[b]]$kan
      rows[[length(rows) + 1]] <- data.frame(dataset = ds_name, seed = seed, layer = l, band = b,
                                             share = with_no_grad(curvature_share(kan, inp[[l]][[b]])))
    }
  }
  cs <- do.call(rbind, rows)
  write.csv(cs, "results/kan_curvature.csv", row.names = FALSE)
  agg <- aggregate(share ~ dataset + band, cs, mean)
  cat("curvature share by band (mean over seeds and layers):\n"); print(agg, row.names = FALSE)

  # --- Figure: curvature share per band, each dataset; dots = seed x layer ---
  sets <- unique(cs$dataset)
  png("figures/kan_nonlinearity.png", width = 1600, height = 700, res = 160)
  setup_par(mfrow = c(1, length(sets)), mar = c(4, 4.5, 3, 1), oma = c(0, 0, 2.4, 0))
  for (s in sets) {
    c1 <- cs[cs$dataset == s, ]
    m <- sapply(1:3, function(b) mean(c1$share[c1$band == b]))
    bp <- barplot(m, names.arg = BAND_NAMES, col = BLUE, border = "white", lwd = 2,
                  ylim = c(0, max(c1$share) * 1.15), space = 0.4,
                  ylab = if (s == sets[1]) "Curvature share of KAN output" else "")
    for (b in 1:3) {
      v <- c1$share[c1$band == b]
      points(bp[b] + seq(-0.2, 0.2, length.out = length(v)), v, pch = 19, cex = 0.6, col = INK)
    }
    text(bp, 0.08, sprintf("%.2f", m), cex = 0.9, col = "white", font = 2)   # mean, inside the bar
    mtext(s, side = 3, adj = 0, line = 0.8, font = 2, col = INK)
  }
  mtext("With equal capacity (order 3 everywhere), how curved is what each band learns?",
        side = 3, outer = TRUE, adj = 0.01, font = 2, cex = 1.05, col = INK, line = 0.8)
  mtext("bars: mean; dots: each seed and layer", side = 3, outer = TRUE, adj = 0.01,
        col = INK2, cex = 0.85, line = -0.4)
  dev.off()
  cat("wrote figures/kan_nonlinearity.png\n")

  # --- Figure: the most important learned edge functions in each band (ETTh1, seed 0, layer 1) ---
  f <- model_files[grepl("ETTh1_96_fixed3_s0", model_files)][1]
  if (!is.na(f)) {
    model <- torch_load(f, device = "cpu"); model$to(device = "cpu"); model$eval()
    inp <- band_inputs(model, test_windows("data/ETTh1.csv"))[[1]]
    png("figures/kan_functions.png", width = 1800, height = 650, res = 160)
    setup_par(mfrow = c(1, 3), mar = c(4, 4, 3.2, 1), oma = c(0, 0, 2.2, 0))
    with_no_grad({
      for (b in 1:3) {
        kan <- model$layers[[1]]$mkans[[b]]$kan
        x_in <- inp[[b]]
        p <- edge_parts(kan, x_in)
        imp <- as.matrix((p$simple + p$curved)$var(dim = 1))       # [D_out, D_in] importance
        top <- order(imp, decreasing = TRUE)[1:8]
        o_idx <- (top - 1) %% nrow(imp) + 1; j_idx <- (top - 1) %/% nrow(imp) + 1
        C <- as.array(kan$coeffs$detach())
        curves <- lapply(seq_along(top), function(k) {
          xr <- as.numeric(x_in[, j_idx[k]])
          grid <- seq(quantile(xr, 0.01), quantile(xr, 0.99), length.out = 200)  # typical input range
          Tg <- as.array(cheb(torch_tensor(grid), kan$order))
          y <- as.numeric(Tg %*% C[o_idx[k], j_idx[k], ])
          list(x = grid, y = y - mean(y))                           # centred: compare shapes
        })
        plot(NA, xlim = range(sapply(curves, function(cv) range(cv$x))),
             ylim = range(sapply(curves, function(cv) range(cv$y))),
             xlab = "Input value (typical range)", ylab = if (b == 1) "Learned function (centred)" else "")
        abline(h = 0, col = GRID)
        for (cv in curves) lines(cv$x, cv$y, col = adjustcolor(BLUE, 0.75), lwd = 2)
        share <- curvature_share(kan, x_in)
        mtext(BAND_NAMES[b], side = 3, adj = 0, line = 1.5, font = 2, col = INK)
        mtext(sprintf("curvature share %.2f", share), side = 3, adj = 0, line = 0.4, col = INK2, cex = 0.85)
      }
    })
    mtext("ETTh1: the 8 most important learned KAN functions per band (layer 1, all bands order 3)",
          side = 3, outer = TRUE, adj = 0.01, font = 2, cex = 1.05, col = INK, line = 0.6)
    dev.off()
    cat("wrote figures/kan_functions.png\n")
  }
}
