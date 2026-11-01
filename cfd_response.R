# Does Cascaded Frequency Decomposition (CFD) really split a series into separate frequency bands?
# Paper, Sec. 3.3: "compared to x_i, x^_i lacks the i-th frequency component", so f_i = x_i - x^_i
# should contain only band i. We check this with the model's own pooling and upsampling (no training).
#
# 1. Feed a pure cosine at every frequency k/96 and measure how much of it ends up in each band
#    (at the input frequency, and at other frequencies, i.e. aliasing).
# 2. On the ETT oil temperature series, measure what share of each band's energy is in its own range.
#
# Writes figures/cfd_response.png, results/cfd_sweep.csv, results/cfd_real_energy.csv
# Usage:  Rscript cfd_response.R

source("timekan.R")   # model + data helpers (training does not run when sourced)
dir.create("figures", showWarnings = FALSE)

T <- 96
pool <- nn_avg_pool1d(2, stride = 2)

# The three bands of the default model (3 levels, window 2), each upsampled back to length T.
# x: [B, 1, T] -> list of three [B, T] matrices
get_bands <- function(x) {
  x2 <- pool(x); x3 <- pool(x2)
  up <- function(t, n) frequency_upsample(t$transpose(2, 3), n)   # works on [B, L, 1]
  f1 <- x$transpose(2, 3) - up(x2, T)
  f2 <- up(x2 - up(x3, T / 2)$transpose(2, 3), T)
  f3 <- up(x3, T)
  lapply(list(f1 = f1, f2 = f2, f3 = f3), function(f) as.matrix(f$squeeze(3)))
}

# Amplitude of each frequency 0..T/2 in each row of M.
amplitudes <- function(M) {
  A <- t(apply(M, 1, function(v) Mod(fft(v))))[, 1:(T / 2 + 1), drop = FALSE] / T
  A[, 2:(T / 2)] <- 2 * A[, 2:(T / 2)]
  A
}

# --- 1. Frequency sweep ---
k <- 1:47
X <- t(sapply(k, function(j) cos(2 * pi * j * (0:(T - 1)) / T)))
bands <- get_bands(torch_tensor(X, dtype = torch_float())$unsqueeze(2))
sweep <- data.frame(k = k, freq = k / T)
for (b in names(bands)) {
  A <- amplitudes(bands[[b]])
  sweep[[paste0(b, "_gain")]] <- A[cbind(k, k + 1)]                        # at the input frequency
  sweep[[paste0(b, "_alias")]] <- sqrt(rowSums(A[, -1]^2) - A[cbind(k, k + 1)]^2)   # everywhere else
}
write.csv(sweep, "results/cfd_sweep.csv", row.names = FALSE)

# Closed form (see the write-up): f1 keeps |sin(pi f)| of every frequency below 1/4.
low <- sweep$freq < 1/4 & sweep$k != 12
cat(sprintf("f1 gain vs |sin(pi f)|: max error %.1e\n", max(abs(sweep$f1_gain[low] - abs(sin(pi * sweep$freq[low]))))))
cat(sprintf("f1 + f2 + f3 = x: max error %.1e\n", max(abs(bands$f1 + bands$f2 + bands$f3 - X))))

# --- 2. Real data: share of each band's energy in low [0, 1/8], mid (1/8, 1/4], high (1/4, 1/2] ---
range_of <- cut(1:(T / 2) / T, c(0, 1/8, 1/4, 1/2), labels = c("low", "mid", "high"))
real <- NULL
for (ds in c("ETTh1", "ETTh2", "ETTm1", "ETTm2")) {
  ot <- read.csv(sprintf("data/%s.csv", ds))$OT
  W <- matrix(ot[1:(T * (length(ot) %/% T))], ncol = T, byrow = TRUE)
  W <- W - rowMeans(W)                                   # non-overlapping windows, mean removed
  all <- c(list(input = W), get_bands(torch_tensor(W, dtype = torch_float())$unsqueeze(2)))
  for (b in names(all)) {
    energy <- tapply(colSums(amplitudes(all[[b]])[, -1]^2), range_of, sum)
    real <- rbind(real, data.frame(dataset = ds, band = b, t(round(energy / sum(energy), 3))))
  }
}
write.csv(real, "results/cfd_real_energy.csv", row.names = FALSE)
print(real, row.names = FALSE)

# --- Figure ---
cols <- c(f1 = "#2a78d6", f2 = "#eb6834", f3 = "#1baf7a")
png("figures/cfd_response.png", width = 1800, height = 1400, res = 170)
par(mfrow = c(2, 1), mar = c(3, 4.5, 3, 1), oma = c(2, 0, 2, 0), bty = "n", las = 1)
for (what in c("gain", "alias")) {
  plot(NA, xlim = c(0, 0.5), ylim = c(0, 1.1), xaxs = "i", xaxt = "n", xlab = "",
       ylab = if (what == "gain") "amplitude kept" else "amplitude elsewhere")
  axis(1, at = c(0, 1/8, 1/4, 3/8, 1/2), labels = c("0", "1/8", "1/4", "3/8", "1/2"))
  rect(1/8, 0, 1/4, 1.1, col = "#f3f2ee", border = NA)
  text(c(1/16, 3/16, 3/8), 1.08, c("low", "mid", "high"), col = "gray40", cex = 0.85)
  for (b in names(cols)) {
    y <- sweep[[paste0(b, "_", what)]]
    if (what == "gain") y[k %in% c(12, 24)] <- NA   # edge bins f = 1/8, 1/4 depend on phase
    lines(sweep$freq, y, type = "o", pch = 19, cex = 0.6, lwd = 1.5, col = cols[[b]])
  }
  mtext(if (what == "gain") "A. How much of the input frequency each band keeps"
        else "B. Aliasing: amplitude at frequencies that are not in the input",
        side = 3, adj = 0, line = 0.6, font = 2)
}
text(0.01, 0.5, "f2 and f3 overlap here: equal and\nopposite, they cancel when summed", adj = 0, cex = 0.8, col = "gray30")
mtext("Input frequency (cycles per step)", side = 1, outer = TRUE, line = 0.5)
par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(NA, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE)
legend("top", c("f1 (\"high\")", "f2 (\"mid\")", "f3 (\"low\")"), col = cols, lwd = 2, pch = 19,
       horiz = TRUE, bty = "n", text.width = 0.12)
dev.off()
