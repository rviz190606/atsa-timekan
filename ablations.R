# Missing paper ablations. Source baseline is unchanged.
source("timekan.R")
original_layer <- timekan_layer
options_ablation <- modifyList(default_options, list(
  upsample = "fft", representation = "kan", temporal = "depthwise",
  heads = 4, mlp_hidden = 32, self_test = 0,
  batch_size = 128, lr = 0.01
))
context <- options_ablation

band_upsampler <- nn_module(
  "BandUpsampler",
  initialize = function(dim, short, long, kind) {
    self$kind <- kind; self$long <- long
    if (kind == "linear_map") self$map <- nn_linear(short, long)
    if (kind == "transpose_conv") {
      ratio <- long / short
      stopifnot(ratio == as.integer(ratio))
      self$map <- nn_conv_transpose1d(dim, dim, kernel_size = ratio,
                                     stride = ratio, bias = TRUE)
    }
    stopifnot(kind %in% c("fft", "linear", "linear_map", "transpose_conv"))
  },
  forward = function(x) {
    if (self$kind == "fft") return(frequency_upsample(x, self$long))
    if (self$kind == "linear") {
      return(nnf_interpolate(x$transpose(2, 3), size = self$long,
                             mode = "linear", align_corners = FALSE)$transpose(2, 3))
    }
    self$map(x$transpose(2, 3))$transpose(2, 3)
  }
)

temporal_attention <- nn_module(
  "TemporalAttention",
  initialize = function(dim, heads) {
    stopifnot(dim %% heads == 0)
    self$heads <- heads; self$width <- dim / heads; self$dim <- dim
    self$q <- nn_linear(dim, dim); self$k <- nn_linear(dim, dim)
    self$v <- nn_linear(dim, dim); self$out <- nn_linear(dim, dim)
  },
  forward = function(x) {
    batch <- x$size(1); len <- x$size(2)
    divide <- function(y) y$reshape(c(batch, len, self$heads, self$width))$permute(c(1, 3, 2, 4))
    q <- divide(self$q(x)); k <- divide(self$k(x)); v <- divide(self$v(x))
    weights <- nnf_softmax(torch_matmul(q, k$transpose(3, 4)) / sqrt(self$width), dim = 4)
    joined <- torch_matmul(weights, v)$permute(c(1, 3, 2, 4))$contiguous()$
      reshape(c(batch, len, self$dim))
    self$out(joined)
  }
)

ablation_mkan <- nn_module(
  "AblationMKAN",
  initialize = function(dim, order, representation, temporal, heads, mlp_hidden) {
    self$temporal <- temporal
    # Preserve baseline parameter initialization order in the default case.
    if (temporal %in% c("depthwise", "standard")) {
      self$conv <- nn_conv1d(dim, dim, 3, padding = 1,
                             groups = if (temporal == "depthwise") dim else 1)
    } else if (temporal == "attention") {
      self$attention <- temporal_attention(dim, heads)
    } else stopifnot(temporal == "none")
    if (representation == "kan") self$representation <- cheby_kan(dim, order)
    else if (representation == "mlp") {
      self$representation <- nn_sequential(nn_linear(dim, mlp_hidden),
                                           nn_gelu(), nn_linear(mlp_hidden, dim))
    } else stop("representation must be kan or mlp")
  },
  forward = function(x) {
    rep <- self$representation(x)
    if (self$temporal == "none") return(rep)
    if (self$temporal == "attention") return(rep + self$attention(x))
    self$conv(x$transpose(2, 3))$transpose(2, 3) + rep
  }
)

timekan_layer <- nn_module(
  "AblationTimeKANLayer",
  initialize = function(dim, num_levels, base_order, fixed_order = 0) {
    k <- num_levels
    lengths <- context$seq_len / context$window^(0:(k - 1))
    order_of <- function(i) if (fixed_order > 0) fixed_order else base_order + k - i
    self$mkans <- nn_module_list(lapply(seq_len(k), function(i)
      ablation_mkan(dim, order_of(i), context$representation, context$temporal,
                    context$heads, context$mlp_hidden)))
    # Independent trainable upsamplers per scale and per DLM pass. Both CFD and
    # mixing use the selected method; parameter-free methods have no weights.
    make_up <- function(i) band_upsampler(dim, lengths[i + 1], lengths[i], context$upsample)
    self$decomp_up <- nn_module_list(lapply(seq_len(k - 1), make_up))
    self$mix_up <- nn_module_list(lapply(seq_len(k - 1), make_up))
  },
  forward = function(xs) {
    k <- length(xs); fs <- vector("list", k)
    for (i in seq_len(k - 1)) fs[[i]] <- xs[[i]] - self$decomp_up[[i]](xs[[i + 1]])
    fs[[k]] <- xs[[k]]
    for (i in seq_len(k)) fs[[i]] <- self$mkans[[i]](fs[[i]])
    out <- vector("list", k); out[[k]] <- fs[[k]]
    for (i in rev(seq_len(k - 1))) out[[i]] <- self$mix_up[[i]](out[[i + 1]]) + fs[[i]]
    out
  }
)

ablation_self_tests <- function() {
  x <- torch_randn(2, 8, 4)
  for (kind in c("fft", "linear", "linear_map", "transpose_conv")) {
    module <- band_upsampler(4, 8, 16, kind)
    y <- module(x)
    stopifnot(identical(as.integer(y$size()), c(2L, 16L, 4L)))
    if (length(module$parameters)) {
      y$pow(2)$mean()$backward()
      stopifnot(all(sapply(module$parameters, function(p)
        !is.null(p$grad) && torch_isfinite(p$grad)$all()$item())))
    }
  }
  for (kind in c("none", "depthwise", "standard", "attention")) {
    module <- ablation_mkan(4, 2, "kan", kind, 2, 8)
    y <- module(x); y$pow(2)$mean()$backward()
    stopifnot(identical(as.integer(y$size()), c(2L, 8L, 4L)),
              all(sapply(module$parameters, function(p)
                !is.null(p$grad) && torch_isfinite(p$grad)$all()$item())))
  }
  module <- ablation_mkan(4, 2, "mlp", "depthwise", 2, 8)
  module(x)$pow(2)$mean()$backward()
  stopifnot(all(sapply(module$parameters, function(p) !is.null(p$grad))))
  # Default ablation architecture must exactly reproduce the existing R layer.
  context <<- modifyList(options_ablation, list(seq_len = 16, window = 2))
  inputs <- list(torch_randn(2, 16, 4), torch_randn(2, 8, 4), torch_randn(2, 4, 4))
  torch_manual_seed(123); original <- original_layer(4, 3, 1)
  torch_manual_seed(123); replacement <- timekan_layer(4, 3, 1)
  a <- original(inputs); b <- replacement(inputs)
  stopifnot(all(sapply(seq_along(a), function(i) torch_allclose(a[[i]], b[[i]]))))
  cat("ABLATION_SHAPES_GRADIENTS_AND_BASELINE_PARITY_OK\n")
}

if (sys.nframe() == 0L) {
  opt <- parse_args(options_ablation)
  context <- opt
  if (opt$self_test > 0) ablation_self_tests() else {
    if (nzchar(opt$out)) dir.create(dirname(opt$out), recursive = TRUE, showWarnings = FALSE)
    if (nzchar(opt$save)) {
      dir.create(dirname(opt$save), recursive = TRUE, showWarnings = FALSE)
      capture.output(dput(opt), file = paste0(opt$save, "_options.R"))
    }
    run_experiment(opt)
  }
}
