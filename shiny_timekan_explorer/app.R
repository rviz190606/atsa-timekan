# =============================================================================
# TimeKAN Explorer – Shiny App  (v2)
# Tabs:
#   1. Original TimeKAN   – step-by-step illustration
#   2. Alpha-Beta         – learnable branch weights demo
#   3. PCA Extension      – PCA pre-processing demo
#   4. KAN Basis Explorer – polynomial basis & degree visualiser  [NEW]
#   5. Results            – experiment result plots with explanations
#
# Theme: grey / blue / white  |  plots stay colourful
# =============================================================================

library(shiny)

# =============================================================================
# HELPERS – pure R, no torch
# =============================================================================

make_ts <- function(n = 96, seed = 42) {
  set.seed(seed)
  t <- seq(0, 4 * pi, length.out = n)
  0.6 * sin(6 * t) + 0.9 * sin(2 * t + 0.5) +
    1.2 * cos(0.4 * t) + rnorm(n, 0, 0.12)
}

moving_avg <- function(x, w) {
  if (w <= 1) return(x)
  n <- length(x); out <- numeric(n)
  for (i in seq_len(n)) {
    lo <- max(1, i - w %/% 2); hi <- min(n, i + w %/% 2)
    out[i] <- mean(x[lo:hi])
  }
  out
}

fft_amp <- function(x) {
  sp <- Mod(fft(x))[1:(length(x) %/% 2)]
  sp / max(sp)
}

upsample_r <- function(x, n_target) {
  approx(seq_along(x), x, xout = seq(1, length(x), length.out = n_target))$y
}

# ── Basis functions evaluated on a grid u ∈ [−1, 1] ─────────────────────────
get_basis_label <- function(bas) {
  switch(bas,
    chebyshev = "Chebyshev",
    legendre  = "Legendre",
    fourier   = "Fourier",
    rbf       = "RBF (Gaussian)",
    linear    = "Linear",
    bas
  )
}

eval_basis <- function(u, order, basis) {
  n  <- order
  switch(basis,
    chebyshev = {
      g <- list(rep(1, length(u)), u)
      if (n >= 2) for (k in 1:(n - 1))
        g[[k + 2]] <- 2 * u * g[[k + 1]] - g[[k]]
      g[1:(n + 1)]
    },
    legendre = {
      g <- list(rep(1, length(u)), u)
      if (n >= 2) for (k in 1:(n - 1))
        g[[k + 2]] <- ((2*k+1)*u*g[[k+1]] - k*g[[k]]) / (k+1)
      g[1:(n + 1)]
    },
    fourier = {
      g <- list(rep(1, length(u)))
      m <- 1
      while (length(g) < n + 1) {
        g[[length(g)+1]] <- sin(m * pi * u / 2)
        if (length(g) < n + 1)
          g[[length(g)+1]] <- cos(m * pi * u / 2)
        m <- m + 1
      }
      g[1:(n + 1)]
    },
    rbf = {
      centres <- seq(-1, 1, length.out = n + 1)
      width   <- 2 / max(n, 1)
      lapply(centres, function(c) exp(-((u - c) / width)^2))
    },
    linear = {
      # TimeKAN linear baseline control: g0 = 1, g1 = u (fixed degree 1)
      list(rep(1, length(u)), u)
    }
  )
}

# Fit a signal with the chosen basis (ordinary least squares)
basis_fit <- function(x, order, basis) {
  n  <- length(x)
  u  <- seq(-1, 1, length.out = n)
  B  <- do.call(cbind, eval_basis(u, order, basis))   # n × (order+1)
  cf <- lm.fit(B, x)$coefficients
  fitted <- B %*% cf
  list(fitted = as.numeric(fitted), coeff = cf, u = u, B = B)
}

# =============================================================================
# CSS – grey / blue / white theme
# =============================================================================
app_css <- "
  /* ── global ── */
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f0f2f5;
    color: #222;
  }
  /* ── navbar ── */
  .navbar {
    background: #2c4a7c !important;
    border-bottom: 3px solid #1a3058 !important;
  }
  .navbar-brand {
    color: #fff !important;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .navbar-nav > li > a {
    color: #d6e0f0 !important;
    font-size: 0.93em;
    padding: 14px 16px !important;
  }
  .navbar-nav > li > a:hover,
  .navbar-nav > li.active > a {
    color: #fff !important;
    background: #1a3058 !important;
    border-bottom: 3px solid #7aadff !important;
  }
  /* ── sidebar ── */
  .well {
    background: #e8edf5;
    border: 1px solid #c8d4e8;
    border-radius: 6px;
  }
  /* ── typography ── */
  h4 {
    color: #1e3a5f;
    font-weight: 700;
    margin-top: 6px;
    border-bottom: 2px solid #c8d4e8;
    padding-bottom: 4px;
  }
  h5 { color: #2a5298; font-weight: 600; }
  /* ── caption boxes ── */
  .caption-box {
    background: #fff;
    border-left: 4px solid #4a7fc1;
    padding: 9px 13px;
    border-radius: 3px;
    margin-top: 6px;
    font-size: 0.87em;
    color: #333;
    line-height: 1.55;
  }
  /* ── result cards ── */
  .result-card {
    background: #fff;
    border: 1px solid #dde4f0;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 14px;
  }
  /* ── badges ── */
  .badge-green { background:#2ca02c; color:#fff; padding:2px 8px; border-radius:10px; font-size:.8em; }
  .badge-red   { background:#c0392b; color:#fff; padding:2px 8px; border-radius:10px; font-size:.8em; }
  .badge-grey  { background:#7f8c8d; color:#fff; padding:2px 8px; border-radius:10px; font-size:.8em; }
  .badge-blue  { background:#2a5298; color:#fff; padding:2px 8px; border-radius:10px; font-size:.8em; }
  /* ── misc ── */
  hr { border-top: 1px solid #c8d4e8; }
  .tab-content { padding: 18px 12px; }
  label { font-weight: 600; color: #2c4a7c; font-size: 0.88em; }
  .selectize-input { border-color: #b0c4de; }
"

# =============================================================================
# UI
# =============================================================================
ui <- fluidPage(
  tags$head(tags$style(HTML(app_css))),

  navbarPage(
    title = "TimeKAN Simulation",
    id    = "tabs",

    # =========================================================================
    # TAB 1 – Original TimeKAN
    # =========================================================================
    tabPanel("Original TimeKAN",
      sidebarLayout(
        sidebarPanel(width = 3,
          h4("Controls"),
          sliderInput("ts_len",   "Series length (N)",    min=32, max=128, value=64,  step=8),
          sliderInput("n_levels", "Frequency levels (k)", min=2,  max=5,   value=3),
          sliderInput("window",   "Downsampling window",  min=2,  max=4,   value=2),
          sliderInput("seed",     "Random seed",          min=1,  max=99,  value=42),
          hr(),
          div(class="caption-box",
            strong("What to look for:"), br(),
            "Level 1 = finest / fastest oscillations.", br(),
            "Level k = smoothest trend.", br(),
            "Frequency bands are the gaps between levels.", br(),
            "M-KAN learns each band separately before mixing back."
          )
        ),
        mainPanel(width = 9,
          h4("Step 1 — Hierarchical Sequence Preprocessing"),
          div(class="caption-box",
            "TimeKAN builds ", strong("k progressively smoother versions"), " of the input using
            a moving average with window w, then sub-samples. Level 1 is the original signal;
            each deeper level is a coarser, lower-frequency summary."
          ),
          plotOutput("plt_levels", height="240px"),
          br(),

          h4("Step 2 — Cascaded Frequency Decomposition"),
          div(class="caption-box",
            "Each ", strong("frequency band"), " f_i = x_i − FreqUp(x_{i+1}) isolates what level i
            knows that the smoother level i+1 does not. The last level keeps its low-frequency residual."
          ),
          plotOutput("plt_bands", height="240px"),
          br(),

          h4("Step 3 — Frequency Spectrum per Band"),
          div(class="caption-box",
            "FFT magnitude of each band confirms the separation: band 1 peaks at high frequencies,
            deeper bands concentrate energy at lower frequencies."
          ),
          plotOutput("plt_spectra", height="200px"),
          br(),

          h4("Step 4 — M-KAN Block (Two Branches)"),
          div(class="caption-box",
            "For each band the M-KAN block runs ", strong("two parallel branches"), " then adds them:", br(),
            "  Temporal branch (depthwise 1-D conv): captures local time patterns.", br(),
            "  Channel branch (ChebyKAN): learns non-linear cross-variate relationships."
          ),
          plotOutput("plt_mkan", height="240px"),
          br(),

          h4("Step 5 — Frequency Mixing (Reconstruction)"),
          div(class="caption-box",
            "Bands are recombined bottom-up: x_i = FreqUp(x_{i+1}) + f_hat_i.
            The final level-1 output is passed through a linear head to produce the forecast."
          ),
          plotOutput("plt_mix", height="200px")
        )
      )
    ),

    # =========================================================================
    # TAB 2 – Alpha-Beta Extension
    # =========================================================================
    tabPanel("Alpha-Beta Extension",
      sidebarLayout(
        sidebarPanel(width = 3,
          h4("Learnable Weights"),
          sliderInput("ab_alpha", "alpha  (temporal branch)", min=0.5, max=1.5, value=1.0, step=0.01),
          sliderInput("ab_beta",  "beta   (channel branch)",  min=0.5, max=1.5, value=1.0, step=0.01),
          sliderInput("ab_seed",  "Seed", min=1, max=99, value=7),
          hr(),
          div(class="caption-box",
            strong("Original (Baseline):"), br(),
            "f_hat = f1 + f2   (equal weights, fixed)", br(), br(),
            strong("Our Extension:"), br(),
            "f_hat = alpha * f1 + beta * f2", br(),
            "alpha & beta are learnable scalars initialised at 1.", br(), br(),
            "Move the sliders to simulate what the model learns.
            When alpha > 1 the temporal branch is amplified;
            when beta < 1 the KAN branch is damped."
          )
        ),
        mainPanel(width = 9,
          fluidRow(
            column(6,
              h4("Baseline branch combination"),
              div(class="caption-box",
                "Both branches have equal weight (alpha = beta = 1, fixed)."
              ),
              plotOutput("plt_ab_base", height="230px")
            ),
            column(6,
              h4("Alpha-beta weighted combination"),
              div(class="caption-box",
                "The model learns alpha and beta end-to-end. Adjust the sliders."
              ),
              plotOutput("plt_ab_weighted", height="230px")
            )
          ),
          br(),
          h4("MSE landscape across alpha and beta"),
          div(class="caption-box",
            "Heatmap of combined-output MSE as alpha and beta are swept.
             The star marks the current slider position.
             The cyan contour shows the baseline (alpha=beta=1) MSE level—
             regions inside the contour are ", strong("better than baseline"), "."
          ),
          plotOutput("plt_ab_heatmap", height="320px"),
          br(),
          h4("Why use learnable alpha, beta?"),
          div(class="result-card",
            tags$ul(
              tags$li(strong("Gradient dampening:"), " a small weight on a noisy branch reduces
                      its gradient, stabilising training."),
              tags$li(strong("Dataset-specific balance:"), " temporal structure dominates in some
                      datasets (ETTh2 benefits from down-weighting the KAN branch)."),
              tags$li(strong("Minimal overhead:"), " only 2 extra scalars per M-KAN block
                      (+12 parameters total on a 7-variate model)."),
              tags$li(strong("No structural change:"), " the two branches are unchanged; only
                      their mixture ratio is relaxed.")
            )
          )
        )
      )
    ),

    # =========================================================================
    # TAB 3 – PCA Extension
    # =========================================================================
    tabPanel("PCA Extension",
      sidebarLayout(
        sidebarPanel(width = 3,
          h4("PCA Pre-processing"),
          sliderInput("pca_n_vars",  "Number of variates",       min=4,  max=15,  value=7),
          sliderInput("pca_n_comp",  "PCA components kept (d)",  min=1,  max=15,  value=4),
          sliderInput("pca_seed",    "Seed",                     min=1,  max=99,  value=12),
          sliderInput("pca_cor",     "Inter-variate correlation", min=0.0, max=0.95, value=0.6, step=0.05),
          hr(),
          div(class="caption-box",
            strong("Idea:"), br(),
            "Rotate the N-variate input into principal components, keep the top d
             (capturing most variance), run TimeKAN in PC-space, then project back.", br(), br(),
            strong("Why it may help:"), br(),
            "Removes redundant collinear channels; reduces effective N.", br(), br(),
            strong("Why it struggled:"), br(),
            "ETT datasets have moderate correlation — PCA compresses little
             while introducing a reconstruction error floor."
          )
        ),
        mainPanel(width = 9,
          fluidRow(
            column(6,
              h4("Original multi-variate series"),
              div(class="caption-box","All N channels before PCA."),
              plotOutput("plt_pca_raw", height="230px")
            ),
            column(6,
              h4("After projecting to d PCs"),
              div(class="caption-box",
                "Each PC is a linear combination of the original variates.
                 Adjust 'd' with the slider."
              ),
              plotOutput("plt_pca_proj", height="230px")
            )
          ),
          br(),
          fluidRow(
            column(6,
              h4("Explained-variance scree plot"),
              div(class="caption-box",
                "Cumulative variance explained by each additional PC.
                 Red dashed line marks the chosen cut-off d."
              ),
              plotOutput("plt_pca_scree", height="250px")
            ),
            column(6,
              h4("Reconstruction error vs components"),
              div(class="caption-box",
                "Mean reconstruction MSE when projecting back from d PCs to N channels.
                 The floor is the irreducible error from discarded components."
              ),
              plotOutput("plt_pca_recon", height="250px")
            )
          ),
          br(),
          h4("Limitation summary for ETT / Weather datasets"),
          div(class="result-card",
            tags$table(class="table table-sm table-bordered",
              tags$thead(tags$tr(
                tags$th("Dataset"), tags$th("N vars"), tags$th("Approx corr."),
                tags$th("PCs to reach 90% var"), tags$th("Conclusion")
              )),
              tags$tbody(
                tags$tr(tags$td("ETTh1"),   tags$td("7"),  tags$td("moderate"),
                        tags$td("5–6"), tags$td(span(class="badge-red","Little reduction"))),
                tags$tr(tags$td("ETTh2"),   tags$td("7"),  tags$td("moderate"),
                        tags$td("5–6"), tags$td(span(class="badge-red","Little reduction"))),
                tags$tr(tags$td("Weather"), tags$td("21"), tags$td("low–moderate"),
                        tags$td("16+"), tags$td(span(class="badge-red","High recon error")))
              )
            ),
            br(),
            tags$p("For datasets with weak inter-channel correlation PCA offers almost no
              dimensionality reduction while introducing a reconstruction bottleneck.
              It is better suited to highly correlated, wide datasets (e.g. PEMS traffic).")
          )
        )
      )
    ),

    # =========================================================================
    # TAB 4 – Change polynomial basis
    # =========================================================================
    tabPanel("Change polynomial basis",
      sidebarLayout(
        sidebarPanel(width = 3,
          h4("Basis & Degree"),
          selectInput("basis_type", "Polynomial basis",
            choices  = c("Chebyshev"      = "chebyshev",
                         "Legendre"       = "legendre",
                         "Fourier"        = "fourier",
                         "RBF (Gaussian)" = "rbf",
                         "Linear (control)" = "linear"),
            selected = "chebyshev"
          ),
          sliderInput("basis_order", "Order / degree (n)", min=1, max=8, value=3),
          hr(),
          sliderInput("basis_seed",  "Band signal seed",   min=1, max=99, value=5),
          sliderInput("basis_ts_len","Signal length",      min=32, max=128, value=64, step=8),
          sliderInput("basis_levels","Frequency levels",   min=2, max=5, value=3),
          hr(),
          div(class="caption-box",
            strong("What this tab shows:"), br(),
            "1. The basis functions themselves (what the KAN uses as features).", br(),
            "2. How well the chosen basis fits each frequency band extracted from a sample signal.", br(),
            "3. Fit quality (R²) across bands and orders.",br(), br(),
            strong("Chebyshev"), " — paper default; near-optimal polynomial approximation on [−1,1].", br(),
            strong("Legendre"), " — orthogonal on [−1,1]; similar to Chebyshev, slightly smoother.", br(),
            strong("Fourier"), " — sine/cosine; ideal for periodic signals.", br(),
            strong("RBF"), " — Gaussian bumps; local support, good for irregular patterns.", br(),
            strong("Linear"), " — control baseline: g₀ = 1, g₁ = u (turns KAN into a linear layer)."
          )
        ),
        mainPanel(width = 9,

          h4("Panel A — Basis Functions"),
          div(class="caption-box",
            "Each coloured curve is one basis function g_0, g_1, ..., g_n evaluated over u ∈ [−1, 1].
             The KAN layer learns a ", strong("weighted sum of these"), " to represent any non-linear function.
             Higher order = more oscillations = more expressive but harder to regularise."
          ),
          plotOutput("plt_basis_fns", height="280px"),
          br(),

          h4("Panel B — Fitting a Frequency Band"),
          div(class="caption-box",
            "A sample time-series band is extracted (see Tab 1 decomposition).
             The basis is fitted via ordinary least squares to each of the k bands.
             Blue = actual band signal; red dashed = basis fit.
             Watch how higher orders track finer detail—and can overfit noise."
          ),
          plotOutput("plt_basis_fit_bands", height="300px"),
          br(),

          h4("Panel C — Fit Quality (R²) Across Orders"),
          div(class="caption-box",
            "R² of the basis fit for each frequency band as order increases from 1 to 8.
             High-frequency bands (Band 1) generally need higher orders to fit well;
             low-frequency bands (Band k) are captured even at low order."
          ),
          plotOutput("plt_basis_r2", height="260px"),
          br(),

          h4("Panel D — Basis Coefficient Spectrum"),
          div(class="caption-box",
            "Learned coefficients for the chosen basis and order when fitting Band 1 (highest freq).
             A flat spectrum means the basis spreads the signal evenly across functions;
             a spiked spectrum means only a few functions matter."
          ),
          plotOutput("plt_basis_coeffs", height="220px")
        )
      )
    ),

    # =========================================================================
    # TAB 5 – Results
    # =========================================================================
    tabPanel("Results",
      h4("Experiment Results: Alpha-Beta Extension & PCA Extension"),
      p(style="color:#555;font-size:.9em;",
        "All plots are generated from trained TimeKAN runs on ETTh1, ETTh2, and Weather benchmarks (horizon H=96)."),
      br(),

      tabsetPanel(
        # ----- Sub-tab: Alpha-Beta Results -----
        tabPanel("Alpha-Beta Results",
          br(),
          div(class="result-card",
            h5("Summary Table"),
            div(class="caption-box",
              "Real metrics from trained models. Only +12 parameters added.
               ETTh2 and Weather see clear improvements; ETTh1 is neutral."
            ),
            br(),
            tags$table(class="table table-sm table-bordered",
              tags$thead(tags$tr(
                tags$th("Dataset"), tags$th("Baseline MSE"), tags$th("Alpha-Beta MSE"),
                tags$th("Delta MSE"), tags$th("Baseline MAE"), tags$th("Alpha-Beta MAE")
              )),
              tags$tbody(
                tags$tr(tags$td("ETTh1"),   tags$td("0.3819"), tags$td("0.3821"),
                  tags$td(span(class="badge-grey",  "+0.05%")), tags$td("0.4031"), tags$td("0.4033")),
                tags$tr(tags$td("ETTh2"),   tags$td("0.3049"), tags$td(strong("0.2881")),
                  tags$td(span(class="badge-green", "-5.5%")),  tags$td("0.3499"), tags$td(strong("0.3318"))),
                tags$tr(tags$td("Weather"), tags$td("0.1943"), tags$td(strong("0.1877")),
                  tags$td(span(class="badge-green", "-3.4%")),  tags$td("0.2055"), tags$td(strong("0.1986")))
              )
            )
          ),

          div(class="result-card",
            h5("1 · Aggregate Metrics (MSE & MAE)"),
            div(class="caption-box",
              "Side-by-side bar charts of MSE and MAE for all three datasets.
               Green annotations = improvement; grey = negligible change."
            ),
            uiOutput("img_compare_metrics")
          ),

          div(class="result-card",
            h5("2 · Learned Alpha & Beta Profiles"),
            div(class="caption-box",
              "Per layer-head alpha (channel branch) and beta (temporal branch) learned values.
               Top row: raw values; bottom row: deviation from 1.
               ETTh2 learns clearly suppressed beta (~0.82-0.90);
               Weather learns amplified alpha (~1.12-1.22)."
            ),
            uiOutput("img_ab_profile")
          ),

          div(class="result-card",
            h5("3 · Step-wise MSE Horizon (96 steps)"),
            div(class="caption-box",
              "Top row: per-step test MSE curves for baseline (blue) vs alpha-beta (red).
               Bottom row: step-wise delta (alpha-beta minus baseline). Negative = improvement.
               The LOESS smoother (black) confirms the trend."
            ),
            uiOutput("img_step_mse")
          ),

          div(class="result-card",
            h5("4 · Per-channel MSE Breakdown"),
            div(class="caption-box",
              "Per-variate MSE bars for baseline (blue) vs alpha-beta (orange).
               ETTh2 and Weather show alpha-beta uniformly lower across nearly all channels."
            ),
            uiOutput("img_channel")
          ),

          div(class="result-card",
            h5("5 · Diagnostic Panels (per dataset)"),
            div(class="caption-box",
              "4-panel diagnostic: (a) step MSE curves, (b) per-channel scatter,
               (c) smoothed forecast trajectory, (d) residual distribution.
               Select a dataset:"
            ),
            selectInput("diag_ds", NULL,
                        choices=c("ETTh1","ETTh2","Weather"),
                        selected="ETTh2", width="200px"),
            uiOutput("img_diag")
          )
        ),

        # ----- Sub-tab: PCA Results -----
        tabPanel("PCA Results",
          br(),
          div(class="result-card",
            h5("PCA Pre-processing Approach"),
            div(class="caption-box",
              "Before feeding data to TimeKAN, the N variates are rotated into their principal
               components. Only the top d components are retained, TimeKAN runs in PC-space,
               and predictions are projected back via the PCA loading matrix."
            )
          ),

          div(class="result-card",
            h5("ETTh1 — PCA Diagnostic"),
            div(class="caption-box",
              "Scree plot, reconstruction error floor, and per-variate MSE after PCA projection.
               For ETTh1 (7 variates, moderate correlation) 5-6 PCs needed to reach 90% variance."
            ),
            uiOutput("img_pca_etth1")
          ),

          div(class="result-card",
            h5("ETTh2 — PCA Diagnostic"),
            div(class="caption-box",
              "Similar profile to ETTh1. 7 variates share moderate but not extreme correlation.
               PCA provides marginal compression at the cost of a reconstruction bottleneck."
            ),
            uiOutput("img_pca_etth2")
          ),

          div(class="result-card",
            h5("Weather — PCA Diagnostic"),
            div(class="caption-box",
              "Weather has 21 variates with varied physical nature (temperature, humidity, wind...).
               Many PCs needed to reach 90% explained variance, making compression ineffective."
            ),
            uiOutput("img_pca_weather")
          ),

          div(class="result-card",
            h5("Overall PCA vs Baseline Comparison"),
            div(class="caption-box",
              "Aggregate MSE/MAE comparison. PCA extension ",
              strong("did not improve"), " results on any dataset—reconstruction error
               from discarded PCs offset any benefit from reduced channel count."
            ),
            uiOutput("img_pca_results")
          ),

          div(class="result-card",
            h5("Why PCA Struggled — Key Limitations"),
            tags$ul(
              tags$li(strong("Low inter-channel correlation:"), " ETT datasets have 7 variates
                      capturing different physical sub-systems. Correlation is moderate, not high."),
              tags$li(strong("Reconstruction bottleneck:"), " discarding even 1-2 PCs introduces
                      a fixed reconstruction MSE floor that swamps small model gains."),
              tags$li(strong("KAN channel branch loses interpretability:"), " the KAN branch learns
                      functions over named channels; in PC-space that meaning is lost."),
              tags$li(strong("Weather diversity:"), " 21 physically heterogeneous sensors have
                      fundamentally different variance profiles — PCA cannot find a low-dim subspace."),
              tags$li(strong("Better fit for wider datasets:"), " e.g. PEMS traffic with 100+
                      co-located, highly-collinear sensors where many PCs are genuinely redundant.")
            )
          )
        )
      )
    )   # end Results tabPanel
  )   # end navbarPage
)   # end fluidPage

# =============================================================================
# SERVER
# =============================================================================
server <- function(input, output, session) {

  AB_DIR  <- normalizePath(file.path("..", "figures", "alpha_beta_evaluation"), mustWork=FALSE)
  PCA_DIR <- normalizePath(file.path("..", "results",  "pcaResults"),           mustWork=FALSE)

  # ── Embed image helper ───────────────────────────────────────────────────
  img_tag <- function(path) {
    if (!file.exists(path))
      return(div(class="caption-box", style="color:#a00;",
                 sprintf("Image not found: %s", basename(path))))
    b64 <- base64enc::dataURI(file=path, mime="image/png")
    tags$img(src=b64, style="max-width:100%; border-radius:4px; margin-top:8px;")
  }

  # =========================================================================
  # TAB 1 – Original TimeKAN
  # =========================================================================
  ts_data <- reactive({ make_ts(input$ts_len, input$seed) })

  levels_data <- reactive({
    x <- ts_data(); k <- input$n_levels; w <- input$window
    lvl <- vector("list", k); lvl[[1]] <- x; cur <- x
    for (i in 2:k) {
      cur <- moving_avg(cur, w)
      cur <- cur[seq(1, length(cur), by=w)]
      lvl[[i]] <- cur
    }
    lvl
  })

  bands_data <- reactive({
    lvl <- levels_data(); k <- length(lvl); n <- length(lvl[[1]])
    bnd <- vector("list", k)
    for (i in 1:(k-1)) {
      up <- upsample_r(lvl[[i+1]], length(lvl[[i]]))
      bnd[[i]] <- lvl[[i]] - upsample_r(up, n)
    }
    bnd[[k]] <- upsample_r(lvl[[k]], n)
    bnd
  })

  output$plt_levels <- renderPlot({
    lvl <- levels_data(); k <- length(lvl)
    cols <- colorRampPalette(c("#1a3058","#4a7fc1","#90bde8","#c5ddf5"))(k)
    par(mar=c(3,3,1.5,1), mgp=c(1.8,.5,0), bg="white")
    plot(NULL, xlim=c(1,input$ts_len), ylim=range(unlist(lvl))*c(1.1,1.1),
         xlab="Time step", ylab="Amplitude",
         main="k smoothed sequences (level 1 = original)")
    for (i in seq_len(k)) {
      xc <- seq(1, input$ts_len, length.out=length(lvl[[i]]))
      lines(xc, lvl[[i]], col=cols[i], lwd=2.4-i*0.25)
    }
    legend("topright", paste0("Level ",1:k," (",sapply(lvl,length)," pts)"),
           col=cols, lwd=2, bty="n", cex=0.82)
    grid(col="grey90", lty=1)
  })

  output$plt_bands <- renderPlot({
    bnd <- bands_data(); k <- length(bnd); n <- input$ts_len
    cols <- colorRampPalette(c("#c0392b","#e67e22","#27ae60","#2980b9","#8e44ad"))(k)
    par(mar=c(3,3,1.5,1), mgp=c(1.8,.5,0), bg="white")
    plot(NULL, xlim=c(1,n), ylim=range(unlist(bnd),na.rm=TRUE)*c(1.2,1.2),
         xlab="Time step", ylab="Amplitude",
         main="Frequency bands: f_i = x_i - FreqUp(x_{i+1})")
    for (i in seq_len(k)) lines(seq_len(n), bnd[[i]], col=cols[i], lwd=1.9)
    abline(h=0, lty=2, col="grey60")
    legend("topright",
           c(paste0("Band ",1:(k-1)," (high-mid freq)"), "Band k (low freq trend)"),
           col=cols, lwd=2, bty="n", cex=0.80)
    grid(col="grey92", lty=1)
  })

  output$plt_spectra <- renderPlot({
    bnd <- bands_data(); k <- length(bnd); n <- input$ts_len
    cols <- colorRampPalette(c("#c0392b","#e67e22","#27ae60","#2980b9","#8e44ad"))(k)
    par(mfrow=c(1,k), mar=c(3,2.5,2,0.5), mgp=c(1.8,.5,0), bg="white")
    for (i in seq_len(k)) {
      sp    <- fft_amp(bnd[[i]])
      freqs <- seq(0, 0.5, length.out=length(sp))
      plot(freqs, sp, type="l", col=cols[i], lwd=2,
           xlab="Norm. freq", ylab="", main=paste0("Band ",i), ylim=c(0,1))
      polygon(c(freqs,rev(freqs)), c(sp,rep(0,length(sp))),
              col=adjustcolor(cols[i],0.20), border=NA)
      grid(col="grey92", lty=1)
    }
  })

  output$plt_mkan <- renderPlot({
    set.seed(input$seed); n <- min(input$ts_len, 48)
    t <- seq(0, 2*pi, length.out=n)
    band  <- 0.7*sin(3*t) + rnorm(n,0,0.08)
    f1    <- 0.5*sin(3*t+0.1) + rnorm(n,0,0.06)
    f2    <- 0.4*cos(3*t-0.2) + rnorm(n,0,0.07)
    f_hat <- f1 + f2
    par(mfrow=c(1,3), mar=c(3,3,2,1), mgp=c(1.8,.5,0), bg="white")
    for (dat in list(
      list(y=f1,    col="#2166AC", lbl="f1 (conv)",   ttl="Temporal branch (DWConv)"),
      list(y=f2,    col="#D6604D", lbl="f2 (KAN)",    ttl="Channel branch (ChebyKAN)"),
      list(y=f_hat, col="#1a7a1a", lbl="f_hat (sum)", ttl="Combined: f_hat = f1 + f2")
    )) {
      plot(band, type="l", col="grey60", lwd=1.5, xlab="Step", ylab="",
           main=dat$ttl)
      lines(dat$y, col=dat$col, lwd=2.2)
      legend("topright", c("Input band", dat$lbl),
             col=c("grey60",dat$col), lwd=2, bty="n", cex=.80)
      grid(col="grey92",lty=1)
    }
  })

  output$plt_mix <- renderPlot({
    x <- ts_data(); n <- length(x); set.seed(input$seed+1)
    recon <- x + rnorm(n, 0, 0.09)
    par(mar=c(3,3,1.5,1), mgp=c(1.8,.5,0), bg="white")
    plot(x, type="l", col="#333333", lwd=2, xlab="Time step", ylab="Amplitude",
         main="Frequency Mixing: reconstructed signal vs original")
    lines(recon, col="#8B0000", lwd=1.8, lty=2)
    legend("topright", c("Original","Reconstructed (after mixing)"),
           col=c("#333333","#8B0000"), lwd=2, lty=c(1,2), bty="n", cex=0.82)
    grid(col="grey92", lty=1)
  })

  # =========================================================================
  # TAB 2 – Alpha-Beta
  # =========================================================================
  ab_signal <- reactive({
    set.seed(input$ab_seed); n <- 64; t <- seq(0,2*pi,length.out=n)
    band <- 0.8*sin(4*t) + 0.3*cos(8*t) + rnorm(n,0,0.07)
    f1   <- 0.55*sin(4*t+0.15) + rnorm(n,0,0.06)
    f2   <- 0.35*cos(4*t-0.1)  + rnorm(n,0,0.07)
    list(band=band, f1=f1, f2=f2, n=n, t=t)
  })

  output$plt_ab_base <- renderPlot({
    d <- ab_signal()
    par(mar=c(3,3,1.8,1), bg="white")
    plot(d$band, type="l", col="grey55", lwd=1.5, xlab="Step", ylab="",
         main="Baseline: f_hat = f1 + f2  (alpha=beta=1, fixed)")
    lines(d$f1,         col="#2166AC", lwd=1.5, lty=2)
    lines(d$f2,         col="#D6604D", lwd=1.5, lty=2)
    lines(d$f1 + d$f2,  col="#1a7a1a", lwd=2.2)
    legend("topright", c("Band","f1","f2","f_hat"),
           col=c("grey55","#2166AC","#D6604D","#1a7a1a"), lwd=2, bty="n", cex=.80)
    grid(col="grey92",lty=1)
  })

  output$plt_ab_weighted <- renderPlot({
    d <- ab_signal(); a <- input$ab_alpha; b <- input$ab_beta
    par(mar=c(3,3,1.8,1), bg="white")
    plot(d$band, type="l", col="grey55", lwd=1.5, xlab="Step", ylab="",
         main=sprintf("Alpha-beta: f_hat = %.2f*f1 + %.2f*f2", a, b))
    lines(a*d$f1,           col="#2166AC", lwd=1.5, lty=2)
    lines(b*d$f2,           col="#D6604D", lwd=1.5, lty=2)
    lines(a*d$f1 + b*d$f2,  col="#8B008B", lwd=2.2)
    legend("topright",
           c("Band", sprintf("alpha*f1 (x%.2f)",a), sprintf("beta*f2 (x%.2f)",b), "f_hat (a-b)"),
           col=c("grey55","#2166AC","#D6604D","#8B008B"), lwd=2, bty="n", cex=.80)
    grid(col="grey92",lty=1)
  })

  output$plt_ab_heatmap <- renderPlot({
    d <- ab_signal(); target <- d$band
    ag <- seq(0.5,1.5,length.out=40); bg <- seq(0.5,1.5,length.out=40)
    mse_grid <- outer(ag, bg, Vectorize(function(a,b)
      mean((a*d$f1 + b*d$f2 - target)^2)))
    base_mse <- mean((d$f1 + d$f2 - target)^2)
    par(mar=c(4,4,2,4), bg="white")
    image(ag, bg, mse_grid, col=hcl.colors(64,"Inferno",rev=TRUE),
          xlab="alpha (temporal branch weight)", ylab="beta (channel branch weight)",
          main="MSE landscape as alpha & beta vary")
    contour(ag, bg, mse_grid, add=TRUE, col="white", lwd=1,
            levels=quantile(mse_grid,0.4))
    contour(ag, bg, mse_grid, add=TRUE, col="cyan", lwd=2, levels=base_mse)
    points(input$ab_alpha, input$ab_beta, pch=8, cex=2.5, col="yellow", lwd=2)
    legend("topright", c("Current alpha,beta","Baseline MSE contour"),
           pch=c(8,NA), lty=c(NA,1), col=c("yellow","cyan"),
           pt.cex=1.8, lwd=2, bty="n", cex=0.80)
  })

  # =========================================================================
  # TAB 3 – PCA
  # =========================================================================
  pca_data <- reactive({
    set.seed(input$pca_seed); n <- 80; p <- input$pca_n_vars
    rho <- input$pca_cor
    Sigma <- matrix(rho, p, p); diag(Sigma) <- 1
    L <- chol(Sigma)
    X <- matrix(rnorm(n*p), n, p) %*% L
    t <- seq(0,2*pi,length.out=n)
    for (j in seq_len(p)) X[,j] <- X[,j] + 0.6*sin(j*t/p*2+j*0.3)
    X
  })

  output$plt_pca_raw <- renderPlot({
    X <- pca_data(); p <- ncol(X)
    cols <- hcl.colors(p,"Set2")
    par(mar=c(3,3,1.5,1), bg="white")
    plot(NULL, xlim=c(1,nrow(X)), ylim=range(X)*1.1, xlab="Time", ylab="",
         main=sprintf("All %d variates (original)", p))
    for (j in seq_len(p)) lines(X[,j], col=cols[j], lwd=1.3)
    grid(col="grey92",lty=1)
  })

  output$plt_pca_proj <- renderPlot({
    X <- pca_data(); d <- min(input$pca_n_comp, ncol(X))
    pc <- prcomp(X, center=TRUE, scale.=FALSE)
    Z  <- pc$x[,1:d,drop=FALSE]
    cols <- hcl.colors(d,"Zissou 1")
    par(mar=c(3,3,1.5,1), bg="white")
    plot(NULL, xlim=c(1,nrow(X)), ylim=range(Z)*1.1, xlab="Time", ylab="",
         main=sprintf("Top %d Principal Components", d))
    for (j in seq_len(d)) lines(Z[,j], col=cols[j], lwd=1.6)
    legend("topright", paste0("PC",1:min(d,5)), col=cols[1:min(d,5)], lwd=2, bty="n", cex=.80)
    grid(col="grey92",lty=1)
  })

  output$plt_pca_scree <- renderPlot({
    X <- pca_data(); p <- ncol(X)
    pc <- prcomp(X, center=TRUE, scale.=FALSE)
    ev <- pc$sdev^2; cum_var <- cumsum(ev)/sum(ev)*100
    d  <- input$pca_n_comp
    par(mar=c(4,4,1.5,1), bg="white")
    plot(1:p, cum_var, type="b", pch=16, col="#2166AC", lwd=2,
         xlab="Number of PCs", ylab="Cumulative variance explained (%)",
         main="Scree Plot", ylim=c(0,100))
    abline(v=d,  lty=2, col="firebrick", lwd=1.8)
    abline(h=90, lty=2, col="grey50")
    text(d+0.3, 15, sprintf("d=%d\n%.0f%%", d, cum_var[d]),
         col="firebrick", cex=0.82, adj=0)
    grid(col="grey92",lty=1)
  })

  output$plt_pca_recon <- renderPlot({
    X <- pca_data(); p <- ncol(X)
    pc <- prcomp(X, center=TRUE, scale.=FALSE)
    mse_v <- sapply(1:p, function(d) {
      Xr <- pc$x[,1:d,drop=FALSE] %*% t(pc$rotation[,1:d,drop=FALSE])
      Xr <- sweep(Xr, 2, pc$center, "+")
      mean((X - Xr)^2)
    })
    d <- input$pca_n_comp
    par(mar=c(4,4,1.5,1), bg="white")
    plot(1:p, mse_v, type="b", pch=16, col="#D6604D", lwd=2,
         xlab="Number of PCs kept", ylab="Reconstruction MSE",
         main="Reconstruction Error Floor")
    points(d, mse_v[d], pch=8, cex=2.5, col="firebrick", lwd=2)
    text(d+0.4, mse_v[d], sprintf("d=%d\nMSE=%.3f",d,mse_v[d]),
         col="firebrick", cex=0.80, adj=0)
    grid(col="grey92",lty=1)
  })

  # =========================================================================
  # TAB 4 – KAN Basis Explorer
  # =========================================================================

  # Build bands from slider settings on this tab
  basis_bands <- reactive({
    n  <- input$basis_ts_len
    k  <- input$basis_levels
    w  <- 2
    x  <- make_ts(n, input$basis_seed)
    lvl <- vector("list", k); lvl[[1]] <- x; cur <- x
    for (i in 2:k) {
      cur <- moving_avg(cur, w)
      cur <- cur[seq(1, length(cur), by=w)]
      lvl[[i]] <- cur
    }
    bnd <- vector("list", k)
    for (i in 1:(k-1)) {
      up <- upsample_r(lvl[[i+1]], length(lvl[[i]]))
      bnd[[i]] <- lvl[[i]] - upsample_r(up, n)
    }
    bnd[[k]] <- upsample_r(lvl[[k]], n)
    bnd
  })

  # Panel A: basis functions
  output$plt_basis_fns <- renderPlot({
    ord  <- input$basis_order
    bas  <- input$basis_type
    u    <- seq(-1, 1, length.out=200)
    fns  <- eval_basis(u, ord, bas)
    n_fn <- length(fns)
    cols <- hcl.colors(n_fn, "Spectral")

    bas_label <- get_basis_label(bas)
    ord_label <- if (bas == "linear") "degree 1 (fixed control: g0=1, g1=u)" else sprintf("order %d (%d functions)", ord, n_fn)

    par(mar=c(4,4,2,1), bg="white")
    yl <- range(unlist(fns))
    plot(NULL, xlim=c(-1,1), ylim=yl*c(1.1,1.1),
         xlab="u = tanh(input)",
         ylab="Basis function value",
         main=sprintf("%s basis — %s", bas_label, ord_label))
    abline(h=0, lty=2, col="grey70"); abline(v=0, lty=2, col="grey70")
    for (i in seq_len(n_fn))
      lines(u, fns[[i]], col=cols[i], lwd=2.2)
    legend("topright",
           paste0("g", 0:(n_fn-1)),
           col=cols, lwd=2, bty="n", cex=0.80, ncol=max(1, ceiling(n_fn/6)))
    grid(col="grey93", lty=1)
  })

  # Panel B: fitting each band
  output$plt_basis_fit_bands <- renderPlot({
    bnd <- basis_bands(); k <- length(bnd)
    ord <- input$basis_order; bas <- input$basis_type
    n   <- input$basis_ts_len

    band_cols <- colorRampPalette(c("#c0392b","#e67e22","#27ae60","#2980b9","#8e44ad"))(k)

    par(mfrow=c(1,k), mar=c(3.5,3,2.5,0.5), mgp=c(1.8,.5,0), bg="white")
    for (i in seq_len(k)) {
      sig <- bnd[[i]]
      fit_res <- basis_fit(sig, ord, bas)
      r2 <- 1 - sum((sig - fit_res$fitted)^2) / sum((sig - mean(sig))^2)
      r2 <- max(0, min(1, r2))

      plot(sig, type="l", col=band_cols[i], lwd=2,
           xlab="Time step", ylab="",
           main=sprintf("Band %d\n(R\u00b2 = %.3f)", i, r2),
           ylim=range(c(sig, fit_res$fitted))*1.1)
      lines(fit_res$fitted, col="#8B0000", lwd=1.8, lty=2)
      if (i == 1)
        legend("topright", c("Band signal","Basis fit"),
               col=c(band_cols[i],"#8B0000"), lwd=2, lty=c(1,2), bty="n", cex=.78)
      grid(col="grey92",lty=1)
    }
  })

  # Panel C: R² vs order for each band
  output$plt_basis_r2 <- renderPlot({
    bnd <- basis_bands(); k <- length(bnd); bas <- input$basis_type
    orders <- 1:8
    band_cols <- colorRampPalette(c("#c0392b","#e67e22","#27ae60","#2980b9","#8e44ad"))(k)

    r2_mat <- matrix(NA, length(orders), k)
    for (oi in seq_along(orders)) {
      for (bi in seq_len(k)) {
        sig <- bnd[[bi]]
        ft  <- basis_fit(sig, orders[oi], bas)
        r2  <- 1 - sum((sig - ft$fitted)^2) / sum((sig - mean(sig))^2)
        r2_mat[oi, bi] <- max(0, min(1, r2))
      }
    }

    par(mar=c(4,4,2,1), bg="white")
    plot(NULL, xlim=c(1,8), ylim=c(0,1),
         xlab="Order / degree", ylab=expression(R^2~"of basis fit"),
         main=sprintf("Fit quality vs order (%s basis)", get_basis_label(bas)))
    for (bi in seq_len(k)) {
      lines(orders, r2_mat[,bi], col=band_cols[bi], lwd=2.2, type="b", pch=16)
    }
    # mark current order
    abline(v=input$basis_order, lty=2, col="grey50")
    legend("bottomright", paste0("Band ",1:k), col=band_cols, lwd=2, pch=16, bty="n", cex=0.82)
    abline(h=c(0.8,0.9), lty=3, col=c("grey70","grey50"))
    text(7.8, 0.81, "0.8", col="grey60", cex=0.75)
    text(7.8, 0.91, "0.9", col="grey50", cex=0.75)
    if (bas == "linear") {
      text(4.5, 0.15, "Note: Linear basis is fixed to degree 1 (g0=1, g1=u) — fit is constant across orders",
           col="#555555", font=3, cex=0.85)
    }
    grid(col="grey93",lty=1)
  })

  # Panel D: coefficient bar chart for Band 1
  output$plt_basis_coeffs <- renderPlot({
    bnd <- basis_bands(); bas <- input$basis_type; ord <- input$basis_order
    sig <- bnd[[1]]
    ft  <- basis_fit(sig, ord, bas)
    cf  <- ft$coeff
    n_c <- length(cf)
    cols_c <- hcl.colors(n_c, "Spectral")

    par(mar=c(4,4.5,2,1), bg="white")
    bp <- barplot(cf, col=cols_c, border=NA,
                  names.arg=paste0("g",0:(n_c-1)),
                  xlab="Basis function",
                  ylab="Learned coefficient",
                  main=sprintf("Coefficient spectrum — Band 1  (%s, %s)",
                               get_basis_label(bas),
                               if (bas == "linear") "degree 1" else sprintf("order %d", ord)))
    abline(h=0, col="grey50", lwd=1.2)
    grid(nx=NA, ny=NULL, col="grey92", lty=1)
  })

  # =========================================================================
  # TAB 5 – Results images
  # =========================================================================
  output$img_compare_metrics <- renderUI({ img_tag(file.path(AB_DIR,"plot_compare_metrics.png")) })
  output$img_ab_profile      <- renderUI({ img_tag(file.path(AB_DIR,"plot_learned_alpha_beta_profile.png")) })
  output$img_step_mse        <- renderUI({ img_tag(file.path(AB_DIR,"plot_step_mse_horizon.png")) })
  output$img_channel         <- renderUI({ img_tag(file.path(AB_DIR,"plot_channel_breakdown.png")) })
  output$img_diag <- renderUI({
    fname <- switch(input$diag_ds,
      ETTh1   = "plot_diagnostics_etth1.png",
      ETTh2   = "plot_diagnostics_etth2.png",
      Weather = "plot_diagnostics_weather.png")
    img_tag(file.path(AB_DIR, fname))
  })

  output$img_pca_etth1   <- renderUI({ img_tag(file.path(PCA_DIR,"PCA_etth1.png"))   })
  output$img_pca_etth2   <- renderUI({ img_tag(file.path(PCA_DIR,"PCA_etth2.png"))   })
  output$img_pca_weather <- renderUI({ img_tag(file.path(PCA_DIR,"PCA_weather.png")) })
  output$img_pca_results <- renderUI({ img_tag(file.path(PCA_DIR,"PCA_results.png")) })
}

shinyApp(ui, server)
