# =============================================================================
# TimeKAN Simulation – Shiny App
# Tabs:
#   1. Original TimeKAN       – step-by-step illustration
#   2. Alpha-Beta Extension   – learnable branch weights demo
#   3. PCA Extension          – PCA pre-processing demo
#   4. Change polynomial basis– polynomial basis & degree visualiser
#   5. Results                – experiment result plots with explanations
#
# Theme: clean grey / blue / white | LaTeX math formatting via MathJax & plotmath
# =============================================================================

library(shiny)

# =============================================================================
# HELPERS – pure R
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
  n <- order
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
        g[[k + 2]] <- ((2 * k + 1) * u * g[[k + 1]] - k * g[[k]]) / (k + 1)
      g[1:(n + 1)]
    },
    fourier = {
      g <- list(rep(1, length(u)))
      m <- 1
      while (length(g) < n + 1) {
        g[[length(g) + 1]] <- sin(m * pi * u / 2)
        if (length(g) < n + 1)
          g[[length(g) + 1]] <- cos(m * pi * u / 2)
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
      list(rep(1, length(u)), u)
    }
  )
}

basis_fit <- function(x, order, basis) {
  n  <- length(x)
  u  <- seq(-1, 1, length.out = n)
  B  <- do.call(cbind, eval_basis(u, order, basis))
  cf <- lm.fit(B, x)$coefficients
  fitted <- B %*% cf
  list(fitted = as.numeric(fitted), coeff = cf, u = u, B = B)
}

# =============================================================================
# CSS STYLING
# =============================================================================
app_css <- "
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f0f2f5;
    color: #222;
  }
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
  .well {
    background: #e8edf5;
    border: 1px solid #c8d4e8;
    border-radius: 6px;
  }
  h4 {
    color: #1e3a5f;
    font-weight: 700;
    margin-top: 8px;
    border-bottom: 2px solid #c8d4e8;
    padding-bottom: 4px;
  }
  h5 { color: #2a5298; font-weight: 600; }
  .caption-box {
    background: #fff;
    border-left: 4px solid #4a7fc1;
    padding: 10px 14px;
    border-radius: 4px;
    margin-top: 6px;
    font-size: 0.88em;
    color: #333;
    line-height: 1.6;
  }
  .math-block {
    text-align: center;
    margin: 8px 0;
    padding: 7px 12px;
    background: #f7f9fd;
    border: 1px solid #d4e0f0;
    border-radius: 5px;
    font-size: 1.04em;
    color: #1e3a5f;
    overflow-x: auto;
  }
  .result-card {
    background: #fff;
    border: 1px solid #dde4f0;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 14px;
  }
  .badge-green { background:#2ca02c; color:#fff; padding:2px 8px; border-radius:10px; font-size:.8em; }
  .badge-red   { background:#c0392b; color:#fff; padding:2px 8px; border-radius:10px; font-size:.8em; }
  .badge-grey  { background:#7f8c8d; color:#fff; padding:2px 8px; border-radius:10px; font-size:.8em; }
  hr { border-top: 1px solid #c8d4e8; }
  .tab-content { padding: 18px 12px; }
  label { font-weight: 600; color: #2c4a7c; font-size: 0.88em; }
"

# =============================================================================
# UI
# =============================================================================
ui <- fluidPage(
  tags$head(
    tags$script(type = "text/x-mathjax-config", HTML("
      MathJax.Hub.Config({
        tex2jax: {
          inlineMath: [['$','$'], ['\\\\(','\\\\)']],
          displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
          processEscapes: true
        }
      });
    ")),
    tags$script(type = "text/javascript", src = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"),
    tags$style(HTML(app_css))
  ),

  withMathJax(),

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
          sliderInput("window",   "Downsampling window (w)", min=2, max=4, value=2),
          sliderInput("seed",     "Random seed",          min=1,  max=99,  value=42),
          hr(),
          div(class="caption-box",
            strong("Pipeline Overview:"), br(),
            "1. Level $x^{(1)}$: finest high-frequency signal.", br(),
            "2. Level $x^{(k)}$: coarsest low-frequency trend.", br(),
            "3. CFD isolates band $f^{(i)}$ via frequency upsampling.", br(),
            "4. M-KAN blocks process each band with temporal & channel branches.", br(),
            "5. Frequency mixing restores full multi-scale representation."
          )
        ),
        mainPanel(width = 9,
          h4("Step 1. Hierarchical Sequence Preprocessing"),
          div(class="caption-box",
            p("TimeKAN generates $k$ progressively smoother sequences using average pooling with window $w$ followed by sub-sampling:"),
            div(class="math-block",
              "$$x^{(i)} = \\text{AvgPool}\\big(\\text{Padding}(x^{(i-1)}), \\, w\\big), \\quad i = 2, \\dots, k$$"
            ),
            p("Level 1 ($x^{(1)}$) retains the raw high-frequency observations; each deeper level $x^{(i)}$ compresses temporal resolution by a factor of $w^{i-1}$.")
          ),
          plotOutput("plt_levels", height="240px"),
          br(),

          h4("Step 2. Cascaded Frequency Decomposition (CFD)"),
          div(class="caption-box",
            p("To isolate the unique oscillatory details of each level, the coarser sequence $x^{(i+1)}$ is upsampled to the length of $x^{(i)}$ via zero-padded spectral interpolation:"),
            div(class="math-block",
              "$$\\tilde{x}^{(i)} = \\text{FreqUp}\\big(x^{(i+1)}\\big) = \\text{IFFT}\\Big(\\text{ZeroPad}\\big(\\text{FFT}(x^{(i+1)})\\big)\\Big)$$"
            ),
            p("The frequency band $f^{(i)}$ represents the residual frequency components present in level $i$ but absent in level $i+1$:"),
            div(class="math-block",
              "$$f^{(i)} = x^{(i)} - \\tilde{x}^{(i)}, \\quad i = 1, \\dots, k-1; \\qquad f^{(k)} = x^{(k)}$$"
            )
          ),
          plotOutput("plt_bands", height="240px"),
          br(),

          h4("Step 3. Frequency Spectrum per Band"),
          div(class="caption-box",
            p("The normalized frequency spectra $|\\text{FFT}(f^{(i)})|$ demonstrate progressive band-pass separation: high-frequency bursts are captured in $f^{(1)}$, intermediate dynamics in $f^{(2)}$, and the smooth macro-trend in $f^{(k)}$.")
          ),
          plotOutput("plt_spectra", height="200px"),
          br(),

          h4("Step 4. M-KAN Block (Two Branches)"),
          div(class="caption-box",
            p("Each frequency band $f^{(i)}$ is processed through two parallel specialized branches:"),
            tags$ul(
              tags$li(strong("Temporal Branch (Depthwise Conv1D): "), "models local temporal patterns: ", tags$span(style="color:#2166AC; font-weight:600;", "$f_{1}^{(i)} = \\text{DWConv1D}(f^{(i)})$")),
              tags$li(strong("Channel Branch (ChebyKAN): "), "learns nonlinear cross-variate correlations via Chebyshev polynomials: ", tags$span(style="color:#D6604D; font-weight:600;", "$f_{2}^{(i)} = \\text{ChebyKAN}(f^{(i)})$"))
            ),
            p("Original TimeKAN rigidly sums the two branch representations with equal weights:"),
            div(class="math-block",
              "$$\\hat{f}^{(i)} = f_{1}^{(i)} + f_{2}^{(i)}$$"
            )
          ),
          plotOutput("plt_mkan", height="240px"),
          br(),

          h4("Step 5. Frequency Mixing (Reconstruction)"),
          div(class="caption-box",
            p("The refined representations $\\hat{f}^{(i)}$ are combined hierarchically in reverse order from coarse to fine:"),
            div(class="math-block",
              "$$x^{(i)} = \\text{FreqUp}\\big(x^{(i+1)}\\big) + \\hat{f}^{(i)}, \\quad i = k-1, \\dots, 1$$"
            ),
            p("The final level-1 sequence is projected through a linear forecasting head to generate horizon predictions $\\hat{y} \\in \\mathbb{R}^{H \\times N}$.")
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
          sliderInput("ab_alpha", HTML("&alpha; (temporal branch weight)"), min=0.5, max=1.5, value=1.0, step=0.01),
          sliderInput("ab_beta",  HTML("&beta; (channel branch weight)"),  min=0.5, max=1.5, value=1.0, step=0.01),
          sliderInput("ab_seed",  "Random seed", min=1, max=99, value=7),
          hr(),
          div(class="caption-box",
            strong("Original Baseline Formulation:"),
            div(class="math-block",
              "$$\\hat{f} = 1.0 \\cdot f_1 + 1.0 \\cdot f_2$$"
            ),
            strong("Our Learnable Extension:"),
            div(class="math-block",
              "$$\\hat{f} = \\alpha \\cdot f_1 + \\beta \\cdot f_2$$"
            ),
            p("Both $\\alpha$ and $\\beta$ are scalar parameters learned end-to-end via gradient descent (initialized at $\\alpha = 1.0, \\beta = 1.0$)."),
            p("When temporal trends dominate, $\\alpha > 1$; when cross-channel KAN outputs exhibit high variance, $\\beta < 1$ dampens noisy gradients.")
          )
        ),
        mainPanel(width = 9,
          fluidRow(
            column(6,
              h4(HTML("Baseline: &alpha; = 1.0, &beta; = 1.0 (Fixed)")),
              div(class="caption-box",
                "Direct summation with rigid equal weighting prevents the model from dampening noisy or misaligned branch outputs."
              ),
              plotOutput("plt_ab_base", height="230px")
            ),
            column(6,
              h4(HTML("Learnable: &alpha;&middot;f<sub>1</sub> + &beta;&middot;f<sub>2</sub>")),
              div(class="caption-box",
                "Learnable scalars dynamically rescale each branch. Use the sliders on the left to inspect the modulated output."
              ),
              plotOutput("plt_ab_weighted", height="230px")
            )
          ),
          br(),
          h4(HTML("Error Landscape Across &alpha; and &beta;")),
          div(class="caption-box",
            p("Mean Squared Error over the prediction interval as $\\alpha$ and $\\beta$ vary:"),
            div(class="math-block",
              "$$\\text{MSE}(\\alpha, \\beta) = \\frac{1}{T} \\sum_{t=1}^{T} \\Big( \\big(\\alpha \\cdot f_1(t) + \\beta \\cdot f_2(t)\\big) - y(t) \\Big)^2$$"
            ),
            p("The yellow star marks your current slider coordinates $(\\alpha, \\beta)$. The cyan contour depicts the baseline error level $\\text{MSE}(1.0, 1.0)$—any coordinate interior to this contour achieves superior accuracy.")
          ),
          plotOutput("plt_ab_heatmap", height="320px"),
          br(),
          h4("Theoretical Advantages of Learnable Weights"),
          div(class="result-card",
            tags$ul(
              tags$li(strong("Gradient Variance Suppression: "), "Reduces gradient noise propagation from the KAN branch when inter-channel dependencies are weak."),
              tags$li(strong("Dataset-Specific Specialization: "), "ETTh2 exhibits strong temporal autocorrelation, allowing the model to suppress $\\beta$ ($0.80 \\le \\beta \\le 0.90$) and reduce forecast error by $5.5\\%$."),
              tags$li(strong("Near-Zero Parameter Footprint: "), "Requires only 2 learnable scalars per M-KAN block ($+12$ parameters total across all layers), maintaining inference efficiency."),
              tags$li(strong("Structural Invariance: "), "Preserves the complete architecture and polynomial bases of M-KAN without introducing extra matrix multiplications.")
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
          sliderInput("pca_n_vars", "Number of variates (N)", min=4,  max=15,  value=7),
          sliderInput("pca_n_comp", "Principal components kept (d)", min=1,  max=15,  value=4),
          sliderInput("pca_cor",    HTML("Inter-channel correlation (&rho;)"), min=0.0, max=0.95, value=0.6, step=0.05),
          sliderInput("pca_seed",   "Random seed", min=1, max=99, value=12),
          hr(),
          div(class="caption-box",
            strong("Mathematical Formulation:"), br(),
            "Given multivariate series $X \\in \\mathbb{R}^{T \\times N}$, compute top $d$ eigenvectors $V_d$ of the covariance matrix:",
            div(class="math-block",
              "$$Z = (X - \\mu) V_d, \\quad Z \\in \\mathbb{R}^{T \\times d}$$"
            ),
            "TimeKAN runs on the reduced space $Z$, then reconstructs predictions via:",
            div(class="math-block",
              "$$\\hat{X} = \\hat{Z} V_d^T + \\mu$$"
            ),
            "Discarding $N - d$ components establishes an irreducible reconstruction floor:",
            div(class="math-block",
              "$$\\text{MSE}_{\\text{recon}} = \\frac{1}{TN} \\|X - (Z V_d^T + \\mu)\\|_F^2 = \\sum_{j=d+1}^N \\lambda_j$$"
            )
          )
        ),
        mainPanel(width = 9,
          fluidRow(
            column(6,
              h4("Original Multivariate Series (N Channels)"),
              div(class="caption-box", "Unprocessed time series observations across all $N$ physical channels before PCA projection."),
              plotOutput("plt_pca_raw", height="230px")
            ),
            column(6,
              h4("Projected Series in Principal Subspace (d PCs)"),
              div(class="caption-box", "Orthogonalized principal component trajectories $Z_j(t) = \\sum_{i=1}^N V_{ij} (X_i(t) - \\mu_i)$."),
              plotOutput("plt_pca_proj", height="230px")
            )
          ),
          br(),
          fluidRow(
            column(6,
              h4("Cumulative Variance Explained (Scree Plot)"),
              div(class="caption-box",
                "Ratio of retained variance $\\sum_{j=1}^d \\lambda_j / \\sum_{j=1}^N \\lambda_j$. The red dashed line denotes the selected cutoff $d$."
              ),
              plotOutput("plt_pca_scree", height="250px")
            ),
            column(6,
              h4("Reconstruction Error Floor"),
              div(class="caption-box",
                "Minimum possible reconstruction error $\\sum_{j=d+1}^N \\lambda_j$ incurred prior to any forecasting loss."
              ),
              plotOutput("plt_pca_recon", height="250px")
            )
          ),
          br(),
          h4("Why PCA Struggled on Benchmarks"),
          div(class="result-card",
            tags$table(class="table table-sm table-bordered",
              tags$thead(tags$tr(
                tags$th("Dataset"), tags$th("Channels (N)"), tags$th("Correlation Profile"),
                tags$th("PCs for 90% Variance"), tags$th("Outcome")
              )),
              tags$tbody(
                tags$tr(tags$td("ETTh1"), tags$td("7"), tags$td("Moderate / Heterogeneous"),
                        tags$td("5–6"), tags$td(span(class="badge-red", "Higher Overall MSE"))),
                tags$tr(tags$td("ETTh2"), tags$td("7"), tags$td("Moderate / Non-collinear"),
                        tags$td("5–6"), tags$td(span(class="badge-red", "Higher Overall MSE"))),
                tags$tr(tags$td("Weather"), tags$td("21"), tags$td("Low / Diverse Sensor Types"),
                        tags$td("16+"), tags$td(span(class="badge-red", "Large Reconstruction Loss")))
              )
            ),
            br(),
            tags$p("Because the benchmark datasets possess heterogeneous physical channels with only moderate cross-correlation, PCA cannot compress dimension without discarding critical signal variance. The resulting reconstruction bottleneck outweighs any model capacity savings.")
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
            choices  = c("Chebyshev"        = "chebyshev",
                         "Legendre"         = "legendre",
                         "Fourier"          = "fourier",
                         "RBF (Gaussian)"   = "rbf",
                         "Linear (control)" = "linear"),
            selected = "chebyshev"
          ),
          sliderInput("basis_order", "Order / degree (n)", min=1, max=8, value=3),
          hr(),
          sliderInput("basis_seed",  "Random seed", min=1, max=99, value=5),
          sliderInput("basis_ts_len","Signal length", min=32, max=128, value=64, step=8),
          sliderInput("basis_levels","Frequency levels (k)", min=2, max=5, value=3),
          hr(),
          div(class="caption-box",
            strong("Basis Types Overview:"), br(),
            tags$ul(
              tags$li(strong("Chebyshev: "), "Default in TimeKAN. Orthogonal polynomials on [-1, 1] minimizing approximation error."),
              tags$li(strong("Legendre: "), "Orthogonal polynomial alternative, providing smooth global approximations."),
              tags$li(strong("Fourier: "), "Harmonic sine and cosine basis, ideal for strictly periodic oscillations."),
              tags$li(strong("RBF (Gaussian): "), "Local radial bumps, well suited for localized non-linear transitions."),
              tags$li(strong("Linear (control): "), "Degree 1 baseline, turning KAN into an ordinary linear layer.")
            ),
            p("Higher degrees capture complex rapid variations in high-frequency bands, while lower degrees provide stable representations for smoother trends.")
          )
        ),
        mainPanel(width = 9,
          h4("Panel A: Basis Functions"),
          div(class="caption-box",
            "Individual basis functions evaluated on the normalized domain [-1, 1]. In ChebyKAN, each neural connection acts as an independent learnable curve parameterized by basis weights."
          ),
          plotOutput("plt_basis_fns", height="280px"),
          br(),

          h4("Panel B: Fitting Frequency Bands"),
          div(class="caption-box",
            p("Each decomposed frequency band is fitted with the selected basis functions via ordinary least squares. The coefficient of determination (R²) quantifies how well the basis reconstructs that band's oscillations.")
          ),
          plotOutput("plt_basis_fit_bands", height="300px"),
          br(),

          h4("Panel C: Fit Quality (R²) Across Orders"),
          div(class="caption-box",
            "High-frequency bands demand higher orders (n ≥ 3) to capture rapid fluctuations, while lower-frequency trends achieve near-perfect R² even at order 1."
          ),
          plotOutput("plt_basis_r2", height="260px"),
          br(),

          h4("Panel D: Basis Coefficient Spectrum"),
          div(class="caption-box",
            "Learned regression weights for Band 1 (highest frequency). A rapid decay in coefficients shows that the basis compactly captures the dominant patterns without high-degree noise."
          ),
          plotOutput("plt_basis_coeffs", height="220px")
        )
      )
    ),

    # =========================================================================
    # TAB 5 – Results
    # =========================================================================
    tabPanel("Results",
      h4("Empirical Performance: Alpha-Beta Weighting and PCA Pre-processing"),
      p(style="color:#555;font-size:.9em;",
        "Comprehensive evaluation benchmarks across ETTh1, ETTh2, and Weather datasets under horizon $H=96$."),
      br(),

      tabsetPanel(
        # ----- Sub-tab: Alpha-Beta Results -----
        tabPanel("Alpha-Beta Results",
          br(),
          div(class="result-card",
            h5("Benchmark Summary Table"),
            div(class="caption-box",
              p("Empirical test set metrics comparing the Baseline TimeKAN ($\\hat{f} = f_1 + f_2$) against our Learnable Extension ($\\hat{f} = \\alpha f_1 + \\beta f_2$):"),
              div(class="math-block",
                "$$\\Delta\\text{MSE} = \\frac{\\text{MSE}_{\\alpha\\beta} - \\text{MSE}_{\\text{base}}}{\\text{MSE}_{\\text{base}}} \\times 100\\%$$"
              )
            ),
            br(),
            tags$table(class="table table-sm table-bordered",
              tags$thead(tags$tr(
                tags$th("Dataset"),
                tags$th(HTML("Baseline MSE")),
                tags$th(HTML("&alpha;&ndash;&beta; MSE")),
                tags$th(HTML("&Delta; MSE (%)")),
                tags$th(HTML("Baseline MAE")),
                tags$th(HTML("&alpha;&ndash;&beta; MAE"))
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
              "Comparative test MSE and MAE. Negative percentage deltas denote superior accuracy achieved by the learnable weighting mechanism."
            ),
            uiOutput("img_compare_metrics")
          ),

          div(class="result-card",
            h5("2 · Learned \\(\\alpha\\) and \\(\\beta\\) Weight Profiles"),
            div(class="caption-box",
              "Inspection of converged weight scalars $\\alpha$ (channel KAN branch) and $\\beta$ (temporal conv branch) reveals systematic dampening of $\\beta$ on ETTh2 and amplification of $\\alpha$ on Weather."
            ),
            uiOutput("img_ab_profile")
          ),

          div(class="result-card",
            h5("3 · Step-Wise MSE Horizon (\\(h = 1, \\dots, 96\\))"),
            div(class="caption-box",
              "Error growth as forecast horizon progresses. Lower panel displays $\\Delta\\text{MSE}(h) = \\text{MSE}_{\\alpha\\beta}(h) - \\text{MSE}_{\\text{base}}(h)$ with LOESS trend line."
            ),
            uiOutput("img_step_mse")
          ),

          div(class="result-card",
            h5("4 · Per-Channel MSE Breakdown"),
            div(class="caption-box",
              "Channel-wise MSE verification confirming that performance gains are distributed across all variates rather than driven by a single dominant channel."
            ),
            uiOutput("img_channel")
          ),

          div(class="result-card",
            h5("5 · Diagnostic Residual Panels"),
            div(class="caption-box",
              "Four-panel validation: (a) step MSE curves, (b) channel scatter relative to identity line $y=x$, (c) forecast trajectory, and (d) residual error distribution."
            ),
            selectInput("diag_ds", "Select dataset:",
                        choices=c("ETTh1","ETTh2","Weather"),
                        selected="ETTh2", width="200px"),
            uiOutput("img_diag")
          )
        ),

        # ----- Sub-tab: PCA Results -----
        tabPanel("PCA Results",
          br(),
          div(class="result-card",
            h5("PCA Pre-processing Evaluation"),
            div(class="caption-box",
              "Comprehensive diagnostic figures illustrating variance distribution, projection fidelity, and residual loss under the PCA extension."
            )
          ),

          div(class="result-card",
            h5("ETTh1 — PCA Diagnostic"),
            div(class="caption-box",
              "Scree plot and reconstruction floor on ETTh1 (7 variates). Moderate cross-correlation prevents meaningful dimension reduction without discarding essential dynamics."
            ),
            uiOutput("img_pca_etth1")
          ),

          div(class="result-card",
            h5("ETTh2 — PCA Diagnostic"),
            div(class="caption-box",
              "Scree profile and error curves on ETTh2. Model performance degrades because the reconstruction penalty $\\sum_{j=d+1}^N \\lambda_j$ exceeds the KAN optimization gain."
            ),
            uiOutput("img_pca_etth2")
          ),

          div(class="result-card",
            h5("Weather — PCA Diagnostic"),
            div(class="caption-box",
              "Weather dataset contains 21 heterogeneous climatological variables. Discarding even a small number of components causes severe reconstruction error across localized sensors."
            ),
            uiOutput("img_pca_weather")
          ),

          div(class="result-card",
            h5("Overall PCA Benchmark Comparison"),
            div(class="caption-box",
              "Direct comparison demonstrating that PCA pre-processing underperformed standard TimeKAN on all benchmarks due to the irreducible dimensionality reconstruction floor."
            ),
            uiOutput("img_pca_results")
          )
        )
      )
    )
  )
)

# =============================================================================
# SERVER
# =============================================================================
server <- function(input, output, session) {

  # Directory resolution: works both locally and when bundled/hosted on shinyapps.io / Docker
  find_asset_dir <- function(sub_www, sub_fallback) {
    if (dir.exists(file.path("www", sub_www))) {
      normalizePath(file.path("www", sub_www), mustWork = FALSE)
    } else if (dir.exists(sub_www)) {
      normalizePath(sub_www, mustWork = FALSE)
    } else {
      normalizePath(file.path("..", sub_fallback), mustWork = FALSE)
    }
  }

  AB_DIR  <- find_asset_dir("alpha_beta_evaluation", file.path("figures", "alpha_beta_evaluation"))
  PCA_DIR <- find_asset_dir("pcaResults",           file.path("results", "pcaResults"))

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
    par(mar=c(3.5, 3.5, 2.2, 1), mgp=c(2.0, 0.6, 0), bg="white")
    plot(NULL, xlim=c(1, input$ts_len), ylim=range(unlist(lvl))*c(1.15, 1.15),
         xlab=expression("Time step " * t),
         ylab=expression("Amplitude " * x^{(i)}(t)),
         main=expression(bold("Hierarchical Preprocessing: ") * x^{(i)} == AvgPool(Padding(x^{(i-1)}), w)))
    for (i in seq_len(k)) {
      xc <- seq(1, input$ts_len, length.out=length(lvl[[i]]))
      lines(xc, lvl[[i]], col=cols[i], lwd=2.5 - i * 0.25)
    }
    leg_labels <- parse(text=paste0("'Level ' * ", 1:k, " * ' (' * x^{(", 1:k, ")} * ')'"))
    legend("topright", legend=leg_labels, col=cols, lwd=2, bty="n", cex=0.82)
    grid(col="grey90", lty=1)
  })

  output$plt_bands <- renderPlot({
    bnd <- bands_data(); k <- length(bnd); n <- input$ts_len
    cols <- colorRampPalette(c("#c0392b","#e67e22","#27ae60","#2980b9","#8e44ad"))(k)
    par(mar=c(3.5, 3.5, 2.2, 1), mgp=c(2.0, 0.6, 0), bg="white")
    plot(NULL, xlim=c(1, n), ylim=range(unlist(bnd), na.rm=TRUE)*c(1.2, 1.2),
         xlab=expression("Time step " * t),
         ylab=expression("Band Amplitude " * f^{(i)}(t)),
         main=expression(bold("Cascaded Frequency Decomposition: ") * f^{(i)} == x^{(i)} - FreqUp(x^{(i+1)})))
    for (i in seq_len(k)) lines(seq_len(n), bnd[[i]], col=cols[i], lwd=1.9)
    abline(h=0, lty=2, col="grey60")
    band_leg <- c(parse(text=paste0("'Band ' * ", 1:(k-1), " * ' (' * f^{(", 1:(k-1), ")} * ')'")),
                  parse(text=paste0("'Trend (' * f^{(", k, ")} * ')'")))
    legend("topright", legend=band_leg, col=cols, lwd=2, bty="n", cex=0.80)
    grid(col="grey92", lty=1)
  })

  output$plt_spectra <- renderPlot({
    bnd <- bands_data(); k <- length(bnd); n <- input$ts_len
    cols <- colorRampPalette(c("#c0392b","#e67e22","#27ae60","#2980b9","#8e44ad"))(k)
    par(mfrow=c(1, k), mar=c(3.5, 3.0, 2.5, 0.8), mgp=c(1.9, 0.6, 0), bg="white")
    for (i in seq_len(k)) {
      sp    <- fft_amp(bnd[[i]])
      freqs <- seq(0, 0.5, length.out=length(sp))
      main_expr <- parse(text=paste0("bold('Band ' * ", i, " * ': ') * '|FFT(' * f^{(", i, ")} * ')|'"))
      plot(freqs, sp, type="l", col=cols[i], lwd=2,
           xlab=expression("Norm. Frequency " * omega / pi),
           ylab=expression("|FFT(" * f^{(i)} * ")|"),
           main=main_expr, ylim=c(0, 1))
      polygon(c(freqs, rev(freqs)), c(sp, rep(0, length(sp))),
              col=adjustcolor(cols[i], 0.20), border=NA)
      grid(col="grey92", lty=1)
    }
  })

  output$plt_mkan <- renderPlot({
    set.seed(input$seed); n <- min(input$ts_len, 48)
    t <- seq(0, 2 * pi, length.out=n)
    band  <- 0.7 * sin(3 * t) + rnorm(n, 0, 0.08)
    f1    <- 0.5 * sin(3 * t + 0.1) + rnorm(n, 0, 0.06)
    f2    <- 0.4 * cos(3 * t - 0.2) + rnorm(n, 0, 0.07)
    f_hat <- f1 + f2
    par(mfrow=c(1, 3), mar=c(3.5, 3.5, 2.5, 1), mgp=c(2.0, 0.6, 0), bg="white")

    plot(band, type="l", col="grey60", lwd=1.5, xlab=expression("Step " * t), ylab="Amplitude",
         main=expression(bold("Temporal Branch: ") * f[1] == DWConv(f)))
    lines(f1, col="#2166AC", lwd=2.2)
    legend("topright", c("Band f", expression(f[1] * " (conv)")),
           col=c("grey60", "#2166AC"), lwd=2, bty="n", cex=0.82)
    grid(col="grey92", lty=1)

    plot(band, type="l", col="grey60", lwd=1.5, xlab=expression("Step " * t), ylab="Amplitude",
         main=expression(bold("Channel Branch: ") * f[2] == ChebyKAN(f)))
    lines(f2, col="#D6604D", lwd=2.2)
    legend("topright", c("Band f", expression(f[2] * " (KAN)")),
           col=c("grey60", "#D6604D"), lwd=2, bty="n", cex=0.82)
    grid(col="grey92", lty=1)

    plot(band, type="l", col="grey60", lwd=1.5, xlab=expression("Step " * t), ylab="Amplitude",
         main=expression(bold("Baseline Sum: ") * hat(f) == f[1] + f[2]))
    lines(f_hat, col="#1a7a1a", lwd=2.2)
    legend("topright", c("Band f", expression(hat(f) == f[1] + f[2])),
           col=c("grey60", "#1a7a1a"), lwd=2, bty="n", cex=0.82)
    grid(col="grey92", lty=1)
  })

  output$plt_mix <- renderPlot({
    x <- ts_data(); n <- length(x); set.seed(input$seed + 1)
    recon <- x + rnorm(n, 0, 0.09)
    par(mar=c(3.5, 3.5, 2.2, 1), mgp=c(2.0, 0.6, 0), bg="white")
    plot(x, type="l", col="#333333", lwd=2, xlab=expression("Time step " * t), ylab="Amplitude",
         main=expression(bold("Frequency Mixing: ") * x^{(i)} == FreqUp(x^{(i+1)}) + hat(f)^{(i)}))
    lines(recon, col="#8B0000", lwd=1.8, lty=2)
    legend("topright", c(expression("Original " * x), expression("Reconstructed " * hat(x))),
           col=c("#333333", "#8B0000"), lwd=2, lty=c(1, 2), bty="n", cex=0.82)
    grid(col="grey92", lty=1)
  })

  # =========================================================================
  # TAB 2 – Alpha-Beta
  # =========================================================================
  ab_signal <- reactive({
    set.seed(input$ab_seed); n <- 64; t <- seq(0, 2 * pi, length.out=n)
    band <- 0.8 * sin(4 * t) + 0.3 * cos(8 * t) + rnorm(n, 0, 0.07)
    f1   <- 0.55 * sin(4 * t + 0.15) + rnorm(n, 0, 0.06)
    f2   <- 0.35 * cos(4 * t - 0.1)  + rnorm(n, 0, 0.07)
    list(band=band, f1=f1, f2=f2, n=n, t=t)
  })

  output$plt_ab_base <- renderPlot({
    d <- ab_signal()
    par(mar=c(3.5, 3.5, 2.2, 1), mgp=c(2.0, 0.6, 0), bg="white")
    plot(d$band, type="l", col="grey55", lwd=1.5, xlab=expression("Step " * t), ylab="Amplitude",
         main=expression(bold("Baseline: ") * hat(f) == 1.00 %*% f[1] + 1.00 %*% f[2]))
    lines(d$f1,         col="#2166AC", lwd=1.5, lty=2)
    lines(d$f2,         col="#D6604D", lwd=1.5, lty=2)
    lines(d$f1 + d$f2,  col="#1a7a1a", lwd=2.2)
    legend("topright", c("Target band", expression(f[1] * " (conv)"), expression(f[2] * " (KAN)"), expression(hat(f) * " (sum)")),
           col=c("grey55", "#2166AC", "#D6604D", "#1a7a1a"), lwd=2, bty="n", cex=0.80)
    grid(col="grey92", lty=1)
  })

  output$plt_ab_weighted <- renderPlot({
    d <- ab_signal(); a <- input$ab_alpha; b <- input$ab_beta
    par(mar=c(3.5, 3.5, 2.2, 1), mgp=c(2.0, 0.6, 0), bg="white")
    main_expr <- bquote(bold("Alpha-Beta: ") * hat(f) == .(sprintf("%.2f", a)) %*% f[1] + .(sprintf("%.2f", b)) %*% f[2])
    plot(d$band, type="l", col="grey55", lwd=1.5, xlab=expression("Step " * t), ylab="Amplitude",
         main=main_expr)
    lines(a * d$f1,           col="#2166AC", lwd=1.5, lty=2)
    lines(b * d$f2,           col="#D6604D", lwd=1.5, lty=2)
    lines(a * d$f1 + b * d$f2, col="#8B008B", lwd=2.2)
    leg_weighted <- c("Target band",
                      as.expression(bquote(alpha %*% f[1] ~ .(sprintf("(x%.2f)", a)))),
                      as.expression(bquote(beta %*% f[2] ~ .(sprintf("(x%.2f)", b)))),
                      expression(hat(f) == alpha %*% f[1] + beta %*% f[2]))
    legend("topright", leg_weighted,
           col=c("grey55", "#2166AC", "#D6604D", "#8B008B"), lwd=2, bty="n", cex=0.80)
    grid(col="grey92", lty=1)
  })

  output$plt_ab_heatmap <- renderPlot({
    d <- ab_signal(); target <- d$band
    ag <- seq(0.5, 1.5, length.out=40); bg <- seq(0.5, 1.5, length.out=40)
    mse_grid <- outer(ag, bg, Vectorize(function(a, b)
      mean((a * d$f1 + b * d$f2 - target)^2)))
    base_mse <- mean((d$f1 + d$f2 - target)^2)
    par(mar=c(4.2, 4.2, 2.5, 4.0), mgp=c(2.4, 0.7, 0), bg="white")
    image(ag, bg, mse_grid, col=hcl.colors(64, "Inferno", rev=TRUE),
          xlab=expression(alpha ~ "(temporal branch weight)"),
          ylab=expression(beta ~ "(channel branch weight)"),
          main=expression(bold("MSE Error Landscape: ") * MSE(alpha, beta) == frac(1, T) * sum((alpha*f[1] + beta*f[2] - y)^2)))
    contour(ag, bg, mse_grid, add=TRUE, col="white", lwd=1,
            levels=quantile(mse_grid, 0.4))
    contour(ag, bg, mse_grid, add=TRUE, col="cyan", lwd=2, levels=base_mse)
    points(input$ab_alpha, input$ab_beta, pch=8, cex=2.5, col="yellow", lwd=2)
    leg_heatmap <- c(as.expression(bquote("Current (" * alpha == .(sprintf("%.2f", input$ab_alpha)) ~ "," ~ beta == .(sprintf("%.2f", input$ab_beta)) * ")")),
                     expression("Baseline contour: " * MSE(1.0, 1.0)))
    legend("topright", leg_heatmap,
           pch=c(8, NA), lty=c(NA, 1), col=c("yellow", "cyan"),
           pt.cex=1.8, lwd=2, bty="n", cex=0.82)
  })

  # =========================================================================
  # TAB 3 – PCA
  # =========================================================================
  pca_data <- reactive({
    set.seed(input$pca_seed); n <- 80; p <- input$pca_n_vars
    rho <- input$pca_cor
    Sigma <- matrix(rho, p, p); diag(Sigma) <- 1
    L <- chol(Sigma)
    X <- matrix(rnorm(n * p), n, p) %*% L
    t <- seq(0, 2 * pi, length.out=n)
    for (j in seq_len(p)) X[, j] <- X[, j] + 0.6 * sin(j * t / p * 2 + j * 0.3)
    X
  })

  output$plt_pca_raw <- renderPlot({
    X <- pca_data(); p <- ncol(X)
    cols <- hcl.colors(p, "Set2")
    par(mar=c(3.5, 3.5, 2.2, 1), mgp=c(2.0, 0.6, 0), bg="white")
    main_expr <- bquote(bold("Raw Observations: ") * X %in% R^{T %*% .(p)})
    plot(NULL, xlim=c(1, nrow(X)), ylim=range(X) * 1.1,
         xlab=expression("Time " * t), ylab=expression("Channel " * X[i](t)),
         main=main_expr)
    for (j in seq_len(p)) lines(X[, j], col=cols[j], lwd=1.3)
    grid(col="grey92", lty=1)
  })

  output$plt_pca_proj <- renderPlot({
    X <- pca_data(); d <- min(input$pca_n_comp, ncol(X))
    pc <- prcomp(X, center=TRUE, scale.=FALSE)
    Z  <- pc$x[, 1:d, drop=FALSE]
    cols <- hcl.colors(d, "Zissou 1")
    par(mar=c(3.5, 3.5, 2.2, 1), mgp=c(2.0, 0.6, 0), bg="white")
    main_expr <- bquote(bold("Principal Subspace: ") * Z == (X - mu) %*% V[d] ~ (d == .(d)))
    plot(NULL, xlim=c(1, nrow(X)), ylim=range(Z) * 1.1,
         xlab=expression("Time " * t), ylab=expression("Score " * Z[j](t)),
         main=main_expr)
    for (j in seq_len(d)) lines(Z[, j], col=cols[j], lwd=1.6)
    legend("topright", parse(text=paste0("PC[", 1:min(d, 5), "]")),
           col=cols[1:min(d, 5)], lwd=2, bty="n", cex=0.82)
    grid(col="grey92", lty=1)
  })

  output$plt_pca_scree <- renderPlot({
    X <- pca_data(); p <- ncol(X)
    pc <- prcomp(X, center=TRUE, scale.=FALSE)
    ev <- pc$sdev^2; cum_var <- cumsum(ev) / sum(ev) * 100
    d  <- input$pca_n_comp
    par(mar=c(4.0, 4.0, 2.2, 1), mgp=c(2.2, 0.6, 0), bg="white")
    plot(1:p, cum_var, type="b", pch=16, col="#2166AC", lwd=2,
         xlab=expression("Number of Principal Components " * d),
         ylab=expression("Cumulative Variance Explained (%)"),
         main=expression(bold("Scree Variance: ") * frac(sum(lambda[j], j=1, d), sum(lambda[j], j=1, N)) %*% 100),
         ylim=c(0, 100))
    abline(v=d,  lty=2, col="firebrick", lwd=1.8)
    abline(h=90, lty=2, col="grey50")
    text(d + 0.3, 15, sprintf("d = %d\n%.1f%%", d, cum_var[d]),
         col="firebrick", cex=0.82, adj=0)
    grid(col="grey92", lty=1)
  })

  output$plt_pca_recon <- renderPlot({
    X <- pca_data(); p <- ncol(X)
    pc <- prcomp(X, center=TRUE, scale.=FALSE)
    mse_v <- sapply(1:p, function(d) {
      Xr <- pc$x[, 1:d, drop=FALSE] %*% t(pc$rotation[, 1:d, drop=FALSE])
      Xr <- sweep(Xr, 2, pc$center, "+")
      mean((X - Xr)^2)
    })
    d <- input$pca_n_comp
    par(mar=c(4.0, 4.0, 2.2, 1), mgp=c(2.2, 0.6, 0), bg="white")
    plot(1:p, mse_v, type="b", pch=16, col="#D6604D", lwd=2,
         xlab=expression("Principal Components Kept " * d),
         ylab=expression("Reconstruction Error " * MSE[recon]),
         main=expression(bold("Irreducible Floor: ") * MSE[recon] == frac(1, T*N) * "||" * X - hat(X) * "||"[F]^2))
    points(d, mse_v[d], pch=8, cex=2.5, col="firebrick", lwd=2)
    text(d + 0.4, mse_v[d], sprintf("d = %d\nMSE = %.3f", d, mse_v[d]),
         col="firebrick", cex=0.80, adj=0)
    grid(col="grey92", lty=1)
  })

  # =========================================================================
  # TAB 4 – Change polynomial basis
  # =========================================================================
  basis_bands <- reactive({
    n   <- input$basis_ts_len
    k   <- input$basis_levels
    w   <- 2
    x   <- make_ts(n, input$basis_seed)
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

  output$plt_basis_fns <- renderPlot({
    ord  <- input$basis_order
    bas  <- input$basis_type
    u    <- seq(-1, 1, length.out=200)
    fns  <- eval_basis(u, ord, bas)
    n_fn <- length(fns)
    cols <- hcl.colors(n_fn, "Spectral")
    bas_label <- get_basis_label(bas)

    par(mar=c(4.0, 4.0, 2.2, 1), mgp=c(2.2, 0.6, 0), bg="white")
    yl <- range(unlist(fns))
    main_expr <- bquote(bold(.(bas_label) ~ "Basis Functions: ") * phi(u) == sum(c[j] * g[j](u), j=0, .(n_fn - 1)))
    plot(NULL, xlim=c(-1, 1), ylim=yl * c(1.15, 1.15),
         xlab=expression("Normalized Coordinate " * u == tanh(x) %in% "[-1, 1]"),
         ylab=expression("Function Value " * g[j](u)),
         main=main_expr)
    abline(h=0, lty=2, col="grey70"); abline(v=0, lty=2, col="grey70")
    for (i in seq_len(n_fn))
      lines(u, fns[[i]], col=cols[i], lwd=2.2)
    legend("topright", parse(text=paste0("g[", 0:(n_fn - 1), "]")),
           col=cols, lwd=2, bty="n", cex=0.82, ncol=max(1, ceiling(n_fn / 6)))
    grid(col="grey93", lty=1)
  })

  output$plt_basis_fit_bands <- renderPlot({
    bnd <- basis_bands(); k <- length(bnd)
    ord <- input$basis_order; bas <- input$basis_type
    n   <- input$basis_ts_len
    band_cols <- colorRampPalette(c("#c0392b","#e67e22","#27ae60","#2980b9","#8e44ad"))(k)

    par(mfrow=c(1, k), mar=c(3.5, 3.2, 2.5, 0.8), mgp=c(1.9, 0.6, 0), bg="white")
    for (i in seq_len(k)) {
      sig <- bnd[[i]]
      fit_res <- basis_fit(sig, ord, bas)
      r2 <- 1 - sum((sig - fit_res$fitted)^2) / sum((sig - mean(sig))^2)
      r2 <- max(0, min(1, r2))

      main_expr <- parse(text=paste0("bold('Band ' * ", i, " * ': ') * R^2 == ", sprintf('%.3f', r2)))
      plot(sig, type="l", col=band_cols[i], lwd=2,
           xlab=expression("Step " * t), ylab=expression("Band " * f^{(i)}(t)),
           main=main_expr,
           ylim=range(c(sig, fit_res$fitted)) * 1.1)
      lines(fit_res$fitted, col="#8B0000", lwd=1.8, lty=2)
      if (i == 1)
        legend("topright", c(expression("Actual " * f^{(i)}), expression("Fitted " * hat(f)^{(i)})),
               col=c(band_cols[i], "#8B0000"), lwd=2, lty=c(1, 2), bty="n", cex=0.80)
      grid(col="grey92", lty=1)
    }
  })

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

    par(mar=c(4.0, 4.0, 2.2, 1), mgp=c(2.2, 0.6, 0), bg="white")
    main_expr <- bquote(bold("Fit Quality vs Order: ") * .(get_basis_label(bas)) ~ "Basis")
    plot(NULL, xlim=c(1, 8), ylim=c(0, 1),
         xlab=expression("Polynomial Order / Degree " * n),
         ylab=expression("Coefficient of Determination " * R^2),
         main=main_expr)
    for (bi in seq_len(k)) {
      lines(orders, r2_mat[, bi], col=band_cols[bi], lwd=2.2, type="b", pch=16)
    }
    abline(v=input$basis_order, lty=2, col="grey50")
    band_leg <- parse(text=paste0("'Band ' * ", 1:k, " * ' (' * f^{(", 1:k, ")} * ')'"))
    legend("bottomright", band_leg, col=band_cols, lwd=2, pch=16, bty="n", cex=0.82)
    abline(h=c(0.8, 0.9), lty=3, col=c("grey70", "grey50"))
    text(7.8, 0.81, expression(R^2 == 0.80), col="grey60", cex=0.75)
    text(7.8, 0.91, expression(R^2 == 0.90), col="grey50", cex=0.75)
    if (bas == "linear") {
      text(4.5, 0.15, expression("Note: Linear control is fixed to degree 1 (" * g[0] == 1 ~ "," ~ g[1] == u * ")"),
           col="#555555", font=3, cex=0.85)
    }
    grid(col="grey93", lty=1)
  })

  output$plt_basis_coeffs <- renderPlot({
    bnd <- basis_bands(); bas <- input$basis_type; ord <- input$basis_order
    sig <- bnd[[1]]
    ft  <- basis_fit(sig, ord, bas)
    cf  <- ft$coeff
    n_c <- length(cf)
    cols_c <- hcl.colors(n_c, "Spectral")

    par(mar=c(4.0, 4.2, 2.2, 1), mgp=c(2.2, 0.6, 0), bg="white")
    main_expr <- bquote(bold("Coefficient Spectrum (Band 1): ") * c[j] ~ "for " * .(get_basis_label(bas)) ~ (n == .(if (bas == "linear") 1 else ord)))
    bp <- barplot(cf, col=cols_c, border=NA,
                  names.arg=parse(text=paste0("g[", 0:(n_c - 1), "]")),
                  xlab=expression("Basis Function " * g[j]),
                  ylab=expression("Regression Coefficient " * c[j]),
                  main=main_expr)
    abline(h=0, col="grey50", lwd=1.2)
    grid(nx=NA, ny=NULL, col="grey92", lty=1)
  })

  # =========================================================================
  # TAB 5 – Results images
  # =========================================================================
  output$img_compare_metrics <- renderUI({ img_tag(file.path(AB_DIR, "plot_compare_metrics.png")) })
  output$img_ab_profile      <- renderUI({ img_tag(file.path(AB_DIR, "plot_learned_alpha_beta_profile.png")) })
  output$img_step_mse        <- renderUI({ img_tag(file.path(AB_DIR, "plot_step_mse_horizon.png")) })
  output$img_channel         <- renderUI({ img_tag(file.path(AB_DIR, "plot_channel_breakdown.png")) })
  output$img_diag <- renderUI({
    fname <- switch(input$diag_ds,
      ETTh1   = "plot_diagnostics_etth1.png",
      ETTh2   = "plot_diagnostics_etth2.png",
      Weather = "plot_diagnostics_weather.png")
    img_tag(file.path(AB_DIR, fname))
  })

  output$img_pca_etth1   <- renderUI({ img_tag(file.path(PCA_DIR, "PCA_etth1.png"))   })
  output$img_pca_etth2   <- renderUI({ img_tag(file.path(PCA_DIR, "PCA_etth2.png"))   })
  output$img_pca_weather <- renderUI({ img_tag(file.path(PCA_DIR, "PCA_weather.png")) })
  output$img_pca_results <- renderUI({ img_tag(file.path(PCA_DIR, "PCA_results.png")) })
}

shinyApp(ui, server)
