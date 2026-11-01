# Extra experiments: R implementation

All experiment logic, training, preprocessing, baseline fitting, scoring and plots
are implemented in R. Python is not required for these experiments. In Colab,
the Python notebook can be used only as a launcher for Rscript commands.

Place these files alongside the existing timekan.R in the repository root:

| File | Purpose |
|---|---|
| extensions12.R | OT-only ETT tests and the two Indian dataset experiments |
| ablations.R | Missing paper component variants and look-back studies |
| classical_baselines.R | AR, MA, ARMA and ARIMA comparisons |
| run_extra.R | R-only experiment orchestration, data preparation, backups and plots |
| EXPERIMENTS.md | This guide |

The original timekan.R and run_experiments.sh are not modified. The new runner
never invokes the existing 26-run baseline suite.

## Installation and data

Install R torch as described in the main README. A CPU installation is sufficient
for the classical comparisons; TimeKAN training can use CPU or an available CUDA
GPU. The existing four ETT CSVs belong in data/ as before.

Indian data sources:

- https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
- https://www.kaggle.com/datasets/yug201/daily-climate-time-series-data-delhi-india

For a new checkout, download city_hour.csv from the first dataset and the weather
CSV from the second. The second collection contains DATE/temp data spanning
2013-2024, rather than the older Delhi dataset with date/meantemp fields.

Run with local source paths:

```bash
Rscript run_extra.R --task extensions --device cuda --air_csv /path/to/city_hour.csv --climate_csv /path/to/kaggel_weather_2013_to_2024.csv
```

Or try the optional public Kaggle ZIP downloader:

```bash
Rscript run_extra.R --task extensions --device cuda --download 1
```

The downloader uses R download.file/unzip, without API tokens. It gives a clear
error if Kaggle requires sign-in/consent; in that case, download the CSVs in your
browser and supply their paths. The public R downloader is optional and has not
been verified end-to-end. Preparing the supplied CSVs and the DATE/temp mapping
have been tested. Already-prepared data/extensions12/ files are reused.

Use --device cpu on a laptop without CUDA. Use --backup /path/to/folder to copy
new results, selected Indian input files and R source after each run. In Colab,
this can point to an already mounted Google Drive folder.

## Run only the work you need

```bash
# Deterministic batching, gradient, baseline-parity and classical forecast checks.
Rscript run_extra.R --task self_test

# Extension 1 (OT-only ETT) and extension 2 (two Indian datasets).
Rscript run_extra.R --task extensions --device cuda --download 1

# Only the missing paper ablations; no original baseline retraining.
Rscript run_extra.R --task ablations --device cuda

# Compare completed univariate results with AR/MA/ARMA/ARIMA on CPU.
Rscript run_extra.R --task compare

# All NEW work in sequence, excluding the old baseline suite.
Rscript run_extra.R --task all --device cuda --download 1
```

Each task skips its completed per-run CSVs. A failed or interrupted run restarts
from the beginning; there is no mid-epoch recovery. If changing hyperparameters,
use a new seed/name/output directory rather than reusing completed results.

Defaults are D=16, two blocks, three bands, base order 1, batch 128, learning
rate .01, seed 0, up to ten epochs and patience three. --epochs and --seed can
be overridden. These match the existing R experiment configuration, not every
dataset-specific setting in the authors' release.

## What the extensions test

| Series | Target | Look-back | Forecast horizons |
|---|---|---|---|
| ETTh1 and ETTh2 | OT only | 96 hours | 96 and 192 hours |
| Delhi city-hour air quality, 2018-2019 | PM2.5 | 96 hours | 24 and 96 hours |
| Delhi daily weather | temp | 96 days | 7 and 30 days |

ETT uses its standard 12/4/4 blocks of 30-day months. Indian data uses a fresh
chronological 70/10/20 split. Scaling uses observed training values only.
Timestamps are sorted and put on a regular grid. Input gaps may be filled forward
from past values for at most three hours (air) or one day (weather). Longer input
gaps and any missing forecast targets exclude that window. No model is trained
or evaluated against fabricated target values. Retained counts are reported.

Every completed extension compares TimeKAN, persistence and seasonal naive on
the same eligible windows. Seasonal naive repeats the latest 24 hours (ETT/air)
or seven days (daily weather). Outputs include standardized MSE/MAE and errors
in original units. An OT-only score is not directly comparable with the paper's
average across seven variables.

## Missing paper ablations

The default reduced-scope plan has twelve runs:

- ETTh1, 96 to 96: linear mapping, linear interpolation and transposed-convolution
  alternatives to FFT upsampling; MLP instead of KAN; no convolution, standard
  convolution and temporal attention instead of depthwise convolution.
- ETTm2, forecast 96: look-backs 48, 192, 336, 512 and 720.

The 96-step baseline is reused from results/main.csv when available. No new
baseline training is scheduled. --component_dataset and --lookback_dataset select
other downloaded ETT files. This does not reproduce Weather, all dataset/table
cells, or the competing models in the paper's look-back figure.

The paper does not fully specify the replacement architectures. Here linear
mapping acts along time; interpolation uses align_corners=FALSE; transposed
convolution is D-to-D, groups=1, kernel=stride=the scale ratio. Learned upsamplers
are separate by scale, layer and decomposition/mixing direction. Both CFD and
mixing use the selected method. The MLP is D -> 32 -> D with GELU. Attention has
four heads and learned Q/K/V/output projections, with no dropout or causal mask;
it attends along the past time axis within each original variable. Parameter
counts are reported; replacement models are not all parameter-matched.

The default ablation layer is tested against the original R layer under identical
seeds. The baseline's existing embedding, normalization and FFT conventions are
retained, including the reviewed old-Nyquist interpolation edge case. This is
an ablation of the existing R implementation, not a corrected/re-tuned baseline.

## Classical baseline comparison

The comparison reads each completed experiment's saved options and reconstructs
the same split, scaling, test origins, history length and horizon. Parameters are
trained only on the training period. Orders are selected on at most 64 evenly
spaced valid validation origins, then all eligible test origins are scored.

| Family | Candidates |
|---|---|
| AR, least squares | p = 4, 12, 24 |
| MA, maximum likelihood | q = 1, 2, 4 |
| ARMA, maximum likelihood | (p,q) = (1,1), (2,1), (2,2) |
| First-difference ARIMA | (p,d,q) = (1,1,0), (0,1,1), (1,1,1) |

At each origin, ARMA-family parameters stay fixed while the latest 96 input
observations are filtered to initialize its state/innovations. No intermediate
future observations are supplied. AR forecasts recurse from fixed lag weights.
ARIMA includes no drift; seasonal ARIMA is outside this initial small search.
MA returns to its fitted mean beyond q steps, which is expected behavior.

The output is a seven-model comparison for each task: TimeKAN, persistence,
seasonal naive, AR, MA, ARMA and ARIMA. A positive TimeKAN MSE reduction indicates
lower error than the best tested alternative; a negative value indicates that
a baseline is better. These are exploratory single-seed results, not claims of
statistically significant or universal superiority.

## Outputs and validation

- results/extensions12/: per-run logs, options, data provenance/counts, checkpoints,
  original-unit/standardized metrics, extension_summary.csv and an R comparison plot.
- results/missing_ablations/: commands/options, logs, checkpoints, individual CSVs,
  missing_ablation_summary_R.csv and an R comparison plot.
- results/univariate_baselines/: selected orders, classical scores, all_models.csv,
  comparison_overview.csv, cached fits and an R comparison plot.

Native R torch batching/missing-target tests, variant shape/gradient checks,
default-layer parity, and fixed-parameter classical forecast checks passed in
Colab. Local R tests also passed for train-only scaling, causal filling, window
selection, the climate schema mapping, aggregation, and base R plotting.
Full missing-ablation training is separate from these deterministic checks;
consult executed logs for which experiments have finished.
