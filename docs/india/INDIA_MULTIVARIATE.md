# Multivariate Indian-data experiments

`run_india_multivariate.R` runs the repository's unchanged `timekan()` model on
multiple measured variables at once, forecasting every retained variable.
It does not rerun the existing ETT suite, ablations, or univariate experiments.
Put this file and `run_india_multivariate.R` in the repository root beside `timekan.R`.

## Experiments

| Source | Selection | History | Horizons |
|---|---|---|---|
| rohanrao/air-quality-data-in-india | Delhi hourly city aggregates, 2018–2019; up to 12 pollutants | 96 hours | 24, 96 hours |
| yug201/daily-climate-time-series-data-delhi-india | Full supplied daily weather table; numeric weather measurements | 96 days | 7, 30 days |

Air fields: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene.
AQI and AQI categories are excluded because AQI is derived from the pollutants.
Climate fields supported in the supplied weather table: tempmax, tempmin, temp,
feelslikemax, feelslikemin, feelslike, humidity, precip, precipprob, precipcover,
windspeed, sealevelpressure. The older date/meantemp format is also supported.
Row indices, categorical descriptions, calendar columns, and supplied engineered
temperature/humidity products are excluded. Constant fields and fields with less
than 50% observed training coverage are dropped, with the reasons saved.

The repository model is **channel independent**: the same learned network processes
each variable separately inside the multivariate batch. This is its ordinary
multivariate input/output setup; these experiments do not add cross-variable
attention or claim that the model learns correlations between pollutants.

## Run

Install the repository's R `torch` and `coro` dependencies first. In Colab use its
existing R CUDA installation. Download the original raw multivariate CSVs; the
earlier files under `data/extensions12` were reduced to one target and cannot be
used for these experiments.

```sh
Rscript run_india_multivariate.R --task self_test

Rscript run_india_multivariate.R --device cuda --batch_size 64 \
  --air_csv /kaggle/input/air-quality-data-in-india/city_hour.csv \
  --climate_csv /kaggle/input/daily-climate-time-series-data-delhi-india/kaggel_weather_2013_to_2024.csv \
  --backup /content/drive/MyDrive/TimeKAN_runs/20261101_091847
```

If the files are not already downloaded, `--download 1` uses Kaggle's public ZIP
endpoint instead. That endpoint may require a manual download if Kaggle changes
its access requirements. Explicit CSV paths are the reliable option.

Use `--task air` or `--task climate` for just one dataset. Use `--device cpu`
without CUDA. Defaults: 10 epochs maximum, early stopping patience 3, seed 0,
Adam learning rate 0.001, three levels, two layers, embedding dimension 16.
The launch above uses batch size 64; script default is 16 for smaller machines.

Completed runs are skipped only when the stored configuration and source hash
match. If changing settings/data, specify a different `--results_dir` to preserve
the previous experiment. Regenerate all images without training or loading torch:

```sh
Rscript run_india_multivariate.R --task plots
```

## Figures

Images are written to `results/india/timekan/` and copied after each run to
`<backup>/results/india/timekan/` when `--backup` is supplied:

- `all_datasets_comparison.png`: standardized MSE and MAE for all completed runs.
- `<run>_comparison.png`: TimeKAN, persistence, and seasonal-naive comparisons.
- `<run>_per_variable.png`: errors for each pollutant/weather variable.
- `<run>_horizon_errors.png`: errors as the forecast lead increases.
- `<run>_training.png`: training/validation losses and selected epoch.
- `<run>_forecasts_1.png`, etc.: observed vs predicted values in original units,
  showing the first eligible test forecast origin, four variables per image.

Forecast examples illustrate one origin; quantitative charts score all eligible
test windows. MSE/MAE charts use standardized units so different physical units
are not pooled. Per-variable original-unit errors are also saved numerically.
CSV tables remain as supporting data; PNGs are the presentation outputs.

## Evaluation and missing data

Chronological splits are 70% training, 10% validation, 20% test. Windows never
place a training target beyond the training boundary or a validation target
beyond the validation boundary. Validation/test history can use earlier rows.
Feature coverage, constant-column decisions, means, standard deviations and
fallback medians are fitted on training observations only.

Timestamps are regularized, identical duplicates merged, and conflicting
duplicates rejected. Missing inputs are carried forward from earlier observed
values; leading gaps use the training median. Long input gaps are also carried
forward, which can reduce forecast quality; per-variable missing fractions are
recorded. Missing **targets are masked from both training loss and evaluation**.
Their filled values are never treated as ground truth. A window with no observed
targets is omitted. Do not interpret these results as forecasts of imputed labels.

All three models use the same test origins and observed-target mask. Persistence
repeats the last input; seasonal-naive repeats the final 24 hours (air) or 7 days
(weather). The best validation checkpoint is selected; test results never select
weights or hyperparameters. One seed is used for the deadline; uncertainty over
multiple independent training seeds has not been estimated.

Each completion RDS saves the resolved settings, source MD5, columns/dropped
columns, split boundaries, missingness, window counts, training curves, metrics,
and original-unit example forecasts. A completion marker is written only after
evaluation and plot generation succeed. Models are saved separately as `.pt`.

## Validation

Local base-R checks cover chronological split boundaries, train-only statistics,
causal missing-value handling, duplicate rejection, multiple-channel preservation,
and all plotting functions. `--task self_test` exercises R torch tensor layout,
masked gradients, unchanged TimeKAN forward/backward, and metric denominators
against a hand-computed persistence error. It runs quickly on CPU inside Colab.


## Repository layout

Checked-in PNGs and newly generated plots are in `figures/india/timekan/` and `figures/india/learned_baselines/`. Numerical tables and lightweight run records are in `results/india/`. Reference sources and licenses are in `third_party/india_baselines/`; the optional Python parity checker is in `tests/india/`. Run the R commands from the repository root. Raw data and model weights are not included.
