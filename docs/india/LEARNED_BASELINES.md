# Learned baselines on the Indian datasets

Two methods from different families in the TimeKAN paper's comparison list:

- **DLinear (linear/MLP family):** a 25-step endpoint-padded moving average splits
  input into seasonal/residual and trend components. Separate shared linear maps
  forecast both components, then their predictions are added. This run uses the
  official `individual=False` option; there is no added RevIN.
- **iTransformer (Transformer family):** each variable's complete history becomes
  a token. Full multi-head self-attention operates across variables, followed by
  feed-forward layers and a forecast projection. Window normalization and its
  inverse follow the original implementation. Unlike channel-independent TimeKAN
  and shared DLinear, it can use relationships between variables.

Architecture sources:

- https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
- https://github.com/thuml/iTransformer/blob/main/model/iTransformer.py
- https://github.com/thuml/iTransformer/blob/main/layers/Transformer_EncDec.py
- https://github.com/thuml/iTransformer/blob/main/layers/SelfAttention_Family.py
- https://github.com/thuml/iTransformer/blob/main/layers/Embed.py

`baseline_reference_manifest.json` records the exact fetched Git blob hashes.
`baseline_official_sources.json` preserves those Python sources for checking.
The R ports are modified implementations, authored for this project. Original
copyright notices and Apache-2.0/MIT license texts are included in
`LICENSE_DLinear.txt` and `LICENSE_iTransformer.txt`; retain them when publishing.

## What is held equal

Eight new runs: each method on Delhi air quality (24/96 hours) and Delhi daily
weather (7/30 days). Both datasets retain the same 12 variables used by TimeKAN.
History length is 96 time steps. The existing TimeKAN models are **not retrained**.

For every run the script asserts that source-file MD5, selected variables,
training means/standard deviations, chronological split boundaries, window
counts and training settings match the saved TimeKAN record. It also checks
that the number of observed test labels and the recomputed persistence and
seasonal-naive errors agree with TimeKAN's records. Results that fail these
checks are not silently accepted as comparable.

All models use the same training-only preprocessing, forward-filled inputs,
masked missing targets, 70/10/20 chronological splits, and eligible test origins
defined in `run_india_multivariate.R`. See `INDIA_MULTIVARIATE.md` for details.
Imputed labels are never scored as observations. Test windows can use the past
observed test history, matching rolling-origin TimeKAN evaluation.

Training budget: seed 0, batch size 64, Adam 0.001, up to 10 epochs, patience 3.
The best validation MSE chooses the checkpoint; test scores never choose models
or settings. No learning-rate search or dataset-specific tuning was performed.

iTransformer uses a **compact configuration**: embedding width 128, feed-forward
width 256, 8 attention heads, 2 encoder layers, dropout 0.1, GELU and final layer
normalization. Calendar covariate tokens are omitted for all datasets. This is
the original architecture with selected dimensions, not the paper's published
benchmark configuration. DLinear uses its original 25-step decomposition and
default random linear initialization (not the commented visualization weights).
Parameter counts, training curves, chosen epochs and run times are saved.

These are deadline-budget, single-seed comparisons. They do not establish a
general ranking, statistical significance, or the best possible performance of
any method. Equal training budgets do not replace extensive hyperparameter
tuning. Results are project experiments on these new datasets, not copied paper
scores or claims to reproduce the paper's benchmark tables.

## Run in R

Place `indian_learned_baselines.R`, `run_india_multivariate.R` and `timekan.R` in
the repository root. Install R `torch` and `coro`; use the existing Colab CUDA
installation. The four completed TimeKAN `_complete.rds` records must remain in
`results/india/timekan/`. Training aborts if a required reference is absent.

```sh
Rscript indian_learned_baselines.R --task self_test

Rscript indian_learned_baselines.R --device cuda \
  --air_csv /kaggle/input/air-quality-data-in-india/city_hour.csv \
  --climate_csv /kaggle/input/daily-climate-time-series-data-delhi-india/kaggel_weather_2013_to_2024.csv \
  --backup /content/drive/MyDrive/TimeKAN_runs/20261101_091847
```

Use `--task air` / `--task climate` or `--model DLinear` / `--model iTransformer`
to select a subset before running. Use a separate `--results_dir` when changing
settings, preserving the previous experiment. Explicit raw CSV paths avoid
depending on Kaggle's public ZIP endpoint. `--device cpu` is also supported.

All preprocessing, model definitions, training, scoring and plotting are in R.
Python is only an optional cross-language verification tool and Colab launcher.
Regenerate figures without torch or training:

```sh
Rscript indian_learned_baselines.R --task plots
```

## Results and plots

`results/india/learned_baselines/` contains:

- `learned_baseline_comparison.png`: all five models, four cases, MSE and MAE.
- `<case>_per_variable.png`: comparisons for each pollutant/weather variable.
- `<case>_horizon_errors.png`: errors as the forecast lead grows.
- `<case>_forecast_comparison_1.png`, etc.: observed vs predicted values in original
  units for TimeKAN, DLinear and iTransformer at the first test origin.
- `<run>_training.png`: training/validation loss and selected epoch.
- `all_models.csv`, `all_models_per_variable.csv`, `all_models_by_horizon.csv`:
  numerical support for figures, including the two original naive baselines.
- `<run>_complete.rds`: settings, data hash, windows, counts, parameter count,
  training duration/curves, metrics and sample predictions.
- `<run>.pt`: saved R torch model.

Scores on comparison plots are standardized; per-variable original-unit scores
remain in supporting CSVs. Illustrative forecast plots show one origin while
reported errors aggregate every eligible observed test target. With `--backup`,
results are copied after every successful run to the Google Drive backup folder.

## Implementation checks

The native R self-test checks multiple-batch/multiple-channel tensor layout,
finite gradients, decomposition reconstruction, constant-series endpoint
padding, DLinear channel independence and iTransformer cross-variable influence.
It also runs the existing masking and evaluation-denominator checks.

For the optional reference parity check, place `baseline_official_sources.json`
and `check_baseline_ports.py` beside the R files, then run:

```sh
Rscript indian_learned_baselines.R --task self_test
python tests/india/check_baseline_ports.py
```

The checker executes the relevant original Python classes extracted by AST,
avoiding unrelated optional dependencies. It assigns identical deterministic
parameters and inputs in R and Python, then compares forecast values and input
gradients. **Reference-check-only adjustment:** the upstream iTransformer uses
an in-place normalization division that current PyTorch rejects when computing
input gradients. The checker changes only that operation to mathematically
equivalent out-of-place division. This does not change forward values or the
architecture; the actual experiment training runs in R. The change is printed
and explicitly recorded here, rather than silently modifying the reference.

`port_checks/official_parity_report.json` saves maximum numerical discrepancies.
This deterministic check supports port correctness; it is not a claim that
independent R/Python training trajectories or full benchmark results are equal.


## Repository layout

Checked-in PNGs and newly generated plots are in `figures/india/timekan/` and `figures/india/learned_baselines/`. Numerical tables and lightweight run records are in `results/india/`. Reference sources and licenses are in `third_party/india_baselines/`; the optional Python parity checker is in `tests/india/`. Run the R commands from the repository root. Raw data and model weights are not included.
