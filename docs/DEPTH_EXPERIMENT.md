# How many TimeKAN layers are useful?

This is a new **single-seed sensitivity experiment in the project's R implementation**,
not a result reported by the authors and not an exact reproduction of their full
Python configuration. No depth result should be quoted until training completes.

## Fixed protocol

- Datasets: ETTh1, ETTh2 and ETTm1, all seven variables. The first two are the
  smallest ETT benchmarks; ETTm1 adds a 15-minute dataset without the many channels
  of Electricity or Weather. ETTm2 has the same size as ETTm1 and can be substituted.
- Layers **L = 1, 2, 3, 4, 5, 10, 20**; seed **0** only: **21 runs**.
- History 96 and forecast horizon 96 samples (96 hours for ETTh; 24 hours for ETTm).
- Standard ETT chronological 12/4/4-month train/validation/test split; full
  benchmark partitions, no subsampling. Train-only z-score scaling.
- Width 16, three frequency bands, pooling factor 2, Chebyshev orders 3/2/1,
  depthwise convolutions, existing R model's `cfd_mode=paper` and window normalization.
  The newer `exact`/`official` CFD alternatives are not varied in this study.
- Adam 0.001, batch 128, maximum 10 epochs, patience 3 for every depth.
  Initial weights use seed 0; shuffle RNG resets to seed+epoch so model size does
  not change the shuffle stream simply by consuming initialization randomness.
- The lowest validation MSE selects each run's checkpoint. MSE and MAE are
  computed over all seven standardized variables, all 96 forecast leads and all
  eligible rolling test origins. Test errors do not select checkpoints/settings.

## Run

An optional [Colab notebook](../notebooks/TimeKAN_Layer_Depth_Study.ipynb)
sets up R torch, mounts Drive, runs only this study, displays results and creates
a full backup ZIP. Its default GPU check stops if a GPU has not been allocated.

On Kaggle, use a Python notebook with **GPU T4 x2** and **Internet on**, install
the CUDA R torch package, then run `python run_depth_kaggle.py` from the repository
root. The optional launcher assigns independent experiments to the two GPUs;
it does not change any model or training settings. Each R worker writes to its
own job folder, and the launcher consolidates completed records serially.
It updates `TimeKAN_depth_results.zip` after every completed run. The active
project notebook is [TimeKAN Layer Depth Study R](https://www.kaggle.com/code/deb044/timekan-layer-depth-study-r/edit).
Kaggle draft files require persistence or a saved version; retain the results ZIP.

The [Kaggle notebook](../notebooks/TimeKAN_Layer_Depth_Kaggle.ipynb) includes the
setup and complete run. Its helper installs official CUDA R torch 0.17.0 into
`/kaggle/working/R-library` and caches the installer under the working directory,
so reconnecting does not require another full download. The installer cache and
R library are outside the repository and excluded from the experiment ZIP.

Use the repository root, with R `torch` and `coro` installed. GPU setup follows
the [official R torch installation instructions](https://torch.mlverse.org/docs/articles/installation).
Only the three required public ETT CSVs are downloaded automatically if absent.

```sh
Rscript run_depth_experiment.R --task self_test
Rscript run_depth_experiment.R --device cuda
```

On a CPU use `--device cpu --threads 2`. Before a long CPU run measure one full
ETTh1 epoch at L=2, saved in a separate benchmark directory:

```sh
Rscript run_depth_experiment.R --task benchmark --device cpu
```

For Colab, add the R commands to Python cells with a leading `!`. To preserve
results on an already mounted Google Drive:

```sh
Rscript run_depth_experiment.R --device cuda --backup /content/drive/MyDrive/TimeKAN_depth
```

The script resumes completed runs, checks source/data hashes and settings before
reuse, and updates tables/plots and the optional backup after every completed run.
It never imports an older L=2 result from a different experiment. Failures are
reported explicitly; they are not converted into invented error values.
Use a new `--results_dir` and `--figures_dir` when changing training settings.
To include ETTm2 as a fourth dataset use `--datasets ETTh1,ETTh2,ETTm1,ETTm2`.

## Compact outputs

- `results/depth/summary.csv`: one row per run with L, validation MSE, test MSE/MAE,
  parameters, measured runtime, selected epoch and training-limit flags.
- `results/depth/learning_curves.csv`: train/validation MSE and duration per epoch.
- `results/depth/records/`: small RDS records with configuration, hashes, scaling,
  window counts and session information. No model weights are published.
- `figures/depth/depth_accuracy_cost.png`: error, runtime and parameter count vs L.
- `figures/depth/depth_validation_curves.png`: convergence check for each dataset.

Only the two figures and two consolidated tables need to be tracked in Git;
per-run records and temporary benchmark files stay ignored. The optional Drive
backup retains the detailed records. Regenerate figures without torch/training:

```sh
Rscript run_depth_experiment.R --task plots
```

## Interpretation

Report the measured accuracy/cost trade-off. If deeper models do not improve,
the supported conclusion is limited to these datasets, horizon, seed and training
budget. This cannot establish that L=3 is universally optimal or explain the
authors' choice. One seed supplies no estimate of seed-to-seed uncertainty.
`hit_epoch_limit` and `best_in_last_two` flag potentially unfinished convergence;
a worse deep model may need more training or different optimization. The fixed
ten-epoch budget intentionally measures performance under the project deadline.
