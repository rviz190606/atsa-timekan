#!/usr/bin/env bash
# Extension: does it matter that CFD's frequency bands overlap? (see cfd_response.R)
# Changes only how the series is split into bands:
#   exact      ideal FFT low-pass before downsampling, so the bands do not overlap
#   official   the authors' upsampling scale (1/4 of the amplitude), as in their code
# on ETTh1 and ETTh2, horizon 96, 3 seeds each (12 runs, ~1 min each on a GPU).
# The baseline is the "paper" runs in results/orders.csv (same settings and seeds).
#   results/cfd.csv   one row per run
# Usage:  ./run_cfd_experiment.sh

set -u
cd "$(dirname "$0")"
mkdir -p results/logs

COMMON="--seq_len 96 --pred_len 96 --lr 0.01 --batch_size 128 --epochs 10 --patience 3"

for D in ETTh1 ETTh2; do
  for SEED in 0 1 2; do
    for MODE in exact official; do
      name="${D}_96_${MODE}_s${SEED}"
      echo "[$(date +%H:%M)] $name"
      Rscript timekan.R $COMMON --csv "data/$D.csv" --seed $SEED \
          --cfd_mode $MODE --tag $MODE --out results/cfd.csv > "results/logs/cfd_$name.log" 2>&1 \
        || echo "  FAILED, see results/logs/cfd_$name.log"
    done
  done
done
echo "[$(date +%H:%M)] all done"
