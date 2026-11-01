#!/usr/bin/env bash
# Extension: is the Chebyshev basis special, or does any flexible function family work?
# Swaps the KAN's basis functions, keeping everything else (and the parameter count) the same:
#   legendre, fourier, rbf   same number of functions per band as Chebyshev -> same parameters
#   linear                   control: an ordinary linear layer (no learnable curves)
# on ETTh1 and ETTh2, horizons 96 and 336, 3 seeds each (48 runs, ~1 min each on a laptop GPU).
# The Chebyshev runs are the "paper" runs in results/orders.csv (run_order_experiment.sh),
# which use exactly the same settings and seeds.
#   results/basis.csv   one row per run
# Usage:  ./run_basis_experiment.sh

set -u
cd "$(dirname "$0")"
mkdir -p results/logs

COMMON="--seq_len 96 --lr 0.01 --batch_size 128 --epochs 10 --patience 3"

for D in ETTh1 ETTh2; do
  for F in 96 336; do
    for SEED in 0 1 2; do
      for BASIS in legendre fourier rbf linear; do
        name="${D}_${F}_${BASIS}_s${SEED}"
        echo "[$(date +%H:%M)] $name"
        Rscript timekan.R $COMMON --csv "data/$D.csv" --pred_len $F --seed $SEED \
            --basis $BASIS --tag $BASIS --out results/basis.csv > "results/logs/basis_$name.log" 2>&1 \
          || echo "  FAILED, see results/logs/basis_$name.log"
      done
    done
  done
done
echo "[$(date +%H:%M)] all done"
