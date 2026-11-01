#!/usr/bin/env bash
# Extension: does giving high-frequency bands higher-order KANs actually help?
# Compares four KAN order assignments over the 3 bands (band 1 = highest frequency):
#   paper     3,2,1   (the paper's design)
#   reversed  1,2,3   (same parameter count as paper)
#   fixed2    2,2,2   (same parameter count as paper)
#   fixed3    3,3,3   (every band gets the same, highest capacity; used by interpret_kan.R)
# on ETTh1 and ETTh2, horizons 96 and 336, 3 seeds each (48 runs, ~1 min each on a laptop GPU).
#   results/orders.csv           one row per run
#   results/orders/*.pt          the trained horizon-96 models (for interpret_kan.R)
# Usage:  ./run_order_experiment.sh

set -u
cd "$(dirname "$0")"
mkdir -p results/logs results/orders

COMMON="--seq_len 96 --lr 0.01 --batch_size 128 --epochs 10 --patience 3"

for D in ETTh1 ETTh2; do
  for F in 96 336; do
    for SEED in 0 1 2; do
      for MODE in paper reversed fixed2 fixed3; do
        case $MODE in
          paper)    ARGS="--order_mode paper" ;;
          reversed) ARGS="--order_mode reversed" ;;
          fixed2)   ARGS="--fixed_order 2" ;;
          fixed3)   ARGS="--fixed_order 3" ;;
        esac
        name="${D}_${F}_${MODE}_s${SEED}"
        SAVE=""
        [ "$F" = 96 ] && SAVE="--save results/orders/$name.pt"
        echo "[$(date +%H:%M)] $name"
        Rscript timekan.R $COMMON --csv "data/$D.csv" --pred_len $F --seed $SEED $ARGS \
            --tag $MODE --out results/orders.csv $SAVE > "results/logs/order_$name.log" 2>&1 \
          || echo "  FAILED, see results/logs/order_$name.log"
      done
    done
  done
done
echo "[$(date +%H:%M)] all done"
