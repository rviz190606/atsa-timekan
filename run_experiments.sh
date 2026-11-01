#!/usr/bin/env bash
# Reproduce part of the TimeKAN paper on the ETT datasets.
#   results/main.csv      Table 1: look-back 96, horizons 96/192/336/720
#   results/ablation.csv  Table 9 (number of frequency bands) and Table 3 (fixed-order KANs)
#   results/logs/         full training log of every run
# Each run appends one row as soon as it finishes, so partial results are usable.
# Usage:  ./run_experiments.sh            (set R_LIBS_USER if torch is in a custom library)

set -u
cd "$(dirname "$0")"
mkdir -p results/logs

# Shared settings for every run (lr / batch size chosen from a quick check on ETTh1).
COMMON="--seq_len 96 --lr 0.01 --batch_size 128 --epochs 10 --patience 3"

run() {  # run <log name> <args...>
  local name=$1; shift
  echo "[$(date +%H:%M)] $name"
  Rscript timekan.R $COMMON "$@" > "results/logs/$name.log" 2>&1 \
    || echo "  FAILED, see results/logs/$name.log"
}

main_table() {  # main_table <dataset>
  for F in 96 192 336 720; do
    run "${1}_${F}" --csv "data/$1.csv" --pred_len $F --out results/main.csv \
        --save "results/${1}_${F}.pt"
  done
}

# 1. Main results on the fast hourly datasets.
main_table ETTh1
main_table ETTh2

# 2. Ablations at horizon 96 (the default model is levels=3 with multi-order KANs).
for D in ETTh1 ETTh2; do
  for K in 2 4 5; do
    run "${D}_levels${K}" --csv "data/$D.csv" --pred_len 96 --levels $K \
        --tag "levels=$K" --out results/ablation.csv
  done
  for O in 2 5; do
    run "${D}_fixed${O}" --csv "data/$D.csv" --pred_len 96 --fixed_order $O \
        --tag "fixed order $O" --out results/ablation.csv
  done
done

# 3. Main results on the larger 15-minute datasets.
main_table ETTm1
main_table ETTm2

echo "[$(date +%H:%M)] all done"
