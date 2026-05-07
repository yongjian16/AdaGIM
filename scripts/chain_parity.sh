#!/usr/bin/env bash
# Run the chain-parity diagnostic: train each model on a (K=1, L=10) chain and
# evaluate length-extrapolation up to L=30. Reproduces the figure showing
# vanishing influence on long chains for contractive GIMs.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"0 1 2"}
OUT=${OUT:-runs_chain_parity}

mkdir -p "$OUT"

for seed in $SEEDS; do
  for model in ignn eignn ignn_finite_loop adagim; do
    log="$OUT/${model}.seed${seed}.log"
    echo "[chain-parity] $model seed=$seed gpu=$GPU"
    "$PYTHON" -m run \
        --dataset chain_parity \
        --model "$model" \
        --gpu "$GPU" \
        --seed "$seed" \
        --out_dir "$OUT" \
        --override "K_train=1 L_train=10 K_test=1 L_test=30 epochs=300 patience=80 hidden=64" \
        > "$log" 2>&1
  done
done
echo "[chain-parity] done; outputs in $OUT/"
