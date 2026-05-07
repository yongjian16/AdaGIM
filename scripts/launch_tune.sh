#!/usr/bin/env bash
# Launch the full Optuna sweep across all (model, dataset) pairs reported in
# the paper. Distributes 40 sweeps (10 models x 4 datasets) across however
# many GPUs are listed in $GPUS, running pairs sequentially on each GPU.
#
# Override defaults via env vars:
#   GPUS="0 1 2 3 4 5 6 7"  N_TRIALS=6  PYTHON=/path/to/python
#   DATASETS_ROOT=/path/with/preloaded/datasets   (default: <repo>/datasets)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
N_TRIALS=${N_TRIALS:-6}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}

mkdir -p logs_tune

read -r -d '' QUEUE <<'EOF' || true
0:ignn:zinc
0:eignn:zinc
0:mgnni:zinc
0:adagim:zinc
1:ignn:peptides_func
1:eignn:peptides_func
1:mgnni:peptides_func
1:adagim:peptides_func
2:ignn:peptides_struct
2:eignn:peptides_struct
2:mgnni:peptides_struct
2:adagim:peptides_struct
3:ignn:ogbg_molhiv
3:eignn:ogbg_molhiv
3:mgnni:ogbg_molhiv
3:adagim:ogbg_molhiv
4:monotone_mignn:zinc
4:monotone_mignn:peptides_func
4:monotone_mignn:peptides_struct
4:monotone_mignn:ogbg_molhiv
5:gind:zinc
5:gind:peptides_func
5:gind:peptides_struct
5:gind:ogbg_molhiv
6:ignn_finite_loop:zinc
6:ignn_finite_loop:peptides_func
6:ignn_finite_loop:peptides_struct
6:ignn_finite_loop:ogbg_molhiv
7:gcn:zinc
7:gcn:peptides_func
7:gcn:peptides_struct
7:gcn:ogbg_molhiv
0:gcnii:zinc
1:gcnii:peptides_func
2:gcnii:peptides_struct
3:gcnii:ogbg_molhiv
4:appnp:zinc
5:appnp:peptides_func
6:appnp:peptides_struct
7:appnp:ogbg_molhiv
EOF

declare -A GPU_QUEUE
ALLOWED=" $GPUS "
while IFS=':' read -r gpu model dataset; do
    [[ -z "$gpu" ]] && continue
    [[ "$ALLOWED" == *" $gpu "* ]] || continue
    GPU_QUEUE["$gpu"]+="${model}:${dataset} "
done <<< "$QUEUE"

launch_gpu_queue() {
    local gpu="$1"
    local pairs="$2"
    local script="logs_tune/_runner_gpu${gpu}.sh"
    local queue_log="logs_tune/_queue_gpu${gpu}.log"
    cat > "$script" <<EOF
#!/usr/bin/env bash
set -u
cd $(pwd)
for pair in ${pairs}; do
    model=\${pair%%:*}
    dataset=\${pair##*:}
    log="logs_tune/\${model}_\${dataset}_gpu${gpu}.log"
    echo "[gpu ${gpu}] starting \${model}/\${dataset}"
    "${PYTHON}" scripts/tune.py --model \${model} --dataset \${dataset} \\
        --gpu ${gpu} --n_trials ${N_TRIALS} >> "\${log}" 2>&1 \\
        || echo "[gpu ${gpu}] \${model}/\${dataset} FAILED (continuing)"
done
echo "[gpu ${gpu}] queue done"
EOF
    chmod +x "$script"
    echo "[launch] gpu=$gpu pairs=[$pairs]"
    nohup "$script" > "$queue_log" 2>&1 &
}

for gpu in $GPUS; do
    [[ -n "${GPU_QUEUE[$gpu]:-}" ]] || continue
    launch_gpu_queue "$gpu" "${GPU_QUEUE[$gpu]}"
done

echo "[launch] sweeps started; monitor with: tail -f logs_tune/*.log"
