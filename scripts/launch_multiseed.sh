#!/usr/bin/env bash
# Re-run each (model, dataset) at its best config across seeds 0, 1, 2 and
# distribute the work across the specified GPUs.
#
# Override defaults via env vars:
#   GPUS="0 1 2 3 4 5 6 7"  SEEDS="0 1 2"  PYTHON=/path/to/python
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}

mkdir -p logs_multiseed

read -r -d '' QUEUE <<'EOF' || true
0:ignn:zinc
0:eignn:zinc
0:mgnni:zinc
1:ignn:peptides_func
1:eignn:peptides_func
1:mgnni:peptides_func
2:ignn:peptides_struct
2:eignn:peptides_struct
2:mgnni:peptides_struct
3:ignn:ogbg_molhiv
3:eignn:ogbg_molhiv
3:mgnni:ogbg_molhiv
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
7:adagim:zinc
7:adagim:peptides_func
7:adagim:peptides_struct
7:adagim:ogbg_molhiv
0:gcn:zinc
1:gcn:peptides_func
2:gcn:peptides_struct
3:gcn:ogbg_molhiv
4:gcnii:zinc
5:gcnii:peptides_func
6:gcnii:peptides_struct
7:gcnii:ogbg_molhiv
0:appnp:zinc
1:appnp:peptides_func
2:appnp:peptides_struct
3:appnp:ogbg_molhiv
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
    local script="logs_multiseed/_runner_gpu${gpu}.sh"
    local queue_log="logs_multiseed/_queue_gpu${gpu}.log"
    cat > "$script" <<EOF
#!/usr/bin/env bash
set -u
cd $(pwd)
for pair in ${pairs}; do
    model=\${pair%%:*}
    dataset=\${pair##*:}
    log="logs_multiseed/\${model}_\${dataset}_gpu${gpu}.log"
    echo "[gpu ${gpu}] starting \${model}/\${dataset}"
    "${PYTHON}" scripts/multiseed.py --model \${model} --dataset \${dataset} \\
        --gpu ${gpu} --seeds ${SEEDS} >> "\${log}" 2>&1 \\
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

echo "[launch] multi-seed runs started; aggregate with: python scripts/aggregate.py"
