

Code accompanying the paper. Reproduces the chain-parity diagnostic, the
real-world graph-level benchmarks, and the AdaGIM ablations.

## Overview

Three groups of experiments:

1. **Chain-parity diagnostic.** Train on `(K=1, L=10)` chains and evaluate on
   `(1, L)` chains up to `L = 30`. Shows the vanishing-influence behaviour
   of contractive graph implicit models (GIMs) and the corresponding failure
   to length-generalise beyond the training length.
2. **Real-world graph benchmarks.** Tune each model with a 6-trial Optuna TPE
   sweep on ZINC, LRGB Peptides-func, LRGB Peptides-struct, and OGBG-MolHIV;
   re-run the best configuration with three random seeds; report mean ± std.
   Implicit baselines: IGNN, EIGNN, MGNNI, GIND, Monotone-MIGNN, IGNN-finite
   (loop variant), AdaGIM. Standard MPGNN baselines: GCN, GCNII, APPNP.
3. **AdaGIM details.** The full AdaGIM model lives in
   [models/adagim.py](models/adagim.py); regulariser weights and gating
   knobs are configurable via the search space in
   [scripts/tune.py](scripts/tune.py).

## Layout

```
release/
├── run.py                  # framework entry point
├── models/                 # model registry; one wrapper per baseline
├── data/                   # PyG dataset loaders
├── trainers/               # PyG and synthetic-mode training loops
├── metrics/  utils/        # evaluation + I/O helpers
├── configs/                # default per-(model, dataset) YAML configs
├── third_party/            # vendored baselines (GIND, MGNNI, MIGNN)
├── helper/                 # chain-parity generator and synthetic-mode models
├── best_configs/
│   ├── implicit.yaml       # winning configs for IGNN..AdaGIM
│   └── gnn.yaml            # winning configs for GCN/GCNII/APPNP
└── scripts/
    ├── tune.py             # one Optuna sweep
    ├── multiseed.py        # re-run best config at multiple seeds
    ├── aggregate.py        # mean ± std over seeds
    ├── launch_tune.sh      # parallel sweeps across GPUs
    ├── launch_multiseed.sh # parallel multi-seed runs across GPUs
    └── chain_parity.sh     # chain-parity diagnostic
```

## Setup

Tested on Python 3.9 with CUDA-enabled PyTorch.

```bash
# Inside a fresh conda or virtualenv environment
pip install -r requirements.txt

# Optional: pre-stage datasets (otherwise PyG/OGB will download into ./datasets/)
export DATASETS_ROOT=/abs/path/to/datasets
```

## Reproduce real-world results

The fastest path uses the bundled best-config YAMLs (no Optuna sweep needed):

```bash
# 1. Re-run all (model, dataset) pairs with seeds 0, 1, 2.
bash scripts/launch_multiseed.sh

# 2. Aggregate seeds into a mean ± std table.
python scripts/aggregate.py
```

Override the GPUs in use by setting `GPUS` in the env:

```bash
GPUS="0 1 2 3" bash scripts/launch_multiseed.sh
```

To reproduce the Optuna sweep itself (~30–90 min per (model, dataset) pair on
a single GPU; full sweep is ~24 hours on 8 GPUs):

```bash
bash scripts/launch_tune.sh
# then re-run multi-seed at the new best configs:
bash scripts/launch_multiseed.sh
```

## Reproduce the chain-parity diagnostic

```bash
GPU=0 bash scripts/chain_parity.sh
```

This runs IGNN, EIGNN, IGNN-finite, and AdaGIM on the `(K=1, L=10) → L=30`
extrapolation protocol with three seeds each. Per-iteration metrics are
written under `runs_chain_parity/`.

## A single experiment

```bash
# Training one model on one dataset
python -m run --dataset zinc --model adagim --gpu 0 --seed 0 \
    --override hidden=128 num_layers=3 s_W=0.9 K_max=10 epochs=200

# Loading a YAML config directly
python -m run --tune_config configs/tune.yaml
```

Outputs go to `runs/<dataset>/<model>/seed_<n>/` and contain `summary.yaml`
(best-val + test-at-best-val), `metrics.csv` (per-epoch), and a learning-curve
plot.

## Notes

* The synthetic chain-parity loader pulls helpers from
  [helper/data/kl_chain_parity_task.py](helper/data/kl_chain_parity_task.py).
  This subdirectory is bundled so the chain-parity diagnostic runs out of
  the box.
* GIND, MGNNI, and Monotone-MIGNN are vendored under
  [third_party/](third_party/). The wrappers in
  [models/gind.py](models/gind.py), [models/mgnni.py](models/mgnni.py), and
  [models/monotone_mignn.py](models/monotone_mignn.py) bridge the framework's
  uniform interface to each upstream's native modules.
* GCN's `virtual_node` augmentation is fixed to `False` in
  [scripts/tune.py](scripts/tune.py) so the comparison with implicit
  baselines is on equal architectural footing (no graph-level augmentation
  on either side).


