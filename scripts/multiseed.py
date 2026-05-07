"""Re-run a (model, dataset) at its best Optuna config across multiple seeds."""
from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
import yaml

REPO = Path(__file__).resolve().parent.parent
PY = os.environ.get("PYTHON", "python")
TIMEOUT_SECONDS = int(os.environ.get("TUNE_TIMEOUT", "7200"))

DATASETS: Dict[str, Dict[str, Any]] = {
    "zinc":            {"higher": False, "batch_size": 128, "epochs": 200, "patience": 30},
    "peptides_func":   {"higher": True,  "batch_size": 128, "epochs": 100, "patience": 30},
    "peptides_struct": {"higher": False, "batch_size": 128, "epochs": 100, "patience": 30},
    "ogbg_molhiv":     {"higher": True,  "batch_size": 128, "epochs": 100, "patience": 30},
}

IMPLICIT_MODELS = ("ignn", "eignn", "mgnni", "monotone_mignn", "gind",
                   "ignn_finite_loop", "adagim")
GNN_MODELS = ("gcn", "gcnii", "appnp")
ALL_MODELS = IMPLICIT_MODELS + GNN_MODELS

def _load_best(model: str, dataset: str) -> Optional[Dict[str, Any]]:
    fname = "implicit.yaml" if model in IMPLICIT_MODELS else "gnn.yaml"
    path = REPO / "best_configs" / fname
    if not path.exists():
        return None
    cfgs = yaml.safe_load(path.read_text()) or {}
    return cfgs.get(f"{model}/{dataset}")

def _load_best_from_optuna(model: str, dataset: str) -> Optional[Dict[str, Any]]:
    storage = f"sqlite:///{REPO}/runs_tune/{model}_{dataset}/optuna.db"
    name = f"{model}_{dataset}"
    try:
        study = optuna.load_study(study_name=name, storage=storage)
    except Exception as e:  # noqa: BLE001
        print(f"[{model}/{dataset}] cannot load Optuna study: {e}")
        return None
    higher = DATASETS[dataset]["higher"]
    finished = [t for t in study.trials
                if t.state.name == "COMPLETE"
                and t.value is not None
                and abs(t.value) < 90.0]
    if not finished:
        print(f"[{model}/{dataset}] no completed trials in Optuna study")
        return None
    finished.sort(key=lambda t: t.value, reverse=higher)
    best = finished[0]
    return {k.removeprefix("knob_"): v for k, v in best.user_attrs.items()
            if k.startswith("knob_")}

def _run_one(model: str, dataset: str, gpu: str, seed: int,
             knobs: Dict[str, Any], out_dir: Path) -> float:
    cfg = DATASETS[dataset]
    overrides = [f"{k}={v}" for k, v in knobs.items()]
    overrides += [
        f"epochs={cfg['epochs']}",
        f"patience={cfg['patience']}",
        f"batch_size={cfg['batch_size']}",
    ]
    cmd = [PY, "-m", "run",
           "--dataset", dataset,
           "--model", model,
           "--gpu", str(gpu),
           "--out_dir", str(out_dir),
           "--seed", str(seed),
           "--override", *overrides]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir.parent / f"{out_dir.name}.seed{seed}.log"
    print(f"[{model}/{dataset} seed {seed}] launching on gpu {gpu}")
    try:
        with open(log_file, "w") as f:
            subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                           timeout=TIMEOUT_SECONDS, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[{model}/{dataset} seed {seed}] rc={e.returncode}; see {log_file}")
        return float("nan")
    except subprocess.TimeoutExpired:
        print(f"[{model}/{dataset} seed {seed}] timed out after {TIMEOUT_SECONDS}s")
        return float("nan")
    except Exception as e:  # noqa: BLE001
        print(f"[{model}/{dataset} seed {seed}] unexpected: {e}")
        return float("nan")

    summary_path = out_dir / dataset / model / f"seed_{seed}" / "summary.yaml"
    if not summary_path.exists():
        print(f"[{model}/{dataset} seed {seed}] no summary at {summary_path}")
        return float("nan")
    try:
        s = yaml.safe_load(summary_path.read_text())
        return float(s.get("test_metric_at_best_val", float("nan")))
    except Exception as e:  # noqa: BLE001
        print(f"[{model}/{dataset} seed {seed}] summary read failed: {e}")
        return float("nan")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    knobs = _load_best(args.model, args.dataset)
    if knobs is None:
        knobs = _load_best_from_optuna(args.model, args.dataset)
    if knobs is None:
        raise SystemExit(
            f"No best config found for {args.model}/{args.dataset}. "
            f"Either populate best_configs/*.yaml or run scripts/tune.py first."
        )

    print(f"[{args.model}/{args.dataset}] knobs: {knobs}")
    out_dir = REPO / "runs_multiseed" / f"{args.model}_{args.dataset}"
    for seed in args.seeds:
        v = _run_one(args.model, args.dataset, args.gpu, seed, knobs, out_dir)
        print(f"[{args.model}/{args.dataset} seed {seed}] test_metric_at_best_val={v}")

if __name__ == "__main__":
    main()
