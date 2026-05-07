"""Optuna sweep for a single (model, dataset) pair."""
from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

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

def _study_name(model: str, dataset: str) -> str:
    return f"{model}_{dataset}"

def _storage_uri(model: str, dataset: str) -> str:
    out_dir = REPO / "runs_tune" / f"{model}_{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{out_dir}/optuna.db"

def _suggest_implicit(trial: optuna.Trial, model: str) -> Dict[str, Any]:
    knobs: Dict[str, Any] = {}
    knobs["hidden"]       = trial.suggest_categorical("hidden", [64, 96, 128])
    knobs["num_layers"]   = trial.suggest_categorical("num_layers", [1, 2, 3])
    if model != "gind":
        knobs["row_norm"] = trial.suggest_categorical("row_norm", [True, False])
    knobs["graph_pool"]   = trial.suggest_categorical("graph_pool", ["mean", "sum"])
    knobs["head_dropout"] = trial.suggest_categorical("head_dropout", [0.0, 0.1, 0.2])
    knobs["lr"]           = trial.suggest_categorical("lr", [1.0e-3, 5.0e-4, 1.0e-4])
    knobs["wd"]           = trial.suggest_categorical("wd", [1.0e-5, 1.0e-4])

    if model == "ignn":
        knobs["kappa"] = trial.suggest_categorical("kappa", [0.7, 0.9, 0.99])
        knobs["phi"] = trial.suggest_categorical("phi", ["tanh", "relu"])
        knobs["fw_mitr"] = 50
    elif model == "eignn":
        knobs["gamma"] = trial.suggest_categorical("gamma", [0.7, 0.9, 0.99])
        knobs["threshold"] = trial.suggest_categorical("threshold", [1.0e-4, 1.0e-6])
        knobs["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.4])
        knobs["max_iter"] = 100
    elif model == "mgnni":
        knobs["gamma"] = trial.suggest_categorical("gamma", [0.7, 0.8, 0.99])
        knobs["threshold"] = trial.suggest_categorical("threshold", [1.0e-4, 1.0e-6])
        knobs["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.4])
        knobs["max_iter"] = 100
    elif model == "monotone_mignn":
        knobs["nonlin"] = trial.suggest_categorical("nonlin", ["relu", "tanh"])
        knobs["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.4])
        knobs["post_ig_act"] = trial.suggest_categorical("post_ig_act", [True, False])
        knobs["head_bias"] = trial.suggest_categorical("head_bias", [True, False])
        knobs["max_iter"] = 100
        knobs["tol"] = 1.0e-5
    elif model == "gind":
        knobs["graph_pool"] = "add"
        knobs["norm"] = trial.suggest_categorical("norm", ["LayerNorm", "InstanceNorm"])
        knobs["alpha"] = trial.suggest_categorical("alpha", [0.1, 0.145, 0.5, 1.0])
        knobs["dropout_imp"] = trial.suggest_categorical("dropout_imp", [0.0, 0.1, 0.4])
        knobs["dropout_exp"] = trial.suggest_categorical("dropout_exp", [0.0, 0.4])
        knobs["act_imp"] = trial.suggest_categorical("act_imp", ["tanh", "relu"])
        knobs["double_linear"] = trial.suggest_categorical("double_linear", [True, False])
    elif model == "ignn_finite_loop":
        knobs["T"] = trial.suggest_categorical("T", [4, 8, 16])
        knobs["activation"] = trial.suggest_categorical("activation", ["tanh", "relu"])
        knobs["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
    elif model == "adagim":
        knobs["s_W"] = trial.suggest_categorical("s_W", [0.7, 0.9, 0.99])
        knobs["alpha"] = trial.suggest_categorical("alpha", [0.5, 1.0])
        knobs["phi"] = trial.suggest_categorical("phi", ["tanh", "relu"])
        knobs["K_max"] = trial.suggest_categorical("K_max", [10, 20, 30])
        knobs["gate_temp"] = trial.suggest_categorical("gate_temp", [0.5, 1.0])
        knobs["gate_tau_init"] = trial.suggest_categorical("gate_tau_init", [0.1, 1.0])
        knobs["gate_lambda_sharp"] = trial.suggest_categorical("gate_lambda_sharp", [0.0, 0.005, 0.01])
        knobs["gate_lambda_nowrite"] = trial.suggest_categorical("gate_lambda_nowrite", [0.0, 0.01, 0.05])
    else:
        raise ValueError(model)
    return knobs

def _suggest_gnn(trial: optuna.Trial, model: str) -> Dict[str, Any]:
    knobs: Dict[str, Any] = {}
    knobs["hidden"] = trial.suggest_categorical("hidden", [64, 96, 128])
    knobs["graph_pool"] = trial.suggest_categorical("graph_pool", ["mean", "sum"])
    knobs["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.4])
    knobs["lr"] = trial.suggest_categorical("lr", [1.0e-3, 5.0e-4, 1.0e-4])
    knobs["wd"] = trial.suggest_categorical("wd", [1.0e-5, 1.0e-4])

    if model == "gcn":
        knobs["num_layers"] = trial.suggest_categorical("num_layers", [2, 4, 6, 8])
        knobs["virtual_node"] = False
    elif model == "gcnii":
        knobs["num_layers"] = trial.suggest_categorical("num_layers", [4, 8, 16])
        knobs["alpha"] = trial.suggest_categorical("alpha", [0.1, 0.2])
        knobs["theta"] = trial.suggest_categorical("theta", [0.5, 1.0])
    elif model == "appnp":
        knobs["K"] = trial.suggest_categorical("K", [5, 10, 20])
        knobs["alpha"] = trial.suggest_categorical("alpha", [0.1, 0.2])
    else:
        raise ValueError(model)
    return knobs

def _suggest(trial: optuna.Trial, model: str) -> Dict[str, Any]:
    if model in IMPLICIT_MODELS:
        return _suggest_implicit(trial, model)
    return _suggest_gnn(trial, model)

def _run_one(out_dir: Path, model: str, dataset: str, gpu: str, seed: int,
             knobs: Dict[str, Any], cfg: Dict[str, Any]) -> float:
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
    try:
        with open(log_file, "w") as f:
            subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                           timeout=TIMEOUT_SECONDS, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[{out_dir.name}] subprocess rc={e.returncode}; see {log_file}")
        return float("nan")
    except subprocess.TimeoutExpired:
        print(f"[{out_dir.name}] timed out after {TIMEOUT_SECONDS}s")
        return float("nan")
    except Exception as e:  # noqa: BLE001
        print(f"[{out_dir.name}] unexpected: {e}")
        return float("nan")

    summary_path = out_dir / dataset / model / f"seed_{seed}" / "summary.yaml"
    if not summary_path.exists():
        print(f"[{out_dir.name}] no summary at {summary_path}")
        return float("nan")
    try:
        s = yaml.safe_load(summary_path.read_text())
        v = float(s.get("best_val_metric", float("nan")))
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception as e:  # noqa: BLE001
        print(f"[{out_dir.name}] summary read failed: {e}")
        return float("nan")

def _objective_factory(model: str, dataset: str, gpu: str, cfg: Dict[str, Any]):
    sentinel = -99.0 if cfg["higher"] else 99.0

    def objective(trial: optuna.Trial) -> float:
        knobs = _suggest(trial, model)
        out_dir = REPO / "runs_tune" / f"{model}_{dataset}" / f"trial_{trial.number:04d}"
        for k, v in knobs.items():
            trial.set_user_attr(f"knob_{k}", v)
        v = _run_one(out_dir, model, dataset, gpu=gpu, seed=0, knobs=knobs, cfg=cfg)
        if math.isnan(v):
            return sentinel
        s_path = out_dir / dataset / model / "seed_0" / "summary.yaml"
        if s_path.exists():
            try:
                s = yaml.safe_load(s_path.read_text())
                trial.set_user_attr("test_metric_at_best_val",
                                    float(s.get("test_metric_at_best_val", float("nan"))))
            except Exception:  # noqa: BLE001
                pass
        return v
    return objective

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--n_trials", type=int, default=6)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    direction = "maximize" if cfg["higher"] else "minimize"
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=15)
    study = optuna.create_study(
        study_name=_study_name(args.model, args.dataset),
        storage=_storage_uri(args.model, args.dataset),
        load_if_exists=True, direction=direction, sampler=sampler, pruner=pruner,
    )
    print(f"[tune {args.model}/{args.dataset} gpu={args.gpu}] "
          f"running {args.n_trials} trials; study has {len(study.trials)} so far. "
          f"direction={direction}")
    study.optimize(_objective_factory(args.model, args.dataset, args.gpu, cfg),
                   n_trials=args.n_trials, gc_after_trial=True)
    if study.trials:
        try:
            print(f"[tune {args.model}/{args.dataset} gpu={args.gpu}] done. "
                  f"best so far: trial {study.best_trial.number}, val={study.best_value:.4f}")
        except Exception:  # noqa: BLE001
            pass

if __name__ == "__main__":
    main()
