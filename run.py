"""Single entry point. Outputs to runs/<dataset>/<model>/seed_<n>/."""
import argparse
import os
from pathlib import Path
from typing import Any, Dict

def _peek_gpu_arg() -> "int | None":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tune_config", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args, _ = parser.parse_known_args()
    if args.gpu is not None:
        return args.gpu
    if args.tune_config:
        try:
            import yaml
            with open(args.tune_config) as f:
                cfg = yaml.safe_load(f) or {}
            if "gpu" in cfg:
                return int(cfg["gpu"])
        except Exception:
            pass
    return None

_GPU_REQUEST = _peek_gpu_arg()
if _GPU_REQUEST is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_GPU_REQUEST)
    print(f"[run] CUDA_VISIBLE_DEVICES = {_GPU_REQUEST} (set from config/CLI)")

import torch

import data as data_registry
import models as model_registry
import trainers
from utils import set_seeds, load_yaml, save_yaml, ensure_dir

def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--model_config", type=str, default=None)
    ap.add_argument("--train_config", type=str, default="configs/train/default.yaml")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--tune_config", type=str, default=None)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent

    tune_cfg: Dict[str, Any] = {}
    if args.tune_config:
        tune_cfg = load_yaml(args.tune_config) or {}
        if "dataset" in tune_cfg and not args.dataset:
            args.dataset = tune_cfg["dataset"]
        if "model" in tune_cfg and not args.model:
            args.model = tune_cfg["model"]
        if "seed" in tune_cfg:
            args.seed = int(tune_cfg["seed"])
        if "out_dir" in tune_cfg and args.out_dir is None:
            args.out_dir = str(tune_cfg["out_dir"])
    if args.out_dir is None:
        args.out_dir = "runs"

    dataset_name = args.dataset
    dataset_cfg = {}
    if args.config:
        dataset_cfg = load_yaml(args.config)
        dataset_name = dataset_cfg.get("name") or dataset_name
    elif dataset_name:
        ds_yaml = repo_root / f"configs/dataset/{dataset_name}.yaml"
        if ds_yaml.exists():
            dataset_cfg = load_yaml(ds_yaml)
    assert dataset_name, "must specify --dataset or --config"

    model_name = args.model
    model_cfg = {}
    if args.model_config:
        model_cfg = load_yaml(args.model_config)
        model_name = model_cfg.get("name") or model_name
    elif model_name:
        m_yaml = repo_root / f"configs/model/{model_name}.yaml"
        if m_yaml.exists():
            model_cfg = load_yaml(m_yaml)
    assert model_name, "must specify --model or --model_config"

    train_cfg = {}
    train_path = repo_root / args.train_config
    if train_path.exists():
        train_cfg = load_yaml(train_path)

    for k, v in tune_cfg.items():
        if k in ("dataset", "model", "seed", "out_dir", "tune_config", "gpu"):
            continue
        dataset_cfg[k] = v
        model_cfg[k] = v
        train_cfg[k] = v

    for kv in args.override:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        for cast in (int, float):
            try:
                v_cast = cast(v)
                break
            except ValueError:
                continue
        else:
            if v.lower() in ("true", "false"):
                v_cast = v.lower() == "true"
            else:
                v_cast = v
        dataset_cfg[k] = v_cast
        model_cfg[k] = v_cast
        train_cfg[k] = v_cast

    log_dir = ensure_dir(repo_root / args.out_dir / dataset_name / model_name / f"seed_{args.seed}")
    set_seeds(args.seed)

    print(f"[run] building dataset: {dataset_name}")
    dataset_dict = data_registry.build(dataset_name, dataset_cfg)
    meta = dataset_dict["meta"]
    print(f"[run] meta: {meta}")

    print(f"[run] building model: {model_name}")
    model = model_registry.build(model_name, model_cfg, meta)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[run] model trainable params: {n_params}")

    save_yaml({
        "dataset_name": dataset_name,
        "model_name": model_name,
        "seed": args.seed,
        "dataset_cfg": dataset_cfg,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "n_params": int(n_params),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }, log_dir / "config.yaml")

    print(f"[run] training {model_name} on {dataset_name} (seed={args.seed})")
    summary = trainers.dispatch(dataset_dict, model, train_cfg, log_dir)
    print(f"[run] summary: {summary}")

if __name__ == "__main__":
    main()
