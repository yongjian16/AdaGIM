"""Aggregate per-seed results from runs_multiseed/ into a mean ± std table."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO = Path(__file__).resolve().parent.parent

DATASETS = ("zinc", "peptides_func", "peptides_struct", "ogbg_molhiv")
MODELS = (
    "ignn", "eignn", "mgnni", "monotone_mignn", "gind",
    "ignn_finite_loop", "adagim",
    "gcn", "gcnii", "appnp",
)

def _seed_test(model: str, dataset: str, seed: int) -> float:
    p = (REPO / "runs_multiseed" / f"{model}_{dataset}"
         / dataset / model / f"seed_{seed}" / "summary.yaml")
    if not p.exists():
        return float("nan")
    try:
        s = yaml.safe_load(p.read_text())
        return float(s.get("test_metric_at_best_val", float("nan")))
    except Exception:  # noqa: BLE001
        return float("nan")

def _mean_std(xs: List[float]) -> Tuple[float, float, int]:
    valid = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
    n = len(valid)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(valid) / n
    if n == 1:
        return mean, float("nan"), 1
    var = sum((x - mean) ** 2 for x in valid) / (n - 1)
    return mean, math.sqrt(var), n

def main(seeds: Tuple[int, ...] = (0, 1, 2)) -> None:
    print(f"{'dataset':<18}{'model':<20}" + "".join(f"seed{s:<5}" for s in seeds)
          + f"{'mean':<10}{'std':<10}{'#':<4}")
    print("-" * (38 + 10 * len(seeds) + 24))
    for ds in DATASETS:
        for m in MODELS:
            vs = [_seed_test(m, ds, s) for s in seeds]
            mean, std, n = _mean_std(vs)
            def fmt(v: float) -> str:
                return f"{v:.4f}" if not (math.isnan(v) or math.isinf(v)) else "—"
            cells = "".join(f"{fmt(v):<10}" for v in vs)
            print(f"{ds:<18}{m:<20}{cells}{fmt(mean):<10}{fmt(std):<10}{n:<4}")
        print()

if __name__ == "__main__":
    main()
