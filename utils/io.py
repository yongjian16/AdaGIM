import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import yaml

PathLike = Union[str, Path]

_DIAG_COLS = ("L_Psi", "L_g_eff", "expected_halt_time", "opt_out_rate",
              "halt_entropy", "D_hat", "R_hat", "avg_iters", "picard_residual",
              "lr", "traj_norm_t1", "traj_norm_tT", "traj_contraction")

def load_yaml(path: PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(obj: Dict[str, Any], path: PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_metrics_csv(rows: Iterable[Dict[str, Any]], path: PathLike) -> None:
    rows = list(rows)
    if not rows:
        return
    ensure_dir(Path(path).parent)
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k); seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def plot_metrics_png(rows: Iterable[Dict[str, Any]], path: PathLike) -> None:
    rows = list(rows)
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(r["epoch"]) for r in rows]
    sample = rows[-1]

    panels = [("train_loss", "tab:blue")]
    if "val_metric" in sample:
        panels.append(("val_metric", "tab:orange"))
    if "test_metric_at_best_val" in sample:
        panels.append(("test_metric_at_best_val", "tab:green"))
    elif "test_metric" in sample:
        panels.append(("test_metric", "tab:green"))
    diag_palette = ["tab:red", "tab:purple", "tab:brown", "tab:pink",
                    "tab:olive", "tab:cyan", "tab:gray", "k", "tab:orange",
                    "tab:green", "darkblue", "darkred", "teal"]
    for col, color in zip(_DIAG_COLS, diag_palette):
        if col in sample:
            panels.append((col, color))

    n = len(panels)
    cols = min(n, 4)
    nrows = (n + cols - 1) // cols
    ensure_dir(Path(path).parent)
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 4 * nrows), squeeze=False)
    for i, (col, color) in enumerate(panels):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        ys = [row.get(col) for row in rows]
        xy = [(e, y) for e, y in zip(epochs, ys) if y is not None and y != ""]
        if not xy:
            ax.axis("off"); continue
        xs, ys = zip(*xy)
        ax.plot(xs, ys, "-o", color=color, ms=3, lw=1)
        ax.set_xlabel("epoch"); ax.set_ylabel(col); ax.set_title(col); ax.grid(alpha=0.3)
        if col == "L_Psi":
            ax.set_ylim(0, 1.05)
            ax.axhline(1.0, color="gray", linestyle="--", lw=0.7)
    for i in range(n, nrows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis("off")
    fig.suptitle(f"{Path(path).parent.name}  ({len(rows)} epochs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
