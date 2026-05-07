"""Pluggable metrics keyed by name."""
from typing import Any, Dict

import torch
import numpy as np

def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def parity_acc(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    target = y_true.argmax(dim=-1) if y_true.dim() > 1 else y_true
    return float((pred == target).float().mean().item())

def ap(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    from sklearn.metrics import average_precision_score
    y = _to_numpy(y_true)
    p = _to_numpy(torch.sigmoid(logits))
    aps = []
    for k in range(y.shape[-1] if y.ndim > 1 else 1):
        yk = y[:, k] if y.ndim > 1 else y
        pk = p[:, k] if p.ndim > 1 else p
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        aps.append(average_precision_score(yk, pk))
    return float(np.mean(aps)) if aps else float("nan")

def mae(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    if logits.dim() != y_true.dim():
        if logits.dim() > y_true.dim():
            logits = logits.squeeze(-1)
        else:
            y_true = y_true.squeeze(-1)
    return float((logits - y_true).abs().mean().item())

def roc_auc(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(_to_numpy(y_true), _to_numpy(torch.sigmoid(logits))))

def macro_f1(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    from sklearn.metrics import f1_score
    pred = logits.argmax(dim=-1)
    return float(f1_score(_to_numpy(y_true), _to_numpy(pred), average="macro"))

def node_acc(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y_true).float().mean().item())

_METRICS: Dict[str, Any] = {
    "parity_acc": parity_acc,
    "ap": ap,
    "mae": mae,
    "roc_auc": roc_auc,
    "macro_f1": macro_f1,
    "node_acc": node_acc,
}

def get(name: str):
    if name not in _METRICS:
        raise KeyError(f"Unknown metric {name!r}. Available: {sorted(_METRICS)}")
    return _METRICS[name]

def higher_is_better(name: str) -> bool:
    return name in {"parity_acc", "ap", "roc_auc", "macro_f1", "node_acc"}
