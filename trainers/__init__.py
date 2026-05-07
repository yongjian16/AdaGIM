from .base import build_loss
from .synthetic import train_synthetic
from .pyg import train_pyg
from .masked_node import train_masked_node

__all__ = ["build_loss", "train_synthetic", "train_pyg", "train_masked_node"]

def dispatch(dataset_dict, model, cfg, log_dir):
    mode = dataset_dict["mode"]
    task_type = dataset_dict.get("meta", {}).get("task_type", "")
    if mode == "synthetic":
        return train_synthetic(dataset_dict, model, cfg, log_dir)
    if mode == "pyg":
        if task_type == "node_classification_masked":
            return train_masked_node(dataset_dict, model, cfg, log_dir)
        return train_pyg(dataset_dict, model, cfg, log_dir)
    raise ValueError(f"Unknown dataset mode {mode!r}")
