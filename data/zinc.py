"""ZINC 12k subset via PyG."""
from typing import Any, Dict

import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

from . import register
from ._paths import resolve as _resolve_root

@register("zinc")
def build(cfg: Dict[str, Any]) -> Dict[str, Any]:
    root = _resolve_root("ZINC", cfg)
    subset = bool(cfg.get("subset", True))  # 12K subset by default
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 2))

    train_ds = ZINC(root=root, subset=subset, split="train")
    val_ds = ZINC(root=root, subset=subset, split="val")
    test_ds = ZINC(root=root, subset=subset, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample = train_ds[0]
    x_is_indices = (sample.x.dtype in (torch.long, torch.int)) and sample.x.size(-1) == 1
    if x_is_indices:
        max_idx = 0
        for i in range(min(len(train_ds), 256)):
            max_idx = max(max_idx, int(train_ds[i].x.max().item()))
        in_dim = max_idx + 1
    else:
        in_dim = sample.x.size(-1)

    edge_attr_is_indices = sample.edge_attr is not None and sample.edge_attr.dim() == 1 and sample.edge_attr.dtype in (torch.long, torch.int)
    if edge_attr_is_indices:
        max_e = 0
        for i in range(min(len(train_ds), 256)):
            if train_ds[i].edge_attr is not None and train_ds[i].edge_attr.numel() > 0:
                max_e = max(max_e, int(train_ds[i].edge_attr.max().item()))
        edge_attr_dim = max_e + 1
    else:
        edge_attr_dim = sample.edge_attr.size(-1) if sample.edge_attr is not None else 0

    return {
        "mode": "pyg",
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "meta": {
            "in_dim": in_dim,
            "edge_attr_dim": edge_attr_dim,
            "out_dim": 1,
            "task_type": "graph_regression",
            "metric": "mae",
            "x_is_indices": x_is_indices,
            "edge_attr_is_indices": edge_attr_is_indices,
        },
    }
