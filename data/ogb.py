"""OGB Graph-level molecular datasets."""
from typing import Any, Dict

import torch
from torch_geometric.loader import DataLoader

try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except Exception:
    pass

from ogb.graphproppred import PygGraphPropPredDataset

from . import register
from ._paths import resolve as _resolve_root

@register("ogbg_molhiv")
def build_molhiv(cfg: Dict[str, Any]) -> Dict[str, Any]:
    root = _resolve_root("OGB", cfg)
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 2))

    ds = PygGraphPropPredDataset(name="ogbg-molhiv", root=root)
    split_idx = ds.get_idx_split()

    train_loader = DataLoader(ds[split_idx["train"]], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ds[split_idx["valid"]], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ds[split_idx["test"]], batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample = ds[0]
    in_dim = sample.x.size(-1)
    edge_attr_dim = sample.edge_attr.size(-1) if sample.edge_attr is not None else 0

    return {
        "mode": "pyg",
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "meta": {
            "in_dim": in_dim,
            "edge_attr_dim": edge_attr_dim,
            "out_dim": ds.num_tasks,  # binary classification, use BCE
            "task_type": "graph_classification_binary",
            "metric": "roc_auc",
            "evaluator": ds.eval if hasattr(ds, "eval") else None,
        },
    }
