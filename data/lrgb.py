"""Long-Range Graph Benchmark (LRGB) datasets via PyG."""
from typing import Any, Dict

import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader

from . import register
from ._paths import resolve as _resolve_root

def _lrgb_meta(name: str, train_ds) -> Dict[str, Any]:
    sample = train_ds[0]
    in_dim = sample.x.size(-1) if sample.x.dim() > 1 else int(sample.x.max().item()) + 1
    edge_attr_dim = sample.edge_attr.size(-1) if sample.edge_attr is not None and sample.edge_attr.dim() > 1 else 0
    if name == "Peptides-func":
        out_dim = sample.y.size(-1) if sample.y.dim() > 1 else int(sample.y.numel())
        return {"in_dim": in_dim, "edge_attr_dim": edge_attr_dim, "out_dim": out_dim,
                "task_type": "graph_classification_multilabel", "metric": "ap"}
    if name == "Peptides-struct":
        out_dim = sample.y.size(-1) if sample.y.dim() > 1 else int(sample.y.numel())
        return {"in_dim": in_dim, "edge_attr_dim": edge_attr_dim, "out_dim": out_dim,
                "task_type": "graph_regression", "metric": "mae"}
    if name == "PascalVOC-SP":
        out_dim = int(train_ds.num_classes) if hasattr(train_ds, "num_classes") else 21
        return {"in_dim": in_dim, "edge_attr_dim": edge_attr_dim, "out_dim": out_dim,
                "task_type": "node_classification", "metric": "macro_f1"}
    raise ValueError(f"Unknown LRGB dataset {name}")

def _build_lrgb(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    root = _resolve_root("LRGB", cfg)
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 2))

    train_ds = LRGBDataset(root=root, name=name, split="train")
    val_ds = LRGBDataset(root=root, name=name, split="val")
    test_ds = LRGBDataset(root=root, name=name, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        "mode": "pyg",
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "meta": _lrgb_meta(name, train_ds),
    }

@register("peptides_func")
def build_peptides_func(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _build_lrgb("Peptides-func", cfg)

@register("peptides_struct")
def build_peptides_struct(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _build_lrgb("Peptides-struct", cfg)
