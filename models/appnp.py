"""APPNP baseline (Klicpera et al. 2019) — PyG built-in APPNP layer."""
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP as PyGAPPNP

from . import register

class APPNPPyG(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 K: int = 10, alpha: float = 0.1, dropout: float = 0.1,
                 graph_pool: str = "mean",
                 x_is_indices: bool = False, num_atom_types: int = 28):
        super().__init__()
        self.x_is_indices = x_is_indices
        if x_is_indices:
            self.input_emb = nn.Embedding(num_atom_types, hidden)
        else:
            self.input_emb = nn.Linear(in_dim, hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.prop = PyGAPPNP(K=K, alpha=alpha)
        self.dropout = dropout
        self.graph_pool = graph_pool
        self.head = nn.Linear(hidden, out_dim)

    def forward_pyg(self, batch) -> torch.Tensor:
        from torch_geometric.utils import scatter

        x = batch.x
        if self.x_is_indices:
            x = x.squeeze(-1) if x.dim() == 2 else x
            h = self.input_emb(x.long())
        else:
            h = self.input_emb(x.float())
        h = F.dropout(F.relu(self.lin1(h)), p=self.dropout, training=self.training)
        h = self.lin2(h)
        h = self.prop(h, batch.edge_index)

        if self.graph_pool == "node":
            return self.head(h)
        if self.graph_pool == "mean":
            g = scatter(h, batch.batch, dim=0, reduce="mean")
        elif self.graph_pool == "sum":
            g = scatter(h, batch.batch, dim=0, reduce="sum")
        else:
            raise ValueError(self.graph_pool)
        return self.head(g)

@register("appnp")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    is_node_task = (dataset_meta.get("task_type") in
                    {"node_classification", "node_classification_masked",
                     "node_classification_parity"})
    return APPNPPyG(
        in_dim=dataset_meta["in_dim"],
        hidden=int(cfg.get("hidden", 128)),
        out_dim=dataset_meta["out_dim"],
        K=int(cfg.get("K", 10)),
        alpha=float(cfg.get("alpha", 0.1)),
        dropout=float(cfg.get("dropout", 0.1)),
        graph_pool="node" if is_node_task else str(cfg.get("graph_pool", "mean")),
        x_is_indices=dataset_meta.get("x_is_indices", False),
    )
