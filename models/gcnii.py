"""GCNII baseline (Chen et al. 2020, "Simple and Deep GCN") — PyG built-in GCN2Conv."""
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv

from . import register

class GCNIIPyG(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 num_layers: int = 8, alpha: float = 0.1, theta: float = 0.5,
                 dropout: float = 0.1, graph_pool: str = "mean",
                 x_is_indices: bool = False, num_atom_types: int = 28):
        super().__init__()
        self.x_is_indices = x_is_indices
        if x_is_indices:
            self.input_emb = nn.Embedding(num_atom_types, hidden)
        else:
            self.input_emb = nn.Linear(in_dim, hidden)

        self.convs = nn.ModuleList([
            GCN2Conv(channels=hidden, alpha=alpha, theta=theta, layer=i + 1, shared_weights=True)
            for i in range(num_layers)
        ])
        self.dropout = dropout
        self.graph_pool = graph_pool
        self.head = nn.Linear(hidden, out_dim)

    def forward_pyg(self, batch) -> torch.Tensor:
        from torch_geometric.utils import scatter

        x = batch.x
        if self.x_is_indices:
            x = x.squeeze(-1) if x.dim() == 2 else x
            h0 = self.input_emb(x.long())
        else:
            h0 = self.input_emb(x.float())

        h = h0
        for conv in self.convs:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, h0, batch.edge_index)
            h = F.relu(h)

        if self.graph_pool == "node":
            return self.head(h)
        if self.graph_pool == "mean":
            g = scatter(h, batch.batch, dim=0, reduce="mean")
        elif self.graph_pool == "sum":
            g = scatter(h, batch.batch, dim=0, reduce="sum")
        else:
            raise ValueError(self.graph_pool)
        return self.head(g)

@register("gcnii")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    is_node_task = (dataset_meta.get("task_type") in
                    {"node_classification", "node_classification_masked",
                     "node_classification_parity"})
    return GCNIIPyG(
        in_dim=dataset_meta["in_dim"],
        hidden=int(cfg.get("hidden", 128)),
        out_dim=dataset_meta["out_dim"],
        num_layers=int(cfg.get("num_layers", 8)),
        alpha=float(cfg.get("alpha", 0.1)),
        theta=float(cfg.get("theta", 0.5)),
        dropout=float(cfg.get("dropout", 0.1)),
        graph_pool="node" if is_node_task else str(cfg.get("graph_pool", "mean")),
        x_is_indices=dataset_meta.get("x_is_indices", False),
    )
