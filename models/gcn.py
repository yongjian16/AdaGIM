"""GCN baseline. Synthetic mode wraps ParityGCN; PyG mode uses PyG GCNConv directly."""
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from ._existing_paths import ParityGCN
from . import register

class GCNSynthetic(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, **kwargs):
        super().__init__()
        self.core = ParityGCN(in_dim=in_dim, hidden=hidden, out_dim=out_dim, **kwargs)

    def set_graph(self, edge_index, edge_weight=None):
        self.core.set_graph(edge_index, edge_weight)

    def forward_synthetic(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return self.core(X)

class GCNPyG(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 4,
                 dropout: float = 0.1, graph_pool: str = "mean",
                 x_is_indices: bool = False, num_atom_types: int = 28,
                 virtual_node: bool = False):
        super().__init__()
        if x_is_indices:
            self.input_emb = nn.Embedding(num_atom_types, hidden)
        else:
            self.input_emb = nn.Linear(in_dim, hidden)
        self.x_is_indices = x_is_indices

        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)]) if virtual_node else None
        self.head = nn.Linear(hidden, out_dim)
        self.dropout = dropout
        self.graph_pool = graph_pool

        self.virtual_node = virtual_node
        if virtual_node:
            self.vn_emb = nn.Embedding(1, hidden)
            nn.init.constant_(self.vn_emb.weight, 0.0)
            self.vn_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden, 2 * hidden), nn.BatchNorm1d(2 * hidden), nn.ReLU(),
                    nn.Linear(2 * hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                ) for _ in range(num_layers - 1)
            ])

    def forward_pyg(self, batch) -> torch.Tensor:
        from torch_geometric.utils import scatter

        x = batch.x
        if self.x_is_indices:
            x = x.squeeze(-1) if x.dim() == 2 else x
            h = self.input_emb(x.long())
        else:
            h = self.input_emb(x.float())

        if self.virtual_node:
            n_graphs = int(batch.batch.max().item()) + 1
            vn = self.vn_emb(torch.zeros(n_graphs, dtype=torch.long, device=h.device))
        else:
            vn = None

        for i, conv in enumerate(self.convs):
            if vn is not None:
                h = h + vn[batch.batch]
            h = conv(h, batch.edge_index)
            if self.bns is not None:
                h = self.bns[i](h)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if vn is not None:
                    vn_input = scatter(h, batch.batch, dim=0, reduce="sum") + vn
                    vn = F.dropout(self.vn_mlps[i](vn_input), p=self.dropout, training=self.training)

        if self.graph_pool == "node":
            return self.head(h)
        if self.graph_pool == "mean":
            g = scatter(h, batch.batch, dim=0, reduce="mean")
        elif self.graph_pool == "sum":
            g = scatter(h, batch.batch, dim=0, reduce="sum")
        else:
            raise ValueError(self.graph_pool)
        return self.head(g)

@register("gcn")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    is_node_task = (dataset_meta.get("task_type") in {
        "node_classification", "node_classification_parity", "node_classification_masked"
    })
    if dataset_meta.get("task_type") == "node_classification_parity":
        return GCNSynthetic(
            in_dim=dataset_meta["in_dim"],
            hidden=int(cfg.get("hidden", 100)),
            out_dim=dataset_meta["out_dim"],
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.0)),
            act=str(cfg.get("act", "relu")),
            activate_last=bool(cfg.get("activate_last", True)),
        )
    return GCNPyG(
        in_dim=dataset_meta["in_dim"],
        hidden=int(cfg.get("hidden", 128)),
        out_dim=dataset_meta["out_dim"],
        num_layers=int(cfg.get("num_layers", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        graph_pool="node" if is_node_task else str(cfg.get("graph_pool", "mean")),
        x_is_indices=dataset_meta.get("x_is_indices", False),
        virtual_node=bool(cfg.get("virtual_node", False)),
    )
