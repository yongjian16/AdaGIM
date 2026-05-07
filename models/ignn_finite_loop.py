"""IGNN-finite (loop variant): unroll the IGNN update for a fixed number of steps,"""
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register

def _matmul_A(A: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    if A.is_sparse:
        return torch.sparse.mm(A.coalesce(), M)
    return A @ M

class IGNNFiniteLoopSynthetic(nn.Module):

    def __init__(self, in_dim: int, hidden: int, out_dim: int, T: int = 4,
                 activation: str = "relu", init_gain: float = 0.99):
        super().__init__()
        self.T = T
        self.V = nn.Linear(in_dim, hidden, bias=True)
        self.W = nn.Parameter(torch.empty(hidden, hidden))
        self.B = nn.Linear(hidden, out_dim, bias=True)
        self.act = nn.ReLU() if activation == "relu" else nn.Tanh()
        nn.init.xavier_uniform_(self.V.weight, gain=init_gain)
        nn.init.xavier_uniform_(self.W, gain=init_gain)
        nn.init.xavier_uniform_(self.B.weight, gain=init_gain)
        nn.init.zeros_(self.V.bias)
        nn.init.zeros_(self.B.bias)

    def forward_synthetic(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        ux = self.V(X)
        h = torch.zeros_like(ux)
        for _ in range(self.T):
            h = self.act(_matmul_A(A, h) @ self.W + ux)
        return self.B(h)

class IGNNFiniteLoopPyG(nn.Module):

    def __init__(self, in_dim: int, hidden: int, out_dim: int, T: int = 4,
                 dropout: float = 0.1, activation: str = "relu", graph_pool: str = "mean",
                 x_is_indices: bool = False, edge_attr_is_indices: bool = False,
                 edge_attr_dim: int = 0, num_atom_types: int = 28, num_bond_types: int = 4,
                 use_ogb_encoders: bool = False, row_norm: bool = False):
        super().__init__()
        self.T = T
        self.x_is_indices = x_is_indices
        self.edge_attr_is_indices = edge_attr_is_indices
        self.use_ogb_encoders = use_ogb_encoders
        self.row_norm = bool(row_norm)

        if use_ogb_encoders:
            from ._ogb_encoder import OGBAtomEncoder, OGBBondEncoder
            self.input_emb = OGBAtomEncoder(emb_dim=hidden)
            self.edge_emb = OGBBondEncoder(emb_dim=hidden)
        elif x_is_indices:
            self.input_emb = nn.Embedding(num_atom_types, hidden)
            if edge_attr_is_indices:
                self.edge_emb = nn.Embedding(edge_attr_dim or num_bond_types, hidden)
            elif edge_attr_dim > 0:
                self.edge_emb = nn.Linear(edge_attr_dim, hidden)
            else:
                self.edge_emb = None
        else:
            self.input_emb = nn.Linear(in_dim, hidden)
            if edge_attr_dim > 0 or edge_attr_is_indices:
                if edge_attr_is_indices:
                    self.edge_emb = nn.Embedding(edge_attr_dim or num_bond_types, hidden)
                else:
                    self.edge_emb = nn.Linear(edge_attr_dim, hidden)
            else:
                self.edge_emb = None

        self.W = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.xavier_uniform_(self.W, gain=0.99)
        self.dropout = dropout
        self.act = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.layer_norm = nn.LayerNorm(hidden) if self.row_norm else nn.Identity()
        self.graph_pool = graph_pool
        self.head = nn.Linear(hidden, out_dim)

    def forward_pyg(self, batch) -> torch.Tensor:
        from torch_geometric.utils import scatter

        x = batch.x
        if self.use_ogb_encoders:
            ux = self.input_emb(x)
        elif self.x_is_indices:
            x = x.squeeze(-1) if x.dim() == 2 else x
            ux = self.input_emb(x.long())
        else:
            ux = self.input_emb(x.float())

        ea = getattr(batch, "edge_attr", None)
        if self.use_ogb_encoders and ea is not None:
            edge_w = torch.sigmoid(self.edge_emb(ea).mean(dim=-1))
        elif self.edge_emb is not None and ea is not None:
            if self.edge_attr_is_indices:
                e = ea.squeeze(-1) if ea.dim() == 2 else ea
                edge_w = torch.sigmoid(self.edge_emb(e.long()).mean(dim=-1))
            else:
                edge_w = torch.sigmoid(self.edge_emb(ea.float()).mean(dim=-1))
        else:
            edge_w = None

        src, dst = batch.edge_index[0], batch.edge_index[1]
        h = torch.zeros_like(ux)
        for _ in range(self.T):
            msg = h[src]
            if edge_w is not None:
                msg = msg * edge_w.unsqueeze(-1)
            agg = scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="mean")
            h = self.act(agg @ self.W + ux)
            h = self.layer_norm(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        if self.graph_pool == "node":
            return self.head(h)
        if self.graph_pool == "mean":
            g = scatter(h, batch.batch, dim=0, reduce="mean")
        elif self.graph_pool == "sum":
            g = scatter(h, batch.batch, dim=0, reduce="sum")
        elif self.graph_pool == "max":
            g = scatter(h, batch.batch, dim=0, reduce="max")
        else:
            raise ValueError(self.graph_pool)
        return self.head(g)

@register("ignn_finite_loop")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    if dataset_meta.get("task_type") == "node_classification_parity":
        return IGNNFiniteLoopSynthetic(
            in_dim=dataset_meta["in_dim"],
            hidden=int(cfg.get("hidden", 100)),
            out_dim=dataset_meta["out_dim"],
            T=int(cfg.get("T", 2)),
            activation=str(cfg.get("activation", "relu")),
            init_gain=float(cfg.get("init_gain", 0.99)),
        )
    is_node_task = (dataset_meta.get("task_type") in {"node_classification", "node_classification_masked"})
    from ._ogb_encoder import is_ogb_format
    use_ogb = bool(cfg.get("use_ogb_encoders", is_ogb_format(dataset_meta)))
    return IGNNFiniteLoopPyG(
        in_dim=dataset_meta["in_dim"],
        hidden=int(cfg.get("hidden", 128)),
        out_dim=dataset_meta["out_dim"],
        T=int(cfg.get("T", 4)),  # match GCN/GINE default num_layers=4
        dropout=float(cfg.get("dropout", 0.1)),
        activation=str(cfg.get("activation", "relu")),
        graph_pool="node" if is_node_task else str(cfg.get("graph_pool", "mean")),
        x_is_indices=dataset_meta.get("x_is_indices", False),
        edge_attr_is_indices=dataset_meta.get("edge_attr_is_indices", False),
        edge_attr_dim=dataset_meta.get("edge_attr_dim", 0),
        use_ogb_encoders=use_ogb,
        row_norm=bool(cfg.get("row_norm", False)),
    )
