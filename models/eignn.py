"""EIGNN wrapper."""
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._existing_paths import ParityEIGNN
from . import mgnni as _mgnni_wrapper  # triggers _layers_mod load
from . import register

import sys
_layers = sys.modules.get("_mgnni_layers")
EIGNN_core = _layers.EIGNN_scale_w_iter if _layers is not None else None

class EIGNNSynthetic(nn.Module):
    def __init__(self, adj, sp_adj, in_dim: int, hidden: int, out_dim: int, **kwargs):
        super().__init__()
        self.core = ParityEIGNN(adj=adj, sp_adj=sp_adj, in_dim=in_dim, hidden=hidden, out_dim=out_dim, **kwargs)

    def set_graph(self, A, A_sp=None):
        self.core.set_graph(A, A_sp)

    def forward_synthetic(self, X: torch.Tensor, A: torch.Tensor, A_sp=None) -> torch.Tensor:
        if A is not None:
            self.core.set_graph(A, A_sp)
        return self.core(X)

    def encode_synthetic(self, X: torch.Tensor, A: torch.Tensor, A_sp=None) -> torch.Tensor:
        if A is not None:
            self.core.set_graph(A, A_sp)
        return self.core.encode(X)

class EIGNNPyG(nn.Module):

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 threshold: float = 1e-4, max_iter: int = 50, gamma: float = 0.99,
                 dropout: float = 0.0, graph_pool: str = "mean",
                 x_is_indices: bool = False, num_atom_types: int = 28,
                 use_ogb_encoders: bool = False, num_layers: int = 1,
                 row_norm: bool = False):
        super().__init__()
        if EIGNN_core is None:
            raise RuntimeError(
                "EIGNN_scale_w_iter could not be loaded from MGNNI. Ensure the MGNNI repo is at "
                "third_party/MGNNI and `from . import mgnni` succeeded."
            )
        self.x_is_indices = x_is_indices
        self.use_ogb_encoders = use_ogb_encoders
        self.num_layers = num_layers
        self.row_norm = bool(row_norm)
        if use_ogb_encoders:
            from ._ogb_encoder import OGBAtomEncoder
            self.input_emb = OGBAtomEncoder(emb_dim=hidden)
        elif x_is_indices:
            self.input_emb = nn.Embedding(num_atom_types, hidden)
        else:
            self.input_emb = nn.Linear(in_dim, hidden)
        self.cores = nn.ModuleList([
            EIGNN_core(m=hidden, k=1, threshold=threshold, max_iter=max_iter, gamma=gamma)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden) if self.row_norm else nn.Identity()
            for _ in range(num_layers)
        ])
        self.dropout = dropout
        self.graph_pool = graph_pool
        self.head = nn.Linear(hidden, out_dim)

    def forward_pyg(self, batch) -> torch.Tensor:
        from torch_geometric.utils import scatter, add_self_loops, degree

        x = batch.x
        if self.use_ogb_encoders:
            X = self.input_emb(x)
        elif self.x_is_indices:
            x = x.squeeze(-1) if x.dim() == 2 else x
            X = self.input_emb(x.long())
        else:
            X = self.input_emb(x.float())

        N = X.size(0)
        ei, _ = add_self_loops(batch.edge_index, num_nodes=N)
        row, col = ei
        deg = degree(row, N, dtype=X.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        ew = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(torch.stack([row, col], dim=0), ew, (N, N), device=X.device).coalesce()

        Z = X
        for core, ln in zip(self.cores, self.layer_norms):
            Zt = core(Z.t(), adj)
            Z = Zt.t()
            Z = ln(Z)
            Z = F.dropout(Z, p=self.dropout, training=self.training)

        if self.graph_pool == "node":
            return self.head(Z)
        if self.graph_pool == "mean":
            g = scatter(Z, batch.batch, dim=0, reduce="mean")
        elif self.graph_pool == "sum":
            g = scatter(Z, batch.batch, dim=0, reduce="sum")
        else:
            raise ValueError(self.graph_pool)
        return self.head(g)

@register("eignn")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    if dataset_meta.get("task_type") == "node_classification_parity":
        hidden = int(cfg.get("hidden", 100))
        import scipy.sparse as _sp_lib
        if bool(cfg.get("bidirectional", False)):
            sp_adj_init = _sp_lib.coo_matrix([[0.0, 1.0], [1.0, 0.0]])  # symmetric
        else:
            sp_adj_init = _sp_lib.coo_matrix([[0.0, 1.0], [0.0, 0.0]])  # asymmetric chain
        adj_init = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long),
            torch.ones(3),
            size=(2, 2),
        ).coalesce() if not bool(cfg.get("bidirectional", False)) else \
        torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            torch.ones(2),
            size=(2, 2),
        ).coalesce()
        return EIGNNSynthetic(
            adj=adj_init, sp_adj=sp_adj_init,
            in_dim=dataset_meta["in_dim"], hidden=hidden, out_dim=dataset_meta["out_dim"],
            threshold=float(cfg.get("threshold", 1e-4)),
            max_iter=int(cfg.get("max_iter", 300)),
            gamma=float(cfg.get("gamma", 0.99)),
            g_type=str(cfg.get("g_type", "psd")),
        )
    is_node_task = (dataset_meta.get("task_type") in {"node_classification", "node_classification_masked"})
    from ._ogb_encoder import is_ogb_format
    use_ogb = bool(cfg.get("use_ogb_encoders", is_ogb_format(dataset_meta)))
    return EIGNNPyG(
        in_dim=dataset_meta["in_dim"],
        hidden=int(cfg.get("hidden", 64)),
        out_dim=dataset_meta["out_dim"],
        threshold=float(cfg.get("threshold", 1e-4)),
        max_iter=int(cfg.get("max_iter", 50)),
        gamma=float(cfg.get("gamma", 0.99)),
        dropout=float(cfg.get("dropout", 0.0)),
        graph_pool="node" if is_node_task else str(cfg.get("graph_pool", "mean")),
        x_is_indices=dataset_meta.get("x_is_indices", False),
        use_ogb_encoders=use_ogb,
        num_layers=int(cfg.get("num_layers", 1)),
        row_norm=bool(cfg.get("row_norm", False)),
    )
