"""MGNNI wrapper. Vendored model lives at third_party/MGNNI/graphclassification/."""
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register

_MGNNI_DIR = Path(__file__).resolve().parents[1] / "third_party" / "MGNNI" / "graphclassification"

def _load_with_alias(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_saved = {}
_NEED_ALIAS = ("normalization", "utils", "functions", "solvers")
for _name in _NEED_ALIAS:
    if _name in sys.modules:
        _saved[_name] = sys.modules[_name]

try:
    _load_with_alias("normalization", _MGNNI_DIR / "normalization.py")
    _load_with_alias("utils", _MGNNI_DIR / "utils.py")
    _load_with_alias("functions", _MGNNI_DIR / "functions.py")
    _load_with_alias("solvers", _MGNNI_DIR / "solvers.py")
    _layers_mod = _load_with_alias("_mgnni_layers", _MGNNI_DIR / "layers.py")
    MGNNI_m_iter = _layers_mod.MGNNI_m_iter
finally:
    for _name in _NEED_ALIAS:
        if _name in _saved:
            sys.modules[_name] = _saved[_name]
        else:
            sys.modules.pop(_name, None)

def _build_sparse_norm_adj(edge_index, num_nodes, device, dtype) -> torch.Tensor:
    from torch_geometric.utils import add_self_loops, degree

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    edge_w = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    indices = torch.stack([row, col], dim=0)
    return torch.sparse_coo_tensor(indices, edge_w, (num_nodes, num_nodes), device=device).coalesce()

class MGNNIPyG(nn.Module):

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 ks: List[int] = (1, 2),
                 threshold: float = 1e-4, max_iter: int = 50, gamma: float = 0.99,
                 dropout: float = 0.0, graph_pool: str = "mean",
                 x_is_indices: bool = False, num_atom_types: int = 28,
                 use_ogb_encoders: bool = False, num_layers: int = 1,
                 row_norm: bool = False):
        super().__init__()
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

        self.layer_cores = nn.ModuleList([
            nn.ModuleList([
                MGNNI_m_iter(hidden, k=int(k), threshold=threshold, max_iter=max_iter, gamma=gamma)
                for k in ks
            ])
            for _ in range(num_layers)
        ])
        self.layer_scale_attn = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden) if self.row_norm else nn.Identity()
            for _ in range(num_layers)
        ])
        self.dropout = dropout
        self.graph_pool = graph_pool
        self.head = nn.Linear(hidden, out_dim)

    def forward_pyg(self, batch) -> torch.Tensor:
        from torch_geometric.utils import scatter

        x = batch.x
        if self.use_ogb_encoders:
            X = self.input_emb(x)
        elif self.x_is_indices:
            x = x.squeeze(-1) if x.dim() == 2 else x
            X = self.input_emb(x.long())
        else:
            X = self.input_emb(x.float())

        N = X.size(0)
        adj = _build_sparse_norm_adj(batch.edge_index, num_nodes=N, device=X.device, dtype=X.dtype)

        h = X
        for layer_idx, cores in enumerate(self.layer_cores):
            Xt = h.t()  # MGNNI_m_iter expects [m, n]
            outs = []
            for core in cores:
                Zt = core(Xt, adj)
                outs.append(Zt.t())
            stacked = torch.stack(outs, dim=1)  # [N, num_scales, hidden]
            attn = torch.softmax(self.layer_scale_attn[layer_idx](stacked).squeeze(-1), dim=1)
            h = (stacked * attn.unsqueeze(-1)).sum(dim=1)  # [N, hidden]
            h = self.layer_norms[layer_idx](h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        if self.graph_pool == "node":
            return self.head(h)
        if self.graph_pool == "mean":
            g = scatter(h, batch.batch, dim=0, reduce="mean")
        elif self.graph_pool == "sum":
            g = scatter(h, batch.batch, dim=0, reduce="sum")
        else:
            raise ValueError(self.graph_pool)
        return self.head(g)

@register("mgnni")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    is_node_task = (dataset_meta.get("task_type") in
                    {"node_classification", "node_classification_masked",
                     "node_classification_parity"})
    from ._ogb_encoder import is_ogb_format
    use_ogb = bool(cfg.get("use_ogb_encoders", is_ogb_format(dataset_meta)))
    return MGNNIPyG(
        in_dim=dataset_meta["in_dim"],
        hidden=int(cfg.get("hidden", 64)),
        out_dim=dataset_meta["out_dim"],
        ks=tuple(cfg.get("ks", [1, 2])),
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
