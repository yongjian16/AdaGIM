"""Monotone-MIGNN wrapper (Baker et al. 2023, ICML)."""
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register

_MIGNN_DIR = Path(__file__).resolve().parents[1] / "third_party" / "MIGNN" / "agg"
if str(_MIGNN_DIR) not in sys.path:
    sys.path.append(str(_MIGNN_DIR))

def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_conv_mod = _load("_mignn_conv", _MIGNN_DIR / "_conv.py")
_deq_mod = _load("_mignn_deq", _MIGNN_DIR / "_deq.py")

MonotoneImplicitGraph = _conv_mod.MonotoneImplicitGraph
MonotoneLinear = _conv_mod.MonotoneLinear
CayleyLinear = _conv_mod.CayleyLinear
MIGNN_ReLU = _conv_mod.ReLU
MIGNN_TanH = _conv_mod.TanH
PeacemanRachford = _deq_mod.PeacemanRachford
ForwardBackward = _deq_mod.ForwardBackward

class MonotoneMIGNNPyG(nn.Module):

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 max_iter: int = 50, tol: float = 1e-5,
                 inv_method: str = "neumann-3",
                 nonlin: str = "relu",
                 graph_pool: str = "mean",
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,         # pre-encoder dropout (Cora repro)
                 post_ig_act: bool = False,           # ReLU after IG step (Cora repro)
                 head_bias: bool = True,              # False to match Cora repro
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
            in_dim = hidden
        elif x_is_indices:
            self.input_emb = nn.Embedding(num_atom_types, hidden)
            in_dim = hidden  # post-embedding feature dim equals hidden
        else:
            self.input_emb = nn.Linear(in_dim, hidden)
            in_dim = hidden

        self.in_dim = in_dim
        self.hidden = hidden
        self.max_iter = max_iter
        self.tol = tol
        self.inv_method = inv_method
        self.nonlin = nonlin
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.post_ig_act = post_ig_act
        self.graph_pool = graph_pool

        self._template_lins = nn.ModuleList([
            CayleyLinear(nfeat=hidden, nhid=hidden, num_node=1,
                         invMethod=inv_method, kappa=0.9)
            for _ in range(num_layers)
        ])
        for idx, lin in enumerate(self._template_lins):
            self.register_buffer(f"_cl_I_{idx}", lin.I)
            self.register_buffer(f"_cl_D_{idx}", lin.D)
            if not isinstance(lin.mu, nn.Parameter):
                self.register_buffer(f"_cl_mu_{idx}", lin.mu)
        self._nonlin_module = MIGNN_ReLU() if nonlin == "relu" else MIGNN_TanH()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden) if self.row_norm else nn.Identity()
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(hidden, out_dim, bias=head_bias)

    def _build_solver(self, adj_sparse, device, layer_idx: int = 0):
        lin = self._template_lins[layer_idx]
        lin.I = getattr(self, f"_cl_I_{layer_idx}")
        lin.D = getattr(self, f"_cl_D_{layer_idx}")
        if hasattr(self, f"_cl_mu_{layer_idx}"):
            lin.mu = getattr(self, f"_cl_mu_{layer_idx}")
        lin.device = device

        lin.set_adj(adj_sparse, sp_adj=None)
        solver = PeacemanRachford(
            lin_module=lin,
            nonlin_module=self._nonlin_module,
            alpha=1.0, tol=self.tol, max_iter=self.max_iter,
            verbose=False, record=False, store=False,
        )
        return MonotoneImplicitGraph(lin, self._nonlin_module, solver)

    def forward_pyg(self, batch) -> torch.Tensor:
        from torch_geometric.utils import scatter

        x = batch.x
        if self.use_ogb_encoders:
            X = self.input_emb(x)
        else:
            if not self.x_is_indices and self.input_dropout > 0:
                x = F.dropout(x.float(), p=self.input_dropout, training=self.training)
            if self.x_is_indices:
                x = x.squeeze(-1) if x.dim() == 2 else x
                X = self.input_emb(x.long())
            else:
                X = self.input_emb(x.float())

        N = X.size(0)
        device = X.device

        from torch_geometric.utils import add_self_loops, degree
        ei, _ = add_self_loops(batch.edge_index, num_nodes=N)
        row, col = ei
        deg = degree(row, N, dtype=X.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        ew = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(torch.stack([row, col], dim=0), ew, (N, N), device=device).coalesce()

        Z = X
        for layer_idx in range(self.num_layers):
            ig = self._build_solver(adj, device, layer_idx)
            Zt = ig(Z.t())
            Z = Zt.t()  # [N, hidden]
            if self.post_ig_act:
                Z = F.relu(Z)
            Z = self.layer_norms[layer_idx](Z)
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

@register("monotone_mignn")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    is_node_task = (dataset_meta.get("task_type") in
                    {"node_classification", "node_classification_masked",
                     "node_classification_parity"})
    from ._ogb_encoder import is_ogb_format
    use_ogb = bool(cfg.get("use_ogb_encoders", is_ogb_format(dataset_meta)))
    return MonotoneMIGNNPyG(
        in_dim=dataset_meta["in_dim"],
        hidden=int(cfg.get("hidden", 64)),
        out_dim=dataset_meta["out_dim"],
        max_iter=int(cfg.get("max_iter", 50)),
        tol=float(cfg.get("tol", 1e-5)),
        inv_method=str(cfg.get("inv_method", "neumann-3")),
        nonlin=str(cfg.get("nonlin", "relu")),
        graph_pool="node" if is_node_task else str(cfg.get("graph_pool", "mean")),
        dropout=float(cfg.get("dropout", 0.0)),
        input_dropout=float(cfg.get("input_dropout", 0.0)),
        post_ig_act=bool(cfg.get("post_ig_act", False)),
        head_bias=bool(cfg.get("head_bias", True)),
        x_is_indices=dataset_meta.get("x_is_indices", False),
        use_ogb_encoders=use_ogb,
        num_layers=int(cfg.get("num_layers", 1)),
        row_norm=bool(cfg.get("row_norm", False)),
    )
