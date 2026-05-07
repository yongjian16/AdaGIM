"""IGNN wrapper using the OFFICIAL `ImplicitGraph` layer from Gu et al. 2020."""
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._existing_paths import ParityIGNN, ImplicitGraph
from . import register

class IGNNSynthetic(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, **kwargs):
        super().__init__()
        self.core = ParityIGNN(in_dim=in_dim, hidden=hidden, out_dim=out_dim, **kwargs)

    def forward_synthetic(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return self.core(X, A)

    def encode_synthetic(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return self.core.encode(X, A)

class IGNNPyG(nn.Module):

    def __init__(self, in_dim: int, hidden: int, out_dim: int, edge_attr_dim: int = 0,
                 kappa: float = 0.99, fw_mitr: int = 50, bw_mitr: int = 50, phi: str = "tanh",
                 graph_pool: str = "mean", x_is_indices: bool = False, edge_attr_is_indices: bool = False,
                 num_atom_types: int = 28, num_bond_types: int = 4,
                 head_bias: bool = True, head_dropout: float = 0.0,
                 use_ogb_encoders: bool = False, num_layers: int = 1,
                 row_norm: bool = False):
        super().__init__()
        self.hidden = hidden
        self.kappa = kappa
        self.fw_mitr = fw_mitr
        self.bw_mitr = bw_mitr
        self.phi_name = phi
        self.phi = torch.tanh if phi == "tanh" else torch.relu
        self.graph_pool = graph_pool
        self.x_is_indices = x_is_indices
        self.edge_attr_is_indices = edge_attr_is_indices
        self.head_dropout = head_dropout
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

        self.ignn_layers = nn.ModuleList([
            ImplicitGraph(in_features=hidden, out_features=hidden, num_node=1,
                          kappa=kappa, b_direct=False)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden) if self.row_norm else nn.Identity()
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(hidden, out_dim, bias=head_bias)

    def _build_norm_adj(self, edge_index: torch.Tensor, num_nodes: int,
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        from torch_geometric.utils import add_self_loops, degree
        ei, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = ei
        deg = degree(row, num_nodes, dtype=dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        ew = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        indices = torch.stack([row, col], dim=0)
        return torch.sparse_coo_tensor(indices, ew, (num_nodes, num_nodes), device=device).coalesce()

    def forward_pyg(self, batch) -> torch.Tensor:
        from torch_geometric.utils import scatter

        x = batch.x
        if self.use_ogb_encoders:
            U = self.input_emb(x)                       # [N, hidden]
        elif self.x_is_indices:
            x = x.squeeze(-1) if x.dim() == 2 else x
            U = self.input_emb(x.long())
        else:
            U = self.input_emb(x.float())

        N = U.size(0)
        device = U.device
        adj = self._build_norm_adj(batch.edge_index, N, device, U.dtype)

        A_rho = 1.0

        U_cur = U  # [N, hidden]
        for ignn_layer, ln in zip(self.ignn_layers, self.layer_norms):
            Z = ignn_layer(
                X_0=None, A=adj, U=U_cur, phi=self.phi,
                A_rho=A_rho, fw_mitr=self.fw_mitr, bw_mitr=self.bw_mitr, A_orig=None,
            )
            if Z.dim() == 2 and Z.size(0) == N and Z.size(1) == self.hidden:
                U_cur = Z
            elif Z.dim() == 2 and Z.size(0) == self.hidden and Z.size(1) == N:
                U_cur = Z.t()
            else:
                raise RuntimeError(f"Unexpected ImplicitGraph output shape {tuple(Z.shape)}")
            U_cur = ln(U_cur)

        h = U_cur                                        # [N, hidden]
        if self.head_dropout > 0:
            h = F.dropout(h, p=self.head_dropout, training=self.training)

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

@register("ignn")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    hidden = int(cfg.get("hidden", 100))
    if dataset_meta.get("task_type") == "node_classification_parity":
        return IGNNSynthetic(
            in_dim=dataset_meta["in_dim"],
            hidden=hidden,
            out_dim=dataset_meta["out_dim"],
            kappa=float(cfg.get("kappa", 0.99)),
            fw_mitr=int(cfg.get("fw_mitr", 300)),
            bw_mitr=int(cfg.get("bw_mitr", 300)),
            A_rho=float(cfg.get("A_rho", 1.0)),
            phi=str(cfg.get("phi", "relu")),
        )
    is_node_task = dataset_meta.get("task_type") in {"node_classification", "node_classification_masked"}
    from ._ogb_encoder import is_ogb_format
    use_ogb = bool(cfg.get("use_ogb_encoders", is_ogb_format(dataset_meta)))
    return IGNNPyG(
        in_dim=dataset_meta["in_dim"],
        hidden=hidden,
        out_dim=dataset_meta["out_dim"],
        edge_attr_dim=dataset_meta.get("edge_attr_dim", 0),
        kappa=float(cfg.get("kappa", 0.99)),
        fw_mitr=int(cfg.get("fw_mitr", 50)),
        bw_mitr=int(cfg.get("bw_mitr", 50)),
        phi=str(cfg.get("phi", "tanh")),
        use_ogb_encoders=use_ogb,
        num_layers=int(cfg.get("num_layers", 1)),
        graph_pool="node" if is_node_task else str(cfg.get("graph_pool", "mean")),
        x_is_indices=dataset_meta.get("x_is_indices", False),
        edge_attr_is_indices=dataset_meta.get("edge_attr_is_indices", False),
        head_bias=bool(cfg.get("head_bias", True)),
        head_dropout=float(cfg.get("head_dropout", 0.0)),
        row_norm=bool(cfg.get("row_norm", False)),
    )
