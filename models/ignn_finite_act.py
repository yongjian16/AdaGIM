"""Generalized IGNN-finite: per-node adaptive halting (Graves-ACT) on the fixed-point iteration."""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register

def _matmul_A(A: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    if A.is_sparse:
        A = A.coalesce()
        return torch.sparse.mm(A, M)
    return A @ M

class IGNNFiniteACT(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        max_iter: int = 30,
        ponder_eps: float = 1e-2,
        ponder_lambda: float = 1e-2,
        activation: str = "relu",
        graph_pool: str = "mean",
        x_is_indices: bool = False,
        edge_attr_is_indices: bool = False,
        edge_attr_dim: int = 0,
        num_atom_types: int = 28,
        num_bond_types: int = 4,
        use_ogb_encoders: bool = False,
        is_synthetic: bool = False,
    ):
        super().__init__()
        self.hidden = hidden
        self.max_iter = max_iter
        self.ponder_eps = ponder_eps
        self.ponder_lambda = ponder_lambda
        self.is_synthetic = is_synthetic
        self.graph_pool = graph_pool
        self.x_is_indices = x_is_indices
        self.edge_attr_is_indices = edge_attr_is_indices
        self.use_ogb_encoders = use_ogb_encoders

        if use_ogb_encoders:
            from ._ogb_encoder import OGBAtomEncoder, OGBBondEncoder
            self.input_emb = OGBAtomEncoder(emb_dim=hidden)
            self.edge_emb = OGBBondEncoder(emb_dim=hidden)
        elif x_is_indices:
            self.input_emb = nn.Embedding(num_atom_types, hidden)
            if edge_attr_dim > 0 or edge_attr_is_indices:
                if edge_attr_is_indices:
                    self.edge_emb = nn.Embedding(num_bond_types, hidden)
                else:
                    self.edge_emb = nn.Linear(edge_attr_dim, hidden)
            else:
                self.edge_emb = None
        else:
            self.input_emb = nn.Linear(in_dim, hidden)
            if edge_attr_dim > 0 or edge_attr_is_indices:
                if edge_attr_is_indices:
                    self.edge_emb = nn.Embedding(num_bond_types, hidden)
                else:
                    self.edge_emb = nn.Linear(edge_attr_dim, hidden)
            else:
                self.edge_emb = None

        self.W = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.xavier_uniform_(self.W, gain=0.99)

        self.halt_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.halt_head.bias)

        self.act = nn.ReLU() if activation == "relu" else nn.Tanh()

        self.head = nn.Linear(hidden, out_dim)

        self.last_ponder_cost: Optional[torch.Tensor] = None
        self.last_avg_iters: Optional[float] = None

    def _act_loop(
        self,
        ux: torch.Tensor,
        propagate_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        N = ux.size(0)
        device = ux.device

        halt_cum = torch.zeros(N, 1, device=device)         # \sum_t p_i^t so far
        weighted = torch.zeros(N, self.hidden, device=device)  # ponder-weighted sum
        n_iters = torch.zeros(N, 1, device=device)          # N_i count for ponder cost
        active = torch.ones(N, 1, device=device, dtype=torch.bool)

        h = torch.zeros_like(ux)

        threshold = 1.0 - self.ponder_eps
        for t in range(1, self.max_iter + 1):
            agg = propagate_fn(h)
            h = self.act(agg @ self.W + ux)

            p = torch.sigmoid(self.halt_head(h))             # [N, 1] in (0, 1)
            new_cum = halt_cum + p

            is_last_step = (t == self.max_iter)
            done_mask = (new_cum >= threshold) | is_last_step
            still_active = active & ~done_mask

            remainder = (1.0 - halt_cum).clamp_min(0.0)
            w_active = p * active.float()                    # weight while active
            w_done = remainder * (active & done_mask).float()  # remainder weight on first done step

            weighted = weighted + (w_active - w_done) * h    # = w_active*h - w_done*h ; effectively use min(p, remainder)

            n_iters = n_iters + active.float()

            halt_cum = new_cum
            active = active & ~done_mask

            if not active.any():
                break

        ponder_cost = (n_iters + (1.0 - halt_cum).clamp_min(0.0)).sum()
        avg_iters = float(n_iters.mean().item())
        return weighted, ponder_cost, avg_iters

    def forward_synthetic(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        if self.x_is_indices:
            ux = self.input_emb(X.long().squeeze(-1) if X.dim() == 2 else X.long())
        else:
            ux = self.input_emb(X.float())

        propagate_fn = lambda h: _matmul_A(A, h)
        weighted, ponder_cost, avg_iters = self._act_loop(ux, propagate_fn)

        self.last_ponder_cost = self.ponder_lambda * ponder_cost
        self.last_avg_iters = avg_iters

        return self.head(weighted)

    def forward_pyg(self, batch) -> torch.Tensor:
        from torch_geometric.utils import scatter

        x = batch.x
        edge_index = batch.edge_index
        edge_attr = getattr(batch, "edge_attr", None)
        batch_idx = batch.batch

        if self.use_ogb_encoders:
            ux = self.input_emb(x)
        elif self.x_is_indices:
            x = x.squeeze(-1) if x.dim() == 2 else x
            ux = self.input_emb(x.long())
        else:
            ux = self.input_emb(x.float())

        if self.use_ogb_encoders and edge_attr is not None:
            edge_w = torch.sigmoid(self.edge_emb(edge_attr).mean(dim=-1))
        elif self.edge_emb is not None and edge_attr is not None:
            if self.edge_attr_is_indices:
                e = edge_attr.squeeze(-1) if edge_attr.dim() == 2 else edge_attr
                edge_w = self.edge_emb(e.long()).mean(dim=-1)
            else:
                edge_w = self.edge_emb(edge_attr.float()).mean(dim=-1)
            edge_w = torch.sigmoid(edge_w)  # in (0, 1)
        else:
            edge_w = None

        src, dst = edge_index[0], edge_index[1]

        def propagate(h: torch.Tensor) -> torch.Tensor:
            msg = h[src]
            if edge_w is not None:
                msg = msg * edge_w.unsqueeze(-1)
            return scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="mean")

        weighted, ponder_cost, avg_iters = self._act_loop(ux, propagate)

        self.last_ponder_cost = self.ponder_lambda * ponder_cost
        self.last_avg_iters = avg_iters

        if self.graph_pool == "node":
            return self.head(weighted)
        if self.graph_pool == "mean":
            g = scatter(weighted, batch_idx, dim=0, reduce="mean")
        elif self.graph_pool == "sum":
            g = scatter(weighted, batch_idx, dim=0, reduce="sum")
        elif self.graph_pool == "max":
            g = scatter(weighted, batch_idx, dim=0, reduce="max")
        else:
            raise ValueError(self.graph_pool)
        return self.head(g)

@register("ignn_finite_act")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    is_synth = (dataset_meta.get("task_type") == "node_classification_parity")
    is_node_task = (dataset_meta.get("task_type") in {
        "node_classification", "node_classification_parity"
    })

    from ._ogb_encoder import is_ogb_format
    use_ogb = bool(cfg.get("use_ogb_encoders", is_ogb_format(dataset_meta)))
    return IGNNFiniteACT(
        in_dim=dataset_meta["in_dim"],
        hidden=int(cfg.get("hidden", 64)),
        out_dim=dataset_meta["out_dim"],
        max_iter=int(cfg.get("max_iter", 30)),
        ponder_eps=float(cfg.get("ponder_eps", 1e-2)),
        ponder_lambda=float(cfg.get("ponder_lambda", 1e-2)),
        activation=str(cfg.get("activation", "relu")),
        graph_pool="node" if is_node_task else str(cfg.get("graph_pool", "mean")),
        x_is_indices=dataset_meta.get("x_is_indices", False),
        edge_attr_is_indices=dataset_meta.get("edge_attr_is_indices", False),
        edge_attr_dim=dataset_meta.get("edge_attr_dim", 0),
        use_ogb_encoders=use_ogb,
        is_synthetic=is_synth,
    )
