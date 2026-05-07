"""Chain-specific IGNN-finite (diagonal-of-unrolled) — wraps existing ParityDiagUnrollIGNN."""
from typing import Any, Dict

import torch
import torch.nn as nn

from ._existing_paths import ParityDiagUnrollIGNN
from . import register

def _transpose_adj(A: torch.Tensor) -> torch.Tensor:
    if A.is_sparse:
        return A.coalesce().t().coalesce()
    return A.t().contiguous()

class IGNNFiniteChain(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, L: int = 10, **kwargs):
        super().__init__()
        self.core = ParityDiagUnrollIGNN(in_dim=in_dim, hidden=hidden, out_dim=out_dim, **kwargs)
        self.L = L

    def forward_synthetic(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        logits, _ = self.core(X, A=_transpose_adj(A), L=self.L)
        return logits

@register("ignn_finite_chain")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    if dataset_meta.get("task_type") != "node_classification_parity":
        raise NotImplementedError(
            "IGNN-finite-chain (diagonal trick) is chain-specific. "
            "For non-chain graphs, use the generalized version: model name 'ignn_finite_act'."
        )
    L = int(cfg.get("L", dataset_meta.get("L_train", 10)))
    return IGNNFiniteChain(
        in_dim=dataset_meta["in_dim"],
        hidden=int(cfg.get("hidden", 100)),
        out_dim=dataset_meta["out_dim"],
        L=L,
        activation=str(cfg.get("activation", "relu")),
        init_gain=float(cfg.get("init_gain", 0.99)),
    )
