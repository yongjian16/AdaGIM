"""OGB-style atom and bond encoders."""
from typing import List

import torch
import torch.nn as nn

ATOM_FEATURE_DIMS: List[int] = [119, 5, 12, 12, 10, 6, 6, 2, 2]
BOND_FEATURE_DIMS: List[int] = [5, 6, 2]

class OGBAtomEncoder(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(d, emb_dim) for d in ATOM_FEATURE_DIMS
        ])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        out = 0
        for i in range(x.shape[1]):
            out = out + self.embeddings[i](x[:, i])
        return out

class OGBBondEncoder(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(d, emb_dim) for d in BOND_FEATURE_DIMS
        ])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr.dtype != torch.long:
            edge_attr = edge_attr.long()
        out = 0
        for i in range(edge_attr.shape[1]):
            out = out + self.embeddings[i](edge_attr[:, i])
        return out

def is_ogb_format(meta: dict) -> bool:
    return bool(meta.get("ogb_atom_features", False)) or (
        meta.get("in_dim", 0) == 9 and meta.get("edge_attr_dim", 0) == 3
    )
