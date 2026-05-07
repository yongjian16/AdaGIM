# save as: data_chains_flex.py
import itertools
import random
from typing import List, Literal, Tuple, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ===== helpers =====
def one_hot_bits(bits: List[int]) -> torch.Tensor:
    t = torch.tensor(bits, dtype=torch.long)
    return torch.nn.functional.one_hot(t, num_classes=2).float()  # [L,2]

def cumulative_parity_onehot(bits: List[int]) -> torch.Tensor:
    t = torch.tensor(bits, dtype=torch.long)
    cum = torch.cumsum(t, dim=0) % 2
    return torch.nn.functional.one_hot(cum, num_classes=2).float()  # [L,2]

def all_binary_chains(L: int) -> List[List[int]]:
    return [list(s) for s in itertools.product([0, 1], repeat=L)]

def random_binary_chains(L: int, num_samples: int) -> List[List[int]]:
    return [[random.randint(0, 1) for _ in range(L)] for _ in range(num_samples)]

# ===== edges / adjacency =====
def build_identity_edges(L: int) -> torch.Tensor:
    if L <= 0:
        return torch.empty(2, 0, dtype=torch.long)
    idx = torch.arange(L, dtype=torch.long)
    return torch.stack([idx, idx], dim=0)  # [2, L]

def build_chain_edges(L: int, directed: bool = True) -> torch.Tensor:
    """
    directed=True  -> edges: (i -> i+1)
    directed=False -> edges: (i <-> i+1) both directions
    """
    if L <= 1:
        return torch.empty(2, 0, dtype=torch.long)
    fwd = [(i, i + 1) for i in range(L - 1)]
    edges = fwd if directed else (fwd + [(j, i) for i, j in fwd])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def dense_adj_from_edge_index(
    num_nodes: int,
    edge_index: torch.Tensor,
    add_self_loops: bool = False,
    normalize: Literal["none", "sym", "row"] = "none",
) -> torch.Tensor:
    """
    Build dense adjacency A [N,N] from edge_index.
    normalize:
      - "none": raw adjacency
      - "sym" : D^{-1/2} (A + I*) D^{-1/2}
      - "row" : D^{-1}   (A + I*)
    If add_self_loops=True, add I before normalization (I*).
    """
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    if edge_index.numel() > 0:
        A[edge_index[0], edge_index[1]] = 1.0
    if add_self_loops:
        A.fill_diagonal_(1.0)

    if normalize == "none":
        return A

    deg = A.sum(dim=1)  # [N]
    if normalize == "row":
        inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        return torch.diag(inv) @ A
    elif normalize == "sym":
        inv_sqrt = torch.where(deg > 0, 1.0 / torch.sqrt(deg), torch.zeros_like(deg))
        D_inv_sqrt = torch.diag(inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt
    else:
        raise ValueError("normalize must be one of: 'none','sym','row'")

# ===== single graph builder =====
def make_chain_graph(
    L: int,
    bits: List[int],
    *,
    identity_adj: bool = False,
    directed: bool = True,
    attach_dense_adj: bool = False,
    dense_add_self_loops: bool = False,
    dense_normalize: Literal["none", "sym", "row"] = "none",
) -> Data:
    """
    Build a chain graph Data with:
      - x: [L,2] one-hot bits
      - y: [L,2] cumulative parity
      - edge_index: identity edges OR real chain edges (directed/undirected)
      - pos: [L] positions 0..L-1
      - (optional) A: dense adjacency [L,L] if attach_dense_adj=True
    """
    x = one_hot_bits(bits)                  # [L,2]
    y = cumulative_parity_onehot(bits)      # [L,2]
    pos = torch.arange(L, dtype=torch.long)

    if identity_adj:
        edge_index = build_identity_edges(L)
    else:
        edge_index = build_chain_edges(L, directed=directed)

    data = Data(x=x, y=y, edge_index=edge_index, pos=pos)

    if attach_dense_adj:
        A = dense_adj_from_edge_index(
            num_nodes=L,
            edge_index=edge_index,
            add_self_loops=dense_add_self_loops,
            normalize=dense_normalize,
        )
        data.A = A  # attach a dense adjacency tensor

    return data

# ===== dataset builders & loaders =====
def build_train_dataset_fixed_length(
    L: int,
    *,
    mode: str = "all",
    num_samples: int = 256,
    identity_adj: bool = False,
    directed: bool = True,
    attach_dense_adj: bool = False,
    dense_add_self_loops: bool = False,
    dense_normalize: Literal["none", "sym", "row"] = "none",
):
    seqs = all_binary_chains(L) if mode == "all" else random_binary_chains(L, num_samples)
    return [
        make_chain_graph(
            L, s,
            identity_adj=identity_adj,
            directed=directed,
            attach_dense_adj=attach_dense_adj,
            dense_add_self_loops=dense_add_self_loops,
            dense_normalize=dense_normalize,
        )
        for s in seqs
    ]

def build_eval_dataset_fixed_length(
    L: int,
    *,
    num_samples: int = 512,
    identity_adj: bool = False,
    directed: bool = True,
    attach_dense_adj: bool = False,
    dense_add_self_loops: bool = False,
    dense_normalize: Literal["none", "sym", "row"] = "none",
):
    seqs = random_binary_chains(L, num_samples)
    return [
        make_chain_graph(
            L, s,
            identity_adj=identity_adj,
            directed=directed,
            attach_dense_adj=attach_dense_adj,
            dense_add_self_loops=dense_add_self_loops,
            dense_normalize=dense_normalize,
        )
        for s in seqs
    ]

def make_loaders(
    train_L: int,
    val_L: int,
    batch_size: int,
    *,
    train_mode: str = "all",
    train_samples: int = 256,
    val_samples: int = 512,
    shuffle: bool = True,
    identity_adj: bool = False,
    directed: bool = True,
    attach_dense_adj: bool = False,
    dense_add_self_loops: bool = False,
    dense_normalize: Literal["none", "sym", "row"] = "none",
) -> Tuple[DataLoader, DataLoader]:
    train_ds = build_train_dataset_fixed_length(
        train_L,
        mode=train_mode,
        num_samples=train_samples,
        identity_adj=identity_adj,
        directed=directed,
        attach_dense_adj=attach_dense_adj,
        dense_add_self_loops=dense_add_self_loops,
        dense_normalize=dense_normalize,
    )
    val_ds = build_eval_dataset_fixed_length(
        val_L,
        num_samples=val_samples,
        identity_adj=identity_adj,
        directed=directed,
        attach_dense_adj=attach_dense_adj,
        dense_add_self_loops=dense_add_self_loops,
        dense_normalize=dense_normalize,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )
