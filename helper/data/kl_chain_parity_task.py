# save as: data_chain_parity_generalize.py
import itertools
from typing import List, Tuple, Optional

import numpy as np
import torch
import scipy.sparse as sp


# -----------------------------
# Graph utilities
# -----------------------------
def build_chain_edges(L: int, bidirectional: bool = True) -> List[Tuple[int, int]]:
    if L <= 1:
        return []
    fwd = [(i, i + 1) for i in range(L - 1)]
    return fwd if not bidirectional else (fwd + [(j, i) for (i, j) in fwd])


def block_diag_adj(edge_lists: List[List[Tuple[int, int]]], lengths: List[int], device=None) -> torch.Tensor:
    N = int(sum(lengths))
    A = torch.zeros(N, N, dtype=torch.float32, device=device)
    off = 0
    for L, edges in zip(lengths, edge_lists):
        for u, v in edges:
            A[off + u, off + v] = 1.0
        off += L
    return A


def normalize_with_self_loops(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    N = A.size(0)
    A_hat = A + torch.eye(N, dtype=A.dtype, device=A.device)
    deg = A_hat.sum(dim=1).clamp_min(eps)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def one_hot_bits(bits01: torch.Tensor) -> torch.Tensor:
    """
    bits01: [N] long in {0,1}
    returns: [N,2] float one-hot
    """
    return torch.nn.functional.one_hot(bits01.to(torch.long), num_classes=2).float()


def parity_label(bits01: torch.Tensor) -> int:
    """bits01: [K] in {0,1}. returns parity in {0,1}."""
    return int(bits01.sum().item() % 2)


# -----------------------------
# Main generator: (K, L) with exactly 2^K sequences
# -----------------------------
@torch.no_grad()
def generate_chain_parity_KL_2powK(
    K: int,
    L: int,
    *,
    bidirectional: bool = True,
    pad_value: int = 0,
    supervise: str = "last",   # "last" (default) or "all_prefix"
    return_dense_adj: bool = True,
    max_graphs: Optional[int] = None,  # if K is huge, optionally cap to first max_graphs sequences
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], sp.coo_matrix, torch.Tensor]:
    """
    Creates EXACTLY 2^K sequences (unless max_graphs is set), each a length-L chain.

    Construction for each graph g:
      - Choose one K-bit string b in {0,1}^K (enumerating all 2^K possibilities).
      - Node i feature (0-indexed):
          if i < K: bit = b[i]
          else:     bit = pad_value  (default 0)
        Feature is one-hot over {0,1}: [L,2]
      - Label:
          parity = sum(b) mod 2
        If supervise == "last":
          Y is zero everywhere except last node has one-hot parity.
        If supervise == "all_prefix":
          Y[i] is one-hot(prefix parity up to i), where padding contributes pad_value.

    Returns merged (block-diagonal) batch:
      X        : [N_total, 2]
      Y        : [N_total, 2]
      A_norm   : [N_total, N_total] or None
      A_sp     : scipy COO adjacency (no self-loops)
      last_mask: [N_total] bool mask selecting last node of each chain

    Requirement:
      - This encoding assumes "K informative positions" sit on first K nodes.
      - Usually you want K <= L. (If K > L, you'd be truncating bits; we forbid it.)
    """
    assert K >= 1, "K must be >= 1"
    assert L >= 1, "L must be >= 1"
    assert 0 <= pad_value <= 1, "pad_value must be 0 or 1"
    assert supervise in {"last", "all_prefix"}, "supervise must be 'last' or 'all_prefix'"
    assert K <= L, f"Need K <= L for 'first K nodes carry bits'. Got K={K}, L={L}."

    # Enumerate all 2^K bitstrings (optionally cap)
    all_bits = list(itertools.product([0, 1], repeat=K))
    if max_graphs is not None:
        all_bits = all_bits[: int(max_graphs)]

    num_graphs = len(all_bits)
    lengths = [L] * num_graphs
    edges_all = [build_chain_edges(L, bidirectional=bidirectional) for _ in range(num_graphs)]

    X_list, Y_list, lm_list = [], [], []
    for b in all_bits:
        b = torch.tensor(b, dtype=torch.long)  # [K]

        # ---- build length-L bit sequence on nodes (first K bits, then padding)
        if L == K:
            x_bits = b
        else:
            pad = torch.full((L - K,), int(pad_value), dtype=torch.long)
            x_bits = torch.cat([b, pad], dim=0)  # [L]

        X_list.append(one_hot_bits(x_bits))      # [L,2]

        # ---- labels
        if supervise == "last":
            y = torch.zeros(L, dtype=torch.long)      # placeholder class 0
            y[-1] = parity_label(b)                   # parity only at last node
            Y_list.append(one_hot_bits(y))            # [L,2]
        else:
            # prefix parity across the length-L node sequence (including padding)
            pref = (torch.cumsum(x_bits, dim=0) % 2).to(torch.long)  # [L]
            Y_list.append(one_hot_bits(pref))                          # [L,2]

        lm = torch.zeros(L, dtype=torch.bool)
        lm[-1] = True
        lm_list.append(lm)

    X = torch.cat(X_list, dim=0)                 # [N_total,2]
    Y = torch.cat(Y_list, dim=0)                 # [N_total,2]
    last_mask = torch.cat(lm_list, dim=0)        # [N_total]

    A = block_diag_adj(edges_all, lengths)       # [N_total,N_total]
    A_sp = sp.coo_matrix(A.cpu().numpy())        # sparse, no self-loops
    # A_norm = normalize_with_self_loops(A) if return_dense_adj else None
    # A = normalize_with_self_loops(A)
    A = A.to_sparse().coalesce() 

    return X, Y, A, A_sp, last_mask


# -----------------------------
# Convenience: build train/test for generalization
# -----------------------------
def build_train_test_generalization(
    K1: int, L1: int,
    K2: int, L2: int,
    *,
    bidirectional: bool = True,
    pad_value: int = 0,
    supervise: str = "last",
    return_dense_adj: bool = True,
    max_train_graphs: Optional[int] = None,
    max_test_graphs: Optional[int] = None,
):
    """
    Returns:
      (X_tr, Y_tr, A_tr, A_sp_tr, last_mask_tr),
      (X_te, Y_te, A_te, A_sp_te, last_mask_te)
    """
    train = generate_chain_parity_KL_2powK(
        K1, L1,
        bidirectional=bidirectional,
        pad_value=pad_value,
        supervise=supervise,
        return_dense_adj=return_dense_adj,
        max_graphs=max_train_graphs,
    )
    test = generate_chain_parity_KL_2powK(
        K2, L2,
        bidirectional=bidirectional,
        pad_value=pad_value,
        supervise=supervise,
        return_dense_adj=return_dense_adj,
        max_graphs=max_test_graphs,
    )
    return train, test
