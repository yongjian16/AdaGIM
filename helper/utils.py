import json
import os
import random
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------------------------
# Repro / loss / sparse utils
# ---------------------------
def set_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sp_add_self_loops(spA):
    # scipy COO/CSR: add I

    n = spA.shape[0]
    return (spA + sp.eye(n, format="coo")).tocoo()

def _sp_row_normalize(spA, eps=1e-12):
    # D^{-1} A
    
    spA = spA.tocsr()
    deg = np.asarray(spA.sum(axis=1)).reshape(-1)
    deg = np.maximum(deg, eps)
    inv = 1.0 / deg
    Dinv = sp.diags(inv, format="csr")
    return (Dinv @ spA).tocoo()

def _sp_sym_normalize(spA, eps=1e-12):
    # D^{-1/2} A D^{-1/2}

    spA = spA.tocsr()
    deg = np.asarray(spA.sum(axis=1)).reshape(-1)
    deg = np.maximum(deg, eps)
    inv_sqrt = 1.0 / np.sqrt(deg)
    Dinv2 = sp.diags(inv_sqrt, format="csr")
    return (Dinv2 @ spA @ Dinv2).tocoo()

def build_torch_adj_from_sp(spA_raw, device, adj_mode: int):
    """
    Returns:
      A_t : torch sparse COO on device (coalesced)
      spA : scipy COO after processing (for EIGNN symmetry check etc.)
    """


    if adj_mode == 1:
        spB = spA_raw.tocoo()
    elif adj_mode == 2:
        spB = _sp_add_self_loops(spA_raw)
        spB = _sp_row_normalize(spB)
    elif adj_mode == 3:
        spB = (spA_raw + spA_raw.T).tocoo()
        spB = _sp_add_self_loops(spB)
        spB = _sp_sym_normalize(spB)
    else:
        raise ValueError(f"adj_mode must be 1, 2, or 3. Got {adj_mode}")

    A_t = ensure_torch_sparse(spB, device).coalesce()
    return A_t, spB

def make_loss(loss_name: str):
    """
    ce  = CrossEntropy on logits
    mse = MSE on softmax(logits) vs one-hot
    """
    loss_name = loss_name.lower()
    if loss_name == "ce":
        ce = nn.CrossEntropyLoss()

        def loss_fn(logits: torch.Tensor, Y_1hot: torch.Tensor) -> torch.Tensor:
            target = Y_1hot.argmax(dim=1).long()
            return ce(logits, target)

        return loss_fn

    if loss_name == "mse":
        return nn.MSELoss()


    raise ValueError(f"Unknown --loss {loss_name}. Use 'ce' or 'mse'.")


def scipy_to_torch_sparse(A: sp.spmatrix, device: torch.device) -> torch.Tensor:
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
    val = torch.tensor(A.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, size=A.shape, device=device).coalesce()


def ensure_torch_sparse(A_sp, device: torch.device) -> torch.Tensor:
    """
    Accepts torch sparse tensor OR scipy sparse matrix.
    Returns torch sparse COO tensor on device.
    """
    if isinstance(A_sp, torch.Tensor):
        return A_sp.to(device).coalesce() if A_sp.is_sparse else A_sp.to(device)
    if sp.issparse(A_sp):
        return scipy_to_torch_sparse(A_sp, device)
    raise TypeError(f"Unsupported A_sp type: {type(A_sp)}")


# ---------------------------
# Output directory + logging
# ---------------------------
def make_out_dir(args, base_dir: str = "results") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # sortable
    title_dir = "directed" if getattr(args, "directed", False) else "bidirectional"
    if args.model == "eignn":
        name = (
            f"{ts}_model={args.model}_{args.g_type}_K{args.K}_L1={args.L1}_L2={args.L2}_"
            f"{title_dir}_loss={args.loss}_seed={args.seed}"
        )
    else:
        name = (
            f"{ts}_model={args.model}_K{args.K}_L1={args.L1}_L2={args.L2}_"
            f"{title_dir}_loss={args.loss}_seed={args.seed}"
        )
    out_dir = os.path.join(base_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_args(out_dir: str, args) -> None:
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def init_metrics(out_dir: str) -> str:
    path = os.path.join(out_dir, "metrics.tsv")
    with open(path, "w") as f:
        f.write("epoch\ttrain_loss\ttest_acc_lastpos\n")
    return path


def append_metrics(metrics_path: str, epoch: int, train_loss: float, test_acc_lastpos: float) -> None:
    with open(metrics_path, "a") as f:
        f.write(f"{epoch}\t{train_loss:.10f}\t{test_acc_lastpos:.6f}\n")


# ---------------------------
# Plotting helpers
# ---------------------------
@torch.no_grad()
def acc_by_pos(logits: torch.Tensor, Y_true_1hot: torch.Tensor, L: int) -> torch.Tensor:
    """
    logits: [N_total,2]
    Y_true_1hot: [N_total,2]
    returns: [L] accuracy averaged over graphs at each position
    """
    N = logits.size(0)
    assert N % L == 0, f"N_total={N} not divisible by L={L}"
    G = N // L
    pred = logits.argmax(dim=1).view(G, L)
    true = Y_true_1hot.argmax(dim=1).view(G, L)
    return (pred == true).float().mean(dim=0)


def plot_acc_by_pos(
    logits: torch.Tensor,
    Y_true_1hot: torch.Tensor,
    L: int,
    L1: int,
    L2: int,
    title: str,
    out_path: str,
):
    a = acc_by_pos(logits, Y_true_1hot, L).detach().cpu().numpy()
    xs = np.arange(1, L + 1)
    mask = (xs >= L1) & (xs <= L2)

    plt.figure(figsize=(8, 4.8))
    plt.plot(xs[mask], a[mask], marker="o")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Node position (1-indexed)")
    plt.ylabel("Accuracy (prefix parity, test)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def pca2_with_params(Z: torch.Tensor):
    mu = Z.mean(dim=0)
    Zc = Z - mu
    _, _, Vh = torch.linalg.svd(Zc, full_matrices=False)
    W2 = Vh[:2].T
    Y2 = Zc @ W2
    return Y2, mu, W2


@torch.no_grad()
def decision_boundary_line_in_pca(mu: torch.Tensor, W2: torch.Tensor, head: nn.Linear):
    W = head.weight                 # [2,H]
    b = head.bias
    if b is None:
        b = torch.zeros(W.size(0), device=W.device, dtype=W.dtype)  # [2]

    d = (W[1] - W[0])              # [H]
    c = (b[1] - b[0])              # scalar
    n = W2.T @ d                   # [2]
    offset = (mu @ d) + c
    return n, offset


def plot_embedding_scatter_pca2_with_boundary(
    Z: torch.Tensor,
    Y_true_1hot: torch.Tensor,
    head: nn.Linear,
    title: str,
    out_path: str,
    max_points: int = 30000,
    plot_boundary: bool = True,
):
    assert Z.dim() == 2
    assert Y_true_1hot.dim() == 2 and Y_true_1hot.size(1) == 2
    N = Z.size(0)
    assert Y_true_1hot.size(0) == N

    Y2_t, mu, W2 = pca2_with_params(Z)
    Y2 = Y2_t.detach().cpu().numpy()

    y_true = Y_true_1hot.argmax(dim=1).detach().cpu().numpy()
    with torch.no_grad():
        y_pred = head(Z).argmax(dim=1).detach().cpu().numpy()
    correct = (y_pred == y_true)

    if N > max_points:
        keep = np.random.choice(N, size=max_points, replace=False)
        Y2 = Y2[keep]
        y_true = y_true[keep]
        correct = correct[keep]

    plt.figure(figsize=(7.8, 6.6))
    m0 = (y_true == 0)
    m1 = (y_true == 1)

    plt.scatter(Y2[m0 & correct, 0], Y2[m0 & correct, 1], s=10, alpha=0.6, label="true=0, correct", marker="o")
    plt.scatter(Y2[m0 & ~correct, 0], Y2[m0 & ~correct, 1], s=18, alpha=0.8, label="true=0, wrong", marker="x")
    plt.scatter(Y2[m1 & correct, 0], Y2[m1 & correct, 1], s=10, alpha=0.6, label="true=1, correct", marker="o")
    plt.scatter(Y2[m1 & ~correct, 0], Y2[m1 & ~correct, 1], s=18, alpha=0.8, label="true=1, wrong", marker="x")

    if plot_boundary:
        n_t, offset_t = decision_boundary_line_in_pca(mu, W2, head)
        n = n_t.detach().cpu().numpy()
        offset = float(offset_t.detach().cpu().item())

        xmin, xmax = np.percentile(Y2[:, 0], [1, 99])
        xline = np.linspace(xmin, xmax, 200)
        if abs(n[1]) > 1e-10:
            yline = -(n[0] * xline + offset) / n[1]
            plt.plot(xline, yline, linewidth=2, label="decision boundary (proj.)")
        else:
            x0 = -offset / (n[0] + 1e-12)
            plt.axvline(x0, linewidth=2, label="decision boundary (proj.)")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_loss_curve(loss_hist, title, out_path):
    plt.figure(figsize=(8, 4.8))
    plt.plot(np.arange(1, len(loss_hist) + 1), loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)