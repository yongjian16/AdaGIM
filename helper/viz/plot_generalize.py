import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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
    """
    Z: [N,H] torch tensor
    Returns:
      Y2: [N,2] torch tensor
      mu: [H] torch tensor
      W2: [H,2] torch tensor
    """
    mu = Z.mean(dim=0)
    Zc = Z - mu
    _, _, Vh = torch.linalg.svd(Zc, full_matrices=False)
    W2 = Vh[:2].T
    Y2 = Zc @ W2
    return Y2, mu, W2


@torch.no_grad()
def decision_boundary_line_in_pca(mu: torch.Tensor, W2: torch.Tensor, head: nn.Linear):
    """
    head: nn.Linear(H->2)
    boundary in PCA coords: n0*x + n1*y + offset = 0
    """
    W = head.weight          # [2,H]
    b = head.bias            # [2]
    d = (W[1] - W[0])        # [H]
    c = (b[1] - b[0])        # scalar
    n = W2.T @ d             # [2]
    offset = (mu @ d) + c
    return n, offset


def plot_embedding_scatter_pca2_with_boundary(
    Z: torch.Tensor,                 # [N,H] embeddings
    Y_true_1hot: torch.Tensor,       # [N,2]
    head: nn.Linear,                # classifier head (H->2)
    title: str,
    out_path: str,
    max_points: int = 30000,
    plot_boundary: bool = True,
):
    assert Z.dim() == 2
    assert Y_true_1hot.dim() == 2 and Y_true_1hot.size(1) == 2
    N = Z.size(0)
    assert Y_true_1hot.size(0) == N

    # PCA projection
    Y2_t, mu, W2 = pca2_with_params(Z)
    Y2 = Y2_t.detach().cpu().numpy()

    # labels + correctness
    y_true = Y_true_1hot.argmax(dim=1).detach().cpu().numpy()
    with torch.no_grad():
        logits = head(Z)
        y_pred = logits.argmax(dim=1).detach().cpu().numpy()
    correct = (y_pred == y_true)

    # downsample if huge
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

