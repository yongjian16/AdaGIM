import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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
    U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
    W2 = Vh[:2].T
    Y2 = Zc @ W2
    return Y2, mu, W2


@torch.no_grad()
def decision_boundary_line_in_pca(mu: torch.Tensor, W2: torch.Tensor, head: nn.Linear):
    """
    For head: logits = Z @ head.weight.T + head.bias  (head.weight: [2,H])
    Returns:
      n: [2] normal vector in PCA coords
      offset: scalar for line n0*x + n1*y + offset = 0
    """
    W = head.weight  # [2,H]
    b = head.bias    # [2]
    d = (W[1] - W[0])            # [H]
    c = (b[1] - b[0])            # scalar
    n = W2.T @ d                 # [2]
    offset = (mu @ d) + c        # scalar
    return n, offset


@torch.no_grad()
def scatter_embeddings_pca2_by_true_label(
    Z: torch.Tensor,
    Y_true_1hot: torch.Tensor,
    head: nn.Linear | None = None,
    title: str = "",
    out_path: str = "scatter.png",
    plot_boundary: bool = False,
    max_points: int = 30000,
    show_correctness: bool = True,
):
    """
    Z: [N,H] torch tensor (embeddings BEFORE classifier)
    Y_true_1hot: [N,2] one-hot labels
    head: nn.Linear mapping H -> 2 (optional; required for correctness/boundary)
    """
    assert Z.dim() == 2
    assert Y_true_1hot.dim() == 2 and Y_true_1hot.size(1) == 2
    N = Z.size(0)
    assert Y_true_1hot.size(0) == N

    Y2_t, mu, W2 = pca2_with_params(Z)
    Y2 = Y2_t.detach().cpu().numpy()

    y_true = Y_true_1hot.argmax(dim=1).detach().cpu().numpy()

    correct = None
    if head is not None and show_correctness:
        logits = head(Z)  # [N,2]
        y_pred = logits.argmax(dim=1).detach().cpu().numpy()
        correct = (y_pred == y_true)

    # downsample (but keep label balance roughly by random)
    if N > max_points:
        keep = np.random.choice(N, size=max_points, replace=False)
        Y2 = Y2[keep]
        y_true = y_true[keep]
        if correct is not None:
            correct = correct[keep]

    plt.figure(figsize=(7.8, 6.6))

    if correct is None:
        m0 = (y_true == 0)
        m1 = (y_true == 1)
        plt.scatter(Y2[m0, 0], Y2[m0, 1], s=10, alpha=0.6, label="true=0")
        plt.scatter(Y2[m1, 0], Y2[m1, 1], s=10, alpha=0.6, label="true=1")
    else:
        m0 = (y_true == 0)
        m1 = (y_true == 1)
        plt.scatter(Y2[m0 & correct, 0], Y2[m0 & correct, 1], s=10, alpha=0.6, label="true=0, correct", marker="o")
        plt.scatter(Y2[m0 & ~correct, 0], Y2[m0 & ~correct, 1], s=18, alpha=0.8, label="true=0, wrong", marker="x")
        plt.scatter(Y2[m1 & correct, 0], Y2[m1 & correct, 1], s=10, alpha=0.6, label="true=1, correct", marker="o")
        plt.scatter(Y2[m1 & ~correct, 0], Y2[m1 & ~correct, 1], s=18, alpha=0.8, label="true=1, wrong", marker="x")

    if plot_boundary:
        if head is None:
            raise ValueError("plot_boundary=True requires head (nn.Linear).")
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
