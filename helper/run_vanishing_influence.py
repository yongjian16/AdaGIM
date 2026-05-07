import argparse
import csv
import json
import os
import time
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import ParityEIGNN, ParityIGNN
from utils import set_seeds, build_torch_adj_from_sp


def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _adj_name(adj_mode: int) -> str:
    return {1: "dir", 2: "dir_row_norm_sl", 3: "undir_sym_norm_sl"}[adj_mode]


def _make_out_dir(args) -> str:
    tag = (
        f"vanish_influence_randinit_fast"
        f"_K{args.K}_L{args.L}"
        f"_ng{args.n_graphs}"
        f"_baseBi{int(args.base_bidirectional)}"
        f"_pad{args.pad}"
        f"_adj{_adj_name(args.adj_mode)}"
        f"_hid{args.hidden}"
        f"_seedData{args.seed_data}"
        f"_seedInit{args.seed_init}"
        f"_flipFirst"
    )
    tag += f"_eignn(gamma{args.gamma}_g{args.g_type}_thr{args.threshold}_it{args.max_iter})"
    tag += f"_ignn(kappa{args.kappa}_fw{args.fw_mitr}_bw{args.bw_mitr}_phi{args.phi}" + ("_bd" if args.b_direct else "") + ")"
    return _mkdir(os.path.join(args.out_root, tag))


def build_blockdiag_chain_adj(L: int, n_graphs: int, bidirectional: bool) -> sp.coo_matrix:
    """
    Raw adjacency (no self-loops), block-diagonal over n_graphs chains.
    Directed edges: i -> i+1. If bidirectional, also add i+1 -> i.
    """
    # edges within one chain
    i = np.arange(L - 1, dtype=np.int64)
    rows = [i]
    cols = [i + 1]
    if bidirectional:
        rows.append(i + 1)
        cols.append(i)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(rows.shape[0], dtype=np.float32)
    A_chain = sp.coo_matrix((data, (rows, cols)), shape=(L, L))

    A_blk = sp.block_diag([A_chain] * n_graphs, format="coo")
    return A_blk


def sample_random_X_onehot(K: int, L: int, n_graphs: int, pad_value: int, device: torch.device) -> torch.Tensor:
    """
    Sample n_graphs random K-bit strings uniformly, embed as one-hot [N_total,2] exactly as your generator:
      - first K nodes carry bits
      - remaining nodes are pad_value (0 or 1)
    """
    bits = torch.randint(0, 2, (n_graphs, K), device=device, dtype=torch.long)  # [G,K]
    x_bits = torch.full((n_graphs, L), int(pad_value), device=device, dtype=torch.long)
    x_bits[:, :K] = bits
    X = F.one_hot(x_bits, num_classes=2).to(torch.float32)  # [G,L,2]
    return X.view(n_graphs * L, 2)  # [N_total,2]


def flip_first_node_onehot(X: torch.Tensor, L: int, n_graphs: int) -> torch.Tensor:
    """
    Flip first node feature for each chain: swap one-hot (0<->1) at node index 0 of each chain.
    """
    Xp = X.clone()
    first_idx = (torch.arange(n_graphs, device=X.device) * L).long()  # [G]
    # swap columns (one-hot over {0,1})
    Xp[first_idx] = Xp[first_idx][:, [1, 0]]
    return Xp


@torch.no_grad()
def encode_ignn(net: ParityIGNN, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    net.eval()
    return net.encode(X, A)


@torch.no_grad()
def encode_eignn(net: ParityEIGNN, X: torch.Tensor, A: torch.Tensor, A_sp: torch.Tensor) -> torch.Tensor:
    net.eval()
    net.set_graph(A, A_sp)
    return net.encode(X)


def diff_by_pos(Z: torch.Tensor, Zp: torch.Tensor, L: int, n_graphs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = Zp - Z
    node_diff = torch.norm(delta, p=2, dim=1)         # [N]
    node_diff = node_diff.view(n_graphs, L)           # [G,L]
    mean = node_diff.mean(dim=0).detach().cpu()       # [L]
    std = node_diff.std(dim=0, unbiased=True).detach().cpu()
    return mean, std


def save_curve_csv(path: str, mean: torch.Tensor, std: torch.Tensor):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pos", "mean", "std"])
        for i in range(mean.numel()):
            w.writerow([i + 1, float(mean[i].item()), float(std[i].item())])


def plot_curves(out_path: str, mean_ignn, std_ignn, mean_eignn, std_eignn, title: str, logy: bool):
    x = np.arange(1, len(mean_ignn) + 1)
    plt.figure()
    plt.plot(x, mean_ignn.numpy(), label="IGNN")
    plt.plot(x, mean_eignn.numpy(), label="EIGNN")
    plt.fill_between(x, (mean_ignn - std_ignn).numpy(), (mean_ignn + std_ignn).numpy(), alpha=0.2)
    plt.fill_between(x, (mean_eignn - std_eignn).numpy(), (mean_eignn + std_eignn).numpy(), alpha=0.2)
    plt.xlabel("Node position along chain")
    plt.ylabel("Avg ||Δembedding||₂ over chains")
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--L", type=int, default=50)
    ap.add_argument("--n_graphs", type=int, default=100)

    ap.add_argument("--pad", type=int, default=0, choices=[0, 1])
    ap.add_argument("--base_bidirectional", action="store_true",
                    help="If set, raw adjacency includes both directions (like bidirectional=True). Default is directed.")

    ap.add_argument(
        "--adj_mode", type=int, default=1, choices=[1, 2, 3],
        help="1: directed raw A; 2: directed (A+I) row-normalized; 3: undirected (A+A^T+I) symmetric-normalized",
    )

    ap.add_argument("--hidden", type=int, default=100)

    ap.add_argument("--seed_data", type=int, default=0, help="Controls random chain sampling.")
    ap.add_argument("--seed_init", type=int, default=0, help="Controls random init for BOTH models.")
    ap.add_argument("--logy", action="store_true")

    # EIGNN
    ap.add_argument("--threshold", type=float, default=1e-4)
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--g_type", type=str, default="psd", choices=["psd", "nonpsd1", "nonpsd2"])

    # IGNN
    ap.add_argument("--kappa", type=float, default=0.99)
    ap.add_argument("--fw_mitr", type=int, default=300)
    ap.add_argument("--bw_mitr", type=int, default=300)
    ap.add_argument("--A_rho", type=float, default=1.0)
    ap.add_argument("--phi", type=str, default="relu", choices=["tanh", "relu"])
    ap.add_argument("--b_direct", action="store_true")

    ap.add_argument("--out_root", type=str, default="results")

    args = ap.parse_args()
    assert args.K <= args.L, "Need K <= L"
    out_dir = _make_out_dir(args)
    print("Results dir:", out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------------
    # Sample 100 random chains (uniform over {0,1}^K)
    # -----------------------------
    set_seeds(args.seed_data)
    X = sample_random_X_onehot(args.K, args.L, args.n_graphs, args.pad, device)
    Xp = flip_first_node_onehot(X, args.L, args.n_graphs)

    # -----------------------------
    # Build block-diagonal chain adjacency (raw, no self-loops)
    # -----------------------------
    A_sp_raw = build_blockdiag_chain_adj(args.L, args.n_graphs, bidirectional=args.base_bidirectional)
    A, A_sp = build_torch_adj_from_sp(A_sp_raw, device, args.adj_mode)

    in_dim = X.size(1)  # should be 2

    # -----------------------------
    # Random init (same scheme): reseed before each instantiation
    # -----------------------------
    set_seeds(args.seed_init)
    ignn = ParityIGNN(
        in_dim=in_dim, hidden=args.hidden, out_dim=2,
        kappa=args.kappa, b_direct=args.b_direct,
        fw_mitr=args.fw_mitr, bw_mitr=args.bw_mitr,
        A_rho=args.A_rho, phi=args.phi
    ).to(device)

    set_seeds(args.seed_init)
    eignn = ParityEIGNN(
        adj=A, sp_adj=A_sp,
        in_dim=in_dim, hidden=args.hidden, out_dim=2,
        threshold=args.threshold, max_iter=args.max_iter,
        gamma=args.gamma, g_type=args.g_type
    ).to(device)

    # -----------------------------
    # Measure vanishing influence: ||Δz|| by position
    # -----------------------------
    t0 = time.time()
    with torch.no_grad():
        Z_ignn = encode_ignn(ignn, X, A)
        Zp_ignn = encode_ignn(ignn, Xp, A)

        Z_eignn = encode_eignn(eignn, X, A, A_sp)
        Zp_eignn = encode_eignn(eignn, Xp, A, A_sp)

    mean_ignn, std_ignn = diff_by_pos(Z_ignn, Zp_ignn, args.L, args.n_graphs)
    mean_eignn, std_eignn = diff_by_pos(Z_eignn, Zp_eignn, args.L, args.n_graphs)
    print(f"Done. Elapsed: {time.time()-t0:.2f}s")

    # -----------------------------
    # Save outputs
    # -----------------------------
    meta = {
        "K": args.K,
        "L": args.L,
        "n_graphs": args.n_graphs,
        "pad_value": args.pad,
        "base_bidirectional": bool(args.base_bidirectional),
        "adj_mode": args.adj_mode,
        "adj_name": _adj_name(args.adj_mode),
        "hidden": args.hidden,
        "seed_data": args.seed_data,
        "seed_init": args.seed_init,
        "flip": "swap one-hot at node 1",
        "eignn": {"threshold": args.threshold, "max_iter": args.max_iter, "gamma": args.gamma, "g_type": args.g_type},
        "ignn": {"kappa": args.kappa, "fw_mitr": args.fw_mitr, "bw_mitr": args.bw_mitr, "A_rho": args.A_rho, "phi": args.phi, "b_direct": bool(args.b_direct)},
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    save_curve_csv(os.path.join(out_dir, "ignn_diff_by_pos.csv"), mean_ignn, std_ignn)
    save_curve_csv(os.path.join(out_dir, "eignn_diff_by_pos.csv"), mean_eignn, std_eignn)

    with open(os.path.join(out_dir, "diff_by_pos_lists.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "pos": list(range(1, args.L + 1)),
                "IGNN": {"mean": [float(x) for x in mean_ignn.tolist()], "std": [float(x) for x in std_ignn.tolist()]},
                "EIGNN": {"mean": [float(x) for x in mean_eignn.tolist()], "std": [float(x) for x in std_eignn.tolist()]},
            },
            f,
            indent=2,
        )

    title = f"Vanishing influence @ random init | (K,L)=({args.K},{args.L}) | n={args.n_graphs} | adj={_adj_name(args.adj_mode)}"
    plot_path = os.path.join(out_dir, "diff_by_pos.png")
    plot_curves(plot_path, mean_ignn, std_ignn, mean_eignn, std_eignn, title, args.logy)
    print("Saved:", plot_path)
    print("Saved CSV/JSON in:", out_dir)


if __name__ == "__main__":
    main()
