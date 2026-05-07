import argparse
import random, numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.nn.utils import clip_grad_value_
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from data_chains_flex import make_loaders, build_eval_dataset_fixed_length
from model_parity_gcn import IGNN_finite


def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    if v in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean (true/false).")


def train_step_mse_iter(net: IGNN_finite, batch: Batch, steps: int, criterion: nn.Module):
    """
    Per-iteration MSE on cumulative parity:
    For each t in [0..steps-1], select nodes with batch.pos == t across the mini-batch,
    then MSE( softmax(logits_t), one_hot_target_t ).
    """
    logits_seq, _ = net(batch, steps=steps, return_all=True)  # [steps, N, C]
    total = 0.0
    active = 0
    for t in range(steps):
        mask_t = (batch.pos == t)
        if mask_t.any():
            probs_t   = torch.softmax(logits_seq[t][mask_t], dim=1)  # [B,2]
            targets_t = batch.y[mask_t]                              # [B,2]
            total += criterion(probs_t, targets_t)
            active += 1
    return (total / max(active, 1)), logits_seq


@torch.no_grad()
def last_node_metrics_mse(net: IGNN_finite, batch: Batch, steps: int, criterion: nn.Module):
    """
    Run for `steps` iterations and evaluate ONLY the last node (pos == steps-1) in each graph.
    Returns: (mse_last, acc_last)
    """
    logits_seq, _ = net(batch, steps=steps, return_all=True)
    last_logits = logits_seq[-1]                               # [N,2]
    last_mask   = (batch.pos == (steps - 1))
    if not last_mask.any():
        return float("nan"), 0.0
    probs_last   = torch.softmax(last_logits[last_mask], dim=1)
    targets_last = batch.y[last_mask]
    mse_last = criterion(probs_last, targets_last).item()
    acc_last = (probs_last.argmax(dim=1) == targets_last.argmax(dim=1)).float().mean().item()
    return mse_last, acc_last


def main():
    ap = argparse.ArgumentParser()

    # --- data shape ---
    ap.add_argument("--train_len", type=int, default=10, help="fixed chain length for TRAIN")
    ap.add_argument("--val_len",   type=int, default=200, help="(unused if --val_lengths is set)")

    # NEW: evaluate on multiple validation lengths
    ap.add_argument("--val_lengths", type=int, nargs="+",
                    default=[25, 50, 100, 125, 150, 175, 200],
                    help="List of validation lengths; overrides --val_len.")

    # --- graph topology flags (wired into data_chains_flex) ---
    ap.add_argument(
        "--identity_adj",
        type=str2bool,
        default=False,
        help="Use identity adjacency (self-loops only). If true, 'directed' is ignored. [default: False]"
    )
    ap.add_argument(
        "--directed",
        type=str2bool,
        default=True,
        help="Use directed forward edges (i->i+1). False -> undirected i<->i+1. Ignored if identity_adj=True. [default: True]"
    )
    ap.add_argument(
        "--attach_dense_adj",
        type=str2bool,
        default=False,
        help="Also attach dense A to data.A (not used by ParityGCN; for experiments). [default: False]"
    )
    ap.add_argument(
        "--dense_add_self_loops",
        type=str2bool,
        default=False,
        help="When building dense A, add I before normalization. [default: False]"
    )
    ap.add_argument("--dense_normalize", type=str, default="none", choices=["none", "sym", "row"],
                    help="Normalization for dense A if attached (not used by ParityGCN).")

    # --- dataset sizes & batching ---
    ap.add_argument("--train_mode", type=str, default="all", choices=["all", "random"],
                    help="Train on all 2^L sequences or a random subset.")
    ap.add_argument("--train_samples", type=int, default=256,
                    help="Used only when train_mode='random'.")
    ap.add_argument("--val_samples", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle train loader.")

    # --- model/opt ---
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.0, help="If >0, clip grad values to this magnitude.")
    ap.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"])
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- TRAIN loader (ignore returned val loader) ----
    train_loader, _ = make_loaders(
        train_L=args.train_len, val_L=args.train_len, batch_size=args.batch_size,
        train_mode=args.train_mode, train_samples=args.train_samples, val_samples=args.val_samples,
        shuffle=args.shuffle,
        identity_adj=args.identity_adj,
        directed=args.directed,
        attach_dense_adj=args.attach_dense_adj,
        dense_add_self_loops=args.dense_add_self_loops,
        dense_normalize=args.dense_normalize,
    )

    # ---- Build one VAL loader per requested length ----
    val_lengths = list(sorted(set(args.val_lengths)))
    val_loaders = {}
    for L in val_lengths:
        val_ds = build_eval_dataset_fixed_length(
            L,
            num_samples=args.val_samples,
            identity_adj=args.identity_adj,
            directed=args.directed,
            attach_dense_adj=args.attach_dense_adj,
            dense_add_self_loops=args.dense_add_self_loops,
            dense_normalize=args.dense_normalize,
        )
        val_loaders[L] = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ---- model / opt ----
    # NOTE: ParityGCN uses GCNConv(add_self_loops=False, normalize=False) internally
    net = IGNN_finite(in_dim=2, hidden=args.hidden, out_dim=2, activation=args.activation).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()

    train_mse_hist = []
    val_mse_hist = {L: [] for L in val_lengths}
    val_acc_hist = {L: [] for L in val_lengths}

    for epoch in range(1, args.epochs + 1):
        # ----- TRAIN -----
        net.train()
        epoch_loss, nb = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss, _ = train_step_mse_iter(net, batch, steps=args.train_len, criterion=criterion)
            loss.backward()
            if args.clip and args.clip > 0:
                clip_grad_value_(net.parameters(), args.clip)
            opt.step()
            epoch_loss += loss.item(); nb += 1

        train_mse = epoch_loss / max(nb, 1)
        train_mse_hist.append(train_mse)

        # ----- VAL (for each requested length, last node only) -----
        net.eval()
        for L, loader in val_loaders.items():
            v_losses, v_accs = [], []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    mse_last, acc_last = last_node_metrics_mse(net, batch, steps=L, criterion=criterion)
                    if not (mse_last != mse_last):  # filter NaN
                        v_losses.append(mse_last)
                    v_accs.append(acc_last)
            val_mse = float(np.mean(v_losses)) if v_losses else float("nan")
            val_acc = float(np.mean(v_accs)) if v_accs else 0.0
            val_mse_hist[L].append(val_mse)
            val_acc_hist[L].append(val_acc)

        if epoch % 50 == 0 or epoch == 1:
            topo = "identity" if args.identity_adj else ("directed" if args.directed else "undirected")
            msg = f"Epoch {epoch:4d} [{topo}] | Train MSE {train_mse:.6f}"
            for L in val_lengths:
                msg += f" | Val@{L} Acc {val_acc_hist[L][-1]:.3f} (MSE {val_mse_hist[L][-1]:.6f})"
            print(msg)

    # ---- plot loss curve (train) ----
    plt.figure(figsize=(8,5))
    plt.plot(train_mse_hist, label="Train MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Training MSE (GCN parity)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("loss_curve_flex_train.png")
    print("Saved training loss curve to loss_curve_flex_train.png")

    # ---- plot validation accuracy per length ----
    plt.figure(figsize=(9,6))
    for L in val_lengths:
        plt.plot(val_acc_hist[L], label=f"Len {L}")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (last node)")
    plt.title("Validation Accuracy vs Epoch (by length)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("val_acc_by_length.png")
    print("Saved validation accuracy curves to val_acc_by_length.png")


if __name__ == "__main__":
    main()
