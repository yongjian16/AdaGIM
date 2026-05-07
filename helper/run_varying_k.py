import argparse
import csv
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple

import torch

from data.kl_chain_parity_task import generate_chain_parity_KL_2powK
from models import ParityEIGNN, ParityIGNN
from utils import (
    set_seeds, make_loss, acc_by_pos,
    build_torch_adj_from_sp, save_args,
)


def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _adj_name(adj_mode: int) -> str:
    return {1: "dir", 2: "dir_row_norm_sl", 3: "undir_sym_norm_sl"}[adj_mode]


def _parse_seeds(seeds_str: Optional[str], base_seed: int, num_seeds: int) -> List[int]:
    """
    seeds_str examples:
      "0,1,2,3,4"
      "0-4"  (inclusive)
    If seeds_str is None, returns [base_seed, base_seed+1, ..., base_seed+num_seeds-1]
    """
    if seeds_str is None:
        return list(range(base_seed, base_seed + num_seeds))

    s = seeds_str.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
        if b < a:
            raise ValueError("seeds range must be like a-b with b>=a")
        return list(range(a, b + 1))

    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _make_base_out_dir(args, seeds: List[int]) -> str:
    if len(seeds) <= 8:
        seeds_tag = "seeds_" + "_".join(str(s) for s in seeds)
    else:
        seeds_tag = f"seeds_{min(seeds)}to{max(seeds)}_n{len(seeds)}"

    tag = (
        f"{seeds_tag}"
        f"_sweepK{args.K_start}to{args.K_end}"
        f"_L{args.L}"
        f"_model{args.model}"
        f"_loss{args.loss}"
        f"_adj{_adj_name(args.adj_mode)}"
        f"_hid{args.hidden}"
        f"_ep{args.epochs}"
        f"_lr{args.lr}"
        f"_wd{args.wd}"
        f"_pad{args.pad}"
    )
    if args.max_graphs is not None:
        tag += f"_maxg{args.max_graphs}"

    if args.model == "eignn":
        tag += f"_gamma{args.gamma}_g{args.g_type}_thr{args.threshold}_it{args.max_iter}"
    else:
        tag += f"_kappa{args.kappa}_fw{args.fw_mitr}_bw{args.bw_mitr}_phi{args.phi}"
        if args.b_direct:
            tag += "_bdirect"

    return _mkdir(os.path.join(args.out_root, tag))


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if len(vals) == 0:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, var ** 0.5


def run_one(args, K: int, seed: int, device: torch.device, out_dir: str) -> Dict[str, Any]:
    set_seeds(seed)

    # -----------------------------
    # Data (train/test at same L)
    # -----------------------------
    X_tr, Y_tr, _, A_sp_tr_raw, _ = generate_chain_parity_KL_2powK(
        K=K, L=args.L,
        bidirectional=False,
        pad_value=args.pad,
        supervise="all_prefix",
        max_graphs=args.max_graphs,
    )
    X_te, Y_te, _, A_sp_te_raw, _ = generate_chain_parity_KL_2powK(
        K=K, L=args.L,
        bidirectional=False,
        pad_value=args.pad,
        supervise="all_prefix",
        max_graphs=args.max_graphs,
    )

    X_tr, Y_tr = X_tr.to(device), Y_tr.to(device)
    X_te, Y_te = X_te.to(device), Y_te.to(device)

    # -----------------------------
    # Adjacency
    # -----------------------------
    A_tr, A_sp_tr = build_torch_adj_from_sp(A_sp_tr_raw, device, args.adj_mode)
    A_te, A_sp_te = build_torch_adj_from_sp(A_sp_te_raw, device, args.adj_mode)

    in_dim = X_tr.size(1)

    # -----------------------------
    # Model
    # -----------------------------
    if args.model == "eignn":
        net = ParityEIGNN(
            adj=A_tr, sp_adj=A_sp_tr,
            in_dim=in_dim, hidden=args.hidden, out_dim=2,
            threshold=args.threshold, max_iter=args.max_iter,
            gamma=args.gamma, g_type=args.g_type
        ).to(device)
    else:
        net = ParityIGNN(
            in_dim=in_dim, hidden=args.hidden, out_dim=2,
            kappa=args.kappa, b_direct=args.b_direct,
            fw_mitr=args.fw_mitr, bw_mitr=args.bw_mitr,
            A_rho=args.A_rho, phi=args.phi
        ).to(device)

    loss_fn = make_loss(args.loss)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    best_last_acc = -1.0
    best_epoch = -1
    final_loss = None
    last_acc = -1.0

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        net.train()
        opt.zero_grad()

        if args.model == "eignn":
            net.set_graph(A_tr, A_sp_tr)
            logits = net(X_tr)
        else:
            logits = net(X_tr, A_tr)

        loss = loss_fn(logits, Y_tr)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

        do_eval = (args.eval_every > 0 and (epoch % args.eval_every == 0)) or (epoch == args.epochs)
        if do_eval:
            net.eval()
            with torch.no_grad():
                if args.model == "eignn":
                    net.set_graph(A_te, A_sp_te)
                    logits_te = net(X_te)
                else:
                    logits_te = net(X_te, A_te)

                acc_vec = acc_by_pos(logits_te, Y_te, args.L)
                last_acc = float(acc_vec[-1].item())

            if last_acc > best_last_acc:
                best_last_acc = last_acc
                best_epoch = epoch

        if args.verbose and do_eval:
            print(f"[seed={seed:3d} K={K:3d}] epoch {epoch:4d} | loss {final_loss:.6f} | acc@L(last) {last_acc:.4f}")

    sec = time.time() - t0

    # Final eval (final model)
    net.eval()
    with torch.no_grad():
        if args.model == "eignn":
            net.set_graph(A_te, A_sp_te)
            logits_te = net(X_te)
        else:
            logits_te = net(X_te, A_te)

        acc_vec = acc_by_pos(logits_te, Y_te, args.L).detach().cpu()
        final_last_acc = float(acc_vec[-1].item())
        mean_acc = float(acc_vec.mean().item())

    _mkdir(out_dir)
    torch.save({"acc_by_pos": acc_vec}, os.path.join(out_dir, "acc_by_pos.pt"))
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": int(seed),
                "K": int(K),
                "L": int(args.L),
                "final_last_acc": float(final_last_acc),
                "mean_acc": float(mean_acc),
                "best_last_acc": float(best_last_acc),
                "best_epoch": int(best_epoch),
                "final_loss": float(final_loss) if final_loss is not None else None,
                "seconds": float(sec),
            },
            f,
            indent=2,
        )

    # cleanup
    del net, X_tr, Y_tr, X_te, Y_te, A_tr, A_te, A_sp_tr, A_sp_te
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "seed": seed,
        "K": K,
        "L": args.L,
        "final_last_acc": final_last_acc,
        "mean_acc": mean_acc,
        "best_last_acc": float(best_last_acc),
        "best_epoch": int(best_epoch),
        "final_loss": float(final_loss) if final_loss is not None else None,
        "seconds": float(sec),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--K_start", type=int, default=2)
    ap.add_argument("--K_end", type=int, default=None, help="Default: K_end=L")

    # multi-seed
    ap.add_argument("--seeds", type=str, default=None,
                    help='Either "0,1,2,3,4" or "0-4". If omitted, uses base_seed..base_seed+num_seeds-1')
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--num_seeds", type=int, default=3)

    # training knobs
    ap.add_argument("--model", type=str, default="eignn", choices=["eignn", "ignn"])
    ap.add_argument("--loss", type=str, default="mse", choices=["ce", "mse"])
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0)

    ap.add_argument("--pad", type=int, default=0, choices=[0, 1])
    ap.add_argument("--max_graphs", type=int, default=None)

    ap.add_argument(
        "--adj_mode",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="1: directed raw A; 2: directed (A+I) row-normalized; 3: undirected (A+A^T+I) symmetric-normalized",
    )

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

    # output/logging
    ap.add_argument("--out_root", type=str, default="results")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if args.K_end is None:
        args.K_end = args.L
    assert 1 <= args.K_start <= args.K_end
    assert args.K_end <= args.L

    seeds = _parse_seeds(args.seeds, args.base_seed, args.num_seeds)

    base_out = _make_base_out_dir(args, seeds)
    print("Multi-seed sweep dir:", base_out)

    save_args(base_out, args)
    with open(os.path.join(base_out, "seeds.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # per (seed, K)
    by_seed_rows: List[Dict[str, Any]] = []
    # aggregated per K
    avg_rows: List[Dict[str, Any]] = []

    for K in range(args.K_start, args.K_end + 1):
        final_last_accs = []
        mean_accs = []
        best_last_accs = []
        best_epochs = []

        for seed in seeds:
            out_dir = os.path.join(base_out, f"seed{seed:03d}", f"K{K:03d}_L{args.L:03d}")
            row = run_one(args, K=K, seed=seed, device=device, out_dir=out_dir)
            by_seed_rows.append(row)

            final_last_accs.append(float(row["final_last_acc"]))
            mean_accs.append(float(row["mean_acc"]))
            best_last_accs.append(float(row["best_last_acc"]))
            best_epochs.append(float(row["best_epoch"]))

            print(f"Done seed={seed} K={K}: final={row['final_last_acc']:.4f} | best={row['best_last_acc']:.4f}")

        m_final, s_final = _mean_std(final_last_accs)
        m_mean, s_mean = _mean_std(mean_accs)
        m_best, s_best = _mean_std(best_last_accs)
        m_be, s_be = _mean_std(best_epochs)

        avg_rows.append(
            {
                "K": K,
                "L": args.L,
                "n_seeds": len(seeds),

                "avg_final_last_acc": m_final,
                "std_final_last_acc": s_final,

                "avg_best_last_acc": m_best,
                "std_best_last_acc": s_best,

                "avg_mean_acc": m_mean,
                "std_mean_acc": s_mean,

                "avg_best_epoch": m_be,
                "std_best_epoch": s_be,
            }
        )

    # per-seed CSV
    by_seed_path = os.path.join(base_out, "summary_by_seed.csv")
    with open(by_seed_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "K", "L",
                "final_last_acc", "mean_acc",
                "best_last_acc", "best_epoch",
                "final_loss", "seconds",
            ],
        )
        w.writeheader()
        for r in by_seed_rows:
            w.writerow(r)
    print("Saved:", by_seed_path)

    # averaged CSV (per K)
    avg_path = os.path.join(base_out, "summary_avg.csv")
    with open(avg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "K", "L", "n_seeds",
                "avg_final_last_acc", "std_final_last_acc",
                "avg_best_last_acc", "std_best_last_acc",
                "avg_mean_acc", "std_mean_acc",
                "avg_best_epoch", "std_best_epoch",
            ],
        )
        w.writeheader()
        for r in avg_rows:
            w.writerow(r)
    print("Saved:", avg_path)

    # -----------------------------
    # NEW: store seed-averaged results as LISTS (indexed by K)
    # (average over seeds only; NOT averaged over K)
    # -----------------------------
    K_list = [int(r["K"]) for r in avg_rows]

    avg_final_last_acc_list = [float(r["avg_final_last_acc"]) for r in avg_rows]
    std_final_last_acc_list = [float(r["std_final_last_acc"]) for r in avg_rows]

    avg_best_last_acc_list = [float(r["avg_best_last_acc"]) for r in avg_rows]
    std_best_last_acc_list = [float(r["std_best_last_acc"]) for r in avg_rows]

    avg_mean_acc_list = [float(r["avg_mean_acc"]) for r in avg_rows]
    std_mean_acc_list = [float(r["std_mean_acc"]) for r in avg_rows]

    avg_best_epoch_list = [float(r["avg_best_epoch"]) for r in avg_rows]
    std_best_epoch_list = [float(r["std_best_epoch"]) for r in avg_rows]

    lists_path = os.path.join(base_out, "avg_lists.json")
    with open(lists_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "L": int(args.L),
                "n_seeds": int(len(seeds)),
                "seeds": seeds,
                "K_list": K_list,

                "avg_final_last_acc": avg_final_last_acc_list,
                "std_final_last_acc": std_final_last_acc_list,

                "avg_best_last_acc": avg_best_last_acc_list,
                "std_best_last_acc": std_best_last_acc_list,

                "avg_mean_acc": avg_mean_acc_list,
                "std_mean_acc": std_mean_acc_list,

                "avg_best_epoch": avg_best_epoch_list,
                "std_best_epoch": std_best_epoch_list,
            },
            f,
            indent=2,
        )
    print("Saved:", lists_path)


if __name__ == "__main__":
    main()
