import argparse
import os
import torch

from data.kl_chain_parity_task import generate_chain_parity_KL_2powK

from models import ParityEIGNN, ParityIGNN
from utils import (
    set_seeds, make_loss, ensure_torch_sparse,
    make_out_dir, save_args, init_metrics, append_metrics,
    acc_by_pos, plot_acc_by_pos, plot_embedding_scatter_pca2_with_boundary, plot_loss_curve,
    build_torch_adj_from_sp,
)



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default="eignn", choices=["eignn", "ignn"])
    ap.add_argument("--loss", type=str, default="mse", choices=["ce", "mse"])

    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--L1", type=int, required=True)
    ap.add_argument("--L2", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=100)

    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pad", type=int, default=0, choices=[0, 1])
    ap.add_argument("--max_graphs", type=int, default=None)

    # NEW: adjacency mode (1/2/3)
    ap.add_argument(
        "--adj_mode",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="1: directed raw A; 2: directed (A+I) row-normalized; 3: undirected (A+A^T+I) symmetric-normalized"
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

    # plotting
    ap.add_argument("--plot_max_points", type=int, default=30000)
    ap.add_argument("--no_boundary", action="store_true")

    args = ap.parse_args()
    assert args.L2 >= args.L1
    assert args.K <= args.L1 and args.K <= args.L2

    out_dir = make_out_dir(args)
    print("Results dir:", out_dir)
    save_args(out_dir, args)
    metrics_path = init_metrics(out_dir)

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------------
    # Data
    # IMPORTANT:
    #   Always generate the BASE directed chain (bidirectional=False),
    #   then adj_mode==3 will symmetrize it.
    # -----------------------------
    X_tr, Y_tr, _, A_sp_tr_raw, _ = generate_chain_parity_KL_2powK(
        K=args.K, L=args.L1,
        bidirectional=False,
        pad_value=args.pad,
        supervise="all_prefix",
        max_graphs=args.max_graphs,
    )
    X_te, Y_te, _, A_sp_te_raw, _ = generate_chain_parity_KL_2powK(
        K=args.K, L=args.L2,
        bidirectional=False,
        pad_value=args.pad,
        supervise="all_prefix",
        max_graphs=args.max_graphs,
    )

    X_tr, Y_tr = X_tr.to(device), Y_tr.to(device)
    X_te, Y_te = X_te.to(device), Y_te.to(device)

    # -----------------------------
    # Build processed adjacencies
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

    loss_hist = []

    # -----------------------------
    # Train
    # -----------------------------
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

        loss_hist.append(float(loss.item()))

        if epoch % 50 == 0 or epoch == 1:
            net.eval()
            with torch.no_grad():
                if args.model == "eignn":
                    net.set_graph(A_te, A_sp_te)
                    logits_te = net(X_te)
                else:
                    logits_te = net(X_te, A_te)

                last_acc = acc_by_pos(logits_te, Y_te, args.L2)[-1].item()

            print(f"Epoch {epoch:4d} | Train {args.loss.upper()} {loss.item():.6f} | Test acc@L2(last) {last_acc:.3f}")
            append_metrics(metrics_path, epoch, float(loss.item()), float(last_acc))

    # -----------------------------
    # Final eval + plots
    # -----------------------------
    net.eval()
    with torch.no_grad():
        if args.model == "eignn":
            net.set_graph(A_te, A_sp_te)
            logits_te = net(X_te)
            Z_te = net.encode(X_te)
        else:
            logits_te = net(X_te, A_te)
            Z_te = net.encode(X_te, A_te)

    adj_name = {1: "dir", 2: "dir_row_norm_sl", 3: "undir_sym_norm_sl"}[args.adj_mode]

    out_acc = os.path.join(out_dir, "acc_by_pos.png")
    plot_acc_by_pos(
        logits=logits_te,
        Y_true_1hot=Y_te,
        L=args.L2,
        L1=args.L1,
        L2=args.L2,
        title=f"Train L1={args.L1} → Test L2={args.L2} | model={args.model} | K={args.K} | adj={adj_name} | loss={args.loss}",
        out_path=out_acc,
    )
    print("Saved:", out_acc)

    out_scat = os.path.join(out_dir, "scatter_pca2_boundary.png")
    plot_embedding_scatter_pca2_with_boundary(
        Z=Z_te,
        Y_true_1hot=Y_te,
        head=net.B,
        title=f"Embeddings (PCA2) + boundary | model={args.model} | K={args.K} | L2={args.L2} | adj={adj_name} | loss={args.loss}",
        out_path=out_scat,
        max_points=args.plot_max_points,
        plot_boundary=(not args.no_boundary),
    )
    print("Saved:", out_scat)

    out_loss = os.path.join(out_dir, "loss_curve.png")
    plot_loss_curve(
        loss_hist=loss_hist,
        title=f"Training Loss Curve | model={args.model} | K={args.K} | L1={args.L1} | adj={adj_name} | loss={args.loss}",
        out_path=out_loss,
    )
    print("Saved:", out_loss)


if __name__ == "__main__":
    main()
