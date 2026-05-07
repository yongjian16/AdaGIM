"""Mini-batch PyG DataLoader training with validation early-stop."""
from pathlib import Path
from typing import Any, Dict, List

import torch

from .base import build_loss
from metrics import get as get_metric, higher_is_better
from utils import write_metrics_csv, save_yaml, plot_metrics_png

_DIAG_FIELDS = (
    "L_Psi", "L_g_eff", "D_hat", "R_hat",
    "expected_halt_time", "opt_out_rate", "halt_entropy",
    "avg_iters", "picard_residual",
    "traj_norm_t1", "traj_norm_tT", "traj_contraction",
)

@torch.no_grad()
def _plot_argmax_dist_pyg(model, loader, device, log_dir: Path, epoch, T: int) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    argmax_all = []
    pis_sum = None       # Σ_v π_v^t              (T,)
    pis_sq_sum = None    # Σ_v (π_v^t)^2          (T,)  — for std across nodes
    n_nodes_total = 0
    for batch in loader:
        batch = batch.to(device)
        _ = model.forward_pyg(batch)
        p = model.cores[-1].last_pis
        if p is None:
            return
        p_cpu = p.detach().cpu()                                # (T, N)
        argmax_all.append(p_cpu.argmax(dim=0).numpy() + 1)      # halt step in {1..T}
        pis_sum = p_cpu.sum(dim=1) if pis_sum is None else pis_sum + p_cpu.sum(dim=1)
        sq = (p_cpu ** 2).sum(dim=1)
        pis_sq_sum = sq if pis_sq_sum is None else pis_sq_sum + sq
        n_nodes_total += p_cpu.shape[1]
    if not argmax_all:
        return

    argmax_all = np.concatenate(argmax_all)
    N = max(n_nodes_total, 1)
    mean_pi = (pis_sum / N).numpy()                              # (T,)
    var_pi = ((pis_sq_sum / N).numpy() - mean_pi ** 2).clip(min=0.0)
    std_pi = np.sqrt(var_pi)                                     # (T,)
    counts = np.bincount(argmax_all, minlength=T + 1)            # bins 0..T; bin 0 always 0
    fracs = counts / max(counts.sum(), 1)
    fracs_std = np.sqrt(fracs * (1.0 - fracs) / N)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xs = np.arange(1, T + 1)

    ax = axes[0]
    ax.bar(xs, fracs[1:], yerr=fracs_std[1:], capsize=4,
           color="steelblue", edgecolor="black", alpha=0.85,
           error_kw={"ecolor": "black", "elinewidth": 1.0})
    for t in xs:
        if fracs[t] > 0.01:
            ax.text(t, fracs[t] + max(fracs_std[t], 0.005),
                    f"{fracs[t]:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xlabel("argmax_t (per-node halt step)")
    ax.set_ylabel("fraction of nodes (±1 binomial-prop std)")
    ax.set_ylim(0, max(fracs[1:].max() * 1.15, 0.05))
    ax.set_title(f"(a) per-node argmax  t∈{{1..{T}}}")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(xs, mean_pi, yerr=std_pi, capsize=4,
           color="seagreen", edgecolor="black", alpha=0.85,
           error_kw={"ecolor": "black", "elinewidth": 1.0})
    for t, (pv, sv) in zip(xs, zip(mean_pi, std_pi)):
        if pv > 0.01:
            ax.text(t, pv + sv + 0.003, f"{pv:.3f}±{sv:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xlabel("halt step t")
    ax.set_ylabel("mean π^t (±1 std across nodes)")
    ax.set_ylim(0, max((mean_pi + std_pi).max() * 1.10, 0.1))
    ax.set_title("(b) per-step π distribution\n(mean ± std across nodes)")
    ax.grid(axis="y", alpha=0.3)

    epoch_tag = f"{epoch:03d}" if isinstance(epoch, int) else str(epoch)
    fig.suptitle(f"Argmax distribution at epoch {epoch} (n_nodes={n_nodes_total:,})",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(log_dir / f"argmax_epoch_{epoch_tag}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    if epoch == "final":
        np.save(log_dir / "argmax_test.npy", argmax_all)
        np.save(log_dir / "mean_pi_test.npy", mean_pi)
        np.save(log_dir / "std_pi_test.npy", std_pi)

@torch.no_grad()
def _evaluate(model, loader, metric_fn, device, task_type: str, out_dim: int = 1) -> float:
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model.forward_pyg(batch)
        if task_type == "node_classification":
            y = batch.y.view(-1)
        elif task_type == "graph_regression":
            y = batch.y.view(-1) if out_dim == 1 else batch.y
        else:
            y = batch.y
        all_logits.append(logits.detach())
        all_y.append(y.detach())
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    return metric_fn(logits, y)

def train_pyg(dataset_dict: Dict[str, Any], model, cfg: Dict[str, Any], log_dir: Path) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = dataset_dict["train_loader"]
    val_loader = dataset_dict["val_loader"]
    test_loader = dataset_dict["test_loader"]
    meta = dataset_dict["meta"]

    loss_fn = build_loss(meta["task_type"])
    metric_fn = get_metric(meta["metric"])
    higher = higher_is_better(meta["metric"])

    epochs = int(cfg.get("epochs", 200))
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("wd", 0.0))
    patience = int(cfg.get("patience", 30))
    min_delta = float(cfg.get("min_delta", 1e-4))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler_kind = str(cfg.get("scheduler", "none")).lower()
    sched_min_lr = float(cfg.get("scheduler_min_lr", 1e-5))
    if scheduler_kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max" if higher else "min",
            factor=float(cfg.get("scheduler_factor", 0.5)),
            patience=int(cfg.get("scheduler_patience", 10)),
            min_lr=sched_min_lr,
        )
    elif scheduler_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=int(cfg.get("scheduler_T_max", epochs)),
            eta_min=sched_min_lr,
        )
    elif scheduler_kind == "warmup_cosine":
        warmup = int(cfg.get("scheduler_warmup_epochs", 5))
        T_max = int(cfg.get("scheduler_T_max", epochs))
        import math as _math
        def _lr_lambda(epoch_idx: int) -> float:
            if epoch_idx < warmup:
                return float(epoch_idx + 1) / float(max(warmup, 1))
            progress = float(epoch_idx - warmup) / float(max(T_max - warmup, 1))
            cos_factor = 0.5 * (1.0 + _math.cos(_math.pi * min(progress, 1.0)))
            return max(sched_min_lr / lr, cos_factor)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    elif scheduler_kind == "step":
        step_size = int(cfg.get("scheduler_step_size", 30))
        factor = float(cfg.get("scheduler_factor", 0.5))
        def _step_lambda(epoch_idx: int) -> float:
            n_drops = epoch_idx // max(step_size, 1)
            return max(factor ** n_drops, sched_min_lr / lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _step_lambda)
    elif scheduler_kind == "multistep":
        milestones = [int(m) for m in cfg.get("scheduler_milestones", [60, 120])]
        factor = float(cfg.get("scheduler_factor", 0.5))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=factor)
    elif scheduler_kind in ("none", "off", "no"):
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler {scheduler_kind!r}; "
                         f"use 'none', 'plateau', 'cosine', 'warmup_cosine', 'step', or 'multistep'")

    rows: List[Dict[str, float]] = []
    best_val = -float("inf") if higher else float("inf")
    best_test = float("nan")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        diag_sum: Dict[str, float] = {k: 0.0 for k in _DIAG_FIELDS}
        diag_count: Dict[str, int] = {k: 0 for k in _DIAG_FIELDS}
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model.forward_pyg(batch)

            if meta["task_type"] == "node_classification":
                y = batch.y.view(-1)
            elif meta["task_type"] == "graph_regression":
                y = batch.y.view(-1) if int(meta.get("out_dim", 1)) == 1 else batch.y
            else:
                y = batch.y

            ponder_cost = getattr(model, "last_ponder_cost", None)
            if getattr(model, "skip_task_loss", False):
                if ponder_cost is None:
                    raise RuntimeError(
                        f"{type(model).__name__} sets skip_task_loss=True but did not expose last_ponder_cost."
                    )
                loss = ponder_cost / max(batch.num_nodes, 1)
            else:
                loss = loss_fn(logits, y)
                if ponder_cost is not None:
                    loss = loss + ponder_cost / max(batch.num_nodes, 1)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * batch.num_graphs
            total_n += batch.num_graphs

            for name in _DIAG_FIELDS:
                v = getattr(model, f"last_{name}", None)
                if v is not None:
                    diag_sum[name] += float(v)
                    diag_count[name] += 1

        train_loss = total_loss / max(total_n, 1)
        val_m = _evaluate(model, val_loader, metric_fn, device, meta["task_type"], int(meta.get("out_dim", 1)))
        improved = ((higher and val_m > best_val + min_delta) or
                    ((not higher) and val_m < best_val - min_delta))
        if improved:
            best_val = val_m
            best_test = _evaluate(model, test_loader, metric_fn, device, meta["task_type"], int(meta.get("out_dim", 1)))
            no_improve = 0
        else:
            no_improve += 1

        row = {"epoch": epoch, "train_loss": train_loss, "val_metric": val_m,
               "test_metric_at_best_val": best_test,
               "lr": float(opt.param_groups[0]["lr"])}
        for name in _DIAG_FIELDS:
            if diag_count[name] > 0:
                row[name] = diag_sum[name] / diag_count[name]
        rows.append(row)

        if scheduler is not None:
            if scheduler_kind == "plateau":
                scheduler.step(val_m)
            else:
                scheduler.step()
        write_metrics_csv(rows, log_dir / "metrics.csv")
        plot_metrics_png(rows, log_dir / "metrics.png")
        argmax_every = int(cfg.get("eval_argmax_every", 0))
        if argmax_every > 0 and epoch % argmax_every == 0 and hasattr(model, "cores"):
            try:
                _plot_argmax_dist_pyg(model, val_loader, device, log_dir, epoch, T=int(cfg.get("T", 10)))
            except Exception as e:
                print(f"[pyg] argmax-dist plot failed at epoch {epoch}: {e}")
        msg = (f"[pyg] epoch={epoch:4d} train_loss={train_loss:.4f} val_{meta['metric']}={val_m:.4f} "
               f"test_{meta['metric']}={best_test:.4f}")
        diag_msg = " ".join(
            f"{name}={diag_sum[name] / diag_count[name]:.4f}"
            for name in ("L_Psi", "L_g_eff", "expected_halt_time", "opt_out_rate", "halt_entropy",
                         "traj_norm_t1", "traj_norm_tT")
            if diag_count[name] > 0
        )
        if diag_msg:
            msg += " | " + diag_msg
        print(msg)

        if no_improve >= patience:
            print(f"[pyg] early stop at epoch {epoch}")
            break

    write_metrics_csv(rows, log_dir / "metrics.csv")
    save_yaml({"best_val_metric": float(best_val), "test_metric_at_best_val": float(best_test),
               "metric_name": meta["metric"]}, log_dir / "summary.yaml")

    if int(cfg.get("eval_argmax_every", 0)) > 0 and hasattr(model, "cores"):
        try:
            _plot_argmax_dist_pyg(model, test_loader, device, log_dir,
                                  epoch="final", T=int(cfg.get("T", 10)))
        except Exception as e:
            print(f"[pyg] final argmax-dist plot failed: {e}")

    return {"best_val_metric": float(best_val), "test_metric_at_best_val": float(best_test),
            "metric_name": meta["metric"]}
