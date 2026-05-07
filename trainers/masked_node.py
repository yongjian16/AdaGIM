"""Masked node-classification trainer for Planetoid-style datasets (Cora, Citeseer, Pubmed)."""
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .base import build_loss
from metrics import get as get_metric, higher_is_better
from utils import write_metrics_csv, save_yaml, plot_metrics_png

@torch.no_grad()
def _eval_masked(model, loader, device, mask_attr: str) -> float:
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        logits = model.forward_pyg(batch)
        mask = getattr(batch, mask_attr)
        pred = logits[mask].argmax(dim=-1)
        y = batch.y[mask]
        return float((pred == y).float().mean().item())
    return float("nan")

def train_masked_node(dataset_dict: Dict[str, Any], model, cfg: Dict[str, Any], log_dir: Path) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = dataset_dict["train_loader"]  # contains the single full graph
    meta = dataset_dict["meta"]

    epochs = int(cfg.get("epochs", 200))
    lr = float(cfg.get("lr", 0.01))
    wd = float(cfg.get("wd", 5e-4))
    patience = int(cfg.get("patience", 50))
    min_delta = float(cfg.get("min_delta", 1e-4))

    if hasattr(model, "params_imp") and hasattr(model, "params_exp"):
        imp_lr = float(cfg.get("imp_lr", lr))
        exp_lr = float(cfg.get("exp_lr", lr))
        imp_wd = float(cfg.get("imp_wd", wd))
        exp_wd = float(cfg.get("exp_wd", wd))
        opt = torch.optim.Adam([
            {"params": model.params_imp, "lr": imp_lr, "weight_decay": imp_wd},
            {"params": model.params_exp, "lr": exp_lr, "weight_decay": exp_wd},
        ])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    rows: List[Dict[str, float]] = []
    best_val, best_test, no_improve = -float("inf"), float("nan"), 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model.forward_pyg(batch)
            mask = batch.train_mask
            loss = F.cross_entropy(logits[mask], batch.y[mask].long())
            ponder = getattr(model, "last_ponder_cost", None)
            if ponder is not None:
                loss = loss + ponder / max(batch.num_nodes, 1)
            loss.backward()
            opt.step()
            train_loss = float(loss.item())
            train_acc = float((logits[mask].argmax(-1) == batch.y[mask]).float().mean().item())

        val_acc = _eval_masked(model, loader, device, "val_mask")
        improved = val_acc > best_val + min_delta  # strict-improvement threshold prevents noise resetting patience
        if improved:
            best_val = val_acc
            best_test = _eval_masked(model, loader, device, "test_mask")
            no_improve = 0
        else:
            no_improve += 1

        rows.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                     "val_acc": val_acc, "test_acc_at_best_val": best_test})
        write_metrics_csv(rows, log_dir / "metrics.csv")
        plot_metrics_png(rows, log_dir / "metrics.png")
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"[masked_node] epoch={epoch:4d} train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} val={val_acc:.4f} test_at_best={best_test:.4f}")

        if no_improve >= patience:
            print(f"[masked_node] early stop at epoch {epoch}")
            break

    write_metrics_csv(rows, log_dir / "metrics.csv")
    save_yaml({"best_val_metric": float(best_val), "test_metric_at_best_val": float(best_test),
               "metric_name": "node_acc"}, log_dir / "summary.yaml")
    return {"best_val_metric": float(best_val), "test_metric_at_best_val": float(best_test),
            "metric_name": "node_acc"}
