"""Full-batch synthetic chain-parity training (mirrors existing run_generalize.py flow)."""
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .base import build_loss
from metrics import get as get_metric, higher_is_better
from utils import write_metrics_csv, save_yaml, plot_metrics_png

_DIAG_FIELDS = (
    "L_Psi", "L_g_eff", "D_hat", "R_hat",
    "expected_halt_time", "opt_out_rate", "halt_entropy",
    "avg_iters", "picard_residual",
)

def _model_forward(model, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_synthetic"):
        return model.forward_synthetic(X, A)
    if hasattr(model, "forward_pyg"):
        from torch_geometric.data import Data, Batch
        edge_index = (A.coalesce().indices() if A.is_sparse
                      else A.nonzero(as_tuple=False).T)
        data = Data(x=X, edge_index=edge_index, num_nodes=X.size(0))
        batch = Batch.from_data_list([data]).to(X.device)
        return model.forward_pyg(batch)
    raise RuntimeError(f"{type(model).__name__} has neither forward_synthetic nor forward_pyg.")

def _per_position_acc(logits: torch.Tensor, y_true: torch.Tensor, L_test: int,
                      seen_len: int) -> Dict[str, Any]:
    pred = logits.argmax(dim=-1)
    target = y_true.argmax(dim=-1) if y_true.dim() > 1 else y_true
    correct = (pred == target).float()  # (N,)
    pos = torch.arange(correct.size(0), device=correct.device) % L_test
    acc_per_pos = []
    for p in range(L_test):
        m = (pos == p)
        acc_per_pos.append(float(correct[m].mean().item()) if m.any() else float("nan"))
    seen_mask = pos < seen_len
    unseen_mask = pos >= seen_len
    return {
        "acc_overall": float(correct.mean().item()),
        "acc_seen": float(correct[seen_mask].mean().item()) if seen_mask.any() else float("nan"),
        "acc_unseen": float(correct[unseen_mask].mean().item()) if unseen_mask.any() else float("nan"),
        "acc_per_pos": acc_per_pos,
    }

def train_synthetic(dataset_dict: Dict[str, Any], model, cfg: Dict[str, Any], log_dir: Path) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X_tr = dataset_dict["X_train"].to(device)
    Y_tr = dataset_dict["Y_train"].to(device)
    A_tr = dataset_dict["A_train"].to(device)
    X_te = dataset_dict["X_test"].to(device)
    Y_te = dataset_dict["Y_test"].to(device)
    A_te = dataset_dict["A_test"].to(device)
    meta = dataset_dict["meta"]

    loss_fn = build_loss(meta["task_type"])
    metric_fn = get_metric(meta["metric"])
    higher = higher_is_better(meta["metric"])

    loss_mask = dataset_dict.get("loss_mask_train")
    if loss_mask is not None:
        loss_mask = loss_mask.to(device)

    epochs = int(cfg.get("epochs", 1000))
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("wd", 1e-4))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    import copy
    rows = []
    best_test_metric = -float("inf") if higher else float("inf")
    best_state: Optional[Dict[str, Any]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        if hasattr(model, "set_graph"):
            try:
                model.set_graph(A_tr, dataset_dict.get("A_sp_train"))
            except TypeError:
                model.set_graph(A_tr)

        if hasattr(model, "set_targets"):
            model.set_targets(Y_tr)

        logits = _model_forward(model, X_tr, A_tr)

        ponder_cost = getattr(model, "last_ponder_cost", None)
        if getattr(model, "skip_task_loss", False):
            if ponder_cost is None:
                raise RuntimeError(
                    f"{type(model).__name__} sets skip_task_loss=True but did not expose last_ponder_cost."
                )
            loss = ponder_cost
        else:
            if loss_mask is not None and meta["task_type"] == "node_classification_parity":
                per_node = ((logits - Y_tr.float()) ** 2).mean(dim=-1)        # (N,)
                denom = loss_mask.sum().clamp_min(1.0)
                loss = (per_node * loss_mask).sum() / denom
            else:
                loss = loss_fn(logits, Y_tr)
            if ponder_cost is not None:
                loss = loss + ponder_cost

        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            if hasattr(model, "set_graph"):
                try:
                    model.set_graph(A_te, dataset_dict.get("A_sp_test"))
                except TypeError:
                    model.set_graph(A_te)
            logits_te = _model_forward(model, X_te, A_te)

        test_m = metric_fn(logits_te, Y_te)
        improved = (higher and test_m > best_test_metric) or ((not higher) and test_m < best_test_metric)
        if improved:
            best_test_metric = test_m
            best_state = copy.deepcopy(model.state_dict())

        row = {"epoch": epoch, "train_loss": float(loss.item()), "test_metric": test_m}
        for name in _DIAG_FIELDS:
            v = getattr(model, f"last_{name}", None)
            if v is not None:
                row[name] = float(v)
        rows.append(row)
        write_metrics_csv(rows, log_dir / "metrics.csv")
        plot_metrics_png(rows, log_dir / "metrics.png")
        if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
            msg = f"[synthetic] epoch={epoch:5d} train_loss={loss.item():.6f} test_{meta['metric']}={test_m:.4f}"
            diag_msg = " ".join(
                f"{name}={row[name]:.4f}"
                for name in ("L_Psi", "L_g_eff", "expected_halt_time", "opt_out_rate", "halt_entropy")
                if name in row
            )
            if diag_msg:
                msg += " | " + diag_msg
            print(msg)

    write_metrics_csv(rows, log_dir / "metrics.csv")

    if best_state is not None:
        model.load_state_dict(best_state)

    per_pos: Dict[str, Any] = {}
    L_test = int(meta.get("L_test", 0))
    L_train = int(meta.get("L_train", 0))
    seen_len = int(meta.get("loss_mask_len") or L_train)
    if L_test > 0 and seen_len > 0:
        model.eval()
        with torch.no_grad():
            if hasattr(model, "set_graph"):
                try:
                    model.set_graph(A_te, dataset_dict.get("A_sp_test"))
                except TypeError:
                    model.set_graph(A_te)
            logits_final = _model_forward(model, X_te, A_te)
        per_pos = _per_position_acc(logits_final, Y_te, L_test=L_test, seen_len=seen_len)

    summary = {"best_test_metric": float(best_test_metric), "metric_name": meta["metric"]}
    summary.update({k: v for k, v in per_pos.items() if k != "acc_per_pos"})
    if "acc_per_pos" in per_pos:
        summary["acc_per_pos"] = per_pos["acc_per_pos"]
    save_yaml(summary, log_dir / "summary.yaml")
    return summary
