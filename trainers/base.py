import torch
import torch.nn.functional as F

def build_loss(task_type: str):
    if task_type == "node_classification_parity":
        def _mse(logits, y):
            return F.mse_loss(logits, y.float())
        return _mse
    if task_type in ("graph_classification_multilabel", "graph_classification_binary"):
        def _bce(logits, y):
            y = y.float()
            if y.dim() < logits.dim():
                y = y.view(logits.shape)
            return F.binary_cross_entropy_with_logits(logits, y)
        return _bce
    if task_type == "graph_regression":
        def _l1(logits, y):
            y = y.float()
            if logits.dim() > y.dim():
                logits = logits.squeeze(-1)
            elif y.dim() > logits.dim():
                y = y.squeeze(-1)
            return F.l1_loss(logits, y)
        return _l1
    if task_type in ("node_classification", "node_classification_masked"):
        def _ce(logits, y):
            return F.cross_entropy(logits, y.long().view(-1))
        return _ce
    raise ValueError(f"Unknown task_type {task_type!r}")
