"""Chain-parity dataset wrapper."""
import importlib.util
from pathlib import Path
from typing import Any, Dict

import torch

_HELPER_TASK = Path(__file__).resolve().parents[1] / "helper" / "data" / "kl_chain_parity_task.py"
_spec = importlib.util.spec_from_file_location("_helper_chain_parity_task", str(_HELPER_TASK))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
generate_chain_parity_KL_2powK = _module.generate_chain_parity_KL_2powK

from . import register

@register("chain_parity")
def build(cfg: Dict[str, Any]) -> Dict[str, Any]:
    K_train = int(cfg["K_train"])
    L_train = int(cfg["L_train"])
    K_test = int(cfg.get("K_test", K_train))
    L_test = int(cfg.get("L_test", L_train))
    bidirectional = bool(cfg.get("bidirectional", False))
    pad_value = int(cfg.get("pad_value", 0))
    supervise = str(cfg.get("supervise", "all_prefix"))
    max_graphs = cfg.get("max_graphs", None)
    loss_mask_len = cfg.get("loss_mask_len", None)
    if loss_mask_len is not None:
        loss_mask_len = int(loss_mask_len)
        if not (0 < loss_mask_len <= L_train):
            raise ValueError(f"loss_mask_len must be in (0, L_train={L_train}]; got {loss_mask_len}")

    X_tr, Y_tr, A_tr, A_sp_tr, last_mask_tr = generate_chain_parity_KL_2powK(
        K=K_train, L=L_train,
        bidirectional=bidirectional,
        pad_value=pad_value,
        supervise=supervise,
        max_graphs=max_graphs,
    )
    X_te, Y_te, A_te, A_sp_te, last_mask_te = generate_chain_parity_KL_2powK(
        K=K_test, L=L_test,
        bidirectional=bidirectional,
        pad_value=pad_value,
        supervise=supervise,
        max_graphs=max_graphs,
    )

    out: Dict[str, Any] = {
        "mode": "synthetic",
        "X_train": X_tr, "Y_train": Y_tr, "A_train": A_tr, "A_sp_train": A_sp_tr, "last_mask_train": last_mask_tr,
        "X_test": X_te, "Y_test": Y_te, "A_test": A_te, "A_sp_test": A_sp_te, "last_mask_test": last_mask_te,
        "meta": {
            "in_dim": int(X_tr.size(1)),
            "out_dim": int(Y_tr.size(1)),
            "task_type": "node_classification_parity",
            "metric": "parity_acc",
            "K_train": K_train, "L_train": L_train, "K_test": K_test, "L_test": L_test,
        },
    }
    if loss_mask_len is not None:
        N_train = X_tr.size(0)
        pos = torch.arange(N_train) % L_train
        out["loss_mask_train"] = (pos < loss_mask_len).to(X_tr.dtype)
        out["meta"]["loss_mask_len"] = loss_mask_len
    return out
