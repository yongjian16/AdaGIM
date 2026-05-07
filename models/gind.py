"""GIND wrapper. Vendored model lives at third_party/GIND/model/gind.py."""
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from . import register

_GIND_DIR = Path(__file__).resolve().parents[1] / "third_party" / "GIND"

if str(_GIND_DIR) not in sys.path:
    sys.path.append(str(_GIND_DIR))

_KEY = "_gind_model"
if _KEY not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_KEY, str(_GIND_DIR / "model" / "gind.py"))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_KEY] = _mod
    _spec.loader.exec_module(_mod)
_GIND = sys.modules[_KEY].GIND

_NORM_KEY = "_gind_norm"
if _NORM_KEY not in sys.modules:
    _nspec = importlib.util.spec_from_file_location(_NORM_KEY, str(_GIND_DIR / "libs" / "normalization.py"))
    _nmod = importlib.util.module_from_spec(_nspec)
    sys.modules[_NORM_KEY] = _nmod
    _nspec.loader.exec_module(_nmod)
_cal_norm = sys.modules[_NORM_KEY].cal_norm

class GINDPyG(nn.Module):

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 num_layers: int = 1, alpha: float = 0.145,
                 total_iter: int = 32, grad_iter: int = 4,
                 dropout_imp: float = 0.0, dropout_exp: float = 0.5,
                 drop_input: bool = True,
                 norm: str = "LayerNorm",
                 linear: bool = True, double_linear: bool = False,
                 act_imp: str = "tanh", act_exp: str = "relu",
                 residual: bool = True, rescale: bool = True,
                 reg_type: str = "", reg_coeff: float = 0.0,
                 self_loop: bool = True,
                 graph_pool: str = "add",
                 x_is_indices: bool = False, num_atom_types: int = 28,
                 use_ogb_encoders: bool = False):
        super().__init__()
        self.x_is_indices = x_is_indices
        self.use_ogb_encoders = use_ogb_encoders
        if use_ogb_encoders:
            from ._ogb_encoder import OGBAtomEncoder
            self.input_emb = OGBAtomEncoder(emb_dim=hidden)
            in_dim = hidden
        elif x_is_indices:
            self.input_emb = nn.Embedding(num_atom_types, hidden)
            in_dim = hidden
        else:
            self.input_emb = nn.Identity()
        self.self_loop = self_loop
        self.core = _GIND(
            in_channels=in_dim, hidden_channels=hidden, out_channels=out_dim,
            num_layers=num_layers, alpha=alpha, iter_nums=(total_iter, grad_iter),
            dropout_imp=dropout_imp, dropout_exp=dropout_exp,
            drop_input=drop_input,
            norm=norm, residual=residual, rescale=rescale,
            linear=linear, double_linear=double_linear,
            act_imp=act_imp, act_exp=act_exp,
            reg_type=reg_type, reg_coeff=reg_coeff,
            final_reduce=graph_pool,
        )
        self.params_imp = self.core.params_imp
        self.params_exp = self.core.params_exp

    def forward_pyg(self, batch) -> torch.Tensor:
        x = batch.x
        if self.use_ogb_encoders:
            x = self.input_emb(x)
        elif self.x_is_indices:
            x = x.squeeze(-1) if x.dim() == 2 else x
            x = self.input_emb(x.long())
        else:
            x = x.float()
        edge_index = batch.edge_index
        norm_factor, edge_index = _cal_norm(edge_index, num_nodes=x.size(0),
                                            self_loop=self.self_loop, cut=True)
        norm_factor = norm_factor.to(x.dtype).to(x.device)
        batch_idx = getattr(batch, "batch", None)
        return self.core(x, edge_index, norm_factor, batch=batch_idx)

@register("gind")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    is_node_task = (dataset_meta.get("task_type") in
                    {"node_classification", "node_classification_masked",
                     "node_classification_parity"})
    from ._ogb_encoder import is_ogb_format
    use_ogb = bool(cfg.get("use_ogb_encoders", is_ogb_format(dataset_meta)))
    return GINDPyG(
        in_dim=dataset_meta["in_dim"],
        hidden=int(cfg.get("hidden", 64)),
        out_dim=dataset_meta["out_dim"],
        num_layers=int(cfg.get("num_layers", 1)),
        use_ogb_encoders=use_ogb,
        alpha=float(cfg.get("alpha", 0.145)),
        total_iter=int(cfg.get("total_iter", 32)),
        grad_iter=int(cfg.get("grad_iter", 4)),
        dropout_imp=float(cfg.get("dropout_imp", cfg.get("dropout", 0.0))),
        dropout_exp=float(cfg.get("dropout_exp", cfg.get("dropout", 0.5))),
        drop_input=bool(cfg.get("drop_input", True)),
        norm=str(cfg.get("norm", "LayerNorm")),
        linear=bool(cfg.get("linear", True)),
        double_linear=bool(cfg.get("double_linear", False)),
        act_imp=str(cfg.get("act_imp", "tanh")),
        act_exp=str(cfg.get("act_exp", "relu")),
        residual=bool(cfg.get("residual", True)),
        rescale=bool(cfg.get("rescale", True)),
        reg_type=str(cfg.get("reg_type", "")),
        reg_coeff=float(cfg.get("reg_coeff", 0.0)),
        self_loop=bool(cfg.get("add_self_loops", True)),
        graph_pool="" if is_node_task else str(cfg.get("graph_pool", "add")),
        x_is_indices=dataset_meta.get("x_is_indices", False),
    )
