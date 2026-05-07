"""AdaGIM: Adaptive Graph Implicit Model."""
from typing import Any, Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register

def _matmul_A(A: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    if A.is_sparse:
        return torch.sparse.mm(A.coalesce(), M)
    return A @ M

def _matmul_AT(A: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    if A.is_sparse:
        return torch.sparse.mm(A.coalesce().t(), M)
    return A.t() @ M

class AdaGIMCore(nn.Module):
    def __init__(self, hidden: int, s_W: float = 0.99, K_max: int = 50,
                 alpha: float = 1.0, tol: float = 1e-3, phi: str = "tanh",
                 row_norm: bool = False, gate_pe_dim: int = 16,
                 gate_hidden: Optional[int] = None,
                 gate_init_bias: float = 1.0,
                 gate_scale: float = 10.0,
                 gate_init_threshold: float = 0.05,
                 gate_scale_learnable: bool = False,
                 gate_tau_init: float = 0.3,
                 gate_time_dim: int = 16,
                 gate_pondercost_beta: float = 0.0,
                 gate_temp: float = 1.0,
                 gate_lambda_sharp: float = 0.0,
                 gate_lambda_nowrite: float = 0.0,
                 gate_lambda_bin: float = 0.0):
        super().__init__()
        self.hidden = hidden
        if not (0.0 < float(s_W) < 1.0):
            raise ValueError(f"s_W must lie in (0, 1); got {s_W!r}")
        self.s_W = float(s_W)
        self.K_max = int(K_max)
        self.alpha = float(alpha)
        self.tol = float(tol)
        self.row_norm = bool(row_norm)
        self.gate_pe_dim = int(gate_pe_dim)
        self.phi = torch.tanh if phi == "tanh" else torch.relu

        self.C = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.normal_(self.C, std=0.01)

        self.layer_norm = nn.LayerNorm(hidden, elementwise_affine=True) if row_norm else nn.Identity()

        del gate_pe_dim, gate_hidden, gate_init_bias
        del gate_scale, gate_init_threshold, gate_scale_learnable
        del gate_time_dim  # legacy: time embedding was a trick; gate now uses H only
        self.halt_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.halt_mlp[0].weight, std=0.01)
        nn.init.zeros_(self.halt_mlp[0].bias)
        nn.init.normal_(self.halt_mlp[-1].weight, std=0.01)
        nn.init.constant_(self.halt_mlp[-1].bias, -3.0)
        if gate_tau_init <= 0:
            raise ValueError(f"gate_tau_init must be positive; got {gate_tau_init!r}")
        self.gate_log_tau = nn.Parameter(
            torch.full((1,), math.log(float(gate_tau_init)))
        )

        if gate_pondercost_beta < 0:
            raise ValueError(
                f"gate_pondercost_beta must be non-negative; got {gate_pondercost_beta!r}"
            )
        self.gate_pondercost_beta = float(gate_pondercost_beta)

        if gate_temp <= 0:
            raise ValueError(f"gate_temp must be positive; got {gate_temp!r}")
        self.gate_temp = float(gate_temp)

        for name, val in (("gate_lambda_sharp", gate_lambda_sharp),
                          ("gate_lambda_nowrite", gate_lambda_nowrite),
                          ("gate_lambda_bin", gate_lambda_bin)):
            if val < 0:
                raise ValueError(f"{name} must be non-negative; got {val!r}")
        self.gate_lambda_sharp = float(gate_lambda_sharp)
        self.gate_lambda_nowrite = float(gate_lambda_nowrite)
        self.gate_lambda_bin = float(gate_lambda_bin)
        self.gate_pe_dim = 0

        self.last_avg_iters_H: Optional[float] = None  # Picard iters to find H*
        self.last_avg_iters_Z: Optional[float] = None  # Picard iters to find Z*
        self.last_picard_residual: Optional[float] = None
        self.last_sigma_mean: Optional[float] = None
        self.last_sigma_at_fixed_point: Optional[torch.Tensor] = None  # (N,) σ(H*, H*)
        self.last_sigma_traj: Optional[torch.Tensor] = None  # (T_done, N)

    def _W(self) -> torch.Tensor:
        I = torch.eye(self.hidden, device=self.C.device, dtype=self.C.dtype)
        S = self.C - self.C.t()
        W_unscaled = torch.linalg.solve(I + S, I - S)
        return self.s_W * W_unscaled

    def _f_theta(self, H: torch.Tensor, A_hat: torch.Tensor, U: torch.Tensor,
                 W: torch.Tensor) -> torch.Tensor:
        agg = _matmul_AT(A_hat, H)
        h = self.phi(agg @ W + U)
        return self.layer_norm(h)

    def _gate(self, H_new: torch.Tensor, H_old: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = H_new - H_old                                       # (N, hidden)
        gate_in = torch.cat([H_new, H_old], dim=-1)                # (N, 2·hidden)
        raw = self.halt_mlp(gate_in).squeeze(-1)                   # (N,)
        lam_tilde = torch.sigmoid(raw / self.gate_temp)            # (N,)
        sq_norm = (diff * diff).sum(dim=-1)                        # (N,)
        tau_sq = torch.exp(2.0 * self.gate_log_tau)
        q = 1.0 - torch.exp(-sq_norm / tau_sq.clamp_min(1e-12))    # (N,)
        alpha = q * lam_tilde                                      # (N,)
        return alpha.unsqueeze(-1), lam_tilde                      # ((N,1), (N,))

    def outer_loop(self, U: torch.Tensor, A_hat: torch.Tensor,
                   node_idx: Optional[torch.Tensor] = None,
                   H_init: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        del node_idx
        W = self._W()
        H = torch.zeros_like(U) if H_init is None else H_init
        Z = torch.zeros_like(U)
        n_Z = 0
        alphas_grad: List[torch.Tensor] = []      # (N,) tensors WITH grad
        lamtildes_grad: List[torch.Tensor] = []   # (N,) tensors WITH grad
        last_rel_H = float("nan")
        for k in range(self.K_max):
            H_new = self._f_theta(H, A_hat, U, W)
            H_new = (1.0 - self.alpha) * H + self.alpha * H_new
            alpha_t, lam_tilde = self._gate(H_new, H)                   # (N,1), (N,)
            Z_new = alpha_t * H_new + (1.0 - alpha_t) * Z
            alphas_grad.append(alpha_t.squeeze(-1))                      # (N,)
            lamtildes_grad.append(lam_tilde)                             # (N,)
            with torch.no_grad():
                rel_H = (H_new - H).norm() / H.norm().clamp_min(1e-8)
                stop = float(rel_H.item()) < self.tol
            H, Z = H_new, Z_new
            n_Z = k + 1
            last_rel_H = float(rel_H.item())
            if stop:
                break
        H_star = H.detach()
        Z_diff = Z

        ponder_terms = []
        if (self.gate_lambda_sharp > 0 or self.gate_lambda_nowrite > 0
                or self.gate_lambda_bin > 0) and len(alphas_grad) > 0:
            T = len(alphas_grad)
            omegas: List[torch.Tensor] = [None] * T  # type: ignore
            tail = torch.ones_like(alphas_grad[0])    # ∏_{s>t}(1 − α_s)
            for t in range(T - 1, -1, -1):
                a_t = alphas_grad[t]
                omegas[t] = a_t * tail               # (N,)
                tail = tail * (1.0 - a_t)            # accumulate for next t
            omega_0 = tail                            # (N,)  no-write mass

            if self.gate_lambda_sharp > 0:
                eps = 1e-12
                ent = -omega_0 * torch.log(omega_0 + eps)
                for w in omegas:
                    ent = ent - w * torch.log(w + eps)
                ponder_terms.append(self.gate_lambda_sharp * ent.mean())

            if self.gate_lambda_nowrite > 0:
                ponder_terms.append(self.gate_lambda_nowrite * omega_0.mean())

            if self.gate_lambda_bin > 0:
                bin_term = sum(l * (1.0 - l) for l in lamtildes_grad)  # (N,)
                ponder_terms.append(
                    self.gate_lambda_bin * bin_term.mean() / max(T, 1)
                )

        if self.gate_pondercost_beta > 0 and len(alphas_grad) > 0:
            survival = torch.ones_like(alphas_grad[0])
            for a_t in alphas_grad:
                survival = survival * (1.0 - a_t)
            ponder_terms.append(self.gate_pondercost_beta * survival.mean())

        if ponder_terms:
            self.last_ponder_cost = sum(ponder_terms)
        else:
            self.last_ponder_cost = None

        sigma_history = [a.detach() for a in alphas_grad]

        with torch.no_grad():
            self.last_avg_iters_H = float(n_Z)   # single phase, same iter count
            self.last_avg_iters_Z = float(n_Z)
            self.last_picard_residual = last_rel_H
            A_fp, _ = self._gate(H_star, H_star)
            A_at_fp = A_fp.squeeze(-1).detach().cpu()
            self.last_sigma_at_fixed_point = A_at_fp
            if sigma_history:
                self.last_sigma_traj = torch.stack(sigma_history, dim=0).cpu()  # (T, N)
            self.last_sigma_mean = float(A_at_fp.mean().item())

        return H_star, Z_diff

class AdaGIMSynthetic(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 s_W: float = 0.99, K_max: int = 50, alpha: float = 1.0,
                 tol: float = 1e-3, phi: str = "tanh", row_norm: bool = False,
                 gate_pe_dim: int = 16, gate_hidden: Optional[int] = None,
                 gate_init_bias: float = 1.0,
                 gate_tau_init: float = 0.3,
                 gate_time_dim: int = 16,
                 gate_pondercost_beta: float = 0.0,
                 gate_temp: float = 1.0,
                 gate_lambda_sharp: float = 0.0,
                 gate_lambda_nowrite: float = 0.0,
                 gate_lambda_bin: float = 0.0,
                 chain_period: Optional[int] = None):
        super().__init__()
        self.input_emb = nn.Linear(in_dim, hidden)
        self.chain_period = int(chain_period) if chain_period else None
        self.core = AdaGIMCore(hidden=hidden, s_W=s_W, K_max=K_max, alpha=alpha,
                                 tol=tol, phi=phi, row_norm=row_norm,
                                 gate_pe_dim=gate_pe_dim, gate_hidden=gate_hidden,
                                 gate_init_bias=gate_init_bias,
                                 gate_tau_init=gate_tau_init,
                                 gate_time_dim=gate_time_dim,
                                 gate_pondercost_beta=gate_pondercost_beta,
                                 gate_temp=gate_temp,
                                 gate_lambda_sharp=gate_lambda_sharp,
                                 gate_lambda_nowrite=gate_lambda_nowrite,
                                 gate_lambda_bin=gate_lambda_bin)
        self.head = nn.Linear(hidden, out_dim)

    def forward_synthetic(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        U = self.input_emb(X.float())
        N = U.size(0)
        if self.chain_period:
            node_idx = torch.arange(N, device=U.device) % self.chain_period
        else:
            node_idx = torch.arange(N, device=U.device)
        _, Z_diff = self.core.outer_loop(U, A, node_idx=node_idx)
        return self.head(Z_diff)

    @property
    def last_ponder_cost(self):
        return getattr(self.core, "last_ponder_cost", None)

class AdaGIMPyG(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 s_W: float = 0.99, K_max: int = 50, alpha: float = 1.0,
                 tol: float = 1e-3, phi: str = "tanh", row_norm: bool = False,
                 gate_pe_dim: int = 16, gate_hidden: Optional[int] = None,
                 gate_init_bias: float = 1.0,
                 gate_tau_init: float = 0.3,
                 gate_time_dim: int = 16,
                 gate_pondercost_beta: float = 0.0,
                 gate_temp: float = 1.0,
                 gate_lambda_sharp: float = 0.0,
                 gate_lambda_nowrite: float = 0.0,
                 gate_lambda_bin: float = 0.0,
                 num_layers: int = 1,
                 graph_pool: str = "mean", x_is_indices: bool = False,
                 num_atom_types: int = 28, use_ogb_encoders: bool = False,
                 head_dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden
        self.graph_pool = graph_pool
        self.x_is_indices = x_is_indices
        self.use_ogb_encoders = use_ogb_encoders
        self.head_dropout = head_dropout
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be ≥ 1; got {num_layers!r}")

        if use_ogb_encoders:
            from ._ogb_encoder import OGBAtomEncoder
            self.input_emb = OGBAtomEncoder(emb_dim=hidden)
        elif x_is_indices:
            self.input_emb = nn.Embedding(num_atom_types, hidden)
        else:
            self.input_emb = nn.Linear(in_dim, hidden)

        self.cores = nn.ModuleList([
            AdaGIMCore(hidden=hidden, s_W=s_W, K_max=K_max, alpha=alpha,
                        tol=tol, phi=phi, row_norm=row_norm,
                        gate_pe_dim=gate_pe_dim, gate_hidden=gate_hidden,
                        gate_init_bias=gate_init_bias,
                        gate_tau_init=gate_tau_init,
                        gate_time_dim=gate_time_dim,
                        gate_pondercost_beta=gate_pondercost_beta,
                        gate_temp=gate_temp,
                        gate_lambda_sharp=gate_lambda_sharp,
                        gate_lambda_nowrite=gate_lambda_nowrite,
                        gate_lambda_bin=gate_lambda_bin)
            for _ in range(self.num_layers)
        ])
        self.core = self.cores[0]
        self.head = nn.Linear(hidden, out_dim)

    @property
    def last_ponder_cost(self):
        terms = [c.last_ponder_cost for c in self.cores
                 if getattr(c, "last_ponder_cost", None) is not None]
        if not terms:
            return None
        out = terms[0]
        for t in terms[1:]:
            out = out + t
        return out

    def _build_norm_adj(self, edge_index: torch.Tensor, num_nodes: int,
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        from torch_geometric.utils import add_self_loops, degree
        ei, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = ei
        deg = degree(row, num_nodes, dtype=dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        ew = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        idx = torch.stack([row, col], dim=0)
        return torch.sparse_coo_tensor(idx, ew, (num_nodes, num_nodes), device=device).coalesce()

    def forward_pyg(self, batch) -> torch.Tensor:
        from torch_geometric.utils import scatter

        x = batch.x
        if self.use_ogb_encoders:
            U = self.input_emb(x)
        elif self.x_is_indices:
            xi = x.squeeze(-1) if x.dim() == 2 else x
            U = self.input_emb(xi.long())
        else:
            U = self.input_emb(x.float())

        N = U.size(0)
        A_hat = self._build_norm_adj(batch.edge_index, N, U.device, U.dtype)
        H_warm = None  # first layer starts H from zeros
        Z_diff = None
        for core in self.cores:
            _, Z_diff = core.outer_loop(U, A_hat, H_init=H_warm)
            H_warm = Z_diff

        if self.head_dropout > 0 and self.training:
            Z_diff = F.dropout(Z_diff, p=self.head_dropout, training=True)

        if self.graph_pool == "node":
            return self.head(Z_diff)
        if self.graph_pool == "mean":
            g = scatter(Z_diff, batch.batch, dim=0, reduce="mean")
        elif self.graph_pool == "sum":
            g = scatter(Z_diff, batch.batch, dim=0, reduce="sum")
        elif self.graph_pool == "max":
            g = scatter(Z_diff, batch.batch, dim=0, reduce="max")
        else:
            raise ValueError(self.graph_pool)
        return self.head(g)

@register("adagim")
def build(cfg: Dict[str, Any], dataset_meta: Dict[str, Any]) -> nn.Module:
    task_type = dataset_meta.get("task_type", "")
    hidden = int(cfg.get("hidden", 128))
    gate_hidden_cfg = cfg.get("gate_hidden", None)
    gate_hidden = int(gate_hidden_cfg) if gate_hidden_cfg is not None else None
    common = dict(
        hidden=hidden,
        s_W=float(cfg.get("s_W", 0.99)),
        K_max=int(cfg.get("K_max", 50)),
        alpha=float(cfg.get("alpha", 1.0)),
        tol=float(cfg.get("tol", 1e-3)),
        phi=str(cfg.get("phi", "tanh")),
        row_norm=bool(cfg.get("row_norm", False)),
        gate_pe_dim=int(cfg.get("gate_pe_dim", 16)),
        gate_hidden=gate_hidden,
        gate_init_bias=float(cfg.get("gate_init_bias", 1.0)),
        gate_tau_init=float(cfg.get("gate_tau_init", 0.3)),
        gate_time_dim=int(cfg.get("gate_time_dim", 16)),
        gate_pondercost_beta=float(cfg.get("gate_pondercost_beta", 0.0)),
        gate_temp=float(cfg.get("gate_temp", 1.0)),
        gate_lambda_sharp=float(cfg.get("gate_lambda_sharp", 0.0)),
        gate_lambda_nowrite=float(cfg.get("gate_lambda_nowrite", 0.0)),
        gate_lambda_bin=float(cfg.get("gate_lambda_bin", 0.0)),
    )

    if task_type == "node_classification_parity":
        default_period = (dataset_meta.get("loss_mask_len")
                          or dataset_meta.get("L_train"))
        chain_period = cfg.get("chain_period", default_period)
        return AdaGIMSynthetic(
            in_dim=dataset_meta["in_dim"],
            out_dim=dataset_meta["out_dim"],
            chain_period=chain_period,
            **common,
        )

    is_node_task = task_type in {"node_classification", "node_classification_masked"}
    from ._ogb_encoder import is_ogb_format
    use_ogb = bool(cfg.get("use_ogb_encoders", is_ogb_format(dataset_meta)))
    return AdaGIMPyG(
        in_dim=dataset_meta["in_dim"],
        out_dim=dataset_meta["out_dim"],
        num_layers=int(cfg.get("num_layers", 1)),
        graph_pool="node" if is_node_task else str(cfg.get("graph_pool", "mean")),
        x_is_indices=dataset_meta.get("x_is_indices", False),
        use_ogb_encoders=use_ogb,
        head_dropout=float(cfg.get("head_dropout", 0.0)),
        **common,
    )
