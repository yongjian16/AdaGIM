import torch
import torch.nn as nn

from eignn_utils.layers import EIGNN_w_iterative_solvers, EIGNN_k_steps
from ignn_utils.layers import ImplicitGraph
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional
from typing import Tuple

def _spmm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Sparse/dense safe A @ X."""
    return torch.sparse.mm(A, X) if A.is_sparse else (A @ X)

def init_named_params_xavier(module: nn.Module, gain: float = 0.99):
    for name, p in module.named_parameters():
        if p is None:
            continue
        if "weight" in name:
            nn.init.xavier_uniform_(p, gain=gain)
        elif "bias" in name:
            nn.init.zeros_(p)


class ParityEIGNN(nn.Module):
    """
    One-model EIGNN wrapper.
    - You can switch the graph via set_graph(adj, sp_adj)
    - encode(X) uses the currently-set graph
    """
    def __init__(self, adj, sp_adj, in_dim: int, hidden: int, out_dim: int = 2,
                 threshold: float = 1e-4, max_iter: int = 300, gamma: float = 0.95, g_type: str = "psd", k_steps=None):
        super().__init__()
        if k_steps is not None:
            self.EIGNN = EIGNN_k_steps(
                adj=adj, m=hidden,
                k_steps=k_steps, gamma=gamma
            )
        else:
            self.EIGNN = EIGNN_w_iterative_solvers(
                adj=adj, sp_adj=sp_adj, m=hidden,
                threshold=threshold, max_iter=max_iter, gamma=gamma, g_type=g_type
            )
        self.W = nn.Linear(in_dim, hidden)
        self.B = nn.Linear(hidden, out_dim, bias=True)
        self.reset_parameters()

    

    def reset_parameters(self, gain_core: float = 0.1, gain_head: float = 0.1):
        # init linear layers
        init_named_params_xavier(self.W, gain=gain_core)
        init_named_params_xavier(self.B, gain=gain_head)

        # init EIGNN core param (name is "F", not "weight")
        nn.init.xavier_uniform_(self.EIGNN.F, gain=gain_core)
        if hasattr(self.EIGNN, 's'):
            torch.nn.init.normal_(self.EIGNN.s, mean=0.0, std=gain_core)

    def set_graph(self, adj, sp_adj=None, k_steps=None):
        """
        Swap the adjacency used inside EIGNN.
        Note: your EIGNN class only uses self.S in forward().
        sp_adj is only used in __init__ for symmetry print / potential eig calls.
        """
        self.EIGNN.S = adj
        # keep for completeness / future use
        if sp_adj is not None:
            self.EIGNN._sp_adj = sp_adj
        if hasattr(self.EIGNN, 'k_steps') and k_steps is not None:
            self.EIGNN.k_steps = k_steps

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        Xh = self.W(X)               # [N,H]
        Z = self.EIGNN(Xh.t()).t()   # [N,H]
        return Z

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.B(self.encode(X))  # [N,2]


class ParityIGNN(nn.Module):
    """
    One-model IGNN wrapper.
    - DOES NOT store adjacency
    - You pass A_sp into encode/forward each call:
         logits = net(X, A_sp)
         Z      = net.encode(X, A_sp)

    NOTE: ImplicitGraph.forward takes A as an argument, and your posted code
    does not use self.n in forward(), so one-model is safe.
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int = 2,
                 kappa: float = 0.99, b_direct: bool = False,
                 fw_mitr: int = 300, bw_mitr: int = 300,
                 A_rho: float = 1.0, phi: str = "tanh"):
        super().__init__()
        self.fw_mitr = fw_mitr
        self.bw_mitr = bw_mitr
        self.A_rho = A_rho

        if phi == "tanh":
            self.phi = torch.tanh
        elif phi == "relu":
            self.phi = torch.relu
        else:
            raise ValueError("phi must be 'tanh' or 'relu'")

        # num_node is required by ctor, but your forward() doesn't use self.n.
        # We'll still set it to a dummy and allow the caller to pass A each time.
        self.IGNN = ImplicitGraph(
            in_features=in_dim,
            out_features=hidden,
            num_node=1,
            kappa=kappa,
            b_direct=b_direct
        )
        self.B = nn.Linear(hidden, out_dim)
        self.reset_parameters()
    
    def reset_parameters(self, gain_core: float = 0.1, gain_head: float = 0.1):
        # classifier
        init_named_params_xavier(self.B, gain=gain_head)

        # IGNN core
        nn.init.xavier_uniform_(self.IGNN.W, gain=gain_core)
        nn.init.xavier_uniform_(self.IGNN.Omega_1, gain=gain_core)
        nn.init.xavier_uniform_(self.IGNN.Omega_2, gain=gain_core)
        nn.init.zeros_(self.IGNN.bias)

    def encode(self, X: torch.Tensor, A_sp: torch.Tensor) -> torch.Tensor:
        Z = self.IGNN(
            X_0=None,
            A=A_sp,
            U=X,
            phi=self.phi,
            A_rho=self.A_rho,
            fw_mitr=self.fw_mitr,
            bw_mitr=self.bw_mitr,
            A_orig=None
        )

        # Some implementations return [N,H], others [H,N]
        if Z.dim() == 2 and Z.size(0) == X.size(0):
            return Z
        if Z.dim() == 2 and Z.size(1) == X.size(0):
            return Z.T
        raise RuntimeError(f"Unexpected IGNN output shape {tuple(Z.shape)} for X shape {tuple(X.shape)}")

    def forward(self, X: torch.Tensor, A_sp: torch.Tensor) -> torch.Tensor:
        return self.B(self.encode(X, A_sp))  # [N,2]
    

class ParityGCN(nn.Module):
    """
    Multi-layer GCN using PyG GCNConv.

    - Call set_graph(edge_index, edge_weight) once per graph.
    - forward(X) uses the stored graph.
    - encode(X) returns node embeddings (hidden).
    - has attribute .B for your plotting pipeline.
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int = 2,
        num_layers: int = 2,
        dropout: float = 0.0,
        act: str = "relu",
        activate_last: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1
        assert act in ("relu", "tanh")

        self.dropout = float(dropout)
        self.act = act
        self.activate_last = bool(activate_last)

        self.convs = nn.ModuleList()
        for l in range(num_layers):
            in_ch = in_dim if l == 0 else hidden

            # IMPORTANT:
            # We set normalize=False and add_self_loops=False because
            # your adj_mode already decides normalization/self-loops.
            self.convs.append(
                GCNConv(
                    in_channels=in_ch,
                    out_channels=hidden,
                    add_self_loops=False,
                    normalize=False,
                )
            )

        # keep this name for plotting code
        self.B = nn.Linear(hidden, out_dim, bias=True)

        self.edge_index = None
        self.edge_weight = None

    def set_graph(self, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
    def _apply_act(self, X: torch.Tensor) -> torch.Tensor:
        return F.relu(X) if self.act == "relu" else torch.tanh(X)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        assert self.edge_index is not None, "Call set_graph(...) before forward/encode."
        h = X
        L = len(self.convs)

        for i, conv in enumerate(self.convs):
            h = conv(h, self.edge_index, self.edge_weight)
            is_last = (i == L - 1)

            if (not is_last) or self.activate_last:
                h = self._apply_act(h)

            # common choice: no dropout on last layer
            if self.dropout > 0 and (not is_last):
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.encode(X)
        return self.B(z)
    






class ParityDiagUnrollIGNN(nn.Module):
    r"""
    Unrolled IGNN with *diagonal readout*:

        H_l = σ( A H_{l-1} W + X V ),   l=1..L
        Z_i = (H_{pos_i+1})[i]          (node i uses state after (pos_i+1) updates)
        logits = Z B

    Notes:
      - This "diagonal" readout is meaningful when each node has a position pos∈{0..L-1}
        (e.g., chains). For a batch of equal-length chains concatenated, use
            pos = torch.arange(N) % L
      - A can be torch sparse COO or dense; it should already contain whatever normalization you want.
    """
    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 64,
        out_dim: int = 2,
        activation: str = "relu",
        init_gain: float = 0.99,
    ):
        super().__init__()
        self.hidden = hidden

        # X V  (your x v^T term)
        self.V = nn.Linear(in_dim, hidden, bias=True)

        # shared W in A H W
        self.W = nn.Parameter(torch.empty(hidden, hidden))

        # decoder
        self.B = nn.Linear(hidden, out_dim, bias=True)

        self.act = nn.ReLU() if activation.lower() == "relu" else nn.Tanh()

        # graph storage (optional convenience)
        self.A: Optional[torch.Tensor] = None

        self.reset_parameters(init_gain)

    def reset_parameters(self, gain: float = 0.99):
        nn.init.xavier_uniform_(self.V.weight, gain=gain)
        nn.init.xavier_uniform_(self.W, gain=gain)
        nn.init.xavier_uniform_(self.B.weight, gain=gain)
        if self.B.bias is not None:
            nn.init.zeros_(self.B.bias)

    def set_graph(self, A: torch.Tensor):
        """Store adjacency (torch sparse COO or dense) to reuse in forward()."""
        self.A = A

    def _matmul_A(self, A: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Compute A @ M supporting sparse/dense A."""
        if A.is_sparse:
            A = A.coalesce()
            return torch.sparse.mm(A, M)
        return A @ M

    def encode(
        self,
        X: torch.Tensor,
        *,
        A: Optional[torch.Tensor] = None,
        L: int,
        pos: Optional[torch.Tensor] = None,
        H0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns diagonal embeddings Z: [N, hidden]
        """
        if A is None:
            assert self.A is not None, "Pass A=... or call set_graph(A) first."
            A = self.A

        N = X.size(0)
        device = X.device

        if pos is None:
            # assumes concatenated equal-length chains
            pos = (torch.arange(N, device=device) % L).long()
        else:
            pos = pos.to(device).long()

        # initialize H_0
        H = torch.zeros(N, self.hidden, device=device) if H0 is None else H0

        Xproj = self.V(X)  # [N,hidden]

        # diagonal output buffer
        Z = torch.zeros(N, self.hidden, device=device)

        # unroll and write diagonal: nodes with pos==(l-1) take H_l
        for l in range(1, L + 1):
            # compute A H_{l-1} W
            HW = H @ self.W              # [N,hidden]
            msg = self._matmul_A(A, HW)  # [N,hidden]
            H = self.act(msg + Xproj)

            m = (pos == (l - 1))
            if m.any():
                Z[m] = H[m]

        return Z

    def forward(
        self,
        X: torch.Tensor,
        *,
        A: Optional[torch.Tensor] = None,
        L: int,
        pos: Optional[torch.Tensor] = None,
        H0: Optional[torch.Tensor] = None,
        return_Z: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          logits: [N, out_dim]
          Z     : [N, hidden] if return_Z else None
        """
        Z = self.encode(X, A=A, L=L, pos=pos, H0=H0)
        logits = self.B(Z)
        return (logits, Z) if return_Z else (logits, None)
