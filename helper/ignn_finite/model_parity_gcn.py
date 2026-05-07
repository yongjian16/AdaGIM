# save as: model_parity_gcn.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class IGNN_finite(nn.Module):
    """
    Iterative cell with GCN as 'W1'-part:
        Z^{l+1} = σ( GCN(Z^{l}) + X W2 + b1 )
        y^{l}   = Z^{l} W3 + b2  (logits per iteration)

    Expectation (per mini-batch):
      batch.x          : [N, 2]
      batch.edge_index : [2, E]  (identity edges here)
      batch.pos        : [N]     (0..L-1 positions)
    """
    def __init__(self, in_dim: int = 2, hidden: int = 64, out_dim: int = 2, activation: str = "relu"):
        super().__init__()
        self.conv = GCNConv(hidden, hidden, add_self_loops=False, normalize=False)
        self.W2 = nn.Linear(in_dim, hidden, bias=False)  # X W2
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W3 = nn.Linear(hidden, out_dim, bias=True)  # Z -> logits
        self.act = nn.ReLU() if activation.lower() == "relu" else nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self, gain: float = 0.99):
        self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.W2.weight, gain=gain)
        nn.init.xavier_uniform_(self.W3.weight, gain=gain)
        if self.W3.bias is not None:
            nn.init.zeros_(self.W3.bias)
        with torch.no_grad():
            self.b1.zero_()

    def forward(
        self,
        batch,
        steps: int,
        Z0: Optional[torch.Tensor] = None,
        return_all: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits_seq: [steps, N, out_dim]
          Z_seq     : [steps, N, hidden]
        """
        x, edge_index = batch.x, batch.edge_index
        N = x.size(0)
        H = self.W3.in_features
        Z = torch.zeros(N, H, device=x.device) if Z0 is None else Z0
        x_proj = self.W2(x)  # [N,H], reused each step

        logits_list, Z_list = [], []
        for i in range(steps):
            z_gcn = self.conv(Z, edge_index)       # [N,H]; with identity edges this is per-node linear map
            # if i > 0:
            #     Z = self.act(z_gcn + self.b1)
            # else:
            Z = self.act(z_gcn + x_proj + self.b1) # update
            logits = self.W3(Z)                    # [N,out_dim]
            logits_list.append(logits)
            Z_list.append(Z)

        logits_seq = torch.stack(logits_list, dim=0)  # [steps,N,C]
        Z_seq = torch.stack(Z_list, dim=0)            # [steps,N,H]
        return (logits_seq, Z_seq) if return_all else (logits_seq[-1], Z_seq[-1])
