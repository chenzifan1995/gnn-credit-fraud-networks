import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

class GATFraudDetector(nn.Module):
    def __init__(self, in_dim_b: int, in_dim_m: int, hidden_dim: int = 64, heads: int = 2, dropout: float = 0.2):
        super().__init__()
        # For simplicity, we lift borrower/merchant into shared dim then do attention on bipartite edges
        self.b_proj = nn.Linear(in_dim_b, hidden_dim)
        self.m_proj = nn.Linear(in_dim_m, hidden_dim)
        self.conv = GATv2Conv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False)
        self.head = nn.Linear(hidden_dim*heads, 1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x_b, x_m, edge_index_bm):
        hb = self.act(self.b_proj(x_b))
        hm = self.act(self.m_proj(x_m))
        x = torch.cat([hb, hm], dim=0)
        # remap indices: borrowers first, then merchants
        offset = hb.size(0)
        edge_index = edge_index_bm.clone()
        edge_index[1] = edge_index_bm[1] + offset
        h = self.conv(x, edge_index)
        h = self.drop(h)
        # predict on edges by dot-product then MLP head
        src, dst = edge_index_bm
        dst = dst + offset
        edge_repr = h[src] * h[dst]
        logits = self.head(edge_repr).squeeze(-1)
        return logits
