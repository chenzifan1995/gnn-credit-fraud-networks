import torch
from torch import nn
from torch_geometric.nn import SAGEConv

class SAGECreditRisk(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim]*num_layers
        for i in range(num_layers):
            layers.append(SAGEConv(dims[i], dims[i+1]))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        h = x
        for conv in self.layers:
            h = self.act(conv(h, edge_index))
            h = self.drop(h)
        logits = self.head(h).squeeze(-1)
        return logits
