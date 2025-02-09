import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims=[16], head_sizes=[4, 1], dropout=0.5):
        super().__init__()
        layers = []

        if len(head_sizes) != len(hidden_dims) + 1:
            raise ValueError("head_sizes must have one more element than hidden_dims")

        for dim, heads in zip(hidden_dims, head_sizes[:-1]):
            layers.append(GATConv(in_channels, dim, heads=heads, dropout=dropout))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = dim * heads  # GAT outputs concatenated heads
        
        layers.append(GATConv(in_channels, out_channels, heads=head_sizes[-1], concat=False, dropout=dropout))
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x, edge_index):
        for layer in self.conv:
            if isinstance(layer, GATConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x
    
    def reset_parameters(self):
        for layer in self.conv:
            if isinstance(layer, GATConv):
                layer.reset_parameters()