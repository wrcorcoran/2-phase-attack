import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims=[16], dropout=0.5):
        super().__init__()
        layers = []

        for dim in hidden_dims:
            layers.append(SAGEConv(in_channels, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = dim
        
        layers.append(SAGEConv(in_channels, out_channels))
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x, edge_index):
        for layer in self.conv:
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x
    
    def reset_parameters(self):
        for layer in self.conv:
            if isinstance(layer, SAGEConv):
                layer.reset_parameters()
