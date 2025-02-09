import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor, fill_diag
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import MessagePassing

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, self_loops=True):
        super().__init__()
        self.self_loops = self_loops

        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, edge_index):
        x = self.lin(x)

        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=None)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = self.propagate(edge_index, x=x, norm=norm)

        out += self.bias

        return out

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dims = [16], dropout = 0.5):
        super().__init__()
        layers = []

        # for dim in hidden_dims:
        #     conv.append(GCNConv(in_channels, dim))
        #     conv.append(nn.ReLU())
        #     conv.append(nn.Dropout(dropout))
        #     in_channels = dim
        # conv.append(GCNConv(in_channels, out_channels))
        # self.conv = nn.ModuleList(conv)
        for dim in hidden_dims:
            layers.append(GCNConv(in_channels, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = dim
        
        layers.append(GCNConv(in_channels, out_channels))
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x, edge_index):
        for layer in self.conv:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

    def reset_parameters(self):
        for layer in self.conv:
            if isinstance(layer, GCNConv):
                layer.reset_parameters()