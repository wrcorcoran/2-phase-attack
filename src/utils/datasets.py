from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import AttributedGraphDataset
import torch

def get_cora(device):
    cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = cora_dataset[0].to(device)

    return data

def edge_index_to_A(edge_index, num_nodes, device):
    A = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1)).to(device), size=(num_nodes, num_nodes)
    )

    return A.to_dense()

def A_to_edge_index(adj_matrix):
    edge_index = adj_matrix.nonzero(as_tuple=False).t()
    return edge_index