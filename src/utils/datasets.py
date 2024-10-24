from torch_geometric.datasets import Planetoid

def get_cora(device):
    cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = cora_dataset[0].to(device)

    return data