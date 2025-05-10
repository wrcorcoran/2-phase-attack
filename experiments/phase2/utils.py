import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.utils import to_networkx, from_networkx
from deeprobust.graph.defense import GCNJaccard
from deeprobust.graph.data import Dataset as DRDataset
from deeprobust.graph.data import Pyg2Dpr
from torch_geometric.nn import GraphSAGE
from torch_geometric.loader import LinkNeighborLoader
from sklearn.linear_model import LogisticRegression
import networkx as nx
import numpy as np
import random, math
import matplotlib.pyplot as plt
from enum import Enum, auto
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Models(Enum):
    GCN = auto()
    GSAGE = auto()
    GSAINT = auto()
    GAT = auto()
    GCNJACCARD = auto()

class Dataset:
    def __init__(self, root='/tmp/Cora', name='Cora', device='cuda'):
        self.root = root
        self.name = name
        self.device = device
        self.data = None
        self.in_feats = None
        self.h_feats = None
        self.num_classes = None
        self.reset_dataset()

    def reset_dataset(self):
        if (self.name == "Cora_ML"):
            cora_dataset = CitationFull(root='/tmp/CoraML', name='Cora_ML')
            self.data = cora_dataset[0]
    
            num_nodes = self.data.num_nodes
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            # Assign masks (this is just an example of random splitting)
            indices = torch.randperm(num_nodes)
            train_indices = indices[:int(0.2 * num_nodes)]
            test_indices = indices[int(0.8 * num_nodes):]
            
            train_mask[train_indices] = True
            test_mask[test_indices] = True
            
            self.data.train_mask = train_mask
            self.data.test_mask = test_mask
        else:
            cora_dataset = Planetoid(root=self.root, name=self.name)
            self.data = cora_dataset[0]
        self.data = self.data.to(self.device)

        self.in_feats = self.data.x.shape[1]
        self.h_feats = 64
        self.num_classes = cora_dataset.num_classes

    def get_data(self):
        self.reset_dataset()
        return self.data, self.in_feats, self.h_feats, self.num_classes

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        h, edge_index = data.x, data.edge_index

        h = F.dropout(h, p=0.6, training=self.training)
        h = F.elu(self.conv1(h, edge_index))
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.conv2(h, edge_index)

        return h

class GSAINT(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super().__init__()
        data, in_feats, h_feats, num_classes = dataset.get_data()
        
        in_channels = in_feats
        out_channels = num_classes
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)

def get_model(path, in_feats, h_feats, num_classes, dataset_name, kind, data=None):
    if kind == Models.GCN:
        model = GCN(in_feats, h_feats, num_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(f'{path}models/gcn/{dataset_name}/{dataset_name}_gcn.pt'))
        model.eval()

        return model

    if kind == Models.GAT:
        heads = 8
        model = GAT(in_feats, h_feats, num_classes, heads)
        model = model.to(device)
        model.load_state_dict(torch.load(f'{path}models/gat/{dataset_name}/{dataset_name}_gat.pt'))
        model.eval()

        return model

    if kind == Models.GSAGE:
        heads = 8
        model = GraphSAGE(
            in_feats,
            h_feats,
            num_layers=2,
        ).to(device)
        model.load_state_dict(torch.load(f'{path}models/gsage/{dataset_name}/{dataset_name}_gsage.pt'))
        model = model.to(device)
        model.eval()

        return model

    if kind == Models.GSAINT:
        model = GSAINT(data, hidden_channels=64).to(device)
        # heads = 8
        # model = GAT(in_feats, h_feats, num_classes, heads)
        # model = model.to(device)
        model.load_state_dict(torch.load(f'{path}models/gsaint/{dataset_name}/{dataset_name}_gsaint.pt'))
        model.eval()

        return model

    if kind == Models.GCNJACCARD:
        data = Pyg2Dpr(Planetoid(root='/tmp/', name=dataset_name))
        # data = DRDataset(root='/tmp/', name=dataset_name)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        
        model = GCNJaccard(nfeat=in_feats, nclass=num_classes,
                nhid=h_feats, device=device)

        model.fit(features, data.adj, labels, idx_train, idx_val, threshold=0.01, verbose=False)
        model.eval()
        # model = GCNJaccard(nfeat=in_feats, nclass=num_classes,
        #         nhid=h_feats, device=device)
        # model = model.to(device)
        
        # model.load(torch.load(f'../../../models/gcnjaccard/{dataset_name}/{dataset_name}_gcn_jaccard.pt'))
        # model.eval()

        return model
    
    raise ValueError("Not valid model type")

@torch.no_grad()
def test_model(model, d, GCNtype, testMask=False):
    model.eval()
    if (GCNtype == Models.GCNJACCARD):
        out = model.test(d)
        return out
    elif GCNtype == Models.GSAGE:
        data = d.to(device, 'x', 'edge_index')
        
        model.eval()
        out = model(data.x, data.edge_index).cpu()
        
        clf = LogisticRegression()
        clf.fit(out[data.train_mask.cpu()].cpu(), data.y[data.train_mask.cpu()].cpu())
        
        predictions = clf.predict(out.cpu())
        correct = torch.tensor(predictions == data.y.cpu().numpy())
        accuracy = correct.sum().item() / len(data.y)
        
        return accuracy, predictions
    elif (GCNtype == Models.GSAINT):
        model.eval()
        model.set_aggr('mean')
    
        out = model(d.x.to(device), d.edge_index.to(device))
        pred = out.argmax(dim=-1)
        correct = pred.eq(d.y.to(device))

        accuracy = correct.sum().item() / len(d.y)

        return accuracy, pred
    
        # accs = []
        # for _, mask in d('test_mask'):
        #     accs.append(correct[mask].sum().item() / mask.sum().item())
        # return accs[0], pred
    else:
        out = model(d)
    pred = out.argmax(dim=1)

    if (testMask):
        acc = (pred[d.test_mask] == d.y[d.test_mask]).sum().item() / d.test_mask.sum().item()
        return acc, pred
    else:
        acc = (pred == d.y).sum().item() / len(d.y)
        return acc, pred

def output_accuracy_change(gt, cv):
    print("\n----")
    if gt != cv:
        print(f'The accuracy has changed by {cv - gt:.4f}')
    else:
        print("The accuracy has not changed.")

def convert_to_networkx(g):
    G = to_networkx(g)
    x = g.x
    y = g.y
    train_mask = g.train_mask
    test_mask = g.test_mask

    return G, x, y, train_mask, test_mask
    
def convert_to_pyg(G, x, y, train_mask, test_mask):
    d = from_networkx(G).to(device)

    d.x = x
    d.y = y
    d.train_mask = train_mask
    d.test_mask = test_mask
    d.num_nodes = None

    return d

def add_edge(g, i, j, undirected):
    if g.has_edge(i, j):
        return;
    
    if undirected:
        g.add_edge(i, j)
        g.add_edge(j, i)
    else:
        g.add_edge(i, j)

def get_ground_truth(model, data, GNNtype, testMask):
    return test_model(model, data, GNNtype, testMask)

def number_added_edges(init, final, is_undirected):
    change = final - init

    if is_undirected:
        percentage_change = (change / init) * 100 if init != 0 else 0
        print("Change in edges: ", change / 2, " | Percentage change: {:.2f}%".format(percentage_change))
    else:
        percentage_change = (change / init) * 100 if init != 0 else 0
        print("Change in edges: ", change, " | Percentage change: {:.2f}%".format(percentage_change))


def print_graph(G):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=1)
    
    nx.draw_networkx_edges(G, pos, alpha=0.05)
    
    plt.title("Large Graph Visualization")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def make_clique(G, s):
    s = list(s)
    for i in range(0, len(s)):
        for j in range(i + 1, len(s)):
            add_edge(G, s[i], s[j], undirected=True)

def get_total_homophily(data):
    G, x, y, train_mask, test_mask = convert_to_networkx(data)
    same = 0
    num_edges = 0
    
    for i in range(0, G.number_of_nodes()):
        edges = G.out_edges(i)
        for e in edges:
            if y[e[0]] == y[e[1]]:
                same += 1
            num_edges += 1

    return same / num_edges

def get_node_homophily(G, n, y, c):
    edges = G.out_edges(n)
    val = 0
    
    for edge in edges:
        if y[edge[1]].item() == c:
            val += 1
            
    return 1 if len(edges) == 0 else val / len(edges)

def get_node_homophily_across_classes(G, n, y, num_classes):
    edges = G.out_edges(n)
    vals = {key: 0 for key in range(num_classes)}
    for edge in edges:
        vals[y[edge[1]].item()] += 1
        
    for v in vals.keys():
        vals[v] = vals[v]/len(edges)
    
    return vals

def isSatisfied(n, c, dict, val):
    # print("satisfied", math.floor(dict[n][0][c] * (1 + val)) - dict[n][1][c])
    return not math.floor(dict[n][0][c] * (1 + val)) - dict[n][1][c] > 0

def getClassConnectivity(G, n, y, num_classes):
    dict = {i: 0 for i in range(0, num_classes)}
    for neighbor in G.adj[n]:
        dict[y[neighbor].item()] += 1
    return dict