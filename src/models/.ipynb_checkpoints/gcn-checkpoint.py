import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from tqdm.auto import tqdm, trange
from copy import deepcopy

import sys
sys.path.append('../')
from utils.datasets import *

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h

    def train_model(self, data, differentiable=False):
        self.train()
        logits = self(data.cuda())
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask]) 

        self.optimizer.zero_grad()
        if differentiable:
            grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            for param, grad in zip(self.parameters(), grads):
                param.grad = grad
        else:
            loss.backward()
    
        self.optimizer.step()
    
        return loss.item()

    def test(self, data):
        self.eval()
        out = self(data.cuda())
        pred = out.argmax(dim=1)
    
        acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        return acc

    def fit(self, data, epochs=200, **kwargs):
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            loss = self.train_model(data, **kwargs)
            acc = self.test(data)