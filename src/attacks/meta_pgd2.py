import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.autograd import grad
from torch.nn import init
from tqdm.auto import tqdm
import scipy.sparse as sp
from torch_geometric.utils import degree, to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
import copy
from torch_geometric.utils import dense_to_sparse
from torch.nn.parameter import Parameter

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import attack_utils as util

# torch.use_deterministic_algorithms(True)

class MetaPGD(torch.nn.Module):
    def __init__(self, data, device='cpu', undirected=True):
        super().__init__()
        self.undirected=undirected
        self.data = data
        self.num_budgets = None
        self.structure_attack = None
        self.device = torch.device(device)
        self.ori_data = data.to(self.device)
        self.modified_adj=None
        self.adjacency_matrix: sp.csr_matrix = to_scipy_sparse_matrix(
            data.edge_index, num_nodes=data.num_nodes).tocsr()

        self._degree = degree(data.edge_index[0], num_nodes=data.num_nodes,
                              dtype=torch.float)

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.num_feats = data.x.size(1)
        self.nodes_set = set(range(self.num_nodes))
        self.label = data.y
        self.surrogate = None
        self.feat = self.ori_data.x
        self.edge_index = self.ori_data.edge_index
        self.edge_weight = self.ori_data.edge_weight
        self.adj_changes = Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes)).to(self.device)
        self.adj_changes.data.fill_(0)


    def setup_surrogate(self, surrogate: torch.nn.Module,
                        labeled_nodes: Tensor, unlabeled_nodes: Tensor,
                        lr: float = 0.1, epochs: int = 100,
                        momentum: float = 0.9, lambda_: float = 0., *,
                        tau: float = 1.0):

        surrogate.eval()
        if hasattr(surrogate, 'cache_clear'):
            surrogate.cache_clear()

        for layer in surrogate.modules():
            if hasattr(layer, 'cached'):
                layer.cached = False
        
        self.surrogate = surrogate.to(self.device)
        self.tau = tau

        if labeled_nodes.dtype == torch.bool:
            labeled_nodes = labeled_nodes.nonzero().view(-1)
        labeled_nodes = labeled_nodes.to(self.device)

        if unlabeled_nodes.dtype == torch.bool:
            unlabeled_nodes = unlabeled_nodes.nonzero().view(-1)
        unlabeled_nodes = unlabeled_nodes.to(self.device)

        #Train nodes
        self.labeled_nodes = labeled_nodes
        #Test nodes
        self.unlabeled_nodes = unlabeled_nodes

        self.y_train = self.label[self.labeled_nodes]
        self.y_stl_unlabeled = self.estimate_self_training_labels(unlabeled_nodes)
        self.adj = self.get_dense_adj()

        # weights = []
        # w_velocities = []
        self.weights, self.w_velocities = [], []

        for para in self.surrogate.parameters():
            if para.ndim == 2:
                para = para.t()
                weight = Parameter(torch.FloatTensor(para.shape[0],para.shape[1]).to(self.device))
                w_velocity = torch.zeros(weight.shape).to(self.device)
                self.weights.append(weight)
                self.w_velocities.append(w_velocity)
                # weights.append(torch.zeros_like(para, requires_grad=True))
                # w_velocities.append(torch.zeros_like(para))

        #self.weights, self.w_velocities = weights, w_velocities
        
        
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.lambda_ = lambda_
        self.reset()

    def _sym(self,triu):
        """
        Constructs a matrix whose values only depend on the upper triangle of "triu", and whose diagonal is
        always zero. This way, each edge in the graph is only parameterized by one number and not two, and the
        diagonal is not parameterized at all. Thereby, we avoid non-symmetric and diagonal adjustments of "triu".
        """
    
        triu = triu.triu(diagonal=1)
        return triu + triu.T

    def estimate_self_training_labels(
            self, nodes = None):
        #The nodes passed need to be the train labels with idx_train
        self_training_labels = self.surrogate(self.feat, self.edge_index)
        if nodes is not None:
            self_training_labels = self_training_labels[nodes]
        return self_training_labels.argmax(-1)
    
    def reset(self):
        #self.adj_changes = torch.zeros_like(self.adj)
        # #self.adj_changes = Parameter(torch.FloatTensor(num_nodes, num_nodes))
        # self.adj_changes.data.fill_(0)
        # self.feat_changes = torch.zeros_like(self.feat)
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        self._removed_edges = {}
        self._added_edges = {}
        self._removed_feats = {}
        self._added_feats = {}
        self.degree = self._degree.clone()
        return self

    
    def get_perturbed_adj(self, ori_adj, adj_changes):
        #This version of the function works with the original adj as input. I dont want to move adj changes around too much and keep self.adj_changes as the only copy that gets modified
        #This version runs on the assumption that the additions during the attack are not symmetric
        adj_changes_square = adj_changes - torch.diag(torch.diag(adj_changes, 0))
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = self.clip(adj_changes_square)
        modified_adj = adj_changes_square + ori_adj
        return modified_adj
    

    def get_perturbed_feat(self, feat_changes=None):
        if feat_changes is None:
            feat_changes = self.feat_changes
        return self.feat + self.clip(feat_changes)

    #Should meta pgd be clipping between 0 and 1?
    def clip(self, matrix):
        clipped_matrix = torch.clamp(matrix, 0., 1.)
        return clipped_matrix

    def reset_parameters(self, seed=42):
        #torch.manual_seed(seed)
        for w, wv in zip(self.weights, self.w_velocities):
            init.xavier_uniform_(w)
            init.zeros_(wv)

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].detach()
            self.weights[i].requires_grad = True
            self.w_velocities[i] = self.w_velocities[i].detach()
            self.w_velocities[i].requires_grad = True

    def filter_potential_singletons(self, modified_adj):
        #Copied this off from deeprobust
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        if self.undirected:
            l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        return flat_mask
        
    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
         #Also copied from deeprobust
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.

        Note that different data type (float, double) can effect the final results.
        """
        t_d_min = torch.tensor(2.0).to(self.device)
        if self.undirected:
            t_possible_edges = np.array(np.triu(np.ones((self.num_nodes, self.num_nodes)), k=1).nonzero()).T
        else:
            t_possible_edges = np.array((np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.nnodes)).nonzero()).T
        allowed_mask, current_ratio = util.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff, undirected=self.undirected)
        return allowed_mask, current_ratio

    
    def forward(self, adj, x):
        """"""
        h = x
        for w in self.weights[:-1]:
            h = adj @ (h @ w)
            h = F.relu(h)

        return adj @ (h @ self.weights[-1])

    #Should be getting adj_norm
    def inner_train(self, adj, feat):
        self.reset_parameters()

        for _ in range(self.epochs):
            out = self(adj, feat)
            #This has to be on the original train ids
            loss = F.cross_entropy(out[self.labeled_nodes], self.y_train)
            grads = torch.autograd.grad(loss, self.weights, create_graph=True)

            self.w_velocities = [
                self.momentum * v + g for v, g in zip(self.w_velocities, grads)
            ]

            self.weights = [
                w - self.lr * v
                for w, v in zip(self.weights, self.w_velocities)
            ]

    def dense_gcn_norm(self, adj):
        """Normalize adjacency tensor matrix.
        """
        device = adj.device
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

    def attack(self, num_budgets=0.05, *, structure_attack=True, symmetric=True,
               feature_attack=False, disable=False, ll_cutoff=0.004, ll_constraint=False,
               iterations=200, base_lr=0.001, xi=1e-5, grad_clip=1.0, sampling_tries=100):

        self.num_budgets = int((self.num_edges // 2) * num_budgets)
        self.structure_attack = structure_attack
        

        # adj_changes = self.adj_changes
        # feat_changes = self.feat_changes
        
        modified_adj = self.adj#.clone()
        num_nodes, num_feats = self.num_nodes, self.num_feats

        for itr in tqdm(range(iterations), desc='Running PGD Attack...', disable=disable):
            # if symmetric:
            #     adj_changes_sym = self._sym(self.adj_changes)
            #     adj_grad = torch.autograd.grad(adj_changes_sym, self.adj_changes, grad_outputs = self.compute_gradients(adj_changes_sym))[0]
            # else:
            #     adj_grad = self.compute_gradients(self.adj_changes)

            modified_adj = self.get_perturbed_adj(self.adj,self.adj_changes)
            adj_grad = self.compute_gradients(self.adj_changes)

            #adj_grad = self.structure_score(modified_adj, adj_grad, self.adj, False, ll_cutoff)
            
            if grad_clip is not None:
                grad_norm = adj_grad.square().sum()
                if grad_norm > grad_clip*grad_clip:
                    adj_grad *= grad_clip / grad_norm.sqrt()


            
            lr = base_lr * self.num_budgets / math.sqrt(itr + 1)
            with torch.no_grad():
                self.adj_changes -= lr * adj_grad
                if self.adj_changes.clamp(0, 1).sum() <= self.num_budgets:  # Keep values in [0,1]
                    self.adj_changes.clamp_(0, 1)  # Keep values in [0,1]
                else:
                # Projection Step (Ensure Budget Constraint)
                # if adj_changes.sum() > self.num_budgets:
                    top = self.adj_changes.max().item()
                    bot = (self.adj_changes.min() - 1).clamp_min(0).item()
                    mu = (top + bot) / 2
                    while (top - bot) / 2 > xi:
                        used_budget = (self.adj_changes - mu).clamp(0, 1).sum()
                        if used_budget == self.num_budgets:
                            break
                        elif used_budget > self.num_budgets:
                            bot = mu
                        else:
                            top = mu
                        mu = (top + bot) / 2
                    self.adj_changes.sub_(mu).clamp_(0, 1)

        best_loss = np.inf
        best_pert = None
        self.adj_changes.detach_()
        k = 0
        pbar = tqdm(total=sampling_tries, leave=False) if not disable else None
        while k < sampling_tries:
            adj_changes_sample = self.adj_changes.bernoulli()
            print(adj_changes_sample.count_nonzero())
            if adj_changes_sample.count_nonzero() <= self.num_budgets:
                k += 1
                if pbar:
                    pbar.update(1)
                loss = self.compute_loss(self._sym(adj_changes_sample) if symmetric else adj_changes_sample)
                print(loss.item(), best_loss)
                if loss < best_loss:
                    best_loss = loss
                    best_pert = adj_changes_sample.nonzero()
        if pbar:
            pbar.close()

        return best_pert, best_loss
            
        
    def unravel_index(self, index, array_shape):
        rows = torch.div(index, array_shape[1], rounding_mode='trunc')
        cols = index % array_shape[1]
        return rows, cols

    def get_dense_adj(self):
        data = self.ori_data
        # adj_t = data.get('adj_t')
        # if isinstance(adj_t, Tensor):
        #     return adj_t.t().to(self.device)
        # elif isinstance(adj_t, SparseTensor):
        #     return adj_t.to_dense().to(self.device)
        
        return self.to_dense_adj(data.edge_index, data.edge_weight,
                            self.num_nodes).to(self.device)

    def to_dense_adj(self,
        edge_index,
        edge_weight,
        num_nodes,
        fill_value = 1.0,
    ):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        if edge_weight is None:
            adj[edge_index[0], edge_index[1]] = fill_value
        else:
            adj[edge_index[0], edge_index[1]] = edge_weight
        return adj

    def add_edge(self, u, v, it = None):
        self._added_edges[(u, v)] = it
        self.degree[u] += 1
        self.degree[v] += 1

    def remove_edge(self, u, v, it = None):
        self._removed_edges[(u, v)] = it
        self.degree[u] -= 1
        self.degree[v] -= 1

    def structure_score(self, modified_adj, adj_grad, ori_adj, ll_constraint, ll_cutoff):
        score = adj_grad * (- 2 * modified_adj + 1)
        score -= score.min()
        #Remove self loops
        score = score - torch.diag(torch.diag(score,0))
        singleton_mask = self.filter_potential_singletons(modified_adj)
        score = score * singleton_mask
        if ll_constraint:
            allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
            allowed_mask = allowed_mask.to(self.device)
            score = score * allowed_mask
        return score

    def compute_loss(self, A_flip):
        """Computes the attack loss after applying perturbations using get_perturbed_adj."""
    
        modified_adj = self.get_perturbed_adj(self.adj,self.adj_changes)
        adj_norm = self.dense_gcn_norm(modified_adj)
        #A_pert = self.adj + A_flip * (1 - 2 * self.adj)
        self.inner_train(adj_norm, self.feat)
    
        scores = self(adj_norm, self.feat)
    
        # def logit_margin_loss(logits, y):
        #     true_class_logits = logits[torch.arange(len(y)), y]  # Logits of correct class
        #     top_wrong_logits = logits.topk(2, dim=1)[0][:, 1]  # 2nd highest logit
        #     return (true_class_logits - top_wrong_logits).mean()
        mloss = self.logit_margin_loss(scores[self.unlabeled_nodes], self.y_stl_unlabeled)
        print("Margin loss is:",mloss)
        return mloss

    def compute_gradients(self, A_flip):
        return torch.autograd.grad(self.compute_loss(A_flip), A_flip, retain_graph=True)[0]

        #Double check this with the code?
    def logit_margin_loss(self, logits, y):
        all_nodes = torch.arange(y.shape[0])
    
        scores_true = logits[all_nodes, y]
    
        # Clone and mask out the true class - agdr does it like this
        scores_mod = logits.clone()
        scores_mod[all_nodes, y] = -float("inf")  # Prevent selecting true class
    
        # Get the highest wrong class logit
        scores_pred_excl_true = scores_mod.amax(dim=-1)
    
        return (scores_true - scores_pred_excl_true).tanh().mean()
    
    # def structure_score(self, modified_adj, adj_grad):
    #     score = adj_grad * (1 - 2 * modified_adj)
    #     score -= score.min()
    #     score = torch.triu(score, diagonal=1)
    #     return score.view(-1)

    def dense_add_self_loops(self, adj, fill_value = 1.0) -> Tensor:
        diag = torch.diag(adj.new_full((adj.size(0), ), fill_value))
        return adj + diag

    def feature_score(self, modified_feat, feat_grad):
        score = feat_grad * (1 - 2 * modified_feat)
        score -= score.min()
        return score.view(-1)


    def accuracy(self, output, labels):
        """Return accuracy of output compared to labels.
    
        Parameters
        ----------
        output : torch.Tensor
            output from model
        labels : torch.Tensor or numpy.array
            node labels
    
        Returns
        -------
        float
            accuracy
        """
        if not hasattr(labels, '__len__'):
            labels = [labels]
        if type(labels) is not torch.Tensor:
            labels = torch.LongTensor(labels)
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

def handle_new_edges(data, attacker, device):

    modified_adj = attacker.get_perturbed_adj(attacker.adj_changes).to(device)

    # Convert the dense adjacency matrix to edge_index format
    edge_index, edge_weight = dense_to_sparse(modified_adj)

    new_data = copy.deepcopy(data)
    new_data.edge_index = edge_index
    new_data.edge_weight = edge_weight

    print(f"Updated edge index: {new_data.edge_index.shape}")

    return new_data