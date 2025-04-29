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
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import attack_utils as util

# torch.use_deterministic_algorithms(True)

import torch
import torch.nn.functional as F
import scipy.sparse as sp
from torch import autograd
from torch_geometric.utils import degree, to_scipy_sparse_matrix
from tqdm.auto import tqdm


class MetaPGD(torch.nn.Module):
    def __init__(self, data, device):
        super().__init__()
        self.data = data
        self.num_budgets = None
        self.structure_attack = None
        self.device = torch.device(device)
        self.ori_data = data.to(self.device)

        self.adjacency_matrix: sp.csr_matrix = to_scipy_sparse_matrix(
            data.edge_index, num_nodes=data.num_nodes
        ).tocsr()

        self._degree = degree(
            data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float
        )

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.num_feats = data.x.size(1)
        self.nodes_set = set(range(self.num_nodes))
        self.label = data.y
        self.surrogate = None

        self.feat = self.ori_data.x
        self.edge_index = self.ori_data.edge_index
        self.edge_weight = self.ori_data.edge_weight

        # Reinitialize adjacency perturbations
        self.adj_changes = torch.zeros(
            (self.num_nodes, self.num_nodes), device=self.device, requires_grad=True
        )

        # Restore Weights and Velocities for Momentum Training
        self.weights = []
        self.w_velocities = []
        self.loss_list = []

    def setup_surrogate(
        self,
        surrogate: torch.nn.Module,
        labeled_nodes: torch.Tensor,
        unlabeled_nodes: torch.Tensor,
        lr: float = 0.1,
        epochs: int = 100,
        momentum: float = 0.9,
        lambda_: float = 0.0,
        *,
        tau: float = 1.0,
    ):
        surrogate.eval()
        if hasattr(surrogate, "cache_clear"):
            surrogate.cache_clear()

        for layer in surrogate.modules():
            if hasattr(layer, "cached"):
                layer.cached = False

        self.surrogate = surrogate.to(self.device)
        self.tau = tau

        if labeled_nodes.dtype == torch.bool:
            labeled_nodes = labeled_nodes.nonzero().view(-1)
        labeled_nodes = labeled_nodes.to(self.device)

        if unlabeled_nodes.dtype == torch.bool:
            unlabeled_nodes = unlabeled_nodes.nonzero().view(-1)
        unlabeled_nodes = unlabeled_nodes.to(self.device)

        self.labeled_nodes = labeled_nodes
        self.unlabeled_nodes = unlabeled_nodes

        self.y_train = self.label[labeled_nodes]
        self.y_self_train = self.estimate_self_training_labels(unlabeled_nodes)
        self.adj = self.get_dense_adj()

        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.lambda_ = lambda_

        # Initialize weights & velocities
        self.weights = []
        self.w_velocities = []
        for para in self.surrogate.parameters():
            if para.ndim == 2:
                para = para.t()
                self.weights.append(torch.zeros_like(para, requires_grad=True))
                self.w_velocities.append(torch.zeros_like(para))

    def estimate_self_training_labels(self, nodes=None):
        """Predicts pseudo-labels for self-training using the surrogate model."""
        self_training_labels = self.surrogate(self.feat, self.edge_index)
        if nodes is not None:
            self_training_labels = self_training_labels[nodes]
        return self_training_labels.argmax(-1)
    
    def reset(self):
        """Resets adjacency perturbations and tracking variables."""
        self.adj_changes = torch.zeros((self.num_nodes, self.num_nodes), device=self.device, requires_grad=True)
        self._removed_edges = {}
        self._added_edges = {}
        self.degree = self._degree.clone()
        return self
    
    def get_perturbed_adj(self, adj_changes=None):
        """Returns the perturbed adjacency matrix after applying the attack."""
        if adj_changes is None:
            adj_changes = self.adj_changes
    
        adj_changes_triu = torch.triu(adj_changes, diagonal=1)
        adj_changes_symm = self.clip(adj_changes_triu + adj_changes_triu.t())
        modified_adj = adj_changes_symm + self.adj
        return modified_adj
    
    def clip(self, matrix):
        """Ensures the perturbation values remain within valid bounds."""
        return torch.clamp(matrix, 0., 1.)  # Updated to fit Meta-PGD constraints
    
    def reset_parameters(self, seed=42):
        """Reinitializes model parameters and velocity terms for momentum updates."""
        torch.manual_seed(seed)
        
        for w, wv in zip(self.weights, self.w_velocities):
            torch.nn.init.xavier_uniform_(w)  # Reinitialize weights
            wv.zero_()  # Reset velocities
    
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].detach().requires_grad_()
            self.w_velocities[i] = self.w_velocities[i].detach()



    def filter_potential_singletons(self, modified_adj, degree):
        modified_degree = degree + modified_adj.sum(1)
        mask = (modified_degree > 0).float()
        return mask.view(-1, 1) * mask.view(1, -1)


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        t_d_min = torch.tensor(2.0).to(self.device)
        t_possible_edges = np.array(np.triu(np.ones((self.num_nodes, self.num_nodes)), k=1).nonzero()).T
        allowed_mask, current_ratio = util.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff)
        return allowed_mask, current_ratio
        
    def forward(self, adj, x):
        """"""
        h = x
        for w in self.weights[:-1]:
            h = adj @ (h @ w)
            h = h.relu()

        return adj @ (h @ self.weights[-1])

    def inner_train(self, adj, feat):
        """Trains the surrogate model on the perturbed adjacency matrix."""
        self.reset_parameters()
        loss = 0
        for _ in range(self.epochs):
            out = self(adj, feat)
    
            # Use Logit Margin Loss instead of CE - agdr paper uses CE for training and margin for eval during grad computation.
            # def logit_margin_loss(logits, y):
            #     true_class_logits = logits[torch.arange(len(y)), y]  # logits of correct class
            #     top_wrong_logits = logits.topk(2, dim=1)[0][:, 1]  # 2nd highest logit
            #     return (true_class_logits - top_wrong_logits).mean()
    
            # loss = logit_margin_loss(out[self.labeled_nodes], self.y_train)
            loss = F.cross_entropy(out[self.labeled_nodes], self.y_train)
    
            grads = torch.autograd.grad(loss, self.weights, create_graph=True)
    
            self.w_velocities = [
                self.momentum * v + g for v, g in zip(self.w_velocities, grads)
            ]
    
            self.weights = [
                w - self.lr * v
                for w, v in zip(self.weights, self.w_velocities)
            ]

        return loss

    def dense_gcn_norm(self, adj, improved = False,
                   add_self_loops = True, rate = -0.5):
        fill_value = 2. if improved else 1.
        if add_self_loops:
            adj = self.dense_add_self_loops(adj, fill_value)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow_(rate)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        norm_src = deg_inv_sqrt.view(1, -1)
        norm_dst = deg_inv_sqrt.view(-1, 1)
        adj = norm_src * adj * norm_dst
        return adj

    def attack(self, num_budgets=0.05, *, structure_attack=True,
               feature_attack=False, disable=False, ll_cutoff=0.004,
               iterations=200, base_lr=0.001, xi=1e-5, grad_clip=1.0, sampling_tries=100):
    
        self.num_budgets = int((self.num_edges // 2) * num_budgets)
        self.structure_attack = structure_attack
    
        if feature_attack:
            raise NotImplementedError("Feature attack is not supported in Meta-PGD.")
    
        adj_changes = self.adj_changes
        modified_adj = self.adj
    
        adj_changes.requires_grad_(True)  # Ensure we can compute gradients
    
        num_nodes = self.num_nodes
    
        # Perform PGD Iterations
        for itr in tqdm(range(iterations), desc='Running PGD Attack...', disable=disable):
            modified_adj = self.get_perturbed_adj(adj_changes)
            adj_norm = self.dense_gcn_norm(modified_adj)
        
            temp_loss = self.inner_train(adj_norm, self.feat)  # Train with the perturbed graph
            self.loss_list.append(temp_loss)

            adj_grad, _ = self.compute_gradients(adj_norm, self.feat)
        
            if grad_clip is not None:
                grad_norm = adj_grad.norm()
                if grad_norm > grad_clip:
                    adj_grad *= grad_clip / grad_norm
        
            lr = base_lr * self.num_budgets / math.sqrt(itr + 1)
            with torch.no_grad():
                adj_changes -= lr * adj_grad  
                adj_changes.clamp_(0, 1)  # Keep values in [0,1]

    
                # Projection Step (Ensure Budget Constraint)
                if adj_changes.sum() > self.num_budgets:
                    top = adj_changes.max().item()
                    bot = (adj_changes.min() - 1).clamp_min(0).item()
                    mu = (top + bot) / 2
                    while (top - bot) / 2 > xi:
                        used_budget = (adj_changes - mu).clamp(0, 1).sum()
                        if used_budget == self.num_budgets:
                            break
                        elif used_budget > self.num_budgets:
                            bot = mu
                        else:
                            top = mu
                        mu = (top + bot) / 2
                    adj_changes.sub_(mu).clamp_(0, 1)
    
        # Sampling Step: Convert Continuous Perturbations to Discrete
        # Is random sampling the best choice?

        best_loss = float("inf")
        best_pert = None
    
        adj_changes.detach_()
        k = 0
        pbar = tqdm(total=sampling_tries, leave=False) if not disable else None
        while k < sampling_tries:
            flip_sample = adj_changes.bernoulli()
            print(self.num_budgets, flip_sample.count_nonzero())
            if flip_sample.count_nonzero() <= self.num_budgets:
                flip_sample = torch.triu(flip_sample, diagonal=1)  # Keep upper triangle
                flip_sample = flip_sample + flip_sample.t()  # Mirror to lower triangle
                k += 1
                if pbar:
                    pbar.update(1)
                loss = self.compute_loss(flip_sample)
                if loss < best_loss:
                    best_loss = loss
                    #best_pert = flip_sample.nonzero()
                    best_pert = flip_sample
            # else:
            #     print("Error with budget control")
    
        if pbar:
            pbar.close()
    
        # Update final perturbations
        self.adj_changes = best_pert
        return best_loss, best_pert


    def compute_loss(self, A_flip):
        """Computes the attack loss after applying perturbations using get_perturbed_adj."""
    
        A_pert = self.get_perturbed_adj(A_flip)
    
        self.inner_train(A_pert, self.feat)
    
        scores = self(A_pert, self.feat)
    
        # def logit_margin_loss(logits, y):
        #     true_class_logits = logits[torch.arange(len(y)), y]  # Logits of correct class
        #     top_wrong_logits = logits.topk(2, dim=1)[0][:, 1]  # 2nd highest logit
        #     return (true_class_logits - top_wrong_logits).mean()
    
        return self.logit_margin_loss(scores[self.unlabeled_nodes], self.y_self_train)

    def logit_margin_loss(self, logits, y):
        all_nodes = torch.arange(y.shape[0])
    
        scores_true = logits[all_nodes, y]
    
        # Clone and mask out the true class - agdr does it like this
        scores_mod = logits.clone()
        scores_mod[all_nodes, y] = -float("inf")  # Prevent selecting true class
    
        # Get the highest wrong class logit
        scores_pred_excl_true = scores_mod.amax(dim=-1)
    
        return (scores_true - scores_pred_excl_true).tanh().mean()

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
    
    def structure_score(self, modified_adj, adj_grad):
        score = adj_grad * (1 - 2 * modified_adj)
        score -= score.min()
        score = torch.triu(score, diagonal=1)
        return score.view(-1)

    def dense_add_self_loops(self, adj, fill_value = 1.0) -> Tensor:
        diag = torch.diag(adj.new_full((adj.size(0), ), fill_value))
        return adj + diag

    def feature_score(self, modified_feat, feat_grad):
        score = feat_grad * (1 - 2 * modified_feat)
        score -= score.min()
        return score.view(-1)

    def compute_gradients(self, modified_adj, modified_feat):
        logit = self(modified_adj, modified_feat) / self.tau
    
        # Use Logit Margin Loss instead of Cross-Entropy
        # def logit_margin_loss(logits, y):
        #     true_class_logits = logits[torch.arange(len(y)), y]  # Logits of correct class
        #     top_wrong_logits = logits.topk(2, dim=1)[0][:, 1]  # 2nd highest logit
        #     return (true_class_logits - top_wrong_logits).mean()
    
        loss = self.logit_margin_loss(logit[self.labeled_nodes], self.y_train)
        #loss = self.logit_margin_loss(logit[self.unlabeled_nodes], self.y_self_train)
        #loss = F.cross_entropy(logit[self.unlabeled_nodes], self.y_self_train)
        
        #Only structure attack is required
        if self.structure_attack:
            return torch.autograd.grad(loss, self.adj_changes)[0], None


# def handle_new_edges(data, attacker, device):
#     added, removed = list(attacker._added_edges.keys()), list(attacker._removed_edges.keys())
#     new_data = copy.deepcopy(data)
#     for u, v in added:
#         edge1 = torch.tensor([[u], [v]]).to(device)
#         edge2 = torch.tensor([[v], [u]]).to(device)
#         new_data.edge_index = torch.cat([new_data.edge_index, edge1], dim=1)
#         new_data.edge_index = torch.cat([new_data.edge_index, edge2], dim=1)

#     print(new_data.edge_index.shape)

#     for u, v in removed:
#         edge_to_delete = torch.tensor([[u, v], [v, u]]).to(device)
#         mask = ~((new_data.edge_index == edge_to_delete[:, 0:1]).all(dim=0) | 
#          (new_data.edge_index == edge_to_delete[:, 1:2]).all(dim=0))
#         new_data.edge_index = new_data.edge_index[:, mask]

#     return new_data

            #Do I need to compute adj_grad_score to get singleton_mask and ll_mask 
            #or can I do it with adj_grad without going through the trouble of masks
            
            # adj_grad_score = self.structure_score(modified_adj, adj_grad)
            # singleton_mask = self.filter_potential_singletons(modified_adj, self.degree)
            # ll_mask = self.log_likelihood_constraint(modified_adj, self.adj, ll_cutoff)[0].to(self.device)
            
            # adj_grad_score *= singleton_mask.view(-1)
            # adj_grad_score *= ll_mask.view(-1)
        
            # adj_grad *= (adj_grad_score > 0).float().view(adj_grad.shape)


def handle_new_edges(data, attacker, device):

    modified_adj = attacker.get_perturbed_adj(attacker.adj_changes).to(device)

    # Convert the dense adjacency matrix to edge_index format
    edge_index, edge_weight = dense_to_sparse(modified_adj)

    new_data = copy.deepcopy(data)
    new_data.edge_index = edge_index
    new_data.edge_weight = edge_weight

    print(f"Updated edge index: {new_data.edge_index.shape}")

    return new_data