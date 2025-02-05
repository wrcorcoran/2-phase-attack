import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
from torch.nn import init
from tqdm.auto import tqdm
import scipy.sparse as sp
from torch_geometric.utils import degree, to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
import copy

class Metattack(torch.nn.Module):
    r"""Implementation of `Metattack` attack from the:
    `"Adversarial Attacks on Graph Neural Networks
    via Meta Learning"
    <https://arxiv.org/abs/1902.08412>`_ paper (ICLR'19)

    Parameters
    ----------
    data : Data
        PyG-like data denoting the input graph
    device : str, optional
        the device of the attack running on, by default "cpu"
    seed : Optional[int], optional
        the random seed for reproducing the attack, by default None
    name : Optional[str], optional
        name of the attacker, if None, it would be
        :obj:`__class__.__name__`, by default None
    kwargs : additional arguments of :class:`greatx.attack.Attacker`,

    """

    # Metattack can also conduct feature attack
    _allow_feature_attack: bool = True
    def __init__(self, data, device):
        super().__init__()
        self.data = data
        self.num_budgets = None
        self.structure_attack = None
        self.feature_attack = None
        self.device = torch.device(device)
        self.ori_data = data.to(self.device)

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

        self.labeled_nodes = labeled_nodes
        self.unlabeled_nodes = unlabeled_nodes

        self.y_train = self.label[labeled_nodes]
        self.y_self_train = self.estimate_self_training_labels(unlabeled_nodes)
        self.adj = self.get_dense_adj()

        weights = []
        w_velocities = []

        for para in self.surrogate.parameters():
            if para.ndim == 2:
                para = para.t()
                weights.append(torch.zeros_like(para, requires_grad=True))
                w_velocities.append(torch.zeros_like(para))

        self.weights, self.w_velocities = weights, w_velocities

        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.lambda_ = lambda_

    def estimate_self_training_labels(
            self, nodes = None):
        self_training_labels = self.surrogate(self.feat, self.edge_index)
        if nodes is not None:
            self_training_labels = self_training_labels[nodes]
        return self_training_labels.argmax(-1)
    
    def reset(self):
        self.adj_changes = torch.zeros_like(self.adj)
        self.feat_changes = torch.zeros_like(self.feat)
        self._removed_edges = {}
        self._added_edges = {}
        self._removed_feats = {}
        self._added_feats = {}
        self.degree = self._degree.clone()
        return self

    def get_perturbed_adj(self, adj_changes=None):
        if adj_changes is None:
            adj_changes = self.adj_changes
        adj_changes_triu = torch.triu(adj_changes, diagonal=1)
        adj_changes_symm = self.clip(adj_changes_triu + adj_changes_triu.t())
        modified_adj = adj_changes_symm + self.adj
        return modified_adj

    def get_perturbed_feat(self, feat_changes=None):
        if feat_changes is None:
            feat_changes = self.feat_changes
        return self.feat + self.clip(feat_changes)

    def clip(self, matrix):
        clipped_matrix = torch.clamp(matrix, -1., 1.)
        return clipped_matrix

    def reset_parameters(self):
        for w, wv in zip(self.weights, self.w_velocities):
            init.xavier_uniform_(w)
            init.zeros_(wv)

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].detach().requires_grad_()
            self.w_velocities[i] = self.w_velocities[i].detach()

    def forward(self, adj, x):
        """"""
        h = x
        for w in self.weights[:-1]:
            h = adj @ (h @ w)
            h = h.relu()

        return adj @ (h @ self.weights[-1])

    def inner_train(self, adj, feat):
        self.reset_parameters()

        for _ in range(self.epochs):
            out = self(adj, feat)
            loss = F.cross_entropy(out[self.labeled_nodes], self.y_train)
            grads = torch.autograd.grad(loss, self.weights, create_graph=True)

            self.w_velocities = [
                self.momentum * v + g for v, g in zip(self.w_velocities, grads)
            ]

            self.weights = [
                w - self.lr * v
                for w, v in zip(self.weights, self.w_velocities)
            ]

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
               feature_attack=False, disable=False):

        self.num_budgets = int((self.num_edges // 2) * num_budgets)
        self.structure_attack = structure_attack
        self.feature_attack = feature_attack
        
        if feature_attack:
            self._check_feature_matrix_binary()

        adj_changes = self.adj_changes
        feat_changes = self.feat_changes
        modified_adj = self.adj
        modified_feat = self.feat

        adj_changes.requires_grad_(bool(structure_attack))
        feat_changes.requires_grad_(bool(feature_attack))

        num_nodes, num_feats = self.num_nodes, self.num_feats

        for it in tqdm(range(self.num_budgets), desc='Peturbing graph...',
                       disable=disable):

            if structure_attack:
                modified_adj = self.get_perturbed_adj(adj_changes)

            if feature_attack:
                modified_feat = self.get_perturbed_feat(feat_changes)

            adj_norm = self.dense_gcn_norm(modified_adj)
            self.inner_train(adj_norm, modified_feat)

            adj_grad, feat_grad = self.compute_gradients(
                adj_norm, modified_feat)

            adj_grad_score = modified_adj.new_zeros(1)
            feat_grad_score = modified_feat.new_zeros(1)

            with torch.no_grad():
                if structure_attack:
                    adj_grad_score = self.structure_score(
                        modified_adj, adj_grad)

                if feature_attack:
                    feat_grad_score = self.feature_score(
                        modified_feat, feat_grad)

                adj_max, adj_argmax = torch.max(adj_grad_score, dim=0)
                feat_max, feat_argmax = torch.max(feat_grad_score, dim=0)

                if adj_max >= feat_max:
                    u, v = divmod(adj_argmax.item(), num_nodes)
                    edge_weight = modified_adj[u, v].data.item()
                    adj_changes[u, v].data.fill_(1 - 2 * edge_weight)
                    adj_changes[v, u].data.fill_(1 - 2 * edge_weight)

                    if edge_weight > 0:
                        self.remove_edge(u, v, it)
                    else:
                        self.add_edge(u, v, it)
                else:
                    u, v = divmod(feat_argmax.item(), num_feats)
                    feat_weight = modified_feat[u, v].data.item()
                    feat_changes[u, v].data.fill_(1 - 2 * feat_weight)
                    if feat_weight > 0:
                        self.remove_feat(u, v, it)
                    else:
                        self.add_feat(u, v, it)

        return self

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

        if self.lambda_ == 1:
            loss = F.cross_entropy(logit[self.labeled_nodes], self.y_train)
        elif self.lambda_ == 0.:
            loss = F.cross_entropy(logit[self.unlabeled_nodes],
                                   self.y_self_train)
        else:
            loss_labeled = F.cross_entropy(logit[self.labeled_nodes],
                                           self.y_train)
            loss_unlabeled = F.cross_entropy(logit[self.unlabeled_nodes],
                                             self.y_self_train)
            loss = self.lambda_ * loss_labeled + \
                (1 - self.lambda_) * loss_unlabeled

        if self.structure_attack and self.feature_attack:
            return grad(loss, [self.adj_changes, self.feat_changes])

        if self.structure_attack:
            return grad(loss, self.adj_changes)[0], None

        if self.feature_attack:
            return None, grad(loss, self.feat_changes)[0]

def handle_new_edges(data, attacker, device):
    added, removed = list(attacker._added_edges.keys()), list(attacker._removed_edges.keys())
    new_data = copy.deepcopy(data)
    for u, v in added:
        edge1 = torch.tensor([[u], [v]]).to(device)
        edge2 = torch.tensor([[v], [u]]).to(device)
        new_data.edge_index = torch.cat([new_data.edge_index, edge1], dim=1)
        new_data.edge_index = torch.cat([new_data.edge_index, edge2], dim=1)

    print(new_data.edge_index.shape)

    for u, v in removed:
        edge_to_delete = torch.tensor([[u, v], [v, u]]).to(device)
        mask = ~((new_data.edge_index == edge_to_delete[:, 0:1]).all(dim=0) | 
         (new_data.edge_index == edge_to_delete[:, 1:2]).all(dim=0))
        new_data.edge_index = new_data.edge_index[:, mask]

    return new_data
