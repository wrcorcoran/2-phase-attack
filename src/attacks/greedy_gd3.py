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

import sys
import os

# Ensure attack_utils is importable, assuming it contains likelihood_ratio_filter
# If attack_utils is in the same directory, this should work.
# Otherwise, adjust the path as needed.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import attack_utils as util
except ImportError:
    print("Warning: attack_utils not found. Likelihood ratio constraint may fail.")
    # Define a dummy function if not found, or handle appropriately
    class DummyUtil:
        def likelihood_ratio_filter(*args, **kwargs):
            print("Warning: Using dummy likelihood_ratio_filter.")
            # Return a mask that allows all edges and a dummy ratio
            num_nodes = args[1].shape[0] # Assuming modified_adj is the second arg
            allowed_mask = torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=args[1].device)
            # Set diagonal to False and make symmetric if needed for consistency
            allowed_mask.fill_diagonal_(False)
            allowed_mask = torch.triu(allowed_mask, diagonal=1)
            return allowed_mask, 0.0
    util = DummyUtil()


# torch.use_deterministic_algorithms(True) # Optional: uncomment for determinism

class Metattack(torch.nn.Module):
    r"""Implementation of `Metattack` attack from the:
    `"Adversarial Attacks on Graph Neural Networks
    via Meta Learning"
    <https://arxiv.org/abs/1902.08412>`_ paper (ICLR'19)

    Adapted implementation.
    """

    _allow_feature_attack: bool = True # Keep feature attack capability

    def __init__(self, data, device):
        super().__init__()
        self.data = data.to(device) # Move data to device early
        self.device = device
        # Store original data separately if needed, but self.data is already on device
        # self.ori_data = data.to(self.device) # Redundant?

        # Use data directly from self.data which is on the correct device
        self.num_nodes = self.data.num_nodes
        self.num_edges = self.data.num_edges
        self.num_feats = self.data.x.size(1)

        # Original adjacency matrix (dense) and features on the device
        self.adj = self.to_dense_adj(self.data.edge_index,
                                     self.data.edge_weight,
                                     self.num_nodes).to(self.device)
        self.feat = self.data.x.clone().detach().to(self.device) # Ensure it's a modifiable copy
        self.label = self.data.y.to(self.device)

        # Keep track of degrees (original)
        self._degree = degree(self.data.edge_index[0], num_nodes=self.num_nodes,
                              dtype=torch.float).to(self.device)
        self.degree = self._degree.clone() # Modifiable degree

        self.surrogate = None
        self.num_budgets = None
        self.structure_attack = True # Defaulting to structure attack
        self.feature_attack = False # Defaulting to no feature attack

        # Perturbation parameters (initialized in reset)
        self.adj_changes = None
        self.feat_changes = None
        self.adj_changes_logical = None # For tracking flips better

        self.reset() # Initialize changes

    def setup_surrogate(self, surrogate: torch.nn.Module,
                        labeled_nodes: Tensor, unlabeled_nodes: Tensor,
                        lr: float = 0.1, epochs: int = 100,
                        momentum: float = 0.9, lambda_: float = 0.5, *, # Default lambda_ to 0.5 like BaseMeta
                        tau: float = 1.0):

        # Ensure surrogate model is properly handled (eval mode, cache)
        surrogate.eval()
        if hasattr(surrogate, 'cache_clear'):
            surrogate.cache_clear()
        for layer in surrogate.modules():
            if hasattr(layer, 'cached'):
                layer.cached = False # Disable caching if applicable

        self.surrogate = surrogate.to(self.device)
        self.tau = tau # Temperature parameter for logits

        # Ensure node indices are on the correct device
        if labeled_nodes.dtype == torch.bool:
            labeled_nodes = labeled_nodes.nonzero(as_tuple=False).view(-1)
        self.labeled_nodes = labeled_nodes.to(self.device)

        if unlabeled_nodes.dtype == torch.bool:
            unlabeled_nodes = unlabeled_nodes.nonzero(as_tuple=False).view(-1)
        self.unlabeled_nodes = unlabeled_nodes.to(self.device)

        # Get labels for training and self-training
        self.y_train = self.label[self.labeled_nodes]
        # Estimate self-training labels *before* attack starts
        with torch.no_grad():
             self.y_self_train = self.estimate_self_training_labels(self.unlabeled_nodes)

        # Setup for inner training loop parameters (weights, velocities)
        # This mimics the structure in Metattack more closely
        weights = []
        w_velocities = []
        # Assuming surrogate parameters can be accessed directly
        # This part needs to align with how your surrogate model stores parameters
        # If surrogate is a simple GCN, access layers/weights like below:
        num_layers = 0
        if hasattr(self.surrogate, 'layers'): # Example: if layers are stored in a list
            param_source = self.surrogate.layers
        elif hasattr(self.surrogate, 'convs'): # Example: PyG GCN style
             param_source = self.surrogate.convs
        else: # Fallback: Iterate through all parameters (might include biases)
             param_source = self.surrogate.parameters()
             print("Warning: Accessing surrogate parameters directly. Ensure order is correct.")

        params_list = list(param_source)

        # We only need the *shapes* to create meta-parameters
        # We don't directly use the surrogate's trained weights here,
        # the inner loop re-trains meta-weights from scratch.
        for para in params_list:
            if para.ndim >= 2: # Typically weight matrices
                # Match dimensionality expectation (some models might not transpose)
                # The reference code transposes weights for its forward pass.
                # Assuming your forward pass expects (in_features, out_features)
                # No transpose needed here if your forward pass matches the weight shape.
                weights.append(torch.zeros_like(para.data, requires_grad=True))
                w_velocities.append(torch.zeros_like(para.data))
                num_layers += 1

        if not weights:
             raise ValueError("Could not extract weight parameters from the surrogate model.")

        print(f"Initialized {num_layers} meta-weight tensors.")
        self.weights = weights
        self.w_velocities = w_velocities

        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.lambda_ = lambda_ # Weighting factor for meta-loss

    def estimate_self_training_labels(self, nodes=None):
        # Use the surrogate model for predictions
        self.surrogate.eval() # Ensure model is in eval mode
        with torch.no_grad():
            # Use the original graph state for initial label estimation
            logits = self.surrogate(self.data.x, self.data.edge_index)
        if nodes is not None:
            logits = logits[nodes]
        return logits.argmax(-1)

    def reset(self):
        # Initialize changes as Parameter for gradient tracking if attacking structure/features
        # Using a dense tensor for adj_changes matches the reference
        self.adj_changes = torch.zeros_like(self.adj, device=self.device, dtype=torch.float)
        # Use Parameter if you intend to optimize adj_changes directly (not the Metattack approach)
        # self.adj_changes = torch.nn.Parameter(torch.zeros_like(self.adj, device=self.device))

        # Use a logical tensor to track actual flips made, separate from gradient targets
        self.adj_changes_logical = torch.zeros_like(self.adj, device=self.device, dtype=torch.bool)

        # Feature changes (if feature attack is enabled)
        self.feat_changes = torch.zeros_like(self.feat, device=self.device, dtype=torch.float)
        # self.feat_changes = torch.nn.Parameter(torch.zeros_like(self.feat, device=self.device))

        self._removed_edges = {}
        self._added_edges = {}
        # Feature modification tracking (if needed)
        # self._removed_feats = {}
        # self._added_feats = {}
        self.degree = self._degree.clone() # Reset degree based on original graph
        return self

    def get_perturbed_adj(self):
        """ Get the perturbed adjacency matrix based on logical flips. """
        # This should reflect the actual flips made so far
        # Add self-loops for GCN normalization later if needed, not here
        modified_adj = self.adj.clone()
        # Apply changes based on adj_changes_logical
        # Symmetrically apply the changes stored in adj_changes_logical
        rows, cols = self.adj_changes_logical.nonzero(as_tuple=True)
        # Determine if the change corresponds to an addition or removal based on original adj
        original_values = self.adj[rows, cols]
        # If original was 0, it's an addition (set to 1). If original was 1, it's a removal (set to 0).
        modified_adj[rows, cols] = 1.0 - original_values

        # We don't clamp here. Clamping happens *implicitly* by flipping 0s to 1s and 1s to 0s.
        # Ensure symmetry explicitly if needed, although flipping should maintain it if done carefully.
        # modified_adj = torch.max(modified_adj, modified_adj.t()) # Ensure symmetry via max

        return modified_adj


    def get_perturbed_feat(self):
        # Similar logic for features if feature attack is enabled
        if not self.feature_attack:
            return self.feat
        # Assuming binary features and flipping logic
        modified_feat = self.feat.clone()
        # Apply feat_changes logic here... needs tracking similar to adj_changes_logical
        # Example: feat_changes_logical = torch.zeros_like(self.feat, dtype=torch.bool)
        # rows, cols = self.feat_changes_logical.nonzero(as_tuple=True)
        # modified_feat[rows, cols] = 1.0 - self.feat[rows, cols]
        # For now, return original if feature attack logic isn't fully implemented yet.
        print("Warning: Feature attack perturbation logic not fully implemented in get_perturbed_feat.")
        return self.feat + self.feat_changes # Basic addition, might need clipping/binary logic


    def reset_parameters(self, seed=None):
        # Re-initialize meta-weights and velocities for the inner loop
        if seed is not None:
             torch.manual_seed(seed) # For reproducibility if needed

        for w, wv in zip(self.weights, self.w_velocities):
            # Use init method consistent with surrogate if known, otherwise Xavier is common
            if hasattr(self.surrogate, 'reset_parameters'):
                 # This is tricky - we need shapes, not the reset function itself
                 # Stick to a standard init like Xavier for meta-weights
                 init.xavier_uniform_(w.data)
            else:
                 init.xavier_uniform_(w.data) # Default initialization
            wv.data.fill_(0.0) # Reset velocities

        # Detach and require gradients for the inner loop optimization
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].detach().clone().requires_grad_(True)
            self.w_velocities[i] = self.w_velocities[i].detach().clone() # No grad needed for velocity


    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes.
        Adapted from mettack.py's BaseMeta.
        """
        # Calculate degrees from the *current* modified adjacency matrix
        degrees = modified_adj.sum(dim=1)
        degree_one = (degrees == 1) # Nodes that *would* have degree 1 if an edge is removed

        # Check pairs where one node has degree 1 and there's an edge between them
        resh = degree_one.view(-1, 1).float() # Shape (N, 1)
        # Element-wise multiply with modified_adj: 1 if node i has degree 1 AND edge (i,j) exists
        l_and = resh * modified_adj
        # If undirected, check the other direction too
        l_and = l_and + l_and.t() # Symmetric check: 1 if edge (i,j) exists and either i or j has degree 1

        # Mask should be 0 for edges that would create a singleton, 1 otherwise
        # If l_and is 1, it means removing edge (i,j) *could* create a singleton.
        # We want to disallow flips for such edges if they currently exist (modified_adj=1).
        # The mask should be applied to the *scores* of potential flips.
        # A score for flipping edge (i,j) should be zeroed out if l_and[i,j] > 0.
        # So, the mask is (1 - l_and). We need to be careful if l_and can exceed 1.
        # Let's clamp it first.
        l_and = torch.clamp(l_and, 0, 1)
        flat_mask = 1.0 - l_and

        # Ensure mask doesn't prevent adding edges that fix a singleton?
        # The original purpose is to prevent removals that create singletons.
        # Let's stick to the original logic: mask is 1 (allowed) if the flip won't create a singleton.
        return flat_mask


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Wrapper for the likelihood ratio filter.
        Ensure attack_utils.likelihood_ratio_filter is available and matches expected signature.
        """
        # Determine potential edges (upper triangle for undirected)
        # This needs to match the expected input of util.likelihood_ratio_filter
        # Assuming it expects indices of potential edges (N_potential, 2)
        rows, cols = torch.triu_indices(self.num_nodes, self.num_nodes, offset=1)
        t_possible_edges = torch.stack([rows, cols], dim=1).to(self.device) # Shape (N_potential, 2)

        # Define minimum degree (typically 2.0)
        t_d_min = torch.tensor(2.0).to(self.device)

        try:
            # Call the utility function
            allowed_mask_dense, current_ratio = util.likelihood_ratio_filter(
                t_possible_edges, # Indices of edges to check
                modified_adj.detach(), # Current adjacency state
                ori_adj.detach(),      # Original adjacency state
                t_d_min,               # Minimum degree threshold
                ll_cutoff              # Likelihood ratio cutoff
            )
            # The utility function likely returns a dense mask (N, N) based on its typical usage
            # Ensure it's returned correctly. If it returns sparse/edge-based, adjust here.
            # Assuming allowed_mask_dense is (N, N), boolean or float
            return allowed_mask_dense, current_ratio
        except Exception as e:
            print(f"Error calling likelihood_ratio_filter: {e}")
            # Fallback: allow all edges if the utility fails
            allowed_mask_dense = torch.ones_like(modified_adj, dtype=torch.float).triu(diagonal=1)
            return allowed_mask_dense, 0.0 # Return 0 ratio as fallback

    # --- Meta-Weight GCN Forward Pass ---
    # This needs to match the structure of your meta-weights (self.weights)
    # Assumes self.weights contains layer weights sequentially.
    def surrogate_forward(self, adj_norm, x):
        h = x
        # Assumes weights are ordered correctly (layer1, layer2, ...)
        num_meta_layers = len(self.weights)
        for i, w in enumerate(self.weights):
            # Apply GCN layer logic: Adj @ Features @ Weight
            # Handle sparse features if necessary (not typical in PyG context usually)
            h = adj_norm @ (h @ w.T)
            # Apply activation (ReLU) except for the last layer
            if i < num_meta_layers - 1:
                h = F.relu(h) # Use ReLU like the reference GCN
        # No softmax needed here, cross_entropy loss expects logits
        return h

    def inner_train(self, adj_norm, feat):
        """ Performs the inner loop training of meta-weights. """
        self.reset_parameters() # Re-initialize meta-weights and velocities

        # Detach parameters that should not be trained in the inner loop
        adj_norm_detached = adj_norm.detach()
        feat_detached = feat.detach()
        y_train_detached = self.y_train.detach()

        # Inner training loop
        for _ in range(self.epochs):
            # --- Forward pass using meta-weights ---
            out = self.surrogate_forward(adj_norm_detached, feat_detached)

            # --- Loss Calculation (only on labeled nodes) ---
            # The inner loop *only* trains on labeled nodes loss
            loss = F.cross_entropy(out[self.labeled_nodes], y_train_detached)

            # --- Gradient Calculation ---
            # Gradients w.r.t. meta-weights
            grads = torch.autograd.grad(loss, self.weights, create_graph=True) # No graph needed for velocity update

            # --- Update Velocities and Weights (SGD with Momentum) ---
            with torch.no_grad(): # Updates should not be part of gradient graph
                # Update velocities
                self.w_velocities = [
                    self.momentum * v + g for v, g in zip(self.w_velocities, grads)
                ]
                # Update weights
                self.weights = [
                    w - self.lr * v for w, v in zip(self.weights, self.w_velocities)
                ]

        # After training, ensure weights require grad again for the outer loop grad calculation
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].detach().clone().requires_grad_(True)


    def dense_gcn_norm(self, adj, improved = False, add_self_loops = True):
        """ Normalizes a dense adjacency matrix. """
        adj_ = adj
        if add_self_loops:
             # Add self-loops only if they don't exist? Or always add?
             # Reference adds fill_value=1 or 2. Let's mimic adding 1.
             adj_ = adj + torch.eye(adj.size(0), device=self.device)

        # Degree calculation
        deg = torch.sum(adj_, dim=1)
        # Handle degrees of zero
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0 # Replace inf with 0

        # Create diagonal matrix D^-0.5
        diag_inv_sqrt = torch.diag(deg_inv_sqrt)

        # Calculate D^-0.5 * A * D^-0.5
        return diag_inv_sqrt @ adj_ @ diag_inv_sqrt


    def attack(self, num_budgets=0.05, *, structure_attack=True,
               feature_attack=False, disable=False, ll_cutoff=0.004):

        # Validate budget
        if isinstance(num_budgets, float) and 0 < num_budgets < 1:
            self.num_budgets = int((self.num_edges // 2) * num_budgets) # Use original edges for budget calc
        elif isinstance(num_budgets, int):
             self.num_budgets = num_budgets
        else:
            raise ValueError("num_budgets must be an int or a float between 0 and 1.")

        print(f"Starting attack with budget: {self.num_budgets} edge flips.")

        self.structure_attack = structure_attack
        self.feature_attack = feature_attack # Store attack type

        if feature_attack:
            raise NotImplementedError("Feature attack scoring and update logic needs review/implementation.")
            # self._check_feature_matrix_binary() # Add checks if needed

        # --- Main Attack Loop ---
        non_improving_streak = 0
        max_non_improving = 5 # Stop after 5 non-improving flips

        # Get the *initial* perturbed state (which is just the original graph at first)
        modified_adj = self.get_perturbed_adj() # Starts as original adj
        modified_feat = self.get_perturbed_feat() # Starts as original feat

        for it in tqdm(range(self.num_budgets), desc='Perturbing graph...', disable=disable):

            # --- 1. Inner Training Loop ---
            # Normalize the *current* adjacency matrix
            # Add self-loops during normalization
            adj_norm = self.dense_gcn_norm(modified_adj, add_self_loops=True)
            self.inner_train(adj_norm, modified_feat) # Trains self.weights

            # --- 2. Compute Meta-Gradients ---
            # Gradients w.r.t. the *input* adjacency (adj_changes) / features (feat_changes)
            # We need gradients of the meta-loss w.r.t the graph structure/features.
            # This requires the outer loss calculation and backprop.

            # Make the *current* adj_norm and features require grad for this step
            adj_norm.requires_grad_(True)
            # modified_feat.requires_grad_(bool(self.feature_attack)) # Only if attacking features

            # Calculate outer loss (meta-objective) using the trained meta-weights
            logits = self.surrogate_forward(adj_norm, modified_feat) / self.tau # Apply temperature

            # Calculate combined loss based on lambda_
            loss_labeled = F.cross_entropy(logits[self.labeled_nodes], self.y_train)
            loss_unlabeled = F.cross_entropy(logits[self.unlabeled_nodes], self.y_self_train)
            loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

            # Calculate gradients w.r.t. the *normalized adjacency matrix*
            # We need the gradient w.r.t. the *original* `adj_changes`, not `adj_norm`.
            # This requires relating `adj_norm` back to `adj_changes` or `modified_adj`.
            # The reference `mettack.py` computes grad w.r.t `self.adj_changes`.
            # Let's try computing grad w.r.t. `modified_adj` first, assuming `adj_norm` depends on it.

            # We need a variable representing the modifiable part of the graph
            # Let's define adj_changes placeholder that affects modified_adj
            # This is tricky because adj_changes isn't directly used in dense_gcn_norm
            # Alternative: Use the gradient w.r.t adj_norm and approximate or derive grad w.r.t changes.

            # --- Approximation based on Reference ---
            # The reference calculates gradient w.r.t. a Parameter `adj_changes`.
            # Let's mimic this by making `adj_changes` a Parameter temporarily for grad computation.
            # This is not ideal but might approximate the reference behaviour.

            adj_grad = None
            feat_grad = None # Placeholder for feature gradient

            # We need grad w.r.t potential flips (0->1 or 1->0)
            # The reference uses grad w.r.t. `self.adj_changes` Parameter.
            # Let's compute grad w.r.t `adj_norm` first.
            grad_outputs = torch.ones_like(loss)
            grads = torch.autograd.grad(outputs=loss,
                                        inputs=[adj_norm] + ([modified_feat] if self.feature_attack else []),
                                        grad_outputs=grad_outputs,
                                        only_inputs=True,
                                        retain_graph=False) # No need to retain graph after grads

            adj_norm_grad = grads[0]
            # Feature grad if applicable
            # feat_grad = grads[1] if self.feature_attack else None

            # --- Score Calculation (using gradient w.r.t. adj_norm) ---
            # This is where `greedy_gd` differs significantly from `mettack`.
            # `mettack` scores based on `adj_grad * (-2 * mod_adj + 1)`
            # We need the gradient w.r.t. the flips, not adj_norm directly.
            # Let's try using adj_norm_grad as an approximation for adj_grad.
            # This IS an approximation. A proper derivation might be needed.
            # Assuming adj_norm_grad ~ adj_grad for scoring purposes:
            adj_grad_approx = adj_norm_grad # *** Approximation Step ***

            adj_score = torch.zeros_like(self.adj)
            feat_score = torch.zeros_like(self.feat) # Placeholder

            if self.structure_attack:
                 # Use the scoring function similar to mettack.py
                 adj_score = self.structure_score(adj_grad_approx, modified_adj, self.adj, ll_cutoff)

            if self.feature_attack:
                 # feat_score = self.feature_score(feat_grad, modified_feat) # Requires feat_grad
                 pass # Skip feature scoring for now

            # --- Find Best Flip ---
            # Flatten scores and find max
            # We should only consider flipping edges where adj_score is defined (upper triangle?)
            # The score function already handles masking (self-loops, constraints)

            # Get the best candidate flip from the scores
            # Consider only upper triangle for undirected graphs to avoid duplicates
            adj_score_upper = torch.triu(adj_score, diagonal=1)
            max_score, flat_idx = torch.max(adj_score_upper.flatten(), dim=0)

            # Convert flat index back to 2D coordinates
            u, v = np.unravel_index(flat_idx.item(), self.adj.shape)

            # Check if the best score is positive (indicates potential improvement)
            if max_score.item() <= 1e-9: # Use a small threshold for floating point
                non_improving_streak += 1
                print(f"Iteration {it+1}: No improving edge flip found (max_score={max_score.item():.4f}). Streak: {non_improving_streak}")
                if non_improving_streak >= max_non_improving:
                    print(f"Stopping early: {max_non_improving} consecutive non-improving flips.")
                    break
            else:
                non_improving_streak = 0 # Reset streak

            # --- Apply Best Flip ---
            # Determine current state of the edge (u, v) in the *modified* graph
            current_edge_state = modified_adj[u, v].item()

            # Update the logical changes tensor
            # If the flip is applied, toggle the state in adj_changes_logical
            # This assumes the score correctly identifies the best *single* flip for this iteration
            self.adj_changes_logical[u, v] = ~self.adj_changes_logical[u, v]
            self.adj_changes_logical[v, u] = ~self.adj_changes_logical[v, u] # Symmetric change

            # Update degrees and edge tracking dictionaries
            if current_edge_state > 0.5: # Edge existed (value was ~1), now removed
                print(f"Iteration {it+1}: Removing edge ({u}, {v}) - Score: {max_score.item():.4f}")
                self.remove_edge(u, v, it)
            else: # Edge didn't exist (value was ~0), now added
                 print(f"Iteration {it+1}: Adding edge ({u}, {v}) - Score: {max_score.item():.4f}")
                 self.add_edge(u, v, it)

            # Update the modified adjacency matrix for the next iteration
            # We can derive it directly from adj_changes_logical and the original adj
            modified_adj = self.get_perturbed_adj() # Recompute based on updated logical changes

            # --- Feature Flip Logic (if enabled) ---
            # if self.feature_attack and feat_max >= adj_max ...
            #    # Find best feature flip and apply it
            #    ...
            #    modified_feat = self.get_perturbed_feat()

            # Detach variables no longer needed for gradient computation
            adj_norm.requires_grad_(False)
            del adj_grad_approx, adj_score, adj_score_upper, loss, logits # Free memory


        print(f"Attack finished after {it+1} iterations.")
        # Store final modified state if needed (optional)
        # self.final_modified_adj = modified_adj.detach()
        return self

    def structure_score(self, adj_grad, modified_adj, ori_adj, ll_cutoff):
        """ Calculate scores for structure perturbations using gradient. """
        # Score = Gradient * Direction_Factor
        # Direction_Factor is (1 - 2 * current_state)
        # If current_state=1 (edge exists), factor is -1 (score = -gradient => higher score if grad is negative)
        # If current_state=0 (no edge), factor is +1 (score = +gradient => higher score if grad is positive)
        score = adj_grad * (1.0 - 2.0 * modified_adj)

        # Normalize score (optional, helps prevent scale issues)
        score = score - score.min() # Make minimum score 0

        # --- Apply Constraints ---
        # 1. Filter self-loops (diagonal)
        score.fill_diagonal_(0.0)

        # 2. Filter potential singletons
        singleton_mask = self.filter_potential_singletons(modified_adj)
        score = score * singleton_mask

        # 3. Likelihood Ratio Constraint (if enabled/available)
        # Use try-except block for safety if util is optional
        try:
             # Get the dense mask (N, N) from the LL constraint function
             ll_mask_dense, ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
             # Ensure the mask is float and on the correct device
             ll_mask_float = ll_mask_dense.float().to(self.device)

             # Apply the mask (element-wise multiplication)
             # Ensure mask aligns with score (upper triangle if score is upper triangle)
             # If score is full matrix, apply full mask
             # Make the mask symmetric if the graph is undirected
             ll_mask_symmetric = torch.max(ll_mask_float, ll_mask_float.t())
             score = score * ll_mask_symmetric

             # print(f"LL constraint applied. Ratio: {ll_ratio:.4f}, Masked edges: {(1-ll_mask_symmetric.triu(1)).sum()}")
        except AttributeError:
            print("Skipping Likelihood Ratio constraint (attack_utils likely not found).")
        except Exception as e:
            print(f"Error during Likelihood Ratio constraint: {e}. Skipping.")

        # Consider only the upper triangle for undirected graphs if selecting best edge later
        # score = torch.triu(score, diagonal=1)
        return score


    # --- Utility Functions (Keep as is) ---
    def get_dense_adj(self):
        # Already computed in __init__ and stored in self.adj
        # This function might be redundant now.
        return self.adj

    def to_dense_adj(self, edge_index, edge_weight, num_nodes, fill_value = 1.0):
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        if edge_weight is None:
            adj[edge_index[0], edge_index[1]] = fill_value
        else:
            # Handle weighted graphs if necessary, Metattack usually assumes unweighted
            adj[edge_index[0], edge_index[1]] = edge_weight # Use weights if provided
        return adj

    def add_edge(self, u, v, it = None):
        key = tuple(sorted((u, v))) # Use sorted tuple for undirected uniqueness
        self._added_edges[key] = it
        # Update degree (only if edge didn't exist before - implicitly handled by calc from modified_adj)
        # self.degree[u] += 1
        # self.degree[v] += 1

    def remove_edge(self, u, v, it = None):
        key = tuple(sorted((u, v)))
        self._removed_edges[key] = it
        # Update degree
        # self.degree[u] -= 1
        # self.degree[v] -= 1

    def dense_add_self_loops(self, adj, fill_value = 1.0):
        # This seems incorrect for standard GCN self-loops.
        # Should add identity matrix, not fill diagonal.
        # return adj + torch.diag(adj.new_full((adj.size(0), ), fill_value))
        return adj + torch.eye(adj.size(0), device=adj.device) * fill_value

    def feature_score(self, modified_feat, feat_grad):
        # Placeholder - needs implementation if feature attack is used
        score = feat_grad * (1 - 2 * modified_feat)
        score -= score.min()
        return score.view(-1)


# --- Helper function to get final graph state ---
# Keep your handle_new_edges function, it seems reasonable.
def handle_new_edges(data, attacker, device):
    """ Creates a new Data object with the perturbed adjacency matrix. """
    # Get the final adjacency matrix based on logical flips
    final_adj = attacker.get_perturbed_adj().detach().to(device) # Ensure it's on the right device

    # Convert the dense adjacency matrix back to edge_index format
    edge_index, edge_weight = dense_to_sparse(final_adj)

    new_data = copy.deepcopy(data) # Start with original data structure
    new_data = new_data.to(device) # Ensure new data object is on device

    new_data.edge_index = edge_index
    # Metattack usually results in unweighted {0,1} adj, set weights to None or 1s
    # If final_adj can have values other than 0/1, keep edge_weight
    # For simplicity assume unweighted flips:
    new_data.edge_weight = None # Or torch.ones(edge_index.size(1), device=device)

    num_added = len(attacker._added_edges)
    num_removed = len(attacker._removed_edges)
    print(f"Created new data object. Added={num_added}, Removed={num_removed}")
    print(f"Original edge index shape: {data.edge_index.shape}")
    print(f"Updated edge index shape: {new_data.edge_index.shape}")

    return new_data