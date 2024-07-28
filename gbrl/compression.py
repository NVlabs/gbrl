##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
import torch as th
import torch.nn as nn 
from torch.distributions import Categorical, Normal
import numpy as np

from typing import Tuple, Union, Type, Optional, Any, Dict

COMPRESSION_METHODS = ['first_k', 'best_k']
DIST_TYPES = ['categorical', 'gaussian', 'supervised_learning', 'deterministic']

def categorical_dist(logits: th.Tensor) -> th.Tensor:
    return Categorical(logits=logits)

def gaussian_dist(mu: th.Tensor, log_std: th.Tensor) -> th.Tensor:
    action_std = th.ones_like(mu) * log_std.exp()
    return Normal(mu, action_std)

class BinarizeSTE(th.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def construct_compression_matrix(tree_selection: th.Tensor, n_leaves_per_tree: th.Tensor) -> th.Tensor:
    """Construct compression matrix according to the tree indices specified to retain 

    Args:
        tree_selection (th.Tensor): indices of trees to retain/discard
        n_leaves_per_tree (th.Tensor): numbe of leaves per tree

    Returns:
        th.Tensor: Compressin matrix C of size LxL', 
        where L is the current number of leaves and L' is the compressed number of leaves.
    """
    selection_mask = th.repeat_interleave(tree_selection, n_leaves_per_tree)
    L, L_prime = len(selection_mask), int(selection_mask.sum())
    # Compute the cumulative sum of the selection_mask
    cumsum_m = selection_mask.cumsum(selection_mask, dim=0) - 1
    # Mask out zero elements
    k = selection_mask * cumsum_m
    # Create an index matrix for columns
    indices = k.unsqueeze(1)   # Convert to zero-based index
    # Construct the binary matrix C
    C = th.zeros((L, L_prime), dtype=th.float32)
    C.scatter_(1, indices.long(), selection_mask.unsqueeze(1)) 
    return C

def get_least_squares_W(C: th.Tensor, A: th.Tensor, V: th.Tensor, epsilon: float = 1e-5):
    CCT = C @ C.T 
    ATA = A.T @ A
    inv_mat = (CCT @ ATA @ CCT)
    inv_mat += epsilon * th.eye(inv_mat.size[0]) 
    I  = th.eye(CCT.size[0])
    return th.inverse(inv_mat) @ CCT @ ATA (I - CCT) @ V 



class TreeCompression:
    def __init__(self, k: int, gradient_steps: int, n_trees: int, n_leaves_per_tree: Union[np.array, th.Tensor], n_leaves: int, output_dim: int, method: str, optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None, least_squares_W: bool = True, temperature: float = 1.0):
        assert method in COMPRESSION_METHODS, f"Compression method: {method} is not supported! Supported compression methods are: {COMPRESSION_METHODS}"
        if isinstance(n_leaves_per_tree, np.array):
            n_leaves_per_tree = th.tensor(n_leaves_per_tree)
        self.compression = BestK(k, n_trees, n_leaves_per_tree, n_leaves, output_dim, least_squares_W) if method == 'first_k' else BestK(k, n_trees, n_leaves_per_tree, n_leaves, output_dim, least_squares_W, temperature)
        self.optimizer = optimizer_class(self.compression.parameters(), **optimizer_kwargs)
        self.gradient_steps = gradient_steps

    
    def compress(self, A: th.Tensor, V: th.Tensor) -> Tuple[np.array, np.array]:
        targets = A @ V 
        for i in range(self.gradient_steps):
            predictions = self.compression(A, V)
            loss = nn.functional.mse_loss(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optmizer.step()
            print(f"{i + 1}/{self.gradient_steps} - compression loss: {loss.item()}")
        return self.compression.get_parameters()
        

class SharedActorCriticCompression(TreeCompression):
    def __init__(self, k: int, gradient_steps: int, dist_type: str, policy_dim: int, n_trees: int, n_leaves_per_tree: Union[np.array, th.Tensor], n_leaves: int, output_dim: int, method: str, optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None, temperature: float = 1.0, vf_coef: float = 0.5):
        assert dist_type in DIST_TYPES, f"Distribution type: {dist_type} is not supported! Supported distributions are: {DIST_TYPES}"
        super(SharedActorCriticCompression, self).__init__(k, gradient_steps, n_trees, n_leaves_per_tree, n_leaves, output_dim, method, optimizer_class, optimizer_kwargs, False, temperature)
        self.policy_dim = policy_dim 
        self.dist_type = dist_type 
        self.vf_coef = vf_coef
    
    def compress(self, A: th.Tensor, V: th.Tensor, actions: th.Tensor, log_std: th.Tensor = None) ->  Tuple[np.array, np.array]:
        assert log_std is not None or self.dist_Type != 'gaussian', "Cannot compress using a Gaussian distribution without log std values!"
        targets = A @ V 
        critic_targets = targets[:, -1]
        for i in range(self.gradient_steps):
            predictions = self.compression(A, V)
            compressed_theta = predictions[:, :self.policy_dim]
            compressed_critic = predictions[:, -1]
            critic_loss = nn.functional.mse_loss(compressed_critic, critic_targets)
            dist = categorical_dist(compressed_theta) if self.dist_type == 'categorical' else gaussian_dist(compressed_theta, log_std)
            log_prob = dist.log_prob(actions)
            loss = -log_prob.mean() + self.vf_coef * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optmizer.step()
            print(f"{i + 1}/{self.gradient_steps} - compression loss: {loss.item()}")
        return self.compression.get_parameters()


class ParametricActorCompression(TreeCompression):
    def __init__(self, k: int, gradient_steps: int, dist_type: str, policy_dim: int, n_trees: int, n_leaves_per_tree: Union[np.array, th.Tensor], n_leaves: int, output_dim: int, method: str, optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None, temperature: float = 1.0):
        assert dist_type in DIST_TYPES, f"Distribution type: {dist_type} is not supported! Supported distributions are: {DIST_TYPES}"
        super(ParametricActorCompression, self).__init__(k, gradient_steps, n_trees, n_leaves_per_tree, n_leaves, output_dim, method, optimizer_class, optimizer_kwargs, dist_type in ['deterministic', 'supervised_learning'] , temperature)
        self.policy_dim = policy_dim 
        self.dist_type = dist_type 
    
    def compress(self, A: th.Tensor, V: th.Tensor, actions: th.Tensor, log_std: th.Tensor = None) ->  Tuple[np.array, np.array]:
        assert log_std is not None or self.dist_Type != 'gaussian', "Cannot compress using a Gaussian distribution without log std values!"
        targets = A @ V 
        for i in range(self.gradient_steps):
            compressed_theta = self.compression(A, V)
            if self.dist_type == 'deterministic':
                loss = nn.functional.mse_loss(compressed_theta, targets)
            else:
                dist = categorical_dist(compressed_theta) if self.dist_type == 'categorical' else gaussian_dist(compressed_theta, log_std)
                log_prob = dist.log_prob(actions)
                loss = -log_prob.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optmizer.step()
            print(f"{i + 1}/{self.gradient_steps} - compression loss: {loss.item()}")
        return self.compression.get_parameters()


class CompressionMethod(nn.Module):
    def __init__(self, k: int, n_trees: int, n_leaves_per_tree: th.Tensor, 
                n_leaves: int, least_squares_W: bool):
        self.k = k
        self.n_trees = n_trees
        self.n_leaves_per_tree = n_leaves_per_tree
        self.n_leaves = n_leaves 
        self.least_squares_W = least_squares_W
        self.W = None 

    def forward(self, A: th.Tensor, V: th.Tensor) -> th.Tensor:
        tree_selection = self.get_tree_selection()
        C = construct_compression_matrix(tree_selection, self.n_leaves_per_tree)
        if self.least_squares_W: 
            self.W = get_least_squares_W(C, A , V)
        return A @ C @ C.t() @ (V + self.W)

    def get_tree_selection(self) -> th.Tensor:
        raise NotImplementedError

    def get_parameters(self) -> Tuple[np.array, np.array]:
        tree_selection = self.get_tree_selection()
        n_compressed_trees = int(np.sum(tree_selection))
        selection_mask = th.repeat_interleave(tree_selection, self.n_leaves_per_tree)
        n_compressed_leaves = int(np.sum(n_compressed_leaves))
        return selection_mask.cpu().numpy().astype(int), tree_selection.cpu().numpy().astype(int), self.W.clone().detach().cpu().numpy().astype(np.single), n_compressed_trees, n_compressed_leaves
    
        
class FirstK(CompressionMethod):
    def __init__(self, k: int, n_trees: int, n_leaves_per_tree: th.Tensor, 
                 n_leaves: int, output_dim: int, least_squares_W: bool):
        super(FirstK, self).__init__(k, n_trees, n_leaves_per_tree, n_leaves, least_squares_W)
        if not least_squares_W:
            self.W = nn.Parameter(th.randn(n_leaves, output_dim), dtype=th.float32, requires_grad=True) 
    
    def get_tree_selection(self) -> th.Tensor:
        tree_selection = th.zeros(self.n_trees, dtype=th.float32)
        tree_selection[:self.k] = 1.0
        return tree_selection
    

class BestK(CompressionMethod):
    def __init__(self, k: int, n_trees: int, n_leaves_per_tree: th.Tensor, 
                 n_leaves: int, output_dim: int, least_squares_W: bool, temperature: float = 1.0):
        super(BestK, self).__init__(k, n_trees, n_leaves_per_tree, n_leaves, least_squares_W)
        if not least_squares_W:
            self.W = nn.Parameter(th.randn(n_leaves, output_dim), dtype=th.float32, requires_grad=True)
        self.logits = nn.Parameter(th.zeros(n_trees), dtype=th.float32, requires_grad=True)
        self.temperature = temperature
    
    def get_tree_selection(self) -> th.Tensor:
        probs = th.sigmoid(self.logits / self.temperature)
        # Binarize using straight through estimator
        tree_selection = BinarizeSTE.apply(probs)
        return tree_selection
    


