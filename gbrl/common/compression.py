##############################################################################
# Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
##############################################################################
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Categorical, Normal

COMPRESSION_METHODS = ['first_k', 'best_k']
DIST_TYPES = ['categorical', 'gaussian', 'supervised_learning', 'deterministic']


def categorical_dist(logits: th.Tensor) -> Categorical:
    """
    Create a categorical distribution from logits.
    
    Args:
        logits (th.Tensor): Unnormalized log probabilities for categorical distribution.
    
    Returns:
        Categorical: PyTorch categorical distribution object.
    """
    return Categorical(logits=logits)


def gaussian_dist(mu: th.Tensor, log_std: th.Tensor) -> Normal:
    """
    Create a Gaussian (normal) distribution from mean and log standard deviation.
    
    Args:
        mu (th.Tensor): Mean values for the normal distribution.
        log_std (th.Tensor): Log of standard deviation (will be exponentiated).
    
    Returns:
        Normal: PyTorch normal distribution object.
    """
    # Broadcasting handles shape matching automatically, no need for ones_like
    return Normal(mu, log_std.exp())


class BinarizeSTE(th.autograd.Function):
    """
    Binarize with Straight-Through Estimator (STE).
    
    A custom autograd function that binarizes values in the forward pass (converting to 0 or 1)
    while allowing gradients to flow through in the backward pass using a clipped identity function.
    This enables gradient-based optimization of discrete/binary parameters.
    
    The forward pass applies a hard threshold, while the backward pass uses hardtanh to clip
    gradients to the range [-1, 1], allowing gradient flow for values near the threshold.
    """
    
    @staticmethod
    def forward(ctx, input: th.Tensor) -> th.Tensor:
        """
        Binarize input tensor using threshold at 0.0.
        
        Args:
            ctx: PyTorch autograd context for saving tensors for backward pass (unused here).
            input (th.Tensor): Input tensor with continuous values to binarize.
        
        Returns:
            th.Tensor: Binarized tensor where values > 0.0 become 1.0, otherwise 0.0.
        """
        return (input > 0.0).float()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> th.Tensor:
        """
        Compute gradient using straight-through estimator with clipping.
        
        Passes gradients through while clipping them to [-1, 1] range using hardtanh.
        This allows gradient-based optimization despite the non-differentiable forward pass.
        
        Args:
            ctx: PyTorch autograd context (unused here).
            *grad_outputs: Tuple containing gradient tensor flowing back from the next layer.
        
        Returns:
            th.Tensor: Clipped gradient tensor, bounded to [-1, 1] range.
        """
        return th.nn.functional.hardtanh(grad_outputs[0], min_val=-1.0, max_val=1.0)


def construct_compression_matrix(tree_selection: th.Tensor, n_leaves_per_tree: th.Tensor) -> th.Tensor:
    """
    Construct compression matrix according to the tree indices specified to retain.
    
    Creates a binary selection matrix that maps from the full leaf space to the compressed
    leaf space based on which trees are selected for retention.

    Args:
        tree_selection (th.Tensor): Binary mask indicating which trees to retain (1) or discard (0).
        n_leaves_per_tree (th.Tensor): Number of leaves per tree.

    Returns:
        th.Tensor: Compression matrix C of size (L, L'), where L is the current number of
            leaves and L' is the compressed number of leaves after selection.
    """
    selection_mask = th.repeat_interleave(tree_selection, n_leaves_per_tree)
    # add 1 for the bias - create directly on correct device to avoid device transfers
    bias_tensor = th.ones(1, dtype=selection_mask.dtype, device=selection_mask.device)
    selection_mask = th.cat((bias_tensor, selection_mask))
    L, L_prime = len(selection_mask), int(selection_mask.sum())
    # Compute the cumulative sum of the selection_mask
    cumsum_m = selection_mask.cumsum(dim=0) - 1
    # Mask out zero elements
    k = selection_mask * cumsum_m
    # Create an index matrix for columns
    indices = k.unsqueeze(1)   # Convert to zero-based index
    # Construct the binary matrix C
    C = th.zeros((L, L_prime), dtype=th.float32, device=n_leaves_per_tree.device)
    C.scatter_(1, indices.long(), selection_mask.unsqueeze(1))
    del selection_mask
    return C


def get_least_squares_W(C: th.Tensor, A: th.Tensor, V: th.Tensor, lambda_reg: float) -> th.Tensor:
    """
    Compute the least squares correction matrix W using regularized optimization.
    
    Solves for W in the optimization problem to minimize reconstruction error when
    compressing the model, using Tikhonov regularization to ensure numerical stability.
    
    Args:
        C (th.Tensor): Compression matrix of size (L, L').
        A (th.Tensor): Input activation/feature matrix.
        V (th.Tensor): Value/weight matrix to be compressed.
        lambda_reg (float): Regularization parameter for numerical stability.
    
    Returns:
        th.Tensor: Correction matrix W that minimizes reconstruction error.
    """
    CCT = C @ C.T
    ATA = A.T @ A
    CCTATA = CCT @ ATA
    del ATA
    # Add regularization and compute pseudo-inverse in one step
    inv_mat = CCTATA @ CCT
    inv_mat.add_(th.eye(inv_mat.size(0), device=A.device, dtype=inv_mat.dtype), alpha=lambda_reg)
    inv_mat = th.linalg.pinv(inv_mat)
    # Compute result with fewer intermediate allocations
    res = inv_mat @ CCTATA
    del inv_mat, CCTATA
    eye = th.eye(CCT.size(0), device=A.device, dtype=CCT.dtype)
    res = res @ (eye - CCT)
    del CCT, eye
    return res @ V


class TreeCompression:
    """
    Base class for compressing gradient boosted tree ensembles.
    
    Reduces the number of trees in an ensemble while maintaining prediction accuracy
    through gradient-based optimization. Supports both 'first_k' (remove first k trees)
    and 'best_k' (learn which k trees to remove) compression methods.
    
    Attributes:
        compression (CompressionMethod): The compression method implementation (FirstK or BestK).
        optimizer (th.optim.Optimizer): Optimizer for learnable parameters.
        device (str): Device for computation ('cpu' or 'cuda').
        method (str): Compression method name ('first_k' or 'best_k').
        use_W (bool): Whether to use learnable correction matrix W.
        gradient_steps (int): Number of optimization steps for compression.
    """
    
    def __init__(self, k: int, gradient_steps: int, n_trees: int, n_leaves_per_tree: Union[np.ndarray, th.Tensor],
                 n_leaves: int, output_dim: int, method: str, optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None, least_squares_W: bool = False,
                 temperature: float = 1.0, lambda_reg: float = 1.0, use_W: bool = True, device: str = 'cpu',
                 actor_critic: bool = False, **kwargs):

        assert method in COMPRESSION_METHODS, \
            f"Compression method: {method} is not supported! Supported compression methods are: {COMPRESSION_METHODS}"
        if isinstance(n_leaves_per_tree, np.ndarray):
            n_leaves_per_tree = th.tensor(n_leaves_per_tree, device=device, dtype=th.int64)
        
        # Select compression method
        if method == 'first_k':
            self.compression = FirstK(k, n_trees, n_leaves_per_tree, n_leaves, output_dim, 
                                     least_squares_W, lambda_reg, use_W, device, actor_critic)
        else:  # best_k
            self.compression = BestK(k, n_trees, n_leaves_per_tree, n_leaves, output_dim,
                                    least_squares_W, temperature, lambda_reg, use_W, device, actor_critic)
        self.optimizer = None
        self.device = device
        self.method = method
        self.use_W = use_W
        if list(self.compression.parameters()):
            self.optimizer = optimizer_class(self.compression.parameters(), **optimizer_kwargs)
        self.gradient_steps = gradient_steps

    def compress(self, A: th.Tensor, V: th.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress the tree ensemble by optimizing tree selection and correction matrix.
        
        Performs gradient descent to minimize reconstruction error between original
        and compressed model predictions.
        
        Args:
            A (th.Tensor): Input activation/feature matrix of shape (n_samples, n_leaves+1).
            V (th.Tensor): Value/weight matrix of shape (n_leaves+1, output_dim).
        
        Returns:
            Tuple containing:
                - compression parameters (selection masks, correction matrix, etc.)
                - list of losses at each gradient step
        """
        targets = A @ V
        losses = []
        if self.method == 'first_k' and not self.use_W:
            return self.compression.get_parameters(A, V), [0]
        if self.optimizer is not None:
            for i in range(self.gradient_steps):
                predictions = self.compression(A, V)
                loss = nn.functional.mse_loss(predictions, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                print(f"{i + 1}/{self.gradient_steps} - compression loss: {loss.item()}")
        else:
            with th.no_grad():
                predictions = self.compression(A, V)
                loss = nn.functional.mse_loss(predictions, targets)
                del predictions
            print(f"Compression loss: {loss.item()}")
            losses.append(loss.item())
        compression_params = self.compression.get_parameters(A, V)
        return compression_params, losses


class SharedActorCriticCompression(TreeCompression):
    """
    Tree compression specialized for shared actor-critic reinforcement learning models.
    
    Compresses tree ensembles used in actor-critic architectures where the tree outputs
    both policy (actor) and value (critic) predictions. The compression optimizes for
    both action distribution accuracy and value function approximation.
    
    Attributes:
        dist_type (str): Type of action distribution ('categorical', 'gaussian', etc.).
        vf_coef (float): Value function loss coefficient (currently unused in loss).
    """
    
    def __init__(self, k: int, gradient_steps: int, dist_type: str, n_trees: int,
                 n_leaves_per_tree: Union[np.ndarray, th.Tensor], n_leaves: int, output_dim: int, method: str,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None, temperature: float = 1.0, vf_coef: float = 0.5,
                 lambda_reg: float = 1.0, use_W: bool = True, device: str = 'cpu', **kwargs):
        assert dist_type in DIST_TYPES, \
            f"Distribution type: {dist_type} is not supported! Supported distributions are: {DIST_TYPES}"
        super(SharedActorCriticCompression, self).__init__(k, gradient_steps, n_trees, n_leaves_per_tree, n_leaves,
                                                           output_dim, method, optimizer_class, optimizer_kwargs, False,
                                                           temperature, lambda_reg, use_W, device, actor_critic=True)
        self.dist_type = dist_type
        self.vf_coef = vf_coef

    def compress(self, A: th.Tensor, V: th.Tensor, actions: th.Tensor,
                 log_std: th.Tensor = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress actor-critic model optimizing both policy and value accuracy.
        
        Minimizes combined loss: actor loss (negative log probability) + critic loss (MSE)
        + regularization, ensuring the compressed model maintains both policy and value quality.
        
        Args:
            A (th.Tensor): Input activation matrix.
            V (th.Tensor): Weight matrix where last column is value function weights.
            actions (th.Tensor): Actions taken (for computing log probabilities).
            log_std (th.Tensor, optional): Log standard deviation for Gaussian distributions.
        
        Returns:
            Tuple of (compression_parameters, losses).
        """
        assert log_std is not None or self.dist_type != 'gaussian', \
            "Cannot compress using a Gaussian distribution without log std values!"
        
        # Move to device once before loop
        targets = A @ V
        critic_targets = targets[:, -1]
        actions = actions.to(self.device)
        if log_std is not None:
            log_std = log_std.to(self.device)

        losses = []
        if self.method == 'first_k' and not self.use_W:
            return self.compression.get_parameters(A, V), [0]
        for i in range(self.gradient_steps):
            predictions = self.compression(A, V)
            compressed_theta = predictions[:, :-1]
            compressed_critic = predictions[:, -1]
            critic_loss = 0.5*nn.functional.mse_loss(compressed_critic, critic_targets)
            dist = categorical_dist(compressed_theta) if self.dist_type == 'categorical' else \
                gaussian_dist(compressed_theta, log_std)
            log_prob = dist.log_prob(actions)
            actor_loss = -log_prob.mean()
            loss = actor_loss + critic_loss + self.compression.reg_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            print(f"{i + 1}/{self.gradient_steps} - compression loss: {loss.item()} with actor loss: "
                  f"{actor_loss.item()} critic loss: {critic_loss} reg loss: {self.compression.reg_loss} ")
        return self.compression.get_parameters(A, V), losses


class ParametricActorCompression(TreeCompression):
    """
    Tree compression for parametric actor (policy) networks.
    
    Compresses tree-based policy networks by optimizing action distribution accuracy.
    Supports categorical, Gaussian, deterministic, and supervised learning modes.
    
    Attributes:
        dist_type (str): Distribution type - 'categorical', 'gaussian', 'deterministic',
            or 'supervised_learning'.
    """
    
    def __init__(self, k: int, gradient_steps: int, dist_type: str, n_trees: int,
                 n_leaves_per_tree: Union[np.ndarray, th.Tensor], n_leaves: int, output_dim: int, method: str,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None, temperature: float = 1.0,
                 lambda_reg: float = 1.0, use_W: bool = True, device: str = 'cpu', **kwargs):
        assert dist_type in DIST_TYPES, \
            f"Distribution type: {dist_type} is not supported! Supported distributions are: {DIST_TYPES}"
        super(ParametricActorCompression, self).__init__(k, gradient_steps, n_trees, n_leaves_per_tree, n_leaves,
                                                         output_dim, method, optimizer_class, optimizer_kwargs,
                                                         dist_type in ['deterministic', 'supervised_learning'],
                                                         temperature, lambda_reg, use_W=use_W, device=device,
                                                         actor_critic=False)
        self.dist_type = dist_type

    def compress(self, A: th.Tensor, V: th.Tensor, actions: th.Tensor,
                 log_std: th.Tensor = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress actor network optimizing for action distribution or target accuracy.
        
        For probabilistic policies (categorical/Gaussian), minimizes negative log likelihood.
        For deterministic/supervised learning, minimizes MSE to target outputs.
        
        Args:
            A (th.Tensor): Input activation matrix.
            V (th.Tensor): Weight matrix.
            actions (th.Tensor): Actions for computing log probabilities (probabilistic modes).
            log_std (th.Tensor, optional): Log standard deviation for Gaussian distributions.
        
        Returns:
            Tuple of (compression_parameters, losses).
        """
        assert log_std is not None or self.dist_type != 'gaussian', \
            "Cannot compress using a Gaussian distribution without log std values!"
        
        # Move to device once before loop
        targets = A @ V
        actions = actions.to(self.device)
        if log_std is not None:
            log_std = log_std.to(self.device)

        losses = []
        if self.method == 'first_k' and not self.use_W:
            return self.compression.get_parameters(A, V), [0]
        for i in range(self.gradient_steps):
            compressed_theta = self.compression(A, V)
            if self.dist_type == 'deterministic':
                loss = nn.functional.mse_loss(compressed_theta, targets)
            else:
                dist = categorical_dist(compressed_theta) if self.dist_type == 'categorical' else \
                    gaussian_dist(compressed_theta, log_std)
                log_prob = dist.log_prob(actions)
                loss = -log_prob.mean()
            loss += self.compression.reg_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss)
            print(f"{i + 1}/{self.gradient_steps} - compression loss: {loss.item()}")
        return self.compression.get_parameters(A, V), losses


class CompressionMethod(nn.Module):
    """
    Abstract base class for tree compression methods.
    
    Defines the interface and common functionality for tree compression algorithms.
    Subclasses implement specific tree selection strategies (FirstK, BestK, etc.).
    
    Attributes:
        k (int): Number of trees to remove from the ensemble.
        n_trees (int): Total number of trees in the ensemble.
        n_leaves_per_tree (th.Tensor): Number of leaves in each tree.
        n_leaves (int): Total number of leaves across all trees.
        least_squares_W (bool): Whether to compute W via least squares or learn it.
        lambda_reg (float): Regularization parameter.
        device (str): Computation device.
        reg_loss (float): Regularization loss term.
        W (th.Tensor): Correction matrix for compression.
        actor_critic (bool): Whether this is for actor-critic models.
    """
    
    def __init__(self, k: int, n_trees: int, n_leaves_per_tree: th.Tensor,
                 n_leaves: int, least_squares_W: bool, lambda_reg: float = 1.0,
                 device: str = 'cpu', actor_critic: bool = False):
        super(CompressionMethod, self).__init__()  # Ensure proper initializatio
        self.k = k
        self.n_trees = n_trees
        self.n_leaves_per_tree = n_leaves_per_tree
        self.n_leaves = n_leaves
        self.least_squares_W = least_squares_W
        self.lambda_reg = lambda_reg
        self.device = device
        self.reg_loss = None
        self.W = None
        self.actor_critic = actor_critic

    def forward(self, A: th.Tensor, V: th.Tensor) -> th.Tensor:
        """
        Forward pass computing compressed model predictions.
        
        Applies compression matrix and correction matrix to produce predictions
        from the compressed tree ensemble.
        
        Args:
            A (th.Tensor): Input activation matrix.
            V (th.Tensor): Value/weight matrix.
        
        Returns:
            th.Tensor: Compressed model predictions.
        """
        tree_selection = self.get_tree_selection()
        C = construct_compression_matrix(tree_selection, self.n_leaves_per_tree)
        if self.least_squares_W:
            if not self.actor_critic:
                self.W = get_least_squares_W(C, A, V, self.lambda_reg)
                W = self.W
            else:
                W_critic = get_least_squares_W(C, A, V[:, -1], self.lambda_reg)
                W = th.cat([self.W, W_critic], dim=1)
        else:
            W = self.W
        # Use .T for consistency and slightly better performance
        C_CT = C @ C.T
        del C, tree_selection
        A_CT = A @ C_CT
        del A
        # In-place addition would modify V, so we keep standard addition
        U = V + W
        del V, W
        return A_CT @ U

    def get_tree_selection(self) -> th.Tensor:
        """
        Get binary tree selection mask.
        
        Returns:
            th.Tensor: Binary mask where 1 indicates tree is retained, 0 indicates removal.
        """
        raise NotImplementedError

    def get_parameters(self, A: th.Tensor, V: th.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract final compression parameters for model export.
        
        Args:
            A (th.Tensor): Input activation matrix.
            V (th.Tensor): Value/weight matrix.
        
        Returns:
            Tuple containing:
                - selection_mask (np.ndarray): Leaf-level selection mask
                - tree_selection (np.ndarray): Tree-level selection mask
                - W (np.ndarray): Correction matrix
                - n_compressed_trees (int): Number of trees after compression
                - n_compressed_leaves (int): Number of leaves after compression
        """
        tree_selection = self.get_tree_selection()
        n_compressed_trees = int(tree_selection.sum())
        selection_mask = th.repeat_interleave(tree_selection, self.n_leaves_per_tree)
        n_compressed_leaves = int(selection_mask.sum())
        if self.least_squares_W:
            C = construct_compression_matrix(tree_selection, self.n_leaves_per_tree)
            if not self.actor_critic:
                W = self.W
            else:
                W_critic = get_least_squares_W(C, A, V[:, -1], self.lambda_reg)
                W = th.cat([self.W, W_critic.unsqueeze(-1)], dim=1)
        return (selection_mask.clone().detach().cpu().numpy().astype(np.int32),
                tree_selection.clone().detach().cpu().numpy().astype(np.int32),
                W.clone().detach().cpu().numpy().astype(np.single), n_compressed_trees, n_compressed_leaves)


class FirstK(CompressionMethod):
    """
    Compression method that removes the first k trees from the ensemble.
    
    Simple deterministic compression that assumes the most important trees are at the end
    (typical for boosting where later trees correct earlier mistakes).
    """
    
    def __init__(self, k: int, n_trees: int, n_leaves_per_tree: th.Tensor,
                 n_leaves: int, output_dim: int, least_squares_W: bool, lambda_reg: float = 1.0, use_W: bool = True,
                 device: str = 'cpu', actor_critic: bool = False):
        super(FirstK, self).__init__(k, n_trees, n_leaves_per_tree, n_leaves, least_squares_W, lambda_reg, device,
                                     actor_critic)
        if not least_squares_W:
            if use_W:
                self.W = nn.Parameter(th.randn(n_leaves + 1, output_dim - 1 if actor_critic and least_squares_W else
                                               output_dim, dtype=th.float32, device=device), requires_grad=True)
            else:
                self.W = th.zeros(n_leaves + 1, output_dim, device=device, dtype=th.float32)

    def get_tree_selection(self) -> th.Tensor:
        """
        Create deterministic tree selection removing first k trees.
        
        Returns:
            th.Tensor: Binary mask with 0s for first k trees, 1s for remaining trees.
        """
        tree_selection = th.zeros(self.n_trees, dtype=th.float32, device=self.device)
        tree_selection[self.k:] = 1.0
        self.reg_loss = 0.0
        return tree_selection


class BestK(CompressionMethod):
    """
    Learned compression method that selects the best k trees to remove.
    
    Uses learnable logits with sigmoid activation and straight-through estimator
    to learn which trees are least important. The top (n_trees - k) most important
    trees are retained.
    
    Attributes:
        logits (nn.Parameter): Learnable importance scores for each tree.
        temperature (float): Temperature for sigmoid activation (controls sharpness).
    """
    
    def __init__(self, k: int, n_trees: int, n_leaves_per_tree: th.Tensor,
                 n_leaves: int, output_dim: int, least_squares_W: bool, temperature: float = 1.0,
                 lambda_reg: float = 1.0, use_W: bool = True, device: str = 'cpu', actor_critic: bool = False):
        super(BestK, self).__init__(k, n_trees, n_leaves_per_tree, n_leaves, least_squares_W, lambda_reg, device,
                                    actor_critic)
        if not least_squares_W:
            if use_W:
                self.W = nn.Parameter(th.randn(n_leaves + 1, output_dim - 1 if actor_critic and least_squares_W else
                                               output_dim, dtype=th.float32, device=device), requires_grad=True)
            else:
                self.W = th.zeros(n_leaves + 1, output_dim, device=device, dtype=th.float32)
        self.logits = nn.Parameter(th.randn(n_trees, dtype=th.float32, device=device), requires_grad=True)
        self.temperature = temperature

    def get_tree_selection(self) -> th.Tensor:
        """
        Learn and select the top (n_trees - k) most important trees.
        
        Computes importance probabilities from learnable logits, selects top trees,
        and binarizes selection using straight-through estimator for gradient flow.
        
        Returns:
            th.Tensor: Binary mask for selected trees (1 = keep, 0 = remove).
        """
        probs = th.sigmoid(self.logits / self.temperature)
        sorted_indices = th.argsort(probs, descending=True)
        # Create a mask for the top n_trees - k probabilities
        mask = th.zeros_like(probs)
        mask[sorted_indices[:self.n_trees - self.k]] = 1.0
        masked_probs = probs * mask
        # Binarize using straight through estimator
        tree_selection = BinarizeSTE.apply(masked_probs)
        # Sigmoid output is always in [0,1], no need for abs()
        self.reg_loss = self.lambda_reg * probs.sum()
        return tree_selection