##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from typing import Dict, Optional, Tuple

import numpy as np
import torch as th
from gbrl.common.utils import NumericalData

from gbrl.learners.gbt_learner import GBTLearner
from gbrl.models.base import BaseGBT
from gbrl.common.utils import (clip_grad_norm, concatenate_arrays, ensure_leaf_tensor_or_array,
                               numerical_dtype,
                               setup_optimizer, validate_array)


class ContinuousCritic(BaseGBT):
    """
    GBRL model for a Continuous Critic ensemble.
    Designed for Q-function approximation in continuous action spaces,
    such as SAC.
    Model is designed to output parameters of 3 types of Q-functions:
    - linear Q(theta(s), a) = <w_theta, a> + b_theta, (<> denotes a dot product).
    - quadratic Q(theta(s), a) = -(<w_theta, a> - b_theta)**2 + c_theta.
    - tanh Q(theta(s), a) = b_theta*tanh(<w_theta, a>)

    This allows to pass derivatives w.r.t to action a while the Q
    parameters are a function of a GBT model theta.
    The target model is approximated as the ensemble without the last <target_update_interval> trees.
    """
    def __init__(self,
                 tree_struct: Dict,
                 input_dim: int,
                 output_dim: int,
                 weights_optimizer: Dict,
                 bias_optimizer: Dict = None,
                 params: Dict = dict(),
                 target_update_interval: int = 100,
                 bias: np.ndarray = None,
                 verbose: int = 0,
                 device: str = 'cpu'):
        """
        Initializes the Continuous Critic model.

        Args:
            tree_struct (Dict): Dictionary containing tree structure information:
                    max_depth (int): maximum tree depth.
                    grow_policy (str): 'greedy' or 'oblivious'.
                    n_bins (int): number of bins per feature for candidate generation.
                    min_data_in_leaf (int): minimum number of samples in a leaf.
                    par_th (int): minimum number of samples for parallelizing on CPU.
            output_dim (int): output dimension.
            weights_optimizer Dict: dictionary containing policy optimizer
            parameters. (see GBRL for optimizer details)
            bias_optimizer Dict: dictionary containing policy optimizer parameters.
            (see GBRL for optimizer details)
            params (Dict, optional): GBRL parameters such as:
                control_variates (bool): use control variates (variance reduction technique CPU only).
                split_score_func (str): "cosine" or "l2"
                generator_type- (str): candidate generation method "Quantile" or "Uniform".
                feature_weights - (list[float]): Per-feature multiplication
                weights used when choosing the best split. Weights should be >= 0
            target_update_interval (int): target update interval
            bias (np.ndarray, optional): manually set a bias. Defaults to None = np.zeros.
            verbose (int, optional): verbosity level. Defaults to 0.
            device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """

        self.weights_optimizer = setup_optimizer(weights_optimizer,
                                                 prefix='weights_')
        self.bias_optimizer = setup_optimizer(bias_optimizer, prefix='bias_')

        super().__init__()
        self.target_learner = None
        bias = bias if bias is not None else np.zeros(output_dim, dtype=numerical_dtype)
        self.target_update_interval = target_update_interval
        # init model
        self.learner = GBTLearner(input_dim, output_dim, tree_struct,
                                  [self.weights_optimizer, self.bias_optimizer],
                                  params, verbose, device)
        self.learner.reset()
        self.learner.set_bias(bias)

    def step(self, observations: Optional[NumericalData] = None,
             weight_grad: Optional[NumericalData] = None,
             bias_grad: Optional[NumericalData] = None,
             q_grad_clip: Optional[float] = None) -> None:
        """
        Performs a single boosting step

        Args:
            observations (NumericalData):
            q_grad_clip (float, optional):. Defaults to None.
            weight_grad (Optional[NumericalData], optional): manually calculated gradients. Defaults to None.
            bias_grad (Optional[NumericalData], optional): manually calculated gradients. Defaults to None.
        """
        if observations is None:
            assert self.input is not None, ("Cannot update trees without input."
                                            "Make sure model is called with requires_grad=True")
            observations = self.input
        n_samples = len(observations)
        weight_grad = weight_grad if weight_grad is not None else self.params[0].grad.detach() * n_samples
        bias_grad = bias_grad if bias_grad is not None else self.params[1].grad.detach() * n_samples

        weight_grad = clip_grad_norm(weight_grad, q_grad_clip)
        bias_grad = clip_grad_norm(bias_grad, q_grad_clip)

        validate_array(weight_grad)
        validate_array(bias_grad)
        theta_grad = concatenate_arrays(weight_grad, bias_grad)

        self.learner.step(observations, theta_grad)
        self.grad = (weight_grad, bias_grad)
        self.input = None

    def predict_target(self, observations: NumericalData,
                       tensor: bool = True) -> Tuple[NumericalData,
                                                     NumericalData]:
        """
        Predict the parameters of a Target Continuous Critic as Tensors.
        Prediction is made by summing the outputs the trees from Continuous
        Critic model up to `n_trees - target_update_interval`.

        Args:
            observations (NumericalData):
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Tuple[th.Tensor, th.Tensor]: weights and bias parameters to thetype of Q-functions

        """
        n_trees = self.learner.get_num_trees()
        theta = self.learner.predict(observations, requires_grad=False,
                                     stop_idx=max(n_trees - self.target_update_interval, 1), tensor=tensor)
        weights = theta[:, self.weights_optimizer['start_idx']:self.weights_optimizer['stop_idx']]
        bias = theta[:, self.bias_optimizer['start_idx']:self.bias_optimizer['stop_idx']]

        return weights, bias

    def __call__(self, observations: NumericalData,
                 requires_grad: bool = True, target: bool = False,
                 start_idx: int = 0, stop_idx: int = None,
                 tensor: bool = True) -> Tuple[NumericalData,
                                               NumericalData]:
        """
        Predict the parameters of a Continuous Critic as Tensors.
        if `requires_grad=True` then stores ifferentiable parameters in self.params.
           Return type/device is identical to the input type/device.

        Args:
            observations (NumericalData)
            requires_grad (bool, optional). Defaults to True.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses
            all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Tuple[NumericalData, NumericalData]: weights and bias parameters to the type of Q-functions
        """
        if target:
            return self.predict_target(observations, tensor)

        theta = self.learner.predict(observations, requires_grad, start_idx,
                                     stop_idx, tensor)
        weights = theta[:, self.weights_optimizer['start_idx']: self.weights_optimizer['stop_idx']].squeeze()
        bias = theta[:, self.bias_optimizer['start_idx']: self.bias_optimizer['stop_idx']].squeeze()
        weights = ensure_leaf_tensor_or_array(weights, tensor=True,
                                              requires_grad=requires_grad, device=self.learner.device)
        bias = ensure_leaf_tensor_or_array(bias, tensor=True, requires_grad=requires_grad, device=self.learner.device)
        if requires_grad:
            self.grad = None
            self.params = weights, bias
            self.input = observations
        return weights, bias

    def __copy__(self) -> "ContinuousCritic":
        """
        Creates a copy of the ContinuousCritic model.
        """
        learner = self.learner.copy()
        copy_ = ContinuousCritic(learner.tree_struct, learner.input_dim,
                                 learner.output_dim, learner.optimizers[0],
                                 learner.optimizers[1], learner.params,
                                 learner.get.bias(), learner.verbose,
                                 learner.device)
        copy_.learner = learner
        return copy_


class DiscreteCritic(BaseGBT):
    """
    GBRL model for a Discrete Critic ensemble.
    Used for Q-function approximation in discrete action spaces.
    The target model is approximated as the ensemble without the last <target_update_interval> trees.
    """
    def __init__(self,
                 tree_struct: Dict,
                 input_dim: int,
                 output_dim: int,
                 critic_optimizer: Dict,
                 params: Dict = dict(),
                 target_update_interval: int = 100,
                 bias: np.ndarray = None,
                 verbose: int = 0,
                 device: str = 'cpu'):
        """
            Initializes the Discrete Critic model.

        Args:
         tree_struct (Dict): Dictionary containing tree structure information:
                max_depth (int): maximum tree depth.
                grow_policy (str): 'greedy' or 'oblivious'.
                n_bins (int): number of bins per feature for candidate generation.
                min_data_in_leaf (int): minimum number of samples in a leaf.
                par_th (int): minimum number of samples for parallelizing on CPU.
        output_dim (int): output dimension.
        critic_optimizer Dict: dictionary containing policy optimizer
        parameters. (see GradientBoostingTrees for optimizer details).
        params (Dict, optional): GBRL parameters such as:
            control_variates (bool): use control variates (variance reduction technique CPU only).
            split_score_func (str): "cosine" or "l2"
            generator_type- (str): candidate generation method "Quantile" or "Uniform".
            feature_weights - (list[float]): Per-feature multiplication
            weights used when choosing the best split. Weights should be >= 0
        verbose (int, optional): verbosity level. Defaults to 0.
        device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """
        critic_optimizer = setup_optimizer(critic_optimizer, prefix='critic_')
        super().__init__()

        self.critic_optimizer = critic_optimizer
        self.target_update_interval = target_update_interval
        bias = bias if bias is not None else np.zeros(output_dim, dtype=numerical_dtype)
        # init model
        self.learner = GBTLearner(input_dim, output_dim, tree_struct,
                                  self.critic_optimizer, params, verbose, device)
        self.learner.reset()
        self.learner.set_bias(bias)

    def step(self, observations: Optional[NumericalData] = None,
             q_grad: Optional[NumericalData] = None,
             max_q_grad_norm: Optional[np.ndarray] = None) -> None:
        """
        Performs a single boosting iterations.

        Args:
            observations (NumericalData):
            max_q_grad_norm (np.ndarray, optional). Defaults to None.
            q_grad (Optional[NumericalData], optional): manually calculated
            gradients. Defaults to None.
        """
        if observations is None:
            assert self.input is not None, ("Cannot update trees without input."
                                            "Make sure model is called with requires_grad=True")
            observations = self.input
        if q_grad is None:
            n_samples = len(observations)
            q_grad = self.params.grad.detach().cpu().numpy() * n_samples
        q_grad = clip_grad_norm(q_grad, max_q_grad_norm)

        self.learner.step(observations, q_grad)
        self.grad = q_grad
        self.input = None

    def __call__(self, observations: NumericalData, requires_grad: bool = True,
                 start_idx: int = 0, stop_idx: Optional[int] = None,
                 tensor: bool = True) -> NumericalData:
        """
        Predict and return Critic's outputs as Tensors. if
        `requires_grad=True` then stores differentiable parameters in self.params.
           Return type/device is identical to the input type/device.

        Args:
            observations (NumericalData)
            requires_grad (bool, optional). Defaults to True.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction
            (uses all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a
            numpy array. Defaults to True.

        Returns:
            NumericalData: Critic's outputs.
        """
        q_values = self.learner.predict(observations, requires_grad, start_idx, stop_idx, tensor)
        if requires_grad:
            self.grad = None
            self.params = q_values
            self.input = observations
        return q_values

    def predict_target(self, observations: NumericalData,
                       tensor: bool = True) -> th.Tensor:
        """
        Predict and return Target Critic's outputs as Tensors.
           Prediction is made by summing the outputs the trees from Continuous
           Critic model up to `n_trees - target_update_interval`.

        Args:
            observations (NumericalData)

        Returns:
            th.Tensor: Target Critic's outputs.
        """
        n_trees = self.learner.get_num_trees()
        return self.learner.predict(observations, requires_grad=False,
                                    stop_idx=max(n_trees - self.target_update_interval, 1), tensor=tensor)

    def __copy__(self) -> "DiscreteCritic":
        """
        Creates a copy of the DiscreteCritic model.
        """
        learner = self.learner.copy()
        copy_ = DiscreteCritic(learner.tree_struct, learner.input_dim,
                               learner.output_dim, learner.optimizers[0],
                               learner.params, self.target_update_interval,
                               learner.get_bias(), learner.verbose,
                               learner.device)
        copy_.learner = learner
        return copy_
