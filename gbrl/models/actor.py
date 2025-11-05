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
"""
Actor Model Module

This module provides actor models for reinforcement learning, including
ParametricActor for deterministic/discrete policies and GaussianActor
for continuous stochastic policies with Gaussian distributions.
"""
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch as th

from gbrl.common.utils import (NumericalData, clip_grad_norm,
                               concatenate_arrays, constant_like,
                               ensure_leaf_tensor_or_array, numerical_dtype,
                               setup_optimizer, validate_array)
from gbrl.learners.gbt_learner import GBTLearner
from gbrl.models.base import BaseGBT


class ParametricActor(BaseGBT):
    """
    GBRL model for a ParametricActor ensemble. ParametricActor outputs
    a single parameter per action dimension, allowing deterministic or
    stochastic behavior
    (e.g., for discrete action spaces).
    """
    def __init__(self,
                 tree_struct: Dict,
                 input_dim: int,
                 output_dim: int,
                 policy_optimizer: Dict,
                 params: Dict = dict(),
                 bias: Optional[Union[float, np.ndarray]] = None,
                 verbose: int = 0,
                 device: str = 'cpu'):
        """
        Initializes the ParametricActor model.

        Args:
            tree_struct (Dict): Dictionary containing tree structure
            information:
                - max_depth (int): Maximum tree depth.
                - grow_policy (str): 'greedy' or 'oblivious'.
                - n_bins (int): Number of bins per feature for candidate
                generation.
                - min_data_in_leaf (int): Minimum number of samples in a leaf.
                - par_th (int): Minimum number of samples for parallelizing on
                CPU.
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.
            policy_optimizer (Dict): Dictionary containing policy optimizer
            parameters.
            params (Dict, optional): Additional GBRL parameters.
            bias (np.ndarray, optional): Bias initialization, defaults to zero.
            verbose (int, optional): Verbosity level. Defaults to 0.
            device (str, optional): Compute device ('cpu' or 'cuda'). Defaults
            to 'cpu'.
        """
        policy_optimizer = setup_optimizer(policy_optimizer, prefix='policy_')
        super().__init__()
        bias = bias if bias is not None else np.zeros(output_dim,
                                                      dtype=numerical_dtype)
        if isinstance(bias, float):
            bias = bias * np.ones(output_dim, dtype=numerical_dtype)
        # init model
        self.learner = GBTLearner(input_dim=input_dim,
                                  output_dim=output_dim,
                                  tree_struct=tree_struct,
                                  optimizers=policy_optimizer,
                                  params=params,
                                  verbose=verbose,
                                  device=device)
        self.learner.reset()
        self.learner.set_bias(bias)
        self.params = None
        self.input = None
        self.grads = None

    def step(self, observations: Optional[NumericalData] = None,
             policy_grads: Optional[NumericalData] = None,
             policy_grad_clip: Optional[float] = None,
             ) -> None:
        """
        Performs a single boosting iteration.

        Args:
            observations (NumericalData):
            policy_grad_clip (float, optional): . Defaults to None.
            policy_grads (Optional[NumericalData], optional): manually
                calculated gradients. Defaults to None.
        """
        if observations is None:
            assert self.input is not None, "Cannot update trees without input."
            "Make sure model is called with requires_grad=True"
            observations = self.input
        n_samples = len(observations)

        if policy_grads is None:
            assert self.params is not None, "params must be set to compute gradients."
            assert isinstance(self.params, th.Tensor), "params must be a Tensor to compute gradients."
            assert self.params.grad is not None, "params.grad must be set to compute gradients."
            policy_grads = self.params.grad.detach() * n_samples

        policy_grads = clip_grad_norm(policy_grads, policy_grad_clip)
        validate_array(policy_grads)

        self.learner.step(inputs=observations, grads=policy_grads)
        self.grads = policy_grads
        self.input = None

    def __call__(self, observations: NumericalData,
                 requires_grad: bool = True, start_idx: Optional[int] = None,
                 stop_idx: Optional[int] = None, tensor: bool = True) -> NumericalData:
        """
        Returns actor output as Tensor. If `requires_grad=True`, stores
        differentiable parameters in `self.params`.

        Args:
            observations (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients.
            Defaults to True.
            start_idx (int, optional): Start tree index for prediction.
            Defaults to 0.
            stop_idx (int, optional): Stop tree index for prediction.
            Defaults to None.
            tensor (bool, optional): Whether to return a PyTorch Tensor.
            Defaults to True.

        Returns:
            NumericalData: GBRL outputs - a single parameter per action
            dimension.
        """
        params = self.learner.predict(observations, requires_grad, start_idx,
                                      stop_idx, tensor)
        if requires_grad:
            self.grads = None
            self.params = params
            self.input = observations
        return params

    def __copy__(self) -> "ParametricActor":
        """
        Creates a copy of the ParametricActor instance.

        Returns:
            ParametricActor: A copy of the current model.
        """
        assert self.learner is not None, "learner must be initialized first."

        learner = self.learner.copy()
        assert isinstance(learner.input_dim, int), "learner.input_dim must be int"
        assert isinstance(learner.output_dim, int), "learner.output_dim must be int"
        assert learner.optimizers is not None, "learner.optimizers must be initialized"
        copy_ = ParametricActor(learner.tree_struct,
                                learner.input_dim,
                                learner.output_dim,
                                learner.optimizers[0],
                                learner.params,
                                learner.get_bias(),  # type: ignore
                                learner.verbose,
                                learner.device)
        copy_.learner = learner
        return copy_


class GaussianActor(BaseGBT):
    """
    GBRL model for an actor ensemble used in algorithms such as SAC.
    This model outputs the mean (`mu`) and log standard deviation (`log_std`)
    of a Gaussian distribution, allowing stochastic action selection.
    """
    def __init__(self,
                 tree_struct: Dict,
                 input_dim: int,
                 output_dim: int,
                 mu_optimizer: Dict,
                 std_optimizer: Optional[Dict] = None,
                 log_std_init: float = -2,
                 params: Dict = dict(),
                 bias: Optional[Union[np.ndarray, float]] = None,
                 verbose: int = 0,
                 device: str = 'cpu'):
        """
        Initializes the GaussianActor model.

        Args:
         tree_struct (Dict): Dictionary containing tree structure information:
                max_depth (int): maximum tree depth.
                grow_policy (str): 'greedy' or 'oblivious'.
                n_bins (int): number of bins per feature for candidate
                generation.
                min_data_in_leaf (int): minimum number of samples in a leaf.
                par_th (int): minimum number of samples for parallelizing on
                CPU.
        output_dim (int): output dimension.
        mu_optimizer Dict: dictionary containing Gaussian mean optimizer
        parameters. (see GradientBoostingTrees for optimizer details)
        std_optimizer Dict: dictionary containing Gaussian sigma optimizer
        parameters. (see GradientBoostingTrees for optimizer details)
        log_std_init (float): initial value of log_std
        params (Dict, optional): GBRL parameters such as:
            control_variates (bool): use control variates (variance reduction
            technique CPU only).
            split_score_func (str): "cosine" or "l2"
            generator_type- (str): candidate generation method "Quantile" or
            "Uniform".
            feature_weights - (list[float]): Per-feature multiplication
            weights used when choosing the best split. Weights should be >= 0
        bias (np.ndarray, optional): manually set a bias. Defaults to None =
        np.zeros.
        verbose (int, optional): verbosity level. Defaults to 0.
        device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to
        'cpu'.
        """
        super().__init__()
        mu_optimizer = setup_optimizer(mu_optimizer, prefix='mu_')

        bias = bias if bias is not None else np.zeros(output_dim,
                                                      dtype=numerical_dtype)
        if isinstance(bias, float):
            bias = bias * np.ones(output_dim, dtype=numerical_dtype)

        policy_dim = output_dim
        if std_optimizer is not None:
            std_optimizer = setup_optimizer(std_optimizer, prefix='std_')
            policy_dim = output_dim // 2
            bias[policy_dim:] = log_std_init*np.ones(policy_dim,  # type: ignore
                                                     dtype=numerical_dtype)
        self.log_std_init = log_std_init
        self.fixed_std = std_optimizer is None
        self.policy_dim = policy_dim

        # init model
        self.learner = GBTLearner(input_dim=input_dim,
                                  output_dim=output_dim,
                                  tree_struct=tree_struct,
                                  optimizers=[mu_optimizer, std_optimizer],
                                  params=params,
                                  verbose=verbose,
                                  device=device)
        self.learner.reset()
        self.learner.set_bias(bias)

    def step(self, observations: Optional[NumericalData] = None,
             mu_grads: Optional[NumericalData] = None,
             log_std_grads: Optional[NumericalData] = None,
             mu_grad_clip: Optional[float] = None,
             log_std_grad_clip: Optional[float] = None
             ) -> None:
        """
        Performs a single boosting iteration.

        Args:
            observations (NumericalData): Input observations.
            mu_grads (Optional[NumericalData], optional):
                Manually computed mean gradients.
            log_std_grads (Optional[NumericalData], optional):
                Manually computed log standard deviation gradients.
            mu_grad_clip (Optional[float], optional): Gradient clipping for
                mean. Defaults to None.
            log_std_grad_clip (Optional[float], optional): Gradient clipping
                for log standard deviation. Defaults to None.

        """
        if observations is None:
            assert self.input is not None, "Cannot update trees without input."
            "Make sure model is called with requires_grad=True"
            observations = self.input
        n_samples = len(observations)

        if mu_grads is None:
            assert self.params is not None, "params must be set to compute gradients."
            assert isinstance(self.params, tuple), "params must be a tuple to compute gradients."
            assert isinstance(self.params[0], th.Tensor), "params[0] must be a Tensor to compute gradients."
            assert self.params[0].grad is not None, "params[0].grad must be set to compute gradients."  # type: ignore
            mu_grads = self.params[0].grad.detach() * n_samples  # type: ignore
        mu_grads = clip_grad_norm(mu_grads, mu_grad_clip)  # type: ignore

        if not self.fixed_std:
            if log_std_grads is None:
                assert self.params is not None, "params must be set to compute gradients."
                assert isinstance(self.params, list), "params must be a list to compute gradients."
                assert isinstance(self.params[1], th.Tensor), "params[1] must be a Tensor to compute gradients."
                assert self.params[1].grad is not None, "params[1].grad must be set to compute gradients."  # type: ignore
                log_std_grads = self.params[1].grad.detach() * n_samples  # type: ignore
            log_std_grads = clip_grad_norm(log_std_grads, log_std_grad_clip)  # type: ignore
            theta_grad = concatenate_arrays(mu_grads, log_std_grads)  # type: ignore
        else:
            theta_grad = mu_grads

        validate_array(theta_grad)

        self.learner.step(observations, theta_grad)
        self.grads = mu_grads
        if not self.fixed_std:
            self.grads = (mu_grads, log_std_grads)
        self.input = None

    def __call__(self, observations: NumericalData,
                 requires_grad: bool = True,
                 start_idx: Optional[int] = None,
                 stop_idx: Optional[int] = None,
                 tensor: bool = True) -> Tuple[NumericalData, NumericalData]:
        """
        Returns actor's outputs as tensor. If `requires_grad=True` then stores
           differentiable parameters in self.params. Return type/device is
           identical to the input type/device.
           Requires_grad is ignored if input is a numpy array.
        Args:
            observations (NumericalData)
            requires_grad bool: Defaults to None.
            start_idx (int, optional): start tree index for prediction.
            Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses
            all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a
            numpy array. Defaults to True.

        Returns:
            NumericalData: Gaussian parameters
        """
        theta = self.learner.predict(observations, requires_grad, start_idx,
                                     stop_idx, tensor)
        mean_actions = theta if self.fixed_std else theta[:, :self.policy_dim]
        if not self.fixed_std:
            mean_actions = ensure_leaf_tensor_or_array(mean_actions, tensor=True, requires_grad=requires_grad, device=self.learner.device)
        log_std = constant_like(theta, self.log_std_init) if self.fixed_std else theta[:, self.policy_dim:]
        log_std = ensure_leaf_tensor_or_array(log_std, tensor=True, requires_grad=False if
                                              self.fixed_std else
                                              requires_grad,
                                              device=self.learner.device)
        if requires_grad:
            self.grads = None
            self.params = mean_actions, log_std
            self.input = observations
        return mean_actions, log_std

    def __copy__(self) -> "GaussianActor":
        """
        Creates a copy of the GaussianActor instance.

        Returns:
            GaussianActor: A copy of the current model.
        """
        assert self.learner is not None, "learner must be initialized first."
        learner = self.learner.copy()
        assert learner.optimizers is not None, "learner.optimizers must be initialized"
        std_optimizer = None if len(learner.optimizers) < 2 else learner.optimizers[1]
        copy_ = GaussianActor(tree_struct=learner.tree_struct,
                              input_dim=learner.input_dim,  # type: ignore
                              output_dim=learner.output_dim,  # type: ignore
                              mu_optimizer=learner.optimizers[0],  # type: ignore
                              std_optimizer=std_optimizer,
                              params=learner.params,
                              bias=learner.get_bias(),  # type: ignore
                              verbose=learner.verbose,
                              device=learner.device)
        copy_.learner = learner
        return copy_
