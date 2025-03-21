##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from typing import Dict, Optional

import numpy as np
from gbrl.common.utils import NumericalData

from gbrl.learners.gbt_learner import GBTLearner
from gbrl.models.base import BaseGBT
from gbrl.common.utils import (clip_grad_norm, concatenate_arrays, constant_like,
                               ensure_leaf_tensor_or_array, numerical_dtype,
                               setup_optimizer, validate_array)


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
                 bias: np.ndarray = None,
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
        # init model
        self.learner = GBTLearner(input_dim, output_dim, tree_struct, policy_optimizer,
                                  params, verbose, device)
        self.learner.reset()
        self.learner.set_bias(bias)
        self.params = None
        self.input = None
        self.grad = None

    def step(self, observations: Optional[NumericalData] = None,
             policy_grad: Optional[NumericalData] = None,
             policy_grad_clip: Optional[float] = None,) -> None:
        """
        Performs a single boosting iteration.

        Args:
            observations (NumericalData):
            policy_grad_clip (float, optional): . Defaults to None.
            policy_grad (Optional[NumericalData], optional): manually
            calculated gradients. Defaults to None.
        """
        if observations is None:
            assert self.input is not None, "Cannot update trees without input."
            "Make sure model is called with requires_grad=True"
            observations = self.input
        n_samples = len(observations)

        policy_grad = policy_grad if policy_grad is not None else \
            self.params.grad.detach() * n_samples
        policy_grad = clip_grad_norm(policy_grad, policy_grad_clip)
        validate_array(policy_grad)

        self.learner.step(observations, policy_grad)
        self.grad = policy_grad
        self.input = None

    def __call__(self, observations: NumericalData,
                 requires_grad: bool = True, start_idx: int = 0,
                 stop_idx: int = None, tensor: bool = True) -> NumericalData:
        """
        Returns actor output as Tensor. If `requires_grad=True`, stores
        differentiable parameters in `self.params`.

        Args:
            observations (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients.
            Defaults to True.
            start_idx (int, optional): Start tree index for prediction.
            Defaults to 0.
            stop_idx (Optional[int], optional): Stop tree index for prediction.
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
        learner = self.learner.copy()
        copy_ = ParametricActor(learner.tree_struct, learner.input_dim,
                                learner.output_dim, learner.optimizers[0],
                                learner.params, learner.get_bias(),
                                learner.verbose, learner.device)
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
                 std_optimizer: Dict = None,
                 log_std_init: float = -2,
                 params: Dict = dict(),
                 bias: np.ndarray = None,
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
        policy_dim = output_dim
        if std_optimizer is not None:
            std_optimizer = setup_optimizer(std_optimizer, prefix='std_')
            policy_dim = output_dim // 2
            bias[policy_dim:] = log_std_init*np.ones(policy_dim,
                                                     dtype=numerical_dtype)
        self.log_std_init = log_std_init
        self.fixed_std = std_optimizer is None
        self.policy_dim = policy_dim

        # init model
        self.learner = GBTLearner(input_dim, output_dim, tree_struct,
                                  [mu_optimizer, std_optimizer], params,
                                  verbose, device)
        self.learner.reset()
        self.learner.set_bias(bias)

    def step(self, observations: Optional[NumericalData] = None,
             mu_grad: Optional[NumericalData] = None,
             log_std_grad: Optional[NumericalData] = None,
             mu_grad_clip: Optional[float] = None,
             log_std_grad_clip: Optional[float] = None) -> None:
        """
        Performs a single boosting iteration.

        Args:
            observations (NumericalData): Input observations.
            mu_grad (Optional[NumericalData], optional):
            Manually computed mean gradients.
            log_std_grad (Optional[NumericalData], optional):
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
        mu_grad = mu_grad if mu_grad is not None else \
            self.params[0].grad.detach() * n_samples
        mu_grad = clip_grad_norm(mu_grad, mu_grad_clip)

        if not self.fixed_std:
            log_std_grad = log_std_grad if log_std_grad is not None else \
                self.params[1].grad.detach() * n_samples
            log_std_grad = clip_grad_norm(log_std_grad, log_std_grad_clip)
            theta_grad = concatenate_arrays(mu_grad, log_std_grad)
        else:
            theta_grad = mu_grad

        validate_array(theta_grad)

        self.learner.step(observations, theta_grad)
        self.grad = mu_grad
        if not self.fixed_std:
            self.grad = (mu_grad, log_std_grad)
        self.input = None

    def __call__(self, observations: NumericalData,
                 requires_grad: bool = True,
                 start_idx: int = 0, stop_idx: int = None,
                 tensor: bool = True) -> NumericalData:
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
            self.grad = None
            self.params = mean_actions, log_std
            self.input = observations
        return mean_actions, log_std

    def __copy__(self) -> "GaussianActor":
        """
        Creates a copy of the GaussianActor instance.

        Returns:
            GaussianActor: A copy of the current model.
        """
        learner = self.learner.copy()
        copy_ = GaussianActor(learner.tree_struct, learner.input_dim,
                              learner.output_dim, learner.optimizers[0],
                              learner.optimizers[1], learner.params,
                              learner.get_bias(), learner.verbose,
                              learner.device)
        copy_.learner = learner
        return copy_
