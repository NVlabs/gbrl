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
Actor-Critic Model Module

This module provides the ActorCritic model, which combines policy and value
function learning in a single or separate tree architecture.
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th

from gbrl.common.utils import (NumericalData, clip_grad_norm, numerical_dtype,
                               setup_optimizer, validate_array)
from gbrl.learners.actor_critic_learner import (SeparateActorCriticLearner,
                                                SharedActorCriticLearner)
from gbrl.models.base import BaseGBT


class ActorCritic(BaseGBT):
    """
    GBRL model for a shared Actor and Critic ensemble.

    Supports both shared and separate actor-critic tree structures.
    """
    def __init__(self,
                 tree_struct: Dict,
                 input_dim: int,
                 output_dim: int,
                 policy_optimizer: Dict,
                 value_optimizer: Dict,
                 shared_tree_struct: bool = True,
                 params: Dict = dict(),
                 bias: Optional[Union[Union[float, NumericalData],
                                List[Union[float, NumericalData]]]] = None,
                 verbose: int = 0,
                 device: str = 'cpu'):
        """
        GBRL model for a shared Actor and Critic ensemble.

        Args:
         tree_struct (Dict): Dictionary containing tree structure information:
                max_depth (int): maximum tree depth.
                grow_policy (str): 'greedy' or 'oblivious'.
                n_bins (int): number of bins per feature for candidate generation.
                min_data_in_leaf (int): minimum number of samples in a leaf.
                par_th (int): minimum number of samples for parallelizing on CPU.
        output_dim (int): output dimension.
        policy_optimizer Dict: dictionary containing policy optimizer
        parameters (see GBRL class for optimizer details).
        value_optimizer Dict: dictionary containing value optimizer parameters
        (see GBRL class for optimizer details).
        shared_tree_struct (bool, optional): sharing actor and critic.
        Defaults to True.
        params (Dict, optional): GBRL parameters such as:
            control_variates (bool): use control variates (variance reduction technique CPU only).
            split_score_func (str): "cosine" or "l2"
            generator_type- (str): candidate generation method "Quantile" or "Uniform".
            feature_weights - (list[float]): Per-feature multiplication
            weights used when choosing the best split. Weights should be >= 0
        bias (np.ndarray, optional): manually set a bias. Defaults to None = np.zeros.
        verbose (int, optional): verbosity level. Defaults to 0.
        device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """
        super().__init__()
        policy_optimizer = setup_optimizer(policy_optimizer, prefix='policy_')
        value_optimizer = setup_optimizer(value_optimizer, prefix='value_')

        self.shared_tree_struct = True if value_optimizer is None else \
            shared_tree_struct

        if bias is None:
            if shared_tree_struct:
                bias = np.zeros(output_dim, dtype=numerical_dtype)
            else:
                bias = [np.zeros(output_dim - 1, dtype=numerical_dtype), 0]

        if not shared_tree_struct and not isinstance(bias, list):
            raise ValueError("When using separate tree structures for actor"
                             " and critic, bias must be a list of two "
                             "elements: [actor_bias, critic_bias]")

        if isinstance(bias, float):
            bias = bias * np.ones(output_dim if shared_tree_struct else output_dim - 1,
                                  dtype=numerical_dtype)

        # init model
        if self.shared_tree_struct:
            self.learner = SharedActorCriticLearner(input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    tree_struct=tree_struct,
                                                    policy_optimizer=policy_optimizer,
                                                    value_optimizer=value_optimizer,
                                                    params=params, verbose=verbose, device=device)
        else:
            self.learner = SeparateActorCriticLearner(input_dim=input_dim,
                                                      output_dim=output_dim,
                                                      tree_struct=tree_struct,
                                                      policy_optimizer=policy_optimizer,
                                                      value_optimizer=value_optimizer,
                                                      params=params, verbose=verbose, device=device)
        self.learner.reset()
        self.learner.set_bias(bias)  # type: ignore
        self.policy_grads = None
        self.value_grads = None

    @classmethod
    def load_learner(cls, load_name: str, device: str) -> "ActorCritic":
        """Loads GBRL model from a file

        Args:
            load_name (str): full path to file name

        Returns:
            ActorCritic: loaded ActorCriticModel
        """
        policy_file = load_name + '_policy.gbrl_model'
        value_file = load_name + '_value.gbrl_model'

        instance = cls.__new__(cls)
        if os.path.isfile(policy_file) and os.path.isfile(value_file):
            instance.learner = SeparateActorCriticLearner.load(load_name,
                                                               device)
            instance.shared_tree_struct = False
        else:
            instance.learner = SharedActorCriticLearner.load(load_name,
                                                             device)
            instance.shared_tree_struct = True

        instance.policy_grads = None
        instance.value_grads = None
        instance.params = None
        instance.input = None
        return instance

    def predict_policy(self, observations: NumericalData,
                       requires_grad: bool = True, start_idx: int = 0,
                       stop_idx: Optional[int] = None, tensor: bool = True) -> \
            NumericalData:
        """
        Predict only policy. If `requires_grad=True` then stores differentiable parameters in self.params
           Return type/device is identical to the input type/device.

        Args:
            observations (NumericalData)
            requires_grad (bool, optional). Defaults to True. Ignored if input is a numpy array.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses
            all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            NumericalData: policy
        """
        policy = self.learner.predict_policy(observations, requires_grad,
                                             start_idx, stop_idx, tensor)
        if requires_grad:
            self.policy_grads = None
            self.params = policy
        return policy

    def predict_values(self, observations: NumericalData,
                       requires_grad: bool = True, start_idx: int = 0,
                       stop_idx: Optional[int] = None, tensor: bool = True) -> \
            NumericalData:
        """
        Predict only values. If `requires_grad=True` then stores differentiable parameters in self.params
           Return type/device is identical to the input type/device.

        Args:
            observations (NumericalData)
            requires_grad (bool, optional). Defaults to True. Ignored if input is a numpy array.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses
            all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            NumericalData: values
        """
        values = self.learner.predict_critic(observations, requires_grad,
                                             start_idx, stop_idx, tensor)
        if requires_grad:
            self.value_grads = None
            self.params = values
        return values

    def __call__(self, observations: NumericalData,
                 requires_grad: bool = True, start_idx: int = 0,
                 stop_idx: Optional[int] = None, tensor: bool = True) -> \
            Tuple[NumericalData, NumericalData]:
        """
        Predicts  and returns actor and critic outputs as tensors.
        If `requires_grad=True` then stores differentiable parameters in self.params
           Return type/device is identical to the input type/device.
        Args:
            observations (NumericalData)
            requires_grad (bool, optional). Defaults to True. Ignored if input is a numpy array.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses
            all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Tuple[NumericalData, NumericalData]: actor and critic output
        """
        params = self.learner.predict(observations, requires_grad, start_idx,
                                      stop_idx, tensor)
        if requires_grad:
            self.policy_grads = None
            self.value_grads = None
            self.params = tuple(params)
            self.inputs = observations
        return params  # type: ignore

    def step(self, observations: Optional[NumericalData] = None,
             policy_grads: Optional[NumericalData] = None,
             value_grads: Optional[NumericalData] = None,
             policy_grad_clip: Optional[float] = None,
             value_grad_clip: Optional[float] = None,
             ) -> None:
        """
        Performs a boosting step for both the actor and critic.

        If `observations` is not provided, it uses the stored input from thelast forward pass.

        Args:
            observations (Optional[NumericalData], optional):Input observations.
            policy_grads (Optional[NumericalData], optional):Manually computed gradients for the policy.
            value_grads (Optional[NumericalData], optional): Manually computed gradients for the value function.
            policy_grad_clip (Optional[float], optional):Gradient clipping value for policy updates.
            value_grad_clip (Optional[float], optional): Gradient clipping value for value updates.
        """
        if observations is None:
            assert self.inputs is not None, ("Cannot update trees without input."
                                             "Make sure model is called with requires_grad=True")
            observations = self.inputs
        
        # Handle 1D observations
        if observations.ndim == 1:
            n_samples = 1 if self.learner.input_dim > 1 else len(observations)
        else:
            n_samples = len(observations)

        if policy_grads is None:
            assert self.params is not None, "params must be set to compute gradients."
            assert isinstance(self.params, tuple), "params must be a tuple to compute gradients."
            assert isinstance(self.params[0], th.Tensor), "params[0] must be a Tensor to compute gradients."
            assert self.params[0].grad is not None, "params[0].grad must be set to compute gradients."  # type: ignore
            policy_grads = self.params[0].grad.detach() * n_samples  # type: ignore

        if value_grads is None:
            assert self.params is not None, "params must be set to compute gradients."
            assert isinstance(self.params, tuple), "params must be a tuple to compute gradients."
            assert isinstance(self.params[1], th.Tensor), "params[1] must be a Tensor to compute gradients."
            assert self.params[1].grad is not None, "params[1].grad must be set to compute gradients."  # type: ignore
            value_grads = self.params[1].grad.detach() * n_samples  # type: ignore

        policy_grads = clip_grad_norm(policy_grads, policy_grad_clip)  # type: ignore
        value_grads = clip_grad_norm(value_grads, value_grad_clip)  # type: ignore

        validate_array(policy_grads)
        validate_array(value_grads)

        self.learner.step(inputs=observations,
                          grads=(policy_grads, value_grads))
        self.policy_grads = policy_grads
        self.value_grads = value_grads
        self.inputs = None

    def actor_step(self, observations: Optional[NumericalData] = None,
                   policy_grads: Optional[NumericalData] = None,
                   policy_grad_clip: Optional[float] = None
                   ) -> None:
        """
        Performs a single boosting step for the actor (should only be used
        if actor and critic use separate models)

        Args:
            observations (NumericalData):
            policy_grad_clip (float, optional): Defaults to None.
            policy_grads (Optional[NumericalData], optional): manually calculated gradients. Defaults to None.

        Returns:
            np.ndarray: policy gradient
        """
        assert not self.shared_tree_struct, "Cannot separate boosting steps"
        "for actor and critic when using separate tree architectures!"
        if observations is None:
            assert self.inputs is not None, ("Cannot update trees without input."
                                             "Make sure model is called with requires_grad=True")
            observations = self.inputs
        
        # Handle 1D observations
        if observations.ndim == 1:
            n_samples = 1 if self.learner.input_dim > 1 else len(observations)
        else:
            n_samples = len(observations)

        if policy_grads is None:
            assert self.params is not None, "params must be set to compute gradients."
            assert isinstance(self.params, tuple), "params must be a tuple to compute gradients."
            assert isinstance(self.params[0], th.Tensor), "params[0] must be a Tensor to compute gradients."
            assert self.params[0].grad is not None, "params[0].grad must be set to compute gradients."  # type: ignore
            policy_grads = self.params[0].grad.detach() * n_samples  # type: ignore

        policy_grads = clip_grad_norm(policy_grads, policy_grad_clip)
        validate_array(policy_grads)

        self.learner.step_actor(inputs=observations,  # type: ignore
                                grads=policy_grads)
        self.policy_grads = policy_grads

    def critic_step(self, observations: Optional[NumericalData] = None,
                    value_grads: Optional[NumericalData] = None,
                    value_grad_clip: Optional[float] = None,
                    ) -> None:
        """
        Performs a single boosting step for the critic (should only be used
        if actor and critic use separate models)

        Args:
            observations (NumericalData):
            value_grad_clip (float, optional): Defaults to None.
            value_grads (Optional[NumericalData], optional): manually calculated gradients. Defaults to None.

        Returns:
            np.ndarray: value gradient
        """
        assert not self.shared_tree_struct, ("Cannot separate boosting steps"
                                             "for actor and critic when using separate tree architectures!")
        if observations is None:
            assert self.inputs is not None, ("Cannot update trees without input."
                                             "Make sure model is called with requires_grad=True")
            observations = self.inputs

        # Handle 1D observations
        if observations.ndim == 1:
            n_samples = 1 if self.learner.input_dim > 1 else len(observations)
        else:
            n_samples = len(observations)

        if value_grads is None:
            assert self.params is not None, "params must be set to compute gradients."
            assert isinstance(self.params, tuple), "params must be a tuple to compute gradients."
            assert isinstance(self.params[1], th.Tensor), "params[1] must be a Tensor to compute gradients."
            assert self.params[1].grad is not None, "params[1].grad must be set to compute gradients."  # type: ignore
            value_grads = self.params[1].grad.detach() * n_samples  # type: ignore

        value_grads = clip_grad_norm(value_grads, value_grad_clip)

        validate_array(value_grads)
        self.learner.step_critic(inputs=observations,  # type: ignore
                                 grads=value_grads)
        self.value_grads = value_grads

    def save_learner(self, save_path: str) -> None:
        """
        Saves model to file

        Args:
            filename (str): Absolute path and name of save filename.
        """
        if self.shared_tree_struct:
            self.learner.save(save_path)
        else:
            self.learner.save(save_path, custom_names=['policy', 'value'])  # type: ignore

    def copy(self) -> "ActorCritic":
        """
        Copy class instance

        Returns:
            ActorCritic: copy of current instance
        """
        return self.__copy__()

    def __copy__(self) -> "ActorCritic":
        assert self.learner is not None, "learner must be initialized first."
        learner = self.learner.copy()
        assert learner.optimizers is not None, "learner.optimizers must be initialized"
        assert len(learner.optimizers) == 2, "learner.optimizers must be initialized"
        copy_ = ActorCritic(tree_struct=learner.tree_struct,
                            input_dim=learner.input_dim,  # type: ignore
                            output_dim=learner.output_dim,  # type: ignore
                            policy_optimizer=learner.optimizers[0],
                            value_optimizer=learner.optimizers[1],
                            shared_tree_struct=self.shared_tree_struct,
                            params=learner.params,
                            bias=learner.get_bias(),  # type: ignore
                            verbose=learner.verbose,
                            device=learner.device)
        copy_.learner = learner
        return copy_
