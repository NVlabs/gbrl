##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
import os
from typing import Dict, Optional, Tuple

import numpy as np
from gbrl.common.utils import NumericalData

from gbrl.learners.actor_critic_learner import (SeparateActorCriticLearner,
                                                SharedActorCriticLearner)
from gbrl.models.base import BaseGBT
from gbrl.common.utils import (clip_grad_norm, numerical_dtype, setup_optimizer,
                               validate_array)


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
                 value_optimizer: Dict = None,
                 shared_tree_struct: bool = True,
                 params: Dict = dict(),
                 bias: np.ndarray = None,
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
        if value_optimizer is not None:
            value_optimizer = setup_optimizer(value_optimizer, prefix='value_')

        self.shared_tree_struct = True if value_optimizer is None else \
            shared_tree_struct
        bias = bias if bias is not None else np.zeros(output_dim if
                                                      shared_tree_struct
                                                      else output_dim - 1,
                                                      dtype=numerical_dtype)
        # init model
        if self.shared_tree_struct:
            self.learner = SharedActorCriticLearner(input_dim, output_dim,
                                                    tree_struct,
                                                    policy_optimizer,
                                                    value_optimizer,
                                                    params, verbose, device)
            self.learner.reset()
            self.learner.set_bias(bias)
        else:
            self.learner = SeparateActorCriticLearner(input_dim, output_dim,
                                                      tree_struct,
                                                      policy_optimizer,
                                                      value_optimizer,
                                                      params, verbose, device)
            self.learner.reset()
            self.learner.set_bias(bias, model_idx=0)
        self.policy_grad = None
        self.value_grad = None

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

        instance.policy_grad = None
        instance.value_grad = None
        instance.params = None
        instance.input = None
        return instance

    def predict_values(self, observations: NumericalData,
                       requires_grad: bool = True, start_idx: int = 0,
                       stop_idx: int = None, tensor: bool = True) -> \
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
            self.value_grad = None
            self.params = values
        return values

    def __call__(self, observations: NumericalData,
                 requires_grad: bool = True, start_idx: int = 0,
                 stop_idx: int = None, tensor: bool = True) -> \
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
            self.policy_grad = None
            self.value_grad = None
            self.params = params
            self.input = observations
        return params

    def step(self, observations: Optional[NumericalData] = None,
             policy_grad: Optional[NumericalData] = None,
             value_grad: Optional[NumericalData] = None,
             policy_grad_clip: Optional[float] = None,
             value_grad_clip: Optional[float] = None) -> None:
        """
        Performs a boosting step for both the actor and critic.

        If `observations` is not provided, it uses the stored input from thelast forward pass.

        Args:
            observations (Optional[NumericalData], optional):Input observations.
            policy_grad (Optional[NumericalData], optional):Manually computed gradients for the policy.
            value_grad (Optional[NumericalData], optional): Manually computed gradients for the value function.
            policy_grad_clip (Optional[float], optional):Gradient clipping value for policy updates.
            value_grad_clip (Optional[float], optional): Gradient clipping value for value updates.
        """
        if observations is None:
            assert self.input is not None, ("Cannot update trees without input."
                                            "Make sure model is called with requires_grad=True")
            observations = self.input
        n_samples = len(observations)

        policy_grad = policy_grad if policy_grad is not None else self.params[0].grad.detach() * n_samples
        value_grad = value_grad if value_grad is not None else self.params[1].grad.detach() * n_samples

        policy_grad = clip_grad_norm(policy_grad, policy_grad_clip)
        value_grad = clip_grad_norm(value_grad, value_grad_clip)

        validate_array(policy_grad)
        validate_array(value_grad)

        self.learner.step(observations, policy_grad, value_grad)
        self.policy_grad = policy_grad
        self.value_grad = value_grad
        self.input = None

    def actor_step(self, observations: Optional[NumericalData]
                   = None, policy_grad: Optional[NumericalData]
                   = None, policy_grad_clip: Optional[float] = None) -> None:
        """
        Performs a single boosting step for the actor (should only be used
        if actor and critic use separate models)

        Args:
            observations (NumericalData):
            policy_grad_clip (float, optional): Defaults to None.
            policy_grad (Optional[NumericalData], optional): manually calculated gradients. Defaults to None.

        Returns:
            np.ndarray: policy gradient
        """
        assert not self.shared_tree_struct, "Cannot separate boosting steps"
        "for actor and critic when using separate tree architectures!"
        if observations is None:
            assert self.input is not None, ("Cannot update trees without input."
                                            "Make sure model is called with requires_grad=True")
            observations = self.input
        n_samples = len(observations)
        policy_grad = policy_grad if policy_grad is not None else self.params[0].grad.detach() * n_samples
        policy_grad = clip_grad_norm(policy_grad, policy_grad_clip)
        validate_array(policy_grad)

        self.learner.step_actor(observations, policy_grad)
        self.policy_grad = policy_grad

    def critic_step(self, observations: Optional[NumericalData] = None,
                    value_grad: Optional[NumericalData] = None,
                    value_grad_clip: Optional[float] = None) -> None:
        """
        Performs a single boosting step for the critic (should only be used
        if actor and critic use separate models)

        Args:
            observations (NumericalData):
            value_grad_clip (float, optional): Defaults to None.
            value_grad (Optional[NumericalData], optional): manually calculated gradients. Defaults to None.

        Returns:
            np.ndarray: value gradient
        """
        assert not self.shared_tree_struct, ("Cannot separate boosting steps"
                                             "for actor and critic when using separate tree architectures!")
        if observations is None:
            assert self.input is not None, ("Cannot update trees without input."
                                            "Make sure model is called with requires_grad=True")
            observations = self.input
        n_samples = len(observations)

        value_grad = value_grad if value_grad is not None else self.params[1].grad.detach() * n_samples
        value_grad = clip_grad_norm(value_grad, value_grad_clip)

        validate_array(value_grad)
        self.learner.step_critic(observations, value_grad)
        self.value_grad = value_grad

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the predicted actor and critic parameters along with their gradients.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Predicted actor and critic outputs.
                - Corresponding policy and value gradients.
        """
        assert self.params is not None, "must run a forward pass first"
        if isinstance(self.params, tuple):
            params = (self.params[0].detach().cpu().numpy(), self.params[1].detach().cpu().numpy())
        else:
            params = self.params
        return params, (self.policy_grad, self.value_grad)

    def save_learner(self, save_path: str) -> None:
        """
        Saves model to file

        Args:
            filename (str): Absolute path and name of save filename.
        """
        if self.shared_tree_struct:
            self.learner.save(save_path)
        else:
            self.learner.save(save_path, custom_names=['policy', 'value'])

    def copy(self) -> "ActorCritic":
        """
        Copy class instance

        Returns:
            ActorCritic: copy of current instance
        """
        return self.__copy__()

    def __copy__(self) -> "ActorCritic":
        learner = self.learner.copy()
        copy_ = ActorCritic(learner.tree_struct, learner.input_dim,
                            learner.output_dim, learner.optimizers[0],
                            learner.optimizers[1], self.shared_tree_struct,
                            learner.params, learner.get_bias(),
                            learner.verbose, learner.device)
        copy_.learner = learner
        return copy_
