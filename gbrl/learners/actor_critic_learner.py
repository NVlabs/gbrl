##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch as th

from gbrl import GBRL_CPP
from gbrl.common.utils import (NumericalData, concatenate_arrays,
                               ensure_leaf_tensor_or_array, ensure_same_type,
                               get_tensor_info, numerical_dtype,
                               preprocess_features)
from gbrl.learners.gbt_learner import GBTLearner
from gbrl.learners.multi_gbt_learner import MultiGBTLearner


class SharedActorCriticLearner(GBTLearner):
    """
    SharedActorCriticLearner is a variant of GBTLearner where a single tree is
    used for both
    actor (policy) and critic (value) learning. It utilizes gradient boosting
    trees (GBTs)
    to estimate both policy and value function parameters efficiently.
    """
    def __init__(self, input_dim: int, output_dim: int, tree_struct: Dict,
                 policy_optimizer: Dict, value_optimizer: Dict,
                 params: Dict = dict(), verbose: int = 0, device: str = 'cpu'):
        """
        Initializes the SharedActorCriticLearner.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output dimensions.
            tree_struct (Dict): Dictionary containing tree structure
            parameters.
            policy_optimizer (Dict): Dictionary with optimization parameters
            for the policy.
            value_optimizer (Dict): Dictionary with optimization parameters
            for the critic.
            params (Dict, optional): Additional model parameters. Defaults to
            an empty dictionary.
            verbose (int, optional): Verbosity level. Defaults to 0.
            device (str, optional): Device to run the model on. Defaults to
            'cpu'.
        """
        if verbose > 0:
            print('****************************************')
            print(f'Shared GBRL Tree with input dim: {input_dim},'
                  f'output dim: {output_dim}, tree_struct: {tree_struct}'
                  f'policy_optimizer: {policy_optimizer}'
                  f'value_optimizer: {value_optimizer}')
            print('****************************************')
        super().__init__(input_dim, output_dim, tree_struct,
                         [policy_optimizer, value_optimizer],
                         params, verbose, device)

    def step(self, obs: NumericalData,
             theta_grad: np.ndarray, value_grad: np.ndarray) -> None:
        """
        Performs a gradient update step for both policy and value function.

        Args:
            obs (NumericalData): Input observations.
            theta_grad (np.ndarray): Gradient of the policy parameters.
            value_grad (np.ndarray): Gradient of the value function parameters.
        """
        grads = concatenate_arrays(theta_grad, value_grad)
        obs, grads = ensure_same_type(obs, grads)
        if isinstance(obs, th.Tensor):
            obs = obs.float()
            grads = grads.float()
            # store data so that data isn't garbage collected
            # while GBRL uses it
            self._save_memory = (obs, grads)
            self._cpp_model.step(get_tensor_info(obs),
                                 None, get_tensor_info(grads))
            self._save_memory = None
        else:
            num_obs, cat_obs = preprocess_features(obs)
            grads = np.ascontiguousarray(grads).astype(numerical_dtype)
            input_dim = 0 if num_obs is None else num_obs.shape[1]
            input_dim += 0 if cat_obs is None else cat_obs.shape[1]
            self._cpp_model.step(num_obs, cat_obs, grads)

        self.iteration = self._cpp_model.get_iteration()
        self.total_iterations += 1

    def distil(self, obs: NumericalData,
               policy_targets: np.ndarray, value_targets: np.ndarray,
               params: Dict, verbose: int) -> Tuple[float, Dict]:
        """
        Distills the trained model into a student model.

        Args:
            obs (NumericalData): Input observations.
            policy_targets (np.ndarray): Target values for the policy (actor).
            value_targets (np.ndarray): Target values for the value function
            (critic).
            params (Dict): Distillation parameters.
            verbose (int): Verbosity level.

        Returns:
            Tuple[float, Dict]: The final loss value and updated parameters
            for distillation
        """
        targets = np.concatenate([policy_targets,
                                  value_targets[:, np.newaxis]], axis=1)
        return super().distil(obs, targets, params, verbose)

    def predict(self, obs: NumericalData,
                requires_grad: bool = True, start_idx: int = 0,
                stop_idx: int = None, tensor: bool = True) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Predicts both policy and value function outputs.

        Args:
            obs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients.
            Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to
            0.
            stop_idx (int, optional): Stop index for prediction. Defaults to
            None.
            tensor (bool, optional): Whether to return a tensor. Defaults to
            True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted policy and value outputs.
        """
        preds = super().predict(obs, requires_grad, start_idx, stop_idx,
                                tensor)
        pred_values = preds[:, -1]
        preds = preds[:, :-1]
        preds = ensure_leaf_tensor_or_array(preds, tensor, requires_grad, self.device)
        pred_values = ensure_leaf_tensor_or_array(pred_values, tensor, requires_grad, self.device)
        return preds, pred_values

    def predict_policy(self, obs: NumericalData,
                       requires_grad: bool = True, start_idx: int = 0,
                       stop_idx: int = None, tensor: bool = True):
        """
        Predicts the policy (actor) output for the given observations.

        Args:
            obs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients.
            Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to
            0.
            stop_idx (int, optional): Stop index for prediction. Defaults to
            None.
            tensor (bool, optional): Whether to return a tensor. Defaults to
            True.

        Returns:
            np.ndarray: Predicted policy outputs.
        """
        preds, _ = self.predict(obs, requires_grad, start_idx, stop_idx,
                                tensor)
        return preds

    def predict_critic(self, obs: NumericalData,
                       requires_grad: bool = True, start_idx: int = 0,
                       stop_idx: int = None, tensor: bool = True):
        """
        Predicts the value function (critic) output for the given observations.

        Args:
            obs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.

        Returns:
            np.ndarray: Predicted value function outputs.
        """
        _, pred_values = self.predict(obs, requires_grad, start_idx, stop_idx,
                                      tensor)
        return pred_values

    def __copy__(self) -> "SharedActorCriticLearner":
        """
        Creates a copy of the SharedActorCriticLearner instance.

        Returns:
            SharedActorCriticLearner: A copy of the current instance.
        """
        copy_ = SharedActorCriticLearner(self.input_dim, self.output_dim,
                                         self.tree_struct.copy(),
                                         self.optimizers[0].copy(),
                                         self.optimizers[1].copy(),
                                         self.params, self.verbose,
                                         self.device)
        copy_.iteration = self.iteration
        copy_.total_iterations = self.total_iterations
        if self._cpp_model is not None:
            copy_._cpp_model = GBRL_CPP(self._cpp_model)
        if self.student_model is not None:
            copy_.student_model = GBRL_CPP(self.student_model)
        return copy_


class SeparateActorCriticLearner(MultiGBTLearner):
    """
    Implements a separate actor-critic learner using two independent gradient
    boosted trees.

    This class extends MultiGBTLearner by maintaining two separate models:
    - One for policy learning (Actor).
    - One for value estimation (Critic).

    It provides separate `step_actor` and `step_critic` methods for updating
    the respective models.
    """
    def __init__(self, input_dim: int, output_dim: int, tree_struct: Dict,
                 policy_optimizer: Dict, value_optimizer: Dict,
                 params: Dict = dict(), verbose: int = 0, device: str = 'cpu'):
        """
        Initializes the SeparateActorCriticLearner with two independent GBT
        models.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output dimensions.
            tree_struct (Dict): Dictionary containing tree structure parameters.
            policy_optimizer (Dict): Optimizer configuration for the policy (actor).
            value_optimizer (Dict): Optimizer configuration for the value function (critic).
            params (Dict, optional): Additional model parameters. Defaults to an empty dictionary.
            verbose (int, optional): Verbosity level for debugging. Defaults to 0.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        if verbose > 0:
            print('****************************************')
            print(f'Separate GBRL Tree with input dim: {input_dim},'
                  f'output dim: {output_dim}, tree_struct: {tree_struct}'
                  f'policy_optimizer: {policy_optimizer}'
                  f'value_optimizer: {value_optimizer}')
            print('****************************************')
        super().__init__(input_dim, [output_dim - 1, 1], tree_struct,
                         [policy_optimizer, value_optimizer],
                         params, 2, verbose, device)

    def step(self, obs: NumericalData,
             theta_grad: NumericalData, value_grad: NumericalData,
             model_idx: Optional[int] = None) -> None:
        """
        Performs a single gradient update step on both the policy and value
        models.

        Args:
            obs (NumericalData): Input observations.
            theta_grad (NumericalData): Gradient update for the policy (actor).
            value_grad (NumericalData): Gradient update for the value function (critic).
            model_idx (Optional[int], optional): Index of the model to update.
            If None, updates both models.
        """
        super().step(obs, [theta_grad, value_grad], model_idx=model_idx)

    def step_actor(self, obs: NumericalData,
                   theta_grad: NumericalData) -> None:
        """
        Performs a gradient update step for the policy (actor) model.

        Args:
            obs (NumericalData): Input observations.
            theta_grad (NumericalData): Gradient update for the policy (actor).
        """
        super().step(obs, theta_grad, model_idx=0)

    def step_critic(self, obs: NumericalData,
                    value_grad: NumericalData) -> None:
        """
        Performs a gradient update step for the value function (critic) model.

        Args:
            obs (NumericalData): Input observations.
            value_grad (NumericalData): Gradient update for the value function (critic).
        """
        super().step(obs, value_grad, model_idx=1)

    def distil(self, obs: NumericalData,
               policy_targets: np.ndarray, value_targets: np.ndarray,
               params: Dict, verbose: int) -> Tuple[List[float], List[Dict]]:
        """
        Distills the trained model into a student model.

        Args:
            obs (NumericalData): Input observations.
            policy_targets (np.ndarray): Target values for the policy (actor).
            value_targets (np.ndarray): Target values for the value function (critic).
            params (Dict): Distillation parameters.
            verbose (int): Verbosity level.

        Returns:
            Tuple[List[float], List[Dict]]: The final loss values and updated parameters for distillation.
        """
        return super().distil(obs, [policy_targets, value_targets], params,
                              verbose)

    def predict(self, obs: NumericalData,
                requires_grad: bool = True, start_idx: int = 0,
                stop_idx: int = None, tensor: bool = True) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Predicts both the policy and value outputs for the given observations.

        Args:
            obs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted policy outputs and value function outputs.
        """
        return super().predict(obs, requires_grad, start_idx, stop_idx, tensor)

    def predict_policy(self, obs: NumericalData,
                       requires_grad: bool = True, start_idx: int = 0,
                       stop_idx: int = None, tensor: bool = True) -> NumericalData:
        """
        Predicts the policy (actor) output for the given observations.

        Args:
            obs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.

        Returns:
            NumericalData: Predicted policy outputs.
        """
        return super().predict(obs, requires_grad, start_idx, stop_idx, tensor,
                               model_idx=0)

    def predict_critic(self, obs: NumericalData,
                       requires_grad: bool = True, start_idx: int = 0,
                       stop_idx: int = None, tensor: bool = True) -> NumericalData:
        """
        Predicts the value function (critic) output for the given observations.

        Args:
            obs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.

        Returns:
            NumericalData: Predicted value function outputs.
        """
        return super().predict(obs, requires_grad, start_idx, stop_idx, tensor,
                               model_idx=1)

    def __copy__(self) -> "SeparateActorCriticLearner":
        """
        Creates a copy of the SeparateActorCriticLearner instance.

        Returns:
            SeparateActorCriticLearner: A new instance with the same parameters and structure.
        """
        opts = [opt.copy() if opt is not None else opt
                for opt in self.optimizers
                ]
        copy_ = SeparateActorCriticLearner(self.input_dim, self.output_dim,
                                           self.tree_struct.copy(),
                                           opts, self.params,
                                           self.n_learners,
                                           self.verbose,
                                           self.device)
        copy_.iteration = self.iteration
        copy_.total_iterations = self.total_iterations
        if self._cpp_models is not None:
            for i in range(self.n_learners):
                copy_._cpp_models[i] = GBRL_CPP(self._cpp_models[i])
        if self.student_models is not None:
            copy_.student_models[i] = GBRL_CPP(self.student_models[i])
        return copy_
