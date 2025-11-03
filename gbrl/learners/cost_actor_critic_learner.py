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

from gbrl import GBRL_CPP
from gbrl.common.utils import NumericalData, ensure_leaf_tensor_or_array
from gbrl.learners.gbt_learner import GBTLearner
from gbrl.learners.actor_critic_learner import SharedActorCriticLearner, SeparateActorCriticLearner
from gbrl.learners.multi_gbt_learner import MultiGBTLearner


class SharedCostActorCriticLearner(SharedActorCriticLearner):
    """
    SharedCostActorCriticLearner is a variant of GBTLearner where a single tree is
    used for the actor (policy), critic (value), and cost critic learning.
    It utilizes gradient boosting trees (GBTs) to estimate policy, value function, and cost function parameters efficiently.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 tree_struct: Dict,
                 policy_optimizer: Dict,
                 value_optimizer: Dict,
                 cost_optimizer: Dict,
                 params: Dict = dict(),
                 verbose: int = 0,
                 device: str = 'cpu'):
        """
        Initializes the SharedCostActorCriticLearner.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output dimensions.
            tree_struct (Dict): Dictionary containing tree structure
            parameters.
            policy_optimizer (Dict): Dictionary with optimization parameters
            for the policy.
            value_optimizer (Dict): Dictionary with optimization parameters
            for the critic.
            cost_optimizer (Dict): Dictionary with optimization parameters
            for the cost critic.
            params (Dict, optional): Additional model parameters. Defaults to
            an empty dictionary.
            verbose (int, optional): Verbosity level. Defaults to 0.
            device (str, optional): Device to run the model on. Defaults to
            'cpu'.
        """
        if verbose > 0:
            print('****************************************')
            print(f'Shared GBRL Tree with input dim: {input_dim}, '
                  f'output dim: {output_dim}, tree_struct: {tree_struct}, '
                  f'policy_optimizer: {policy_optimizer}, '
                  f'value_optimizer: {value_optimizer}, '
                  f'cost_optimizer: {cost_optimizer}')
            print('****************************************')
        GBTLearner.__init__(self, input_dim, output_dim,
                            tree_struct=tree_struct,
                            optimizers=[policy_optimizer, value_optimizer, cost_optimizer],
                            params=params,
                            policy_dim=output_dim - 1,
                            verbose=verbose,
                            device=device)

    def distil(self, obs: np.ndarray,  # type: ignore
               policy_targets: np.ndarray,
               value_targets: np.ndarray,
               cost_targets: np.ndarray,
               params: Dict,
               verbose: int = 0) -> Tuple[float, Dict]:
        """
        Distills the trained model into a student model.

        Args:
            obs (np.ndarray): Input observations.
            policy_targets (np.ndarray): Target values for the policy (actor).
            value_targets (np.ndarray): Target values for the value function (critic).
            cost_targets (np.ndarray): Target values for the cost value function (critic).
            params (Dict): Distillation parameters.
            verbose (int): Verbosity level.

        Returns:
            Tuple[float, Dict]: The final loss value and updated parameters
            for distillation
        """
        targets = np.concatenate([policy_targets,
                                  value_targets,
                                  cost_targets[:, np.newaxis]], axis=1)
        return GBTLearner.distil(self, obs, targets, params, verbose)

    def predict(self,  # type: ignore
                inputs: NumericalData,
                requires_grad: bool = True, start_idx: Optional[int] = 0,
                stop_idx: Optional[int] = None, tensor: bool = True) -> \
            Tuple[NumericalData, ...]:
        """
        Predicts both policy and value function outputs.

        Args:
            inputs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients.
            Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to
            0.
            stop_idx (int, optional): Stop index for prediction. Defaults to
            None.
            tensor (bool, optional): Whether to return a tensor. Defaults to
            True.

        Returns:
            Tuple[NumericalData, ...]: Predicted policy and value outputs.
        """
        preds = GBTLearner.predict(self, inputs, requires_grad, start_idx, stop_idx, tensor)
        pred_values = preds[:, -2]
        pred_costs = preds[:, -1]
        preds = preds[:, :-2]
        preds = ensure_leaf_tensor_or_array(preds, tensor, requires_grad, self.device)
        pred_values = ensure_leaf_tensor_or_array(pred_values, tensor, requires_grad, self.device)
        pred_costs = ensure_leaf_tensor_or_array(pred_costs, tensor, requires_grad, self.device)
        return preds, pred_values, pred_costs

    def predict_policy(self, obs: NumericalData,
                       requires_grad: bool = True, start_idx: Optional[int] = 0,
                       stop_idx: Optional[int] = None, tensor: bool = True) -> NumericalData:
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
            NumericalData: Predicted policy outputs.
        """
        preds, _, _ = self.predict(obs, requires_grad, start_idx, stop_idx,
                                   tensor)
        return preds

    def predict_critic(self, obs: NumericalData,
                       requires_grad: bool = True, start_idx: Optional[int] = 0,
                       stop_idx: Optional[int] = None, tensor: bool = True) -> NumericalData:
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
        _, pred_values, _ = self.predict(obs, requires_grad, start_idx, stop_idx,
                                         tensor)
        return pred_values

    def predict_cost(self, obs: NumericalData,
                     requires_grad: bool = True, start_idx: Optional[int] = 0,
                     stop_idx: Optional[int] = None, tensor: bool = True) -> NumericalData:
        """
        Predicts the cost value function output for the given observations.

        Args:
            obs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.

        Returns:
            NumericalData: Predicted cost value function outputs.
        """
        _, _, pred_costs = self.predict(obs, requires_grad, start_idx, stop_idx,
                                        tensor)
        return pred_costs

    def __copy__(self) -> "SharedCostActorCriticLearner":
        """
        Creates a copy of the SharedCostActorCriticLearner instance.

        Returns:
            SharedCostActorCriticLearner: A copy of the current instance.
        """
        assert isinstance(self.input_dim, int), "input_dim should be an integer"
        assert isinstance(self.output_dim, int), "output_dim should be an integer"
        copy_ = SharedCostActorCriticLearner(input_dim=self.input_dim,
                                             output_dim=self.output_dim,
                                             tree_struct=self.tree_struct.copy(),
                                             policy_optimizer=self.optimizers[0].copy(),
                                             value_optimizer=self.optimizers[1].copy(),
                                             cost_optimizer=self.optimizers[2].copy(),
                                             params=self.params,
                                             verbose=self.verbose,
                                             device=self.device)
        copy_.iteration = self.iteration
        copy_.total_iterations = self.total_iterations
        if self._cpp_model is not None:
            copy_._cpp_model = GBRL_CPP(self._cpp_model)
        if self.student_model is not None:
            copy_.student_model = GBRL_CPP(self.student_model)
        return copy_


class SeparateCostActorCriticLearner(SeparateActorCriticLearner):
    """
    Implements a separate cost-actor-critic learner using three independent gradient
    boosted trees.

    This class extends MultiGBTLearner by maintaining three separate models:
    - One for policy learning (Actor).
    - One for value estimation (Critic).
    - One for cost value estimation (Critic).

    It provides separate `step_actor`, `step_critic`, and `step_cost` methods for updating
    the respective models.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 tree_struct: Dict,
                 policy_optimizer: Dict,
                 value_optimizer: Dict,
                 cost_optimizer: Dict,
                 params: Dict = dict(),
                 verbose: int = 0,
                 device: str = 'cpu'):
        """
        Initializes the SeparateCostActorCriticLearner with two independent GBT
        models.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output dimensions.
            tree_struct (Dict): Dictionary containing tree structure parameters.
            policy_optimizer (Dict): Optimizer configuration for the policy (actor).
            value_optimizer (Dict): Optimizer configuration for the value function (critic).
            cost_optimizer (Dict): Optimizer configuration for the cost value function (critic).
            params (Dict, optional): Additional model parameters. Defaults to an empty dictionary.
            verbose (int, optional): Verbosity level for debugging. Defaults to 0.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        if verbose > 0:
            print('****************************************')
            print(f'Separate GBRL Tree with input dim: {input_dim}, '
                  f'output dim: {output_dim}, tree_struct: {tree_struct}, '
                  f'policy_optimizer: {policy_optimizer}, '
                  f'value_optimizer: {value_optimizer}, '
                  f'cost_optimizer: {cost_optimizer}')
            print('****************************************')
        MultiGBTLearner.__init__(self,
                                 input_dim,
                                 output_dim=[output_dim - 2, 1, 1],
                                 tree_struct=tree_struct,
                                 optimizers=[policy_optimizer,
                                             value_optimizer,
                                             cost_optimizer],
                                 params=params,
                                 n_learners=3,
                                 policy_dim=[output_dim - 2, 0, 0],
                                 verbose=verbose, device=device)

    def step_cost(self, inputs: NumericalData,
                  grads: NumericalData) -> None:
        """
        Performs a gradient update step for the cost value function (critic) model.

        Args:
            obs (NumericalData): Input observations.
            value_grad (NumericalData): Gradient update for the cost value function (critic).
        """
        super().step(inputs=inputs, grads=grads, model_idx=2)

    def distil(self, obs: NumericalData,  # type: ignore
               policy_targets: np.ndarray,
               value_targets: np.ndarray,
               cost_targets: np.ndarray,
               params: Dict,
               verbose: int = 0) -> Tuple[List[float], List[Dict]]:
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
        return MultiGBTLearner.distil(self, obs, [policy_targets, value_targets, cost_targets], params, verbose)

    def predict_cost(self,
                     obs: NumericalData,
                     requires_grad: bool = True,
                     start_idx: Optional[int] = 0,
                     stop_idx: Optional[int] = None,
                     tensor: bool = True) -> NumericalData:
        """
        Predicts the cost value function (critic) output for the given observations.

        Args:
            obs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.

        Returns:
            NumericalData: Predicted cost value function outputs.
        """
        return super().predict(obs, requires_grad, start_idx, stop_idx, tensor, model_idx=2)  # type: ignore

    def __copy__(self) -> "SeparateCostActorCriticLearner":
        """
        Creates a copy of the SeparateCostActorCriticLearner instance.

        Returns:
            SeparateCostActorCriticLearner: A new instance with the same parameters and structure.
        """
        opts = [opt.copy() if opt is not None else opt
                for opt in self.optimizers
                ]
        assert isinstance(self.input_dim, int), "input_dim should be an integer"
        assert isinstance(self.output_dim, int), "output_dim should be an integer"

        copy_ = SeparateCostActorCriticLearner(input_dim=self.input_dim,
                                               output_dim=self.output_dim,
                                               tree_struct=self.tree_struct.copy(),
                                               policy_optimizer=opts[0],
                                               value_optimizer=opts[1],
                                               cost_optimizer=opts[2],
                                               params=self.params,
                                               verbose=self.verbose,
                                               device=self.device)
        copy_.iteration = self.iteration
        copy_.total_iterations = self.total_iterations

        if self.student_models is not None:
            copy_.student_models = [None] * self.n_learners
        if self._cpp_models is not None:
            copy_._cpp_models = [None] * self.n_learners
        if self._cpp_models is not None:
            for i in range(self.n_learners):
                copy_._cpp_models[i] = GBRL_CPP(self._cpp_models[i])  # type: ignore
                if self.student_models is not None:
                    copy_.student_models[i] = GBRL_CPP(self.student_models[i])  # type: ignore
        return copy_
