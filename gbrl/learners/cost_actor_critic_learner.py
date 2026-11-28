##############################################################################
# Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
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
from typing import Dict, List, Optional, Tuple

import numpy as np

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
                            policy_dim=output_dim - 2,
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
        # Scalar critics: accept (n,) or (n, 1) targets.
        targets = np.concatenate([policy_targets,
                                  np.reshape(value_targets, (len(value_targets), -1)),
                                  np.reshape(cost_targets, (len(cost_targets), -1))], axis=1)
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

    def __copy__(self):
        return GBTLearner.__copy__(self)


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
        self.n_objs_per_model = [self.n_objs, 1, 1]

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

    def __copy__(self):
        return MultiGBTLearner.__copy__(self)
