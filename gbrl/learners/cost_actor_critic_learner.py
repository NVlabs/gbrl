##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th

from gbrl import GBRL_CPP
from gbrl.common.compression import SharedActorCriticCompression, TreeCompression
from gbrl.common.constraints import Constraint
from gbrl.common.utils import (NumericalData, concatenate_arrays,
                               ensure_leaf_tensor_or_array, ensure_same_type,
                               get_index_mapping, get_tensor_info, numerical_dtype,
                               preprocess_features)
from gbrl.learners.gbt_learner import GBTLearner
from gbrl.learners.multi_gbt_learner import MultiGBTLearner


class SharedCostActorCriticLearner(GBTLearner):
    """
    SharedCostActorCriticLearner is a variant of GBTLearner where a single tree is
    used for the
    actor (policy), reward critic (value), and cost critic learning. It utilizes gradient boosting
    trees (GBTs)
    to estimate the policy and both value functions parameters efficiently.
    """
    def __init__(self, input_dim: int, output_dim: int, tree_struct: Dict,
                 policy_optimizer: Dict, value_optimizer: Dict, cost_optimizer: Dict,
                 params: Dict = dict(), verbose: int = 0, device: str = 'cpu',
                 constraints: Optional[Union[Constraint, List[Dict]]] = None):
        """
        Initializes the SharedActorCriticLearner.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output dimensions.
            tree_struct (Dict): Dictionary containing tree structure parameters.
            policy_optimizer (Dict): Dictionary with optimization parameters for the policy.
            value_optimizer (Dict): Dictionary with optimization parameters for the reward critic.
            cost_optimizer (Dict): Dictionary with optimization parameters for the cost critic.
            params (Dict, optional): Additional model parameters. Defaults to an empty dictionary.
            verbose (int, optional): Verbosity level. Defaults to 0.
            device (str, optional): Device to run the model on. Defaults to 'cpu'.
            constraints (Union[Constraint, List[Dict], optional): feature constraints. Defaults to None.
        """
        if verbose > 0:
            print('****************************************')
            print(f'Shared GBRL Tree with input dim: {input_dim}, '
                  f'output dim: {output_dim}, tree_struct: {tree_struct} '
                  f'policy_optimizer: {policy_optimizer} '
                  f'value_optimizer: {value_optimizer} '
                  f'cost_optimizer: {cost_optimizer}'
                  )
            print('****************************************')
        if cost_optimizer['stop_idx'] != output_dim:
            output_dim = cost_optimizer['stop_idx']
        super().__init__(input_dim, output_dim, tree_struct,
                         [policy_optimizer, value_optimizer, cost_optimizer],
                         params, verbose, device,
                         constraints)

    def step(self, obs: NumericalData,
             theta_grad: np.ndarray, value_grad: np.ndarray,
             cost_grad: np.ndarray) -> None:
        """
        Performs a gradient update step for both policy and value function.

        Args:
            obs (NumericalData): Input observations.
            theta_grad (np.ndarray): Gradient of the policy parameters.
            value_grad (np.ndarray): Gradient of the value function parameters.
            cost_grad (np.ndarray): Gradient of the cost value function parameters.
        """
        grads = concatenate_arrays(theta_grad, value_grad)
        grads = concatenate_arrays(grads, cost_grad)
        obs, grads = ensure_same_type(obs, grads)
        if self.total_iterations == 0:
            mapping, is_numeric = get_index_mapping(obs)
            self.mapping = (mapping, is_numeric)
            self._cpp_model.set_feature_mapping(np.ascontiguousarray(mapping),
                                                np.ascontiguousarray(is_numeric))
            self._set_constraints()

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
               cost_targets: np.ndarray,
               params: Dict, verbose: int) -> Tuple[float, Dict]:
        """
        Distills the trained model into a student model.

        Args:
            obs (NumericalData): Input observations.
            policy_targets (np.ndarray): Target values for the policy (actor).
            value_targets (np.ndarray): Target values for the value function (critic).
            cost_targets (np.ndarray): Target values for the cost value function (cost critic).
            params (Dict): Distillation parameters.
            verbose (int): Verbosity level.

        Returns:
            Tuple[float, Dict]: The final loss value and updated parameters
            for distillation
        """
        targets = np.concatenate([policy_targets,
                                  value_targets[:, np.newaxis],
                                  cost_targets[:, np.newaxis]], axis=1)
        return super().distil(obs, targets, params, verbose)

    def predict(self, obs: NumericalData,
                requires_grad: bool = True, start_idx: int = 0,
                stop_idx: int = None, tensor: bool = True) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Predicted policy and critic outputs.
        """
        preds = super().predict(obs, requires_grad, start_idx, stop_idx, tensor)
        preds_policy = ensure_leaf_tensor_or_array(preds[:, :-2], tensor, requires_grad, self.device)
        pred_values = ensure_leaf_tensor_or_array(preds[:, -2], tensor, requires_grad, self.device)
        pred_costs = ensure_leaf_tensor_or_array(preds[:, -1], tensor, requires_grad, self.device)
        return preds_policy, pred_values, pred_costs

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
        preds, _, _ = self.predict(obs, requires_grad, start_idx, stop_idx, tensor)
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
        _, pred_values, _ = self.predict(obs, requires_grad, start_idx, stop_idx,
                                         tensor)
        return pred_values

    def predict_cost(self, obs: NumericalData,
                     requires_grad: bool = True, start_idx: int = 0,
                     stop_idx: int = None, tensor: bool = True):
        """
        Predicts the cost value function (cost critic) output for the given observations.

        Args:
            obs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.

        Returns:
            np.ndarray: Predicted value function outputs.
        """
        _, _, pred_costs = self.predict(obs, requires_grad, start_idx, stop_idx,
                                        tensor)
        return pred_costs

    def compress(self, trees_to_keep: int, gradient_steps: int, features: NumericalData,
                 actions: th.Tensor = None, log_std: th.Tensor = None,
                 method: str = 'first_k', dist_type: str = 'deterministic',
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 temperature: float = 1.0, lambda_reg: float = 1.0, **kwargs):
        """
        Compresses the tree ensemble by selecting and retraining a subset of trees.

        Args:
            trees_to_keep (int): Number of trees to retain in the compressed model.
            gradient_steps (int): Number of optimization steps during compression.
            features (NumericalData): Input feature matrix (n_samples, n_features).
            actions (th.Tensor, optional): Target actions (for policy compression). Required if dist_type
                is not 'supervised_learning'.
            log_std (th.Tensor, optional): Log standard deviation (only used for certain policy types).
            method (str): Tree selection method. Defaults to 'first_k'.
            dist_type (str): Compression type ('supervised_learning', 'actor', etc.).
            optimizer_kwargs (dict, optional): Optimizer configuration.
            temperature (float): Temperature parameter for soft selection.
            lambda_reg (float): L2 regularization coefficient on weights.
            **kwargs: Additional keyword arguments passed to the compressor.

        Returns:
            float: Final loss value after compression.
        """
        assert actions is not None, "Cannot compress a shared actor-critic policy without actions"
        assert dist_type != 'supervised_learning', \
            "Cannot compress a shared actor-critic policy using supervised learning compression methods"
        A, V, n_leaves_per_tree, n_leaves, n_trees = self.get_matrix_representation(features)
        A, V = th.tensor(A, dtype=th.float32, device=self.device), th.tensor(V, dtype=th.float32,
                                                                             device=self.device)
        n_leaves_per_tree = th.tensor(n_leaves_per_tree, device=self.device)
        k = self.get_num_trees() - trees_to_keep
        compression_params = {'k': k, 'gradient_steps': gradient_steps,
                              'method': method,
                              'optimizer_kwargs': optimizer_kwargs,
                              'temperature': temperature, 'n_leaves': n_leaves, 'n_trees': n_trees,
                              'n_leaves_per_tree': n_leaves_per_tree,
                              'lambda_reg': lambda_reg,
                              'output_dim': self.output_dim,
                              'device': self.device}
        if 'policy_only' in kwargs:
            del kwargs['policy_only']
        compression_params.update(kwargs)

        compression_params['dist_type'] = dist_type
        if dist_type == 'deterministic':
            compressor = TreeCompression(**compression_params)
            parameters, losses = compressor.compress(A, V)
        else:
            compressor = SharedActorCriticCompression(**compression_params)
            parameters, losses = compressor.compress(A, V, actions, log_std)
        leaves_selection, tree_selection, W, n_compressed_trees, n_compressed_leaves = parameters
        # indices of selected leaves / trees in original indexing
        compressed_leaf_indices = np.where(leaves_selection > 0)[0].astype(np.int32)
        compressed_tree_indices = np.where(tree_selection > 0)[0].astype(np.int32)
        # indices of the start of each leaf according to the compressed model
        new_tree_indices = np.zeros(n_compressed_trees)
        new_tree_indices[1:] = np.cumsum(n_leaves_per_tree[compressed_tree_indices].detach().cpu().numpy())[:-1]
        self._cpp_model.compress(n_compressed_leaves, n_compressed_trees, compressed_leaf_indices,
                                 compressed_tree_indices, new_tree_indices.astype(np.int32), W)
        print(f"Finished compressing - compressed model has {self.get_num_trees()} trees")
        return losses[-1]

    def __copy__(self) -> "SharedCostActorCriticLearner":
        """
        Creates a copy of the SharedCostActorCriticLearner instance.

        Returns:
            SharedCostActorCriticLearner: A copy of the current instance.
        """
        copy_ = SharedCostActorCriticLearner(self.input_dim, self.output_dim,
                                             self.tree_struct.copy(),
                                             self.optimizers[0].copy(),
                                             self.optimizers[1].copy(),
                                             self.optimizers[2].copy(),
                                             self.params, self.verbose,
                                             self.device)
        copy_.iteration = self.iteration
        copy_.total_iterations = self.total_iterations
        if self._cpp_model is not None:
            copy_._cpp_model = GBRL_CPP(self._cpp_model)
        if self.student_model is not None:
            copy_.student_model = GBRL_CPP(self.student_model)
        return copy_


class SeparateCostActorCriticLearner(MultiGBTLearner):
    """
    Implements a separate cost actor-critic learner using three independent gradient
    boosted trees.

    This class extends MultiGBTLearner by maintaining two separate models:
    - One for policy learning (Actor).
    - One for value estimation (Critic).
    - One for cost value estimation (Cost Critic).

    It provides separate `step_actor` `step_critic` `step_cost` methods for updating
    the respective models.
    """
    def __init__(self, input_dim: int, output_dim: int, tree_struct: Dict,
                 policy_optimizer: Dict, value_optimizer: Dict, cost_optimizer: Dict,
                 params: Dict = dict(), verbose: int = 0, device: str = 'cpu',
                 constraints: Optional[Union[Constraint, List[Dict]]] = None):
        """
        Initializes the SeparateActorCriticLearner with two independent GBT
        models.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output dimensions.
            tree_struct (Dict): Dictionary containing tree structure parameters.
            policy_optimizer (Dict): Optimizer configuration for the policy (actor).
            value_optimizer (Dict): Optimizer configuration for the value function (critic).
            cost_optimizer (Dict): Optimizer configuration for the cost value function (cost critic).
            params (Dict, optional): Additional model parameters. Defaults to an empty dictionary.
            verbose (int, optional): Verbosity level for debugging. Defaults to 0.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
            constraints (Union[Constraint, List[Dict], optional): feature constraints. Defaults to None.
        """
        if verbose > 0:
            print('****************************************')
            print(f'Separate GBRL Tree with input dim: {input_dim}, '
                  f'output dim: {output_dim}, tree_struct: {tree_struct} '
                  f'policy_optimizer: {policy_optimizer} '
                  f'value_optimizer: {value_optimizer} '
                  f'cost_optimizer: {cost_optimizer}'
                  )
            print('****************************************')
        super().__init__(input_dim, [output_dim - 1, 1], tree_struct,
                         [policy_optimizer, value_optimizer, cost_optimizer],
                         params, 3, verbose, device,
                         constraints)

    def step(self, obs: NumericalData,
             theta_grad: NumericalData, value_grad: NumericalData,
             cost_grad: NumericalData,
             model_idx: Optional[int] = None) -> None:
        """
        Performs a single gradient update step on both the policy and value
        models.

        Args:
            obs (NumericalData): Input observations.
            theta_grad (NumericalData): Gradient update for the policy (actor).
            value_grad (NumericalData): Gradient update for the value function (critic).
            cost_grad (NumericalData): Gradient update for the cost value function (cost critic).
            model_idx (Optional[int], optional): Index of the model to update.
            If None, updates both models.
        """
        super().step(obs, [theta_grad, value_grad, cost_grad], model_idx=model_idx)

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

    def step_cost(self, obs: NumericalData,
                  cost_grad: NumericalData) -> None:
        """
        Performs a gradient update step for the cost value function (cost critic) model.

        Args:
            obs (NumericalData): Input observations.
            cost_grad (NumericalData).
        """
        super().step(obs, cost_grad, model_idx=2)

    def distil(self, obs: NumericalData,
               policy_targets: np.ndarray, value_targets: np.ndarray,
               cost_targets: np.ndarray,
               params: Dict, verbose: int) -> Tuple[List[float], List[Dict]]:
        """
        Distills the trained model into a student model.

        Args:
            obs (NumericalData): Input observations.
            policy_targets (np.ndarray): Target values for the policy (actor).
            value_targets (np.ndarray): Target values for the value function (critic).
            cost_targets (np.ndarray): Target values for the cost value function (cost critic).
            params (Dict): Distillation parameters.
            verbose (int): Verbosity level.

        Returns:
            Tuple[List[float], List[Dict]]: The final loss values and updated parameters for distillation.
        """
        return super().distil(obs, [policy_targets, value_targets, cost_targets], params,
                              verbose)

    def predict(self, obs: NumericalData,
                requires_grad: bool = True, start_idx: int = 0,
                stop_idx: int = None, tensor: bool = True) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts both the policy and value outputs for the given observations.

        Args:
            obs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Predicted policy outputs and value functions outputs.
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
        return super().predict(obs, requires_grad, start_idx, stop_idx, tensor, model_idx=0)

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
        return super().predict(obs, requires_grad, start_idx, stop_idx, tensor, model_idx=1)

    def predict_cost(self, obs: NumericalData,
                     requires_grad: bool = True, start_idx: int = 0,
                     stop_idx: int = None, tensor: bool = True) -> NumericalData:
        """
        Predicts the cost value function (cost critic) output for the given observations.

        Args:
            obs (NumericalData): Input observations.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.

        Returns:
            NumericalData: Predicted value function outputs.
        """
        return super().predict(obs, requires_grad, start_idx, stop_idx, tensor, model_idx=2)

    def __copy__(self) -> "SeparateCostActorCriticLearner":
        """
        Creates a copy of the SeparateCostActorCriticLearner instance.

        Returns:
            SeparateCostActorCriticLearner: A new instance with the same parameters and structure.
        """
        opts = [opt.copy() if opt is not None else opt
                for opt in self.optimizers
                ]
        copy_ = SeparateCostActorCriticLearner(self.input_dim, self.output_dim,
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
