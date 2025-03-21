##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch as th

from gbrl import GBRL_CPP
from gbrl.learners.base import BaseLearner
from gbrl.common.utils import (NumericalData, ensure_leaf_tensor_or_array,
                               ensure_same_type, get_poly_vectors, get_tensor_info,
                               numerical_dtype, preprocess_features, to_numpy)


class GBTLearner(BaseLearner):
    """
    GBTLearner is a gradient boosted tree learner that utilizes a C++ backend
    for efficient computation.
    It supports training, prediction, saving, loading,
    and SHAP value computation.
    """
    def __init__(self, input_dim: int, output_dim: int, tree_struct: Dict,
                 optimizers: Union[Dict, List], params: Dict,
                 verbose: int = 0, device: str = 'cpu'):
        """
        Initializes the GBTLearner.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output dimensions.
            tree_struct (Dict): A dictionary containing tree structure parameters.
            optimizers (Union[Dict, List]): A dictionary or list of dictionaries containing optimizer parameters.
            params (Dict): A dictionary containing model parameters.
            verbose (int, optional): Verbosity level. Defaults to 0.
            device (str, optional): The device to run the model on. Defaults to 'cpu'.
        """
        super().__init__(input_dim, output_dim, tree_struct, params, verbose, device)
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        self.optimizers = optimizers
        self._cpp_model = None
        self.student_model = None

    def reset(self) -> None:
        """
        Resets the learner to its initial state,
        reinitializing the C++ model and optimizers.
        """
        if self._cpp_model is not None:
            lrs = self._cpp_model.get_scheduler_lrs()
            for i in range(len(self.optimizers)):
                self.optimizers[i]['init_lr'] = lrs[i]

        self._cpp_model = GBRL_CPP(**self.params)
        self._cpp_model.set_feature_weights(self.feature_weights)
        if self.student_model is not None:
            for i in range(len(self.optimizers)):
                self.optimizers[i]['T'] -= self.total_iterations
        else:
            self.total_iterations = 0
        try:
            for opt in self.optimizers:
                self._cpp_model.set_optimizer(**opt)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def step(self, features: Union[np.ndarray, th.Tensor, Tuple], grads: NumericalData) -> None:
        """
        Performs a single gradient update step (e.g, adding a single decision tree).

        Args:
            features (Union[np.ndarray, th.Tensor, Tuple]): Input features.
            grads (NumericalData): Gradients.
        """
        features, grads = ensure_same_type(features, grads)
        if isinstance(features, th.Tensor):
            features = features.float()
            grads = grads.float()
            self._save_memory = (features, grads)
            num_features = get_tensor_info(features)
            cat_features = None
            grads = get_tensor_info(grads)
            self._save_memory = None
        else:
            num_features, cat_features = preprocess_features(features)
            grads = np.ascontiguousarray(grads.reshape((len(grads), self.params['output_dim'])))
            grads = grads.astype(numerical_dtype)

        self._cpp_model.step(num_features, cat_features, grads)
        self.iteration = self._cpp_model.get_iteration()
        self.total_iterations += 1

    def fit(self, features: NumericalData,
            targets: NumericalData, iterations: int,
            shuffle: bool = True, loss_type: str = 'MultiRMSE') -> float:
        """
        Fits the model to the provided features and targets for a
        given number of iterations.

        Args:
            features (NumericalData): Input features.
            targets (NumericalData): Target values.
            iterations (int): Number of training iterations.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            loss_type (str, optional): Type of loss function. Defaults to 'MultiRMSE'.

        Returns:
            float: The final loss value.
        """
        if isinstance(features, th.Tensor):
            features = features.detach().cpu().numpy()
        num_features, cat_features = preprocess_features(features)
        targets = to_numpy(targets)
        targets = targets.reshape((len(targets), self.params['output_dim']))
        loss = self._cpp_model.fit(num_features, cat_features,
                                   targets.astype(numerical_dtype),
                                   iterations, shuffle, loss_type)
        self.iteration = self._cpp_model.get_iteration()
        return loss

    def save(self, filename: str) -> None:
        """
        Saves the model to a file.

        Args:
            filename (str): The filename to save the model to.
        """
        filename = filename.rstrip('.')
        filename += '.gbrl_model'
        assert self._cpp_model is not None, "Can't save non-existent model!"
        status = self._cpp_model.save(filename)
        assert status == 0, "Failed to save model"

    def export(self, filename: str, modelname: str = None) -> None:
        """
        Exports the model to a C header file.

        Args:
            filename (str): The filename to export the model to.
            modelname (str, optional): The name of the model in the C code. Defaults to None.
        """
        filename = filename.rstrip('.')
        filename += '.h'
        assert self._cpp_model is not None, "Can't export non-existent model!"
        if modelname is None:
            modelname = ""
        try:
            status = self._cpp_model.export(filename, modelname)
            assert status == 0, "Failed to export model"
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    @classmethod
    def load(cls, filename: str, device: str) -> "GBTLearner":
        """
        Loads a GBTLearner model from a file.

        Args:
            filename (str): The filename to load the model from.
            device (str): The device to load the model onto.

        Returns:
            GBTLearner: The loaded GBTLearner instance.
        """
        filename = filename.rstrip('.')
        if '.gbrl_model' not in filename:
            filename += '.gbrl_model'
        assert os.path.isfile(filename), "filename doesn't exist!"
        try:
            instance = cls.__new__(cls)
            instance._cpp_model = GBRL_CPP.load(filename)
            instance.set_device(device)
            metadata = instance._cpp_model.get_metadata()
            instance.tree_struct = {'max_depth': metadata['max_depth'],
                                    'min_data_in_leaf':
                                    metadata['min_data_in_leaf'],
                                    'n_bins': metadata['n_bins'],
                                    'par_th': metadata['par_th'],
                                    'batch_size': metadata['batch_size'],
                                    'grow_policy': metadata['grow_policy']}
            instance.params = {'input_dim': metadata['input_dim'],
                               'output_dim': metadata['output_dim'],
                               'split_score_func':
                               metadata['split_score_func'],
                               'generator_type': metadata['generator_type'],
                               'use_control_variates':
                               metadata['use_control_variates'],
                               'verbose': metadata['verbose'],
                               'device': instance._cpp_model.get_device(),
                               **instance.tree_struct
                               }
            instance.output_dim = metadata['output_dim']
            instance.input_dim = metadata['input_dim']
            instance.verbose = metadata['verbose']
            instance.optimizers = instance._cpp_model.get_optimizers()
            instance.iteration = metadata['iteration']
            instance.total_iterations = metadata['iteration']
            instance.student_model = None
            instance.feature_weights = instance._cpp_model.get_feature_weights()
            instance.device = instance.params['device']
            return instance
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")
            return None

    def get_schedule_learning_rates(self) -> Union[int, Tuple[int, int]]:
        """
        Returns the learning rates of the schedulers.

        Returns:
            Union[int, Tuple[int, int]]: The learning rates.
        """
        return self._cpp_model.get_scheduler_lrs()

    def get_iteration(self) -> int:
        """
        Returns the current iteration number.

        Returns:
            int: The current iteration number.
        """
        return self._cpp_model.get_iteration()

    def get_num_trees(self) -> int:
        """
        Returns the total number of trees in the ensemble.

        Returns:
            int: The total number of trees.
        """
        num_trees = self._cpp_model.get_num_trees()
        if self.student_model is not None:
            num_trees += self.student_model.get_num_trees()
        return num_trees

    def set_bias(self, bias: Union[np.ndarray, float]) -> None:
        """
        Sets the bias of the model.

        Args:
            bias (Union[np.ndarray, float]): The bias value.
        """
        if not isinstance(bias, np.ndarray) and not isinstance(bias, float):
            raise TypeError("Input should be a numpy array or float")

        if isinstance(bias, float):
            bias = np.ndarray([float])

        if bias.ndim > 1:
            bias = bias.ravel()
        elif bias.ndim == 0:
            bias = np.ndarray([bias.item()])  # Converts 0D arrays to 1D
        try:
            bias = bias.astype(np.single)
            self._cpp_model.set_bias(bias)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def set_feature_weights(self, feature_weights: Union[np.ndarray, float]) -> None:
        """
        Sets the feature weights of the model.

        Args:
            feature_weights (Union[np.ndarray, float]): The feature weights.
        """
        if not isinstance(feature_weights, np.ndarray) and not isinstance(feature_weights, float):
            raise TypeError("Input should be a numpy array or float")

        if isinstance(feature_weights, float):
            feature_weights = np.array([float])

        if feature_weights.ndim > 1:
            feature_weights = feature_weights.ravel()
        elif feature_weights.ndim == 0:
            # Converts 0D arrays to 1D
            feature_weights = np.array([feature_weights.item()])
        assert len(feature_weights) == self.input_dim, ("feature weights has to have the "
                                                        "same number of elements as features")
        assert np.all(feature_weights >= 0), "feature weights contains non-positive values"
        try:
            self._cpp_model.set_feature_weights(feature_weights)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def get_bias(self) -> np.ndarray:
        """
        Returns the bias of the model.

        Returns:
            np.ndarray: The bias.
        """
        return self._cpp_model.get_bias()

    def get_feature_weights(self) -> np.ndarray:
        """
        Returns the feature weights of the model.

        Returns:
            np.ndarray: The feature weights.
        """
        return self._cpp_model.get_feature_weights()

    def get_device(self) -> str:
        """
        Returns the device the model is running on.

        Returns:
            str: The device.
        """
        return self._cpp_model.get_device()

    def print_tree(self, tree_idx: int) -> None:
        """
        Prints the tree at the given index.

        Args:
            tree_idx (int): The index of the tree to print.
        """
        self._cpp_model.print_tree(tree_idx)

    def plot_tree(self, tree_idx: int, filename: str) -> None:
        """
        Plots the tree at the given index and saves it to a file.

        Args:
            tree_idx (int): The index of the tree to plot.
            filename (str): The filename to save the plot to.
        """
        filename = filename.rstrip('.')
        try:
            self._cpp_model.plot_tree(tree_idx, filename)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def tree_shap(self, tree_idx: int, features:
                  NumericalData) -> np.ndarray:
        """
        Computes SHAP values for a single tree.

        Implementation based on - https://github.com/yupbank/linear_tree_shap
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192
        Args:
            tree_idx (int): tree index
            features (NumericalData):

        Returns:
            np.ndarray: shap values
        """
        if isinstance(features, th.Tensor):
            features = features.detach().cpu().numpy()
        num_features, cat_features = preprocess_features(features)
        poly_vectors = get_poly_vectors(self.params['max_depth'], numerical_dtype)
        base_poly, norm_values, offset = poly_vectors

        base_poly = np.ascontiguousarray(base_poly)
        norm_values = np.ascontiguousarray(norm_values)
        offset = np.ascontiguousarray(offset)
        return self._cpp_model.tree_shap(tree_idx, num_features, cat_features,
                                         norm_values, base_poly, offset)

    def shap(self, features: NumericalData) -> np.ndarray:
        """
        Computes SHAP values for the entire ensemble.

        Uses Linear tree shap for each tree in the ensemble (sequentially)
        Implementation based on - https://github.com/yupbank/linear_tree_shap
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192
        Args:
            features (NumericalData):

        Returns:
            np.ndarray: shap values
        """
        if isinstance(features, th.Tensor):
            features = features.detach().cpu().numpy()
        num_features, cat_features = preprocess_features(features)
        poly_vectors = get_poly_vectors(self.params['max_depth'], numerical_dtype)
        base_poly, norm_values, offset = poly_vectors

        base_poly = np.ascontiguousarray(base_poly)
        norm_values = np.ascontiguousarray(norm_values)
        offset = np.ascontiguousarray(offset)
        return self._cpp_model.ensemble_shap(num_features, cat_features,
                                             norm_values, base_poly, offset)

    def set_device(self, device: Union[str, th.device]) -> None:
        """
        Sets the device the model should run on.

        Args:
            device (Union[str, th.device]): The device to set.
        """
        if isinstance(device, th.device):
            device = device.type
        try:
            self._cpp_model.to_device(device)
            self.device = device
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def predict(self, features: NumericalData, requires_grad: bool = True,
                start_idx: int = 0, stop_idx: int = None, tensor: bool = True) -> np.ndarray:
        """
        Predicts the output for the given features.

        Args:
            features (NumericalData): Input features.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.

        Returns:
            np.ndarray: The predicted output.
        """
        if stop_idx is None:
            stop_idx = 0

        if isinstance(features, th.Tensor):
            features = features.float()
            # store features so that data isn't garbage
            # collected while GBRL uses it
            self._save_memory = features
            num_features = get_tensor_info(features)
            cat_features = None
        else:
            num_features, cat_features = preprocess_features(features)

        preds = self._cpp_model.predict(num_features, cat_features, start_idx, stop_idx)
        preds = th.from_dlpack(preds) if not isinstance(preds, np.ndarray) else preds

        # Add student model predictions if available
        if self.student_model is not None:
            student_preds = self.student_model.predict(num_features,
                                                       cat_features,
                                                       start_idx,
                                                       stop_idx)
            student_preds = th.from_dlpack(student_preds) if not \
                isinstance(student_preds, np.ndarray) else student_preds
            preds += student_preds

        preds = ensure_leaf_tensor_or_array(preds, tensor, requires_grad, self.device)
        return preds

    def distil(self, obs: NumericalData, targets: np.ndarray,
               params: Dict, verbose: int = 0) -> Tuple[int, Dict]:
        """
        Distills the model into a student model.

        Args:
            obs (NumericalData): Input observations.
            targets (np.ndarray): Target values.
            params (Dict): Distillation parameters.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            Tuple[int, Dict]: The final loss and updated parameters.
        """
        num_obs, cat_obs = preprocess_features(obs)
        distil_params = {'output_dim': self.params['output_dim'],
                         'split_score_func': 'L2',
                         'generator_type': 'Quantile',
                         'use_control_variates': False, 'device': self.device,
                         'max_depth': params.get('distil_max_depth', 6),
                         'verbose': verbose, 'batch_size':
                         self.params.get('distil_batch_size', 2048)}
        self.student_model = GBRL_CPP(**distil_params)
        distil_optimizer = {'algo': 'SGD',
                            'init_lr': params.get('distil_lr', 0.1)}
        try:
            self.student_model.set_optimizer(**distil_optimizer)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

        bias = np.mean(targets, axis=0)
        if isinstance(bias, float):
            bias = np.ndarray([bias])
        self.student_model.set_bias(bias.astype(numerical_dtype))
        tr_loss = self.student_model.fit(num_obs, cat_obs,
                                         targets, params['min_steps'])
        while tr_loss > params.get('min_distillation_loss', 0.1):
            if params['min_steps'] < params['limit_steps']:
                steps_to_add = min(500, params['limit_steps'] - params['min_steps'])
                tr_loss = self.student_model.fit(num_obs, cat_obs,
                                                 targets, steps_to_add,
                                                 shuffle=False)
                params['min_steps'] += steps_to_add
            else:
                break
        self.reset()
        return tr_loss, params

    def print_ensemble_metadata(self):
        """Prints the metadata of the ensemble."""
        self._cpp_model.print_ensemble_metadata()

    def __copy__(self):
        """Creates a copy of the GBTLearner instance."""
        opts = [opt.copy() if opt is not None else opt
                for opt in self.optimizers
                ]
        copy_ = GBTLearner(self.input_dim, self.output_dim,
                           self.tree_struct.copy(),
                           opts, self.params, self.verbose,
                           self.device)
        copy_.iteration = self.iteration
        copy_.total_iterations = self.total_iterations
        if self._cpp_model is not None:
            copy_._cpp_model = GBRL_CPP(self._cpp_model)
        if self.student_model is not None:
            copy_.student_model = GBRL_CPP(self.student_model)
        return copy_
