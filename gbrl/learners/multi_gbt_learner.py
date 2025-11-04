##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th

from gbrl import GBRL_CPP
from gbrl.common.utils import (NumericalData, ensure_leaf_tensor_or_array,
                               get_poly_vectors, get_tensor_info,
                               numerical_dtype, preprocess_features, to_numpy)
from gbrl.learners.base import BaseLearner


class MultiGBTLearner(BaseLearner):
    """
    MultiGBTLearner is a gradient boosted tree learner that
    utilizes a C++ backend for efficient computation that contains
    multiple GBT models.
    It supports training, prediction, saving, loading,
    and SHAP value computation.
    """
    def __init__(self, input_dim: Union[int, List[int]],
                 output_dim: Union[int, List[int]],
                 tree_struct: Dict,
                 optimizers: Union[Dict, List[Dict]],
                 params: Dict,
                 n_learners: int,
                 policy_dim: Optional[Union[int, List[int]]] = None,
                 verbose: int = 0,
                 device: str = 'cpu'):
        """
        Initializes the MultiGBTLearner.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output dimensions.
            tree_struct (Dict): A dictionary containing tree structure parameters.
            optimizers (Union[Dict, List]): A dictionary or list of
            dictionaries containing optimizer parameters.
            params (Dict): A dictionary containing model parameters.
            n_learners (int): Number of GBT learners.
            policy_dim (Optional[Union[int, List[int]]]): The dimension of the policy space.
            verbose (int, optional): Verbosity level. Defaults to 0.
            device (str, optional): The device to run the model on. Defaults to 'cpu'.
        """

        assert len(optimizers) == 1 or len(optimizers) == n_learners
        if isinstance(input_dim, list):
            assert len(input_dim) == n_learners
            if isinstance(output_dim, int):
                output_dim = [output_dim] * n_learners
        if isinstance(output_dim, list):
            assert len(output_dim) == n_learners
            if isinstance(input_dim, int):
                input_dim = [input_dim] * n_learners

        if policy_dim is None:
            policy_dim = output_dim

        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         tree_struct=tree_struct,
                         params=params,
                         policy_dim=policy_dim,
                         verbose=verbose,
                         device=device)
        if isinstance(optimizers, dict):
            optimizers = [optimizers for _ in range(n_learners)]
        self.optimizers = optimizers
        self._cpp_models = None
        self.student_models = None
        self.n_learners = n_learners

    def reset(self) -> None:
        """Resets the learner to its initial state, reinitializing the C++ model and optimizers."""
        if self._cpp_models:
            for i in range(self.n_learners):
                lr = self._cpp_models[i].get_scheduler_lrs()
                self.optimizers[i]['init_lr'] = lr

        self._cpp_models = []
        params = self.params.copy()
        for i in range(self.n_learners):
            if isinstance(self.input_dim, list):
                params['input_dim'] = self.input_dim[i]  #  type: ignore
                params['output_dim'] = self.output_dim[i]  #  type: ignore
                params['policy_dim'] = self.policy_dim[i]  #  type: ignore
            cpp_model = GBRL_CPP(**params)
            cpp_model.set_feature_weights(self.feature_weights)
            if self.student_models is not None:
                self.optimizers[i]['T'] -= self.total_iterations
            try:
                cpp_model.set_optimizer(**self.optimizers[i])
                self._cpp_models.append(cpp_model)
            except RuntimeError as e:
                print(f"Caught an exception in GBRL: {e}")

        if self.student_models is None:
            self.total_iterations = 0
        self.iteration = [0] * self.n_learners

    def step(self,
             inputs: NumericalData,
             grads: Union[Sequence[NumericalData], NumericalData],
             model_idx: Optional[int] = None) -> None:
        """
        Performs a single gradient update step (e.g, adding a single decision tree).

        Args:
            inputs (Union[np.ndarray, th.Tensor, Tuple]): Input features.
            grads (Union[Sequence[NumericalData], NumericalData]): Gradients.
            model_idx (int, optional): The index of the model.
        """
        assert model_idx is not None or ((isinstance(grads, list) or isinstance(grads, tuple)) and
                                         len(grads) == self.n_learners), "Invalid model index or gradients"
        assert self._cpp_models is not None, "Model not initialized."

        num_inputs, cat_inputs = preprocess_features(inputs)

        if model_idx is not None:
            assert not isinstance(grads, list), "When model_idx is specified, grads should not be a list"
            self._memory = []
            self._cpp_models[model_idx].step(obs=self.transform_data(num_inputs),
                                             categorical_obs=cat_inputs,
                                             grads=self.transform_data(grads),  # type: ignore
                                             )

            self._memory = []
            self.iteration[model_idx] = self._cpp_models[model_idx].get_iteration()
        else:
            self._memory = []
            for i in range(self.n_learners):
                self._cpp_models[i].step(obs=self.transform_data(num_inputs),
                                         categorical_obs=cat_inputs,
                                         grads=self.transform_data(grads[i]),  # type: ignore
                                         )

                self._memory = []

                self.iteration[i] = self._cpp_models[i].get_iteration()
        self.total_iterations += 1

    def fit(self, inputs: NumericalData,
            targets: Union[List[NumericalData], NumericalData],
            iterations: int, shuffle: bool = True, loss_type: str = 'MultiRMSE',
            model_idx: Optional[int] = None) -> Union[float, List[float]]:
        """
        Fits the model to the provided inputs and targets for a
        given number of iterations.

        Args:
            inputs (NumericalData): Input features.
            targets (Union[List[NumericalData], NumericalData]): Target values.
            iterations (int): Number of training iterations.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            loss_type (str, optional): Type of loss function. Defaults to 'MultiRMSE'.
            model_idx (int, optional): The index of the model.

        Returns:
            Union[float, List[float]]: The final loss value.
        """

        assert model_idx is not None or (isinstance(targets, list) and
                                         len(targets) == self.n_learners)
        assert self._cpp_models is not None, "Model not initialized."
        if isinstance(inputs, th.Tensor):
            inputs = inputs.detach().cpu().numpy()
        num_inputs, cat_inputs = preprocess_features(inputs)

        self.total_iterations += iterations

        if model_idx is not None:
            assert not isinstance(targets, list), \
                "when model_idx is specified, targets should not be a list"
            targets = to_numpy(targets)
            targets = targets.reshape((len(targets),
                                       self.params['output_dim']))
            loss = self._cpp_models[model_idx].fit(num_inputs, cat_inputs,
                                                   targets.astype(
                                                       numerical_dtype),
                                                   iterations, shuffle,
                                                   loss_type)
            self.iteration[model_idx] = self._cpp_models[model_idx].get_iteration()
            return loss

        assert isinstance(targets, list) and len(targets) == self.n_learners, \
            "when model_idx is not specified, targets should be a list with length equal to n_learners"
        losses = []
        for i in range(self.n_learners):
            targets[i] = to_numpy(targets[i])
            targets[i] = targets[i].reshape((len(targets[i]),
                                            self.params['output_dim']))
            loss = self._cpp_models[i].fit(num_inputs, cat_inputs,
                                           targets[i],
                                           iterations, shuffle, loss_type)
            self.iteration[i] = self._cpp_models[i].get_iteration()
            losses.append(loss)
        return losses

    def save(self, filename: str, custom_names: Optional[List] = None) -> None:
        """
        Saves the models to a file.

        Args:
            filename (str): The filename to save the model to.
        """
        assert self._cpp_models is not None, "Model not initialized."

        filename = filename.rstrip('.')
        assert custom_names is None or len(custom_names) == self.n_learners, "Custom names must be per learner"
        for i in range(self.n_learners):
            if custom_names is None:
                savename = filename + f'_{i}.gbrl_model'
            else:
                savename = filename + f'_{custom_names[i]}.gbrl_model'
            assert self._cpp_models[i] is not None, "Can't save non-existent model!"
            status = self._cpp_models[i].save(savename)
            assert status == 0, "Failed to save model"

        metadata = {
            "n_learners": self.n_learners,
            'custom_names': custom_names,
            }
        meta_filename = filename + ".gbrl_meta"
        with open(meta_filename, "w") as meta_file:
            json.dump(metadata, meta_file, indent=4)

        print(f"Saved {self.n_learners} models with metadata to {meta_filename}")

    def export(self, filename: str, modelname: Optional[str] = None) -> None:
        """
        Exports the model to a C header file.

        Args:
            filename (str): The filename to export the model to.
            modelname (str, optional): The name of the model in the C code.
            Defaults to None.
        """
        assert self._cpp_models is not None, "Model not initialized."

        filename = filename.rstrip('.')
        for i in range(self.n_learners):
            exportname = filename + f'_{i}.h'
            assert self._cpp_models[i] is not None, "Can't export non-existent model!"
            if modelname is None:
                modelname = ""
            try:
                status = self._cpp_models[i].export(exportname, modelname)
                assert status == 0, "Failed to export model"
            except RuntimeError as e:
                print(f"Caught an exception in GBRL: {e}")

    @classmethod
    def load(cls, filename: str, device: str) -> "MultiGBTLearner":
        """
        Loads a MultiGBTLearner model from files.

        Args:
            filename (str): The filename to load the model from.
            device (str): The device to load the model onto.

        Returns:
            GBTLearner: The loaded GBTLearner instance.
        """
        filename = filename.rstrip('.')

        meta_filename = filename + ".gbrl_meta"
        assert os.path.exists(meta_filename), f"Metadata file {meta_filename} not found!"
        with open(meta_filename, "r") as meta_file:
            metadata = json.load(meta_file)

        n_learners = metadata['n_learners']
        custom_names = metadata['custom_names']
        assert custom_names is None or len(custom_names) == n_learners, "Custom names must be per learner"
        try:
            instance = cls.__new__(cls)
            instance.n_learners = n_learners
            instance._cpp_models = []
            instance.optimizers = []

            instance.input_dim = []
            instance.output_dim = []
            instance.policy_dim = []

            for i in range(n_learners):
                if custom_names is None:
                    loadname = filename + f'_{i}'
                else:
                    loadname = filename + f'_{custom_names[i]}'
                loadname += '.gbrl_model'
                cpp_model = GBRL_CPP.load(loadname)
                instance.optimizers.extend(cpp_model.get_optimizers())
                instance._cpp_models.append(cpp_model)
                model_metadata = cpp_model.get_metadata()
                print(model_metadata)
                instance.output_dim.append(model_metadata['output_dim'])
                instance.input_dim.append(model_metadata['input_dim'])
                instance.policy_dim.append(model_metadata['policy_dim'])

            instance.set_device(device)
            metadata = instance._cpp_models[0].get_metadata()
            instance.tree_struct = {'max_depth': metadata['max_depth'],
                                    'min_data_in_leaf':
                                    metadata['min_data_in_leaf'],
                                    'n_bins': metadata['n_bins'],
                                    'par_th': metadata['par_th'],
                                    'batch_size': metadata['batch_size'],
                                    'grow_policy': metadata['grow_policy']}
            instance.params = {'input_dim': instance.input_dim,
                               'output_dim': instance.output_dim,
                               'policy_dim': instance.policy_dim,
                               'split_score_func':
                               metadata['split_score_func'],
                               'generator_type': metadata['generator_type'],
                               'use_control_variates':
                               metadata['use_control_variates'],
                               'verbose': metadata['verbose'],
                               'device': device,
                               **instance.tree_struct
                               }
            instance.output_dim = metadata['output_dim']
            instance.input_dim = metadata['input_dim']
            instance.policy_dim = metadata['policy_dim']
            instance.verbose = metadata['verbose']
            instance.params = {'split_score_func':
                               metadata['split_score_func'],
                               'generator_type':
                               metadata['generator_type'],
                               'use_control_variates':
                               metadata['use_control_variates'],
                               }

            instance.iteration = metadata['iteration']
            instance.total_iterations = metadata['iteration']
            instance.student_models = None
            instance.feature_weights = instance._cpp_models[0].get_feature_weights()
            return instance
        except RuntimeError as e:
            raise RuntimeError(f"Caught an exception in GBRL: {e}")

    def get_schedule_learning_rates(self, model_idx: Optional[int] = None) -> Union[np.ndarray,
                                                                                    Tuple[np.ndarray,
                                                                                          ...]]:
        """
        Returns the learning rates of the schedulers.

        Args:
            model_idx (int, optional): The index of the model.

        Returns:
            Union[int, Tuple[int, int]]: The learning rates.
        """
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."

        if model_idx is not None:
            return self._cpp_models[model_idx].get_scheduler_lrs()
        return (cpp_model.get_scheduler_lrs() for cpp_model in self._cpp_models)  # type: ignore

    def get_iteration(self, model_idx: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        """
        Returns the current iteration number.

        Args:
            model_idx (int, optional): The index of the model.

        Returns:
            Union[int, Tuple[int, int]]: The current iteration number.
        """
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."
        if model_idx is not None:
            return self._cpp_models[model_idx].get_iteration()
        return (cpp_model.get_iteration() for cpp_model in self._cpp_models)  # type: ignore

    def get_num_trees(self, model_idx: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        """
        Returns the total number of trees in the ensemble.

        Args:
            model_idx (int, optional): The index of the model.

        Returns:
            Union[int, Tuple[int, int]]: The total number of trees.
        """
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."

        if model_idx is not None:
            _num_trees = self._cpp_models[model_idx].get_num_trees()
            if self.student_models is not None:
                _num_trees += self.student_models[model_idx].get_num_trees()
            return _num_trees

        num_trees = []
        for i in range(self.n_learners):
            _num_trees = self._cpp_models[i].get_num_trees()
            if self.student_models is not None:
                _num_trees += self.student_models[i].get_num_trees()
            num_trees.append(_num_trees)
        return tuple(num_trees)

    def set_bias(self, bias: Union[np.ndarray, float],
                 model_idx: Optional[int] = None) -> None:
        """
        Sets the bias of the model.

        Args:
            bias (Union[np.ndarray, float]): The bias value.
            model_idx (int, optional): model index to set bias to.
        """
        if not isinstance(bias, np.ndarray) and not isinstance(bias, float):
            raise TypeError("Input should be a numpy array or float")
        assert self._cpp_models is not None, "Model not initialized."

        if isinstance(bias, float):
            bias = np.array([float])

        if bias.ndim > 1:
            bias = bias.ravel()
        elif bias.ndim == 0:
            bias = np.ndarray([bias.item()])  # Converts 0D arrays to 1D
        try:
            if model_idx is None:
                for i in range(self.n_learners):
                    self._cpp_models[i].set_bias(bias)
            else:
                self._cpp_models[model_idx].set_bias(bias)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def set_feature_weights(self, feature_weights:
                            Union[np.ndarray, float],
                            model_idx: Optional[int] = None) -> None:
        """
        Sets the feature weights of the model.

        Args:
            feature_weights (Union[np.ndarray, float]): The feature weights.
            model_idx (int, optional): The index of the model.
        """
        if not isinstance(feature_weights, np.ndarray) and not isinstance(feature_weights, float):
            raise TypeError("Input should be a numpy array or float")

        assert self._cpp_models is not None, "Model not initialized."

        if isinstance(feature_weights, float):
            feature_weights = np.array([float])

        if feature_weights.ndim > 1:
            feature_weights = feature_weights.ravel()
        elif feature_weights.ndim == 0:
            # Converts 0D arrays to 1D
            feature_weights = np.array([feature_weights.item()])
        assert len(feature_weights) == self.input_dim, \
            "feature weights has to have the same number of elements as features"
        assert np.all(feature_weights >= 0), "feature weights contains non-positive values"
        try:
            if model_idx is None:
                for i in range(self.n_learners):
                    self._cpp_models[i].set_feature_weights(feature_weights)
            else:
                self._cpp_models[model_idx].set_feature_weights(
                    feature_weights)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def get_bias(self, model_idx: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Returns the bias of the model.

        Args:
            model_idx (int, optional): The index of the model.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ...]]: The bias.
        """
        if model_idx is None:
            return (cpp_model.get_bias() for cpp_model in self._cpp_models)  # type: ignore
        return self._cpp_models[model_idx].get_bias()  # type: ignore

    def get_feature_weights(self, model_idx: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Returns the feature weights of the model.

        Args:
            model_idx (int, optional): The index of the model.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ...]]: The feature weights.
        """
        if model_idx is None:
            return (cpp_model.get_feature_weights() for cpp_model in self._cpp_models)  # type: ignore
        return self._cpp_models[model_idx].get_feature_weights()  # type: ignore

    def get_device(self, model_idx: Optional[int] = None) -> Union[str, Tuple[str, ...]]:
        """
        Returns the device the model is running on.

        Args:
            model_idx (int, optional): The index of the model.

        Returns:
            Union[str, Tuple[str, ...]]: The device.
        """
        if model_idx is None:
            return (cpp_model.get_device() for cpp_model in self._cpp_models)  # type: ignore
        return self._cpp_models[model_idx].get_device()  # type: ignore

    def print_tree(self, tree_idx: int,
                   model_idx: Optional[int] = None) -> None:
        """
        Prints the tree at the given index.

        Args:
            tree_idx (int): The index of the tree to print.
            model_idx (int, optional): The index of the model to print.
        """
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."
        if model_idx is None:
            for i in range(self.n_learners):
                self._cpp_models[i].print_tree(tree_idx)
        else:
            self._cpp_models[model_idx].print_tree(tree_idx)

    def plot_tree(self, tree_idx: int, filename: str, model_idx: Optional[int] = None) -> None:
        """
        Plots the tree at the given index and saves it to a file.

        Args:
            tree_idx (int): The index of the tree to plot.
            filename (str): The filename to save the plot to.
            model_idx (int, optional): The index of the model to print.
        """
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."

        filename = filename.rstrip('.')
        try:
            if model_idx is not None:
                self._cpp_models[model_idx].plot_tree(tree_idx, filename)
            else:
                for i in range(self.n_learners):
                    self._cpp_models[i].plot_tree(tree_idx, filename + f'_model_{i}')
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def tree_shap(self, tree_idx: int, features:
                  NumericalData,
                  model_idx: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Computes SHAP values for a single tree.

        Implementation based on - https://github.com/yupbank/linear_tree_shap
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192
        Args:
            tree_idx (int): tree index
            features (NumericalData):
            model_idx (int, optional): The index of the model to print.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ...]: shap values
        """
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."

        if isinstance(features, th.Tensor):
            features = features.detach().cpu().numpy()
        num_inputs, cat_inputs = preprocess_features(features)
        poly_vectors = get_poly_vectors(self.params['max_depth'], numerical_dtype)
        base_poly, norm_values, offset = poly_vectors
        base_poly = np.ascontiguousarray(base_poly)
        norm_values = np.ascontiguousarray(norm_values)
        offset = np.ascontiguousarray(offset)
        if model_idx is not None:
            return self._cpp_models[model_idx].tree_shap(tree_idx,
                                                         num_inputs,
                                                         cat_inputs,
                                                         norm_values,
                                                         base_poly,
                                                         offset)
        shap_values = []
        for i in range(self.n_learners):
            shap_values.append(self._cpp_models[i].tree_shap(tree_idx,
                                                             num_inputs,
                                                             cat_inputs,
                                                             norm_values,
                                                             base_poly,
                                                             offset))
        return shap_values  # type: ignore

    def shap(self, features: NumericalData,
             model_idx: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Computes SHAP values for the entire ensemble.

        Uses Linear tree shap for each tree in the ensemble (sequentially)
        Implementation based on - https://github.com/yupbank/linear_tree_shap
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192
        Args:
            features (NumericalData):
            model_idx (int, optional): The index of the model to print.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ...]: shap values
        """
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."

        if isinstance(features, th.Tensor):
            features = features.detach().cpu().numpy()
        num_inputs, cat_inputs = preprocess_features(features)
        poly_vectors = get_poly_vectors(self.params['max_depth'], numerical_dtype)
        base_poly, norm_values, offset = poly_vectors
        base_poly = np.ascontiguousarray(base_poly)
        norm_values = np.ascontiguousarray(norm_values)
        offset = np.ascontiguousarray(offset)
        if model_idx is not None:
            return self._cpp_models[model_idx].ensemble_shap(num_inputs,
                                                             cat_inputs,
                                                             norm_values,
                                                             base_poly,
                                                             offset)
        shap_values = []
        for i in range(self.n_learners):
            shap_values.append(self._cpp_models[i].ensemble_shap(num_inputs,
                                                                 cat_inputs,
                                                                 norm_values,
                                                                 base_poly,
                                                                 offset))
        return shap_values  # type: ignore

    def set_device(self, device: Union[str, th.device],
                   model_idx: Optional[int] = None) -> None:
        """
        Sets the device the model should run on.

        Args:
            device (Union[str, th.device]): The device to set.
            model_idx (int, optional): The index of the model to print.
        """
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."

        if isinstance(device, th.device):
            device = device.type
        try:
            if model_idx is not None:
                self._cpp_models[model_idx].to_device(device)
            else:
                for i in range(self.n_learners):
                    self._cpp_models[i].to_device(device)
            self.device = device
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def predict(self, features: NumericalData,  # type: ignore
                requires_grad: bool = True, start_idx: Optional[int] = 0,
                stop_idx: Optional[int] = None, tensor: bool = True,
                model_idx: Optional[int] = None) -> Union[NumericalData, List[NumericalData]]:
        """
        Predicts the output for the given features.

        Args:
            features (NumericalData): Input features.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.
            model_idx (int, optional): The index of the model to print.

        Returns:
            Union[NumericalData, List[NumericalData]]: The predicted output.
        """
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."
        assert self.n_learners > 0, "No learners in the model."

        if stop_idx is None:
            stop_idx = 0

        if isinstance(features, th.Tensor):
            features = features.float()
            # store features so that data isn't garbage
            # collected while GBRL uses it
            self._save_memory = features
            num_inputs = get_tensor_info(features)
            cat_inputs = None
        else:
            num_inputs, cat_inputs = preprocess_features(features)

        def predict_single_model(model, student_model, device):
            preds = model.predict(num_inputs, cat_inputs, start_idx, stop_idx)
            preds = th.from_dlpack(preds) if not isinstance(preds, np.ndarray) else preds  # type: ignore

            # Add student model predictions if available
            if student_model is not None:
                student_preds = student_model.predict(num_inputs,
                                                      cat_inputs,
                                                      start_idx,
                                                      stop_idx)
                student_preds = th.from_dlpack(student_preds) if not \
                    isinstance(student_preds, np.ndarray) else student_preds
                preds += student_preds
            return ensure_leaf_tensor_or_array(preds, tensor, requires_grad, device)

        if model_idx is not None:
            total_preds = predict_single_model(self._cpp_models[model_idx],
                                               None if self.student_models is
                                               None else
                                               self.student_models[model_idx],
                                               self.device)
        else:
            total_preds = []
            for i in range(self.n_learners):
                preds = predict_single_model(self._cpp_models[i], None if
                                             self.student_models is None else
                                             self.student_models[i],
                                             self.device)
                total_preds.append(preds)

        if isinstance(features, th.Tensor):
            self._save_memory = None

        return total_preds

    def distil(self,
               obs: NumericalData,
               targets: List[np.ndarray],  # type: ignore
               params: Dict,
               verbose: int = 0) -> Tuple[List[float], List[Dict]]:
        """
        Distills the model into a student model.

        Args:
            obs (NumericalData): Input observations.
            targets (np.ndarray): Target values.
            params (Dict): Distillation parameters.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            Tuple[List[float], List[Dict]]: The final loss and updated parameters.
        """
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."
        num_obs, cat_obs = preprocess_features(obs)
        distil_params = {'output_dim': self.params['output_dim'],
                         'split_score_func': 'L2',
                         'generator_type': 'Quantile',
                         'use_control_variates': False, 'device': self.device,
                         'max_depth': params.get('distil_max_depth', 6),
                         'verbose': verbose, 'batch_size':
                         self.params.get('distil_batch_size', 2048)}

        distil_optimizer = {'algo': 'SGD', 'init_lr': params.get('distil_lr', 0.1)}
        self.student_models = []
        tr_losses = []
        distil_params = []
        for i in range(self.n_learners):
            student_model = GBRL_CPP(**distil_params)  # type: ignore
            try:
                student_model.set_optimizer(**distil_optimizer)
            except RuntimeError as e:
                print(f"Caught an exception in GBRL: {e}")

            bias = np.mean(targets[i], axis=0)
            if isinstance(bias, float):
                bias = np.array([bias])

            student_model.set_bias(bias.astype(numerical_dtype))
            tr_loss = student_model.fit(num_obs, cat_obs, targets[i], params['min_steps'])
            while tr_loss > params.get('min_distillation_loss', 0.1):
                if params['min_steps'] < params['limit_steps']:
                    steps_to_add = min(500, params['limit_steps'] - params['min_steps'])
                    tr_loss = student_model.fit(num_obs, cat_obs,
                                                targets[i], steps_to_add,
                                                shuffle=False)
                    params['min_steps'] += steps_to_add
                else:
                    break
            tr_losses.append(tr_loss)
            distil_params.append(params)
            self.student_models.append(student_model)
        self.reset()
        return tr_losses, distil_params

    def print_ensemble_metadata(self, model_idx: Optional[int] = None) -> None:
        """
        Prints the metadata of the ensemble.

        Args:
            model_idx (int, optional): The index of the model.
        """
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."

        if model_idx is not None:
            self._cpp_models[model_idx].print_ensemble_metadata()
        else:
            for i in range(self.n_learners):
                self._cpp_models[i].print_ensemble_metadata()

    def __copy__(self):
        """Creates a copy of the MultiGBTLearner instance."""
        assert self._cpp_models is not None and isinstance(self._cpp_models, list), \
            "Model not initialized."

        opts = [opt.copy() if opt is not None else opt
                for opt in self.optimizers
                ]
        copy_ = MultiGBTLearner(input_dim=self.input_dim,
                                output_dim=self.output_dim,
                                tree_struct=self.tree_struct.copy(),
                                optimizers=opts,
                                params=self.params,
                                n_learners=self.n_learners,
                                policy_dim=self.policy_dim,
                                verbose=self.verbose,
                                device=self.device)
        copy_.iteration = self.iteration
        copy_.total_iterations = self.total_iterations
        if self.student_models is not None:
            copy_.student_models = [None] * self.n_learners

        if self._cpp_models is not None:
            copy_._cpp_models = [None] * self.n_learners
            for i in range(self.n_learners):
                copy_._cpp_models[i] = GBRL_CPP(self._cpp_models[i])
                if self.student_models[i] is not None:  # type: ignore
                    copy_.student_models[i] = GBRL_CPP(self.student_models[i])  # type: ignore
        return copy_
