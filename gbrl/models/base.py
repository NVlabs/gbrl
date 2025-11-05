##############################################################################
# Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
"""
Base Model Module

This module provides the abstract base class for all GBRL models,
defining the common interface for gradient boosting tree models.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch as th

from gbrl.common.utils import NumericalData
from gbrl.learners.gbt_learner import GBTLearner


class BaseGBT(ABC):
    """
    Abstract base class for gradient boosting tree models.

    This class defines the fundamental interface for all GBRL models and manages
    common functionality such as parameter storage, gradient tracking, and
    model serialization.

    Attributes:
        learner: The underlying GBTLearner instance.
        grads: Stored gradients from the last backward pass.
        params: Stored parameters (predictions) from the last forward pass.
        input: Stored input from the last forward pass (when requires_grad=True).
    """
    def __init__(self):
        """
        Initializes the BaseGBT model.

        Sets up internal state for gradient tracking and parameter storage.
        """
        self.learner = None
        self.grads: Optional[Union[NumericalData, Tuple[Optional[NumericalData], ...]]] = None
        self.params: Optional[Union[NumericalData, Tuple[NumericalData, ...]]] = None
        self.input = None

    def set_bias(self, *args, **kwargs) -> None:
        """
        Sets the bias term for the GBRL model.

        This method should be implemented by subclasses to set the initial
        bias value for predictions.

        Raises:
            NotImplementedError: This is an abstract method that must be
                implemented by subclasses.
        """
        assert self.learner is not None, "learner must be initialized first"
        self.learner.set_bias(*args, **kwargs)

    def set_feature_weights(self, feature_weights: NumericalData) -> None:
        """
        Sets per-feature importance weights for split selection.

        Feature weights are used to scale the contribution of each feature
        when selecting the best split during tree construction. Higher weights
        give features more importance in the splitting process.

        Args:
            feature_weights (NumericalData): Array of weights (one per feature).
                All weights must be >= 0.

        Raises:
            AssertionError: If learner is not initialized or weights are invalid.
        """
        assert self.learner is not None, "learner must be initialized first"

        if isinstance(feature_weights, th.Tensor):
            feature_weights = feature_weights.clone().detach().cpu().numpy()
        # GBRL works with 2D numpy arrays.
        if len(feature_weights.shape) == 1:
            feature_weights = feature_weights[:, np.newaxis]
        self.learner.set_feature_weights(feature_weights.astype(np.single))

    def get_iteration(self) -> Union[int, Tuple[int, ...]]:
        """
        Gets the current number of boosting iterations completed.

        Returns:
            Union[int, Tuple[int, ...]]: Number of boosting iterations per learner.
                Returns a single int for single models or a tuple for multi-models.

        Raises:
            AssertionError: If learner is not initialized.
        """
        assert self.learner is not None, "learner must be initialized first"

        return self.learner.get_iteration()

    def get_total_iterations(self) -> int:
        """
        Gets the total cumulative number of boosting iterations.

        For actor-critic models with separate learners, this returns the sum
        of iterations across both actor and critic. For single models or shared
        architectures, this equals get_iteration().

        Returns:
            int: Total number of boosting iterations across all learners.

        Raises:
            AssertionError: If learner is not initialized.
        """
        assert self.learner is not None, "learner must be initialized first"

        return self.learner.get_total_iterations()

    def get_schedule_learning_rates(self) -> Union[float, Tuple[float, ...]]:
        """
        Gets current scheduled learning rate values for all optimizers.

        For constant schedules, returns the initial learning rate unchanged.
        For linear schedules, returns the learning rate adjusted based on the
        number of trees in the ensemble relative to the total expected iterations.

        Returns:
            Union[float, Tuple[float, ...]]: Current learning rate(s) per optimizer.

        Raises:
            AssertionError: If learner is not initialized.
        """
        assert self.learner is not None, "learner must be initialized first"

        return self.learner.get_schedule_learning_rates()

    @abstractmethod
    def step(self, *args, **kwargs) -> None:
        """
        Performs a single boosting step by fitting one tree to gradients.

        This method should be implemented by subclasses to add a new tree to
        the ensemble based on the computed gradients.

        Raises:
            NotImplementedError: This is an abstract method that must be
                implemented by subclasses.
        """
        pass

    def fit(self, *args, **kwargs) -> Union[float, Tuple[float, ...]]:
        """
        Fits the model for multiple iterations (supervised learning mode).

        This method performs batch training by fitting multiple trees
        sequentially on the provided data.

        Returns:
            Union[float, Tuple[float, ...]]: Final loss value(s) per learner
                averaged over all examples.

        Raises:
            NotImplementedError: This method must be implemented by subclasses
                that support supervised learning.
        """
        raise NotImplementedError

    def get_num_trees(self, *args, **kwargs) -> Union[int, Tuple[int, ...]]:
        """
        Gets the total number of trees in the ensemble.

        Returns:
            Union[int, Tuple[int, ...]]: Number of trees in the ensemble per learner.
                Returns a single int for single models or a tuple for multi-models.

        Raises:
            AssertionError: If learner is not initialized.
        """
        assert self.learner is not None, "learner must be initialized first"

        return self.learner.get_num_trees(*args, **kwargs)

    def tree_shap(self, tree_idx: int, features: NumericalData, *args, **kwargs) -> \
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculates SHAP values for a single tree in the ensemble.

        Implementation based on Linear TreeShap algorithm by Yu et al., 2023.
        See: https://arxiv.org/pdf/2209.08192 and
        https://github.com/yupbank/linear_tree_shap

        Args:
            tree_idx (int): Index of the tree to compute SHAP values for.
            features (NumericalData): Input features for SHAP computation.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: SHAP values with
                shape [n_samples, n_features, n_outputs]. Returns a tuple of SHAP
                values for separate actor-critic models.

        Raises:
            AssertionError: If learner is not initialized.
        """
        assert self.learner is not None, "learner must be initialized first"

        return self.learner.tree_shap(tree_idx, features, *args, **kwargs)

    def shap(self, features: NumericalData, *args, **kwargs) -> \
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculates SHAP values for the entire ensemble.

        Implementation based on Linear TreeShap algorithm by Yu et al., 2023.
        Computes SHAP values sequentially for each tree and aggregates them.
        See: https://arxiv.org/pdf/2209.08192 and
        https://github.com/yupbank/linear_tree_shap

        Args:
            features (NumericalData): Input features for SHAP computation.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: SHAP values with
                shape [n_samples, n_features, n_outputs]. Returns a tuple of SHAP
                values for separate actor-critic models.

        Raises:
            AssertionError: If learner is not initialized.
        """
        assert self.learner is not None, "learner must be initialized first"

        return self.learner.shap(features, *args, **kwargs)

    def save_learner(self, save_path: str) -> None:
        """
        Saves the model to disk.

        Args:
            save_path (str): Absolute path and filename for saving the model.
                The .gbrl_model extension will be added automatically.

        Raises:
            AssertionError: If learner is not initialized.
        """
        assert self.learner is not None, "learner must be initialized first"

        self.learner.save(save_path)

    def export_learner(self, filename: str, modelname: Optional[str] = None) -> None:
        """
        Exports the model as a C header file for embedded deployment.

        Args:
            filename (str): Absolute path and filename for the exported header.
                The .h extension will be added automatically.
            modelname (Optional[str], optional): Name to use for the model in
                the C code. Defaults to None (empty string).

        Raises:
            AssertionError: If learner is not initialized.
        """
        assert self.learner is not None, "learner must be initialized first"

        self.learner.export(filename, modelname)

    @classmethod
    def load_learner(cls, load_name: str, device: str) -> "BaseGBT":
        """
        Loads a model from disk.

        Args:
            load_name (str): Full path to the saved model file.
            device (str): Device to load the model onto ('cpu' or 'cuda').

        Returns:
            BaseGBT: Loaded model instance of the appropriate subclass.
        """
        instance = cls.__new__(cls)
        instance.learner = GBTLearner.load(load_name, device)
        instance.grads = None
        instance.params = None
        instance.input = None
        return instance

    def get_params(self) -> Optional[Union[NumericalData, Tuple[NumericalData, ...]]]:
        """
        Gets a copy of the model's predicted parameters from the last forward pass.

        Returns:
            Optional[Union[NumericalData, Tuple[NumericalData, ...]]]: Cloned/copied
                parameters or None if no forward pass has been performed.
        """
        if self.params is None:
            return None

        if isinstance(self.params, tuple):
            return tuple(p.detach().clone() if isinstance(p, th.Tensor) else p.copy() for p in self.params)

        return self.params.detach().clone() if isinstance(self.params, th.Tensor) else self.params.copy()

    def get_grads(self) -> Optional[Union[NumericalData, Tuple[NumericalData, ...]]]:
        """
        Gets a copy of the gradients from the last backward pass.

        Returns:
            Optional[Union[NumericalData, Tuple[NumericalData, ...]]]: Cloned/copied
                gradients or None if no backward pass has been performed.
        """
        if self.grads is None:
            return None

        if isinstance(self.grads, tuple):
            grads = []
            for g in self.grads:
                if g is not None:
                    grads.append(g.detach().clone() if isinstance(g, th.Tensor) else g.copy())
                else:
                    grads.append(None)
            return tuple(grads)

        return self.grads.detach().clone() if isinstance(self.grads, th.Tensor) else self.grads.copy()

    def set_device(self, device: str):
        """
        Sets the computation device for the GBRL model.

        Args:
            device (str): Target device, must be either 'cpu' or 'cuda'.

        Raises:
            AssertionError: If device is not 'cpu' or 'cuda', or if learner
                is not initialized.
        """
        assert device in ['cpu', 'cuda'], "device must be in ['cpu', 'cuda']"
        assert self.learner is not None, "learner must be initialized first"

        self.learner.set_device(device)

    def get_device(self) -> Union[str, Tuple[str, ...]]:
        """
        Gets the current computation device(s) for the model.

        Returns:
            Union[str, Tuple[str, ...]]: Device string ('cpu' or 'cuda') per learner.
                Returns a single string for single models or a tuple for multi-models.

        Raises:
            AssertionError: If learner is not initialized.
        """
        assert self.learner is not None, "learner must be initialized first"

        return self.learner.get_device()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Union[NumericalData, Tuple[NumericalData, ...]]:
        """
        Performs forward pass through the model.

        This method should be implemented by subclasses to compute predictions
        and optionally store parameters for gradient computation.

        Returns:
            Union[NumericalData, Tuple[NumericalData, ...]]: Model predictions
                (Tensor or numpy array) per learner.

        Raises:
            NotImplementedError: This is an abstract method that must be
                implemented by subclasses.
        """
        pass

    def print_tree(self, tree_idx: int, *args, **kwargs) -> None:
        """
        Prints detailed information about a specific tree to stdout.

        Args:
            tree_idx (int): Index of the tree to print.

        Raises:
            AssertionError: If learner is not initialized.
        """
        assert self.learner is not None, "learner must be initialized first"

        self.learner.print_tree(tree_idx, *args, **kwargs)

    def plot_tree(self, tree_idx: int, filename: str, *args, **kwargs) -> None:
        """
        Visualizes a tree and saves it as a PNG image.

        Note: Only works if GBRL was compiled with Graphviz support.

        Args:
            tree_idx (int): Index of the tree to visualize.
            filename (str): Output filename for the PNG image (extension optional).

        Raises:
            AssertionError: If learner is not initialized.
        """
        assert self.learner is not None, "learner must be initialized first"

        self.learner.plot_tree(tree_idx, filename, *args, **kwargs)

    def copy(self) -> "BaseGBT":
        """
        Creates a deep copy of the model instance.

        Returns:
            BaseGBT: A new instance with copied parameters and state.
        """
        return self.__copy__()

    @abstractmethod
    def __copy__(self) -> "BaseGBT":
        """
        Copy constructor implementation.

        This method should be implemented by subclasses to create a proper
        deep copy of the model.

        Returns:
            BaseGBT: A new instance with copied parameters and state.

        Raises:
            NotImplementedError: This is an abstract method that must be
                implemented by subclasses.
        """
        pass
