##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import torch as th

from gbrl.common.utils import NumericalData, to_numpy


class BaseLearner(ABC):
    """
    Abstract base class for gradient boosting tree learners.

    This class defines the fundamental interface for gradient boosting
    learners and serves as a wrapper for the C++ backend.

    Attributes:
        input_dim (int): The number of input features.
        output_dim (int): The number of output dimensions.
        tree_struct (Dict): Dictionary containing tree structure parameters.
        params (Union[Dict, List[Dict]]): Dictionary containing model parameters.
        verbose (int): Verbosity level (0 = silent, 1 = debug).
        device (str): The device the model runs on (e.g., 'cpu' or 'cuda').
        iteration (int): The current training iteration.
        total_iterations (int): Total number of training iterations.
        feature_weights (np.ndarray): Feature importance weights.
    """
    def __init__(self, input_dim: int, output_dim: int, tree_struct: Dict,
                 params: Dict, verbose: int = 0, device: str = 'cpu'):
        """
        Initializes the BaseLearner.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output dimensions.
            tree_struct (Dict): Dictionary containing tree
            structure parameters.
            params (Dict): Dictionary containing additional model parameters.
            verbose (int, optional): Verbosity level (0 = silent, 1 = debug).
            Defaults to 0.
            device (str, optional): Device to run the model on
            ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.tree_struct = tree_struct
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.params = {'input_dim': input_dim,
                       'output_dim': output_dim,
                       'split_score_func': params.get('split_score_func',
                                                      'Cosine'),
                       'generator_type': params.get('generator_type',
                                                    'Quantile'),
                       'use_control_variates': params.get('control_variates',
                                                          False),
                       'verbose': verbose, 'device': device, **tree_struct}

        self.iteration = 0
        self.total_iterations = 0
        self.verbose = verbose
        feature_weights = params.get('feature_weights', None)
        if feature_weights is not None:
            feature_weights = to_numpy(feature_weights)
            feature_weights = feature_weights.flatten()
            message = "feature weights contains non-positive values"
            assert np.all(feature_weights >= 0), message
        else:
            weights = np.ones(input_dim, dtype=np.single)
            feature_weights = np.ascontiguousarray(weights)
        self.feature_weights = feature_weights

    @abstractmethod
    def reset(self) -> None:
        """Resets the model, reinitializing internal states and parameters."""
        pass

    @abstractmethod
    def step(self, *args, **kwargs) -> None:
        """Performs a single update step using provided gradients."""
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> Union[float, List[float]]:
        """
        Trains the model on provided data.

        Returns:
            Union[float, List[float]]: The final loss after training per model.
        """
        pass

    @abstractmethod
    def save(self, filename: str, *args, **kwargs) -> None:
        """
        Saves the model to a file.

        Args:
            filename (str): The filename to save the model to.
        """
        pass

    def export(self, filename: str, modelname: str = None) -> None:
        # exports model to C
        filename = filename.rstrip('.')
        filename += '.h'
        assert self.cpp_model is not None, "Can't export non-existent model!"
        if modelname is None:
            modelname = ""
        try:
            status = self.cpp_model.export(filename, modelname)
            assert status == 0, "Failed to export model"
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    @classmethod
    @abstractmethod
    def load(cls, filename: str, device: str, *args, **kwargs) -> "BaseLearner":
        """
        Loads a model from a file.

        Args:
            filename (str): Path to the model file.
            device (str): Device to load the model onto.

        Returns:
            BaseLearner: Loaded model instance.
        """
        pass

    @abstractmethod
    def get_schedule_learning_rates(self, *args, **kwargs) -> Union[int, Tuple[int, ...]]:
        """
        Retrieves the scheduled learning rates.

        Returns:
            Union[int, Tuple[int, ...]]: Learning rate(s).
        """
        pass

    def get_total_iterations(self) -> int:
        """
        Returns the total number of iterations performed.

        Returns:
            int: The total number of iterations.
        """
        return self.total_iterations

    @abstractmethod
    def get_iteration(self, *args, **kwargs) -> Union[int, Tuple[int, ...]]:
        """
        Retrieves the current iteration count.

        Returns:
            Union[int, Tuple[int, ...]]: The current iteration number.
        """
        pass

    @abstractmethod
    def get_num_trees(self, *args, **kwargs) -> Union[int, Tuple[int, ...]]:
        """
        Retrieves the number of trees in the model.

        Returns:
            Union[int, Tuple[int, ...]]: Number of trees.
        """
        pass

    @abstractmethod
    def set_bias(self, bias: Union[np.ndarray, float], *args, **kwargs) -> None:
        """
        Sets the bias term for the model.

        Args:
            bias (Union[np.ndarray, float]): Bias value(s).
        """
        pass

    @abstractmethod
    def set_feature_weights(self, feature_weights: Union[np.ndarray, float], *args, **kwargs) -> None:
        """
        Sets the feature importance weights.

        Args:
            feature_weights (Union[np.ndarray, float]): Feature weights.
        """
        pass

    @abstractmethod
    def get_bias(self, *args, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Retrieves the model bias.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ...]]: Bias values.
        """
        pass

    @abstractmethod
    def get_feature_weights(self, *args, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Retrieves the feature importance weights.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ...]]: Feature weights.
        """
        pass

    @abstractmethod
    def get_device(self, *args, **kwargs) -> Union[str, Tuple[str, ...]]:
        """
        Retrieves the current device the model is running on.

        Returns:
            Union[str, Tuple[str, ...]]: Device (e.g., 'cpu', 'cuda').
        """
        pass

    @abstractmethod
    def print_tree(self, tree_idx: int, *args, **kwargs) -> None:
        """
        Prints the structure of a specific decision tree.

        Args:
            tree_idx (int): Index of the tree.
        """
        pass

    @abstractmethod
    def plot_tree(self, tree_idx: int, filename: str, *args, **kwargs) -> None:
        """
        Plots a decision tree and saves it to a file.

        Args:
            tree_idx (int): Index of the tree.
            filename (str): Path to save the tree visualization.
        """
        pass

    @abstractmethod
    def tree_shap(self, tree_idx: int, features: NumericalData, *args, **kwargs) -> \
            Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Computes SHAP values for a specific tree.

        Implementation based on - https://github.com/yupbank/linear_tree_shap
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192
        Args:
            tree_idx (int): tree index
            features (NumericalData):

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ...]]: shap values
        """
        pass

    @abstractmethod
    def shap(self, features: NumericalData, *args, **kwargs) -> \
            Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Computes SHAP values for the entire model.

        Uses Linear tree shap for each tree in the ensemble (sequentially)
        Implementation based on - https://github.com/yupbank/linear_tree_shap
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192
        Args:
            features (NumericalData):

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ...]]: shap values
        """
        pass

    @abstractmethod
    def set_device(self, device: Union[str, th.device], *args, **kwargs) -> None:
        """
        Sets the device the model should run on.

        Args:
            device (Union[str, th.device]): Target device.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Generates predictions using the trained model.

        Returns:
            np.ndarray: Model predictions.
        """
        pass

    @abstractmethod
    def distil(self, *args, **kwargs) -> Tuple[int, Dict]:
        """
        Distills the model into a smaller, simplified version.

        Returns:
            Tuple[int, Dict]: Final loss and updated parameters.
        """
        pass

    def copy(self):
        """
        Creates a copy of the model instance.

        Returns:
            BaseLearner: A copy of the model.
        """
        return self.__copy__()

    @abstractmethod
    def print_ensemble_metadata(self) -> None:
        """Prints metadata information about the entire ensemble."""
        pass

    @abstractmethod
    def __copy__(self) -> "BaseLearner":
        """Creates and returns a copy of the learner instance."""
        pass
