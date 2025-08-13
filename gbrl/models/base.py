##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union

import numpy as np
import torch as th

from gbrl.common.utils import NumericalData
from gbrl.learners.gbt_learner import GBTLearner


class BaseGBT(ABC):
    def __init__(self):
        """General class for gradient boosting trees
        """
        self.learner: Optional[GBTLearner] = None
        self.grad: Optional[NumericalData] = None
        self.params: Optional[NumericalData] = None
        self.input: Optional[NumericalData] = None

    def set_bias(self, *args, **kwargs) -> None:
        """Sets GBRL bias

        Args:
            *args: Variable length argument list (implementation dependent)
            **kwargs: Arbitrary keyword arguments (implementation dependent)
        """
        raise NotImplementedError

    def set_feature_weights(self, feature_weights: NumericalData) -> None:
        """Sets GBRL feature weights

        Args:
            feature_weights (NumericalData): Feature weights to set
        """
        assert self.learner is not None, "Learner is not initialized"

        if isinstance(feature_weights, th.Tensor):
            feature_weights = feature_weights.clone().detach().cpu().numpy()
        # GBRL works with 2D numpy arrays.
        if len(feature_weights.shape) == 1:
            feature_weights = feature_weights[:, np.newaxis]
        self.learner.set_feature_weights(feature_weights.astype(np.single))

    def get_iteration(self) -> Union[int, Tuple[int, ...]]:
        """Gets the number of boosting iterations per learner

        Returns:
            Union[int, Tuple[int, ...]]: Number of boosting iterations per learner
        """
        assert self.learner is not None, "Learner is not initialized"
        return self.learner.get_iteration()

    def get_total_iterations(self) -> int:
        """Gets the total number of boosting iterations

        Returns:
            int: Total number of boosting iterations
                (sum of actor and critic if they are not shared, otherwise equals get_iteration())
        """
        assert self.learner is not None, "Learner is not initialized"
        return self.learner.get_total_iterations()

    def get_schedule_learning_rates(self) -> Union[float, Tuple[float, ...]]:
        """Gets learning rate values for optimizers according to schedule of ensemble

        Constant schedule - no change in values.
        Linear schedule - learning rate value according to number of trees in the ensemble.

        Returns:
            Union[float, Tuple[float, ...]]: Learning rate schedule per optimizer
        """
        assert self.learner is not None, "Learner is not initialized"
        return self.learner.get_schedule_learning_rates()

    @abstractmethod
    def step(self, *args, **kwargs) -> None:
        """Performs a boosting step (fits a single tree on the gradients)

        Args:
            *args: Variable length argument list (implementation dependent)
            **kwargs: Arbitrary keyword arguments (implementation dependent)
        """
        pass

    def fit(self, *args, **kwargs) -> Union[float, Tuple[float, ...]]:
        """Fits multiple iterations (as in supervised learning)

        Args:
            *args: Variable length argument list (implementation dependent)
            **kwargs: Arbitrary keyword arguments (implementation dependent)

        Returns:
            Union[float, Tuple[float, ...]]: Final loss per learner over all examples
        """
        raise NotImplementedError

    def get_num_trees(self, *args, **kwargs) -> Union[int, Tuple[int, ...]]:
        """Gets the number of trees in the ensemble per learner

        Args:
            *args: Variable length argument list (implementation dependent)
            **kwargs: Arbitrary keyword arguments (implementation dependent)

        Returns:
            Union[int, Tuple[int, ...]]: Number of trees in the ensemble per learner
        """
        assert self.learner is not None, "Learner is not initialized"
        return self.learner.get_num_trees(*args, **kwargs)

    def tree_shap(self, tree_idx: int, features: NumericalData, *args, **kwargs) -> \
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Calculates SHAP values for a single tree

        Implementation based on - https://github.com/yupbank/linear_tree_shap.
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192.

        Args:
            tree_idx (int): Tree index
            features (NumericalData): Input features
            *args: Variable length argument list (implementation dependent)
            **kwargs: Arbitrary keyword arguments (implementation dependent)

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: SHAP values of shape
                [n_samples, number of input features, number of outputs]. The output
                is a tuple of SHAP values per model only in the case of a separate
                actor-critic model.
        """
        assert self.learner is not None, "Learner is not initialized"
        return self.learner.tree_shap(tree_idx, features, *args, **kwargs)

    def shap(self, features: NumericalData, *args, **kwargs) -> \
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Calculates SHAP values for the entire ensemble

        Implementation based on - https://github.com/yupbank/linear_tree_shap.
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192.

        Args:
            features (NumericalData): Input features
            *args: Variable length argument list (implementation dependent)
            **kwargs: Arbitrary keyword arguments (implementation dependent)

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: SHAP values of shape
                [n_samples, number of input features, number of outputs]. The output
                is a tuple of SHAP values per model only in the case of a separate
                actor-critic model.
        """
        assert self.learner is not None, "Learner is not initialized"
        return self.learner.shap(features, *args, **kwargs)

    def save_learner(self, save_path: str) -> None:
        """Saves model to file

        Args:
            save_path (str): Absolute path and name of save filename
        """
        assert self.learner is not None, "Learner is not initialized"
        self.learner.save(save_path)

    def export_learner(self, filename: str, modelname: Optional[str] = None) -> None:
        """Exports learner model as a C-header file

        Args:
            filename (str): Absolute path and name of exported filename
            modelname (Optional[str], optional): Model name for export. Defaults to None.
        """
        assert self.learner is not None, "Learner is not initialized"
        self.learner.export(filename, modelname)

    @classmethod
    def load_learner(cls, load_name: str, device: str) -> "BaseGBT":
        """Loads a BaseGBT model from a file

        Args:
            load_name (str): Full path to the saved model file
            device (str): Device to load the model onto ('cpu' or 'cuda')

        Returns:
            BaseGBT: Loaded BaseGBT model instance
        """
        instance = cls.__new__(cls)
        instance.learner = GBTLearner.load(load_name, device)
        instance.grad = None
        instance.params = None
        instance.input = None
        return instance

    def get_params(self) -> Tuple[Union[NumericalData, Tuple[NumericalData, ...]],
                                  Optional[Union[NumericalData, Tuple[NumericalData, ...]]]]:
        """Gets predicted model parameters and their respective gradients

        Returns:
            Tuple[Union[NumericalData, Tuple[NumericalData, ...]],
                  Optional[Union[NumericalData, Tuple[NumericalData, ...]]]]:
                Model parameters and their gradients
        """
        assert self.params is not None, "must run a forward pass first"
        params = self.params
        if isinstance(self.params, tuple):
            params = (params[0].detach().cpu().numpy(), params[1].detach().cpu().numpy())
        return params, self.grad

    def set_device(self, device: str):
        """Sets GBRL device (either cpu or cuda)

        Args:
            device (str): Device choice, must be in ['cpu', 'cuda']
        """
        assert device in ['cpu', 'cuda'], "device must be in ['cpu', 'cuda']"
        assert self.learner is not None, "Learner is not initialized"
        self.learner.set_device(device)

    def get_device(self) -> Union[str, Tuple[str, str]]:
        """Gets GBRL device/devices per learner

        Returns:
            Union[str, Tuple[str, str]]: GBRL device per model
        """
        assert self.learner is not None, "Learner is not initialized"
        return self.learner.get_device()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Union[NumericalData, Tuple[NumericalData, ...]]:
        """Returns GBRL's output as either a Tensor or a numpy array per learner

        Args:
            *args: Variable length argument list (implementation dependent)
            **kwargs: Arbitrary keyword arguments (implementation dependent)

        Returns:
            Union[NumericalData, Tuple[NumericalData, ...]]: Model output per learner
        """
        pass

    def print_tree(self, tree_idx: int, *args, **kwargs) -> None:
        """Prints tree information

        Args:
            tree_idx (int): Tree index to print
            *args: Variable length argument list (implementation dependent)
            **kwargs: Arbitrary keyword arguments (implementation dependent)
        """
        assert self.learner is not None, "Learner is not initialized"
        self.learner.print_tree(tree_idx, *args, **kwargs)

    def plot_tree(self, tree_idx: int, filename: str, *args, **kwargs) -> None:
        """Plots tree using graphviz (only works if GBRL was compiled with graphviz)

        Args:
            tree_idx (int): Tree index to plot
            filename (str): .png filename to save
            *args: Variable length argument list (implementation dependent)
            **kwargs: Arbitrary keyword arguments (implementation dependent)
        """
        assert self.learner is not None, "Learner is not initialized"
        self.learner.plot_tree(tree_idx, filename, *args, **kwargs)

    def copy(self) -> "BaseGBT":
        """Creates a copy of the class instance

        Returns:
            BaseGBT: Copy of the current instance
        """
        return self.__copy__()

    @abstractmethod
    def __copy__(self) -> "BaseGBT":
        """Copy constructor

        Returns:
            BaseGBT: Copy of the current instance
        """
        pass
