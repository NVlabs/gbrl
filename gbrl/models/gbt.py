##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from typing import Dict, List, Optional, Union

import numpy as np
import torch as th

from gbrl.learners.gbt_learner import GBTLearner
from gbrl.models.base import BaseGBT
from gbrl.common.utils import (NumericalData, clip_grad_norm, setup_optimizer,
                               validate_array)


class GBTModel(BaseGBT):
    """
    General class for gradient boosting trees

    """
    def __init__(self,
                 tree_struct: Dict,
                 input_dim: int,
                 output_dim: int,
                 optimizers: Union[Dict, List[Dict]],
                 params: Dict = dict(),
                 verbose: int = 0,
                 device: str = 'cpu'):
        """
        Initializes the GBT model.

        Args:
            tree_struct (Dict): Dictionary containing
            tree structure information:
                max_depth (int): maximum tree depth.
                grow_policy (str): 'greedy' or 'oblivious'.
                n_bins (int): number of bins per feature for candidate generation.
                min_data_in_leaf (int): minimum number of samples in a leaf.
                par_th (int): minimum number of samples for parallelizing on CPU.
            output_dim (int): output dimension.
            optimizers (Union[Dict, List[Dict]]): dictionary containing
            optimizer parameters or a list of dictionaries containing
            optimizer parameters.
                SGD optimizer must contain as:
                    'algo': 'SGD'
                    'lr' Union[float, str]: learning rate value. Constant learning rate is used by default
                Adam optimizer must contain (CPU only):
                    'algo' (str): 'Adam'
                    'lr' Union[float, str]: learning rate value. Constant learning rate is used by default
                    'beta_1' (float): 0.99
                    'beta_2' (float): 0.999
                    'eps' (float): 1.0e-8
                All optimizers must contain:
                    'start_idx'
                    'stop_idx'
                Setting scheduler type:
                Available schedulers are Constant and Linear. Constant is default, Linear is CPU only.
                To specify a linear scheduler, 3 additional arguments must be
                added to an optimizer dict.
                    'init_lr' (str): "lin_<value>"
                    'stop_lr' (float): minimum lr value
                    'T' (int): number of total expected boosting trees,
                               used to calculate the linear scheduling internally.
                               Can be manually calculated per algorithm
                               according to the total number of training steps.
             params (Dict, optional): GBRL parameters such as:
                control_variates (bool): use control variates (variance
                reduction technique CPU only).
                split_score_func (str): "cosine" or "l2"
                generator_type - (str): candidate generation method "Quantile" or "Uniform".
                feature_weights - (list[float]): Per-feature multiplication
                weights used when choosing the best split. Weights should be >= 0
            verbose (int, optional): verbosity level. Defaults to 0.
            device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """
        super().__init__()
        if optimizers is not None:
            if isinstance(optimizers, dict):
                optimizers = [optimizers]
            optimizers = [setup_optimizer(opt) for opt in optimizers]

        self.learner = GBTLearner(input_dim, output_dim,
                                  tree_struct, optimizers,
                                  params, verbose, device)
        self.learner.reset()
        self.grad = None
        self.input = None
        self.params = None

    def set_bias(self, bias: NumericalData) -> None:
        """Sets GBRL bias

        Args:
            bias (NumericalData)
        """
        if isinstance(bias, th.Tensor):
            bias = bias.clone().detach().cpu().numpy()
        # GBRL works with 2D numpy arrays.
        if len(bias.shape) == 1:
            bias = bias[:, np.newaxis]
        self.learner.set_bias(bias.astype(np.single))

    def set_bias_from_targets(self, targets: Union[np.ndarray,
                                                   th.Tensor]) -> None:
        """Sets bias as mean of targets

        Args:
            targets (NumericalData): Targets
        """
        if isinstance(targets, th.Tensor):
            arr = targets.clone().detach().cpu().numpy()
        else:
            arr = targets.copy()
        # GBRL works with 2D numpy arrays.
        if len(arr.shape) == 1:
            arr = arr[:, np.newaxis]
        mean_arr = np.mean(arr, axis=0)
        if isinstance(mean_arr, float):
            mean_arr = np.ndarray([mean_arr])
        # GBRL only works with float32
        self.learner.set_bias(mean_arr)

    def step(self,  X: Optional[NumericalData] = None,
             grad: Optional[NumericalData] = None,
             max_grad_norm: Optional[float] = None) -> None:
        """
        Perform a boosting step (fits a single tree on the gradients)

        Args:
            X (NumericalData): inputs
            max_grad_norm (float, optional): perform gradient clipping by norm. Defaults to None.
            grad (Optional[NumericalData], optional): manually calculated gradients. Defaults to None.
        """
        if X is None:
            assert self.input is not None, ("Cannot update trees without input."
                                            "Make sure model is called with requires_grad=True")
            X = self.input
        n_samples = len(X)
        grad = grad if grad is not None else self.params.grad.detach() * n_samples

        grad = clip_grad_norm(grad, max_grad_norm)
        validate_array(grad)
        self.learner.step(X, grad)
        self.grad = grad
        self.input = None

    def fit(self, X: NumericalData,
            targets: NumericalData,
            iterations: int, shuffle: bool = True,
            loss_type: str = 'MultiRMSE') -> float:
        """
        Fit multiple iterations (as in supervised learning)

        Args:
            X (NumericalData): inputs
            targets (NumericalData): targets
            iterations (int): number of boosting iterations
            shuffle (bool, optional): Shuffle dataset. Defaults to True.
            loss_type (str, optional): Loss to use (only MultiRMSE is currently implemented ). Defaults to 'MultiRMSE'.

        Returns:
            float: final loss over all examples.
        """
        return self.learner.fit(X, targets, iterations, shuffle, loss_type)

    @classmethod
    def load_learner(cls, load_name: str, device: str) -> "GBTModel":
        """Loads GBRL model from a file

        Args:
            load_name (str): full path to file name

        Returns:
            GBRL instance
        """
        instance = cls.__new__(cls)
        instance.learner = GBTLearner.load(load_name, device)
        instance.params = None
        return instance

    def __call__(self, X: Union[th.Tensor, np.ndarray],
                 requires_grad: bool = True, start_idx: int = 0,
                 stop_idx: int = None, tensor: bool = True) -> Union[th.Tensor, np.ndarray]:
        """
        Returns GBRL's output as either a Tensor or a numpy array. if
        `requires_grad=True` then stores differentiable parameters
        in self.params. Return type/device is identical to the input type/device.

        Args:
            X (Union[th.Tensor, np.ndarray]): Input
            requires_grad (bool, optional). Defaults to True.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction
            (uses all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Union[th.Tensor, np.ndarray]: Returns model predictions
        """
        y_pred = self.learner.predict(X, requires_grad, start_idx, stop_idx, tensor)
        if requires_grad:
            self.grad = None
            self.params = y_pred
            self.input = X
        return y_pred

    def print_tree(self, tree_idx: int) -> None:
        """Prints tree information

        Args:
            tree_idx (int): tree index to print
        """
        self.learner.print_tree(tree_idx)

    def plot_tree(self, tree_idx: int, filename: str) -> None:
        """Plots tree using (only works if GBRL was compiled with graphviz)

        Args:
            tree_idx (int): tree index to plot
            filename (str): .png filename to save
        """
        self.learner.plot_tree(tree_idx, filename)

    def copy(self) -> "GBTModel":
        """Copy class instance

        Returns:
            GradientBoostingTrees: copy of current instance. The actual type
            will be the type of the subclass that calls this method.
        """
        return self.__copy__()

    def __copy__(self) -> "GBTModel":
        learner = self.learner.copy()
        copy_ = GBTModel(learner.tree_struct, learner.input_dim,
                         learner.output_dim, learner.optimizers,
                         learner.params, learner.verbose, learner.device)
        copy_.learner = learner
        return copy_
