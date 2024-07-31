##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from typing import Dict, List, Union, Tuple, Optional

import numpy as np
import torch as th


from .gbrl_wrapper import GBTWrapper
from .utils import setup_optimizer, clip_grad_norm


class GBRL:
    def __init__(self, 
                 tree_struct: Dict,
                 output_dim: int,
                 optimizer: Union[Dict, List[Dict]],
                 gbrl_params: Dict = dict(),
                 verbose: int=0,
                 device: str = 'cpu'):
        """General class for gradient boosting trees

        Args:
            tree_struct (Dict): Dictionary containing tree structure information:
                max_depth (int): maximum tree depth.
                grow_policy (str): 'greedy' or 'oblivious'.
                n_bins (int): number of bins per feature for candidate generation.
                min_data_in_leaf (int): minimum number of samples in a leaf.
                par_th (int): minimum number of samples for parallelizing on CPU.
            output_dim (int): output dimension.
            optimizer (Union[Dict, List[Dict]]): dictionary containing optimizer parameters or a list of dictionaries containing optimizer parameters.
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
                To specify a linear scheduler, 3 additional arguments must be added to an optimizer dict.
                    'init_lr' (str): "lin_<value>"
                    'stop_lr' (float): minimum lr value 
                    'T' (int): number of total expected boosting trees, 
                               used to calculate the linear scheduling internally.
                               Can be manually calculated per algorithm according 
                               to the total number of training steps.
             gbrl_params (Dict, optional): GBRL parameters such as:
                control_variates (bool): use control variates (variance reduction technique CPU only).
                split_score_func (str): "cosine" or "l2"
                generator_type - (str): candidate generation method "Quantile" or "Uniform".
                feature_weights - (list[float]): Per-feature multiplication weights used when choosing the best split. Weights should be >= 0
            verbose (int, optional): verbosity level. Defaults to 0.
            device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """
        if optimizer is not None:
            if isinstance(optimizer, dict):
                    optimizer = [optimizer]
            optimizer = [setup_optimizer(opt) for opt in optimizer]

        self.optimizer = optimizer
        self.output_dim = output_dim
        self.verbose = verbose
        self.tree_struct = tree_struct
        self._model = None 
        self.gbrl_params = gbrl_params
        self.device = device
        self.params = None
        if type(self) is GBRL:
            self._model = GBTWrapper(self.output_dim, self.tree_struct, self.optimizer, self.gbrl_params, self.verbose, self.device)
            self._model.reset()
        self.grad = None

    def reset_params(self):
        """Resets param attributes
        """
        self.params = None

    def set_bias(self, bias: Union[np.array, th.Tensor]):
        """Sets GBRL bias

        Args:
            y (Union[np.array, th.Tensor]): _description_
        """
        if isinstance(bias, th.Tensor):
            bias = bias.clone().detach().cpu().numpy()
        # GBRL works with 2D numpy arrays.
        if len(bias.shape) == 1:
            bias = bias[:, np.newaxis]
        self._model.set_bias(bias.astype(np.single))
    
    def set_bias_from_targets(self, targets: Union[np.array, th.Tensor]):
        """Sets bias as mean of targets

        Args:
            targets (Union[np.array, th.Tensor]): Targets
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
            mean_arr = np.array([mean_arr])
        self._model.set_bias(mean_arr.astype(np.single)) # GBRL only works with float32
    
    def get_iteration(self) -> int:
        """
        Returns:
            int : number of boosting iterations
        """
        return self._model.get_iteration()

    def get_total_iterations(self) -> int:
        """
        Returns:
            int: total number of boosting iterations 
            (sum of actor and critic if they are not shared otherwise equals 
            get_iteration())
        """
        return self._model.get_total_iterations()

    def get_schedule_learning_rates(self) -> Tuple[float, float]:
        """
        Gets learning rate values for optimizers according to schedule of ensemble.
        Constant schedule - no change in values.
        Linear schedule - learning rate value accordign to number of trees in the ensemble.
        Returns:
            Tuple[float, float]: learning rate schedule per optimizer.
        """
        return self._model.get_schedule_learning_rates()

    def step(self,  X: Union[np.array, th.Tensor], max_grad_norm: float = None, grad: Optional[Union[np.array, th.tensor]] = None) -> None:
        """Perform a boosting step (fits a single tree on the gradients)

        Args:
            X (Union[np.array, th.Tensor]): inputs
            max_grad_norm (float, optional): perform gradient clipping by norm. Defaults to None.
            grad (Optional[Union[np.array, th.tensor]], optional): manually calculated gradients. Defaults to None.
        """
        if grad is None:
            assert self.params is not None, "must run a forward pass first"
            n_samples = len(X)
            grad = self.params.grad.detach().cpu().numpy() * n_samples

        grad = clip_grad_norm(grad, max_grad_norm)
        self._model.step(X, grad)
        self.grad = grad
        
    def get_params(self) -> Tuple[np.array, np.array]:
        """Returns predicted model parameters and their respective gradients

        Returns:
            Tuple[np.array, np.array]
        """
        assert self.params is not None, "must run a forward pass first"
        params = self.params
        if isinstance(self.params, tuple):
            params = (params[0].detach().cpu().numpy(), params[1].detach().cpu().numpy()) 
        return params, self.grad

    def predict(self, X: np.array, start_idx:int =0, stop_idx: int=None) -> np.array:
        """Predict 

        Args:
            x (np.array): inputs
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses all trees in the ensemble if set to 0). Defaults to None.

        Returns:
            np.narray: prediction
        """
        return self._model.predict(X, start_idx, stop_idx)
    
    def fit(self, X: Union[np.array, th.tensor], targets: Union[np.array, th.tensor], iterations: int, shuffle: bool=True, loss_type: str='MultiRMSE') -> float:
        """Fit multiple iterations (as in supervised learning)

        Args:
            x (np.array): inputs
            targets (np.array): targets
            iterations (int): number of boosting iterations
            shuffle (bool, optional): Shuffle dataset. Defaults to True.
            loss_type (str, optional): Loss to use (only MultiRMSE is currently implemented ). Defaults to 'MultiRMSE'.

        Returns:
            float: final loss over all examples.
        """
        return self._model.fit(X, targets, iterations, shuffle, loss_type)
    
    def get_num_trees(self) -> int:
        """
        Returns number of trees in the ensemble

        Returns:
            int: number of trees in the ensemble
        """
        return self._model.get_num_trees()
    
    def tree_shap(self, tree_idx: int, features: Union[np.array, th.Tensor]) -> Union[np.array, Tuple[np.array, np.array]]:
        """Calculates SHAP values for a single tree
            Implementation based on - https://github.com/yupbank/linear_tree_shap
            See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192 

        Args:
            tree_idx (int): tree index
            features (Union[np.array, th.Tensor])

        Returns:
            Union[np.array, Tuple[np.array, np.array]]: SHAP values of shap [n_samples, number of input features, number of outputs]. The output is a tuple of SHAP values per model only in the case of a separate actor-critic model.
        """
        return self._model.tree_shap(tree_idx, features)
    
    def shap(self, features: Union[np.array, th.Tensor]) -> Union[np.array, Tuple[np.array, np.array]]:
        """Calculates SHAP values for the entire ensemble
            Implementation based on - https://github.com/yupbank/linear_tree_shap
            See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192 
            
        Args:
            features (Union[np.array, th.Tensor])

        Returns:
            Union[np.array, Tuple[np.array, np.array]]: SHAP values of shap [n_samples, number of input features, number of outputs]. The output is a tuple of SHAP values per model only in the case of a separate actor-critic model.
        """

        return self._model.shap(features)

    def save_model(self, save_path: str) -> None:
        """
        Saves model to file

        Args:
            filename (str): Absolute path and name of save filename.
        """
        self._model.save(save_path) 

    def export_model(self, filename: str, modelname: str = None) -> None:
        """
        Exports model as a C-header file

        Args:
            filename (str): Absolute path and name of exported filename.
        """
        self._model.export(filename, modelname) 

    @classmethod
    def load_model(cls, load_name: str) -> "GBRL":
        """Loads GBRL model from a file

        Args:
            load_name (str): full path to file name

        Returns:
            GBRL instance 
        """
        instance = cls.__new__(cls)
        instance._model = GBTWrapper.load(load_name)
        instance.optimizer =  instance._model.optimizer
        instance.output_dim = instance._model.output_dim
        instance.verbose = instance._model.verbose
        instance.tree_struct = instance._model.tree_struct
        instance.gbrl_params = instance._model.gbrl_params
        instance.device = instance._model.get_device()
        return instance

    def set_device(self, device: str):
        """Sets GBRL device (either cpu or cuda)

        Args:
            device (str): choices are ['cpu', 'cuda']
        """
        assert device in ['cpu', 'cuda'], "device must be in ['cpu', 'cuda']"
        self._model.set_device(device)
        self.device = device

    def get_device(self) -> Union[str, Tuple[str, str]]:
        """Returns GBRL device/devices (if multiple GBRL models)

        Returns:
            Union[str, Tuple[str, str]]: GBRL device per model
        """
        return self._model.get_device()

    def __call__(self, X: np.array, requires_grad: bool = True) -> th.Tensor:
        """Returns GBRL's output as Tensor. if `requires_grad=True` then stores 
           differentiable parameters in self.params. 

        Args:
            X (np.array): Input
            requires_grad (bool, optional). Defaults to True.

        Returns:
            th.Tensor: _description_
        """
        y_pred = self.predict(X)
        params = th.tensor(y_pred, requires_grad=requires_grad)
        if requires_grad:
            self.grad = None
            self.params = params
        return params

    def print_tree(self, tree_idx: int) -> None:
        """Prints tree information

        Args:
            tree_idx (int): tree index to print
        """
        self._model.print_tree(tree_idx)

    def plot_tree(self, tree_idx: int, filename: str) -> None:
        """Plots tree using (only works if GBRL was compiled with graphviz)

        Args:
            tree_idx (int): tree index to plot
            filename (str): .png filename to save
        """
        self._model.plot_tree(tree_idx, filename)
    
    def copy(self) -> "GBRL":
        """Copy class instance 

        Returns:
            GradientBoostingTrees: copy of current instance. The actual type will be the type
            of the subclass that calls this method.
        """
        return self.__copy__()
    
    def __copy__(self) -> "GBRL":
        copy_ = GBRL(self.tree_struct.copy(), self.output_dim, self.optimizer.copy(), self.gbrl_params, self.verbose)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_


if __name__ == '__main__':
    pass