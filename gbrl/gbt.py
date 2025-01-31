##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from typing import Dict, List, Union, Tuple, Optional, Any

import numpy as np
import torch as th


from gbrl.gbrl_wrapper import GBTWrapper
from gbrl.utils import setup_optimizer, clip_grad_norm, validate_array


class GBRL:
    def __init__(self, 
                 tree_struct: Dict,
                 input_dim: int,
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
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.verbose = verbose
        self.tree_struct = tree_struct
        self._model = None 
        self.gbrl_params = gbrl_params
        self.device = device
        self.params = None
        if type(self) is GBRL:
            self._model = GBTWrapper(self.input_dim, self.output_dim, self.tree_struct, self.optimizer, self.gbrl_params, self.verbose, self.device)
            self._model.reset()
        self.grad = None
        self.input = None

    def reset_params(self):
        """Resets param attributes
        """
        self.params = None

    def set_bias(self, bias: Union[np.ndarray, th.Tensor]):
        """Sets GBRL bias

        Args:
            bias (Union[np.ndarray, th.Tensor])
        """
        if isinstance(bias, th.Tensor):
            bias = bias.clone().detach().cpu().numpy()
        # GBRL works with 2D numpy arrays.
        if len(bias.shape) == 1:
            bias = bias[:, np.newaxis]
        self._model.set_bias(bias.astype(np.single))

    def set_feature_weights(self, feature_weights: Union[np.ndarray, th.Tensor]):
        """Sets GBRL feature_weights

        Args:
            feature_weights (Union[np.ndarray, th.Tensor])
        """
        if isinstance(feature_weights, th.Tensor):
            feature_weights = feature_weights.clone().detach().cpu().numpy()
        # GBRL works with 2D numpy arrays.
        if len(feature_weights.shape) == 1:
            feature_weights = feature_weights[:, np.newaxis]
        self._model.set_feature_weights(feature_weights.astype(np.single))
    
    def set_bias_from_targets(self, targets: Union[np.ndarray, th.Tensor]):
        """Sets bias as mean of targets

        Args:
            targets (Union[np.ndarray, th.Tensor]): Targets
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

    def step(self,  X: Optional[Union[np.ndarray, th.Tensor]] = None, grad: Optional[Union[np.ndarray, th.Tensor]] = None, max_grad_norm: Optional[float] = None) -> None:
        """Perform a boosting step (fits a single tree on the gradients)

        Args:
            X (Union[np.ndarray, th.Tensor]): inputs
            max_grad_norm (float, optional): perform gradient clipping by norm. Defaults to None.
            grad (Optional[Union[np.ndarray, th.Tensor]], optional): manually calculated gradients. Defaults to None.
        """
        if X is None:
            assert self.input is not None, "Cannot update trees without input. Make sure model is called with requires_grad=True"
            X = self.input
        n_samples = len(X)
        grad = grad if grad is not None else self.params.grad.detach() * n_samples

        grad = clip_grad_norm(grad, max_grad_norm)
        validate_array(grad)
        self._model.step(X, grad)
        self.grad = grad
        self.input = None
        
    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns predicted model parameters and their respective gradients

        Returns:
            Tuple[np.ndarray, np.ndarray]
        """
        assert self.params is not None, "must run a forward pass first"
        params = self.params
        if isinstance(self.params, tuple):
            params = (params[0].detach().cpu().numpy(), params[1].detach().cpu().numpy()) 
        return params, self.grad
    
    def fit(self, X: Union[np.ndarray, th.Tensor], targets: Union[np.ndarray, th.Tensor], iterations: int, shuffle: bool = True, loss_type: str = 'MultiRMSE') -> float:
        """Fit multiple iterations (as in supervised learning)

        Args:
            X (Union[np.ndarray, th.Tensor]): inputs
            targets (Union[np.ndarray, th.Tensor]): targets
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
    
    def tree_shap(self, tree_idx: int, features: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Calculates SHAP values for a single tree
            Implementation based on - https://github.com/yupbank/linear_tree_shap
            See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192 

        Args:
            tree_idx (int): tree index
            features (Union[np.ndarray, th.Tensor])

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: SHAP values of shap [n_samples, number of input features, number of outputs]. The output is a tuple of SHAP values per model only in the case of a separate actor-critic model.
        """
        return self._model.tree_shap(tree_idx, features)
    
    def shap(self, features: Union[np.ndarray, th.Tensor]) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Calculates SHAP values for the entire ensemble
            Implementation based on - https://github.com/yupbank/linear_tree_shap
            See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192 
            
        Args:
            features (Union[np.ndarray, th.Tensor])

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: SHAP values of shap [n_samples, number of input features, number of outputs]. The output is a tuple of SHAP values per model only in the case of a separate actor-critic model.
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
    def load_model(cls, load_name: str, device: str) -> "GBRL":
        """Loads GBRL model from a file

        Args:
            load_name (str): full path to file name

        Returns:
            GBRL instance 
        """
        instance = cls.__new__(cls)
        instance._model = GBTWrapper.load(load_name, device)
        instance.optimizer =  instance._model.optimizer
        instance.input_dim = instance._model.input_dim
        instance.output_dim = instance._model.output_dim
        instance.verbose = instance._model.verbose
        instance.tree_struct = instance._model.tree_struct
        instance.gbrl_params = instance._model.gbrl_params
        instance.device = instance._model.get_device()
        instance._model.device = instance.device
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

    def __call__(self, X: Union[th.Tensor, np.ndarray], requires_grad: bool = True, start_idx: int = 0, stop_idx: int = None, tensor: bool = True) -> Union[th.Tensor, np.ndarray]:
        """Returns GBRL's output as either a Tensor or a numpy array. if `requires_grad=True` then stores 
           differentiable parameters in self.params. 
           Return type/device is identical to the input type/device.

        Args:
            X (Union[th.Tensor, np.ndarray]): Input
            requires_grad (bool, optional). Defaults to True.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Union[th.Tensor, np.ndarray]: Returns model predictions
        """
        y_pred = self._model.predict(X, requires_grad, start_idx, stop_idx, tensor)
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
        copy_ = GBRL(self.tree_struct.copy(), self.input_dim, self.output_dim, self.optimizer.copy(), self.gbrl_params, self.verbose)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_


if __name__ == '__main__':
    pass