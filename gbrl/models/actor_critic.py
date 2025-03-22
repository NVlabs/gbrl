##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from typing import Dict, Tuple, Union, Optional
import os 
import numpy as np
import torch as th

from gbrl.gbrl_wrapper import GBTWrapper, SeparateActorCriticWrapper, SharedActorCriticWrapper
from gbrl.models.gbt import GBRL
from gbrl.utils import (setup_optimizer, clip_grad_norm, numerical_dtype, 
                    concatenate_arrays, validate_array, constant_like,
                    tensor_to_leaf)
from gbrl.constraints import Constraint


class ActorCritic(GBRL):
    def __init__(self, 
                 tree_struct: Dict,
                 input_dim: int,
                 output_dim: int, 
                 policy_optimizer: Dict,
                 value_optimizer: Dict= None,
                 shared_tree_struct: bool=True,
                 gbrl_params: Dict=dict(),
                 bias: np.ndarray = None,
                 verbose: int=0,
                 device: str='cpu',
                 constraints: Constraint = None):
        
        """ GBRL model for a shared Actor and Critic ensemble.

        Args:
         tree_struct (Dict): Dictionary containing tree structure information:
                max_depth (int): maximum tree depth.
                grow_policy (str): 'greedy' or 'oblivious'.
                n_bins (int): number of bins per feature for candidate generation.
                min_data_in_leaf (int): minimum number of samples in a leaf.
                par_th (int): minimum number of samples for parallelizing on CPU.
        output_dim (int): output dimension.
        policy_optimizer Dict: dictionary containing policy optimizer parameters (see GradientBoostingTrees for optimizer details).
        value_optimizer Dict: dictionary containing value optimizer parameters (see GradientBoostingTrees for optimizer details).
        shared_tree_struct (bool, optional): sharing actor and critic. Defaults to True.
        gbrl_params (Dict, optional): GBRL parameters such as:
            control_variates (bool): use control variates (variance reduction technique CPU only).
            split_score_func (str): "cosine" or "l2"
            generator_type- (str): candidate generation method "Quantile" or "Uniform".
            feature_weights - (list[float]): Per-feature multiplication weights used when choosing the best split. Weights should be >= 0
        bias (np.ndarray, optional): manually set a bias. Defaults to None = np.zeros.
        verbose (int, optional): verbosity level. Defaults to 0.
        device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """
        policy_optimizer = setup_optimizer(policy_optimizer, prefix='policy_')
        if value_optimizer is not None:
            value_optimizer = setup_optimizer(value_optimizer, prefix='value_')
        super().__init__(tree_struct,
                         input_dim,
                         output_dim,
                         None,
                         gbrl_params,
                         verbose,
                         device,
                         None)
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer

        self.shared_tree_struct = True if value_optimizer is None else shared_tree_struct
        self.bias = bias if bias is not None else np.zeros(self.output_dim if shared_tree_struct else self.output_dim - 1, dtype=numerical_dtype)
        # init model
        if self.shared_tree_struct:
            self._model = SharedActorCriticWrapper(self.input_dim, self.output_dim, self.tree_struct, self.policy_optimizer, self.value_optimizer, self.gbrl_params, self.verbose, self.device, constraints) 
            self._model.reset()
            self._model.set_bias(self.bias)
        else:
            self._model = SeparateActorCriticWrapper(self.input_dim, self.output_dim, self.tree_struct, self.policy_optimizer, self.value_optimizer, self.gbrl_params, self.verbose, self.device, constraints)
            self._model.reset()
            self._model.set_policy_bias(self.bias)
        self.policy_grad = None 
        self.value_grad = None
        
    @classmethod
    def load_model(cls, load_name: str, device: str) -> "ActorCritic":
        """Loads GBRL model from a file

        Args:
            load_name (str): full path to file name

        Returns:
            ActorCritic: loaded ActorCriticModel
        """
        policy_file = load_name + '_policy.gbrl_model'
        value_file = load_name + '_value.gbrl_model'

        instance = cls.__new__(cls)
        if os.path.isfile(policy_file) and os.path.isfile(value_file):
            instance._model = SeparateActorCriticWrapper.load(load_name, device)
            instance.shared_tree_struct = False
            instance.bias = instance._model.policy_model.get_bias() 
        else:
            instance._model = SharedActorCriticWrapper.load(load_name, device)
            instance.shared_tree_struct = True 
            instance.bias = instance._model.get_bias()
        instance.value_optimizer = instance._model.value_optimizer
        instance.policy_optimizer = instance._model.policy_optimizer
        instance.input_dim = instance._model.input_dim
        instance.output_dim = instance._model.output_dim
        instance.verbose = instance._model.verbose
        instance.tree_struct = instance._model.tree_struct
        instance.gbrl_params = instance._model.gbrl_params
        instance.device = instance._model.get_device()
        if isinstance(instance.device, tuple):
            instance.device = instance.device[0]
        return instance
    
    def get_num_trees(self) -> Union[int, Tuple[int, int]]:
        """ Returns number of trees in the ensemble.
        If separate actor and critic return number of trees per ensemble.
        Returns:
            Union[int, Tuple[int, int]]
        """
        return self._model.get_num_trees()
     
    def predict_values(self, observations: Union[np.ndarray, th.Tensor], requires_grad: bool = True, start_idx: int = 0, stop_idx: int = None, tensor: bool = True) -> Union[np.ndarray, th.Tensor]:
        """Predict only values. If `requires_grad=True` then stores 
           differentiable parameters in self.params 
           Return type/device is identical to the input type/device.

        Args:
            observations (Union[np.ndarray, th.Tensor])
            requires_grad (bool, optional). Defaults to True. Ignored if input is a numpy array.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Union[np.ndarray, th.Tensor]: values
        """
        values = self._model.predict_critic(observations, requires_grad, start_idx, stop_idx, tensor)
        if requires_grad:
            self.value_grad = None
            self.params = values
        return values

    def __call__(self, observations: Union[np.ndarray, th.Tensor], requires_grad: bool = True, start_idx: int = 0, stop_idx: int = None, tensor: bool = True) -> Tuple[Union[np.ndarray, th.Tensor], Union[np.ndarray, th.Tensor]]:
        """ Predicts  and returns actor and critic outputs as tensors. If `requires_grad=True` then stores 
           differentiable parameters in self.params 
           Return type/device is identical to the input type/device.
        Args:
            observations (Union[np.ndarray, th.Tensor])
            requires_grad (bool, optional). Defaults to True. Ignored if input is a numpy array.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Tuple[Union[np.ndarray, th.Tensor], Union[np.ndarray, th.Tensor]]: actor and critic output
        """
        params = self._model.predict(observations, requires_grad, start_idx, stop_idx, tensor)
        if requires_grad:
            self.policy_grad = None
            self.value_grad = None
            self.params = params
            self.input = observations
        return params
    
    def step(self, observations: Optional[Union[np.ndarray, th.Tensor]] = None, policy_grad: Optional[Union[np.ndarray, th.Tensor]] = None, value_grad: Optional[Union[np.ndarray, th.Tensor]] = None, policy_grad_clip: Optional[float] = None, value_grad_clip : Optional[float] = None) -> None:
        """Performs a boosting step for both actor and critic

        Args:
            observations (Union[np.ndarray, th.Tensor]):
            policy_grad_clip (float, optional): . Defaults to None.
            value_grad_clip (float, optional):. Defaults to None.
            policy_grad (Optional[Union[np.ndarray, th.Tensor]], optional): manually calculated gradients. Defaults to None.
            value_grad (Optional[Union[np.ndarray, th.Tensor]], optional): manually calculated gradients. Defaults to None.
        """
        if observations is None:
            assert self.input is not None, "Cannot update trees without input. Make sure model is called with requires_grad=True"
            observations = self.input
        n_samples = len(observations)

        policy_grad = policy_grad if policy_grad is not None else self.params[0].grad.detach() * n_samples
        value_grad = value_grad if value_grad is not None else self.params[1].grad.detach() * n_samples

        policy_grad = clip_grad_norm(policy_grad, policy_grad_clip)
        value_grad = clip_grad_norm(value_grad, value_grad_clip)

        validate_array(policy_grad)
        validate_array(value_grad)

        self._model.step(observations, policy_grad, value_grad)
        self.policy_grad = policy_grad
        self.value_grad = value_grad
        self.input = None
    
    def actor_step(self, observations: Optional[Union[np.ndarray, th.Tensor]] = None, policy_grad: Optional[Union[np.ndarray, th.Tensor]] = None, policy_grad_clip: Optional[float] = None) -> None:
        """Performs a single boosting step for the actor (should only be used if actor and critic use separate models)

        Args:
            observations (Union[np.ndarray, th.Tensor]):
            policy_grad_clip (float, optional): Defaults to None.
            policy_grad (Optional[Union[np.ndarray, th.Tensor]], optional): manually calculated gradients. Defaults to None.

        Returns:
            np.ndarray: policy gradient
        """
        assert not self.shared_tree_struct, "Cannot separate boosting steps for actor and critic when using separate tree architectures!"
        if observations is None:
            assert self.input is not None, "Cannot update trees without input. Make sure model is called with requires_grad=True"
            observations = self.input
        n_samples = len(observations)
        policy_grad = policy_grad if policy_grad is not None else self.params[0].grad.detach() * n_samples
        policy_grad = clip_grad_norm(policy_grad, policy_grad_clip)
        validate_array(policy_grad)

        self._model.step_policy(observations, policy_grad)
        self.policy_grad = policy_grad
    
    def critic_step(self, observations: Optional[Union[np.ndarray, th.Tensor]] = None, value_grad: Optional[Union[np.ndarray, th.Tensor]] = None, value_grad_clip: Optional[float] = None) -> None:
        """Performs a single boosting step for the critic (should only be used if actor and critic use separate models)

        Args:
            observations (Union[np.ndarray, th.Tensor]):
            value_grad_clip (float, optional): Defaults to None.
            value_grad (Optional[Union[np.ndarray, th.Tensor]], optional): manually calculated gradients. Defaults to None.

        Returns:
            np.ndarray: value gradient
        """
        assert not self.shared_tree_struct, "Cannot separate boosting steps for actor and critic when using separate tree architectures!"
        if observations is None:
            assert self.input is not None, "Cannot update trees without input. Make sure model is called with requires_grad=True"
            observations = self.input
        n_samples = len(observations)
        
        value_grad = value_grad if value_grad is not None else self.params[1].grad.detach() * n_samples
        value_grad = clip_grad_norm(value_grad, value_grad_clip)

        validate_array(value_grad)
        self._model.step_critic(observations, value_grad)
        self.value_grad = value_grad

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns predicted actor and critic parameters and their respective gradients

        Returns:
            Tuple[np.ndarray, np.ndarray]
        """
        assert self.params is not None, "must run a forward pass first"
        if isinstance(self.params, tuple):
            return (self.params[0].detach().cpu().numpy(),self.params[1].detach().cpu().numpy()) , (self.policy_grad, self.value_grad)
        return self.params, (self.policy_grad, self.value_grad)
    
    def copy(self) -> "ActorCritic":
        """Copy class instance 

        Returns:
            ActorCritic: copy of current instance
        """
        return self.__copy__()

    def __copy__(self) -> "ActorCritic":
        value_optimizer = None if self.value_optimizer is None else self.value_optimizer.copy()
        copy_ = ActorCritic(self.tree_struct.copy(), self.input_dim, self.output_dim, self.policy_optimizer.copy(), value_optimizer, self.shared_tree_struct, self.gbrl_params, self.bias, self.verbose, self.device)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_
