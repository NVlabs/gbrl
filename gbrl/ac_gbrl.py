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
from gbrl.gbt import GBRL
from gbrl.utils import (setup_optimizer, clip_grad_norm, numerical_dtype, 
                    concatenate_arrays, validate_array, constant_like,
                    tensor_to_leaf)


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
                 device: str='cpu'):
        
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
                         device)
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer

        self.shared_tree_struct = True if value_optimizer is None else shared_tree_struct
        self.bias = bias if bias is not None else np.zeros(self.output_dim if shared_tree_struct else self.output_dim - 1, dtype=numerical_dtype)
        # init model
        if self.shared_tree_struct:
            self._model = SharedActorCriticWrapper(self.input_dim, self.output_dim, self.tree_struct, self.policy_optimizer, self.value_optimizer, self.gbrl_params, self.verbose, self.device) 
            self._model.reset()
            self._model.set_bias(self.bias)
        else:
            self._model = SeparateActorCriticWrapper(self.input_dim, self.output_dim, self.tree_struct, self.policy_optimizer, self.value_optimizer, self.gbrl_params, self.verbose, self.device)
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
    
class ParametricActor(GBRL):
    def __init__(self, 
                 tree_struct: Dict,
                 input_dim: int, 
                 output_dim: int, 
                 policy_optimizer: Dict,
                 gbrl_params: Dict=dict(),
                 bias: np.ndarray = None,
                 verbose: int=0,
                 device: str='cpu'):
        """ GBRL model for a ParametricActor ensemble. ParametricActor outputs a single parameter per action dimension.
            Therefore it can be determinstic or stochastic (e.g. Discrete action space).

        Args:
         tree_struct (Dict): Dictionary containing tree structure information:
                max_depth (int): maximum tree depth.
                grow_policy (str): 'greedy' or 'oblivious'.
                n_bins (int): number of bins per feature for candidate generation.
                min_data_in_leaf (int): minimum number of samples in a leaf.
                par_th (int): minimum number of samples for parallelizing on CPU.
        output_dim (int): output dimension.
        policy_optimizer Dict: dictionary containing policy optimizer parameters (see GradientBoostingTrees for optimizer details).
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
        super().__init__(tree_struct,
                         input_dim,
                         output_dim,
                         None,
                         gbrl_params,
                         verbose,
                         device)
        self.policy_optimizer = policy_optimizer
        self.bias = bias if bias is not None else np.zeros(self.output_dim, dtype=numerical_dtype)
        # init model
        self._model = GBTWrapper(self.output_dim, self.tree_struct, self.policy_optimizer, self.gbrl_params, self.verbose, self.device) 
        self._model.reset()
        self._model.set_bias(self.bias)

    def step(self, observations: Optional[Union[np.ndarray, th.Tensor]] = None, policy_grad: Optional[Union[np.ndarray, th.Tensor]] = None, policy_grad_clip: Optional[float] = None,) -> None:
        """Performs a single boosting iteration.

        Args:
            observations (Union[np.ndarray, th.Tensor]):
            policy_grad_clip (float, optional): . Defaults to None.
            policy_grad (Optional[Union[np.ndarray, th.Tensor]], optional): manually calculated gradients. Defaults to None.
        """
        if observations is None:
            assert self.input is not None, "Cannot update trees without input. Make sure model is called with requires_grad=True"
            observations = self.input
        n_samples = len(observations)
            
        policy_grad = policy_grad if policy_grad is not None else self.params.grad.detach() * n_samples
        policy_grad = clip_grad_norm(policy_grad, policy_grad_clip)
        validate_array(policy_grad)

        self._model.step(observations, policy_grad)
        self.grad = policy_grad
        self.input = None

    @classmethod
    def load_model(cls, load_name: str, device: str) -> "ParametricActor":
        """Loads GBRL model from a file

        Args:
            load_name (str): full path to file name

        Returns:
            ParametricActor: loaded ActorCriticModel
        """
        instance = cls.__new__(cls)
        instance._model = GBTWrapper.load(load_name, device)
        instance.bias = instance._model.get_bias()
        instance.policy_optimizer = instance._model.optimizer
        instance.input_dim = instance._model.input_dim
        instance.output_dim = instance._model.output_dim
        instance.verbose = instance._model.verbose
        instance.tree_struct = instance._model.tree_struct
        instance.gbrl_params = instance._model.gbrl_params
        instance.device = instance._model.get_device()
        return instance
    
    def get_num_trees(self) -> int:
        """ Returns number of trees in the ensemble.
        Returns:
            int
        """
        return self._model.get_num_trees()
     
    def __call__(self, observations: Union[np.ndarray, th.Tensor], requires_grad : bool = True, start_idx: int = 0, stop_idx: int = None, tensor: bool = True) -> Union[np.ndarray, th.Tensor]:
        """ Returns actor output as Tensor. If `requires_grad=True` then stores 
           differentiable parameters in self.params 
           Return type/device is identical to the input type/device.
           Requires_grad is ignored if input is a numpy array.
        Args:
            observations (Union[np.ndarray, th.Tensor])
            requires_grad bool: Defaults to None.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Union[np.ndarray, th.Tensor]: GBRL outputs - a single parameter per action dimension.
        """
        params = self._model.predict(observations, requires_grad, start_idx, stop_idx, tensor)
        if requires_grad:
            self.grads = None
            self.params = params
            self.input = observations
        return params
    
    def __copy__(self) -> "ParametricActor":
        copy_ = ParametricActor(self.tree_struct.copy(), self.input_dim, self.output_dim, self.policy_optimizer.copy(),  self.gbrl_params, self.bias, self.verbose, self.device)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_
    
class GaussianActor(GBRL):
    def __init__(self, 
                 tree_struct: Dict,
                 input_dim: int,
                 output_dim: int, 
                 mu_optimizer: Dict,
                 std_optimizer: Dict = None,
                 log_std_init: float = -2,
                 gbrl_params: Dict = dict(),
                 bias: np.ndarray = None,
                 verbose: int=0,
                 device: str='cpu'):
        """ GBRL model for a Actor ensemble used in algorithms such as: SAC.
        Model outputs mu and log_std of a Gaussian distribution. 
        Actor optimizer can be shared for both parameters or separate. 
        
        Args:
         tree_struct (Dict): Dictionary containing tree structure information:
                max_depth (int): maximum tree depth.
                grow_policy (str): 'greedy' or 'oblivious'.
                n_bins (int): number of bins per feature for candidate generation.
                min_data_in_leaf (int): minimum number of samples in a leaf.
                par_th (int): minimum number of samples for parallelizing on CPU.
        output_dim (int): output dimension.
        mu_optimizer Dict: dictionary containing Gaussian mean optimizer parameters. (see GradientBoostingTrees for optimizer details)
        std_optimizer Dict: dictionary containing Gaussian sigma optimizer parameters. (see GradientBoostingTrees for optimizer details)
        log_std_init (float): initial value of log_std
        gbrl_params (Dict, optional): GBRL parameters such as:
            control_variates (bool): use control variates (variance reduction technique CPU only).
            split_score_func (str): "cosine" or "l2"
            generator_type- (str): candidate generation method "Quantile" or "Uniform".
            feature_weights - (list[float]): Per-feature multiplication weights used when choosing the best split. Weights should be >= 0
        bias (np.ndarray, optional): manually set a bias. Defaults to None = np.zeros.
        verbose (int, optional): verbosity level. Defaults to 0.
        device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """
        mu_optimizer = setup_optimizer(mu_optimizer, prefix='mu_')

        self.bias = bias if bias is not None else np.zeros(output_dim, dtype=numerical_dtype)

        policy_dim = output_dim
        if std_optimizer is not None:
            std_optimizer = setup_optimizer(std_optimizer, prefix='std_')
            policy_dim = output_dim // 2
            self.bias[policy_dim:] = log_std_init*np.ones(policy_dim, dtype=numerical_dtype)
        
        self.mu_optimizer = mu_optimizer
        self.std_optimizer = std_optimizer
        self.log_std_init = log_std_init

        super().__init__(tree_struct,
                         input_dim,
                         output_dim,
                         None, 
                         gbrl_params,
                         verbose,
                         device)

        
        self.policy_dim = policy_dim
        # init model
        self._model = GBTWrapper(self.input_dim, self.output_dim, self.tree_struct, [mu_optimizer, std_optimizer], self.gbrl_params, self.verbose, self.device)
        self._model.reset()
        self._model.set_bias(self.bias)
        
    def step(self, observations: Optional[Union[np.ndarray, th.Tensor]] = None, mu_grad: Optional[Union[np.ndarray, th.Tensor]] = None, log_std_grad: Optional[Union[np.ndarray, th.Tensor]] = None, mu_grad_clip: Optional[float] = None, log_std_grad_clip: Optional[float] = None) -> None:
        """Performs a single boosting iteration.

        Args:
            observations (Union[np.ndarray, th.Tensor])
            mu_grad_clip (float, optional). Defaults to None.
            log_std_grad_clip (float, optional). Defaults to None.
            mu_grad (Optional[Union[np.ndarray, th.Tensor]], optional): manually calculated gradients. Defaults to None.
            log_std_grad (Optional[Union[np.ndarray, th.Tensor]], optional): manually calculated gradients. Defaults to None.
        """
        if observations is None:
            assert self.input is not None, "Cannot update trees without input. Make sure model is called with requires_grad=True"
            observations = self.input
        n_samples = len(observations)
        mu_grad = mu_grad if mu_grad is not None else self.params[0].grad.detach() * n_samples
        mu_grad = clip_grad_norm(mu_grad, mu_grad_clip)
  
        if self.std_optimizer is not None:
            log_std_grad = log_std_grad if log_std_grad is not None else self.params[1].grad.detach() * n_samples
            log_std_grad = clip_grad_norm(log_std_grad, log_std_grad_clip)
            theta_grad = concatenate_arrays(mu_grad, log_std_grad)
        else:
            theta_grad = mu_grad
        
        validate_array(theta_grad)

        self._model.step(observations, theta_grad)
        self.grad = mu_grad
        if self.std_optimizer is not None:
            self.grad = (mu_grad, log_std_grad)
        self.input = None
    
    def get_num_trees(self) -> int:
        """Returns the number of trees in the ensemble

        Returns:
            int: number of trees
        """
        return self._model.get_num_trees()

    def __call__(self, observations: Union[np.ndarray, th.Tensor], requires_grad : bool = True, start_idx: int = 0, stop_idx: int = None, tensor: bool = True) -> Union[np.ndarray, th.Tensor]:
        """Returns actor's outputs as tensor. If `requires_grad=True` then stores 
           differentiable parameters in self.params. Return type/device is identical to the input type/device.
           Requires_grad is ignored if input is a numpy array.
        Args:
            observations (Union[np.ndarray, th.Tensor])
            requires_grad bool: Defaults to None.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Union[np.ndarray, th.Tensor]: Gaussian parameters
        """
        theta = self._model.predict(observations, requires_grad, start_idx, stop_idx, tensor)
        mean_actions = theta if self.std_optimizer is None else theta[:, :self.policy_dim]
        log_std = constant_like(theta, self.log_std_init) if self.std_optimizer is None else theta[:, self.policy_dim:]
        mean_actions = tensor_to_leaf(mean_actions, requires_grad)
        log_std = tensor_to_leaf(log_std, requires_grad = False if self.std_optimizer is None else requires_grad)
        if requires_grad:
            self.grad = None
            self.params = mean_actions, log_std
            self.input = observations
        return mean_actions, log_std
    
    def __copy__(self) -> "GaussianActor":
        std_optimizer = None if self.std_optimizer is None else self.std_optimizer.copy()
        copy_ = GaussianActor(self.tree_struct.copy(), self.input_dim, self.output_dim, self.mu_optimizer.copy(), std_optimizer, self.gbrl_params, self.bias, self.verbose, self.device)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_
    

class ContinuousCritic(GBRL):
    def __init__(self, 
                 tree_struct: Dict,
                 input_dim: int,
                 output_dim: int, 
                 weights_optimizer: Dict,
                 bias_optimizer: Dict =  None,
                 gbrl_params: Dict=dict(),
                 target_update_interval: int = 100,
                 bias: np.ndarray = None,
                 verbose: int=0,
                 device: str='cpu'):         
        """ GBRL model for a Continuous Critic ensemble.
            Usage example Q-function in SAC.
            Model is designed to output parameters of 3 types of Q-functions:
                - linear Q(theta(s), a) = <w_theta, a> + b_theta, (<> denotes a dot product).
                - quadratic Q(theta(s), a) = -(<w_theta, a> - b_theta)**2 + c_theta  
                - tanh Q(theta(s), a) = b_theta*tanh(<w_theta, a>)
            This allows to pass derivatives w.r.t to action a while the Q parameters are a function of a GBT model theta. 
            The target model is approximated as the ensemble without the last <target_update_interval> trees.
        
        Args:
         tree_struct (Dict): Dictionary containing tree structure information:
                max_depth (int): maximum tree depth.
                grow_policy (str): 'greedy' or 'oblivious'.
                n_bins (int): number of bins per feature for candidate generation.
                min_data_in_leaf (int): minimum number of samples in a leaf.
                par_th (int): minimum number of samples for parallelizing on CPU.
        output_dim (int): output dimension.
        weights_optimizer Dict: dictionary containing policy optimizer parameters. (see GradientBoostingTrees for optimizer details)
        bias_optimizer Dict: dictionary containing policy optimizer parameters. (see GradientBoostingTrees for optimizer details)
        gbrl_params (Dict, optional): GBRL parameters such as:
            control_variates (bool): use control variates (variance reduction technique CPU only).
            split_score_func (str): "cosine" or "l2"
            generator_type- (str): candidate generation method "Quantile" or "Uniform".
            feature_weights - (list[float]): Per-feature multiplication weights used when choosing the best split. Weights should be >= 0
        target_update_interval (int): target update interval
        bias (np.ndarray, optional): manually set a bias. Defaults to None = np.zeros.
        verbose (int, optional): verbosity level. Defaults to 0.
        device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """

        weights_optimizer = setup_optimizer(weights_optimizer, prefix='weights_')
        bias_optimizer = setup_optimizer(bias_optimizer, prefix='bias_')

        super().__init__(tree_struct,
                         input_dim,
                         output_dim,
                         None, 
                         gbrl_params,
                         verbose,
                         device)
        self.weights_optimizer = weights_optimizer
        self.bias_optimizer = bias_optimizer
        self.target_model = None
        self.bias = bias if bias is not None else np.zeros(self.output_dim, dtype=numerical_dtype)
        self.target_update_interval = target_update_interval
        # init model
        self._model = GBTWrapper(self.input_dim, self.output_dim, self.tree_struct, [weights_optimizer, bias_optimizer], self.gbrl_params, self.verbose, self.device)
        self._model.reset()
        self._model.set_bias(self.bias)
        
    def step(self, observations: Optional[Union[np.ndarray, th.Tensor]] = None, weight_grad: Optional[Union[np.ndarray, th.Tensor]] = None, bias_grad: Optional[Union[np.ndarray, th.Tensor]] = None, q_grad_clip: Optional[float] = None) -> None:
        """Performs a single boosting step

        Args:
            observations (Union[np.ndarray, th.Tensor]):
            q_grad_clip (float, optional):. Defaults to None.
        weight_grad (Optional[Union[np.ndarray, th.Tensor]], optional): manually calculated gradients. Defaults to None.
            bias_grad (Optional[Union[np.ndarray, th.Tensor]], optional): manually calculated gradients. Defaults to None.           
        """
        if observations is None:
            assert self.input is not None, "Cannot update trees without input. Make sure model is called with requires_grad=True"
            observations = self.input
        n_samples = len(observations)
        weight_grad = weight_grad if weight_grad is not None else self.params[0].grad.detach() * n_samples
        bias_grad = bias_grad if bias_grad is not None else self.params[1].grad.detach() * n_samples

        weight_grad = clip_grad_norm(weight_grad, q_grad_clip)
        bias_grad = clip_grad_norm(bias_grad, q_grad_clip)

        validate_array(weight_grad)
        validate_array(bias_grad)
        theta_grad = concatenate_arrays(weight_grad, bias_grad)

        self._model.step(observations, theta_grad)
        self.grad = (weight_grad, bias_grad)
        self.input = None

    def predict_target(self, observations: Union[np.ndarray, th.Tensor], tensor: bool = True) -> Tuple[Union[np.ndarray, th.Tensor], Union[np.ndarray, th.Tensor]]:
        """Predict the parameters of a Target Continuous Critic as Tensors.
        Prediction is made by summing the outputs the trees from Continuous Critic model up to `n_trees - target_update_interval`.

        Args:
            observations (Union[np.ndarray, th.Tensor]):
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Tuple[th.Tensor, th.Tensor]: weights and bias parameters to the type of Q-functions
        """
        n_trees = self._model.get_num_trees()
        theta = self._model.predict(observations, requires_grad=False, stop_idx=max(n_trees - self.target_update_interval, 1), tensor=tensor)
        weights, bias = theta[:, self.weights_optimizer['start_idx']:self.weights_optimizer['stop_idx']], theta[:, self.bias_optimizer['start_idx']:self.bias_optimizer['stop_idx']]
        return weights, bias

    def get_num_trees(self) -> int:
        """Get number of trees in model.

        Returns:
            int: return number of trees
        """
        return self._model.get_num_trees()

    def __call__(self, observations: Union[np.ndarray, th.Tensor], requires_grad: bool =  True, target: bool = False, start_idx: int = 0, stop_idx: int = None, tensor: bool = True) -> Tuple[Union[np.ndarray, th.Tensor], Union[np.ndarray, th.Tensor]]:
        """Predict the parameters of a Continuous Critic as Tensors. if `requires_grad=True` then stores 
           differentiable parameters in self.params. 
           Return type/device is identical to the input type/device.

        Args:
            observations (Union[np.ndarray, th.Tensor])
            requires_grad (bool, optional). Defaults to True.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Tuple[Union[np.ndarray, th.Tensor], Union[np.ndarray, th.Tensor]]: weights and bias parameters to the type of Q-functions
        """
        if target: 
            return self.predict_target(observations, tensor)

        theta = self._model.predict(observations, requires_grad, start_idx, stop_idx, tensor)
        weights, bias = theta[:, self.weights_optimizer['start_idx']:self.weights_optimizer['stop_idx']], theta[:, self.bias_optimizer['start_idx']:self.bias_optimizer['stop_idx']]
        if requires_grad:
            self.grad = None
            self.params = weights, bias
            self.input = observations
        return weights, bias
    
    def __copy__(self) -> "ContinuousCritic":
        copy_ = ContinuousCritic(self.tree_struct.copy(), self.input_dim, self.output_dim, self.weights_optimizer.copy(), self.bias_optimizer.copy() if isinstance(self.critic_optimizer, dict) else {'weights_optimizer': self.critic_optimizer[0], 'bias_optimizer': self.critic_optimizer[1]}, self.gbrl_params, self.target_update_interval, self.bias, self.verbose, self.device)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_
    

class DiscreteCritic(GBRL):
    def __init__(self, 
                 tree_struct: Dict,
                 input_dim: int,
                 output_dim: int, 
                 critic_optimizer: Dict,
                 gbrl_params: Dict=dict(),
                 target_update_interval: int = 100,
                 bias: np.ndarray = None,
                 verbose: int=0,
                 device: str='cpu'):  
        """ GBRL model for a Discrete Critic ensemble.
            Usage example: Q-function in GBRL.
            The target model is approximated as the ensemble without the last <target_update_interval> trees.
        Args:
         tree_struct (Dict): Dictionary containing tree structure information:
                max_depth (int): maximum tree depth.
                grow_policy (str): 'greedy' or 'oblivious'.
                n_bins (int): number of bins per feature for candidate generation.
                min_data_in_leaf (int): minimum number of samples in a leaf.
                par_th (int): minimum number of samples for parallelizing on CPU.
        output_dim (int): output dimension.
        critic_optimizer Dict: dictionary containing policy optimizer parameters. (see GradientBoostingTrees for optimizer details).
        gbrl_params (Dict, optional): GBRL parameters such as:
            control_variates (bool): use control variates (variance reduction technique CPU only).
            split_score_func (str): "cosine" or "l2"
            generator_type- (str): candidate generation method "Quantile" or "Uniform".
            feature_weights - (list[float]): Per-feature multiplication weights used when choosing the best split. Weights should be >= 0
        verbose (int, optional): verbosity level. Defaults to 0.
        device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """
        critic_optimizer = setup_optimizer(critic_optimizer, prefix='critic_')
        super().__init__(tree_struct,
                         input_dim,
                         output_dim,
                         None, 
                         gbrl_params,
                         verbose,
                         device)
        self.critic_optimizer = critic_optimizer
        self.target_update_interval = target_update_interval
        self.bias = bias if bias is not None else np.zeros(self.output_dim, dtype=numerical_dtype)
        # init model
        self._model = GBTWrapper(self.input_dim, self.output_dim, self.tree_struct, self.critic_optimizer, self.gbrl_params, self.verbose, self.device)
        self._model.reset()
        self._model.set_bias(self.bias)

    def step(self, observations: Optional[Union[np.ndarray, th.Tensor]] = None, q_grad: Optional[Union[np.ndarray, th.Tensor]] = None, max_q_grad_norm: Optional[np.ndarray] = None) -> None:
        """Performs a single boosting iterations.

        Args:
            observations (Union[np.ndarray, th.Tensor]):
            max_q_grad_norm (np.ndarray, optional). Defaults to None.
            q_grad (Optional[Union[np.ndarray, th.Tensor]], optional): manually calculated gradients. Defaults to None.
        """
        if observations is None:
            assert self.input is not None, "Cannot update trees without input. Make sure model is called with requires_grad=True"
            observations = self.input
        if q_grad is None:
            n_samples = len(observations)
            q_grad = self.params.grad.detach().cpu().numpy() * n_samples
        q_grad = clip_grad_norm(q_grad, max_q_grad_norm)
      
        self._model.step(observations, q_grad)
        self.grad = q_grad
        self.input = None

    def __call__(self, observations: Union[np.ndarray, th.Tensor], requires_grad: bool = True, start_idx: int = 0, stop_idx: Optional[int] = None, tensor: bool = True) -> Union[np.ndarray, th.Tensor]:
        """Predict and return Critic's outputs as Tensors. if `requires_grad=True` then stores 
           differentiable parameters in self.params. 
           Return type/device is identical to the input type/device.

        Args:
            observations (Union[np.ndarray, th.Tensor])
            requires_grad (bool, optional). Defaults to True.
            start_idx (int, optional): start tree index for prediction. Defaults to 0.
            stop_idx (_type_, optional): stop tree index for prediction (uses all trees in the ensemble if set to 0). Defaults to None.
            tensor (bool, optional): Return PyTorch Tensor, False returns a numpy array. Defaults to True.

        Returns:
            Union[np.ndarray, th.Tensor]: Critic's outputs.
        """
        q_values =self._model.predict(observations, requires_grad, start_idx, stop_idx, tensor)
        if requires_grad:
            self.grad = None
            self.params = q_values
            self.input = observations
        return q_values

    def predict_target(self, observations: Union[np.ndarray, th.Tensor], tensor: bool = True) -> th.Tensor:
        """Predict and return Target Critic's outputs as Tensors.
           Prediction is made by summing the outputs the trees from Continuous Critic model up to `n_trees - target_update_interval`.
        
        Args:
            observations (Union[np.ndarray, th.Tensor])
        
        Returns:
            th.Tensor: Target Critic's outputs.
        """
        n_trees = self._model.get_num_trees()
        return self._model.predict(observations, requires_grad=False, stop_idx=max(n_trees - self.target_update_interval, 1), tensor=tensor)

    def __copy__(self) -> "DiscreteCritic":
        copy_ = DiscreteCritic(self.tree_struct.copy(), self.input_dim, self.output_dim, self.critic_optimizer.copy(), self.gbrl_params, self.target_update_interval, self.bias, self.verbose, self.device)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_

if __name__ == '__main__':
    pass
        




