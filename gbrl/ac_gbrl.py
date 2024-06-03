"""
Module for different GBT models for RL
"""
from typing import Dict, Tuple, Union
import os 
import numpy as np
import torch as th

from .gbrl_wrapper import (GBTWrapper, SeparateActorCriticWrapper,
                           SharedActorCriticWrapper, numerical_dtype)
from .gbt import GradientBoostingTrees
from .utils import setup_optimizer, clip_grad_norm


class ActorCritic(GradientBoostingTrees):
    def __init__(self, 
                 tree_struct: Dict,
                 output_dim: int, 
                 policy_optimizer: Dict,
                 value_optimizer: Dict= None,
                 shared_tree_struct: bool=True,
                 gbrl_params: Dict=dict(),
                 bias: np.array = None,
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
        bias (np.array, optional): manually set a bias. Defaults to None = np.zeros.
        verbose (int, optional): verbosity level. Defaults to 0.
        device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """
        policy_optimizer = setup_optimizer(policy_optimizer, prefix='policy_')
        policy_optimizer['T'] = gbrl_params.get('T', 10000)
        if value_optimizer is not None:
            value_optimizer = setup_optimizer(value_optimizer, prefix='value_')
            value_optimizer['T'] = gbrl_params.get('T', 10000)
        super().__init__(tree_struct,
                         output_dim if shared_tree_struct else output_dim - 1,
                         None,
                         gbrl_params,
                         verbose,
                         device)
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer

        self.shared_tree_struct = True if value_optimizer is None else shared_tree_struct
        self.bias = bias if bias is not None else np.zeros(self.output_dim, dtype=numerical_dtype)
        # init model
        if self.shared_tree_struct:
            self._model = SharedActorCriticWrapper(self.output_dim, self.tree_struct, self.policy_optimizer, self.value_optimizer, self.gbrl_params, self.verbose, self.device) 
        else:
            self._model = SeparateActorCriticWrapper(self.output_dim, self.tree_struct, self.policy_optimizer, self.value_optimizer, self.gbrl_params, self.verbose, self.device)
        self._model.reset()
        self._model.set_bias(self.bias)
        
    @classmethod
    def load_model(cls, load_name: str):

        policy_file = load_name + '_policy.gbrl_model'
        value_file = load_name + '_value.gbrl_model'

        instance = cls.__new__(cls)
        print("loading model")
        if os.path.isfile(policy_file) and os.path.isfile(value_file):
            instance._model = SeparateActorCriticWrapper.load(load_name)
            instance.shared_tree_struct = False
            instance.bias = instance._model.policy_model.get_bias() 
        else:
            print("is shared model")
            instance._model = SharedActorCriticWrapper.load(load_name)
            instance.shared_tree_struct = True 
            instance.bias = instance._model.get_bias()
        instance.value_optimizer = instance._model.value_optimizer
        instance.policy_optimizer = instance._model.policy_optimizer
        instance.output_dim = instance._model.output_dim
        instance.verbose = instance._model.verbose
        instance.tree_struct = instance._model.tree_struct
        instance.gbrl_params = instance._model.gbrl_params
        instance.device = instance._model.get_device()
        return instance
    
    def predict(self, observations: Union[np.array, th.Tensor]) -> Tuple[np.array, np.array]:
        """Predict method
        Args:
            observations (Union[np.array, th.Tensor])

        Returns:
            Tuple[np.array, np.array]: actor and value predictions.
        """
        return self._model.predict(observations)
    
    def get_num_trees(self) -> Union[int, Tuple[int, int]]:
        """ Returns number of trees in the ensemble.
            If separate actor and critic return number of trees per ensemble.
        Returns:
            Union[int, Tuple[int, int]]
        """
        return self._model.get_num_trees()
     
    def predict_values(self, observations: Union[np.array, th.Tensor], requires_grad: bool=True) -> th.Tensor:
        """Predict only values

        Args:
            observations (Union[np.array, th.Tensor])

        Returns:
            th.Tensor: values
        """
        _, values = self._model.predict(observations)
        if len(values.shape) == 0:
            values = values[np.newaxis]
        self.params = th.tensor(values, requires_grad=requires_grad)
        return self.params

    def __call__(self, observations: Union[np.array, th.Tensor], requires_grad: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        """ Predicts  and returns actor and critic outputs as tensors.
        Args:
            observations (Union[np.array, th.Tensor])

        Returns:
            Tuple[th.Tensor, th.Tensor]: actor and critic output
        """
        theta, values = self.predict(observations)
        params = (th.tensor(theta.astype(np.single), requires_grad=requires_grad), th.tensor(values.astype(np.single), requires_grad=requires_grad))
        if requires_grad:
            self.params = params
        return params
    
    def step(self, observations: Union[np.array, th.Tensor], policy_grad_clip: float = None, value_grad_clip : float = None) -> None:
        n_samples = len(observations)
        theta_grad = self.params[0].grad.detach().cpu().numpy() * n_samples
        value_grad = self.params[1].grad.detach().cpu().numpy() * n_samples
        # print(value_grad)
        theta_grad = clip_grad_norm(theta_grad, policy_grad_clip)
        value_grad = clip_grad_norm(value_grad, value_grad_clip)

        assert ~np.isnan(theta_grad).any(), "nan in assigned grads"
        assert ~np.isinf(theta_grad).any(), "infinity in assigned grads"
        assert ~np.isnan(value_grad).any(), "nan in assigned grads"
        assert ~np.isinf(value_grad).any(), "infinity in assigned grads"
        self._model.step(observations, theta_grad, value_grad)
        return theta_grad, value_grad
    
    def actor_step(self, observations: Union[np.array, th.Tensor], policy_grad_clip: float = None) -> None:
        n_samples = len(observations)
        theta_grad = self.params[0].grad.detach().cpu().numpy() * n_samples
        theta_grad = clip_grad_norm(theta_grad, policy_grad_clip)
        assert ~np.isnan(theta_grad).any(), "nan in assigned grads"
        assert ~np.isinf(theta_grad).any(), "infinity in assigned grads"

        self._model.step_policy(observations, theta_grad)
        return self.params[0].detach().cpu().numpy(), theta_grad
    
    def critic_step(self, observations: Union[np.array, th.Tensor], value_grad_clip : float = None) -> None:
        n_samples = len(observations)
        value_grad = self.params[1].grad.detach().cpu().numpy() * n_samples
        value_grad = clip_grad_norm(value_grad, value_grad_clip)

        assert ~np.isnan(value_grad).any(), "nan in assigned grads"
        assert ~np.isinf(value_grad).any(), "infinity in assigned grads"
        self._model.step_critic(observations, value_grad)
        return self.params[1].detach().cpu().numpy(), value_grad

    def __copy__(self) -> "ActorCritic":
        value_optimizer = None if self.value_optimizer is None else self.value_optimizer.copy()
        copy_ = ActorCritic(self.tree_struct.copy(), self.output_dim, self.policy_optimizer.copy(), value_optimizer, self.shared_tree_struct, self.gbrl_params, self.bias, self.verbose, self.device)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_
    
class ParametricActor(GradientBoostingTrees):
    def __init__(self, 
                 tree_struct: Dict,
                 output_dim: int, 
                 policy_optimizer: Dict,
                 gbrl_params: Dict=dict(),
                 bias: np.array = None,
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
        bias (np.array, optional): manually set a bias. Defaults to None = np.zeros.
        verbose (int, optional): verbosity level. Defaults to 0.
        device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """
        policy_optimizer = setup_optimizer(policy_optimizer, prefix='policy_')
        policy_optimizer['T'] = gbrl_params.get('T', 10000)
        super().__init__(tree_struct,
                         output_dim,
                         None,
                         gbrl_params,
                         verbose,
                         device)
        self.policy_optimizer = policy_optimizer
        self.bias = bias if bias is not None else np.zeros(self.output_dim, dtype=numerical_dtype)
        # init model
        self._model = GBTWrapper(self.output_dim, self.output_dim, self.tree_struct, self.policy_optimizer, self.gbrl_params, self.verbose, self.device) 
        self._model.reset()
        self._model.set_bias(self.bias)

    def step(self, observations: Union[np.array, th.Tensor], policy_grad_clip: float = None) -> None:
        n_samples = len(observations)
        theta_grad = self.params.grad.detach().cpu().numpy() * n_samples
        theta_grad = clip_grad_norm(theta_grad, policy_grad_clip)
        assert ~np.isnan(theta_grad).any(), "nan in assigned grads"
        assert ~np.isinf(theta_grad).any(), "infinity in assigned grads"

        self._model.step(observations, theta_grad)
        return theta_grad

    @classmethod
    def load_model(cls, load_name: str):
        instance = cls.__new__(cls)
        print("loading model")
        instance._model = GBTWrapper.load(load_name)
        instance.bias = instance._model.get_bias()
        instance.policy_optimizer = instance._model.optimizer
        instance.output_dim = instance._model.output_dim
        instance.verbose = instance._model.verbose
        instance.tree_struct = instance._model.tree_struct
        instance.gbrl_params = instance._model.gbrl_params
        instance.device = instance._model.get_device()
        return GBTWrapper.load(load_name)
    
    def predict(self, observations: Union[np.array, th.Tensor]) -> np.array:
        """Predict method
        Args:
            observations (Union[np.array, th.Tensor])

        Returns:
            np.array: actor predictions.
        """
        return self._model.predict(observations)
    
    def get_num_trees(self) -> int:
        """ Returns number of trees in the ensemble.
        Returns:
            int
        """
        return self._model.get_num_trees()
     
    def __call__(self, observations: Union[np.array, th.Tensor], requires_grad : bool = False) -> th.Tensor:
        """ Predicts  and returns actor output as tensor.
        Args:
            observations (Union[np.array, th.Tensor])

        Returns:
            th.Tensor: actor output
        """
        theta = self._model.predict(observations)
        params = th.tensor(theta, requires_grad=requires_grad)
        if requires_grad:
            self.params = params
        return params
    
    def __copy__(self) -> "ParametricActor":
        copy_ = ParametricActor(self.tree_struct.copy(), self.output_dim, self.policy_optimizer.copy(),  self.gbrl_params, self.bias, self.verbose, self.device)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_
    
class GaussianActor(GradientBoostingTrees):
    def __init__(self, 
                 tree_struct: Dict,
                 output_dim: int, 
                 mu_optimizer: Dict,
                 std_optimizer: Dict = None,
                 log_std_init: float = -2,
                 gbrl_params: Dict = dict(),
                 bias: np.array = None,
                 verbose: int=0,
                 device: str='cpu'):
        """ GBRL model for a Actor ensemble used in SAC.
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
        bias (np.array, optional): manually set a bias. Defaults to None = np.zeros.
        verbose (int, optional): verbosity level. Defaults to 0.
        device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """
        mu_optimizer = setup_optimizer(mu_optimizer, prefix='mu_')
        mu_optimizer['T'] = gbrl_params.get('T', 10000)

        self.bias = bias if bias is not None else np.zeros(output_dim, dtype=numerical_dtype)

        policy_dim = output_dim
        if std_optimizer is not None:
            std_optimizer = setup_optimizer(std_optimizer, prefix='std_')
            std_optimizer['T'] = gbrl_params.get('T', 10000)
            policy_dim = output_dim // 2
            self.bias[policy_dim:] = log_std_init*np.ones(policy_dim, dtype=numerical_dtype)
        
        self.mu_optimizer = mu_optimizer
        self.std_optimizer = std_optimizer
        self.log_std_init = log_std_init

        super().__init__(tree_struct,
                         output_dim,
                         None, 
                         gbrl_params,
                         verbose,
                         device)

        
        self.policy_dim = policy_dim
        # init model
        self._model = GBTWrapper(self.output_dim, self.policy_dim, self.tree_struct, [mu_optimizer, std_optimizer], self.gbrl_params, self.verbose, self.device)
        self._model.reset()
        self._model.set_bias(self.bias)
        
    def step(self, observations: Union[np.array, th.Tensor], mu_grad_clip: float = None, std_grad_clip: float = None) -> None:
        n_samples = len(observations)
        mu_grad = self.params[0].grad.detach().cpu().numpy() * n_samples
        mu_grad = clip_grad_norm(mu_grad, mu_grad_clip)
        log_std_grad = None
        if self.std_optimizer is not None:
            log_std_grad = self.params[1].grad.detach().cpu().numpy() * n_samples
            log_std_grad = clip_grad_norm(log_std_grad, std_grad_clip)
            theta_grad = np.concatenate([mu_grad, log_std_grad], axis=1)
        else:
            theta_grad = mu_grad
        assert ~np.isnan(theta_grad).any(), "nan in assigned grads"
        assert ~np.isinf(theta_grad).any(), "infinity in assigned grads"

        self._model.step(observations, theta_grad)
        return mu_grad, log_std_grad
        
    def predict(self, observations: Union[np.array, th.Tensor]) -> np.array:
        theta = self._model.predict(observations)
        if self.std_optimizer is not None:
            return theta[:, :self.policy_dim], theta[:, self.policy_dim:]
        return theta, np.ones_like(theta) * self.log_std_init
    
    def get_num_trees(self) -> int:
        return self._model.get_num_trees()

    def __call__(self, observations: Union[np.array, th.Tensor], requires_grad : bool = False) -> th.Tensor:
        mean_actions, log_std = self.predict(observations)
        params = (th.tensor(mean_actions, requires_grad=requires_grad), th.tensor(log_std, requires_grad=requires_grad if self.std_optimizer is not None else False))
        if requires_grad:
            self.params = params
        return params
    
    def __copy__(self) -> "GaussianActor":
        std_optimizer = None if self.std_optimizer is None else self.std_optimizer.copy()
        copy_ = GaussianActor(self.tree_struct.copy(), self.output_dim, self.mu_optimizer.copy(), std_optimizer, self.gbrl_params, self.bias, self.verbose, self.device)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_
    

class ContinuousCritic(GradientBoostingTrees):
    def __init__(self, 
                 tree_struct: Dict,
                 output_dim: int, 
                 weights_optimizer: Dict,
                 bias_optimizer: Dict =  None,
                 gbrl_params: Dict=dict(),
                 target_update_interval: int = 100,
                 bias: np.array = None,
                 verbose: int=0,
                 device: str='cpu'):         
        """ GBRL model for a Continuous Critic ensemble (Q-function) used in GBRL SAC.
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
        bias (np.array, optional): manually set a bias. Defaults to None = np.zeros.
        verbose (int, optional): verbosity level. Defaults to 0.
        device (str, optional): GBRL device 'cpu' or 'cuda/gpu'. Defaults to 'cpu'.
        """
        weights_optimizer = setup_optimizer(weights_optimizer, prefix='weights_')
        bias_optimizer = setup_optimizer(bias_optimizer, prefix='bias_')
        weights_optimizer['T'] = gbrl_params.get('T', 10000)
        bias_optimizer['T'] = gbrl_params.get('T', 10000)

        policy_dim = output_dim - 1
        super().__init__(tree_struct,
                         output_dim,
                         None, 
                         gbrl_params,
                         verbose,
                         device)
        self.policy_dim = policy_dim
        self.weights_optimizer = weights_optimizer
        self.bias_optimizer = bias_optimizer
        self.target_model = None
        self.bias = bias if bias is not None else np.zeros(self.output_dim, dtype=numerical_dtype)
        self.target_update_interval = target_update_interval
        # init model
        self._model = GBTWrapper(self.output_dim, self.policy_dim, self.tree_struct, [weights_optimizer, bias_optimizer], self.gbrl_params, self.verbose, self.device)
        self._model.reset()
        self._model.set_bias(self.bias)
        
    def step(self, observations: Union[np.array, th.Tensor], q_grad_clip: float = None) -> None:
        n_samples = len(observations)
        weight_grad = self.params[0].grad.detach().cpu().numpy() * n_samples
        bias_grad = self.params[1].grad.detach().cpu().numpy() * n_samples
        weight_grad = clip_grad_norm(weight_grad, q_grad_clip)
        bias_grad = clip_grad_norm(bias_grad, q_grad_clip)

        assert ~np.isnan(weight_grad).any(), "nan in assigned grads"
        assert ~np.isinf(weight_grad).any(), "infinity in assigned grads"
        assert ~np.isnan(bias_grad).any(), "nan in assigned grads"
        assert ~np.isinf(bias_grad).any(), "infinity in assigned grads"
        theta_grad = np.concatenate([weight_grad, bias_grad], axis=1)
        self._model.step(observations, theta_grad)
        return weight_grad, bias_grad
        
    def predict(self, observations: Union[np.array, th.Tensor]) -> np.array:
        return self._model.predict(observations)

    def predict_target(self, observations: Union[np.array, th.Tensor]) -> np.array:
        n_trees = self._model.get_num_trees()
        theta = self._model.predict(observations, stop_idx=max(n_trees - self.target_update_interval, 1))
        return th.tensor(theta[:, :self.policy_dim]), th.tensor(theta[:, self.policy_dim:])

    def get_num_trees(self) -> int:
        return self._model.get_num_trees()

    def __call__(self, observations: Union[np.array, th.Tensor], requires_grad: bool =  False) -> Tuple[th.Tensor, th.Tensor]:
        theta = self.predict(observations)
        weights, bias = theta[:, :self.policy_dim], theta[:, self.policy_dim:]
        params = (th.tensor(weights, requires_grad=requires_grad), th.tensor(bias, requires_grad=requires_grad))
        if requires_grad:
            self.params = params
        return params

    def copy(self) -> "ContinuousCritic":
        return self.__copy__()
    
    def __copy__(self) -> "ContinuousCritic":
        copy_ = ContinuousCritic(self.tree_struct.copy(), self.output_dim, self.weights_optimizer.copy(), self.bias_optimizer.copy() if isinstance(self.critic_optimizer, dict) else {'weights_optimizer': self.critic_optimizer[0], 'bias_optimizer': self.critic_optimizer[1]}, self.gbrl_params, self.target_update_interval, self.bias, self.verbose, self.device)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_
    

class DiscreteCritic(GradientBoostingTrees):
    def __init__(self, 
                 tree_struct: Dict,
                 output_dim: int, 
                 critic_optimizer: Dict,
                 gbrl_params: Dict=dict(),
                 target_update_interval: int = 100,
                 bias: np.array = None,
                 verbose: int=0,
                 device: str='cpu'):  
        """ GBRL model for a Discrete Critic ensemble (Q-function) used in GBRL DQN.
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
        critic_optimizer['T'] = gbrl_params.get('T', 10000)
        super().__init__(tree_struct,
                         output_dim,
                         None, 
                         gbrl_params,
                         verbose,
                         device)
        self.critic_optimizer = critic_optimizer
        self.target_update_interval = target_update_interval
        self.bias = bias if bias is not None else np.zeros(self.output_dim, dtype=numerical_dtype)
        # init model
        self._model = GBTWrapper(self.output_dim, self.output_dim, self.tree_struct, self.critic_optimizer, self.gbrl_params, self.verbose, self.device)
        self._model.reset()
        self._model.set_bias(self.bias)

    def step(self, observations: Union[np.array, th.Tensor], max_q_grad_norm: np.array = None) -> None:
        n_samples = len(observations)
        q_grad = self.params.grad.detach().cpu().numpy() * n_samples
        q_grad = clip_grad_norm(q_grad, max_q_grad_norm)
      
        self._model.step(observations, q_grad)
        return q_grad


    def predict(self, observations: Union[np.array, th.Tensor]) -> np.array:
        return self._model.predict(observations)

    def __call__(self, observations: Union[np.array, th.Tensor], requires_grad: bool=False) -> th.Tensor:
        q_values = self.predict(observations)
        params = th.tensor(q_values, requires_grad=requires_grad)
        if requires_grad:
            self.params = params
        return params

    def predict_target(self, observations: Union[np.array, th.Tensor]) -> th.Tensor:
        n_trees = self._model.get_num_trees()
        q_values = self._model.predict(observations, stop_idx=max(n_trees - self.target_update_interval, 1))
        return th.tensor(q_values)

    def copy(self) -> "DiscreteCritic":
        return self.__copy__()
    
    def __copy__(self) -> "DiscreteCritic":
        copy_ = DiscreteCritic(self.tree_struct.copy(), self.output_dim, self.critic_optimizer.copy(), self.gbrl_params, self.target_update_interval, self.bias, self.verbose, self.device)
        if self._model is not None:
            copy_._model = self._model.copy()
        return copy_


        




