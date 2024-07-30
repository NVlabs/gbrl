
##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
import os
from typing import Dict, List, Union, Tuple

import numpy as np
import torch as th

from .gbrl_cpp import GBRL as GBRL_CPP
from .utils import get_input_dim, get_poly_vectors, process_array, to_numpy, numerical_dtype


def features_to_numpy(arr: Union[np.array, th.Tensor]) -> Tuple[np.array, np.array]:
    if isinstance(arr, th.Tensor):
        arr = arr.detach().cpu().numpy()
        return np.ascontiguousarray(arr, dtype=numerical_dtype), None 
    elif isinstance(arr, tuple):
        num_arr, _ = process_array(arr[0])
        _, cat_arr = process_array(arr[1])
        return num_arr, cat_arr
    elif isinstance(arr, list):
        return process_array(np.array(arr))
    elif isinstance(arr, dict):
        num_arr, _ = process_array(arr['numerical_data'])
        _, cat_arr = process_array(arr['categorical_data'])
        return num_arr, cat_arr
    else:
        return process_array(arr)
       
def preprocess_features(arr: Union[np.array, th.Tensor]) -> Tuple[np.array, np.array]:
    """Preprocess array such that the dimensions and the data type match.
    Returns numerical and categorical features. 
    May return None for each if purely numerical or purely categorical.
    Args:

    Returns:
        Tuple[np.array, np.array]
    """
    input_dim = get_input_dim(arr)
    num_arr, cat_arr = features_to_numpy(arr)
    if num_arr is not None and len(num_arr.shape) == 1:
        if input_dim == 1:
            num_arr = num_arr[np.newaxis, :]
        else:
            num_arr = num_arr[:, np.newaxis]
    if num_arr is not None and len(num_arr.shape) > 2:
        num_arr = num_arr.squeeze()
    if cat_arr is not None and len(cat_arr.shape) == 1:
        if input_dim == 1:
            cat_arr = cat_arr[np.newaxis, :]
        else:
            cat_arr = cat_arr[:, np.newaxis]
    if cat_arr is not None and len(cat_arr.shape) > 2:
        cat_arr = cat_arr.squeeze()
    return num_arr, cat_arr


class GBTWrapper:
    def __init__(self, output_dim: int, tree_struct: Dict, optimizer: Union[Dict, List], gbrl_params: Dict, verbose: int = 0, device: str = 'cpu'):
        if 'T' in gbrl_params:
            del gbrl_params['T']
        self.params = {'output_dim': output_dim,
                       'split_score_func': gbrl_params.get('split_score_func', 'Cosine'), 
                       'generator_type': gbrl_params.get('generator_type', 'Quantile'), 
                       'use_control_variates': gbrl_params.get('control_variates', False), 
                       'verbose': verbose, 'device': device, **tree_struct}
        self.tree_struct = tree_struct
        self.output_dim = output_dim
        self.device = device
        self.optimizer = optimizer if isinstance(optimizer, list) or optimizer is None else [optimizer]
        self.student_model = None
        self.cpp_model = None
        self.iteration = 0
        self.total_iterations = 0
        self.gbrl_params = gbrl_params
        self.verbose = verbose
        feature_weights = gbrl_params.get('feature_weights', None)
        if feature_weights is not None:
            feature_weights = to_numpy(feature_weights)
            feature_weights = feature_weights.flatten()
            assert np.all(feature_weights >= 0), "feature weights contains non-positive values"
        self.feature_weights = feature_weights

    def reset(self) -> None:
        if self.cpp_model is not None:
            lrs = self.cpp_model.get_scheduler_lrs()
            for i in range(len(self.optimizer)):
                self.optimizer[i]['init_lr'] = lrs[i]

        self.cpp_model = GBRL_CPP(**self.params)
        if self.student_model is not None:
            for i in range(len(self.optimizer)):
                self.optimizer[i]['T'] -= self.total_iterations
        else:
            self.total_iterations = 0
        try:
            for opt in self.optimizer:
                self.cpp_model.set_optimizer(**opt)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")
        feature_weights = self.gbrl_params.get('feature_weights', None)
        if feature_weights is not None:
            feature_weights = to_numpy(feature_weights)
            feature_weights = feature_weights.flatten()
            assert np.all(feature_weights >= 0), "feature weights contains non-positive values"
        self.feature_weights = feature_weights

    def step(self, features: Union[np.array, th.Tensor, Tuple], grads: Union[np.array, th.Tensor]) -> None:
        num_features, cat_features = preprocess_features(features)
        grads = to_numpy(grads)
        grads = grads.reshape((len(grads), self.params['output_dim']))
        input_dim = 0 if num_features is None else num_features.shape[1]
        input_dim += 0 if cat_features is None else cat_features.shape[1]
        if self.feature_weights is None:
            self.feature_weights = np.ones(input_dim, dtype=numerical_dtype)
        assert len(self.feature_weights) == input_dim, "feature weights has to have the same number of elements as features"
        assert np.all(self.feature_weights >= 0), "feature weights contains non-positive values"
        self.cpp_model.step(num_features, cat_features, grads, self.feature_weights)
        self.iteration = self.cpp_model.get_iteration()
        self.total_iterations += 1

    def fit(self, features: Union[np.array, th.Tensor], targets: Union[np.array, th.Tensor], iterations: int, shuffle: bool=True, loss_type: str='MultiRMSE') -> float:
        num_features, cat_features = preprocess_features(features)
        targets = to_numpy(targets)
        targets = targets.reshape((len(targets), self.params['output_dim'])).astype(numerical_dtype)
        input_dim = 0 if num_features is None else num_features.shape[1]
        input_dim += 0 if cat_features is None else cat_features.shape[1]
        if self.feature_weights is None:
            self.feature_weights = np.ones(input_dim, dtype=numerical_dtype)
        assert len(self.feature_weights) == input_dim, "feature weights has to have the same number of elements as features"
        assert np.all(self.feature_weights >= 0), "feature weights contains non-positive values"
        loss = self.cpp_model.fit(num_features, cat_features, targets, self.feature_weights, iterations, shuffle, loss_type)
        self.iteration = self.cpp_model.get_iteration()
        return loss

    def save(self, filename: str) -> None:
        filename = filename.rstrip('.')
        filename += '.gbrl_model'
        assert self.cpp_model is not None, "Can't save non-existent model!"
        status = self.cpp_model.save(filename)
        assert status == 0, "Failed to save model"

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
    def load(cls, filename: str) -> "GBTWrapper":
        filename = filename.rstrip('.')
        if '.gbrl_model' not in filename:
            filename += '.gbrl_model'
        assert os.path.isfile(filename), "filename doesn't exist!"
        try:
            instance = cls.__new__(cls)
            instance.cpp_model = GBRL_CPP.load(filename)
            metadata =  instance.cpp_model.get_metadata()
            instance.tree_struct = {'max_depth': metadata['max_depth'], 
                            'min_data_in_leaf': metadata['min_data_in_leaf'],
                            'n_bins': metadata['n_bins'], 
                            'par_th': metadata['par_th'],
                            'batch_size': metadata['batch_size'], 
                            'grow_policy': metadata['grow_policy']}
            instance.params = {'output_dim': metadata['output_dim'],
                            'split_score_func': metadata['split_score_func'], 
                            'generator_type': metadata['generator_type'], 
                            'use_control_variates': metadata['use_control_variates'], 
                            'verbose': metadata['verbose'], 
                            'device': instance.cpp_model.get_device(),
                            **instance.tree_struct
                            }
            instance.output_dim = metadata['output_dim']
            instance.verbose = metadata['verbose']
            instance.gbrl_params = {'split_score_func': metadata['split_score_func'], 
                                    'generator_type': metadata['generator_type'], 
                                    'use_control_variates': metadata['use_control_variates'], 
                                    }
            instance.optimizer = instance.cpp_model.get_optimizers()
            instance.iteration = metadata['iteration']
            instance.total_iterations = metadata['iteration']
            instance.student_model = None
            return instance
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")
            return None

    def get_schedule_learning_rates(self) -> Union[int, Tuple[int, int]]:
        return self.cpp_model.get_scheduler_lrs()

    def get_total_iterations(self) -> int:
        return self.total_iterations

    def get_iteration(self) -> int:
        return self.cpp_model.get_iteration()
    
    def get_num_trees(self) -> int:
        if self.student_model is not None:
            return self.cpp_model.get_num_trees() + self.student_model.get_num_trees()
        return self.cpp_model.get_num_trees()
    
    def set_bias(self, bias: Union[np.array, float]) -> None:
        if not isinstance(bias, np.ndarray) and not isinstance(bias, float):
            raise TypeError("Input should be a numpy array or float")

        if isinstance(bias, float):
            bias = np.array([float])

        if bias.ndim > 1:
            bias = bias.ravel()
        elif bias.ndim == 0:
            bias = np.array([bias.item()])  # Converts 0D arrays to 1D
        try:
            self.cpp_model.set_bias(bias)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def get_bias(self) -> np.array:
        return self.cpp_model.get_bias()

    def get_device(self) -> str:
        return self.cpp_model.get_device()

    def print_tree(self, tree_idx: int) -> None:
        self.cpp_model.print_tree(tree_idx)
    
    def plot_tree(self, tree_idx: int, filename: str) -> None:
        try:
            self.cpp_model.plot_tree(tree_idx, filename)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def tree_shap(self, tree_idx: int, features: Union[np.array, th.Tensor]) -> np.array:
        """  
        Implementation based on - https://github.com/yupbank/linear_tree_shap
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192 
        Args:
            tree_idx (int): tree index
            features (Union[np.array, th.Tensor]):

        Returns:
            np.array: shap values
        """
        num_features, cat_features = preprocess_features(features)
        base_poly, norm_values, offset = get_poly_vectors(self.params['max_depth'], numerical_dtype)
        return self.cpp_model.tree_shap(tree_idx, num_features, cat_features, np.ascontiguousarray(norm_values), np.ascontiguousarray(base_poly), np.ascontiguousarray(offset)) 
    
    def shap(self, features: Union[np.array, th.Tensor]) -> np.array:
        """  
        Uses Linear tree shap for each tree in the ensemble (sequentially)
        Implementation based on - https://github.com/yupbank/linear_tree_shap
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192 
        Args:
            features (Union[np.array, th.Tensor]):

        Returns:
            np.array: shap values
        """
        num_features, cat_features = preprocess_features(features)
        base_poly, norm_values, offset = get_poly_vectors(self.params['max_depth'], numerical_dtype)
        return self.cpp_model.ensemble_shap(num_features, cat_features, np.ascontiguousarray(norm_values), np.ascontiguousarray(base_poly), np.ascontiguousarray(offset)) 
    
    def set_device(self, device: str) -> None:
        try:
            self.cpp_model.to_device(device)
            self.device = device
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")
    
    def predict(self, features: Union[np.array, th.Tensor], start_idx: int=0, stop_idx: int=None) -> np.array:
        num_features, cat_features = preprocess_features(features)
        if stop_idx is None:
            stop_idx = 0

        if self.student_model is not None:
            preds = self.student_model.predict(num_features, cat_features)
            if num_features is not None and not num_features.flags['WRITEABLE']:
                num_features = num_features.copy()
            if cat_features is not None and not cat_features.flags['WRITEABLE']:
                cat_features = cat_features.copy()
            self.cpp_model.predict(num_features, cat_features, preds, start_idx, stop_idx) # modifies it inplace
        else:
            preds = self.cpp_model.predict(num_features, cat_features, start_idx, stop_idx)
        return preds
    
    def distil(self, obs: Union[np.array, th.Tensor], targets: np.array, params: Dict, verbose: int=0) -> Tuple[int, Dict]:
        num_obs, cat_obs = preprocess_features(obs)
        distil_params = {'output_dim': self.params['output_dim'], 'split_score_func': 'L2',
                         'generator_type': 'Quantile',  'use_control_variates': False, 'device': self.device,
                        'max_depth': params.get('distil_max_depth', 6), 'verbose': verbose, 'batch_size': self.params.get('distil_batch_size', 2048)}
        self.student_model = GBRL_CPP(**distil_params)
        distil_optimizer = {'algo': 'SGD', 'init_lr': params.get('distil_lr', 0.1)}
        try:
            self.student_model.set_optimizer(**distil_optimizer)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

        bias = np.mean(targets, axis=0)
        if isinstance(bias, float):
            bias = np.array([bias])
        self.student_model.set_bias(bias.astype(numerical_dtype))
        tr_loss = self.student_model.fit(num_obs, cat_obs, targets, params['min_steps'])
        while tr_loss> params.get('min_distillation_loss', 0.1):
            if params['min_steps'] < params['limit_steps']:
                steps_to_add = min(500, params['limit_steps'] - params['min_steps'])
                tr_loss = self.student_model.fit(num_obs, cat_obs, targets, steps_to_add, shuffle=False)
                params['min_steps'] += steps_to_add
            else:
                break
        self.reset()
        return tr_loss, params

    def copy(self):
        return self.__copy__()
    
    def __copy__(self):
        copy_ = GBTWrapper(self.output_dim, self.tree_struct.copy(), [opt.copy() if opt is not None else opt for opt in self.optimizer], self.gbrl_params, self.verbose, self.device)
        copy_.iteration = self.iteration 
        copy_.total_iterations = self.total_iterations
        if self.cpp_model is not None:
            copy_.cpp_model = GBRL_CPP(self.cpp_model)
        if self.student_model is not None:
            copy_.student_model = GBRL_CPP(self.student_model)
        return copy_

class SeparateActorCriticWrapper:
    def __init__(self, output_dim: int, tree_struct: Dict, policy_optimizer: Dict, value_optimizer: Dict, gbrl_params: Dict, verbose: int = 0, device: str = 'cpu'):
        print('****************************************')
        print(f'Separate GBRL Tree with output dim: {output_dim}, tree_struct: {tree_struct} policy_optimizer: {policy_optimizer} value_optimizer: {value_optimizer}')
        print('****************************************')
        self.policy_model = GBTWrapper(output_dim - 1, tree_struct, policy_optimizer, gbrl_params, verbose, device)
        self.value_model = GBTWrapper(1, tree_struct, value_optimizer, gbrl_params, verbose, device)
        self.tree_struct = tree_struct
        self.total_iterations = 0
        self.output_dim = output_dim
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.gbrl_params = gbrl_params
        self.verbose = verbose
        self.device = device
 
    def step(self, observations: Union[np.array, th.Tensor], theta_grad: Union[np.array, th.Tensor], value_grad: Union[np.array, th.Tensor]):
        self.step_policy(observations, theta_grad)
        self.step_critic(observations, value_grad)
        self.total_iterations += 1
        
    def step_policy(self, observations: Union[np.array, th.Tensor], theta_grad: Union[np.array, th.Tensor]):
        self.policy_model.step(observations, theta_grad)
    
    def step_critic(self, observations: Union[np.array, th.Tensor], value_grad: Union[np.array, th.Tensor]):
        self.value_model.step(observations, value_grad)

    def set_device(self, device:str) -> None:
        self.policy_model.set_device(device)
        self.value_model.set_device(device)
        self.device = device

    def tree_shap(self, tree_idx: int, observations: Union[np.array, th.Tensor]) -> Tuple[np.array, np.array]:
        policy_shap = self.policy_model.tree_shap(tree_idx, observations)
        value_shap = self.value_model.tree_shap(tree_idx, observations)
        return policy_shap, value_shap

    def get_total_iterations(self) -> int:
        return self.total_iterations
    
    def get_device(self) -> Tuple[str, str]:
        return self.policy_model.get_device(), self.value_model.get_device()

    def get_schedule_learning_rates(self) -> Tuple[int, int]:
        policy_lr, _ = self.policy_model.get_schedule_learning_rates()
        value_lr, _ = self.value_model.get_schedule_learning_rates()
        return policy_lr, value_lr
    
    def predict(self, observations: Union[np.array, th.Tensor], start_idx: int=0, stop_idx: int=None) -> Tuple[np.array, np.array]:
        preds = self.policy_model.predict(observations, start_idx, stop_idx)
        pred_values = self.value_model.predict(observations, start_idx, stop_idx).squeeze()
        if len(preds.shape) == 1:
            preds = preds[:, np.newaxis]
        return preds, pred_values
    
    def reset(self) -> None:
        self.policy_model.reset()
        self.value_model.reset()

    def save(self, filename: str) -> None:
        self.policy_model.save(filename + '_policy')
        self.value_model.save(filename + '_value')

    def export(self, filename: str) -> None:
        self.policy_model.export(filename + '_policy')
        self.value_model.export(filename + '_value')
    
    def print_tree(self, tree_idx: int) -> None:
        print("Policy ensemble")
        self.policy_model.print_tree(tree_idx)
        print("Value ensemble")
        self.value_model.print_tree(tree_idx)
    
    def plot_tree(self, tree_idx: int, filename: str) -> None:
        print("Policy ensemble")
        self.policy_model.plot_tree(tree_idx, filename.rstrip(".") + "_policy")
        print("Value ensemble")
        self.value_model.plot_tree(tree_idx,  filename.rstrip(".") + "_value")

    @classmethod
    def load(cls, filename: str) -> "SeparateActorCriticWrapper":
        instance = cls.__new__(cls)
        instance.policy_model = GBTWrapper.load(filename + '_policy')
        instance.value_model = GBTWrapper.load(filename + '_value')
        instance.tree_struct = instance.policy_model.tree_struct
        instance.total_iterations = instance.policy_model.iteration + instance.value_model.iteration
        instance.output_dim = instance.policy_model.output_dim
        instance.policy_optimizer = instance.policy_model.optimizer[0]
        instance.value_optimizer = instance.value_model.optimizer[0]
        instance.gbrl_params = instance.policy_model.gbrl_params
        instance.verbose = instance.policy_model.verbose
        instance.device = instance.policy_model.get_device()
        return instance

    def distil_policy(self, obs: Union[np.array, th.Tensor], targets: np.array, params: Dict) -> Tuple[int, Dict]:
        return self.policy_model.distil(obs, targets, params)

    def distil_value(self, obs: Union[np.array, th.Tensor], targets: np.array, params: Dict) -> Tuple[int, Dict]:
        return self.value_model.distil(obs, targets, params)

    def get_iteration(self) -> Tuple[int, int]:
        return self.policy_model.get_iteration(), self.value_model.get_iteration()
    
    def get_num_trees(self) -> Tuple[int, int]:
        return self.policy_model.get_num_trees(), self.value_model.get_num_trees()
    
    def set_bias(self, bias: np.array) -> None:
        self.policy_model.set_bias(bias)

    def copy(self) -> "SeparateActorCriticWrapper":
        return self.__copy__()
    
    def __copy__(self) -> "SeparateActorCriticWrapper":
        copy_ = SeparateActorCriticWrapper(self.output_dim, self.tree_struct.copy(), self.policy_optimizer.copy(), self.value_optimizer.copy(), self.gbrl_params, self.verbose, self.device)
        copy_.total_iterations = self.total_iterations
        copy_.policy_model = self.policy_model.copy()
        copy_.value_model = self.value_model.copy()
        return copy_
    

class SharedActorCriticWrapper(GBTWrapper):
    def __init__(self, output_dim: int, tree_struct: Dict, policy_optimizer: Dict, value_optimizer: Dict=None, gbrl_params: Dict=dict(), verbose: int = 0, device: str = 'cpu'):
        print('****************************************')
        print(f'Shared GBRL Tree with output dim: {output_dim}, tree_struct: {tree_struct} policy_optimizer: {policy_optimizer} value_optimizer: {value_optimizer}')
        print('****************************************')
        self.value_optimizer = value_optimizer
        self.policy_optimizer = policy_optimizer
        super().__init__(output_dim, tree_struct, policy_optimizer, gbrl_params, verbose, device)

    def reset(self) -> None:
        if self.cpp_model is not None:
            policy_lr, value_lr = self.get_schedule_learning_rates()
            self.policy_optimizer['init_lr'] = policy_lr
            if self.value_optimizer:
                self.value_optimizer['init_lr'] = value_lr
        self.cpp_model = GBRL_CPP(**self.params)
        if self.student_model is not None:
            self.policy_optimizer['T'] -= self.total_iterations
        else:
            self.total_iterations = 0
        try:
            self.cpp_model.set_optimizer(**self.policy_optimizer)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")
        if self.value_optimizer:
            if self.student_model is not None:
                self.value_optimizer['T'] -= self.total_iterations
            try:
                self.cpp_model.set_optimizer(**self.value_optimizer)
            except RuntimeError as e:
                print(f"Caught an exception in GBRL: {e}")
        
    def step(self, observations: Union[np.array, th.Tensor], theta_grad: np.array, value_grad: np.array=None) -> None:
        num_observations, cat_observations = preprocess_features(observations)
        target_grads = theta_grad 
        if value_grad is not None:
            value_grad = value_grad[:, np.newaxis]
            target_grads = np.concatenate([theta_grad, value_grad], axis=-1)
        target_grads = target_grads.astype(numerical_dtype)
        target_grads = target_grads.reshape((len(theta_grad), self.params['output_dim']))
        input_dim = 0 if num_observations is None else num_observations.shape[1]
        input_dim += 0 if cat_observations is None else cat_observations.shape[1]
        if self.feature_weights is None:
            self.feature_weights = np.ones(input_dim, dtype=numerical_dtype)
        assert len(self.feature_weights) == input_dim, "feature weights has to have the same number of elements as features"
        assert np.all(self.feature_weights >= 0), "feature weights contains non-positive values"
        self.cpp_model.step(num_observations, cat_observations, target_grads, self.feature_weights)
        self.iteration = self.cpp_model.get_iteration()
        self.total_iterations += 1

    def distil(self, obs: Union[np.array, th.Tensor], policy_targets: np.array, value_targets: np.array, params: Dict, verbose: int) -> Tuple[float, Dict]:
        targets = policy_targets.squeeze() 
        if self.value_optimizer is not None:
            targets = np.concatenate([policy_targets, value_targets[:, np.newaxis]], axis=1)
        return super().distil(obs, targets, params, verbose)
                                     
    def predict(self, observations: Union[np.array, th.Tensor], start_idx: int=0, stop_idx: int=None) -> Tuple[np.array, np.array]:
        if stop_idx is None:
            stop_idx = 0
        num_observations, cat_observations = preprocess_features(observations)
        if self.student_model is not None:
            preds = self.student_model.predict(num_observations, cat_observations)
            if num_observations is not None and num_observations.flags['WRITEABLE']:
                num_observations = num_observations.copy()
            if cat_observations is not None and not cat_observations.flags['WRITEABLE']:
                cat_observations = cat_observations.copy()
            self.cpp_model.predict(num_observations, cat_observations, preds, start_idx, stop_idx)
        else:
            preds = self.cpp_model.predict(num_observations, cat_observations, start_idx, stop_idx)
        pred_values = np.zeros(len(preds), dtype=np.single)
        if self.value_optimizer:
            pred_values = preds[:, -1]
            preds = preds[:, :-1]
        return preds, pred_values
    
    @classmethod
    def load(cls, filename: str) -> "SharedActorCriticWrapper":
        instance = super().load(filename) 
        instance.policy_optimizer = instance.optimizer[0]
        instance.value_optimizer = None if len(instance.optimizer) != 2 else instance.optimizer[1]
        return instance

    def copy(self) -> "SharedActorCriticWrapper":
        return self.__copy__()
    
    def __copy__(self) -> "SharedActorCriticWrapper":
        copy_ = SharedActorCriticWrapper(self.output_dim, self.tree_struct.copy(), self.policy_optimizer.copy(), None if self.value_optimizer is None else self.value_optimizer.copy(), self.gbrl_params, self.verbose, self.device)
        copy_.iteration = self.iteration 
        copy_.total_iterations = self.total_iterations
        if self.cpp_model is not None:
            copy_.model = GBRL_CPP(self.cpp_model)
        if self.student_model is not None:
            copy_.student_model = GBRL_CPP(self.student_model)
        return copy_


    
        
        

