##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from typing import Dict, Union, Tuple
import numpy as np
import torch as th
from scipy.special import binom

from .config import APPROVED_OPTIMIZERS, VALID_OPTIMIZER_ARGS

import numpy as np 
# Define custom dtypes
numerical_dtype = np.dtype('float32')
categorical_dtype = np.dtype('S128')  


def process_array(arr: np.array)-> Tuple[np.array, np.array]:
    """ Formats numpy array for C++ GBRL.
    """
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=numerical_dtype), None 
    else:
        fixed_str = np.char.encode(arr.astype(str), 'utf-8').astype(categorical_dtype)
        return None, np.ascontiguousarray(fixed_str)
    
def to_numpy(arr: Union[np.array, th.Tensor]) -> np.array:
    if isinstance(arr, th.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.ascontiguousarray(arr, dtype=numerical_dtype)

def setup_optimizer(optimizer: Dict, prefix: str='') -> Dict:
    """Setup optimizer to correctly allign with GBRL C++ module

    Args:
        optimizer (Dict): optimizer dictionary
        prefix (str, optional): optimizer parameter prefix names such as: mu_lr, mu_algo, std_algo, policy_algo, value_algo, etc. Defaults to ''.
    Returns:
        Dict: modified optimizer dictionary
    """
    assert isinstance(optimizer, dict), 'optimization must be a dictionary'
    assert 'start_idx' in optimizer, "optimizer must have a start idx"
    assert 'stop_idx' in optimizer, "optimizer must have a stop idx"
    if prefix:
        optimizer = {k.replace(prefix, ''): v for k, v in optimizer.items()}
    lr = optimizer.get('lr', 1.0) if 'init_lr' not in optimizer else optimizer['init_lr']
    # setup scheduler
    optimizer['scheduler'] = 'Const'
    assert isinstance(lr, int) or isinstance(lr, float) or isinstance(lr, str), "lr must be a float or string"
    if isinstance(lr, str) and 'lin_' in lr:
        assert 'T' in optimizer, "Linear optimizer must contain T the total number of iterations used for scheduling"
        lr = lr.replace('lin_', '')
        optimizer['scheduler'] = 'Linear'
    optimizer['init_lr'] = float(lr)
    optimizer['algo'] = optimizer.get('algo', 'SGD')
    assert optimizer['algo'] in APPROVED_OPTIMIZERS, f"optimization algo has to be in {APPROVED_OPTIMIZERS}"
    # optimizer['stop_lr'] = optimizer.get('stop_lr', 1.0e-8)
    # optimizer['beta_1'] = optimizer.get('beta_1', 0.9)
    # optimizer['beta_2'] = optimizer.get('beta_2', 0.999)
    # optimizer['eps'] = optimizer.get('eps', 1.0e-5)
    # optimizer['shrinkage'] = optimizer.get('shrinkage', 0.0)
    # if optimizer['shrinkage'] is None:
    #     optimizer['shrinkage'] = 0.0

    return {k: v for k, v in optimizer.items() if k in VALID_OPTIMIZER_ARGS and v is not None}


def clip_grad_norm(grads: np.array, grad_clip: float) -> np.array:
    """clip per sample gradients according to their norm

    Args:
        grads (np.array): gradients
        grad_clip (float): gradient clip value

    Returns:
        np.array: clipped gradients
    """
    grads = to_numpy(grads)
    if grad_clip is None or grad_clip == 0.0:
        return grads
    if len(grads.shape) == 1:
        grads = np.clip(grads, a_min=-grad_clip, a_max=grad_clip)
        return grads 
    grad_norms = np.linalg.norm(grads, axis=1)
    grads[grad_norms > grad_clip] = grad_clip*grads[grad_norms > grad_clip] / grads[grad_norms > grad_clip]
    return grads


def get_input_dim(arr: Union[np.array, th.Tensor]) -> int:
    """Returns the column dimension of a 2D array

    Args:
        arr (np.array):input array

    Returns:
        int: input dimension
    """
    if isinstance(arr, Tuple):
        num_arr, cat_arr = arr
        return get_input_dim(num_arr) + get_input_dim(cat_arr)
    return 1 if len(arr.shape) == 1 else arr.shape[1]


def get_norm_values(base_poly: np.array) -> np.array:
    """Precompute normalization values for linear tree shap
    See https://github.com/yupbank/linear_tree_shap/blob/main/linear_tree_shap/utils.py

    Args:
        base_poly (np.array): base polynomial

    Returns:
        np.array: normalization values
    """
    depth = base_poly.shape[0]
    norm_values = np.zeros((depth+1, depth))
    for i in range(1, depth+1):
        norm_weights = binom(i-1, np.arange(i))
        norm_values[i,:i] = np.linalg.inv(np.vander(base_poly[:i]).T).dot(1./norm_weights) 
    return norm_values


def get_poly_vectors(max_depth: int, dtype: np.dtype) -> Tuple[np.array, np.array, np.array]:
    """Returns polynomial vectors/matrices used in the calculation of linear tree shap
    See https://arxiv.org/pdf/2209.08192
    Args:
        max_depth (int)
        dtype (np.dtype)

    Returns:
        Tuple[np.array, np.array, np.array]: base_polynomial (chebyshev of the second kind), normalization values, offset
    """
    base_poly = np.polynomial.chebyshev.chebpts2(max_depth).astype(dtype)
    a = 2  # Lower bound of the new interval
    b = 3  # Upper bound of the new interval
    base_poly = (base_poly + 1) * (b - a) / 2 + a
    norm_values = get_norm_values(base_poly).astype(dtype)
    offset = np.vander(base_poly + 1).T[::-1].astype(dtype)
    return base_poly, norm_values, offset