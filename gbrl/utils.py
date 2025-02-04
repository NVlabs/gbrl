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

from gbrl.config import APPROVED_OPTIMIZERS, VALID_OPTIMIZER_ARGS

import numpy as np 
# Define custom dtypes
numerical_dtype = np.dtype('float32')
categorical_dtype = np.dtype('S128')  

def get_tensor_info(tensor: th.Tensor) -> Tuple[int, Tuple[int, ...], str, str]:
    """Extracts pytorch tensor information for usage in C++

    Args:
        tensor (th.Tensor): input tensor

    Returns:
        Tuple[int, Tuple[int, ...], str, str]: raw data pointer, tensor shape, tensor dtype, device
    """
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    data_ptr = tensor.data_ptr()
    shape = tuple(tensor.size())  # Convert torch.Size to tuple
    dtype = str(tensor.dtype)
    device = 'cuda' if tensor.is_cuda else 'cpu'
    return (data_ptr, shape, dtype, device)

def process_array(arr: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    """ Formats numpy array for C++ GBRL.
    """
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
        return np.ascontiguousarray(arr, dtype=numerical_dtype), None 
    elif arr.dtype == categorical_dtype or np.issubdtype(arr.dtype, np.str_):
        fixed_str = np.char.encode(arr.astype(str), 'utf-8').astype(categorical_dtype)
        return None, np.ascontiguousarray(fixed_str)
    elif arr.dtype == object:
        # Get the first row
        if arr.ndim == 1:
            # For 1D array, use the array itself as first_row
            first_row = arr
        else:
            # For 2D array, get the first row
            first_row = arr[0]
        # Vectorized function to check if a type is numerical
        is_numerical_type = np.vectorize(
            lambda x: isinstance(x, (int, float, np.integer, np.floating))
        )(first_row)

        # Create masks for numerical and categorical columns
        numerical_mask = is_numerical_type
        categorical_mask = ~is_numerical_type
        # Check if there are any numerical columns
        if np.any(numerical_mask):
            # Select numerical columns and convert to numerical_dtype
            numerical_array = np.ascontiguousarray(arr[numerical_mask]).astype(numerical_dtype) if arr.ndim == 1 else np.ascontiguousarray(
                arr[:, numerical_mask].astype(numerical_dtype)
            )
        else:
            numerical_array = None
        # Check if there are any categorical columns
        if np.any(categorical_mask):
            # Select categorical columns and convert to categorical_dtype
            categorical_array = arr[categorical_mask] if arr.ndim == 1 else arr[:, categorical_mask]
            categorical_array = np.char.encode(categorical_array.astype(str), 'utf-8').astype(categorical_dtype)
        else:
            categorical_array = None

        return numerical_array, categorical_array

    else:
        raise ValueError(f"Unsupported array data type: {arr.dtype}")
    
def to_numpy(arr: Union[np.ndarray, th.Tensor]) -> np.ndarray:
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
    return {k: v for k, v in optimizer.items() if k in VALID_OPTIMIZER_ARGS and v is not None}


def clip_grad_norm(grads: Union[np.ndarray, th.Tensor], grad_clip: float) -> Union[np.ndarray, th.Tensor]:
    """clip per sample gradients according to their norm

    Args:
        grads (Union[np.ndarray, th.Tensor]): gradients
        grad_clip (float): gradient clip value

    Returns:
        Union[np.ndarray, th.Tensor]: clipped gradients
    """
    if grad_clip is None or grad_clip == 0.0:
        return grads
    if len(grads.shape) == 1:
        if isinstance(grads, th.Tensor):
            grads = th.clamp(grads, min=-grad_clip, max=grad_clip)
        else:
            grads = np.clip(grads, a_min=-grad_clip, a_max=grad_clip)
        return grads 
    if isinstance(grads, th.Tensor):
        grad_norms = th.norm(grads, p=2, dim=1, keepdim=True)
    else:
        grad_norms = np.linalg.norm(grads, axis=1, ord=2, keepdims=True)
    mask = (grad_norms > grad_clip).squeeze()
    grads[mask] = grad_clip * grads[mask] / grad_norms[mask]
    return grads


def get_input_dim(arr: Union[np.ndarray, th.Tensor]) -> int:
    """Returns the column dimension of a 2D array

    Args:
        arr (Union[np.ndarray, th.Tensor]):input array

    Returns:
        int: input dimension
    """
    if isinstance(arr, Tuple):
        num_arr, cat_arr = arr
        return get_input_dim(num_arr) + get_input_dim(cat_arr)
    return 1 if len(arr.shape) == 1 else arr.shape[1]


def get_norm_values(base_poly: np.ndarray) -> np.ndarray:
    """Precompute normalization values for linear tree shap
    See https://github.com/yupbank/linear_tree_shap/blob/main/linear_tree_shap/utils.py

    Args:
        base_poly (np.ndarray): base polynomial

    Returns:
        np.ndarray: normalization values
    """
    depth = base_poly.shape[0]
    norm_values = np.zeros((depth+1, depth))
    for i in range(1, depth+1):
        norm_weights = binom(i-1, np.arange(i))
        norm_values[i,:i] = np.linalg.inv(np.vander(base_poly[:i]).T).dot(1./norm_weights) 
    return norm_values


def get_poly_vectors(max_depth: int, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns polynomial vectors/matrices used in the calculation of linear tree shap
    See https://arxiv.org/pdf/2209.08192
    Args:
        max_depth (int)
        dtype (np.dtype)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: base_polynomial (chebyshev of the second kind), normalization values, offset
    """
    base_poly = np.polynomial.chebyshev.chebpts2(max_depth).astype(dtype)
    a = 2  # Lower bound of the new interval
    b = 3  # Upper bound of the new interval
    base_poly = (base_poly + 1) * (b - a) / 2 + a
    norm_values = get_norm_values(base_poly).astype(dtype)
    offset = np.vander(base_poly + 1).T[::-1].astype(dtype)
    return base_poly, norm_values, offset


def ensure_same_type(arr_a: Union[th.Tensor, np.ndarray], arr_b: Union[th.Tensor, np.ndarray]) -> Tuple[Union[th.Tensor, np.ndarray], Union[th.Tensor, np.ndarray]]:
    """Ensures both arrays are of the same type (either Tensor or ndarray).
       If not, transforms array B to the type and device of array A.
    Args:
        arr_a (Union[th.Tensor, np.ndarray]): array A
        arr_b (Union[th.Tensor, np.ndarray]): array B
    Returns:
        Tuple[Union[th.Tensor, np.ndarray], Union[th.Tensor, np.ndarray]]: _description_
    """
    if isinstance(arr_a, th.Tensor) and not isinstance(arr_b, th.Tensor):
        arr_b = th.tensor(arr_b, device=arr_a.device).float()
    elif isinstance(arr_a, np.ndarray) and not isinstance(arr_b, np.ndarray):
        arr_b = np.ascontiguousarray(arr_b.detach().cpu().numpy(), dtype=numerical_dtype)
    return arr_a, arr_b


def concatenate_arrays(arr_a: Union[th.Tensor, np.ndarray], arr_b: Union[th.Tensor, np.ndarray], axis: int = 1) -> Union[th.Tensor, np.ndarray]:
    """Concatenates to arrays together. If both arrays are not of the same type then transforms array B to the type and device of array A.

    Args:
        arr_a (Union[th.Tensor, np.ndarray]): array A
        arr_b (Union[th.Tensor, np.ndarray]): Array B
        axis (int, optional): concatenation axis. Defaults to 1.

    Returns:
        Union[th.Tensor, np.ndarray]: concatenated array of device and type of array A
    """
    arr_a, arr_b = ensure_same_type(arr_a, arr_b)
        # Check if we need to add an axis to match dimensionality
    def add_axis_if_needed(array, target_ndim, axis):
        if array.ndim < target_ndim:
            if isinstance(array, th.Tensor):
                array = array.unsqueeze(axis)
            else:  # For NumPy array
                array = np.expand_dims(array, axis=axis)
        return array

    # Ensure both arrays have at least the right number of dimensions for concatenation
    max_ndim = max(arr_a.ndim, arr_b.ndim)
    arr_a = add_axis_if_needed(arr_a, max_ndim, axis)
    arr_b = add_axis_if_needed(arr_b, max_ndim, axis)
    
    return th.cat([arr_a, arr_b], dim=axis) if isinstance(arr_a, th.Tensor) else np.concatenate([arr_a, arr_b], axis=axis)


def validate_array(arr: Union[th.Tensor, np.ndarray]) -> None:
    """Checks for NaN and Inf values in an array/tensor.

    Args:
        arr (Union[th.Tensor, np.ndarray]): array/tensor
    """
    if isinstance(arr, np.ndarray):
        assert not np.isnan(arr).any(), "nan in array"
        assert not np.isinf(arr).any(), "infinity in array"
    else:
        assert not th.isnan(arr).any(), "nan in tensor"
        assert not th.isinf(arr).any(), "infinity in tensor"

def constant_like(arr: Union[th.Tensor, np.ndarray], constant: float = 1) -> None:
    """Returns a ones array with the same shape as arr multiplid by a constant

    Args:
        arr (Union[th.Tensor, np.ndarray]): array
    """
    if isinstance(arr, th.Tensor):
        return th.ones_like(arr, device=arr.device) * constant 
    else:
        return np.ones_like(arr) * constant
    
def separate_numerical_categorical(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
    """Separate a numpy array to a categorical and numerical numpy arrays.
    Args:
        arr (np.ndarray): array

    Returns:
        Tuple[np.ndarray, np.ndarray]: numerical and categorical arrays
    """
    if isinstance(arr, tuple):
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
       
def preprocess_features(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess array such that the dimensions and the data type match.
    Returns numerical and categorical features. 
    May return None for each if purely numerical or purely categorical.
    Args:

    Returns:
        Tuple[np.ndarray, np.ndarray]
    """
    input_dim = get_input_dim(arr)
    num_arr, cat_arr = separate_numerical_categorical(arr)
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


def tensor_to_leaf(array: Union[th.Tensor, np.ndarray], requires_grad: bool = True) -> Union[th.Tensor, np.ndarray]:
    """Ensure a tensor requiring a gradient is a leaf tensor.
    Numpy arrays are ignored

    Args:
        array (Union[th.Tensor, np.ndarray]): input array
        requires_grad (bool, optional) Defaults to True.

    Returns:
        Union[th.Tensor, np.ndarray]: leaf tensor 
    """
    if isinstance(array, np.ndarray):
        return array
    array = array.detach()
    array.requires_grad_(requires_grad)
    return array

