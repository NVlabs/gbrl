##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
from typing import Dict, Sequence, Optional, Tuple, Union

import numpy as np
import torch as th
from scipy.special import binom

from gbrl.common.config import APPROVED_OPTIMIZERS, VALID_OPTIMIZER_ARGS

numerical_dtype = np.dtype('float32')
categorical_dtype = np.dtype('S128')
NumericalData = Union[np.ndarray, th.Tensor]
TensorInfo = Tuple[int, Tuple[int, ...], str, str]


def get_tensor_info(tensor: th.Tensor) -> TensorInfo:
    """Extracts pytorch tensor information for usage in C++

    Args:
        tensor (th.Tensor): input tensor

    Returns:
        Tuple[int, Tuple[int, ...], str, str]: raw data pointer, tensor shape,
        tensor dtype, device.
    """
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    data_ptr = tensor.data_ptr()
    shape = tuple(tensor.size())  # Convert torch.Size to tuple
    dtype = str(tensor.dtype)
    device = 'cuda' if tensor.is_cuda else 'cpu'
    return (data_ptr, shape, dtype, device)


def process_array(arr: np.ndarray) -> Tuple[Optional[np.ndarray],
                                            Optional[np.ndarray]]:
    """ Formats numpy array for C++ GBRL.
    """
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype,
                                                              np.integer):
        return np.ascontiguousarray(arr, dtype=numerical_dtype), None
    elif arr.dtype == categorical_dtype or np.issubdtype(arr.dtype, np.str_):
        fixed_str = np.char.encode(arr.astype(str), 'utf-8'
                                   ).astype(categorical_dtype)
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
            numerical_array = np.ascontiguousarray(arr[numerical_mask]).astype(
                numerical_dtype) if arr.ndim == 1 else np.ascontiguousarray(
                arr[:, numerical_mask].astype(numerical_dtype)
            )
        else:
            numerical_array = None
        # Check if there are any categorical columns
        if np.any(categorical_mask):
            # Select categorical columns and convert to categorical_dtype
            categorical_array = arr[categorical_mask] if arr.ndim == 1 else \
                arr[:, categorical_mask]
            categorical_array = np.char.encode(categorical_array.astype(str),
                                               'utf-8').astype(
                                                   categorical_dtype)
        else:
            categorical_array = None

        return numerical_array, categorical_array

    else:
        raise ValueError(f"Unsupported array data type: {arr.dtype}")


def to_numpy(arr: NumericalData) -> np.ndarray:
    if isinstance(arr, th.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.ascontiguousarray(arr, dtype=numerical_dtype)


def setup_optimizer(optimizer: Dict, prefix: str = '') -> Dict:
    """Setup optimizer to correctly align with GBRL C++ module.

    Processes and validates optimizer configuration dictionary, ensuring it contains
    the required parameters and is compatible with the GBRL C++ backend. Handles
    learning rate scheduling and parameter prefixes.

    Args:
        optimizer (Dict): Optimizer configuration dictionary containing parameters
            like 'start_idx', 'stop_idx', 'lr', 'algo', etc.
        prefix (str, optional): Optimizer parameter prefix names such as:
            'mu_', 'std_', 'policy_', 'value_', etc. Defaults to ''.

    Returns:
        Dict: Modified optimizer dictionary with validated and processed parameters.

    Raises:
        AssertionError: If required parameters are missing or invalid.
    """
    assert isinstance(optimizer, dict), 'optimization must be a dictionary'
    assert 'start_idx' in optimizer, "optimizer must have a start idx"
    assert 'stop_idx' in optimizer, "optimizer must have a stop idx"
    if prefix:
        optimizer = {k.replace(prefix, ''): v for k, v in optimizer.items()}
    lr = optimizer.get('lr', 1.0) if 'init_lr' not in optimizer else \
        optimizer['init_lr']
    # setup scheduler
    optimizer['scheduler'] = 'Const'
    assert isinstance(lr, (int, float, str)), "lr must be a float or string"
    if isinstance(lr, str) and 'lin_' in lr:
        assert 'T' in optimizer, "Linear optimizer must contain T the total"
        "   number of iterations used for scheduling"
        lr = lr.replace('lin_', '')
        optimizer['scheduler'] = 'Linear'
    optimizer['init_lr'] = float(lr)
    optimizer['algo'] = optimizer.get('algo', 'SGD')
    assert optimizer['algo'] in APPROVED_OPTIMIZERS, \
        f"optimization algo has to be in {APPROVED_OPTIMIZERS}"
    return {k: v for k, v in optimizer.items() if k in VALID_OPTIMIZER_ARGS
            and v is not None}


def clip_grad_norm(grads: NumericalData, grad_clip: Optional[float]) ->\
      NumericalData:
    """clip per sample gradients according to their norm

    Args:
        grads (NumericalData): gradients
        grad_clip (float, optional): gradient clip value

    Returns:
        NumericalData: clipped gradients
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


def get_input_dim(arr: NumericalData) -> int:
    """Returns the column dimension of a 2D array

    Args:
        arr (NumericalData):input array

    Returns:
        int: Number of input features/dimensions.
    """
    if isinstance(arr, Tuple):
        num_arr, cat_arr = arr
        return get_input_dim(num_arr) + get_input_dim(cat_arr)
    return 1 if len(arr.shape) == 1 else arr.shape[1]


def get_norm_values(base_poly: np.ndarray) -> np.ndarray:
    """
    Precompute normalization values for linear tree SHAP computation.

    Calculates normalization weights and values used in the linear tree SHAP
    algorithm. These values are precomputed to optimize SHAP value calculations
    across multiple tree evaluations.

    Args:
        base_poly (np.ndarray): Base polynomial coefficients (typically Chebyshev points).

    Returns:
        np.ndarray: Normalization values matrix of shape (depth+1, depth).

    References:
        https://github.com/yupbank/linear_tree_shap/blob/main/linear_tree_shap/utils.py
    """
    depth = base_poly.shape[0]
    norm_values = np.zeros((depth+1, depth))
    for i in range(1, depth+1):
        norm_weights = binom(i-1, np.arange(i))
        norm_values[i, :i] = np.linalg.inv(np.vander(base_poly[:i]).T).dot(
            1. / norm_weights)
    return norm_values


def get_poly_vectors(max_depth: int, dtype: np.dtype) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns polynomial vectors/matrices used in the calculation of linear tree SHAP.

    Generates Chebyshev polynomial vectors and associated matrices required for
    computing SHAP values using the linear tree SHAP algorithm. Based on the
    implementation described in "Linear TreeShap" by Yu et al, 2023.

    Args:
        max_depth (int): Maximum tree depth for generating polynomials.
        dtype (np.dtype): Data type for the output arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - base_polynomial: Chebyshev points of the second kind, scaled to [2,3]
            - normalization_values: Precomputed normalization values for SHAP
            - offset: Vandermonde matrix for polynomial evaluation

    References:
        https://arxiv.org/pdf/2209.08192
    """
    base_poly = np.polynomial.chebyshev.chebpts2(max_depth).astype(dtype)
    a = 2  # Lower bound of the new interval
    b = 3  # Upper bound of the new interval
    base_poly = (base_poly + 1) * (b - a) / 2 + a
    norm_values = get_norm_values(base_poly).astype(dtype)
    offset = np.vander(base_poly + 1).T[::-1].astype(dtype)
    return base_poly, norm_values, offset


def ensure_same_type(arr_a: NumericalData,
                     arr_b: NumericalData) -> \
                        Tuple[NumericalData, NumericalData]:
    """Ensures both arrays are of the same type (either Tensor or ndarray).

    If the arrays are of different types, transforms array B to match the type
    and device of array A. This is useful for operations that require both
    operands to be of the same type.

    Args:
        arr_a (NumericalData): array A
        arr_b (NumericalData): array B
    Returns:
        Tuple[NumericalData, NumericalData]
    """
    if isinstance(arr_a, th.Tensor) and not isinstance(arr_b, th.Tensor):
        arr_b = th.tensor(arr_b, device=arr_a.device).float()
    elif isinstance(arr_a, np.ndarray) and not isinstance(arr_b, np.ndarray):
        arr_b = np.ascontiguousarray(arr_b.detach().cpu().numpy(),
                                     dtype=numerical_dtype)
    return arr_a, arr_b


def concatenate_arrays(arrays: Sequence[NumericalData],
                       axis: int = 1) -> \
                        NumericalData:
    """
    Concatenates multiple arrays along a specified axis. All arrays must be of the same type
    (either all NumPy arrays or all PyTorch tensors). If an array has fewer dimensions than
    required for concatenation, an axis is added to match the dimensionality.

    Args:
        arrays (Sequence[NumericalData]): Sequence of arrays (NumPy or PyTorch) to concatenate.
        axis (int, optional): Axis along which to concatenate. Defaults to 1.

    Returns:
        NumericalData: Concatenated array with the type and device of the first array.

    Raises:
        AssertionError: If fewer than two arrays are provided or if array types do not match.
    """
    assert len(arrays) > 1, "Need at least two arrays to concatenate"
    sequence_type = type(arrays[0])
    for arr in arrays[1:]:
        assert isinstance(arr, sequence_type), "All arrays must be of the same type"

    # Check if we need to add an axis to match dimensionality
    def add_axis_if_needed(array, target_ndim, axis):
        if array.ndim < target_ndim or array.ndim == 1:
            if isinstance(array, th.Tensor):
                array = array.unsqueeze(axis)
            else:  # For NumPy array
                array = np.expand_dims(array, axis=axis)
        return array

    # Ensure all arrays have at least the right number of dimensions for
    # concatenation
    max_ndim = max([arr.ndim for arr in arrays])
    arrays = [add_axis_if_needed(arr, max_ndim, axis) for arr in arrays]

    if isinstance(arrays[0], th.Tensor):
        return th.cat(arrays, dim=axis)  # type: ignore
    return np.concatenate(arrays, axis=axis)


def pad_array(array: NumericalData, n_dims: int, pad_value: float = 0.0, axis: int = -1) -> NumericalData:
    """
    Pads an array with singleton dimensions to ensure it has at least
    `n_dims` dimensions along the specified axis.

    Args:
        array (NumericalData): Input array (NumPy or PyTorch).
        n_dims (int): Minimum number of dimensions required.
        pad_value (float, optional): Value to use for padding. Defaults to 0.0.
        axis (int, optional): Axis along which to pad. Defaults to 1.

    Returns:
        NumericalData: Padded array with the same type as the input.
    """
    if isinstance(array, th.Tensor):
        return concatenate_arrays([array, pad_value*th.ones((len(array), n_dims), dtype=array.dtype, device=array.device)],
                                  axis=axis)
    return concatenate_arrays([array, pad_value*np.ones((len(array), n_dims), dtype=array.dtype)], axis=axis)


def validate_array(arr: NumericalData) -> None:
    """Checks for NaN and Inf values in an array/tensor.

    Args:
        arr (NumericalData): array/tensor
    """
    if isinstance(arr, np.ndarray):
        assert not np.isnan(arr).any(), "nan in array"
        assert not np.isinf(arr).any(), "infinity in array"
    else:
        assert not th.isnan(arr).any(), "nan in tensor"
        assert not th.isinf(arr).any(), "infinity in tensor"


def constant_like(arr: NumericalData,
                  constant: float = 1) -> NumericalData:
    """Returns a ones array with the same shape as arr multiplid by a constant

    Args:
        arr (NumericalData): array
    """
    if isinstance(arr, th.Tensor):
        return th.ones_like(arr, device=arr.device) * constant
    else:
        return np.ones_like(arr) * constant


def separate_numerical_categorical(arr: np.ndarray) -> Tuple[Optional[np.ndarray],
                                                             Optional[np.ndarray]]:
    """Separate a numpy array to a categorical and numerical numpy arrays.
    Args:
        arr (np.ndarray): Input array, tuple, list, or dictionary to separate.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing:
            - numerical_array: Array with numerical data or None
            - categorical_array: Array with categorical data or None
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


def preprocess_features(arr: NumericalData) -> Tuple[Optional[NumericalData],
                                                     Optional[np.ndarray]]:
    """
    Preprocess array such that dimensions and data types match GBRL requirements.

    Separates input array into numerical and categorical features, ensuring proper
    dimensionality for the C++ GBRL backend. Handles 1D to 2D conversion and
    squeezing of excess dimensions.

    Args:
        arr (np.ndarray): Input array containing features to preprocess.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing:
            - numerical_features: Processed numerical features or None
            - categorical_features: Processed categorical features or None
    """
    if isinstance(arr, th.Tensor):
        return arr, None

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


def ensure_leaf_tensor_or_array(array: NumericalData,
                                tensor: bool,
                                requires_grad: bool,
                                device: str) -> NumericalData:
    """
    Ensures that the output is:
    1) A PyTorch **leaf tensor** if both `tensor=True` and
    `requires_grad=True`.
    2) A PyTorch tensor (detached) if `tensor=True` and `requires_grad=False`.
    3) A NumPy array if `tensor=False`.

    - If `tensor=True` and `requires_grad=True`, ensures the tensor is a
    **leaf tensor**.
    - If `tensor=True` and `requires_grad=False`, ensures `requires_grad` is
    disabled.
    - If `tensor=False`, converts to NumPy if necessary.

    Args:
        array (NumericalData): Input array (NumPy array or PyTorch tensor).
        tensor (bool): If True, ensures output is a PyTorch tensor.
        requires_grad (bool): If True and `tensor=True`, ensures output is a
        **leaf tensor**.

    Returns:
        NumericalData: A PyTorch tensor (if `tensor=True`) or a NumPy array
        (if `tensor=False`).
    """
    if tensor:
        if isinstance(array, np.ndarray):
            array = th.from_numpy(array).to(device)
        else:
            array = array.detach()
        array.requires_grad_(requires_grad)
    elif not tensor and isinstance(array, th.Tensor):
        array = array.detach().cpu().numpy()

    return array
