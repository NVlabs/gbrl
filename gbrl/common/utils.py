##############################################################################
# Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
##############################################################################
"""
GBRL Utility Functions

This module provides utility functions for data preprocessing, array manipulation,
tensor operations, optimizer setup, and SHAP value computation used throughout
the GBRL library.
"""
from typing import Dict, List, Sequence, Optional, Tuple, Union

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
    """
    Formats numpy array for C++ GBRL by separating numerical and categorical data.

    This function processes input arrays and separates them into numerical and
    categorical components based on their data types. It handles various dtype
    formats including floating point, integer, string, and object arrays.

    Args:
        arr (np.ndarray): Input array to process.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing:
            - numerical_array: Array with numerical data (float32) or None
            - categorical_array: Array with categorical data (S128) or None

    Raises:
        ValueError: If the array has an unsupported data type.
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


def get_index_mapping(arr: NumericalData) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a mapping from original column indices to their new \
        indices after separating numerical and categorical features."""
    if not isinstance(arr, th.Tensor):
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

        numerical_indices = np.where(numerical_mask)[0]
        categorical_indices = np.where(categorical_mask)[0]
        # Create the index mapping array
        index_mapping = np.empty_like(np.arange(arr.shape[-1]), dtype=int)
        index_mapping[numerical_indices] = np.arange(len(numerical_indices))
        index_mapping[categorical_indices] = np.arange(len(categorical_indices))

        # Boolean mask: True for categorical, False for numerical
        numerical_mask = np.zeros(arr.shape[-1], dtype=bool)
        numerical_mask[numerical_indices] = True

        return index_mapping, numerical_mask
    else:
        return np.arange(arr.shape[-1]), np.ones(arr.shape[-1], dtype=bool)


def to_numpy(arr: Union[np.ndarray, th.Tensor]) -> np.ndarray:
    if isinstance(arr, th.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.ascontiguousarray(arr, dtype=numerical_dtype)


def normalize_vector_input(data: Union[float, NumericalData]) -> Union[np.ndarray, TensorInfo]:
    """
    Normalizes scalar, numpy array, or torch tensor input to a 1D vector for C++ pybind.

    This function handles conversion and reshaping of various input types to ensure
    the output is always a 1D vector suitable for passing to C++ pybind interfaces.
    It converts scalars to numpy arrays, flattens multi-dimensional arrays, and
    reshapes 0-dimensional arrays. For PyTorch tensors, returns TensorInfo tuple
    containing raw pointer information for direct C++ access.

    Args:
        data (Union[float, NumericalData]): Input data which can be:
            - A Python float or int
            - A 0D, 1D, or multi-dimensional numpy array
            - A 0D, 1D, or multi-dimensional PyTorch tensor

    Returns:
        Union[np.ndarray, TensorInfo]:
            - For float/int/numpy: A contiguous 1D numpy array with dtype float32
            - For PyTorch tensors: TensorInfo tuple (data_ptr, shape, dtype, device)
              containing raw pointer information for C++ backend

    Examples:
        >>> normalize_vector_input(3.14)
        # Returns: np.array([3.14], dtype=float32)

        >>> normalize_vector_input(np.array(5.0))
        # Returns: np.array([5.0], dtype=float32)

        >>> normalize_vector_input(np.array([[1, 2], [3, 4]]))
        # Returns: np.array([1, 2, 3, 4], dtype=float32)

        >>> normalize_vector_input(torch.tensor([[1.0, 2.0]]))
        # Returns: TensorInfo(data_ptr, (2,), 'torch.float32', 'cpu')
    """
    # Convert float to numpy array
    if isinstance(data, (float, int)):
        return np.ascontiguousarray(np.array([data], dtype=numerical_dtype))

    assert isinstance(data, (np.ndarray, th.Tensor)), \
        "Input must be a float, numpy array, or PyTorch tensor"

    # Handle array shape
    if data.ndim == 0:
        # 0D tensor: reshape to 1D
        data = data.reshape(1)
    elif data.ndim > 1:
        # Multi-dimensional: flatten
        data = data.flatten()

    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data.astype(numerical_dtype))
    return get_tensor_info(data)


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
    # setup scheduler - check if explicitly set, otherwise default to Const
    if 'scheduler' not in optimizer:
        optimizer['scheduler'] = 'Const'
    assert isinstance(lr, (int, float, str)), "lr must be a float or string"
    if isinstance(lr, str) and 'lin_' in lr:
        if 'T' not in optimizer:
            raise ValueError("Linear scheduler requires 'T' (total number of iterations) to be specified.")
        lr = lr.replace('lin_', '')
        optimizer['scheduler'] = 'Linear'
    # Validate scheduler type before normalization
    sched_value = optimizer.get('scheduler', 'Const')
    if not isinstance(sched_value, str):
        raise ValueError("scheduler must be a string ('linear', 'const', or 'constant')")
    # Normalize scheduler name (linear -> Linear, const/constant -> Const)
    sched = sched_value.lower()
    if sched == 'linear':
        optimizer['scheduler'] = 'Linear'
    elif sched in ('const', 'constant'):
        optimizer['scheduler'] = 'Const'
    else:
        raise ValueError(f"Unknown scheduler '{sched}'. Must be 'linear', 'const', or 'constant'.")
    # Validate 'T' is present and usable for Linear scheduler
    if optimizer['scheduler'] == 'Linear':
        if 'T' not in optimizer:
            raise ValueError("Linear scheduler requires 'T' (total number of iterations) to be specified.")
        # T divides in get_lr(), so a non-positive or non-integer T is unusable.
        T = optimizer['T']
        if isinstance(T, bool) or not isinstance(T, (int, np.integer)) or T < 1:
            raise ValueError(f"Linear scheduler 'T' must be an integer >= 1, got {T!r}")
    optimizer['init_lr'] = float(lr)
    if optimizer['init_lr'] <= 0:
        raise ValueError("init_lr must be > 0")
    # A schedule crossing zero flips the sign of the leaf-value -> prediction
    # transform partway through the ensemble, which the monotonic projection
    # assumes is constant.  NaN/inf must be rejected explicitly: NaN fails every
    # comparison, so a bare `<= 0` test lets it through.
    if optimizer.get('stop_lr') is not None:
        stop_lr = float(optimizer['stop_lr'])
        if not np.isfinite(stop_lr) or stop_lr <= 0:
            raise ValueError(f"stop_lr must be a finite value > 0, got {optimizer['stop_lr']}")
        optimizer['stop_lr'] = stop_lr
    if not np.isfinite(optimizer['init_lr']):
        raise ValueError(f"init_lr must be finite, got {optimizer['init_lr']}")
    optimizer['algo'] = optimizer.get('algo', 'SGD')
    assert optimizer['algo'] in APPROVED_OPTIMIZERS, \
        f"optimization algo has to be in {APPROVED_OPTIMIZERS}"
    return {k: v for k, v in optimizer.items() if k in VALID_OPTIMIZER_ARGS
            and v is not None}


def validate_optimizer_ranges(optimizers: Union[Dict, List[Dict]]) -> None:
    """Reject optimizers whose output ranges overlap.

    Each output dimension must be covered by at most one optimizer. Overlapping
    ranges make the per-dimension update ambiguous and produce incorrect SHAP
    values, so they are rejected before the C++ model is built.

    Args:
        optimizers (Union[Dict, List[Dict]]): One optimizer dict or a list of them.

    Raises:
        ValueError: If any two optimizers cover the same output dimension.
    """
    if isinstance(optimizers, dict):
        optimizers = [optimizers]
    seen = []
    for opt in optimizers:
        start, stop = opt.get('start_idx'), opt.get('stop_idx')
        if start is None or stop is None:
            continue
        # An empty or reversed interval never overlaps anything, so the check
        # below would wave it through even though C++ rejects it later.
        if (isinstance(start, bool) or isinstance(stop, bool)
                or not isinstance(start, (int, np.integer))
                or not isinstance(stop, (int, np.integer))):
            raise ValueError(
                f"optimizer start_idx/stop_idx must be integers, got "
                f"{start!r}/{stop!r}")
        if start < 0 or start >= stop:
            raise ValueError(
                f"optimizer output range must satisfy 0 <= start_idx < stop_idx, "
                f"got [{start}, {stop})")
        for prev_start, prev_stop in seen:
            if start < prev_stop and stop > prev_start:
                raise ValueError(
                    f"Overlapping optimizer output ranges are not supported: "
                    f"[{start}, {stop}) overlaps [{prev_start}, {prev_stop}). "
                    f"Each output dimension may be covered by at most one optimizer."
                )
        seen.append((start, stop))


def is_valid_feature_mapping(mapping, input_dim: int, n_num_features: int,
                             n_cat_features: int) -> bool:
    """Check that a feature mapping can identify the input column of every split.

    Numerical and categorical features each index from 0 internally, so the
    mapping is what turns a split's internal index back into an input column.
    Models trained by versions whose fit() never installed one carry an all-zero
    mapping, which sends every feature to column 0 -- invisibly, since additivity
    is unaffected by moving attribution between columns.

    A mapping is valid when it covers every input column and the two halves are
    exactly 0..n_num_features-1 and 0..n_cat_features-1.

    Args:
        mapping: (feature_mapping, numerical_mask) as returned by
            get_feature_mapping(), or None.
        input_dim (int): number of input columns.
        n_num_features (int): number of numerical columns the model was trained on.
        n_cat_features (int): number of categorical columns the model was trained on.

    Returns:
        bool: True if the mapping is usable.
    """
    if mapping is None:
        return False
    # A model that has never seen data reports 0 features of both kinds, so there
    # is nothing to check the mapping against.
    if n_num_features + n_cat_features != input_dim:
        return False
    try:
        indices = np.asarray(mapping[0]).ravel()
        mask = np.asarray(mapping[1]).ravel().astype(bool)
    except (TypeError, ValueError, IndexError):
        return False
    if indices.size != input_dim or mask.size != input_dim:
        return False
    if int(mask.sum()) != n_num_features or int((~mask).sum()) != n_cat_features:
        return False
    numerical = np.sort(indices[mask])
    categorical = np.sort(indices[~mask])
    return (np.array_equal(numerical, np.arange(n_num_features))
            and np.array_equal(categorical, np.arange(n_cat_features)))


def validate_monotonic_features_numerical(constraints, numerical_mask) -> None:
    """Reject monotonic constraints placed on categorical features.

    Monotonicity is an ordering property, so it is only defined for numerical
    features; a categorical feature has no order to be monotone in. Feature types
    are not known until the first batch is seen, so this runs then rather than at
    construction time.

    Args:
        constraints (Dict): feature index -> (direction, output_dims), or None.
        numerical_mask (np.ndarray): per-global-column bool, True for numerical.

    Raises:
        ValueError: If any constrained feature is categorical.
    """
    if not constraints:
        return
    bad = [int(f) for f in constraints
           if 0 <= int(f) < len(numerical_mask) and not bool(numerical_mask[int(f)])]
    if bad:
        raise ValueError(
            f"Monotonic constraints were set on categorical feature(s) {bad}. "
            f"Monotonicity requires an ordering, so it is only defined for "
            f"numerical features."
        )


def validate_monotonic_optimizer_compat(constraints, optimizers) -> None:
    """Reject monotonic constraints on any model using Adam.

    The projection orders the raw leaf gradients g, but the contribution Adam
    actually adds is

        delta = -alpha * (b1*m + (1-b1)*g) / (sqrt(b2*v + (1-b2)*g^2) + eps)

    whose derivative w.r.t. g has sign proportional to

        (1-b1)*b2*v - (1-b2)*b1*m*g

    so a larger gradient does not reliably give a larger contribution. m and v are
    also sample-specific (they depend on the path each sample took through the
    earlier trees), while leaf values are shared, so no ordering of leaf values can
    make every sample's contribution monotone. Only SGD, whose transform is a fixed
    signed scale, preserves the ordering.

    Args:
        constraints (Dict): feature index -> (direction, output_dims), or None.
        optimizers (Union[Dict, List[Dict]]): optimizer configuration(s).

    Raises:
        ValueError: If any optimizer uses Adam.
    """
    if not constraints:
        return
    if isinstance(optimizers, dict):
        optimizers = [optimizers]
    if any(str(opt.get('algo', 'SGD')).lower() == 'adam' for opt in optimizers):
        raise ValueError(
            "Monotonic constraints are not supported with the Adam optimizer. "
            "Adam's update is non-linear in the leaf gradient and depends on "
            "per-sample optimizer state, so ordering leaf values does not make "
            "predictions monotone. Use SGD when applying monotonic constraints."
        )


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


def get_input_dim(arr: Union[NumericalData, Tuple[NumericalData, ...]]) -> int:
    """
    Returns the column dimension of a 2D array (number of features).

    This function handles both single arrays and tuples of arrays,
    summing their dimensions when necessary.

    Args:
        arr (Union[NumericalData, Tuple[NumericalData, ...]]): Input array or tuple of arrays.

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
    """
    Returns an array of the same shape as input, filled with a constant value.

    Args:
        arr (NumericalData): Reference array for shape and type.
        constant (float, optional): Value to fill the array with. Defaults to 1.

    Returns:
        NumericalData: Array of ones (or constant value) matching input type and shape.
    """
    if isinstance(arr, th.Tensor):
        return th.ones_like(arr, device=arr.device) * constant
    else:
        return np.ones_like(arr) * constant


def separate_numerical_categorical(arr: np.ndarray) -> Tuple[Optional[np.ndarray],
                                                             Optional[np.ndarray]]:
    """
    Separates a numpy array into categorical and numerical components.

    This function handles various input formats including tuples, lists, dictionaries,
    and raw arrays, extracting and processing numerical and categorical data separately.

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
    Preprocesses array to match GBRL requirements for dimensions and data types.

    Separates input array into numerical and categorical features, ensuring proper
    dimensionality for the C++ GBRL backend. Handles 1D to 2D conversion and
    squeezing of excess dimensions. For PyTorch tensors, returns them as-is for
    numerical data.

    Args:
        arr (NumericalData): Input array containing features to preprocess.

    Returns:
        Tuple[Optional[NumericalData], Optional[np.ndarray]]: A tuple containing:
            - numerical_features: Processed numerical features (Tensor or ndarray) or None
            - categorical_features: Processed categorical features (ndarray) or None
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
    Ensures the output is either a PyTorch leaf tensor or a NumPy array.

    This function converts between PyTorch tensors and NumPy arrays while ensuring
    proper gradient tracking for leaf tensors when needed.

    Behavior:
    1) If tensor=True and requires_grad=True, returns a **leaf tensor** with gradients enabled.
    2) If tensor=True and requires_grad=False, returns a detached PyTorch tensor.
    3) If tensor=False, converts to and returns a NumPy array.

    Args:
        array (NumericalData): Input array (NumPy array or PyTorch tensor).
        tensor (bool): If True, ensures output is a PyTorch tensor.
        requires_grad (bool): If True and tensor=True, ensures output is a
            **leaf tensor** with gradient tracking enabled.
        device (str): Device for PyTorch tensor ('cpu' or 'cuda').

    Returns:
        NumericalData: A PyTorch tensor (if tensor=True) or a NumPy array
            (if tensor=False).
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


def process_monotonic_constraints(
    constraints: Dict[int, Tuple[str, Union[int, Sequence[int]]]],
    policy_dim: int,
    input_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process user-friendly monotonic constraint specification into C++ format.

    Converts a dictionary mapping feature indices to constraint specifications
    into three C-contiguous numpy arrays that can be passed to the C++ backend.

    The input format is designed to be user-friendly:
    ```python
    constraints = {
        0: ("increasing", [0, 1]),    # Feature 0 increases actions 0 and 1
        3: ("decreasing", 0),         # Feature 3 decreases action 0
        5: (1, [0, 1, 2]),            # Feature 5 increases all actions (numeric)
    }
    ```

    Args:
        constraints: Dictionary mapping feature indices to constraint specs.
            Keys are feature indices (int).
            Values are tuples of (direction, output_dims) where:
                - direction: "increasing"/"+"/1 or "decreasing"/"-"/-1
                - output_dims: Single int or list of output dimension indices
        policy_dim: Number of policy dimensions (constraints must be < policy_dim)
        input_dim: Number of input features (for validation)

    Returns:
        Tuple of three C-contiguous int32 numpy arrays:
            - feature_indices: Expanded feature indices for each constraint
            - output_indices: Output dimension for each constraint
            - constraint_dirs: Direction for each constraint (+1 or -1)

    Raises:
        ValueError: If constraints are invalid (bad feature index, output index,
            or direction specification)

    Example:
        >>> constraints = {0: ("increasing", [0, 1]), 3: ("decreasing", 0)}
        >>> feat, out, dirs = process_monotonic_constraints(constraints, 2, 10)
        >>> feat  # array([0, 0, 3], dtype=int32)
        >>> out   # array([0, 1, 0], dtype=int32)
        >>> dirs  # array([1, 1, -1], dtype=int32)
    """
    # Type check for constraints parameter
    if constraints is not None and not isinstance(constraints, dict):
        raise ValueError(
            f"constraints must be a dict mapping feature_index->(direction, output_dims), "
            f"got {type(constraints).__name__}"
        )
    
    if not constraints:
        return (np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32))

    feature_indices = []
    output_indices = []
    constraint_dirs = []

    direction_map = {
        "increasing": 1, "inc": 1, "+": 1, 1: 1,
        "decreasing": -1, "dec": -1, "-": -1, -1: -1
    }

    for feat_idx, (direction, output_dims) in constraints.items():
        # Validate feature index
        if not isinstance(feat_idx, (int, np.integer)) or feat_idx < 0 or feat_idx >= input_dim:
            raise ValueError(
                f"Invalid feature index {feat_idx}. "
                f"Must be an integer in [0, {input_dim})"
            )

        # Parse direction
        if direction not in direction_map:
            raise ValueError(
                f"Invalid constraint direction '{direction}' for feature {feat_idx}. "
                f"Use 'increasing'/'+'/1 or 'decreasing'/'-'/-1"
            )
        dir_val = direction_map[direction]

        # Normalize output_dims to list (handle numpy arrays, scalars and integers)
        if isinstance(output_dims, np.ndarray):
            # Use atleast_1d to handle 0-D arrays (e.g., np.array(3))
            output_dims = np.atleast_1d(output_dims).tolist()
        elif isinstance(output_dims, (int, np.integer)) or np.isscalar(output_dims):
            output_dims = np.atleast_1d(output_dims).tolist()

        # Check for empty output_dims after normalization
        if len(output_dims) == 0:
            raise ValueError(
                f"No output indices provided for feature {feat_idx}. "
                f"output_dims cannot be empty."
            )

        # Validate and add each output dimension
        for out_idx in output_dims:
            if not isinstance(out_idx, (int, np.integer)) or out_idx < 0 or out_idx >= policy_dim:
                raise ValueError(
                    f"Invalid output index {out_idx} for feature {feat_idx}. "
                    f"Must be an integer in [0, {policy_dim})"
                )
            feature_indices.append(feat_idx)
            output_indices.append(out_idx)
            constraint_dirs.append(dir_val)

    return (
        np.ascontiguousarray(feature_indices, dtype=np.int32),
        np.ascontiguousarray(output_indices, dtype=np.int32),
        np.ascontiguousarray(constraint_dirs, dtype=np.int32)
    )
