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
Gradient Boosted Tree Learner Module

This module provides the GBTLearner class, which wraps the C++ GBRL backend
for single gradient boosted tree models. It supports training, prediction,
SHAP computation, and model serialization.
"""
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th

from gbrl import GBRL_CPP
from gbrl.common.compression import ParametricActorCompression, TreeCompression
from gbrl.common.utils import (NumericalData, concatenate_arrays,
                               ensure_leaf_tensor_or_array, get_poly_vectors,
                               is_valid_feature_mapping,
                               normalize_vector_input, numerical_dtype,
                               preprocess_features, process_monotonic_constraints,
                               get_index_mapping, to_numpy,
                               normalize_device,
                               validate_monotonic_features_numerical,
                               validate_monotonic_optimizer_compat,
                               validate_optimizer_ranges)
from gbrl.learners.base import BaseLearner


def warn_on_projection_limit(cpp_model) -> None:
    """Raise a Python warning if a monotonic projection hit its pass limit.

    The C++ side only prints to stderr, which nothing in Python can observe. The
    leaf values are still monotone; they are just no longer guaranteed to be the
    closest monotone values to the ones the trees produced.

    Args:
        cpp_model: The C++ model that just trained; the projection count is read
            off it directly.
    """
    n_hits = cpp_model.get_monotonic_nonconverged()
    if n_hits > 0:
        warnings.warn(
            f"{n_hits} monotonic projection(s) hit the pass limit. Leaf values "
            f"satisfy the constraints but may not be the closest values that do.",
            RuntimeWarning, stacklevel=3)


class GBTLearner(BaseLearner):
    """
    GBTLearner is a gradient boosted tree learner that utilizes a C++ backend
    for efficient computation.
    It supports training, prediction, saving, loading,
    and SHAP value computation.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 tree_struct: Dict,
                 optimizers: Union[Dict, List],
                 params: Dict,
                 policy_dim: Optional[int] = None,
                 verbose: int = 0,
                 device: str = 'cpu',
                 name: str = 'GBRL'):
        """
        Initializes the GBTLearner.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output dimensions.
            tree_struct (Dict): A dictionary containing tree structure parameters.
            optimizers (Union[Dict, List]): A dictionary or list of dictionaries containing optimizer parameters.
            params (Dict): A dictionary containing model parameters.
            policy_dim (Optional[int]): The dimension of the policy output. Defaults to None -> if None
                assumes to equal output dim.
            verbose (int, optional): Verbosity level. Defaults to 0.
            device (str, optional): The device to run the model on. Defaults to 'cpu'.
            name (str, optional): Name identifier for this learner. Defaults to 'GBRL'.
        """
        super().__init__(input_dim, output_dim, tree_struct, params, policy_dim, verbose, device)
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        validate_optimizer_ranges(optimizers)
        validate_monotonic_optimizer_compat(self.monotonic_constraints, optimizers)
        self.optimizers = optimizers
        self.student_model = None
        self._feature_mapping_installed = False
        self.learner_name = name

    def reset(self) -> None:
        """
        Resets the learner to its initial state,
        reinitializing the C++ model and optimizers.
        """
        # State this method changes is computed into a `next_*` local first and
        # assigned only once the rebuild has fully succeeded.
        next_optimizers = [opt.copy() for opt in self.optimizers]

        # Carry the decayed LR forward only when training continues after
        # distillation.  A plain reset must restart the scheduler from the
        # originally configured init_lr, not from wherever the last run left it.
        if self._cpp_model is not None and self.student_model is not None:
            lrs = self._cpp_model.get_scheduler_lrs()
            for i in range(len(next_optimizers)):
                next_optimizers[i]['init_lr'] = lrs[i]

        # load() builds instances via __new__ and bypasses __init__, so the
        # Adam/monotonic check has to run here too.
        validate_monotonic_optimizer_compat(self.monotonic_constraints, next_optimizers)

        # Process monotonic constraints first to know size for allocation
        n_mono_constraints = 0
        mono_data = None
        if self.monotonic_constraints is not None:
            feat_idx, out_idx, dirs = process_monotonic_constraints(
                self.monotonic_constraints,
                policy_dim=self.policy_dim,
                input_dim=self.input_dim
            )
            if len(feat_idx) > 0:
                n_mono_constraints = len(feat_idx)
                mono_data = (feat_idx, out_idx, dirs)
        
        # Publish only after every optimizer is set, so a failure keeps the old model.
        cpp_model = GBRL_CPP(**self.params, learner_name=self.learner_name, n_mono_constraints=n_mono_constraints)
        cpp_model.set_feature_weights(self.feature_weights)

        # Set monotonic constraint data if provided
        if mono_data is not None:
            feat_idx, out_idx, dirs = mono_data
            cpp_model.set_monotonic_constraints(feat_idx, out_idx, dirs)

        next_total_iterations = (self.total_iterations
                                 if self.student_model is not None else 0)

        # Copy the configs handed to C++: writing the reduced horizon back into the
        # optimizers would subtract total_iterations again on every reset().
        configs = []
        for i, opt in enumerate(next_optimizers):
            cfg = opt.copy()
            if (self.student_model is not None and
                    str(cfg.get('scheduler', 'Const')).lower() == 'linear'):
                horizon = cfg.get('T')
                if horizon is None:
                    raise ValueError(
                        "Linear scheduler requires 'T' (total number of iterations)")
                remaining = horizon - next_total_iterations
                if remaining <= 0:
                    # The schedule documents lr(t >= T) == stop_lr, so an exhausted
                    # horizon is not an error: hold the final rate.
                    cfg['scheduler'] = 'Const'
                    cfg['init_lr'] = cfg.get('stop_lr', cfg['init_lr'])
                    cfg.pop('T', None)
                    cfg.pop('stop_lr', None)
                else:
                    cfg['T'] = remaining
            configs.append(cfg)
        try:
            for opt in configs:
                cpp_model.set_optimizer(**opt)
        except RuntimeError as exc:
            # The previous model is still installed, so the learner stays usable.
            raise ValueError(f"Invalid GBRL optimizer configuration: {exc}") from exc

        # Publish the new model and its Python state together. The cached mapping
        # belongs to the old model: a fresh one reports no feature counts, so a
        # stale layout would be installed unchecked on the next batch.
        self._cpp_model = cpp_model
        self.optimizers = next_optimizers
        self.total_iterations = next_total_iterations
        self.feature_mapping = None
        self._feature_mapping_installed = False

    def _ensure_feature_mapping(self, features) -> None:
        """Compute and install the numerical/categorical feature mapping.

        SHAP maps each split's type-local index through the reverse mappings to
        recover the original input column.  Those arrays are zero-initialised in
        C++, so without this every feature collapses onto column 0 -- silently,
        because completeness is unaffected by moving attribution between columns.
        Monotonic constraints and feature-weighted scoring read them too.

        Called from step(), fit() and the SHAP entry points; keyed on the C++
        model rather than on total_iterations, because distillation swaps in a
        fresh model while leaving total_iterations non-zero.
        """
        # Derived from the batch in hand and kept local until every check and the
        # C++ setter succeed, so a rejected batch leaves no mapping behind.
        candidate = get_index_mapping(self._mapping_input(features))
        feature_mapping, numerical_mask = candidate
        # A batch with a different numerical/categorical mix than the model was
        # trained on would name the wrong column for every split, so reject it.
        metadata = self._cpp_model.get_metadata()
        n_num = int(metadata.get('n_num_features', 0))
        n_cat = int(metadata.get('n_cat_features', 0))
        if n_num + n_cat > 0 and not is_valid_feature_mapping(
                (feature_mapping, numerical_mask), self.input_dim, n_num, n_cat):
            raise ValueError(
                f"The given batch does not match the data this model was trained "
                f"on: it expects {n_num} numerical and {n_cat} categorical "
                f"columns. Pass a representative batch.")
        validate_monotonic_features_numerical(self.monotonic_constraints, numerical_mask)
        self._cpp_model.set_feature_mapping(np.ascontiguousarray(feature_mapping),
                                            np.ascontiguousarray(numerical_mask))
        self.feature_mapping = candidate

    def step(self,
             inputs: NumericalData,
             grads: Union[NumericalData, Tuple[NumericalData, ...]],
             ) -> None:
        """
        Performs a single gradient update step by adding a decision tree to the ensemble.

        Args:
            inputs (NumericalData): Input features (NumPy array or PyTorch tensor).
            grads (NumericalData or Tuple[NumericalData, ...]): Gradients for the update step.

        Returns:
            None
        """
        assert self._cpp_model is not None, "Model not initialized!"
        assert isinstance(grads, (list, tuple, np.ndarray, th.Tensor)), \
            "Invalid gradients type"

        super().step(inputs)
        if not self._feature_mapping_installed:
            self._ensure_feature_mapping(inputs)
            self._feature_mapping_installed = True

        if isinstance(grads, tuple):
            grads = concatenate_arrays(grads)

        if inputs.ndim == 1:
            inputs = inputs.reshape((1, self.input_dim)) if self.input_dim > 1 else inputs.reshape((len(inputs), 1))   # type: ignore

        grads = grads.reshape((len(inputs), self.output_dim))  # type: ignore
        inputs = self._mapping_input(inputs)
        num_inputs, cat_inputs = preprocess_features(inputs)

        self._memory = []
        self._cpp_model.step(obs=self.transform_data(num_inputs),
                             categorical_obs=cat_inputs,
                             grads=self.transform_data(grads),  # type: ignore
                             )

        self._memory = []

        self.iteration = self._cpp_model.get_iteration()
        self.total_iterations += 1
        warn_on_projection_limit(self._cpp_model)

    def fit(self, features: NumericalData,
            targets: NumericalData, iterations: int,
            shuffle: bool = True, loss_type: str = 'MultiRMSE') -> float:
        """
        Fits the model to the provided features and targets for a
        given number of iterations.

        Args:
            features (NumericalData): Input features.
            targets (NumericalData): Target values.
            iterations (int): Number of training iterations.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            loss_type (str, optional): Type of loss function. Defaults to 'MultiRMSE'.

        Returns:
            float: The final loss value.
        """
        if isinstance(features, th.Tensor):
            features = features.detach().cpu().numpy()
        features = self._mapping_input(features)
        num_features, cat_features = preprocess_features(features)
        # Without the mapping, SHAP attributes every feature to column 0
        # (see _ensure_feature_mapping).
        if not self._feature_mapping_installed:
            self._ensure_feature_mapping(features)
            self._feature_mapping_installed = True
        targets = to_numpy(targets)

        # Handle 1D targets
        if targets.ndim == 1:
            n_samples = 1 if self.params['output_dim'] > 1 else len(targets)
            targets = targets.reshape((n_samples, self.params['output_dim']))
        else:
            targets = targets.reshape((len(targets), self.params['output_dim']))

        # Accumulate the delta: after distillation the main C++ model restarts at
        # zero trees while total_iterations keeps the teacher's history.
        # In a finally block: a rejected monotonic projection can throw after
        # trees were already added and kept.
        iters_before = self._cpp_model.get_iteration()
        try:
            loss = self._cpp_model.fit(num_features, cat_features,
                                       targets.astype(numerical_dtype),
                                       iterations, shuffle, loss_type)
        finally:
            self.iteration = self._cpp_model.get_iteration()
            self.total_iterations += self.iteration - iters_before
        warn_on_projection_limit(self._cpp_model)
        return loss

    def save(self, filename: str) -> None:
        """
        Saves the model to a file.

        Args:
            filename (str): The filename to save the model to.
        """
        filename = filename.rstrip('.')
        filename += '.gbrl_model'
        assert self._cpp_model is not None, "Can't save non-existent model!"
        if self.student_model is not None:
            raise ValueError(
                "save() is not supported when a student model is attached. "
                "The student contributes to every prediction but would be omitted "
                "from the file, causing a silent prediction mismatch after load. "
                "Resolve distillation before saving.")
        status = self._cpp_model.save(filename)
        assert status == 0, "Failed to save model"

    def export(self, filename: str, modelname: Optional[str] = None) -> None:
        """
        Exports the model to a C header file.

        Args:
            filename (str): The filename to export the model to.
            modelname (str, optional): The name of the model in the C code. Defaults to None.
        """
        filename = filename.rstrip('.')
        filename += '.h'
        assert self._cpp_model is not None, "Can't export non-existent model!"
        if self.student_model is not None:
            raise ValueError(
                "export() is not supported when a student model is attached. "
                "The student contributes to every prediction but would be omitted "
                "from the export.")
        if modelname is None:
            modelname = ""
        try:
            status = self._cpp_model.export(filename, modelname)
            assert status == 0, "Failed to export model"
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    @classmethod
    def load(cls, filename: str, device: str) -> "GBTLearner":
        """
        Loads a GBTLearner model from a file.

        Args:
            filename (str): The filename to load the model from.
            device (str): The device to load the model onto.

        Returns:
            GBTLearner: The loaded GBTLearner instance.
        """
        filename = filename.rstrip('.')
        if '.gbrl_model' not in filename:
            filename += '.gbrl_model'
        assert os.path.isfile(filename), "filename doesn't exist!"
        try:
            instance = cls.__new__(cls)
            device = normalize_device(device)
            instance._cpp_model = GBRL_CPP.load(filename)
            # Check Adam/CUDA before calling set_device: instance.optimizers
            # isn't populated yet, so the check in set_device() can't fire.
            _temp_opts = instance._cpp_model.get_optimizers()
            if device == 'cuda' and any(
                    str(opt.get('algo', 'SGD')).lower() == 'adam'
                    for opt in _temp_opts):
                raise ValueError(
                    "Adam models are CPU-only and cannot be loaded onto CUDA. "
                    "Load with device='cpu' instead.")
            instance.set_device(device)
            metadata = instance._cpp_model.get_metadata()
            instance.tree_struct = {'max_depth': metadata['max_depth'],
                                    'min_data_in_leaf':
                                    metadata['min_data_in_leaf'],
                                    'n_bins': metadata['n_bins'],
                                    'par_th': metadata['par_th'],
                                    'batch_size': metadata['batch_size'],
                                    'grow_policy': metadata['grow_policy']}
            instance.params = {'input_dim': metadata['input_dim'],
                               'output_dim': metadata['output_dim'],
                               'policy_dim': metadata['policy_dim'],
                               'split_score_func':
                               metadata['split_score_func'],
                               'generator_type': metadata['generator_type'],
                               'use_control_variates':
                               metadata['use_control_variates'],
                               'verbose': metadata['verbose'],
                               'device': instance._cpp_model.get_device(),
                               **instance.tree_struct
                               }
            instance.output_dim = metadata['output_dim']
            instance.input_dim = metadata['input_dim']
            instance.policy_dim = metadata['policy_dim']
            instance.verbose = metadata['verbose']
            instance.optimizers = instance._cpp_model.get_optimizers()
            instance.iteration = metadata['iteration']
            instance.total_iterations = metadata['iteration']
            instance.student_model = None
            instance.feature_weights = instance._cpp_model.get_feature_weights()
            instance.device = instance.params['device']
            instance.feature_mapping = instance._cpp_model.get_feature_mapping()
            # A checkpoint without an installed mapping carries an all-zero one,
            # which makes SHAP attribute every feature to column 0. Validate it
            # against the trained feature counts and force a rebuild if it fails.
            instance._feature_mapping_installed = is_valid_feature_mapping(
                instance.feature_mapping, instance.input_dim,
                int(metadata.get('n_num_features', 0)),
                int(metadata.get('n_cat_features', 0)))
            if not instance._feature_mapping_installed:
                instance.feature_mapping = None
            # __new__ bypasses __init__, so the checks it runs are repeated here.
            validate_optimizer_ranges(instance.optimizers)
            # Rebuild the Python constraint dict from the serialized arrays so
            # reset()/distil() recreate a model with the same constraints.
            instance.monotonic_constraints = None
            if int(metadata.get('n_mono_constraints', 0)) > 0:
                feats, outs, dirs = instance._cpp_model.get_monotonic_constraints()
                restored = {}
                for f, o, d in zip(feats.tolist(), outs.tolist(), dirs.tolist()):
                    direction = 'increasing' if d == 1 else 'decreasing'
                    if f in restored:
                        # Same feature, more output dims: extend the existing entry.
                        prev_dir, prev_outs = restored[f]
                        if prev_dir != direction:
                            raise RuntimeError(
                                f"Loaded model has conflicting monotonic directions "
                                f"for feature {f}")
                        prev_outs.append(o)
                    else:
                        restored[f] = (direction, [o])
                instance.monotonic_constraints = restored
            # The monotonic checks __init__ runs; step()/fit() can be reached
            # straight after load(). The projection only exists for oblivious
            # trees, so a checkpoint claiming otherwise would train unconstrained.
            if instance.monotonic_constraints and \
                    str(instance.tree_struct.get('grow_policy', '')).lower() != 'oblivious':
                raise ValueError(
                    f"Monotonic constraints require oblivious grow_policy, got "
                    f"'{instance.tree_struct.get('grow_policy')}'")
            validate_monotonic_optimizer_compat(instance.monotonic_constraints,
                                                instance.optimizers)
            if instance._feature_mapping_installed and instance.feature_mapping is not None:
                # Feature types are only known once a mapping exists; a rebuilt
                # mapping is checked in _ensure_feature_mapping instead.
                validate_monotonic_features_numerical(instance.monotonic_constraints,
                                                      instance.feature_mapping[1])
            instance._memory = []
            instance.learner_name = instance._cpp_model.get_learner_name()
            return instance
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")
            raise e

    def get_schedule_learning_rates(self) -> Union[np.ndarray,
                                                   Tuple[np.ndarray, ...]]:
        """
        Returns the learning rates of the schedulers.

        Returns:
            Union[int, Tuple[int, int]]: The learning rates.
        """
        return self._cpp_model.get_scheduler_lrs()

    def get_iteration(self) -> int:
        """
        Returns the current iteration number.

        Returns:
            int: The current iteration number.
        """
        return self._cpp_model.get_iteration()

    def get_num_trees(self) -> int:
        """
        Returns the total number of trees in the ensemble.

        Returns:
            int: The total number of trees.
        """
        num_trees = self._cpp_model.get_num_trees()
        if self.student_model is not None:
            num_trees += self.student_model.get_num_trees()
        return num_trees

    def set_bias(self, bias: Union[NumericalData, float]) -> None:
        """
        Sets the bias of the model.

        Args:
            bias (Union[NumericalData, float]): The bias value.
        """
        try:
            self._cpp_model.set_bias(normalize_vector_input(bias))
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def set_feature_weights(self, feature_weights: Union[NumericalData, float]) -> None:
        """
        Sets the feature weights of the model.

        Args:
            feature_weights (Union[NumericalData, float]): The feature weights.
        """
        assert self._cpp_model is not None, "Model not initialized!"
        # Normalize to 1D vector (handles float, numpy, torch, 0D, and multi-D)
        if isinstance(feature_weights, th.Tensor):
            assert (feature_weights >= 0).all(), "feature weights contains non-positive values"
        elif isinstance(feature_weights, np.ndarray):
            assert np.all(feature_weights >= 0), "feature weights contains non-positive values"
        else:
            assert feature_weights >= 0, "feature weights contains non-positive values"

        normalized = normalize_vector_input(feature_weights)
        try:
            self._cpp_model.set_feature_weights(normalized)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")
            return
        # reset(), __copy__() and distil() rebuild from this, so keep it in sync
        # with C++.
        self.feature_weights = normalized

    def get_bias(self) -> np.ndarray:
        """
        Returns the bias of the model.

        Returns:
            np.ndarray: The bias.
        """
        return self._cpp_model.get_bias()

    def get_feature_weights(self) -> np.ndarray:
        """
        Returns the feature weights of the model.

        Returns:
            np.ndarray: The feature weights.
        """
        return self._cpp_model.get_feature_weights()

    def get_device(self) -> str:
        """
        Returns the device the model is running on.

        Returns:
            str: The device.
        """
        return self._cpp_model.get_device()

    def _reject_student_tree_access(self, what: str) -> None:
        """get_num_trees() counts main + student, but these only see the main model.

        distil() resets the main ensemble to zero trees, so straight after it
        get_num_trees() reports the student's count while the main model is
        empty: an index that is valid per the public count raises "Invalid tree
        index" here. Reject rather than answer for a different ensemble.
        """
        if self.student_model is not None:
            raise ValueError(
                f"{what} is not supported when a student model is attached: it "
                f"only sees the main ensemble, while get_num_trees() counts both, "
                f"so tree indices do not line up.")

    def print_tree(self, tree_idx: int) -> None:
        """
        Prints the tree at the given index.

        Args:
            tree_idx (int): The index of the tree to print.
        """
        self._reject_student_tree_access("print_tree()")
        self._cpp_model.print_tree(tree_idx)

    def plot_tree(self, tree_idx: int, filename: str) -> None:
        """
        Plots the tree at the given index and saves it to a file.

        Args:
            tree_idx (int): The index of the tree to plot.
            filename (str): The filename to save the plot to.
        """
        self._reject_student_tree_access("plot_tree()")
        filename = filename.rstrip('.')
        try:
            self._cpp_model.plot_tree(tree_idx, filename)
        except RuntimeError as e:
            print(f"Caught an exception in GBRL: {e}")

    def tree_shap(self, tree_idx: int, features:
                  NumericalData, return_base: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Computes SHAP values for a single tree.

        Implementation based on - https://github.com/yupbank/linear_tree_shap
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192

        For Adam models, tree_shap(tree_idx, x) explains tree tree_idx using the
        Adam optimizer state as it actually was before that tree was applied
        (built up by all preceding trees), not tree tree_idx in isolation.

        Args:
            tree_idx (int): tree index
            features (NumericalData): input data
            return_base (bool): if True, return (phi, base) where base has shape
                (n_samples, output_dim). base[s] + phi[s].sum(axis=0) equals the
                actual contribution of tree tree_idx for sample s. For Adam,
                base is sample-specific.

        Returns:
            np.ndarray or tuple: shap values, or (shap_values, base_values) if
                return_base=True
        """
        if self.student_model is not None:
            raise RuntimeError(
                "tree_shap() is not supported when a student model is attached. "
                "predict() sums both the main and student ensembles, so a single-model "
                "SHAP result would not reconstruct the prediction."
            )
        if isinstance(features, th.Tensor):
            features = features.detach().cpu().numpy()
        # A model saved before fit() installed a mapping carries an unusable one;
        # rebuild it from the batch being explained.
        if not self._feature_mapping_installed:
            self._ensure_feature_mapping(features)
            self._feature_mapping_installed = True
        features = self._mapping_input(features)
        num_features, cat_features = preprocess_features(features)
        poly_vectors = get_poly_vectors(self.params['max_depth'], numerical_dtype)
        base_poly, norm_values, offset = poly_vectors

        base_poly = np.ascontiguousarray(base_poly)
        norm_values = np.ascontiguousarray(norm_values)
        offset = np.ascontiguousarray(offset)
        if return_base:
            return self._cpp_model.tree_shap_and_base(tree_idx, num_features, cat_features,
                                                      norm_values, base_poly, offset)
        return self._cpp_model.tree_shap(tree_idx, num_features, cat_features,
                                         norm_values, base_poly, offset)

    def shap(self, features: NumericalData, return_base: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Computes SHAP values for the entire ensemble.

        Uses Linear tree shap for each tree in the ensemble (sequentially)
        Implementation based on - https://github.com/yupbank/linear_tree_shap
        See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192

        For Adam models, GBRL explains each tree using the Adam optimizer state
        as it actually was before that tree was applied. The numbers always add up:
        base + phi.sum(axis=1) == predict(x). Per-feature scores are approximate
        because GBRL uses the optimizer state from the real path, not from
        hypothetical alternative paths.

        Args:
            features: input data
            return_base: if True, return (phi, base) where base has shape
                (n_samples, output_dim) and satisfies base[s] + phi[s].sum(axis=0) == predict(x_s).
                For Adam, base is sample-specific. For SGD, it is shared across samples by construction.

        Returns:
            np.ndarray or tuple: shap values, or (shap_values, base_values) if
                return_base=True
        """
        if self.student_model is not None:
            raise RuntimeError(
                "shap() is not supported when a student model is attached. "
                "predict() sums both the main and student ensembles, so a single-model "
                "SHAP result would not reconstruct the prediction."
            )
        if isinstance(features, th.Tensor):
            features = features.detach().cpu().numpy()
        # A model saved before fit() installed a mapping carries an unusable one;
        # rebuild it from the batch being explained.
        if not self._feature_mapping_installed:
            self._ensure_feature_mapping(features)
            self._feature_mapping_installed = True
        features = self._mapping_input(features)
        num_features, cat_features = preprocess_features(features)
        poly_vectors = get_poly_vectors(self.params['max_depth'], numerical_dtype)
        base_poly, norm_values, offset = poly_vectors

        base_poly = np.ascontiguousarray(base_poly)
        norm_values = np.ascontiguousarray(norm_values)
        offset = np.ascontiguousarray(offset)
        if return_base:
            return self._cpp_model.ensemble_shap_and_base(num_features, cat_features,
                                                          norm_values, base_poly, offset)
        return self._cpp_model.ensemble_shap(num_features, cat_features,
                                             norm_values, base_poly, offset)

    def set_device(self, device: Union[str, th.device]) -> None:
        """
        Sets the device the model should run on.

        Args:
            device (Union[str, th.device]): The device to set.
        """
        # Normalised before to_device(): 'gpu' is an alias for 'cuda', and the
        # CPU fallback reallocates the ensemble, dropping the trained trees.
        device = normalize_device(device)
        # Adam is CPU-only; the GPU predictor represents every optimizer as SGD
        # and discards moment state, so an Adam model on CUDA gives wrong predictions.
        # Use getattr so this is safe in GBTLearner.load(), which calls set_device
        # before instance.optimizers is assigned.
        if device == 'cuda' and any(
                str(opt.get('algo', 'SGD')).lower() == 'adam'
                for opt in (getattr(self, 'optimizers', None) or [])):
            raise ValueError(
                "Adam models are CPU-only and cannot be moved to CUDA. "
                "The GPU predictor does not implement Adam; predictions would be wrong.")
        # load() calls set_device() before student_model is assigned, so use
        # getattr to avoid an AttributeError.
        student = getattr(self, 'student_model', None)
        origin = self._cpp_model.get_device()
        moved = []
        try:
            self._cpp_model.to_device(device)
            moved.append(self._cpp_model)
            if student is not None:
                student.to_device(device)
                moved.append(student)
        except RuntimeError as e:
            # predict() sums the main and student models, so leaving them on
            # different devices would feed one of them the wrong buffer. Put
            # back whatever already moved.
            for component in moved:
                try:
                    component.to_device(origin)
                except RuntimeError as rollback_error:
                    raise RuntimeError(
                        f"Failed to move model to device '{device}' ({e}), and "
                        f"could not restore it to '{origin}' ({rollback_error}). "
                        f"The learner is now inconsistent and should be reloaded."
                    ) from e
            raise RuntimeError(
                f"Failed to move model to device '{device}': {e}") from e
        # to_device() falls back to CPU (printing to stderr) when CUDA is
        # unavailable rather than raising, so the requested string is not
        # authoritative. Take the device the backend actually ended up on.
        actual_device = self._cpp_model.get_device()
        self.device = actual_device
        # reset() rebuilds the C++ model with GBRL_CPP(**self.params), so params
        # has to carry the real device too.
        # getattr: load() calls set_device() before it builds params, and fills
        # in the device from the C++ model itself a few lines later.
        if getattr(self, 'params', None) is not None:
            self.params['device'] = actual_device
        if actual_device != device:
            warnings.warn(
                f"Requested device '{device}' but GBRL is on '{actual_device}'. "
                f"The model and all future reset() calls will use "
                f"'{actual_device}'.", RuntimeWarning, stacklevel=2)

    def predict(self,
                inputs: NumericalData,
                requires_grad: bool = True,
                start_idx: Optional[int] = None,
                stop_idx: Optional[int] = None,
                tensor: bool = True) -> NumericalData:
        """
        Predicts the output for the given features.

        Args:
            inputs (NumericalData): Input features.
            requires_grad (bool, optional): Whether to compute gradients. Defaults to True.
            start_idx (int, optional): Start index for prediction. Defaults to 0.
            stop_idx (int, optional): Stop index for prediction. Defaults to None.
            tensor (bool, optional): Whether to return a tensor. Defaults to True.

        Returns:
            NumericalData: The predicted output.
        """
        assert self._cpp_model is not None, "No model loaded!"
        # 0 and None both mean "all trees", so they are not a real range.
        has_range = start_idx not in (None, 0) or stop_idx not in (None, 0)
        if self.student_model is not None and has_range:
            raise ValueError(
                "Ranged prediction (start_idx/stop_idx) is not supported when a "
                "student model is attached. The combined tree sequence has no defined "
                "ordering. Call predict() without range arguments.")
        if stop_idx is None:
            stop_idx = 0

        inputs = self._mapping_input(inputs)
        num_inputs, cat_inputs = preprocess_features(inputs)

        self._memory = []
        preds = self._cpp_model.predict(obs=self.transform_data(num_inputs),
                                        categorical_obs=cat_inputs,
                                        start_tree_idx=start_idx,
                                        stop_tree_idx=stop_idx)
        self._memory = []

        preds = th.from_dlpack(preds) if not isinstance(preds, np.ndarray) else preds  # type: ignore

        # Add student model predictions if available
        if self.student_model is not None:
            student_preds = self.student_model.predict(obs=self.transform_data(num_inputs),
                                                       categorical_obs=cat_inputs,
                                                       start_tree_idx=start_idx,
                                                       stop_tree_idx=stop_idx)
            if not isinstance(student_preds, np.ndarray):
                student_preds = th.from_dlpack(student_preds)  # type: ignore
            preds += student_preds

        preds = ensure_leaf_tensor_or_array(preds, tensor, requires_grad, self.device)
        return preds

    def distil(self,
               obs: np.ndarray,
               targets: np.ndarray,
               params: Dict,
               verbose: int = 0) -> Tuple[float, Dict]:
        """
        Distills the model into a student model.

        Args:
            obs (np.ndarray): Input observations.
            targets (np.ndarray): Target values.
            params (Dict): Distillation parameters.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            Tuple[float, Dict]: The final loss and updated parameters.
        """
        # predict() adds the student's output to the main model's, so the combined
        # prediction is only monotone if the student is too, and it is not.
        if self.monotonic_constraints:
            raise ValueError(
                "Distillation is not supported for models with monotonic "
                "constraints. predict() adds the student model's output to the "
                "main model's, and the student is not constrained, so the result "
                "would not be guaranteed monotone.")
        # Checked before anything is built: params['min_steps'] is only read once
        # the student exists.
        for required in ('min_steps', 'limit_steps'):
            if required not in params:
                raise ValueError(
                    f"Distillation requires '{required}' in params, got keys "
                    f"{sorted(params)}")
        obs = self._mapping_input(obs)
        num_obs, cat_obs = preprocess_features(obs)
        distil_params = {'input_dim': self.input_dim,
                         'output_dim': self.output_dim,
                         'policy_dim': self.policy_dim,
                         'split_score_func': 'L2',
                         'generator_type': 'Quantile',
                         'use_control_variates': False, 'device': self.device,
                         'max_depth': params.get('distil_max_depth', 6),
                         'verbose': verbose, 'batch_size':
                         self.params.get('distil_batch_size', 2048)}
        # Published only after training and reset() both succeed, so a failure
        # leaves the learner unchanged.
        student = GBRL_CPP(**distil_params)
        # A raw C++ model starts with zero feature weights and an uninitialised
        # feature mapping.  Zero weights collapse every split score, so the student
        # would train on essentially no signal.
        student.set_feature_weights(
            np.ascontiguousarray(self.feature_weights, dtype=numerical_dtype))
        student_mapping, student_mask = get_index_mapping(obs)
        student.set_feature_mapping(np.ascontiguousarray(student_mapping),
                                    np.ascontiguousarray(student_mask))
        targets = np.ascontiguousarray(targets, dtype=numerical_dtype)
        # start_idx/stop_idx are required: stop_idx defaults to 0 in the binding
        # and C++ rejects stop_idx <= 0, which would leave the student with no
        # optimizer and make it predict only its bias.
        distil_optimizer = {'algo': 'SGD',
                            'init_lr': params.get('distil_lr', 0.1),
                            'start_idx': 0,
                            'stop_idx': self.output_dim}
        try:
            student.set_optimizer(**distil_optimizer)
        except RuntimeError as exc:
            raise ValueError(
                f"Invalid GBRL distillation optimizer configuration: {exc}") from exc

        bias = np.mean(targets, axis=0)
        # np.mean returns a NumPy scalar (e.g. np.float32), not a Python float,
        # so 1-D targets need an explicit promotion to a vector.
        bias = np.atleast_1d(bias).astype(numerical_dtype, copy=False)
        student.set_bias(bias.astype(numerical_dtype))
        tr_loss = student.fit(num_obs, cat_obs, targets, params['min_steps'])
        while tr_loss > params.get('min_distillation_loss', 0.1):
            if params['min_steps'] < params['limit_steps']:
                steps_to_add = min(500, params['limit_steps'] - params['min_steps'])
                tr_loss = student.fit(num_obs, cat_obs,
                                      targets, steps_to_add,
                                      shuffle=False)
                params['min_steps'] += steps_to_add
            else:
                break
        # reset() reads student_model to decide whether to keep total_iterations
        # and to shorten a linear schedule, so publish it first. reset() leaves the
        # main ensemble untouched on failure, so restoring the student undoes all.
        previous_student = self.student_model
        self.student_model = student
        try:
            self.reset()
        except Exception:
            self.student_model = previous_student
            raise
        return tr_loss, params

    def _reject_unsupported_matrix_representation(self, what: str) -> None:
        """Adam has no fixed per-leaf contribution, so it has no leaf-value matrix.

        get_matrix_representation() builds V with Optimizer::copy_and_scale, which
        is -lr * raw_leaf_value. That is not virtual, so Adam uses it too. An Adam
        tree's contribution depends on each sample's accumulated moments, which one
        shared value per leaf cannot express, so A @ V does not reconstruct
        predict(X) and anything optimized against it is optimizing the wrong
        function.

        Args:
            what (str): Name of the calling API, used in the error message.

        Raises:
            ValueError: If any optimizer uses Adam.
        """
        for opt in (self.optimizers or []):
            if str(opt.get('algo', 'SGD')).lower() == 'adam':
                raise ValueError(
                    f"{what} is not supported for models using the Adam optimizer. "
                    f"Adam's per-tree contribution is sample-specific, so it cannot "
                    f"be represented as one value per leaf. Use SGD.")

    def _reject_constrained_compression(self) -> None:
        """Compression rewrites leaf values through the learned W matrix and never
        re-projects them, while the compressed model keeps advertising the
        constraints. The result would claim a monotonicity it no longer has.
        """
        if self.monotonic_constraints:
            raise ValueError(
                "Compression is not supported for models with monotonic "
                "constraints. Compression rewrites leaf values and does not "
                "re-apply the monotonic projection, so the compressed model "
                "would no longer satisfy them.")

    def get_matrix_representation(self, features: NumericalData) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Converts input features into matrix representations required for compression.

        Args:
            features (NumericalData): Input feature batch of shape (n_samples, n_features),
                either as a NumPy array or a PyTorch tensor.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]: A tuple containing:
                - A (np.ndarray): Feature-to-leaf assignment matrix.
                - V (np.ndarray): Leaf value matrix.
                - n_leaves_per_tree (np.ndarray): Number of leaves in each tree.
                - n_leaves (int): Total number of leaves.
                - n_trees (int): Total number of trees.
        """
        self._reject_unsupported_matrix_representation("get_matrix_representation()")
        if self.student_model is not None:
            raise ValueError(
                "get_matrix_representation() is not supported when a student model "
                "is attached. The matrix A@V would not match predict(), which also "
                "sums the student ensemble.")
        if isinstance(features, th.Tensor):
            features = features.detach().cpu().numpy().astype(np.single)
        features = self._mapping_input(features)
        num_features, cat_features = preprocess_features(features)
        # Ensure float32 dtype for C++ backend
        if num_features is not None:
            num_features = num_features.astype(np.single)
        return self._cpp_model.get_matrix_representation(num_features, cat_features)

    def compress(self, trees_to_keep: int, gradient_steps: int, features: NumericalData,
                 actions: Optional[th.Tensor] = None, log_std: Optional[th.Tensor] = None,
                 method: str = 'first_k', dist_type: str = 'supervised_learning',
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 least_squares_W: bool = True, temperature: float = 1.0, lambda_reg: float = 1.0, **kwargs) -> float:
        """
        Compresses the tree ensemble by selecting and retraining a subset of trees.

        Args:
            trees_to_keep (int): Number of trees to retain in the compressed model.
            gradient_steps (int): Number of optimization steps during compression.
            features (NumericalData): Input feature matrix (n_samples, n_features).
            actions (th.Tensor, optional): Target actions (for policy compression). Required if dist_type
                is not 'supervised_learning'.
            log_std (th.Tensor, optional): Log standard deviation (only used for certain policy types).
            method (str): Tree selection method. Defaults to 'first_k'.
            dist_type (str): Compression type ('supervised_learning', 'actor', etc.).
            optimizer_kwargs (dict, optional): Optimizer configuration.
            least_squares_W (bool): Whether to use least-squares to estimate weights (for supervised compression).
            temperature (float): Temperature parameter for soft selection.
            lambda_reg (float): L2 regularization coefficient on weights.
            **kwargs: Additional keyword arguments passed to the compressor.

        Returns:
            float: Final loss value after compression.
        """
        assert actions is not None or dist_type == 'supervised_learning', \
            "Cannot compress a policy without actions unless using supervised_learning mode"
        self._reject_constrained_compression()
        self._reject_unsupported_matrix_representation("compress()")
        
        # Validate model state and trees_to_keep before expensive matrix computation
        assert self._cpp_model is not None, "Cannot compress: no model has been trained"
        total_trees = self._cpp_model.get_num_trees()
        if trees_to_keep <= 0:
            raise ValueError(f"trees_to_keep must be > 0, got {trees_to_keep}")
        if trees_to_keep >= total_trees:
            raise ValueError(f"trees_to_keep must be < total number of trees ({total_trees}), got {trees_to_keep}")
        
        A, V, n_leaves_per_tree, n_leaves, n_trees = self.get_matrix_representation(features)
        
        # Convert to tensors with explicit dtype
        A = th.tensor(A, dtype=th.float32, device=self.device)
        V = th.tensor(V, dtype=th.float32, device=self.device)
        n_leaves_per_tree = th.tensor(n_leaves_per_tree, dtype=th.int64, device=self.device)
        trees_to_remove = total_trees - trees_to_keep
        compression_params = {'k': trees_to_remove, 'gradient_steps': gradient_steps,
                              'method': method,
                              'optimizer_kwargs': optimizer_kwargs,
                              'temperature': temperature, 'n_leaves': n_leaves, 'n_trees': n_trees,
                              'n_leaves_per_tree': n_leaves_per_tree,
                              'lambda_reg': lambda_reg,
                              'output_dim': self.output_dim,
                              'device': self.device}
        compression_params.update(kwargs)

        if actions is not None:
            # Only require log_std for Gaussian-like distributions
            if dist_type in ('gaussian', 'normal', 'diag_gaussian'):
                assert log_std is not None, "log_std must be provided for Gaussian policy compression"
            compression_params['dist_type'] = dist_type
            compressor = ParametricActorCompression(**compression_params)
            parameters, losses = compressor.compress(A, V, actions, log_std)
        else:
            compression_params['least_squares_W'] = least_squares_W
            compressor = TreeCompression(**compression_params)
            parameters, losses = compressor.compress(A, V)
        
        leaves_selection, tree_selection, W, n_compressed_trees, n_compressed_leaves = parameters
        
        # Get indices of selected leaves/trees using nonzero (more efficient than where)
        compressed_leaf_indices = leaves_selection.nonzero()[0].astype(np.int32)
        compressed_tree_indices = tree_selection.nonzero()[0].astype(np.int32)
        
        # Reorder W from original leaf order to compressed leaf order
        # W has shape (n_leaves+1, output_dim): row 0 is bias, rows 1+ are leaves
        # Select bias row (0) plus rows for compressed leaves (indices+1)
        W_indices = np.concatenate([[0], compressed_leaf_indices + 1])
        W_compressed = W[W_indices].astype(np.single)
        
        # Compute new tree indices for compressed model
        new_tree_indices = np.zeros(n_compressed_trees, dtype=np.int32)
        if n_compressed_trees > 1:
            # Convert tensor to CPU numpy before indexing with numpy array
            n_leaves_per_tree_np = n_leaves_per_tree.cpu().numpy()
            new_tree_indices[1:] = np.cumsum(
                n_leaves_per_tree_np[compressed_tree_indices]
            )[:-1].astype(np.int32)

        self._cpp_model.compress(n_compressed_leaves, n_compressed_trees, compressed_leaf_indices,
                                 compressed_tree_indices, new_tree_indices, W_compressed)
        if self.verbose > 0:
            print(f"Finished compressing - compressed model has {self.get_num_trees()} trees")
        
        # Defensive check for empty losses
        if not losses:
            del compressor, A, V, n_leaves_per_tree
            del leaves_selection, tree_selection, W, parameters
            raise RuntimeError("No losses computed by compressor during compression operation")
        
        final_loss = losses[-1]
        # Clean up large tensors
        del compressor, A, V, n_leaves_per_tree
        del leaves_selection, tree_selection, W, W_compressed, parameters, losses
        
        return final_loss

    def print_ensemble_metadata(self):
        """Prints the metadata of the ensemble."""
        if self._cpp_model is None:
            print("No model loaded!")
            return
        self._cpp_model.print_ensemble_metadata()

    def __copy__(self):
        """Creates a copy of the GBTLearner instance."""
        opts = [opt.copy() if opt is not None else opt
                for opt in self.optimizers
                ]

        copy_ = GBTLearner(input_dim=self.input_dim,  # type: ignore
                           output_dim=self.output_dim,  # type: ignore
                           tree_struct=self.tree_struct.copy(),
                           optimizers=opts,
                           params=self.params,
                           policy_dim=self.policy_dim,  # type: ignore
                           verbose=self.verbose,
                           device=self.device,
                           name=self.learner_name)
        # params does not describe these, so a rebuild from it alone would drop
        # them and copy_.reset() would produce an unconstrained model.
        copy_.monotonic_constraints = (dict(self.monotonic_constraints)
                                       if self.monotonic_constraints else None)
        copy_.feature_weights = np.array(self.feature_weights, copy=True)
        copy_.feature_mapping = self.feature_mapping
        copy_._feature_mapping_installed = self._feature_mapping_installed
        copy_.iteration = self.iteration
        copy_.total_iterations = self.total_iterations
        if self._cpp_model is not None:
            copy_._cpp_model = GBRL_CPP(self._cpp_model)
        if self.student_model is not None:
            copy_.student_model = GBRL_CPP(self.student_model)
        return copy_
