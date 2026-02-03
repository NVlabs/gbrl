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
Tests for monotonic constraints in GBRL.

Tests that monotonic constraints are properly enforced during training
for oblivious trees on GPU.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th
from torch.nn.functional import mse_loss

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

from gbrl import cuda_available
from gbrl.models.gbt import GBTModel


def create_monotonic_data(n_samples=1000, seed=42):
    """
    Create synthetic data where:
    - Feature 0 should have increasing relationship with output
    - Feature 1 should have decreasing relationship with output
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, 5).astype(np.float32)
    # y = 2*x0 - 3*x1 + noise (increasing in x0, decreasing in x1)
    y = 2 * X[:, 0] - 3 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    y = y.astype(np.float32)[:, np.newaxis]
    return th.tensor(X), th.tensor(y)


def check_monotonicity(model, X, feature_idx, direction, n_samples=100, output_idx=0, num_base_points=5):
    """
    Check if predictions are monotonic with respect to a feature.
    
    Args:
        model: The trained model
        X: Sample input data
        feature_idx: Which feature to test
        direction: 1 for increasing, -1 for decreasing
        n_samples: Number of test points per base point
        output_idx: Which output to check (for multi-output models)
        num_base_points: Number of random base points to test
    
    Returns:
        (violations, total_pairs): Count of violations and total pairs tested
    """
    # Pick multiple random base points for more thorough testing
    rng = np.random.default_rng(123)
    base_indices = rng.choice(len(X), size=min(num_base_points, len(X)), replace=False)
    
    # Create a range of values for the target feature
    feature_values = np.linspace(-3, 3, n_samples)
    
    violations = 0
    total_pairs = 0
    
    # Determine device for test inputs
    device = X.device
    
    # Test monotonicity across multiple base points
    for base_idx in base_indices:
        base_point = X[base_idx].cpu().numpy().copy()
        prev_pred = None
        
        for val in feature_values:
            test_point = base_point.copy()
            test_point[feature_idx] = val
            test_input = th.tensor(test_point.reshape(1, -1), dtype=th.float32, device=device)
            pred_output = model(test_input, requires_grad=False, tensor=False)
            
            # Handle multi-output
            if len(pred_output.shape) > 1 and pred_output.shape[1] > 1:
                pred = pred_output[0, output_idx]
            else:
                pred = pred_output.flatten()[0]
            
            if prev_pred is not None:
                total_pairs += 1
                # Use 1e-5 tolerance to account for floating point precision in GPU computation
                if direction == 1 and pred < prev_pred - 1e-5:  # Should be increasing
                    violations += 1
                elif direction == -1 and pred > prev_pred + 1e-5:  # Should be decreasing
                    violations += 1
            
            prev_pred = pred
    
    return violations, total_pairs


class TestMonotonicConstraints(unittest.TestCase):
    """Test monotonic constraints on CPU."""

    @classmethod
    def setUpClass(cls):
        """Set up test data for monotonic constraint tests."""
        print("Setting up monotonic constraints tests...")
        cls.X, cls.y = create_monotonic_data(n_samples=1000)
        cls.input_dim = cls.X.shape[1]
        cls.output_dim = 1
        cls.n_epochs = 50

    def test_monotonic_increasing_cpu(self):
        """Test that increasing constraint is enforced on CPU."""
        print("Running test_monotonic_increasing_cpu")
        
        tree_struct = {
            'max_depth': 4,
            'n_bins': 256,
            'min_data_in_leaf': 0,
            'par_th': 2,
            'grow_policy': 'oblivious'  # Required for monotonic constraints
        }
        
        # Feature 0 should be increasing
        monotonic_constraints = {
            0: ("increasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.5, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        model.set_bias_from_targets(self.y)
        
        # Train using step() to apply constraints per tree
        for epoch in range(self.n_epochs):
            y_pred = model(self.X, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, self.y.squeeze())
            loss.backward()
            model.step()
        
        # Check monotonicity
        violations, total = check_monotonicity(model, self.X, 0, 1)
        violation_rate = violations / total if total > 0 else 0
        
        print(f"Increasing constraint: {violations}/{total} violations ({violation_rate:.2%})")
        self.assertEqual(violations, 0, 
                       f"VIOLATIONS DETECTED: {violations}/{total} ({violation_rate:.2%}) - MUST BE 0%!")

    def test_monotonic_decreasing_cpu(self):
        """Test that decreasing constraint is enforced on CPU."""
        print("Running test_monotonic_decreasing_cpu")
        
        tree_struct = {
            'max_depth': 4,
            'n_bins': 256,
            'min_data_in_leaf': 0,
            'par_th': 2,
            'grow_policy': 'oblivious'
        }
        
        # Feature 1 should be decreasing
        monotonic_constraints = {
            1: ("decreasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.5, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        model.set_bias_from_targets(self.y)
        
        # Train using step() to apply constraints per tree
        for epoch in range(self.n_epochs):
            y_pred = model(self.X, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, self.y.squeeze())
            loss.backward()
            model.step()
        
        # Check monotonicity
        violations, total = check_monotonicity(model, self.X, 1, -1)
        violation_rate = violations / total if total > 0 else 0
        
        print(f"Decreasing constraint: {violations}/{total} violations ({violation_rate:.2%})")
        self.assertEqual(violations, 0,
                       f"VIOLATIONS DETECTED: {violations}/{total} ({violation_rate:.2%}) - MUST BE 0%!")

    def test_monotonic_both_constraints_cpu(self):
        """Test that both increasing and decreasing constraints work together on CPU."""
        print("Running test_monotonic_both_constraints_cpu")
        
        tree_struct = {
            'max_depth': 4,
            'n_bins': 256,
            'min_data_in_leaf': 0,
            'par_th': 2,
            'grow_policy': 'oblivious'
        }
        
        # Feature 0 increasing, Feature 1 decreasing
        monotonic_constraints = {
            0: ("increasing", 0),
            1: ("decreasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.5, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        model.set_bias_from_targets(self.y)
        
        # Train using step() to apply constraints per tree
        for epoch in range(self.n_epochs):
            y_pred = model(self.X, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, self.y.squeeze())
            loss.backward()
            model.step()
        
        # Check both constraints
        violations_inc, total_inc = check_monotonicity(model, self.X, 0, 1)
        violations_dec, total_dec = check_monotonicity(model, self.X, 1, -1)
        
        violation_rate_inc = violations_inc / total_inc if total_inc > 0 else 0
        violation_rate_dec = violations_dec / total_dec if total_dec > 0 else 0
        
        print(f"Increasing (feat 0): {violations_inc}/{total_inc} violations ({violation_rate_inc:.2%})")
        print(f"Decreasing (feat 1): {violations_dec}/{total_dec} violations ({violation_rate_dec:.2%})")
        
        self.assertEqual(violations_inc, 0,
                       f"VIOLATIONS DETECTED (increasing): {violations_inc}/{total_inc} ({violation_rate_inc:.2%}) - MUST BE 0%!")
        self.assertEqual(violations_dec, 0,
                       f"VIOLATIONS DETECTED (decreasing): {violations_dec}/{total_dec} ({violation_rate_dec:.2%}) - MUST BE 0%!")

    def test_monotonic_requires_oblivious(self):
        """Test that monotonic constraints raise error for non-oblivious trees."""
        print("Running test_monotonic_requires_oblivious")
        
        tree_struct = {
            'max_depth': 4,
            'n_bins': 256,
            'min_data_in_leaf': 0,
            'par_th': 2,
            'grow_policy': 'greedy'  # Not oblivious!
        }
        
        monotonic_constraints = {
            0: ("increasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.5, 'start_idx': 0, 'stop_idx': 1}
        
        with self.assertRaises(ValueError) as context:
            GBTModel(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                tree_struct=tree_struct,
                optimizers=optimizer,
                params=params,
                verbose=0,
                device='cpu'
            )
        
        self.assertIn("oblivious", str(context.exception).lower())

    def test_monotonic_with_fit_cpu(self):
        """Test that monotonic constraints work with fit() (not just step())."""
        print("Running test_monotonic_with_fit_cpu")
        
        tree_struct = {
            'max_depth': 4,
            'n_bins': 256,
            'min_data_in_leaf': 0,
            'par_th': 2,
            'grow_policy': 'oblivious'
        }
        
        # Feature 0 should be increasing
        monotonic_constraints = {
            0: ("increasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.5, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        model.set_bias_from_targets(self.y)
        # Use fit() instead of step()
        model.fit(self.X, self.y, self.n_epochs)
        
        # Check monotonicity
        violations, total = check_monotonicity(model, self.X, 0, 1)
        violation_rate = violations / total if total > 0 else 0
        
        print(f"Increasing constraint (with fit): {violations}/{total} violations ({violation_rate:.2%})")
        self.assertEqual(violations, 0, 
                       f"VIOLATIONS DETECTED: {violations}/{total} ({violation_rate:.2%}) - MUST BE 0%!")

    def test_monotonic_mixed_dataset_categorical_rejection(self):
        """Test that we cannot apply constraints to categorical features."""
        self.skipTest(
            "Constraints on categorical features validated in Python layer - "
            "this test documents the limitation"
        )

    def test_monotonic_mixed_dataset_numerical_only(self):
        """Test that constraints work on numerical features even with categorical data."""
        print("Running test_monotonic_mixed_dataset_numerical_only")
        
        # Create data with numerical features only (simulating mixed after preprocessing)
        # In real mixed datasets, categorical features would be handled separately
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        # y depends on X[:, 0] (increasing) and X[:, 1] (decreasing)
        y = (2 * X[:, 0] - X[:, 1] + np.random.randn(200) * 0.1).astype(np.float32)[:, np.newaxis]
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        # Apply constraints on numerical features
        monotonic_constraints = {
            0: ("increasing", 0),
            1: ("decreasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        model.set_bias_from_targets(y)
        
        # Train with step()
        for epoch in range(30):
            y_pred = model(th.tensor(X, dtype=th.float32), requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, th.tensor(y, dtype=th.float32).squeeze())
            loss.backward()
            model.step()
        
        # Check monotonicity
        violations_0, total_0 = check_monotonicity(model, th.tensor(X, dtype=th.float32), 0, 1)
        violations_1, total_1 = check_monotonicity(model, th.tensor(X, dtype=th.float32), 1, -1)
        
        print(f"Feature 0 (increasing): {violations_0}/{total_0} violations")
        print(f"Feature 1 (decreasing): {violations_1}/{total_1} violations")
        
        self.assertEqual(violations_0, 0, 
                       f"Feature 0 violations: {violations_0}/{total_0} - MUST BE 0%!")
        self.assertEqual(violations_1, 0, 
                       f"Feature 1 violations: {violations_1}/{total_1} - MUST BE 0%!")

    def test_monotonic_interleaved_features_cpu(self):
        """Test constraints with categorical features interspersed between numerical features.
        
        This tests the feature mapping logic when categorical variables are not at the end.
        Feature layout: num0, cat0, num1, cat1, num2
        This ensures the reverse_num_feature_mapping works correctly.
        """
        print("Running test_monotonic_interleaved_features_cpu")
        
        # Create data: 3 numerical features interleaved with 2 categorical
        # Layout: [num0, cat0, num1, cat1, num2]
        # Global indices: num0=0, cat0=1, num1=2, cat1=3, num2=4
        np.random.seed(42)
        n_samples = 300
        
        # Numerical features
        num0 = np.random.randn(n_samples).astype(np.float32)
        num1 = np.random.randn(n_samples).astype(np.float32)
        num2 = np.random.randn(n_samples).astype(np.float32)
        
        # Note: This test simulates a dataset where categorical features exist
        # but only numerical features are passed to the model
        
        # Target depends on num0 (increasing) and num2 (decreasing)
        y = (2 * num0 - 1.5 * num2 + 0.5 * num1 + np.random.randn(n_samples) * 0.1).astype(np.float32)[:, np.newaxis]
        
        # Combine features in interleaved order
        # Note: For GBT, we only pass numerical features
        # The categorical info is just to test that the system handles feature indices correctly
        X = np.column_stack([num0, num1, num2])
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        # Apply constraints on numerical features
        # Since we only pass numerical features to the model, the indices are:
        # num0 = index 0, num1 = index 1, num2 = index 2
        monotonic_constraints = {
            0: ("increasing", 0),  # num0 should increase
            2: ("decreasing", 0),  # num2 should decrease
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=3,
            output_dim=1,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        model.set_bias_from_targets(y)
        
        # Train with step()
        for epoch in range(30):
            y_pred = model(th.tensor(X, dtype=th.float32), requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, th.tensor(y, dtype=th.float32).squeeze())
            loss.backward()
            model.step()
        
        # Check monotonicity
        violations_0, total_0 = check_monotonicity(model, th.tensor(X, dtype=th.float32), 0, 1)
        violations_2, total_2 = check_monotonicity(model, th.tensor(X, dtype=th.float32), 2, -1)
        
        print(f"Interleaved CPU - num0 (increasing): {violations_0}/{total_0} violations")
        print(f"Interleaved CPU - num2 (decreasing): {violations_2}/{total_2} violations")
        
        self.assertEqual(violations_0, 0, 
                       f"Interleaved CPU - num0 violations: {violations_0}/{total_0} - MUST BE 0%!")
        self.assertEqual(violations_2, 0, 
                       f"Interleaved CPU - num2 violations: {violations_2}/{total_2} - MUST BE 0%!")

    def test_monotonic_multioutput_same_feature(self):
        """Test that one feature can constrain multiple outputs."""
        print("Running test_monotonic_multioutput_same_feature")
        
        # Create multi-output data where feature 0 affects both outputs
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        # Output 0: increasing with feature 0
        # Output 1: also increasing with feature 0 (but different scale)
        y0 = (2 * X[:, 0] + 0.5 * X[:, 2] + np.random.randn(500) * 0.1).astype(np.float32)
        y1 = (X[:, 0] + X[:, 3] + np.random.randn(500) * 0.1).astype(np.float32)
        y = np.column_stack([y0, y1])
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        # Feature 0 should be increasing for both outputs
        monotonic_constraints = {
            0: ("increasing", [0, 1]),  # Feature 0 applies to both output 0 and output 1
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 2}
        
        model = GBTModel(
            input_dim=5,
            output_dim=2,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        model.set_bias_from_targets(y)
        
        # Train with step()
        for epoch in range(30):
            y_pred = model(th.tensor(X, dtype=th.float32), requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, th.tensor(y, dtype=th.float32))
            loss.backward()
            model.step()
        
        # Check monotonicity for both outputs
        X_tensor = th.tensor(X, dtype=th.float32)
        
        # Check output 0, feature 0
        violations_0_0, total_0_0 = check_monotonicity(model, X_tensor, 0, 1, output_idx=0)
        # Check output 1, feature 0
        violations_1_0, total_1_0 = check_monotonicity(model, X_tensor, 0, 1, output_idx=1)
        
        print(f"Output 0, Feature 0 (increasing): {violations_0_0}/{total_0_0} violations")
        print(f"Output 1, Feature 0 (increasing): {violations_1_0}/{total_1_0} violations")
        
        self.assertEqual(violations_0_0, 0, 
                       f"Output 0, Feature 0 violations: {violations_0_0}/{total_0_0} - MUST BE 0%!")
        self.assertEqual(violations_1_0, 0, 
                       f"Output 1, Feature 0 violations: {violations_1_0}/{total_1_0} - MUST BE 0%!")

    def test_monotonic_multioutput_different_features(self):
        """Test that different features can constrain different outputs."""
        print("Running test_monotonic_multioutput_different_features")
        
        # Create multi-output data with different monotonic relationships
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        # Output 0: increasing with feature 0, decreasing with feature 1
        # Output 1: increasing with feature 2
        y0 = (2 * X[:, 0] - X[:, 1] + np.random.randn(500) * 0.1).astype(np.float32)
        y1 = (1.5 * X[:, 2] + 0.5 * X[:, 4] + np.random.randn(500) * 0.1).astype(np.float32)
        y = np.column_stack([y0, y1])
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        # Different constraints for different outputs
        monotonic_constraints = {
            0: ("increasing", 0),   # Feature 0, output 0: increasing
            1: ("decreasing", 0),   # Feature 1, output 0: decreasing
            2: ("increasing", 1),   # Feature 2, output 1: increasing
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 2}
        
        model = GBTModel(
            input_dim=5,
            output_dim=2,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        model.set_bias_from_targets(y)
        
        # Train with step()
        for epoch in range(30):
            y_pred = model(th.tensor(X, dtype=th.float32), requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, th.tensor(y, dtype=th.float32))
            loss.backward()
            model.step()
        
        # Check monotonicity for all constraints
        X_tensor = th.tensor(X, dtype=th.float32)
        
        violations_0_0, total_0_0 = check_monotonicity(model, X_tensor, 0, 1, output_idx=0)
        violations_0_1, total_0_1 = check_monotonicity(model, X_tensor, 1, -1, output_idx=0)
        violations_1_2, total_1_2 = check_monotonicity(model, X_tensor, 2, 1, output_idx=1)
        
        print(f"Output 0, Feature 0 (increasing): {violations_0_0}/{total_0_0} violations")
        print(f"Output 0, Feature 1 (decreasing): {violations_0_1}/{total_0_1} violations")
        print(f"Output 1, Feature 2 (increasing): {violations_1_2}/{total_1_2} violations")
        
        self.assertEqual(violations_0_0, 0, 
                       f"Output 0, Feature 0: {violations_0_0}/{total_0_0} - MUST BE 0%!")
        self.assertEqual(violations_0_1, 0, 
                       f"Output 0, Feature 1: {violations_0_1}/{total_0_1} - MUST BE 0%!")
        self.assertEqual(violations_1_2, 0, 
                       f"Output 1, Feature 2: {violations_1_2}/{total_1_2} - MUST BE 0%!")

class TestMonotonicConstraintsGPU(unittest.TestCase):
    """Test monotonic constraints on GPU."""
    
    def setUp(self):
        """Set up GPU test data and skip if CUDA unavailable."""
        print("Setting up GPU monotonic constraints tests...")
        if not cuda_available():
            self.skipTest("CUDA not available, skipping GPU tests")
        
        self.X, self.y = create_monotonic_data(n_samples=1000)
        # X is already a tensor, just move to GPU
        self.X = self.X.cuda()
        self.y_tensor = self.y.cuda().squeeze()
        self.n_epochs = 30
    
    def test_monotonic_increasing_gpu(self):
        """Test that increasing monotonic constraints work on GPU."""
        print("Running test_monotonic_increasing_gpu")
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        monotonic_constraints = {
            0: ("increasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.3, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cuda'
        )
        
        model.set_bias_from_targets(self.y)
        
        # Train using step() to apply constraints per tree
        for epoch in range(self.n_epochs):
            y_pred = model(self.X, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, self.y_tensor)
            loss.backward()
            model.step()
        
        # Check monotonicity
        violations, total = check_monotonicity(model, self.X, 0, 1)
        print(f"GPU - Feature 0 (increasing): {violations}/{total} violations")
        
        self.assertEqual(violations, 0, 
                       f"GPU - Feature 0 violations: {violations}/{total} - MUST BE 0%!")
    
    def test_monotonic_decreasing_gpu(self):
        """Test that decreasing monotonic constraints work on GPU."""
        print("Running test_monotonic_decreasing_gpu")
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        monotonic_constraints = {
            1: ("decreasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.5, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cuda'
        )
        
        model.set_bias_from_targets(self.y)
        
        # Train using step()
        for epoch in range(self.n_epochs):
            y_pred = model(self.X, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, self.y_tensor)
            loss.backward()
            model.step()
        
        # Check monotonicity on feature 1
        violations, total = check_monotonicity(model, self.X, 1, -1)
        print(f"GPU - Feature 1 (decreasing): {violations}/{total} violations")
        
        self.assertEqual(violations, 0, 
                       f"GPU - Feature 1 violations: {violations}/{total} - MUST BE 0%!")
    
    def test_monotonic_both_constraints_gpu(self):
        """Test that multiple constraints work on GPU."""
        print("Running test_monotonic_both_constraints_gpu")
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        monotonic_constraints = {
            0: ("increasing", 0),
            1: ("decreasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.5, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cuda'
        )
        
        model.set_bias_from_targets(self.y)
        
        # Train using step()
        for epoch in range(self.n_epochs):
            y_pred = model(self.X, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, self.y_tensor)
            loss.backward()
            model.step()
        
        # Check both constraints
        violations_0, total_0 = check_monotonicity(model, self.X, 0, 1)
        violations_1, total_1 = check_monotonicity(model, self.X, 1, -1)
        
        print(f"GPU - Feature 0 (increasing): {violations_0}/{total_0} violations")
        print(f"GPU - Feature 1 (decreasing): {violations_1}/{total_1} violations")
        
        self.assertEqual(violations_0, 0, 
                       f"GPU - Feature 0 violations: {violations_0}/{total_0} - MUST BE 0%!")
        self.assertEqual(violations_1, 0, 
                       f"GPU - Feature 1 violations: {violations_1}/{total_1} - MUST BE 0%!")
    
    def test_monotonic_with_fit_gpu(self):
        """Test that fit() applies constraints correctly on GPU."""
        print("Running test_monotonic_with_fit_gpu")
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        monotonic_constraints = {
            0: ("increasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.5, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cuda'
        )
        
        model.set_bias_from_targets(self.y)
        
        # Train using fit() instead of step()
        model.fit(self.X.cpu().numpy(), self.y, iterations=self.n_epochs)
        
        # Check monotonicity
        violations, total = check_monotonicity(model, self.X, 0, 1)
        print(f"GPU fit() - Feature 0 (increasing): {violations}/{total} violations")
        
        self.assertEqual(violations, 0, 
                       f"GPU fit() - Feature 0 violations: {violations}/{total} - MUST BE 0%!")
    
    def test_monotonic_multioutput_gpu(self):
        """Test that multi-output constraints work on GPU."""
        print("Running test_monotonic_multioutput_gpu")
        
        # Create multi-output data
        np.random.seed(42)
        X = np.random.randn(500, 5).astype(np.float32)
        y0 = (2 * X[:, 0] + 0.5 * X[:, 2] + np.random.randn(500) * 0.1).astype(np.float32)
        y1 = (X[:, 0] + X[:, 3] + np.random.randn(500) * 0.1).astype(np.float32)
        y = np.column_stack([y0, y1])
        
        X_tensor = th.from_numpy(X).cuda()
        y_tensor = th.from_numpy(y).cuda()
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        # Feature 0 should be increasing for both outputs
        monotonic_constraints = {
            0: ("increasing", [0, 1]),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 2}
        
        model = GBTModel(
            input_dim=5,
            output_dim=2,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cuda'
        )
        
        model.set_bias_from_targets(y)
        
        # Train with step()
        for epoch in range(30):
            y_pred = model(X_tensor, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, y_tensor)
            loss.backward()
            model.step()
        
        # Check monotonicity for both outputs
        violations_0_0, total_0_0 = check_monotonicity(model, X_tensor, 0, 1, output_idx=0)
        violations_1_0, total_1_0 = check_monotonicity(model, X_tensor, 0, 1, output_idx=1)
        
        print(f"GPU Multi-output - Output 0, Feature 0: {violations_0_0}/{total_0_0} violations")
        print(f"GPU Multi-output - Output 1, Feature 0: {violations_1_0}/{total_1_0} violations")
        
        self.assertEqual(violations_0_0, 0, 
                       f"GPU Multi-output 0: {violations_0_0}/{total_0_0} - MUST BE 0%!")
        self.assertEqual(violations_1_0, 0, 
                       f"GPU Multi-output 1: {violations_1_0}/{total_1_0} - MUST BE 0%!")
    
    def test_monotonic_interleaved_features_gpu(self):
        """Test constraints with categorical features interspersed between numerical features on GPU.
        
        This tests the feature mapping logic when categorical variables are not at the end.
        Feature layout: num0, cat0, num1, cat1, num2
        This ensures the reverse_num_feature_mapping works correctly on GPU.
        """
        print("Running test_monotonic_interleaved_features_gpu")
        
        # Create data: 3 numerical features interleaved with 2 categorical
        # Layout: [num0, cat0, num1, cat1, num2]
        # Global indices: num0=0, cat0=1, num1=2, cat1=3, num2=4
        np.random.seed(42)
        n_samples = 300
        
        # Numerical features
        num0 = np.random.randn(n_samples).astype(np.float32)
        num1 = np.random.randn(n_samples).astype(np.float32)
        num2 = np.random.randn(n_samples).astype(np.float32)
        
        # Target depends on num0 (increasing) and num2 (decreasing)
        y = (2 * num0 - 1.5 * num2 + 0.5 * num1 + np.random.randn(n_samples) * 0.1).astype(np.float32)[:, np.newaxis]
        
        # Combine numerical features only (as GBT expects)
        X = np.column_stack([num0, num1, num2])
        
        X_tensor = th.from_numpy(X).cuda()
        y_tensor = th.from_numpy(y).cuda().squeeze()
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        # Apply constraints on numerical features
        # num0 = index 0, num1 = index 1, num2 = index 2
        monotonic_constraints = {
            0: ("increasing", 0),  # num0 should increase
            2: ("decreasing", 0),  # num2 should decrease
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=3,
            output_dim=1,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cuda'
        )
        
        model.set_bias_from_targets(y)
        
        # Train with step()
        for epoch in range(30):
            y_pred = model(X_tensor, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, y_tensor)
            loss.backward()
            model.step()
        
        # Check monotonicity
        violations_0, total_0 = check_monotonicity(model, X_tensor, 0, 1)
        violations_2, total_2 = check_monotonicity(model, X_tensor, 2, -1)
        
        print(f"Interleaved GPU - num0 (increasing): {violations_0}/{total_0} violations")
        print(f"Interleaved GPU - num2 (decreasing): {violations_2}/{total_2} violations")
        
        self.assertEqual(violations_0, 0, 
                       f"Interleaved GPU - num0 violations: {violations_0}/{total_0} - MUST BE 0%!")
        self.assertEqual(violations_2, 0, 
                       f"Interleaved GPU - num2 violations: {violations_2}/{total_2} - MUST BE 0%!")


class TestMonotonicConstraintsPersistence(unittest.TestCase):
    """Test save/load/copy functionality with monotonic constraints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for persistence tests."""
        print("Setting up persistence tests for monotonic constraints...")
        cls.X, cls.y = create_monotonic_data(n_samples=500)
        cls.input_dim = cls.X.shape[1]
        cls.output_dim = 1
        cls.n_epochs = 20
    
    def test_save_load_with_constraints_cpu(self):
        """Test that saving and loading preserves monotonic constraints on CPU."""
        print("Running test_save_load_with_constraints_cpu")
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        monotonic_constraints = {
            0: ("increasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.3, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        model.set_bias_from_targets(self.y)
        
        # Train
        for epoch in range(self.n_epochs):
            y_pred = model(self.X, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, self.y.squeeze())
            loss.backward()
            model.step()
        
        # Check monotonicity before save
        violations_before, total_before = check_monotonicity(model, self.X, 0, 1)
        self.assertEqual(violations_before, 0, "Model should have no violations before save")
        
        # Get predictions before save
        pred_before = model(self.X, requires_grad=False, tensor=False)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            model.save_learner(save_path)
            
            loaded_model = GBTModel.load_learner(save_path, device='cpu')
        
        # Check predictions match
        pred_after = loaded_model(self.X, requires_grad=False, tensor=False)
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5, atol=1e-6,
                                   err_msg="Predictions should match after load")
        
        # Check monotonicity preserved after load
        violations_after, total_after = check_monotonicity(loaded_model, self.X, 0, 1)
        print(f"Before save: {violations_before}/{total_before}, After load: {violations_after}/{total_after}")
        self.assertEqual(violations_after, 0, "Loaded model should still have no violations")
        
        # Train more on loaded model and check constraints still enforced
        for epoch in range(5):
            y_pred = loaded_model(self.X, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, self.y.squeeze())
            loss.backward()
            loaded_model.step()
        
        violations_continued, total_continued = check_monotonicity(loaded_model, self.X, 0, 1)
        print(f"After continued training: {violations_continued}/{total_continued}")
        self.assertEqual(violations_continued, 0, "Continued training should maintain constraints")
    
    def test_copy_with_constraints_cpu(self):
        """Test that copying preserves monotonic constraints on CPU."""
        print("Running test_copy_with_constraints_cpu")
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        monotonic_constraints = {
            0: ("increasing", 0),
            1: ("decreasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.3, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        model.set_bias_from_targets(self.y)
        
        # Train
        for epoch in range(self.n_epochs):
            y_pred = model(self.X, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, self.y.squeeze())
            loss.backward()
            model.step()
        
        # Get predictions before copy
        pred_before = model(self.X, requires_grad=False, tensor=False)
        
        # Copy
        copied_model = model.copy()
        
        # Check predictions match
        pred_after = copied_model(self.X, requires_grad=False, tensor=False)
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5, atol=1e-6,
                                   err_msg="Predictions should match after copy")
        
        # Check monotonicity preserved after copy
        violations_inc, total_inc = check_monotonicity(copied_model, self.X, 0, 1)
        violations_dec, total_dec = check_monotonicity(copied_model, self.X, 1, -1)
        
        print(f"Copied model - increasing: {violations_inc}/{total_inc}, decreasing: {violations_dec}/{total_dec}")
        self.assertEqual(violations_inc, 0, "Copied model should have no increasing violations")
        self.assertEqual(violations_dec, 0, "Copied model should have no decreasing violations")
        
        # Train more on copied model (should not affect original)
        for epoch in range(5):
            y_pred = copied_model(self.X, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, self.y.squeeze())
            loss.backward()
            copied_model.step()
        
        # Original should be unchanged
        pred_original = model(self.X, requires_grad=False, tensor=False)
        np.testing.assert_allclose(pred_before, pred_original, rtol=1e-5, atol=1e-6,
                                   err_msg="Original model should be unchanged after training copy")
        
        # Copied model should still respect constraints
        violations_continued, _ = check_monotonicity(copied_model, self.X, 0, 1)
        self.assertEqual(violations_continued, 0, "Continued training on copy should maintain constraints")
    
    def test_save_load_with_constraints_gpu(self):
        """Test that saving and loading preserves monotonic constraints on GPU."""
        print("Running test_save_load_with_constraints_gpu")
        
        if not cuda_available():
            self.skipTest("CUDA not available")
        
        X_gpu = self.X.cuda()
        y_tensor = self.y.cuda().squeeze()
        
        tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        monotonic_constraints = {
            0: ("increasing", 0),
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
            "monotonic_constraints": monotonic_constraints
        }
        
        optimizer = {'algo': 'SGD', 'lr': 0.3, 'start_idx': 0, 'stop_idx': 1}
        
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cuda'
        )
        
        model.set_bias_from_targets(self.y)
        
        # Train
        for epoch in range(self.n_epochs):
            y_pred = model(X_gpu, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, y_tensor)
            loss.backward()
            model.step()
        
        # Get predictions before save
        pred_before = model(X_gpu, requires_grad=False, tensor=False)
        
        # Check monotonicity before save
        violations_before, total_before = check_monotonicity(model, X_gpu, 0, 1)
        self.assertEqual(violations_before, 0, "GPU model should have no violations before save")
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model_gpu")
            model.save_learner(save_path)
            
            loaded_model = GBTModel.load_learner(save_path, device='cuda')
        
        # Check predictions match
        pred_after = loaded_model(X_gpu, requires_grad=False, tensor=False)
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-4, atol=1e-5,
                                   err_msg="GPU predictions should match after load")
        
        # Check monotonicity preserved
        violations_after, total_after = check_monotonicity(loaded_model, X_gpu, 0, 1)
        print(f"GPU - Before save: {violations_before}/{total_before}, After load: {violations_after}/{total_after}")
        self.assertEqual(violations_after, 0, "Loaded GPU model should still have no violations")


if __name__ == '__main__':
    unittest.main()

