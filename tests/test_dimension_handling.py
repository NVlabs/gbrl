##############################################################################
# Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
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
Unit tests for dimension handling across Python and C++ binding layers.

Tests ensure that both 1D arrays (n_features,) and 2D arrays (1, n_features)
are handled correctly when n_samples=1, across all relevant GBTModel methods.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch as th

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

from gbrl.models.gbt import GBTModel  # noqa: E402


class TestDimensionHandling(unittest.TestCase):
    """Test dimension handling for GBTModel methods."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.input_dim = 4
        cls.output_dim = 2
        cls.n_samples = 10

        # Generate random training data
        np.random.seed(42)
        cls.X_train = np.random.randn(cls.n_samples, cls.input_dim).astype(np.float32)
        cls.y_train = np.random.randn(cls.n_samples, cls.output_dim).astype(np.float32)

        # Standard tree structure and optimizer
        cls.tree_struct = {
            'max_depth': 3,
            'n_bins': 256,
            'min_data_in_leaf': 0,
            'par_th': 2,
            'grow_policy': 'greedy'
        }

        cls.sgd_optimizer = {
            'algo': 'SGD',
            'lr': 1.0,
            'start_idx': 0,
            'stop_idx': cls.output_dim
        }

        cls.params = {
            "split_score_func": "Cosine",
            "generator_type": "Quantile"
        }

    def test_fit_single_sample_1d(self):
        """Test learner.step() with single sample as 1D array."""
        print("Running test_fit_single_sample_1d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        obs_1d = np.random.randn(self.input_dim).astype(np.float32)
        grad_1d = np.random.randn(self.output_dim).astype(np.float32)

        # Should not raise an exception
        model.learner.step(obs_1d, grad_1d)

    def test_fit_single_sample_2d(self):
        """Test learner.step() with single sample as 2D array (1, n_features)."""
        print("Running test_fit_single_sample_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        obs_2d = np.random.randn(1, self.input_dim).astype(np.float32)
        grad_2d = np.random.randn(1, self.output_dim).astype(np.float32)

        # Should not raise an exception
        model.learner.step(obs_2d, grad_2d)

    def test_fit_multiple_samples_2d(self):
        """Test learner.step() with multiple samples as 2D array."""
        print("Running test_fit_multiple_samples_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        n = 5
        obs_2d = np.random.randn(n, self.input_dim).astype(np.float32)
        grad_2d = np.random.randn(n, self.output_dim).astype(np.float32)

        # Should not raise an exception
        model.learner.step(obs_2d, grad_2d)

    def test_fit_consistency_1d_vs_2d(self):
        """Test that 1D and 2D single samples in learner.step() produce same results."""
        print("Running test_fit_consistency_1d_vs_2d")

        # Create identical data in different shapes
        obs_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        obs_2d = obs_1d.reshape(1, -1)

        grad_1d = np.array([0.5, -0.5], dtype=np.float32)
        grad_2d = grad_1d.reshape(1, -1)

        # Create two identical models
        model1 = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )
        model2 = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        # Fit with 1D
        model1.learner.step(obs_1d, grad_1d)

        # Fit with 2D
        model2.learner.step(obs_2d, grad_2d)

        # Predictions should be identical
        pred1_1d = model1(obs_1d, requires_grad=False, tensor=False)
        pred1_2d = model1(obs_2d, requires_grad=False, tensor=False)
        pred2_1d = model2(obs_1d, requires_grad=False, tensor=False)
        pred2_2d = model2(obs_2d, requires_grad=False, tensor=False)

        # All predictions should be close (accounting for floating point precision)
        np.testing.assert_allclose(pred1_1d, pred2_1d, rtol=1e-5)
        np.testing.assert_allclose(pred1_2d.flatten(), pred2_2d.flatten(), rtol=1e-5)

    def test_predict_single_sample_1d(self):
        """Test model() prediction with single sample as 1D array."""
        print("Running test_predict_single_sample_1d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        obs_1d = np.random.randn(self.input_dim).astype(np.float32)

        # Should not raise an exception
        result = model(obs_1d, requires_grad=False, tensor=False)

        # With multi-output and 1D input, predict returns (1, output_dim)
        self.assertEqual(result.shape, (1, self.output_dim))

    def test_predict_single_sample_2d(self):
        """Test model() prediction with single sample as 2D array (1, n_features)."""
        print("Running test_predict_single_sample_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        obs_2d = np.random.randn(1, self.input_dim).astype(np.float32)

        # Should not raise an exception
        result = model(obs_2d, requires_grad=False, tensor=False)

        # Should return 2D array (1, output_dim) for 2D input
        self.assertEqual(result.shape, (1, self.output_dim))

    def test_predict_multiple_samples_2d(self):
        """Test model() prediction with multiple samples as 2D array."""
        print("Running test_predict_multiple_samples_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        n = 5
        obs_2d = np.random.randn(n, self.input_dim).astype(np.float32)

        # Should not raise an exception
        result = model(obs_2d, requires_grad=False, tensor=False)

        # Should return 2D array
        self.assertEqual(result.shape, (n, self.output_dim))

    def test_predict_consistency_1d_vs_2d(self):
        """Test that 1D and 2D single sample predictions are identical."""
        print("Running test_predict_consistency_1d_vs_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        obs_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        obs_2d = obs_1d.reshape(1, -1)

        pred_1d = model(obs_1d, requires_grad=False, tensor=False)
        pred_2d = model(obs_2d, requires_grad=False, tensor=False)

        # Both should return same shape for single sample
        np.testing.assert_allclose(pred_1d, pred_2d, rtol=1e-6)

    def test_tree_shap_single_sample_1d(self):
        """Test tree_shap() with single sample as 1D array."""
        print("Running test_tree_shap_single_sample_1d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        # Train at least one tree
        model.learner.step(self.X_train, self.y_train)

        obs_1d = np.random.randn(self.input_dim).astype(np.float32)

        # Should not raise an exception
        shap_values = model.tree_shap(0, obs_1d)

        # Shape should be (1, n_features, output_dim)
        self.assertEqual(shap_values.shape, (1, self.input_dim, self.output_dim))

    def test_tree_shap_single_sample_2d(self):
        """Test tree_shap() with single sample as 2D array (1, n_features)."""
        print("Running test_tree_shap_single_sample_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        # Train at least one tree
        model.learner.step(self.X_train, self.y_train)

        obs_2d = np.random.randn(1, self.input_dim).astype(np.float32)

        # Should not raise an exception
        shap_values = model.tree_shap(0, obs_2d)

        # Shape should be (1, n_features, output_dim)
        self.assertEqual(shap_values.shape, (1, self.input_dim, self.output_dim))

    def test_tree_shap_multiple_samples_2d(self):
        """Test tree_shap() with multiple samples as 2D array."""
        print("Running test_tree_shap_multiple_samples_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        # Train at least one tree
        model.learner.step(self.X_train, self.y_train)

        n = 5
        obs_2d = np.random.randn(n, self.input_dim).astype(np.float32)

        # Should not raise an exception
        shap_values = model.tree_shap(0, obs_2d)

        # Shape should be (n, n_features, output_dim)
        self.assertEqual(shap_values.shape, (n, self.input_dim, self.output_dim))

    def test_tree_shap_consistency_1d_vs_2d(self):
        """Test that 1D and 2D single samples produce same SHAP values."""
        print("Running test_tree_shap_consistency_1d_vs_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        # Train at least one tree
        model.learner.step(self.X_train, self.y_train)

        obs_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        obs_2d = obs_1d.reshape(1, -1)

        shap_1d = model.tree_shap(0, obs_1d)
        shap_2d = model.tree_shap(0, obs_2d)

        # SHAP values should be identical
        np.testing.assert_allclose(shap_1d, shap_2d, rtol=1e-6)

    def test_ensemble_shap_single_sample_1d(self):
        """Test model.shap() with single sample as 1D array."""
        print("Running test_ensemble_shap_single_sample_1d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        # Train at least one tree
        model.learner.step(self.X_train, self.y_train)

        obs_1d = np.random.randn(self.input_dim).astype(np.float32)

        # Should not raise an exception
        shap_values = model.shap(obs_1d)

        # Shape should be (1, n_features, output_dim)
        self.assertEqual(shap_values.shape, (1, self.input_dim, self.output_dim))

    def test_ensemble_shap_single_sample_2d(self):
        """Test model.shap() with single sample as 2D array (1, n_features)."""
        print("Running test_ensemble_shap_single_sample_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        # Train at least one tree
        model.learner.step(self.X_train, self.y_train)

        obs_2d = np.random.randn(1, self.input_dim).astype(np.float32)

        # Should not raise an exception
        shap_values = model.shap(obs_2d)

        # Shape should be (1, n_features, output_dim)
        self.assertEqual(shap_values.shape, (1, self.input_dim, self.output_dim))

    def test_ensemble_shap_multiple_samples_2d(self):
        """Test model.shap() with multiple samples as 2D array."""
        print("Running test_ensemble_shap_multiple_samples_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        # Train at least one tree
        model.learner.step(self.X_train, self.y_train)

        n = 5
        obs_2d = np.random.randn(n, self.input_dim).astype(np.float32)

        # Should not raise an exception
        shap_values = model.shap(obs_2d)

        # Shape should be (n, n_features, output_dim)
        self.assertEqual(shap_values.shape, (n, self.input_dim, self.output_dim))

    def test_ensemble_shap_consistency_1d_vs_2d(self):
        """Test that 1D and 2D single sample SHAP values are identical."""
        print("Running test_ensemble_shap_consistency_1d_vs_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        # Train at least one tree
        model.learner.step(self.X_train, self.y_train)

        obs_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        obs_2d = obs_1d.reshape(1, -1)

        shap_1d = model.shap(obs_1d)
        shap_2d = model.shap(obs_2d)

        # SHAP values should be identical
        np.testing.assert_allclose(shap_1d, shap_2d, rtol=1e-6)

    def test_torch_tensor_single_sample_1d(self):
        """Test learner.step() with single sample as 1D PyTorch tensor."""
        print("Running test_torch_tensor_single_sample_1d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        obs_1d = th.randn(self.input_dim)
        grad_1d = th.randn(self.output_dim)

        # Should not raise an exception
        model.learner.step(obs_1d, grad_1d)

    def test_torch_tensor_single_sample_2d(self):
        """Test learner.step() with single sample as 2D PyTorch tensor (1, n_features)."""
        print("Running test_torch_tensor_single_sample_2d")
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=self.sgd_optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )

        obs_2d = th.randn(1, self.input_dim)
        grad_2d = th.randn(1, self.output_dim)

        # Should not raise an exception
        model.learner.step(obs_2d, grad_2d)


if __name__ == '__main__':
    unittest.main()
