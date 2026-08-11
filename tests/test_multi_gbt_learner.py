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
Unit tests for MultiGBTLearner class.

Tests specifically target the fit() method and multi-learner functionality
with different output dimensions per learner.
"""
import os
import shutil
import tempfile
import unittest
import sys

import numpy as np
import torch as th
from sklearn import datasets
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

from gbrl import cuda_available
from gbrl.learners.multi_gbt_learner import MultiGBTLearner


class TestMultiGBTLearner(unittest.TestCase):
    """Test suite for MultiGBTLearner class."""

    @classmethod
    def setUpClass(cls):
        """Set up test data and common parameters."""
        print('Loading data for MultiGBTLearner tests...')
        try:
            X, y = datasets.load_diabetes(return_X_y=True, as_frame=False,
                                          scaled=False)
        except TypeError:
            X, y = datasets.load_diabetes(return_X_y=True, as_frame=False)
        
        cls.X = X
        cls.y = y[:, np.newaxis] if len(y.shape) == 1 else y
        cls.input_dim = X.shape[1]
        cls.n_samples = len(X)
        cls.test_dir = tempfile.mkdtemp()
        
        cls.tree_struct = {
            'max_depth': 4,
            'n_bins': 256,
            'min_data_in_leaf': 0,
            'par_th': 2,
            'grow_policy': 'greedy'
        }
        
        cls.params = {
            'control_variates': False,
            'split_score_func': 'L2',
            'generator_type': 'Quantile'
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test directory."""
        shutil.rmtree(cls.test_dir)

    def test_fit_with_same_output_dims(self):
        """Test fit() with multiple learners having same output dimension."""
        print("Running test_fit_with_same_output_dims")
        
        n_learners = 3
        output_dim = 2
        
        optimizers = [
            {'algo': 'SGD', 'init_lr': 1.0, 'start_idx': 0, 'stop_idx': output_dim}
            for _ in range(n_learners)
        ]
        
        learner = MultiGBTLearner(
            input_dim=self.input_dim,
            output_dim=output_dim,
            tree_struct=self.tree_struct,
            optimizers=optimizers,
            params=self.params,
            n_learners=n_learners,
            verbose=0,
            device='cpu'
        )
        learner.reset()
        
        # Create targets for each learner
        targets = [
            np.random.randn(self.n_samples, output_dim).astype(np.float32)
            for _ in range(n_learners)
        ]
        
        # Test fit without model_idx (all learners)
        losses = learner.fit(self.X, targets, iterations=10, shuffle=False)
        
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), n_learners)
        for loss in losses:
            self.assertIsInstance(loss, float)
            self.assertGreater(loss, 0)

    def test_fit_with_different_output_dims(self):
        """Test fit() with multiple learners having different output dimensions."""
        print("Running test_fit_with_different_output_dims")
        
        n_learners = 3
        output_dims = [1, 2, 3]  # Different output dimensions per learner
        
        optimizers = [
            {'algo': 'SGD', 'init_lr': 1.0, 'start_idx': 0, 'stop_idx': dim}
            for dim in output_dims
        ]
        
        learner = MultiGBTLearner(
            input_dim=self.input_dim,
            output_dim=output_dims,
            tree_struct=self.tree_struct,
            optimizers=optimizers,
            params=self.params,
            n_learners=n_learners,
            verbose=0,
            device='cpu'
        )
        learner.reset()
        
        # Create targets with matching dimensions
        targets = [
            np.random.randn(self.n_samples, dim).astype(np.float32)
            for dim in output_dims
        ]
        
        # Test fit without model_idx (all learners)
        losses = learner.fit(self.X, targets, iterations=10, shuffle=False)
        
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), n_learners)
        for loss in losses:
            self.assertIsInstance(loss, float)
            self.assertGreater(loss, 0)

    def test_fit_with_model_idx(self):
        """Test fit() with specific model_idx."""
        print("Running test_fit_with_model_idx")
        
        n_learners = 3
        output_dims = [1, 2, 3]
        
        optimizers = [
            {'algo': 'SGD', 'init_lr': 1.0, 'start_idx': 0, 'stop_idx': dim}
            for dim in output_dims
        ]
        
        learner = MultiGBTLearner(
            input_dim=self.input_dim,
            output_dim=output_dims,
            tree_struct=self.tree_struct,
            optimizers=optimizers,
            params=self.params,
            n_learners=n_learners,
            verbose=0,
            device='cpu'
        )
        learner.reset()
        
        # Test each learner individually
        for idx, dim in enumerate(output_dims):
            target = np.random.randn(self.n_samples, dim).astype(np.float32)
            loss = learner.fit(self.X, target, iterations=10, 
                             shuffle=False, model_idx=idx)
            
            self.assertIsInstance(loss, float)
            self.assertGreater(loss, 0)

    def test_fit_1d_targets(self):
        """Test fit() with 1D targets (edge case)."""
        print("Running test_fit_1d_targets")
        
        n_learners = 2
        output_dim = 1
        
        optimizers = [
            {'algo': 'SGD', 'init_lr': 1.0, 'start_idx': 0, 'stop_idx': output_dim}
            for _ in range(n_learners)
        ]
        
        learner = MultiGBTLearner(
            input_dim=self.input_dim,
            output_dim=output_dim,
            tree_struct=self.tree_struct,
            optimizers=optimizers,
            params=self.params,
            n_learners=n_learners,
            verbose=0,
            device='cpu'
        )
        learner.reset()
        
        # Create 1D targets (will be reshaped internally)
        targets = [
            np.random.randn(self.n_samples).astype(np.float32)
            for _ in range(n_learners)
        ]
        
        # Should not crash - targets will be reshaped to (n_samples, 1)
        losses = learner.fit(self.X, targets, iterations=5, shuffle=False)
        
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), n_learners)

    def test_fit_tensor_inputs(self):
        """Test fit() with PyTorch tensor inputs."""
        print("Running test_fit_tensor_inputs")
        
        n_learners = 2
        output_dim = 2
        
        optimizers = [
            {'algo': 'SGD', 'init_lr': 1.0, 'start_idx': 0, 'stop_idx': output_dim}
            for _ in range(n_learners)
        ]
        
        learner = MultiGBTLearner(
            input_dim=self.input_dim,
            output_dim=output_dim,
            tree_struct=self.tree_struct,
            optimizers=optimizers,
            params=self.params,
            n_learners=n_learners,
            verbose=0,
            device='cpu'
        )
        learner.reset()
        
        # Use tensor inputs
        X_tensor = th.tensor(self.X, dtype=th.float32)
        targets = [
            th.randn(self.n_samples, output_dim, dtype=th.float32)
            for _ in range(n_learners)
        ]
        
        losses = learner.fit(X_tensor, targets, iterations=5, shuffle=False)
        
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), n_learners)

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_fit_gpu(self):
        """Test fit() on GPU."""
        print("Running test_fit_gpu")
        
        n_learners = 2
        output_dims = [1, 2]
        
        optimizers = [
            {'algo': 'SGD', 'init_lr': 1.0, 'start_idx': 0, 'stop_idx': dim}
            for dim in output_dims
        ]
        
        learner = MultiGBTLearner(
            input_dim=self.input_dim,
            output_dim=output_dims,
            tree_struct=self.tree_struct,
            optimizers=optimizers,
            params=self.params,
            n_learners=n_learners,
            verbose=0,
            device='cuda'
        )
        learner.reset()
        
        targets = [
            np.random.randn(self.n_samples, dim).astype(np.float32)
            for dim in output_dims
        ]
        
        losses = learner.fit(self.X, targets, iterations=5, shuffle=False)
        
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), n_learners)

    def test_step_with_different_output_dims(self):
        """Test step() method with different output dimensions."""
        print("Running test_step_with_different_output_dims")
        
        n_learners = 2
        output_dims = [2, 3]
        
        optimizers = [
            {'algo': 'SGD', 'init_lr': 1.0, 'start_idx': 0, 'stop_idx': dim}
            for dim in output_dims
        ]
        
        learner = MultiGBTLearner(
            input_dim=self.input_dim,
            output_dim=output_dims,
            tree_struct=self.tree_struct,
            optimizers=optimizers,
            params=self.params,
            n_learners=n_learners,
            verbose=0,
            device='cpu'
        )
        learner.reset()
        
        # Test step with all learners
        grads = [
            np.random.randn(self.n_samples, dim).astype(np.float32)
            for dim in output_dims
        ]
        
        learner.step(self.X, grads)
        
        # Verify trees were added
        num_trees = learner.get_num_trees()
        self.assertEqual(len(num_trees), n_learners)

    def test_predict_with_different_output_dims(self):
        """Test predict() with different output dimensions."""
        print("Running test_predict_with_different_output_dims")
        
        n_learners = 2
        output_dims = [1, 3]
        
        optimizers = [
            {'algo': 'SGD', 'init_lr': 1.0, 'start_idx': 0, 'stop_idx': dim}
            for dim in output_dims
        ]
        
        learner = MultiGBTLearner(
            input_dim=self.input_dim,
            output_dim=output_dims,
            tree_struct=self.tree_struct,
            optimizers=optimizers,
            params=self.params,
            n_learners=n_learners,
            verbose=0,
            device='cpu'
        )
        learner.reset()
        
        # Fit some data first
        targets = [
            np.random.randn(self.n_samples, dim).astype(np.float32)
            for dim in output_dims
        ]
        learner.fit(self.X, targets, iterations=5, shuffle=False)
        
        # Predict without model_idx (all learners)
        predictions = learner.predict(self.X[:10], tensor=False)
        
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), n_learners)
        # When output_dim=1, shape is squeezed to (n_samples,) instead of (n_samples, 1)
        self.assertEqual(predictions[0].shape[0], 10)
        self.assertEqual(predictions[1].shape, (10, output_dims[1]))
        
        # Predict with specific model_idx
        pred_0 = learner.predict(self.X[:10], model_idx=0, tensor=False)
        pred_1 = learner.predict(self.X[:10], model_idx=1, tensor=False)
        
        self.assertEqual(pred_0.shape[0], 10)
        self.assertEqual(pred_1.shape, (10, output_dims[1]))

    def test_save_load_with_different_output_dims(self):
        """Test save/load with different output dimensions."""
        print("Running test_save_load_with_different_output_dims")
        
        n_learners = 2
        output_dims = [1, 2]
        
        optimizers = [
            {'algo': 'SGD', 'init_lr': 1.0, 'start_idx': 0, 'stop_idx': dim}
            for dim in output_dims
        ]
        
        learner = MultiGBTLearner(
            input_dim=self.input_dim,
            output_dim=output_dims,
            tree_struct=self.tree_struct,
            optimizers=optimizers,
            params=self.params,
            n_learners=n_learners,
            verbose=0,
            device='cpu'
        )
        learner.reset()
        
        # Train
        targets = [
            np.random.randn(self.n_samples, dim).astype(np.float32)
            for dim in output_dims
        ]
        learner.fit(self.X, targets, iterations=10, shuffle=False)
        
        # Save
        save_path = os.path.join(self.test_dir, 'multi_learner_test')
        learner.save(save_path)
        
        # Load
        loaded_learner = MultiGBTLearner.load(save_path, device='cpu')
        
        # Verify
        self.assertEqual(loaded_learner.n_learners, n_learners)
        self.assertEqual(loaded_learner.output_dim, output_dims)
        
        # Compare predictions
        pred_original = learner.predict(self.X[:10], tensor=False)
        pred_loaded = loaded_learner.predict(self.X[:10], tensor=False)
        
        for i in range(n_learners):
            np.testing.assert_allclose(pred_original[i], pred_loaded[i],
                                       rtol=1e-5)


class TestMultiGBTLearnerDistilRestrictions(unittest.TestCase):
    """Verify that unsupported operations raise when student models are attached."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        n, d = 60, 4
        cls.X = rng.normal(size=(n, d)).astype(np.float32)
        cls.input_dim = d
        cls.output_dim = 2
        cls.n_learners = 2
        cls.tree_struct = {
            'max_depth': 2, 'n_bins': 32, 'min_data_in_leaf': 1,
            'par_th': 1, 'grow_policy': 'oblivious'
        }
        cls.params = {
            'control_variates': False, 'split_score_func': 'L2',
            'generator_type': 'Quantile'
        }
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def _make_trained_learner(self):
        optimizers = [
            {'algo': 'SGD', 'init_lr': 0.1, 'start_idx': 0, 'stop_idx': self.output_dim}
            for _ in range(self.n_learners)
        ]
        learner = MultiGBTLearner(
            input_dim=self.input_dim, output_dim=self.output_dim,
            tree_struct=self.tree_struct, optimizers=optimizers,
            params=self.params, n_learners=self.n_learners, verbose=0, device='cpu'
        )
        learner.reset()
        targets = [np.random.randn(len(self.X), self.output_dim).astype(np.float32)
                   for _ in range(self.n_learners)]
        learner.fit(self.X, targets, iterations=10, shuffle=False)
        return learner

    def _attach_students(self, learner):
        targets = [
            np.asarray(learner.predict(self.X, tensor=False, model_idx=i))
            for i in range(self.n_learners)
        ]
        params = {'min_steps': 5, 'limit_steps': 10, 'min_distillation_loss': 0.0,
                  'distil_max_depth': 3, 'distil_lr': 0.1}
        learner.distil(self.X, targets, params)

    def test_multi_save_raises_with_students(self):
        learner = self._make_trained_learner()
        self._attach_students(learner)
        with self.assertRaises(ValueError):
            learner.save(os.path.join(self.test_dir, 'student_save'))

    def test_multi_export_raises_with_students(self):
        learner = self._make_trained_learner()
        self._attach_students(learner)
        with self.assertRaises(ValueError):
            learner.export(os.path.join(self.test_dir, 'student_export'))

    def test_multi_ranged_predict_raises_with_students(self):
        learner = self._make_trained_learner()
        self._attach_students(learner)
        with self.assertRaises(ValueError):
            learner.predict(self.X, start_idx=0, stop_idx=3)

    def test_multi_full_predict_works_with_students(self):
        """Default (unranged) predict must still work after distillation."""
        learner = self._make_trained_learner()
        self._attach_students(learner)
        preds = learner.predict(self.X, tensor=False)
        self.assertEqual(len(preds), self.n_learners)

    def test_multi_matrix_representation_raises_with_students(self):
        learner = self._make_trained_learner()
        self._attach_students(learner)
        with self.assertRaises(ValueError):
            learner.get_matrix_representation(self.X)

    def test_multi_set_device_moves_students(self):
        """set_device must not raise for SGD models with students."""
        learner = self._make_trained_learner()
        self._attach_students(learner)
        learner.set_device('cpu')
        preds = learner.predict(self.X, tensor=False)
        self.assertEqual(len(preds), self.n_learners)

    def test_multi_print_and_plot_tree_raise_with_students(self):
        """Tree indices from get_num_trees() do not line up with the main-only
        ensembles once students are attached."""
        learner = self._make_trained_learner()
        self._attach_students(learner)
        with self.assertRaises(ValueError):
            learner.print_tree(0)
        with self.assertRaises(ValueError):
            learner.plot_tree(0, os.path.join(self.test_dir, 'student_plot'))

    def test_multi_failed_reset_leaves_python_state_untouched(self):
        """_consumed_steps is a read-modify-write against models that are NOT
        replaced when reset() raises, so mutating it in place made a retried
        reset() count the same trees twice and shrink the scheduler horizon."""
        learner = self._make_trained_learner()
        before_opts = [dict(o) for o in learner.optimizers]
        before_consumed = list(learner._consumed_steps)
        before_total = learner.total_iterations
        before_models = learner._cpp_models

        saved_algo = learner.optimizers[0]['algo']
        learner.optimizers[0]['algo'] = 'NONEXISTENT_ALGO'
        try:
            with self.assertRaises((ValueError, RuntimeError)):
                learner.reset()
            learner.optimizers[0]['algo'] = saved_algo
            self.assertEqual([dict(o) for o in learner.optimizers], before_opts,
                             'failed reset() mutated self.optimizers')
            self.assertEqual(learner._consumed_steps, before_consumed,
                             'failed reset() mutated _consumed_steps')
            self.assertEqual(learner.total_iterations, before_total,
                             'failed reset() mutated total_iterations')
            self.assertIs(learner._cpp_models, before_models,
                          'failed reset() replaced the trained models')
        finally:
            learner.optimizers[0]['algo'] = saved_algo

    def test_multi_ensemble_shap_completeness(self):
        """Ensemble-level multi shap() has its own Python dispatch, separate from
        tree_shap(); without this it ships untested."""
        learner = self._make_trained_learner()
        results = learner.shap(self.X, return_base=True)
        self.assertEqual(len(results), self.n_learners,
                         'shap() with model_idx=None returns one entry per sub-model')
        for i, (phi, base) in enumerate(results):
            pred = np.asarray(learner.predict(self.X, tensor=False, model_idx=i))
            pred = pred.reshape(len(self.X), self.output_dim)
            max_err = float(np.abs(base + phi.sum(axis=1) - pred).max())
            self.assertLess(max_err, 1e-3,
                            f'sub-model {i}: base+sum(shap) != predict, err={max_err:.5f}')

    def test_multi_set_device_rejects_model_idx(self):
        """Partial placement cannot be expressed: one shared self.device drives
        transform_data() for every sub-model, so a CUDA tensor would reach a
        sub-model still on CPU."""
        learner = self._make_trained_learner()
        with self.assertRaises(ValueError):
            learner.set_device('cpu', model_idx=0)

    def test_multi_set_device_then_reset_keeps_device(self):
        """reset() rebuilds every sub-model from self.params, so set_device()
        must update params too."""
        learner = self._make_trained_learner()
        learner.set_device('cpu')
        learner.reset()
        self.assertEqual(learner.params['device'], 'cpu')
        self.assertEqual(learner.get_device(), tuple(['cpu'] * self.n_learners))

    def test_multi_distil_transactional_on_failure(self):
        """If reset() inside distil() fails, student_models must be restored to its
        pre-distillation value (None when no prior distillation has been run)."""
        learner = self._make_trained_learner()
        targets = [
            np.asarray(learner.predict(self.X, tensor=False, model_idx=i))
            for i in range(self.n_learners)
        ]

        # Corrupt one optimizer so that reset() (called at the end of distil())
        # raises a ValueError when set_optimizer() receives an invalid algo name.
        # This causes failure AFTER students are trained but BEFORE the main model
        # is replaced, exercising the rollback path.
        saved_cfg = learner.optimizers[0].copy()
        learner.optimizers[0]['algo'] = 'NONEXISTENT_ALGO'

        params = {'min_steps': 5, 'limit_steps': 10, 'min_distillation_loss': 0.0,
                  'distil_max_depth': 3, 'distil_lr': 0.1}
        try:
            with self.assertRaises((ValueError, RuntimeError)):
                learner.distil(self.X, targets, params)
            # student_models must be restored to None after the failed distil
            self.assertIsNone(learner.student_models,
                              "student_models must be None after failed distil()")
        finally:
            # Always restore the valid optimizer so teardown succeeds
            learner.optimizers[0] = saved_cfg


if __name__ == '__main__':
    unittest.main()
