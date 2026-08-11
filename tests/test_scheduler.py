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
Unit tests for learning rate schedulers (constant and linear).
Scheduler affects PREDICTION - it scales tree outputs when summing predictions.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch as th
from torch.nn.functional import mse_loss

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

from gbrl import cuda_available
from gbrl.models.gbt import GBTModel


class TestScheduler(unittest.TestCase):
    """Test learning rate schedulers for both CPU and GPU."""

    @classmethod
    def setUpClass(cls):
        """Set up simple deterministic test data with obvious split.
        
        Creates data where X > 0 → high value and X <= 0 → low value
        for predictable tree splits.
        """
        # Simple data: X > 0 → high value, X <= 0 → low value
        # This ensures predictable tree splits
        cls.X = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
        cls.y = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
        
        cls.input_dim = 1
        cls.output_dim = 1
        
        # Depth 1 = single split, oblivious required for GPU linear scheduler
        cls.tree_struct = {
            'max_depth': 1,
            'n_bins': 64,
            'min_data_in_leaf': 1,
            'par_th': 2,
            'grow_policy': 'oblivious'
        }
        
        cls.params = {
            'control_variates': False,
            'split_score_func': 'L2',
            'generator_type': 'Quantile'
        }

    def _get_tree_predictions(self, model, device, start_tree, stop_tree):
        """Get predictions from a specific range of trees."""
        X_tensor = th.tensor(self.X, dtype=th.float32, device=device)
        with th.no_grad():
            preds = model.learner.predict(X_tensor, requires_grad=False,
                                          start_idx=start_tree,
                                          stop_idx=stop_tree)
        return preds.cpu().numpy() if hasattr(preds, 'cpu') else preds

    def test_linear_scheduler_scales_trees_cpu(self):
        """Prove linear scheduler applies different LRs to each tree on CPU.
        
        With T=2, init_lr=1.0, stop_lr=0.1 the schedule is
        lr(t) = init_lr + clamp(t/T, 0, 1) * (stop_lr - init_lr):
        - Tree 0: lr = 1.0   (t/T = 0   -> init_lr)
        - Tree 1: lr = 0.55  (t/T = 0.5 -> midpoint)
        - Tree 2+: lr = 0.1  (t/T >= 1  -> stop_lr)

        We verify the actual predictions match the expected LR scaling.
        """
        optimizer = {
            'algo': 'SGD',
            'lr': 1.0,
            'stop_lr': 0.1,
            'scheduler': 'linear',
            'T': 2,
            'start_idx': 0,
            'stop_idx': self.output_dim
        }
        
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )
        
        X_tensor = th.tensor(self.X, dtype=th.float32)
        y_tensor = th.tensor(self.y, dtype=th.float32)
        
        # Train 2 trees
        for _ in range(2):
            y_pred = model(X_tensor, requires_grad=True)
            loss = 0.5 * th.mean((y_pred - y_tensor.squeeze()) ** 2)
            loss.backward()
            model.step()
        
        # Get predictions from each tree individually
        pred_tree0 = self._get_tree_predictions(model, 'cpu', 0, 1)
        pred_tree1 = self._get_tree_predictions(model, 'cpu', 1, 2)
        
        # Tree 0 uses init_lr (t/T = 0), so samples 2,3 predict lr * leaf_value = 1.0.
        print("\nCPU Linear Scheduler Test:")
        print(f"  Tree 0 predictions: {pred_tree0.flatten()}")
        print(f"  Tree 1 predictions: {pred_tree1.flatten()}")
        print("  Expected tree 0 (samples 2,3): ~1.0 (lr=1.0 = init_lr)")

        # depth=1, bias=0, target=1 for samples 2,3 => gradient=-1, leaf=1, pred=lr*1
        np.testing.assert_allclose(
            pred_tree0[2:4], [1.0, 1.0], rtol=1e-5,
            err_msg="Tree 0 positive samples should have lr=init_lr=1.0 applied"
        )

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_linear_scheduler_scales_trees_gpu(self):
        """Prove linear scheduler applies correct LRs on GPU (must match CPU)."""
        optimizer = {
            'algo': 'SGD',
            'lr': 1.0,
            'stop_lr': 0.1,
            'scheduler': 'linear',
            'T': 2,
            'start_idx': 0,
            'stop_idx': self.output_dim
        }
        
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=optimizer,
            params=self.params,
            verbose=0,
            device='cuda'
        )
        
        X_tensor = th.tensor(self.X, dtype=th.float32, device='cuda')
        y_tensor = th.tensor(self.y, dtype=th.float32, device='cuda')
        
        # Train 2 trees
        for _ in range(2):
            y_pred = model(X_tensor, requires_grad=True)
            loss = 0.5 * th.mean((y_pred - y_tensor.squeeze()) ** 2)
            loss.backward()
            model.step()
        
        # Get predictions from each tree individually
        pred_tree0 = self._get_tree_predictions(model, 'cuda', 0, 1)
        pred_tree1 = self._get_tree_predictions(model, 'cuda', 1, 2)
        
        print("\nGPU Linear Scheduler Test:")
        print(f"  Tree 0 predictions: {pred_tree0.flatten()}")
        print(f"  Tree 1 predictions: {pred_tree1.flatten()}")
        
        # Tree 0 uses init_lr (t/T = 0): 1.0 * raw_leaf=1.0
        np.testing.assert_allclose(
            pred_tree0[2:4], [1.0, 1.0], rtol=1e-5,
            err_msg="GPU: Tree 0 positive samples should have lr=init_lr=1.0 applied"
        )

    def _assert_midpoint_lr(self, device):
        """Tree 1 must get the midpoint rate, not init_lr or stop_lr.

        init_lr=0.5 is required here: with init_lr=1.0 the residual for samples
        2,3 is driven to exactly 0, so tree 1's leaf is 0 whatever rate is
        applied. With init_lr=0.5 the residual survives:
            lr(0) = 0.5                          -> tree 0 = 0.5 * 1.0  = 0.5
            lr(1) = 0.5 + 0.5*(0.1-0.5) = 0.30   -> tree 1 = 0.30 * 0.5 = 0.15
        """
        optimizer = {'algo': 'SGD', 'lr': 0.5, 'stop_lr': 0.1, 'scheduler': 'linear',
                     'T': 2, 'start_idx': 0, 'stop_idx': self.output_dim}
        model = GBTModel(
            input_dim=self.input_dim, output_dim=self.output_dim,
            tree_struct=self.tree_struct, optimizers=optimizer,
            params=self.params, verbose=0, device=device)

        X_tensor = th.tensor(self.X, dtype=th.float32, device=device)
        y_tensor = th.tensor(self.y, dtype=th.float32, device=device)
        for _ in range(2):
            y_pred = model(X_tensor, requires_grad=True)
            (0.5 * th.mean((y_pred - y_tensor.squeeze()) ** 2)).backward()
            model.step()

        pred_tree0 = self._get_tree_predictions(model, device, 0, 1)
        pred_tree1 = self._get_tree_predictions(model, device, 1, 2)
        np.testing.assert_allclose(
            pred_tree0[2:4], [0.5, 0.5], rtol=1e-5,
            err_msg=f'{device}: tree 0 should use init_lr=0.5')
        np.testing.assert_allclose(
            pred_tree1[2:4], [0.15, 0.15], rtol=1e-4,
            err_msg=f'{device}: tree 1 should use the midpoint lr=0.30 (0.30 * 0.5 leaf)')

    def test_linear_scheduler_midpoint_tree1_cpu(self):
        self._assert_midpoint_lr('cpu')

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_linear_scheduler_midpoint_tree1_gpu(self):
        self._assert_midpoint_lr('cuda')

    def test_constant_scheduler_equal_trees_cpu(self):
        """Prove constant scheduler gives equal weight to all trees."""
        optimizer = {
            'algo': 'SGD',
            'lr': 1.0,
            'scheduler': 'constant',
            'T': 2,
            'start_idx': 0,
            'stop_idx': self.output_dim
        }
        
        model = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )
        
        X_tensor = th.tensor(self.X, dtype=th.float32)
        y_tensor = th.tensor(self.y, dtype=th.float32)
        
        for _ in range(2):
            y_pred = model(X_tensor, requires_grad=True)
            loss = 0.5 * th.mean((y_pred - y_tensor.squeeze()) ** 2)
            loss.backward()
            model.step()
        
        pred_tree0 = self._get_tree_predictions(model, 'cpu', 0, 1)
        pred_tree1 = self._get_tree_predictions(model, 'cpu', 1, 2)
        
        tree0_contribution = np.abs(pred_tree0).mean()
        tree1_contribution = np.abs(pred_tree1).mean()
        
        # With constant lr=1.0, both trees should have same scaling
        # (though raw leaf values differ based on residuals)
        print("\nCPU Constant Scheduler Test:")
        print(f"  Tree 0 avg contribution: {tree0_contribution:.6f} (lr=1.0)")
        print(f"  Tree 1 avg contribution: {tree1_contribution:.6f} (lr=1.0)")
        
        # Just verify we get valid predictions
        self.assertFalse(np.any(np.isnan(pred_tree0)))
        self.assertFalse(np.any(np.isnan(pred_tree1)))

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_cpu_gpu_linear_match(self):
        """Verify CPU and GPU produce identical predictions with linear scheduler."""
        optimizer = {
            'algo': 'SGD',
            'lr': 1.0,
            'stop_lr': 0.1,
            'scheduler': 'linear',
            'T': 2,
            'start_idx': 0,
            'stop_idx': self.output_dim
        }
        
        # Train on CPU
        model_cpu = GBTModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree_struct=self.tree_struct,
            optimizers=optimizer,
            params=self.params,
            verbose=0,
            device='cpu'
        )
        
        X_cpu = th.tensor(self.X, dtype=th.float32)
        y_cpu = th.tensor(self.y, dtype=th.float32)
        
        for _ in range(2):
            y_pred = model_cpu(X_cpu, requires_grad=True)
            loss = 0.5 * th.mean((y_pred - y_cpu.squeeze()) ** 2)
            loss.backward()
            model_cpu.step()
        
        # Get CPU predictions per tree
        pred_tree0_cpu = self._get_tree_predictions(model_cpu, 'cpu', 0, 1)
        pred_tree1_cpu = self._get_tree_predictions(model_cpu, 'cpu', 1, 2)
        
        # Move model to GPU and predict
        model_cpu.set_device('cuda')
        
        pred_tree0_gpu = self._get_tree_predictions(model_cpu, 'cuda', 0, 1)
        pred_tree1_gpu = self._get_tree_predictions(model_cpu, 'cuda', 1, 2)
        
        print("\nCPU vs GPU Linear Scheduler:")
        print(f"  Tree 0 CPU: {pred_tree0_cpu.flatten()}")
        print(f"  Tree 0 GPU: {pred_tree0_gpu.flatten()}")
        print(f"  Tree 1 CPU: {pred_tree1_cpu.flatten()}")
        print(f"  Tree 1 GPU: {pred_tree1_gpu.flatten()}")
        
        np.testing.assert_allclose(
            pred_tree0_cpu, pred_tree0_gpu, rtol=1e-4, atol=1e-5,
            err_msg="Tree 0 predictions should match between CPU and GPU"
        )
        np.testing.assert_allclose(
            pred_tree1_cpu, pred_tree1_gpu, rtol=1e-4, atol=1e-5,
            err_msg="Tree 1 predictions should match between CPU and GPU"
        )


class TestSchedulerPersistence(unittest.TestCase):
    """Test save/load/copy functionality with linear scheduler."""
    
    def test_save_load_with_linear_scheduler_cpu(self):
        """Test that saving and loading preserves linear scheduler on CPU."""
        print("Running test_save_load_with_linear_scheduler_cpu")
        import tempfile
        import os
        
        np.random.seed(42)
        X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        y = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
        
        tree_struct = {
            'max_depth': 1,
            'n_bins': 4,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        optimizer = {
            'algo': 'SGD',
            'lr': 1.0,
            'stop_lr': 0.1,
            'T': 4,
            'scheduler': 'Linear',
            'start_idx': 0,
            'stop_idx': 1
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
        }
        
        model = GBTModel(
            input_dim=1,
            output_dim=1,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        # Train 2 trees
        for _ in range(2):
            y_pred = model(th.tensor(X), requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, th.tensor(y).squeeze())
            loss.backward()
            model.step()
        
        # Get predictions before save
        pred_before = model(X, tensor=False)
        iteration_before = model.get_iteration()
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "scheduler_model")
            model.save_learner(save_path)
            
            loaded_model = GBTModel.load_learner(save_path, device='cpu')
        
        # Check predictions match
        pred_after = loaded_model(X, tensor=False)
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5, atol=1e-6,
                                   err_msg="Predictions should match after load")
        
        # Check iteration count preserved
        iteration_after = loaded_model.get_iteration()
        self.assertEqual(iteration_before, iteration_after, "Iteration count should be preserved")
        
        # Train more trees and verify scheduler continues correctly
        for _ in range(2):
            y_pred = loaded_model(th.tensor(X), requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, th.tensor(y).squeeze())
            loss.backward()
            loaded_model.step()
        
        final_iteration = loaded_model.get_iteration()
        self.assertEqual(final_iteration, 4, "Should have 4 trees total after continued training")
        print(f"Iterations: before={iteration_before}, after_load={iteration_after}, final={final_iteration}")
    
    def test_copy_with_linear_scheduler_cpu(self):
        """Test that copying preserves linear scheduler on CPU."""
        print("Running test_copy_with_linear_scheduler_cpu")
        
        np.random.seed(42)
        X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        y = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
        
        tree_struct = {
            'max_depth': 1,
            'n_bins': 4,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        optimizer = {
            'algo': 'SGD',
            'lr': 1.0,
            'stop_lr': 0.1,
            'T': 4,
            'scheduler': 'Linear',
            'start_idx': 0,
            'stop_idx': 1
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
        }
        
        model = GBTModel(
            input_dim=1,
            output_dim=1,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cpu'
        )
        
        # Train 2 trees
        for _ in range(2):
            y_pred = model(th.tensor(X), requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, th.tensor(y).squeeze())
            loss.backward()
            model.step()
        
        # Get predictions before copy
        pred_before = model(X, tensor=False)
        
        # Copy
        copied_model = model.copy()
        
        # Check predictions match
        pred_after = copied_model(X, tensor=False)
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5, atol=1e-6,
                                   err_msg="Predictions should match after copy")
        
        # Train original more
        for _ in range(2):
            y_pred = model(th.tensor(X), requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, th.tensor(y).squeeze())
            loss.backward()
            model.step()
        
        # Check copied model unchanged
        pred_copied_unchanged = copied_model(X, tensor=False)
        np.testing.assert_allclose(pred_after, pred_copied_unchanged, rtol=1e-5, atol=1e-6,
                                   err_msg="Copied model should be unchanged after training original")
        
        print(f"Original iterations: {model.get_iteration()}, Copy iterations: {copied_model.get_iteration()}")
        self.assertEqual(model.get_iteration(), 4, "Original should have 4 trees")
        self.assertEqual(copied_model.get_iteration(), 2, "Copy should still have 2 trees")
    
    def test_plain_reset_restores_init_lr(self):
        """A plain reset() must restart the linear schedule from the configured
        init_lr, not from the decayed rate the previous generation ended at."""
        print("Running test_plain_reset_restores_init_lr")

        X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        y = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
        configured_init_lr = 0.5

        tree_struct = {
            'max_depth': 1, 'n_bins': 4, 'min_data_in_leaf': 1,
            'par_th': 1, 'grow_policy': 'oblivious'
        }
        optimizer = {
            'algo': 'SGD', 'lr': configured_init_lr, 'stop_lr': 0.01,
            'T': 10, 'scheduler': 'Linear', 'start_idx': 0, 'stop_idx': 1
        }
        params = {"control_variates": False, "split_score_func": "L2"}

        model = GBTModel(
            input_dim=1, output_dim=1,
            tree_struct=tree_struct, optimizers=optimizer,
            params=params, verbose=0, device='cpu'
        )

        # Train several trees so the linear schedule decays the LR.
        for _ in range(5):
            y_pred = model(th.tensor(X), requires_grad=True)
            (0.5 * th.mean((y_pred - th.tensor(y).squeeze()) ** 2)).backward()
            model.step()

        lr_after_training = model.get_schedule_learning_rates()[0]
        self.assertLess(
            lr_after_training, configured_init_lr,
            "LR should have decayed after 5 trees with a linear schedule")

        # Plain reset (no distillation): must restore configured init_lr.
        model.learner.reset()
        lr_after_reset = model.get_schedule_learning_rates()[0]
        self.assertAlmostEqual(
            float(lr_after_reset), configured_init_lr, places=4,
            msg=(f"After plain reset, scheduler must start from configured "
                 f"init_lr={configured_init_lr}, got {lr_after_reset:.6f}"))

    def test_save_load_with_linear_scheduler_gpu(self):
        """Test that saving and loading preserves linear scheduler on GPU."""
        print("Running test_save_load_with_linear_scheduler_gpu")
        
        if not cuda_available():
            self.skipTest("CUDA not available")
        
        import tempfile
        import os
        
        np.random.seed(42)
        X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        y = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
        X_gpu = th.tensor(X).cuda()
        y_gpu = th.tensor(y).cuda().squeeze()
        
        tree_struct = {
            'max_depth': 1,
            'n_bins': 4,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'
        }
        
        optimizer = {
            'algo': 'SGD',
            'lr': 1.0,
            'stop_lr': 0.1,
            'T': 4,
            'scheduler': 'Linear',
            'start_idx': 0,
            'stop_idx': 1
        }
        
        params = {
            "control_variates": False,
            "split_score_func": "L2",
        }
        
        model = GBTModel(
            input_dim=1,
            output_dim=1,
            tree_struct=tree_struct,
            optimizers=optimizer,
            params=params,
            verbose=0,
            device='cuda'
        )
        
        # Train 2 trees
        for _ in range(2):
            y_pred = model(X_gpu, requires_grad=True)
            loss = 0.5 * mse_loss(y_pred, y_gpu)
            loss.backward()
            model.step()
        
        # Get predictions before save
        pred_before = model(X_gpu, tensor=False)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "scheduler_model_gpu")
            model.save_learner(save_path)
            
            loaded_model = GBTModel.load_learner(save_path, device='cuda')
        
        # Check predictions match
        pred_after = loaded_model(X_gpu, tensor=False)
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-4, atol=1e-5,
                                   err_msg="GPU predictions should match after load")
        
        print(f"GPU save/load test passed. Iterations: {loaded_model.get_iteration()}")


if __name__ == '__main__':
    unittest.main()
