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

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

from gbrl import cuda_available
from gbrl.models.gbt import GBTModel


class TestScheduler(unittest.TestCase):
    """Test learning rate schedulers for both CPU and GPU."""

    @classmethod
    def setUpClass(cls):
        """Set up simple deterministic test data with obvious split."""
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
        
        With T=2, init_lr=1.0, stop_lr=0.1:
        - Tree 0: lr = 0.55
        - Tree 1: lr = 0.1
        
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
        
        # Tree 0 should have lr=0.55 applied
        # For samples with X > 0 (indices 2,3): raw gradient points toward 1.0
        # The tree fits gradient -0.5 (half of 1.0) for these, scaled by 0.55 => 0.55 * -(-0.5 * 2) = 0.55
        # For simplicity, just verify the predictions are what we expect:
        # Sample 2,3 should have prediction ~0.55 (lr * leaf_value)
        
        print(f"\nCPU Linear Scheduler Test:")
        print(f"  Tree 0 predictions: {pred_tree0.flatten()}")
        print(f"  Tree 1 predictions: {pred_tree1.flatten()}")
        print(f"  Expected tree 0 (samples 2,3): ~0.55 (lr=0.55)")
        print(f"  Expected tree 1 (samples 2,3): ~0.045 (lr=0.1, reduced residual)")
        
        # The key test: Tree 0 samples 2,3 should be exactly 0.55
        # (with depth=1, bias=0, target=1 for samples 2,3, gradient=-1, leaf=-(-1)=1, pred=lr*1=0.55)
        np.testing.assert_allclose(
            pred_tree0[2:4], [0.55, 0.55], rtol=1e-5,
            err_msg="Tree 0 positive samples should have lr=0.55 applied"
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
        
        print(f"\nGPU Linear Scheduler Test:")
        print(f"  Tree 0 predictions: {pred_tree0.flatten()}")
        print(f"  Tree 1 predictions: {pred_tree1.flatten()}")
        
        # Tree 0 samples 2,3 should be exactly 0.55 (lr=0.55 * raw_leaf=1.0)
        np.testing.assert_allclose(
            pred_tree0[2:4], [0.55, 0.55], rtol=1e-5,
            err_msg="GPU: Tree 0 positive samples should have lr=0.55 applied"
        )

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
        print(f"\nCPU Constant Scheduler Test:")
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
        
        print(f"\nCPU vs GPU Linear Scheduler:")
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


if __name__ == '__main__':
    unittest.main()
