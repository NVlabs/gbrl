##############################################################################
# Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
"""
Tests for multi-objective / Split-RL functionality in GBRL.

Split-RL uses multi-objective gradients with shape (n_objs, n_samples, output_dim).
Each sample has an obj_label indicating which objective it belongs to.
The conflict penalty penalizes splits where the mean gradients of different 
objectives point in different directions.

Key formula:
- rho = 1 - ||sum(mu_k)||^2 / sum(||mu_k||^2)
- rho = 0 when all objectives agree (parallel gradients)
- rho = 1 when objectives perfectly conflict (opposite gradients)

IMPORTANT: Split-RL penalty is ONLY implemented on GPU (CUDA).
CPU tests will verify the data flow works but penalty has no effect.
"""
import unittest
import numpy as np
import torch as th

from gbrl import cuda_available
from gbrl.models.gbt import GBTModel


class TestMultiObjectiveBasic(unittest.TestCase):
    """Basic tests for multi-objective gradient handling."""

    def test_multi_objective_gradient_shape(self):
        """
        Test that model can accept 3D gradients (n_objs, n_samples, output_dim).
        """
        n = 10
        X = np.arange(n, dtype=np.float32).reshape(-1, 1)
        
        # Multi-objective gradients: shape (n_objs=2, n_samples=10, output_dim=1)
        grads = np.zeros((2, n, 1), dtype=np.float32)
        grads[0, :, 0] = 1.0   # Objective 0: all positive
        grads[1, :, 0] = -1.0  # Objective 1: all negative
        
        obj_labels = np.array([i % 2 for i in range(n)], dtype=np.float32)
        
        model = GBTModel(
            input_dim=1, output_dim=1,
            tree_struct={'max_depth': 2, 'n_bins': 16, 'min_data_in_leaf': 1,
                        'par_th': 1, 'grow_policy': 'greedy'},
            optimizers={'algo': 'SGD', 'lr': 1.0, 'start_idx': 0, 'stop_idx': 1},
            params={'n_objs': 2, 'split_score_func': 'Cosine', 'lambda_penalty': 0.0},
            verbose=0, device='cpu'
        )
        
        X_t = th.tensor(X)
        grads_t = th.tensor(grads)
        obj_labels_t = th.tensor(obj_labels)
        
        # Should not raise
        model.step(X=X_t, grads=grads_t, obj_labels=obj_labels_t)
        
        # Verify tree was grown
        num_trees = model.learner.get_num_trees()
        self.assertGreater(num_trees, 0, "Should grow at least one tree")

    def test_multi_objective_without_obj_labels(self):
        """
        Test that model works with multi-objective gradients but no obj_labels.
        """
        n = 10
        X = np.arange(n, dtype=np.float32).reshape(-1, 1)
        
        # Multi-objective gradients with agreeing direction (no conflict)
        grads = np.zeros((2, n, 1), dtype=np.float32)
        grads[0, :, 0] = 1.0
        grads[1, :, 0] = 1.0  # Same direction
        
        model = GBTModel(
            input_dim=1, output_dim=1,
            tree_struct={'max_depth': 2, 'n_bins': 16, 'min_data_in_leaf': 1,
                        'par_th': 1, 'grow_policy': 'greedy'},
            optimizers={'algo': 'SGD', 'lr': 1.0, 'start_idx': 0, 'stop_idx': 1},
            params={'n_objs': 2, 'split_score_func': 'Cosine', 'lambda_penalty': 1.0},
            verbose=0, device='cpu'
        )
        
        X_t = th.tensor(X)
        grads_t = th.tensor(grads)
        
        # Should not raise even without obj_labels
        model.step(X=X_t, grads=grads_t, obj_labels=None)
        
        num_trees = model.learner.get_num_trees()
        self.assertGreater(num_trees, 0, "Should grow at least one tree")


@unittest.skipIf(not cuda_available(), "CUDA not available - Split-RL only implemented on GPU")
class TestMultiObjectiveSplitRL(unittest.TestCase):
    """GPU tests for multi-objective Split-RL functionality."""

    def test_conflicting_gradients_with_penalty(self):
        """
        Test that lambda_penalty affects split scores when objectives conflict.
        
        Setup:
        - X = linspace(0, 1, 40) - 40 samples with spatial variation
        - 2 objectives with OPPOSITE gradient directions but varying magnitudes
        - Objective 0: gradients from +3 to +1 (decreasing)
        - Objective 1: gradients from -3 to -1 (increasing toward 0)
        - obj_labels interleaved with some clustering
        
        With lambda_penalty > 0, the split should prefer locations that 
        better separate the conflicting objectives.
        """
        n = 40
        X = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)
        
        # Multi-objective gradients with spatial variation
        grads = np.zeros((2, n, 1), dtype=np.float32)
        grads[0, :, 0] = np.linspace(3.0, 1.0, n)  # Objective 0: +3 to +1
        grads[1, :, 0] = np.linspace(-3.0, -1.0, n)  # Objective 1: -3 to -1 (CONFLICT!)
        
        # Interleaved objective labels with some clustering
        # Similar to test_splitrl.py pattern
        obj_labels = np.zeros(n, dtype=np.float32)
        obj_labels[2] = 1  # Some early samples for obj 1
        obj_labels[4:7] = 1  # Cluster of obj 1
        obj_labels[8] = 1
        obj_labels[12] = 1
        obj_labels[14] = 1
        obj_labels[16] = 1
        obj_labels[18] = 1
        obj_labels[20] = 1
        obj_labels[22] = 1
        obj_labels[24] = 1
        obj_labels[26] = 1
        obj_labels[30:32] = 1  # Another cluster
        
        tree_struct = {
            'max_depth': 1,
            'n_bins': 20,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'greedy'
        }
        
        # Train with NO penalty
        model_no_penalty = GBTModel(
            input_dim=1, output_dim=1,
            tree_struct=tree_struct,
            optimizers={'algo': 'SGD', 'lr': 1.0, 'start_idx': 0, 'stop_idx': 1},
            params={'n_objs': 2, 'split_score_func': 'Cosine', 'lambda_penalty': 0.0},
            verbose=0, device='cuda'
        )
        
        X_t = th.tensor(X)
        grads_t = th.tensor(grads)
        obj_labels_t = th.tensor(obj_labels)
        
        # Single step with manual gradients
        model_no_penalty.step(X=X_t, grads=grads_t, obj_labels=obj_labels_t)
        
        # Train with HIGH penalty
        model_high_penalty = GBTModel(
            input_dim=1, output_dim=1,
            tree_struct=tree_struct,
            optimizers={'algo': 'SGD', 'lr': 1.0, 'start_idx': 0, 'stop_idx': 1},
            params={'n_objs': 2, 'split_score_func': 'Cosine', 'lambda_penalty': 0.9},
            verbose=0, device='cuda'
        )
        
        model_high_penalty.step(X=X_t, grads=grads_t, obj_labels=obj_labels_t)
        
        pred_no_penalty = model_no_penalty(X_t, requires_grad=False, tensor=False)
        pred_high_penalty = model_high_penalty(X_t, requires_grad=False, tensor=False)
        
        # Calculate objective separation for each model
        obj0_mask = obj_labels == 0
        obj1_mask = obj_labels == 1
        
        sep_no_penalty = abs(pred_no_penalty[obj0_mask].mean() - pred_no_penalty[obj1_mask].mean())
        sep_high_penalty = abs(pred_high_penalty[obj0_mask].mean() - pred_high_penalty[obj1_mask].mean())
        
        print(f"\nConflicting gradients test (with spatial variation):")
        print(f"  n_samples: {n}, n_obj0: {obj0_mask.sum()}, n_obj1: {obj1_mask.sum()}")
        print(f"  Objective separation (no penalty): {sep_no_penalty:.4f}")
        print(f"  Objective separation (high penalty): {sep_high_penalty:.4f}")
        print(f"  Prediction variance (no penalty): {pred_no_penalty.var():.4f}")
        print(f"  Prediction variance (high penalty): {pred_high_penalty.var():.4f}")
        
        # With high penalty, the model should choose splits that better separate objectives
        # This manifests as either better objective separation or different variance
        pred_diff = np.abs(pred_no_penalty - pred_high_penalty).max()
        print(f"  Max prediction difference: {pred_diff:.4f}")
        
        # The predictions should differ when penalty affects split choice
        self.assertGreater(pred_diff, 0.01, 
            "Lambda penalty should affect split choice when objectives conflict")

    def test_agreeing_gradients_no_conflict(self):
        """
        Test that agreeing gradients produce no conflict (rho = 0).
        
        When both objectives have gradients pointing in the same direction,
        there is no conflict, so penalty should not affect splits.
        """
        n = 10
        X = np.arange(n, dtype=np.float32).reshape(-1, 1)
        
        # Both objectives agree (same direction)
        grads = np.zeros((2, n, 1), dtype=np.float32)
        grads[0, :, 0] = 1.0   # Objective 0: positive
        grads[1, :, 0] = 1.0   # Objective 1: also positive (NO CONFLICT)
        
        obj_labels = np.array([i % 2 for i in range(n)], dtype=np.float32)
        
        tree_struct = {
            'max_depth': 2,
            'n_bins': 16,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'greedy'
        }
        
        # Train with NO penalty
        model_no_penalty = GBTModel(
            input_dim=1, output_dim=1,
            tree_struct=tree_struct,
            optimizers={'algo': 'SGD', 'lr': 1.0, 'start_idx': 0, 'stop_idx': 1},
            params={'n_objs': 2, 'split_score_func': 'Cosine', 'lambda_penalty': 0.0},
            verbose=0, device='cuda'
        )
        
        X_t = th.tensor(X)
        grads_t = th.tensor(grads)
        obj_labels_t = th.tensor(obj_labels)
        
        model_no_penalty.step(X=X_t, grads=grads_t, obj_labels=obj_labels_t)
        
        # Train with HIGH penalty
        model_high_penalty = GBTModel(
            input_dim=1, output_dim=1,
            tree_struct=tree_struct,
            optimizers={'algo': 'SGD', 'lr': 1.0, 'start_idx': 0, 'stop_idx': 1},
            params={'n_objs': 2, 'split_score_func': 'Cosine', 'lambda_penalty': 10.0},
            verbose=0, device='cuda'
        )
        
        model_high_penalty.step(X=X_t, grads=grads_t, obj_labels=obj_labels_t)
        
        pred_no_penalty = model_no_penalty(X_t, requires_grad=False, tensor=False)
        pred_high_penalty = model_high_penalty(X_t, requires_grad=False, tensor=False)
        
        print(f"\nAgreeing gradients test (no conflict):")
        print(f"  grads[0]: {grads[0, :, 0].tolist()} (obj 0)")
        print(f"  grads[1]: {grads[1, :, 0].tolist()} (obj 1)")
        print(f"  Predictions (no penalty): {pred_no_penalty.flatten()}")
        print(f"  Predictions (high penalty): {pred_high_penalty.flatten()}")
        
        pred_diff = np.abs(pred_no_penalty - pred_high_penalty).max()
        print(f"  Max prediction difference: {pred_diff:.4f}")
        
        # When gradients agree, penalty should have minimal effect
        # (rho = 0 means penalty_factor = 0)
        np.testing.assert_allclose(pred_no_penalty, pred_high_penalty, atol=0.01,
            err_msg="Agreeing gradients should not be affected by penalty")

    def test_l2_split_score_with_conflict(self):
        """
        Test L2 split score function with conflicting objectives.
        """
        n = 10
        X = np.arange(n, dtype=np.float32).reshape(-1, 1)
        
        grads = np.zeros((2, n, 1), dtype=np.float32)
        grads[0, :, 0] = 2.0    # Strong positive
        grads[1, :, 0] = -2.0   # Strong negative (conflict)
        
        obj_labels = np.array([i % 2 for i in range(n)], dtype=np.float32)
        
        tree_struct = {
            'max_depth': 2,
            'n_bins': 16,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'oblivious'  # Test with oblivious trees
        }
        
        model = GBTModel(
            input_dim=1, output_dim=1,
            tree_struct=tree_struct,
            optimizers={'algo': 'SGD', 'lr': 1.0, 'start_idx': 0, 'stop_idx': 1},
            params={'n_objs': 2, 'split_score_func': 'L2', 'lambda_penalty': 1.0},
            verbose=0, device='cuda'
        )
        
        X_t = th.tensor(X)
        grads_t = th.tensor(grads)
        obj_labels_t = th.tensor(obj_labels)
        
        model.step(X=X_t, grads=grads_t, obj_labels=obj_labels_t)
        
        num_trees = model.learner.get_num_trees()
        print(f"\nL2 split score test:")
        print(f"  Trees grown: {num_trees}")
        
        self.assertGreater(num_trees, 0, "L2 split should grow trees")

    def test_varying_gradient_magnitudes(self):
        """
        Test conflict detection with varying gradient magnitudes.
        
        Setup gradients that increase linearly to create spatial variation.
        """
        n = 10
        X = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)
        
        grads = np.zeros((2, n, 1), dtype=np.float32)
        # Objective 0: gradients increase from 1 to 3
        grads[0, :, 0] = np.linspace(1.0, 3.0, n)
        # Objective 1: gradients decrease from -1 to -3 (conflict but varying)
        grads[1, :, 0] = np.linspace(-3.0, -1.0, n)
        
        # First half -> obj 0, second half -> obj 1
        obj_labels = np.array([0 if i < n//2 else 1 for i in range(n)], dtype=np.float32)
        
        tree_struct = {
            'max_depth': 2,
            'n_bins': 16,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': 'greedy'
        }
        
        model = GBTModel(
            input_dim=1, output_dim=1,
            tree_struct=tree_struct,
            optimizers={'algo': 'SGD', 'lr': 1.0, 'start_idx': 0, 'stop_idx': 1},
            params={'n_objs': 2, 'split_score_func': 'Cosine', 'lambda_penalty': 0.0},
            verbose=0, device='cuda'
        )
        
        X_t = th.tensor(X)
        grads_t = th.tensor(grads)
        obj_labels_t = th.tensor(obj_labels)
        
        model.step(X=X_t, grads=grads_t, obj_labels=obj_labels_t)
        
        num_trees = model.learner.get_num_trees()
        pred = model(X_t, requires_grad=False, tensor=False)
        
        print(f"\nVarying gradient magnitudes test:")
        print(f"  grads[0]: {grads[0, :, 0].tolist()}")
        print(f"  grads[1]: {grads[1, :, 0].tolist()}")
        print(f"  obj_labels: {obj_labels.tolist()}")
        print(f"  Trees grown: {num_trees}")
        print(f"  Predictions: {pred.flatten()}")
        
        self.assertGreater(num_trees, 0, "Should grow trees with varying gradients")


if __name__ == '__main__':
    unittest.main()
