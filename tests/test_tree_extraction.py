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
Unit tests for tree extraction and insertion (get_tree/add_tree functionality).
Tests distributed training use case where trees are extracted from one model
and added to another.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch as th
from sklearn import datasets

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

from gbrl import cuda_available
from gbrl.models.gbt import GBTModel


class TestTreeExtraction(unittest.TestCase):
    """Test tree extraction and insertion for distributed training scenarios"""

    def setUp(self):
        """Set up test data"""
        # Simple regression dataset
        X, y = datasets.make_regression(
            n_samples=200,
            n_features=5,
            n_informative=3,
            noise=10.0,
            random_state=42
        )
        self.X = th.tensor(X, dtype=th.float32)
        self.y = th.tensor(y, dtype=th.float32)
        # Common model config
        self.tree_struct = {
            'max_depth': 3,
            'n_bins': 64,
            'min_data_in_leaf': 5,
            'par_th': 2,
            'grow_policy': 'greedy'
        }
        self.optimizer = {
            'algo': 'SGD',
            'lr': 0.1,
            'start_idx': 0,
            'stop_idx': 1
        }
        self.params = {
            'split_score_func': 'Cosine',
            'generator_type': 'Quantile'
        }
        self.n_epochs = 10

    def _train_and_extract_trees(self, device='cpu'):
        """Train a model normally and extract all trees"""
        model1 = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=self.tree_struct,
            optimizers=self.optimizer,
            params=self.params,
            device=device
        )
        X = self.X.to(device)
        y = self.y.to(device)

        # Train for n_epochs
        for epoch in range(self.n_epochs):
            pred = model1(X, requires_grad=True)
            loss = th.nn.functional.mse_loss(pred, y)
            loss.backward()
            model1.step()

        # Extract all trees
        trees = []
        for i in range(self.n_epochs):
            tree_data = model1.get_tree(i)
            trees.append(tree_data)
        
        return model1, trees

    def _build_from_extracted_trees(self, trees, device='cpu'):
        """Build a model by adding extracted trees one by one"""
        model2 = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=self.tree_struct,
            optimizers=self.optimizer,
            params=self.params,
            device=device
        )
        
        # Initialize model2 by doing a dummy forward pass so metadata gets set up
        X = self.X.to(device)
        _ = model2(X, requires_grad=False)
        
        for tree_data in trees:
            model2.add_tree(tree_data)
        
        return model2

    def _compare_models(self, model1, model2, X, device='cpu'):
        """Compare predictions and structure of two models"""
        X = X.to(device)
        
        # Compare predictions
        with th.no_grad():
            pred1 = model1(X, requires_grad=False)
            pred2 = model2(X, requires_grad=False)
        
        pred1_np = pred1.cpu().numpy()
        pred2_np = pred2.cpu().numpy()
        
        # Predictions should be identical
        np.testing.assert_allclose(
            pred1_np, pred2_np, rtol=1e-5, atol=1e-6,
            err_msg=f"Predictions differ on {device}"
        )
        
        # Compare tree structures - both should have same number of trees
        n_trees = self.n_epochs
        for i in range(n_trees):
            tree1 = model1.get_tree(i)
            tree2 = model2.get_tree(i)
            
            # Compare tree metadata
            self.assertEqual(
                tree1['n_leaves'], tree2['n_leaves'],
                f"Tree {i}: n_leaves differ on {device}"
            )
            self.assertEqual(
                tree1['tree_depth'], tree2['tree_depth'],
                f"Tree {i}: tree_depth differ on {device}"
            )
            self.assertEqual(
                tree1['is_oblivious'], tree2['is_oblivious'],
                f"Tree {i}: is_oblivious differ on {device}"
            )
            
            # Compare tree arrays
            np.testing.assert_array_equal(
                tree1['depths'], tree2['depths'],
                err_msg=f"Tree {i}: depths differ on {device}"
            )
            np.testing.assert_allclose(
                tree1['values'], tree2['values'], rtol=1e-5, atol=1e-6,
                err_msg=f"Tree {i}: leaf values differ on {device}"
            )
            np.testing.assert_allclose(
                tree1['edge_weights'], tree2['edge_weights'], rtol=1e-5, atol=1e-6,
                err_msg=f"Tree {i}: edge_weights differ on {device}"
            )
            np.testing.assert_array_equal(
                tree1['feature_indices'], tree2['feature_indices'],
                err_msg=f"Tree {i}: feature_indices differ on {device}"
            )
            np.testing.assert_allclose(
                tree1['feature_values'], tree2['feature_values'], rtol=1e-5, atol=1e-6,
                err_msg=f"Tree {i}: feature_values differ on {device}"
            )

    def test_tree_extraction_insertion_cpu(self):
        """Test extracting trees from one model and adding to another on CPU"""
        print("\n=== Testing tree extraction/insertion on CPU ===")
        
        # Train model1 and extract trees
        model1, trees = self._train_and_extract_trees(device='cpu')
        print(f"Model 1: Trained {len(trees)} trees")
        
        # Build model2 from extracted trees
        model2 = self._build_from_extracted_trees(trees, device='cpu')
        print(f"Model 2: Built from {len(trees)} extracted trees")
        
        # Compare models
        self._compare_models(model1, model2, self.X, device='cpu')
        print("✓ CPU: Models produce identical predictions and have identical tree structures")

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_tree_extraction_insertion_gpu(self):
        """Test extracting trees from one model and adding to another on GPU"""
        print("\n=== Testing tree extraction/insertion on GPU ===")
        
        # Train model1 and extract trees
        model1, trees = self._train_and_extract_trees(device='cuda')
        print(f"Model 1 (GPU): Trained {len(trees)} trees")
        
        # Build model2 from extracted trees
        model2 = self._build_from_extracted_trees(trees, device='cuda')
        print(f"Model 2 (GPU): Built from {len(trees)} extracted trees")
        
        # Compare models
        self._compare_models(model1, model2, self.X, device='cuda')
        print("✓ GPU: Models produce identical predictions and have identical tree structures")

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_tree_extraction_cpu_to_gpu(self):
        """Test extracting trees from CPU model and adding to GPU model"""
        print("\n=== Testing tree extraction CPU→GPU ===")
        
        # Train on CPU
        model_cpu, trees = self._train_and_extract_trees(device='cpu')
        print(f"CPU Model: Trained {len(trees)} trees")
        
        # Build on GPU from CPU-extracted trees
        model_gpu = self._build_from_extracted_trees(trees, device='cuda')
        print(f"GPU Model: Built from CPU-extracted trees")
        
        # Compare (predictions on same device for comparison)
        with th.no_grad():
            pred_cpu = model_cpu(self.X.cpu()).cpu().numpy()
            pred_gpu = model_gpu(self.X.cuda()).cpu().numpy()
        
        np.testing.assert_allclose(
            pred_cpu, pred_gpu, rtol=1e-5, atol=1e-6,
            err_msg="CPU→GPU predictions differ"
        )
        print("✓ CPU→GPU: Models produce identical predictions")

    def test_incremental_tree_addition_cpu(self):
        """Test adding trees incrementally on CPU (simulating distributed training)"""
        print("\n=== Testing incremental tree addition on CPU ===")
        
        model_full = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=self.tree_struct,
            optimizers=self.optimizer,
            params=self.params,
            device='cpu'
        )
        model_incremental = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=self.tree_struct,
            optimizers=self.optimizer,
            params=self.params,
            device='cpu'
        )
        
        X = self.X.cpu()
        y = self.y.cpu()
        
        # Initialize model_incremental with a dummy forward pass
        _ = model_incremental(X, requires_grad=False)
        
        # Train and add trees one by one
        for epoch in range(self.n_epochs):
            # Train one tree in model_full
            pred = model_full(X, requires_grad=True)
            loss = th.nn.functional.mse_loss(pred, y)
            loss.backward()
            model_full.step()
            
            # Extract the new tree and add to model_incremental
            tree = model_full.get_tree(epoch)
            model_incremental.add_tree(tree)
            
            # Predictions should match after each tree
            with th.no_grad():
                pred_full = model_full(X).numpy()
                pred_inc = model_incremental(X).numpy()
            
            np.testing.assert_allclose(
                pred_full, pred_inc, rtol=1e-5, atol=1e-6,
                err_msg=f"Predictions differ after adding tree {epoch}"
            )
        
        print(f"✓ CPU: Incremental addition of {self.n_epochs} trees successful")

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_incremental_tree_addition_gpu(self):
        """Test adding trees incrementally on GPU (simulating distributed training)"""
        print("\n=== Testing incremental tree addition on GPU ===")
        
        model_full = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=self.tree_struct,
            optimizers=self.optimizer,
            params=self.params,
            device='cuda'
        )
        model_incremental = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=self.tree_struct,
            optimizers=self.optimizer,
            params=self.params,
            device='cuda'
        )
        
        X = self.X.cuda()
        y = self.y.cuda()
        
        # Initialize model_incremental with a dummy forward pass
        _ = model_incremental(X, requires_grad=False)
        
        # Train and add trees one by one
        for epoch in range(self.n_epochs):
            # Train one tree in model_full
            pred = model_full(X, requires_grad=True)
            loss = th.nn.functional.mse_loss(pred, y)
            loss.backward()
            model_full.step()
            
            # Extract the new tree and add to model_incremental
            tree = model_full.get_tree(epoch)
            model_incremental.add_tree(tree)
            
            # Predictions should match after each tree
            with th.no_grad():
                pred_full = model_full(X).cpu().numpy()
                pred_inc = model_incremental(X).cpu().numpy()
            
            np.testing.assert_allclose(
                pred_full, pred_inc, rtol=1e-5, atol=1e-6,
                err_msg=f"Predictions differ after adding tree {epoch} on GPU"
            )
        
        print(f"✓ GPU: Incremental addition of {self.n_epochs} trees successful")

    def test_same_bias_initialization(self):
        """Test that models with same bias produce same results when trees are transferred"""
        print("\n=== Testing same bias initialization ===")
        
        # Create two models with explicit same bias
        bias_value = np.array([0.5], dtype=np.float32)
        model1 = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=self.tree_struct,
            optimizers=self.optimizer,
            params=self.params,
            device='cpu'
        )
        model1.set_bias(bias_value)
        
        model2 = GBTModel(
            input_dim=5,
            output_dim=1,
            tree_struct=self.tree_struct,
            optimizers=self.optimizer,
            params=self.params,
            device='cpu'
        )
        model2.set_bias(bias_value)
        
        X = self.X.cpu()
        y = self.y.cpu()
        
        # Initial predictions should be identical (just bias)
        with th.no_grad():
            pred1 = model1(X).numpy()
            pred2 = model2(X).numpy()
        np.testing.assert_allclose(pred1, pred2, rtol=1e-7, atol=1e-8)
        print(f"✓ Initial predictions identical (bias={bias_value})")
        
        # Train model1
        for epoch in range(self.n_epochs):
            pred = model1(X, requires_grad=True)
            loss = th.nn.functional.mse_loss(pred, y)
            loss.backward()
            model1.step()
        
        # Extract all trees from model1 and add to model2
        for i in range(self.n_epochs):
            tree = model1.get_tree(i)
            model2.add_tree(tree)
        
        # Final predictions should be identical
        with th.no_grad():
            pred1_final = model1(X).numpy()
            pred2_final = model2(X).numpy()
        
        np.testing.assert_allclose(
            pred1_final, pred2_final, rtol=1e-5, atol=1e-6,
            err_msg="Final predictions differ despite same bias and transferred trees"
        )
        print(f"✓ Final predictions identical after transferring {self.n_epochs} trees")


if __name__ == '__main__':
    unittest.main(verbosity=2)
