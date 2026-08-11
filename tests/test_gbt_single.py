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
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th
from sklearn import datasets
from torch.nn.functional import mse_loss

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

from gbrl import cuda_available
from gbrl.common.utils import (cuda_usable, get_poly_vectors, normalize_device,
                               numerical_dtype, preprocess_features)
from gbrl.models.gbt import GBTModel
from tests import CATEGORICAL_INPUTS, CATEGORICAL_OUTPUTS

N_EPOCHS = 100


def rmse_model(model, X, y, n_epochs, device='cpu'):
    y_ = th.tensor(y, dtype=th.float32, device=device).squeeze()
    if isinstance(X, th.Tensor):
        X_ = X.clone().to(device)
    else:
        X_ = X.copy()
    epoch = 0
    while epoch < n_epochs:
        y_pred = model(X_, requires_grad=True)
        loss = 0.5*mse_loss(y_pred, y_)
        loss.backward()
        model.step()
        print(f"epoch: {epoch} loss: {loss.sqrt()}")
        epoch += 1
    y_pred = model(X_)
    loss = (0.5*mse_loss(y_pred, y_)).sqrt().item()
    return loss


def to_utf8(s):
    return s.encode('utf-8').decode('utf-8') if isinstance(s, str) else s


class TestGBTSingle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('Loading data...')
        # Imagine this loads your actual data
        try:
            X, y = datasets.load_diabetes(return_X_y=True,
                                          as_frame=False, scaled=False)
        except TypeError:  # python3.7 uses older version of scikit-learn
            X, y = datasets.load_diabetes(return_X_y=True, as_frame=False)

        X = th.tensor(X, dtype=th.float32)
        out_dim = 1 if len(y.shape) == 1 else y.shape[1]
        if out_dim == 1:
            y = y[:, np.newaxis]
        input_dim = X.shape[1]
        cls.single_data = (X, y)
        cls.out_dim = out_dim
        cls.input_dim = input_dim
        cls.n_epochs = 100
        cls.test_dir = tempfile.mkdtemp()
        # Setup categorical data
        X_categorical = np.array(CATEGORICAL_INPUTS, dtype=str)
        X_categorical = np.char.encode(X_categorical, encoding='utf-8',
                                       errors=None)
        X_categorical = np.char.decode(X_categorical, encoding='utf-8',
                                       errors=None)
        y_categorical = np.array(CATEGORICAL_OUTPUTS,
                                 dtype=np.single)[:, np.newaxis]
        cls.cat_data = (X_categorical, y_categorical)
        cls.tree_struct = {'max_depth': 4,
                           'n_bins': 256, 'min_data_in_leaf': 0,
                           'par_th': 2,
                           'grow_policy': 'greedy'}
        cls.sgd_optimizer = {'algo': 'SGD',
                             'lr': 1.0,
                             'start_idx': 0,
                             'stop_idx': out_dim
                             }

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test
        shutil.rmtree(cls.test_dir)

    def test_cosine_cpu(self):
        print("Running test_cosine_cpu")
        X, y = self.single_data
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine",
                       "generator_type": "Quantile"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 5
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_learner(os.path.join(self.test_dir, 'test_cosine_cpu'))

        model.learner.reset()
        model.set_bias_from_targets(y)
        train_loss = model.fit(X, y, self.n_epochs)
        self.assertTrue(train_loss < value, f'Expected loss = {train_loss} '
                        f'< {value}')

        X_categorical, y_categorical = self.cat_data
        model = GBTModel(input_dim=X_categorical.shape[1],
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value, f'Expected Categorical loss = '
                        f'{loss} < {value}')

    def test_matrix_representation_cpu(self):
        """Test matrix representation (A, V) generation on CPU."""
        print("Running test_matrix_representation_cpu")
        X, y = self.single_data
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine",
                       "generator_type": "Quantile"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y)
        _ = rmse_model(model, X, y, self.n_epochs)
        
        A, V, n_leaves_per_tree, n_leaves, n_trees = model.learner.get_matrix_representation(X)
        preds_representation = (A @ V).squeeze()
        self.assertTrue(np.allclose(preds_representation, model(X, tensor=False)),
                        "Matrix representation A @ V should equal model predictions")

    def test_compress_cpu(self):
        """Test tree ensemble compression on CPU."""
        print("Running test_compress_cpu")
        X, y = self.single_data
        k = 50  # Number of trees to discard (keep trees from k onwards)
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine",
                       "generator_type": "Quantile"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y)
        _ = rmse_model(model, X, y, self.n_epochs)
        
        A, V, n_leaves_per_tree, n_leaves, n_trees = model.learner.get_matrix_representation(X)
        
        # Get predictions starting from tree k (what compressed model should produce)
        y_pred_k = model(X, tensor=False, start_idx=k)
        
        # Select trees k onwards
        tree_selection = th.zeros(n_trees, dtype=th.float32, device='cpu')
        tree_selection[k:] = 1.0
        n_compressed_trees = int(tree_selection.sum())
        
        # Map tree selection to leaf selection
        selection_mask = th.repeat_interleave(
            tree_selection, th.tensor(n_leaves_per_tree, device='cpu'))
        n_compressed_leaves = int(selection_mask.sum())
        
        selection_mask = selection_mask.detach().cpu().numpy()
        tree_selection = tree_selection.detach().cpu().numpy()
        
        compressed_leaf_indices = np.where(selection_mask > 0)[0].astype(np.int32)
        compressed_tree_indices = np.where(tree_selection > 0)[0].astype(np.int32)
        
        # Calculate new tree indices for compressed model
        new_tree_indices = np.zeros(n_compressed_trees)
        new_tree_indices[1:] = np.cumsum(n_leaves_per_tree[compressed_tree_indices])[:-1]
        
        # Create zero correction matrix W_compressed with shape (n_compressed_leaves + 1, output_dim)
        # Row 0 is bias, remaining rows are for compressed leaves only
        W_compressed = np.zeros((n_compressed_leaves + 1, self.out_dim), dtype=np.single)
        
        model.learner._cpp_model.compress(
            n_compressed_leaves, n_compressed_trees, compressed_leaf_indices,
            compressed_tree_indices, new_tree_indices.astype(np.int32), W_compressed)
        
        compressed_y = model(X, tensor=False)
        self.assertTrue(np.allclose(compressed_y, y_pred_k),
                        "Discarding trees should be equal to prediction without them")

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over "
                     "gpu tests")
    def test_matrix_representation_gpu(self):
        """Test matrix representation (A, V) generation on GPU."""
        print("Running test_matrix_representation_gpu")
        X, y = self.single_data
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cuda')
        model.set_bias_from_targets(y)
        _ = rmse_model(model, X, y, self.n_epochs, device='cuda')
        
        A, V, n_leaves_per_tree, n_leaves, n_trees = model.learner.get_matrix_representation(X)
        preds_representation = (A @ V).squeeze()
        self.assertTrue(np.allclose(preds_representation, model(X, tensor=False)),
                        "Matrix representation A @ V should equal model predictions")

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over "
                     "gpu tests")
    def test_compress_gpu(self):
        """Test tree ensemble compression on GPU."""
        print("Running test_compress_gpu")
        X, y = self.single_data
        k = 50  # Number of trees to discard (keep trees from k onwards)
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cuda')
        model.set_bias_from_targets(y)
        _ = rmse_model(model, X, y, self.n_epochs, device='cuda')
        
        A, V, n_leaves_per_tree, n_leaves, n_trees = model.learner.get_matrix_representation(X)
        
        # Get predictions starting from tree k (what compressed model should produce)
        y_pred_k = model(X, tensor=False, start_idx=k)
        
        # Select trees k onwards
        tree_selection = th.zeros(n_trees, dtype=th.float32, device='cuda')
        tree_selection[k:] = 1.0
        n_compressed_trees = int(tree_selection.sum())
        
        # Map tree selection to leaf selection
        selection_mask = th.repeat_interleave(
            tree_selection, th.tensor(n_leaves_per_tree, device='cuda'))
        n_compressed_leaves = int(selection_mask.sum())
        
        selection_mask = selection_mask.detach().cpu().numpy()
        tree_selection = tree_selection.detach().cpu().numpy()
        
        compressed_leaf_indices = np.where(selection_mask > 0)[0].astype(np.int32)
        compressed_tree_indices = np.where(tree_selection > 0)[0].astype(np.int32)
        
        # Calculate new tree indices for compressed model
        new_tree_indices = np.zeros(n_compressed_trees)
        new_tree_indices[1:] = np.cumsum(n_leaves_per_tree[compressed_tree_indices])[:-1]
        
        # Create zero correction matrix W_compressed with shape (n_compressed_leaves + 1, output_dim)
        # Row 0 is bias, remaining rows are for compressed leaves only
        W_compressed = np.zeros((n_compressed_leaves + 1, self.out_dim), dtype=np.single)
        
        model.learner._cpp_model.compress(
            n_compressed_leaves, n_compressed_trees, compressed_leaf_indices,
            compressed_tree_indices, new_tree_indices.astype(np.int32), W_compressed)
        
        compressed_y = model(X, tensor=False)
        self.assertTrue(np.allclose(compressed_y, y_pred_k),
                        "Discarding trees should be equal to prediction without them")

    def test_matrix_representation_oblivious_cpu(self):
        """Test matrix representation for oblivious trees on CPU."""
        print("Running test_matrix_representation_oblivious_cpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4,
                       'n_bins': 256, 'min_data_in_leaf': 0,
                       'par_th': 2,
                       'grow_policy': 'oblivious'}
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y)
        _ = rmse_model(model, X, y, self.n_epochs)
        
        A, V, _, _, _ = model.learner.get_matrix_representation(X)
        preds_representation = (A @ V).squeeze()
        self.assertTrue(np.allclose(preds_representation, model(X, tensor=False)),
                        "Matrix representation A @ V should equal model predictions for oblivious trees")

    def test_copy_cpu(self):
        print("Running test_copy_cpu")
        X, y = self.single_data
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine",
                       "generator_type": "Quantile"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y)
        _ = rmse_model(model, X, y, self.n_epochs)
        copy_model = model.copy()
        y_pred = model(X, requires_grad=False, tensor=False)
        y_copy_pred = copy_model(X, requires_grad=False, tensor=False)
        assert np.allclose(y_pred, y_copy_pred), (
            "Expected copied GBRL model to be equal to original"
        )

    def test_continuation_cpu(self):
        print("Running test_continuation_cpu")
        X, y = self.single_data
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine",
                       "generator_type": "Quantile"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs // 2)
        model.save_learner(os.path.join(self.test_dir,
                                        'test_continuation_cpu'))
        new_model = GBTModel.load_learner(os.path.join(self.test_dir,
                                                       'test_continuation_cpu'),
                                          device='cpu')
        loss = rmse_model(new_model, X, y, self.n_epochs // 2)
        value = 5
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over "
                     "gpu tests")
    def test_continuation_gpu(self):
        print("Running test_continuation_gpu")
        X, y = self.single_data
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine",
                       "generator_type": "Quantile"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs // 2)
        model.save_learner(os.path.join(self.test_dir,
                                        'test_continuation_gpu'))
        new_model = GBTModel.load_learner(os.path.join(self.test_dir,
                                                       'test_continuation_gpu'),
                                          device='cuda')
        loss = rmse_model(new_model, X, y, self.n_epochs // 2, device='cuda')
        value = 5
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')

    def test_shap_cpu(self):
        """tree_shap completeness: base + sum_f shap_f(x) == predict(x) for one SGD tree."""
        print("Running test_shap_cpu")
        X, y = self.single_data
        X_cpu = X.detach().clone().cpu().numpy()
        tree_struct = {'max_depth': 3,
                       'n_bins': 256, 'min_data_in_leaf': 1,
                       'par_th': 2,
                       'grow_policy': 'greedy'}
        params = dict({"control_variates": False,
                       "split_score_func": "L2",
                       "generator_type": "Uniform"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.learner.step(X, y)
        pred = model(X_cpu).detach().cpu().numpy().reshape(len(X_cpu), -1)
        shap_vals, base = model.tree_shap(0, X_cpu, return_base=True)
        reconstructed = base + shap_vals.sum(axis=1)
        max_err = float(np.abs(reconstructed - pred).max())
        self.assertLess(
            max_err, 2e-3,
            f'tree_shap completeness violated: max |base+sum(shap)-pred|={max_err:.4f}')

    def test_ensemble_shap_completeness_sgd(self):
        """ensemble shap(): base + sum_f shap[f] == predict(x) for SGD constant and linear lr."""
        print("Running test_ensemble_shap_completeness_sgd")
        rng = np.random.default_rng(42)
        n, d = 200, 6
        X = rng.normal(size=(n, d)).astype(np.float32)
        Y = (2 * X[:, 0] - X[:, 1]).astype(np.float32)[:, np.newaxis]
        n_iters = 30

        # Constant lr schedules.
        for lr in (0.1, 0.05):
            for output_dim, Y_use in [(1, Y), (2, np.hstack([Y, -Y]))]:
                opt = {'algo': 'SGD', 'lr': lr, 'start_idx': 0, 'stop_idx': output_dim}
                model = GBTModel(
                    input_dim=d, output_dim=output_dim,
                    tree_struct={'max_depth': 4, 'n_bins': 256, 'min_data_in_leaf': 1,
                                 'grow_policy': 'oblivious'},
                    optimizers=opt,
                    params={'split_score_func': 'Cosine', 'generator_type': 'Quantile'},
                    device='cpu', verbose=0,
                )
                model.set_bias_from_targets(Y_use)
                target = th.as_tensor(Y_use)
                for _ in range(n_iters):
                    pred_t = model(X, requires_grad=True)
                    loss = ((pred_t.reshape(target.shape) - target) ** 2).mean()
                    loss.backward()
                    model.step(X)

                pred = model(X).detach().cpu().numpy().reshape(n, output_dim)
                shap_vals, base = model.shap(X, return_base=True)
                max_err = float(np.abs(base + shap_vals.sum(axis=1) - pred).max())
                self.assertLess(
                    max_err, 1e-3,
                    f'SGD lr={lr} output_dim={output_dim}: max |base+sum(shap)-pred|={max_err:.4f}')

        # Linear lr schedule: off-by-one in tree_idx would show up here.
        output_dim = 1
        opt_linear = {'algo': 'SGD', 'lr': 0.1, 'stop_lr': 0.01, 'T': n_iters,
                      'start_idx': 0, 'stop_idx': output_dim}
        model = GBTModel(
            input_dim=d, output_dim=output_dim,
            tree_struct={'max_depth': 4, 'n_bins': 256, 'min_data_in_leaf': 1,
                         'grow_policy': 'oblivious'},
            optimizers=opt_linear,
            params={'split_score_func': 'Cosine', 'generator_type': 'Quantile'},
            device='cpu', verbose=0,
        )
        model.set_bias_from_targets(Y)
        target = th.as_tensor(Y)
        for _ in range(n_iters):
            pred_t = model(X, requires_grad=True)
            loss = ((pred_t.reshape(target.shape) - target) ** 2).mean()
            loss.backward()
            model.step(X)

        pred = model(X).detach().cpu().numpy().reshape(n, output_dim)
        shap_vals, base = model.shap(X, return_base=True)
        max_err = float(np.abs(base + shap_vals.sum(axis=1) - pred).max())
        self.assertLess(
            max_err, 1e-3,
            f'SGD linear-lr: max |base+sum(shap)-pred|={max_err:.4f}')

    def test_ensemble_shap_completeness_adam(self):
        """Adam local TreeSHAP: full ensemble completeness with sample-specific base."""
        print("Running test_ensemble_shap_completeness_adam")
        rng = np.random.default_rng(7)
        n, d = 200, 6
        X = rng.normal(size=(n, d)).astype(np.float32)
        Y = (2 * X[:, 0] - X[:, 1]).astype(np.float32)[:, np.newaxis]

        for output_dim, Y_use in [(1, Y), (2, np.hstack([Y, -Y]))]:
            opt = {'algo': 'Adam', 'lr': 0.01, 'start_idx': 0, 'stop_idx': output_dim}
            model = GBTModel(
                input_dim=d, output_dim=output_dim,
                tree_struct={'max_depth': 4, 'n_bins': 256, 'min_data_in_leaf': 1,
                             'grow_policy': 'oblivious'},
                optimizers=opt,
                params={'split_score_func': 'Cosine', 'generator_type': 'Quantile'},
                device='cpu', verbose=0,
            )
            model.set_bias_from_targets(Y_use)
            target = th.as_tensor(Y_use)
            for _ in range(30):
                pred_t = model(X, requires_grad=True)
                loss = ((pred_t.reshape(target.shape) - target) ** 2).mean()
                loss.backward()
                model.step(X)

            pred = model(X).detach().cpu().numpy().reshape(n, output_dim)

            # --- Test 1: ensemble completeness with sample-specific base ---
            # This exercises Adam moment state accumulated across all trees.
            phi, base = model.shap(X, return_base=True)      # (n,d,out), (n,out)
            reconstructed = base + phi.sum(axis=1)
            max_err = float(np.abs(reconstructed - pred).max())
            self.assertLess(
                max_err, 1e-4,
                f'Adam ensemble completeness violated for output_dim={output_dim}: '
                f'max |base+sum(shap)-pred|={max_err:.4f}')

            # --- Test 2: tree_shap(1) has a sample-specific base different from tree_shap(0) ---
            # For Adam, tree 1's effective leaf values depend on the Adam state produced by tree 0,
            # which is sample-specific. So base_t1 != base_t0 in general.
            phi_t0, base_t0 = model.tree_shap(0, X, return_base=True)  # (n,d,out), (n,out)
            phi_t1, base_t1 = model.tree_shap(1, X, return_base=True)

            # base_t1 varies across samples via each sample's frozen Adam state from tree 0.
            # std over the sample axis only: flattening mixes in the output-dim spread.
            self.assertGreater(
                float(np.max(np.std(np.asarray(base_t1), axis=0))),
                1e-6,
                f'Adam tree_shap(1) base should be sample-specific for output_dim={output_dim}')

            # Consistency: sum of per-tree (base + phi.sum) should give non-trivial contributions
            tree_contributions = (base_t0 + phi_t0.sum(axis=1)) + (base_t1 + phi_t1.sum(axis=1))
            self.assertGreater(
                float(np.abs(tree_contributions).mean()),
                1e-6,
                f'Adam tree_shap contributions should be non-trivial for output_dim={output_dim}')

    def test_cosine_adam_cpu(self):
        print("Running test_cosine_adam_cpu")
        X, y = self.single_data
        optimizer = {'algo': 'Adam',
                     'lr': 1.0,
                     'start_idx': 0,
                     'stop_idx': self.out_dim
                     }
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 50
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_learner(os.path.join(self.test_dir, 'test_cosine_adam_cpu'))

    @unittest.skipIf(not cuda_available(), "cuda not available skipping "
                     "over gpu tests")
    def test_cosine_gpu(self):
        print("Running test_cosine_gpu")
        X, y = self.single_data
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cuda')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs, device='cuda')
        value = 2
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_learner(os.path.join(self.test_dir, 'test_cosine_gpu'))

        model.learner.reset()
        model.set_bias_from_targets(y)
        train_loss = model.fit(X, y, self.n_epochs)
        value = 2
        self.assertTrue(train_loss < value, 'Expected loss = '
                        f'{train_loss} < {value}')

        X_categorical, y_categorical = self.cat_data
        model = GBTModel(input_dim=X_categorical.shape[1],
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cuda')
        model.set_bias_from_targets(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical,
                          self.n_epochs, device='cuda')
        value = 5000
        self.assertTrue(loss < value,
                        f'Expected Categorical loss = {loss} < {value}')

    def test_cosine_oblivious_cpu(self):
        print("Running test_cosine_oblivious_cpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4,
                       'n_bins': 256, 'min_data_in_leaf': 0,
                       'par_th': 2,
                       'grow_policy': 'oblivious'}
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 13
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_learner(os.path.join(self.test_dir,
                                        'test_cosine_oblivious_gpu'))
        model.learner.reset()
        model.set_bias_from_targets(y)
        train_loss = model.fit(X, y, self.n_epochs)
        self.assertTrue(train_loss < value,
                        f'Expected loss = {train_loss} < {value}')
        X_categorical, y_categorical = self.cat_data
        model = GBTModel(input_dim=X_categorical.shape[1],
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value,
                        f'Expected Categorical loss = {loss} < {value}')

    @unittest.skipIf(not cuda_available(),
                     "cuda not available skipping over gpu tests")
    def test_cosine_oblivious_gpu(self):
        print("Running test_cosine_oblivious_gpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4,
                       'n_bins': 256, 'min_data_in_leaf': 0,
                       'par_th': 2,
                       'grow_policy': 'oblivious'}
        params = dict({"control_variates": False,
                       "split_score_func": "Cosine"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cuda')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs, device='cuda')
        value = 12
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_learner(os.path.join(self.test_dir,
                                        'test_cosine_oblivious_gpu'))
        model.learner.reset()
        model.set_bias_from_targets(y)
        train_loss = model.fit(X, y, self.n_epochs)
        self.assertTrue(train_loss < value,
                        f'Expected loss = {train_loss} < {value}')

        X_categorical, y_categorical = self.cat_data
        model = GBTModel(input_dim=X_categorical.shape[1],
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cuda')
        model.set_bias_from_targets(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs,
                          device='cuda')
        value = 5000
        self.assertTrue(loss < value,
                        f'Expected Categorical loss = {loss} < {value}')

    def test_l2_cpu(self):
        print("Running test_l2_cpu")
        X, y = self.single_data
        params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GBTModel(
                    input_dim=self.input_dim,
                    output_dim=self.out_dim,
                    tree_struct=self.tree_struct,
                    optimizers=self.sgd_optimizer,
                    params=params,
                    verbose=1,
                    device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        self.assertTrue(loss < 0.5, f'Expected loss = {loss} < 0.5')
        model.save_learner(os.path.join(self.test_dir, 'test_l2_cpu'))
        model.learner.reset()
        model.set_bias_from_targets(y)
        train_loss = model.fit(X, y, self.n_epochs)
        value = 6.0
        self.assertTrue(train_loss < value,
                        f'Expected loss = {train_loss} < {value}')
        X_categorical, y_categorical = self.cat_data
        model = GBTModel(input_dim=X_categorical.shape[1],
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value,
                        f'Expected Categorical loss = {loss} < {value}')

    @unittest.skipIf(not cuda_available(),
                     "cuda not available skipping over gpu tests")
    def test_l2_gpu(self):
        print("Running test_l2_gpu")
        X, y = self.single_data
        params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GBTModel(
                    input_dim=self.input_dim,
                    output_dim=self.out_dim,
                    tree_struct=self.tree_struct,
                    optimizers=self.sgd_optimizer,
                    params=params,
                    verbose=0,
                    device='cuda')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs, device='cuda')

        self.assertTrue(loss < 0.5, f'Expected loss = {loss} < 0.5')
        model.save_learner(os.path.join(self.test_dir, 'test_l2_gpu'))
        model.learner.reset()
        model.set_bias_from_targets(y)
        train_loss = model.fit(X, y, self.n_epochs)
        value = 0.5
        self.assertTrue(train_loss < value,
                        f'Expected loss = {train_loss} < {value}')
        X_categorical, y_categorical = self.cat_data
        model = GBTModel(input_dim=X_categorical.shape[1],
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cuda')
        model.set_bias_from_targets(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs,
                          device='cuda')
        value = 5000
        self.assertTrue(loss < value,
                        f'Expected Categorical loss = {loss} < {value}')

    def test_l2_oblivious_cpu(self):
        print("Running test_l2_oblivious_cpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4,
                       'n_bins': 256, 'min_data_in_leaf': 0,
                       'par_th': 2,
                       'grow_policy': 'oblivious'}
        params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GBTModel(
                    input_dim=self.input_dim,
                    output_dim=self.out_dim,
                    tree_struct=tree_struct,
                    optimizers=self.sgd_optimizer,
                    params=params,
                    verbose=0,
                    device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        self.assertTrue(loss < 10.0, f'Expected loss = {loss} < 10.0')
        model.save_learner(os.path.join(self.test_dir,
                                        'test_l2_oblivious_cpu'))

        model.learner.reset()
        model.set_bias_from_targets(y)
        train_loss = model.fit(X, y, self.n_epochs)
        value = 10.0
        self.assertTrue(train_loss < value,
                        f'Expected loss = {train_loss} < {value}')

        X_categorical, y_categorical = self.cat_data

        model = GBTModel(input_dim=X_categorical.shape[1],
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model.set_bias_from_targets(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value,
                        f'Expected Categorical loss = {loss} < {value}')

    @unittest.skipIf(not cuda_available(),
                     "cuda not available skipping over gpu tests")
    def test_l2_oblivious_gpu(self):
        print("Running test_l2_oblivious_gpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4,
                       'n_bins': 256, 'min_data_in_leaf': 0,
                       'par_th': 2,
                       'grow_policy': 'oblivious'}
        params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cuda')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs, device='cuda')
        self.assertTrue(loss < 10.0, f'Expected loss = {loss} < 10.0')
        model.save_learner(os.path.join(self.test_dir,
                                        'test_l2_oblivious_gpu'))
        model.learner.reset()
        model.set_bias_from_targets(y)
        train_loss = model.fit(X, y, self.n_epochs)
        value = 10.0
        self.assertTrue(train_loss < value,
                        f'Expected loss = {train_loss} < {value}')
        X_categorical, y_categorical = self.cat_data
        model = GBTModel(input_dim=X_categorical.shape[1],
                         output_dim=self.out_dim,
                         tree_struct=self.tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cuda')
        model.set_bias_from_targets(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical,
                          self.n_epochs, device='cuda')
        value = 5000
        self.assertTrue(loss < value,
                        f'Expected Categorical loss = {loss} < {value}')

    def test_loading(self):
        X, y = self.single_data
        tree_struct = {'max_depth': 4,
                       'n_bins': 256, 'min_data_in_leaf': 0,
                       'par_th': 2,
                       'grow_policy': 'oblivious'}
        params = dict({"control_variates": False,
                       "split_score_func": "cosine"})
        model = GBTModel(input_dim=self.input_dim,
                         output_dim=self.out_dim,
                         tree_struct=tree_struct,
                         optimizers=self.sgd_optimizer,
                         params=params,
                         verbose=0,
                         device='cpu')
        model = GBTModel.load_learner(os.path.join(self.test_dir,
                                                   'test_cosine_cpu'),
                                      device='cpu')
        y_pred = model(X, requires_grad=False, tensor=False)
        loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
        self.assertTrue(loss < 2.0, f'Expected loss = {loss} < 2.0')

        model = GBTModel.load_learner(os.path.join(self.test_dir, 'test_l2_cpu'),
                                      device='cpu')
        y_pred = model(X, requires_grad=False, tensor=False)
        loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
        self.assertTrue(loss < 0.5, f'Expected loss = {loss} < 0.5')
        if (cuda_available()):
            model = GBTModel.load_learner(os.path.join(self.test_dir,
                                                       'test_cosine_gpu'),
                                          device='cuda')
            y_pred = model(X, requires_grad=False, tensor=False)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 2.0, f'Expected loss = {loss} < 2.0')
            model = GBTModel.load_learner(
                os.path.join(self.test_dir,
                             'test_cosine_oblivious_gpu'), device='cuda')
            y_pred = model(X, requires_grad=False, tensor=False)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 12.0, f'Expected loss = {loss} < 12.0')
            model = GBTModel.load_learner(
                os.path.join(self.test_dir, 'test_l2_gpu'), device='cuda')
            y_pred = model(X, requires_grad=False, tensor=False)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 0.5, f'Expected loss = {loss} < 0.5')
            model = GBTModel.load_learner(os.path.join(self.test_dir,
                                                       'test_l2_oblivious_gpu'),
                                          device='cuda')
            y_pred = model(X, requires_grad=False, tensor=False)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 10.0, f'Expected loss = {loss} < 10.0')
        model = GBTModel.load_learner(os.path.join(self.test_dir,
                                                   'test_cosine_adam_cpu'),
                                      device='cpu')
        y_pred = model(X, requires_grad=False, tensor=False)
        loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
        value = 50.0
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')


    def test_shap_rejects_student_model(self):
        """shap() and tree_shap() must raise as soon as a student model is attached."""
        model = GBTModel(
            input_dim=self.input_dim, output_dim=self.out_dim,
            tree_struct={'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 1,
                         'grow_policy': 'oblivious'},
            optimizers=self.sgd_optimizer,
            params={'split_score_func': 'Cosine', 'generator_type': 'Quantile'},
            device='cpu', verbose=0,
        )
        # A sentinel student model suffices: the guard fires before any C++ call.
        model.learner.student_model = object()
        self.addCleanup(setattr, model.learner, 'student_model', None)

        obs = np.zeros((1, self.input_dim), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, 'student model'):
            model.shap(obs, return_base=True)
        with self.assertRaisesRegex(RuntimeError, 'student model'):
            model.tree_shap(0, obs, return_base=True)

    def test_overlapping_optimizer_raises(self):
        """Overlapping optimizer output ranges must raise to the caller."""
        overlapping = [
            {'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 2},
            {'algo': 'SGD', 'lr': 0.05, 'start_idx': 1, 'stop_idx': 3},
        ]
        with self.assertRaisesRegex(ValueError, 'Overlapping optimizer'):
            GBTModel(
                input_dim=self.input_dim, output_dim=3,
                tree_struct={'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 1,
                             'grow_policy': 'oblivious'},
                optimizers=overlapping,
                params={'split_score_func': 'Cosine', 'generator_type': 'Quantile'},
                device='cpu', verbose=0,
            )

    def test_shap_float64_input(self):
        """Low-level pybind binding must not dangle when obs requires float32 conversion.

        The high-level API pre-converts inputs before calling the binding, so it never
        triggers the temporary py::array_t<float> lifetime. This test calls the C++
        binding directly with a float64 array, forcing the conversion inside parse_shap_args,
        and checks that the result matches the float32 baseline.
        """
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, self.input_dim)).astype(np.float32)
        Y = rng.normal(size=(20, self.out_dim)).astype(np.float32)
        model = GBTModel(
            input_dim=self.input_dim, output_dim=self.out_dim,
            tree_struct={'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 1,
                         'grow_policy': 'oblivious'},
            optimizers=self.sgd_optimizer,
            params={'split_score_func': 'Cosine', 'generator_type': 'Quantile'},
            device='cpu', verbose=0,
        )
        target = th.as_tensor(Y)
        for _ in range(5):
            pred_t = model(X, requires_grad=True)
            loss = ((pred_t.reshape(target.shape) - target) ** 2).mean()
            loss.backward()
            model.step(X)

        cpp_model = model.learner._cpp_model
        num_inputs, cat_inputs = preprocess_features(X)
        base_poly, norm_values, offset = get_poly_vectors(model.learner.params['max_depth'], numerical_dtype)
        base_poly   = np.ascontiguousarray(base_poly)
        norm_values = np.ascontiguousarray(norm_values)
        offset      = np.ascontiguousarray(offset)

        phi32, base32 = cpp_model.ensemble_shap_and_base(
            num_inputs, cat_inputs, norm_values, base_poly, offset)

        # float64 obs forces py::cast<py::array_t<float>> to make a temporary float32
        # copy inside parse_shap_args; ShapArgs owns that copy so it cannot dangle.
        phi64, base64 = cpp_model.ensemble_shap_and_base(
            num_inputs.astype(np.float64), cat_inputs, norm_values, base_poly, offset)

        np.testing.assert_allclose(phi32, phi64, rtol=1e-5,
                                   err_msg="float64 obs gives different SHAP values than float32")
        np.testing.assert_allclose(base32, base64, rtol=1e-5,
                                   err_msg="float64 obs gives different base values than float32")


class TestLearnerLifecycle(unittest.TestCase):
    """Guards the bug class behind most lifecycle regressions: load() and
    __copy__ build instances via __new__, so they silently miss fields that
    __init__ sets. Diffing the attribute sets catches all of them at once."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.ts = {'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 1,
                  'par_th': 2, 'grow_policy': 'oblivious'}
        cls.pr = {'split_score_func': 'Cosine', 'generator_type': 'Quantile'}
        cls.X = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def _trained_single(self):
        m = GBTModel(input_dim=4, output_dim=1, tree_struct=self.ts,
                     optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
                     params=dict(self.pr), device='cpu', verbose=0)
        y = th.as_tensor(self.X[:, :1].copy())
        for _ in range(3):
            p = m(self.X, requires_grad=True)
            ((p.reshape(y.shape) - y) ** 2).mean().backward()
            m.step(self.X)
        return m

    def _trained_multi(self):
        from gbrl.learners.multi_gbt_learner import MultiGBTLearner
        from gbrl.common.utils import setup_optimizer
        opt = setup_optimizer({'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1})
        mo = MultiGBTLearner(
            input_dim=4, output_dim=1, tree_struct=self.ts, optimizers=dict(opt),
            params={**self.pr, **self.ts, 'input_dim': 4, 'output_dim': 1, 'policy_dim': 1},
            n_learners=2, policy_dim=1)
        mo.reset()
        g = np.random.default_rng(1).normal(size=(20, 1)).astype(np.float32)
        mo.step(self.X, [g, g])
        return mo

    def test_load_preserves_all_init_attributes(self):
        """load() must set every attribute __init__ does; it bypasses __init__."""
        from gbrl.learners.multi_gbt_learner import MultiGBTLearner
        m = self._trained_single()
        m.save_learner(os.path.join(self.test_dir, 'lc_single'))
        loaded = GBTModel.load_learner(os.path.join(self.test_dir, 'lc_single'), device='cpu')
        missing = sorted(set(vars(m.learner)) - set(vars(loaded.learner)))
        self.assertEqual(missing, [], f'GBTLearner.load() missing attrs: {missing}')

        mo = self._trained_multi()
        mo.save(os.path.join(self.test_dir, 'lc_multi'))
        mol = MultiGBTLearner.load(os.path.join(self.test_dir, 'lc_multi'), 'cpu')
        missing = sorted(set(vars(mo)) - set(vars(mol)))
        self.assertEqual(missing, [], f'MultiGBTLearner.load() missing attrs: {missing}')

    def test_copy_without_student_models(self):
        """__copy__ must not assume distillation has run."""
        import copy as _copy
        _copy.copy(self._trained_single().learner)
        _copy.copy(self._trained_multi())

    def test_loaded_model_can_reset(self):
        """grow_policy must round-trip: the emitted string has to parse back."""
        m = self._trained_single()
        m.save_learner(os.path.join(self.test_dir, 'lc_reset'))
        loaded = GBTModel.load_learner(os.path.join(self.test_dir, 'lc_reset'), device='cpu')
        loaded.learner.reset()

    def test_multi_accepts_int_policy_dim(self):
        """policy_dim is documented as int-or-list; output_dim is normalized, so
        policy_dim must be too or the base-class type assertion fires."""
        self._trained_multi()


class TestLinearScheduler(unittest.TestCase):
    """The linear schedule must honour its documented endpoints in BOTH
    directions: decay (stop_lr < lr) and warmup (stop_lr > lr), including
    once t exceeds T, where the lr must stay clamped at stop_lr."""

    def _model(self, lr, stop_lr, T):
        return GBTModel(
            input_dim=4, output_dim=1,
            tree_struct={'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 1,
                         'par_th': 2, 'grow_policy': 'oblivious'},
            optimizers={'algo': 'SGD', 'lr': lr, 'stop_lr': stop_lr, 'T': T,
                        'scheduler': 'Linear', 'start_idx': 0, 'stop_idx': 1},
            params={'split_score_func': 'Cosine', 'generator_type': 'Quantile'},
            device='cpu', verbose=0)

    def _train(self, model, n):
        X = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
        y = th.as_tensor(X[:, :1].copy())
        for _ in range(n):
            p = model(X, requires_grad=True)
            ((p.reshape(y.shape) - y) ** 2).mean().backward()
            model.step(X)

    def test_warmup_never_exceeds_stop_lr(self):
        """stop_lr > lr must ramp up to stop_lr and stay there, not overshoot."""
        T = 5
        model = self._model(lr=0.1, stop_lr=0.5, T=T)
        self._train(model, T * 2)
        lr = float(np.asarray(model.learner.get_schedule_learning_rates()).ravel()[0])
        self.assertLessEqual(lr, 0.5 + 1e-6,
                             f'warmup lr {lr} exceeded stop_lr past T')
        self.assertGreaterEqual(lr, 0.1 - 1e-6, f'warmup lr {lr} below init_lr')

    def test_decay_never_undershoots_stop_lr(self):
        T = 5
        model = self._model(lr=0.1, stop_lr=0.01, T=T)
        self._train(model, T * 2)
        lr = float(np.asarray(model.learner.get_schedule_learning_rates()).ravel()[0])
        self.assertGreaterEqual(lr, 0.01 - 1e-6, f'decay lr {lr} below stop_lr')
        self.assertLessEqual(lr, 0.1 + 1e-6, f'decay lr {lr} above init_lr')


class TestLowLevelBindingContract(unittest.TestCase):
    """gbrl_cpp is internal and does not convert its inputs.

    It borrows the caller's buffer and rejects a strided or wrong-dtype array
    instead of forcecasting it: a temporary copy would be owned by the binding
    frame and freed on return, so the backend would read a dangling pointer
    after releasing the GIL.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.X = rng.normal(size=(64, 4)).astype(np.float32)
        cls.y = (cls.X[:, 0] * 2.0).reshape(-1, 1).astype(np.float32)
        cls.model = GBTModel(
            input_dim=4, output_dim=1,
            tree_struct={'max_depth': 3, 'n_bins': 32, 'min_data_in_leaf': 0,
                         'par_th': 2, 'grow_policy': 'greedy'},
            optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
            params={'split_score_func': 'L2', 'generator_type': 'Quantile'},
            device='cpu', verbose=0)
        cls.model.fit(cls.X, cls.y, iterations=3, shuffle=False)

    def test_non_contiguous_is_rejected_not_silently_copied(self):
        strided = np.ascontiguousarray(self.X[:, ::-1])[:, ::-1]
        self.assertFalse(strided.flags['C_CONTIGUOUS'])
        self.assertTrue(np.array_equal(strided, self.X))   # same values
        with self.assertRaises(Exception):
            self.model.learner._cpp_model.predict(strided, None, 0, 0)

    def test_wrong_dtype_is_rejected(self):
        with self.assertRaises(Exception):
            self.model.learner._cpp_model.predict(
                np.ascontiguousarray(self.X, dtype=np.float64), None, 0, 0)

    def test_contiguous_float32_still_works(self):
        preds = np.asarray(
            self.model.learner._cpp_model.predict(np.ascontiguousarray(self.X), None, 0, 0))
        self.assertTrue(np.all(np.isfinite(preds)))


class TestRepeatedFit(unittest.TestCase):
    """fit() must boost against every tree already in the model.

    The tree range is counted from the start of the ensemble, not the start of
    the call, so splitting a run into two calls yields the same model as doing
    it in one. Distillation depends on this: it calls fit() repeatedly on the
    same student, and each chunk past the first must boost against the previous
    ones.

    These tests use the default batch_size, which is larger than the sample
    count, so every tree here sees the full dataset and the comparison isolates
    the tree-range behaviour.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.X = rng.normal(size=(256, 4)).astype(np.float32)
        cls.y = (cls.X[:, 0] * 2.0 - cls.X[:, 1]).reshape(-1, 1).astype(np.float32)
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def _model(self, device='cpu', grow_policy='greedy'):
        return GBTModel(
            input_dim=4, output_dim=1,
            tree_struct={'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 0,
                         'par_th': 2, 'grow_policy': grow_policy},
            optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
            params={'split_score_func': 'L2', 'generator_type': 'Quantile'},
            device=device, verbose=0)

    def _assert_split_matches_single(self, device, grow_policy='greedy'):
        one = self._model(device, grow_policy)
        one.fit(self.X, self.y, iterations=5, shuffle=False)

        two = self._model(device, grow_policy)
        two.fit(self.X, self.y, iterations=3, shuffle=False)
        two.fit(self.X, self.y, iterations=2, shuffle=False)

        self.assertEqual(one.learner.get_num_trees(), two.learner.get_num_trees())
        np.testing.assert_allclose(
            np.asarray(one.learner.predict(self.X, requires_grad=False, tensor=False)),
            np.asarray(two.learner.predict(self.X, requires_grad=False, tensor=False)),
            rtol=1e-4, atol=1e-5,
            err_msg=f'{device}/{grow_policy}: 3+2 iterations disagree with 5')

    def test_split_fit_matches_single_fit_cpu(self):
        self._assert_split_matches_single('cpu')

    def test_split_fit_matches_single_fit_cpu_oblivious(self):
        self._assert_split_matches_single('cpu', grow_policy='oblivious')

    @unittest.skipUnless(cuda_available(), 'CUDA not available')
    def test_split_fit_matches_single_fit_cuda(self):
        self._assert_split_matches_single('cuda')

    def _assert_many_chunks_match_single(self, device):
        """Six chunks of 5 must equal one call of 30; drift compounds across
        chunks, so many small calls is the sharper check."""
        chunked = self._model(device)
        for _ in range(6):
            chunked.fit(self.X, self.y, iterations=5, shuffle=False)

        single = self._model(device)
        single.fit(self.X, self.y, iterations=30, shuffle=False)

        self.assertEqual(chunked.learner.get_num_trees(), single.learner.get_num_trees())
        np.testing.assert_allclose(
            np.asarray(chunked.learner.predict(self.X, requires_grad=False, tensor=False)),
            np.asarray(single.learner.predict(self.X, requires_grad=False, tensor=False)),
            rtol=1e-4, atol=1e-5,
            err_msg=f'{device}: 6x5 iterations disagree with 30')

    def _assert_split_matches_single_cv(self, device):
        """Control variates key off the ensemble size, not the loop counter, so
        the first tree of a second fit() still uses them."""
        def mk():
            return GBTModel(
                input_dim=4, output_dim=1,
                tree_struct={'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 0,
                             'par_th': 2, 'grow_policy': 'greedy'},
                optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
                params={'split_score_func': 'L2', 'generator_type': 'Quantile',
                        'control_variates': True},
                device=device, verbose=0)

        one = mk()
        one.fit(self.X, self.y, iterations=5, shuffle=False)
        two = mk()
        two.fit(self.X, self.y, iterations=3, shuffle=False)
        two.fit(self.X, self.y, iterations=2, shuffle=False)
        np.testing.assert_allclose(
            np.asarray(one.learner.predict(self.X, requires_grad=False, tensor=False)),
            np.asarray(two.learner.predict(self.X, requires_grad=False, tensor=False)),
            rtol=1e-4, atol=1e-5,
            err_msg=f'{device}: 3+2 disagree with 5 under control variates')

    def test_split_fit_matches_single_fit_cpu_control_variates(self):
        self._assert_split_matches_single_cv('cpu')

    def test_categorical_fit_cpu(self):
        """Categorical candidate generation is host code; fit() must feed it a
        host buffer, on CUDA as well as on CPU."""
        self._assert_categorical_fit('cpu')

    @unittest.skipUnless(cuda_available(), 'CUDA not available')
    def test_categorical_fit_cuda(self):
        self._assert_categorical_fit('cuda')

    def _assert_categorical_fit(self, device):
        rng = np.random.default_rng(3)
        cats = np.array(['apple', 'apricot', 'banana', 'cherry'])
        X = np.column_stack([
            rng.normal(size=200).astype(object),
            rng.choice(cats, size=200).astype(object),
        ])
        y = np.where(X[:, 1] == 'apple', 1.0, -1.0).astype(np.float32)[:, None]
        model = GBTModel(
            input_dim=2, output_dim=1,
            tree_struct={'max_depth': 3, 'n_bins': 32, 'min_data_in_leaf': 0,
                         'par_th': 2, 'grow_policy': 'greedy'},
            optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
            params={'split_score_func': 'L2', 'generator_type': 'Quantile'},
            device=device, verbose=0)
        loss = model.fit(X, y, iterations=5, shuffle=True)
        self.assertTrue(np.isfinite(loss), f'{device}: categorical fit loss is not finite')
        preds = np.asarray(model.learner.predict(X, requires_grad=False, tensor=False))
        self.assertTrue(np.all(np.isfinite(preds)), f'{device}: categorical predictions not finite')


    def test_failed_distillation_leaves_learner_untouched(self):
        """A failed distil() must leave no student attached: a half-built one
        would add a bias-only term to predict() and block shap()."""
        model = self._model()
        model.fit(self.X, self.y, iterations=5, shuffle=False)
        before = np.asarray(model.learner.predict(self.X, requires_grad=False, tensor=False))
        targets = before.copy()

        with self.assertRaises(ValueError):
            model.learner.distil(self.X, targets, {'limit_steps': 20})   # no min_steps

        self.assertIsNone(model.learner.student_model,
                          'a failed distillation left a student attached')
        after = np.asarray(model.learner.predict(self.X, requires_grad=False, tensor=False))
        np.testing.assert_allclose(before, after, rtol=0, atol=0,
                                   err_msg='a failed distillation changed predictions')
        model.learner.shap(self.X)   # must still work

    def test_mapping_retry_after_rejected_batch(self):
        """A rejected batch must not stick: the inferred mapping is stored only
        after _ensure_feature_mapping() validates it, so the retry the error
        message asks for can succeed."""
        model = self._model()
        model.fit(self.X, self.y, iterations=3, shuffle=False)
        learner = model.learner
        learner.feature_mapping = None
        learner._feature_mapping_installed = False

        bad = np.column_stack([
            self.X[:8, 0].astype(object), self.X[:8, 1].astype(object),
            self.X[:8, 2].astype(object),
            np.array(['a', 'b'] * 4, dtype=object),
        ])
        with self.assertRaises(ValueError):
            learner.shap(bad)
        self.assertIsNone(learner.feature_mapping,
                          'the rejected mapping was kept and poisons the retry')

        phi = learner.shap(self.X)
        self.assertTrue(learner._feature_mapping_installed)
        self.assertTrue(np.all(np.isfinite(np.asarray(phi))))

    def test_copy_preserves_constraints_and_weights(self):
        """__copy__ rebuilds from params, which describes neither constraints nor
        feature weights, so both must be carried over separately or a reset()
        copy silently drops the monotonic guarantee."""
        import copy as _copy
        weights = np.array([0.1, 2.0, 0.5, 1.5], dtype=np.float32)
        model = GBTModel(
            input_dim=4, output_dim=1,
            tree_struct={'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 0,
                         'par_th': 2, 'grow_policy': 'oblivious'},
            optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
            params={'split_score_func': 'L2', 'generator_type': 'Quantile',
                    'feature_weights': weights,
                    'monotonic_constraints': {0: ('increasing', 0)}},
            device='cpu', verbose=0)
        model.fit(self.X, self.y, iterations=3, shuffle=False)

        copied = _copy.copy(model.learner)
        copied.reset()      # rebuilds from Python state

        self.assertEqual(copied.monotonic_constraints, {0: ('increasing', 0)},
                         'copy lost its monotonic constraints')
        np.testing.assert_allclose(np.asarray(copied.get_feature_weights()).ravel(),
                                   weights, rtol=1e-6,
                                   err_msg='copy lost its feature weights')

    def test_set_feature_weights_survives_reset(self):
        """set_feature_weights() must also update Python state, since reset()
        rebuilds the C++ model from it."""
        model = self._model()
        model.fit(self.X, self.y, iterations=2, shuffle=False)
        weights = np.array([0.25, 1.75, 0.5, 1.0], dtype=np.float32)
        model.learner.set_feature_weights(weights)
        model.learner.reset()
        np.testing.assert_allclose(np.asarray(model.learner.get_feature_weights()).ravel(),
                                   weights, rtol=1e-6,
                                   err_msg='reset() restored stale feature weights')

    def test_adam_rejects_matrix_representation_and_compression(self):
        """V is -lr * raw_leaf_value, which cannot express Adam's sample-specific
        contribution."""
        model = GBTModel(
            input_dim=4, output_dim=1,
            tree_struct={'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 0,
                         'par_th': 2, 'grow_policy': 'greedy'},
            optimizers={'algo': 'Adam', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
            params={'split_score_func': 'L2', 'generator_type': 'Quantile'},
            device='cpu', verbose=0)
        with self.assertRaises(ValueError) as ctx:
            model.learner.get_matrix_representation(self.X)
        self.assertIn('Adam', str(ctx.exception))

        # Adam does not support fit(); use step-based training to add trees.
        X_t = th.tensor(self.X)
        y_t = th.tensor(self.y.squeeze())
        for _ in range(5):
            pred = model(X_t, requires_grad=True)
            ((pred - y_t) ** 2).mean().backward()
            model.step()
        with self.assertRaises(ValueError) as ctx:
            model.learner.compress(trees_to_keep=2, gradient_steps=1, features=self.X)
        self.assertIn('Adam', str(ctx.exception))

    def test_constrained_model_rejects_compression(self):
        """Compression rewrites leaf values and never re-projects them, so the
        compressed model would advertise constraints it no longer satisfies."""
        model = GBTModel(
            input_dim=4, output_dim=1,
            tree_struct={'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 0,
                         'par_th': 2, 'grow_policy': 'oblivious'},
            optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
            params={'split_score_func': 'L2', 'generator_type': 'Quantile',
                    'monotonic_constraints': {0: ('increasing', 0)}},
            device='cpu', verbose=0)
        model.fit(self.X, self.y, iterations=5, shuffle=False)
        with self.assertRaises(ValueError) as ctx:
            model.learner.compress(trees_to_keep=2, gradient_steps=1, features=self.X)
        self.assertIn('monotonic', str(ctx.exception).lower())

    def test_many_chunks_match_single_fit_cpu(self):
        self._assert_many_chunks_match_single('cpu')

    @unittest.skipUnless(cuda_available(), 'CUDA not available')
    def test_many_chunks_match_single_fit_cuda(self):
        self._assert_many_chunks_match_single('cuda')

    def test_distillation_over_multiple_chunks(self):
        """distil() calls the student's fit() again for each chunk past the
        first, so it exercises exactly the continued-fit path."""
        model = self._model()
        model.fit(self.X, self.y, iterations=10, shuffle=False)
        targets = np.asarray(model.learner.predict(self.X, requires_grad=False, tensor=False))
        # A loss threshold of 0 forces the while loop to keep adding chunks
        # until limit_steps, so more than one fit() call always happens.
        params = {'min_steps': 5, 'limit_steps': 15, 'min_distillation_loss': 0.0,
                  'distil_lr': 0.1, 'distil_max_depth': 4}
        loss, out_params = model.learner.distil(self.X, targets, params)
        self.assertGreater(out_params['min_steps'], 5,
                           'distillation did not run more than one fit() chunk')
        self.assertTrue(np.isfinite(loss))
        preds = np.asarray(model.learner.predict(self.X, requires_grad=False, tensor=False))
        self.assertTrue(np.all(np.isfinite(preds)))


class TestDistilledModelRestrictions(unittest.TestCase):
    """Verify that unsupported operations raise when a student model is attached."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        n, d = 80, 4
        cls.X = rng.normal(size=(n, d)).astype(np.float32)
        cls.y = (cls.X[:, 0] - cls.X[:, 1]).astype(np.float32)[:, np.newaxis]
        cls.input_dim = d
        cls.output_dim = 1
        cls.tree_struct = {
            'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 1,
            'par_th': 2, 'grow_policy': 'oblivious'
        }
        cls.params = {
            'split_score_func': 'L2', 'generator_type': 'Quantile',
            'control_variates': False
        }
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def _make_trained_model(self):
        opt = {'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': self.output_dim}
        model = GBTModel(
            input_dim=self.input_dim, output_dim=self.output_dim,
            tree_struct=self.tree_struct, optimizers=opt,
            params=self.params, verbose=0, device='cpu')
        model.fit(self.X, self.y, 15)
        return model

    def _attach_student(self, model):
        targets = np.asarray(model.learner.predict(self.X, requires_grad=False, tensor=False))
        params = {'min_steps': 5, 'limit_steps': 10, 'min_distillation_loss': 0.0,
                  'distil_max_depth': 3, 'distil_lr': 0.1}
        model.learner.distil(self.X, targets, params)

    def test_save_raises_with_student(self):
        model = self._make_trained_model()
        self._attach_student(model)
        with self.assertRaises(ValueError):
            model.save_learner(os.path.join(self.test_dir, 'student_save'))

    def test_export_raises_with_student(self):
        model = self._make_trained_model()
        self._attach_student(model)
        with self.assertRaises(ValueError):
            model.export_learner(os.path.join(self.test_dir, 'student_export'))

    def test_ranged_predict_stop_idx_raises_with_student(self):
        model = self._make_trained_model()
        self._attach_student(model)
        with self.assertRaises(ValueError):
            model(self.X, start_idx=0, stop_idx=5)

    def test_ranged_predict_start_idx_raises_with_student(self):
        model = self._make_trained_model()
        self._attach_student(model)
        with self.assertRaises(ValueError):
            model(self.X, start_idx=3)

    def test_full_predict_works_with_student(self):
        """Default (unranged) predict must still work after distillation."""
        model = self._make_trained_model()
        self._attach_student(model)
        pred = model(self.X, tensor=False)
        self.assertEqual(pred.size, self.y.size)

    def test_matrix_representation_raises_with_student(self):
        model = self._make_trained_model()
        self._attach_student(model)
        with self.assertRaises(ValueError):
            model.learner.get_matrix_representation(self.X)

    def test_full_range_sentinels_allowed_with_student(self):
        """stop_idx=0 means "all trees", so it is the full prediction, not a range."""
        model = self._make_trained_model()
        self._attach_student(model)
        expected = model(self.X, tensor=False)
        for kwargs in ({'stop_idx': 0}, {'start_idx': 0}, {'start_idx': 0, 'stop_idx': 0}):
            np.testing.assert_allclose(
                model(self.X, tensor=False, **kwargs), expected, rtol=1e-6,
                err_msg=f'{kwargs} should equal the full prediction')

    def test_print_and_plot_tree_raise_with_student(self):
        """get_num_trees() counts main+student, but these only see the main model.

        distil() resets the main ensemble to zero trees, so an index that is
        valid per get_num_trees() would otherwise raise "Invalid tree index".
        """
        model = self._make_trained_model()
        self._attach_student(model)
        self.assertGreater(model.learner.get_num_trees(), 0)
        with self.assertRaises(ValueError):
            model.learner.print_tree(0)
        with self.assertRaises(ValueError):
            model.learner.plot_tree(0, os.path.join(self.test_dir, 'student_plot'))

    def test_failed_reset_leaves_python_state_untouched(self):
        """reset() must publish Python state only after the rebuild succeeds.

        Mutating optimizers/total_iterations up front would leave the still-installed
        old model carrying half-updated scheduler state after a failure.
        """
        model = self._make_trained_model()
        learner = model.learner
        before_opts = [dict(o) for o in learner.optimizers]
        before_total = learner.total_iterations
        before_pred = np.asarray(learner.predict(self.X, requires_grad=False, tensor=False))

        saved_algo = learner.optimizers[0]['algo']
        learner.optimizers[0]['algo'] = 'NONEXISTENT_ALGO'
        try:
            with self.assertRaises((ValueError, RuntimeError)):
                learner.reset()
            learner.optimizers[0]['algo'] = saved_algo
            self.assertEqual([dict(o) for o in learner.optimizers], before_opts,
                             'failed reset() mutated self.optimizers')
            self.assertEqual(learner.total_iterations, before_total,
                             'failed reset() mutated total_iterations')
            np.testing.assert_allclose(
                np.asarray(learner.predict(self.X, requires_grad=False, tensor=False)),
                before_pred, rtol=1e-6,
                err_msg='failed reset() disturbed the installed model')
        finally:
            learner.optimizers[0]['algo'] = saved_algo

    def test_set_device_moves_student(self):
        """set_device on an SGD model with a student must not raise and must move both."""
        model = self._make_trained_model()
        self._attach_student(model)
        model.set_device('cpu')
        pred = model(self.X, tensor=False)
        self.assertEqual(pred.size, self.y.size)

    @unittest.skipUnless(cuda_available(), 'CUDA not available')
    def test_set_device_moves_student_to_cuda(self):
        """set_device('cuda') must move a distilled model across devices.

        Main and student must land on the same device: predict() sums both, so a
        split placement would feed one of them the wrong buffer type.
        """
        model = self._make_trained_model()
        self._attach_student(model)
        model.set_device('cuda')
        self.assertEqual(model.learner._cpp_model.get_device(), 'cuda')
        self.assertEqual(model.learner.student_model.get_device(), 'cuda')
        self.assertEqual(model.learner.device, 'cuda')

    def test_set_device_then_reset_keeps_device(self):
        """reset() rebuilds from self.params, so set_device() must update it.

        Updating only self.device would let reset() rebuild on the original
        device while transform_data() routes tensors for the requested one.
        """
        model = self._make_trained_model()
        model.set_device('cpu')
        model.learner.reset()
        self.assertEqual(model.learner.params['device'], 'cpu')
        self.assertEqual(model.get_device(), 'cpu')
        self.assertEqual(model.learner.device, model.learner._cpp_model.get_device())

    @unittest.skipUnless(cuda_available(), 'CUDA not available')
    def test_set_device_cuda_then_reset_keeps_cuda(self):
        """A device set to cuda must survive a reset()."""
        model = self._make_trained_model()
        model.set_device('cuda')
        model.learner.reset()
        self.assertEqual(model.learner.params['device'], 'cuda')
        self.assertEqual(model.get_device(), 'cuda',
                         'reset() rebuilt on the stale device from params')
        self.assertEqual(model.learner.device, model.learner._cpp_model.get_device())


class TestAdamCUDAGuard(unittest.TestCase):
    """Adam is CPU-only; verify the Python API rejects CUDA transitions."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n, d = 50, 4
        cls.X = rng.normal(size=(n, d)).astype(np.float32)
        cls.y = cls.X[:, 0:1].astype(np.float32)
        cls.input_dim = d
        cls.output_dim = 1
        cls.tree_struct = {
            'max_depth': 2, 'n_bins': 32, 'min_data_in_leaf': 1,
            'par_th': 1, 'grow_policy': 'oblivious'
        }
        cls.params = {
            'split_score_func': 'L2', 'generator_type': 'Quantile',
            'control_variates': False
        }
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def _make_adam_model(self):
        import torch as th_local
        opt = {'algo': 'Adam', 'lr': 0.01, 'start_idx': 0, 'stop_idx': self.output_dim}
        model = GBTModel(
            input_dim=self.input_dim, output_dim=self.output_dim,
            tree_struct=self.tree_struct, optimizers=opt,
            params=self.params, verbose=0, device='cpu')
        X_t = th_local.tensor(self.X)
        y_t = th_local.tensor(self.y)
        for _ in range(10):
            pred = model(X_t, requires_grad=True)
            ((pred - y_t.squeeze()) ** 2).mean().backward()
            model.step()
        return model

    # No CUDA skip: the Adam check is pure Python and runs before any device
    # transfer, so it must hold on CPU-only machines too.
    def test_adam_set_device_to_cuda_raises(self):
        model = self._make_adam_model()
        with self.assertRaises(ValueError):
            model.set_device('cuda')

    def test_adam_load_on_cuda_raises(self):
        model = self._make_adam_model()
        path = os.path.join(self.test_dir, 'adam_cuda_test')
        model.save_learner(path)
        with self.assertRaises((ValueError, RuntimeError)):
            GBTModel.load_learner(path, device='cuda')

    def test_rejected_cuda_request_leaves_device_on_cpu(self):
        """A refused CUDA move must not leave Python claiming 'cuda'.

        transform_data() routes tensors on self.device, so a stale 'cuda' here
        would hand a CUDA tensor to a CPU model.
        """
        model = self._make_adam_model()
        with self.assertRaises(ValueError):
            model.set_device('cuda')
        self.assertEqual(model.learner.device, 'cpu')
        self.assertEqual(model.learner.params['device'], 'cpu')
        self.assertEqual(model.get_device(), 'cpu')


class TestCudaUnavailableIsNonDestructive(unittest.TestCase):
    """Requesting CUDA when it is unusable must raise before touching the backend.

    The C++ to_device() falls back to CPU by reallocating the ensemble, which
    drops the trained trees, so the check has to happen in Python first.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.X = rng.normal(size=(60, 4)).astype(np.float32)
        cls.y = (cls.X[:, 0] - cls.X[:, 1]).astype(np.float32)[:, np.newaxis]
        cls.tree_struct = {'max_depth': 3, 'n_bins': 64, 'min_data_in_leaf': 1,
                           'par_th': 2, 'grow_policy': 'oblivious'}
        cls.params = {'split_score_func': 'L2', 'generator_type': 'Quantile',
                      'control_variates': False}
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def _trained(self):
        model = GBTModel(
            input_dim=4, output_dim=1, tree_struct=self.tree_struct,
            optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
            params=self.params, verbose=0, device='cpu')
        model.fit(self.X, self.y, 15)
        return model

    @unittest.skipIf(cuda_usable(), 'CUDA is usable here; this covers the fallback path')
    def test_set_device_cuda_raises_and_preserves_model(self):
        model = self._trained()
        before = model(self.X, tensor=False)
        with self.assertRaises(ValueError):
            model.set_device('cuda')
        np.testing.assert_allclose(
            model(self.X, tensor=False), before, rtol=1e-6,
            err_msg='a refused CUDA move changed the model')
        self.assertEqual(model.get_device(), 'cpu')

    @unittest.skipIf(cuda_usable(), 'CUDA is usable here; this covers the fallback path')
    def test_load_on_cuda_raises(self):
        model = self._trained()
        path = os.path.join(self.test_dir, 'cuda_unavailable')
        model.save_learner(path)
        with self.assertRaises(ValueError):
            GBTModel.load_learner(path, device='cuda')

    @unittest.skipIf(cuda_usable(), 'CUDA is usable here; this covers the fallback path')
    def test_construction_on_cuda_raises(self):
        with self.assertRaises(ValueError):
            GBTModel(
                input_dim=4, output_dim=1, tree_struct=self.tree_struct,
                optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
                params=self.params, verbose=0, device='cuda')

    def test_cuda_usable_matches_runtime(self):
        """cuda_usable() must reflect the runtime, not just the build."""
        self.assertEqual(cuda_usable(), cuda_available() and th.cuda.is_available())

    @unittest.skipIf(cuda_usable(), 'CUDA is usable here; this covers the fallback path')
    def test_gpu_alias_raises_and_preserves_model(self):
        """'gpu' is a documented alias for 'cuda' and must hit the same guard."""
        model = self._trained()
        before = model(self.X, tensor=False)
        with self.assertRaises(ValueError):
            model.set_device('gpu')
        with self.assertRaises(ValueError):
            model.learner.set_device('gpu')
        np.testing.assert_allclose(
            model(self.X, tensor=False), before, rtol=1e-6,
            err_msg="a refused 'gpu' move changed the model")
        self.assertEqual(model.get_device(), 'cpu')

    @unittest.skipIf(cuda_usable(), 'CUDA is usable here; this covers the fallback path')
    def test_gpu_alias_load_and_construction_raise(self):
        model = self._trained()
        path = os.path.join(self.test_dir, 'gpu_alias')
        model.save_learner(path)
        with self.assertRaises(ValueError):
            GBTModel.load_learner(path, device='gpu')
        with self.assertRaises(ValueError):
            GBTModel(
                input_dim=4, output_dim=1, tree_struct=self.tree_struct,
                optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
                params=self.params, verbose=0, device='gpu')

    @unittest.skipUnless(cuda_usable(), 'CUDA required')
    def test_gpu_alias_normalizes_to_cuda(self):
        """With CUDA present, 'gpu' must resolve to the canonical 'cuda'."""
        model = self._trained()
        model.set_device('gpu')
        self.assertEqual(model.get_device(), 'cuda')
        self.assertEqual(model.learner.device, 'cuda')
        self.assertEqual(model.learner.params['device'], 'cuda')

    def test_failed_transfer_rolls_back(self):
        """A mid-transfer failure must not leave main and student on different
        devices: predict() sums both, so one would get the wrong buffer type."""
        model = self._trained()
        learner = model.learner

        class _Boom:
            """Stands in for a student whose transfer fails."""
            def __init__(self):
                self.devices = []

            def to_device(self, device):
                self.devices.append(device)
                if len(self.devices) == 1:
                    raise RuntimeError('simulated transfer failure')

            def get_device(self):
                return 'cpu'

        before = np.asarray(learner.predict(self.X, requires_grad=False, tensor=False))
        origin = learner.device
        boom = _Boom()
        learner.student_model = boom
        try:
            with self.assertRaises(RuntimeError):
                learner.set_device(origin)
        finally:
            learner.student_model = None

        # The main model was put back, so it still agrees with self.device.
        self.assertEqual(learner._cpp_model.get_device(), origin)
        self.assertEqual(learner.device, origin)
        np.testing.assert_allclose(
            np.asarray(learner.predict(self.X, requires_grad=False, tensor=False)),
            before, rtol=1e-6,
            err_msg='a failed transfer changed the predictions')

    def test_normalize_device_contract(self):
        """Aliases and casing resolve; unknown names and bad types raise."""
        self.assertEqual(normalize_device('cpu'), 'cpu')
        self.assertEqual(normalize_device('CPU'), 'cpu')
        self.assertEqual(normalize_device(th.device('cpu')), 'cpu')
        for bad in ('tpu', 'cuda:0', ''):
            with self.assertRaises(ValueError, msg=f'{bad!r} should be rejected'):
                normalize_device(bad)
        with self.assertRaises(TypeError):
            normalize_device(0)

    @unittest.skipUnless(cuda_usable(), 'CUDA required')
    def test_normalize_device_gpu_alias(self):
        self.assertEqual(normalize_device('gpu'), 'cuda')
        self.assertEqual(normalize_device('GPU'), 'cuda')
        self.assertEqual(normalize_device(th.device('cuda')), 'cuda')


if __name__ == '__main__':
    unittest.main()
