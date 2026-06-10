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
Tests for cross-device save/load and device-transfer correctness.

Scenarios covered:
  1. GPU-trained model loaded onto CPU — must not OOM/crash.
  2. GPU-trained model loaded onto CPU then continues training on CPU.
  3. CPU-trained model loaded onto GPU then continues training on GPU.
  4. Model loaded onto CPU, moved to GPU via set_device, continues training.
  5. Metadata capacity invariant: max_trees == n_trees after load/transfer.
  6. Capacity expansion behavior: validated by test_cpu_load_capacity_expands_correctly,
     which checks that the first new tree triggers a realloc sized at n_trees + batch,
     not a reset to any default constant.
"""
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
from gbrl.models.gbt import GBTModel


def _make_model(input_dim, out_dim, tree_struct, optimizer, device):
    return GBTModel(
        input_dim=input_dim,
        output_dim=out_dim,
        tree_struct=tree_struct,
        optimizers=optimizer,
        params={"control_variates": False, "split_score_func": "Cosine",
                "generator_type": "Quantile"},
        verbose=0,
        device=device,
    )


def _train_steps(model, X, y, n_steps, device='cpu'):
    y_ = th.tensor(y, dtype=th.float32, device=device).squeeze()
    X_ = X.clone().to(device) if isinstance(X, th.Tensor) else X.copy()
    for _ in range(n_steps):
        y_pred = model(X_, requires_grad=True)
        loss = 0.5 * mse_loss(y_pred, y_)
        loss.backward()
        model.step()
    y_pred = model(X_)
    return (0.5 * mse_loss(y_pred, y_)).sqrt().item()


class TestDeviceTransfer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            X, y = datasets.load_diabetes(return_X_y=True, as_frame=False,
                                          scaled=False)
        except TypeError:
            X, y = datasets.load_diabetes(return_X_y=True, as_frame=False)

        X = th.tensor(X, dtype=th.float32)
        out_dim = 1 if len(y.shape) == 1 else y.shape[1]
        if out_dim == 1:
            y = y[:, np.newaxis]

        cls.X = X
        cls.y = y
        cls.input_dim = X.shape[1]
        cls.out_dim = out_dim
        cls.test_dir = tempfile.mkdtemp()

        cls.tree_struct = {
            'max_depth': 4, 'n_bins': 256, 'min_data_in_leaf': 0,
            'par_th': 2, 'grow_policy': 'greedy',
        }
        cls.optimizer = {
            'algo': 'SGD', 'lr': 1.0,
            'start_idx': 0, 'stop_idx': out_dim,
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    # ------------------------------------------------------------------
    # Helper: assert capacity invariant
    # ------------------------------------------------------------------
    def _assert_exact_capacity(self, model, label=""):
        """After load or to_device, max_trees/max_leaves must equal n_trees/n_leaves."""
        md = model.learner._cpp_model.get_metadata()
        self.assertEqual(
            md['max_trees'], md['n_trees'],
            f"{label}: max_trees ({md['max_trees']}) != n_trees ({md['n_trees']})"
        )
        self.assertEqual(
            md['max_leaves'], md['n_leaves'],
            f"{label}: max_leaves ({md['max_leaves']}) != n_leaves ({md['n_leaves']})"
        )

    # ------------------------------------------------------------------
    # 1. GPU-trained model loaded onto CPU — must not OOM/crash
    # ------------------------------------------------------------------
    @unittest.skipIf(not cuda_available(), "cuda not available")
    def test_gpu_save_cpu_load_no_crash(self):
        """Loading a GPU-trained model on CPU must not OOM or segfault."""
        model = _make_model(self.input_dim, self.out_dim,
                            self.tree_struct, self.optimizer, 'cuda')
        model.set_bias_from_targets(self.y)
        _train_steps(model, self.X, self.y, 20, device='cuda')

        path = os.path.join(self.test_dir, 'gpu_save_cpu_load')
        model.save_learner(path)

        loaded = GBTModel.load_learner(path, device='cpu')
        y_pred = loaded(self.X.cpu(), requires_grad=False, tensor=False)
        loss = np.sqrt(np.mean((y_pred.squeeze() - self.y.squeeze()) ** 2))
        self.assertLess(loss, 100.0, f"Unexpected loss after cross-device load: {loss}")

    # ------------------------------------------------------------------
    # 2. GPU-trained → load on CPU → capacity invariant
    # ------------------------------------------------------------------
    @unittest.skipIf(not cuda_available(), "cuda not available")
    def test_gpu_save_cpu_load_capacity_invariant(self):
        """After loading a GPU model on CPU, max_trees must equal n_trees."""
        model = _make_model(self.input_dim, self.out_dim,
                            self.tree_struct, self.optimizer, 'cuda')
        model.set_bias_from_targets(self.y)
        _train_steps(model, self.X, self.y, 20, device='cuda')

        path = os.path.join(self.test_dir, 'gpu_save_cap_check')
        model.save_learner(path)

        loaded = GBTModel.load_learner(path, device='cpu')
        self._assert_exact_capacity(loaded, "GPU→CPU load")

    # ------------------------------------------------------------------
    # 3. GPU-trained → load on CPU → continue training on CPU
    # ------------------------------------------------------------------
    @unittest.skipIf(not cuda_available(), "cuda not available")
    def test_gpu_save_cpu_load_continue_training(self):
        """GPU-trained model loaded on CPU can continue training without crash."""
        model = _make_model(self.input_dim, self.out_dim,
                            self.tree_struct, self.optimizer, 'cuda')
        model.set_bias_from_targets(self.y)
        _train_steps(model, self.X, self.y, 20, device='cuda')

        path = os.path.join(self.test_dir, 'gpu_save_cpu_continue')
        model.save_learner(path)

        loaded = GBTModel.load_learner(path, device='cpu')
        loss = _train_steps(loaded, self.X, self.y, 30, device='cpu')
        self.assertLess(loss, 20.0,
                        f"Loss after cross-device load + CPU training: {loss}")

    # ------------------------------------------------------------------
    # 4. CPU-trained → load on GPU → capacity invariant
    # ------------------------------------------------------------------
    @unittest.skipIf(not cuda_available(), "cuda not available")
    def test_cpu_save_gpu_load_capacity_invariant(self):
        """After loading a CPU model on GPU, max_trees must equal n_trees."""
        model = _make_model(self.input_dim, self.out_dim,
                            self.tree_struct, self.optimizer, 'cpu')
        model.set_bias_from_targets(self.y)
        _train_steps(model, self.X, self.y, 20, device='cpu')

        path = os.path.join(self.test_dir, 'cpu_save_gpu_cap_check')
        model.save_learner(path)

        loaded = GBTModel.load_learner(path, device='cuda')
        self._assert_exact_capacity(loaded, "CPU→GPU load")

    # ------------------------------------------------------------------
    # 5. CPU-trained → load on GPU → continue training on GPU
    # ------------------------------------------------------------------
    @unittest.skipIf(not cuda_available(), "cuda not available")
    def test_cpu_save_gpu_load_continue_training(self):
        """CPU-trained model loaded on GPU can continue training without crash."""
        model = _make_model(self.input_dim, self.out_dim,
                            self.tree_struct, self.optimizer, 'cpu')
        model.set_bias_from_targets(self.y)
        _train_steps(model, self.X, self.y, 20, device='cpu')

        path = os.path.join(self.test_dir, 'cpu_save_gpu_continue')
        model.save_learner(path)

        loaded = GBTModel.load_learner(path, device='cuda')
        loss = _train_steps(loaded, self.X, self.y, 30, device='cuda')
        self.assertLess(loss, 20.0,
                        f"Loss after CPU load → GPU training: {loss}")

    # ------------------------------------------------------------------
    # 6. GPU-trained → load on CPU → set_device GPU → continue training
    # ------------------------------------------------------------------
    @unittest.skipIf(not cuda_available(), "cuda not available")
    def test_gpu_save_cpu_load_move_to_gpu_continue(self):
        """Load GPU model on CPU, move to GPU with set_device, continue training."""
        model = _make_model(self.input_dim, self.out_dim,
                            self.tree_struct, self.optimizer, 'cuda')
        model.set_bias_from_targets(self.y)
        _train_steps(model, self.X, self.y, 20, device='cuda')

        path = os.path.join(self.test_dir, 'gpu_save_cpu_move_gpu')
        model.save_learner(path)

        loaded = GBTModel.load_learner(path, device='cpu')
        # Explicit device move after load
        loaded.set_device('cuda')

        # Capacity must still be exact after to_device
        self._assert_exact_capacity(loaded, "GPU→CPU load→to_device(GPU)")

        loss = _train_steps(loaded, self.X, self.y, 30, device='cuda')
        self.assertLess(loss, 20.0,
                        f"Loss after load→CPU→GPU→train: {loss}")

    # ------------------------------------------------------------------
    # 7. Capacity expands on first new tree, not to default constants
    # ------------------------------------------------------------------
    def test_cpu_load_capacity_expands_correctly(self):
        """After load on CPU, the first training step expands by batch size,
        not by resetting to INITIAL_MAX_TREES."""
        model = _make_model(self.input_dim, self.out_dim,
                            self.tree_struct, self.optimizer, 'cpu')
        model.set_bias_from_targets(self.y)
        _train_steps(model, self.X, self.y, 10, device='cpu')

        path = os.path.join(self.test_dir, 'cpu_capacity_expand')
        model.save_learner(path)

        loaded = GBTModel.load_learner(path, device='cpu')
        md_before = loaded.learner._cpp_model.get_metadata()
        n_trees_before = md_before['n_trees']

        # One more training step triggers allocate_ensemble_memory.
        _train_steps(loaded, self.X, self.y, 1, device='cpu')

        md_after = loaded.learner._cpp_model.get_metadata()
        # max_trees must have grown beyond the loaded n_trees (expansion fired).
        self.assertGreater(md_after['max_trees'], n_trees_before,
                           "max_trees should have grown after the first new tree")
        # max_trees must NOT have jumped to INITIAL_MAX_TREES if n_trees was
        # already above it — it must be n_trees_before + batch_size.
        expected_max = n_trees_before + md_before['max_trees_batch']
        self.assertEqual(md_after['max_trees'], expected_max,
                         f"Expected max_trees={expected_max}, got {md_after['max_trees']}")

    # ------------------------------------------------------------------
    # 8. Predictions are identical before and after save/load on same device
    # ------------------------------------------------------------------
    def test_cpu_load_predictions_match(self):
        """Loaded CPU model produces identical predictions to the saved one."""
        model = _make_model(self.input_dim, self.out_dim,
                            self.tree_struct, self.optimizer, 'cpu')
        model.set_bias_from_targets(self.y)
        _train_steps(model, self.X, self.y, 20, device='cpu')

        path = os.path.join(self.test_dir, 'cpu_pred_match')
        model.save_learner(path)

        y_orig = model(self.X, requires_grad=False, tensor=False)

        loaded = GBTModel.load_learner(path, device='cpu')
        y_loaded = loaded(self.X, requires_grad=False, tensor=False)

        self.assertTrue(np.allclose(y_orig, y_loaded),
                        "Predictions changed after CPU save/load")

    @unittest.skipIf(not cuda_available(), "cuda not available")
    def test_gpu_load_predictions_match(self):
        """Loaded GPU model produces identical predictions to the saved one."""
        model = _make_model(self.input_dim, self.out_dim,
                            self.tree_struct, self.optimizer, 'cuda')
        model.set_bias_from_targets(self.y)
        _train_steps(model, self.X, self.y, 20, device='cuda')

        path = os.path.join(self.test_dir, 'gpu_pred_match')
        model.save_learner(path)

        y_orig = model(self.X.cuda(), requires_grad=False, tensor=False)

        loaded = GBTModel.load_learner(path, device='cuda')
        y_loaded = loaded(self.X.cuda(), requires_grad=False, tensor=False)

        self.assertTrue(np.allclose(y_orig, y_loaded),
                        "Predictions changed after GPU save/load")

    @unittest.skipIf(not cuda_available(), "cuda not available")
    def test_gpu_save_cpu_load_predictions_match(self):
        """GPU-trained model loaded on CPU produces same predictions."""
        model = _make_model(self.input_dim, self.out_dim,
                            self.tree_struct, self.optimizer, 'cuda')
        model.set_bias_from_targets(self.y)
        _train_steps(model, self.X, self.y, 20, device='cuda')

        path = os.path.join(self.test_dir, 'gpu_save_cpu_pred')
        model.save_learner(path)

        # Reference: move GPU model to CPU
        model.set_device('cpu')
        y_ref = model(self.X.cpu(), requires_grad=False, tensor=False)

        loaded = GBTModel.load_learner(path, device='cpu')
        y_loaded = loaded(self.X.cpu(), requires_grad=False, tensor=False)

        self.assertTrue(np.allclose(y_ref, y_loaded, atol=1e-5),
                        "Cross-device load changed predictions")


if __name__ == '__main__':
    unittest.main()
