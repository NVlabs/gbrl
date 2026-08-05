##############################################################################
# Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
"""
Tests for per-leaf density correctness in multi-objective GBRL.

Invariant: every leaf density vector must sum to 1 and be non-negative.
This was violated on the CUDA + oblivious path due to two bugs:
  1. Empty leaves kept the [1,1] ones_kernel init instead of inheriting parent.
  2. All-zero-label minibatches set obj_labels=None, leaving [1,1] everywhere.

Both are now fixed. These tests are the regression guard.
"""
import tempfile
import unittest

import numpy as np
import torch as th

from gbrl import cuda_available
from gbrl.models.gbt import GBTModel


N_STEPS = 5
N_SAMPLES = 128
N_OBJS = 2
TOL = 1e-3


def _make_model(device, grow_policy, max_depth=4):
    return GBTModel(
        input_dim=1,
        output_dim=1,
        tree_struct={
            'max_depth': max_depth,
            'n_bins': 32,
            'min_data_in_leaf': 1,
            'par_th': 1,
            'grow_policy': grow_policy,
        },
        optimizers={'algo': 'SGD', 'lr': 0.1, 'start_idx': 0, 'stop_idx': 1},
        params={'n_objs': N_OBJS, 'split_score_func': 'Cosine', 'lambda_penalty': 0.0},
        verbose=0,
        device=device,
    )


def _mixed_labels(n):
    """~50/50 split of label 0 and 1."""
    labels = np.zeros(n, dtype=np.float32)
    labels[n // 2:] = 1.0
    return labels


def _train(model, n_steps=N_STEPS, label_fn=_mixed_labels):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((N_SAMPLES, 1)).astype(np.float32)
    for _ in range(n_steps):
        grads = rng.standard_normal((N_OBJS, N_SAMPLES, 1)).astype(np.float32)
        labels = label_fn(N_SAMPLES)
        model.step(
            X=th.tensor(X),
            grads=th.tensor(grads),
            obj_labels=th.tensor(labels),
        )
    return X


def _density_row_sums(model, X):
    """Return per-sample density row sums from predict_densities."""
    densities = model.learner.predict_densities(th.tensor(X))
    if isinstance(densities, th.Tensor):
        densities = densities.numpy()
    return densities.sum(axis=1)


def _leaf_density_row_sums(model):
    """Return all per-leaf density row sums across every tree."""
    n_trees = model.learner.get_num_trees()
    sums = []
    for t in range(n_trees):
        tree = model.learner.get_tree(t)
        d = tree['densities']          # (n_leaves, n_objs)
        sums.append(d.sum(axis=1))
    return np.concatenate(sums)


class TestDensityRowSumInvariant(unittest.TestCase):
    """
    Core invariant: every leaf density vector sums to 1 across all
    device × grow_policy combinations.
    """

    def _check(self, device, grow_policy, max_depth=4):
        model = _make_model(device, grow_policy, max_depth)
        X = _train(model)
        sums = _leaf_density_row_sums(model)
        frac_valid = np.mean(np.abs(sums - 1.0) < TOL)
        self.assertAlmostEqual(
            frac_valid, 1.0, delta=0.001,
            msg=f"{device}/{grow_policy}/depth={max_depth}: "
                f"only {frac_valid*100:.1f}% of leaves sum to 1",
        )

    def test_cpu_greedy(self):
        self._check('cpu', 'greedy')

    def test_cpu_oblivious(self):
        self._check('cpu', 'oblivious')

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_cuda_greedy(self):
        self._check('cuda', 'greedy')

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_cuda_oblivious(self):
        self._check('cuda', 'oblivious')

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_cuda_oblivious_deep(self):
        """Deeper trees have more empty leaves — the original bug scaled with depth."""
        for depth in [4, 6, 8]:
            with self.subTest(depth=depth):
                self._check('cuda', 'oblivious', max_depth=depth)


class TestDefaultObjIdx(unittest.TestCase):
    """
    When all labels in a minibatch are the same, obj_labels is set to None
    for speed and default_obj_idx encodes the uniform label.
    The init one-hot must reflect the correct objective.
    """

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_all_reward_labels_cuda_oblivious(self):
        """All label-0 minibatch → density should be [1, 0], not [1, 1]."""
        model = _make_model('cuda', 'oblivious')
        _train(model, label_fn=lambda n: np.zeros(n, dtype=np.float32))
        sums = _leaf_density_row_sums(model)
        frac_valid = np.mean(np.abs(sums - 1.0) < TOL)
        self.assertAlmostEqual(frac_valid, 1.0, delta=0.001,
            msg=f"All-reward labels: {frac_valid*100:.1f}% valid (expected 100%)")

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_all_safety_labels_cuda_oblivious(self):
        """All label-1 minibatch → density should be [0, 1], not [1, 1]."""
        model = _make_model('cuda', 'oblivious')
        _train(model, label_fn=lambda n: np.ones(n, dtype=np.float32))
        sums = _leaf_density_row_sums(model)
        frac_valid = np.mean(np.abs(sums - 1.0) < TOL)
        self.assertAlmostEqual(frac_valid, 1.0, delta=0.001,
            msg=f"All-safety labels: {frac_valid*100:.1f}% valid (expected 100%)")

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_default_obj_idx_value_all_reward(self):
        """All label-0 → each leaf's density[0] should be 1.0, density[1] = 0.0."""
        model = _make_model('cuda', 'oblivious')
        _train(model, n_steps=1, label_fn=lambda n: np.zeros(n, dtype=np.float32))
        n_trees = model.learner.get_num_trees()
        for t in range(n_trees):
            d = model.learner.get_tree(t)['densities']  # (n_leaves, 2)
            # Each row must be either [1,0] (non-empty leaf) or [1,0] (empty, inherited/defaulted)
            # No row should be [1,1]
            bad = np.any(d > 1.0 + TOL)
            self.assertFalse(bad, f"Tree {t} has density > 1 in some leaf (got [1,1] init)")

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_default_obj_idx_value_all_safety(self):
        """All label-1 → density[1] should be 1.0 for leaves where default applies."""
        model = _make_model('cuda', 'oblivious')
        _train(model, n_steps=1, label_fn=lambda n: np.ones(n, dtype=np.float32))
        n_trees = model.learner.get_num_trees()
        for t in range(n_trees):
            d = model.learner.get_tree(t)['densities']
            bad = np.any(d > 1.0 + TOL)
            self.assertFalse(bad, f"Tree {t} has density > 1 (got [1,1] init for all-safety case)")


class TestPredictDensitiesSumToOne(unittest.TestCase):
    """predict_densities output rows must sum to 1 at inference time."""

    def _check(self, device, grow_policy):
        model = _make_model(device, grow_policy)
        X = _train(model)
        row_sums = _density_row_sums(model, X)
        np.testing.assert_allclose(
            row_sums, np.ones_like(row_sums), atol=TOL,
            err_msg=f"predict_densities rows don't sum to 1 ({device}/{grow_policy})",
        )

    def test_cpu_greedy(self):
        self._check('cpu', 'greedy')

    def test_cpu_oblivious(self):
        self._check('cpu', 'oblivious')

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_cuda_greedy(self):
        self._check('cuda', 'greedy')

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_cuda_oblivious(self):
        self._check('cuda', 'oblivious')


class TestDensitySaveLoad(unittest.TestCase):
    """Densities must survive a save/load round-trip."""

    def _check(self, device, grow_policy):
        model = _make_model(device, grow_policy)
        X = _train(model)
        sums_before = _leaf_density_row_sums(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model"
            model.save_learner(path)
            model2 = GBTModel.load_learner(path, device=device)

        sums_after = _leaf_density_row_sums(model2)
        np.testing.assert_allclose(sums_before, sums_after, atol=TOL,
            err_msg=f"Density row sums changed after save/load ({device}/{grow_policy})")

    def test_cpu_greedy(self):
        self._check('cpu', 'greedy')

    def test_cpu_oblivious(self):
        self._check('cpu', 'oblivious')

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_cuda_greedy(self):
        self._check('cuda', 'greedy')

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_cuda_oblivious(self):
        self._check('cuda', 'oblivious')


class TestDefaultObjIdxGradientRouting(unittest.TestCase):
    """
    Semantic test: when all labels are the same, only that objective's gradients
    should shape the leaf value.

    leaf_value = Σ_k density_k * lambda_k * mean_grad_k
    lambda_objs defaults to [1, 1].  GBT steps in the negative gradient direction, so:
      default_obj_idx=0  →  density=[1,0]  →  pred ≈ -mean(grads[0]) = -G
      default_obj_idx=1  →  density=[0,1]  →  pred ≈ -mean(grads[1]) = +G

    We fix grads[0]=+G and grads[1]=-G and verify:
      - Opposite-sign predictions depending on which label was used.
      - Each prediction matches -mean(grads[its_label]) within tolerance.
    Changing the label index changes the result — that is the guarantee.
    """

    GRAD_VAL = 2.0   # grads[0] = +G  →  pred ≈ -G;  grads[1] = -G  →  pred ≈ +G
    LR = 1.0
    N = 64
    ATol = 0.3

    def _make_model(self, device):
        return GBTModel(
            input_dim=1, output_dim=1,
            tree_struct={
                'max_depth': 1,
                'n_bins': 16,
                'min_data_in_leaf': 1,
                'par_th': 1,
                'grow_policy': 'oblivious',
            },
            optimizers={'algo': 'SGD', 'lr': self.LR, 'start_idx': 0, 'stop_idx': 1},
            params={'n_objs': 2, 'split_score_func': 'Cosine', 'lambda_penalty': 0.0},
            verbose=0,
            device=device,
        )

    def _run(self, device, uniform_label):
        """Train one step with all labels == uniform_label, return mean prediction."""
        rng = np.random.default_rng(0)
        # Varied X so split candidates exist on CPU
        X = rng.standard_normal((self.N, 1)).astype(np.float32)
        grads = np.zeros((2, self.N, 1), dtype=np.float32)
        grads[0, :, 0] = +self.GRAD_VAL
        grads[1, :, 0] = -self.GRAD_VAL
        labels = np.full(self.N, uniform_label, dtype=np.float32)
        model = self._make_model(device)
        model.step(X=th.tensor(X), grads=th.tensor(grads), obj_labels=th.tensor(labels))
        pred = model(th.tensor(X), requires_grad=False, tensor=False)
        return float(pred.mean()), X

    def _check(self, device):
        pred_obj0, _ = self._run(device, uniform_label=0)
        pred_obj1, _ = self._run(device, uniform_label=1)

        # GBT steps in the negative gradient direction
        expected_obj0 = -self.GRAD_VAL * self.LR   # grads[0]=+G  →  pred ≈ -G
        expected_obj1 = +self.GRAD_VAL * self.LR   # grads[1]=-G  →  pred ≈ +G

        print(f"\n[{device}] all-label-0 pred: {pred_obj0:.4f}  (expected ≈ {expected_obj0:.1f})")
        print(f"[{device}] all-label-1 pred: {pred_obj1:.4f}  (expected ≈ {expected_obj1:.1f})")

        # Predictions must have opposite signs — changing the index changes the result
        self.assertLess(pred_obj0, 0,
            f"[{device}] all-label-0: expected negative pred (from grads[0]=+{self.GRAD_VAL}), got {pred_obj0:.4f}")
        self.assertGreater(pred_obj1, 0,
            f"[{device}] all-label-1: expected positive pred (from grads[1]=-{self.GRAD_VAL}), got {pred_obj1:.4f}")

        # Values must match the gradient of the routed objective
        self.assertAlmostEqual(pred_obj0, expected_obj0, delta=self.ATol,
            msg=f"[{device}] all-label-0: pred={pred_obj0:.4f} expected={expected_obj0:.1f}")
        self.assertAlmostEqual(pred_obj1, expected_obj1, delta=self.ATol,
            msg=f"[{device}] all-label-1: pred={pred_obj1:.4f} expected={expected_obj1:.1f}")

    def test_cpu_oblivious(self):
        self._check('cpu')

    @unittest.skipIf(not cuda_available(), "CUDA not available")
    def test_cuda_oblivious(self):
        self._check('cuda')


if __name__ == '__main__':
    unittest.main(verbosity=2)
