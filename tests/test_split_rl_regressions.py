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
"""Regressions for Split-RL checkpoint, export, and model contracts."""
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

import numpy as np
import torch as th

from gbrl import cuda_available
from gbrl.learners.gbt_learner import GBTLearner
from gbrl.learners.multi_gbt_learner import MultiGBTLearner
from gbrl.models.actor import GaussianActor
from gbrl.models import ActorCritic, CostActorCritic
from gbrl.models.gbt import GBTModel


TREE = dict(max_depth=2, n_bins=8, min_data_in_leaf=1, par_th=100,
            grow_policy='oblivious')
OPT = dict(algo='SGD', init_lr=1.0, start_idx=0, stop_idx=1)
X = np.arange(8, dtype=np.float32).reshape(-1, 1)
LABELS = np.array([0, 1] * 4, dtype=np.float32)
GRADS = np.ones((2, 8, 1), dtype=np.float32)
GRADS[1] *= 3


def model(width=1, **params):
    return GBTModel(TREE, 1, width, dict(OPT, stop_idx=width), params=params)


class TestSplitRLContracts(unittest.TestCase):
    def test_imported_tree_validation_is_atomic(self):
        for policy in ('greedy', 'oblivious'):
            source = GBTModel(dict(TREE, grow_policy=policy), 1, 1, OPT,
                              params=dict(n_objs=2, lambda_penalty=0))
            source.step(X, np.stack((X + 1, 2 * X + 1)), obj_labels=LABELS)
            tree = source.get_tree(0)
            self.assertGreater(tree['tree_depth'], 0)
            target = GBTModel(dict(TREE, grow_policy=policy), 1, 1, OPT,
                              params=dict(n_objs=2, lambda_penalty=0))
            expected = target(X, tensor=False).copy()
            cases = [
                ('depths', 0, -1, 'depths'),
                ('depths', 0, TREE['max_depth'] + 1, 'depths'),
                ('feature_indices', 0, -1, 'feature index'),
                ('feature_indices', 0, 1, 'feature index'),
                ('densities', 0, np.nan, 'densities'),
                ('densities', 0, np.inf, 'densities'),
                ('densities', 0, -0.1, 'densities'),
                ('densities', 0, 1.1, 'densities'),
            ]
            for field, index, value, message in cases:
                invalid = {k: v.copy() if isinstance(v, np.ndarray) else v
                           for k, v in tree.items()}
                invalid[field].flat[index] = value
                with self.subTest(policy=policy, field=field, value=value):
                    with self.assertRaisesRegex(RuntimeError, message):
                        target.add_tree(invalid)
                    self.assertEqual(target.get_num_trees(), 0)
                    self.assertEqual(target.get_iteration(), 0)
                    np.testing.assert_array_equal(target(X, tensor=False), expected)
            invalid = {k: v.copy() if isinstance(v, np.ndarray) else v
                       for k, v in tree.items()}
            invalid['densities'][0] = 0
            with self.assertRaisesRegex(RuntimeError, 'sum to one'):
                target.add_tree(invalid)
            invalid = {k: v.copy() if isinstance(v, np.ndarray) else v
                       for k, v in tree.items()}
            if policy == 'oblivious':
                invalid['tree_depth'] -= 1
                invalid['depths'][0] = invalid['tree_depth']
                message = 'leaf count'
            else:
                invalid['tree_depth'] -= 1
                message = 'tree_depth'
            with self.assertRaisesRegex(RuntimeError, message):
                target.add_tree(invalid)
            target.add_tree(tree)
            np.testing.assert_allclose(target(X, tensor=False), source(X, tensor=False))
            np.testing.assert_allclose(target.learner.predict_densities(X), source.learner.predict_densities(X))

    def test_imported_tree_checks_only_active_features_and_accepts_stumps(self):
        for policy in ('greedy', 'oblivious'):
            for depth in (0, 3):
                tree_struct = dict(TREE, grow_policy=policy, max_depth=depth, min_data_in_leaf=4)
                source = GBTModel(tree_struct, 1, 1, OPT, params=dict(n_objs=2))
                source.step(X, np.stack((X + 1, 2 * X + 1)), obj_labels=LABELS)
                tree = source.get_tree(0)
                for row, actual_depth in enumerate(tree['depths']):
                    tree['feature_indices'][row, actual_depth:] = -999
                target = GBTModel(tree_struct, 1, 1, OPT, params=dict(n_objs=2))
                target(X, tensor=False)
                target.add_tree(tree)
                np.testing.assert_allclose(target(X, tensor=False), source(X, tensor=False))

    def test_legacy_checkpoint_migration(self):
        checkpoint = Path(__file__).parent / 'fixtures' / 'gbrl_1_1_9.gbrl_model'
        learner = GBTLearner.load(str(checkpoint), 'cpu')
        expected = np.array([-1.5] * 4 + [-5.5] * 4, dtype=np.float32)
        np.testing.assert_allclose(learner.predict(X, tensor=False), expected)
        np.testing.assert_allclose(learner.predict_densities(X), 1)
        learner.step(X, np.ones((8, 1), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'migrated')
            learner.save(filename)
            loaded = GBTLearner.load(filename, 'cpu')
            np.testing.assert_allclose(loaded.predict(X, tensor=False), expected - 1)
            loaded.copy().reset()

    def test_legacy_layout_boundaries(self):
        checkpoint = Path(__file__).parent / 'fixtures' / 'gbrl_1_1_9.gbrl_model'
        original = bytearray(checkpoint.read_bytes())
        name_size = struct.unpack_from('<Q', original, 110)[0]
        with tempfile.TemporaryDirectory() as directory:
            for patch in range(2, 9):
                data = original.copy()
                struct.pack_into('<H', data, 4, patch)
                if patch < 7:
                    del data[170:173]
                    if patch < 6:
                        del data[110:118 + name_size]
                    del data[104:108]
                filename = Path(directory) / 'legacy.gbrl_model'
                filename.write_bytes(data)
                with self.subTest(version=f'1.1.{patch}'):
                    learner = GBTLearner.load(str(filename), 'cpu')
                    np.testing.assert_allclose(learner.predict(X, tensor=False),
                                               [-1.5] * 4 + [-5.5] * 4)

    def test_checkpoint_rejects_invalid_header_counts_and_truncation(self):
        m = model(n_objs=2)
        m.step(X, GRADS, obj_labels=LABELS)
        with tempfile.TemporaryDirectory() as directory:
            name = str(Path(directory) / 'model')
            m.save_learner(name)
            data = bytearray(Path(name + '.gbrl_model').read_bytes())
            bad_version = data.copy()
            struct.pack_into('<H', bad_version, 12, 99)
            bad_count = data.copy()
            struct.pack_into('<i', bad_count, 24, 2**30)
            bad_objectives = data.copy()
            struct.pack_into('<i', bad_objectives, 24 + 13 * 4, 2**30)
            for damaged in (bad_version, bad_count, bad_objectives, data[:30], data[:-20]):
                Path(name + '.gbrl_model').write_bytes(damaged)
                with self.assertRaises(RuntimeError):
                    GBTLearner.load(name, 'cpu')

    def test_label_validation_precedes_uniform_optimization(self):
        m = model(n_objs=2)
        for labels in (np.full(8, -1), np.full(8, 2), np.ones(1),
                       np.array([0.5, 1] * 4), np.full(8, np.nan),
                       np.ones((2, 4))):
            for backend in (np.asarray, th.tensor):
                with self.subTest(labels=labels, backend=backend), self.assertRaises(ValueError):
                    m.step(X, GRADS, obj_labels=backend(labels))
        m.step(X, GRADS, obj_labels=np.ones(8, dtype=np.float32))
        np.testing.assert_allclose(m.learner.predict_densities(X), np.tile([0, 1], (8, 1)))

    def test_copy_preserves_weights_and_independent_state(self):
        m = model(n_objs=2, lambda_objs=[1, 4])
        m.step(X, GRADS, obj_labels=LABELS)
        c = m.copy()
        np.testing.assert_allclose(c(X, tensor=False), m(X, tensor=False))
        c.learner.reset()
        np.testing.assert_allclose(c.learner._cpp_model.get_lambda_objs(), [1, 4])
        self.assertEqual(m.learner.get_num_trees(), 1)
        c.step(X, GRADS, obj_labels=LABELS)
        with tempfile.TemporaryDirectory() as directory:
            name = str(Path(directory) / 'copy')
            m.save_learner(name)
            loaded = GBTLearner.load(name, 'cpu').copy()
            loaded.reset()
            np.testing.assert_allclose(loaded._cpp_model.get_lambda_objs(), [1, 4])

    def test_actor_critic_gradients_and_copy(self):
        for cls, width in ((ActorCritic, 2), (CostActorCritic, 3)):
            for shared in (False, True):
                for backend in ('numpy', 'torch'):
                    for as_tuple in (False, True):
                        with self.subTest(cls=cls, shared=shared, backend=backend, as_tuple=as_tuple):
                            opts = [dict(OPT, start_idx=i, stop_idx=i + 1) if shared else OPT
                                    for i in range(width)]
                            m = cls(TREE, 1, width, *opts, shared_tree_struct=shared,
                                    params=dict(n_objs=2, lambda_objs=[1, 4]))
                            x, policy, critic, labels = X, GRADS.copy(), np.ones(8, np.float32), LABELS
                            if backend == 'torch':
                                x, policy, critic, labels = map(th.tensor, (x, policy, critic, labels))
                            if as_tuple:
                                policy = tuple(policy)
                            args = [x, policy] + [critic] * (width - 1)
                            m.step(*args, obj_labels=labels)
                            self.assertEqual(len(m.get_grads()), width)
                            for gradient in m.get_grads():
                                self.assertIsNotNone(gradient)
                            c = m.copy()
                            for actual, expected in zip(c(x, tensor=False), m(x, tensor=False)):
                                np.testing.assert_allclose(actual, expected)
                            if not shared:
                                self.assertIsNot(c.learner.iteration, m.learner.iteration)
                                c.critic_step(x, critic)
                                if width == 3:
                                    c.cost_critic_step(x, critic)
                                    self.assertIsNotNone(c.get_grads()[2])
                            c.learner.reset()
                            native = c.learner._cpp_model if shared else c.learner._cpp_models[0]
                            np.testing.assert_allclose(native.get_lambda_objs(), [1, 4])

    def test_gaussian_tuple_clipping(self):
        for backend in (np.asarray, th.tensor):
            for fixed in (False, True):
                width = 1 if fixed else 2
                std_opt = None if fixed else dict(OPT, start_idx=1, stop_idx=2)
                m = GaussianActor(TREE, 1, width, OPT, std_opt, params=dict(n_objs=2))
                g = tuple(backend(np.full((8, 1), v, np.float32)) for v in (2, 3))
                m.step(backend(X), g, None if fixed else g,
                       mu_grad_clip=1, log_std_grad_clip=1, obj_labels=backend(LABELS))
                self.assertEqual(m.learner.get_num_trees(), 1)

    def test_generic_multi_learner_objectives_survive_copy_load_reset(self):
        learner = MultiGBTLearner(1, [1, 1], TREE, [dict(OPT, init_lr=1.0), dict(OPT, init_lr=1.0)],
                                  dict(n_objs=2, lambda_objs=[1, 4]), n_learners=2)
        learner.reset()
        learner.step(X, (GRADS, GRADS), obj_idx=1, obj_labels=LABELS)
        for native in learner._cpp_models:
            self.assertEqual(native.get_metadata()['n_objs'], 2)
        c = learner.copy()
        self.assertIsNot(c.iteration, learner.iteration)
        with tempfile.TemporaryDirectory() as directory:
            name = str(Path(directory) / 'multi')
            c.save(name)
            loaded = MultiGBTLearner.load(name, 'cpu')
            loaded.reset()
            self.assertEqual(loaded.n_objs_per_model, [2, 2])
            loaded.step(X, GRADS, model_idx=1, obj_labels=LABELS)
            np.testing.assert_allclose(loaded._cpp_models[1].get_lambda_objs(), [1, 4])

    def test_density_stumps_and_ranges(self):
        for policy in ('greedy', 'oblivious'):
            for depth, observations in ((0, X), (2, np.zeros_like(X))):
                m = GBTModel(dict(TREE, grow_policy=policy, max_depth=depth),
                             1, 1, OPT, params=dict(n_objs=2))
                np.testing.assert_allclose(m.learner.predict_densities(observations), 0)
                m.step(observations, GRADS, obj_labels=LABELS)
                np.testing.assert_allclose(m(observations, tensor=False), -2)
                np.testing.assert_allclose(m.learner.predict_densities(observations), 0.5)
                np.testing.assert_allclose(m.learner.predict_densities(observations, start_idx=1), 0)
                for start, stop in ((-1, None), (0, -1), (2, None), (0, 2), (0.5, None)):
                    with self.assertRaises(ValueError):
                        m.learner.predict_densities(observations, start, stop)
                with self.assertRaises(RuntimeError):
                    m.learner._cpp_model.predict_densities(observations, None, -1, 0)

    def test_packed_export_matches_native_and_c_header(self):
        m = GBTModel(dict(TREE, max_depth=3), 1, 2, OPT)
        m.step(np.array([[0], [1]], np.float32), np.array([[-1, 100], [1, 100]], np.float32))
        m.step(X, np.concatenate((np.sin(X), np.full_like(X, 100)), axis=1))
        exported = m.get_export_data()
        self.assertEqual(exported['tree_depths'].tolist(), [1, 1])
        result = np.tile(exported['bias'], (len(X), 1))
        split_offset, leaf_offset = 0, 0
        for depth, count in zip(exported['tree_depths'], exported['tree_leaf_counts']):
            indices = np.zeros(len(X), dtype=np.int32)
            for d in range(depth):
                idx = split_offset + d
                indices = 2 * indices + (X[:, exported['feature_indices'][idx]] > exported['feature_values'][idx])
            result += exported['leaf_values'][leaf_offset + indices]
            split_offset += depth
            leaf_offset += count
        np.testing.assert_allclose(result, m(X, tensor=False).reshape(8, 2))
        np.testing.assert_allclose(exported['leaf_values'][:, 1], 0)
        with tempfile.TemporaryDirectory() as directory:
            name = str(Path(directory) / 'export')
            m.export_learner(name)
            with self.assertRaises(RuntimeError):
                m.export_learner(name + '_compact', export_type='compact')
            if shutil.which('c++'):
                source = Path(directory) / 'main.cpp'
                source.write_text('#include <cstdio>\n#include <cstdint>\n'
                                  'typedef uint8_t uint8; typedef uint16_t uint16;\n'
                                  'typedef int16_t int16; typedef int32_t int32;\n'
                                  '#include "export.h"\n'
                                  'int main(){for(int i=0;i<8;i++){float x[]={float(i)},y[2]={123,456};'
                                  'gbrl_predict(y,x);printf("%.8g %.8g\\n",y[0],y[1]);}}')
                subprocess.run(['c++', str(source), '-o', name], check=True, capture_output=True)
                output = subprocess.check_output([name], text=True)
                np.testing.assert_allclose(np.loadtxt(output.splitlines()), result, rtol=1e-5, atol=1e-6)

    def test_export_rejects_unsupported_models(self):
        for algo, observations in (('Adam', X), ('SGD', np.array([['a'], ['b']] * 4))):
            m = GBTModel(TREE, 1, 1, dict(OPT, algo=algo))
            m.step(observations, np.ones((8, 1), np.float32))
            with self.assertRaises(RuntimeError):
                m.get_export_data()
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(RuntimeError):
                m.export_learner(str(Path(directory) / 'unsupported'))

    def test_cosine_split_matches_numerical_reference(self):
        rng = np.random.default_rng(11)
        gradients = rng.normal(size=(2, 8, 3)).astype(np.float32) + 0.4
        scores = []
        for cut in range(7):
            left, right = gradients[:, :cut + 1], gradients[:, cut + 1:]
            gain = np.sqrt((left.sum(axis=1)**2).sum(axis=1) / left.shape[1] +
                           (right.sum(axis=1)**2).sum(axis=1) / right.shape[1])
            scores.append(gain.mean())
        def expected_leaves(grads, left_mask):
            # Leaf value = -lr * sum_k density_k * lambda_k * mean_k (lr = lambda = 1).
            out = np.zeros((len(left_mask), grads.shape[2]), np.float32)
            for mask in (left_mask, ~left_mask):
                labels = LABELS[mask].astype(int)
                value = sum((labels == k).mean() * grads[k, mask].mean(axis=0) for k in range(2))
                out[mask] = -value
            return out

        for policy in ('greedy', 'oblivious'):
            with self.subTest(policy=policy):
                m = GBTModel(dict(TREE, max_depth=1, grow_policy=policy, n_bins=8),
                             1, 3, dict(OPT, stop_idx=3),
                             params=dict(n_objs=2, lambda_penalty=0, generator_type='Uniform'))
                m.step(X, gradients, obj_labels=LABELS)
                threshold = m.get_tree(0)['feature_values'][0, 0]
                chosen, expected = int(np.floor(threshold)), int(np.argmax(scores))
                if chosen != expected:
                    # Diagnostics for platform-specific failures: does the model's
                    # leaf data match the true gradients or a reversed pairing?
                    predictions = m(X, requires_grad=False, tensor=False)
                    left = X[:, 0] <= threshold
                    detail = (f"{policy}: threshold={threshold} (cut {chosen}, expected {expected}); "
                              f"left samples={np.flatnonzero(left).tolist()}; "
                              f"leaf values match true grads="
                              f"{np.allclose(predictions, expected_leaves(gradients, left), atol=1e-4)}, "
                              f"reversed grads="
                              f"{np.allclose(predictions, expected_leaves(gradients[:, ::-1], left), atol=1e-4)}; "
                              f"reference scores={np.round(scores, 4).tolist()}")
                    self.fail(detail)


class TestReleaseReviewFixes(unittest.TestCase):
    """Regressions for issues found in the 1.2.0 release review."""

    @staticmethod
    def _devices():
        return ['cpu'] + (['cuda'] if cuda_available() and th.cuda.is_available() else [])

    def test_concatenate_arrays_treats_1d_inputs_as_columns(self):
        from gbrl.common.utils import concatenate_arrays
        a, b = np.zeros(4, np.float32), np.ones(4, np.float32)
        self.assertEqual(concatenate_arrays([a, b]).shape, (4, 2))
        self.assertEqual(concatenate_arrays([th.zeros(4), th.ones(4)]).shape, (4, 2))
        self.assertEqual(concatenate_arrays([np.zeros((4, 3)), b]).shape, (4, 4))
        self.assertEqual(concatenate_arrays([np.zeros((2, 4, 3)), np.zeros((2, 4, 1))]).shape,
                         (2, 4, 4))

    def test_pad_array_appends_n_dims_columns(self):
        from gbrl.common.utils import pad_array
        self.assertEqual(pad_array(np.zeros((4, 3)), 2).shape, (4, 5))
        self.assertEqual(pad_array(np.zeros(4), 2).shape, (4, 3))
        self.assertEqual(pad_array(np.zeros((2, 4, 3)), 2).shape, (2, 4, 5))
        self.assertEqual(tuple(pad_array(th.zeros(4, 3), 1, pad_value=7.0).shape), (4, 4))

    def test_control_variates_rejected_with_multiple_objectives(self):
        with self.assertRaises(ValueError):
            model(n_objs=2, lambda_objs=[1.0, 1.0], control_variates=True)

    def test_fresh_cpp_model_has_unit_objective_weights(self):
        # Internally created models (e.g. a distillation student) never call
        # set_lambda_objs; zero weights would zero every score and policy leaf.
        from gbrl import GBRL_CPP
        for device in self._devices():
            with self.subTest(device=device):
                cpp = GBRL_CPP(input_dim=1, output_dim=1, policy_dim=1,
                               split_score_func='L2', generator_type='Quantile',
                               use_control_variates=False, device=device,
                               max_depth=2, verbose=0, n_objs=3)
                np.testing.assert_array_equal(np.asarray(cpp.get_lambda_objs()),
                                              np.ones(3, np.float32))


@unittest.skipUnless(cuda_available() and th.cuda.is_available(), 'CUDA runtime unavailable')
class TestSplitRLCUDAParity(unittest.TestCase):
    def test_conflict_widths_and_l2_objective_preprocessing(self):
        for width in (1, 2, 3, 5, 1025):
            for score in ('Cosine', 'L2'):
                for policy in ('greedy', 'oblivious'):
                    with self.subTest(width=width, score=score, policy=policy):
                        rng = np.random.default_rng(42)
                        gradients = np.zeros((2, 8, width), np.float32)
                        gradients[0, :, -1] = rng.normal(size=8) + 0.5
                        gradients[1, :, -1] = (-gradients[0, :, -1] if score == 'Cosine'
                                                  else -(gradients[0, :, -1] + 0.4) * 50)
                        predictions, splits = [], []
                        for device in ('cpu', 'cuda'):
                            m = GBTModel(dict(TREE, max_depth=1, grow_policy=policy),
                                         1, width, dict(OPT, stop_idx=width),
                                         params=dict(n_objs=2, lambda_penalty=0.8, split_score_func=score),
                                         device=device)
                            m.step(X, gradients, obj_labels=LABELS)
                            predictions.append(m(X, tensor=False))
                            tree = m.get_tree(0)
                            splits.append((tree['tree_depth'], tree['feature_values'][0, :tree['tree_depth']]))
                        self.assertEqual(splits[0][0], splits[1][0])
                        np.testing.assert_allclose(splits[0][1], splits[1][1], atol=1e-6)
                        np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-4, rtol=1e-5)
