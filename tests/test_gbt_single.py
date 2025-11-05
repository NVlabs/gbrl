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
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import shap
import torch as th
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from torch.nn.functional import mse_loss

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

from gbrl import cuda_available
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
        gbrl_shap = model.tree_shap(0, X_cpu[0, :])[0].flatten()
        clf = DecisionTreeRegressor(max_depth=3).fit(X_cpu, y)

        target_shap = shap.TreeExplainer(clf).shap_values(X_cpu[0])
        self.assertTrue(np.allclose(gbrl_shap, target_shap, rtol=1e-3),
                        'GBRL SHAP values are not close to target SHAP values')

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


if __name__ == '__main__':
    unittest.main()
