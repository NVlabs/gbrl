##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
import os
import shutil
import tempfile
import unittest
from typing import Union, Tuple

import numpy as np
import torch as th
from sklearn.tree import DecisionTreeRegressor
import shap
from torch.nn.functional import mse_loss

from sklearn import datasets

from gbrl import cuda_available
from gbrl.gbt import GBRL
from gbrl.ac_gbrl import ActorCritic

def rmse(preds: Union[np.array, th.Tensor], targets: Union[np.array, th.Tensor]) -> Tuple[float, np.array]:
    if isinstance(preds, th.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, th.Tensor):
        targets = targets.detach().cpu().numpy()
        
    preds = preds.squeeze()
    targets = targets.squeeze()

    loss = np.sqrt(np.mean((preds - targets)**2))
    return loss, (preds - targets)

N_EPOCHS = 100

def rmse_model(model, X, y, n_epochs, device='cpu'):
    y_ = th.tensor(y, dtype=th.float32, device=device)
    X_ = X.copy()
    epoch = 0
    while epoch < n_epochs:
        y_pred = model(X_, requires_grad=True)
        loss = 0.5*mse_loss(y_pred, y_) * y_.shape[1]
        loss.backward() 
        model.step()
        print(f"epoch: {epoch} loss: {(loss / y_.shape[1]).sqrt()}")
        epoch += 1
    y_pred = model(X_)
    loss = (0.5*mse_loss(y_pred, y_)).sqrt().item()
    return loss

def ac_rmse_model(model, X, y, n_epochs, device='cpu'):
    y_ac = th.tensor(y[:, :-1], dtype=th.float32, device=device)
    y_value = th.tensor(y[:, -1], dtype=th.float32, device=device)
    X_ = X.copy()
    epoch = 0
    while epoch < n_epochs:
        theta, value = model(X_, requires_grad=True)
        loss_theta = 0.5*mse_loss(theta, y_ac) * y_ac.shape[1]
        loss_theta.backward()
        loss_value = 0.5*mse_loss(value, y_value)
        loss_value.backward()
        model.step()
        print(f"epoch: {epoch} loss_theta: {loss_theta.sqrt():.5f} loss_value: {(loss_value).sqrt():.5f}")
        epoch += 1
    theta, value = model(X_)
    loss_theta = (0.5*mse_loss(theta, y_ac)).sqrt().item()
    loss_value = (0.5*mse_loss(value, y_value)).sqrt().item()
    return loss_theta, loss_value

class TestGBTMulti(unittest.TestCase):

    @classmethod 
    def setUpClass(cls):
        print('Loading data...')
        # Imagine this loads your actual data
        try:
            X ,y = datasets.load_diabetes(return_X_y=True, as_frame=False, scaled=False)
        except TypeError: #python3.7 uses older version of scikit-learn
            X ,y = datasets.load_diabetes(return_X_y=True, as_frame=False)
        out_dim = 1 if len(y.shape) == 1  else  y.shape[1]
        if out_dim == 1:
            y = y[:, np.newaxis]
        y_fake = y.copy()
        n_cols = 10
        for _ in range(n_cols - 1):
            y = np.concatenate([y, y_fake], axis= 1)
        out_dim = y.shape[1]
        cls.data = (X, y)
        cls.out_dim = out_dim
        cls.input_dim = X.shape[1]
        cls.n_epochs = 100
        cls.test_dir = tempfile.mkdtemp()
        cls.tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'greedy'}
        cls.sgd_optimizer = { 'algo': 'SGD',
                    'lr': 1.0,
                    'start_idx': 0,
                    'stop_idx': out_dim
                }
    
    @classmethod 
    def tearDownClass(cls):
        # Remove the directory after the test
        shutil.rmtree(cls.test_dir)


    def test_cosine_cpu(self):
        print("Running Multi test_cosine_cpu")
        X, y = self.data
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine"})
        model = GBRL(input_dim=self.input_dim,
                            output_dim=self.out_dim,
                            tree_struct=self.tree_struct,
                            optimizer=self.sgd_optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 2.0
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_cosine_cpu'))

    def test_shap_cpu(self):
        print("Running test_shap_cpu")
        X, y = self.data
        tree_struct = {'max_depth': 3, 
                'n_bins': 256,'min_data_in_leaf': 1,
                'par_th': 2,
                'grow_policy': 'greedy'}

        gbrl_params = dict({"control_variates": False, "split_score_func": "L2",
                            "generator_type": "Uniform"})
        model = GBRL(input_dim=self.input_dim,
                    output_dim=self.out_dim,
                    tree_struct=tree_struct,
                    optimizer=self.sgd_optimizer,
                    gbrl_params=gbrl_params,
                    verbose=0,
                    device='cpu')
        model._model.step(X, y)
        gbrl_shap = model.tree_shap(0, X[0, :])
        clf = DecisionTreeRegressor(max_depth=3).fit(X, y)
        target_shap = shap.TreeExplainer(clf).shap_values(X[0])
        self.assertTrue(np.allclose(gbrl_shap, target_shap, rtol=1e-3), f'GBRL sHAP values are not close to target SHAP values')

    def test_cosine_adam_cpu(self):
        print("Running Multi test_cosine_adam_cpu")
        X, y = self.data
        optimizer = {'algo': 'Adam',
                    'lr': 1.0,
                    'start_idx': 0,
                    'stop_idx': self.out_dim
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine"})
        model = GBRL(input_dim=self.input_dim,
                    output_dim=self.out_dim,
                    tree_struct=self.tree_struct,
                    optimizer=optimizer,
                    gbrl_params=gbrl_params,
                    verbose=0,
                    device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 50.0
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_cosine_adam_cpu'))

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_cosine_gpu(self):
        print("Running Multi test_cosine_gpu")
        X, y = self.data
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine"})
        model = GBRL(input_dim=self.input_dim,
                    output_dim=self.out_dim,
                    tree_struct=self.tree_struct,
                    optimizer=self.sgd_optimizer,
                    gbrl_params=gbrl_params,
                    verbose=0,
                    device='cuda')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs, device='cuda')
        value = 2.0
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_cosine_gpu'))

    def test_cosine_oblivious_cpu(self):
        print("Running Multi test_cosine_oblivious_cpu")
        X, y = self.data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine"})
        model = GBRL(input_dim=self.input_dim,
                    output_dim=self.out_dim,
                    tree_struct=tree_struct,
                    optimizer=self.sgd_optimizer,
                    gbrl_params=gbrl_params,
                    verbose=0,
                    device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        self.assertTrue(loss < 12, f'Expected loss = {loss} < 12')
        model.save_model(os.path.join(self.test_dir, 'test_cosine_oblivious_cpu'))
    
    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_cosine_oblivious_gpu(self):
        print("Running Multi test_cosine_oblivious_gpu")
        X, y = self.data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine"})
        model = GBRL(input_dim=self.input_dim,
                    output_dim=self.out_dim,
                    tree_struct=tree_struct,
                    optimizer=self.sgd_optimizer,
                    gbrl_params=gbrl_params,
                    verbose=0,
                    device='cuda')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs, device='cuda')
        value = 12
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_cosine_oblivious_gpu'))

    def test_l2_cpu(self):
        print("Running Multi test_l2_cpu")
        X, y = self.data
        gbrl_params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GBRL(input_dim=self.input_dim,
                    output_dim=self.out_dim,
                    tree_struct=self.tree_struct,
                    optimizer=self.sgd_optimizer,
                    gbrl_params=gbrl_params,
                    verbose=0,
                    device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 0.5
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_l2_cpu'))

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_l2_gpu(self):
        print("Running Multi test_l2_gpu")
        X, y = self.data
        gbrl_params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GBRL(input_dim=self.input_dim,
                    output_dim=self.out_dim,
                    tree_struct=self.tree_struct,
                    optimizer=self.sgd_optimizer,
                    gbrl_params=gbrl_params,
                    verbose=0,
                    device='cuda')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs, device='cuda')
        value = 0.5
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_l2_gpu'))

    def test_l2_oblivious_cpu(self):
        print("Running Multi test_l2_oblivious_cpu")
        X, y = self.data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}
        gbrl_params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GBRL(input_dim=self.input_dim,
                    output_dim=self.out_dim,
                    tree_struct=tree_struct,
                    optimizer=self.sgd_optimizer,
                    gbrl_params=gbrl_params,
                    verbose=0,
                    device='cpu')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 10.0
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_l2_oblivious_cpu'))

    def test_1shared_cpu(self):
        print("Running shared_cpu")
        X, y = self.data
        policy_optimizer = {
            'policy_algo': 'SGD',
            'policy_lr': 1.0,
            'start_idx': 0,
            'stop_idx': self.out_dim - 1
        }
        value_optimizer = {
                    'value_algo': 'SGD',
                    'value_lr': 0.1,
                    'start_idx': self.out_dim - 1,
                    'stop_idx': self.out_dim
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "cosine"})
        model = ActorCritic(
                            input_dim=self.input_dim,
                            output_dim=self.out_dim,
                            tree_struct=self.tree_struct,
                            policy_optimizer=policy_optimizer,
                            shared_tree_struct=True,
                            value_optimizer=value_optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cpu')
        policy_loss, value_loss = ac_rmse_model(model, X, y, self.n_epochs)
        policy_value = 10.0
        value_value = 30
        self.assertTrue(policy_loss < policy_value, f'Expected loss = {policy_loss} < {policy_value}')
        self.assertTrue(value_loss < value_value, f'Expected loss = {value_loss} < {value_value}')
        model.save_model(os.path.join(self.test_dir, 'test_shared_cpu'))

    def test_1separate_cpu(self):
        print("Running separate_cpu")
        X, y = self.data
        policy_optimizer = {
            'policy_algo': 'SGD',
            'policy_lr': 1.0,
            'start_idx': 0,
            'stop_idx': self.out_dim - 1
        }
        value_optimizer = {
            'value_algo': 'SGD',
            'value_lr': 0.1,
            'start_idx': 0,
            'stop_idx': 1
        }
        gbrl_params = dict({"control_variates": False, "split_score_func": "cosine"})
        model = ActorCritic(input_dim=self.input_dim,
                            output_dim=self.out_dim,
                            tree_struct=self.tree_struct,
                            policy_optimizer=policy_optimizer,
                            shared_tree_struct=False,
                            value_optimizer=value_optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cpu')
        policy_loss, value_loss = ac_rmse_model(model, X, y, self.n_epochs)
        policy_value = 3.0
        value_value = 30
        self.assertTrue(policy_loss < policy_value, f'Expected loss = {policy_loss} < {policy_value}')
        self.assertTrue(value_loss < value_value, f'Expected loss = {value_loss} < {value_value}')
        model.save_model(os.path.join(self.test_dir, 'test_separate_cpu'))

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_1separate_gpu(self):
        print("Running separate_gpu")
        X, y = self.data
        policy_optimizer = {
            'policy_algo': 'SGD',
            'policy_lr': 1.0,
            'start_idx': 0,
            'stop_idx': self.out_dim -1
        }
        value_optimizer = {
                    'value_algo': 'SGD',
                    'value_lr': 0.1,
                    'start_idx': 0,
                    'stop_idx': 1
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "cosine"})
        model = ActorCritic(
                            input_dim=self.input_dim,
                            output_dim=self.out_dim,
                            tree_struct=self.tree_struct,
                            policy_optimizer=policy_optimizer,
                            shared_tree_struct=False,
                            value_optimizer=value_optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cuda')
        policy_loss, value_loss = ac_rmse_model(model, X, y, self.n_epochs, device='cuda')
        policy_value = 2.0
        value_value = 30
        self.assertTrue(policy_loss < policy_value, f'Expected loss = {policy_loss} < {policy_value}')
        self.assertTrue(value_loss < value_value, f'Expected loss = {value_loss} < {value_value}')
        model.save_model(os.path.join(self.test_dir, 'test_separate_gpu'))

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_1shared_gpu(self):
        print("Running shared_gpu")
        X, y = self.data
        policy_optimizer = {
            'policy_algo': 'SGD',
            'policy_lr': 1.0,
            'start_idx': 0,
            'stop_idx': self.out_dim - 1
        }
        value_optimizer = {
                    'value_algo': 'SGD',
                    'value_lr': 0.1,
                    'start_idx': self.out_dim - 1,
                    'stop_idx': self.out_dim
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "cosine"})
        model = ActorCritic(input_dim=self.input_dim,
                            output_dim=self.out_dim,
                            tree_struct=self.tree_struct,
                            policy_optimizer=policy_optimizer,
                            shared_tree_struct=True,
                            value_optimizer=value_optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cuda')
        policy_loss, value_loss = ac_rmse_model(model, X, y, self.n_epochs, device='cuda')
        policy_value = 10.0
        value_value = 30
        self.assertTrue(policy_loss < policy_value, f'Expected loss = {policy_loss} < {policy_value}')
        self.assertTrue(value_loss < value_value, f'Expected loss = {value_loss} < {value_value}')
        model.save_model(os.path.join(self.test_dir, 'test_shared_gpu'))

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_l2_oblivious_gpu(self):
        print("Running Multi test_l2_oblivious_gpu")
        X, y = self.data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}
        gbrl_params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GBRL(input_dim=self.input_dim,
                        output_dim=self.out_dim,
                        tree_struct=tree_struct,
                        optimizer=self.sgd_optimizer,
                        gbrl_params=gbrl_params,
                        verbose=0,
                        device='cuda')
        model.set_bias_from_targets(y)
        loss = rmse_model(model, X, y, self.n_epochs, device='cuda')
        value = 10.0
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_l2_oblivious_gpu'))

    def test_loading(self):
        X, y = self.data

        model = GBRL.load_model(os.path.join(self.test_dir, 'test_cosine_cpu'), device='cpu')
        y_pred = model(X, requires_grad=False, tensor=False)
        loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
        self.assertTrue(loss < 2.0, f'Expected loss = {loss} < 2.0')
        
        
        model = GBRL.load_model(os.path.join(self.test_dir, 'test_l2_cpu'), device='cpu')
        y_pred = model(X, requires_grad=False, tensor=False)
        loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
        self.assertTrue(loss < 0.5, f'Expected loss = {loss} < 0.5')

        model = ActorCritic.load_model(os.path.join(self.test_dir, 'test_shared_cpu'), device='cpu')
        policy_y, value_y = model(X, requires_grad=False, tensor=False)
        policy_loss, _ = rmse(policy_y, y[:, :-1])
        value_loss, _ = rmse(value_y, y[:, -1])
        policy_value = 10.0
        value_value = 30
        self.assertTrue(policy_loss < policy_value, f'Expected loss = {policy_loss} < {policy_value}')
        self.assertTrue(value_loss < value_value, f'Expected loss = {value_loss} < {value_value}')

        model = ActorCritic.load_model(os.path.join(self.test_dir, 'test_separate_cpu'), device='cpu')
        policy_y, value_y = model(X, requires_grad=False, tensor=False)
        policy_loss, _ = rmse(policy_y, y[:, :-1])
        value_loss, _ = rmse(value_y, y[:, -1])
        policy_value = 2.0
        value_value = 30
        self.assertTrue(policy_loss < policy_value, f'Expected loss = {policy_loss} < {policy_value}')
        self.assertTrue(value_loss < value_value, f'Expected loss = {value_loss} < {value_value}')
        
        if (cuda_available()):
            model = GBRL.load_model(os.path.join(self.test_dir, 'test_cosine_gpu'), device='cuda')
            y_pred = model(X, requires_grad=False, tensor=False)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 2.0, f'Expected loss = {loss} < 2.0')
            
            model = GBRL.load_model(os.path.join(self.test_dir, 'test_cosine_oblivious_gpu'), device='cuda')
            y_pred = model(X, requires_grad=False, tensor=False)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 12.0, f'Expected loss = {loss} < 12.0')
            
            model = GBRL.load_model(os.path.join(self.test_dir, 'test_l2_gpu'), device='cuda')
            y_pred = model(X, requires_grad=False, tensor=False)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 0.5, f'Expected loss = {loss} < 0.5')

            model = GBRL.load_model(os.path.join(self.test_dir, 'test_l2_oblivious_gpu'), device='cuda')
            y_pred = model(X, requires_grad=False, tensor=False)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 10.0, f'Expected loss = {loss} < 10.0')

            model = ActorCritic.load_model(os.path.join(self.test_dir, 'test_shared_gpu'), device='cuda')
            policy_y, value_y = model(X, requires_grad=False, tensor=False)
            policy_loss, _ = rmse(policy_y, y[:, :-1])
            value_loss, _ = rmse(value_y, y[:, -1])
            policy_value = 10.0
            value_value = 30
            self.assertTrue(policy_loss < policy_value, f'Expected loss = {policy_loss} < {policy_value}')
            self.assertTrue(value_loss < value_value, f'Expected loss = {value_loss} < {value_value}')

            model = ActorCritic.load_model(os.path.join(self.test_dir, 'test_separate_gpu'), device='cuda')
            policy_y, value_y = model(X, requires_grad=False, tensor=False)
            policy_loss, _ = rmse(policy_y, y[:, :-1])
            value_loss, _ = rmse(value_y, y[:, -1])
            policy_value = 2.0
            value_value = 30
            self.assertTrue(policy_loss < policy_value, f'Expected loss = {policy_loss} < {policy_value}')
            self.assertTrue(value_loss < value_value, f'Expected loss = {value_loss} < {value_value}')

        model = GBRL.load_model(os.path.join(self.test_dir, 'test_cosine_adam_cpu'), device='cpu')
        y_pred = model(X, requires_grad=False, tensor=False)
        loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
        value = 50.0
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')


if __name__ == '__main__':
    unittest.main()