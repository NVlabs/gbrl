import os
import shutil
import tempfile
import unittest

import numpy as np
from sklearn import datasets
import torch as th
from torch.nn.functional import mse_loss

from gbrl import GradientBoostingTrees, cuda_available
from tests import CATEGORICAL_INPUTS, CATEGORICAL_OUTPUTS

N_EPOCHS = 100

def rmse_model(model, X, y, n_epochs):
    y_ = th.tensor(y, dtype=th.float32)
    X_ = X.copy()
    epoch = 0
    while epoch < n_epochs:
        y_pred = model(X_, requires_grad=True)
        loss = 0.5*mse_loss(y_pred, y_)
        loss.backward()
        model.step(X_)
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
        X ,y = datasets.load_diabetes(return_X_y=True, as_frame=False, scaled=False)
        out_dim = 1 if len(y.shape) == 1  else  y.shape[1]
        if out_dim == 1:
            y = y[:, np.newaxis]
        cls.single_data = (X, y)
        cls.out_dim = out_dim
        cls.n_epochs = 100
        cls.test_dir = tempfile.mkdtemp()
        # Setup categorical data
        X_categorical = np.array(CATEGORICAL_INPUTS, dtype=str)
        X_categorical = np.char.encode(X_categorical, encoding='utf-8', errors=None)
        X_categorical = np.char.decode(X_categorical, encoding='utf-8', errors=None)
        y_categorical = np.array(CATEGORICAL_OUTPUTS, dtype=np.single)[:, np.newaxis]
        cls.cat_data = (X_categorical, y_categorical)
        
    
    @classmethod 
    def tearDownClass(cls):
        # Remove the directory after the test
        shutil.rmtree(cls.test_dir)


    def test_cosine_cpu(self):
        print("Running test_cosine_cpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'greedy'}
        optimizer = { 'algo': 'SGD',
                    'lr': 1.0,
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine",
                            "generator_type": "Quantile"})
        model = GradientBoostingTrees(
                            output_dim=self.out_dim,
                            tree_struct=tree_struct,
                            optimizer=optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cpu')
        model.set_bias(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 2
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_cosine_cpu'))

        model._model.reset()
        model.set_bias(y)
        train_loss = model.fit(X, y, self.n_epochs)
        self.assertTrue(train_loss < value, f'Expected loss = {train_loss} < {value}')

        X_categorical, y_categorical = self.cat_data
        model._model.reset()
        model.set_bias(y)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value, f'Expected Categorical loss = {loss} < {value}')

    def test_cosine_adam_cpu(self):
        print("Running test_cosine_adam_cpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'greedy'}
        optimizer = { 'algo': 'Adam',
                    'lr': 1.0,
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine"})
        model = GradientBoostingTrees(
                            output_dim=self.out_dim,
                            tree_struct=tree_struct,
                            optimizer=optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cpu')
        model.set_bias(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 50
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_cosine_adam_cpu'))
    
    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_cosine_gpu(self):
        print("Running test_cosine_gpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4, 
                       'n_bins': 256,'min_data_in_leaf': 0,
                       'par_th': 2,
                       'grow_policy': 'greedy'}
        optimizer = { 'algo': 'SGD',
                    'lr': 1.0,
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine"})
        model = GradientBoostingTrees(
                            output_dim=self.out_dim,
                            tree_struct=tree_struct,
                            optimizer=optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cuda')
        model.set_bias(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 2
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_cosine_gpu'))

        model._model.reset()
        model.set_bias(y)
        train_loss = model.fit(X, y, self.n_epochs)
        value = 2
        self.assertTrue(train_loss < value, f'Expected loss = {train_loss} < {value}')

        X_categorical, y_categorical = self.cat_data
        model._model.reset()
        model.set_bias(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value, f'Expected Categorical loss = {loss} < {value}')

    def test_cosine_oblivious_cpu(self):
        print("Running test_cosine_oblivious_cpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}
        optimizer = { 'algo': 'SGD',
                    'lr': 1.0,
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine"})
        model = GradientBoostingTrees(
                            output_dim=self.out_dim,
                            tree_struct=tree_struct,
                            optimizer=optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cpu')
        model.set_bias(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 10
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_cosine_oblivious_gpu'))
        model._model.reset()
        model.set_bias(y)
        train_loss = model.fit(X, y, self.n_epochs)
        self.assertTrue(train_loss < value, f'Expected loss = {train_loss} < {value}')
        X_categorical, y_categorical = self.cat_data
        model._model.reset()
        model.set_bias(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value, f'Expected Categorical loss = {loss} < {value}')

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_cosine_oblivious_gpu(self):
        print("Running test_cosine_oblivious_gpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}
        optimizer = { 'algo': 'SGD',
                    'lr': 1.0,
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine"})
        model = GradientBoostingTrees(
                            output_dim=self.out_dim,
                            tree_struct=tree_struct,
                            optimizer=optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cuda')
        model.set_bias(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        value = 12
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')
        model.save_model(os.path.join(self.test_dir, 'test_cosine_oblivious_gpu'))
        model._model.reset()
        model.set_bias(y)
        train_loss = model.fit(X, y, self.n_epochs)
        self.assertTrue(train_loss < value, f'Expected loss = {train_loss} < {value}')

        X_categorical, y_categorical = self.cat_data
        model._model.reset()
        model.set_bias(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value, f'Expected Categorical loss = {loss} < {value}')

    def test_l2_cpu(self):
        print("Running test_l2_cpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'greedy'}
        optimizer = { 'algo': 'SGD',
                    'lr': 1.0,
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GradientBoostingTrees(
                            output_dim=self.out_dim,
                            tree_struct=tree_struct,
                            optimizer=optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cpu')
        model.set_bias(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        self.assertTrue(loss < 0.5, f'Expected loss = {loss} < 0.5')
        model.save_model(os.path.join(self.test_dir, 'test_l2_cpu'))
        model._model.reset()
        model.set_bias(y)
        train_loss = model.fit(X, y, self.n_epochs)
        value = 0.5
        self.assertTrue(train_loss < value, f'Expected loss = {train_loss} < {value}')
        X_categorical, y_categorical = self.cat_data
        model._model.reset()
        model.set_bias(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value, f'Expected Categorical loss = {loss} < {value}')

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_l2_gpu(self):
        print("Running test_l2_gpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'greedy'}
        optimizer = { 'algo': 'SGD',
                    'lr': 1.0,
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GradientBoostingTrees(
                            output_dim=self.out_dim,
                            tree_struct=tree_struct,
                            optimizer=optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cuda')
        model.set_bias(y)
        loss = rmse_model(model, X, y, self.n_epochs)

        self.assertTrue(loss < 0.5, f'Expected loss = {loss} < 0.5')
        model.save_model(os.path.join(self.test_dir, 'test_l2_gpu'))
        model._model.reset()
        model.set_bias(y)
        train_loss = model.fit(X, y, self.n_epochs)
        value = 0.5
        self.assertTrue(train_loss < value, f'Expected loss = {train_loss} < {value}')
        X_categorical, y_categorical = self.cat_data
        model._model.reset()
        model.set_bias(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value, f'Expected Categorical loss = {loss} < {value}')

    def test_l2_oblivious_cpu(self):
        print("Running test_l2_oblivious_cpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}
        optimizer = { 'algo': 'SGD',
                    'lr': 1.0,
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GradientBoostingTrees(
                            output_dim=self.out_dim,
                            tree_struct=tree_struct,
                            optimizer=optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cuda')
        model.set_bias(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        self.assertTrue(loss < 10.0, f'Expected loss = {loss} < 10.0')
        model.save_model(os.path.join(self.test_dir, 'test_l2_oblivious_cpu'))
        model._model.reset()
        model.set_bias(y)
        train_loss = model.fit(X, y, self.n_epochs)
        value = 10.0
        self.assertTrue(train_loss < value, f'Expected loss = {train_loss} < {value}')

        X_categorical, y_categorical = self.cat_data
        model._model.reset()
        model.set_bias(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value, f'Expected Categorical loss = {loss} < {value}')

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_l2_oblivious_gpu(self):
        print("Running test_l2_oblivious_gpu")
        X, y = self.single_data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}
        optimizer = { 'algo': 'SGD',
                    'lr': 1.0,
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "L2"})
        model = GradientBoostingTrees(
                            output_dim=self.out_dim,
                            tree_struct=tree_struct,
                            optimizer=optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cuda')
        model.set_bias(y)
        loss = rmse_model(model, X, y, self.n_epochs)
        self.assertTrue(loss < 10.0, f'Expected loss = {loss} < 10.0')
        model.save_model(os.path.join(self.test_dir, 'test_l2_oblivious_gpu'))
        model._model.reset()
        model.set_bias(y)
        train_loss = model.fit(X, y, self.n_epochs)
        value = 10.0
        self.assertTrue(train_loss < value, f'Expected loss = {train_loss} < {value}')
        X_categorical, y_categorical = self.cat_data
        model._model.reset()
        model.set_bias(y_categorical)
        loss = rmse_model(model, X_categorical, y_categorical, self.n_epochs)
        value = 5000
        self.assertTrue(loss < value, f'Expected Categorical loss = {loss} < {value}')

    def test_loading(self):
        X, y = self.single_data
        tree_struct = {'max_depth': 4, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}
        optimizer = { 'algo': 'SGD',
                    'lr': 1.0,
                }
        gbrl_params = dict({"control_variates": False, "split_score_func": "cosine"})
        model = GradientBoostingTrees(
                            output_dim=self.out_dim,
                            tree_struct=tree_struct,
                            optimizer=optimizer,
                            gbrl_params=gbrl_params,
                            verbose=0,
                            device='cpu')
        model = GradientBoostingTrees.load_model(os.path.join(self.test_dir, 'test_cosine_cpu'))
        y_pred = model.predict(X)
        loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
        self.assertTrue(loss < 2.0, f'Expected loss = {loss} < 2.0')
        
        model = GradientBoostingTrees.load_model(os.path.join(self.test_dir, 'test_l2_cpu'))
        y_pred = model.predict(X)
        loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
        self.assertTrue(loss < 0.5, f'Expected loss = {loss} < 0.5')
        if (cuda_available()):
            model = GradientBoostingTrees.load_model(os.path.join(self.test_dir, 'test_cosine_gpu'))
            y_pred = model.predict(X)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 2.0, f'Expected loss = {loss} < 2.0')
            model = GradientBoostingTrees.load_model(os.path.join(self.test_dir, 'test_cosine_oblivious_gpu'))
            y_pred = model.predict(X)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 12.0, f'Expected loss = {loss} < 12.0')
            model = GradientBoostingTrees.load_model(os.path.join(self.test_dir, 'test_l2_gpu'))
            y_pred = model.predict(X)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 0.5, f'Expected loss = {loss} < 0.5')
            model = GradientBoostingTrees.load_model(os.path.join(self.test_dir, 'test_l2_oblivious_gpu'))
            y_pred = model.predict(X)
            loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
            self.assertTrue(loss < 10.0, f'Expected loss = {loss} < 10.0')
        model = GradientBoostingTrees.load_model(os.path.join(self.test_dir, 'test_cosine_adam_cpu'))
        y_pred = model.predict(X)
        loss = np.sqrt(np.mean((y_pred.squeeze() - y.squeeze())**2))
        value = 50.0
        self.assertTrue(loss < value, f'Expected loss = {loss} < {value}')

if __name__ == '__main__':
    unittest.main()