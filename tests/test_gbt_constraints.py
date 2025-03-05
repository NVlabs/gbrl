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

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
import shap
import torch as th
from torch.nn.functional import mse_loss

FILE_PATH = os.path.dirname(os.path.dirname(__file__))

from gbrl.gbt import GBRL
from gbrl.constraints import Constraint
from gbrl import cuda_available

N_EPOCHS = 100

def rmse_model(model, X, y, n_epochs, device='cpu'):
    y_ = th.tensor(y, dtype=th.float32, device=device).squeeze()
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

class TestGBTConstraints(unittest.TestCase):

    @classmethod 
    def setUpClass(cls):
        print('Generating data...')
        n_samples = 100
        categories = ["A", "B", "C", "D", "E"]
        n_categories = len(categories)
        # Imagine this loads your actual data
        np.random.seed(42)  # Set a seed for the first-time generation
        X_rand= np.random.uniform(-100, 100, size=100).tolist()  # Uniformly spaced from -100 to 100
        X_float = [-1 + (2 * i / (n_samples - 1)) for i in range(n_samples)]  # Uniformly spaced from -1 to 1
        X_str = [categories[i // (n_samples // n_categories)] for i in range(n_samples)]  # Categorical assignment

        # Convert to numpy arrays
        X = np.array(list(zip(X_float, X_rand, X_str)), dtype=object)
        y = np.array(X_float, dtype=np.single)  # Target is the first column
        out_dim = 1 if len(y.shape) == 1  else  y.shape[1]
        if out_dim == 1:
            y = y[:, np.newaxis]
        input_dim = X.shape[1]
        cls.single_data = (X, y)
        cls.out_dim = out_dim
        cls.input_dim = input_dim
        cls.test_dir = tempfile.mkdtemp()
        # Setup categorical data
        cls.tree_struct = {'max_depth': 3, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'greedy'}
        cls.sgd_optimizer = { 'algo': 'SGD',
                    'lr': 1.0,
                    'start_idx': 0,
                    'stop_idx': out_dim
                }
        cls.oblivious_tree_struct = {'max_depth': 3, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}
    
    @classmethod 
    def tearDownClass(cls):
        # Remove the directory after the test
        shutil.rmtree(cls.test_dir)

    def test_hierarchy_constraints_cpu(self):
        print("Running test_hierarchy_constraints_cpu")
        X, y = self.single_data
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine",
                            "generator_type": "Quantile"})
        model = GBRL(input_dim=self.input_dim,
                     output_dim=self.out_dim,
                     tree_struct=self.tree_struct,
                     optimizer=self.sgd_optimizer,
                     gbrl_params=gbrl_params,
                     verbose=0,
                     device='cpu')
        model.set_bias_from_targets(y)
        _ = rmse_model(model, X, y, 1)
        edata = model._model.cpp_model.get_ensemble_data()
        assert (edata['feature_indices'].flatten() == 0).all(), "Unconstrained model should only use feature idx: 0"

        constraint = Constraint()
        constraint.add_constraint(constraint_type="hierarchy", feature_idx=0, dependent_features=[2, 1])
        constraint.add_constraint(constraint_type="hierarchy", feature_idx=2, dependent_features=[1])
        model = GBRL(input_dim=self.input_dim,
                     output_dim=self.out_dim,
                     tree_struct=self.oblivious_tree_struct,
                     optimizer=self.sgd_optimizer,
                     gbrl_params=gbrl_params,
                     verbose=0,
                     device='cpu',
                     constraints=constraint)
        model.set_bias_from_targets(y)
        _ = rmse_model(model, X, y, 1)
        edata = model._model.cpp_model.get_ensemble_data()
        assert (edata['feature_indices'][:, 0].flatten() == 1).all(), "Model should use feature index 1 for the root node"
        assert (edata['feature_indices'][:, 1].flatten() + (edata['mapping_numerics'] == True).sum() == 2).all(), "Model should use feature index 2 for the second layer"
        assert (edata['feature_indices'][:, 2].flatten() == 0).all(), "Model should use feature index 0 for the final layer"

    def test_threshold_constraints_cpu(self):
        print("Running test_threshold_constraints_cpu")
        X, y = self.single_data
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine",
                            "generator_type": "Quantile"})

        constraint = Constraint()
        constraint.add_constraint(constraint_type="threshold", feature_idx=0,
                        feature_value=-0.5, op_is_positive=False, constraint_value=10)
        constraint.add_constraint(constraint_type="threshold", feature_idx=2,
                        feature_value='B', op_is_positive=True, constraint_value=10)
        model = GBRL(input_dim=self.input_dim,
                     output_dim=self.out_dim,
                     tree_struct=self.oblivious_tree_struct,
                     optimizer=self.sgd_optimizer,
                     gbrl_params=gbrl_params,
                     verbose=0,
                     device='cpu',
                     constraints=constraint)
        model.set_bias_from_targets(y)
        _ = rmse_model(model, X, y, 1)
        edata = model._model.cpp_model.get_ensemble_data()
        assert (edata['feature_indices'][:, 0].flatten() == 0).all() and (edata['feature_values'][:, 0] < -0.5).all(), "Model should use feature index 1 for the root node"
        assert (edata['feature_indices'][:, 1].flatten() + (edata['mapping_numerics'] == True).sum() == 2).all() and (edata['categorical_values'][:, 1] == b'B').all(), "Model should use feature index 2 for the second layer"

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_hierarchy_constraints_gpu(self):
        print("Running test_hierarchy_constraints_gpu")
        X, y = self.single_data
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine",
                            "generator_type": "Quantile"})
        model = GBRL(input_dim=self.input_dim,
                     output_dim=self.out_dim,
                     tree_struct=self.tree_struct,
                     optimizer=self.sgd_optimizer,
                     gbrl_params=gbrl_params,
                     verbose=0,
                     device='cuda')
        model.set_bias_from_targets(y)
        _ = rmse_model(model, X, y, 1, device='cuda')
        edata = model._model.cpp_model.get_ensemble_data()
        assert (edata['feature_indices'].flatten() == 0).all(), "Unconstrained model should only use feature idx: 0"

        constraint = Constraint()
        constraint.add_constraint(constraint_type="hierarchy", feature_idx=0, dependent_features=[2, 1])
        constraint.add_constraint(constraint_type="hierarchy", feature_idx=2, dependent_features=[1])
        model = GBRL(input_dim=self.input_dim,
                     output_dim=self.out_dim,
                     tree_struct=self.tree_struct,
                     optimizer=self.sgd_optimizer,
                     gbrl_params=gbrl_params,
                     verbose=0,
                     device='cuda',
                     constraints=constraint)
        model.set_bias_from_targets(y)
        model.set_feature_weights([10.0, 1.0, 1.0])
        _ = rmse_model(model, X, y, 1, device='cuda')
        edata = model._model.cpp_model.get_ensemble_data()
        assert (edata['feature_indices'][:, 0].flatten() == 1).all(), "Model should use feature index 1 for the root node"
        assert (edata['feature_indices'][:, 1].flatten() + (edata['mapping_numerics'] == True).sum() == 2).all(), "Model should use feature index 2 for the second layer"
        assert (edata['feature_indices'][:, 2].flatten() == 0).all(), "Model should use feature index 0 for the final layer"
        

    @unittest.skipIf(not cuda_available(), "cuda not available skipping over gpu tests")
    def test_threshold_constraints_gpu(self):
        print("Running test_threshold_constraints_gpu")
        X, y = self.single_data
        gbrl_params = dict({"control_variates": False, "split_score_func": "Cosine",
                            "generator_type": "Quantile"})
        tree_struct = {'max_depth': 3, 
                'n_bins': 256,'min_data_in_leaf': 0,
                'par_th': 2,
                'grow_policy': 'oblivious'}

        constraint = Constraint()
        constraint.add_constraint(constraint_type="threshold", feature_idx=0,
                        feature_value=-0.5, op_is_positive=False, constraint_value=10)
        constraint.add_constraint(constraint_type="threshold", feature_idx=2,
                        feature_value='B', op_is_positive=True, constraint_value=10)
        model = GBRL(input_dim=self.input_dim,
                     output_dim=self.out_dim,
                     tree_struct=tree_struct,
                     optimizer=self.sgd_optimizer,
                     gbrl_params=gbrl_params,
                     verbose=0,
                     device='cuda',
                     constraints=constraint)
        model.set_bias_from_targets(y)
        _ = rmse_model(model, X, y, 1, device='cuda')
        edata = model._model.cpp_model.get_ensemble_data()
        assert (edata['feature_indices'][:, 0].flatten() == 0).all() and (edata['feature_values'][:, 0] < -0.5).all(), "Model should use feature index 1 for the root node"
        assert (edata['feature_indices'][:, 1].flatten() + (edata['mapping_numerics'] == True).sum() == 2).all() and (edata['categorical_values'][:, 1] == b'B').all(), "Model should use feature index 2 for the second layer"

if __name__ == '__main__':
    # unittest.main()
    unittest.main()
