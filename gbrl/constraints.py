##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#

from typing import Union, List, Optional
import numpy as np
import torch as th

from gbrl.utils import get_index_mapping

CONSTRAINTS = ['threshold', 'hierarchy', 'output']


class Constraint:
    def __init__(self):
        self.constraints = []
        self.used = False

    def parse_dataset(self, dataset: Union[np.ndarray, th.Tensor]) -> None:
        index_mapping = get_index_mapping(dataset)
        for i in range(len(self.constraints)):
            self.constraints[i]['feature_idx'] = index_mapping[self.constraints[i]['feature_idx']]
    
    def get_constraints(self):
        return self.constraints
    def add_constraint(self, constraint_type: str, feature_idx: int, 
                 feature_value: Optional[Union[float, str]] = None, 
                 op_is_positive: bool = True,
                 constraint_value: float = None,
                 output_values: Optional[Union[np.ndarray, List[float]]] = None, 
                 dependent_features: Optional[Union[np.ndarray, List[int]]] = None) -> None: 
        """Represents a constraint that can be one of three types:
        
        - **THRESHOLD:** Enforces a condition on a feature's value.
          - `op_is_positive = True` means `>` for numeric features and `==` for categorical features.
          - `op_is_positive = False` means `<=` for numeric features and `!=` for categorical features.
        
        - **HIERARCHY:** Enforces that a feature must be split before other dependent features.
          - `dependent_features` is a list of feature indices that must come after this one.
        
        - **OUTPUT:** Forces a specific action when the constraint is met.
          - `output_value` is a NumPy array containing the action constraints.

        Args:
            constraint_type (str): "THRESHOLD", "HIERARCHY", or "OUTPUT".
            feature_index (int): Index of the feature being constrained.
            feature_value (Optional[Union[float, str]], optional):  Threshold value (numeric or categorical).. Defaults to None.
            op_is_positive (bool, optional): Defines the operation for the Threshold value. Defaults to True.
            output_value (Optional[Union[np.ndarray, List[float]]], optional): Specifies the constraints node values for OUTPUT constraints. Defaults to None.
            dependent_features (Optional[Union[np.ndarray, List[int]]], optional): List of dependent features for HIERARCHY constraints. Defaults to None.
        """        
        constraint_type = constraint_type.lower()
        is_numeric = not isinstance(feature_value, str)
        assert constraint_type in CONSTRAINTS, f"Constraint type must be one of {CONSTRAINTS}"
        if constraint_type != 'hierarchy':
            assert dependent_features is None, "Can only set constraints on dependent features using a hierarchy constraint"
        constraint = {'feature_idx': feature_idx, 'feature_value': feature_value if is_numeric else 0.0,
                            'categorical_value': None if is_numeric else feature_value.encode('utf-8').ljust(128, '\0'),
                            'constraint_type': constraint_type, 'is_numeric': is_numeric,
                            'op_is_positive': op_is_positive, 
                            }
        if dependent_features is not None:
            if isinstance(dependent_features, np.ndarray):
                dependent_features = dependent_features.flatten().astype(np.intc)
            else:
                dependent_features = np.asarray(dependent_features, dtype=np.intc)
                    
        constraint['dependent_features'] = dependent_features
        constraint['n_features'] = 0 if dependent_features is None else len(dependent_features)

        if output_values is not None:
            if isinstance(output_values, np.ndarray):
                output_values = output_values.flatten().astype(np.single)
            else:
                output_values = np.asarray(output_values, dtype=np.single)
        constraint['output_values'] = output_values
        if constraint_value is not None:
            assert constraint_value >= 0, "constraint value must be a positive float"
        constraint['constraint_value'] = constraint_value if constraint_value is not None else 0.0
        self.constraints.append(constraint)
