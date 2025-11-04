##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
"""
GBRL Configuration Constants

This module contains configuration constants for optimizer validation
and parameter settings used throughout the GBRL library.
"""
# shrinkage is not implemented yet
VALID_OPTIMIZER_ARGS = ['init_lr', 'scheduler', 'shrinkage', 'algo', 'beta_1',
                        'beta_2', 'eps', 'T', 'start_idx', 'stop_idx']
APPROVED_OPTIMIZERS = ["Adam", "SGD"]
