##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html 
#
##############################################################################
__version__ = "1.0.4"

import platform
import os

if platform.system() == "Darwin":
    libomp_path = os.path.join(os.path.dirname(__file__), "include", "libomp")
    if "DYLD_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_LIBRARY_PATH"] = libomp_path + ":" + os.environ["DYLD_LIBRARY_PATH"]
    else:
        os.environ["DYLD_LIBRARY_PATH"] = libomp_path

from .ac_gbrl import (ActorCritic, GaussianActor, ContinuousCritic,
                   DiscreteCritic, ParametricActor)
from .gbt import GBRL
from .gbrl_cpp import GBRL as GBRL_CPP

cuda_available = GBRL_CPP.cuda_available

