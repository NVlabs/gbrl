__version__ = "1.0.0"

from .ac_gbrl import (ActorCritic, GaussianActor, ContinuousCritic,
                   DiscreteCritic, ParametricActor)
from .gbt import GradientBoostingTrees
from .gbrl_cpp import GBRL

cuda_available = GBRL.cuda_available

