# Gradient Boosting Reinforcement Learning (GBRL)
GBRL is a Python-based Gradient Boosting Trees (GBT) library, similar to popular packages such as [XGBoost](https://xgboost.readthedocs.io/en/stable/), [CatBoost](https://catboost.ai/), but specifically designed and optimized for reinforcement learning (RL). GBRL is implemented in C++/CUDA aimed to seamlessly integrate within popular RL libraries. 

<!-- [![Build Status](https://img.shields.io/github/workflow/status/Nvlabs/gbrl/CI)](https://github.com/NVlabs/gbrl/actions) -->
[![License](https://img.shields.io/badge/license-NVIDIA-green.svg)](https://nvlabs.github.io/gbrl/license.htm)
[![PyPI version](https://badge.fury.io/py/gbrl.svg)](https://badge.fury.io/py/gbrl)
<!-- [![Python Coverage](https://codecov.io/gh/Nvlabs/gbrl/branch/master/graph/badge.svg?flag=python)](https://codecov.io/gh/Nvlabs/gbrl)
[![C++ Coverage](https://codecov.io/gh/Nvlabs/gbrl/branch/master/graph/badge.svg?flag=cpp)](https://codecov.io/gh/Nvlabs/gbrl) -->

## Overview

GBRL adapts the power of Gradient Boosting Trees to the unique challenges of RL environments, including non-stationarity and the absence of predefined targets. The following diagram illustrates how GBRL uses gradient boosting trees in RL:

![GBRL Diagram](https://github.com/NVlabs/gbrl/raw/master/docs/images/gbrl_diagram.png)

GBRL features a shared tree-based structure for policy and value functions, significantly reducing memory and computational overhead, enabling it to tackle complex, high-dimensional RL problems.

## Key Features: 
- GBT Tailored for RL: GBRL adapts the power of Gradient Boosting Trees to the unique challenges of RL environments, including non-stationarity and the absence of predefined targets.
- Optimized Actor-Critic Architecture: GBRL features a shared tree-based structure for policy and value functions. This significantly reduces memory and computational overhead, enabling it to tackle complex, high-dimensional RL problems.
- Hardware Acceleration: GBRL leverages CUDA for hardware-accelerated computation, ensuring efficiency and speed.
- Seamless Integration: GBRL is designed for easy integration with popular RL libraries. We implemented GBT-based actor-critic algorithm implementations (A2C, PPO, and AWR) in stable_baselines3 [GBRL_SB3](https://github.com/NVlabs/gbrl_sb3). 

## Performance

The following results, obtained using the `GBRL_SB3` repository, demonstrate the performance of PPO with GBRL compared to neural-networks across various scenarios and environments:

![PPO GBRL results in stable_baselines3](https://github.com/NVlabs/gbrl/raw/master/docs/images/relative_ppo_performance.png)

## Getting started
### Prerequisites
- Python 3.9 or higher
- LLVM and OpenMP (macOS).

### Installation
To install GBRL via pip, use the following command:
```
pip install gbrl
```

For further installation details and dependencies see the documentation. 

### Usage Example
For a detailed usage example, see `tutorial.ipynb`

## Current Supported Features
### Tree Fitting
- Greedy (Depth-wise) tree building - (CPU/GPU)  
- Oblivious (Symmetric) tree building - (CPU/GPU)  
- L2 split score - (CPU/GPU)  
- Cosine split score - (CPU/GPU) 
- Uniform based candidate generation - (CPU/GPU)
- Quantile based candidate generation - (CPU/GPU)
- Supervised learning fitting / Multi-iteration fitting - (CPU/GPU)
    - MultiRMSE loss (only)
- Categorical inputs
- Input feature weights - (CPU/GPU)
### GBT Inference
- SGD optimizer - (CPU/GPU)
- ADAM optimizer - (CPU only)
- Control Variates (gradient variance reduction technique) - (CPU only)
- Shared Tree for policy and value function - (CPU/GPU)
- Linear and constant learning rate scheduler - (CPU/GPU only constant)
- Support for up to two different optimizers (e.g, policy/value) - **(CPU/GPU if both are SGD)
- SHAP value calculation

# Documentation 
For comprehensive documentation, visit the [GBRL documentation](https://nvlabs.github.io/gbrl/).

# Citation
``` 
@article{gbrl,
  title={Gradient Boosting Reinforcement Learning},
  author={Benjamin Fuhrer, Chen Tessler, Gal Dalal},
  year={2024},
  eprint={2407.08250},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.08250}, 
}
```
# Licenses
Copyright © 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](https://nvlabs.github.io/gbrl/license.htm). to view a copy of this license.

