# Gradient Boosting Reinforcement Learning (GBRL)
GBRL is a Python-based GBT library designed and optimized for reinforcement learning (RL). GBRL is implemented in C++/CUDA aimed to seamlessly integrate within popular RL libraries. 

<!-- [![Build Status](https://img.shields.io/github/workflow/status/Nvlabs/gbrl/CI)](https://github.com/NVlabs/gbrl/actions) -->
[![License](https://img.shields.io/badge/license-NVIDIA-green.svg)](https://nvlabs.github.io/gbrl/license.htm)
[![PyPI version](https://badge.fury.io/py/gbrl.svg)](https://badge.fury.io/py/gbrl)

## Key Features: 
- GBTs Tailored for RL: GBRL adapts the power of Gradient Boosting Trees to the unique challenges of RL environments, including non-stationarity and delayed feedback.
- Optimized Actor-Critic Architecture: GBRL features a shared tree-based structure for policy and value functions. This significantly reduces memory and computational overhead, enabling it to tackle complex, high-dimensional RL problems.
- Hardware Acceleration: GBRL leverages CUDA for hardware-accelerated computation, ensuring efficiency and speed.
- Seamless Integration: GBRL is designed for easy integration with popular RL libraries, making it readily accessible for practitioners.


## Getting started
### Prerequisites
- Python 3.9 or higher

### Installation
To install GBRL via pip, use the following command:
```
pip install gbrl
```

For furthere installation details and dependencies see the documentation. 

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

# Documentation 
For comprehensive documentation, visit the [GBRL documentation](https://effective-adventure-22v795q.pages.github.io/index.html).

# Citation
``` 
TODO
@article{gbrl,
  title={Gradient Boosting Reinforcement Learning},
  author={},
  journal={arXiv preprint arXiv:2406.xxxxx},
  year={2024}
}
```
# Licenses
Copyright © 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](https://nvlabs.github.io/gbrl/license.htm). to view a copy of this license.

