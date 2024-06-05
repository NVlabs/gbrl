# Gradient Boosting Reinforcement Learning (GBRL)
GBRL is a Python-based GBT library designed and optimized for reinforcement learning (RL). GBRL is implemented in C++/CUDA aimed to seamlessly integrate within popular RL libraries. 

### Key Features: 
- GBTs Tailored for RL: GBRL adapts the power of Gradient Boosting Trees to the unique challenges of RL environments, including non-stationarity and delayed feedback.
- Optimized Actor-Critic Architecture: GBRL features a shared tree-based structure for policy and value functions. This significantly reduces memory and computational overhead, enabling it to tackle complex, high-dimensional RL problems.
- Hardware Acceleration: GBRL leverages CUDA for hardware-accelerated computation, ensuring efficiency and speed.
- Seamless Integration: GBRL is designed for easy integration with popular RL libraries, making it readily accessible for practitioners.


## Getting started

### Dependencies 
#### MAC OS 
```
llvm
openmp
```

Make sure to run:
```
brew install libomp
brew install llvm
 ```

xcode command line tools should be installed installed 

### Installation
```
pip install gbrl
``` 

Verify that GPU is visible by running
```
import gbrl

gbrl.cuda_available()
```

GBRL can be compiled and installed with a CPU version only even on CUDA capable machines by setting `CPU_ONLY=1` as an environment variable. 

*OPTIONAL*  
For tree visualization make sure graphviz is installed before compilation. 

***Usage Example see `tutorial.ipynb`***

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

# Citation

# Licenses
Copyright © 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](https://nvlabs.github.io/gbrl/license.htm). to view a copy of this license.

