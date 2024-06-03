from typing import Dict 
import numpy as np

from .config import APPROVED_OPTIMIZERS, VALID_OPTIMIZER_ARGS

def setup_optimizer(optimizer: Dict, prefix: str='') -> Dict:
    """Setup optimizer to correctly allign with GBRL C++ module

    Args:
        optimizer (Dict): optimizer dictionary
        prefix (str, optional): optimizer parameter prefix names such as: mu_lr, mu_algo, std_algo, policy_algo, value_algo, etc. Defaults to ''.

    Returns:
        Dict: modified optimizer dictionary
    """
    if prefix:
        optimizer = {k.replace(prefix, ''): v for k, v in optimizer.items()}
    lr = optimizer.get('lr', 1.0) if 'init_lr' not in optimizer else optimizer['init_lr']
    # setup scheduler
    optimizer['scheduler'] = 'Const'
    assert isinstance(lr, int) or isinstance(lr, float) or isinstance(lr, str), "lr must be a float or string"
    if isinstance(lr, str) and 'lin_' in lr:
        lr = lr.replace('lin_', '')
        optimizer['scheduler'] = 'Linear'
    optimizer['init_lr'] = float(lr)
    optimizer['algo'] = optimizer.get('algo', 'SGD')
    assert optimizer['algo'] in APPROVED_OPTIMIZERS, f"optimization algo has to be in {APPROVED_OPTIMIZERS}"
    optimizer['stop_lr'] = optimizer.get('stop_lr', 1.0e-8)
    optimizer['beta_1'] = optimizer.get('beta_1', 0.9)
    optimizer['beta_2'] = optimizer.get('beta_2', 0.999)
    optimizer['eps'] = optimizer.get('eps', 1.0e-5)
    optimizer['shrinkage'] = optimizer.get('shrinkage', 0.0)
    if optimizer['shrinkage'] is None:
        optimizer['shrinkage'] = 0.0

    return {k: v for k, v in optimizer.items() if k in VALID_OPTIMIZER_ARGS}


def clip_grad_norm(grads: np.array, grad_clip: float) -> np.array:
    """clip per sample gradients according to their norm

    Args:
        grads (np.array): gradients
        grad_clip (float): gradient clip value

    Returns:
        np.array: clipped gradients
    """
    if grad_clip is None or grad_clip == 0.0:
        return grads
    if len(grads.shape) == 1:
        grads = np.clip(grads, a_min=-grad_clip, a_max=grad_clip)
        return grads 
    grad_norms = np.linalg.norm(grads, axis=1)
    grads[grad_norms > grad_clip] = grad_clip*grads[grad_norms > grad_clip] / grads[grad_norms > grad_clip]
    return grads




