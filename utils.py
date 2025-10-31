import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

try:
    from torch.autograd import Variable
except Exception:
    Variable = None

import pandas as pd


def get_device():
    """Return the preferred device (cuda if available, else cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# module-level device (evaluated at import-time)
DEVICE = get_device()


def ensure_model_on_device(model, device=None):
    """Move model to device if it's not already there.

    Returns the model moved (in-place) and the device used.
    """
    if device is None:
        device = DEVICE
    # If model has parameters, check their device
    try:
        p = next(model.parameters())
    except StopIteration:
        # model has no parameters; still safe to call .to()
        return model.to(device)
    current_device = p.device
    if current_device != device:
        return model.to(device)
    return model


def _to_tensor(x, device=None, non_blocking=True):
    """Convert numpy array or Python scalar to torch tensor on device."""
    if device is None:
        device = DEVICE
    if torch.is_tensor(x):
        return x.to(device=device, non_blocking=non_blocking)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device=device, non_blocking=non_blocking)
    if isinstance(x, (int, float)):
        return torch.tensor(x, device=device)
    return x


def to_var(var, device=None, non_blocking=True):
    """Recursively move tensors/numpy arrays inside structures to the target device.

    - If `var` is a torch tensor -> moved to device
    - If `var` is a numpy array -> converted to torch tensor and moved
    - If `var` is dict/list/tuple -> recursively applied
    - If `var` is a Python scalar or string -> returned unchanged (or converted to tensor for ints/floats)
    """
    if device is None:
        device = DEVICE

    if torch.is_tensor(var):
        return var.to(device=device, non_blocking=non_blocking)

    if isinstance(var, np.ndarray):
        return torch.from_numpy(var).float().to(device=device, non_blocking=non_blocking)

    if isinstance(var, (int, float)):
        # return a scalar tensor on device
        return torch.tensor(var, device=device)

    if isinstance(var, str):
        return var

    if isinstance(var, dict):
        return {k: to_var(v, device=device, non_blocking=non_blocking) for k, v in var.items()}

    if isinstance(var, list):
        return [to_var(x, device=device, non_blocking=non_blocking) for x in var]

    if isinstance(var, tuple):
        return tuple(to_var(x, device=device, non_blocking=non_blocking) for x in var)

    return var


def stop_gradient(x):
    # Return a detached tensor(s)
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, tuple):
        return tuple(y.detach() if torch.is_tensor(y) else y for y in x)
    if torch.is_tensor(x):
        return x.detach()
    return x


def zero_var(sz, device=None):
    if device is None:
        device = DEVICE
    return torch.zeros(sz, device=device)
