"""Utility functions."""

import random
import numpy as np
import torch


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name):
    """Get device."""
    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
