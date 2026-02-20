"""Loss functions."""

import torch.nn as nn


def get_loss_fn():
    """TODO: Define your loss function."""
    return nn.CrossEntropyLoss()
