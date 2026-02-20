"""Model architecture."""

import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple linear model."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)


def create_model(config):
    """Create model from config."""
    return SimpleModel(config.input_dim, config.output_dim)
