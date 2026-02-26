"""Model architecture."""

import torch.nn as nn


class CNNModel(nn.Module):
    """CNN model architecture."""

    def __init__(self, kernel_size, output_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=5, stride=5)
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.bn2 = nn.BatchNorm2d(256)
        self.bn1 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveMaxPool2d((3, 1))
        self.fc = nn.Linear(256 * 3 * 1, output_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)


def create_model(config):
    """Create model from config."""
    return CNNModel(
        config.kernel_size, config.output_dim
    )
