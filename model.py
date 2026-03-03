"""Model architecture."""

import torch
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


class CNNWithTransformerModel(nn.Module):
    def __init__(self, num_classes=15, d_model=256, nhead=8, num_layers=3):
        super().__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.Conv2d(128, d_model, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
        )

        self.projection = nn.Linear(d_model * 12, d_model)

        self.pos_embedding = nn.Parameter(torch.zeros(1, 26, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.db = nn.Linear(d_model, num_classes)

    def forward(self, x):

        x = self.cnn_block(x)

        b, c, h, w = x.shape

        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)

        x = self.projection(x)
        x = x + self.pos_embedding

        x = self.transformer_encoder(x)

        x = x.mean(dim=1)
        x = self.db(x)

        return x


def create_model(config):
    """Create model from config."""
    if config.model_type == "cnn":
        return CNNModel(config.kernel_size, config.output_dim)
    elif config.model_type == "cnn_transformer":
        return CNNWithTransformerModel(
            num_classes=config.num_classes,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
        )
