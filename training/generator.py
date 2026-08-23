import torch
import torch.nn as nn

from training.config import (
    GENERATOR_FEATURES,
    IMAGE_CHANNELS,
    LATENT_DIM,
)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        features: int = GENERATOR_FEATURES,
        image_channels: int = IMAGE_CHANNELS,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim,
                features * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                features * 8,
                features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                features * 4,
                features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                features * 2,
                features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                features,
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)