import torch
import torch.nn as nn

from training.config import (
    DISCRIMINATOR_FEATURES,
    IMAGE_CHANNELS,
)


class Discriminator(nn.Module):
    def __init__(
        self,
        features: int = DISCRIMINATOR_FEATURES,
        image_channels: int = IMAGE_CHANNELS,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                image_channels,
                features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                features,
                features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                features * 2,
                features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                features * 4,
                features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                features * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)