from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image


def save_generated_samples(
    generator: nn.Module,
    fixed_noise: torch.Tensor,
    path: str,
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    was_training = generator.training

    generator.eval()

    with torch.no_grad():
        fake_images = generator(fixed_noise)

    save_image(
        fake_images,
        output_path,
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )

    if was_training:
        generator.train()