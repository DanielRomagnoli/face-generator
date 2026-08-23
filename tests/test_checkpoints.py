import torch

from training.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from training.discriminator import Discriminator
from training.generator import Generator


def test_checkpoint_round_trip(tmp_path):
    generator = Generator()
    discriminator = Discriminator()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=0.0002,
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=0.0002,
    )

    path = tmp_path / "checkpoint.pt"

    original_generator = [
        parameter.detach().clone()
        for parameter in generator.parameters()
    ]

    save_checkpoint(
        path=str(path),
        epoch=3,
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
    )

    for parameter in generator.parameters():
        parameter.data.zero_()

    loaded_epoch = load_checkpoint(
        path=str(path),
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=torch.device("cpu"),
    )

    assert loaded_epoch == 3

    for original, restored in zip(
        original_generator,
        generator.parameters(),
    ):
        assert torch.equal(
            original,
            restored.detach(),
        )