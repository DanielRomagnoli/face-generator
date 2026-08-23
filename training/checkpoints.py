from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
        },
        checkpoint_path,
    )

def load_checkpoint(
    path: str,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    discriminator.load_state_dict(
        checkpoint["discriminator_state_dict"]
    )

    optimizer_g.load_state_dict(
        checkpoint["optimizer_g_state_dict"]
    )

    optimizer_d.load_state_dict(
        checkpoint["optimizer_d_state_dict"]
    )

    return checkpoint["epoch"]