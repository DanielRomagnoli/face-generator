import torch

def generate_noise(
    batch_size: int,
    latent_dim: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(
        batch_size,
        latent_dim,
        1,
        1,
        device=device,
    )