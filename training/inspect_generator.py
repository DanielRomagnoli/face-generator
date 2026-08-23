import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from training.device import get_device
from training.generator import Generator
from training.noise import generate_noise


def main() -> None:
    device = get_device()

    generator = Generator().to(device)

    noise = generate_noise(
        batch_size=64,
        latent_dim=100,
        device=device,
    )

    with torch.no_grad():
        fake_images = generator(noise)

    grid = make_grid(
        fake_images.cpu(),
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import torch

    main()