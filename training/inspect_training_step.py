import torch

from training.config import (
    BETA1,
    BETA2,
    LATENT_DIM,
    LEARNING_RATE,
)
from training.dataloader import get_dataloader
from training.device import get_device
from training.discriminator import Discriminator
from training.generator import Generator
from training.losses import get_adversarial_loss
from training.noise import generate_noise
from training.train_steps import train_gan_step
from training.weights import initialize_weights


def main() -> None:
    device = get_device()

    print(f"Device: {device}")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    criterion = get_adversarial_loss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
    )

    dataloader = get_dataloader(
        batch_size=8,
        shuffle=True,
    )

    real_images = next(iter(dataloader)).to(device)

    noise = generate_noise(
        batch_size=real_images.size(0),
        latent_dim=LATENT_DIM,
        device=device,
    )

    metrics = train_gan_step(
        discriminator=discriminator,
        generator=generator,
        real_images=real_images,
        noise=noise,
        criterion=criterion,
        optimizer_d=optimizer_d,
        optimizer_g=optimizer_g,
    )

    print("\nTraining metrics")

    for name, value in metrics.items():
        print(f"{name:15}: {value:.4f}")


if __name__ == "__main__":
    main()