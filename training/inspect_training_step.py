import torch

from training.dataloader import get_dataloader
from training.device import get_device
from training.discriminator import Discriminator
from training.generator import Generator
from training.losses import get_adversarial_loss
from training.noise import generate_noise
from training.weights import initialize_weights
from training.config import (
    LATENT_DIM,
    LEARNING_RATE,
    BETA1,
    BETA2,
)

def main() -> None:
    device = get_device()

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

    batch_size = real_images.size(0)

    noise = generate_noise(
        batch_size=batch_size,
        latent_dim=LATENT_DIM,
        device=device,
    )

    fake_images = generator(noise)

    optimizer_d.zero_grad()

    real_predictions = discriminator(real_images)

    real_labels = torch.ones_like(real_predictions)

    loss_d_real = criterion(
        real_predictions,
        real_labels,
    )

    fake_predictions = discriminator(
        fake_images.detach()
    )

    fake_labels = torch.zeros_like(fake_predictions)

    loss_d_fake = criterion(
        fake_predictions,
        fake_labels,
    )

    loss_d = loss_d_real + loss_d_fake
    loss_d.backward()
    optimizer_d.step()

    optimizer_g.zero_grad()
    generator_predictions = discriminator(fake_images)

    generator_labels = torch.ones_like(
        generator_predictions
    )

    loss_g = criterion(
        generator_predictions,
        generator_labels,
    )
    loss_g.backward()
    optimizer_g.step()

    print(f"Discriminator loss: {loss_d.item():.4f}")
    print(f"  Real loss:        {loss_d_real.item():.4f}")
    print(f"  Fake loss:        {loss_d_fake.item():.4f}")

    print(f"Generator loss:     {loss_g.item():.4f}")

    print(
        "D(real):",
        real_predictions.mean().item(),
    )

    print(
        "D(fake) before G update:",
        fake_predictions.mean().item(),
    )

    print(
        "D(fake) for G:",
        generator_predictions.mean().item(),
    )

if __name__ == "__main__":
    main()