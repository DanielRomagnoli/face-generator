import torch

from training.checkpoints import save_checkpoint
from training.config import (
    BATCH_SIZE,
    BETA1,
    BETA2,
    CHECKPOINT_DIR,
    LATENT_DIM,
    LEARNING_RATE,
    LOG_INTERVAL,
    NUM_EPOCHS,
    SAMPLE_DIR,
    SAMPLE_SIZE,
)
from training.dataloader import get_dataloader
from training.device import get_device
from training.discriminator import Discriminator
from training.generator import Generator
from training.losses import get_adversarial_loss
from training.noise import generate_noise
from training.samples import save_generated_samples
from training.train_steps import train_gan_step
from training.weights import initialize_weights

def main(
    num_epochs: int = NUM_EPOCHS,
    max_batches: int | None = None,
) -> None:
    device = get_device()

    print(f"Training device: {device}")

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
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    fixed_noise = generate_noise(
        batch_size=SAMPLE_SIZE,
        latent_dim=LATENT_DIM,
        device=device,
    )

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}"
        )

        for batch_index, real_images in enumerate(dataloader):
            real_images = real_images.to(device)

            batch_size = real_images.size(0)

            noise = generate_noise(
                batch_size=batch_size,
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

            if (
                max_batches is not None
                and batch_index + 1 >= max_batches
            ):
                break
            
            if batch_index % LOG_INTERVAL == 0:
                print(
                    f"[{batch_index:04d}/{len(dataloader):04d}] "
                    f"Loss D: {metrics['loss_d']:.4f} | "
                    f"Loss G: {metrics['loss_g']:.4f} | "
                    f"D(real): {metrics['d_real']:.3f} | "
                    f"D(fake): {metrics['d_fake']:.3f}"
                )

        sample_path = (
            f"{SAMPLE_DIR}/epoch_{epoch + 1:03d}.png"
        )

        save_generated_samples(
            generator=generator,
            fixed_noise=fixed_noise,
            path=sample_path,
        )

        checkpoint_path = (
            f"{CHECKPOINT_DIR}/"
            f"checkpoint_epoch_{epoch + 1:03d}.pt"
        )

        save_checkpoint(
            path=checkpoint_path,
            epoch=epoch + 1,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
        )

    print("\nTraining complete.")

if __name__ == "__main__":
    main()

    