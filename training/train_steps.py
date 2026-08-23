import torch
import torch.nn as nn


def train_discriminator_step(
    discriminator: nn.Module,
    generator: nn.Module,
    real_images: torch.Tensor,
    noise: torch.Tensor,
    criterion: nn.Module,
    optimizer_d: torch.optim.Optimizer,
) -> dict[str, float]:
    optimizer_d.zero_grad()

    fake_images = generator(noise)

    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images.detach())

    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)

    loss_real = criterion(
        real_logits,
        real_labels,
    )

    loss_fake = criterion(
        fake_logits,
        fake_labels,
    )

    loss_d = loss_real + loss_fake

    loss_d.backward()
    optimizer_d.step()

    return {
        "loss_d": loss_d.item(),
        "loss_d_real": loss_real.item(),
        "loss_d_fake": loss_fake.item(),
        "d_real": torch.sigmoid(real_logits).mean().item(),
        "d_fake": torch.sigmoid(fake_logits).mean().item(),
    }

def train_generator_step(
    discriminator: nn.Module,
    generator: nn.Module,
    noise: torch.Tensor,
    criterion: nn.Module,
    optimizer_g: torch.optim.Optimizer,
) -> dict[str, float]:
    optimizer_g.zero_grad()

    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    fake_images = generator(noise)

    fake_logits = discriminator(fake_images)

    real_labels = torch.ones_like(fake_logits)

    loss_g = criterion(
        fake_logits,
        real_labels,
    )

    loss_g.backward()
    optimizer_g.step()

    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)

    return {
        "loss_g": loss_g.item(),
        "d_fake_for_g": torch.sigmoid(
            fake_logits
        ).mean().item(),
    }

def train_gan_step(
    discriminator: nn.Module,
    generator: nn.Module,
    real_images: torch.Tensor,
    noise: torch.Tensor,
    criterion: nn.Module,
    optimizer_d: torch.optim.Optimizer,
    optimizer_g: torch.optim.Optimizer,
) -> dict[str, float]:
    discriminator_metrics = train_discriminator_step(
        discriminator=discriminator,
        generator=generator,
        real_images=real_images,
        noise=noise,
        criterion=criterion,
        optimizer_d=optimizer_d,
    )

    generator_metrics = train_generator_step(
        discriminator=discriminator,
        generator=generator,
        noise=noise,
        criterion=criterion,
        optimizer_g=optimizer_g,
    )

    return {
        **discriminator_metrics,
        **generator_metrics,
    }