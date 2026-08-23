import torch

from training.discriminator import Discriminator
from training.generator import Generator
from training.losses import get_adversarial_loss
from training.train_steps import (
    train_discriminator_step,
    train_generator_step,
    train_gan_step,
)


def clone_parameters(model):
    return [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]


def parameters_changed(before, model):
    after = list(model.parameters())

    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, after)
    )

def test_discriminator_step_updates_only_discriminator():
    generator = Generator()
    discriminator = Discriminator()

    criterion = get_adversarial_loss()

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )

    real_images = torch.randn(
        4,
        3,
        64,
        64,
    )

    noise = torch.randn(
        4,
        100,
        1,
        1,
    )

    generator_before = clone_parameters(generator)
    discriminator_before = clone_parameters(discriminator)

    train_discriminator_step(
        discriminator=discriminator,
        generator=generator,
        real_images=real_images,
        noise=noise,
        criterion=criterion,
        optimizer_d=optimizer_d,
    )

    assert parameters_changed(
        discriminator_before,
        discriminator,
    )

    assert not parameters_changed(
        generator_before,
        generator,
    )

def test_generator_step_updates_generator():
    generator = Generator()
    discriminator = Discriminator()

    criterion = get_adversarial_loss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )

    noise = torch.randn(
        4,
        100,
        1,
        1,
    )

    generator_before = clone_parameters(generator)

    train_generator_step(
        discriminator=discriminator,
        generator=generator,
        noise=noise,
        criterion=criterion,
        optimizer_g=optimizer_g,
    )

    assert parameters_changed(
        generator_before,
        generator,
    )

def test_complete_gan_step_returns_expected_metrics():
    generator = Generator()
    discriminator = Discriminator()

    criterion = get_adversarial_loss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )

    real_images = torch.randn(
        4,
        3,
        64,
        64,
    )

    noise = torch.randn(
        4,
        100,
        1,
        1,
    )

    from training.train_steps import train_gan_step

    metrics = train_gan_step(
        discriminator=discriminator,
        generator=generator,
        real_images=real_images,
        noise=noise,
        criterion=criterion,
        optimizer_d=optimizer_d,
        optimizer_g=optimizer_g,
    )

    expected_keys = {
        "loss_d",
        "loss_d_real",
        "loss_d_fake",
        "d_real",
        "d_fake",
        "loss_g",
        "d_fake_for_g",
    }

    assert set(metrics.keys()) == expected_keys

def test_gan_step_metrics_are_finite():
    generator = Generator()
    discriminator = Discriminator()

    criterion = get_adversarial_loss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )

    real_images = torch.randn(
        4,
        3,
        64,
        64,
    )

    noise = torch.randn(
        4,
        100,
        1,
        1,
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

    for value in metrics.values():
        assert torch.isfinite(
            torch.tensor(value)
        )

def test_generator_step_does_not_update_discriminator():
    generator = Generator()
    discriminator = Discriminator()

    criterion = get_adversarial_loss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )

    noise = torch.randn(
        4,
        100,
        1,
        1,
    )

    discriminator_before = clone_parameters(
        discriminator
    )

    train_generator_step(
        discriminator=discriminator,
        generator=generator,
        noise=noise,
        criterion=criterion,
        optimizer_g=optimizer_g,
    )

    assert not parameters_changed(
        discriminator_before,
        discriminator,
    )

def test_generator_step_restores_discriminator_gradients():
    generator = Generator()
    discriminator = Discriminator()

    criterion = get_adversarial_loss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )

    noise = torch.randn(
        4,
        100,
        1,
        1,
    )

    train_generator_step(
        discriminator=discriminator,
        generator=generator,
        noise=noise,
        criterion=criterion,
        optimizer_g=optimizer_g,
    )

    assert all(
        parameter.requires_grad
        for parameter in discriminator.parameters()
    )

