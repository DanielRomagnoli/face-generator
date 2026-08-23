import torch

from training.discriminator import Discriminator
from training.generator import Generator


def test_detached_fake_images_do_not_update_generator_gradients():
    generator = Generator()
    discriminator = Discriminator()

    noise = torch.randn(2, 100, 1, 1)

    fake_images = generator(noise)

    predictions = discriminator(
        fake_images.detach()
    )

    loss = predictions.mean()
    loss.backward()

    generator_gradients = [
        parameter.grad
        for parameter in generator.parameters()
    ]

    assert all(
        gradient is None
        for gradient in generator_gradients
    )