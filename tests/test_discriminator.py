import torch

from training.discriminator import Discriminator


def test_discriminator_output_shape():
    discriminator = Discriminator()

    images = torch.randn(4, 3, 64, 64)

    output = discriminator(images)

    assert output.shape == (4, 1, 1, 1)


def test_discriminator_output_range():
    discriminator = Discriminator()

    images = torch.randn(4, 3, 64, 64)

    output = discriminator(images)

    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0


def test_discriminator_preserves_batch_size():
    discriminator = Discriminator()

    images = torch.randn(7, 3, 64, 64)

    output = discriminator(images)

    assert output.shape[0] == 7


def test_discriminator_has_trainable_parameters():
    discriminator = Discriminator()

    trainable_parameters = [
        parameter
        for parameter in discriminator.parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) > 0

def test_discriminator_backpropagation():
    discriminator = Discriminator()

    images = torch.randn(4, 3, 64, 64)

    output = discriminator(images)

    loss = output.mean()

    loss.backward()

    gradients_exist = [
        parameter.grad is not None
        for parameter in discriminator.parameters()
        if parameter.requires_grad
    ]

    assert all(gradients_exist)

