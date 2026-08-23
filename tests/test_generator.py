import torch

from training.generator import Generator


def test_generator_output_shape():
    generator = Generator()

    noise = torch.randn(4, 100, 1, 1)

    output = generator(noise)

    assert output.shape == (4, 3, 64, 64)


def test_generator_output_range():
    generator = Generator()

    noise = torch.randn(4, 100, 1, 1)

    output = generator(noise)

    assert output.min().item() >= -1.0
    assert output.max().item() <= 1.0


def test_generator_preserves_batch_size():
    generator = Generator()

    noise = torch.randn(7, 100, 1, 1)

    output = generator(noise)

    assert output.shape[0] == 7


def test_generator_has_trainable_parameters():
    generator = Generator()

    trainable_parameters = [
        parameter
        for parameter in generator.parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) > 0

def test_generator_backpropagation():
    generator = Generator()

    noise = torch.randn(4, 100, 1, 1)

    output = generator(noise)

    loss = output.mean()

    loss.backward()

    gradients_exist = [
        parameter.grad is not None
        for parameter in generator.parameters()
        if parameter.requires_grad
    ]

    assert all(gradients_exist)