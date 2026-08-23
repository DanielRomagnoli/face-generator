from training.generator import Generator
from training.discriminator import Discriminator
from training.weights import initialize_weights


def test_generator_weight_initialization():
    generator = Generator()

    generator.apply(initialize_weights)

    first_conv = generator.network[0]

    mean = first_conv.weight.mean().item()
    std = first_conv.weight.std().item()

    assert abs(mean) < 0.01
    assert 0.015 < std < 0.025

def test_discriminator_weight_initialization():
    discriminator = Discriminator()

    discriminator.apply(initialize_weights)

    first_conv = discriminator.network[0]

    mean = first_conv.weight.mean().item()
    std = first_conv.weight.std().item()

    assert abs(mean) < 0.01
    assert 0.015 < std < 0.025