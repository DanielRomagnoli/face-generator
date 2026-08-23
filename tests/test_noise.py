import torch

from training.noise import generate_noise

def test_generate_noise_shape():
    device = torch.device("cpu")

    noise = generate_noise(
        batch_size=32,
        latent_dim=100,
        device=device,
    )

    assert noise.shape == (32, 100, 1, 1)


def test_generate_noise_device():
    device = torch.device("cpu")

    noise = generate_noise(
        batch_size=32,
        latent_dim=100,
        device=device,
    )

    assert noise.device.type == "cpu"


def test_generate_noise_is_random():
    device = torch.device("cpu")

    noise_1 = generate_noise(32, 100, device)
    noise_2 = generate_noise(32, 100, device)

    assert not torch.equal(noise_1, noise_2)