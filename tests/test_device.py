import torch

from training.device import get_device

def test_get_device_returns_torch_device():
    device = get_device()

    assert isinstance(device, torch.device)

def test_get_device_returns_supported_device():
    device = get_device()

    assert device.type in {"mps", "cuda", "cpu"}

def test_get_device_prefers_mps_when_available():
    if torch.backends.mps.is_available():
        assert get_device().type == "mps"