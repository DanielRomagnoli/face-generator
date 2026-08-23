from training.dataloader import get_dataloader
import torch

def test_dataloader_batch_shape():
    dataloader = get_dataloader(
        batch_size=16,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch.shape == (16, 3, 64, 64)


def test_dataloader_batch_dtype():
    dataloader = get_dataloader(
        batch_size=16,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch.dtype.is_floating_point


def test_dataloader_batch_range():
    dataloader = get_dataloader(
        batch_size=16,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch.min().item() >= -1.0
    assert batch.max().item() <= 1.0


def test_dataloader_dataset_size():
    dataloader = get_dataloader(
        batch_size=16,
        shuffle=False,
    )

    assert len(dataloader.dataset) == 202599

def test_dataloader_contains_only_finite_values():
    dataloader = get_dataloader(
        batch_size=16,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert torch.isfinite(batch).all()