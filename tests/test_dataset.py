from pathlib import Path

from PIL import Image

from training.dataset import CelebADataset
from training.transforms import get_image_transform

IMAGE_DIR = "data/celeba/img_align_celeba"


def test_dataset_length():
    dataset = CelebADataset(IMAGE_DIR)

    assert len(dataset) == 202599


def test_dataset_first_image_is_rgb():
    dataset = CelebADataset(IMAGE_DIR)

    image = dataset[0]

    assert image.mode == "RGB"


def test_dataset_first_image_size():
    dataset = CelebADataset(IMAGE_DIR)

    image = dataset[0]

    assert image.size == (178, 218)


def test_dataset_index_matches_expected_filename():
    dataset = CelebADataset(IMAGE_DIR)

    first_path = dataset.image_paths[0]
    last_path = dataset.image_paths[-1]

    assert first_path.name == "000001.jpg"
    assert last_path.name == "202599.jpg"


def test_dataset_returns_pil_image():
    dataset = CelebADataset(IMAGE_DIR)

    image = dataset[0]

    assert isinstance(image, Image.Image)

def test_transformed_image_shape():
    dataset = CelebADataset(
        IMAGE_DIR,
        transform=get_image_transform(),
    )

    image = dataset[0]

    assert image.shape == (3, 64, 64)


def test_transformed_image_range():
    dataset = CelebADataset(
        IMAGE_DIR,
        transform=get_image_transform(),
    )

    image = dataset[0]

    assert image.min().item() >= -1.0
    assert image.max().item() <= 1.0