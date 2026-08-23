from torch.utils.data import DataLoader

from training.config import BATCH_SIZE
from training.dataset import CelebADataset
from training.transforms import get_image_transform


IMAGE_DIR = "data/celeba/img_align_celeba"


def get_dataloader(
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    dataset = CelebADataset(
        IMAGE_DIR,
        transform=get_image_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )