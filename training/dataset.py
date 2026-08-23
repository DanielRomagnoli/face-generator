from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.image_dir}"
            )

        self.image_paths = sorted(self.image_dir.glob("*.jpg"))

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No JPEG images found in: {self.image_dir}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image