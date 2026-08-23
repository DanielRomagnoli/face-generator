from torchvision import transforms

from training.config import IMAGE_SIZE


def get_image_transform():
    return transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )