import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from training.dataloader import get_dataloader


def main() -> None:
    dataloader = get_dataloader(
        batch_size=64,
        shuffle=True,
    )

    batch = next(iter(dataloader))

    grid = make_grid(
        batch,
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()