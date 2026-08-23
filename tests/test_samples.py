import torch

from training.generator import Generator
from training.samples import save_generated_samples


def test_save_generated_samples_creates_file(tmp_path):
    generator = Generator()

    fixed_noise = torch.randn(
        8,
        100,
        1,
        1,
    )

    path = tmp_path / "samples.png"

    save_generated_samples(
        generator=generator,
        fixed_noise=fixed_noise,
        path=str(path),
    )

    assert path.exists()
    assert path.stat().st_size > 0