"""Experiment configuration."""

from dataclasses import dataclass, field


@dataclass
class Config:
    """Hyper-parameters and settings for a linear-regression experiment."""

    # Data
    n_samples: int = 1000
    n_features: int = 1
    noise: float = 0.1
    test_size: float = 0.2

    # Training
    lr: float = 0.01
    epochs: int = 100
    batch_size: int = 32

    # Reproducibility
    seed: int = 42

    # I/O
    output_dir: str = "outputs"
    run_name: str = "run"

    # Derived (computed after init)
    n_train: int = field(init=False)
    n_test: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_test = int(self.n_samples * self.test_size)
        self.n_train = self.n_samples - self.n_test
