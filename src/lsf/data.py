"""Synthetic dataset generation."""

from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def make_regression_data(
    n_samples: int = 1000,
    n_features: int = 1,
    noise: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Tensor, Tensor]:
    """Generate a synthetic linear regression dataset.

    Returns
    -------
    train_loader, test_loader, true_weights, true_bias
    """
    rng = torch.Generator().manual_seed(seed)

    X = torch.randn(n_samples, n_features, generator=rng)
    true_w = torch.randn(n_features, 1, generator=rng)
    true_b = torch.randn(1, generator=rng)
    noise_tensor = noise * torch.randn(n_samples, 1, generator=rng)
    y = X @ true_w + true_b + noise_tensor

    n_test = int(n_samples * test_size)
    X_train, X_test = X[n_test:], X[:n_test]
    y_train, y_test = y[n_test:], y[:n_test]

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=rng)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    return train_loader, test_loader, true_w, true_b
