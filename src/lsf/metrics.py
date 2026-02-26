"""Regression metrics."""

import torch
from torch import Tensor


def mse(preds: Tensor, targets: Tensor) -> float:
    """Mean Squared Error."""
    return torch.mean((preds - targets) ** 2).item()


def mae(preds: Tensor, targets: Tensor) -> float:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(preds - targets)).item()


def r2_score(preds: Tensor, targets: Tensor) -> float:
    """Coefficient of determination R²."""
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return (1 - ss_res / ss_tot).item()
