"""Model definitions."""

import torch.nn as nn
from torch import Tensor


class LinearRegression(nn.Module):
    """Single-layer linear regression: y = X W + b."""

    def __init__(self, in_features: int = 1, out_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
