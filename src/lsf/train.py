"""Training utilities."""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Run one full pass over *loader* and return the mean training loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds: Tensor = model(X_batch)
        loss: Tensor = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Tensor, Tensor]:
    """Evaluate *model* on *loader*.

    Returns
    -------
    mean_loss, all_predictions, all_targets
    """
    model.eval()
    preds_list = []
    targets_list = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item()
            preds_list.append(preds.cpu())
            targets_list.append(y_batch.cpu())
            n_batches += 1

    all_preds = torch.cat(preds_list)
    all_targets = torch.cat(targets_list)
    return total_loss / max(n_batches, 1), all_preds, all_targets
