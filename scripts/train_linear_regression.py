#!/usr/bin/env python
"""Train a linear regression model on synthetic data."""

import argparse
import os

import torch
import torch.nn as nn

from lsf import (
    Config,
    LinearRegression,
    evaluate,
    make_regression_data,
    mae,
    mse,
    r2_score,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear regression with PyTorch")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise=args.noise,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, true_w, true_b = make_regression_data(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        noise=cfg.noise,
        test_size=cfg.test_size,
        seed=cfg.seed,
    )

    model = LinearRegression(in_features=cfg.n_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    print(f"Training for {cfg.epochs} epochs …")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{cfg.epochs}  train_loss={train_loss:.6f}")

    _, preds, targets = evaluate(model, test_loader, criterion, device)
    print("\n── Test metrics ──")
    print(f"  MSE : {mse(preds, targets):.6f}")
    print(f"  MAE : {mae(preds, targets):.6f}")
    print(f"  R²  : {r2_score(preds, targets):.4f}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, "linear_regression.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    print("\nLearned parameters vs. true parameters:")
    learned_w = model.linear.weight.detach().cpu()
    learned_b = model.linear.bias.detach().cpu()
    print(f"  weight  learned={learned_w.squeeze().item():.4f}  "
          f"true={true_w.squeeze().item():.4f}")
    print(f"  bias    learned={learned_b.item():.4f}  "
          f"true={true_b.item():.4f}")


if __name__ == "__main__":
    main()
