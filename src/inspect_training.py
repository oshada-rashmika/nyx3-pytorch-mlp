"""Demonstrate one NYX-3 training step."""

import torch
from torch import nn

from src.data_pipeline import create_dataloaders
from src.model import NYX3
from src.training import create_optimizer, train_step


def main():
    torch.manual_seed(42)

    # Initialize NYX-3
    model = NYX3()

    # Configure classification loss and Adam
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model)

    # Load one batch of real Fashion-MNIST images
    train_loader, _, _ = create_dataloaders()
    images, labels = next(iter(train_loader))

    print("=== NYX-3 Training Inspection ===")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")

    # Save original first-layer weights
    weights_before = model.network[1].weight.detach().clone()

    # Perform one optimization step
    loss = train_step(
        model,
        images,
        labels,
        criterion,
        optimizer,
    )

    # Compare weights before and after training
    weights_after = model.network[1].weight.detach()

    weights_changed = not torch.equal(
        weights_before,
        weights_after,
    )

    print(f"\nTraining loss: {loss:.4f}")
    print(f"Weights changed: {weights_changed}")

    assert torch.isfinite(torch.tensor(loss))
    assert weights_changed

    print("\nSingle training step completed!")

if __name__ == "__main__":
    main()