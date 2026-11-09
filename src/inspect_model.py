"""Inspect the NYX-3 neural network architecture."""

import torch
from torch import nn

from src.model import NYX3


def main():
    model = NYX3()

    print("=== NYX-3 Architecture ===")
    print(model)

    # Count only trainable Linear layers.
    linear_layers = [
        layer
        for layer in model.modules()
        if isinstance(layer, nn.Linear)
    ]

    print(f"\nTrainable Linear Layers: {len(linear_layers)}")

    # Count all trainable parameters.
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable Parameters: {parameter_count:,}")

    # Simulate a batch of 64 grayscale images.
    sample_images = torch.rand(64, 1, 28, 28)

    # Perform a forward pass.
    with torch.no_grad():
        outputs = model(sample_images)

    print(f"\nInput Shape: {sample_images.shape}")
    print(f"Output Shape: {outputs.shape}")

if __name__ == "__main__":
    main()