"""Training utilities for NYX-3."""

import torch

LEARNING_RATE = 0.001

def create_optimizer(model):
    """Create an Adam optimizer for the model."""
    return torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

def train_step(model, images, labels, criterion, optimizer):
    """Train the model on a single batch of images."""
    model.train()

    # Clear gradients from previous steps
    optimizer.zero_grad()

    # Forward pass: generate class scores
    outputs = model(images)

    # Calculate classification loss
    loss = criterion(outputs, labels)

    # Backward pass: calculate gradients
    loss.backward()

    # Update trainable parameters
    optimizer.step()

    return loss.item()