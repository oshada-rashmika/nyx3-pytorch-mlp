"""Training and evaluation utilities for NYX-3."""

import torch

LEARNING_RATE = 0.001

def create_optimizer(model):
    """Create an Adam optimizer."""
    return torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

def train_step(model, images, labels, criterion, optimizer):
    """Update model parameters using one training batch."""
    model.train()

    optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item()

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one full epoch and calculate metrics."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Collect training metrics
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples

    return average_loss, accuracy

def evaluate(model, loader, criterion, device):
    """Evaluate without updating model parameters."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()

            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples

    return average_loss, accuracy