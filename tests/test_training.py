"""Tests for NYX-3 training operations."""

import torch
from torch import nn

from src.model import NYX3
from src.training import create_optimizer, train_step


def test_optimizer_configuration():
    """Verify Adam is configured correctly."""
    model = NYX3()
    optimizer = create_optimizer(model)

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 0.001


def test_cross_entropy_loss():
    """Verify classification loss is finite."""
    criterion = nn.CrossEntropyLoss()

    logits = torch.tensor([
        [2.0, 0.5, -1.0],
        [0.1, 1.5, 0.2],
    ])

    labels = torch.tensor([0, 1])

    loss = criterion(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_training_step_updates_weights():
    """Verify training modifies model parameters."""
    torch.manual_seed(42)

    model = NYX3()
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model)

    images = torch.rand(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))

    weights_before = model.network[1].weight.detach().clone()

    loss = train_step(
        model,
        images,
        labels,
        criterion,
        optimizer,
    )

    weights_after = model.network[1].weight.detach()

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(weights_before, weights_after)