"""Tests for full-epoch training and evaluation."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import NYX3
from src.training import (
    create_optimizer,
    evaluate,
    train_one_epoch,
)


def create_test_loader():
    """Create synthetic batches without dataset downloads."""
    torch.manual_seed(42)

    images = torch.rand(12, 1, 28, 28)
    labels = torch.randint(0, 10, (12,))

    dataset = TensorDataset(images, labels)

    return DataLoader(dataset, batch_size=4)


def test_train_one_epoch():
    """Verify epoch training produces valid metrics."""
    model = NYX3()
    loader = create_test_loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model)

    loss, accuracy = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
    )

    assert loss >= 0
    assert torch.isfinite(torch.tensor(loss))
    assert 0 <= accuracy <= 100


def test_evaluate_does_not_update_weights():
    """Evaluation must preserve the model's parameters."""
    model = NYX3()
    loader = create_test_loader()

    criterion = nn.CrossEntropyLoss()

    weights_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    loss, accuracy = evaluate(
        model,
        loader,
        criterion,
        torch.device("cpu"),
    )

    weights_after = list(model.parameters())

    assert loss >= 0
    assert 0 <= accuracy <= 100

    for before, after in zip(
        weights_before,
        weights_after,
        strict=True,
    ):
        assert torch.equal(before, after)