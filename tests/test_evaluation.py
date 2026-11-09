"""Automated tests for NYX-3 model evaluation."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import (
    calculate_metrics,
    collect_predictions,
)
from src.model import NYX3


def test_perfect_predictions():
    """Perfect predictions should produce perfect metrics."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]

    class_names = ["Shirt", "Coat", "Sneaker"]

    metrics = calculate_metrics(
        y_true,
        y_pred,
        class_names,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"].shape == (3, 3)


def test_incorrect_predictions():
    """Incorrect predictions should reduce accuracy."""
    y_true = [0, 1, 2, 0]
    y_pred = [0, 2, 2, 0]

    class_names = ["Shirt", "Coat", "Sneaker"]

    metrics = calculate_metrics(
        y_true,
        y_pred,
        class_names,
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] < 1.0


def test_prediction_collection():
    """Verify predictions can be collected from a loader."""
    torch.manual_seed(42)

    model = NYX3()

    images = torch.rand(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    y_true, y_pred = collect_predictions(
        model,
        loader,
        torch.device("cpu"),
    )

    assert len(y_true) == 8
    assert len(y_pred) == 8

    assert all(0 <= label < 10 for label in y_pred)