"""Evaluation utilities for the NYX-3 classifier."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def collect_predictions(model, loader, device):
    """Collect true and predicted labels from a DataLoader."""
    model.eval()

    all_true = []
    all_pred = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(predictions.cpu().tolist())

    return all_true, all_pred


def calculate_metrics(y_true, y_pred, class_names):
    """Calculate classification metrics."""
    labels = list(range(len(class_names)))

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "report": report,
        "confusion_matrix": matrix,
    }


def save_confusion_matrix(matrix, class_names, output_path):
    """Save a confusion matrix visualization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False,
    )

    ax.set_title("NYX-3 | Fashion-MNIST Confusion Matrix")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)