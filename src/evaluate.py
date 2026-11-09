"""Evaluate the trained NYX-3 model."""

from pathlib import Path

import torch
from torchvision.datasets import FashionMNIST

from src.data_pipeline import create_dataloaders
from src.evaluation import (
    calculate_metrics,
    collect_predictions,
    save_confusion_matrix,
)
from src.model import NYX3

MODEL_PATH = Path("models/nyx3_best.pth")
RESULTS_DIR = Path("results")


def main():
    print("=== NYX-3 Model Evaluation ===")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run 'python -m src.train' first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # Load the saved architecture and trained weights.
    model = NYX3().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("Best model checkpoint loaded.")

    # Load the held-out test dataset.
    _, _, test_loader = create_dataloaders()

    class_names = FashionMNIST.classes

    print(f"Test images: {len(test_loader.dataset):,}")

    # Generate predictions.
    y_true, y_pred = collect_predictions(
        model,
        test_loader,
        device,
    )

    # Calculate classification metrics.
    metrics = calculate_metrics(
        y_true,
        y_pred,
        class_names,
    )

    print("\n=== Overall Performance ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    print("\n=== Classification Report ===")
    print(metrics["report"])

    # Save evaluation results.
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = RESULTS_DIR / "classification_report.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write("NYX-3 | Fashion-MNIST Evaluation\n\n")
        file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        file.write(
            f"Macro Precision: "
            f"{metrics['macro_precision']:.4f}\n"
        )
        file.write(
            f"Macro Recall: {metrics['macro_recall']:.4f}\n"
        )
        file.write(
            f"Macro F1: {metrics['macro_f1']:.4f}\n\n"
        )
        file.write(metrics["report"])

    save_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        RESULTS_DIR / "confusion_matrix.png",
    )

    print("\n=== Evaluation Complete ===")
    print(f"Report: {report_path}")
    print("Matrix: results/confusion_matrix.png")


if __name__ == "__main__":
    main()