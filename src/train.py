"""Train NYX-3 on Fashion-MNIST."""

import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.data_pipeline import create_dataloaders
from src.model import NYX3
from src.training import (
    create_optimizer,
    evaluate,
    train_one_epoch,
)

EPOCHS = 10
SEED = 42

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


def save_training_plot(history):
    """Save training and validation learning curves."""
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss graph
    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        label="Training",
    )
    axes[0].plot(
        epochs,
        [row["val_loss"] for row in history],
        label="Validation",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy graph
    axes[1].plot(
        epochs,
        [row["train_accuracy"] for row in history],
        label="Training",
    )
    axes[1].plot(
        epochs,
        [row["val_accuracy"] for row in history],
        label="Validation",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("NYX-3 Training Performance")
    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "training_curves.png",
        dpi=150,
    )
    plt.close(fig)


def save_training_history(history):
    """Export epoch metrics to CSV."""
    path = RESULTS_DIR / "training_history.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ],
        )

        writer.writeheader()
        writer.writerows(history)


def main():
    # Reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Select CPU or CUDA
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== NYX-3 Model Training ===")
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}")

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load training and validation data.
    # The test set is reserved for Step 13.
    train_loader, val_loader, _ = create_dataloaders()

    # Initialize model
    model = NYX3().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model)

    best_val_accuracy = -1.0
    best_epoch = 0
    history = []

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        # Training phase
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # Validation phase
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        # Record results
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                MODEL_DIR / "nyx3_best.pth",
            )

        # Display epoch metrics
        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    # Export training metrics and graphs
    save_training_history(history)
    save_training_plot(history)

    print("\n=== Training Completed ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print("Model: models/nyx3_best.pth")
    print("History: results/training_history.csv")
    print("Graph: results/training_curves.png")


if __name__ == "__main__":
    main()