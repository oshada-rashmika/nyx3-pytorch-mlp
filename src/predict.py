"""Visualize individual NYX-3 predictions."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torchvision import datasets

from src.data_pipeline import get_transform
from src.inference import load_model, predict_image

MODEL_PATH = Path("models/nyx3_best.pth")
RESULTS_DIR = Path("results/predictions")


def visualize_prediction(
    image,
    actual_label,
    predicted_label,
    confidence,
    probabilities,
    class_names,
    output_path,
):
    """Save the input image and its classification scores."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": [1, 1.8]},
    )

    # Original image
    axes[0].imshow(
        image.squeeze().numpy(),
        cmap="gray",
    )

    axes[0].set_title(
        f"Actual: {class_names[actual_label]}\n"
        f"Predicted: {class_names[predicted_label]}"
    )
    axes[0].axis("off")

    # Confidence scores for all 10 classes
    scores = probabilities.numpy() * 100

    colors = [
        "seagreen" if i == predicted_label else "steelblue"
        for i in range(len(class_names))
    ]

    axes[1].barh(
        class_names,
        scores,
        color=colors,
    )

    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Softmax score (%)")
    axes[1].set_title(
        f"NYX-3 Confidence: {confidence * 100:.2f}%"
    )

    fig.suptitle("NYX-3 | Image Classification")
    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Predict a Fashion-MNIST image using NYX-3."
    )

    parser.add_argument(
        "--index",
        type=int,
        default=42,
        help="Fashion-MNIST test image index (0-9999).",
    )

    args = parser.parse_args()

    if not 0 <= args.index < 10_000:
        parser.error("--index must be between 0 and 9999.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(MODEL_PATH, device)

    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=get_transform(),
    )

    image, actual_label = test_dataset[args.index]
    class_names = test_dataset.classes

    predicted_label, confidence, probabilities = (
        predict_image(model, image, device)
    )

    print("=== NYX-3 Image Prediction ===")
    print(f"Image index: {args.index}")
    print(f"Actual: {class_names[actual_label]}")
    print(f"Predicted: {class_names[predicted_label]}")
    print(f"Confidence: {confidence * 100:.2f}%")

    correct = predicted_label == actual_label
    print(f"Correct: {correct}")

    output_path = (
        RESULTS_DIR / f"prediction_{args.index}.png"
    )

    visualize_prediction(
        image,
        actual_label,
        predicted_label,
        confidence,
        probabilities,
        class_names,
        output_path,
    )

    print(f"Visualization saved: {output_path}")


if __name__ == "__main__":
    main()