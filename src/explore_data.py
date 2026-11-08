"""Download and explore the Fashion-MNIST dataset."""

import matplotlib.pyplot as plt
import torch
from torchvision import datasets


def main():
    # Download the training dataset.
    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
    )

    # Download the test dataset.
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
    )

    # Explore the dataset.
    print("=== NYX-3 Dataset Exploration ===")
    print(f"Training images: {len(train_dataset):,}")
    print(f"Testing images: {len(test_dataset):,}")
    print(f"Classes: {train_dataset.classes}")

    # Inspect the first training example.
    image, label = train_dataset[0]

    print("\n=== First Image ===")
    print(f"Image type: {type(image).__name__}")
    print(f"Image dimensions: {image.size}")
    print(f"Class ID: {label}")
    print(f"Class name: {train_dataset.classes[label]}")

    # Inspect how many examples belong to each class.
    print("\n=== Class Distribution ===")
    counts = torch.bincount(
        train_dataset.targets,
        minlength=len(train_dataset.classes),
    )

    for class_name, count in zip(
        train_dataset.classes,
        counts.tolist(),
    ):
        print(f"{class_name}: {count:,}")

    # Visualize 10 example images.
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for index, ax in enumerate(axes.flat):
        image, label = train_dataset[index]

        ax.imshow(image, cmap="gray")
        ax.set_title(train_dataset.classes[label])
        ax.axis("off")

    fig.suptitle("NYX-3 | Fashion-MNIST Samples")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()