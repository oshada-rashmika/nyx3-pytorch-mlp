"""Fashion-MNIST preprocessing and data loading for NYX-3."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2

BATCH_SIZE = 64
RANDOM_SEED = 42

TRAIN_SIZE = 54_000
VAL_SIZE = 6_000


def get_transform():
    """Convert grayscale images into float tensors."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def create_dataloaders(batch_size=BATCH_SIZE):
    """Create training, validation, and test DataLoaders."""

    transform = get_transform()

    # Load the original training dataset.
    full_train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    # Load the separate test dataset.
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    # Reproducible train/validation split.
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [TRAIN_SIZE, VAL_SIZE],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    # Create batches.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def main():
    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    print("=== NYX-3 Data Pipeline ===")

    print(f"Training images: {len(train_loader.dataset):,}")
    print(f"Validation images: {len(val_loader.dataset):,}")
    print(f"Testing images: {len(test_loader.dataset):,}")

    # Retrieve one batch for inspection.
    images, labels = next(iter(train_loader))

    print("\n=== Batch Inspection ===")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Minimum pixel value: {images.min().item():.4f}")
    print(f"Maximum pixel value: {images.max().item():.4f}")

    print("\nData pipeline ready!")


if __name__ == "__main__":
    main()