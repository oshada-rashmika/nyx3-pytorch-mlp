"""Tests for NYX-3 image preprocessing and batching."""

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.data_pipeline import get_transform


def test_image_preprocessing():
    """Images should become scaled float tensors."""
    image = Image.new("L", (28, 28), color=255)

    tensor = get_transform()(image)

    assert tensor.shape == (1, 28, 28)
    assert tensor.dtype == torch.float32
    assert torch.all(tensor == 1.0)


def test_batch_shapes():
    """DataLoader should group images and labels."""
    images = torch.zeros(10, 1, 28, 28)
    labels = torch.arange(10)

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4)

    batch_images, batch_labels = next(iter(loader))

    assert batch_images.shape == (4, 1, 28, 28)
    assert batch_labels.shape == (4,)