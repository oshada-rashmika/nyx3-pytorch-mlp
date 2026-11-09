"""Automated tests for the NYX-3 MLP architecture."""

import torch
from torch import nn

from src.model import NYX3

def test_model_has_three_trainable_layers():
    """NYX-3 must contain exactly three Linear layers."""
    model = NYX3()

    linear_layers = [
        layer
        for layer in model.modules()
        if isinstance(layer, nn.Linear)
    ]

    assert len(linear_layers) == 3

    assert [
        (layer.in_features, layer.out_features)
        for layer in linear_layers
    ] == [
        (784, 256),
        (256, 128),
        (128, 10),
    ]

def test_model_parameter_count():
    """Verify the total number of trainable parameters."""
    model = NYX3()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert total_parameters == 235_146

def test_model_output_shape():
    """A batch of 64 images should produce 64 class-score vectors."""
    model = NYX3()

    images = torch.rand(64, 1, 28, 28)

    outputs = model(images)

    assert outputs.shape == (64, 10)

def test_model_single_image():
    """The model should support a batch containing one image."""
    model = NYX3()

    image = torch.rand(1, 1, 28, 28)

    outputs = model(image)

    assert outputs.shape == (1, 10)

def test_model_backward_pass():
    """Gradients should reach all trainable parameters."""
    model = NYX3()

    images = torch.rand(8, 1, 28, 28)

    outputs = model(images)
    loss = outputs.square().mean()

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()