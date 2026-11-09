"""Automated tests for NYX-3 inference."""

import pytest
import torch

from src.inference import load_model, predict_image
from src.model import NYX3


def test_prediction_output():
    """Prediction should return a valid class and confidence."""
    torch.manual_seed(42)

    model = NYX3()
    image = torch.rand(1, 28, 28)

    predicted_class, confidence, probabilities = (
        predict_image(
            model,
            image,
            torch.device("cpu"),
        )
    )

    assert 0 <= predicted_class < 10
    assert 0 <= confidence <= 1
    assert probabilities.shape == (10,)

    assert torch.isclose(
        probabilities.sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )

    assert predicted_class == probabilities.argmax().item()


def test_invalid_image_shape():
    """Reject incorrectly shaped images."""
    model = NYX3()
    image = torch.rand(3, 28, 28)

    with pytest.raises(ValueError):
        predict_image(
            model,
            image,
            torch.device("cpu"),
        )


def test_load_checkpoint(tmp_path):
    """Verify model weights can be saved and reloaded."""
    torch.manual_seed(42)

    original_model = NYX3()

    checkpoint = tmp_path / "test_model.pth"

    torch.save(
        original_model.state_dict(),
        checkpoint,
    )

    loaded_model = load_model(
        checkpoint,
        torch.device("cpu"),
    )

    for original, loaded in zip(
        original_model.parameters(),
        loaded_model.parameters(),
        strict=True,
    ):
        assert torch.equal(original, loaded)

    assert not loaded_model.training