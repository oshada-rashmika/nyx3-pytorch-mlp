"""Inference utilities for the trained NYX-3 classifier."""

from pathlib import Path

import torch

from src.model import NYX3


def load_model(checkpoint_path, device):
    """Load NYX-3 weights from a trusted checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run 'python -m src.train' first."
        )

    model = NYX3().to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict_image(model, image, device):
    """Predict one image and return class probabilities."""
    if image.shape != (1, 28, 28):
        raise ValueError(
            "Expected a grayscale tensor with shape (1, 28, 28)."
        )

    model.eval()

    with torch.inference_mode():
        # (1, 28, 28) -> (1, 1, 28, 28)
        batch = image.unsqueeze(0).to(device)

        # Produce 10 class scores
        logits = model(batch)

        # Convert logits to probabilities
        probabilities = torch.softmax(
            logits,
            dim=1,
        ).squeeze(0)

        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()

    return predicted_class, confidence, probabilities.cpu()