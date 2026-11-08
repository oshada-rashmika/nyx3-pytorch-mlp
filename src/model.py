"""NYX-3: A three-layer MLP image classifier."""

from torch import nn


class NYX3(nn.Module):
    """Multi-Layer Perceptron for Fashion-MNIST."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # Convert each image into 784 input features.
            nn.Flatten(),

            # Trainable Layer 1
            nn.Linear(784, 256),
            nn.ReLU(),

            # Trainable Layer 2
            nn.Linear(256, 128),
            nn.ReLU(),

            # Trainable Layer 3
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """Perform a forward pass through the MLP."""
        return self.network(x)