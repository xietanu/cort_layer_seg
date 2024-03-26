from typing import Protocol

import torch
import numpy as np


class ModelProtocol(Protocol):
    step: int
    device: torch.device

    def train_one_step(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        targets: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> tuple[float, float]:  # type: ignore
        """Train the model on one step."""

    def save(self, path: str):
        """Save the model to a file."""

    @classmethod
    def restore(cls, path: str):
        """Restore the model from a file."""

    def validate(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        targets: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> tuple[float, float]:  # type: ignore
        """Validate the model."""

    def predict(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> np.ndarray:
        """Predict the outputs for given inputs."""
