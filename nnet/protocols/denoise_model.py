from typing import Protocol

import torch
import numpy as np

import datasets


class DenoiseModelProtocol(Protocol):
    step: int
    device: torch.device

    def train_one_step(
        self,
        logits: torch.Tensor,
        gt_segmentation: torch.Tensor,
        probs: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
    ) -> tuple[float, float]:  # type: ignore
        """Train the model on one step."""

    def save(self, path: str):
        """Save the model to a file."""

    @classmethod
    def restore(cls, path: str):
        """Restore the model from a file."""

    def load(self, path: str):
        """Load the model from a file."""

    def validate(
        self,
        logits: torch.Tensor,
        gt_segmentation: torch.Tensor,
        probs: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
    ) -> tuple[float, float]:  # type: ignore
        """Validate the model."""

    def predict(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the outputs for given outputs."""
