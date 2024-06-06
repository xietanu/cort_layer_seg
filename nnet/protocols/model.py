from typing import Protocol

import torch
import numpy as np

import datasets
import nnet.dataclasses


class ModelProtocol(Protocol):
    step: int
    device: torch.device

    def train_one_step(
        self,
        inputs: datasets.datatypes.DataInputs,
        ground_truths: datasets.datatypes.GroundTruths,
    ) -> tuple[float, float, float]:  # type: ignore
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
        inputs: datasets.datatypes.DataInputs,
        ground_truths: datasets.datatypes.GroundTruths,
    ) -> tuple[float, float, float]:  # type: ignore
        """Validate the model."""

    def predict(
        self,
        inputs: datasets.datatypes.DataInputs,
    ) -> datasets.datatypes.Predictions:
        """Predict the outputs for given outputs."""
