from enum import Enum, auto
from typing import Protocol

import torch


class LossType(Enum):
    SEGMENTATION = auto()
    DEPTH = auto()
    DENOISE = auto()
    ACCURACY_ESTIMATE = auto()
    AUTOENCODE = auto()
    CONSISTENCY = auto()


class Loss(Protocol):
    @property
    def total(self) -> torch.Tensor:
        """Return the total loss."""

    def add(self, loss_type: LossType, loss: torch.Tensor, alpha: float = 1.0) -> None:
        """Add a loss to the total loss."""

    def __getitem__(self, loss_type: LossType) -> torch.Tensor:
        """Return the loss of the given type."""

    def to_dict(self) -> dict[str, float]:
        """Return the loss as a dictionary."""

    def __add__(self, other):
        """Add two losses together."""

    def detach(self):
        """Detach the loss."""

    def cpu(self):
        """Move the loss to the CPU."""

    def backward(self):
        """Backward the loss."""
