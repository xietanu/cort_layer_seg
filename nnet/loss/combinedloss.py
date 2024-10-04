from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import nnet.protocols

import torch


@dataclass
class LossItem:
    raw_value: torch.Tensor
    alpha: float = 1.0

    @property
    def value(self) -> torch.Tensor:
        return self.raw_value * self.alpha


class CombinedLoss(nnet.protocols.Loss):
    def __init__(self, losses: dict[nnet.protocols.LossType, LossItem] = None):
        if losses is None:
            losses = {}
        self.losses = losses

    @property
    def total(self) -> torch.Tensor:
        """Return the total loss."""
        if len(self.losses) == 0:
            return torch.tensor(0.0)
        return torch.sum(torch.stack([loss.value for loss in self.losses.values()]))

    def add(
        self, loss_type: nnet.protocols.LossType, loss: torch.Tensor, alpha: float = 1.0
    ) -> None:
        """Add a loss to the total loss."""
        self.losses[loss_type] = LossItem(loss, alpha)

    def __getitem__(self, loss_type: nnet.protocols.LossType) -> torch.Tensor:
        """Return the loss of the given type."""
        if loss_type not in self.losses:
            return torch.tensor(0.0)
        return self.losses[loss_type].raw_value

    def to_dict(self) -> dict[str, float]:
        """Return the loss as a dictionary."""
        output = {
            loss_type.name.lower(): loss.raw_value.item()
            for loss_type, loss in self.losses.items()
        }
        if len(self.losses) > 1:
            output["total"] = self.total.item()
        return output

    def __add__(self, other):
        """Add two losses together."""
        new_losses = {}
        for loss_type, loss in self.losses.items():
            new_losses[loss_type] = loss
        for loss_type, loss in other.losses.items():
            if loss_type in new_losses:
                new_losses[loss_type] = LossItem(
                    new_losses[loss_type].raw_value + loss.raw_value,
                    new_losses[loss_type].alpha,
                )
            else:
                new_losses[loss_type] = loss
        return CombinedLoss(new_losses)

    def detach(self) -> CombinedLoss:
        """Detach the loss."""
        new_losses = {}
        for loss_type, loss in self.losses.items():
            new_losses[loss_type] = LossItem(loss.raw_value.detach(), loss.alpha)
        return CombinedLoss(new_losses)

    def cpu(self):
        """Move the loss to the CPU."""
        new_losses = {}
        for loss_type, loss in self.losses.items():
            new_losses[loss_type] = LossItem(loss.raw_value.cpu(), loss.alpha)
        return CombinedLoss(new_losses)

    def backward(self):
        """Backward the loss."""
        self.total.backward()

    def __lt__(self, other):
        if isinstance(other, CombinedLoss):
            return self.total < other.total
        return self.total < other

    def __gt__(self, other):
        if isinstance(other, CombinedLoss):
            return self.total > other.total
        return self.total > other

    def __le__(self, other):
        if isinstance(other, CombinedLoss):
            return self.total <= other.total
        return self.total <= other

    def __ge__(self, other):
        if isinstance(other, CombinedLoss):
            return self.total >= other.total
        return self.total >= other

    def __eq__(self, other):
        if isinstance(other, CombinedLoss):
            return self.total == other.total
        return self.total == other

    def __ne__(self, other):
        if isinstance(other, CombinedLoss):
            return self.total != other.total
        return self.total != other

    @classmethod
    def mean(cls, losses: list[CombinedLoss]) -> CombinedLoss:
        """Return the mean of the losses."""
        new_losses = {}
        for loss in losses:
            for loss_type, loss_item in loss.losses.items():
                if loss_type in new_losses:
                    new_losses[loss_type] = LossItem(
                        new_losses[loss_type].raw_value + loss_item.raw_value,
                        new_losses[loss_type].alpha,
                    )
                else:
                    new_losses[loss_type] = loss_item
        for loss_type, loss_item in new_losses.items():
            new_losses[loss_type] = LossItem(
                loss_item.raw_value / len(losses), loss_item.alpha
            )
        return CombinedLoss(new_losses)
