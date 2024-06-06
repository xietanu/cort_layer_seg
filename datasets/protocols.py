from dataclasses import dataclass
from typing import Protocol

import torch
import torch.utils.data

import cort


class Condition(Protocol):
    def __call__(self, cort_patch: cort.CorticalPatch) -> torch.Tensor:
        """Return the condition for the given patch."""


@dataclass
class Fold:
    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader
    test_dataset: torch.utils.data.Dataset
    test_dataloader: torch.utils.data.DataLoader
    siibra_test_dataset: torch.utils.data.Dataset
    siibra_test_dataloader: torch.utils.data.DataLoader
