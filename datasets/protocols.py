from typing import Protocol

import torch

import cort


class Condition(Protocol):
    def __call__(self, cort_patch: cort.CorticalPatch) -> torch.Tensor:
        """Return the condition for the given patch."""
