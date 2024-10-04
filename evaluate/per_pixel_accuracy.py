import numpy as np
import torch

import cort


def per_pixel_accuracy(
    pred: torch.Tensor | list[cort.CorticalPatch] | np.ndarray,
    target: torch.Tensor | list[cort.CorticalPatch] | np.ndarray,
    ignore_index: int,
    epsilon: float = 1e-6,
) -> float:
    """Compute the per-pixel accuracy."""

    if isinstance(pred, list):
        pred = np.array([patch.mask for patch in pred])
    if isinstance(target, list):
        target = np.array([patch.mask for patch in target])

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
        target = torch.from_numpy(target)

    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    if len(pred.shape) != 3:
        raise ValueError(f"Invalid input shape, got {pred.shape}")

    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes do not match, got {pred.shape} and {target.shape}"
        )

    acc_mask = (target != ignore_index) & (pred == target)
    acc = acc_mask.float().sum() / ((target != ignore_index).float().sum() + epsilon)

    return acc.item()
