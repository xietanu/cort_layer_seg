import numpy as np
import torch

import cort


def f1_score(
    pred: torch.Tensor | list[cort.CorticalPatch] | np.ndarray,
    target: torch.Tensor | list[cort.CorticalPatch] | np.ndarray,
    n_classes: int,
    ignore_index: int,
    epsilon: float = 1e-6,
    average_over_classes: bool = True,
) -> float | np.ndarray:
    """Compute the F1 score."""

    if isinstance(pred, list):
        pred = np.stack([p.mask for p in pred])
        target = np.stack([t.mask for t in target])

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
        target = torch.from_numpy(target)

    if len(pred.shape) == 3:
        flat_pred = torch.nn.functional.one_hot(
            pred.flatten().to(torch.int64), num_classes=n_classes + 1
        )
        flat_pred[target.flatten() == ignore_index] = 0
    else:
        raise ValueError(f"Invalid input shape, got {pred.shape}")

    flat_target = torch.nn.functional.one_hot(
        target.flatten().to(torch.int64), num_classes=n_classes + 1
    )

    flat_target[target.flatten() == ignore_index] = 0

    individual_f1_scores = torch.stack(
        [
            single_class_f1_score(flat_pred, flat_target, i, epsilon)
            for i in range(n_classes)
            if i != ignore_index
        ]
    )

    if not average_over_classes:
        return individual_f1_scores.numpy()

    return individual_f1_scores.mean().item()


def single_class_f1_score(
    flat_pred: torch.Tensor,
    flat_target: torch.Tensor,
    class_index: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute the F1 score for a single class."""
    tp = torch.sum(flat_target[:, class_index] * flat_pred[:, class_index])
    fp = torch.sum((1 - flat_target[:, class_index]) * flat_pred[:, class_index])
    fn = torch.sum(flat_target[:, class_index] * (1 - flat_pred[:, class_index]))

    f1 = 2 * tp / (2 * tp + fp + fn + epsilon)

    return f1


def mean_dice(
    pred: torch.Tensor | list[cort.CorticalPatch] | np.ndarray,
    target: torch.Tensor | list[cort.CorticalPatch] | np.ndarray,
    n_classes: int,
    ignore_index: int,
    epsilon: float = 1e-6,
) -> float:
    if isinstance(pred, list):
        pred = np.stack([p.mask for p in pred])
        target = np.stack([t.mask for t in target])

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
        target = torch.from_numpy(target)

    f1_scores = [
        f1_score(pred[None, ...], target[None, ...], n_classes, ignore_index, epsilon)
        for pred, target in zip(pred, target)
    ]

    return np.mean(f1_scores)
