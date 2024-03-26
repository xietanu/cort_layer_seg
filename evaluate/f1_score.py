import torch


def f1_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    softmax_based: bool = False,
    epsilon: float = 1e-6,
) -> float:
    """Compute the F1 score."""
    n_classes = pred.shape[1]

    flat_pred = pred.permute(0, 2, 3, 1).reshape(-1, n_classes)
    flat_pred[target.flatten() == ignore_index] = 0

    if not softmax_based:
        flat_pred = torch.nn.functional.one_hot(
            torch.argmax(flat_pred, dim=1), num_classes=n_classes
        )

    flat_target = torch.nn.functional.one_hot(
        target.flatten().to(torch.int64), num_classes=n_classes
    )

    flat_target[target.flatten() == ignore_index] = 0

    individual_f1_scores = torch.stack(
        [
            single_class_f1_score(flat_pred, flat_target, i, epsilon)
            for i in range(n_classes)
            if i != ignore_index
        ]
    )

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
