import torch


def tversky_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute the Dice loss."""
    n_classes = outputs.shape[1]

    flat_inputs = outputs.permute(0, 2, 3, 1).reshape(-1, n_classes)
    flat_inputs[targets.flatten() == ignore_index] = 0
    flat_inputs = torch.nn.functional.softmax(flat_inputs, dim=1)

    flat_targets = torch.nn.functional.one_hot(
        targets.flatten().to(torch.int64), num_classes=n_classes + 1
    )
    flat_targets = flat_targets[:, :n_classes]

    flat_targets[targets.flatten() == ignore_index] = 0

    loss = -torch.sum(
        torch.sum(flat_targets * flat_inputs, dim=0)
        / (
            torch.sum(flat_targets * flat_inputs, dim=0)
            + 0.5 * torch.sum(flat_targets * (1 - flat_inputs), dim=0)
            + 0.5 * torch.sum((1 - flat_targets) * flat_inputs, dim=0)
            + epsilon
        ),
        dim=0,
    ) / (n_classes - 1)

    return loss
