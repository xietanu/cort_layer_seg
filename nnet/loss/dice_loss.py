import torch


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute the Dice loss."""
    n_classes = inputs.shape[1]

    flat_inputs = inputs.permute(0, 2, 3, 1).reshape(-1, n_classes)
    flat_inputs[targets.flatten() == ignore_index] = 0
    flat_inputs = torch.nn.functional.softmax(flat_inputs, dim=1)

    flat_targets = torch.nn.functional.one_hot(
        targets.flatten().to(torch.int64), num_classes=n_classes
    )

    flat_targets[targets.flatten() == ignore_index] = 0

    loss = (
        -2
        / n_classes
        * torch.sum(
            torch.sum(flat_targets * flat_inputs, dim=0)
            / (
                torch.sum(flat_targets, dim=0) + torch.sum(flat_inputs, dim=0) + epsilon
            ),
            dim=0,
        )
    )

    return loss
