import torch


def calculate_entropy(pred: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Calculate the entropy of the prediction."""
    pred = torch.softmax(pred, dim=-3)
    return -torch.sum(pred * torch.log(pred + epsilon), dim=-3)
