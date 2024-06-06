import torch


def send_to_device(
    inputs: torch.Tensor | tuple[torch.Tensor, ...], device: torch.device
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Send outputs to the device."""
    if isinstance(inputs, (tuple, list)):
        return tuple(send_to_device(i, device) for i in inputs)  # type: ignore
    else:
        return inputs.to(device)
