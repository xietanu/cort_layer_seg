from dataclasses import dataclass

import torch


@dataclass
class PrerenderedVolume:
    volume: torch.Tensor
    alpha_mask: torch.Tensor
    light_masks: list[torch.Tensor]
    light_dirs: list[torch.Tensor]
    corners: torch.Tensor