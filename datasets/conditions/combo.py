import cort
import datasets
import nnet.modules.conditional
import torch


def combo_condition(patch: cort.CorticalPatch) -> torch.Tensor:
    return torch.cat(
        (
            torch.tensor(patch.region_probs, dtype=torch.float64),
            torch.tensor([patch.x, patch.y, patch.z], dtype=torch.float64),
        )
    )
