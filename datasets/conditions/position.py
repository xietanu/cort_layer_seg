import cort
import datasets
import nnet.modules.conditional
import torch

EMBED_DIM = 48


def pos_condition(patch: cort.CorticalPatch) -> torch.Tensor:
    return torch.tensor(
        [
            patch.x,
            patch.y,
            patch.z,
        ],
        dtype=torch.float64,
    )
