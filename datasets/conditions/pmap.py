import cort
import datasets
import nnet.modules.conditional
import torch


def pmap_condition(patch: cort.CorticalPatch) -> torch.Tensor:
    return torch.tensor(patch.region_probs, dtype=torch.float64)
