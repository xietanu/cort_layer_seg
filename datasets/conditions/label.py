import cort
import datasets.protocols
import json
import torch

AREA_MAPPING = json.load(open("data/preprocessed/area_mapping.json"))


def label_condition(cort_patch: cort.CorticalPatch) -> torch.Tensor:
    return torch.tensor(
        AREA_MAPPING[cort_patch.brain_area],
        dtype=torch.float64,
    )
