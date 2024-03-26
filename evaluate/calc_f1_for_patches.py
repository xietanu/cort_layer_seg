import torch
import numpy as np
import json

import cort
import nnet
import nnet.models
import evaluate

AREA_MAPPING_PATH = "data/cort_patches/area_mapping.json"


def calc_f1_for_patches(
    model: nnet.protocols.ModelProtocol,
    patches: list[cort.CorticalPatch],
    conditional: bool = False,
    batch_size: int = 4,
    ignore_index: int = cort.constants.PADDING_MASK_VALUE,
    softmax_based: bool = False,
) -> float:
    """Predict the labels for the given patches."""
    inputs = np.stack([patch.image for patch in patches], axis=0)[:, None, :, :]
    inputs = torch.tensor(inputs, dtype=torch.float64)
    inputs = inputs.to(model.device)

    if conditional:
        area_mapping = json.load(open(AREA_MAPPING_PATH, "r"))
        condition = torch.tensor(
            [area_mapping[patch.brain_area] for patch in patches],
            dtype=torch.float64,
        )
        condition = condition.to(model.device)

    outputs = []
    for i in range(0, len(inputs), batch_size):
        if conditional:
            output = model.network(
                inputs[i : i + batch_size].float(),
                condition[i : i + batch_size].float(),
            )
        else:
            output = model.network(inputs[i : i + batch_size].float())
        outputs.append(output)

    outputs = torch.cat(outputs, dim=0)

    adjusted_outputs = torch.nn.functional.softmax(outputs, dim=1)

    masks = torch.tensor(np.stack([patch.mask for patch in patches], axis=0))

    return evaluate.f1_score(
        adjusted_outputs.cpu(),
        masks.cpu(),
        ignore_index=cort.constants.PADDING_MASK_VALUE,
        softmax_based=softmax_based,
    )
