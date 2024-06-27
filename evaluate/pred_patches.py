import torch
import numpy as np
import json

import cort
import nnet
import nnet.models
import datasets.protocols

AREA_MAPPING_PATH = "data/cort_patches/area_mapping.json"


def pred_patches(
    model: nnet.protocols.SegModelProtocol,
    patches: list[cort.CorticalPatch],
    condition: datasets.protocols.Condition | None = None,
    batch_size: int = 4,
) -> list[cort.CorticalPatch]:
    """Predict the labels for the given patches."""
    inputs = np.stack([patch.image for patch in patches], axis=0)[:, None, :, :]
    inputs = torch.tensor(inputs, dtype=torch.float64)
    inputs = inputs.to(model.device)

    if condition is not None:
        # area_mapping = json.load(open(AREA_MAPPING_PATH, "r"))
        condition = torch.stack([condition(patch) for patch in patches], dim=0)

    outputs = []
    for i in range(0, len(inputs), batch_size):
        if condition is not None:
            output = model.predict(
                (inputs[i : i + batch_size], condition[i : i + batch_size])
            )
        else:
            output = model.predict(inputs[i : i + batch_size])
        outputs.append(output)

    outputs = np.concatenate(outputs, axis=0)

    masks = np.stack([patch.mask for patch in patches], axis=0)

    outputs[masks == cort.constants.PADDING_MASK_VALUE] = (
        cort.constants.PADDING_MASK_VALUE
    )

    predicted_patches = [
        cort.CorticalPatch(
            image=patch.image,
            mask=output,
            inplane_resolution_micron=patch.inplane_resolution_micron,
            section_thickness_micron=patch.section_thickness_micron,
            brain_area=patch.brain_area,
            brain_id=patch.brain_id,
            patch_id=patch.patch_id,
            section_id=patch.section_id,
            x=patch.x,
            y=patch.y,
            z=patch.z,
            region_probs=patch.region_probs,
        )
        for patch, output in zip(patches, outputs)
    ]

    return predicted_patches
