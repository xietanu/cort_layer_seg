import numpy as np
import torch
from tqdm import tqdm

import datasets
import cort
import nnet.models


def predict(
    models: list[nnet.models.SemantSegUNetModel | str],
    patches: list[cort.CorticalPatch],
    batch_size: int = 5,
):
    max_width = max(patch.image.shape[1] for patch in patches)
    max_height = max(patch.image.shape[0] for patch in patches)

    # max_width += (16 - max_width % 16) % 16
    # max_height += (16 - max_height % 16) % 16

    max_width = 128
    max_height = 256

    print(f"Max width: {max_width}, max height: {max_height}")

    batches = []
    for i in range(0, len(patches), batch_size):
        cur_patches = patches[i : i + batch_size]

        images = torch.zeros(len(cur_patches), 1, max_height, max_width)
        probabilities = torch.zeros(
            len(cur_patches), cur_patches[0].region_probs.shape[0]
        )
        points = torch.zeros(len(cur_patches), 3)

        for j, patch in enumerate(cur_patches):
            x_offset = (max_width - patch.image.shape[1]) // 2
            y_offset = (max_height - patch.image.shape[0]) // 2
            images[
                j,
                0,
                y_offset : y_offset + patch.image.shape[0],
                x_offset : x_offset + patch.image.shape[1],
            ] = torch.from_numpy(patch.image)
            probabilities[j] = torch.from_numpy(patch.region_probs)
            points[j] = torch.tensor([patch.x, patch.y, patch.z])

        inputs = datasets.datatypes.SegInputs(
            images,
            probabilities,
            points,
        )

        batches.append(inputs)

    outputs = []

    for i, model in enumerate(models):
        if len(models) > 1:
            print(f"Predicting with model {i}")

        if isinstance(model, str):
            model = nnet.models.SemantSegUNetModel.restore(model)

        outputs.append(None)
        for batch in tqdm(batches, desc="Predicting segmentations"):
            cur_outputs = model.predict(batch)

            cur_outputs.logits = torch.softmax(cur_outputs.logits, dim=1)
            cur_outputs.denoised_logits = torch.softmax(
                cur_outputs.denoised_logits, dim=1
            )

            if outputs[i] is None:
                outputs[i] = cur_outputs
            else:
                outputs[i] = outputs[i] + cur_outputs

    combined_outputs = outputs[0]

    for i in range(1, len(outputs)):
        combined_outputs.logits += outputs[i].logits
        combined_outputs.denoised_logits += outputs[i].denoised_logits

    print(combined_outputs.logits.shape)
    print(combined_outputs.denoised_logits.shape)

    combined_outputs.segmentation = torch.argmax(combined_outputs.logits, dim=1)
    combined_outputs.denoised_segementation = torch.argmax(
        combined_outputs.denoised_logits, dim=1
    )

    print(combined_outputs.segmentation.shape)
    print(combined_outputs.denoised_segementation.shape)

    return combined_outputs
