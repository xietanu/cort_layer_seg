import numpy as np
import torch
import torch.utils.data
import torchvision

import cort
import datasets.protocols


class PatchDataset(torch.utils.data.Dataset):
    """A dataset for patches."""

    def __init__(
        self,
        patches: list[cort.CorticalPatch],
        transform: torchvision.transforms.Compose,
        condition: datasets.protocols.Condition | None = None,
        img_transform: torchvision.transforms.Compose | None = None,
    ):
        self.patches = patches
        self.transform = transform
        self.img_transform = img_transform
        self.condition = condition

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]

        image_and_label = np.stack((patch.image, patch.mask), axis=-1)

        transformed_image_and_label = self.transform(image_and_label)

        image = transformed_image_and_label[None, 0, :, :]
        label = transformed_image_and_label[None, 1, :, :]

        image[image == cort.constants.PADDING_MASK_VALUE] = 0

        if self.img_transform is not None:
            image = self.img_transform(image)

        if self.condition is not None:
            condition = self.condition(patch)
            return (
                (
                    image,
                    condition,
                ),
                label,
            )

        return (
            image,
            label,
        )
