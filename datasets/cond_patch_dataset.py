import numpy as np
import torch
import torch.utils.data
import torchvision

import cort


class ConditionalPatchDataset(torch.utils.data.Dataset):
    """A dataset for patches."""

    def __init__(
        self,
        patches: list[cort.CorticalPatch],
        transform: torchvision.transforms.Compose,
        img_transform: torchvision.transforms.Compose | None = None,
    ):
        self.patches = patches
        self.transform = transform
        self.img_transform = img_transform

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        patch = self.patches[idx]

        image_and_label = np.stack((patch.image, patch.mask), axis=-1)

        transformed_image_and_label = self.transform(image_and_label)

        image = transformed_image_and_label[None, 0, :, :]
        label = transformed_image_and_label[None, 1, :, :]

        image[image == cort.constants.PADDING_MASK_VALUE] = 0

        if self.img_transform is not None:
            image = self.img_transform(image)

        condition = torch.tensor([patch.x, patch.y, patch.z], dtype=torch.float32)

        return (
            (image, condition),
            label,
        )
