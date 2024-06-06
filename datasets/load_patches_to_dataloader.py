import json

import torch
import torch.utils.data
import torchvision.transforms

import datasets

BRAIN_AREA_ENCODING_PATH = "data/cort_patches/area_mapping.json"


def load_patches_to_dataloader(
    split: datasets.enums.Split,
    downscale_factor: float,
    batch_size: int,
    transform: torchvision.transforms.Compose,
    img_transform: torchvision.transforms.Compose | None = None,
    shuffle: bool = False,
    condition: datasets.protocols.Condition | None = None,
) -> torch.utils.data.DataLoader:
    """Create training, validation, and test loaders."""
    patches = datasets.load_split_patches(split, downscale_factor)

    dataset = datasets.PatchDataset(
        patches,
        transform=transform,
        img_transform=img_transform,
        condition=condition,
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
