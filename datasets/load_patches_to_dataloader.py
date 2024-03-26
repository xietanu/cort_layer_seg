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
    conditional: bool = False,
) -> torch.utils.data.DataLoader:
    """Create train, validation, and test loaders."""
    patches = datasets.load_split_patches(split, downscale_factor)

    if not conditional:
        dataset = datasets.PatchDataset(
            patches, transform=transform, img_transform=img_transform
        )
    else:
        with open(BRAIN_AREA_ENCODING_PATH, "r") as f:
            area_encoding_lookup = json.load(f)
        dataset = datasets.PatchDatasetConditionedOnArea(
            patches,
            transform=transform,
            img_transform=img_transform,
            area_encoding_lookup=area_encoding_lookup,
        )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
