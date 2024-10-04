import random

import numpy as np
import torch
from tqdm import tqdm
import os

import cort.load


def main():
    gaussian_paths = cort.load.find_all_corner_patches("data/random_masks")

    print(f"Found {len(gaussian_paths)} gaussian patches.")

    gaussian_patches = cort.load.load_mask_patches(
        "data/random_masks",
        gaussian_paths,
        report_progress=True,
        max_width=128,
        max_height=256,
    )
    print(f"Loaded {len(gaussian_patches)} gaussian patches.")

    gaussian_patches = [
        patch for patch in gaussian_patches if np.unique(patch.mask).shape[0] > 3
    ]

    gaussian_patches = [
        patch
        for patch in gaussian_patches
        if np.sum(patch.mask == 0) < 0.5 * patch.mask.size
    ]

    print(f"Filtered to {len(gaussian_patches)} gaussian patches.")

    os.makedirs("data/preprocessed_gaussian", exist_ok=True)

    for patch in tqdm(gaussian_patches, desc="Saving gaussian patches"):
        patch.save("data/preprocessed_gaussian")

    print("Saved all gaussian patches. Done!")


if __name__ == "__main__":
    main()
