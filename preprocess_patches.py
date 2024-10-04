import random

import numpy as np
import torch
from tqdm import tqdm
import os

import cort.load


def main():
    patch_paths = cort.load.find_all_patches("data/all_cort_patches")
    corner_paths = cort.load.find_all_corner_patches("data/man_patches")

    print(f"Found {len(patch_paths)} patches, {len(corner_paths)} corner patches.")

    corner_patches = cort.load.load_corner_patches(
        corner_paths, report_progress=True, max_width=128, max_height=256
    )
    patches = cort.load.load_patches(patch_paths, 20, report_progress=True)

    patches.extend(corner_patches)

    print(f"Loaded {len(patches)} patches.")

    os.makedirs("data/preprocessed", exist_ok=True)

    for patch in tqdm(patches, desc="Saving patches"):
        patch.save("data/preprocessed")

    print("Saved all patches. Done!")


if __name__ == "__main__":
    main()
