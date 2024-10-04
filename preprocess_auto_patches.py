import random

import numpy as np
import torch
from tqdm import tqdm
import os

import cort.load


def main():
    random_paths = cort.load.find_all_corner_patches("data/random_imgs")

    print(f"Found {len(random_paths)} random patches.")

    random_patches = cort.load.load_corner_patches(
        random_paths, report_progress=True, max_width=256, max_height=256
    )

    print(f"Loaded {len(random_patches)} random patches.")

    random_patches = [
        patch
        for patch in random_patches
        if np.sum(patch.image > 0.95) < 0.5 * patch.image.size
    ]

    print(f"Filtered to {len(random_patches)} random patches.")

    os.makedirs("data/preprocessed_random", exist_ok=True)

    for patch in tqdm(random_patches, desc="Saving random patches"):
        patch.save("data/preprocessed_random")

    print("Saved all random patches. Done!")


if __name__ == "__main__":
    main()
