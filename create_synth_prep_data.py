import os

import numpy as np

import cort.load

import cv2

BASE_DIR = "pytorch-CycleGAN-and-pix2pix/datasets/to_synth/test"

PATCHES_DIR = "data/preprocessed_gaussian"


def create_cycle_data():
    """Create a cycle dataset."""
    os.makedirs(BASE_DIR, exist_ok=True)

    patches = cort.load.load_pre_mask_patches(PATCHES_DIR, report_progress=True)

    print(f"{len(patches)} patches loaded.")

    for i, patch in enumerate(patches):
        cell_mask = patch.mask
        output = np.zeros(
            (cell_mask.shape[0], cell_mask.shape[1] * 2, 1), dtype=np.uint8
        )
        output[:, : cell_mask.shape[1], 0] = cell_mask
        output[:, cell_mask.shape[1] :, 0] = cell_mask * (255 // 8)

        cell_path = os.path.join(BASE_DIR, f"{i}.png")
        cv2.imwrite(cell_path, output)


if __name__ == "__main__":
    create_cycle_data()
