import os

import numpy as np
from tqdm import tqdm

import cort.load

import cv2

RESULTS_DIR = (
    "pytorch-CycleGAN-and-pix2pix/results/seg_to_cell_pix2pix/test_latest/images"
)
BASE_DIR = "data/synth"

PATCHES_DIR = "data/preprocessed_gaussian"


def create_cycle_data():
    """Create a cycle dataset."""
    os.makedirs(BASE_DIR, exist_ok=True)

    patches = cort.load.load_pre_mask_patches(PATCHES_DIR, report_progress=True)

    print(f"{len(patches)} patches loaded.")

    synth_images = [
        cv2.imread(f"{RESULTS_DIR}/{i}_fake_B.png", cv2.IMREAD_GRAYSCALE)
        for i in range(len(patches))
    ]

    for i, (patch, synth_image) in tqdm(
        enumerate(zip(patches, synth_images)), total=len(patches)
    ):
        synth_image = cv2.resize(
            synth_image, (patch.mask.shape[1], patch.mask.shape[0])
        )
        new_patch = cort.CorticalPatch(
            synth_image,
            patch.mask,
            inplane_resolution_micron=patch.inplane_resolution_micron,
            section_thickness_micron=patch.section_thickness_micron,
            brain_id=patch.brain_id,
            section_id=patch.section_id,
            patch_id=patch.patch_id,
            brain_area=patch.brain_area,
            x=patch.x,
            y=patch.y,
            z=patch.z,
            depth_maps=patch.depth_maps,
            borders=patch.borders,
            region_probs=patch.region_probs,
            is_corner_patch=False,
        )
        new_patch.save(BASE_DIR)


if __name__ == "__main__":
    create_cycle_data()
