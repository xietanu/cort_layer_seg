"""Load a CorticalPatch from a folder."""

import os
from dataclasses import dataclass

import nibabel as nib
import cv2
import numpy as np

import cort
import cort.load

COORDS_FILENAME = "coords.txt"
REGION_PROBS_FILENAME = "region_probs.npy"
IMAGE_FILENAME = "image.nii.gz"
MASK_FILENAME = "layermask.png"
RESOLUTION_KEY = "inplane_resolution_micron"


def load_patch(folder_path: str, downscale: float = 1) -> cort.CorticalPatch:
    """Load a CorticalPatch from a folder."""
    # LAYER AND MASK

    image = nib.load(os.path.join(folder_path, IMAGE_FILENAME))

    layer = image.get_fdata()
    layer = (layer - np.min(layer)) / (np.max(layer) - np.min(layer))

    mask = cv2.imread(os.path.join(folder_path, MASK_FILENAME), cv2.IMREAD_UNCHANGED)

    dscale_layer, dscale_mask = downscale_image_and_mask(layer, mask, downscale)

    dscale_layer = dscale_layer[:, 1:]
    dscale_mask = dscale_mask[:, 1:]

    relabel_bg(dscale_mask)

    # SIIBRA IMAGES

    siibra_images = cort.SiibraImages.from_folder(folder_path)

    # DATA

    data = cort.load.read_info(os.path.join(folder_path, "info.txt"))

    data[RESOLUTION_KEY] = data[RESOLUTION_KEY] * downscale  # type: ignore

    coords = image.affine[:3, 3]

    if os.path.exists(os.path.join(folder_path, REGION_PROBS_FILENAME)):
        region_probs = np.load(os.path.join(folder_path, REGION_PROBS_FILENAME))
    else:
        region_probs = np.zeros(1)

    patch = cort.CorticalPatch(
        dscale_layer,
        dscale_mask,
        y=coords[0],
        x=coords[1],
        z=coords[2],
        region_probs=region_probs,
        siibra_images=siibra_images,
        **data,
    )

    return patch


def relabel_bg(dscale_mask):
    mask_top_half = dscale_mask[: dscale_mask.shape[0] // 2, :]
    mask_top_half[mask_top_half == 7] = 0
    dscale_mask[: dscale_mask.shape[0] // 2, :] = mask_top_half


def downscale_image_and_mask(layer: np.ndarray, mask: np.ndarray, downscale: float):
    layer = cv2.resize(
        layer, None, fx=1 / downscale, fy=1 / downscale, interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, None, fx=1 / downscale, fy=1 / downscale, interpolation=cv2.INTER_NEAREST
    )

    return layer, mask


def read_coords(folder_path):
    with open(os.path.join(folder_path, COORDS_FILENAME), "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return tuple(int(line) for line in lines)
