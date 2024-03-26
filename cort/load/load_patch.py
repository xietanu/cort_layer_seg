"""Load a CorticalPatch from a folder."""

import os

import cv2
import numpy as np

import cort
import cort.load

COORDS_FILENAME = "coords.txt"
REGION_PROBS_FILENAME = "region_probs.npy"
IMAGE_FILENAME = "image.png"
MASK_FILENAME = "layermask.png"
RESOLUTION_KEY = "inplane_resolution_micron"


def load_patch(folder_path: str, downscale: float = 1) -> cort.CorticalPatch:
    """Load a CorticalPatch from a folder."""
    layer, mask = read_image_and_mask(folder_path)

    dscale_layer, dscale_mask = downscale_image_and_mask(layer, mask, downscale)

    data = cort.load.read_info(os.path.join(folder_path, "info.txt"))

    data[RESOLUTION_KEY] = data[RESOLUTION_KEY] * downscale  # type: ignore

    coords = read_coords(folder_path)

    region_probs = np.load(os.path.join(folder_path, REGION_PROBS_FILENAME))

    return cort.CorticalPatch(
        dscale_layer,
        dscale_mask,
        y=coords[0],
        x=coords[1],
        z=coords[2],
        region_probs=region_probs,
        **data
    )


def read_image_and_mask(folder_path):
    layer = (
        2
        * cv2.imread(
            os.path.join(folder_path, IMAGE_FILENAME), cv2.IMREAD_GRAYSCALE
        ).astype(np.float32)
        / 255
        - 1
    )
    mask = cv2.imread(os.path.join(folder_path, MASK_FILENAME), cv2.IMREAD_UNCHANGED)
    return layer, mask


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
