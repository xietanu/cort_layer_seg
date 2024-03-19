"""Load a CorticalPatch from a folder."""

import os

import cv2
import numpy as np

import cort
import cort.load

IMAGE_FILENAME = "image.png"
MASK_FILENAME = "layermask.png"
RESOLUTION_KEY = "inplane_resolution_micron"


def load_patch(folder_path: str, downscale: float = 1) -> cort.CorticalPatch:
    """Load a CorticalPatch from a folder."""
    layer, mask = read_image_and_mask(folder_path)

    dscale_layer, dscale_mask = downscale_image_and_mask(layer, mask, downscale)

    data = cort.load.read_info(os.path.join(folder_path, "info.txt"))

    data[RESOLUTION_KEY] = data[RESOLUTION_KEY] * downscale  # type: ignore

    return cort.CorticalPatch(dscale_layer, dscale_mask, **data)  # type: ignore


def read_image_and_mask(folder_path):
    layer = 2 * cv2.imread(os.path.join(folder_path, IMAGE_FILENAME), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 - 1
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
