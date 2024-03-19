"""Load a CorticalPatch from a folder."""
import os

import cv2

import cort
import cort.load

IMAGE_FILENAME = "image.png"
MASK_FILENAME = "layermask.png"
RESOLUTION_KEY = "inplane_resolution_micron"


def load_patch(folder_path: str, downscale: float = 1) -> cort.CorticalPatch:
    """Load a CorticalPatch from a folder."""
    layer = cv2.imread(os.path.join(folder_path, IMAGE_FILENAME), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(folder_path, MASK_FILENAME), cv2.IMREAD_GRAYSCALE)

    layer = cv2.resize(
        layer, None, fx=1 / downscale, fy=1 / downscale, interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, None, fx=1 / downscale, fy=1 / downscale, interpolation=cv2.INTER_AREA
    )

    data = cort.load.read_info(os.path.join(folder_path, "info.txt"))
    
    data[RESOLUTION_KEY] = data[RESOLUTION_KEY] * downscale # type: ignore

    return cort.CorticalPatch(layer, mask, **data)  # type: ignore
