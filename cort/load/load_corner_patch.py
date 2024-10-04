"""Load a CorticalPatch from a folder."""

import json
import os
from dataclasses import dataclass

import nibabel as nib
import cv2
import numpy as np

import cort
import cort.load

COORDS_FILENAME = "coords.txt"
REGION_PROBS_FILENAME = "probs.npy"
IMAGE_FILENAME = "img.png"
MASK_FILENAME = "mask.png"
RESOLUTION_KEY = "inplane_resolution_micron"

TOP = 0
BOTTOM = 1
LEFT = 2
RIGHT = 3


def load_corner_patch(
    folder_path: str, max_width: int = 128, max_height: int = 256
) -> cort.CorticalPatch:
    """Load a CorticalPatch from a folder."""
    # LAYER AND MASK

    image = cv2.imread(os.path.join(folder_path, IMAGE_FILENAME), cv2.IMREAD_UNCHANGED)
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)

    if os.path.exists(os.path.join(folder_path, MASK_FILENAME)):
        mask = cv2.imread(
            os.path.join(folder_path, MASK_FILENAME), cv2.IMREAD_UNCHANGED
        )
    else:
        mask = None

    image, mask = orient_image_and_mask(image, mask)

    excess_y = max(image.shape[0] - max_height, 0) // 2
    excess_x = max(image.shape[1] - max_width, 0) // 2

    if np.random.rand() < 0.5:
        excess_y = max(
            0,
            excess_y - np.random.randint(0, 75),
        )
    else:
        excess_y = min(
            excess_y * 2,
            excess_y + np.random.randint(0, 125),
        )

    image = image[
        excess_y : excess_y + max_height,
        excess_x : excess_x + max_width,
    ]
    image = image[:max_height, :max_width]
    if mask is not None:
        mask = mask[
            excess_y : excess_y + max_height,
            excess_x : excess_x + max_width,
        ]
        mask = mask[:max_height, :max_width]

    # DATA

    data = json.load(open(os.path.join(folder_path, "info.json")))

    coords = np.load(os.path.join(folder_path, "coords.npy"))

    if os.path.exists(os.path.join(folder_path, REGION_PROBS_FILENAME)):
        region_probs = np.load(os.path.join(folder_path, REGION_PROBS_FILENAME))
    else:
        region_probs = np.zeros(1)
        print("No region probs found.")

    patch = cort.CorticalPatch(
        image,
        mask,
        y=coords[0],
        x=coords[1],
        z=coords[2],
        region_probs=region_probs,
        siibra_imgs=None,
        inplane_resolution_micron=20.0,
        section_thickness_micron=1.0,
        brain_id="brain1",
        section_id=data["section_n"],
        patch_id=0,
        brain_area=data["area"] + "_corner",
        is_corner_patch=True,
    )

    return patch


def orient_image_and_mask(image, mask):
    """Orient the image and mask so that they are in the same orientation."""
    orientation = determine_orientation(image)
    image = orient_to_top(image, orientation)
    if mask is not None:
        mask = orient_to_top(mask, orientation)
    return image, mask


def determine_orientation(img):
    white_mask = img > 0.95

    top = (np.sum(white_mask[: white_mask.shape[0] // 2, :])) / (np.prod(img.shape))
    bottom = (np.sum(white_mask[white_mask.shape[0] // 2 :, :])) / (np.prod(img.shape))

    left = (np.sum(white_mask[:, : white_mask.shape[1] // 2])) / (np.prod(img.shape))
    right = (np.sum(white_mask[:, white_mask.shape[1] // 2 :])) / (np.prod(img.shape))

    return np.argmax([top, bottom, left, right])


def orient_to_top(img, orientation):
    if orientation == TOP:
        return img
    elif orientation == BOTTOM:
        return np.flipud(img)
    elif orientation == LEFT:
        return np.swapaxes(img, 0, 1)
    elif orientation == RIGHT:
        return np.flipud(np.swapaxes(img, 0, 1))
    else:
        raise ValueError("Invalid orientation")
