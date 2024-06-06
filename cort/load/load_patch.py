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

SIIBRA_IMAGE = "siibra_img.png"
SIIBRA_MASK = "full_aligned_mask.png"
MATCHED_IMAGE = "full_aligned_img.png"
BASIC_MASK = "basic_aligned_mask.png"
BASIC_IMAGE = "basic_aligned_img.png"
SIIBRA_DEPTH_MAPS = "full_aligned_depth_maps.npy"
BASIC_DEPTH_MAPS = "basic_aligned_depth_maps.npy"

TOP = 0
BOTTOM = 1
LEFT = 2
RIGHT = 3


@dataclass
class SiibraImages:
    image: np.ndarray
    mask: np.ndarray
    matched_image: np.ndarray
    depth_maps: np.ndarray
    affine_matched_image: np.ndarray
    affine_mask: np.ndarray
    uses_affine: bool = False

    def __post_init__(self):
        self.image = (self.image - np.min(self.image)) / (
            np.max(self.image) - np.min(self.image)
        )
        self.matched_image = (self.matched_image - np.min(self.matched_image)) / (
            np.max(self.matched_image) - np.min(self.matched_image)
        )
        self.affine_matched_image = (
            self.affine_matched_image - np.min(self.affine_matched_image)
        ) / (np.max(self.affine_matched_image) - np.min(self.affine_matched_image))
        if np.all(self.mask == cort.constants.PADDING_MASK_VALUE):
            self.use_affine()

    def use_affine(self):
        self.image = self.affine_matched_image
        self.mask = self.affine_mask
        self.uses_affine = True

    def trim_to_border(self):
        top, bottom, left, right = find_borders(self.mask)

        if bottom - top < 50 or right - left < 50:
            self.use_affine()
            top, bottom, left, right = find_borders(self.mask)

        self.image = self.image[top:bottom, left:right]
        self.mask = self.mask[top:bottom, left:right]
        self.matched_image = self.matched_image[top:bottom, left:right]
        self.depth_maps = self.depth_maps[top:bottom, left:right]
        self.affine_matched_image = self.affine_matched_image[top:bottom, left:right]
        self.affine_mask = self.affine_mask[top:bottom, left:right]

    def trim_to_size(self, size: tuple[int, int]):
        self.image = self.image[: size[0], : size[1]]
        self.mask = self.mask[: size[0], : size[1]]
        self.matched_image = self.matched_image[: size[0], : size[1]]
        self.depth_maps = self.depth_maps[: size[0], : size[1]]
        self.affine_matched_image = self.affine_matched_image[: size[0], : size[1]]
        self.affine_mask = self.affine_mask[: size[0], : size[1]]

    @classmethod
    def from_folder(cls, folder_path: str, auto_fix: bool = True):
        siibra_image = cv2.imread(
            os.path.join(folder_path, SIIBRA_IMAGE), cv2.IMREAD_UNCHANGED
        )
        siibra_mask = cv2.imread(
            os.path.join(folder_path, SIIBRA_MASK), cv2.IMREAD_UNCHANGED
        )
        matched_image = cv2.imread(
            os.path.join(folder_path, MATCHED_IMAGE), cv2.IMREAD_UNCHANGED
        )
        siibra_depth_maps = np.load(os.path.join(folder_path, SIIBRA_DEPTH_MAPS))
        basic_matched_image = cv2.imread(
            os.path.join(folder_path, BASIC_IMAGE), cv2.IMREAD_UNCHANGED
        )
        basic_mask = cv2.imread(
            os.path.join(folder_path, BASIC_MASK), cv2.IMREAD_UNCHANGED
        )

        output = cls(
            siibra_image,
            siibra_mask,
            matched_image,
            siibra_depth_maps,
            basic_matched_image,
            basic_mask,
        )

        if not auto_fix:
            return output

        output.trim_to_border()
        output.auto_orient()
        output.trim_to_size((256, 128))

        return output

    def auto_orient(self):
        orientation = determine_orientation(self.mask)
        self.image = orient_to_top(self.image, orientation)
        self.mask = orient_to_top(self.mask, orientation)
        self.matched_image = orient_to_top(self.matched_image, orientation)
        self.depth_maps = orient_to_top(self.depth_maps, orientation)
        self.affine_matched_image = orient_to_top(
            self.affine_matched_image, orientation
        )
        self.affine_mask = orient_to_top(self.affine_mask, orientation)


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

    siibra_images = SiibraImages.from_folder(folder_path)

    # DATA

    data = cort.load.read_info(os.path.join(folder_path, "info.txt"))

    data[RESOLUTION_KEY] = data[RESOLUTION_KEY] * downscale  # type: ignore

    coords = image.affine[:3, 3]

    if os.path.exists(os.path.join(folder_path, REGION_PROBS_FILENAME)):
        region_probs = np.load(os.path.join(folder_path, REGION_PROBS_FILENAME))
    else:
        region_probs = np.zeros(1)

    return cort.CorticalPatch(
        dscale_layer,
        dscale_mask,
        siibra_images.image,
        siibra_images.mask,
        siibra_images.matched_image,
        y=coords[0],
        x=coords[1],
        z=coords[2],
        region_probs=region_probs,
        siibra_depth_maps=siibra_images.depth_maps,
        basic_matched_image=siibra_images.affine_matched_image,
        basic_siibra_mask=siibra_images.affine_mask,
        **data,
    )


def relabel_bg(dscale_mask):
    mask_top_half = dscale_mask[: dscale_mask.shape[0] // 2, :]
    mask_top_half[mask_top_half == 7] = 0
    dscale_mask[: dscale_mask.shape[0] // 2, :] = mask_top_half


def find_borders(siibra_mask):
    left = 0
    while np.all(siibra_mask[:, left] == cort.constants.PADDING_MASK_VALUE):
        left += 1
    right = siibra_mask.shape[1] - 1
    while np.all(siibra_mask[:, right] == cort.constants.PADDING_MASK_VALUE):
        right -= 1
    top = 0
    while np.all(siibra_mask[top, :] == cort.constants.PADDING_MASK_VALUE):
        top += 1
    bottom = siibra_mask.shape[0] - 1
    while np.all(siibra_mask[bottom, :] == cort.constants.PADDING_MASK_VALUE):
        bottom -= 1
    return bottom, left, right, top


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


def determine_orientation(mask):
    top = (
        np.sum(mask[: mask.shape[0] // 2, :])
        - np.sum(mask[: mask.shape[0] // 2, :] == cort.constants.PADDING_MASK_VALUE)
        * cort.constants.PADDING_MASK_VALUE
    ) / (np.sum(mask[: mask.shape[0] // 2, :] != cort.constants.PADDING_MASK_VALUE) + 1)
    bottom = (
        np.sum(mask[mask.shape[0] // 2 :, :])
        - np.sum(mask[mask.shape[0] // 2 :, :] == cort.constants.PADDING_MASK_VALUE)
        * cort.constants.PADDING_MASK_VALUE
    ) / (np.sum(mask[mask.shape[0] // 2 :, :] != cort.constants.PADDING_MASK_VALUE) + 1)
    left = (
        np.sum(mask[:, : mask.shape[1] // 2])
        - np.sum(mask[:, : mask.shape[1] // 2] == cort.constants.PADDING_MASK_VALUE)
        * cort.constants.PADDING_MASK_VALUE
    ) / (np.sum(mask[:, : mask.shape[1] // 2] != cort.constants.PADDING_MASK_VALUE) + 1)
    right = (
        np.sum(mask[:, mask.shape[1] // 2 :])
        - np.sum(mask[:, mask.shape[1] // 2 :] == cort.constants.PADDING_MASK_VALUE)
        * cort.constants.PADDING_MASK_VALUE
    ) / (np.sum(mask[:, mask.shape[1] // 2 :] != cort.constants.PADDING_MASK_VALUE) + 1)

    return np.argmin([top, bottom, left, right])


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
