import os
from dataclasses import dataclass
import json

import numpy as np
import matplotlib.pyplot as plt

import cort
import cort.constants
import cort.display
import cort.depth


@dataclass
class CorticalPatch:
    """A cortical image patch."""

    image: np.ndarray
    mask: np.ndarray
    inplane_resolution_micron: float
    section_thickness_micron: float
    brain_id: str
    section_id: int
    patch_id: int
    brain_area: str
    x: float
    y: float
    z: float
    region_probs: np.ndarray
    borders: np.ndarray = None
    depth_maps: np.ndarray = None
    siibra_imgs: cort.SiibraImages | None = None

    def __post_init__(self):
        """Post-initialization."""
        if self.image.shape != self.mask.shape:
            raise ValueError("Image and mask must have the same shape.")

        if self.borders is None:
            self.borders = cort.depth.find_borders(self.mask)

        if self.depth_maps is None:
            self.depth_maps = cort.depth.map_distance_from_border(
                self.mask, self.borders
            )

    def __eq__(self, other):
        """Check if two patches are equal."""
        if not isinstance(other, CorticalPatch):
            return False
        return (
            self.inplane_resolution_micron == other.inplane_resolution_micron
            and self.section_thickness_micron == other.section_thickness_micron
            and self.brain_id == other.brain_id
            and self.section_id == other.section_id
            and self.patch_id == other.patch_id
            and self.brain_area == other.brain_area
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
        )

    @property
    def shape(self):
        """Return the shape of the patch."""
        return self.image.shape

    @property
    def name(self):
        """Return the name of the patch."""
        return f"{self.brain_id}-{self.brain_area}-{self.section_id}-{self.patch_id}"

    def display(self, ax=None, show_padding=False):
        """Display the patch."""
        new_ax = cort.display.display_patch(
            self.image if show_padding else self.image_without_padding,
            self.mask if show_padding else self.mask_without_padding,
            ax=ax,
        )
        new_ax.set_title(self.name)
        if ax is None:
            plt.show()
        return ax

    @property
    def image_without_padding(self):
        """Return the image without padded."""
        unpadded_width = np.sum(self.mask[0, :] != cort.constants.PADDING_MASK_VALUE)
        unpadded_height = np.sum(self.mask[:, 0] != cort.constants.PADDING_MASK_VALUE)
        return self.image[:unpadded_height, :unpadded_width]

    @property
    def mask_without_padding(self):
        """Return the mask without padded."""
        unpadded_width = np.sum(self.mask[0, :] != cort.constants.PADDING_MASK_VALUE)
        unpadded_height = np.sum(self.mask[:, 0] != cort.constants.PADDING_MASK_VALUE)
        return self.mask[:unpadded_height, :unpadded_width]

    def save(self, folder):
        """Save the patch to a folder."""
        name = f"{self.brain_area}-{self.section_id}-{self.patch_id}"

        images_stack = np.concatenate(
            [
                self.image[:, :, None],
                self.mask[:, :, None],
                # self.border_flow_map[:, :, None],
                self.borders,
                self.depth_maps,
            ],
            axis=2,
        )

        siibra_stack = None
        if self.siibra_imgs is not None:
            siibra_stack = np.concatenate(
                [
                    self.siibra_imgs.image[:, :, None],
                    self.siibra_imgs.mask[:, :, None],
                    self.siibra_imgs.matched_image[:, :, None],
                    self.siibra_imgs.affine_matched_image[:, :, None],
                    self.siibra_imgs.affine_mask[:, :, None],
                    self.siibra_imgs.depth_maps,
                ],
                axis=2,
            )

        other_data = {
            "inplane_resolution_micron": self.inplane_resolution_micron,
            "section_thickness_micron": self.section_thickness_micron,
            "brain_id": self.brain_id,
            "section_id": self.section_id,
            "patch_id": self.patch_id,
            "brain_area": self.brain_area,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "region_probs": self.region_probs.tolist(),
            "n_borders": self.borders.shape[2],
            "n_depth_maps": self.depth_maps.shape[2],
        }

        np.save(f"{folder}/{name}_images.npy", images_stack)
        if siibra_stack is not None:
            np.save(f"{folder}/{name}_siibra.npy", siibra_stack)
            np.save(
                f"{folder}/{name}_prev_mask.npy", self.siibra_imgs.existing_cort_layers
            )
        with open(f"{folder}/{name}_data.json", "w") as f:
            json.dump(other_data, f)

    @classmethod
    def load(cls, folder, name):
        """Load a patch from a folder."""
        images_stack = np.load(f"{folder}/{name}_images.npy")
        siibra_imgs = None
        if os.path.exists(f"{folder}/{name}_siibra.npy"):
            siibra_stack = np.load(f"{folder}/{name}_siibra.npy")
            prev_mask = np.load(f"{folder}/{name}_prev_mask.npy")
            siibra_imgs = cort.SiibraImages(
                image=siibra_stack[:, :, 0],
                existing_cort_layers=prev_mask,
                mask=siibra_stack[:, :, 1],
                matched_image=siibra_stack[:, :, 2],
                depth_maps=siibra_stack[:, :, 5:],
                affine_matched_image=siibra_stack[:, :, 3],
                affine_mask=siibra_stack[:, :, 4],
            )
        with open(f"{folder}/{name}_data.json", "r") as f:
            other_data = json.load(f)
        return cls(
            image=images_stack[:, :, 0],
            mask=images_stack[:, :, 1],
            borders=images_stack[:, :, 2 : 2 + other_data["n_borders"]],
            depth_maps=images_stack[:, :, 2 + other_data["n_borders"] :],
            inplane_resolution_micron=other_data["inplane_resolution_micron"],
            section_thickness_micron=other_data["section_thickness_micron"],
            brain_id=other_data["brain_id"],
            section_id=other_data["section_id"],
            patch_id=other_data["patch_id"],
            brain_area=other_data["brain_area"],
            x=other_data["x"],
            y=other_data["y"],
            z=other_data["z"],
            region_probs=np.array(other_data["region_probs"]),
            siibra_imgs=siibra_imgs,
        )
