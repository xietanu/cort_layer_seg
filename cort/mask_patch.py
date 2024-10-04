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
class MaskCorticalPatch:
    """A cortical image patch."""

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

    def __post_init__(self):
        """Post-initialization."""
        if self.borders is None and self.mask is not None:
            self.borders = cort.depth.find_borders(self.mask)

        if self.depth_maps is None and self.mask is not None:
            self.depth_maps = cort.depth.map_distance_from_border(
                self.mask, self.borders
            )

    def __eq__(self, other):
        """Check if two patches are equal."""
        if not isinstance(other, MaskCorticalPatch):
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
        return self.mask.shape

    @property
    def name(self):
        """Return the name of the patch."""
        return f"{self.brain_id}-{self.brain_area}-{self.section_id}-{self.patch_id}"

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
                self.mask[:, :, None],
                # self.border_flow_map[:, :, None],
                (
                    self.borders
                    if self.borders is not None
                    else np.zeros_like(self.mask)[:, :, None]
                ),
                (
                    self.depth_maps
                    if self.depth_maps is not None
                    else np.zeros_like(self.mask)[:, :, None]
                ),
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
            "n_borders": self.borders.shape[2] if self.borders is not None else 1,
            "n_depth_maps": (
                self.depth_maps.shape[2] if self.depth_maps is not None else 1
            ),
        }

        np.save(f"{folder}/{name}_images.npy", images_stack)

        with open(f"{folder}/{name}_data.json", "w") as f:
            json.dump(other_data, f)

    @classmethod
    def load(cls, folder, name):
        """Load a patch from a folder."""
        with open(f"{folder}/{name}_data.json", "r") as f:
            other_data = json.load(f)

        images_stack = np.load(f"{folder}/{name}_images.npy")
        mask = images_stack[:, :, 0]
        borders = images_stack[:, :, 1 : 1 + other_data["n_borders"]]
        depth_maps = images_stack[:, :, 1 + other_data["n_borders"] :]

        if np.all(mask == 0):
            mask = None
            borders = None
            depth_maps = None

        return cls(
            mask=mask,
            borders=borders,
            depth_maps=depth_maps,
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
        )
