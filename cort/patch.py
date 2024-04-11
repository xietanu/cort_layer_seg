from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import cort.constants
import cort.display


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

    def __post_init__(self):
        """Post-initialization."""
        if self.image.shape != self.mask.shape:
            raise ValueError("Image and mask must have the same shape.")

        self.borders = np.zeros((7, self.mask.shape[0], self.mask.shape[1]))

        offset_mask = np.roll(self.mask, -1, axis=0)

        for i in range(7):
            self.borders[i, (self.mask == i) & (offset_mask != i)] = 1

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
        """Return the image without padding."""
        unpadded_width = np.sum(self.mask[0, :] != cort.constants.PADDING_MASK_VALUE)
        unpadded_height = np.sum(self.mask[:, 0] != cort.constants.PADDING_MASK_VALUE)
        return self.image[:unpadded_height, :unpadded_width]

    @property
    def mask_without_padding(self):
        """Return the mask without padding."""
        unpadded_width = np.sum(self.mask[0, :] != cort.constants.PADDING_MASK_VALUE)
        unpadded_height = np.sum(self.mask[:, 0] != cort.constants.PADDING_MASK_VALUE)
        return self.mask[:unpadded_height, :unpadded_width]
