from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

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

    @property
    def shape(self):
        """Return the shape of the patch."""
        return self.image.shape
    
    @property
    def name(self):
        """Return the name of the patch."""
        return f"{self.brain_id}-{self.brain_area}-{self.section_id}-{self.patch_id}"

    def display(self,ax=None):
        """Display the patch."""
        new_ax = cort.display.display_patch(self.image, self.mask, ax=ax)
        new_ax.set_title(self.name)
        if ax is None:
            plt.show()
        return ax
    