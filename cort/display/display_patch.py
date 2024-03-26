import matplotlib.pyplot as plt
import numpy as np

import cort.constants
import cort.display


def display_patch(patch_img: np.ndarray, patch_mask: np.ndarray, ax=None):
    """Display a CorticalPatch."""
    if ax is None:
        ax = plt.gca()

    img = cort.display.colour_patch(patch_img, patch_mask)

    ax.imshow(img)
    ax.axis("off")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
