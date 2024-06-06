import numpy as np

import cort
import cort.display

import matplotlib.pyplot as plt


def display_patch_diff(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, ax=None):
    """Display a CorticalPatch."""
    if ax is None:
        ax = plt.gca()

    comb_img = cort.display.colour_patch(img, gt)
    comb_img[pred != gt] = [
        255,
        0,
        0,
    ]

    ax.imshow(comb_img)
    ax.axis("off")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
