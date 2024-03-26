import cort
import cort.display

import matplotlib.pyplot as plt


def display_patch_diff(
    pred_patch: cort.CorticalPatch, true_patch: cort.CorticalPatch, ax=None
):
    """Display a CorticalPatch."""
    if ax is None:
        ax = plt.gca()

    img = cort.display.colour_patch(
        true_patch.image_without_padding, true_patch.mask_without_padding
    )
    img[pred_patch.mask_without_padding != true_patch.mask_without_padding] = [
        255,
        0,
        0,
    ]

    ax.imshow(img)
    ax.axis("off")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
