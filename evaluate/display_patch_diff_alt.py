import numpy as np

import cort
import cort.display

import matplotlib.pyplot as plt


def display_patch_diff_alt(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, ax=None):
    """Display a CorticalPatch."""
    if ax is None:
        ax = plt.gca()

    img_alt = img.copy()
    img_alt[gt != pred] = 1

    gt_img = cort.display.colour_patch(img, gt)
    pred_img = cort.display.colour_patch(img_alt, pred)

    mesh_grid = np.meshgrid(np.arange(gt_img.shape[1]), np.arange(gt_img.shape[0]))
    stripe1_mask = np.zeros_like(gt)
    stripe1_mask[(mesh_grid[0] + mesh_grid[1]) // 2 % 2 == 0] = 1
    stripe2_mask = np.zeros_like(pred)
    stripe2_mask[(mesh_grid[0] + mesh_grid[1]) // 2 % 2 == 1] = 1

    comb_img = np.zeros_like(gt_img)
    comb_img[stripe1_mask == 1, 0] = 225
    comb_img[stripe1_mask == 1, 1] = 0
    comb_img[stripe1_mask == 1, 2] = 0
    comb_img[stripe2_mask == 1] = pred_img[stripe2_mask == 1]
    comb_img[gt == pred] = gt_img[gt == pred]

    ax.imshow(comb_img)
    ax.axis("off")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
