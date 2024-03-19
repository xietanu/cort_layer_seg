import matplotlib.pyplot as plt
import cv2
import numpy as np

import cort.constants


def display_patch(patch_img: np.ndarray, patch_mask: np.ndarray, ax=None):
    """Display a CorticalPatch."""
    if ax is None:
        ax = plt.gca()

    img = np.stack([patch_img, patch_img, patch_img], axis=-1)
    img = ((img + 1) / 2 * 255).astype(np.uint8)

    for i, colour in enumerate(cort.constants.COLOURS):
        img[patch_mask == i] = (
            img[patch_mask == i].astype(np.float32)
            / 255
            * np.array(colour)[None, None, :]
        ).astype(np.uint8)
        
    ksize = patch_img.shape[0] // 150
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    ksize = min(ksize, 31)

    borders = cv2.Sobel(src=patch_mask, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)

    img[borders > 0] = 0

    ax.imshow(img)
    
    return ax
