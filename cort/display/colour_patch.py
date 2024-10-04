import numpy as np
import cv2

import cort


def colour_patch(patch_img, patch_mask, normalize=True) -> np.ndarray:
    img = np.stack([patch_img, patch_img, patch_img], axis=-1)
    if normalize:
        img = ((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6) * 255).astype(
            np.uint8
        )
    elif np.max(img) <= 1:
        img = (img * 255).astype(np.uint8)

    for i, colour in enumerate(cort.constants.COLOURS):
        img[patch_mask == i] = (
            img[patch_mask == i].astype(np.float32)
            / 255
            * np.array(colour)[None, None, :]
        ).astype(np.uint8)

    # ksize = patch_img.shape[0] // 150
    # ksize = ksize + 1 if ksize % 2 == 0 else ksize
    # ksize = min(ksize, 31)

    # borders = cv2.Sobel(src=patch_mask, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)

    # img[borders > 0] = 0
    return img
