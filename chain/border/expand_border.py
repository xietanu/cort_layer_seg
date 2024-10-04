import cv2
import numpy as np


def expand_border(border, n=1):
    border = border.copy()
    for r, c in np.argwhere(border):
        cv2.circle(border, (c, r), n, 1, -1)

    return border
