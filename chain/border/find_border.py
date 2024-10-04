import numpy as np


def find_border(image: np.ndarray, value: float):
    border = np.zeros_like(image)
    for dir, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
        rolled = np.roll(image, dir, axis)
        border[(image < value) & (rolled >= value)] = 1
    border[0, :] = 0
    border[-1, :] = 0
    border[:, 0] = 0
    border[:, -1] = 0
    return border
