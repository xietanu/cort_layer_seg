import numpy as np
import chain.border


def find_below_surface_points(
    image: np.ndarray, distance_px: int, threshold: float, padding: int = 0
) -> np.ndarray:
    border = chain.border.find_border(image, threshold)
    expanded_bg = chain.border.expand_border(border, n=distance_px)
    expanded_bg[image >= threshold] = 1

    expanded_border = chain.border.find_border(expanded_bg, 0.5)
    expanded_border[:padding, :] = 0
    expanded_border[-padding:, :] = 0
    expanded_border[:, :padding] = 0
    expanded_border[:, -padding:] = 0

    return np.argwhere(expanded_border)
