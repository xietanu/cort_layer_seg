import numpy as np


OFFSETS = [(0, 1), (1, 1), (0, -1), (1, -1)]


def find_borders(
    mask: np.ndarray,
):
    """Find the borders of a binary mask."""
    borders = np.zeros((mask.shape[0], mask.shape[1], 7))

    for axis, offset in OFFSETS:
        offset_mask = np.roll(mask, offset, axis=axis)
        if axis == 0 and offset == 1:
            offset_mask[0, :] = 0
        elif axis == 0 and offset == -1:
            offset_mask[-1, :] = 0
        elif axis == 1 and offset == 1:
            offset_mask[:, 0] = 0
        elif axis == 1 and offset == -1:
            offset_mask[:, -1] = 0

        for i in range(7):
            borders[(mask == i) & (offset_mask > i), i] = 1

    if np.sum(borders[:, :, 0]) == 0:
        borders[0, :, 0] = 1

    for i in range(1, 6):
        if np.sum(borders[:, :, i]) == 0:
            borders[:, :, i] = borders[:, :, i - 1]
    if np.sum(borders[:, :, 6]) == 0:
        borders[-1, :, 6] = 1

    return borders
