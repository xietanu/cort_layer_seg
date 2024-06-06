import numpy as np

OFFSETS = [(0, 1), (1, 1), (0, -1), (1, -1)]


def map_distance_from_border(
    mask: np.ndarray,
    border: np.ndarray,
):
    """Map the distance from the border of a binary mask."""
    distance = np.full(border.shape, np.inf)
    distance[border > 0] = 0

    valid_layers = np.any(border > 0, axis=(0, 1))

    cur_value = 0

    while np.inf in distance[:, :, valid_layers]:
        next_distance = np.full(border.shape, np.inf)
        next_distance[distance <= cur_value] = cur_value + 1

        for axis, offset in OFFSETS:
            offset_distance = np.roll(next_distance, offset, axis=axis)
            if axis == 0 and offset == 1:
                offset_distance[0, :, :] = np.inf
            elif axis == 0 and offset == -1:
                offset_distance[-1, :, :] = np.inf
            elif axis == 1 and offset == 1:
                offset_distance[:, 0, :] = np.inf
            elif axis == 1 and offset == -1:
                offset_distance[:, -1, :] = np.inf
            distance = np.minimum(distance, offset_distance)

        cur_value += 1

    for i in range(7):
        distance[mask <= i, i] *= -1

    distance = np.tanh(distance / 10)

    return distance
