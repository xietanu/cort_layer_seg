import numpy as np

# [x,y,z]
# x - z
# |
# y

TOP = 0
BOTTOM = 1
LEFT = 2
RIGHT = 3
FORE = 4
BACK = 5


def determine_orientation(
    volume: np.ndarray,
) -> int:
    white_mask = volume >= 0.95 * volume.max()

    height = volume.shape[1]
    width = volume.shape[2]
    depth = volume.shape[0]

    top_sum = np.sum(white_mask[:, : height // 2, :])
    bottom_sum = np.sum(white_mask[:, height // 2 :, :])
    left_sum = np.sum(white_mask[:, :, : width // 2])
    right_sum = np.sum(white_mask[:, :, width // 2 :])
    fore_sum = np.sum(white_mask[: depth // 2, :, :])
    back_sum = np.sum(white_mask[depth // 2 :, :, :])

    cur_orientation = np.argmax(
        [top_sum, bottom_sum, left_sum, right_sum, fore_sum, back_sum]
    )

    return cur_orientation


def orient_volume(
    volume: np.ndarray,
    cur_orientation: int | None = None,
) -> np.ndarray:
    if cur_orientation is None:
        cur_orientation = determine_orientation(volume)

    if cur_orientation == TOP:
        return volume
    elif cur_orientation == BOTTOM:
        return np.rot90(volume, 2, (1, 2))
    elif cur_orientation == LEFT:
        return np.rot90(volume, 1, (2, 1))
    elif cur_orientation == RIGHT:
        return np.rot90(volume, 1, (1, 2))
    elif cur_orientation == FORE:
        return np.rot90(volume, 1, (0, 1))
    elif cur_orientation == BACK:
        return np.rot90(volume, 3, (0, 1))
    else:
        raise ValueError("Invalid orientation")
