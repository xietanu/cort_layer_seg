import numpy as np


def rotate_volume(volume: np.ndarray, rev=False) -> np.ndarray:
    if volume.ndim == 3:
        if rev:
            axis = (2, 0)
        else:
            axis = (0, 2)
    elif volume.ndim == 4:
        if rev:
            axis = (3, 1)
        else:
            axis = (1, 3)
    else:
        raise ValueError(f"Invalid volume shape: {volume.shape}")
    return np.rot90(volume, 1, axis)
