import numpy as np
import cv2

import cort


def colour_volume(volume, volume_seg) -> np.ndarray:
    col_vol = np.stack([volume, volume, volume, volume], axis=-1)
    col_vol = (col_vol - np.min(col_vol)) / (np.max(col_vol) - np.min(col_vol) + 1e-6)

    print("volume_seg", volume_seg.shape)
    print("volume", volume.shape)
    print("col_vol", col_vol.shape)

    for i, colour in enumerate(cort.constants.COLOURS):
        col_vol[volume_seg == i, :3] = (
            col_vol[volume_seg == i, :3].astype(np.float32)
            * np.array(colour)[None, None, None, :]
            / 255.0
        )

    col_vol[:, :, :, 3] = 0
    col_vol[((volume_seg != 0) | (volume < 0.95 * np.max(volume))), 3] = 1
    col_vol[((volume_seg != 0) & (volume >= 0.95 * np.max(volume))), 3] = 0.1

    return col_vol
